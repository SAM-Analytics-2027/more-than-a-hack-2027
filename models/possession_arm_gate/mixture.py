"""Mixture-of-experts gate (probability-space blend) for possession_arm_gate.

Instead of the log-linear meta-regression (which behaves like a geometric mean),
this variant routes per CLASS a convex mixture over the ingredient families:

    P_gate[:, c] = sum_f w[f, c | x] * P_f[:, c],    x = (entropies, ctx tiers)

w comes from a tiny LINEAR routing net (9 features -> 3 classes x 3 families,
softmax over the family axis per class) so the parameters stay few (~30) and
the blend always yields a valid distribution. Trained with Adam on labeled
frames only; early stopping mirrors the family protocol (the NEXT fold as val,
patience/floor, final artifact = LOO-p80 fixed epochs).

Everything downstream reuses this family's own protocol artifacts: the LOO
posteriors saved by eval_external.py feed the shared external decoder search.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from models.possession_arm_gate.train import GATE_FAMILIES, ARTIFACT_DIR, MODEL_TAG

# mixture hyper-parameters (fixed family constants, like every other family's base params)
MIX_EPOCHS = 600
MIX_PATIENCE = 30
MIX_FLOOR = 80
MIX_LR = 1e-2
MIX_WD = 1e-5

_ROUTE_KEYS = ("impact_loud", "control_loud", "in_occlusion",
               "stop_strong", "stop_lowtempo", "restart_cue")


def _route_features_ctx(ctx: Dict[str, np.ndarray]) -> np.ndarray:
    """Routing features [n, 6]: only the raw context tiers (no posterior info)."""
    return np.stack([np.asarray(ctx[k], dtype=np.float64) for k in _ROUTE_KEYS], axis=1)


def route_feature_names() -> List[str]:
    return list(_ROUTE_KEYS)


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _route_tensors(files, post_cache):
    """Concat the mixture tensors of the given files: (feats [N,6], y [N], Pf [F,N,3])."""
    import torch

    fs, ys = [], []
    logPf = []
    for f in files:
        key = (f["match_id"], bool(f["dirty"]))
        fs.append(torch.from_numpy(_route_features_ctx(f["ctx"])).float())
        y = torch.from_numpy(np.asarray(f["labels"], dtype=np.int64))
        ys.append(y)
        per_fam = [torch.from_numpy(np.asarray(post_cache[key][fam], dtype=np.float32))
                   for fam in GATE_FAMILIES]
        logPf.append(torch.stack(per_fam, 0))          # [F, n, 3]
    feats = torch.cat(fs, 0)
    y = torch.cat(ys, 0)
    PF = torch.cat(logPf, 1)                            # [F, N, 3]
    y = y.to(_device())
    return feats, y, PF


def _blend_P(w: "torch.Tensor", PF: "torch.Tensor") -> "torch.Tensor":
    """w [N, 3C, F] softmax rows -> P_gate [N, 3C] = sum_f w * PF."""
    return (w * PF.permute(1, 2, 0)).sum(-1)


def mixture_logits(feats: "torch.Tensor", W: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    """Routing logits [N, 3C, F] from features [N, F_in] and weight [3C, F_in]."""
    z = feats @ W.T + b                                # [N, 3C]
    return z.reshape(feats.shape[0], 3, len(GATE_FAMILIES))


def mixture_posterior_t(w_rows: np.ndarray, PF_rows: np.ndarray) -> np.ndarray:
    """numpy convenience: w [N, 3C, F] + PF [F, N, 3] -> P [N, 3C]."""
    return (w_rows * PF_rows).sum(-1).astype(np.float32)


def apply_mixture(Ps: List[np.ndarray], ctx: Dict[str, np.ndarray],
                  mean: np.ndarray, scale: np.ndarray, W: np.ndarray,
                  b: np.ndarray) -> np.ndarray:
    """Competition/eval blend of per-file feature rows (no torch needed)."""
    F_in = mean.shape[0]
    x = ((_route_features_ctx(ctx) - mean) / scale).astype(np.float64)
    logits = (x @ W.T + b).reshape(-1, 3, len(Ps))
    w = np.exp(logits - logits.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)                      # [n, 3, F]
    return (w * np.stack(Ps, -1)).sum(-1).astype(np.float32)


def fit_mixture(train_files, post_cache, stop_files=None,
                epochs_override: Optional[int] = None, verbose: bool = True) -> Dict:
    """Fit the mixture router; returns params dict (W, b, mean, scale, info...)."""
    import torch
    import torch.nn.functional as TNF

    torch.manual_seed(2027)
    n_fam = len(GATE_FAMILIES)
    F_in = _route_features_ctx(train_files[0]["ctx"]).shape[1]     # 6
    device = _device()
    if verbose:
        print(f"mixture gate device: {device}")

    feats, y, PF = _route_tensors(train_files, post_cache)
    feats, PF = feats.to(device), PF.to(device)
    keep = y >= 0
    feats, y, PF = feats[keep], y[keep], PF[:, keep]
    n_rows = int(keep.sum())

    mean = feats.mean(0, keepdim=True)
    std = feats.std(0, keepdim=True).clamp(min=1e-6)
    feats_n = (feats - mean) / std
    route_out = 3 * n_fam                               # 3 classes x F families

    stop_state = None
    if stop_files:
        sf, sy, sPF = _route_tensors(stop_files, post_cache)
        sf, sy, sPF = sf.to(device), sy.to(device), sPF.to(device)
        sk = sy >= 0
        sf, sy, sPF = sf[sk], sy[sk], sPF[:, sk]
        n_val = int(sk.sum())
        stop_state = ((sf - mean) / std, sy, sPF)
        if verbose:
            print(f"  val rows: {n_val}")

    max_ep = int(epochs_override or MIX_EPOCHS)
    W = torch.zeros(route_out, F_in, device=device, requires_grad=True)
    b = torch.zeros(route_out, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=MIX_LR, weight_decay=MIX_WD)

    best = {"ep": 0, "val": float("inf"), "state": None}

    for ep in range(1, max_ep + 1):
        opt.zero_grad()
        w = TNF.softmax(mixture_logits(feats_n, W, b), dim=-1)     # [N, 3, F]
        P = _blend_P(w, PF)                                        # [N, 3]
        loss = TNF.cross_entropy(torch.log(P.clamp_min(1e-7)), y)
        loss.backward()
        opt.step()
        if stop_state is not None and ep >= MIX_FLOOR:
            sf_n, sy, sPF = stop_state
            with torch.no_grad():
                w = TNF.softmax(mixture_logits(sf_n, W, b), dim=-1)
                Pv = _blend_P(w, sPF)
                val = float(TNF.cross_entropy(torch.log(Pv.clamp_min(1e-7)), sy).item())
            if verbose and ep % 50 == 0:
                print(f"  ep{ep:>3} val_logloss={val:.4f} ({loss.item():.4f})")
            if val < best["val"] - 1e-5:
                best = {"ep": ep, "val": val,
                        "state": (W.detach().clone(), b.detach().clone())}
            if (ep - best["ep"]) >= MIX_PATIENCE:
                if verbose:
                    print(f"  early stop at ep{ep} (best ep{best['ep']} val={best['val']:.4f})")
                break
    if best["state"] is not None:
        W = torch.nn.Parameter(best["state"][0])
        b = torch.nn.Parameter(best["state"][1])
    info = {"best_epoch": best["ep"] or max_ep, "best_val": best["val"]
            if best["val"] < float("inf") else None,
            "epochs_run": ep, "n_train_rows": n_rows}
    return {"W": W.detach().cpu().numpy().astype(np.float64),
            "b": b.detach().cpu().numpy().astype(np.float64),
            "mean": mean.detach().cpu().numpy().astype(np.float64),
            "scale": std.detach().cpu().numpy().astype(np.float64),
            **info}


def fit_final_and_save(files, post_cache, epochs_final: int, note: str = "") -> None:
    """Fit on all 20 files with fixed epochs; save gate_final.npz as the MIXTURE form."""
    params = fit_mixture(files, post_cache, stop_files=None, epochs_override=epochs_final)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    npz_path = os.path.join(ARTIFACT_DIR, "gate_final.npz")
    np.savez(npz_path, W=params["W"], b=params["b"],
             mean=params["mean"], scale=params["scale"])
    meta = {"tag": MODEL_TAG, "blend": "mixture", "families": list(GATE_FAMILIES),
            "route_features": route_feature_names(),
            "epochs_final": int(epochs_final),
            "n_train_rows": params["n_train_rows"],
            "note": note or "per-class convex mixture over ingredient posteriors; linear router"}
    with open(os.path.splitext(npz_path)[0] + "_meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    print(f"final mixture gate saved (epochs={epochs_final}, "
          f"rows={params['n_train_rows']}, route feats={len(route_feature_names())})")


def load_final() -> Dict:
    """Artifact loader for predict.py: returns blend='mixture' params or raises."""
    npz_path = os.path.join(ARTIFACT_DIR, "gate_final.npz")
    meta_path = os.path.splitext(npz_path)[0] + "_meta.json"
    if not os.path.isfile(npz_path):
        raise SystemExit(f"missing gate artifact: {npz_path}\n"
                         "run: python models/possession_arm_gate/train.py --mode final")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("blend") != "mixture":
        raise ValueError(f"artifact is blend={meta.get('blend')!r}, not mixture")
    with np.load(npz_path, allow_pickle=False) as z:
        return {"W": np.asarray(z["W"], dtype=np.float64),
                "b": np.asarray(z["b"], dtype=np.float64),
                "mean": np.asarray(z["mean"], dtype=np.float64),
                "scale": np.asarray(z["scale"], dtype=np.float64),
                "meta": meta}
