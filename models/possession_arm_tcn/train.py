"""Stage-1 training for possession_arm_tcn: causal dilated TCN (torch) on WINDOWS
of the identical v2 features, same external-only rules as every other family.

    python models/possession_arm_tcn/train.py --mode final                all-10-matches fit
    python models/possession_arm_tcn/train.py --mode loo --fold 3        one LOO fit

Protocol mapping (documented so the comparison stays honest): the GBDT families'
"rounds" become EPOCHS here; early stopping watches the NEXT fold's weighted
multi-class logloss with patience 6 epochs and a floor of 8 epochs (a boosting
floor of 150 rounds over ~1.4M frames is roughly 1 epoch in GBDT terms, so an
epoch-scale floor is the honest equivalent, not a tweak). Class weights use the
same balanced scheme as v1 (per-epoch class-weighted cross entropy).
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_tcn/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from models.modules.cache import load_all_files
from models.modules.training_protocol import fold_of  # noqa: E402
from models.possession_arm_tcn.network import TCN, window_split

MODEL_TAG = "possession_arm_tcn"

ARTIFACT_DIR = os.path.join(HERE, "artifacts")

# Architecture / optimization constants. W=33 (1.3 s) is a documented choice, not
# tuned on the competition data; the receptive field (125 frames, 5 s each side)
# comes from the dilation ladder. Seed is fixed for reproducibility.
TCN_PARAMS = {
    "window": 33,
    "width": 96,
    "dilations": [1, 2, 4, 8, 16],
    "dropout": 0.10,
    "batch": 4096,
    "lr": 1.5e-3,
    "weight_decay": 1e-4,
    "max_epochs": 40,
    "patience": 6,          # epochs without val improvement
    "floor_epochs": 8,      # small-epoch floor (analogue of the GBDT 150-round floor)
    "seed": 17,
}
MAX_FRAME_CHUNK = 24576   # frames per unfold call (RAM-bound, not a tuning knob)

DEVICE: Optional[torch.device] = None


def resolve_device() -> torch.device:
    """CUDA when available (same accelerator policy as the GBDT families)."""
    global DEVICE
    if DEVICE is None:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DEVICE


def _seed_everything(fold_key: int = 0) -> None:
    seed = TCN_PARAMS["seed"] + fold_key
    torch.manual_seed(seed)
    np.random.seed(seed)


def class_counts(labels: np.ndarray) -> torch.Tensor:
    """Balanced per-class weights (v1 alpha scheme) -> tensor [3] on the fit device."""
    codes = labels[labels >= 0]
    counts = {c: max(1, int((codes == c).sum())) for c in (0, 1, 2)}
    n = int(codes.size)
    w = np.array([n / (3.0 * counts[c]) for c in (0, 1, 2)], dtype=np.float32)
    w = w / w.mean()          # scale-invariant for cross entropy
    return torch.from_numpy(w).to(resolve_device())


def file_splits(f: Dict) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """One cache file -> (filled, mask) CPU tensors + labels tensor on device.

    Labels are None at competition time (no GT) -- only training/LOO windows need them.
    """
    X = torch.from_numpy(np.ascontiguousarray(f["X"], dtype=np.float32))
    filled, mask = window_split(X)
    y = None if f["labels"] is None else torch.from_numpy(f["labels"].astype(np.int64))
    return filled, mask, y, torch.from_numpy(np.asarray(f["blocks"], dtype=np.int64))


def batch_windows(filled: torch.Tensor, mask: torch.Tensor, y: torch.Tensor,
                  blocks: torch.Tensor, W: int, device: torch.device,
                  chunk: int = MAX_FRAME_CHUNK, labeled_only: bool = False):
    """Yield (windows [K, W, 2C], targets [K]) chunks over one file.

    Blocks are (half, start, END) with END INCLUSIVE (half_blocks.py convention),
    so the frame range of a block is a..b closed. Windows stay inside their own
    half block; edges are replicated with the missing-mask set to one, so padded
    context is honest 'unseen' evidence, not fabricated ball data.
    """
    h = W // 2
    if y is None:
        # competition-time file: no GT; a dummy all-ungraded tensor keeps the
        # window slicing intact (targets are ignored by predict_with)
        y = torch.full((int(filled.shape[0]),), -1, dtype=torch.int64)
    for a, b, _half in [(int(r[1]), int(r[2]), int(r[0])) for r in blocks]:
        b_in = min(b, filled.shape[0] - 1)
        if b_in < a:
            continue
        fa, ma = filled[a:b_in + 1], mask[a:b_in + 1]
        t_all = y[a:b_in + 1]
        C = fa.shape[1]
        if fa.shape[0] < W:
            # block shorter than one window (rare): build each window explicitly
            for i in range(fa.shape[0]):
                if labeled_only and int(t_all[i]) < 0:
                    continue
                win_f, win_m = _center_window(fa, ma, i, W)
                w_win = torch.cat([win_f, win_m], dim=1).unsqueeze(0).to(device)
                yield w_win, t_all[i:i + 1].to(device)
            continue
        fpad = torch.cat([torch.zeros((h, fa.shape[1]), dtype=fa.dtype), fa,
                          torch.zeros((h, fa.shape[1]), dtype=fa.dtype)], 0)
        mpad = torch.cat([torch.ones((h, fa.shape[1]), dtype=fa.dtype), ma,
                          torch.ones((h, fa.shape[1]), dtype=fa.dtype)], 0)
        both = torch.cat([fpad, mpad], dim=1)            # [len+2h, 2C]
        Xw = both.unfold(0, W, 1)                         # [len, 2C, W]
        Xw = Xw.transpose(1, 2).contiguous()              # [len, W, 2C]
        del both
        n_block = fa.shape[0]
        m = 0
        while m < n_block:
            hi = min(m + chunk, n_block)
            w = Xw[m:hi].to(device, non_blocking=True)
            t = t_all[m:hi].to(device, non_blocking=True)
            if labeled_only:
                keep = t >= 0
                if keep.any():
                    yield w[keep], t[keep]
            else:
                yield w, t
            m = hi
        del Xw


def _center_window(fa: torch.Tensor, ma: torch.Tensor, i: int, W: int):
    """One (W, C) filled + mask window centered at i: real frames where available,
    all-missing (mask=1, filled=...) padding where the block is too short."""
    h = W // 2
    C = fa.shape[1]
    lo = max(0, i - h)
    hi = min(fa.shape[0], i + h + 1)
    pad_pre_f = torch.zeros((max(0, h - (i - lo)), C), dtype=fa.dtype)
    pad_post_f = torch.zeros((max(0, (i + h + 1) - hi), C), dtype=fa.dtype)
    pad_pre_m = torch.ones((pad_pre_f.shape[0], C), dtype=fa.dtype)
    pad_post_m = torch.ones((pad_post_f.shape[0], C), dtype=fa.dtype)
    win_f = torch.cat([pad_pre_f, fa[lo:hi], pad_post_f], 0)[:W]
    win_m = torch.cat([pad_pre_m, ma[lo:hi], pad_post_m], 0)[:W]
    return win_f, win_m


def _forward_loss(model, w_win, t, cw) -> torch.Tensor:
    logits = model(w_win)
    keep = t >= 0
    total = keep.sum()
    if total == 0:
        return None
    logits = logits[keep]
    t = t[keep]
    return F.cross_entropy(logits, t, weight=cw)


def _val_logloss(model, files: List[Dict], cw) -> Tuple[float, float]:
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for f in files:
            filled, mask, y, blocks = file_splits(f)
            for w_win, t in batch_windows(filled, mask, y, blocks, TCN_PARAMS["window"],
                                          resolve_device(), labeled_only=True):
                logits = model(w_win)
                loss_sum += float(F.cross_entropy(logits, t, weight=cw, reduction="sum"))
                tot += int(logits.shape[0])
                correct += int((logits.argmax(1) == t).sum())
            del filled, mask, y, blocks
    model.train()
    return loss_sum / max(1, tot), correct / max(1, tot)


def predict_with(model, f: Dict, W: int) -> np.ndarray:
    """Full-FILE posterior [n, 3] for ONE cache file dict (all frames).

    The windowed pass covers the half blocks; frames outside any block get the
    uninformative 1/3 row (the seg/CRF convention) -- the shared decoder
    abstains there. On the external caches the blocks cover every frame, so
    this is a no-op for the LOO artifacts; it only matters for the competition
    match (BAR-ATH has stoppage gaps).
    """
    model.eval()
    filled, mask, y, blocks = file_splits(f)
    n = int(filled.shape[0])
    P_full = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
    rows = 0
    with torch.no_grad():
        for (a, b, _half) in [(int(r[1]), int(r[2]), int(r[0])) for r in blocks]:
            fb, mb = filled[a:b + 1], mask[a:b + 1]
            tb = y[a:b + 1] if y is not None else None
            outs = []
            for w_win, _t in batch_windows(fb, mb, tb, [(_half, 0, int(fb.shape[0]) - 1)],
                                           W, resolve_device()):
                p = F.softmax(model(w_win), dim=1)
                outs.append(p.cpu())
                del w_win
            P_block = torch.cat(outs, 0).numpy().astype(np.float32)
            b_in = min(b, n - 1)
            P_full[a:b_in + 1] = P_block[:b_in + 1 - a]
            rows += P_block.shape[0]
    return P_full


def fit_fold(train_files: List[Dict], eval_files: Optional[List[Dict]] = None,
             epochs_override: Optional[int] = None, verbose: bool = True) -> Tuple:
    """Train one TCN; returns (model, names, best_epoch_info dict).

    Early stopping: weighted multiclass logloss on the supplied eval files (the
    NEXT fold), patience/max/floor epochs as documented in the module docstring.
    For the FINAL artifact pass epochs_override (the LOO p80): the run then has
    no val split and no early stopping, exactly `epochs_override` epochs.
    """
    _seed_everything()
    device = resolve_device()
    names = train_files[0]["names"]
    W = TCN_PARAMS["window"]

    cw = class_counts(np.concatenate([f["labels"] for f in train_files]))
    model = TCN(in_ch=TCN.IN_CH, width=TCN_PARAMS["width"],
                dilations=tuple(TCN_PARAMS["dilations"]),
                dropout=TCN_PARAMS["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TCN_PARAMS["lr"],
                            weight_decay=TCN_PARAMS["weight_decay"])
    max_ep_wanted = TCN_PARAMS["max_epochs"]
    max_ep = int(epochs_override or max_ep_wanted)
    patience, floor_ep = TCN_PARAMS["patience"], TCN_PARAMS["floor_epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_ep_wanted, eta_min=TCN_PARAMS["lr"] * 0.05)

    file_order: List[Dict] = list(train_files)
    best = {"epoch": None, "val": float("inf"), "state": None}
    history = []
    for epoch in range(1, max_ep + 1):
        rng = np.random.default_rng(TCN_PARAMS["seed"] + epoch)
        rand_perm = rng.permutation(len(file_order))
        model.train()
        t0 = time.time()
        seen = 0
        for idx in rand_perm:
            f = file_order[idx]
            filled, mask, y, blocks = file_splits(f)
            for w_win, t in batch_windows(filled, mask, y, blocks, W, device,
                                          labeled_only=True):
                opt.zero_grad(set_to_none=True)
                loss = _forward_loss(model, w_win, t, cw)
                if loss is None:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                seen += int(t.shape[0])
                del w_win,t
            del filled, mask, y, blocks
        sched.step()
        val_loss, val_acc = (None, None)
        if eval_files:
            val_loss, val_acc = _val_logloss(model, eval_files, cw)
            improved = val_loss < best["val"] - 1e-5
            if improved:
                best = {"epoch": epoch, "val": val_loss,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            history.append({"epoch": epoch, "val": val_loss, "acc": val_acc})
            if verbose:
                print(f"  ep{epoch:>3} n={seen:>8} val_logloss={val_loss:.4f} "
                      f"acc={val_acc:.3f} {' *' if improved else ''} ({time.time() - t0:.0f}s)")
            if epoch >= floor_ep and (epoch - best["epoch"]) >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} "
                          f"(best ep{best['epoch']} val={best['val']:.4f})")
                break
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    info = {"best_epoch": best["epoch"] or max_ep, "best_val": best["val"]
            if best["val"] < float("inf") else None, "epochs_run": len(history)}
    return model, names, info


def make_fit_predict():
    """Adapter for models.modules.training_protocol.run_nested_loo (file-dict contract)."""

    def fit_predict(train_files: List[Dict], stop_files: List[Dict]) -> Tuple:
        model, names, info = fit_fold(train_files, eval_files=stop_files)

        def predict_fn(file_dict: Dict) -> np.ndarray:
            return predict_with(model, file_dict, TCN_PARAMS["window"])

        return predict_fn, {"best_iteration": info["best_epoch"]}

    return fit_predict


def predict_final(f: Dict) -> Tuple[np.ndarray, List[str]]:
    """Competition-time posterior from the saved final artifact (file-dict contract)."""
    path = model_path("final")
    with open(os.path.splitext(path)[0] + "_meta.json", "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("window") != TCN_PARAMS["window"]:
        raise RuntimeError("window mismatch: meta json disagrees with TCN_PARAMS")
    device = resolve_device()
    model = TCN(in_ch=TCN.IN_CH, width=meta["width"], dilations=tuple(meta["dilations"]),
                dropout=meta["dropout"]).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    P = predict_with(model, f, TCN_PARAMS["window"])
    return P, meta["feature_names"]


def model_path(mode: str) -> str:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    return os.path.join(ARTIFACT_DIR, f"tcn_{mode}.pt")


def save_model(model, names: List[str], meta: Dict, path: str) -> None:
    torch.save(model.state_dict(), path)
    with open(os.path.splitext(path)[0] + "_meta.json", "w", encoding="utf-8") as handle:
        json.dump({"feature_names": names, **meta}, handle, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("loo", "final"), default="final")
    parser.add_argument("--fold", type=int, default=None, help="held-out fold for loo mode (0..9)")
    args = parser.parse_args()

    device = resolve_device()
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    files = load_all_files(clean_and_dirty=True)
    print(f"cache: {len(files)} files, feature-dim={len(files[0]['names'])}")

    if args.mode == "final":
        t0 = time.time()
        # final epochs: same p80-from-LOO convention as the GBDT families, but at
        # epoch scale (nothing is tuned on the competition data)
        n_epochs = None
        summary_path = os.path.join(ARTIFACT_DIR, "loo_rounds_summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                rounds = json.load(handle)["rounds"]
            n_epochs = int(np.percentile([r for r in rounds if r], 80))
            print(f"final epochs from LOO p80: {n_epochs}")
        model, names, info = fit_fold(files, epochs_override=n_epochs)
        save_model(
            model, names,
            {"mode": "final", "model_tag": MODEL_TAG, **TCN_PARAMS,
             "training_files":
                 [f"{f['match_id']}{'_dirty' if f['dirty'] else ''}" for f in files],
             "best_epoch": info["best_epoch"] if info and info.get("best_epoch") else None},
            model_path("final"),
        )
        print(f"final model saved ({time.time() - t0:.1f}s)")
    else:
        fold = args.fold if args.fold is not None else -1
        if not 0 <= fold < 10:
            parser.error("--mode loo needs --fold in 0..9")
        train_files = [f for f in files if fold_of(f) != fold]
        eval_files = [f for f in files if fold_of(f) == fold]
        print(f"fold {fold}: train on {len(train_files)} files, hold out {len(eval_files)}")
        model, names, info = fit_fold(train_files, eval_files=eval_files)
        save_model(
            model, names,
            {"mode": "loo", "fold": fold, **TCN_PARAMS}, model_path("loo"),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
