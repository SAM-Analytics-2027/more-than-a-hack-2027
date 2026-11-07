"""Stage-1 training for possession_arm_crf: frame-level emission MLP + per-fold
duration model, then the duration-state forward-backward for class marginals.

    python models/possession_arm_crf/train.py --mode final                 all-10-matches fit
    python models/possession_arm_crf/train.py --mode loo --fold 3          one LOO fold
    python models/possession_arm_crf/train.py --mode loo --fold 3 --device cpu

Protocol mapping (documented, as in the TCN family): the GBDT "rounds" become
EPOCHS of the frame-level emission fit; early stopping watches the NEXT fold's
weighted logloss with patience 6 / floor 8 / max 40 epochs. The duration model
(hazards, sb, A, pi) re-learns from each fold's TRAIN files' clean GT spells
only. Emissions are frame-level (NO temporal windows): temporal structure is
exclusively the expanded chain, that is the paradigm.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_crf/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F

from models.modules.cache import load_all_files
from models.modules.data_external import EXTERNAL_IDS
from models.modules.training_protocol import fold_of
from models.possession_arm_crf.decode import learn_duration_model, spell_marginals
from models.possession_arm_crf.network import EmissionMLP, window_split

MODEL_TAG = "possession_arm_crf"

ARTIFACT_DIR = os.path.join(HERE, "artifacts")

CRF_PARAMS = {
    "hidden": 64,
    "dropout": 0.05,
    "batch": 65536,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-4,
    "max_epochs": 40,
    "patience": 6,          # epochs without val improvement
    "floor_epochs": 8,      # epoch-scale floor (analogue of the 150-round GBDT floor)
    "grad_clip": 5.0,
    "seed": 17,
}
PREDICT_CHUNK = 262144


def resolve_device(force: str = "auto") -> torch.device:
    """cuda when available; explicit --device cpu/cuda overrides (cuda falls back)."""
    if force == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if force == "cuda":
        print("device: cuda requested but unavailable -> cpu")
    return torch.device("cpu")


def file_tensor(f: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """One cache file -> (Z [n,218] filled+mask CPU float32, y int64)."""
    X = np.ascontiguousarray(f["X"], dtype=np.float32)
    filled, mask = window_split(torch.from_numpy(X))
    Z = torch.cat([filled, mask], dim=1)
    return Z, torch.from_numpy(np.asarray(f["labels"], dtype=np.int64))


def class_weight_tensor(y: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Balanced per-class weights (v1 alpha scheme) from THIS training set only."""
    counts = {c: max(1, int((y == c).sum())) for c in (0, 1, 2)}
    n = int(y.numel())
    w = np.asarray([n / (3.0 * counts[c]) for c in (0, 1, 2)], dtype=np.float32)
    return torch.from_numpy(w).to(device)


def _val_metrics(model, files: List[Dict], cw, dev) -> Tuple[float, float]:
    model.eval()
    tot, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for f in files:
            Z, y = file_tensor(f)
            keep = y >= 0
            Z, y = Z[keep], y[keep]
            C = Z.shape[1] // 2
            for s in range(0, Z.shape[0], PREDICT_CHUNK):
                zb = Z[s:s + PREDICT_CHUNK].to(dev)
                yb = y[s:s + PREDICT_CHUNK].to(dev)
                logits = model(zb[:, :C], zb[:, C:])
                loss_sum += float(F.cross_entropy(logits, yb, weight=cw, reduction="sum"))
                tot += int(yb.numel())
                correct += int((logits.argmax(1) == yb).sum())
    model.train()
    return loss_sum / max(1, tot), correct / max(1, tot)


def fit_fold(train_files: List[Dict], eval_files: Optional[List[Dict]] = None,
             device: str = "auto", epochs_override: Optional[int] = None,
             verbose: bool = True) -> Tuple:
    """Train the emission MLP on labeled frames + learn the duration model.

    Returns (model, names, info) with info = {best_epoch, best_val, epochs_run,
    duration} (duration re-learned from this fold's clean GT spells).
    """
    t0 = time.time()
    dev = resolve_device(device)
    names = train_files[0]["names"]
    in_ch = 2 * len(names)
    torch.manual_seed(CRF_PARAMS["seed"])
    np.random.seed(CRF_PARAMS["seed"])

    dur = learn_duration_model(train_files)

    Zs, ys = [], []
    for f in train_files:
        Z, y = file_tensor(f)
        keep = y >= 0
        Zs.append(Z[keep])
        ys.append(y[keep])
    Z = torch.cat(Zs, 0)
    y = torch.cat(ys, 0)
    del Zs, ys
    cw = class_weight_tensor(y, dev)
    C = in_ch // 2
    if verbose:
        print(f"  crf fit: {Z.shape[0]:,} labeled frames, device={dev}, "
              f"duration from {sum(1 for f in train_files if not f['dirty'])} clean files")

    model = EmissionMLP(in_ch=in_ch, hidden=CRF_PARAMS["hidden"],
                        dropout=CRF_PARAMS["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CRF_PARAMS["lr"],
                            weight_decay=CRF_PARAMS["weight_decay"])

    best = {"epoch": None, "val": float("inf"), "state": None}
    max_ep = int(epochs_override or CRF_PARAMS["max_epochs"])
    epochs_run = 0
    stop_ok = bool(eval_files) and epochs_override is None
    n = Z.shape[0]
    for epoch in range(1, max_ep + 1):
        epochs_run = epoch
        model.train()
        te = time.time()
        perm = torch.randperm(n)
        for s in range(0, n, CRF_PARAMS["batch"]):
            idx = perm[s:s + CRF_PARAMS["batch"]]
            zb = Z[idx].to(dev)
            yb = y[idx].to(dev)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(zb[:, :C], zb[:, C:]), yb, weight=cw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CRF_PARAMS["grad_clip"])
            opt.step()
        if stop_ok:
            val_loss, val_acc = _val_metrics(model, eval_files, cw, dev)
            improved = val_loss < best["val"] - 1e-5
            if improved:
                best = {"epoch": epoch, "val": val_loss,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            if verbose:
                print(f"  ep{epoch:>3} val_logloss={val_loss:.4f} acc={val_acc:.3f} "
                      f"{' *' if improved else ''} ({time.time() - te:.0f}s)")
            if epoch >= CRF_PARAMS["floor_epochs"] and (epoch - best["epoch"]) >= CRF_PARAMS["patience"]:
                if verbose:
                    print(f"  early stop at epoch {epoch} "
                          f"(best ep{best['epoch']} val={best['val']:.4f})")
                break
        elif verbose:
            print(f"  ep{epoch:>3} train-only ({time.time() - te:.0f}s)")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    del Z, y

    info = {"best_epoch": best["epoch"] or epochs_run,
            "best_val": None if best["val"] == float("inf") else best["val"],
            "epochs_run": epochs_run, "duration": dur}
    return model, names, info


def predict_with(model, dur: Dict, f: Dict) -> np.ndarray:
    """Full-file posterior [n, 3] for ONE cache file dict (emissions -> FB marginals)."""
    dev = next(model.parameters()).device
    model.eval()
    X = np.ascontiguousarray(f["X"], dtype=np.float32)
    filled, mask = window_split(torch.from_numpy(X))
    n = filled.shape[0]
    logits = []
    with torch.no_grad():
        for s in range(0, n, PREDICT_CHUNK):
            zf = filled[s:s + PREDICT_CHUNK].to(dev)
            zm = mask[s:s + PREDICT_CHUNK].to(dev)
            logits.append(model(zf, zm).cpu())
    logits = torch.cat(logits, 0)
    log_emit = torch.log_softmax(logits, dim=1).numpy().astype(np.float64)
    return spell_marginals(log_emit, dur, f["blocks"])


def make_fit_predict(device: str = "auto"):
    """Adapter for models.modules.training_protocol.run_nested_loo (file-dict contract)."""

    def fit_predict(train_files: List[Dict], stop_files: List[Dict]) -> Tuple:
        model, names, info = fit_fold(train_files, eval_files=stop_files, device=device)

        def predict_fn(file_dict: Dict) -> np.ndarray:
            return predict_with(model, info["duration"], file_dict)

        return predict_fn, {"best_iteration": info["best_epoch"]}

    return fit_predict


def model_path() -> str:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    return os.path.join(ARTIFACT_DIR, "crf_final.pt")


def save_model(model, names: List[str], dur: Dict, meta: Dict) -> str:
    path = model_path()
    torch.save(model.state_dict(), path)
    with open(os.path.splitext(path)[0] + "_meta.json", "w", encoding="utf-8") as handle:
        json.dump({"feature_names": names, "model_tag": MODEL_TAG,
                   "params": CRF_PARAMS, "duration": dur, **meta},
                  handle, indent=2, sort_keys=True)
    return path


def predict_final(f: Dict, device: str = "auto") -> Tuple[np.ndarray, List[str]]:
    """Competition-time posterior from the saved final artifact (file-dict contract)."""
    path = model_path()
    with open(os.path.splitext(path)[0] + "_meta.json", "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    dev = resolve_device(device)
    model = EmissionMLP(in_ch=2 * len(meta["feature_names"]),
                        hidden=meta["params"]["hidden"],
                        dropout=meta["params"]["dropout"]).to(dev)
    model.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    P = predict_with(model, meta["duration"], f)
    return P, meta["feature_names"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("loo", "final"), default="final")
    parser.add_argument("--fold", type=int, default=None, help="held-out fold for loo mode (0..9)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    dev = resolve_device(args.device)
    print(f"device: {dev} ({torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'})")
    files = load_all_files(clean_and_dirty=True)
    print(f"cache: {len(files)} files, feature-dim={len(files[0]['names'])}")

    if args.mode == "final":
        t0 = time.time()
        # final epochs: same p80-from-LOO convention as the other documented families
        n_epochs = None
        summary_path = os.path.join(ARTIFACT_DIR, "loo_rounds_summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                rounds = json.load(handle)["rounds"]
            rounds = [int(r) for r in rounds if r]
            if rounds:
                n_epochs = int(np.percentile(rounds, 80))
                print(f"final epochs from LOO p80: {n_epochs}")
        model, names, info = fit_fold(files, epochs_override=n_epochs, device=args.device)
        save_model(model, names, info["duration"],
                   {"mode": "final", "training_files":
                       [f"{f['match_id']}{'_dirty' if f['dirty'] else ''}" for f in files],
                    "epochs_run": info["epochs_run"]})
        print(f"final model saved ({time.time() - t0:.1f}s)")
    else:
        fold = args.fold if args.fold is not None else -1
        if not 0 <= fold < len(EXTERNAL_IDS):
            parser.error("--mode loo needs --fold in 0..9")
        train_files = [f for f in files if fold_of(f) != fold]
        eval_files = [f for f in files if fold_of(f) == fold]
        print(f"fold {fold} ({EXTERNAL_IDS[fold]}): train on {len(train_files)} files, "
              f"hold out {len(eval_files)}")
        model, names, info = fit_fold(train_files, eval_files=eval_files, device=args.device)
        save_model(model, names, info["duration"], {"mode": "loo", "fold": fold})
    return 0


if __name__ == "__main__":
    sys.exit(main())
