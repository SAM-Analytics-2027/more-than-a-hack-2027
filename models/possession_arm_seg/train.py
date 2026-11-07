"""Stage-1 for possession_arm_seg: bottom-up change-point segmentation + segment
logistic classifier, then frame posteriors by segment broadcast.

    python models/possession_arm_seg/train.py --mode final                 all-10-matches fit
    python models/possession_arm_seg/train.py --mode loo --fold 3         one LOO fold

Everything below is trained ONLY on the external fold's training files; the brand
new constants (L0, penalty, cap, dilation) are documented units, not tuned on any
held-out or competition data.
"""

import argparse
import heapq
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))    # models/possession_arm_seg/ -> models/ -> repo
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from models.modules.cache import load_all_files
from models.modules.data_external import EXTERNAL_IDS
from models.modules.training_protocol import fold_of

MODEL_TAG = "possession_arm_seg"

ARTIFACT_DIR = os.path.join(HERE, "artifacts")

# -- documented constants (units of the v2 feature scales; never tuned on holdouts)
L0 = 25                    # initial seed length, 1 s
MAX_SEG = 300              # max segment length, 12 s
PENALTY = 0.30             # bottom-up merge stops when the level-shift gain < PENALTY
IMPACT_DILATION = 5        # forced boundaries +-5 frames around verified impacts
# Evidence columns are looked up BY NAME at fit time (robust to feature-order
# changes); this list is the fixed set the paradigm uses.
EVIDENCE_NAMES = (
    "ctrl_score_diff", "sm10_ctrl_score_diff", "sm40_ctrl_pass_diff", "ctrl_pass_diff",
    "ctrl_impact_h", "ctrl_impact_a", "ball_exit_signal", "ball_frames_unseen",
    "ball_speed", "sm40_ball_speed", "tm_walk_ratio", "n_visible",
    "sm40_tm_territory_h", "sm40_tm_pitch_control_h", "sm40_tm_drift_h",
)
# SEGMENT_FEATURES: the aggregates the segment classifier sees.
SEGMENT_TRACE_NAMES = ("ctrl_score_diff", "sm10_ctrl_score_diff", "sm40_ctrl_pass_diff",
                       "ctrl_pass_diff")
SEGMENT_EXTRA_NAMES = ("ball_exit_signal", "ball_frames_unseen", "ball_speed",
                       "sm40_ball_speed", "tm_walk_ratio", "n_visible",
                       "sm40_tm_territory_h", "sm40_tm_pitch_control_h", "sm40_tm_drift_h")

_COLS_CACHE: Dict[Tuple[str, ...], Dict[str, int]] = {}


def evidence_cols(names: List[str]) -> Dict[str, int]:
    """name -> column index for the current v2 feature order (cached per order)."""
    key = tuple(names)
    if key not in _COLS_CACHE:
        missing = [n for n in EVIDENCE_NAMES if n not in names]
        if missing:
            raise RuntimeError(f"v2 features missing: {missing} -- update EVIDENCE_NAMES")
        _COLS_CACHE[key] = {n: names.index(n) for n in EVIDENCE_NAMES}
    return _COLS_CACHE[key]

LOGISTIC_PARAMS = {
    "epochs": 300,
    "lr": 0.05,
    "weight_decay": 1e-3,
    "seed": 17,
}


def evidence_trace(f: Dict) -> np.ndarray:
    """Signed 'who has the ball' scalar per frame (NaN-safe, no GT involved)."""
    col = evidence_cols(f["names"])
    get = lambda name: f["X"][:, col[name]]  # noqa: E731
    c = np.nan_to_num(get("ctrl_score_diff"), nan=0.0)
    c10 = np.nan_to_num(get("sm10_ctrl_score_diff"), nan=0.0)
    p40 = np.nan_to_num(get("sm40_ctrl_pass_diff"), nan=0.0)
    pd = np.nan_to_num(get("ctrl_pass_diff"), nan=0.0)
    return 0.5 * c + 0.25 * c10 + 0.15 * p40 + 0.10 * pd


def impact_boundaries(f: Dict) -> np.ndarray:
    """Frames that must start a new segment: the shared context tier `impact_loud`
    (the provided detector's OWN threshold + 5-frame dilation, not a re-derivation)."""
    return np.asarray(f["ctx"]["impact_loud"], dtype=bool)


def _smooth(e: np.ndarray, width: int = 40) -> np.ndarray:
    """Centered moving average (NaN-aware: NaNs -> 0 weighted out) for STABLE
    segment medians; boundaries are still detected on this smoothed trace."""
    kernel = np.ones(width, dtype=np.float64) / width
    pad = width // 2
    return np.convolve(np.concatenate([np.full(pad, e[0] if len(e) else 0.0),
                                       np.nan_to_num(e, nan=0.0),
                                       np.full(width - pad - 1, e[-1] if len(e) else 0.0)]),
                       kernel, mode="valid")[: len(e)]


def bottom_up_segments(e: np.ndarray, forced: np.ndarray, l0: int = L0,
                       penalty: float = PENALTY, cap: int = MAX_SEG) -> List[Tuple[int, int]]:
    """Bottom-up segmentation of ONE frame range (indices absolute into e).

    Start with L0-length seeds cut at every forced boundary, then repeatedly merge
    the least-contrasting adjacent pair (MEDIANS ON THE SMOOTHED TRACE) while the
    pair's level shift stays under PENALTY and the merge does not exceed cap
    frames. Classic bottom-up with a heap; ascending gains mean the first pair at
    or above the penalty ends the loop.
    """
    cuts = {0, len(e)}
    for idx in np.flatnonzero(forced):
        cuts.add(int(idx))
    for s in range(l0, len(e), l0):
        cuts.add(s)
    cut_list = sorted(cuts)
    e_hat = _smooth(e)

    med = [float(np.median(e_hat[cut_list[i]:cut_list[i + 1]]))
           for i in range(len(cut_list) - 1)]
    widths = [cut_list[i + 1] - cut_list[i] for i in range(len(cut_list) - 1)]

    heap = []

    def push(i: int) -> None:
        if 0 <= i < len(cut_list) - 2:
            heapq.heappush(heap, (abs(med[i] - med[i + 1]), i))

    for i in range(len(cut_list) - 2):
        push(i)

    while heap:
        g, i = heapq.heappop(heap)
        if not 0 <= i < len(med) - 1:
            continue
        cur = abs(med[i] - med[i + 1])
        if abs(cur - g) > 1e-12:
            # stale entry: re-queue with the CURRENT gain (or drop if hopeless)
            if cur < penalty:
                heapq.heappush(heap, (cur, i))
            continue
        if widths[i] + widths[i + 1] > cap:
            continue
        if cur >= penalty:
            break
        # merge i and i+1: boundary at i+1 disappears
        w_new = widths[i] + widths[i + 1]
        m_new = (med[i] * widths[i] + med[i + 1] * widths[i + 1]) / w_new
        med[i] = m_new
        med.pop(i + 1)
        widths[i] = w_new
        widths.pop(i + 1)
        cut_list.pop(i + 1)
        push(i - 1)
        push(i)

    segs = [(cut_list[i], cut_list[i + 1] - 1) for i in range(len(cut_list) - 1)]
    return segs


def segment_features(e: np.ndarray, f: Dict, segs: List[Tuple[int, int]]) -> np.ndarray:
    """Aggregate features [n_segs, d] on the SEGMENT-LEVEL scale."""
    col = evidence_cols(f["names"])
    cols = {name: np.nan_to_num(f["X"][:, col[name]], nan=0.0)
            for name in SEGMENT_TRACE_NAMES + SEGMENT_EXTRA_NAMES}
    dur_s = []
    rows = []
    for a, b in segs:
        sl = slice(a, b + 1)
        feats = []
        for name in SEGMENT_TRACE_NAMES:
            v = cols[name][sl]
            feats.extend([np.median(v), np.mean(v), np.std(v)])
        feats.append((b - a + 1) / 25.0)                     # duration seconds
        for name in SEGMENT_EXTRA_NAMES:
            v = cols[name][sl]
            feats.extend([np.median(v), np.mean(v)])
        rows.append(feats)
        dur_s.append((b - a + 1) / 25.0)
    return np.asarray(rows, dtype=np.float32)


def majority_label(labels: np.ndarray, segs: List[Tuple[int, int]]) -> np.ndarray:
    """Majority clean-GT code per segment (-1 when the segment has no graded frames)."""
    out = np.empty(len(segs), dtype=np.int64)
    for k, (a, b) in enumerate(segs):
        c = labels[a:b + 1]
        c = c[c >= 0]
        if c.size == 0:
            out[k] = -1
        else:
            out[k] = int(np.bincount(c, minlength=3).argmax())
    return out


def standardize(fit_files_rows: List[np.ndarray], apply_to: List[np.ndarray]):
    """Train-fold mean/std standardization (alpha scheme of the segment matrices)."""
    allX = np.concatenate(fit_files_rows, 0)
    mu = allX.mean(0)
    sd = allX.std(0)
    sd[sd < 1e-6] = 1.0
    return ([(r - mu) / sd for r in apply_to], mu, sd)


def fit_logistic(Xs: List[np.ndarray], ys: List[np.ndarray], device: str = "cpu") -> Dict:
    """Balanced multinomial logistic on SEGMENT rows (torch, CPU is plenty)."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(LOGISTIC_PARAMS["seed"])
    labels_avail = [y for y in ys if y is not None]
    X = np.concatenate([x for x, y in zip(Xs, ys) if y is not None], 0)
    y = np.concatenate(labels_avail, 0)
    keep = y >= 0
    X, y = X[keep], y[keep]
    n = len(y)
    counts = {c: max(1, int((y == c).sum())) for c in (0, 1, 2)}
    cw = np.array([n / (3.0 * counts[c]) for c in (0, 1, 2)], dtype=np.float32)
    cw_t = torch.from_numpy(cw)
    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y).to(dev)
    Wlin = torch.zeros((3, X.shape[1]), device=dev, requires_grad=True)
    blin = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.AdamW([Wlin, blin], lr=LOGISTIC_PARAMS["lr"],
                            weight_decay=LOGISTIC_PARAMS["weight_decay"])
    rng = np.random.default_rng(LOGISTIC_PARAMS["seed"])
    for _ep in range(LOGISTIC_PARAMS["epochs"]):
        perm = rng.permutation(n)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(Xt[perm] @ Wlin.T + blin, yt[perm], weight=cw_t)
        loss.backward()
        opt.step()
    return {"W": Wlin.detach().cpu().numpy().tolist(),
            "b": blin.detach().cpu().numpy().tolist(),
            "class_weights": cw.tolist()}


def load_final(path: Optional[str] = None) -> Optional[Dict]:
    p = path or os.path.join(ARTIFACT_DIR, "seg_final.json")
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_final(snap: Dict, path: Optional[str] = None) -> str:
    p = path or os.path.join(ARTIFACT_DIR, "seg_final.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as handle:
        json.dump(snap, handle, indent=2)
    return p


def fit_fold(train_files: List[Dict], eval_files: Optional[List[Dict]] = None,
             verbose: bool = True) -> Tuple:
    """Segment + class-weighted-logistic fit over the TRAIN files only."""
    t0 = time.time()
    # integrity: name-based lookups enforce the v2 feature order (see evidence_cols)
    evidence_cols(train_files[0]["names"])
    seg_tables, seg_labels = [], []
    for f in train_files:
        e = evidence_trace(f)
        forced = impact_boundaries(f)
        blocks = [(int(r[1]), int(r[2])) for r in np.asarray(f["blocks"])]
        for a, b in blocks:
            sub_e = e[a:b + 1]
            sub_f = impact_boundaries(f)[a:b + 1]
            segs = bottom_up_segments(sub_e, sub_f)
            segs = [(a + p, a + q) for p, q in segs]
            seg_tables.append(segment_features(e, f, segs))
            seg_labels.append(majority_label(f["labels"], segs))
    Xs, mu, sd = standardize(seg_tables, seg_tables)
    weights = fit_logistic(Xs, seg_labels)
    model = {"W": weights["W"], "b": weights["b"], "mu": mu.tolist(), "sd": sd.tolist(),
             "class_weights": weights["class_weights"]}
    if verbose:
        print(f"  seg fit: {sum(x.shape[0] for x in seg_tables)} segments, "
              f"weights {time.time() - t0:.1f}s")
    return model


def predict_with(model: Dict, f: Dict) -> np.ndarray:
    """Frame posterior [n, 3] from segment posteriors over the file's frames."""
    import torch
    import torch.nn.functional as F

    e = evidence_trace(f)
    forced = impact_boundaries(f)
    blocks = [(int(r[1]), int(r[2])) for r in np.asarray(f["blocks"])]
    n = f["X"].shape[0]
    P = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
    Wt = torch.from_numpy(np.asarray(model["W"], dtype=np.float32))
    bt = torch.from_numpy(np.asarray(model["b"], dtype=np.float32))
    mu = np.asarray(model["mu"], dtype=np.float32)
    sd = np.asarray(model["sd"], dtype=np.float32)
    sd[sd < 1e-6] = 1.0
    with torch.no_grad():
        for a, b in blocks:
            sub_e = e[a:b + 1]
            sub_f = forced[a:b + 1]
            segs = bottom_up_segments(sub_e, sub_f)
            segs_abs = [(a + p, a + q) for p, q in segs]
            Zs = segment_features(e, f, segs_abs)
            Zs = (Zs - mu) / sd
            logits = torch.from_numpy(Zs) @ Wt.T + bt
            probs = F.softmax(logits, dim=1).numpy().astype(np.float32)
            for k, (pa, pb) in enumerate(segs_abs):
                P[pa:pb + 1] = probs[k]
    return P


def make_fit_predict():
    """Adapter for models.modules.training_protocol.run_nested_loo (file-dict contract)."""

    def fit_predict(train_files: List[Dict], stop_files: List[Dict]) -> Tuple:
        model = fit_fold(train_files, stop_files)

        def predict_fn(file_dict: Dict) -> np.ndarray:
            return predict_with(model, file_dict)

        return predict_fn, {"best_iteration": None, "model": model}

    return fit_predict


def predict_final(f: Dict) -> np.ndarray:
    """Competition-time posterior from the saved final JSON artifact."""
    snap = load_final()
    if snap is None:
        raise FileNotFoundError(f"no final artifact: {ARTIFACT_DIR}/seg_final.json "
                                f"(run --mode final first)")
    return predict_with(snap, f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("loo", "final"), default="final")
    parser.add_argument("--fold", type=int, default=None, help="held-out fold for loo mode (0..9)")
    args = parser.parse_args()

    files = load_all_files(clean_and_dirty=True)
    print(f"cache: {len(files)} files, feature-dim={len(files[0]['names'])}")
    if args.mode == "final":
        t0 = time.time()
        model = fit_fold(files)
        path = save_final({"mode": "final", "model_tag": MODEL_TAG,
                           "n_files": len(files), **model})
        print(f"final artifact saved -> {path} ({time.time() - t0:.1f}s)")
    else:
        fold = args.fold if args.fold is not None else -1
        if not 0 <= fold < len(EXTERNAL_IDS):
            parser.error("--mode loo needs --fold in 0..9")
        train_files = [f for f in files if fold_of(f) != fold]
        eval_files = [f for f in files if fold_of(f) == fold]
        print(f"fold {fold} ({EXTERNAL_IDS[fold]}): train on {len(train_files)}, "
              f"hold out {len(eval_files)}")
        model = fit_fold(train_files, eval_files)
        snapshot = {"mode": "loo", "fold": fold, "held_out": EXTERNAL_IDS[fold], **model}
        out_path = os.path.join(ARTIFACT_DIR, f"seg_loo_fold{fold:02d}.json")
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2)
        print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
