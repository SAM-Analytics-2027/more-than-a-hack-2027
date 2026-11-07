"""Model-agnostic training protocol (More-than-a-Hack 2027, branch AM).

Every model runs the SAME list-one-match-out protocol:
  - data: the shared v2 feature cache (20 files = 10 external matches x {clean, dirtied}),
    clean ground truth in both cases (labels always come from `*_pulite`).
  - folds: one per logical external match; clean and dirtied copies share the fold index.
  - per fold i: fit on every file except fold i and the NEXT fold (i+1) % 10 -- the next
    fold is the early-stopping set -- then predict and score every file of fold i.
  - posteriors of each held-out file are saved to
      data/_arm_cache/looP_v2_<model_tag>_<branch>_<subset>_<match_id>.npz
    so the decoder-parameter search and this loop never re-fit anything by accident.

The caller supplies one adapter function:
      fit_predict(train_files, stop_files) -> (predict_fn, info)
  where predict_fn(file_dict) -> P: [n, 3] float32 posterior of (H, A, D) for the
  file's frames (the FULL dict is passed: GBDT families need X, temporal families
  also need blocks/ctx for windowing), and info is a small dict (best_iteration
  etc.) for the per-fold log.

Files are the dicts produced by cache loading (see models/modules/cache.py):
  {match_id, dirty, X, labels, blocks, ctx, names, ...}.
"""

import math
import os
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

from models.modules.cache import loo_posterior_path  # noqa: E402
from models.modules.context import CONTEXT_KEYS  # noqa: E402
from models.modules.data_external import EXTERNAL_IDS  # noqa: E402
from models.modules.decoder_support import DecoderParams, decode_match  # noqa: E402
from models.modules.scoring import score_file  # noqa: E402

FIT_BACKENDS: Dict[str, Callable] = {}   # registry for `--backend` style dispatch


def fold_of(file: Dict) -> int:
    """Fold index of a cache file dict (clean and dirtied share it)."""
    return EXTERNAL_IDS.index(file["match_id"])


def balanced_class_weights(labels: np.ndarray) -> np.ndarray:
    """Balanced per-class sample weights from THIS training set only (alpha scheme).

    Copied from possession_arm_v1/train.py:class_weights so GBDT families weight
    the same way and the nested comparison stays consistent.
    """
    codes = labels[labels >= 0]
    n = len(codes)
    counts = {c: max(1, int((codes == c).sum())) for c in (0, 1, 2)}
    weights = np.empty(n, dtype=np.float32)
    for c, cnt in counts.items():
        mask = codes == c
        weights[mask] = (n / (3.0 * cnt))
    return weights


def stack_labeled(train_files: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate training files into one fit matrix, dropping X (ungraded) rows.

    Returns (X, y, w) where w is the balanced per-class weight array.
    """
    X = np.concatenate([f["X"] for f in train_files]).astype(np.float32)
    y = np.concatenate([f["labels"] for f in train_files]).astype(np.int32)
    keep = y >= 0                       # X (ungraded) rows never enter the fit
    X, y = X[keep].astype(np.float32, copy=False), y[keep]
    w = balanced_class_weights(y)
    return X, y, w


def run_one_fold(i: int, files: List[Dict], model_tag: str,
                 fit_predict: Callable, params: DecoderParams) -> Dict:
    """Fit fold i, decode+score its held-out files, save posteriors. Returns the result
    dict {rows, best_iteration, fit_s} with rows = (i, (match_id, dirty), scores)."""
    stop_fold = (i + 1) % 10
    train_files = [f for f in files if fold_of(f) != i and fold_of(f) != stop_fold]
    stop_files = [f for f in files if fold_of(f) == stop_fold]
    hold_files = [f for f in files if fold_of(f) == i]

    t0 = time.time()
    predict_fn, info = fit_predict(train_files, stop_files)
    fit_s = time.time() - t0

    rows, saved = [], []
    for f in hold_files:
        P = np.asarray(predict_fn(f), dtype=np.float32)
        P = P.reshape(-1, 3)
        tags = (f["match_id"], bool(f["dirty"]))
        meta_ctx = f["ctx"]
        blocks = [tuple(map(int, b)) for b in f["blocks"]]
        states = decode_match(P, meta_ctx, blocks, params)
        rows.append((i, tags, score_file(f["labels"], states)))
        p_path = loo_posterior_path(tags[0], tags[1], model_tag=model_tag)
        np.savez_compressed(p_path, P=P, labels=f["labels"],
                            blocks=np.asarray(blocks, dtype=np.int64))
        saved.append(os.path.basename(p_path))
    return {"rows": rows, "best_iteration": info.get("best_iteration"),
            "fit_s": fit_s, "saved": saved, "fold": i}


def run_nested_loo(files: List[Dict], model_tag: str, fit_predict: Callable,
                   params: DecoderParams, folds: List[int] | None = None,
                   log: Callable[[str], None] = print) -> Dict:
    """The full 10-fold protocol = possession_arm_v1's eval_external.py main loop,
    without any GPU/GBDT assumption. Returns aggregated rows for summary writing."""
    if folds is None:
        folds = list(range(10))
    all_rows: List = []
    best_iterations: List = []
    for i in folds:
        res = run_one_fold(i, files, model_tag, fit_predict, params)
        for _, tags, s in res["rows"]:
            all_rows.append((i, tags, s))
            label = f"{tags[0]}{'_d' if tags[1] else ''}"
            log(f"  fold {i} hold {label:<20}: spellF1={s['spell_f1']:.3f} "
                f"macro={s['macro_f1']:.3f} spells {s['gt_spells']}/{s['pred_spells']}")
        best_iterations.append(res["best_iteration"])
        log(f"fold {i} fit={res['fit_s']:.0f}s rounds={res['best_iteration']}")

    spell = np.array([s["spell_f1"] for _, _, s in all_rows])
    macro = np.array([s["macro_f1"] for _, _, s in all_rows])
    log(f"\n mean spellF1 = {spell.mean():.4f}  macroF1 = {macro.mean():.4f}  "
        f"(n={len(spell)})")
    return {"rows": all_rows, "best_iterations": best_iterations,
            "mean_spell_f1": float(spell.mean()), "mean_macro_f1": float(macro.mean())}


def loo_rounds_final(rounds: List[int], quantile: float = 0.8,
                     floor: int = 150) -> Tuple[int, int]:
    """Final-fit round count from the LOO best iterations (p80 default, v1 protocol).
    Returns (n_rounds, n_rounds_max) with min 300 to stay near the LOO sweet spot."""
    import math

    it = max(int(r) for r in rounds if r is not None) if rounds else floor
    n = max(int(math.floor(quantile * it)), min(300, it), 1)
    return n, max(n, it)
