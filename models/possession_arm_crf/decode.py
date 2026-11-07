"""Duration-state CRF core for possession_arm_crf (clean rewrite).

Chain model (the family's paradigm). States are EXPANDED with a duration
regime: (class c in {H, A, D}, regime k in 0..K-1). A spell picks its
duration regime at START -- the log-spaced bucket of its length (K=24
buckets, 1 .. 3022 frames; spells longer than the last edge clamp into the
last regime) -- and keeps it for its whole life, ending each frame
independently with the per-frame hazard h[c, k]. The chain is therefore an
exact mixture-of-geometrics duration model: WITHIN a regime the age
dependence is collapsed (documented approximation -- per-frame state
resolution was traded for a 3 x K = 72-state inference that stays trivial
in memory), while ACROSS regimes the spell-length distribution is
reproduced.

Inference: log-domain forward-backward over the 72 states, run SEPARATELY
inside each half block (no spell crosses the half-time break); output is
the per-frame CLASS marginals [n, 3] handed to the shared Viterbi.

Learning (per fold, TRAIN files only, clean copies): per-frame hazards h
per regime (spell ends / spell frames, Laplace-smoothed, capped at
HAZARD_CAP), the start-regime distribution sb, the spell-to-spell class
transition A, and the first-spell prior pi. Unlabeled (-1) GT frames
terminate a spell in GT parsing and take no side in the chain.
"""

from typing import Dict, List, Tuple

import numpy as np

BUCKET_EDGES = np.asarray([1, 3, 5, 8, 12, 17, 24, 33, 45, 60, 80, 106, 140,
                           185, 245, 324, 428, 566, 748, 989, 1308, 1729,
                           2286, 3022], dtype=np.float64)   # spell lengths, frames (25 fps)
K = int(len(BUCKET_EDGES))        # 24 regimes
EPS_HAZARD = 1e-3                 # Laplace floor for the hazard estimate
HAZARD_CAP = 0.80                 # a spell never ends with more than this probability
NEG_INF = -1e18


def _lse(a: np.ndarray, axis: int) -> np.ndarray:
    """Log-sum-exp along one axis (inputs must be finite; NEG_INF entries ok)."""
    m = np.max(a, axis=axis)
    return m + np.log(np.sum(np.exp(a - m[..., None]), axis=axis))


def _row_norm(row: np.ndarray) -> np.ndarray:
    """Shift a (3, K) log-domain matrix so its total logmass is 0 (stability)."""
    return row - float(_lse(row.reshape(-1), 0))


def bucket_index(lengths: np.ndarray) -> np.ndarray:
    """Spell length(s) in frames -> regime index (long tail clamps to K-1)."""
    return np.clip(np.searchsorted(BUCKET_EDGES, lengths, side="right") - 1, 0, K - 1)


def spells_from_labels(labels: np.ndarray) -> List[Tuple[int, int]]:
    """Graded runs of the label stream as (class, length).

    A frame with label < 0 (ungraded) terminates (and does not belong to) any
    spell: the duration model sees only frames the GT actually grades.
    """
    spells: List[Tuple[int, int]] = []
    c_run, n_run = -1, 0
    for c in np.asarray(labels, dtype=np.int64):
        c = int(c)
        if c < 0:
            if n_run:
                spells.append((c_run, n_run))
            c_run, n_run = -1, 0
        elif c == c_run:
            n_run += 1
        else:
            if n_run:
                spells.append((c_run, n_run))
            c_run, n_run = c, 1
    if n_run:
        spells.append((c_run, n_run))
    return spells


def learn_duration_model(files: List[Dict]) -> Dict:
    """h [3,K], sb [3,K], A [3,3], pi [3] from the CLEAN copies' GT spells only.

    Dirty copies carry identical labels, so they are skipped to avoid
    double-counting spells. Empty regimes inherit their class's mean hazard so
    the chain never hits a degenerate value; hazards are Laplace-smoothed and
    capped; sb / A / pi get small Dirichlet floors.
    """
    h_ends = np.zeros((3, K), dtype=np.float64)   # spells ASSIGNED regime k
    h_frames = np.zeros((3, K), dtype=np.float64)  # their total frames
    sb_counts = np.zeros((3, K), dtype=np.float64)
    pi_counts = np.zeros(3, dtype=np.float64)
    A_counts = np.zeros((3, 3), dtype=np.float64)
    n_spells, n_transitions = 0, 0

    for f in files:
        if bool(f.get("dirty", False)):
            continue
        spells = spells_from_labels(f["labels"])
        if not spells:
            continue
        n_spells += len(spells)
        pi_counts[spells[0][0]] += 1.0
        for c, length in spells:
            k = int(bucket_index(np.asarray([length]))[0])
            sb_counts[c, k] += 1.0
            h_frames[c, k] += float(length)
            h_ends[c, k] += 1.0
        for (c0, _), (c1, _) in zip(spells, spells[1:]):
            A_counts[c0, c1] += 1.0
            n_transitions += 1
    if n_spells == 0:
        raise RuntimeError("duration model: no graded spells in the training files")

    h = np.empty((3, K), dtype=np.float64)
    for c in range(3):
        known = h_frames[c] > 0
        fallback = float((h_ends[c][known] / h_frames[c][known]).mean()) if known.any() else 0.02
        for k in range(K):
            if h_frames[c, k] > 0:
                h[c, k] = (h_ends[c, k] + EPS_HAZARD) / (h_frames[c, k] + 2.0 * EPS_HAZARD)
            else:
                h[c, k] = fallback
        h[c] = np.clip(h[c], 1.0 / float(BUCKET_EDGES[-1]), HAZARD_CAP)
    sb = (sb_counts + 0.25) / (sb_counts.sum(axis=1, keepdims=True) + 0.25 * K)
    A = (A_counts + 0.5) / (A_counts.sum(axis=1, keepdims=True) + 1.5)
    pi = (pi_counts + 0.5) / (pi_counts.sum() + 1.5)

    return {
        "hazard": h.tolist(), "sb": sb.tolist(), "A": A.tolist(), "pi": pi.tolist(),
        "bucket_edges": BUCKET_EDGES.tolist(), "k": K,
        "eps_hazard": EPS_HAZARD, "hazard_cap": HAZARD_CAP,
        "n_spells": int(n_spells), "n_transitions": int(n_transitions),
    }


def _forward_pass(e: np.ndarray, log_h: np.ndarray, log_surv: np.ndarray,
                  log_sb: np.ndarray, log_A: np.ndarray, log_pi: np.ndarray) -> np.ndarray:
    """alpha over frames: [n, 3, K] float32, each row log-normalized."""
    n = e.shape[0]
    alpha_all = np.zeros((n, 3, K), dtype=np.float32)
    row = log_pi[:, None] + log_sb + e[0][:, None]
    row = _row_norm(row)
    alpha_all[0] = row
    for t in range(1, n):
        exit_mass = _lse(row + log_h, axis=1)                     # [3] per c1
        start_mass = _lse(log_A + exit_mass[:, None], axis=0)     # [3] per new class
        cont = row + log_surv                                     # spell continues
        new = start_mass[:, None] + log_sb                        # spell restarts
        row = _row_norm(np.logaddexp(cont, new) + e[t][:, None])
        alpha_all[t] = row
    return alpha_all


def _backward_pass(e: np.ndarray, log_h: np.ndarray, log_surv: np.ndarray,
                   log_sb: np.ndarray, log_A: np.ndarray) -> np.ndarray:
    """beta over frames: [n, 3, K] float32, each row log-normalized."""
    n = e.shape[0]
    beta_all = np.zeros((n, 3, K), dtype=np.float32)
    brow = np.zeros((3, K), dtype=np.float64)
    for t in range(n - 2, -1, -1):
        start_pot = _lse(log_sb + e[t + 1][:, None] + brow, axis=1)   # [3] new spell at t+1
        exit_pot = _lse(log_A + start_pot[None, :], axis=1)           # [3] then switch class
        brow_prev = np.logaddexp(log_h + exit_pot[:, None],
                                 log_surv + e[t + 1][:, None] + brow)
        brow_prev = _row_norm(brow_prev)
        beta_all[t] = brow_prev
        brow = brow_prev
    return beta_all


def forward_backward_marginals(log_emit: np.ndarray, model: Dict) -> np.ndarray:
    """Class marginals P(class of frame t's spell) from the expanded chain.

    log_emit: [n, 3] per-frame class log-likelihoods (log-softmax of the
    emission MLP logits; unseen frames get their features' softmax). Per-frame
    row normalizations of alpha/beta cancel inside the per-frame posterior, so
    the returned marginals are exact for the normalized chain.
    """
    e = np.asarray(log_emit, dtype=np.float64)
    if e.ndim != 2 or e.shape[1] != 3:
        raise ValueError(f"log_emit must be [n, 3], got {e.shape}")
    h = np.asarray(model["hazard"], dtype=np.float64)
    sb = np.asarray(model["sb"], dtype=np.float64)
    A = np.asarray(model["A"], dtype=np.float64)
    pi = np.asarray(model["pi"], dtype=np.float64)
    log_h = np.log(np.clip(h, 1e-9, 1.0))
    log_surv = np.log(np.clip(1.0 - np.clip(h, 0.0, 0.999999), 1e-9, 1.0))
    log_sb = np.log(np.clip(sb, 1e-12, None))
    log_A = np.log(np.clip(A, 1e-9, None))
    log_pi = np.log(np.clip(pi, 1e-12, None))

    alpha_all = _forward_pass(e, log_h, log_surv, log_sb, log_A, log_pi)
    beta_all = _backward_pass(e, log_h, log_surv, log_sb, log_A)
    gamma = np.exp(alpha_all.astype(np.float64) + beta_all.astype(np.float64))
    P = gamma.sum(axis=2)                                   # [n, 3]
    P /= np.maximum(P.sum(axis=1, keepdims=True), 1e-30)
    return P.astype(np.float32)


def spell_marginals(log_emit: np.ndarray, model: Dict, blocks) -> np.ndarray:
    """Per-half-block forward-backward over the whole timeline -> [n, 3] float32.

    Blocks are (half, start, END) with END inclusive (half_blocks convention);
    the chain resets at each block start, so no spell crosses the break.
    Frames outside any block (none on the real data) get a flat 1/3 row.
    """
    n = log_emit.shape[0]
    P = np.full((n, 3), np.nan, dtype=np.float32)
    for r in (np.asarray(blocks) if len(blocks) else []):
        a, b = int(r[1]), int(r[2])
        b_in = min(b, n - 1)
        if b_in < a:
            continue
        P[a:b_in + 1] = forward_backward_marginals(log_emit[a:b_in + 1], model)
    miss = np.isnan(P).all(axis=1)
    if miss.any():
        P[miss] = 1.0 / 3.0
    return P
