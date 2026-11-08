"""Coarse relabeling of external GT to the competition annotation style (track E2).

Style gap (read-only quantification): external touch-granular GT spells
(mean 8.0 s, p50 4.0 s) vs competition GT actions (public slice mean 14.3 s,
p50 8.1 s). The external annotator splits one play into multiple short
possessions interrupted by brief dead-ball frames; the competition annotator
folds those into a single action.

Merge rule (documented, deterministic):
  two consecutive action spells of the SAME team separated ONLY by ONE D
  interruption are merged into a single spell; the intervening D frames are
  relabeled to that team's code. Opponent-touch boundaries are NEVER touched.
  Unlabeled (X) frames and half-block boundaries break every merge.

Threshold policy (LEGIT: derived from training data alone, never the target):
  pick_threshold takes the largest multiplicative jump in the sorted UNIQUE
  two-sided D-gap lengths of the external GT and returns the value just below
  the jump (the top of the short-gap cluster). The procedure is deterministic
  and only reads the external labels.

MEASURED (read-only external stats; see the E2 diagnostic run): over the 10
external matches there are 278 two-sided same-team D gaps, mean 29.0 s
(p50 25.1 s, p90 54.0 s); ONLY 5 gaps fall below the recorded valley (<= 2.2 s)
and even a permissive 5 s threshold merges ~55 of them, shifting the per-match
action count by <1%. The short-D merge is therefore a DOCUMENTED near no-op:
the external annotation style differs from the competition GT mainly in touch
granularity (p50 spell 4.0 s vs 8.1 s public slice), which no D-gap merge can
emulate. The module stays as protocol-clean tooling for any future relabel
experiment; no model in the repository currently consumes it.
"""

from typing import List, Optional, Tuple

import numpy as np

# Label codes (cache labels): H=0, A=1, D=2, unlabeled/X=-1.

TEAM_CODES = (0, 1)


def runs_of(labels: np.ndarray, start: int, end: int) -> List[Tuple[int, int, int]]:
    """Runs of identical labels in the inclusive span [start, end]:
    (label, start_exclusive, end_exclusive)."""
    out: List[Tuple[int, int, int]] = []
    i = max(0, int(start))
    stop = min(len(labels) - 1, int(end))
    n = stop + 1
    while i < n:
        j = i + 1
        code = int(labels[i])
        while j < n and int(labels[j]) == code:
            j += 1
        out.append((code, i, j))
        i = j
    return out


def two_sided_d_gaps(labels: np.ndarray, blocks) -> List[int]:
    """Lengths (frames) of D interruptions that separate two same-team
    action runs inside a half block. This is the population the merge rule
    applies to; the threshold is a recorded valley of this list."""
    gaps: List[int] = []
    for _no, a, b in blocks:
        rs = runs_of(labels, a, b)
        for k in range(1, len(rs) - 1):
            left, mid, right = rs[k - 1], rs[k], rs[k + 1]
            if mid[0] != 2:
                continue
            if left[0] in TEAM_CODES and left[0] == right[0]:
                gaps.append(mid[2] - mid[1])
    return gaps


def merge_spells(labels: np.ndarray, blocks, max_gap_frames: int) -> np.ndarray:
    """Coarse labels: same-team runs split by ONE D run of <= max_gap_frames
    merged to a single spell (cascade until fixpoint per block, deterministic)."""
    out = np.asarray(labels, dtype=np.int8).copy()
    for _no, a, b in blocks:
        while True:
            merged = False
            rs = runs_of(out, a, b)
            for k in range(1, len(rs) - 1):
                left, mid, right = rs[k - 1], rs[k], rs[k + 1]
                gap = mid[2] - mid[1]
                if (mid[0] == 2 and left[0] in TEAM_CODES and left[0] == right[0]
                        and gap <= max_gap_frames):
                    out[mid[1]:mid[2]] = np.int8(left[0])
                    merged = True
                    break
            if not merged:
                break
    return out


def gap_jump_table(gaps: List[int], min_gap_frames: int = 5) -> List[dict]:
    """Ratio jumps between consecutive sorted UNIQUE gap lengths (>= min_gap_frames),
    largest first: the candidates a recorded valley can be picked from."""
    uniq = sorted({int(g) for g in gaps if g >= min_gap_frames})
    table = []
    for k in range(len(uniq) - 1):
        table.append({"below": uniq[k], "above": uniq[k + 1],
                      "ratio": uniq[k + 1] / max(uniq[k], 1),
                      "n_at_or_below": int(sum(1 for g in gaps if g <= uniq[k])),
                      "n_above": int(sum(1 for g in gaps if g >= uniq[k + 1]))})
    table.sort(key=lambda row: -row["ratio"])
    return table


def pick_threshold(gaps: List[int], min_gap_frames: int = 5) -> Tuple[Optional[int], dict]:
    """Recorded valley: the biggest multiplicative jump in the sorted unique gap
    lengths; the threshold is the value just BELOW the jump (top of the short
    cluster). Returns (threshold_frames_or_None, explanation) deterministically."""
    table = gap_jump_table(gaps, min_gap_frames)
    if not table:
        return None, {"reason": "no two-sided same-team D gaps found"}
    top = table[0]
    threshold = int(top["below"])
    return threshold, {"rule": "largest ratio jump between consecutive unique gap lengths",
                       "below": top["below"], "above": top["above"],
                       "ratio": round(top["ratio"], 4),
                       "n_merged_at_threshold": top["n_at_or_below"],
                       "n_kept_above": top["n_above"]}


def action_stats(labels: np.ndarray, blocks) -> dict:
    """Spell stats for graded frames only: n actions, mean/p50/p90 duration,
    D share of graded frames (diagnostics for the style comparison)."""
    durs: List[float] = []
    graded = 0
    d_frames = 0
    for _no, a, b in blocks:
        rs = runs_of(labels, a, b)
        for code, s, e in rs:
            if code < 0:
                continue
            frames = e - s
            graded += frames
            if code == 2:
                d_frames += frames
            else:
                durs.append(frames)
    arr = np.asarray(durs, dtype=np.float64)
    return {"n_actions": int(len(durs)),
            "mean_s": float(arr.mean() / 25.0) if len(durs) else float("nan"),
            "p50_s": float(np.median(arr) / 25.0) if len(durs) else float("nan"),
            "p90_s": float(np.percentile(arr, 90) / 25.0) if len(durs) else float("nan"),
            "d_share": float(d_frames / graded) if graded else float("nan")}
