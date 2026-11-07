"""Locate the half-time break of a stitched two-half timeline from tracking only.

External matches run continuously across the break (frame ids do not reset), so the
boundary must come from the movement data. The physical signature that cannot be
confused with anything else is **which goal each team defends**: in one half the
team-A players appear in the left goal zone and team-B in the right; in the next half
they swap. Players visit their OWN goal zone far more (goal kicks, defensive presses,
build-ups) and their opponents' zone rarely. Cumulative goal-zone counts therefore
grow differentially up to the true split and reverse afterwards, and they do so for
BOTH teams simultaneously. Any mid-half restart (a kick-off after a goal keeps both
teams in their own halves, like a half-time kick-off) does not move the defended goal,
so it cannot manufacture the signature.

The detector maximises the summed two-team flip score over the timeline in an exact
O(n) scan over cumulative sums, under three guards: each side of the split must hold
at least 10 minutes of footage, the flip score must reach a clear positive level, and
each team's flip must point in the opposite direction (they swap ends, so the two
flips are always antiparallel). Otherwise -- conservative fallback -- it returns None
and the match decodes as a single block, which only degrades two direction features
instead of inventing a boundary that would corrupt the decoded windows.
"""

from typing import Optional, Tuple

import numpy as np

# Goal zone: within 9 m of the goal line, y within a 25 m band around the centre-line.
# These are pitch geometry, not tuned scene parameters.
GOAL_ZONE_X = 9.0
GOAL_ZONE_Y_HALF = 12.5
MIN_HALF_S = 10 * 60.0        # each side of the split must hold at least 10 minutes
MIN_FLIP_SCORE = 200.0        # player-frames of net flipped goal-zone presence


def detect_half_split(player_xy: np.ndarray, slot_teams: np.ndarray,
                      pitch: Tuple[float, float], fps: float = 25.0) -> Optional[int]:
    """Return the absolute frame index where the second half starts, or None.

    Teams are identified ONLY by `slot_teams` (0 = team-A slots, 1 = team-B slots):
    which goal each defends first is inferred, never assumed.
    """
    n = len(player_xy)

    def cum_delta(side: int) -> np.ndarray:
        x = player_xy[:, slot_teams == side, 0]
        y = player_xy[:, slot_teams == side, 1]
        vis = np.isfinite(x) & np.isfinite(y)
        left = (x < GOAL_ZONE_X) & (np.abs(y - 0.5 * pitch[1]) <= GOAL_ZONE_Y_HALF) & vis
        right = (x > pitch[0] - GOAL_ZONE_X) & (np.abs(y - 0.5 * pitch[1]) <= GOAL_ZONE_Y_HALF) & vis
        dt = left.astype(np.float64) - right.astype(np.float64)
        return np.cumsum(dt.sum(axis=1))

    cum_a = cum_delta(0)
    cum_b = cum_delta(1)
    d_a_tot = float(cum_a[-1])
    d_b_tot = float(cum_b[-1])

    # flip_A(s) = pre_delta - post_delta, with post = total - pre  =>  2*cum_a[s] - d_a_tot.
    # The away flip mirrors the home one, so the aligned joint score is
    #   score(s) = (2*cum_a[s] - d_a_tot) - (2*cum_b[s] - d_b_tot)
    # and at the TRUE split both terms peak together, while at any mid-half frame they
    # cancel. The reversed orientation yields the negative of this score, hence |best|.
    frame = np.arange(n, dtype=np.int64)
    score = (2.0 * cum_a - d_a_tot) - (2.0 * cum_b - d_b_tot)
    lo = int(MIN_HALF_S * fps)
    valid = (frame >= lo) & (frame <= n - lo)
    magnitude = np.abs(score)
    magnitude[~valid] = -np.inf

    best = int(np.argmax(magnitude))
    if not np.isfinite(magnitude[best]) or magnitude[best] <= MIN_FLIP_SCORE:
        return None

    # Flip guard: the two teams' flips must be strongly antiparallel (they swap ends),
    # each carrying its own absolute level.
    flip_a = 2.0 * float(cum_a[best]) - d_a_tot
    flip_b = 2.0 * float(cum_b[best]) - d_b_tot
    if (abs(flip_a) < MIN_FLIP_SCORE or abs(flip_b) < MIN_FLIP_SCORE
            or np.sign(flip_a) == np.sign(flip_b)):
        return None
    return best


def half_blocks_from_tracking(n: int, split_frame: Optional[int]):
    """Assemble [(1, s, e), (2, s, e)] blocks for a stitched two-half timeline."""
    if split_frame is None:
        return [(1, 0, n - 1)]
    return [(1, 0, split_frame - 1), (2, split_frame, n - 1)]
