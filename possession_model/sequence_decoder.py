"""Duration-Constrained Viterbi Dynamic-Programming Sequence Decoder.

Finds the globally optimal possession state sequence y_0..y_{T-1} in {HOME=0, AWAY=1, DEAD=2}
minimizing total per-frame negative log-likelihood plus context-dependent state-transition
penalties, subject to minimum physical segment durations for controlled possessions and
dead-ball stoppages:

    Cost(y | x) = sum_t C_frame(t, y_t) + sum_t C_trans(t, y_{t-1}, y_t)

Solved exactly in O(T) time via an expanded-state Viterbi dynamic program with compact
uint8 backpointers.
"""

import numpy as np

HOME = 0
AWAY = 1
DEAD = 2


def decode_possession_sequence(
    frame_costs: np.ndarray,
    turnover_cost: np.ndarray,
    stoppage_entry_cost: np.ndarray,
    restart_cost: np.ndarray,
    min_possession_frames: int = 35,
    min_stoppage_frames: int = 85,
    initial_cost_offset: np.ndarray = None,
) -> np.ndarray:
    """Find the exact minimum-cost state sequence y_0..y_{T-1} in {0 (H), 1 (A), 2 (D)}.

    Parameters
    ----------
    frame_costs : (T, 3) float64
        Per-frame state costs C_frame(t, s) for s in {H=0, A=1, D=2}. Lower cost = higher likelihood.
    turnover_cost : (T,) float64
        Transition penalty for direct open-play turnover H -> A or A -> H at frame t.
    stoppage_entry_cost : (T,) float64
        Transition penalty for live play -> stoppage (H/A -> D) at frame t.
    restart_cost : (T,) float64
        Transition penalty for stoppage -> play restart (D -> H/A) at frame t.
    min_possession_frames : int
        Minimum segment length (in frames) for an H or A controlled possession spell.
    min_stoppage_frames : int
        Minimum segment length (in frames) for a D dead-ball stoppage interval.
    initial_cost_offset : (3,) float64, optional
        Initial cost offset at t = 0 for states (H, A, D).

    Returns
    -------
    states : (T,) int8
        Optimal decoded state sequence in {0, 1, 2}.
    """
    T = frame_costs.shape[0]
    if T == 0:
        return np.empty(0, dtype=np.int8)

    lh = max(1, int(min_possession_frames))
    la = max(1, int(min_possession_frames))
    ld = max(1, int(min_stoppage_frames))

    # Expanded state layout of length K = lh + la + ld:
    #   H dwell chain: [0 .. lh-1]               mature index: m_h = lh - 1
    #   A dwell chain: [lh .. lh+la-1]           mature index: m_a = lh + la - 1
    #   D dwell chain: [lh+la .. lh+la+ld-1]     mature index: m_d = lh + la + ld - 1
    off_h = 0
    off_a = lh
    off_d = lh + la
    m_h = off_h + lh - 1
    m_a = off_a + la - 1
    m_d = off_d + ld - 1
    K = lh + la + ld

    INF = 1e18
    dp = np.full(K, INF, dtype=np.float64)
    init = (
        np.zeros(3, dtype=np.float64)
        if initial_cost_offset is None
        else np.asarray(initial_cost_offset, dtype=np.float64)
    )
    dp[off_h] = frame_costs[0, HOME] + init[HOME]
    dp[off_a] = frame_costs[0, AWAY] + init[AWAY]
    dp[off_d] = frame_costs[0, DEAD] + init[DEAD]

    # Compact backpointers for the 6 branching transitions per frame
    bp = np.zeros((T, 6), dtype=np.uint8)

    c_h = frame_costs[:, HOME]
    c_a = frame_costs[:, AWAY]
    c_d = frame_costs[:, DEAD]

    dp_next = np.empty(K, dtype=np.float64)

    for t in range(1, T):
        ch_t = c_h[t]
        ca_t = c_a[t]
        cd_t = c_d[t]
        pen_turn = turnover_cost[t]
        pen_stop = stoppage_entry_cost[t]
        pen_rest = restart_cost[t]

        val_mh = dp[m_h]
        val_ma = dp[m_a]
        val_md = dp[m_d]

        # 1. Entry into H (from mature A or mature D)
        cand_a_to_h = val_ma + pen_turn
        cand_d_to_h = val_md + pen_rest
        if cand_a_to_h <= cand_d_to_h:
            dp_next[off_h] = cand_a_to_h + ch_t
            bp[t, 0] = 0
        else:
            dp_next[off_h] = cand_d_to_h + ch_t
            bp[t, 0] = 1

        # 2. Entry into A (from mature H or mature D)
        cand_h_to_a = val_mh + pen_turn
        cand_d_to_a = val_md + pen_rest
        if cand_h_to_a <= cand_d_to_a:
            dp_next[off_a] = cand_h_to_a + ca_t
            bp[t, 1] = 0
        else:
            dp_next[off_a] = cand_d_to_a + ca_t
            bp[t, 1] = 1

        # 3. Entry into D (from mature H or mature A)
        cand_h_to_d = val_mh + pen_stop
        cand_a_to_d = val_ma + pen_stop
        if cand_h_to_d <= cand_a_to_d:
            dp_next[off_d] = cand_h_to_d + cd_t
            bp[t, 2] = 0
        else:
            dp_next[off_d] = cand_a_to_d + cd_t
            bp[t, 2] = 1

        # 4. Deterministic interior shifts along each minimum-duration chain
        if lh > 2:
            dp_next[off_h + 1 : m_h] = dp[off_h : m_h - 1] + ch_t
        if la > 2:
            dp_next[off_a + 1 : m_a] = dp[off_a : m_a - 1] + ca_t
        if ld > 2:
            dp_next[off_d + 1 : m_d] = dp[off_d : m_d - 1] + cd_t

        # 5. Mature absorbing states (stay in mature vs advance from penultimate)
        if lh > 1:
            prev_h = dp[m_h - 1]
            if val_mh <= prev_h:
                dp_next[m_h] = val_mh + ch_t
                bp[t, 3] = 0
            else:
                dp_next[m_h] = prev_h + ch_t
                bp[t, 3] = 1
        else:
            stay_h = val_mh + ch_t
            if stay_h <= dp_next[off_h]:
                dp_next[m_h] = stay_h
                bp[t, 3] = 0
            else:
                bp[t, 3] = 2

        if la > 1:
            prev_a = dp[m_a - 1]
            if val_ma <= prev_a:
                dp_next[m_a] = val_ma + ca_t
                bp[t, 4] = 0
            else:
                dp_next[m_a] = prev_a + ca_t
                bp[t, 4] = 1
        else:
            stay_a = val_ma + ca_t
            if stay_a <= dp_next[off_a]:
                dp_next[m_a] = stay_a
                bp[t, 4] = 0
            else:
                bp[t, 4] = 2

        if ld > 1:
            prev_d = dp[m_d - 1]
            if val_md <= prev_d:
                dp_next[m_d] = val_md + cd_t
                bp[t, 5] = 0
            else:
                dp_next[m_d] = prev_d + cd_t
                bp[t, 5] = 1
        else:
            stay_d = val_md + cd_t
            if stay_d <= dp_next[off_d]:
                dp_next[m_d] = stay_d
                bp[t, 5] = 0
            else:
                bp[t, 5] = 2

        dp, dp_next = dp_next, dp

    # Backtrack from the minimum-cost state at t = T - 1
    curr = int(np.argmin(dp))
    out = np.empty(T, dtype=np.int8)

    for t in range(T - 1, 0, -1):
        if curr < off_a:
            out[t] = HOME
            if lh == 1:
                if bp[t, 3] == 0:
                    curr = m_h
                else:
                    curr = m_a if bp[t, 0] == 0 else m_d
            elif curr == off_h:
                curr = m_a if bp[t, 0] == 0 else m_d
            elif curr == m_h:
                curr = m_h if bp[t, 3] == 0 else m_h - 1
            else:
                curr = curr - 1
        elif curr < off_d:
            out[t] = AWAY
            if la == 1:
                if bp[t, 4] == 0:
                    curr = m_a
                else:
                    curr = m_h if bp[t, 1] == 0 else m_d
            elif curr == off_a:
                curr = m_h if bp[t, 1] == 0 else m_d
            elif curr == m_a:
                curr = m_a if bp[t, 4] == 0 else m_a - 1
            else:
                curr = curr - 1
        else:
            out[t] = DEAD
            if ld == 1:
                if bp[t, 5] == 0:
                    curr = m_d
                else:
                    curr = m_h if bp[t, 2] == 0 else m_a
            elif curr == off_d:
                curr = m_h if bp[t, 2] == 0 else m_a
            elif curr == m_d:
                curr = m_d if bp[t, 5] == 0 else m_d - 1
            else:
                curr = curr - 1

    out[0] = HOME if curr < off_a else (AWAY if curr < off_d else DEAD)
    return out
