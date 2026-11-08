"""Exact Duration-Constrained Viterbi Dynamic-Programming Sequence Decoder.

Finds the globally optimal possession state sequence y_0..y_{T-1} in {HOME=0, AWAY=1, DEAD=2}
minimizing total per-frame negative log-likelihood plus state-transition self-information:

    Cost(y | x) = sum_{t=0}^{T-1} C_frame(t, y_t) + sum_{t=1}^{T-1} C_trans(t, y_{t-1}, y_t)

subject to:
  - Minimum physical segment durations for controlled open-play turnovers (`min_possession_frames`).
  - Pre-stoppage semi-Markov dwell guard (`min_pre_stop_frames` and `pen_pre_stop`), preventing
    cheap 1*tc pre-stoppage flip hallucinations right before dead balls while preserving rapid
    open-play turnover detection.
  - Minimum interior dead-ball stoppage durations (`min_stoppage_frames`).

Solved exactly in O(T) time and O(T) memory via an expanded minimum-dwell Markov chain.
"""

from typing import Optional
import numpy as np

from possession_model.signal_ops import AWAY, DEAD, HOME


def decode_possession_sequence(
    frame_costs: np.ndarray,
    turnover_cost: np.ndarray,
    stoppage_entry_cost: np.ndarray,
    restart_cost: np.ndarray,
    min_possession_frames: int = 20,
    min_stoppage_frames: int = 75,
    min_pre_stop_frames: int = 55,
    pen_pre_stop: float = 6.0,
    initial_cost_offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Find the exact minimum-cost state sequence y_0..y_{T-1} in {HOME=0, AWAY=1, DEAD=2}."""
    T = frame_costs.shape[0]
    if T == 0:
        return np.empty(0, dtype=np.int8)

    lh = max(1, int(min_possession_frames))
    lp = max(lh, int(min_pre_stop_frames))
    ld = max(1, int(min_stoppage_frames))

    off_h = 0
    off_a = lp
    off_d = 2 * lp
    m_h = off_h + lp - 1
    m_a = off_a + lp - 1
    m_d = off_d + ld - 1
    K = 2 * lp + ld

    INF = 1e18
    dp = np.full(K, INF, dtype=np.float64)
    init = (
        np.zeros(3, dtype=np.float64)
        if initial_cost_offset is None
        else np.asarray(initial_cost_offset, dtype=np.float64)
    )
    dp[off_h] = frame_costs[0, HOME] + init[HOME]
    dp[off_a] = frame_costs[0, AWAY] + init[AWAY]
    # Dead ball at t = 0 starts mature to allow immediate kickoff restart
    dp[m_d] = frame_costs[0, DEAD] + init[DEAD]

    bp_entry_h = np.zeros(T, dtype=np.int16)
    bp_entry_a = np.zeros(T, dtype=np.int16)
    bp_entry_d = np.zeros(T, dtype=np.int16)
    bp_mature = np.zeros((T, 3), dtype=np.uint8)

    c_h = frame_costs[:, HOME]
    c_a = frame_costs[:, AWAY]
    c_d = frame_costs[:, DEAD]
    dp_next = np.empty(K, dtype=np.float64)

    s_elig_h = off_h + lh - 1
    s_elig_a = off_a + lh - 1

    for t in range(1, T):
        ch_t = c_h[t]
        ca_t = c_a[t]
        cd_t = c_d[t]
        pen_turn = turnover_cost[t]
        pen_stop = stoppage_entry_cost[t]
        pen_rest = restart_cost[t]

        # 1. Entry into H (from any eligible A dwell state [s_elig_a .. m_a] or mature D)
        min_a_rel = int(np.argmin(dp[s_elig_a : m_a + 1]))
        val_best_a = dp[s_elig_a + min_a_rel] + pen_turn

        cand_d_to_h = dp[m_d] + pen_rest
        if val_best_a <= cand_d_to_h:
            dp_next[off_h] = val_best_a + ch_t
            bp_entry_h[t] = s_elig_a + min_a_rel
        else:
            dp_next[off_h] = cand_d_to_h + ch_t
            bp_entry_h[t] = m_d

        # 2. Entry into A (from any eligible H dwell state [s_elig_h .. m_h] or mature D)
        min_h_rel = int(np.argmin(dp[s_elig_h : m_h + 1]))
        val_best_h = dp[s_elig_h + min_h_rel] + pen_turn

        cand_d_to_a = dp[m_d] + pen_rest
        if val_best_h <= cand_d_to_a:
            dp_next[off_a] = val_best_h + ca_t
            bp_entry_a[t] = s_elig_h + min_h_rel
        else:
            dp_next[off_a] = cand_d_to_a + ca_t
            bp_entry_a[t] = m_d

        # 3. Entry into D (mature H/A pays pen_stop; semi-mature dwell pays pen_stop + pen_pre_stop)
        val_mat_hd = dp[m_h] + pen_stop
        if m_h > s_elig_h:
            min_semi_rel_h = int(np.argmin(dp[s_elig_h:m_h]))
            val_semi_hd = dp[s_elig_h + min_semi_rel_h] + pen_stop + pen_pre_stop
            if val_mat_hd <= val_semi_hd:
                val_best_hd = val_mat_hd
                best_hd_idx = m_h
            else:
                val_best_hd = val_semi_hd
                best_hd_idx = s_elig_h + min_semi_rel_h
        else:
            val_best_hd = val_mat_hd
            best_hd_idx = m_h

        val_mat_ad = dp[m_a] + pen_stop
        if m_a > s_elig_a:
            min_semi_rel_a = int(np.argmin(dp[s_elig_a:m_a]))
            val_semi_ad = dp[s_elig_a + min_semi_rel_a] + pen_stop + pen_pre_stop
            if val_mat_ad <= val_semi_ad:
                val_best_ad = val_mat_ad
                best_ad_idx = m_a
            else:
                val_best_ad = val_semi_ad
                best_ad_idx = s_elig_a + min_semi_rel_a
        else:
            val_best_ad = val_mat_ad
            best_ad_idx = m_a

        if val_best_hd <= val_best_ad:
            dp_next[off_d] = val_best_hd + cd_t
            bp_entry_d[t] = best_hd_idx
        else:
            dp_next[off_d] = val_best_ad + cd_t
            bp_entry_d[t] = best_ad_idx

        # 4. Interior shifts along minimum-duration dwell chains
        dp_next[off_h + 1 : m_h] = dp[off_h : m_h - 1] + ch_t
        dp_next[off_a + 1 : m_a] = dp[off_a : m_a - 1] + ca_t
        dp_next[off_d + 1 : m_d] = dp[off_d : m_d - 1] + cd_t

        # 5. Mature absorbing states
        prev_h = dp[m_h - 1]
        val_mh = dp[m_h]
        if val_mh <= prev_h:
            dp_next[m_h] = val_mh + ch_t
            bp_mature[t, 0] = 0
        else:
            dp_next[m_h] = prev_h + ch_t
            bp_mature[t, 0] = 1

        prev_a = dp[m_a - 1]
        val_ma = dp[m_a]
        if val_ma <= prev_a:
            dp_next[m_a] = val_ma + ca_t
            bp_mature[t, 1] = 0
        else:
            dp_next[m_a] = prev_a + ca_t
            bp_mature[t, 1] = 1

        prev_d = dp[m_d - 1]
        val_md = dp[m_d]
        if val_md <= prev_d:
            dp_next[m_d] = val_md + cd_t
            bp_mature[t, 2] = 0
        else:
            dp_next[m_d] = prev_d + cd_t
            bp_mature[t, 2] = 1

        dp, dp_next = dp_next, dp

    # Backtrack optimal sequence
    curr = int(np.argmin(dp))
    out = np.empty(T, dtype=np.int8)

    for t in range(T - 1, 0, -1):
        if curr < off_a:
            out[t] = HOME
            if curr == off_h:
                curr = int(bp_entry_h[t])
            elif curr == m_h:
                curr = m_h if bp_mature[t, 0] == 0 else m_h - 1
            else:
                curr = curr - 1
        elif curr < off_d:
            out[t] = AWAY
            if curr == off_a:
                curr = int(bp_entry_a[t])
            elif curr == m_a:
                curr = m_a if bp_mature[t, 1] == 0 else m_a - 1
            else:
                curr = curr - 1
        else:
            out[t] = DEAD
            if curr == off_d:
                curr = int(bp_entry_d[t])
            elif curr == m_d:
                curr = m_d if bp_mature[t, 2] == 0 else m_d - 1
            else:
                curr = curr - 1

    out[0] = HOME if curr < off_a else (AWAY if curr < off_d else DEAD)
    return out
