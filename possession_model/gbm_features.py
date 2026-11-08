"""Declarative Multi-Scale Feature Matrices for Cascaded Possession LightGBM Models.

Three clean, leak-free feature stages:
  1. `extract_dead_features`: Domain-invariant 22-player multi-scale kinematics & boundary exits
     for Dead-Ball probability P(D | x_t) (free of DFL frozen-ball / visibility shortcuts).
  2. `extract_team_features`: Stoppage-barrier-clipped, exact H <-> A equivariant multi-scale
     ball-interaction & collective formation features for P(H | alive, x_t).
  3. `extract_transition_features`: Team-symmetric kinematic & posterior-derivative features
     for live turnover probability P(turnover | alive, x_t).
"""

from typing import Dict, List, Tuple
import numpy as np

from possession_model.signal_ops import (
    barrier_box_mean_blocks,
    barrier_event_timers,
    box_mean_blocks,
)


def extract_dead_features(
    ball_data: Dict[str, np.ndarray],
    team_dynamics: Dict[str, np.ndarray],
    half_blocks: List[Tuple[int, int, int]],
    fps: float = 25.0,
) -> Dict[str, object]:
    """Build domain-invariant Dead-Ball feature matrix X_dead (N, D_dead) float32."""
    n = len(ball_data["obs_mask"])
    oob = ball_data["out_of_bounds"].astype(np.float32)
    exit_sig = ball_data["boundary_exit_signal"].astype(np.float32)

    rms = team_dynamics["rms_player_speed"]
    p85 = team_dynamics["p85_player_speed"]
    walk = team_dynamics["walking_ratio"]
    run = team_dynamics["running_ratio"]
    cent = team_dynamics["centroid_speed"]
    align = team_dynamics["velocity_alignment"]
    tot_width = (team_dynamics["std_y_h"] + team_dynamics["std_y_a"]).astype(np.float32)
    tot_len = (team_dynamics["std_x_h"] + team_dynamics["std_x_a"]).astype(np.float32)
    in_box = team_dynamics["max_in_penalty_box"]
    touch_cnt = team_dynamics["near_touchline"]
    half_sep = team_dynamics["half_separation_frac"]
    n_vis = team_dynamics["n_vis_tot"]

    no_bar = np.zeros(n, dtype=bool)
    oob_bool = ball_data["out_of_bounds"] | (exit_sig > 0.5)
    dt_prev_oob, dt_next_oob = barrier_event_timers(oob_bool, no_bar, half_blocks, fps=fps, horizon_s=25.0)

    cols: List[np.ndarray] = []
    names: List[str] = []

    def _add(name: str, arr: np.ndarray) -> None:
        names.append(name)
        cols.append(np.asarray(arr, dtype=np.float32))

    # Instantaneous Signals
    for nm, sig in [
        ("oob", oob),
        ("exit_sig", exit_sig),
        ("dt_prev_oob", dt_prev_oob),
        ("dt_next_oob", dt_next_oob),
        ("rms_0", rms),
        ("p85_0", p85),
        ("walk_0", walk),
        ("run_0", run),
        ("cent_0", cent),
        ("align_0", align),
        ("tot_width_0", tot_width),
        ("tot_len_0", tot_len),
        ("in_box_0", in_box),
        ("touchline_0", touch_cnt),
        ("half_sep_0", half_sep),
        ("n_vis_0", n_vis),
    ]:
        _add(nm, sig)

    # Multi-scale symmetric windows (0.6s, 1.5s, 3.0s, 6.0s, 10.0s)
    core_series = [
        ("rms", rms),
        ("p85", p85),
        ("walk", walk),
        ("run", run),
        ("cent", cent),
        ("align", align),
        ("tot_w", tot_width),
        ("tot_l", tot_len),
        ("in_box", in_box),
        ("touch", touch_cnt),
        ("half_sep", half_sep),
    ]
    for sec in (0.6, 1.5, 3.0, 6.0, 10.0):
        w = max(1, int(round(sec * fps)))
        tag = f"pm{int(round(sec * 10)):02d}"
        for s_name, s_arr in core_series:
            _add(f"{s_name}_{tag}", box_mean_blocks(s_arr, w, w, half_blocks))

    # Asymmetric past vs future split windows (1.5s and 3.5s) for sharp stoppage entry/exit boundaries
    split_series = [("rms", rms), ("p85", p85), ("walk", walk), ("cent", cent), ("align", align)]
    for sec in (1.5, 3.5):
        w = max(1, int(round(sec * fps)))
        tag = f"{int(round(sec * 10)):02d}"
        for s_name, s_arr in split_series:
            m_past = box_mean_blocks(s_arr, w, 0, half_blocks)
            m_fut = box_mean_blocks(s_arr, 0, w, half_blocks)
            _add(f"{s_name}_past{tag}", m_past)
            _add(f"{s_name}_fut{tag}", m_fut)
            _add(f"{s_name}_jump{tag}", m_fut - m_past)

    X_dead = np.column_stack(cols).astype(np.float32)
    np.nan_to_num(X_dead, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return {"X_dead": X_dead, "dead_feature_names": names}


def extract_team_features(
    ball_data: Dict[str, np.ndarray],
    ball_kin: Dict[str, np.ndarray],
    team_dynamics: Dict[str, np.ndarray],
    p_dead: np.ndarray,
    half_blocks: List[Tuple[int, int, int]],
    pitch: Tuple[float, float],
    fps: float = 25.0,
) -> Dict[str, object]:
    """Build stoppage-barrier-clipped, exact H <-> A equivariant feature matrix X_team (N, D_team)."""
    length, _ = pitch
    oob = ball_data["out_of_bounds"]
    obs_in = (ball_data["obs_mask"] & (~oob)).astype(np.float32)
    kspd = np.where(np.isfinite(ball_data["kin_speed"]), ball_data["kin_speed"], 0.0).astype(np.float32)
    b_imp = ball_data["ball_impulse"].astype(np.float32)

    w_bar = max(1, int(round(0.25 * fps)))
    p_dead_sm = box_mean_blocks(np.asarray(p_dead, dtype=np.float32), w_bar, w_bar, half_blocks)
    barrier = (p_dead_sm >= 0.55) | oob

    cols: List[np.ndarray] = []
    names: List[str] = []
    swap_pairs: List[Tuple[int, int]] = []
    negate_cols: List[int] = []

    def _add_pair(nm_h: str, arr_h: np.ndarray, nm_a: str, arr_a: np.ndarray, add_diff: bool = True) -> None:
        ih = len(cols)
        cols.append(np.asarray(arr_h, dtype=np.float32))
        names.append(nm_h)
        ia = len(cols)
        cols.append(np.asarray(arr_a, dtype=np.float32))
        names.append(nm_a)
        swap_pairs.append((ih, ia))
        if add_diff:
            idiff = len(cols)
            cols.append(np.asarray(arr_h, dtype=np.float32) - np.asarray(arr_a, dtype=np.float32))
            names.append(f"diff_{nm_h}")
            negate_cols.append(idiff)

    def _add_anti(nm: str, arr: np.ndarray) -> None:
        idx = len(cols)
        cols.append(np.asarray(arr, dtype=np.float32))
        names.append(nm)
        negate_cols.append(idx)

    def _add_neutral(nm: str, arr: np.ndarray) -> None:
        cols.append(np.asarray(arr, dtype=np.float32))
        names.append(nm)

    d1_h, d1_a = ball_kin["d1_h"], ball_kin["d1_a"]
    d2_h, d2_a = ball_kin["d2_h"], ball_kin["d2_a"]
    rv_h, rv_a = ball_kin["rv_h"], ball_kin["rv_a"]
    aff_h, aff_a = ball_kin["aff_h"], ball_kin["aff_a"]
    imp_h, imp_a = ball_kin["imp_aff_h"], ball_kin["imp_aff_a"]
    arc_h, arc_a = ball_kin["arc_h"], ball_kin["arc_a"]

    att_dir = team_dynamics["attack_dir_h"]
    fwd_off_h = ball_kin["near_dx_h"] * att_dir
    fwd_off_a = ball_kin["near_dx_a"] * (-att_dir)

    # 1. Instantaneous Ball-Player Interaction
    _add_pair("d1_h", d1_h, "d1_a", d1_a)
    _add_pair("d2_h", d2_h, "d2_a", d2_a)
    _add_pair("rv_h", rv_h, "rv_a", rv_a)
    _add_pair("aff_h", aff_h, "aff_a", aff_a)
    _add_pair("imp_h", imp_h, "imp_a", imp_a)
    _add_pair("arc_h", arc_h, "arc_a", arc_a)
    _add_pair("fwd_off_h", fwd_off_h, "fwd_off_a", fwd_off_a, add_diff=False)

    # Ballistic longitudinal ball progression relative to Home attacking goal
    bx = np.where(np.isfinite(ball_data["kin_xy"][:, 0]), ball_data["kin_xy"][:, 0], 0.5 * length)
    bvx = np.where(np.isfinite(ball_data["kin_vel"][:, 0]), ball_data["kin_vel"][:, 0], 0.0)
    _add_anti("ball_prog_h", ((bx - 0.5 * length) / (0.5 * length)) * att_dir)
    _add_anti("ball_vx_att_h", (bvx / 20.0) * att_dir)

    # 2. Short-to-Medium Barrier-Clipped Causal, Anticausal & Symmetric Interaction Windows
    #    Crucial for rapid 0.6s-3.0s turnovers AND rejecting 1-touch grazing deflections!
    for sec in (0.28, 0.72, 1.40, 2.80):
        w = max(1, int(round(sec * fps)))
        t_lbl = f"{int(round(sec * 100)):03d}"
        ah_p = barrier_box_mean_blocks(aff_h, w, 0, barrier, half_blocks)
        aa_p = barrier_box_mean_blocks(aff_a, w, 0, barrier, half_blocks)
        ah_f = barrier_box_mean_blocks(aff_h, 0, w, barrier, half_blocks)
        aa_f = barrier_box_mean_blocks(aff_a, 0, w, barrier, half_blocks)
        ah_pm = barrier_box_mean_blocks(aff_h, w, w, barrier, half_blocks)
        aa_pm = barrier_box_mean_blocks(aff_a, w, w, barrier, half_blocks)

        _add_pair(f"aff_h_past{t_lbl}", ah_p, f"aff_a_past{t_lbl}", aa_p)
        _add_pair(f"aff_h_fut{t_lbl}", ah_f, f"aff_a_fut{t_lbl}", aa_f)
        _add_pair(f"aff_h_pm{t_lbl}", ah_pm, f"aff_a_pm{t_lbl}", aa_pm)

        # Geometric mean of past & future affinity: high on sustained control, ~0 on 1-frame deflection
        sus_h = np.sqrt(np.maximum(0.0, ah_p * ah_f))
        sus_a = np.sqrt(np.maximum(0.0, aa_p * aa_f))
        _add_pair(f"sustained_h_{t_lbl}", sus_h, f"sustained_a_{t_lbl}", sus_a)

        arc_h_pm = barrier_box_mean_blocks(arc_h, w, w, barrier, half_blocks)
        arc_a_pm = barrier_box_mean_blocks(arc_a, w, w, barrier, half_blocks)
        _add_pair(f"arc_h_pm{t_lbl}", arc_h_pm, f"arc_a_pm{t_lbl}", arc_a_pm)

    for sec in (0.24, 0.64):
        w = max(1, int(round(sec * fps)))
        t_lbl = f"{int(round(sec * 100)):03d}"
        ih_pm = barrier_box_mean_blocks(imp_h, w, w, barrier, half_blocks)
        ia_pm = barrier_box_mean_blocks(imp_a, w, w, barrier, half_blocks)
        _add_pair(f"imp_h_pm{t_lbl}", ih_pm, f"imp_a_pm{t_lbl}", ia_pm)

    # 3. Barrier-Clipped Elapsed Time to Nearest Physical Contact (Past & Future)
    dt_p_h, dt_n_h = barrier_event_timers(ball_kin["contact_h"], barrier, half_blocks, fps=fps, horizon_s=18.0)
    dt_p_a, dt_n_a = barrier_event_timers(ball_kin["contact_a"], barrier, half_blocks, fps=fps, horizon_s=18.0)
    _add_pair("dt_prev_contact_h", dt_p_h, "dt_prev_contact_a", dt_p_a)
    _add_pair("dt_next_contact_h", dt_n_h, "dt_next_contact_a", dt_n_a)

    # 4. Barrier-Clipped Collective Spatial Formation & Pitch Control (1.6s, 3.5s, 6.0s)
    pc_h = team_dynamics["pitch_control_h"]
    prog_h, prog_a = team_dynamics["prog_h"], team_dynamics["prog_a"]
    wid_h, wid_a = team_dynamics["std_y_h"], team_dynamics["std_y_a"]
    len_h, len_a = team_dynamics["std_x_h"], team_dynamics["std_x_a"]
    drf_h, drf_a = team_dynamics["drift_h"], team_dynamics["drift_a"]

    _add_anti("pc_h_0", pc_h)
    _add_pair("wid_h_0", wid_h, "wid_a_0", wid_a)
    _add_pair("prog_h_0", prog_h, "prog_a_0", prog_a)
    _add_pair("n_vis_h", team_dynamics["n_vis_h"], "n_vis_a", team_dynamics["n_vis_a"], add_diff=False)

    for sec in (1.6, 3.5, 6.0):
        w = max(1, int(round(sec * fps)))
        t_lbl = f"{int(round(sec * 10)):02d}"
        _add_anti(f"pc_h_pm{t_lbl}", barrier_box_mean_blocks(pc_h, w, w, barrier, half_blocks))
        _add_pair(
            f"wid_h_pm{t_lbl}", barrier_box_mean_blocks(wid_h, w, w, barrier, half_blocks),
            f"wid_a_pm{t_lbl}", barrier_box_mean_blocks(wid_a, w, w, barrier, half_blocks),
        )
        _add_pair(
            f"prog_h_pm{t_lbl}", barrier_box_mean_blocks(prog_h, w, w, barrier, half_blocks),
            f"prog_a_pm{t_lbl}", barrier_box_mean_blocks(prog_a, w, w, barrier, half_blocks),
        )
        _add_pair(
            f"len_h_pm{t_lbl}", barrier_box_mean_blocks(len_h, w, w, barrier, half_blocks),
            f"len_a_pm{t_lbl}", barrier_box_mean_blocks(len_a, w, w, barrier, half_blocks),
            add_diff=False,
        )
        _add_pair(
            f"drf_h_fut{t_lbl}", barrier_box_mean_blocks(drf_h, 0, w, barrier, half_blocks),
            f"drf_a_fut{t_lbl}", barrier_box_mean_blocks(drf_a, 0, w, barrier, half_blocks),
        )

    # 5. Neutral Stoppage & Visibility Context (invariant under H <-> A swap)
    no_bar = np.zeros(len(obs_in), dtype=bool)
    dt_p_dead, dt_n_dead = barrier_event_timers(barrier, no_bar, half_blocks, fps=fps, horizon_s=20.0)
    dt_p_obs, dt_n_obs = barrier_event_timers(obs_in > 0.5, barrier, half_blocks, fps=fps, horizon_s=18.0)

    _add_neutral("ctx_obs_in", obs_in)
    _add_neutral("ctx_ball_speed", kspd)
    _add_neutral("ctx_ball_impulse", b_imp)
    _add_neutral("ctx_p_dead", p_dead_sm)
    _add_neutral("ctx_dt_prev_dead", dt_p_dead)
    _add_neutral("ctx_dt_next_dead", dt_n_dead)
    _add_neutral("ctx_dt_prev_obs", dt_p_obs)
    _add_neutral("ctx_dt_next_obs", dt_n_obs)

    X_team = np.column_stack(cols).astype(np.float32)
    np.nan_to_num(X_team, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "X_team": X_team,
        "team_feature_names": names,
        "team_swap_pairs": swap_pairs,
        "team_negate_cols": negate_cols,
    }


def swap_team_feature_matrix(
    X_team: np.ndarray,
    swap_pairs: List[Tuple[int, int]],
    negate_cols: List[int],
) -> np.ndarray:
    """Exact H <-> A involution on X_team so f_sym(X) = 0.5 * (f(X) + 1 - f(swap(X))) is unbiased."""
    X_sw = X_team.copy()
    for c_h, c_a in swap_pairs:
        X_sw[:, c_h] = X_team[:, c_a]
        X_sw[:, c_a] = X_team[:, c_h]
    for c_neg in negate_cols:
        X_sw[:, c_neg] = -X_team[:, c_neg]
    return X_sw


def extract_transition_features(
    ball_data: Dict[str, np.ndarray],
    ball_kin: Dict[str, np.ndarray],
    p_dead: np.ndarray,
    p_home: np.ndarray,
    half_blocks: List[Tuple[int, int, int]],
    fps: float = 25.0,
) -> Dict[str, object]:
    """Build team-symmetric feature matrix X_turn (N, D_turn) for Turnover Probability P(turn | x_t).
    Because every feature is symmetric under H <-> A exchange, P(turn | x_t) is strictly H/A unbiased.
    """
    n = len(p_home)
    oob = ball_data["out_of_bounds"]
    obs_in = (ball_data["obs_mask"] & (~oob)).astype(np.float32)
    kspd = np.where(np.isfinite(ball_data["kin_speed"]), ball_data["kin_speed"], 0.0).astype(np.float32)
    b_imp = ball_data["ball_impulse"].astype(np.float32)

    d1_h, d1_a = ball_kin["d1_h"], ball_kin["d1_a"]
    aff_h, aff_a = ball_kin["aff_h"], ball_kin["aff_a"]
    barrier = (p_dead >= 0.55) | oob

    w7 = max(1, int(round(0.28 * fps)))
    w18 = max(1, int(round(0.72 * fps)))
    ah_p7 = barrier_box_mean_blocks(aff_h, w7, 0, barrier, half_blocks)
    aa_p7 = barrier_box_mean_blocks(aff_a, w7, 0, barrier, half_blocks)
    ah_f7 = barrier_box_mean_blocks(aff_h, 0, w7, barrier, half_blocks)
    aa_f7 = barrier_box_mean_blocks(aff_a, 0, w7, barrier, half_blocks)

    ah_p18 = barrier_box_mean_blocks(aff_h, w18, 0, barrier, half_blocks)
    aa_p18 = barrier_box_mean_blocks(aff_a, w18, 0, barrier, half_blocks)
    ah_f18 = barrier_box_mean_blocks(aff_h, 0, w18, barrier, half_blocks)
    aa_f18 = barrier_box_mean_blocks(aff_a, 0, w18, barrier, half_blocks)

    # Cross-team flip product: positive when past favored one team and future favors the opposite team
    flip_028 = (ah_p7 - aa_p7) * (aa_f7 - ah_f7)
    flip_072 = (ah_p18 - aa_p18) * (aa_f18 - ah_f18)
    max_aff_past072 = np.maximum(ah_p18, aa_p18)
    max_aff_fut072 = np.maximum(ah_f18, aa_f18)
    two_sided_aff_072 = np.minimum(max_aff_past072, max_aff_fut072)

    w25 = max(1, int(round(1.00 * fps)))
    obs_past10 = barrier_box_mean_blocks(obs_in, w25, 0, barrier, half_blocks)
    obs_fut10 = barrier_box_mean_blocks(obs_in, 0, w25, barrier, half_blocks)

    ph32 = np.asarray(p_home, dtype=np.float32)
    pd32 = np.asarray(p_dead, dtype=np.float32)

    dp_03 = np.abs(
        box_mean_blocks(ph32, 0, max(1, int(round(0.32 * fps))), half_blocks)
        - box_mean_blocks(ph32, max(1, int(round(0.32 * fps))), 0, half_blocks)
    )
    dp_08 = np.abs(
        box_mean_blocks(ph32, 0, max(1, int(round(0.80 * fps))), half_blocks)
        - box_mean_blocks(ph32, max(1, int(round(0.80 * fps))), 0, half_blocks)
    )
    dp_16 = np.abs(
        box_mean_blocks(ph32, 0, max(1, int(round(1.60 * fps))), half_blocks)
        - box_mean_blocks(ph32, max(1, int(round(1.60 * fps))), 0, half_blocks)
    )
    dp_dead_06 = (
        box_mean_blocks(pd32, 0, max(1, int(round(0.60 * fps))), half_blocks)
        - box_mean_blocks(pd32, max(1, int(round(0.60 * fps))), 0, half_blocks)
    )

    # Elapsed time to nearest posterior 0.50 zero-crossing
    zc_mask = np.zeros(n, dtype=bool)
    for _, s_f, e_f in half_blocks:
        if e_f > s_f:
            c = ph32[s_f : e_f + 1] - 0.5
            zc = (c[:-1] * c[1:]) <= 0.0
            zc_mask[s_f + 1 : e_f + 1] |= zc
            zc_mask[s_f:e_f] |= zc
    no_bar = np.zeros(n, dtype=bool)
    dt_zc_p, dt_zc_n = barrier_event_timers(zc_mask, no_bar, half_blocks, fps=fps, horizon_s=10.0)
    dt_zc_min = np.minimum(dt_zc_p, dt_zc_n)
    dt_prev_dead, dt_next_dead = barrier_event_timers(barrier, no_bar, half_blocks, fps=fps, horizon_s=20.0)

    # Physical impulse coupled with ownership flip: strictly zero on same-team passes,
    # positive ONLY when ball impulse is accompanied by cross-team change in affinity.
    flip_imp_028 = (b_imp * np.maximum(0.0, flip_028)).astype(np.float32)
    flip_imp_072 = (b_imp * np.maximum(0.0, flip_072)).astype(np.float32)

    cols = [
        ("obs_in", obs_in),
        ("obs_past10", obs_past10),
        ("obs_fut10", obs_fut10),
        ("flip_imp_028", flip_imp_028),
        ("flip_imp_072", flip_imp_072),
        ("min_d1", np.minimum(d1_h, d1_a)),
        ("abs_d1_diff", np.abs(d1_h - d1_a)),
        ("max_aff", np.maximum(aff_h, aff_a)),
        ("min_aff", np.minimum(aff_h, aff_a)),
        ("max_aff_past072", max_aff_past072),
        ("max_aff_fut072", max_aff_fut072),
        ("two_sided_aff_072", two_sided_aff_072),
        ("flip_028", flip_028),
        ("flip_072", flip_072),
        ("dp_team_03", dp_03),
        ("dp_team_08", dp_08),
        ("dp_team_16", dp_16),
        ("dt_zero_cross", dt_zc_min),
        ("team_uncertainty", 1.0 - 2.0 * np.abs(ph32 - 0.5)),
        ("p_dead", pd32),
        ("dp_dead_06", dp_dead_06),
        ("dt_prev_dead", dt_prev_dead),
        ("dt_next_dead", dt_next_dead),
    ]
    X_turn = np.column_stack([c[1] for c in cols]).astype(np.float32)
    np.nan_to_num(X_turn, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return {"X_turn": X_turn, "turn_feature_names": [c[0] for c in cols]}
