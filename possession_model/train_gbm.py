"""Leak-Free 7-Match External Training, Nested Leave-One-Match-Out Cross-Validation,
Probabilistic Viterbi Calibration, and Production 3-Booster Bundle Export.

Strict Competition Compliance (RULES.md & CLARIFICATIONS.md):
  - Trains and calibrates EXCLUSIVELY on external matches in `data/training/` (DFL open dataset).
  - Never reads or touches `BAR-ATH` ground-truth files or video.
  - Groups original (`dfl_sporcate`) and domain-adapted (`mtah_aug`) copies strictly by match ID
    so no match ever leaks across train/validation folds.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from possession_model.ball_tracking import (  # noqa: E402
    compute_ball_player_kinematics,
    reconstruct_ball_trajectory,
)
from possession_model.dfl_dataset import (  # noqa: E402
    augment_dfl_to_mtah_domain,
    default_training_dir,
    evaluate_dfl_predictions,
    filter_ricochets_by_physics,
    list_dfl_matches,
    load_dfl_gt,
    load_dfl_match,
)
from possession_model.gbm_features import (  # noqa: E402
    extract_dead_features,
    extract_team_features,
    extract_transition_features,
)
from possession_model.gbm_model import (  # noqa: E402
    CalibrationParams,
    PossessionGBMBundle,
    build_turnover_targets,
    compute_probabilistic_sequence_costs,
    compute_spell_balanced_weights,
    decode_match_sequence,
    select_training_frame_indices,
)
from possession_model.sequence_decoder import decode_possession_sequence  # noqa: E402
from possession_model.signal_ops import runs_of_labels  # noqa: E402
from possession_model.team_dynamics import compute_team_dynamics  # noqa: E402

DEFAULT_MODEL_DIR = os.path.join(HERE, "models")
BUNDLE_VERSION = "v5_cascaded_3gbm"
TURNOVER_RADIUS_FRAMES = 6  # +/- 0.24 s physical foot-ball contact window at 25 Hz


def _sample_turn_slice(
    X_turn: np.ndarray,
    y_turn: np.ndarray,
    half_blocks: List[Tuple[int, int, int]],
    neg_stride: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep all positive turnover frames (weight 1.0) and subsample negative frames every `neg_stride`
    frames with exact importance weight `float(neg_stride)` so Bayes posterior calibration is preserved.
    """
    sel_parts: List[np.ndarray] = []
    w_parts: List[np.ndarray] = []
    for _, s_f, e_f in half_blocks:
        sub_y = y_turn[s_f : e_f + 1]
        pos_rel = np.flatnonzero(sub_y > 0.5)
        neg_rel = np.flatnonzero(sub_y <= 0.5)[:: max(1, neg_stride)]
        idx_h = np.concatenate([s_f + pos_rel, s_f + neg_rel])
        w_h = np.concatenate([
            np.ones(len(pos_rel), dtype=np.float32),
            np.full(len(neg_rel), float(max(1, neg_stride)), dtype=np.float32),
        ])
        order = np.argsort(idx_h, kind="mergesort")
        sel_parts.append(idx_h[order])
        w_parts.append(w_h[order])

    sel = np.concatenate(sel_parts)
    weights = np.concatenate(w_parts)
    return X_turn[sel], y_turn[sel], weights


def _count_spurious_pre_stoppage(
    pred: np.ndarray,
    gt_phys: np.ndarray,
    half_blocks: List[Tuple[int, int, int]],
    fps: float = 25.0,
) -> int:
    """Count false T1 -> short T2 (<= 3.6 s) -> D flips where T2 had <25% GT overlap."""
    max_len = int(round(3.6 * fps))
    cnt = 0
    for _, s_f, e_f in half_blocks:
        runs = runs_of_labels(pred[s_f : e_f + 1])
        for i in range(1, len(runs) - 1):
            a, b, lbl = runs[i]
            prev_lbl = runs[i - 1][2]
            next_lbl = runs[i + 1][2]
            if (
                lbl in ("H", "A")
                and prev_lbl in ("H", "A")
                and prev_lbl != lbl
                and next_lbl == "D"
                and (b - a) <= max_len
            ):
                if float((gt_phys[s_f + a : s_f + b] == lbl).mean()) < 0.25:
                    cnt += 1
    return cnt


def _compute_duration_bin_recall(
    bundles: List[dict],
    oof_preds: Dict[str, np.ndarray],
    oof_ph: Dict[str, np.ndarray],
) -> Dict[str, dict]:
    """Diagnostic breakdown of live possession recall by spell duration bin on gt_phys."""
    bins = {
        "0.5-1.5s": [0, 0, 0],
        "1.5-3.0s": [0, 0, 0],
        "3.0-6.0s": [0, 0, 0],
        ">6.0s": [0, 0, 0],
    }
    for b in bundles:
        m = b["match_id"]
        fps = b["fps"]
        pred = oof_preds[m]
        ph = oof_ph[m]
        for _, s_f, e_f in b["half_blocks"]:
            for a, e_excl, lbl in runs_of_labels(b["gt_phys"][s_f : e_f + 1]):
                if lbl not in ("H", "A"):
                    continue
                dur_s = (e_excl - a) / fps
                if dur_s < 0.5:
                    continue
                key = (
                    "0.5-1.5s" if dur_s <= 1.5
                    else ("1.5-3.0s" if dur_s <= 3.0 else ("3.0-6.0s" if dur_s <= 6.0 else ">6.0s"))
                )
                bins[key][0] += 1
                ph_seg = ph[s_f + a : s_f + e_excl]
                gbm_hit = float(((ph_seg >= 0.5) if lbl == "H" else (ph_seg < 0.5)).mean()) >= 0.35
                vit_hit = float((pred[s_f + a : s_f + e_excl] == lbl).mean()) >= 0.35
                bins[key][1] += int(gbm_hit)
                bins[key][2] += int(vit_hit)

    return {
        k: {
            "total_spells": tot,
            "gbm_recall": round(g_ok / max(1, tot), 4),
            "viterbi_recall": round(v_ok / max(1, tot), 4),
        }
        for k, (tot, g_ok, v_ok) in bins.items()
    }


def prepare_match_bundle(
    match_id: str,
    training_dir: str,
    build_augmented: bool = True,
    aug_seed: int = 2027,
) -> dict:
    """Load `dfl_sporcate` (+ `dfl_pulite` for physical ricochet GT filtering) and extract X_dead."""
    tac, half_blocks, pitch, fps = load_dfl_match(match_id, kind="dfl_sporcate", training_dir=training_dir)
    tac_clean, _, _, _ = load_dfl_match(match_id, kind="dfl_pulite", training_dir=training_dir)
    gt_raw = load_dfl_gt(match_id, training_dir=training_dir)
    gt_phys = filter_ricochets_by_physics(gt_raw, tac_clean, half_blocks, fps=fps)
    del tac_clean

    ball_data = reconstruct_ball_trajectory(tac["ball_xy"], None, pitch=pitch, fps=fps)
    td = compute_team_dynamics(
        tac["player_xy"], tac["player_speed"], tac["player_vel"],
        tac["slot_teams"], ball_data["kin_xy"], half_blocks, pitch, fps=fps,
    )
    ball_kin = compute_ball_player_kinematics(
        tac["player_xy"], tac["player_vel"], tac["slot_teams"], ball_data, fps=fps
    )
    dead_ft = extract_dead_features(ball_data, td, half_blocks, fps=fps)
    del tac

    aug_state = None
    if build_augmented:
        tac_orig, _, _, _ = load_dfl_match(match_id, kind="dfl_sporcate", training_dir=training_dir)
        tac_aug = augment_dfl_to_mtah_domain(tac_orig, pitch=pitch, fps=fps, seed=aug_seed)
        del tac_orig
        bd_a = reconstruct_ball_trajectory(tac_aug["ball_xy"], None, pitch=pitch, fps=fps)
        td_a = compute_team_dynamics(
            tac_aug["player_xy"], tac_aug["player_speed"], tac_aug["player_vel"],
            tac_aug["slot_teams"], bd_a["kin_xy"], half_blocks, pitch, fps=fps,
        )
        bk_a = compute_ball_player_kinematics(
            tac_aug["player_xy"], tac_aug["player_vel"], tac_aug["slot_teams"], bd_a, fps=fps
        )
        aug_dead_ft = extract_dead_features(bd_a, td_a, half_blocks, fps=fps)
        del tac_aug
        aug_state = {
            "ball_data": bd_a,
            "ball_kin": bk_a,
            "team_dynamics": td_a,
            "X_dead": aug_dead_ft["X_dead"],
        }

    return {
        "match_id": match_id,
        "half_blocks": half_blocks,
        "pitch": pitch,
        "fps": fps,
        "gt_raw": gt_raw,
        "gt_phys": gt_phys,
        "y_turn": build_turnover_targets(gt_phys, half_blocks, radius_frames=TURNOVER_RADIUS_FRAMES),
        "ball_data": ball_data,
        "ball_kin": ball_kin,
        "team_dynamics": td,
        "X_dead": dead_ft["X_dead"],
        "dead_feature_names": dead_ft["dead_feature_names"],
        "aug_state": aug_state,
    }


def _collect_dead_slices(
    bundles: List[dict],
    use_augmented: bool = True,
    stride: int = 5,
    aug_stride: int = 8,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    Xd_list, yd_list = [], []
    for b in bundles:
        idx = select_training_frame_indices(b["gt_phys"], b["half_blocks"], stride=stride, boundary_radius=16)
        Xd_list.append(b["X_dead"][idx])
        yd_list.append((b["gt_raw"][idx] == "D").astype(np.float32))
        if use_augmented:
            if b.get("aug_state") is not None:
                idx_a = select_training_frame_indices(
                    b["gt_phys"], b["half_blocks"], stride=aug_stride, boundary_radius=12
                )
                Xd_list.append(b["aug_state"]["X_dead"][idx_a])
                yd_list.append((b["gt_raw"][idx_a] == "D").astype(np.float32))
            elif b.get("aug_dead_slice") is not None:
                Xd_list.append(b["aug_dead_slice"][0])
                yd_list.append(b["aug_dead_slice"][1])
    return Xd_list, yd_list


def _collect_team_slices(
    bundles: List[dict],
    use_augmented: bool = True,
    stride: int = 5,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    Xt_list, yt_list, wt_list = [], [], []
    for b in bundles:
        gt_p = b["gt_phys"]
        idx = select_training_frame_indices(gt_p, b["half_blocks"], stride=stride, boundary_radius=18)
        alive = idx[np.isin(gt_p[idx], ["H", "A"])]
        Xt_list.append(b["X_team"][alive])
        yt_list.append((gt_p[alive] == "H").astype(np.float32))
        wt_list.append(compute_spell_balanced_weights(gt_p, alive, b["half_blocks"]))

        if use_augmented and b.get("aug_team_slice") is not None:
            Xt_a, yt_a, wt_a = b["aug_team_slice"]
            Xt_list.append(Xt_a)
            yt_list.append(yt_a)
            wt_list.append(wt_a)
    return Xt_list, yt_list, wt_list


def _collect_turn_slices(
    bundles: List[dict],
    use_augmented: bool = True,
    neg_stride: int = 4,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    Xtr_list, ytr_list, wtr_list = [], [], []
    for b in bundles:
        Xs, ys, ws = _sample_turn_slice(b["X_turn"], b["y_turn"], b["half_blocks"], neg_stride=neg_stride)
        Xtr_list.append(Xs)
        ytr_list.append(ys)
        wtr_list.append(ws)
        if use_augmented and b.get("aug_turn_slice") is not None:
            Xa, ya, wa = b["aug_turn_slice"]
            Xtr_list.append(Xa)
            ytr_list.append(ya)
            wtr_list.append(wa)
    return Xtr_list, ytr_list, wtr_list


def _decode_match_coarse(
    p_dead: np.ndarray,
    p_home: np.ndarray,
    p_turn: np.ndarray,
    ball_data: Dict[str, np.ndarray],
    half_blocks: List[Tuple[int, int, int]],
    params: CalibrationParams,
    fps: float = 25.0,
    step: int = 2,
) -> np.ndarray:
    """Strided Viterbi (step=2 -> 0.08 s) for fast coordinate search, exact when step=1."""
    if step <= 1:
        return decode_match_sequence(p_dead, p_home, p_turn, ball_data, half_blocks, params, fps=fps)

    fc, tc, sc, rc = compute_probabilistic_sequence_costs(
        p_dead, p_home, p_turn, ball_data, half_blocks, params, fps=fps
    )
    labels = np.full(len(p_dead), "X", dtype="<U1")
    cmap = np.array(["H", "A", "D"], dtype="<U1")
    min_l = max(1, int(round(params.min_live_frames / step)))
    min_d = max(1, int(round(params.min_dead_frames / step)))
    min_p = max(min_l, int(round(params.min_pre_stop_frames / step)))

    for _, s_f, e_f in half_blocks:
        m = e_f - s_f + 1
        nb = m // step
        trim = nb * step
        fc_b = fc[s_f : s_f + trim].reshape(nb, step, 3).sum(axis=1)
        tc_b = tc[s_f : s_f + trim].reshape(nb, step).min(axis=1)
        sc_b = sc[s_f : s_f + trim].reshape(nb, step).min(axis=1)
        rc_b = rc[s_f : s_f + trim].reshape(nb, step).min(axis=1)

        st_b = decode_possession_sequence(
            fc_b,
            tc_b,
            sc_b,
            rc_b,
            min_possession_frames=min_l,
            min_stoppage_frames=min_d,
            min_pre_stop_frames=min_p,
            pen_pre_stop=params.pen_pre_stop,
        )
        labels[s_f : s_f + trim] = cmap[np.repeat(st_b, step)]
        if trim < m:
            labels[s_f + trim : e_f + 1] = cmap[st_b[-1]]
    return labels


def score_calibration_on_bundles(
    bundles: List[dict],
    posteriors: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    params: CalibrationParams,
    coarse_step: int = 1,
    memo: Optional[Dict[tuple, dict]] = None,
) -> dict:
    """Evaluate `CalibrationParams` across `bundles` on both physics-filtered and raw DFL GT."""
    sp_phys, mf_phys, sp_raw, mf_raw, ad_f1, ta_alive, n_pred_l, spur_l = (
        [], [], [], [], [], [], [], []
    )
    key_base = tuple(sorted(asdict(params).items()))

    for b in bundles:
        m = b["match_id"]
        ck = (key_base, int(coarse_step), m) if memo is not None else None
        if memo is not None and ck in memo:
            item = memo[ck]
        else:
            pd_m, ph_m, pt_m = posteriors[m]
            pred = _decode_match_coarse(
                pd_m, ph_m, pt_m, b["ball_data"], b["half_blocks"], params, fps=b["fps"], step=coarse_step
            )
            rp = evaluate_dfl_predictions(b["gt_phys"], pred, b["half_blocks"])
            rr = evaluate_dfl_predictions(b["gt_raw"], pred, b["half_blocks"]) if coarse_step <= 1 else rp
            spur = _count_spurious_pre_stoppage(pred, b["gt_phys"], b["half_blocks"], fps=b["fps"])
            item = {
                "spell_phys": rp["spell_f1_mean"],
                "macro_phys": rp["macro_f1"],
                "spell_raw": rr["spell_f1_mean"],
                "macro_raw": rr["macro_f1"],
                "alive_dead_f1": rp["alive_dead_f1"],
                "team_acc_alive": rp["team_acc_both_alive"],
                "n_pred_spells": rp["n_pred_spells"],
                "spur_pre_stop": spur,
            }
            if memo is not None:
                memo[ck] = item

        sp_phys.append(item["spell_phys"])
        mf_phys.append(item["macro_phys"])
        sp_raw.append(item["spell_raw"])
        mf_raw.append(item["macro_raw"])
        ad_f1.append(item["alive_dead_f1"])
        ta_alive.append(item["team_acc_alive"])
        n_pred_l.append(item["n_pred_spells"])
        spur_l.append(item["spur_pre_stop"])

    mean_sp_phys = float(np.mean(sp_phys))
    mean_sp_raw = float(np.mean(sp_raw))
    mean_mf_phys = float(np.mean(mf_phys))
    mean_spur = float(np.mean(spur_l))
    composite = 0.55 * mean_sp_phys + 0.25 * mean_sp_raw + 0.20 * mean_mf_phys - 0.003 * mean_spur

    return {
        "composite": composite,
        "spell_phys": mean_sp_phys,
        "spell_raw": mean_sp_raw,
        "macro_phys": mean_mf_phys,
        "macro_raw": float(np.mean(mf_raw)),
        "alive_dead_f1": float(np.mean(ad_f1)),
        "team_acc_alive": float(np.mean(ta_alive)),
        "n_pred_spells": float(np.mean(n_pred_l)),
        "spurious_pre_stop_total": int(np.sum(spur_l)),
    }


def optimize_calibration_params(
    bundles: List[dict],
    posteriors: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    base: Optional[CalibrationParams] = None,
    coarse_step: int = 5,
    n_passes: int = 1,
    memo: Optional[Dict[tuple, dict]] = None,
) -> Tuple[CalibrationParams, dict]:
    """Coordinate ascent on `CalibrationParams` over external DFL training folds."""
    start_p = base or CalibrationParams()
    best_p = start_p
    best_s = score_calibration_on_bundles(bundles, posteriors, best_p, coarse_step=coarse_step, memo=memo)

    search_grid = [
        ("scale_team", [1.25, 1.55, 1.85]),
        ("alpha_turn", [5.50, 7.50, 9.50]),
        ("beta_turn", [1.00, 2.00, 3.20]),
        ("min_live_frames", [14, 18, 22]),
        ("min_pre_stop_frames", [45, 55, 68]),
        ("pen_pre_stop", [4.00, 6.50, 9.00]),
        ("scale_dead", [0.90, 1.05]),
        ("bias_dead", [-0.10, 0.05]),
        ("alpha_stop", [3.50, 5.50]),
        ("beta_stop", [2.50, 4.50]),
        ("beta_rest", [2.50, 4.50]),
        ("min_dead_frames", [70, 82]),
    ]

    for _pass in range(n_passes):
        improved = False
        for field_name, candidates in search_grid:
            for val in candidates:
                if getattr(best_p, field_name) == val:
                    continue
                cand_p = replace(best_p, **{field_name: val})
                cand_s = score_calibration_on_bundles(
                    bundles, posteriors, cand_p, coarse_step=coarse_step, memo=memo
                )
                if cand_s["composite"] > best_s["composite"] + 1e-5:
                    best_p = cand_p
                    best_s = cand_s
                    improved = True
        if not improved:
            break

    if coarse_step > 1:
        exact_best = score_calibration_on_bundles(bundles, posteriors, best_p, coarse_step=1, memo=memo)
        exact_start = score_calibration_on_bundles(bundles, posteriors, start_p, coarse_step=1, memo=memo)
        if exact_start["composite"] > exact_best["composite"]:
            return start_p, exact_start
        return best_p, exact_best

    return best_p, best_s


def run_training_and_cv(
    training_dir: Optional[str] = None,
    out_model_dir: str = DEFAULT_MODEL_DIR,
    use_augmented: bool = True,
    tune_calibration: bool = True,
) -> dict:
    t0 = time.time()
    tdir = training_dir or default_training_dir()
    matches = list_dfl_matches(tdir, kind="dfl_sporcate")
    print(f"Found {len(matches)} external training matches in {tdir}: {matches}", flush=True)

    bundles: List[dict] = []
    for idx, m in enumerate(matches):
        tm = time.time()
        b = prepare_match_bundle(m, tdir, build_augmented=use_augmented, aug_seed=2027 + idx)
        bundles.append(b)
        n_raw_sp = sum(
            sum(1 for _, _, lbl in runs_of_labels(b["gt_raw"][s : e + 1]) if lbl in ("H", "A"))
            for _, s, e in b["half_blocks"]
        )
        n_phys_sp = sum(
            sum(1 for _, _, lbl in runs_of_labels(b["gt_phys"][s : e + 1]) if lbl in ("H", "A"))
            for _, s, e in b["half_blocks"]
        )
        print(
            f"  [Prep {idx+1}/{len(matches)}] {m} ({len(b['gt_raw'])} frames, {time.time()-tm:.1f}s) | "
            f"live spells: raw={n_raw_sp} -> physics_controlled={n_phys_sp}",
            flush=True,
        )

    dead_names = bundles[0]["dead_feature_names"]
    n_dead_rounds = 380
    n_team_rounds = 420
    n_turn_rounds = 260

    # Stage 1: 7-Fold LOMO-CV for Dead-Ball LightGBM
    print("\n=== Stage 1: 7-Fold LOMO-CV for Dead-Ball LightGBM ===", flush=True)
    oof_p_dead: Dict[str, np.ndarray] = {}
    oof_aug_p_dead: Dict[str, np.ndarray] = {}

    for fold_idx, val_b in enumerate(bundles):
        tf = time.time()
        val_m = val_b["match_id"]
        tr_bs = [b for b in bundles if b["match_id"] != val_m]
        Xd_tr, yd_tr = _collect_dead_slices(tr_bs, use_augmented=use_augmented)
        dead_fold = PossessionGBMBundle.fit_dead_booster(Xd_tr, yd_tr, dead_names, n_rounds=n_dead_rounds)
        pd_val = PossessionGBMBundle.predict_dead_with_booster(dead_fold, val_b["X_dead"], val_b["half_blocks"])
        oof_p_dead[val_m] = pd_val
        if use_augmented and val_b["aug_state"] is not None:
            oof_aug_p_dead[val_m] = PossessionGBMBundle.predict_dead_with_booster(
                dead_fold, val_b["aug_state"]["X_dead"], val_b["half_blocks"]
            )

        v_idx = select_training_frame_indices(val_b["gt_phys"], val_b["half_blocks"], stride=5)
        d_acc = float(((pd_val[v_idx] >= 0.5) == (val_b["gt_raw"][v_idx] == "D")).mean())
        print(f"  [Dead Fold {fold_idx+1}/7 -> {val_m}] ({time.time()-tf:.1f}s) | D_acc={d_acc:.4f}", flush=True)

    # Stage 2: Extract Barrier-Clipped Team Features & Run 7-Fold LOMO-CV for Spell-Weighted Team GBM
    print("\n=== Stage 2: 7-Fold LOMO-CV for Spell-Weighted Equivariant Team LightGBM ===", flush=True)
    for b in bundles:
        m = b["match_id"]
        tft = extract_team_features(
            b["ball_data"], b["ball_kin"], b["team_dynamics"],
            oof_p_dead[m], b["half_blocks"], b["pitch"], fps=b["fps"],
        )
        b["X_team"] = tft["X_team"]
        b["team_feature_names"] = tft["team_feature_names"]
        b["team_swap_pairs"] = tft["team_swap_pairs"]
        b["team_negate_cols"] = tft["team_negate_cols"]

        b["aug_team_slice"] = None
        if use_augmented and b["aug_state"] is not None:
            ast = b["aug_state"]
            aug_tft = extract_team_features(
                ast["ball_data"], ast["ball_kin"], ast["team_dynamics"],
                oof_aug_p_dead[m], b["half_blocks"], b["pitch"], fps=b["fps"],
            )
            ast["X_team"] = aug_tft["X_team"]
            idx_a = select_training_frame_indices(b["gt_phys"], b["half_blocks"], stride=8, boundary_radius=14)
            alive_a = idx_a[np.isin(b["gt_phys"][idx_a], ["H", "A"])]
            b["aug_team_slice"] = (
                ast["X_team"][alive_a],
                (b["gt_phys"][alive_a] == "H").astype(np.float32),
                compute_spell_balanced_weights(b["gt_phys"], alive_a, b["half_blocks"]),
            )
            b["aug_dead_slice"] = (
                ast["X_dead"][idx_a],
                (b["gt_raw"][idx_a] == "D").astype(np.float32),
            )

    team_names = bundles[0]["team_feature_names"]
    swap_pairs = bundles[0]["team_swap_pairs"]
    negate_cols = bundles[0]["team_negate_cols"]

    oof_p_home: Dict[str, np.ndarray] = {}
    oof_aug_p_home: Dict[str, np.ndarray] = {}

    for fold_idx, val_b in enumerate(bundles):
        tf = time.time()
        val_m = val_b["match_id"]
        tr_bs = [b for b in bundles if b["match_id"] != val_m]
        Xt_tr, yt_tr, wt_tr = _collect_team_slices(tr_bs, use_augmented=use_augmented)
        team_fold = PossessionGBMBundle.fit_team_booster(
            Xt_tr, yt_tr, wt_tr, team_names, swap_pairs, negate_cols, n_rounds=n_team_rounds
        )
        ph_val = PossessionGBMBundle.predict_team_with_booster(
            team_fold, val_b["X_team"], swap_pairs, negate_cols, val_b["half_blocks"]
        )
        oof_p_home[val_m] = ph_val
        if use_augmented and val_b["aug_state"] is not None:
            oof_aug_p_home[val_m] = PossessionGBMBundle.predict_team_with_booster(
                team_fold, val_b["aug_state"]["X_team"], swap_pairs, negate_cols, val_b["half_blocks"]
            )

        v_idx = select_training_frame_indices(val_b["gt_phys"], val_b["half_blocks"], stride=5)
        alive_v = v_idx[np.isin(val_b["gt_phys"][v_idx], ["H", "A"])]
        t_acc = float(((ph_val[alive_v] >= 0.5) == (val_b["gt_phys"][alive_v] == "H")).mean())
        print(f"  [Team Fold {fold_idx+1}/7 -> {val_m}] ({time.time()-tf:.1f}s) | alive T_acc={t_acc:.4f}", flush=True)

    # Stage 3: Extract Team-Symmetric X_turn & Run 7-Fold LOMO-CV for Turnover Boundary LightGBM
    print("\n=== Stage 3: 7-Fold LOMO-CV for Team-Symmetric Turnover Boundary LightGBM ===", flush=True)
    for b in bundles:
        m = b["match_id"]
        trn_ft = extract_transition_features(
            b["ball_data"], b["ball_kin"], oof_p_dead[m], oof_p_home[m], b["half_blocks"], fps=b["fps"]
        )
        b["X_turn"] = trn_ft["X_turn"]
        b["turn_feature_names"] = trn_ft["turn_feature_names"]
        b["aug_turn_slice"] = None
        if use_augmented and b["aug_state"] is not None:
            ast = b["aug_state"]
            aug_trn = extract_transition_features(
                ast["ball_data"], ast["ball_kin"], oof_aug_p_dead[m], oof_aug_p_home[m],
                b["half_blocks"], fps=b["fps"],
            )
            b["aug_turn_slice"] = _sample_turn_slice(
                aug_trn["X_turn"], b["y_turn"], b["half_blocks"], neg_stride=6
            )
            b["aug_state"] = None  # Free full-length augmented arrays

    turn_names = bundles[0]["turn_feature_names"]
    oof_posteriors: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for fold_idx, val_b in enumerate(bundles):
        tf = time.time()
        val_m = val_b["match_id"]
        tr_bs = [b for b in bundles if b["match_id"] != val_m]
        Xtr_l, ytr_l, wtr_l = _collect_turn_slices(tr_bs, use_augmented=use_augmented, neg_stride=4)
        turn_fold = PossessionGBMBundle.fit_turn_booster(
            Xtr_l, ytr_l, wtr_l, turn_names, n_rounds=n_turn_rounds
        )
        pt_val = PossessionGBMBundle.predict_turn_with_booster(turn_fold, val_b["X_turn"], val_b["half_blocks"])
        oof_posteriors[val_m] = (oof_p_dead[val_m], oof_p_home[val_m], pt_val)

        pos_m = val_b["y_turn"] > 0.5
        neg_m = ~pos_m
        print(
            f"  [Turn Fold {fold_idx+1}/7 -> {val_m}] ({time.time()-tf:.1f}s) | "
            f"mean_p(turn)={float(pt_val[pos_m].mean()):.4f} vs mean_p(non_turn)={float(pt_val[neg_m].mean()):.4f}",
            flush=True,
        )

    # Stage 4: Nested 7-Fold LOMO-CV Calibration of Probabilistic Viterbi Parameters
    eval_memo: Dict[tuple, dict] = {}
    oof_default_summary = score_calibration_on_bundles(
        bundles, oof_posteriors, CalibrationParams(), coarse_step=1, memo=eval_memo
    )

    nested_sp_phys, nested_sp_raw, nested_macro, nested_ad, nested_ta, nested_spur = (
        [], [], [], [], [], []
    )
    nested_preds: Dict[str, np.ndarray] = {}
    best_global_params = CalibrationParams()

    if tune_calibration:
        print("\n=== Stage 4: Nested 7-Fold LOMO-CV Calibration of Viterbi Log-Likelihood Scales ===", flush=True)
        for fold_idx, val_b in enumerate(bundles):
            tf = time.time()
            val_m = val_b["match_id"]
            tr_bs = [b for b in bundles if b["match_id"] != val_m]
            fold_p, _ = optimize_calibration_params(tr_bs, oof_posteriors, coarse_step=5, n_passes=1, memo=eval_memo)
            val_s = score_calibration_on_bundles([val_b], oof_posteriors, fold_p, coarse_step=1, memo=eval_memo)
            nested_sp_phys.append(val_s["spell_phys"])
            nested_sp_raw.append(val_s["spell_raw"])
            nested_macro.append(val_s["macro_phys"])
            nested_ad.append(val_s["alive_dead_f1"])
            nested_ta.append(val_s["team_acc_alive"])
            nested_spur.append(val_s["spurious_pre_stop_total"])

            pd_m, ph_m, pt_m = oof_posteriors[val_m]
            nested_preds[val_m] = decode_match_sequence(
                pd_m, ph_m, pt_m, val_b["ball_data"], val_b["half_blocks"], fold_p, fps=val_b["fps"]
            )
            print(
                f"  [Nested Fold {fold_idx+1}/7 -> {val_m}] ({time.time()-tf:.1f}s) | "
                f"spell_phys={val_s['spell_phys']:.4f} | spurPreD={val_s['spurious_pre_stop_total']}",
                flush=True,
            )

        best_global_params, oof_tuned_summary = optimize_calibration_params(
            bundles, oof_posteriors, coarse_step=5, n_passes=2, memo=eval_memo
        )
    else:
        oof_tuned_summary = oof_default_summary
        for b in bundles:
            m = b["match_id"]
            pd_m, ph_m, pt_m = oof_posteriors[m]
            nested_preds[m] = decode_match_sequence(
                pd_m, ph_m, pt_m, b["ball_data"], b["half_blocks"], best_global_params, fps=b["fps"]
            )

    bin_recall = _compute_duration_bin_recall(bundles, nested_preds, oof_p_home)

    print("\n================ 7-MATCH EXTERNAL BENCHMARK SUMMARY (1,002,644 FRAMES) ================")
    print(
        f"  Pure-ML v5 OOF(def) : spell_phys={oof_default_summary['spell_phys']:.4f} | "
        f"spell_raw={oof_default_summary['spell_raw']:.4f} | macro={oof_default_summary['macro_phys']:.4f} | "
        f"spurPreD={oof_default_summary['spurious_pre_stop_total']}"
    )
    if tune_calibration:
        print(
            f"  Pure-ML v5 NestedCV : spell_phys={np.mean(nested_sp_phys):.4f} | "
            f"spell_raw={np.mean(nested_sp_raw):.4f} | macro={np.mean(nested_macro):.4f} | "
            f"adF1={np.mean(nested_ad):.4f} | teamAcc={np.mean(nested_ta):.4f} | "
            f"spurPreD={int(np.sum(nested_spur))}"
        )
    print("  Short-Spell Recall by Duration Bin (Nested OOF):")
    for b_key, b_val in bin_recall.items():
        print(
            f"    {b_key:8s} (n={b_val['total_spells']:3d}): "
            f"GBM sees {100.0 * b_val['gbm_recall']:5.1f}% -> Viterbi keeps {100.0 * b_val['viterbi_recall']:5.1f}%"
        )
    print(f"  Calibrated Params   : {asdict(best_global_params)}", flush=True)

    # Stage 5: Train Final Production 3-Booster Bundle on All 7 External DFL Matches
    print(
        f"\n=== Stage 5: Training Final Production 3-Booster v5 Bundle on All 7 Matches "
        f"(dead={n_dead_rounds}, team={n_team_rounds}, turn={n_turn_rounds}) ===",
        flush=True,
    )
    Xd_all, yd_all = _collect_dead_slices(bundles, use_augmented=use_augmented)
    Xt_all, yt_all, wt_all = _collect_team_slices(bundles, use_augmented=use_augmented)
    Xtr_all, ytr_all, wtr_all = _collect_turn_slices(bundles, use_augmented=use_augmented, neg_stride=4)

    dead_final = PossessionGBMBundle.fit_dead_booster(Xd_all, yd_all, dead_names, n_rounds=n_dead_rounds)
    team_final = PossessionGBMBundle.fit_team_booster(
        Xt_all, yt_all, wt_all, team_names, swap_pairs, negate_cols, n_rounds=n_team_rounds
    )
    turn_final = PossessionGBMBundle.fit_turn_booster(Xtr_all, ytr_all, wtr_all, turn_names, n_rounds=n_turn_rounds)

    final_bundle = PossessionGBMBundle(
        dead_booster=dead_final,
        team_booster=team_final,
        turn_booster=turn_final,
        dead_feature_names=dead_names,
        team_feature_names=team_names,
        turn_feature_names=turn_names,
        team_swap_pairs=swap_pairs,
        team_negate_cols=negate_cols,
        calibration=best_global_params,
    )
    final_bundle.save(out_model_dir)

    dead_imp = sorted(
        zip(dead_names, dead_final.feature_importance(importance_type="gain")), key=lambda x: -x[1]
    )[:10]
    team_imp = sorted(
        zip(team_names, team_final.feature_importance(importance_type="gain")), key=lambda x: -x[1]
    )[:10]
    turn_imp = sorted(
        zip(turn_names, turn_final.feature_importance(importance_type="gain")), key=lambda x: -x[1]
    )[:10]

    print("  Top Dead Features (gain):", [(n, round(float(g), 1)) for n, g in dead_imp[:6]])
    print("  Top Team Features (gain):", [(n, round(float(g), 1)) for n, g in team_imp[:6]])
    print("  Top Turn Features (gain):", [(n, round(float(g), 1)) for n, g in turn_imp[:6]])

    cv_report = {
        "bundle_version": BUNDLE_VERSION,
        "n_matches": len(matches),
        "matches": matches,
        "use_augmented": use_augmented,
        "oof_default": oof_default_summary,
        "nested_cv": {
            "spell_phys": float(np.mean(nested_sp_phys)) if tune_calibration else oof_default_summary["spell_phys"],
            "spell_raw": float(np.mean(nested_sp_raw)) if tune_calibration else oof_default_summary["spell_raw"],
            "macro_phys": float(np.mean(nested_macro)) if tune_calibration else oof_default_summary["macro_phys"],
            "alive_dead_f1": float(np.mean(nested_ad)) if tune_calibration else oof_default_summary["alive_dead_f1"],
            "team_acc_alive": float(np.mean(nested_ta)) if tune_calibration else oof_default_summary["team_acc_alive"],
            "spurious_pre_stop_total": int(np.sum(nested_spur)) if tune_calibration else oof_default_summary["spurious_pre_stop_total"],
        },
        "duration_bin_recall": bin_recall,
        "calibration": asdict(best_global_params),
        "top_dead_features": [(n, float(g)) for n, g in dead_imp],
        "top_team_features": [(n, float(g)) for n, g in team_imp],
        "top_turn_features": [(n, float(g)) for n, g in turn_imp],
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_model_dir, "cv_report.json"), "w", encoding="utf-8") as fh:
        json.dump(cv_report, fh, indent=2)

    print(f"\nSaved production v5 3-GBM bundle + CV report -> {out_model_dir} (total {time.time()-t0:.1f}s)", flush=True)
    return cv_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Train & Cross-Validate 3-Stage Possession LightGBM on DFL Data")
    parser.add_argument("--training-dir", default=None, help="Path to data/training directory")
    parser.add_argument("--out-dir", default=DEFAULT_MODEL_DIR, help="Output model bundle directory")
    parser.add_argument("--no-augment", action="store_true", help="Disable optical domain-adaptation augmentation")
    parser.add_argument("--no-tune-calibration", action="store_true", help="Skip nested Viterbi calibration")
    args = parser.parse_args()

    run_training_and_cv(
        training_dir=args.training_dir,
        out_model_dir=args.out_dir,
        use_augmented=not args.no_augment,
        tune_calibration=not args.no_tune_calibration,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
