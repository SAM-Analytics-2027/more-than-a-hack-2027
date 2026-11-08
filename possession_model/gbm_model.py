"""3-Stage Cascaded LightGBM Possession Bundle & Pure Probabilistic Viterbi Coupling.

Architecture:
  1. Dead-Ball Booster: P(D | X_dead)
  2. Spell-Balanced H <-> A Equivariant Team Booster:
         P_sym(H | Alive, X_team) = 0.5 * (f(X_team) + 1 - f(swap(X_team)))
  3. Team-Symmetric Turnover Boundary Booster:
         P(turnover | X_turn), trained with exact importance-weighted negative subsampling
         so predicted probabilities require zero manual post-hoc shift.
  4. Pure Probabilistic Viterbi Cost Coupling:
         Every frame emission cost and transition penalty is a negative log-probability
         scaled/shifted solely by `CalibrationParams` fitted on external DFL matches.
"""

import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np

from possession_model.gbm_features import swap_team_feature_matrix
from possession_model.sequence_decoder import decode_possession_sequence
from possession_model.signal_ops import AWAY, DEAD, HOME, box_mean_blocks, logit, sigmoid, softplus


@dataclass
class CalibrationParams:
    """Probabilistic emission & transition calibration parameters learned via 7-fold LOMO-CV on DFL."""
    scale_dead: float = 1.00
    bias_dead: float = 0.00
    scale_team: float = 1.55
    alpha_turn: float = 6.00
    beta_turn: float = 1.80
    alpha_stop: float = 3.50
    beta_stop: float = 3.00
    beta_rest: float = 2.50
    min_live_frames: int = 18
    min_dead_frames: int = 75
    min_pre_stop_frames: int = 55
    pen_pre_stop: float = 6.00


DEFAULT_DEAD_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.045,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 260,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 2.0,
    "reg_lambda": 8.0,
    "max_bin": 255,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "seed": 2027,
    "feature_fraction_seed": 2027,
    "bagging_seed": 2027,
    "verbose": -1,
}

DEFAULT_TEAM_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.045,
    "num_leaves": 63,
    "max_depth": 7,
    "min_child_samples": 140,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.72,
    "reg_alpha": 1.5,
    "reg_lambda": 5.0,
    "max_bin": 255,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "seed": 2027,
    "feature_fraction_seed": 2027,
    "bagging_seed": 2027,
    "verbose": -1,
}

DEFAULT_TURN_LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.045,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 90,
    "subsample": 0.80,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "max_bin": 255,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 4,
    "seed": 2027,
    "feature_fraction_seed": 2027,
    "bagging_seed": 2027,
    "verbose": -1,
}


def select_training_frame_indices(
    gt_labels: np.ndarray,
    half_blocks: List[Tuple[int, int, int]],
    stride: int = 5,
    boundary_radius: int = 16,
) -> np.ndarray:
    """Subsample interior frames at `stride` while keeping 25 Hz dense frames within `+/- boundary_radius`
    of every ground-truth transition.
    """
    keep = np.zeros(len(gt_labels), dtype=bool)
    for _, s, e in half_blocks:
        sub = gt_labels[s : e + 1]
        m = len(sub)
        local_keep = np.zeros(m, dtype=bool)
        local_keep[:: max(1, stride)] = True
        chg = np.flatnonzero(sub[1:] != sub[:-1]) + 1
        for c in chg:
            local_keep[max(0, int(c) - boundary_radius) : min(m, int(c) + boundary_radius + 1)] = True
        keep[s : e + 1] = local_keep
    return np.flatnonzero(keep)


def compute_spell_balanced_weights(
    gt_labels: np.ndarray,
    sample_idx: np.ndarray,
    half_blocks: List[Tuple[int, int, int]],
) -> np.ndarray:
    """Compute geometric-mean frame/spell weights w_t = clip(sqrt(L_median / L_spell(t)), 1/3, 3.0)
    so short controlled possessions contribute on par with long spells without hardcoded reference lengths.
    """
    n = len(gt_labels)
    spell_durs: List[int] = []
    spans: List[Tuple[int, int, int]] = []

    for _, s_f, e_f in half_blocks:
        sub = gt_labels[s_f : e_f + 1]
        if len(sub) == 0:
            continue
        chg = np.flatnonzero(sub[1:] != sub[:-1]) + 1
        starts = np.concatenate(([0], chg))
        ends = np.concatenate((chg, [len(sub)]))
        for a, b in zip(starts, ends):
            if sub[a] in ("H", "A"):
                dur = int(b - a)
                spell_durs.append(dur)
                spans.append((s_f + int(a), s_f + int(b), dur))

    med_dur = float(np.median(spell_durs)) if spell_durs else 250.0
    w_full = np.ones(n, dtype=np.float32)
    for g_a, g_b, dur in spans:
        w_full[g_a:g_b] = float(np.clip(np.sqrt(med_dur / max(1.0, float(dur))), 1.0 / 3.0, 3.0))
    return w_full[sample_idx]


def build_turnover_targets(
    gt_labels: np.ndarray,
    half_blocks: List[Tuple[int, int, int]],
    radius_frames: int = 6,
) -> np.ndarray:
    """Binary target: 1.0 within +/- `radius_frames` (0.24 s at 25 Hz) of a live H <-> A turnover."""
    y = np.zeros(len(gt_labels), dtype=np.float32)
    for _, s_f, e_f in half_blocks:
        sub = gt_labels[s_f : e_f + 1]
        chg = np.flatnonzero(
            (sub[1:] != sub[:-1]) & np.isin(sub[:-1], ["H", "A"]) & np.isin(sub[1:], ["H", "A"])
        ) + 1
        for c in chg:
            a = max(0, int(c) - radius_frames)
            b = min(len(sub), int(c) + radius_frames + 1)
            y[s_f + a : s_f + b] = 1.0
    return y


class PossessionGBMBundle:
    """Serializable bundle containing Dead-Ball, Equivariant Team, and Turnover LightGBM models."""

    def __init__(
        self,
        dead_booster: lgb.Booster,
        team_booster: lgb.Booster,
        turn_booster: lgb.Booster,
        dead_feature_names: List[str],
        team_feature_names: List[str],
        turn_feature_names: List[str],
        team_swap_pairs: List[Tuple[int, int]],
        team_negate_cols: List[int],
        calibration: Optional[CalibrationParams] = None,
    ):
        self.dead_booster = dead_booster
        self.team_booster = team_booster
        self.turn_booster = turn_booster
        self.dead_feature_names = list(dead_feature_names)
        self.team_feature_names = list(team_feature_names)
        self.turn_feature_names = list(turn_feature_names)
        self.team_swap_pairs = [tuple(p) for p in team_swap_pairs]
        self.team_negate_cols = [int(i) for i in team_negate_cols]
        self.calibration = calibration or CalibrationParams()

    @staticmethod
    def fit_dead_booster(
        X_dead_list: List[np.ndarray],
        y_dead_list: List[np.ndarray],
        dead_feature_names: List[str],
        n_rounds: int = 380,
        params: Optional[dict] = None,
    ) -> lgb.Booster:
        X_d = np.vstack(X_dead_list)
        y_d = np.concatenate(y_dead_list).astype(np.float32)
        cfg = {**DEFAULT_DEAD_LGB_PARAMS, **(params or {})}
        dset = lgb.Dataset(X_d, label=y_d, feature_name=list(dead_feature_names), free_raw_data=True)
        return lgb.train(cfg, dset, num_boost_round=n_rounds)

    @staticmethod
    def fit_team_booster(
        X_team_list: List[np.ndarray],
        y_home_list: List[np.ndarray],
        w_team_list: List[np.ndarray],
        team_feature_names: List[str],
        team_swap_pairs: List[Tuple[int, int]],
        team_negate_cols: List[int],
        n_rounds: int = 420,
        params: Optional[dict] = None,
    ) -> lgb.Booster:
        X_orig = np.vstack(X_team_list)
        y_orig = np.concatenate(y_home_list).astype(np.float32)
        w_orig = np.concatenate(w_team_list).astype(np.float32)
        X_swap = swap_team_feature_matrix(X_orig, team_swap_pairs, team_negate_cols)

        X_sym = np.vstack([X_orig, X_swap])
        y_sym = np.concatenate([y_orig, 1.0 - y_orig])
        w_sym = np.concatenate([w_orig, w_orig])
        del X_orig, X_swap

        cfg = {**DEFAULT_TEAM_LGB_PARAMS, **(params or {})}
        dset = lgb.Dataset(
            X_sym, label=y_sym, weight=w_sym, feature_name=list(team_feature_names), free_raw_data=True
        )
        return lgb.train(cfg, dset, num_boost_round=n_rounds)

    @staticmethod
    def fit_turn_booster(
        X_turn_list: List[np.ndarray],
        y_turn_list: List[np.ndarray],
        w_turn_list: List[np.ndarray],
        turn_feature_names: List[str],
        n_rounds: int = 260,
        params: Optional[dict] = None,
    ) -> lgb.Booster:
        X_tr = np.vstack(X_turn_list)
        y_tr = np.concatenate(y_turn_list).astype(np.float32)
        w_tr = np.concatenate(w_turn_list).astype(np.float32)
        cfg = {**DEFAULT_TURN_LGB_PARAMS, **(params or {})}
        dset = lgb.Dataset(
            X_tr, label=y_tr, weight=w_tr, feature_name=list(turn_feature_names), free_raw_data=True
        )
        return lgb.train(cfg, dset, num_boost_round=n_rounds)

    @staticmethod
    def predict_dead_with_booster(
        booster: lgb.Booster,
        X_dead: np.ndarray,
        half_blocks: Optional[List[Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        n = len(X_dead)
        out = np.full(n, 0.5, dtype=np.float64)
        for _, s, e in (half_blocks or [(1, 0, n - 1)]):
            out[s : e + 1] = booster.predict(X_dead[s : e + 1], num_threads=1)
        return out

    @staticmethod
    def predict_team_with_booster(
        booster: lgb.Booster,
        X_team: np.ndarray,
        swap_pairs: List[Tuple[int, int]],
        negate_cols: List[int],
        half_blocks: Optional[List[Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        n = len(X_team)
        out = np.full(n, 0.5, dtype=np.float64)
        for _, s, e in (half_blocks or [(1, 0, n - 1)]):
            X_sub = X_team[s : e + 1]
            X_sw = swap_team_feature_matrix(X_sub, swap_pairs, negate_cols)
            pf = booster.predict(X_sub, num_threads=1)
            pr = booster.predict(X_sw, num_threads=1)
            out[s : e + 1] = 0.5 * (pf + (1.0 - pr))
        return out

    @staticmethod
    def predict_turn_with_booster(
        booster: lgb.Booster,
        X_turn: np.ndarray,
        half_blocks: Optional[List[Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        n = len(X_turn)
        out = np.full(n, 1e-3, dtype=np.float64)
        for _, s, e in (half_blocks or [(1, 0, n - 1)]):
            out[s : e + 1] = booster.predict(X_turn[s : e + 1], num_threads=1)
        return out

    def predict_dead_posterior(self, X_dead: np.ndarray, half_blocks: List[Tuple[int, int, int]]) -> np.ndarray:
        return self.predict_dead_with_booster(self.dead_booster, X_dead, half_blocks)

    def predict_team_posterior(self, X_team: np.ndarray, half_blocks: List[Tuple[int, int, int]]) -> np.ndarray:
        return self.predict_team_with_booster(
            self.team_booster, X_team, self.team_swap_pairs, self.team_negate_cols, half_blocks
        )

    def predict_turn_posterior(self, X_turn: np.ndarray, half_blocks: List[Tuple[int, int, int]]) -> np.ndarray:
        return self.predict_turn_with_booster(self.turn_booster, X_turn, half_blocks)

    def save(self, model_dir: str) -> None:
        os.makedirs(model_dir, exist_ok=True)
        self.dead_booster.save_model(os.path.join(model_dir, "dead_ball_gbm.lgb"))
        self.team_booster.save_model(os.path.join(model_dir, "team_poss_gbm.lgb"))
        self.turn_booster.save_model(os.path.join(model_dir, "turn_bound_gbm.lgb"))
        meta = {
            "bundle_format": "v5_cascaded_3gbm",
            "dead_feature_names": self.dead_feature_names,
            "team_feature_names": self.team_feature_names,
            "turn_feature_names": self.turn_feature_names,
            "team_swap_pairs": [list(p) for p in self.team_swap_pairs],
            "team_negate_cols": self.team_negate_cols,
            "calibration": asdict(self.calibration),
        }
        with open(os.path.join(model_dir, "gbm_bundle_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> "PossessionGBMBundle":
        meta_path = os.path.join(model_dir, "gbm_bundle_meta.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"Trained LightGBM bundle not found at {meta_path}. Run `python possession_model/train_gbm.py` first."
            )
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        dead_b = lgb.Booster(model_file=os.path.join(model_dir, "dead_ball_gbm.lgb"))
        team_b = lgb.Booster(model_file=os.path.join(model_dir, "team_poss_gbm.lgb"))
        turn_b = lgb.Booster(model_file=os.path.join(model_dir, "turn_bound_gbm.lgb"))
        valid_fields = {f.name for f in fields(CalibrationParams)}
        cal_dict = {k: v for k, v in meta.get("calibration", {}).items() if k in valid_fields}
        return cls(
            dead_booster=dead_b,
            team_booster=team_b,
            turn_booster=turn_b,
            dead_feature_names=meta["dead_feature_names"],
            team_feature_names=meta["team_feature_names"],
            turn_feature_names=meta["turn_feature_names"],
            team_swap_pairs=[tuple(p) for p in meta["team_swap_pairs"]],
            team_negate_cols=meta["team_negate_cols"],
            calibration=CalibrationParams(**cal_dict),
        )


def compute_probabilistic_sequence_costs(
    p_dead: np.ndarray,
    p_home: np.ndarray,
    p_turn: np.ndarray,
    ball_data: Dict[str, np.ndarray],
    half_blocks: List[Tuple[int, int, int]],
    params: CalibrationParams,
    tag_adjustments: Optional[Dict[str, np.ndarray]] = None,
    fps: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map calibrated posteriors (p_dead, p_home, p_turn) and Empirical Bayes tag LLRs into
    negative log-likelihood frame costs and transition self-information penalties.
    """
    n = len(p_dead)
    oob = ball_data["out_of_bounds"]

    z_dead = params.scale_dead * logit(p_dead) + params.bias_dead
    z_team = logit(p_home)

    if tag_adjustments is not None:
        z_dead = z_dead + tag_adjustments["llr_dead"]
        z_team = z_team + tag_adjustments["llr_team_h"]
        z_dead = np.where(tag_adjustments["pre_kickoff_mask"], np.maximum(z_dead, 6.0), z_dead)

    # FIFA Law 9: ball beyond pitch boundary lines is physically out of play
    z_dead = np.where(oob, np.maximum(z_dead, 4.5), z_dead)
    z_dead = np.clip(z_dead, -10.0, 10.0)
    z_team = np.clip(z_team, -10.0, 10.0)

    c_alive = softplus(z_dead)
    c_dead = softplus(-z_dead)
    frame_costs = np.empty((n, 3), dtype=np.float64)
    frame_costs[:, HOME] = c_alive + params.scale_team * softplus(-z_team)
    frame_costs[:, AWAY] = c_alive + params.scale_team * softplus(+z_team)
    frame_costs[:, DEAD] = c_dead

    # Turnover transition cost from Shannon self-information -log P(turnover | x_t)
    pt_clip = np.clip(np.asarray(p_turn, dtype=np.float64), 1e-4, 0.999)
    turnover_cost = np.maximum(0.5, -params.alpha_turn * np.log(pt_clip) + params.beta_turn)

    # Stoppage entry (Alive -> Dead) and restart (Dead -> Alive) self-information from P(D | x_t)
    pd_post = sigmoid(z_dead).astype(np.float32)
    w_tr = max(1, int(round(0.60 * fps)))
    pd_past = box_mean_blocks(pd_post, w_tr, 0, half_blocks, fill=0.5).astype(np.float64)
    pd_fut = box_mean_blocks(pd_post, 0, w_tr, half_blocks, fill=0.5).astype(np.float64)

    q_stop = np.clip((1.0 - pd_past) * np.maximum(pd_fut, oob.astype(np.float64)), 1e-4, 0.999)
    q_rest = np.clip(pd_past * (1.0 - pd_fut), 1e-4, 0.999)

    stoppage_entry_cost = np.maximum(0.5, -params.alpha_stop * np.log(q_stop) + params.beta_stop)
    restart_cost = np.maximum(0.5, -params.alpha_stop * np.log(q_rest) + params.beta_rest)

    return frame_costs, turnover_cost, stoppage_entry_cost, restart_cost


def decode_match_sequence(
    p_dead: np.ndarray,
    p_home: np.ndarray,
    p_turn: np.ndarray,
    ball_data: Dict[str, np.ndarray],
    half_blocks: List[Tuple[int, int, int]],
    params: CalibrationParams,
    tag_adjustments: Optional[Dict[str, np.ndarray]] = None,
    fps: float = 25.0,
) -> np.ndarray:
    """Decode full-match possession labels ('H', 'A', 'D', 'X') across graded halves."""
    n = len(p_dead)
    fc, tc, sc, rc = compute_probabilistic_sequence_costs(
        p_dead, p_home, p_turn, ball_data, half_blocks, params, tag_adjustments=tag_adjustments, fps=fps
    )
    labels = np.full(n, "X", dtype="<U1")
    code_map = np.array(["H", "A", "D"], dtype="<U1")

    for _, s_f, e_f in half_blocks:
        states = decode_possession_sequence(
            fc[s_f : e_f + 1],
            tc[s_f : e_f + 1],
            sc[s_f : e_f + 1],
            rc[s_f : e_f + 1],
            min_possession_frames=params.min_live_frames,
            min_stoppage_frames=params.min_dead_frames,
            min_pre_stop_frames=params.min_pre_stop_frames,
            pen_pre_stop=params.pen_pre_stop,
        )
        labels[s_f : e_f + 1] = code_map[states]

    return labels
