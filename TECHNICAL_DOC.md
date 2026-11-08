> Rename to `<team_name>.md` (or convert to PDF) before the form upload; see rules/SUBMISSION.md.

# Two-Stage Ball-Possession Model for BAR-ATH — possession_arm_v1

More-than-a-Hack 2027 — Phase 1 deliverable. The data-preparation workstream material is
acknowledged below; the model, the code and all numbers in this document are reproducible
from the public repository.

## 1. The approach, and why

The competition asks for a per-frame label (H/A/D/X) over the whole tactical video of
Barça–Athletic, scored primarily by **spellF1**: possession spells are extracted from the
prediction and matched against the ground-truth spells across durations, so the *sequence
structure* matters as much as the per-frame answer.

We built an independent two-stage model:

**Stage 1 — per-frame posterior.** A gradient-boosted decision tree (XGBoost,
`multi:softprob`, native NaN handling, GPU `tree_method=hist / device=cuda`) maps a
feature vector built from the automatic tracking to `P(H, A, D | x_t)`.

**Stage 2 — duration-constrained sequence decode.** The frame posteriors become
`C_frame(t, s) = -log P(s | x_t)` (optionally shifted by a learned class prior), and an
exact dynamic program finds the globally optimal state sequence under minimum-duration
constraints:

    Cost(y) = Σ_t C_frame(t, y_t) + Σ_t C_trans(tier(t), y_{t-1}, y_t)

subject to minimum lengths for possession spells (H/A) and dead-ball intervals (D). The
DP itself is the **provided** duration-constrained Viterbi (`possession_model.sequence_decoder`),
reused verbatim — one code path, no re-derivation. What we changed is everything around it:
in the provided baseline the transition penalties are a dozen hand-set constants (e.g.
turnover cost 4.5/13.5/54/42 across four context tiers); in our version **every one of these
constants is a learned parameter**, fitted by coordinate descent on held-out data of the
external matches (Section 5). The frame costs also gain an explicit, learned class prior
weight `w_prior`, which stabilises the dead-ball class.

Two design choices worth stating up front, both forced by the compliance rules:

- **Tactical camera only.** The broadcast camera exists only for the competition match, so
  fusing it into features would put the target outside the training distribution. The two
  cameras never mix anywhere in the model.
- **No Smart Tagging.** The provided Smart Tagging is legal input (rules/FAQ.md), but the
  external training matches have none — tag-derived features would again be out of
  distribution at serve time.

## 2. What runs when a frame is scored

For the competition match (`models/possession_arm_v1/predict.py`):

1. Load tactical ATD with the provided loader semantics (`load_target_match`).
2. Build features for all 199,319 frames (NaN where the ball is unseen — XGBoost routes
   missing values natively; no competition-side interpolation).
3. Build six coarse **context tiers** from tracking only: `impact_loud`,
   `control_loud`, `in_occlusion`, `stop_strong`, `stop_lowtempo`, `restart_cue`.
4. Stage-1 posterior `P(H/A/D/x_t)` with the final artifact
   (`models/possession_arm_v1/artifacts/xgb_final.json`).
5. Stage-2 decode per half (the two published half blocks), stitch; everything outside
   the blocks is `X` (40,462 rows = exactly the ungraded remainder, 199,319 − 158,857).
6. Write the submission CSV; validate with `evaluation/validate_submission.py`.

## 3. Training data (none of it from the provided game)

Every labelled dataset used, with source and who produced it:

| dataset | matches | frames (approx.) | labels | provenance |
|---|---|---|---|---|
| DFL 2022/23, clean + dirtied copies | 7 (2 Bundesliga, 5 2. Bundesliga) | ~1.005 M | per-frame H/A/D, touch-granular (mean spell ≈ 5.6 s, ≈ 500 spells/match) | prepared, cleaned and dirtied by the team's data-preparation workstream from the licensed DFL tracking release («sporcata» replicates the imperfections of the provided tracking); see `data/analisi/LEGGIMI.md` |
| Metrica sample games, clean + dirtied copies | 3 (`Sample_Game_1..3`) | ~0.431 M | per-frame H/A/D, mean spell ≈ 6.6 s | the public Metrica Sports sample dataset, reworked by the same team workstream into the identical ATD format, cleaned then dirtied |

Ground truth is always the **clean** file; the dirtied copy of the same match supplies the
*tracking* the model sees (the competition tracking is imperfect too). Clean and dirtied
versions of one match always share the same cross-validation fold. The provided game and
its ground truth are never in any training, tuning or selection step.

## 4. The possession rules we implemented

The decoder's tier system is where the football rules live. Every boundary below is a
learned parameter (coordinate search, Section 5); the *tier definitions* are fixed
tracking geometry/biomechanics, identical for training and the competition match.

- **Possession (H/A) spells.** A direct H↔A flip costs least right after a verified
  kinematic ball impact (= kick/challenge contact), more inside verified controlled
  possession, more inside a same-team occlusion bridge, most in quiet open play.
  Final learned values: 1.5 / 19 / 54 / 20 (frames of evidence tiers, monotone soft
  ordering comes out of the search, not from where we started).
- **Minimum spell lengths.** Possession spell ≥ 18 frames (0.72 s), dead-ball interval
  ≥ 55 frames (2.2 s) — learned, in both cases lower than the provided baseline's
  hand-set 45/85, as expected for touch-granular external labels.
- **Dead ball (D).** Stoppage entry is cheap when the ball is out of bounds, shows a
  boundary-exit signal, or is unseen while the whole collective walks (learned ≈ 3);
  much more expensive otherwise (≈ 23).
- **Restarts.** D→live costs least right after a restart cue (ball observed in play after
  a stop window; learned ≈ 2), otherwise ≈ 18.5 — so the decoder prefers anchoring a
  restart on real ball evidence instead of timing it mid-nowhere.
- **Frame-cost prior.** `w_prior = 0.75` shifts weakly-informed frames toward the
  aggregate external class prior H/A/D = 0.300/0.270/0.430, which stabilises the
  dead-ball class without hand-adjusting anything per match.

## 5. Validation discipline and results

Experiments touch the competition data exactly twice, both read-only:
`predict.py` (write the submission), and the public 5-minute ground truth **as a
sanity report only** — no parameter, model, threshold or decision was ever chosen on it.

Cross-validation is leave-one-match-out over the 10 external matches (20 files). For fold
*i*, the classifier trains on the other 9 matches and early-stops on fold *(i+1) mod 10*,
so the held-out match is touched by neither the gradients nor the round-count selection
(`MIN_BOOST_ROUNDS` floor: 150). Stage-2 parameters are searched against these held-out
posteriors, never against training posteriors.

| configuration | mean spellF1 (20 external files) | macroF1 |
|---|---|---|
| Stage 1 only + default decoder (v1 features) | 0.5973 | 0.9118 |
| v1 features + learned decoder | 0.6148 | 0.9138 |
| v2 features + default decoder | 0.6100 | 0.9163 |
| **v2 features + learned decoder (submitted config)** | **0.6298** | **0.9182** |

v2 features add the outputs of the provided verified-control machinery as classifier
inputs (control scores, pass/occlusion bridges, proximity, verified mask) — the same
evidence the rule-based model hand-scales into logits, now learned. Per-file held-out
spellF1 ranges 0.526–0.795 (worst: dirtied DFL match; best: Metrica clean), so the
configuration is not tuned to any single league's annotation style.

Public-slice sanity of the submitted file: spellF1 0.503 (macroF1 0.776; per-class
H 0.864, A 0.840, D 0.624; 18 predicted spells against the ground truth's 17). For
comparison, the provided rule-based model, reproduced at the current repository HEAD,
scores ≈ 0.565 spellF1 on the same public slice (its saved `possession_v2.csv` file
scores 0.4471 — the file predates the current code; we did not retune it).

**Known gap, stated plainly.** The external ground truths label possession in a
touch-granular style (mean spell 5.6–6.6 s), while the competition ground truth is much
coarser (17 spells in the 300 s public slice, mean ≈ 17.6 s). The decoder is tuned on
external labels, so its duration constraints match the external style; spellF1 across an
annotation-style gap punishes extra boundaries. We did **not** tune any of it on the
public slice (that would break the tuning rule); the honest outcome is that the model is
external-optimal and carries this residual risk privately. The clear fix with more time:
relabel the DFL touch sequences into coarser spells with an explicit, documented
aggregation rule, and re-learn the duration constraint on the new target style.

## 6. Limitations

- **Label-style mismatch** as above — the single biggest residual risk on the private
  ground truth.
- **Class D.** The weakest F1 everywhere (public slice 0.624). D is a "no player controls
  the ball" abstraction; its evidence (ball unobserved + collective walking) saturates
  early, and the prior shift partly trades boundaries for it.
- **No Smart Tagging, no broadcast camera.** Deliberate (out-of-distribution risk); both
  would likely add signal if the external data ever gains them.
- **Feature coverage.** A tracking-only distribution diagnostic shows 76/109 features with
  ≥ 90 % quantile-box coverage of the competition match against pooled dirtied-external
  training data; the misses are upper tails of the control-evidence features (the target
  game has slightly sharper control spikes). No catastrophic skew found.
- **Single artefact per battery.** We fit one GBDT; no seed bagging. Averaging posteriors
  across seeds would smooth stage-1 noise cheaply.
- **Half-time boundary placements** inherit the published block offsets unchanged.

## 7. Reproduction

With the repository checked out and the data bundle in place (`requirements.txt`,
`xgboost` with CUDA):

    python models/modules/cache.py --force                                  # 1. features + context
    python models/possession_arm_v1/eval_external.py                        # 2. nested LOO + posteriors
    python models/possession_arm_v1/train.py --mode final                   # 3. final Stage-1 artifact
    python models/possession_arm_v1/search_decoder.py --rounds 4            # 4. Stage-2 parameter search
    python models/possession_arm_v1/predict.py --out evaluation/results/possession_arm_v1.csv   # 5. submission + sanity
    python evaluation/validate_submission.py evaluation/results/possession_arm_v1.csv # 6. format check

`models/possession_arm_v1/README.md` documents the module map; shared building
blocks for every model live in `models/modules/`; `models/possession_arm_v1/artifacts/`
contains the models, the decoder parameters with provenance, and the search history.
Total wall time for the full chain: under two hours on one RTX 5070 Ti (the Stage-2
search dominates; the provided Viterbi runs ≈ 277 k frames/s, so it is never the
bottleneck).

## 8. Family sweep on branch AM (identical protocol)

From the 2027 refactor on, every model family lives in `models/<name>/` with its own
artifacts and both a `QUICK_NAME.md` and a README; the building blocks they all share
are in `models/modules/`. The comparison protocol is IDENTICAL for every family:

- same v2 feature base (`models/modules/features_base.py`, feature-set tag `v2`);
- same nested leave-one-match-out outer folds over the 10 external matches, clean GT
  in both copies, early stopping on the NEXT fold (`models/modules/training_protocol.py`);
- same Viterbi decoder core and same external-only coordinate search over the shared
  `PARAM_GRID` (`models/modules/decoder_search.py`), on each family's own posteriors;
- the repository metric (`evaluation/possession_metrics.py`, primary taus 12/25);
  every completed family writes `evaluation/results/<name>.csv` for the dashboard.

Selection rule (compliance): the family to submit is the best **external LOO mean
spellF1**; the public 5-minute GT is reported only as a sanity number, never as a
tuning or selection signal.

Completed families so far:

| model | Stage-1 family | external LOO mean spellF1 |
|---|---|---|
| `possession_arm_v1` | XGBoost GBDT (depth-wise hist) | 0.6298 |
| `possession_arm_seg` | bottom-up change-point segmentation + segment voting | 0.4755 |
| `possession_arm_crf` | duration-state CRF: emission MLP (218→64→3) + expanded (class, duration-regime) chain, forward-backward per half block | 0.6448 |
| `possession_arm_tcn` | temporal conv net (window 33, filled+missing-mask channels), posterior = full-file window sweep | 0.5104 |

