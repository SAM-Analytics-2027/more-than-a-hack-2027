# possession_arm_rule — the provided rule pipeline as a fit-free posterior family

Model family 9. The provided pipeline documents its per-frame negative
log-likelihood costs exactly (`possession_model.pipeline.
compute_frame_and_transition_costs`: `frame_costs = -log P(s | x_t)`), so
`P = exp(-frame_costs)` is a posterior family — no fit anywhere. The rule is
TAGLESS: Smart Tagging does not exist on the external matches, so the tactical
arrays carry the exact no-events defaults of `compute_tactical_tag_priors`
(zeros + `stop_impulse = exit_impulse.copy()` + `restart_team = NOBODY`), and
the SAME defaults are used on the competition match at predict time. One
identical rule for training files and the target.

Inputs are the shared v2 feature base's intermediate dicts (`side` = ball_data
/ dynamics / control, produced by `models/modules/features_base.py` with the
same call sequence as `run_possession_inference`; the broadcast camera is not
used, per feature-base policy). The build drift-checks the recomputed X
against the cache (`build_posteriors.py`, atol 2e-5) and uses the tracking-based
half split exactly like `cache.py build_one`.

## Results (external nested LOO, 20 held-out files, shared metric)

| step | mean spellF1 | macroF1 | notes |
|---|---|---|---|
| default decoder (min_pos 45, min_stop 85) | 0.4701 | 0.8291 | over-merges: 268–333 pred spells vs 370–511 GT |
| own decoder search (4 rounds) | **0.4770** | 0.8325 | min_pos 35, min_stop 120, turn_quiet 54, stop_quiet 32, restart_quiet 25, w_prior 0.25, turn_control 9, stop_loud 5.5, restart_loud 0 |

The family is weak standalone (the rule's merge/stop logic produces far longer
spells than the touch-granular external GT wants), but it carries the strongest
dead-ball detector in the repo (the provided `z_dead` recipe) and its
competition-target annotation style matches the provided pipeline's
(56.5/88.9 on the public slice for the with-tags repro). Therefore its role is
NOT as a standalone submission but as an ingredient:

| ensemble screen (default decoder) | composition | mean spellF1 |
|---|---|---|
| probability-space mean | trio + rule (equal) | 0.6158 |
| probability-space `--weights 1 1 1 0.4286` | rule w ≈ 0.125 | 0.5416* |
| **log-space `--log-blend --weights .3167 .3167 .3167 .05`** | rule γ = 0.05 | **0.6367** vs trio 0.6324 |

*superseded recording: weight miscount; explicit-weight screens 0.5864 (γ≈0.125)
and 0.5250 (γ≈0.375) — probability averaging dilutes the rule's peaky D signal.
The geometric blend preserves the trio's log-dominance and was screened at
γ ∈ {0.025…0.3}: peak at γ = 0.05 (0.6366/0.6367/0.6361/0.6356/0.6355/0.6323/
0.6243 for γ = 0.025/0.05/0.065/0.075/0.1/0.2/0.3).

## Layout

| file | role |
|---|---|
| `rule_costs.py` | tagless-tactical defaults + `rule_posterior(side)` via the provided cost function |
| `build_posteriors.py` | static npz for all 20 external files (+ X cache-drift check) |
| `train.py` | fit-free adapter (`prediction_only_adapter`, `predict_final(X, side)`) |
| `eval_external.py` | shared nested-LOO engine over the static posteriors |
| `search_decoder.py` | shared coordinate descent (external only) → `artifacts/decoder_params.json` |
| `predict.py` | competition target (+ public-GT sanity report) |

The rule is deterministic and per-file: LOO posteriors equal the static ones,
so the nested protocol holds trivially (no fold-time fitting anywhere).
