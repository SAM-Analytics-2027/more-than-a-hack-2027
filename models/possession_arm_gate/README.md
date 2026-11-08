# possession_arm_gate — situation-gated meta-blend (family 8, investigation)

The natural next step after the equal-weight ensemble: same three production
ingredients (lgbm + catboost + crf), but a SMALL learned model recomposes their
posteriors per frame as a function of the situation, instead of a fixed 1/3
mean. The situation signal is the six decoder-context tiers the shared decoder
already consumes (impact_loud, control_loud, in_occlusion, stop_strong,
stop_lowtempo, restart_cue), plus each ingredient's own normalized entropy.

## Leakage geometry (why this is protocol-clean)

- the gate's training inputs are each ingredient's CACHED nested-LOO posterior
  (`looP_v2_possession_arm_lgbm_*` etc.), which by construction is out-of-sample
  for the training file's own match;
- the gate itself re-runs the identical 10-fold engine
  (models/modules/training_protocol.run_nested_loo): fold i fits on 18 files and
  scores fold i only (`looP_v2_possession_arm_gate_*`, own tag);
- the shared external-only decoder-parameter search then tunes the decode on
  those gate posteriors; no competition/public-GT signal enters anywhere.

## Design variants tried (default decoder params, external LOO mean spellF1)

| variant | features | C | LOO default |
|---|---|---|---|
| additive gate | logP(9) + ent(3) + ctx(6) | 0.3 | 0.6296 |
| cross gate (MoE) | + ctx x logP interactions (54) | 0.2 | 0.6293 |
| mixture gate (GPU) | prob-space convex per-class router over ctx | — | 0.6259 |
| (reference) equal-weight trio ensemble | — | — | 0.6324 |

All three gate variants land within ~6 thousandths of the fixed mean at default
parameters and below it: with only 10 matches the learned trust does not
extract more signal than the ingredients' posteriors already encode. The
production form of this family stays the CROSS gate (best default with the
real situation-conditioning behaviour); mixture.py records the negative
prob-space result.

Post-search (external-only coordinate search `--rounds 4` on the cross gate's
LOO posteriors; converged after round 1, see `artifacts/decoder_search_log.json`):

| variant | LOO default | LOO post-search | decoder params found |
|---|---|---|---|
| cross gate (MoE) | 0.6293 | **0.6495** | min_pos 25, min_stop 55, turn_quiet 20, turn_loud 1.5, turn_control 9, turn_occlusion 30, w_prior 0.0 |
| (reference) equal-weight trio ensemble | 0.6324 | **0.6543** | min_pos 18, min_stop 55, w_prior 0.5 |

VERDICT: 0.6495 < 0.6543 — the gate is NOT the submission;
`possession_arm_ens` stays the production model. Honest close-out: the
situation-conditioned trust learns nothing beyond what the ingredients'
posteriors already encode at this sample size (10 matches, ~3.3M labeled
frames). The family is kept as a documented negative result with complete,
reproducible artifacts.

Competition-time reproduction: the gate artifact
(`artifacts/gate_final.npz` + `_meta.json`, fitted on all 20 files) blends the
ingredients' FINAL artifacts with the same features recorded in the artifact
metadata. NOTE the honest caveat: ingredient LOO posteriors are cooler than
final-artifact posteriors (more training matches, less shrinkage); the
ensemble's arithmetic mean is robust to this, the gate's log-linear blend may
be less so — the public-GT sanity in `predict.py` reports the effect (report
only, never a tuning signal).

## Run book

| step | script |
|---|---|
| LOO | `python models/possession_arm_gate/eval_external.py [--interactions --C 0.2]` |
| decoder search | `python models/possession_arm_gate/search_decoder.py --rounds 4` |
| final gate fit | `python models/possession_arm_gate/train.py --mode final [--interactions --C 0.2]` |
| submission | `python models/possession_arm_gate/predict.py` |
| format check | `python evaluation/validate_submission.py evaluation/results/possession_arm_gate.csv` |

Dependencies: sklearn (1.9.1, pip wheel installed offline from files.pythonhosted.org).
