# models — More-than-a-Hack 2027, branch AM

One folder per ball-possession model, plus shared building blocks.

    models/modules/           shared foundation used by EVERY model:
                              data_external, half_blocks, features_base (v2),
                              context, target_match, cache, decoder_support.
    models/<model_name>/      one model: its own scripts, its own artifacts/,
                              QUICK_NAME.md (<= 5 lines: what the model is),
                              README.md (how to reproduce it).

## Identical comparison protocol (all models)

- Same v2 feature base from `models/modules/features_base.py` (feature-set tag `v2`).
- Same nested leave-one-match-out outer folds over the 10 external matches
  (`models/modules/data_external.py`), early stopping on the NEXT fold, floor 150 rounds.
- Same repository metric (`evaluation/possession_metrics.py`), primary taus 12/25.
- LOO posteriors are cached per model in `data/_arm_cache/` under
  `looP_v2_<model_tag>_*` so families never overwrite each other's files
  (`models/modules/cache.py: loo_posterior_path(..., model_tag=...)`; the legacy
  untagged name `looP_v2_*` belongs to possession_arm_v1).
- Every completed model writes `evaluation/results/<model_name>.csv` (dashboard
  discovers them) and is committed on branch `AM` as `model_<nome>: <riassunto>`.

## Current models

| model | family | external LOO mean spellF1 |
|---|---|---|
| `possession_arm_v1` | XGBoost GBDT + provided Viterbi | 0.6298 |
| `possession_arm_seg` | change-point segmentation + segment voting | 0.4755 |
| `possession_arm_crf` | duration-state CRF (emission MLP + expanded chain) | 0.6448 |
