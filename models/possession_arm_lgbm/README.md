# possession_arm_lgbm — LightGBM GBDT + provided Viterbi (same v2 base as v1)

Model family 2 of the branch-AM model sweep. The ONLY change vs `possession_arm_v1`
is the Stage-1 hypothesis: LightGBM's leaf-wise histogram GBDT instead of XGBoost's
depth-wise `hist`. Everything else is the shared, identical protocol:

| step | script | notes |
|---|---|---|
| 1 | `models/modules/cache.py --force` | shared v2 feature cache (already built) |
| 2 | `python models/possession_arm_lgbm/eval_external.py` | nested LOO via `models/modules/training_protocol`; posteriors → `data/_arm_cache/looP_v2_possession_arm_lgbm_*` |
| 3 | `python models/possession_arm_lgbm/train.py --mode final` | all-10-matches artifact, rounds = LOO p80 |
| 4 | `python models/possession_arm_lgbm/search_decoder.py --rounds 4` | shared coordinate descent on THIS family's posteriors |
| 5 | `python models/possession_arm_lgbm/predict.py --out evaluation/results/possession_arm_lgbm.csv` | submission + public-GT sanity |
| 6 | `python evaluation/validate_submission.py evaluation/results/possession_arm_lgbm.csv` | format check |

Hyper-parameters are set on the same scale as v1 (eta 0.08, 127 leaves vs depth 8,
min_data_in_leaf 5 vs min_child_weight 5, 0.95/0.75 sampling, lambda 2, max_bin 512);
GPU is probed and falls back to CPU. Nothing is tuned on the competition data.
