# possession_arm_catboost — CatBoost GBDT + provided Viterbi (same v2 base as v1)

Model family 6 of the branch-AM sweep. The ONLY change vs `possession_arm_v1`
is the Stage-1 hypothesis: CatBoost's oblivious (symmetric) trees with native
NaN routing instead of XGBoost's depth-wise `hist`. Everything else is the
shared, identical protocol.

| step | script | notes |
|---|---|---|
| 1 | `models/modules/cache.py --force` | shared v2 feature cache (already built) |
| 2 | `python models/possession_arm_catboost/eval_external.py` | nested LOO via `models/modules/training_protocol`; posteriors → `data/_arm_cache/looP_v2_possession_arm_catboost_*` |
| 3 | `python models/possession_arm_catboost/train.py --mode final` | all-10-matches artifact, rounds = LOO p80 |
| 4 | `python models/possession_arm_catboost/search_decoder.py --rounds 4` | shared coordinate descent on THIS family's posteriors |
| 5 | `python models/possession_arm_catboost/predict.py --out evaluation/results/possession_arm_catboost.csv` | submission + public-GT sanity |
| 6 | `python evaluation/validate_submission.py evaluation/results/possession_arm_catboost.csv` | format check |

Hyper-parameters on the same scale as v1 (eta 0.08, depth 8, l2 2.0, subsample
0.95 via Bernoulli bootstrap, border_count 512, random_seed 2027); the balanced
class weights of v1 are passed as CatBoost `class_weights`. CUDA is probed with
CPU fallback. Nothing is tuned on the competition data.
