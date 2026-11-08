# possession_arm_tcn — causal dilated TCN + provided Viterbi (same v2 base)

Model family 3 of the branch-AM sweep. Stage 1 is a temporal end-to-end network
(torch, CUDA) instead of a GBDT: a causal dilated TCN over windows of the SAME v2
features (window 33 frames = 1.3 s; dilations 1/2/4/8/16 -> receptive field
125 frames = 5 s per side; 202k parameters). NaNs become 0 + explicit
missing-mask channels, so "ball unseen" stays an honest input signal.

| step | script | notes |
|---|---|---|
| 1 | `models/modules/cache.py --force` | shared v2 feature cache (already built) |
| 2 | `python models/possession_arm_tcn/eval_external.py` | nested LOO via `models/modules/training_protocol`; posteriors → `data/_arm_cache/looP_v2_possession_arm_tcn_*` |
| 3 | `python models/possession_arm_tcn/train.py --mode final` | all-10-matches artifact, epochs = LOO p80 |
| 4 | `python models/possession_arm_tcn/search_decoder.py --rounds 4` | shared coordinate descent on THIS family's posteriors |
| 5 | `python models/possession_arm_tcn/predict.py --out evaluation/results/possession_arm_tcn.csv` | submission + public-GT sanity |
| 6 | `python evaluation/validate_submission.py evaluation/results/possession_arm_tcn.csv` | format check |

Protocol mapping (documented for honesty): GBDT "rounds" -> epochs; early stopping
watches the NEXT fold's weighted multi-class logloss (patience 6, floor 8, max 40
epochs); balanced class weights as in v1; window edges padded all-missing (mask
= 1), never fabricated ball data; seed 17 fixed. Nothing is tuned on the
competition data.
