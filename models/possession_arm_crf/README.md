# possession_arm_crf — duration-state CRF (expanded linear chain) + shared Viterbi

Model family 5 of the branch-AM sweep. Two stacked parts, both external-only:

1. **Emissions**: a frame-level MLP (218 -> 64 -> 3) on the SAME v2 features,
   split into filled + missing-mask channels; balanced class weights like v1;
   epoch-based early stopping on the NEXT fold (patience 6 / floor 8 / max 40,
   the documented mapping of the GBDT rounds; seed 17).
2. **Duration chain**: the exact linear chain over (class, duration-regime)
   states — a spell picks its log-spaced length bucket at start and holds it,
   ending each frame with per-frame hazard h[c,k]. Hazards, start regimes sb,
   spell transitions A and prior pi are learned from the TRAIN folds' clean GT
   spells ONLY (mixture-of-geometrics; the within-regime age dependence is
   collapsed in exchange for a trivial 3x24-state inference). Forward-backward
   runs per half block; CLASS MARGINALS go to the shared Viterbi.

| step | script | notes |
|---|---|---|
| 1 | `models/modules/cache.py --force` | shared v2 feature cache (already built) |
| 2 | `python models/possession_arm_crf/eval_external.py` | nested LOO via the shared protocol; posteriors → `data/_arm_cache/looP_v2_possession_arm_crf_*` |
| 3 | `python models/possession_arm_crf/train.py --mode final` | final emission artifact + duration model (`crf_final.pt` + meta), epochs = LOO p80 |
| 4 | `python models/possession_arm_crf/search_decoder.py --rounds 4` | shared coordinate descent on THIS family's posteriors |
| 5 | `python models/possession_arm_crf/predict.py --out evaluation/results/possession_arm_crf.csv` | submission + public-GT sanity |
| 6 | `python evaluation/validate_submission.py evaluation/results/possession_arm_crf.csv` | format check |

`--device cpu|cuda|auto` is available on train/eval/predict (the emission MLP is
small enough to run comfortably on CPU while the GPU serves the TCN family).
Nothing is tuned on the competition data.
