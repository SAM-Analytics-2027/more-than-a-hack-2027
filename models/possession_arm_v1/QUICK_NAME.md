possession_arm_v1 — XGBoost GBDT (feature set v2) + provided duration-constrained Viterbi.
Stage 1: per-frame H/A/D posterior from tracking-only features, native NaN handling, GPU hist.
Stage 2: context-tiered transition costs + minimum segment durations, learned ONLY on external matches.
External nested-LOO mean spellF1 = 0.6298 (macroF1 0.9182); public-GT sanity = 0.5029.
