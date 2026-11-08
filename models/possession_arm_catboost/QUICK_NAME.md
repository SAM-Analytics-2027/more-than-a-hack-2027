possession_arm_catboost — CatBoost multiclass (oblivious trees) on the SAME v2 features + same Viterbi.
Third GBDT inductive-bias sibling of possession_arm_v1's XGBoost; native NaN routing, depth-8 oblivious trees.
Fit: balanced class_weights, early stop on the NEXT fold (patience 40, floor 150 rounds), seed 2027.
GPU probed with CPU fallback; decoder params learned only from this family's external LOO posteriors.
Train/tune strictly on the 10 external matches; nothing tuned on the competition data.
