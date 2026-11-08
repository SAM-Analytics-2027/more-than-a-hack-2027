possession_arm_lgbm — LightGBM GBDT (leaf-wise) on the SAME v2 features + same Viterbi decoder.
Purpose: inductive-bias sibling of possession_arm_v1's depth-wise XGBoost; identical external protocol.
Fit: balanced class weights, early stop on the NEXT fold's multi_logloss (patience 40, floor 150 rounds).
Decoder params learned only from this family's own external LOO posteriors (shared coordinate search).
