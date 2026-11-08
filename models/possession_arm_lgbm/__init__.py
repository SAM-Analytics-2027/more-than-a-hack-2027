"""possession_arm_lgbm -- LightGBM sibling of possession_arm_v1 on the SAME v2 base.

Stage 1  LightGBM GBDT (leaf-wise, histogram) multiclass H/A/D posterior from the
         identical v2 feature matrix; balanced class weights as in v1.
Stage 2  the same duration-constrained Viterbi decoder core, with parameters found
         by the shared external-only coordinate search on THIS family's own LOO
         posteriors.

The comparison with possession_arm_v1 is entirely external: same features, same
folds, same early-stopping rule, same metric.
"""
