"""possession_arm_catboost -- CatBoost sibling of possession_arm_v1 (third GBDT).

Stage 1  CatBoost multiclass (oblivious trees, native NaN routing) posterior from
         the identical v2 feature matrix; balanced class_weights like v1.
Stage 2  the same duration-constrained Viterbi decoder core, parameters found by
         the shared external-only coordinate search on THIS family's own LOO
         posteriors.

Same external protocol: nested LOO with early stopping on the NEXT fold via
models/modules/training_protocol; repository metric; no competition asset.
"""
