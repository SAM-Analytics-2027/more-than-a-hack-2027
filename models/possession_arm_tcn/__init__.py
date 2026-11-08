"""possession_arm_tcn -- temporal CNN (causal dilated TCN) sibling of possession_arm_v1.

Stage 1  a 1D causal dilated TCN trained on WINDOWS of the identical v2 feature
         matrix (NaNs -> 0 + explicit missing-mask channels), producing the H/A/D
         posterior of each CENTER frame; temporal context is learned end-to-end
         instead of being hand-crafted as in the v2 features.
Stage 2  the same duration-constrained Viterbi decoder core, parameters found by
         the shared external-only coordinate search on THIS family's LOO
         posteriors.

Same external-only protocol as every family: nested LOO with early stopping on
the NEXT fold, repository metric, no competition asset anywhere.
"""
