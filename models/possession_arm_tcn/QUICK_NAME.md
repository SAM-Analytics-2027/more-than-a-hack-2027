possession_arm_tcn — causal dilated TCN (torch) learned directly on v2-feature WINDOWS, same Viterbi.
Stage 1: NaN -> 0 + missing-mask channels; window W=33 (1.3 s), receptive field 125 frames (5 s).
Stage 2: same decoder core; params from the shared external search on this family's LOO posteriors.
Epoch-based early stop (documented mapping of the GBDT rounds) on the NEXT external fold.
CUDA 2.11 training, 202k params, seed 17; no competition asset anywhere.
