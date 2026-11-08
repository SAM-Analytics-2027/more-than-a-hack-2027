possession_arm_ens — posterior-averaging ensemble closer over the family sweep (family 7).
LOO: equal-weight mean of per-family protocol-compliant held-out posteriors; NO new fitts, no leakage.
Shared Viterbi + own external coordinate search on the COMPOSED posteriors; repo metric throughout.
Production blend: lgbm + catboost + crf (selected on external LOO only; see README screen table).
Target predict: equal-weight mean of the selected ingredients' FINAL artifacts on the shared v2 features.
Submission = mean posterior of the production trio by default (--families overrides).
