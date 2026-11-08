possession_arm_gate — situation-gated meta-blend over the production trio (family 8).
Inputs: each ingredient's OWN protocol-compliant held-out posterior (cache, blind per fold)
+ its entropy + the six context tiers; blend learned per frame by regularized
multinomial logistic regression (C fixed, no extra external tuning).
Double nesting: ingredient LOO posteriors are out-of-sample for their own match, and the
gate itself re-runs the same 10-fold protocol through the shared engine.
Same shared Viterbi + own external coordinate search as every other family.
Competition: gate artifact blends the trio's FINAL artifacts on the shared v2 features.
