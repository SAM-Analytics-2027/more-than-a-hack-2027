"""possession_arm_v1 -- per-frame ball possession classifier + sequence decoder.

Two stages, trained on EXTERNAL matches only (7 DFL + 3 Metrica, provided clean and
artificially dirtied by data/analisi):

    Stage 1  XGBoost (GBDT, tree_method='hist', device='cuda'):
             multiclass H/A/D posterior per frame from tracking-derived features.
             Native NaN handling: missing ball / missing slots go in as NaN.

    Stage 2  Duration-constrained Viterbi sequence decoder (HMM-style):
             frame costs = -log posterior, transition penalties and minimum segment
             durations learned from the external ground truths -- never from the
             competition match or its ground truth.

Rules compliance (RULES.md / CLARIFICATIONS.md):
  - No hand labels on the provided game; no tuning on the competition ground truth.
  - Every scalar that is not pitch geometry or a documented unit conversion lives in
    constants.json with its provenance (external stat / geometry / forced default).
  - The labels for the competition match are produced end-to-end by this code.
"""
