"""possession_arm_crf -- duration-state CRF sibling (expanded-state linear chain).

Stage 1  frame-level emission MLP (filled + missing-mask channels, 218 -> 64 -> 3)
         trained with balanced class weights on the same v2 features.
Stage 2  a duration-capable linear chain over EXPANDED states (class, dwell bucket):
         hazards come from the training fold's own GT spells (log-spaced buckets),
         spell-to-spell class transitions from the same GT. Inference is a
         log-domain forward-backward over (3 x K) states; the CLASS MARGINALS are
         the posterior handed to the shared Viterbi (stop tiers keep doing the
         boundary cleaning there).

Same external-only protocol (nested LOO via models/modules/training_protocol,
posteriors under looP_v2_possession_arm_crf_*). Seed fixed; nothing tuned on the
competition data.
"""
