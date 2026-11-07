possession_arm_crf — duration-state CRF: emission MLP + expanded (class, duration-regime) chain.
72-state log-domain forward-backward per half block; hazards/A/pi/start-regimes from train-fold GT spells.
Emissions: 218→64→3 MLP (filled + missing-mask), balanced CE, epoch early stop on the NEXT fold.
Class marginals feed the SHARED Viterbi with this family's own external coordinate-search parameters.
CPU/GPU torch, seed 17; trained and tuned strictly on the 10 external matches.
