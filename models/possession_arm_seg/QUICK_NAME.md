possession_arm_seg — change-point segmentation + segment voting, no per-frame classifier.
Bottom-up segmentation on a signed possession-evidence trace (verified-control/pass diffs).
Forced boundaries at dilated verified ball impacts; segment cap 12 s; merge penalty 0.30.
Per-segment features -> multinomial logistic (torch) trained on external segment GT majority.
Segment posteriors broadcast per frame, then the SHARED Viterbi + own external coordinate search.
