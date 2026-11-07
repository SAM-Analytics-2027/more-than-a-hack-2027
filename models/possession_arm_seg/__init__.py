"""possession_arm_seg -- change-point segmentation + segment voting (paradigm 4).

Instead of classifying frames, this family:
  1. builds a SIGNED possession-evidence trace from a small, fixed set of v2
     columns (verified-control diffs, pass diffs, smoothed copies, stop/visibility
     evidence);
  2. segments the timeline BOTTOM-UP on level shifts of that trace, with forced
     boundaries at dilated verified ball-impact moments and a max segment length;
  3. aggregates segment-level features and classifies each SEGMENT with a small
     modal multinomial logistic (balanced class weights, torch), labels being the
     majority clean-GT code of the segment;
  4. broadcasts the segment posterior to the segment's frames and lets the SHARED
     Viterbi + its per-family coordinate search do the sequence smoothing.

Same external-only protocol as every family (nested LOO via
models/modules/training_protocol, posteriors under looP_v2_possession_arm_seg_*).
"""
