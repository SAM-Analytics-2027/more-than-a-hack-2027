# possession_arm_seg — bottom-up segmentation + segment voting + shared Viterbi

Model family 4: a different PARADIGM. Frames are never classified; the timeline is
segmented on evidence, each SEGMENT is classified once, and the constant posterior
per segment enters the shared decoder. The per-frame families pay per-frame noise;
this family pays segmentation errors, which the decoder's stop/impact tiers then
partially clean.

| step | script | notes |
|---|---|---|
| 1 | `models/modules/cache.py --force` | shared v2 feature cache (already built) |
| 2 | `python models/possession_arm_seg/eval_external.py` | nested LOO via the shared protocol; posteriors → `data/_arm_cache/looP_v2_possession_arm_seg_*` |
| 3 | `python models/possession_arm_seg/train.py --mode final` | final segment classifier JSON artifact |
| 4 | `python models/possession_arm_seg/search_decoder.py --rounds 4` | shared coordinate descent (durations relax relative to the per-frame families) |
| 5 | `python models/possession_arm_seg/predict.py --out evaluation/results/possession_arm_seg.csv` | submission + public-GT sanity |
| 6 | `python evaluation/validate_submission.py evaluation/results/possession_arm_seg.csv` | format check |

Documented constants (all fixed, none tuned on held-outs or the competition data):
L0 = 25 frames (1 s seeds), MAX_SEG = 300 (12 s cap), PENALTY = 0.30 on the median
level-shift of the signed evidence trace, IMPACT_DILATION = 5 frames (the same
0.2 s dilation the provided pipeline uses for impact windows). The evidence trace
is 0.5*ctrl_score_diff + 0.25*sm10_ctrl_score_diff + 0.15*sm40_ctrl_pass_diff +
0.10*ctrl_pass_diff.
