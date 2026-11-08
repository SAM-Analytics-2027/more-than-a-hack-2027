"""possession_arm_rule — the provided rule-based pipeline as a posterior family.

Unlike every other family there is NO fit: the posterior is the exact provided
pipeline's frame-cost recipe P = exp(-frame_costs), computed on the shared v2
feature base's intermediate dicts (ball_data / dynamics / control) with the
EXACT no-Smart-Tagging tactical defaults that external matches imply
(compute_tactical_tag_priors with empty events: zeros + stop_impulse =
exit_impulse copy + restart_team = NOBODY). The same functional form runs on
the competition match at predict time, so training and target share one rule.
"""
