"""Ball Possession Inference Model for More than a Hack 2027.

Combines dual-camera tracking synchronization, ball trajectory & control analysis,
collective team movement & spatial dynamics, Smart Tagging tactical priors, and
duration-constrained Viterbi sequence decoding.
"""

from possession_model.pipeline import run_possession_inference

__all__ = ["run_possession_inference"]
