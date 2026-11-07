"""Shared foundation for all possession models (More-than-a-Hack 2027).

Modules:
    data_external.py     loaders for the EXTERNAL matches under data/altre_partite/
    half_blocks.py       half-time split detection from tracking only
    features_base.py     the shared v2 feature base (all models use tag "v2")
    context.py           context-tier features (referee, score, phase)
    target_match.py      loader for the COMPETITION match (read-only, never trained on)
    cache.py             npz feature/posterior caches + cache-name helpers
    decoder_support.py   provided Viterbi decoder + helper utilities

Every model package imports from `models.modules.*`; nothing here is model-specific.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
