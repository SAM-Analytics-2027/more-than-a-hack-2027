"""One sub-package per ball-possession model, More-than-a-Hack 2027.

Layout:
    models/modules/            shared foundation (external loaders, half blocks,
                               feature base, context tiers, cache, decoder core,
                               target loader). Feature set is v2 for all models.
    models/<model_name>/       the model's own code + its artifacts/ + QUICK_NAME.md
                               + README.md

Everything is runnable directly (scripts insert the repository root on sys.path):

    python models/<model_name>/train.py --mode final
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
