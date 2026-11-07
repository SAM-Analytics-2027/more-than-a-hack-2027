"""Emission MLP for the duration-state CRF family.

Frame-level (NO temporal windows -- temporal structure lives in the expanded
chain, that is the paradigm). Input: v2 features split into filled (NaN->0) and
missing-mask channels, like the TCN family for identical NaN semantics.
"""

from typing import Tuple

import torch
import torch.nn as nn


def window_split(X: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
    """[n, C] float32 features with NaNs -> (filled, mask) [2 x n, C]."""
    mask = torch.isnan(X).to(X.dtype)
    filled = torch.nan_to_num(X, nan=0.0)
    return filled, mask


class EmissionMLP(nn.Module):
    """218 -> hidden -> 3 logits; small and smooth by design."""

    def __init__(self, in_ch: int = 218, hidden: int = 64, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, filled: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
        x = torch.cat([filled, mask], dim=1)
        return self.net(x)
