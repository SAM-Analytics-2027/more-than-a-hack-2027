"""Causal dilated TCN for the possession task (temporal sibling of the v1 GBDT).

Input per frame: the v2 feature vector, split into
    filled:  NaN -> 0.0                                (C = 109 channels)
    mask:    1.0 where the v2 feature was NaN           (C = 109 channels)
A window of W centered frames (W odd, edge-padded at block boundaries; padded
positions are all-mask) enters a stack of causal dilated conv blocks with
residual connections; the head reads the CENTER timestep -> 3 logits, and the
posteriors sum (softmax) to 1 like every other family's Stage 1.

The window length W is the ONLY extra temporal hand-holding; the network learns
everything else from the external matches.
"""

from typing import Tuple

import torch
import torch.nn as nn


def window_split(X: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
    """[n, C] float32 feature matrix with NaNs -> (filled, mask) [2 x n, C]."""
    mask = torch.isnan(X).to(X.dtype)
    filled = torch.nan_to_num(X, nan=0.0)
    return filled, mask


class CausalBlock(nn.Module):
    """Conv1d (causal, dilated) -> BN -> SiLU -> Dropout + residual projection."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int, kernel: int = 3,
                 dropout: float = 0.10) -> None:
        super().__init__()
        self.pad = (kernel - 1) * dilation          # left-only padding = causal
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        y = nn.functional.pad(x, (self.pad, 0))
        y = self.conv(y)
        y = self.drop(self.act(self.bn(y)))
        return y if self.res is None else y + self.res(x)


class TCN(nn.Module):
    """Small causal TCN: [B, W, C] -> 3 logits at the CENTER frame.

    Dilation ladder (1, 2, 4, 8, 16) with width 96: receptive field
    1 + 2 * (kernel-1) * sum(dilations) = 1 + 4 * 31 = 125 frames (5 s @ 25 fps)
    each side, comfortably wider than the spell-decadal dynamics it must smooth.
    """

    IN_CH = 218          # 109 filled + 109 missing-mask channels

    def __init__(self, in_ch: int = IN_CH, width: int = 96,
                 dilations: Tuple[int, ...] = (1, 2, 4, 8, 16), out: int = 3,
                 dropout: float = 0.10) -> None:
        super().__init__()
        blocks = []
        prev = in_ch
        for d in dilations:
            blocks.append(CausalBlock(prev, width, d, dropout=dropout))
            prev = width
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Conv1d(prev, 64, 1), nn.SiLU(), nn.Conv1d(64, out, 1),
        )

    def forward(self, win: "torch.Tensor") -> "torch.Tensor":
        """win: [B, W, C] -> center-frame logits [B, 3]."""
        x = win.transpose(1, 2)                    # [B, C, W] for Conv1d
        x = self.blocks(x)
        return self.head(x)[:, :, 0]               # center timestep only
