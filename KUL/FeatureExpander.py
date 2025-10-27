
"""
Temporal Convolutional Stem ("feature lift-off") for EEG sequences.

Input  shape: [B, T, C_in]  (e.g., C_in=64 electrodes)
Output shape: [B, T, C_out] (default C_out=256)

Design:
- Depthwise Conv1D over time (one filter per channel)
- Pointwise Conv1D to mix channels and expand to C_out
- SiLU activation + LayerNorm + Dropout
- Optional causal mode (left padding only) for streaming
"""

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _TimeConv1d(nn.Module):
    """1D convolution over time with optional left-only padding for causal mode.

    Expects input as [B, C, T].
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1, bias=False, causal=False):
        super().__init__()
        self.causal = causal
        self.k = int(kernel_size)
        pad = 0 if causal else self.k // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=self.k, stride=stride, padding=pad, groups=groups, bias=bias)

    def forward(self, x):
        """x: [B, C, T] → conv(x)."""
        if self.causal:
            x = F.pad(x, (self.k - 1, 0))
        return self.conv(x)


class TemporalStem(nn.Module):
    """Temporal convolutional stem for EEG feature lift-off.

    Args:
        in_ch:   number of input EEG channels (e.g., 64)
        out_ch:  number of output features (e.g., 256)
        k_dw:    depthwise kernel size over time (odd)
        k_pw:    pointwise kernel size (kept at 1)
        dropout: dropout prob after activations
        causal:  if True, use left padding only
        use_ln:  apply LayerNorm over feature dimension
    """
    def __init__(self, in_ch=64, out_ch=256, k_dw=9, k_pw=1, dropout=0.1, causal=False, use_ln=True):
        super().__init__()
        assert k_dw % 2 == 1
        assert k_pw == 1
        self.dw = _TimeConv1d(in_ch, in_ch, kernel_size=k_dw, groups=in_ch, causal=causal)
        self.pw = _TimeConv1d(in_ch, out_ch, kernel_size=k_pw, groups=1, causal=False)
        self.act = nn.SiLU()
        self.use_ln = use_ln
        self.ln = nn.LayerNorm(out_ch) if use_ln else nn.Identity()
        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.dw.conv, self.pw.conv]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """x: [B, T, C_in] → y: [B, T, C_out]."""
        B, T, C = x.shape
        x = x.transpose(1, 2)
        x = self.dw(x)
        x = self.pw(x)
        x = self.act(x)
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = self.drop(x)
        return x


def main():
    
    parser = argparse.ArgumentParser(description="Test TemporalStem on random EEG data")
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--time', type=int, default=512)
    parser.add_argument('--in_ch', type=int, default=64)
    parser.add_argument('--out_ch', type=int, default=256)
    parser.add_argument('--k_dw', type=int, default=9)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--causal', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn(args.batch, args.time, args.in_ch)

    stem = TemporalStem(in_ch=args.in_ch, out_ch=args.out_ch, k_dw=args.k_dw, dropout=args.dropout, causal=args.causal)
    y = stem(x)

    print(f"Input  shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Causal mode:  {args.causal}")

    t_probe = args.time // 2
    if args.causal:
        x2 = x.clone()
        x2[:, t_probe+1:, :] = 0.0
        y1 = stem(x)[:, t_probe, :]
        y2 = stem(x2)[:, t_probe, :]
        diff = (y1 - y2).abs().max().item()
        print(f"Max abs diff at t={t_probe} after zeroing future: {diff:.3e}")

if __name__ == "__main__":
    main()
