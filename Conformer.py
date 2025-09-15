#!/usr/bin/env python3
"""
Conformer encoder with causal MHSA and latency-aware bias.

Blocks:
- MacaronFFN (pre)
- Causal MHSA with latency-aware bias:
    bias_mode="rel": learn per-head bias for past offsets 0..max_rel (clamped for longer lags)
    bias_mode="alibi": per-head linear penalty proportional to distance to the past
- Conformer ConvModule: PW-GLU -> DepthwiseConv1d (causal) -> BN -> SiLU -> PW
- MacaronFFN (post)
- Pre-norm residual wiring

I/O:
- Input  : X [B, T, d_model]
- Output : Y [B, T, d_model]

Run this file to test on random input.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MacaronFFN(nn.Module):
    """
    Macaron-style FFN with 0.5 residual scaling.
    """
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, expansion * d_model)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(expansion * d_model, d_model)
        self.res_scale = 0.5

    def forward(self, x):
        y = self.ln(x)
        y = self.fc2(self.drop(self.act(self.fc1(y))))
        return x + self.res_scale * self.drop(y)


class LatencyAwareBias(nn.Module):
    """
    Builds an additive attention bias for causal self-attention.

    bias_mode="rel":
      learn per-head biases b[h, d] for past offsets d=0..max_rel
      bias[i,j] = b[h, min(i-j, max_rel)] for j<=i else 0

    bias_mode="alibi":
      per-head slopes s[h] >= 0
      bias[i,j] = -s[h] * (i - j) for j<=i else 0
    """
    def __init__(self, heads, max_rel=128, bias_mode="rel"):
        super().__init__()
        self.heads = heads
        self.max_rel = max_rel
        self.bias_mode = bias_mode
        if bias_mode == "rel":
            self.rel = nn.Parameter(torch.zeros(heads, max_rel + 1))
            with torch.no_grad():
                for h in range(heads):
                    decay = torch.linspace(0.0, -2.0, max_rel + 1)
                    self.rel[h].copy_(decay)
        elif bias_mode == "alibi":
            slopes = []
            base = 2 ** (-8.0 / heads)
            for h in range(heads):
                slopes.append(base ** h)
            self.slopes = nn.Parameter(torch.tensor(slopes).float(), requires_grad=True)
        else:
            raise ValueError("bias_mode must be 'rel' or 'alibi'")

    def forward(self, T, device):
        i = torch.arange(T, device=device)
        j = torch.arange(T, device=device)
        dist = (i.view(T, 1) - j.view(1, T)).clamp(min=0)
        if self.bias_mode == "rel":
            idx = dist.clamp(max=self.max_rel)
            b = self.rel[:, idx]  # [H, T, T]
        else:
            b = -self.slopes.view(self.heads, 1, 1) * dist.view(1, T, T)
        return b


class CausalMHSA(nn.Module):
    """
    Multi-head self-attention with causal masking and latency-aware bias.
    """
    def __init__(self, d_model, heads=4, dropout=0.1, bias_mode="rel", max_rel=128):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads
        self.dh = d_model // heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.bias = LatencyAwareBias(heads=heads, max_rel=max_rel, bias_mode=bias_mode)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x0 = x
        x = self.ln(x)
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        q = q / math.sqrt(self.dh)
        scores = torch.matmul(q, k.transpose(-1, -2))
        bias = self.bias(T, x.device).unsqueeze(0)
        scores = scores + bias
        mask = torch.ones(T, T, device=x.device).triu(1).bool()
        scores = scores.masked_fill(mask.view(1, 1, T, T), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        y = self.o(y)
        y = self.drop(y)
        return x0 + y


class ConformerConvModule(nn.Module):
    """
    Conformer convolution module (causal):
    PW-GLU → DepthwiseConv1d → BN → SiLU → PW.
    """
    def __init__(self, d_model, kernel_size=9, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.pw_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1, bias=True)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model, bias=True)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw_out = nn.Conv1d(d_model, d_model, kernel_size=1, bias=True)
        self.drop = nn.Dropout(dropout)
        self.ks = kernel_size

    def forward(self, x):
        x0 = x
        x = self.ln(x)
        x = x.transpose(1, 2)
        z = self.pw_in(x)
        a, b = torch.chunk(z, 2, dim=1)
        x = a * torch.sigmoid(b)
        x = F.pad(x, (self.ks - 1, 0))
        x = self.dw(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pw_out(x)
        x = x.transpose(1, 2)
        x = self.drop(x)
        return x0 + x


class ConformerBlock(nn.Module):
    """
    Conformer block: FFN → MHSA → Conv → FFN.
    """
    def __init__(self, d_model, heads=4, ff_expansion=4, dropout=0.1,
                 kernel_size=9, bias_mode="rel", max_rel=128):
        super().__init__()
        self.ff1 = MacaronFFN(d_model, expansion=ff_expansion, dropout=dropout)
        self.mhsa = CausalMHSA(d_model, heads=heads, dropout=dropout,
                               bias_mode=bias_mode, max_rel=max_rel)
        self.conv = ConformerConvModule(d_model, kernel_size=kernel_size, dropout=dropout)
        self.ff2 = MacaronFFN(d_model, expansion=ff_expansion, dropout=dropout)

    def forward(self, x):
        x = self.ff1(x)
        x = self.mhsa(x)
        x = self.conv(x)
        x = self.ff2(x)
        return x


class ConformerEncoder(nn.Module):
    """
    Stack of Conformer blocks.
    """
    def __init__(self, d_model=128, depth=4, heads=4, ff_expansion=4,
                 dropout=0.1, kernel_size=9, bias_mode="rel", max_rel=128):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, heads=heads, ff_expansion=ff_expansion,
                           dropout=dropout, kernel_size=kernel_size,
                           bias_mode=bias_mode, max_rel=max_rel)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def main():
    torch.manual_seed(0)
    B, T, D = 2, 256, 128
    x = torch.randn(B, T, D)
    enc_rel = ConformerEncoder(d_model=D, depth=3, heads=4, dropout=0.1,
                               kernel_size=9, bias_mode="rel", max_rel=64)
    y_rel = enc_rel(x)
    print("rel:", y_rel.shape)
    enc_alibi = ConformerEncoder(d_model=D, depth=3, heads=4, dropout=0.1,
                                 kernel_size=9, bias_mode="alibi", max_rel=64)
    y_alibi = enc_alibi(x)
    print("alibi:", y_alibi.shape)


if __name__ == "__main__":
    main()
