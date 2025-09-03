#!/usr/bin/env python3
"""
EEG GraphFormer for envelope reconstruction.

Pipeline
  EEG [B, T, N]
→ TemporalStem: causal 1D time convs → H0 [B, T, d_stem]
→ Linear lift: Lf [B, T, d_lift]
→ Per-node features: concat(broadcast(Lf), raw EEG) → Xin [B, T, N, d_in]
→ GraphEncoder (vectorized fixed-K GATv2 + mixer + gate), memory-friendly via bt_chunk
     returns node states Z [B, T, N, d_model], pooled tokens S [B, T, d_model], adjacency A [N, N]
→ ConformerEncoder (Macaron FFN → causal MHSA with latency-aware bias → depthwise conv → Macaron FFN)
→ Head: per-time regression → envelope ŷ [B, T]

Memory aspects
- Uses the sparse, vectorized fixed-K GraphEncoder (no dense N×N attention)
- Forward accepts bt_chunk to process B×T in chunks inside the graph encoder
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from FeatureExpander import TemporalStem
from GraphEncoder1 import GraphEncoder


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
    Additive attention bias for causal self-attention.

    bias_mode="rel": learn per-head bias over past offsets 0..max_rel (clamped)
    bias_mode="alibi": per-head linear penalty proportional to past distance
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
                    self.rel[h].copy_(torch.linspace(0.0, -2.0, max_rel + 1))
        elif bias_mode == "alibi":
            base = 2 ** (-8.0 / heads)
            slopes = [base ** h for h in range(heads)]
            self.slopes = nn.Parameter(torch.tensor(slopes).float(), requires_grad=True)
        else:
            raise ValueError("bias_mode must be 'rel' or 'alibi'")

    def forward(self, T, device):
        i = torch.arange(T, device=device)
        j = torch.arange(T, device=device)
        dist = (i.view(T, 1) - j.view(1, T)).clamp(min=0)
        if self.bias_mode == "rel":
            idx = dist.clamp(max=self.max_rel)
            b = self.rel[:, idx]
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
    Conformer convolution module (causal): PW-GLU → depthwise conv → BN → SiLU → PW.
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


class EEGGraphConformer(nn.Module):
    """
    Full model: TemporalStem → GraphEncoder → Conformer → Head.

    forward(eeg, bt_chunk=None) returns (ŷ, A)
      eeg: [B, T, N]
      ŷ  : [B, T]
      A  : [N, N]
    """
    def __init__(self, n_ch, pos, d_stem=256, d_lift=127, d_in=128, d_model=128,
                 L_graph=3, k=8, graph_heads=4, graph_dropout=0.1,
                 conf_depth=4, conf_heads=4, ff_expansion=4, conf_dropout=0.1,
                 kernel_size=9, bias_mode="rel", max_rel=128, causal=True):
        super().__init__()
        if not torch.is_tensor(pos):
            pos = torch.tensor(pos, dtype=torch.float32)
        self.stem = TemporalStem(in_ch=n_ch, out_ch=d_stem, causal=causal, dropout=graph_dropout)
        self.lift = nn.Linear(d_stem, d_lift)
        self.graph = GraphEncoder(pos=pos, d_in=d_in, d_model=d_model, L=L_graph,
                                  k=k, heads=graph_heads, dropout=graph_dropout)
        self.enc = ConformerEncoder(d_model=d_model, depth=conf_depth, heads=conf_heads,
                                    ff_expansion=ff_expansion, dropout=conf_dropout,
                                    kernel_size=kernel_size, bias_mode=bias_mode, max_rel=max_rel)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, eeg, bt_chunk=None):
        H0 = self.stem(eeg)
        Lf = self.lift(H0)
        B, T, _ = H0.shape
        N = eeg.shape[-1]
        Xin = torch.cat([Lf.unsqueeze(2).expand(B, T, N, -1),
                         eeg.unsqueeze(-1)], dim=-1)
        try:
            _, S, A = self.graph(Xin, bt_chunk=bt_chunk)
        except TypeError:
            _, S, A = self.graph(Xin)
        Y = self.enc(S)
        y = self.head(Y).squeeze(-1)
        return y, A


def _ring_positions(n_ch):
    theta = np.linspace(0, 2 * np.pi, n_ch, endpoint=False)
    return np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1).astype(np.float32)


def main():
    torch.manual_seed(0)
    B, T, N = 2, 512, 64
    eeg = torch.randn(B, T, N)
    pos = _ring_positions(N)
    model = EEGGraphConformer(
        n_ch=N, pos=pos,
        d_stem=256, d_lift=127, d_in=128, d_model=128,
        L_graph=3, k=8, graph_heads=4, graph_dropout=0.1,
        conf_depth=3, conf_heads=4, ff_expansion=4, conf_dropout=0.1,
        kernel_size=9, bias_mode="rel", max_rel=64, causal=True
    )
    y, A = model(eeg, bt_chunk=128)
    print("ŷ:", y.shape, "A:", A.shape)


if __name__ == "__main__":
    main()
