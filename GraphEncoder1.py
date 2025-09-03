#!/usr/bin/env python3
"""
Memory-efficient Graph Encoder for EEG.

- Builds an anatomical prior A0 (RBF + kNN + row-norm) from sensor positions
- Adds a nonnegative low-rank residual on existing edges
- Fixed mixing: H_mix = A @ H
- Edge-wise GATv2 (O(E) memory) over kNN neighbors
- Per-node Squeeze–Excite gate
- Stack L blocks, return node features, pooled summary, and final A

Input : X [B, T, N, d_in]
Output: Z [B, T, N, d_model], S [B, T, d_model], A [N, N]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_dist(x):
    x2 = (x**2).sum(-1, keepdim=True)
    d2 = x2 + x2.t() - 2 * (x @ x.t())
    d2 = d2.clamp_min(0)
    return d2.sqrt()


def build_anatomical_prior(pos, k=8, sigma=None, self_w=0.1):
    N = pos.size(0)
    D = pairwise_dist(pos)
    if sigma is None:
        with torch.no_grad():
            triu = D[torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)]
            sigma = torch.median(triu).item() + 1e-6
    W = torch.exp(-(D**2) / (2 * (sigma**2)))
    W.fill_diagonal_(0)
    if self_w and self_w > 0:
        W = W + torch.eye(N, device=W.device) * self_w
    topk = torch.topk(W, k=k, dim=-1).values[:, -1].unsqueeze(-1)
    M = W >= topk
    M = M | torch.eye(N, dtype=torch.bool, device=W.device)
    W = W * M.float()
    W = 0.5 * (W + W.t())
    A0 = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
    M = A0 > 0
    return A0, M


def mask_to_edges(mask):
    idx = mask.nonzero(as_tuple=False)
    return idx[:, 0], idx[:, 1]


class GraphMixer(nn.Module):
    """
    Blend A0 with a nonnegative low-rank residual on prior edges:
      A = row_norm( A0 ⊙ (1 + alpha * softplus(PQ^T)) )
    Then mix: H_mix = A @ H
    """
    def __init__(self, A0, mask, rank=8, alpha=0.5):
        super().__init__()
        N = A0.size(0)
        self.register_buffer('A0', A0)
        self.register_buffer('mask', mask)
        self.P = nn.Parameter(torch.randn(N, rank) * 0.05)
        self.Q = nn.Parameter(torch.randn(N, rank) * 0.05)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def blended_A(self):
        R = self.P @ self.Q.t()
        A_delta = F.softplus(R) * self.mask.float()
        A = self.A0 * (1.0 + self.alpha * A_delta)
        A = A / (A.sum(-1, keepdim=True) + 1e-8)
        return A

    def forward(self, H):
        A = self.blended_A()
        return torch.einsum('ij, bjd -> bid', A, H), A


class GATv2Edge(nn.Module):
    """
    Edge-wise GATv2 over neighbor lists (no NxN tensors).
    """
    def __init__(self, d, heads=4, dropout=0.1, use_anat_bias=True, A0=None, mask=None):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        self.lin = nn.Linear(d, d, bias=False)
        self.val = nn.Linear(d, d, bias=False)
        self.a = nn.Parameter(torch.randn(heads, self.dh))
        self.out = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.use_anat_bias = use_anat_bias
        self.register_buffer('A0', A0)
        self.register_buffer('mask', mask)
        src, dst = mask_to_edges(mask)
        self.register_buffer('src', src)
        self.register_buffer('dst', dst)
        self.lrelu = nn.LeakyReLU(0.2)
        self.N = mask.size(0)

    def forward(self, H):
        BT, N, d = H.shape
        Xq = self.lin(H).view(BT, N, self.h, self.dh).permute(0, 2, 1, 3)
        Xv = self.val(H).view(BT, N, self.h, self.dh).permute(0, 2, 1, 3)
        Y = H.new_zeros(BT, self.h, N, self.dh)
        for i in range(N):
            mask_i = (self.src == i)
            nbr = self.dst[mask_i]
            if nbr.numel() == 0:
                continue
            Qi = Xq[:, :, i, :]
            Kj = Xq[:, :, nbr, :]
            e = self.lrelu(Qi.unsqueeze(2) + Kj)
            e = (e * self.a.view(1, self.h, 1, self.dh)).sum(-1)
            if self.use_anat_bias:
                bias = torch.log(self.A0[i, nbr] + 1e-8).view(1, 1, -1)
                e = e + bias
            attn = F.softmax(e, dim=-1)
            attn = self.dropout(attn)
            Vj = Xv[:, :, nbr, :]
            Yi = (attn.unsqueeze(-1) * Vj).sum(dim=2)
            Y[:, :, i, :] = Yi
        Y = Y.permute(0, 2, 1, 3).contiguous().view(BT, N, d)
        Y = self.out(Y)
        return Y


class NodeGate(nn.Module):
    """Per-node Squeeze–Excite (scalar gate 0..1)."""
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(1, d)
        self.fc2 = nn.Linear(d, 1)

    def forward(self, U):
        s = U.mean(dim=-1, keepdim=True)
        g = torch.sigmoid(self.fc2(F.silu(self.fc1(s))))
        return U * g


class GraphBlock(nn.Module):
    """LN → Mixing + GATv2Edge → Residual → Gate → MLP → Residual."""
    def __init__(self, d, mixer, heads=4, dropout=0.1, use_anat_bias=True, mlp_expansion=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.gat = GATv2Edge(d, heads=heads, dropout=dropout, use_anat_bias=use_anat_bias, A0=mixer.A0, mask=mixer.mask)
        self.mixer = mixer
        self.gate = NodeGate(d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_expansion * d),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_expansion * d, d),
            nn.Dropout(dropout),
        )

    def forward(self, Z):
        H = self.ln1(Z)
        H_mix, A = self.mixer(H)
        H_attn = self.gat(H)
        U = Z + H_mix + H_attn
        U = self.gate(U)
        V = self.ln2(U)
        V = self.mlp(V)
        return U + V, A


class GraphEncoder(nn.Module):
    """
    Stack L graph blocks with mean-pool over nodes.
    forward(X, bt_chunk=None) supports chunking over B*T for low memory.

    Returns:
      Z [B,T,N,d], S [B,T,d], A [N,N]
    """
    def __init__(self, pos, d_in=128, d_model=128, L=3, k=8, sigma=None, self_w=0.1,
                 rank=8, alpha=0.5, heads=4, dropout=0.1, use_anat_bias=True):
        super().__init__()
        if not torch.is_tensor(pos):
            pos = torch.tensor(pos, dtype=torch.float32)
        if pos.dim() == 3:  # if passed as [N,3], ok; if [N,2], ok
            pos = pos[:, :2]
        A0, mask = build_anatomical_prior(pos, k=k, sigma=sigma, self_w=self_w)
        self.register_buffer('A0', A0)
        self.register_buffer('mask', mask)
        self.proj_in = nn.Linear(d_in, d_model)
        self.mixer = GraphMixer(A0, mask, rank=rank, alpha=alpha)
        self.blocks = nn.ModuleList([
            GraphBlock(d_model, self.mixer, heads=heads, dropout=dropout, use_anat_bias=use_anat_bias)
            for _ in range(L)
        ])

    def forward(self, X, bt_chunk=None):
        B, T, N, D = X.shape
        X = self.proj_in(X)
        Z = X.reshape(B * T, N, -1)
        lastA = None
        if bt_chunk is None:
            for blk in self.blocks:
                Z, lastA = blk(Z)
        else:
            outs = []
            for s in range(0, Z.size(0), bt_chunk):
                z = Z[s:s+bt_chunk]
                for blk in self.blocks:
                    z, lastA = blk(z)
                outs.append(z)
            Z = torch.cat(outs, dim=0)
        Z = Z.view(B, T, N, -1)
        S = Z.mean(dim=2)
        return Z, S, lastA
