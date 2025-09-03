#!/usr/bin/env python3
"""
Graph Encoder for EEG sequences with an anatomical prior, fixed mixing, GATv2 attention, and node gating.

Input:  X ∈ [B, T, N, d_in]
Output: Z ∈ [B, T, N, d_model], and pooled S ∈ [B, T, d_model] (mean-pool)

Components per block (repeated L times):
1) Mixing: A @ H (A from anatomical prior + learnable residual), H=LayerNorm(Z)
2) GATv2: masked neighbor attention over nodes
3) Residual merge and node gating (Squeeze–Excite per node)
4) Optional per-node MLP with residual

Anatomical prior A0 is built from electrode positions via RBF + k-NN + row-normalization.
A low-rank nonnegative residual (softplus(PQᵀ)) masked to A0 edges is blended: A = row_norm(A0 ⊙ (1 + α·AΔ)).
"""
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_dist(x):
    """x: [N, D] → distances [N, N]."""
    x2 = (x**2).sum(-1, keepdim=True)
    d2 = x2 + x2.t() - 2 * (x @ x.t())
    d2 = d2.clamp_min(0)
    return d2.sqrt()


def build_anatomical_prior(pos, k=8, sigma=None, self_w=0.1):
    """pos: [N,dim]. Returns A0[row-normalized] and mask M (bool).
    RBF weights with k-NN sparsity, symmetric, with small self-loops, then row-norm.
    """
    N = pos.size(0)
    D = pairwise_dist(pos)
    if sigma is None:
        with torch.no_grad():
            triu = D[torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)]
            sigma = torch.median(triu).item() + 1e-6
    W = torch.exp(-(D**2) / (2 * (sigma**2)))
    W.fill_diagonal_(0)
    if self_w is not None and self_w > 0:
        W = W + torch.eye(N, device=W.device) * self_w
    topk = torch.topk(W, k=k, dim=-1).values[:, -1].unsqueeze(-1)
    M = W >= topk
    M = M | torch.eye(N, dtype=torch.bool, device=W.device)
    W = W * M.float()
    W = 0.5 * (W + W.t())
    A0 = W / (W.sum(dim=-1, keepdim=True) + 1e-8)
    M = A0 > 0
    return A0, M


class GraphMixer(nn.Module):
    """Fixed neighbor mixing: H_mix = A @ H."""
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
        A_delta = F.softplus(R)
        A_delta = A_delta * self.mask.float()
        A = self.A0 * (1.0 + self.alpha * A_delta)
        A = A / (A.sum(-1, keepdim=True) + 1e-8)
        return A

    def forward(self, H):
        A = self.blended_A()
        return torch.einsum('ij, bjd -> bid', A, H), A


class GATv2Layer(nn.Module):
    """Neighbor-masked GATv2 with optional anatomical log-bias.

    Inputs: H ∈ [B, N, d]
    Outputs: Y ∈ [B, N, d]
    """
    def __init__(self, d, heads=4, dropout=0.1, use_anat_bias=True):
        super().__init__()
        self.h = heads
        self.dh = d // heads
        self.lin = nn.Linear(d, d, bias=False)
        self.a = nn.Parameter(torch.randn(heads, self.dh))
        self.out = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.use_anat_bias = use_anat_bias
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, H, mask, A0):
        B, N, d = H.shape
        X = self.lin(H).view(B, N, self.h, self.dh).permute(0, 2, 1, 3)
        Xi = X.unsqueeze(3)
        Xj = X.unsqueeze(2)
        e = (Xi + Xj)
        e = self.lrelu(e)
        e = torch.einsum('bhijd, hd -> bhij', e, self.a)
        if self.use_anat_bias:
            anat = torch.log(A0 + 1e-8)
            e = e + anat.unsqueeze(0).unsqueeze(0)
        e = e.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(e, dim=-1)
        attn = self.dropout(attn)
        V = X
        Y = torch.einsum('bhij, bhjd -> bhid', attn, V)
        Y = Y.permute(0, 2, 1, 3).contiguous().view(B, N, d)
        Y = self.out(Y)
        return Y


class NodeGate(nn.Module):
    """Squeeze–Excite gate per node (scalar 0..1)."""
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(1, d)
        self.fc2 = nn.Linear(d, 1)

    def forward(self, U):
        s = U.mean(dim=-1, keepdim=True)
        g = torch.sigmoid(self.fc2(F.silu(self.fc1(s))))
        return U * g


class GraphBlock(nn.Module):
    """One block: LN → Mixing + GATv2 → Residual → Gate → MLP + Residual."""
    def __init__(self, d, mixer, heads=4, dropout=0.1, use_anat_bias=True, mlp_expansion=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.gat = GATv2Layer(d, heads=heads, dropout=dropout, use_anat_bias=use_anat_bias)
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
        H_attn = self.gat(H, self.mixer.mask, self.mixer.A0)
        U = Z + H_mix + H_attn
        U = self.gate(U)
        V = self.ln2(U)
        V = self.mlp(V)
        return U + V, A


class GraphEncoder(nn.Module):
    """Stack of L graph blocks with mean-pooling over nodes.

    Forward returns (Z, S, A):
      Z ∈ [B, T, N, d_model]  node features after L blocks
      S ∈ [B, T, d_model]     mean-pooled spatial summary
      A ∈ [N, N]              blended adjacency used
    """
    def __init__(self, pos, d_in=128, d_model=128, L=3, k=8, sigma=None, self_w=0.1, rank=8, alpha=0.5, heads=4, dropout=0.1, use_anat_bias=True):
        super().__init__()
        A0, mask = build_anatomical_prior(pos, k=k, sigma=sigma, self_w=self_w)
        self.register_buffer('A0', A0)
        self.register_buffer('mask', mask)
        self.proj_in = nn.Linear(d_in, d_model)
        self.mixer = GraphMixer(A0, mask, rank=rank, alpha=alpha)
        self.blocks = nn.ModuleList([
            GraphBlock(d_model, self.mixer, heads=heads, dropout=dropout, use_anat_bias=use_anat_bias)
            for _ in range(L)
        ])

    def forward(self, X):
        B, T, N, D = X.shape
        X = self.proj_in(X)
        Z = X.reshape(B * T, N, -1)
        lastA = None
        for blk in self.blocks:
            Z, lastA = blk(Z)
        Z = Z.view(B, T, N, -1)
        S = Z.mean(dim=2)
        return Z, S, lastA


def main():
    """Quick shape and behavior check with random data and random 2D positions."""
    parser = argparse.ArgumentParser(description="Test GraphEncoder on random EEG node features")
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--time', type=int, default=128)
    parser.add_argument('--nodes', type=int, default=64)
    parser.add_argument('--d_in', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(0)
    N = args.nodes
    theta = torch.linspace(0, 2*math.pi, N+1)[:-1]
    pos = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

    X = torch.randn(args.batch, args.time, N, args.d_in)
    enc = GraphEncoder(pos, d_in=args.d_in, d_model=args.d_model, L=args.blocks, k=args.k, rank=args.rank, heads=args.heads, dropout=args.dropout)
    Z, S, A = enc(X)

    print(f"X: {tuple(X.shape)}")
    print(f"Z: {tuple(Z.shape)}  (node features)")
    print(f"S: {tuple(S.shape)}  (mean-pooled)")
    print(f"A: {tuple(A.shape)}  (blended adjacency)")
    print(f"Row sums (first 5): {A.sum(-1)[:5].tolist()}")
    print(f"Avg degree (nonzeros per row): {int((A>0).sum(-1).float().mean().item())}")

if __name__ == '__main__':
    main()
