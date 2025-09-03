#!/usr/bin/env python3
"""
Train the Temporal Stem + Graph Encoder for envelope reconstruction on one subject,
then visualize the learned graph (adjacency heatmap, sensor-edge plot, degree topomap).

Usage (example):
  python train_stem_graph_demo.py \
    --preproc_dir /path/to/preproc \
    --subj_id 1 \
    --epochs 5 --batch 32 --win_sec 10 --hop_sec 5

Requires:
  - helper.subject_eeg_env_ab(PREPROC_DIR, subj_id) -> (eeg[T,C], env[T], fs, att_AB)
  - temporal_stem.py and graph_encoder.py in the same folder
  - mne, torch, numpy, matplotlib
"""
import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

from FeatureExpander import TemporalStem
from GraphEncoder import GraphEncoder
import helper


def zscore_train(x):
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True) + 1e-8
    return (x - m) / s, m, s


def window_indices(T, win, hop):
    idx = []
    t = 0
    while t + win <= T:
        idx.append((t, t+win))
        t += hop
    return np.array(idx, dtype=int)


class EEGEnvDataset(Dataset):
    """Creates sliding windows from continuous EEG and envelope.

    X: [T, C], y: [T], returns (EEG_win [W,C], env_win [W])
    """
    def __init__(self, X, y, win, hop):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.win = int(win)
        self.hop = int(hop)
        self.spans = window_indices(len(X), self.win, self.hop)

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, i):
        a, b = self.spans[i]
        return self.X[a:b], self.y[a:b]


def make_biosemi64_info(n_ch=64, sfreq=64.0):
    """Build MNE Info with BioSemi64 montage if channel count=64, else synthetic ring."""
    if n_ch == 64:
        montage = mne.channels.make_standard_montage('biosemi64')
        ch_names = montage.ch_names
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        info.set_montage(montage)
        pos = np.stack([montage.get_positions()['ch_pos'][ch] for ch in ch_names])
        return info, ch_names, pos
    else:
        ch_names = [f'EEG{i}' for i in range(n_ch)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        theta = np.linspace(0, 2*np.pi, n_ch, endpoint=False)
        pos = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
        montage = mne.channels.make_dig_montage(ch_pos={ch: p for ch, p in zip(ch_names, pos)})
        info.set_montage(montage)
        return info, ch_names, pos


class ERGraphModel(nn.Module):
    """Temporal Stem → Graph Encoder → MLP head to reconstruct envelope.

    Node feature build: project stem features to d_lift and broadcast to all nodes,
    concatenate with per-node raw EEG value to get d_in for the graph encoder.
    """
    def __init__(self, n_ch, d_stem=256, d_lift=127, d_in=128, d_model=128, L=3, k=8, heads=4, dropout=0.1, causal=False, pos=None):
        super().__init__()
        self.stem = TemporalStem(in_ch=n_ch, out_ch=d_stem, causal=causal, dropout=dropout)
        self.lift = nn.Linear(d_stem, d_lift)
        if pos is None:
            theta = torch.linspace(0, 2*math.pi, n_ch+1)[:-1]
            pos = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        self.graph = GraphEncoder(pos=torch.tensor(pos, dtype=torch.float32), d_in=d_in, d_model=d_model, L=L, k=k, heads=heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, eeg):
        H0 = self.stem(eeg)
        Lf = self.lift(H0)
        B, T, _ = H0.shape
        N = eeg.shape[-1]
        LfN = Lf.unsqueeze(2).expand(B, T, N, -1)
        raw = eeg.unsqueeze(-1)
        Xin = torch.cat([LfN, raw], dim=-1)
        Z, S, A = self.graph(Xin)
        yhat = self.head(S)
        return yhat.squeeze(-1), A


def pearsonr_loss(yhat, y, eps=1e-8):
    yhat = yhat - yhat.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (yhat * y).sum(dim=1)
    den = torch.sqrt((yhat**2).sum(dim=1) * (y**2).sum(dim=1) + eps)
    r = num / (den + eps)
    return 1 - r.mean()


def train_one_subject(preproc_dir, subj_id, epochs, batch, win_sec, hop_sec, lr, device, outdir):
    eeg, env, fs, attAB = helper.subject_eeg_env_ab(preproc_dir, subj_id)
    X = eeg.astype(np.float32)
    y = env.astype(np.float32)
    X, mX, sX = zscore_train(X)
    y, my, sy = zscore_train(y[:, None])
    y = y[:, 0]

    win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))

    n_ch = X.shape[1]
    info, ch_names, pos = make_biosemi64_info(n_ch=n_ch, sfreq=fs)

    ds = EEGEnvDataset(X, y, win, hop)
    n_train = int(0.8 * len(ds))
    n_val = len(ds) - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=batch, shuffle=False)

    model = ERGraphModel(n_ch=n_ch, d_stem=256, d_lift=127, d_in=128, d_model=128, L=3, k=8, heads=4, dropout=0.1, causal=True, pos=pos)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val = 1e9
    hist = {'train': [], 'val': []}

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_ld:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            yhat, _ = model(xb)
            loss = 0.7 * pearsonr_loss(yhat, yb) + 0.3 * F.l1_loss(yhat, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ld.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_ld:
                xb = xb.to(device)
                yb = yb.to(device)
                yhat, _ = model(xb)
                loss = 0.7 * pearsonr_loss(yhat, yb) + 0.3 * F.l1_loss(yhat, yb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= len(val_ld.dataset)

        hist['train'].append(tr_loss)
        hist['val'].append(va_loss)
        print(f"Epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(outdir, 'last.pt'))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), os.path.join(outdir, 'best.pt'))

    model.load_state_dict(torch.load(os.path.join(outdir, 'best.pt'), map_location=device))
    model.eval()
    with torch.no_grad():
        xb, yb = next(iter(val_ld))
        xb = xb.to(device)
        yhat, A = model(xb)
        A = A.detach().cpu().numpy()

    os.makedirs(outdir, exist_ok=True)
    plot_training_curve(hist, outdir)
    plot_adjacency_heatmap(A, ch_names, outdir)
    plot_sensor_edges(A, info, outdir, topk=300)
    plot_degree_topomap(A, info, outdir)


def plot_training_curve(hist, outdir):
    plt.figure(figsize=(6,4))
    plt.plot(hist['train'], label='train')
    plt.plot(hist['val'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss (0.7·1−r + 0.3·L1)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_curve.png'), dpi=150)
    plt.close()


def plot_adjacency_heatmap(A, ch_names, outdir):
    plt.figure(figsize=(6,5))
    plt.imshow(A, cmap='viridis')
    plt.colorbar(label='weight')
    plt.title('Blended adjacency A')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'A_heatmap.png'), dpi=150)
    plt.close()


def plot_sensor_edges(A, info, outdir, topk=300):
    pos = np.array([info.get_montage().get_positions()['ch_pos'][ch] for ch in info.ch_names])
    P2 = pos[:, :2]
    W = A.copy()
    np.fill_diagonal(W, 0.0)
    triu = np.triu_indices_from(W, k=1)
    vals = W[triu]
    k = min(topk, (len(vals)))
    thr = np.partition(vals, -k)[-k] if k > 0 else 0.0
    sel = W >= thr

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(P2[:,0], P2[:,1], s=30, zorder=3, c='k')
    for i, ch in enumerate(info.ch_names):
        ax.text(P2[i,0], P2[i,1], ch, fontsize=6, ha='center', va='center', color='white', zorder=4)
    idx = np.array(np.where(np.triu(sel, k=1))).T
    for i,j in idx:
        x = [P2[i,0], P2[j,0]]
        y = [P2[i,1], P2[j,1]]
        lw = 0.5 + 2.5 * (A[i,j] / (A.max() + 1e-8))
        ax.plot(x, y, '-', color='tab:blue', lw=lw, alpha=0.5, zorder=2)
    circ = plt.Circle((0,0), 1.05, color='k', fill=False, lw=1)
    ax.add_artist(circ)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'Top {k} edges on sensor map')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'A_edges_sensors.png'), dpi=150)
    plt.close()


def plot_degree_topomap(A, info, outdir):
    W = A.copy()
    np.fill_diagonal(W, 0.0)
    deg = W.sum(axis=1)
    data = deg
    fig, ax = plt.subplots(figsize=(5,4))
    mne.viz.plot_topomap(data, info, axes=ax, show=False)
    ax.set_title('Node strength (row-sum of A)')
    fig.savefig(os.path.join(outdir, 'degree_topomap.png'), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preproc_dir', type=str, required=True)
    parser.add_argument('--subj_id', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--win_sec', type=float, default=10.0)
    parser.add_argument('--hop_sec', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--outdir', type=str, default='outputs')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    train_one_subject(args.preproc_dir, args.subj_id, args.epochs, args.batch, args.win_sec, args.hop_sec, args.lr, args.device, args.outdir)


if __name__ == '__main__':
    main()
