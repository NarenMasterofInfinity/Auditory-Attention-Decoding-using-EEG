#!/usr/bin/env python3
"""
visualize_graph_encoder.py

Visualize envelope reconstruction using the GraphEncoder-only model:
TemporalStem → GraphEncoder → MLP head

What it does:
- Loads EEG, envelope, fs via helper.subject_eeg_env_ab(preproc_dir, subj_id)
- Z-scores EEG and envelope over the whole sequence (matches TestingGraphMemEfficient.py)
- Builds sliding windows (win_sec, hop_sec)
- Loads best_model.pt for this subject
- Runs inference over all windows, stitches predictions with overlap-averaging
- Plots:
    (1) EEG snippet (first 3 channels, offset),
    (2) Full predicted vs. true envelope,
    (3) Zoomed overlay segment
- Saves into outputs/SX/vis/

Usage:
  python visualize_graph_encoder.py \
    --preproc_dir $DATASET \
    --subj_id 1 \
    --ckpt outputs/S1/best_model.pt \
    --outdir outputs \
    --win_sec 5 --hop_sec 2.5 \
    --batch 32 --bt_chunk 256 --device cuda \
    --k 8 --heads 4 --blocks 3 --amp bf16
"""

import os
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

import helper

# ----- imports that mirror your training script -----
try:
    from FeatureExpander import TemporalStem
except Exception:
    from temporal_stem import TemporalStem

try:
    from GraphEncoder1 import GraphEncoder
except Exception:
    from graph_encoder_sparse import GraphEncoder


def zscore_train(x):
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True) + 1e-8
    return (x - m) / s, m, s


def window_indices(T, win, hop):
    idx, t = [], 0
    while t + win <= T:
        idx.append((t, t + win))
        t += hop
    return np.array(idx, dtype=int)


class EEGWindows(Dataset):
    """Yields (EEG_win[W,C], env_win[W]) for sequential stitching."""
    def __init__(self, X, y, spans):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.spans = spans
    def __len__(self):
        return len(self.spans)
    def __getitem__(self, i):
        a, b = self.spans[i]
        return self.X[a:b], self.y[a:b]


def make_biosemi64_info(n_ch=64, sfreq=64.0):
    if n_ch == 64:
        montage = mne.channels.make_standard_montage('biosemi64')
        ch_names = montage.ch_names
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        info.set_montage(montage)
        pos = np.stack([montage.get_positions()['ch_pos'][ch] for ch in ch_names])
        return info, ch_names, pos
    ch_names = [f'EEG{i}' for i in range(n_ch)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    theta = np.linspace(0, 2*np.pi, n_ch, endpoint=False)
    pos = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
    montage = mne.channels.make_dig_montage(ch_pos={ch: p for ch, p in zip(ch_names, pos)})
    info.set_montage(montage)
    return info, ch_names, pos


class ERGraphModel(nn.Module):
    """Same as in TestingGraphMemEfficient.py: Stem → GraphEncoder → MLP head."""
    def __init__(self, n_ch, pos, d_stem=256, d_lift=127, d_in=128, d_model=128,
                 L=3, k=8, heads=4, dropout=0.1, causal=True):
        super().__init__()
        self.stem = TemporalStem(in_ch=n_ch, out_ch=d_stem, causal=causal, dropout=dropout)
        self.lift = nn.Linear(d_stem, d_lift)
        self.graph = GraphEncoder(pos=torch.tensor(pos, dtype=torch.float32),
                                  d_in=d_in, d_model=d_model, L=L, k=k, heads=heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.SiLU(), nn.Linear(64, 1)
        )
    def forward(self, eeg, bt_chunk=None):
        H0 = self.stem(eeg)                        # [B,T,256]
        Lf = self.lift(H0)                         # [B,T,127]
        B, T, _ = H0.shape
        N = eeg.shape[-1]
        Xin = torch.cat([Lf.unsqueeze(2).expand(B, T, N, -1),
                         eeg.unsqueeze(-1)], dim=-1)  # [B,T,N,128]
        try:
            _, S, _ = self.graph(Xin, bt_chunk=bt_chunk)
        except TypeError:
            _, S, _ = self.graph(Xin)
        yhat = self.head(S).squeeze(-1)           # [B,T]
        return yhat


def stitch_predict(model, ds, device, batch, bt_chunk, amp_dtype):
    ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)
    T = ds.X.shape[0]
    sum_pred = np.zeros(T, dtype=np.float32)
    count = np.zeros(T, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i, (xb, yb) in enumerate(ld):
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                yhat = model(xb, bt_chunk=bt_chunk)   # [B, W] (likely bfloat16 when amp=bf16)
            # >>> cast to float32 before moving to numpy <<<
            yhat = yhat.to(torch.float32).detach().cpu().numpy()

            B, W = yhat.shape
            a0 = i * B
            for b in range(B):
                if a0 + b >= len(ds.spans): break
                a, e = ds.spans[a0 + b]
                sum_pred[a:e] += yhat[b]
                count[a:e] += 1.0

    pred = np.zeros_like(sum_pred)
    m = count > 0
    pred[m] = sum_pred[m] / count[m]
    return pred, m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preproc_dir', type=str, required=True)
    p.add_argument('--subj_id', type=int, required=True)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--outdir', type=str, default='outputs')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    p.add_argument('--win_sec', type=float, default=5.0)
    p.add_argument('--hop_sec', type=float, default=2.5)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--bt_chunk', type=int, default=256)

    # graph hyperparams (must match training)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--blocks', type=int, default=3)

    p.add_argument('--amp', type=str, choices=['none','fp16','bf16'], default='bf16')
    p.add_argument('--zoom_sec', type=float, default=20.0)
    args = p.parse_args()

    eeg, env, fs, attAB = helper.subject_eeg_env_ab(args.preproc_dir, args.subj_id)
    X = eeg.astype(np.float32)
    y = env.astype(np.float32)
    Xz, Xm, Xs = zscore_train(X)
    yz, ym, ys = zscore_train(y[:, None]); yz = yz[:, 0]

    win = int(round(args.win_sec * fs))
    hop = int(round(args.hop_sec * fs))
    spans = window_indices(len(Xz), win, hop)
    ds = EEGWindows(Xz, yz, spans)

    info, ch_names, pos = make_biosemi64_info(n_ch=X.shape[1], sfreq=fs)
    model = ERGraphModel(n_ch=X.shape[1], pos=pos, d_stem=256, d_lift=127, d_in=128, d_model=128,
                         L=args.blocks, k=args.k, heads=args.heads, dropout=0.1, causal=True).to(args.device)

    state = torch.load(args.ckpt, map_location='cpu')
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) or len(unexpected):
        print("[warn] state_dict mismatch:", len(missing), "missing |", len(unexpected), "unexpected")

    use_fp16 = (args.amp == 'fp16'); use_bf16 = (args.amp == 'bf16')
    amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)

    pred_norm, mask = stitch_predict(model, ds, args.device, args.batch, args.bt_chunk, amp_dtype)

    # de-normalize to original envelope scale
    pred = pred_norm * ys.item() + ym.item()
    true_env = env.copy()

    # stitched Pearson r over valid region
    valid = mask
    r = np.nan
    if valid.sum() > 10:
        ph = pred[valid]; th = true_env[valid]
        ph = (ph - ph.mean()) / (ph.std() + 1e-8)
        th = (th - th.mean()) / (th.std() + 1e-8)
        r = float(np.corrcoef(ph, th)[0, 1])
    print(f"[viz] stitched Pearson r = {r:.3f}" if np.isfinite(r) else "[viz] r = nan")

    # plots
    sx = os.path.join(args.outdir, f"S{args.subj_id}", "vis")
    os.makedirs(sx, exist_ok=True)

    T = len(env); t = np.arange(T) / fs
    zoom = int(round(args.zoom_sec * fs))
    t0 = 0
    t1 = min(T, t0 + zoom)

    fig = plt.figure(figsize=(12, 9))

    ax1 = plt.subplot(3,1,1)
    if X.shape[1] >= 3:
        z1 = (eeg[:, 0] - eeg[:, 0].mean()) / (eeg[:, 0].std() + 1e-8)
        z2 = (eeg[:, 1] - eeg[:, 1].mean()) / (eeg[:, 1].std() + 1e-8)
        z3 = (eeg[:, 2] - eeg[:, 2].mean()) / (eeg[:, 2].std() + 1e-8)
        ax1.plot(t[t0:t1], z1[t0:t1], lw=0.8, label=ch_names[0] if ch_names else 'CH0')
        ax1.plot(t[t0:t1], z2[t0:t1] + 3.0, lw=0.8, label=ch_names[1] if len(ch_names)>1 else 'CH1')
        ax1.plot(t[t0:t1], z3[t0:t1] + 6.0, lw=0.8, label=ch_names[2] if len(ch_names)>2 else 'CH2')
        ax1.set_ylabel('EEG (z, offsets)')
        ax1.legend(fontsize=8)
    else:
        ax1.plot(t[t0:t1], eeg[t0:t1, 0], lw=0.8, label='EEG0')
        ax1.legend(fontsize=8)
        ax1.set_ylabel('EEG')

    ax1.set_title(f"S{args.subj_id} | EEG snippet ({args.zoom_sec:.0f}s)")

    ax2 = plt.subplot(3,1,2)
    ax2.plot(t, true_env, lw=0.8, label='True envelope', alpha=0.9)
    ax2.plot(t, pred, lw=0.8, label='Predicted (stitched)', alpha=0.9)
    ax2.set_ylabel('Envelope')
    ax2.legend()
    ax2.set_title(f"Full envelopes | stitched r={r:.3f}" if np.isfinite(r) else "Full envelopes")

    ax3 = plt.subplot(3,1,3)
    ax3.plot(t[t0:t1], true_env[t0:t1], lw=1.0, label='True env', alpha=0.9)
    ax3.plot(t[t0:t1], pred[t0:t1], lw=1.0, label='Pred env', alpha=0.9)
    ax3.set_ylabel('Envelope'); ax3.set_xlabel('Time (s)')
    ax3.legend()
    ax3.set_title(f"Zoomed overlay ({args.zoom_sec:.0f}s)")

    plt.tight_layout()
    fig_path = os.path.join(sx, f"env_compare_S{args.subj_id}.png")
    plt.savefig(fig_path, dpi=150); plt.close(fig)

    # Save timeseries
    np.savez_compressed(
        os.path.join(sx, f"env_timeseries_S{args.subj_id}.npz"),
        t=t, pred=pred, true_env=true_env, mask=mask, fs=fs, r=r
    )
    print(f"[viz] wrote {fig_path}")
    print(f"[viz] wrote env_timeseries_S{args.subj_id}.npz")


if __name__ == '__main__':
    main()
