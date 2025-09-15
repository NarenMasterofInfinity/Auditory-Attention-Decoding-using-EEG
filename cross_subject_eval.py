#!/usr/bin/env python3
"""
Subject-independent evaluation for the *old configuration*:
TemporalStem → GraphEncoder → MLP head (ERGraphModel).

For each train subject S_train in 1..18:
  load outputs/S{S_train}/best.pt
  for each test subject S_test in 1..18:
    evaluate on ALL windows of S_test (no split)
    append a row to subject_indeptendt/cross_subject_results.csv

Matches what you trained with TestingGraphMemEfficient.py.
"""

import os, csv, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mne

# --- modules that match your training ---
try:
    from FeatureExpander import TemporalStem
except Exception:
    from temporal_stem import TemporalStem

from GraphEncoder1 import GraphEncoder  # memory-efficient fixed-K

# ---------------- data helpers ----------------

def zscore_per_subject(x):
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True) + 1e-8
    return (x - m) / s

def window_indices(T, win, hop):
    spans, t = [], 0
    while t + win <= T:
        spans.append((t, t + win))
        t += hop
    return np.array(spans, dtype=int)

class EEGEnvDataset(Dataset):
    def __init__(self, X, y, win, hop):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.spans = window_indices(len(X), int(win), int(hop))
    def __len__(self): return len(self.spans)
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

# ---------------- metric ----------------

def pearsonr_batch(yhat, y, eps=1e-8):
    yhat = yhat - yhat.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (yhat * y).sum(dim=1)
    den = torch.sqrt((yhat**2).sum(dim=1) * (y**2).sum(dim=1) + eps)
    return num / (den + eps)

# ---------------- model (matches old ERGraphModel) ----------------

class ERGraphModel(nn.Module):
    """
    TemporalStem → GraphEncoder → tiny MLP head producing envelope per time step.
    """
    def __init__(self, n_ch, pos, d_stem=256, d_lift=127, d_in=128, d_model=128,
                 L=3, k=8, heads=4, dropout=0.1, causal=True):
        super().__init__()
        if not torch.is_tensor(pos):
            pos = torch.tensor(pos, dtype=torch.float32)
        self.stem = TemporalStem(in_ch=n_ch, out_ch=d_stem, causal=causal, dropout=dropout)
        self.lift = nn.Linear(d_stem, d_lift)
        self.graph = GraphEncoder(pos=pos, d_in=d_in, d_model=d_model, L=L, k=k, heads=heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.SiLU(), nn.Linear(64, 1)
        )

    def forward(self, eeg, bt_chunk=None):
        H0 = self.stem(eeg)                       # [B,T,256]
        Lf = self.lift(H0)                        # [B,T,127]
        B, T, _ = H0.shape; N = eeg.shape[-1]
        Xin = torch.cat([Lf.unsqueeze(2).expand(B, T, N, -1),
                         eeg.unsqueeze(-1)], dim=-1)  # [B,T,N,128]
        try:
            Z, S, A = self.graph(Xin, bt_chunk=bt_chunk)
        except TypeError:
            Z, S, A = self.graph(Xin)
        yhat = self.head(S).squeeze(-1)           # [B,T]
        return yhat, A

# ---------------- loading utils ----------------

def load_state_dict_flexible(model, ckpt_path, device):
    """
    Load a state dict saved by TestingGraphMemEfficient.py best_state dump.
    Strips common wrappers like 'module.' or 'model.' if present.
    """
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    # strip common prefixes
    def strip_prefix(d, prefix):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in d.items()}
    for pref in ['module.', 'model.']:
        sd = strip_prefix(sd, pref)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) < 10: print(" missing keys:", missing)
        if len(unexpected) < 10: print(" unexpected keys:", unexpected)
    return model

# ---------------- evaluation ----------------

def eval_on_full_subject(model, preproc_dir, subj_id, win_sec, hop_sec, bt_chunk, device, amp):
    import helper
    eeg, env, fs, attAB = helper.subject_eeg_env_ab(preproc_dir, subj_id)
    X = zscore_per_subject(eeg.astype(np.float32))
    y = zscore_per_subject(env.astype(np.float32)[:, None])[:, 0]
    win = int(round(win_sec * fs)); hop = int(round(hop_sec * fs))
    ds = EEGEnvDataset(X, y, win, hop)
    ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    use_fp16 = (amp == 'fp16'); use_bf16 = (amp == 'bf16')
    amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)

    model.eval()
    r_sum, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in ld:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                yhat, _ = model(xb, bt_chunk=bt_chunk)
                r = pearsonr_batch(yhat, yb).mean()
            r_sum += r.item() * xb.size(0); n += xb.size(0)
    return r_sum / max(1, n)

# ---------------- main ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preproc_dir', type=str, required=True)
    p.add_argument('--models_root', type=str, default='outputs')
    p.add_argument('--outdir', type=str, default='subject_indeptendt')
    p.add_argument('--win_sec', type=float, default=5.0)
    p.add_argument('--hop_sec', type=float, default=2.5)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # graph hyperparams MUST match training
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--blocks', type=int, default=3)
    p.add_argument('--bt_chunk', type=int, default=256)
    p.add_argument('--amp', type=str, choices=['none','fp16','bf16'], default='bf16')
    p.add_argument('--skip_same', action='store_true')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, 'cross_subject_results.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['train_subject', 'test_subject', 'r_mean'])

    # use BioSemi64 positions (or ring fallback)
    info, ch_names, pos = make_biosemi64_info(n_ch=64, sfreq=64.0)

    for s_train in range(1, 19):
        ckpt = os.path.join(args.models_root, f"S{s_train}", "best_model.pt")
        if not os.path.isfile(ckpt):
            print(f"[WARN] Missing {ckpt}; skip S{s_train}"); continue

        # Build model EXACTLY like training
        model = ERGraphModel(
            n_ch=64, pos=pos, d_stem=256, d_lift=127, d_in=128, d_model=128,
            L=args.blocks, k=args.k, heads=args.heads, dropout=0.1, causal=True
        ).to(args.device)

        try:
            load_state_dict_flexible(model, ckpt, device=args.device)
        except Exception as e:
            print(f"[ERR ] Could not load {ckpt}: {e}")
            # write NA rows for this train subject to keep matrix rectangular (optional)
            for s_test in range(1, 19):
                with open(csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([s_train, s_test, 'NA'])
            continue

        print(f"\nLoaded train model S{s_train} from {ckpt}")
        for s_test in range(1, 19):
            if args.skip_same and s_test == s_train:
                continue
            try:
                r = eval_on_full_subject(model, args.preproc_dir, s_test,
                                         args.win_sec, args.hop_sec, args.bt_chunk,
                                         args.device, args.amp)
                with open(csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([s_train, s_test, r])
                print(f"[OK  ] train S{s_train} → test S{s_test}: r={r:.4f}")
            except Exception as e:
                with open(csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([s_train, s_test, 'NA'])
                print(f"[FAIL] train S{s_train} → test S{s_test}: {e}")

    print(f"\nWrote {csv_path}")

if __name__ == '__main__':
    main()
