#!/usr/bin/env python3
"""
finetune_from_best.py

Resume training from an existing checkpoint (e.g., outputs/S1/best_model.pt)
for a few more epochs using the same ERGraphModel (TemporalStem → GraphEncoder → MLP).

Saves:
- finetune_best.pt (new best after finetune)
- finetune_training_curve.png, finetune_training_r.png
- finetune_A_heatmap.png, finetune_A0_heatmap.png, finetune_A_edges_sensors.png, finetune_A_edges_delta.png, finetune_in_strength_topomap.png
- vis/finetune_test_compare_window.png

Usage:
  python finetune_from_best.py --preproc_dir $DATASET --subject 1 \
      --ckpt outputs/S1/best_model.pt --outdir outputs/S1 --epochs_more 20 \
      --lr 1e-4 --batch 8 --bt_chunk 128 --amp bf16
"""

import os, math, argparse, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

import helper

try:
    from FeatureExpander import TemporalStem
except Exception:
    from temporal_stem import TemporalStem

try:
    from GraphEncoder1 import GraphEncoder
except Exception:
    from graph_encoder_sparse import GraphEncoder


def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def zscore_train(x):
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True) + 1e-8
    return (x - m) / s, m, s

def window_indices(T, win, hop):
    idx, t = [], 0
    while t + win <= T:
        idx.append((t, t + win)); t += hop
    return np.array(idx, dtype=int)

class EEGEnvDataset(Dataset):
    def __init__(self, X, y, win, hop):
        self.X = X.astype(np.float32); self.y = y.astype(np.float32)
        self.win = int(win); self.hop = int(hop)
        self.spans = window_indices(len(X), self.win, self.hop)
    def __len__(self): return len(self.spans)
    def __getitem__(self, i):
        a, b = self.spans[i]; return self.X[a:b], self.y[a:b]

def split_indices(num_items, train_ratio=0.7, val_ratio=0.1, seed=123):
    all_idx = np.arange(num_items); rng = np.random.default_rng(seed); rng.shuffle(all_idx)
    n_train = int(round(train_ratio * num_items))
    n_val   = int(round(val_ratio * num_items))
    train_idx = all_idx[:n_train]
    val_idx   = all_idx[n_train:n_train+n_val]
    test_idx  = all_idx[n_train+n_val:]
    return train_idx, val_idx, test_idx

def subset_dataset(ds, idx):
    class _Sub(Dataset):
        def __init__(self, base, sel): self.base = base; self.sel = np.array(sel, int)
        def __len__(self): return len(self.sel)
        def __getitem__(self, i): return self.base[self.sel[i]]
    return _Sub(ds, idx)

def make_biosemi64_info(n_ch=64, sfreq=64.0):
    if n_ch == 64:
        montage = mne.channels.make_standard_montage('biosemi64')
        ch_names = montage.ch_names
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg'); info.set_montage(montage)
        pos = np.stack([montage.get_positions()['ch_pos'][ch] for ch in ch_names]); return info, ch_names, pos
    ch_names = [f'EEG{i}' for i in range(n_ch)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    theta = np.linspace(0, 2*np.pi, n_ch, endpoint=False)
    pos = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
    montage = mne.channels.make_dig_montage(ch_pos={ch: p for ch, p in zip(ch_names, pos)})
    info.set_montage(montage); return info, ch_names, pos

class ERGraphModel(nn.Module):
    def __init__(self, n_ch, pos, d_stem=256, d_lift=127, d_in=128, d_model=128,
                 L=3, k=8, heads=4, dropout=0.1, causal=True):
        super().__init__()
        self.stem = TemporalStem(in_ch=n_ch, out_ch=d_stem, causal=causal, dropout=dropout)
        self.lift = nn.Linear(d_stem, d_lift)
        self.graph = GraphEncoder(pos=torch.tensor(pos, dtype=torch.float32),
                                  d_in=d_in, d_model=d_model, L=L, k=k, heads=heads, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.SiLU(), nn.Linear(64, 1))
    def forward(self, eeg, bt_chunk=None):
        H0 = self.stem(eeg)                       # [B,T,256]
        Lf = self.lift(H0)                        # [B,T,127]
        B, T, _ = H0.shape; N = eeg.shape[-1]
        Xin = torch.cat([Lf.unsqueeze(2).expand(B, T, N, -1), eeg.unsqueeze(-1)], dim=-1)  # [B,T,N,128]
        try: _, S, A = self.graph(Xin, bt_chunk=bt_chunk)
        except TypeError: _, S, A = self.graph(Xin)
        yhat = self.head(S).squeeze(-1)           # [B,T]
        return yhat, A

def pearsonr_batch(yhat, y, eps=1e-8):
    yhat = yhat - yhat.mean(dim=1, keepdim=True)
    y    = y    - y.mean(dim=1, keepdim=True)
    num = (yhat * y).sum(dim=1)
    den = torch.sqrt((yhat**2).sum(dim=1) * (y**2).sum(dim=1) + eps)
    return num / (den + eps)

def slope_loss_simple(yhat, y, beta=0.01):
    dyh = yhat[:, 1:] - yhat[:, :-1]
    dy  = y[:,   1:] - y[:,   :-1]
    return F.smooth_l1_loss(dyh, dy, beta=beta)

def make_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=0, final_lr_pct=0.2):
    def lr_lambda(step):
        step = min(step, total_steps)
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if total_steps == warmup_steps: return final_lr_pct
        prog = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * prog))
        return final_lr_pct + (1 - final_lr_pct) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def plot_curves(hist, outdir):
    plt.figure(figsize=(6,4))
    plt.plot(hist['train_loss'], label='train'); plt.plot(hist['val_loss'], label='val')
    plt.xlabel('epoch'); plt.ylabel('loss (0.7·(1−r)+0.3·MSE+λΔ)')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir, 'finetune_training_curve.png'), dpi=150); plt.close()
    plt.figure(figsize=(6,4))
    plt.plot(hist['train_r'], label='train r'); plt.plot(hist['val_r'], label='val r')
    plt.xlabel('epoch'); plt.ylabel('Pearson r')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(outdir, 'finetune_training_r.png'), dpi=150); plt.close()

def plot_graphs(A, A0, info, outdir):
    plt.figure(figsize=(6,5)); plt.imshow(A, cmap='viridis'); plt.colorbar(label='weight'); plt.title('Finetune: A')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'finetune_A_heatmap.png'), dpi=150); plt.close()
    if A0 is not None:
        plt.figure(figsize=(6,5)); plt.imshow(A0, cmap='viridis'); plt.colorbar(label='weight'); plt.title('A0 (prior)')
        plt.tight_layout(); plt.savefig(os.path.join(outdir, 'finetune_A0_heatmap.png'), dpi=150); plt.close()

    pos = np.array([info.get_montage().get_positions()['ch_pos'][ch] for ch in info.ch_names])
    P2 = pos[:, :2]
    W = A.copy(); np.fill_diagonal(W, 0.0)
    triu = np.triu_indices_from(W, k=1); vals = W[triu]
    k = min(120, len(vals)); thr = np.partition(vals, -k)[-k] if k>0 else 0.0
    sel = W >= thr
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(P2[:,0], P2[:,1], s=30, zorder=3, c='k')
    for i, ch in enumerate(info.ch_names):
        ax.text(P2[i,0], P2[i,1], ch, fontsize=6, ha='center', va='center', color='white', zorder=4)
    idx = np.array(np.where(np.triu(sel, k=1))).T
    for i, j in idx:
        w = A[i,j]
        ax.plot([P2[i,0], P2[j,0]], [P2[i,1], P2[j,1]], '-', lw=0.5 + 3*(w/(A.max()+1e-8)), color='tab:blue', alpha=0.5)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title('Finetune: Top edges on sensor map')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'finetune_A_edges_sensors.png'), dpi=150); plt.close()

    if A0 is not None:
        D = np.clip(A - A0, 0, None); np.fill_diagonal(D, 0.0)
        triu = np.triu_indices_from(D, k=1); vals = D[triu]
        k = min(80, len(vals)); thr = np.partition(vals, -k)[-k] if k>0 else 0.0
        sel = D >= thr
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(P2[:,0], P2[:,1], s=30, zorder=3, c='k')
        for i, ch in enumerate(info.ch_names):
            ax.text(P2[i,0], P2[i,1], ch, fontsize=6, ha='center', va='center', color='white', zorder=4)
        idx = np.array(np.where(np.triu(sel, k=1))).T
        for i, j in idx:
            w = D[i,j]
            ax.plot([P2[i,0], P2[j,0]], [P2[i,1], P2[j,1]], '-', lw=0.5 + 3*(w/(D.max()+1e-8)), color='tab:orange', alpha=0.8)
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title('Strengthened edges (A−A0)+')
        plt.tight_layout(); plt.savefig(os.path.join(outdir, 'finetune_A_edges_delta.png'), dpi=150); plt.close()

    W = A.copy(); np.fill_diagonal(W, 0.0); in_strength = W.sum(axis=0)
    fig, ax = plt.subplots(figsize=(5,4))
    mne.viz.plot_topomap(in_strength, info, axes=ax, show=False)
    ax.set_title('Finetune: In-strength'); fig.savefig(os.path.join(outdir, 'finetune_in_strength_topomap.png'), dpi=150); plt.close(fig)

def plot_test_window_compare(x_win, y_win, yhat_win, fs, ch_names, outdir, eeg_channels_to_show=6, fname='finetune_test_compare_window.png'):
    ensure_dir(outdir)
    W, C = x_win.shape; t = np.arange(W) / float(fs)
    k = min(max(4, eeg_channels_to_show), min(6, C))
    chans = list(range(k))
    Xz = (x_win[:, chans] - x_win[:, chans].mean(0, keepdims=True)) / (x_win[:, chans].std(0, keepdims=True) + 1e-8)
    offsets = np.arange(k) * 3.0
    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(2,1,1)
    for i, ch in enumerate(chans):
        name = ch_names[ch] if ch_names and ch < len(ch_names) else f'CH{ch}'
        ax1.plot(t, Xz[:, i] + offsets[i], lw=0.8, label=name)
    ax1.set_ylabel('EEG (z, offsets)'); ax1.set_title(f'EEG window (first {k} channels)')
    ax1.legend(fontsize=8, ncol=min(3, k), loc='upper right'); ax1.grid(alpha=0.2)
    ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax2.plot(t, y_win, lw=1.2, label='True env'); ax2.plot(t, yhat_win, lw=1.2, label='Pred env')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Envelope'); ax2.legend(); ax2.grid(alpha=0.2)
    plt.tight_layout(); path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150); plt.close(fig); return path

def evaluate(model, loader, device, bt_chunk, amp_dtype):
    model.eval(); r_sum, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                yhat, _ = model(xb, bt_chunk=bt_chunk); r = pearsonr_batch(yhat, yb).mean()
            r_sum += r.item() * xb.size(0); n += xb.size(0)
    return r_sum / max(1, n)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preproc_dir', type=str, required=True)
    p.add_argument('--subject', type=int, default=1)
    p.add_argument('--ckpt', type=str, default='outputs/S1/best_model.pt')
    p.add_argument('--outdir', type=str, default='outputs_loss_updated/S1')
    p.add_argument('--epochs_more', type=int, default=50)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--win_sec', type=float, default=5.0)
    p.add_argument('--hop_sec', type=float, default=2.5)
    p.add_argument('--lr', type=float, default=1e-4)      # smaller LR for finetune
    p.add_argument('--weight_decay', type=float, default=5e-5)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--blocks', type=int, default=3)
    p.add_argument('--bt_chunk', type=int, default=128)
    p.add_argument('--amp', type=str, choices=['none','fp16','bf16'], default='bf16')
    p.add_argument('--warmup_pct', type=float, default=0.1)
    p.add_argument('--final_lr_pct', type=float, default=0.2)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--min_delta', type=float, default=1e-3)
    # loss weights
    p.add_argument('--w_r', type=float, default=0.7)
    p.add_argument('--w_abs', type=float, default=0.3)
    p.add_argument('--lambda_slope', type=float, default=0.03)  # smaller for stable finetune
    p.add_argument('--huber_beta', type=float, default=0.01)
    args = p.parse_args()

    ensure_dir(args.outdir)

    eeg, env, fs, _ = helper.subject_eeg_env_ab(args.preproc_dir, args.subject)
    X = eeg.astype(np.float32); y = env.astype(np.float32)
    X, _, _ = zscore_train(X); y, _, _ = zscore_train(y[:, None]); y = y[:, 0]

    win = int(round(args.win_sec * fs)); hop = int(round(args.hop_sec * fs))
    ds = EEGEnvDataset(X, y, win, hop)
    tr_idx, va_idx, te_idx = split_indices(len(ds), 0.7, 0.1, seed=1000 + args.subject)
    train_ds = subset_dataset(ds, tr_idx); val_ds = subset_dataset(ds, va_idx); test_ds = subset_dataset(ds, te_idx)

    workers, prefetch = 4, 2
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
                          num_workers=workers, pin_memory=True,
                          persistent_workers=(workers>0), prefetch_factor=prefetch)
    val_ld   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                          num_workers=workers, pin_memory=True,
                          persistent_workers=(workers>0), prefetch_factor=prefetch)
    test_ld  = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                          num_workers=workers, pin_memory=True,
                          persistent_workers=(workers>0), prefetch_factor=prefetch)

    n_ch = X.shape[1]
    info, ch_names, pos = make_biosemi64_info(n_ch=n_ch, sfreq=fs)

    model = ERGraphModel(n_ch=n_ch, pos=pos, d_stem=256, d_lift=127, d_in=128, d_model=128,
                         L=args.blocks, k=args.k, heads=args.heads, dropout=0.1, causal=True).to(args.device)

    # Load the existing checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu')
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print("Loaded with key diffs. Missing:", len(missing), "Unexpected:", len(unexpected))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_ld))
    total_steps = args.epochs_more * steps_per_epoch
    warmup_steps = int(args.warmup_pct * total_steps)
    sched = make_warmup_cosine_scheduler(opt, total_steps, warmup_steps, args.final_lr_pct)

    use_fp16 = (args.amp == 'fp16'); use_bf16 = (args.amp == 'bf16')
    amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    best_val = float('inf'); best_state = None; wait = 0
    hist = {'train_loss': [], 'val_loss': [], 'train_r': [], 'val_r': []}

    def _to(x): return x.to(args.device, non_blocking=True)

    for ep in range(1, args.epochs_more + 1):
        model.train()
        tr_loss_sum, tr_r_sum, tr_n = 0.0, 0.0, 0
        for xb, yb in train_ld:
            xb = _to(xb); yb = _to(yb)
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                yhat, _ = model(xb, bt_chunk=args.bt_chunk)
                r = pearsonr_batch(yhat, yb).mean()
                mse_abs = F.mse_loss(yhat, yb)
                mse_slope = slope_loss_simple(yhat, yb, beta=args.huber_beta)
                loss = args.w_r * (1 - r) + args.w_abs * mse_abs + args.lambda_slope * mse_slope
            if use_fp16:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            opt.zero_grad(set_to_none=True); sched.step()
            bs = xb.size(0); tr_loss_sum += loss.item() * bs; tr_r_sum += r.item() * bs; tr_n += bs
        tr_loss = tr_loss_sum / tr_n; tr_r = tr_r_sum / tr_n

        model.eval()
        va_loss_sum, va_r_sum, va_n = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_ld:
                xb = _to(xb); yb = _to(yb)
                with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                    yhat, _ = model(xb, bt_chunk=args.bt_chunk)
                    r = pearsonr_batch(yhat, yb).mean()
                    mse_abs = F.mse_loss(yhat, yb)
                    mse_slope = slope_loss_simple(yhat, yb, beta=args.huber_beta)
                    loss = args.w_r * (1 - r) + args.w_abs * mse_abs + args.lambda_slope * mse_slope
                bs = xb.size(0); va_loss_sum += loss.item() * bs; va_r_sum += r.item() * bs; va_n += bs
        va_loss = va_loss_sum / va_n; va_r = va_r_sum / va_n

        hist['train_loss'].append(tr_loss); hist['val_loss'].append(va_loss)
        hist['train_r'].append(tr_r);       hist['val_r'].append(va_r)
        print(f"[FT] Ep{ep:02d} | train {tr_loss:.4f} r={tr_r:.3f} | val {va_loss:.4f} r={va_r:.3f}")

        improved = (best_val - va_loss) > float(args.min_delta)
        if improved:
            best_val = va_loss; wait = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(args.outdir, 'finetune_best.pt'))
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[FT] Early stop at ep {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    plot_curves(hist, args.outdir)

    with torch.no_grad():
        xb, yb = next(iter(val_ld)) if len(val_ld)>0 else next(iter(test_ld))
        xb = _to(xb); yhat, A_t = model(xb, bt_chunk=args.bt_chunk)
        A = A_t.detach().cpu().numpy()

    info, ch_names, _ = make_biosemi64_info(n_ch=n_ch, sfreq=fs)
    try:
        A0 = model.graph.A0.detach().cpu().numpy()
    except Exception:
        A0 = None
    plot_graphs(A, A0, info, args.outdir)

    with torch.no_grad():
        for xb_te, yb_te in test_ld:
            xb_te = _to(xb_te); yhat_te, _ = model(xb_te, bt_chunk=args.bt_chunk)
            x_win = xb_te[0].detach().cpu().numpy(); y_win = yb_te[0].detach().cpu().numpy()
            yhat_win = yhat_te[0].detach().float().cpu().numpy(); break
    vis_dir = ensure_dir(os.path.join(args.outdir, 'vis'))
    path = plot_test_window_compare(x_win, y_win, yhat_win, fs, ch_names, vis_dir)
    print(f"[FT] wrote {path}")

    val_r = evaluate(model, val_ld, args.device, args.bt_chunk, amp_dtype)
    test_r = evaluate(model, test_ld, args.device, args.bt_chunk, amp_dtype)
    print(f"[FT] Final Val r={val_r:.4f} | Test r={test_r:.4f}")


if __name__ == '__main__':
    main()
