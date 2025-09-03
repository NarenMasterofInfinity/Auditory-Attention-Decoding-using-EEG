#!/usr/bin/env python3
"""
main.py

Run EEGGraphConformer across subjects (1..18) with 70/10/20 Train/Val/Test split.
Per subject outputs in output_main/SX:
- training_curve.png, training_r.png
- A0.csv, A0_heatmap.png (prior adjacency if available)
- A_final.csv, A_heatmap.png
- A_edges_sensors.png, A_edges_delta.png, in_strength_topomap.png

Appends test Pearson r to output_main/summary_pearsonr.csv immediately after each subject finishes.
"""

import os, math, argparse, csv, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

from model import EEGGraphConformer


def zscore_train(x):
    """Channel-wise z-score using training statistics."""
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True) + 1e-8
    return (x - m) / s, m, s


def window_indices(T, win, hop):
    """Return array of (start, end) indices with step=hop and length=win."""
    idx, t = [], 0
    while t + win <= T:
        idx.append((t, t + win))
        t += hop
    return np.array(idx, dtype=int)


class EEGEnvDataset(Dataset):
    """From continuous arrays to overlapping windows."""
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
    """Return (mne.Info, ch_names, pos[N,3]). Falls back to ring if n_ch != 64."""
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


def pearsonr_batch(yhat, y, eps=1e-8):
    """Mean Pearson r over batch."""
    yhat = yhat - yhat.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (yhat * y).sum(dim=1)
    den = torch.sqrt((yhat**2).sum(dim=1) * (y**2).sum(dim=1) + eps)
    return num / (den + eps)


def pearsonr_loss(yhat, y):
    """1 - mean Pearson r."""
    return 1 - pearsonr_batch(yhat, y).mean()


def make_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=0, final_lr_pct=0.1):
    """Per-step LambdaLR with linear warm-up then cosine decay."""
    final_lr_pct = float(final_lr_pct)
    def lr_lambda(step):
        step = min(step, total_steps)
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if total_steps == warmup_steps:
            return final_lr_pct
        prog = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * prog))
        return final_lr_pct + (1 - final_lr_pct) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def plot_training_curves(hist, outdir):
    """Save loss and r curves."""
    plt.figure(figsize=(6,4))
    plt.plot(hist['train_loss'], label='train')
    plt.plot(hist['val_loss'], label='val')
    plt.xlabel('epoch'); plt.ylabel('loss (0.7·(1−r)+0.3·L1)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_curve.png'), dpi=150); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(hist['train_r'], label='train r')
    plt.plot(hist['val_r'], label='val r')
    plt.xlabel('epoch'); plt.ylabel('Pearson r')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_r.png'), dpi=150); plt.close()


def plot_adjacency_heatmap(A, outdir, name='A_heatmap.png', title='Blended adjacency A'):
    """Save adjacency heatmap."""
    plt.figure(figsize=(6,5))
    plt.imshow(A, cmap='viridis')
    plt.colorbar(label='weight')
    plt.title(title)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, name), dpi=150); plt.close()


def plot_sensor_edges(A, info, outdir, topk=120, name='A_edges_sensors.png', title='Top edges on sensor map'):
    """Plot strongest edges on 2D scalp."""
    pos = np.array([info.get_montage().get_positions()['ch_pos'][ch] for ch in info.ch_names])
    P2 = pos[:, :2]
    W = A.copy(); np.fill_diagonal(W, 0.0)
    triu = np.triu_indices_from(W, k=1); vals = W[triu]
    k = min(topk, len(vals)); thr = np.partition(vals, -k)[-k] if k > 0 else 0.0
    sel = W >= thr
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(P2[:,0], P2[:,1], s=30, zorder=3, c='k')
    for i, ch in enumerate(info.ch_names):
        ax.text(P2[i,0], P2[i,1], ch, fontsize=6, ha='center', va='center', color='white', zorder=4)
    idx = np.array(np.where(np.triu(sel, k=1))).T
    for i, j in idx:
        w = A[i, j]
        ax.plot([P2[i,0], P2[j,0]], [P2[i,1], P2[j,1]], '-', lw=0.5 + 3.0*(w/(A.max()+1e-8)), color='tab:blue', alpha=0.5, zorder=2)
    circ = plt.Circle((0,0), 1.05, color='k', fill=False, lw=1); ax.add_artist(circ)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, name), dpi=150); plt.close()


def plot_sensor_edges_delta(A, A0, info, outdir, topk=80):
    """Plot strengthened edges (A−A0)+."""
    if A0 is None:
        return
    D = np.clip(A - A0, 0, None)
    pos = np.array([info.get_montage().get_positions()['ch_pos'][ch] for ch in info.ch_names])
    P2 = pos[:, :2]
    np.fill_diagonal(D, 0.0)
    triu = np.triu_indices_from(D, k=1); vals = D[triu]
    k = min(topk, len(vals)); thr = np.partition(vals, -k)[-k] if k > 0 else 0.0
    sel = D >= thr
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(P2[:,0], P2[:,1], s=30, zorder=3, c='k')
    for i, ch in enumerate(info.ch_names):
        ax.text(P2[i,0], P2[i,1], ch, fontsize=6, ha='center', va='center', color='white', zorder=4)
    idx = np.array(np.where(np.triu(sel, k=1))).T
    for i, j in idx:
        w = D[i, j]
        ax.plot([P2[i,0], P2[j,0]], [P2[i,1], P2[j,1]], '-', lw=0.5 + 3.0*(w/(D.max()+1e-8)), color='tab:orange', alpha=0.8, zorder=2)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title('Top strengthened edges (A−A₀)+')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'A_edges_delta.png'), dpi=150); plt.close()


def plot_in_strength_topomap(A, info, outdir):
    """Column-sum topomap (who receives more)."""
    W = A.copy(); np.fill_diagonal(W, 0.0)
    in_strength = W.sum(axis=0)
    fig, ax = plt.subplots(figsize=(5,4))
    mne.viz.plot_topomap(in_strength, info, axes=ax, show=False)
    ax.set_title('In-strength (col-sum of A)')
    fig.savefig(os.path.join(outdir, 'in_strength_topomap.png'), dpi=150); plt.close(fig)


def split_indices(num_items, train_ratio=0.7, val_ratio=0.1, seed=123):
    """Return train/val/test index splits."""
    all_idx = np.arange(num_items)
    rng = np.random.default_rng(seed)
    rng.shuffle(all_idx)
    n_train = int(round(train_ratio * num_items))
    n_val = int(round(val_ratio * num_items))
    n_test = num_items - n_train - n_val
    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train+n_val]
    test_idx = all_idx[n_train+n_val:]
    return train_idx, val_idx, test_idx


def subset_dataset(ds, idx):
    """Dataset wrapper for index selection."""
    class _Sub(Dataset):
        def __init__(self, base, sel):
            self.base = base; self.sel = np.array(sel, dtype=int)
        def __len__(self):
            return len(self.sel)
        def __getitem__(self, i):
            return self.base[self.sel[i]]
    return _Sub(ds, idx)


def evaluate(model, loader, device, bt_chunk, amp_dtype):
    """Mean Pearson r over loader."""
    model.eval(); r_sum, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                yhat, _ = model(xb, bt_chunk=bt_chunk)
                r = pearsonr_batch(yhat, yb).mean()
            r_sum += r.item() * xb.size(0)
            n += xb.size(0)
    return r_sum / max(1, n)


def train_one_subject(preproc_dir, subj_id, epochs, batch, win_sec, hop_sec, lr, device,
                      outdir, k, heads_graph, blocks_graph, conf_depth, heads_conf, bt_chunk, amp,
                      workers, prefetch, patience, min_delta, warmup_pct, final_lr_pct, bias_mode):
    """Train one subject, save figures and CSVs, return (val_r, test_r)."""
    import helper
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
    try: torch.backends.cudnn.benchmark = True
    except Exception: pass

    eeg, env, fs, attAB = helper.subject_eeg_env_ab(preproc_dir, subj_id)
    X = eeg.astype(np.float32)
    y = env.astype(np.float32)
    X, _, _ = zscore_train(X)
    y, _, _ = zscore_train(y[:, None]); y = y[:, 0]

    win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))

    n_ch = X.shape[1]
    info, ch_names, pos = make_biosemi64_info(n_ch=n_ch, sfreq=fs)

    ds = EEGEnvDataset(X, y, win, hop)
    tr_idx, va_idx, te_idx = split_indices(len(ds), 0.7, 0.1, seed=1000 + subj_id)
    train_ds = subset_dataset(ds, tr_idx)
    val_ds   = subset_dataset(ds, va_idx)
    test_ds  = subset_dataset(ds, te_idx)

    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True,
                          num_workers=workers, pin_memory=True,
                          persistent_workers=(workers > 0), prefetch_factor=prefetch)
    val_ld = DataLoader(val_ds, batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=True,
                        persistent_workers=(workers > 0), prefetch_factor=prefetch)
    test_ld = DataLoader(test_ds, batch_size=batch, shuffle=False,
                         num_workers=workers, pin_memory=True,
                         persistent_workers=(workers > 0), prefetch_factor=prefetch)

    model = EEGGraphConformer(
        n_ch=n_ch, pos=pos,
        d_stem=256, d_lift=127, d_in=128, d_model=128,
        L_graph=blocks_graph, k=k, graph_heads=heads_graph, graph_dropout=0.1,
        conf_depth=conf_depth, conf_heads=heads_conf, ff_expansion=4, conf_dropout=0.1,
        kernel_size=9, bias_mode=bias_mode, max_rel=128, causal=True
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    steps_per_epoch = max(1, len(train_ld))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(warmup_pct * total_steps)
    sched = make_warmup_cosine_scheduler(opt, total_steps, warmup_steps=warmup_steps, final_lr_pct=final_lr_pct)

    use_fp16 = (amp == 'fp16'); use_bf16 = (amp == 'bf16')
    amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    best_val = float('inf'); best_state = None; wait = 0
    hist = {'train_loss': [], 'val_loss': [], 'train_r': [], 'val_r': []}
    def _to(x): return x.to(device, non_blocking=True)

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss_sum, tr_r_sum, tr_n = 0.0, 0.0, 0
        for xb, yb in train_ld:
            xb = _to(xb); yb = _to(yb)
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                yhat, _ = model(xb, bt_chunk=bt_chunk)
                r = pearsonr_batch(yhat, yb).mean()
                loss = 0.7 * (1 - r) + 0.3 * F.l1_loss(yhat, yb)
            if use_fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            tr_loss_sum += loss.item() * xb.size(0)
            tr_r_sum += r.item() * xb.size(0)
            tr_n += xb.size(0)
        tr_loss = tr_loss_sum / tr_n
        tr_r = tr_r_sum / tr_n

        model.eval()
        va_loss_sum, va_r_sum, va_n = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_ld:
                xb = _to(xb); yb = _to(yb)
                with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                    yhat, _ = model(xb, bt_chunk=bt_chunk)
                    r = pearsonr_batch(yhat, yb).mean()
                    loss = 0.7 * (1 - r) + 0.3 * F.l1_loss(yhat, yb)
                va_loss_sum += loss.item() * xb.size(0)
                va_r_sum += r.item() * xb.size(0)
                va_n += xb.size(0)
        va_loss = va_loss_sum / va_n
        va_r = va_r_sum / va_n

        hist['train_loss'].append(tr_loss); hist['val_loss'].append(va_loss)
        hist['train_r'].append(tr_r);       hist['val_r'].append(va_r)

        improved = (best_val - va_loss) > float(min_delta)
        if improved:
            best_val = va_loss; wait = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    val_r_final = evaluate(model, val_ld, device, bt_chunk, amp_dtype)
    test_r_final = evaluate(model, test_ld, device, bt_chunk, amp_dtype)

    with torch.no_grad():
        ref_loader = val_ld if len(val_ld) > 0 else test_ld
        xb, yb = next(iter(ref_loader))
        xb = xb.to(device, non_blocking=True)
        yhat, A_t = model(xb, bt_chunk=bt_chunk)
        A = A_t.detach().cpu().numpy()

    os.makedirs(outdir, exist_ok=True)
    plot_training_curves(hist, outdir)
    plot_adjacency_heatmap(A, outdir, name='A_heatmap.png', title='Blended adjacency A')
    plot_sensor_edges(A, info, outdir, topk=120, name='A_edges_sensors.png')

    try:
        A0 = model.graph.A0.detach().cpu().numpy()
        np.savetxt(os.path.join(outdir, 'A0.csv'), A0, delimiter=',')
        plot_adjacency_heatmap(A0, outdir, name='A0_heatmap.png', title='Initial prior A0')
    except Exception:
        A0 = None
    np.savetxt(os.path.join(outdir, 'A_final.csv'), A, delimiter=',')

    plot_sensor_edges_delta(A, A0, info, outdir, topk=80)
    plot_in_strength_topomap(A, info, outdir)

    return float(val_r_final), float(test_r_final)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preproc_dir', type=str, required=True)
    p.add_argument('--outdir', type=str, default='output_main')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--win_sec', type=float, default=5.0)
    p.add_argument('--hop_sec', type=float, default=2.5)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--heads_graph', type=int, default=4)
    p.add_argument('--blocks_graph', type=int, default=3)
    p.add_argument('--conf_depth', type=int, default=3)
    p.add_argument('--heads_conf', type=int, default=4)
    p.add_argument('--bt_chunk', type=int, default=128)
    p.add_argument('--amp', type=str, choices=['none','fp16','bf16'], default='bf16')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--prefetch', type=int, default=2)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--min_delta', type=float, default=1e-3)
    p.add_argument('--warmup_pct', type=float, default=0.05)
    p.add_argument('--final_lr_pct', type=float, default=0.1)
    p.add_argument('--bias_mode', type=str, choices=['rel','alibi'], default='rel')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, "summary_pearsonr.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['subject', 'val_r', 'test_r'])

    for subj in range(1, 19):
        sx = os.path.join(args.outdir, f"S{subj}")
        print(f"\n=== Subject {subj} → {sx} ===")
        val_r, test_r = train_one_subject(
            args.preproc_dir, subj, args.epochs, args.batch,
            args.win_sec, args.hop_sec, args.lr, args.device, sx,
            args.k, args.heads_graph, args.blocks_graph, args.conf_depth, args.heads_conf,
            args.bt_chunk, args.amp, args.workers, args.prefetch,
            args.patience, args.min_delta, args.warmup_pct, args.final_lr_pct, args.bias_mode
        )
        print(f"Subject {subj}: Val r={val_r:.4f} | Test r={test_r:.4f}")
        with open(summary_path, 'a', newline='') as f:
            w = csv.writer(f); w.writerow([subj, val_r, test_r])
        print(f"Appended results to {summary_path}")


if __name__ == '__main__':
    main()
