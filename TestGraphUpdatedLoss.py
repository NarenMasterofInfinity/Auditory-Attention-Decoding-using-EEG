#!/usr/bin/env python3
"""
Train every subject for audio envelope reconstruction with:
TemporalStem → GraphEncoder → MLP head

Adds:
- 70/10/20 Train/Val/Test split (window-level, reproducible)
- Early stopping on Val; best-weight restore
- Test-set evaluation after training
- Subject loop (S12..S18, skipping S14 per your last run)
- Create per-subject outdir outputs/SX (auto-created)
- Save initial A0.csv and final A_final.csv for the graph
- Save plots (training curves, A heatmaps, edge maps, topomap)
- NEW: Slope-aware loss
- NEW: Test visualization comparing EEG (4–6 channels), true env, predicted env

Loss:
  loss = w_r * (1 − PearsonR) + w_abs * MSE(ŷ, y) + λ_slope * MSE(Δŷ, Δy)

Assumes:
  helper.subject_eeg_env_ab(PREPROC_DIR, subj_id) -> (eeg[T,C], env[T], fs, att_AB)
  TemporalStem and GraphEncoder modules are available (as in your project)
"""

import os, math, time, argparse, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

try:
    from FeatureExpander import TemporalStem
except Exception:
    from temporal_stem import TemporalStem

try:
    from GraphEncoder1 import GraphEncoder
except Exception:
    from graph_encoder_sparse import GraphEncoder

import helper


# ---------------------- utils ----------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

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

class EEGEnvDataset(Dataset):
    """Returns (EEG_win[W,C], env_win[W])"""
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


# ---------------------- model ----------------------

class ERGraphModel(nn.Module):
    """TemporalStem → GraphEncoder → MLP head → envelope per time step"""
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
        yhat = self.head(S).squeeze(-1)
        return yhat, A


# ---------------------- loss / metrics ----------------------

def pearsonr_batch(yhat, y, eps=1e-8):
    yhat = yhat - yhat.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (yhat * y).sum(dim=1)
    den = torch.sqrt((yhat**2).sum(dim=1) * (y**2).sum(dim=1) + eps)
    return num / (den + eps)

def slope_mse(yhat, y, fs, beta =0.02):
    """MSE between first temporal differences Δŷ and Δy."""
    dyh = (yhat[:, 2:] - yhat[:, :-2]) * (fs / 2.0)
    dy  = (y[:, 2:]   - y[:, :-2])   * (fs / 2.0)
    return F.smooth_l1_loss(dyh, dy, beta=beta)

def make_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=0, final_lr_pct=0.1):
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


# ---------------------- plots ----------------------

def plot_training_curves(hist, outdir):
    ensure_dir(outdir)
    plt.figure(figsize=(6,4))
    plt.plot(hist['train_loss'], label='train')
    plt.plot(hist['val_loss'], label='val')
    plt.xlabel('epoch'); plt.ylabel('loss (w_r·(1−r) + w_abs·MSE + λ_slope·MSE(Δ))')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_curve.png'), dpi=150); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(hist['train_r'], label='train r')
    plt.plot(hist['val_r'], label='val r')
    plt.xlabel('epoch'); plt.ylabel('Pearson r')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_r.png'), dpi=150); plt.close()

def plot_adjacency_heatmap(A, outdir, name='A_heatmap.png', title='Blended adjacency A'):
    ensure_dir(outdir)
    plt.figure(figsize=(6,5))
    plt.imshow(A, cmap='viridis')
    plt.colorbar(label='weight')
    plt.title(title)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, name), dpi=150); plt.close()

def plot_sensor_edges(A, info, outdir, topk=120, name='A_edges_sensors.png', title='Top edges on sensor map'):
    ensure_dir(outdir)
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
        ax.plot([P2[i,0], P2[j,0]], [P2[i,1], P2[j,1]],
                '-', lw=0.5 + 3.0*(w/(A.max()+1e-8)), color='tab:blue', alpha=0.5, zorder=2)
    circ = plt.Circle((0,0), 1.05, color='k', fill=False, lw=1); ax.add_artist(circ)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, name), dpi=150); plt.close()

def plot_sensor_edges_delta(A, A0, info, outdir, topk=80):
    ensure_dir(outdir)
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
        ax.plot([P2[i,0], P2[j,0]], [P2[i,1], P2[j,1]],
                '-', lw=0.5 + 3.0*(w/(D.max()+1e-8)), color='tab:orange', alpha=0.8, zorder=2)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(f'Top strengthened edges (A−A₀)+')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'A_edges_delta.png'), dpi=150); plt.close()

def plot_in_strength_topomap(A, info, outdir):
    ensure_dir(outdir)
    W = A.copy(); np.fill_diagonal(W, 0.0)
    in_strength = W.sum(axis=0)
    fig, ax = plt.subplots(figsize=(5,4))
    mne.viz.plot_topomap(in_strength, info, axes=ax, show=False)
    ax.set_title('In-strength (col-sum of A)')
    fig.savefig(os.path.join(outdir, 'in_strength_topomap.png'), dpi=150); plt.close(fig)

def plot_test_window_compare(x_win, y_win, yhat_win, fs, ch_names, outdir,
                             eeg_channels_to_show=6, fname='test_compare_window.png'):
    """
    Save a figure with (top) 4-6 EEG channels (z-scored & offset) and (bottom) true vs predicted env.
    """
    ensure_dir(outdir)
    W, C = x_win.shape
    t = np.arange(W) / float(fs)

    k = min(max(4, eeg_channels_to_show), min(6, C))
    chans = list(range(k))  # first k channels; customize if needed

    # z-score each selected channel within the window
    Xz = (x_win[:, chans] - x_win[:, chans].mean(0, keepdims=True)) / (x_win[:, chans].std(0, keepdims=True) + 1e-8)

    # offsets for visibility
    offsets = np.arange(k) * 3.0

    fig = plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(2,1,1)
    for i, ch in enumerate(chans):
        ax1.plot(t, Xz[:, i] + offsets[i], lw=0.8, label=(ch_names[ch] if ch_names and ch < len(ch_names) else f'CH{ch}'))
    ax1.set_ylabel('EEG (z, offsets)')
    ax1.set_title(f'EEG window (first {k} channels)')
    ax1.legend(fontsize=8, ncol=min(3, k), loc='upper right')
    ax1.grid(alpha=0.2)

    ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax2.plot(t, y_win, lw=1.2, label='True env')
    ax2.plot(t, yhat_win, lw=1.2, label='Pred env')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Envelope')
    ax2.legend(); ax2.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150); plt.close(fig)
    return path


# ---------------------- train / eval ----------------------

def split_indices(num_items, train_ratio=0.7, val_ratio=0.1, seed=123):
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
    class _Sub(Dataset):
        def __init__(self, base, sel):
            self.base = base
            self.sel = np.array(sel, dtype=int)
        def __len__(self):
            return len(self.sel)
        def __getitem__(self, i):
            return self.base[self.sel[i]]
    return _Sub(ds, idx)

def evaluate(model, loader, device, bt_chunk, amp_dtype):
    model.eval()
    r_sum, n = 0.0, 0
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
                      outdir, k, heads, blocks, bt_chunk, amp,
                      workers, prefetch, accum, compile_flag,
                      patience, min_delta, warmup_pct, final_lr_pct,
                      w_r=0.7, w_abs=0.3, lambda_slope=0.2):

    # create subject dir early
    ensure_dir(outdir)

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

    model = ERGraphModel(n_ch=n_ch, pos=pos, d_stem=256, d_lift=127, d_in=128, d_model=128,
                         L=blocks, k=k, heads=heads, dropout=0.1, causal=True).to(device)

    if compile_flag:
        try:
            model = torch.compile(model, mode='max-autotune')
        except Exception as e:
            print("torch.compile failed (continuing):", str(e))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = max(1, math.ceil(len(train_ld) / max(1, accum)))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(warmup_pct * total_steps)
    sched = make_warmup_cosine_scheduler(opt, total_steps, warmup_steps=warmup_steps, final_lr_pct=final_lr_pct)

    use_fp16 = (amp == 'fp16'); use_bf16 = (amp == 'bf16')
    amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    best_val = float('inf'); best_state = None; best_epoch = -1; wait = 0
    hist = {'train_loss': [], 'val_loss': [], 'train_r': [], 'val_r': []}
    def _to(x): return x.to(device, non_blocking=True)

    global_step = 0
    for ep in range(1, epochs + 1):
        print(f"EPOCH {ep:02d}")
        model.train()
        tr_loss_sum, tr_r_sum, tr_n = 0.0, 0.0, 0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        for step, (xb, yb) in enumerate(train_ld, 1):
            xb = _to(xb); yb = _to(yb)
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                yhat, _ = model(xb, bt_chunk=bt_chunk)
                r = pearsonr_batch(yhat, yb).mean()
                mse_abs = F.mse_loss(yhat, yb)
                mse_slope = slope_mse(yhat, yb, fs)
                loss = w_r * (1 - r) + w_abs * mse_abs + lambda_slope * mse_slope
                loss = loss / max(1, accum)

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step % max(1, accum)) == 0:
                if use_fp16: scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_fp16:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                global_step += 1

            tr_loss_sum += loss.item() * xb.size(0) * max(1, accum)
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
                    mse_abs = F.mse_loss(yhat, yb)
                    mse_slope = slope_mse(yhat, yb, fs)
                    loss = w_r * (1 - r) + w_abs * mse_abs + lambda_slope * mse_slope
                va_loss_sum += loss.item() * xb.size(0)
                va_r_sum += r.item() * xb.size(0)
                va_n += xb.size(0)

        va_loss = va_loss_sum / va_n
        va_r = va_r_sum / va_n
        hist['train_loss'].append(tr_loss); hist['val_loss'].append(va_loss)
        hist['train_r'].append(tr_r);       hist['val_r'].append(va_r)

        sec = time.time() - t0
        print(f"train {tr_loss:.4f} (r={tr_r:.3f}) | val {va_loss:.4f} (r={va_r:.3f}) | {sec:.1f}s")

        improved = (best_val - va_loss) > float(min_delta)
        if improved:
            best_val = va_loss; best_epoch = ep; wait = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(outdir, 'best_model.pt'))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {ep} (best epoch {best_epoch}, val {best_val:.4f})")
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # final metrics
    val_r_final = evaluate(model, val_ld, device, bt_chunk, amp_dtype)
    test_r_final = evaluate(model, test_ld, device, bt_chunk, amp_dtype)

    # one val (or test) batch for A and test visualization
    with torch.no_grad():
        pick_loader = val_ld if len(val_ld) > 0 else test_ld
        xb, yb = next(iter(pick_loader))
        xb = xb.to(device, non_blocking=True)
        yhat, A_t = model(xb, bt_chunk=bt_chunk)
        A = A_t.detach().cpu().numpy()

    # ensure outdir exists
    ensure_dir(outdir)

    # core plots & graph CSVs
    plot_training_curves(hist, outdir)
    plot_adjacency_heatmap(A, outdir, name='A_heatmap.png', title='Blended adjacency A')
    plot_sensor_edges(A, info, outdir, topk=120, name='A_edges_sensors.png')

    # save graph priors/finals
    A0 = None
    try:
        A0 = model.graph.A0.detach().cpu().numpy()
        np.savetxt(os.path.join(outdir, 'A0.csv'), A0, delimiter=',')
        plot_adjacency_heatmap(A0, outdir, name='A0_heatmap.png', title='Initial prior A0')
    except Exception:
        pass
    np.savetxt(os.path.join(outdir, 'A_final.csv'), A, delimiter=',')

    plot_sensor_edges_delta(A, A0, info, outdir, topk=80)
    plot_in_strength_topomap(A, info, outdir)

    # ---------- NEW: Test visualization (EEG + true vs pred envelope) ----------
    # Take a real window from the test split and plot it.
    with torch.no_grad():
        for xb_te, yb_te in test_ld:
            xb_te = xb_te.to(device, non_blocking=True)
            yhat_te, _ = model(xb_te, bt_chunk=bt_chunk)
            # Use the first item of this batch for the plot
            x_win = xb_te[0].detach().cpu().numpy()      # [W,C]
            y_win = yb_te[0].detach().cpu().numpy()      # [W]
            yhat_win = yhat_te[0].detach().float().cpu().numpy()  # [W]
            break

    # Make sure we have channel names and fs
    _, ch_names, _ = make_biosemi64_info(n_ch=n_ch, sfreq=fs)
    vis_dir = ensure_dir(os.path.join(outdir, "vis"))
    comp_path = plot_test_window_compare(x_win, y_win, yhat_win, fs, ch_names, vis_dir,
                                         eeg_channels_to_show=6, fname='test_compare_window.png')
    print(f"[viz] wrote {comp_path}")

    return float(val_r_final), float(test_r_final)


# ---------------------- main ----------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preproc_dir', type=str, required=True)
    p.add_argument('--outdir', type=str, default='outputs')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--win_sec', type=float, default=5.0)
    p.add_argument('--hop_sec', type=float, default=2.5)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--blocks', type=int, default=3)
    p.add_argument('--bt_chunk', type=int, default=128)
    p.add_argument('--amp', type=str, choices=['none', 'fp16', 'bf16'], default='bf16')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--prefetch', type=int, default=2)
    p.add_argument('--accum', type=int, default=2)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--min_delta', type=float, default=1e-3)
    p.add_argument('--warmup_pct', type=float, default=0.05)
    p.add_argument('--final_lr_pct', type=float, default=0.1)
    # loss weights
    p.add_argument('--w_r', type=float, default=0.7, help='weight on (1 - Pearson r)')
    p.add_argument('--w_abs', type=float, default=0.3, help='weight on absolute MSE')
    p.add_argument('--lambda_slope', type=float, default=0.2, help='weight on slope MSE')
    args = p.parse_args()

    ensure_dir(args.outdir)

    import csv
    summ_path = os.path.join(args.outdir, "summary_pearsonr_best.csv")
    if not os.path.exists(summ_path):
        ensure_dir(os.path.dirname(summ_path))
        with open(summ_path, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['subject', 'val_r', 'test_r'])

    for subj in range(1, 19):
        
        sx = ensure_dir(os.path.join(args.outdir, f"S{subj}"))
        print(f"\n=== Subject {subj} → {sx} ===")
        val_r, test_r = train_one_subject(
            args.preproc_dir, subj, args.epochs, args.batch,
            args.win_sec, args.hop_sec, args.lr, args.device, sx,
            args.k, args.heads, args.blocks, args.bt_chunk, args.amp,
            args.workers, args.prefetch, args.accum, args.compile,
            args.patience, args.min_delta, args.warmup_pct, args.final_lr_pct,
            args.w_r, args.w_abs, args.lambda_slope
        )
        print(f"Subject {subj}: Val r={val_r:.4f} | Test r={test_r:.4f}")

        with open(summ_path, 'a', newline='') as f:
            csv.writer(f).writerow([subj, val_r, test_r])
        print(f"Appended results to {summ_path}")

    print(f"\nWrote {summ_path}")


if __name__ == '__main__':
    main()
