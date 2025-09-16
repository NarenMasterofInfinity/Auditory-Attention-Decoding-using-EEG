#!/usr/bin/env python3
"""
Temporal-first → Spatial (Graph) training:
Shared per-channel GRU → GraphEncoder → MLP head

- 70/10/20 window split (fixed seed per subject)
- Early stopping (best weights restored)
- Loss: w_r * (1 - Pearson r) + w_abs * MSE
- Mixed precision (bf16/fp16)
- Memory-friendly graph time-chunking (bt_chunk)
- Saves A0.csv, A_final.csv, plots, and a test comparison figure
- Loops over subjects and appends per-subject r to a CSV

Assumes:
  helper.subject_eeg_env_ab(PREPROC_DIR, subj_id) -> (eeg[T,C], env[T], fs, att_AB)
  GraphEncoder in GraphEncoder1.py or graph_encoder_sparse.py
"""

import os, math, time, argparse, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import csv
from GraphEncoder1 import GraphEncoder

import helper


# -------------------- utils --------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True); return path

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
    """Yields: EEG_win [W,C], env_win [W]"""
    def __init__(self, X, y, win, hop):
        self.X = X.astype(np.float32); self.y = y.astype(np.float32)
        self.win = int(win); self.hop = int(hop)
        self.spans = window_indices(len(X), self.win, self.hop)
    def __len__(self): return len(self.spans)
    def __getitem__(self, i):
        a, b = self.spans[i]
        return self.X[a:b], self.y[a:b]

def split_indices(num_items, train_ratio=0.7, val_ratio=0.1, seed=123):
    all_idx = np.arange(num_items)
    rng = np.random.default_rng(seed); rng.shuffle(all_idx)
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


# -------------------- plotting --------------------

def plot_training_curves(hist, outdir):
    ensure_dir(outdir)
    plt.figure(figsize=(6,4))
    plt.plot(hist['train_loss'], label='train')
    plt.plot(hist['val_loss'], label='val')
    plt.xlabel('epoch'); plt.ylabel('loss (w_r·(1−r) + w_abs·MSE)')
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
                '-', lw=0.5 + 3.0*(w/(A.max()+1e-8)), color='tab:blue', alpha=0.5)
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
                '-', lw=0.5 + 3.0*(w/(D.max()+1e-8)), color='tab:orange', alpha=0.8)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title('Top strengthened edges (A−A₀)+')
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


# -------------------- temporal-first model --------------------

class SharedGRUTemporal(nn.Module):
    """
    Shared per-channel GRU.
    Input:  X [B, T, C]
    Output: H [B, T, C, d_t]
    The same GRU weights are applied to each channel sequence (input size = 1).
    """
    def __init__(self, hidden=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=num_layers,
                          batch_first=True, dropout=0.0 if num_layers == 1 else dropout)
        self.do = nn.Dropout(dropout)
    def forward(self, X):
        B, T, C = X.shape
        x1 = X.transpose(1,2).reshape(B*C, T, 1)
        h, _ = self.gru(x1)            # [B*C, T, hidden]
        h = self.do(h)
        H = h.reshape(B, C, T, -1).transpose(1,2)  # [B, T, C, hidden]
        return H
class CausalDWConv(nn.Module):
    """Depthwise causal conv (kernel=9 by default) for local smoothing."""
    def __init__(self, d, kernel=9, dropout=0.1):
        super().__init__()
        self.pad = kernel - 1
        self.dw  = nn.Conv1d(d, d, kernel, groups=d)
        self.pw  = nn.Conv1d(d, d, 1)
        self.do  = nn.Dropout(dropout)
        self.act = nn.SiLU()
    def forward(self, x):                            # x: [B,T,D]
        x = x.transpose(1,2)                         # [B,D,T]
        y = F.pad(x, (self.pad, 0))                  # causal
        y = self.pw(self.act(self.dw(y)))
        y = y[:, :, :x.size(-1)].transpose(1,2)      # [B,T,D]
        return self.do(y)

class MultiShiftCausalMHSA(nn.Module):
    """
    Local causal attention with ALiBi and a small set of integer time shifts for K/V.
    Shared across channels if you fold B*C.
    """
    def __init__(self, d, heads=2, window=64, shifts=(0,4,8,12), dropout=0.1):
        super().__init__()
        assert d % heads == 0
        self.d = d; self.h = heads; self.dh = d // heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d); self.v = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
        self.do = nn.Dropout(dropout)
        self.window = window
        self.shifts = list(shifts)
        # per-head logits over shifts (learned softmax)
        self.shift_logits = nn.Parameter(torch.zeros(heads, len(self.shifts)))
        # ALiBi slopes per head (positive -> larger past penalty)
        self.alibi_slope = nn.Parameter(torch.ones(heads) * 0.01)

    def _split(self, x):  # [B,T,D] -> [B,H,T,dh]
        B,T,_ = x.shape; x = x.view(B,T,self.h,self.dh).transpose(1,2)
        return x
    def _merge(self, x):  # [B,H,T,dh] -> [B,T,D]
        B,H,T,dh = x.shape; return x.transpose(1,2).contiguous().view(B,T,H*dh)

    def forward(self, x):  # x: [B,T,D], causal local
        B,T,D = x.shape
        q = self._split(self.q(x))                   # [B,H,T,dh]
        k0 = self._split(self.k(x))
        v0 = self._split(self.v(x))

        # build local causal mask indices
        w = self.window
        # We’ll compute attention in full then zero out logits outside the window to keep code short.
        # (T is small in your windows; otherwise implement sliding blocks.)
        # Build ALiBi: per-head linear penalty with distance (only past allowed)
        pos = torch.arange(T, device=x.device)
        dist = (pos[None, :] - pos[:, None]).clamp(min=0).float()  # [T,T], 0 for future
        alibi = (-self.alibi_slope.view(1,self.h,1,1)) * dist      # [1,H,T,T]

        # Build multi-shift K/V and combine
        # Shift along time (right shift => attend further back)
        Ks, Vs = [], []
        for s in self.shifts:
            if s == 0:
                Ks.append(k0); Vs.append(v0)
            else:
                pad = torch.zeros_like(k0[:, :, :s, :])
                Ks.append(torch.cat([pad, k0[:, :, :-s, :]], dim=2))
                padv = torch.zeros_like(v0[:, :, :s, :])
                Vs.append(torch.cat([padv, v0[:, :, :-s, :]], dim=2))
        K = torch.stack(Ks, dim=2)  # [B,H,S,T,dh]
        V = torch.stack(Vs, dim=2)  # [B,H,S,T,dh]

        # attention logits per shift
        # q @ k^T over time: [B,H,T,dh] x [B,H,S,T,dh] -> [B,H,S,T,T]
        qn = q / math.sqrt(self.dh)
        att = torch.einsum('bhtd,bhsTd->bhstT', qn, K)  # s indexes shifts

        # Window mask (keep only last w steps)
        if w is not None and w < T:
            idx = torch.arange(T, device=x.device)
            keep = (idx[None,:] - idx[:,None])  # [T,T], >=0 past, <w window
            keep = (keep >= 0) & (keep < w)
            mask = (~keep).float() * -1e4       # large negative outside window
            att = att + mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # broadcast to [B,H,S,T,T]

        # Causal mask (no future)
        causal = (dist == 0)  # dist>0 for past, 0 on diag; we already masked future via dist<0 in 'keep'
        # Add ALiBi bias (only over T,T dims)
        att = att + alibi.unsqueeze(2)          # [1,H,1,T,T] -> broadcast

        # Mix shifts with learned softmax per head
        pi = torch.softmax(self.shift_logits, dim=-1)     # [H,S]
        att = att + torch.log(pi[None, :, :, None, None] + 1e-8)  # add log-weights

        # Softmax over keys’ time dimension
        att = att.logsumexp(dim=2)                # sum over shifts in log-space -> [B,H,T,T]
        att = torch.softmax(att, dim=-1)

        # Context
        ctx = torch.einsum('bhTT,bhsTd->bhtd', att, V)    # soft mix over shifts already applied
        out = self._merge(ctx)                            # [B,T,D]
        out = self.proj(out)
        return out
class TemporalFirstGraphModel(nn.Module):
    """
    Shared per-channel GRU (temporal) → GraphEncoder (spatial) → residual bypass to head.

    Input:  eeg [B,T,C]
    Output: yhat [B,T], A [C,C]
    """
    def __init__(self, n_ch, pos, d_t=64, use_raw=True,
                 d_model=128, L=3, k=8, heads=4, dropout=0.1):
        super().__init__()
        # temporal encoder (shared GRU across channels)
        self.gru = nn.GRU(input_size=1, hidden_size=d_t, num_layers=1, batch_first=True)
        self.do  = nn.Dropout(dropout)
        self.use_raw = use_raw

        # graph encoder sees per-node features of size d_t (+1 if raw EEG appended)
        d_in = d_t + (1 if use_raw else 0)
        self.graph = GraphEncoder(
            pos=torch.tensor(pos, dtype=torch.float32),
            d_in=d_in, d_model=d_model, L=L, k=k, heads=heads, dropout=dropout
        )

        # bypass: pool GRU features across channels → project to d_model
        self.bypass_proj = nn.Linear(d_t, d_model)

        # prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.SiLU(), nn.Linear(64, 1)
        )

    def forward(self, eeg, bt_chunk=None):
        B, T, C = eeg.shape
        # per-channel GRU with shared weights
        x = eeg.transpose(1, 2).reshape(B*C, T, 1)  # [B*C,T,1]
        h, _ = self.gru(x)                          # [B*C,T,d_t]
        h = self.do(h)
        H = h.reshape(B, C, T, -1).transpose(1, 2)  # [B,T,C,d_t]

        # graph input (optionally concat raw eeg)
        Xin = H if not self.use_raw else torch.cat([H, eeg.unsqueeze(-1)], dim=-1)  # [B,T,C,d_in]

        # graph path → S_graph [B,T,d_model], adjacency A [C,C]
        try:
            _, S_graph, A = self.graph(Xin, bt_chunk=bt_chunk)
        except TypeError:
            _, S_graph, A = self.graph(Xin)

        # bypass: mean over channels, project to d_model
        S0 = H.mean(dim=2)                 # [B,T,d_t]
        S0 = self.bypass_proj(S0)          # [B,T,d_model]

        # residual fusion then head
        S = S_graph + S0                   # [B,T,d_model]
        yhat = self.head(S).squeeze(-1)    # [B,T]
        return yhat, A


# -------------------- train / eval --------------------

def pearsonr_batch(yhat, y, eps=1e-8):
    yhat = yhat - yhat.mean(dim=1, keepdim=True)
    y    = y    - y.mean(dim=1, keepdim=True)
    num = (yhat * y).sum(dim=1)
    den = torch.sqrt((yhat**2).sum(dim=1) * (y**2).sum(dim=1) + eps)
    return num / (den + eps)

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

def evaluate(model, loader, device, bt_chunk, amp_dtype):
    model.eval(); r_sum, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                yhat, _ = model(xb, bt_chunk=bt_chunk); r = pearsonr_batch(yhat, yb).mean()
            r_sum += r.item() * xb.size(0); n += xb.size(0)
    return r_sum / max(1, n)

def train_one_subject(preproc_dir, subj_id, epochs, batch, win_sec, hop_sec, lr, device,
                      outdir, k, heads, blocks, bt_chunk, amp,
                      workers, prefetch, accum, compile_flag,
                      patience, min_delta, warmup_pct, final_lr_pct,
                      w_r=0.7, w_abs=0.3, d_t=64, use_raw=True):

    ensure_dir(outdir)

    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
    try: torch.backends.cudnn.benchmark = True
    except Exception: pass

    eeg, env, fs, _ = helper.subject_eeg_env_ab(preproc_dir, subj_id)
    X = eeg.astype(np.float32); y = env.astype(np.float32)
    X, _, _ = zscore_train(X)
    y, _, _ = zscore_train(y[:, None]); y = y[:, 0]

    win = int(round(win_sec * fs)); hop = int(round(hop_sec * fs))
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

    model = TemporalFirstGraphModel(n_ch=n_ch, pos=pos, d_t=d_t, use_raw=use_raw,
                                    d_model=128, L=blocks, k=k, heads=heads, dropout=0.1).to(device)

    if compile_flag:
        try: model = torch.compile(model, mode='max-autotune')
        except Exception as e: print("torch.compile failed (continuing):", str(e))

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
                loss = w_r * (1 - r) + w_abs * mse_abs
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
                    loss = w_r * (1 - r) + w_abs * mse_abs
                va_loss_sum += loss.item() * xb.size(0)
                va_r_sum += r.item() * xb.size(0)
                va_n += xb.size(0)

        va_loss = va_loss_sum / va_n
        va_r = va_r_sum / va_n

        hist['train_loss'].append(tr_loss); hist['val_loss'].append(va_loss)
        hist['train_r'].append(tr_r);       hist['val_r'].append(va_r)
        sec = time.time() - t0
        print(f"Ep{ep:02d} | train {tr_loss:.4f} r={tr_r:.3f} | val {va_loss:.4f} r={va_r:.3f} | {sec:.1f}s")

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

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # final metrics
    val_r_final = evaluate(model, val_ld, device, bt_chunk, amp_dtype)
    test_r_final = evaluate(model, test_ld, device, bt_chunk, amp_dtype)

    # graphs & visualizations (use one batch)
    with torch.no_grad():
        pick_loader = val_ld if len(val_ld) > 0 else test_ld
        xb, yb = next(iter(pick_loader))
        xb = xb.to(device, non_blocking=True)
        yhat, A_t = model(xb, bt_chunk=bt_chunk)
        A = A_t.detach().cpu().numpy()

    plot_training_curves(hist, outdir)
    plot_adjacency_heatmap(A, outdir, name='A_heatmap.png', title='Blended adjacency A')
    info, ch_names, _pos = make_biosemi64_info(n_ch=n_ch, sfreq=fs)
    plot_sensor_edges(A, info, outdir, topk=120, name='A_edges_sensors.png')

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

    # test window comparison
    with torch.no_grad():
        for xb_te, yb_te in test_ld:
            xb_te = xb_te.to(device, non_blocking=True)
            yhat_te, _ = model(xb_te, bt_chunk=bt_chunk)
            x_win = xb_te[0].detach().cpu().numpy()
            y_win = yb_te[0].detach().cpu().numpy()
            yhat_win = yhat_te[0].detach().float().cpu().numpy()
            break
    vis_dir = ensure_dir(os.path.join(outdir, "vis"))
    _, chn, _ = make_biosemi64_info(n_ch=n_ch, sfreq=fs)
    comp_path = plot_test_window_compare(x_win, y_win, yhat_win, fs, chn, vis_dir,
                                         eeg_channels_to_show=6, fname='test_compare_window.png')
    print(f"[viz] wrote {comp_path}")

    return float(val_r_final), float(test_r_final)


# -------------------- mne utils --------------------

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


# -------------------- main --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preproc_dir', type=str, default='/home/naren-root/Dataset/DATA_preproc')
    p.add_argument('--outdir', type=str, default='outputs_full')
    p.add_argument('--subjects', type=str, default='1-18', help='e.g., "1-18" or "1,3,5"')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--win_sec', type=float, default=5.0)
    p.add_argument('--hop_sec', type=float, default=2.5)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--blocks', type=int, default=3)
    p.add_argument('--bt_chunk', type=int, default=256)
    p.add_argument('--amp', type=str, choices=['none', 'fp16', 'bf16'], default='bf16')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--prefetch', type=int, default=2)
    p.add_argument('--accum', type=int, default=2)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--min_delta', type=float, default=1e-3)
    p.add_argument('--warmup_pct', type=float, default=0.05)
    p.add_argument('--final_lr_pct', type=float, default=0.1)
    p.add_argument('--w_r', type=float, default=0.7)
    p.add_argument('--w_abs', type=float, default=0.3)
    p.add_argument('--d_t', type=int, default=64)
    p.add_argument('--use_raw', action='store_true')
    args = p.parse_args()

    ensure_dir(args.outdir)
    summ_path = os.path.join(args.outdir, "summary_pearsonr_temporal_first.csv")
    if not os.path.exists(summ_path):
        with open(summ_path, 'w', newline='') as f:
            csv.writer(f).writerow(['subject', 'val_r', 'test_r'])

    # parse subjects
    subs = []
    if '-' in args.subjects:
        a, b = args.subjects.split('-'); subs = list(range(int(a), int(b) + 1))
    else:
        subs = [int(s) for s in args.subjects.split(',') if s.strip()]

    for subj in subs:
        sx = ensure_dir(os.path.join(args.outdir, f"S{subj}"))
        print(f"\n=== Subject {subj} → {sx} ===")
        val_r, test_r = train_one_subject(
            args.preproc_dir, subj, args.epochs, args.batch,
            args.win_sec, args.hop_sec, args.lr, args.device, sx,
            args.k, args.heads, args.blocks, args.bt_chunk, args.amp,
            args.workers, args.prefetch, args.accum, args.compile,
            args.patience, args.min_delta, args.warmup_pct, args.final_lr_pct,
            args.w_r, args.w_abs, args.d_t, args.use_raw
        )
        print(f"Subject {subj}: Val r={val_r:.4f} | Test r={test_r:.4f}")
        with open(summ_path, 'a', newline='') as f:
            csv.writer(f).writerow([subj, val_r, test_r])

    print(f"\nWrote {summ_path}")


if __name__ == '__main__':
    main()
