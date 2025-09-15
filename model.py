#!/usr/bin/env python3
"""
model.py (consolidated, overfit-reduced)

End-to-end: EEG → TemporalStem → GraphEncoder (memory-friendly fixed-K) → Conformer (causal, ALiBi) → ŷ
Training features:
- 70/10/20 contiguous split by time with boundary gap to avoid window leakage
- Train-only normalization (z-score from train segment, applied to val/test)
- On-the-fly EEG augment (channel dropout, light Gaussian noise, optional time-masking)
- Temporal smoothness penalty on outputs (TV-L1)
- Cosine LR with warm-up, early stopping, best-weight restore
- Mixed precision (fp16/bf16), gradient clipping, AdamW with stronger weight decay
- Per-epoch TEST evaluation; rows appended to outputs/test_scores_per_epoch.csv
- Per-subject artifacts in outputs/SX: curves, adjacency visuals, A0/A_final CSVs, best.pt, last.pt

Assumes:
  helper.subject_eeg_env_ab(PREPROC_DIR, subj_id) -> (eeg[T,C], env[T], fs, att_AB)
  graph_encoder_sparse.GraphEncoder implements forward(Xin[B,T,N,d_in], bt_chunk=None) -> (Z, S, A)
  temporal_stem.TemporalStem implements forward(eeg[B,T,N]) -> H0[B,T,d_stem]
"""

import os, math, time, argparse, csv, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import mne
import time
try:
    from FeatureExpander import TemporalStem
except Exception:
    from temporal_stem import TemporalStem

from GraphEncoder1 import GraphEncoder


class MacaronFFN(nn.Module):
    """Macaron FFN with 0.5 residual scaling."""
    def __init__(self, d_model, expansion=4, dropout=0.2):
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
    """ALiBi or learned relative bias for causal attention."""
    def __init__(self, heads, max_rel=128, bias_mode="alibi"):
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
    """Causal multi-head self-attention with latency-aware bias."""
    def __init__(self, d_model, heads=4, dropout=0.2, bias_mode="alibi", max_rel=128):
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
    """Pointwise-GLU → causal depthwise Conv1d → BN → SiLU → pointwise."""
    def __init__(self, d_model, kernel_size=9, dropout=0.2):
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
    """FFN → MHSA → Conv → FFN."""
    def __init__(self, d_model, heads=4, ff_expansion=4, dropout=0.2,
                 kernel_size=9, bias_mode="alibi", max_rel=128):
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
    """Stack of Conformer blocks."""
    def __init__(self, d_model=128, depth=2, heads=4, ff_expansion=4,
                 dropout=0.2, kernel_size=9, bias_mode="alibi", max_rel=128):
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
    """TemporalStem → GraphEncoder → Conformer → Head."""
    def __init__(self, n_ch, pos, d_stem=256, d_lift=127, d_in=128, d_model=128,
                 L_graph=2, k=8, graph_heads=4, graph_dropout=0.2,
                 conf_depth=2, conf_heads=4, ff_expansion=4, conf_dropout=0.2,
                 kernel_size=9, bias_mode="alibi", max_rel=128, causal=True):
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


class EEGTrainAugment(nn.Module):
    """Channel dropout, light Gaussian noise, optional temporal masking."""
    def __init__(self, p_chdrop=0.10, noise_std=0.02, p_timemask=0.10, max_mask=32):
        super().__init__()
        self.p_chdrop = p_chdrop
        self.noise_std = noise_std
        self.p_timemask = p_timemask
        self.max_mask = max_mask
    def forward(self, x):
        if self.p_chdrop > 0:
            B, T, N = x.shape
            drop = (torch.rand(B, 1, N, device=x.device) < self.p_chdrop).float()
            x = x * (1.0 - drop)
        if self.noise_std > 0:
            x = x + self.noise_std * torch.randn_like(x)
        if self.p_timemask > 0 and self.max_mask > 0:
            B, T, N = x.shape
            mlen = min(self.max_mask, T)
            for b in range(B):
                if torch.rand(1, device=x.device) < self.p_timemask:
                    t0 = torch.randint(0, T - mlen + 1, (1,), device=x.device).item()
                    x[b, t0:t0+mlen, :] = 0
        return x


def window_indices(T, win, hop):
    idx, t = [], 0
    while t + win <= T:
        idx.append((t, t + win))
        t += hop
    return np.array(idx, dtype=int)


def split_windows_by_time(spans, T_total, train_ratio=0.7, val_ratio=0.1, gap=0):
    t_train_end = int(T_total * train_ratio)
    t_val_end   = int(T_total * (train_ratio + val_ratio))
    tr = np.where(spans[:, 1] <= (t_train_end - gap))[0]
    va = np.where((spans[:, 0] >= (t_train_end + gap)) & (spans[:, 1] <= (t_val_end - gap)))[0]
    te = np.where(spans[:, 0] >= (t_val_end + gap))[0]
    return tr, va, te


class EEGEnvDataset(Dataset):
    """Dataset from pre-normalized continuous arrays and window spans."""
    def __init__(self, X, y, spans, lag):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.spans = spans
        self.lag = lag
    def __len__(self):
        return len(self.spans)
    def __getitem__(self, i):
        a, b = self.spans[i]          # window on EEG
        return self.X[a:b], self.y[(a - self.lag):(b - self.lag)]



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


def pearsonr_batch(yhat, y, eps=1e-8):
    yhat = yhat - yhat.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
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


def plot_training_curves(hist, outdir):
    plt.figure(figsize=(6,4))
    plt.plot(hist['train_loss'], label='train')
    plt.plot(hist['val_loss'], label='val')
    plt.xlabel('epoch'); plt.ylabel('loss (0.7·(1−r)+0.3·L1 + λ·TV)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_curve.png'), dpi=150); plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(hist['train_r'], label='train r')
    plt.plot(hist['val_r'], label='val r')
    plt.plot(hist['test_r'], label='test r')
    plt.xlabel('epoch'); plt.ylabel('Pearson r')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_r.png'), dpi=150); plt.close()


def plot_adjacency_heatmap(A, outdir, name='A_heatmap.png', title='Blended adjacency A'):
    plt.figure(figsize=(6,5))
    plt.imshow(A, cmap='viridis')
    plt.colorbar(label='weight')
    plt.title(title)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, name), dpi=150); plt.close()


def plot_sensor_edges(A, info, outdir, topk=120, name='A_edges_sensors.png', title='Top edges on sensor map'):
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
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title('Top strengthened edges (A−A₀)+')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'A_edges_delta.png'), dpi=150); plt.close()


def plot_in_strength_topomap(A, info, outdir):
    W = A.copy(); np.fill_diagonal(W, 0.0)
    in_strength = W.sum(axis=0)
    fig, ax = plt.subplots(figsize=(5,4))
    mne.viz.plot_topomap(in_strength, info, axes=ax, show=False)
    ax.set_title('In-strength (col-sum of A)')
    fig.savefig(os.path.join(outdir, 'in_strength_topomap.png'), dpi=150); plt.close(fig)


def evaluate(model, loader, device, bt_chunk, amp_dtype, lambda_tv):
    model.eval(); r_sum, loss_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                yhat, _ = model(xb, bt_chunk=bt_chunk)
                r = pearsonr_batch(yhat, yb).mean()
                l_main = 0.7 * (1 - r) + 0.3 * F.l1_loss(yhat, yb)
                tv = (yhat[:, 1:] - yhat[:, :-1]).abs().mean()
                loss = l_main + lambda_tv * tv
            r_sum += r.item() * xb.size(0)
            loss_sum += loss.item() * xb.size(0)
            n += xb.size(0)
    return (r_sum / max(1, n)), (loss_sum / max(1, n))


def train_one_subject(preproc_dir, subj_id, epochs, batch, win_sec, hop_sec, lr, device,
                      outdir, k, heads_graph, blocks_graph, conf_depth, heads_conf, bt_chunk, amp,
                      workers, prefetch, patience, min_delta, warmup_pct, final_lr_pct, bias_mode,
                      p_chdrop, noise_std, p_timemask, lambda_tv, weight_decay, lag_ms):
    import helper
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
    try: torch.backends.cudnn.benchmark = True
    except Exception: pass

    eeg, env, fs, attAB = helper.subject_eeg_env_ab(preproc_dir, subj_id)
    lag = int(round(lag_ms * fs / 1000.0))

    T_total = len(eeg)
    win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))
    spans = []
    t = lag
    while t + win <= T_total:
        spans.append((t, t + win))  # X-window [t,t+win)
        t += hop
    spans = np.array(spans, dtype=int)

    tr_idx, va_idx, te_idx = split_windows_by_time(spans, T_total, 0.7, 0.1, gap=win//2)

    t_train_end = int(T_total * 0.7)
    X_train_stat = eeg[:t_train_end].astype(np.float32)
    y_train_stat = env[:t_train_end].astype(np.float32)
    Xm = X_train_stat.mean(axis=0, keepdims=True)
    Xs = X_train_stat.std(axis=0, keepdims=True) + 1e-8
    ym = y_train_stat.mean(keepdims=True)
    ys = y_train_stat.std(keepdims=True) + 1e-8

    X = (eeg.astype(np.float32) - Xm) / Xs
    y = (env.astype(np.float32) - ym) / ys

    n_ch = X.shape[1]
    info, ch_names, pos = make_biosemi64_info(n_ch=n_ch, sfreq=fs)

    train_ds = EEGEnvDataset(X, y, spans[tr_idx], lag)
    val_ds   = EEGEnvDataset(X, y, spans[va_idx], lag)
    test_ds  = EEGEnvDataset(X, y, spans[te_idx], lag)

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
        L_graph=blocks_graph, k=k, graph_heads=heads_graph, graph_dropout=0.2,
        conf_depth=conf_depth, conf_heads=heads_conf, ff_expansion=4, conf_dropout=0.2,
        kernel_size=9, bias_mode=bias_mode, max_rel=128, causal=True
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, len(train_ld))
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(warmup_pct * total_steps)
    sched = make_warmup_cosine_scheduler(opt, total_steps, warmup_steps=warmup_steps, final_lr_pct=final_lr_pct)

    use_fp16 = (amp == 'fp16'); use_bf16 = (amp == 'bf16')
    amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    augment = EEGTrainAugment(p_chdrop=p_chdrop, noise_std=noise_std, p_timemask=p_timemask, max_mask=int(0.1*fs))

    best_val = float('inf'); best_state = None; wait = 0
    hist = {'train_loss': [], 'val_loss': [], 'train_r': [], 'val_r': [], 'test_r': []}

    os.makedirs(outdir, exist_ok=True)
    test_csv = os.path.join('outputs', 'test_scores_per_epoch.csv')
    if not os.path.exists(test_csv):
        with open(test_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['subject','epoch','train_loss','val_loss','train_r','val_r','test_r','lr'])

    for ep in range(1, epochs + 1):
        # st = time.time()
        model.train()
        tr_loss_sum, tr_r_sum, tr_n = 0.0, 0.0, 0
        t0 = time.time()
        for xb, yb in train_ld:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            xb = augment(xb)
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                yhat, _ = model(xb, bt_chunk=bt_chunk)
                r = pearsonr_batch(yhat, yb).mean()
                l_main = 0.7 * (1 - r) + 0.3 * F.l1_loss(yhat, yb)
                tv = (yhat[:, 1:] - yhat[:, :-1]).abs().mean()
                loss = l_main + lambda_tv * tv
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
        va_r, va_loss = evaluate(model, val_ld, device, bt_chunk, amp_dtype, lambda_tv)
        te_r, te_loss = evaluate(model, test_ld, device, bt_chunk, amp_dtype, lambda_tv)

        hist['train_loss'].append(tr_loss); hist['val_loss'].append(va_loss)
        hist['train_r'].append(tr_r);       hist['val_r'].append(va_r)
        hist['test_r'].append(te_r)
        # sec = time.time() - st
        # print(f"train {tr_loss:.4f} (r = {tr_r : .3f}) | val {va_loss : .4f} (r = {va_r : .3f}) | Took {sec :.1f}s")

        lr_now = opt.param_groups[0]['lr']
        with open(test_csv, 'a', newline='') as f:
            csv.writer(f).writerow([subj_id, ep, tr_loss, va_loss, tr_r, va_r, te_r, lr_now])

        improved = (best_val - va_loss) > float(min_delta)
        if improved:
            best_val = va_loss; wait = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(outdir, 'best.pt'))
        else:
            wait += 1
            if wait >= patience:
                break
        dt = time.time() - t0
        print(f"S{subj_id} Ep{ep:02d} | tr {tr_loss:.4f} r={tr_r:.3f} | va {va_loss:.4f} r={va_r:.3f} | te r={te_r:.3f} | {dt:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    torch.save(model.state_dict(), os.path.join(outdir, 'last.pt'))

    with torch.no_grad():
        ref_loader = val_ld if len(val_ld) > 0 else test_ld
        xb, yb = next(iter(ref_loader))
        xb = xb.to(device, non_blocking=True)
        yhat, A_t = model(xb, bt_chunk=bt_chunk)
        A = A_t.detach().cpu().numpy()

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

    return float(hist['val_r'][-1]), float(hist['test_r'][-1])


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
    p.add_argument('--heads_graph', type=int, default=4)
    p.add_argument('--blocks_graph', type=int, default=3)
    p.add_argument('--conf_depth', type=int, default=2)
    p.add_argument('--heads_conf', type=int, default=4)
    p.add_argument('--bt_chunk', type=int, default=256)
    p.add_argument('--amp', type=str, choices=['none','fp16','bf16'], default='bf16')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--prefetch', type=int, default=2)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--min_delta', type=float, default=1e-3)
    p.add_argument('--warmup_pct', type=float, default=0.05)
    p.add_argument('--final_lr_pct', type=float, default=0.1)
    p.add_argument('--bias_mode', type=str, choices=['rel','alibi'], default='alibi')
    p.add_argument('--p_chdrop', type=float, default=0.05 )
    p.add_argument('--lag_ms', type=float, default=120.0)
    p.add_argument('--noise_std', type=float, default=0.01)
    p.add_argument('--p_timemask', type=float, default=0)
    p.add_argument('--lambda_tv', type=float, default=0.01)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    args = p.parse_args()
    lag = args.lag_ms
    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, "summary_pearson_full.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, 'w', newline='') as f:
            csv.writer(f).writerow(['subject', 'final_val_r', 'final_test_r'])

    for subj in range(1, 19):
        sx = os.path.join(args.outdir, f"S{subj}_FULL")
        print(f"\n=== Subject {subj} → {sx} ===")
        val_r, test_r = train_one_subject(
            args.preproc_dir, subj, args.epochs, args.batch,
            args.win_sec, args.hop_sec, args.lr, args.device, sx,
            args.k, args.heads_graph, args.blocks_graph, args.conf_depth, args.heads_conf,
            args.bt_chunk, args.amp, args.workers, args.prefetch,
            args.patience, args.min_delta, args.warmup_pct, args.final_lr_pct, args.bias_mode,
            args.p_chdrop, args.noise_std, args.p_timemask, args.lambda_tv, args.weight_decay, args.lag_ms
        )
        with open(summary_path, 'a', newline='') as f:
            csv.writer(f).writerow([subj, val_r, test_r])
        print(f"Subject {subj}: Val r={val_r:.4f} | Test r={test_r:.4f}")

    print(f"\nWrote {summary_path}")


if __name__ == '__main__':
    main()
