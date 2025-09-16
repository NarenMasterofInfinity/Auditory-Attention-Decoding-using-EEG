#!/usr/bin/env python3
"""
train_aad_graph_xattn.py

Auditory Attention Decoding (AAD) with:
  EEG encoder: TemporalStem → GraphEncoder  (spatial per-time)
  Audio encoder: causal TCN on audio envelope(s)
  Fusion: Multi-Head Cross Attention (EEG queries → Audio keys/values)
  Head: Binary classifier over candidates (A vs B)

Data expectation (preferred):
  helper.subject_eeg_env_pair_ab(PREPROC_DIR, subj_id)
    -> eeg[T,C], env_A[T], env_B[T], fs, att_AB  # att_AB per sample: 'A' or 'B'

Fallback (plumbing only; replace with real B for proper AAD):
  helper.subject_eeg_env_ab(PREPROC_DIR, subj_id)
    -> eeg[T,C], env_att[T], fs, att_AB
  env_B is synthesized by circularly shifting env_att by 1 second.

Saves per subject (outputs_aad/SX):
  - best_model.pt
  - A0.csv, A_final.csv (+ heatmaps/edges/topomap)
  - training curves (loss/acc)
  - test window viz: EEG (4–6 ch) + candidate probabilities over time
Appends per-subject accuracy, ROC-AUC to outputs_aad/summary_aad.csv
"""

import os, math, time, argparse, csv, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
from sklearn.metrics import roc_auc_score

# ----- Your modules -----
try:
    from FeatureExpander import TemporalStem
except Exception:
    from temporal_stem import TemporalStem

try:
    from GraphEncoder1 import GraphEncoder
except Exception:
    from graph_encoder_sparse import GraphEncoder

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

def maj_vote_label(chars):
    # chars: array/list of 'A' or 'B'
    a = np.sum(np.array(chars) == 'A')
    b = len(chars) - a
    return 1 if a >= b else 0   # 1→A attended, 0→B attended


# -------------------- dataset --------------------

class AADDataset(Dataset):
    """
    Returns:
      eeg_win [W,C], envA_win [W], envB_win [W], label (0/1)
    """
    def __init__(self, eeg, envA, envB, att_AB, fs, win_sec, hop_sec):
        self.X = eeg.astype(np.float32)
        self.a = envA.astype(np.float32)
        self.b = envB.astype(np.float32)
        self.fs = float(fs)
        self.win = int(round(win_sec * fs))
        self.hop = int(round(hop_sec * fs))
        T = len(self.a)
        self.spans = window_indices(T, self.win, self.hop)
        self.labels = self._window_labels(att_AB)

    def _window_labels(self, att_AB):
        labs = []
        for a, b in self.spans:
            labs.append(maj_vote_label(att_AB[a:b]))
        return np.array(labs, dtype=np.int64)

    def __len__(self): return len(self.spans)

    def __getitem__(self, i):
        a, b = self.spans[i]
        return self.X[a:b], self.a[a:b], self.b[a:b], self.labels[i]


# -------------------- plotting --------------------

def plot_training_curves(hist, outdir):
    ensure_dir(outdir)
    # loss
    plt.figure(figsize=(6,4)); plt.plot(hist['train_loss'], label='train'); plt.plot(hist['val_loss'], label='val')
    plt.xlabel('epoch'); plt.ylabel('CE loss'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_curve.png'), dpi=150); plt.close()
    # acc
    plt.figure(figsize=(6,4)); plt.plot(hist['train_acc'], label='train acc'); plt.plot(hist['val_acc'], label='val acc')
    plt.xlabel('epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_acc.png'), dpi=150); plt.close()

def plot_adjacency_heatmap(A, outdir, name='A_heatmap.png', title='Blended adjacency A'):
    ensure_dir(outdir)
    plt.figure(figsize=(6,5)); plt.imshow(A, cmap='viridis'); plt.colorbar(label='weight'); plt.title(title)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, name), dpi=150); plt.close()

def plot_sensor_edges(A, info, outdir, topk=120, name='A_edges_sensors.png', title='Top edges on sensor map'):
    ensure_dir(outdir)
    pos = np.array([info.get_montage().get_positions()['ch_pos'][ch] for ch in info.ch_names]); P2 = pos[:, :2]
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
        ax.plot([P2[i,0], P2[j,0]], [P2[i,1], P2[j,1]], '-', lw=0.5 + 3.0*(w/(A.max()+1e-8)), color='tab:blue', alpha=0.5)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, name), dpi=150); plt.close()

def plot_sensor_edges_delta(A, A0, info, outdir, topk=80):
    ensure_dir(outdir)
    if A0 is None: return
    D = np.clip(A - A0, 0, None); np.fill_diagonal(D, 0.0)
    pos = np.array([info.get_montage().get_positions()['ch_pos'][ch] for ch in info.ch_names]); P2 = pos[:, :2]
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

def plot_test_probabilities(probs, labels, outdir, fname='test_probs.png'):
    ensure_dir(outdir)
    plt.figure(figsize=(8,3))
    plt.plot(probs[:,1], label='P(att=A)')
    plt.plot(labels, label='label (A=1, B=0)', alpha=0.6)
    plt.xlabel('window'); plt.ylabel('probability / label'); plt.legend(); plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150); plt.close()
    return path


# -------------------- encoders & fusion --------------------

class EEGEncoder(nn.Module):
    """
    TemporalStem → Linear(lift) → GraphEncoder
    Returns tokens S [B,T,d_model] and adjacency A [C,C]
    """
    def __init__(self, n_ch, pos, d_stem=256, d_lift=127, d_in=128, d_model=128,
                 L=3, k=8, heads=4, dropout=0.1, causal=True):
        super().__init__()
        self.stem = TemporalStem(in_ch=n_ch, out_ch=d_stem, causal=causal, dropout=dropout)
        self.lift = nn.Linear(d_stem, d_lift)
        self.graph = GraphEncoder(pos=torch.tensor(pos, dtype=torch.float32),
                                  d_in=d_in, d_model=d_model, L=L, k=k, heads=heads, dropout=dropout)
    def forward(self, eeg, bt_chunk=None):
        H0 = self.stem(eeg)             # [B,T,256]
        Lf = self.lift(H0)              # [B,T,127]
        B, T, _ = H0.shape; N = eeg.shape[-1]
        Xin = torch.cat([Lf.unsqueeze(2).expand(B, T, N, -1), eeg.unsqueeze(-1)], dim=-1)  # [B,T,N,128]
        try:
            _, S, A = self.graph(Xin, bt_chunk=bt_chunk)     # S [B,T,d_model]
        except TypeError:
            _, S, A = self.graph(Xin)
        return S, A

class AudioTCN(nn.Module):
    """
    Causal depthwise-separable TCN on envelope.
    Input: env [B,T] → Output tokens [B,T,d_a]
    """
    def __init__(self, d_a=64, hidden=128, blocks=4, kernel=9, dropout=0.1):
        super().__init__()
        self.inp = nn.Conv1d(1, hidden, 1)
        layers = []
        for i in range(blocks):
            dil = 2**i
            pad = (kernel-1)*dil
            layers += [
                nn.Conv1d(hidden, hidden, kernel, padding=pad, dilation=dil, groups=hidden),  # depthwise causal
                nn.Conv1d(hidden, hidden, 1),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
        self.tcn = nn.Sequential(*layers)
        self.out = nn.Conv1d(hidden, d_a, 1)
    def forward(self, env):
        x = env.unsqueeze(1)            # [B,1,T]
        h = self.inp(x)
        h = self.tcn(h)[:, :, :env.size(1)]  # causal trim
        z = self.out(h).transpose(1,2)  # [B,T,d_a]
        return z

class CrossAttention(nn.Module):
    """
    Multi-Head Cross Attention:
      Q = EEG tokens [B,T_e,d]
      K,V = Audio tokens [B,T_a,d_a] (projected to d)
    Returns context [B,T_e,d] after one cross-attn block (pre-norm).
    """
    def __init__(self, d_eeg=128, d_audio=64, heads=4, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(d_eeg, d_eeg)
        self.k = nn.Linear(d_audio, d_eeg)
        self.v = nn.Linear(d_audio, d_eeg)
        self.h = heads
        self.d = d_eeg
        self.do = nn.Dropout(dropout)
        self.ln_q = nn.LayerNorm(d_eeg)
        self.ln_out = nn.LayerNorm(d_eeg)
        self.proj = nn.Linear(d_eeg, d_eeg)

    def _split(self, x):
        B,T,D = x.shape
        H = self.h; d = D // H
        return x.view(B,T,H,d).transpose(1,2)   # [B,H,T,d]

    def _merge(self, x):
        B,H,T,d = x.shape
        return x.transpose(1,2).contiguous().view(B,T,H*d)

    def forward(self, eeg_tok, aud_tok):
        q = self.ln_q(eeg_tok)
        k = aud_tok; v = aud_tok
        q = self._split(self.q(q))
        k = self._split(self.k(k))
        v = self._split(self.v(v))
        scale = 1.0 / math.sqrt(q.size(-1))
        att = torch.softmax(torch.matmul(q, k.transpose(-1,-2))*scale, dim=-1)  # [B,H,T_e,T_a]
        ctx = torch.matmul(att, v)                                             # [B,H,T_e,d]
        ctx = self._merge(ctx)
        out = self.ln_out(eeg_tok + self.do(self.proj(ctx)))                   # resid+proj+ln
        return out

class AADModel(nn.Module):
    """
    EEG encoder (TemporalStem+Graph) + Audio TCN encoders + Cross-attn → logits[2]
    """
    def __init__(self, n_ch, pos, d_model=128, d_audio=64, L=3, k=8, heads_graph=4, heads_xattn=4, dropout=0.1):
        super().__init__()
        self.eeg_enc = EEGEncoder(n_ch=n_ch, pos=pos, d_model=d_model, L=L, k=k, heads=heads_graph, dropout=dropout)
        self.audA = AudioTCN(d_a=d_audio, hidden=128, blocks=4, kernel=9, dropout=dropout)
        self.audB = AudioTCN(d_a=d_audio, hidden=128, blocks=4, kernel=9, dropout=dropout)
        self.xA = CrossAttention(d_eeg=d_model, d_audio=d_audio, heads=heads_xattn, dropout=dropout)
        self.xB = CrossAttention(d_eeg=d_model, d_audio=d_audio, heads=heads_xattn, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clsA = nn.Linear(d_model, 1)
        self.clsB = nn.Linear(d_model, 1)

    def forward(self, eeg, envA, envB, bt_chunk=None):
        S, A = self.eeg_enc(eeg, bt_chunk=bt_chunk)   # [B,T,d_model]
        ZA = self.audA(envA)                          # [B,T,d_audio]
        ZB = self.audB(envB)                          # [B,T,d_audio]
        CA = self.xA(S, ZA)                           # [B,T,d_model]
        CB = self.xB(S, ZB)                           # [B,T,d_model]
        PA = self.pool(CA.transpose(1,2)).squeeze(-1) # [B,d_model]
        PB = self.pool(CB.transpose(1,2)).squeeze(-1) # [B,d_model]
        logitA = self.clsA(PA).squeeze(-1)            # [B]
        logitB = self.clsB(PB).squeeze(-1)            # [B]
        logits = torch.stack([logitB, logitA], dim=-1) # [B,2]  idx0->B, idx1->A
        return logits, A


# -------------------- train/eval --------------------

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

def split_indices(num_items, train_ratio=0.7, val_ratio=0.1, seed=123):
    all_idx = np.arange(num_items); rng = np.random.default_rng(seed); rng.shuffle(all_idx)
    n_train = int(round(train_ratio * num_items)); n_val = int(round(val_ratio * num_items))
    train_idx = all_idx[:n_train]; val_idx = all_idx[n_train:n_train+n_val]; test_idx = all_idx[n_train+n_val:]
    return train_idx, val_idx, test_idx

def subset_dataset(ds, idx):
    class _Sub(Dataset):
        def __init__(self, base, sel): self.base = base; self.sel = np.array(sel, int)
        def __len__(self): return len(self.sel)
        def __getitem__(self, i): return self.base[self.sel[i]]
    return _Sub(ds, idx)

def make_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps=0, final_lr_pct=0.1):
    final_lr_pct = float(final_lr_pct)
    def lr_lambda(step):
        step = min(step, total_steps)
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if total_steps == warmup_steps: return final_lr_pct
        prog = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * prog))
        return final_lr_pct + (1 - final_lr_pct) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def evaluate_cls(model, loader, device, bt_chunk, amp_dtype):
    model.eval(); acc_sum, n = 0.0, 0; probs_all = []; labs_all = []
    with torch.no_grad():
        for xb, a, b, yb in loader:
            xb = xb.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)
            b = b.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                logits, _ = model(xb, a, b, bt_chunk=bt_chunk)
                pred = logits.argmax(-1)
                acc = (pred == yb).float().sum().item()
                prob = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            acc_sum += acc; n += xb.size(0)
            probs_all.append(prob); labs_all.append(yb.cpu().numpy())
    probs = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0,2))
    labs  = np.concatenate(labs_all, axis=0) if labs_all else np.zeros((0,))
    try:
        auc = roc_auc_score(labs, probs[:,1])
    except Exception:
        auc = float('nan')
    return acc_sum / max(1,n), auc, probs, labs

def train_one_subject(preproc_dir, subj_id, epochs, batch, win_sec, hop_sec, lr, device,
                      outdir, k, heads_graph, blocks_graph, heads_xattn, bt_chunk, amp,
                      workers, prefetch, accum, compile_flag,
                      patience, min_delta, warmup_pct, final_lr_pct):

    ensure_dir(outdir)

    # ----- data load (prefer both envelopes) -----
    use_fallback = False
    try:
        eeg, envA, envB, fs, attAB = helper.subject_eeg_env_ab_aad(preproc_dir, subj_id)
    except Exception:
        eeg, env_att, fs, attAB = helper.subject_eeg_env_ab(preproc_dir, subj_id)
        shift = int(round(1.0 * fs))
        envB = np.roll(env_att, shift)
        envA = env_att
        use_fallback = True
        print("WARNING: Using fallback envB (circularly shifted). Replace with real env_B for AAD.")

    X = eeg.astype(np.float32)
    X, _, _ = zscore_train(X)

    win = int(round(win_sec * fs)); hop = int(round(hop_sec * fs))
    n_ch = X.shape[1]
    info, ch_names, pos = make_biosemi64_info(n_ch=n_ch, sfreq=fs)

    ds = AADDataset(X, envA, envB, attAB, fs, win_sec, hop_sec)
    tr_idx, va_idx, te_idx = split_indices(len(ds), 0.7, 0.1, seed=2000 + subj_id)
    train_ds, val_ds, test_ds = subset_dataset(ds, tr_idx), subset_dataset(ds, va_idx), subset_dataset(ds, te_idx)

    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True,
                          num_workers=workers, pin_memory=True,
                          persistent_workers=(workers>0), prefetch_factor=prefetch)
    val_ld = DataLoader(val_ds, batch_size=batch, shuffle=False,
                        num_workers=workers, pin_memory=True,
                        persistent_workers=(workers>0), prefetch_factor=prefetch)
    test_ld = DataLoader(test_ds, batch_size=batch, shuffle=False,
                         num_workers=workers, pin_memory=True,
                         persistent_workers=(workers>0), prefetch_factor=prefetch)

    # ----- model -----
    model = AADModel(n_ch=n_ch, pos=pos, d_model=128, d_audio=64,
                     L=blocks_graph, k=k, heads_graph=heads_graph, heads_xattn=heads_xattn, dropout=0.1).to(device)

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

    best_val = float('inf'); best_state = None; wait = 0
    hist = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def _to(x): return x.to(device, non_blocking=True)

    # ----- train -----
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss_sum, tr_acc_sum, tr_n = 0.0, 0.0, 0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        for step, (xb, a, b, yb) in enumerate(train_ld, 1):
            xb = _to(xb); a = _to(a); b = _to(b); yb = _to(yb)
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                logits, _ = model(xb, a, b, bt_chunk=bt_chunk)
                loss = F.cross_entropy(logits, yb)
                loss = loss / max(1, accum)

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step % max(1, accum)) == 0:
                if use_fp16: scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_fp16: scaler.step(opt); scaler.update()
                else: opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()

            pred = logits.argmax(-1)
            tr_acc_sum += (pred == yb).float().sum().item()
            tr_loss_sum += loss.item() * xb.size(0) * max(1, accum)
            tr_n += xb.size(0)

        tr_loss = tr_loss_sum / tr_n
        tr_acc = tr_acc_sum / tr_n

        # ----- validate -----
        model.eval()
        va_loss_sum, va_acc_sum, va_n = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, a, b, yb in val_ld:
                xb = _to(xb); a = _to(a); b = _to(b); yb = _to(yb)
                with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                    logits, _ = model(xb, a, b, bt_chunk=bt_chunk)
                    loss = F.cross_entropy(logits, yb)
                va_loss_sum += loss.item() * xb.size(0)
                va_acc_sum += (logits.argmax(-1) == yb).float().sum().item()
                va_n += xb.size(0)
        va_loss = va_loss_sum / va_n
        va_acc = va_acc_sum / va_n

        hist['train_loss'].append(tr_loss); hist['val_loss'].append(va_loss)
        hist['train_acc'].append(tr_acc);   hist['val_acc'].append(va_acc)
        sec = time.time() - t0
        print(f"Ep{ep:02d} | train {tr_loss:.4f} acc={tr_acc:.3f} | val {va_loss:.4f} acc={va_acc:.3f} | {sec:.1f}s")

        improved = (best_val - va_loss) > float(min_delta)
        if improved:
            best_val = va_loss; wait = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(outdir, 'best_model.pt'))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {ep} (best val {best_val:.4f})")
                break

    # ----- restore best -----
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # ----- metrics & artifacts -----
    val_acc, val_auc, _, _ = evaluate_cls(model, val_ld, device, bt_chunk, amp_dtype)
    test_acc, test_auc, te_probs, te_labs = evaluate_cls(model, test_ld, device, bt_chunk, amp_dtype)
    print(f"Final: Val acc={val_acc:.3f} AUC={val_auc:.3f} | Test acc={test_acc:.3f} AUC={test_auc:.3f}")

    # graphs on one batch
    with torch.no_grad():
        xb, a, b, yb = next(iter(val_ld)) if len(val_ld) > 0 else next(iter(test_ld))
        xb = _to(xb); a = _to(a); b = _to(b)
        logits, A_t = model(xb, a, b, bt_chunk=bt_chunk)
        A = A_t.detach().cpu().numpy()

    plot_training_curves(hist, outdir)
    plot_adjacency_heatmap(A, outdir, name='A_heatmap.png', title='Blended adjacency A')
    info, ch_names, _pos = make_biosemi64_info(n_ch=n_ch, sfreq=fs)
    plot_sensor_edges(A, info, outdir, topk=120, name='A_edges_sensors.png')
    A0 = None
    try:
        A0 = model.eeg_enc.graph.A0.detach().cpu().numpy()
        np.savetxt(os.path.join(outdir, 'A0.csv'), A0, delimiter=',')
        plot_adjacency_heatmap(A0, outdir, name='A0_heatmap.png', title='Initial prior A0')
    except Exception:
        pass
    np.savetxt(os.path.join(outdir, 'A_final.csv'), A, delimiter=',')
    plot_sensor_edges_delta(A, A0, info, outdir, topk=80)
    plot_in_strength_topomap(A, info, outdir)

    # test probability timeline
    vis_dir = ensure_dir(os.path.join(outdir, "vis"))
    prob_path = plot_test_probabilities(te_probs, te_labs, vis_dir, fname='test_probs.png')
    print(f"[viz] wrote {prob_path}")

    # note any fallback
    if use_fallback:
        with open(os.path.join(outdir, 'README_FALLBACK.txt'), 'w') as f:
            f.write("env_B was synthesized by circular shift of env_att by 1 second. Replace with real env_B for proper AAD.\n")

    return float(val_acc), float(val_auc), float(test_acc), float(test_auc)


# -------------------- main --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preproc_dir', type=str, default='/home/naren-root/Dataset/DATA_preproc')
    p.add_argument('--outdir', type=str, default='AAD/outputs_aad')
    p.add_argument('--subjects', type=str, default='1-18', help='e.g., "1-18" or "1,3,5"')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--win_sec', type=float, default=5.0)
    p.add_argument('--hop_sec', type=float, default=2.5)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--heads_graph', type=int, default=4)
    p.add_argument('--blocks_graph', type=int, default=3)
    p.add_argument('--heads_xattn', type=int, default=4)
    p.add_argument('--bt_chunk', type=int, default=128)
    p.add_argument('--amp', type=str, choices=['none', 'fp16', 'bf16'], default='bf16')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--prefetch', type=int, default=2)
    p.add_argument('--accum', type=int, default=1)
    p.add_argument('--compile', action='store_true')
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--min_delta', type=float, default=5e-4)
    p.add_argument('--warmup_pct', type=float, default=0.05)
    p.add_argument('--final_lr_pct', type=float, default=0.1)
    args = p.parse_args()

    ensure_dir(args.outdir)
    summ_path = os.path.join(args.outdir, "summary_aad.csv")
    if not os.path.exists(summ_path):
        with open(summ_path, 'w', newline='') as f:
            csv.writer(f).writerow(['subject', 'val_acc', 'val_auc', 'test_acc', 'test_auc'])

    # parse subjects
    if '-' in args.subjects:
        a, b = args.subjects.split('-'); subs = list(range(int(a), int(b) + 1))
    else:
        subs = [int(s) for s in args.subjects.split(',') if s.strip()]

    for subj in subs:
        sx = ensure_dir(os.path.join(args.outdir, f"S{subj}"))
        print(f"\n=== Subject {subj} → {sx} ===")
        val_acc, val_auc, test_acc, test_auc = train_one_subject(
            args.preproc_dir, subj, args.epochs, args.batch,
            args.win_sec, args.hop_sec, args.lr, args.device, sx,
            args.k, args.heads_graph, args.blocks_graph, args.heads_xattn, args.bt_chunk, args.amp,
            args.workers, args.prefetch, args.accum, args.compile,
            args.patience, args.min_delta, args.warmup_pct, args.final_lr_pct
        )
        with open(summ_path, 'a', newline='') as f:
            csv.writer(f).writerow([subj, val_acc, val_auc, test_acc, test_auc])
        print(f"Subject {subj}: Val acc={val_acc:.3f} AUC={val_auc:.3f} | Test acc={test_acc:.3f} AUC={test_auc:.3f}")

    print(f"\nWrote {summ_path}")


if __name__ == '__main__':
    main()
