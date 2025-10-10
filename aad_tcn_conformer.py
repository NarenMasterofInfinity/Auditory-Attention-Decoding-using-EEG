#!/usr/bin/env python3
"""
train_aad_eegformer.py

Auditory Attention Decoding (AAD)
  EEG encoder : TemporalStem → GraphEncoder → EEGFormer(Conformer)
  Audio enc   : causal TCN on speech envelopes (A & B)
  Fusion      : Multi-Head Cross-Attention (EEG queries ← Audio keys/values)
  Head        : Binary classifier (A vs B)

Data expectation (preferred):
  helper.subject_eeg_env_pair_ab(PREPROC_DIR, subj_id)
    -> eeg[T,C], env_A[T], env_B[T], fs, att_AB   # att_AB in {'A','B'} per sample

Windowing:
  Sliding windows of length win_sec with hop hop_sec (in seconds), z-scored per-channel (train stats).

Logging:
  - CSV at outputs_dir/{subject}/training_log.csv
  - best checkpoint at outputs_dir/{subject}/best_model.pt
  - optional TensorBoard (--tb)

Usage (example):
  python train_aad_eegformer.py --preproc_dir ./preproc --subject S01 --epochs 30 \
    --win_sec 5 --hop_sec 1 --batch_size 16 --lr 2e-4 --amp --tb

"""

import os, sys, math, csv, time, json, random
from pathlib import Path
from dataclasses import dataclass
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

import mne

# --- Your modules ---
from FeatureExpander import TemporalStem                # Temporal feature extractor
from GraphEncoder1 import GraphEncoder                  # Spatial graph encoder
from Conformer import ConformerEncoder as EEGFormer     # EEG temporal encoder

# --------- Utils ---------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def to_device(batch, device):
    return tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)

def zscore_train_stats(x: np.ndarray, eps=1e-8):
    # x: [T,C]
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True) + eps
    return m, s

def apply_zscore(x: np.ndarray, m: np.ndarray, s: np.ndarray):
    return (x - m) / s

def sliding_indices(T, win, hop):
    idx = []
    t = 0
    while t + win <= T:
        idx.append((t, t+win))
        t += hop
    if not idx and T >= win:  # edge case
        idx.append((0, win))
    return idx

def make_info_and_positions(n_ch: int, sfreq: float = 64.0, prefer_biosemi: bool = True):
    if prefer_biosemi and n_ch == 64:
        montage = mne.channels.make_standard_montage('biosemi64')
        ch_names = montage.ch_names
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        info.set_montage(montage)
        pos = np.stack([montage.get_positions()['ch_pos'][ch] for ch in ch_names]).astype(np.float32)
        return info, ch_names, pos
    montage = mne.channels.make_standard_montage('standard_1020')
    all_names = montage.ch_names[:n_ch]
    info = mne.create_info(ch_names=all_names, sfreq=sfreq, ch_types='eeg')
    info.set_montage(montage)
    pos_map = montage.get_positions()['ch_pos']
    pos = np.stack([pos_map[ch] for ch in all_names]).astype(np.float32)
    return info, all_names, pos

# --------- Audio TCN & Cross-Attn & Model ---------

class AudioTCN(nn.Module):
    """Causal depthwise-separable TCN on envelope: env [B,T] -> tokens [B,T,d_a]."""
    def __init__(self, d_a=64, hidden=128, blocks=4, kernel=9, dropout=0.1):
        super().__init__()
        self.inp = nn.Conv1d(1, hidden, 1)
        layers = []
        for i in range(blocks):
            dil = 2 ** i
            pad = (kernel - 1) * dil  # causal padding; we'll trim after
            layers += [
                nn.Conv1d(hidden, hidden, kernel, padding=pad, dilation=dil, groups=hidden),  # depthwise
                nn.Conv1d(hidden, hidden, 1),  # pointwise
                nn.GELU(),
                nn.Dropout(dropout)
            ]
        self.tcn = nn.Sequential(*layers)
        self.out = nn.Conv1d(hidden, d_a, 1)

    def forward(self, env):  # env: [B,T]
        x = env.unsqueeze(1)                 # [B,1,T]
        h = self.inp(x)
        h = self.tcn(h)[:, :, :env.size(1)]  # causal trim
        z = self.out(h).transpose(1, 2)      # [B,T,d_a]
        return z

class CrossAttention(nn.Module):
    """
    Multi-Head Cross Attention (Q=EEG, K/V=Audio):
      EEG tokens [B,T_e,d_eeg], Audio tokens [B,T_a,d_a] → context [B,T_e,d_eeg].
    """
    def __init__(self, d_eeg=128, d_audio=64, heads=4, dropout=0.1):
        super().__init__()
        assert d_eeg % heads == 0
        self.h = heads
        self.dh = d_eeg // heads
        self.q = nn.Linear(d_eeg, d_eeg)
        self.k = nn.Linear(d_audio, d_eeg)
        self.v = nn.Linear(d_audio, d_eeg)
        self.proj = nn.Linear(d_eeg, d_eeg)
        self.do = nn.Dropout(dropout)
        self.ln_q = nn.LayerNorm(d_eeg)
        self.ln_out = nn.LayerNorm(d_eeg)

    def _split(self, x):  # [B,T,D] -> [B,H,T,d]
        B, T, D = x.shape
        return x.view(B, T, self.h, self.dh).transpose(1, 2)

    def _merge(self, x):  # [B,H,T,d] -> [B,T,D]
        B, H, T, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d)

    def forward(self, eeg_tok, aud_tok):
        q = self._split(self.q(self.ln_q(eeg_tok)))
        k = self._split(self.k(aud_tok))
        v = self._split(self.v(aud_tok))
        scale = 1.0 / math.sqrt(self.dh)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale   # [B,H,T_e,T_a]
        att = torch.softmax(att, dim=-1)
        ctx = torch.matmul(att, v)                            # [B,H,T_e,d]
        ctx = self._merge(ctx)
        y = self.ln_out(eeg_tok + self.do(self.proj(ctx)))   # residual + proj + LN
        return y

class EEGEncoderPlusFormer(nn.Module):
    """
    TemporalStem → Linear(lift) → GraphEncoder → mean over nodes → EEGFormer(Conformer).
    Returns tokens F [B,T,d_model] and adjacency A [N,N].
    """
    def __init__(self, n_ch, pos, d_stem=256, d_lift=127, d_in=128, d_model=128,
                 L=3, k=8, heads_graph=4, dropout=0.1,
                 eegformer_depth=2, eegformer_heads=4, ff_expansion=4,
                 eeg_kernel=9, bias_mode="rel", max_rel=128, causal=True):
        super().__init__()
        self.stem = TemporalStem(in_ch=n_ch, out_ch=d_stem, causal=causal, dropout=dropout)
        self.lift = nn.Linear(d_stem, d_lift)
        self.graph = GraphEncoder(pos=torch.tensor(pos, dtype=torch.float32),
                                  d_in=d_in, d_model=d_model, L=L, k=k, heads=heads_graph, dropout=dropout)
        self.former = EEGFormer(d_model=d_model,
                                depth=eegformer_depth, heads=eegformer_heads,
                                ff_expansion=ff_expansion, dropout=dropout,
                                kernel_size=eeg_kernel, bias_mode=bias_mode, max_rel=max_rel)

    def forward(self, eeg, bt_chunk=None):
        H0 = self.stem(eeg)     # [B,T,256]
        Lt = self.lift(H0)      # [B,T,127]
        B, T, _ = H0.shape; N = eeg.shape[-1]
        Xin = torch.cat([Lt.unsqueeze(2).expand(B, T, N, -1), eeg.unsqueeze(-1)], dim=-1)  # [B,T,N,128]
        try:
            _, S, A = self.graph(Xin, bt_chunk=bt_chunk)   # S: [B,T,d_model]
        except TypeError:
            _, S, A = self.graph(Xin)
        F = self.former(S)      # [B,T,d_model]
        return F, A

class AADModel_EEGFormer(nn.Module):
    """
    EEG encoder (TemporalStem+Graph → EEGFormer) + Audio TCN (A,B) + Cross-attn → logits[2]
    """
    def __init__(self, n_ch, pos, d_model=128, d_audio=64,
                 L=3, k=8, heads_graph=4, heads_xattn=4, dropout=0.1,
                 eegformer_depth=2, eegformer_heads=4, ff_expansion=4,
                 eeg_kernel=9, bias_mode="rel", max_rel=128):
        super().__init__()
        self.eeg_enc = EEGEncoderPlusFormer(
            n_ch=n_ch, pos=pos, d_model=d_model, L=L, k=k, heads_graph=heads_graph, dropout=dropout,
            eegformer_depth=eegformer_depth, eegformer_heads=eegformer_heads, ff_expansion=ff_expansion,
            eeg_kernel=eeg_kernel, bias_mode=bias_mode, max_rel=max_rel, causal=True
        )
        self.audA = AudioTCN(d_a=d_audio, hidden=128, blocks=4, kernel=9, dropout=dropout)
        self.audB = AudioTCN(d_a=d_audio, hidden=128, blocks=4, kernel=9, dropout=dropout)
        self.xA = CrossAttention(d_eeg=d_model, d_audio=d_audio, heads=heads_xattn, dropout=dropout)
        self.xB = CrossAttention(d_eeg=d_model, d_audio=d_audio, heads=heads_xattn, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.clsA = nn.Linear(d_model, 1)
        self.clsB = nn.Linear(d_model, 1)

    def forward(self, eeg, envA, envB, bt_chunk=None):
        F, A = self.eeg_enc(eeg, bt_chunk=bt_chunk)     # [B,T,d_model], [C,C]
        ZA = self.audA(envA)                            # [B,T,d_audio]
        ZB = self.audB(envB)                            # [B,T,d_audio]
        CA = self.xA(F, ZA)                             # [B,T,d_model]
        CB = self.xB(F, ZB)                             # [B,T,d_model]
        PA = self.pool(CA.transpose(1, 2)).squeeze(-1)  # [B,d_model]
        PB = self.pool(CB.transpose(1, 2)).squeeze(-1)  # [B,d_model]
        logitA = self.clsA(PA).squeeze(-1)              # [B]
        logitB = self.clsB(PB).squeeze(-1)              # [B]
        logits = torch.stack([logitB, logitA], dim=-1)  # [B,2] idx0→B, idx1→A
        return logits, A

# --------- Dataset / Loader ---------

def majority_label(att_seq, s, e):
    """att_seq: array of 'A'/'B' per-sample; window [s:e) → 0 for B, 1 for A."""
    seg = att_seq[s:e]
    a_count = np.sum(seg == 'A'); b_count = np.sum(seg == 'B')
    return 1 if a_count >= b_count else 0

class AADDataset(Dataset):
    """
    Windows EEG/envA/envB with label y in {0(B),1(A)}
    """
    def __init__(self, eeg, envA, envB, att_AB, fs, win_sec, hop_sec, z_m=None, z_s=None, split="train", train_stats=None):
        super().__init__()
        self.fs = int(fs)
        self.win = int(round(win_sec * fs))
        self.hop = int(round(hop_sec * fs))
        self.eeg = eeg.astype(np.float32)     # [T,C]
        self.envA = envA.astype(np.float32)   # [T]
        self.envB = envB.astype(np.float32)   # [T]
        self.att = np.array(att_AB)
        self.idxs = sliding_indices(len(self.eeg), self.win, self.hop)
        # z-score: use provided stats (computed on train split)
        self.m, self.s = (z_m, z_s) if (z_m is not None and z_s is not None) else (None, None)
        self.split = split
        self.train_stats = train_stats

    def __len__(self): return len(self.idxs)

    def __getitem__(self, i):
        s, e = self.idxs[i]
        X = self.eeg[s:e]           # [win,C]
        A = self.envA[s:e]          # [win]
        B = self.envB[s:e]          # [win]
        y = majority_label(self.att, s, e)   # 0/1
        if self.m is not None:
            X = (X - self.m) / self.s
        return torch.from_numpy(X), torch.from_numpy(A), torch.from_numpy(B), torch.tensor(y, dtype=torch.long)

# --------- Data loading (helper integration) ---------

def load_subject_pair(preproc_dir: str, subj_id: str):
    """
    Preferred loader using user's helper module.
    Falls back to ImportError if helper not found.
    """
    try:
        import helper
    except Exception as e:
        raise RuntimeError("Could not import your 'helper' module. Place it on PYTHONPATH.") from e

    # expected: eeg[T,C], env_A[T], env_B[T], fs, att_AB ('A'/'B' per sample)
    out = helper.subject_eeg_env_ab_aad(preproc_dir, subj_id)
    eeg, envA, envB, fs, att_AB = out
    return eeg, envA, envB, fs, att_AB

# --------- Metrics, Train/Eval, Logging ---------

def accuracy_from_logits(logits, y):
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()

@dataclass
class TrainState:
    epoch: int = 0
    best_val_acc: float = -1.0
    best_path: str = ""

def save_csv_header(path, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)

def append_csv_row(path, row):
    with open(path, "a", newline="") as f:
        w = csv.writer(f); w.writerow(row)

# --------- Main train loop ---------

def train_one_epoch(model, loader, opt, scaler, device, amp=False, grad_clip=None):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    ce = nn.CrossEntropyLoss()
    for X, A, B, y in loader:
        X, A, B, y = to_device((X, A, B, y), device)
        opt.zero_grad(set_to_none=True)
        if amp:
            with torch.cuda.amp.autocast():
                logits, _ = model(X, A, B)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update()
        else:
            logits, _ = model(X, A, B)
            loss = ce(logits, y)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        acc = accuracy_from_logits(logits.detach(), y)
        bsz = X.size(0)
        total_loss += loss.item() * bsz
        total_acc += acc * bsz
        n += bsz
    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for X, A, B, y in loader:
        X, A, B, y = to_device((X, A, B, y), device)
        logits, _ = model(X, A, B)
        loss = ce(logits, y)
        acc = accuracy_from_logits(logits, y)
        bsz = X.size(0)
        total_loss += loss.item() * bsz
        total_acc += acc * bsz
        n += bsz
    return total_loss / n, total_acc / n

# --------- Argument parsing ---------

def build_argparser():
    p = argparse.ArgumentParser(description="Train AAD: GraphEncoder + EEGFormer + Audio TCN + CrossAttn")
    # paths
    p.add_argument("--preproc_dir", type=str, required=True, help="Preprocessed dataset folder")
    p.add_argument("--subject", type=str, required=True, help="Subject ID (e.g., S01)")
    p.add_argument("--outputs_dir", type=str, default="outputs_aad_eegformer", help="Where to save logs/checkpoints")
    # windowing
    p.add_argument("--win_sec", type=float, default=5.0)
    p.add_argument("--hop_sec", type=float, default=2.0)
    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience (epochs)")
    # amp / device
    p.add_argument("--amp", action="store_true", help="Use mixed precision (autocast + GradScaler)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # EEG encoder dims
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--L_graph", type=int, default=3)
    p.add_argument("--k_graph", type=int, default=8)
    p.add_argument("--heads_graph", type=int, default=4)
    # EEGFormer
    p.add_argument("--eegformer_depth", type=int, default=2)
    p.add_argument("--eegformer_heads", type=int, default=4)
    p.add_argument("--ff_expansion", type=int, default=4)
    p.add_argument("--eeg_kernel", type=int, default=9)
    p.add_argument("--bias_mode", type=str, default="rel", choices=["none", "rel", "alibi"])
    p.add_argument("--max_rel", type=int, default=128)
    # Audio TCN
    p.add_argument("--d_audio", type=int, default=64)
    p.add_argument("--tcn_blocks", type=int, default=4)
    p.add_argument("--tcn_kernel", type=int, default=9)
    # cross-attn
    p.add_argument("--heads_xattn", type=int, default=4)
    # logging
    p.add_argument("--tb", action="store_true", help="Enable TensorBoard logging")
    p.add_argument("--save_every", type=int, default=0, help="If >0, save checkpoint every N epochs")
    # resume
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume (best_model.pt)")
    return p

# --------- Main ---------

def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    import re
    sub = int(re.findall(r'\d+', args.subject)[0])
    eeg, envA, envB, fs, att_AB = load_subject_pair(args.preproc_dir, sub)
    T, C = eeg.shape
    print(f"[DATA] Subject {args.subject}: EEG shape {eeg.shape}, fs={fs}, env len={len(envA)}")

  
    split_t = int(0.8 * T)
    eeg_tr, eeg_va = eeg[:split_t], eeg[split_t:]
    envA_tr, envA_va = envA[:split_t], envA[split_t:]
    envB_tr, envB_va = envB[:split_t], envB[split_t:]
    att_tr, att_va = np.array(att_AB[:split_t]), np.array(att_AB[split_t:])

    # ---- Z-score stats from train ----
    m, s = zscore_train_stats(eeg_tr)
    eeg_tr = apply_zscore(eeg_tr, m, s)
    eeg_va = apply_zscore(eeg_va, m, s)

    # ---- Datasets / Loaders ----
    train_ds = AADDataset(eeg_tr, envA_tr, envB_tr, att_tr, fs, args.win_sec, args.hop_sec, z_m=None, z_s=None, split="train")
    val_ds   = AADDataset(eeg_va, envA_va, envB_va, att_va, fs, args.win_sec, args.hop_sec, z_m=None, z_s=None, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

    # ---- MNE positions ----
    _, ch_names, pos = make_info_and_positions(n_ch=C, sfreq=float(fs), prefer_biosemi=(C==64))
    print(f"[MNE] Using {len(ch_names)} channels; e.g., {ch_names[:5]}")

    # ---- Build model ----
    model = AADModel_EEGFormer(
        n_ch=C, pos=pos, d_model=args.d_model, d_audio=args.d_audio,
        L=args.L_graph, k=args.k_graph, heads_graph=args.heads_graph, heads_xattn=args.heads_xattn,
        dropout=args.dropout,
        eegformer_depth=args.eegformer_depth, eegformer_heads=args.eegformer_heads,
        ff_expansion=args.ff_expansion, eeg_kernel=args.eeg_kernel,
        bias_mode=args.bias_mode, max_rel=args.max_rel
    ).to(device)

    # ---- Optimizer & Scheduler ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Cosine schedule with no warmup for simplicity; adjust if needed
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # ---- AMP scaler ----
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ---- Outputs & logging ----
    out_dir = Path(args.outputs_dir) / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "training_log.csv"
    save_csv_header(csv_path, ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    writer = None
    if args.tb and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    # ---- Resume ----
    state = TrainState()
    best_path = out_dir / "best_model.pt"
    if args.resume and os.path.isfile(args.resume):
        print(f"[RESUME] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        state.epoch = ckpt.get("epoch", 0)
        state.best_val_acc = ckpt.get("best_val_acc", -1.0)
        print(f"[RESUME] Resumed at epoch={state.epoch}, best_val_acc={state.best_val_acc:.4f}")

    # ---- Train loop with early stopping ----
    bad_epochs = 0
    for epoch in range(state.epoch, args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, scaler, device, amp=args.amp, grad_clip=args.grad_clip)
        va_loss, va_acc = evaluate(model, val_loader, device)
        sched.step()

        lr_curr = sched.get_last_lr()[0]
        append_csv_row(csv_path, [epoch+1, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{va_loss:.6f}", f"{va_acc:.6f}", f"{lr_curr:.6e}"])
        if writer is not None:
            writer.add_scalar("loss/train", tr_loss, epoch+1)
            writer.add_scalar("loss/val", va_loss, epoch+1)
            writer.add_scalar("acc/train", tr_acc, epoch+1)
            writer.add_scalar("acc/val", va_acc, epoch+1)
            writer.add_scalar("lr", lr_curr, epoch+1)

        # Save best
        if va_acc > state.best_val_acc:
            state.best_val_acc = va_acc
            torch.save({
                "epoch": epoch+1,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "best_val_acc": state.best_val_acc,
                "args": vars(args),
            }, best_path)
            bad_epochs = 0
            print(f"[EPOCH {epoch+1:03d}] train_acc={tr_acc:.4f} val_acc={va_acc:.4f}  (BEST ✓)  time={time.time()-t0:.1f}s")
        else:
            bad_epochs += 1
            print(f"[EPOCH {epoch+1:03d}] train_acc={tr_acc:.4f} val_acc={va_acc:.4f}  (patience {bad_epochs}/{args.patience})  time={time.time()-t0:.1f}s")

        # Optional periodic save
        if args.save_every and (epoch+1) % args.save_every == 0:
            ep_path = out_dir / f"epoch_{epoch+1:03d}.pt"
            torch.save({"epoch": epoch+1, "model": model.state_dict()}, ep_path)

        # Early stopping
        if bad_epochs >= args.patience:
            print("[STOP] Early stopping triggered.")
            break

    print(f"[DONE] Best val acc: {state.best_val_acc:.4f}; Best model: {best_path}")
    if writer is not None:
        writer.close()

if __name__ == "__main__":
    main()
