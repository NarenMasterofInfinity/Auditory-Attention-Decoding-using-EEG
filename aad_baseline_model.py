#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
test.py — AAD baseline (EEG + envA + envB) — 1s windows version
- Encoders: CNN_EEG, CNN_A, CNN_B → concat → Linear → logits[2] (idx1 = A)
- Data root fixed: D:\FYP\DATA_preproc
- Output root fixed: aad_baseline_1s\
- Subject folders: aad_baseline_1s\s1 ... \s18
- Global logs: aad_baseline_1s\Logs\ + aad_baseline_1s\aad_baseline_1s.log
- Mirrors your previous 5s script, just with win_sec=1.0 (hop_sec=0.5 by default).
"""
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os, sys, math, time, csv, argparse, logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ====== FIXED DEFAULTS (edit here if needed) ======
DEFAULT_PREPROC_DIR = r'D:\FYP\DATA_preproc'  # your data root (EEG + envA + envB)
DEFAULT_OUTDIR      = r'aad_baseline_1s'      # output root for 1s run
DEFAULT_SUBJECTS    = list(range(1, 19))      # s1..s18
DEFAULT_WIN_SEC     = 1.0                     # 1-second window
DEFAULT_HOP_SEC     = 0.5                     # 0.5-second hop (50% overlap)
# ==================================================

# Optional CLI overrides (not required to run)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--preproc_dir", type=str, default=DEFAULT_PREPROC_DIR)
parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
parser.add_argument("--subjects", type=str, default="1-18")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch", type=int, default=16)      # smaller windows allow larger batch
parser.add_argument("--win_sec", type=float, default=DEFAULT_WIN_SEC)
parser.add_argument("--hop_sec", type=float, default=DEFAULT_HOP_SEC)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument("--amp", type=str, choices=["none","fp16","bf16"], default="bf16")
parser.add_argument("--workers", type=int, default=0)     # 0 by default for Windows stability
parser.add_argument("--prefetch", type=int, default=2)
parser.add_argument("--accum", type=int, default=1)
parser.add_argument("--patience", type=int, default=30)   # faster convergence on 1s windows
parser.add_argument("--min_delta", type=float, default=5e-4)
parser.add_argument("--warmup_pct", type=float, default=0.05)
parser.add_argument("--final_lr_pct", type=float, default=0.1)
parser.add_argument("--compile", action="store_true")
args, _ = parser.parse_known_args()

# Your project helper (unchanged)
import helper  # must be importable from repo

# -------------------- filesystem & logging --------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def setup_logging(out_root):
    ensure_dir(out_root)
    log_path = os.path.join(out_root, "aad_baseline_1s.log")
    logger = logging.getLogger("aad_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8"); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout);              sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger, log_path

# -------------------- small utilities --------------------

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
    a = np.sum(np.array(chars) == 'A')
    b = len(chars) - a
    return 1 if a >= b else 0   # 1→A, 0→B

# -------------------- dataset --------------------

class AADDataset(Dataset):
    """
    Returns: eeg_win [W,C], envA_win [W], envB_win [W], label (0/1)
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
    plt.figure(figsize=(6,4)); plt.plot(hist['train_loss'], label='train'); plt.plot(hist['val_loss'], label='val')
    plt.xlabel('epoch'); plt.ylabel('CE loss'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_curve.png'), dpi=150); plt.close()

    plt.figure(figsize=(6,4)); plt.plot(hist['train_acc'], label='train acc'); plt.plot(hist['val_acc'], label='val acc')
    plt.xlabel('epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_acc.png'), dpi=150); plt.close()

def plot_test_probabilities(probs, labels, outdir, fname='test_probs.png'):
    ensure_dir(outdir)
    plt.figure(figsize=(8,3))
    plt.plot(probs[:,1], label='P(att=A)')
    plt.plot(labels, label='label (A=1, B=0)', alpha=0.6)
    plt.xlabel('window'); plt.ylabel('probability / label'); plt.legend(); plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150); plt.close()
    return path

# -------------------- CNN encoders --------------------

class EEGCNN(nn.Module):
    def __init__(self, in_ch, hidden=128, emb=128, kernel=9, blocks=3, dropout=0.1):
        super().__init__()
        layers = [nn.Conv1d(in_ch, hidden, kernel, padding=kernel//2),
                  nn.GELU(), nn.Dropout(dropout)]
        for _ in range(blocks-1):
            layers += [nn.Conv1d(hidden, hidden, kernel, padding=kernel//2),
                       nn.GELU(), nn.Dropout(dropout)]
        self.net  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden, emb)

    def forward(self, eeg):               # eeg: [B,W,C]
        x = eeg.transpose(1,2)            # [B,C,W]
        h = self.net(x)                   # [B,H,W]
        g = self.pool(h).squeeze(-1)      # [B,H]
        z = self.proj(g)                  # [B,emb]
        return z

class AudioCNN(nn.Module):
    def __init__(self, hidden=64, emb=64, kernel=9, blocks=3, dropout=0.1):
        super().__init__()
        layers = [nn.Conv1d(1, hidden, kernel, padding=kernel//2),
                  nn.GELU(), nn.Dropout(dropout)]
        for _ in range(blocks-1):
            layers += [nn.Conv1d(hidden, hidden, kernel, padding=kernel//2),
                       nn.GELU(), nn.Dropout(dropout)]
        self.net  = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden, emb)

    def forward(self, env):               # env: [B,W]
        x = env.unsqueeze(1)              # [B,1,W]
        h = self.net(x)                   # [B,H,W]
        g = self.pool(h).squeeze(-1)      # [B,H]
        z = self.proj(g)                  # [B,emb]
        return z

# -------------------- Model --------------------

class AADBaselineCNN(nn.Module):
    def __init__(self, n_ch, eeg_emb=128, aud_emb=64, eeg_hidden=128, aud_hidden=64,
                 eeg_blocks=3, aud_blocks=3, dropout=0.1):
        super().__init__()
        self.eeg_enc = EEGCNN(n_ch, eeg_hidden, eeg_emb, 9, eeg_blocks, dropout)
        self.audA    = AudioCNN(aud_hidden, aud_emb, 9, aud_blocks, dropout)
        self.audB    = AudioCNN(aud_hidden, aud_emb, 9, aud_blocks, dropout)
        self.head    = nn.Linear(eeg_emb + 2*aud_emb, 2)  # 0->B, 1->A

    def forward(self, eeg, envA, envB):
        z_eeg = self.eeg_enc(eeg)
        z_a   = self.audA(envA)
        z_b   = self.audB(envB)
        z     = torch.cat([z_eeg, z_a, z_b], dim=-1)
        return self.head(z)

# -------------------- train/eval helpers --------------------

def split_indices(n, train_ratio=0.7, val_ratio=0.1, seed=123):
    idx = np.arange(n); rng = np.random.default_rng(seed); rng.shuffle(idx)
    ntr = int(round(train_ratio*n)); nv = int(round(val_ratio*n))
    return idx[:ntr], idx[ntr:ntr+nv], idx[ntr+nv:]

def subset_dataset(ds, idx):
    if isinstance(idx, np.ndarray):
        idx = idx.tolist()
    return Subset(ds, idx)

def make_sched(optimizer, total_steps, warmup_steps=0, final_lr_pct=0.1):
    def lr_lambda(step):
        step = min(step, total_steps)
        if warmup_steps>0 and step<warmup_steps: return (step+1)/warmup_steps
        if total_steps==warmup_steps: return final_lr_pct
        prog = (step-warmup_steps)/max(1,(total_steps-warmup_steps))
        cosine = 0.5*(1+math.cos(math.pi*prog))
        return final_lr_pct + (1-final_lr_pct)*cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

@torch.no_grad()
def evaluate_cls(model, loader, device, amp_dtype):
    model.eval(); acc_sum, n = 0.0, 0; probs_all, labs_all = [], []
    for xb, a, b, yb in loader:
        xb=xb.to(device, non_blocking=True); a=a.to(device, non_blocking=True)
        b=b.to(device, non_blocking=True);   yb=yb.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
            logits = model(xb, a, b)
            pred   = logits.argmax(-1)
            acc    = (pred==yb).float().sum().item()
            prob   = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        acc_sum += acc; n += xb.size(0)
        probs_all.append(prob); labs_all.append(yb.cpu().numpy())
    probs = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0,2))
    labs  = np.concatenate(labs_all, axis=0) if labs_all else np.zeros((0,))
    try: auc = roc_auc_score(labs, probs[:,1])
    except Exception: auc = float('nan')
    return acc_sum/max(1,n), auc, probs, labs

# -------------------- per-subject train --------------------

def train_one_subject(preproc_dir, subj_id, epochs, batch, win_sec, hop_sec, lr, device,
                      subj_outdir, logs_root, workers, prefetch, accum,
                      patience, min_delta, warmup_pct, final_lr_pct, amp, compile_flag, logger):

    ensure_dir(subj_outdir)
    # data load (prefer both envs)
    use_fallback = False
    try:
        eeg, envA, envB, fs, attAB = helper.subject_eeg_env_ab_aad(preproc_dir, subj_id)
    except Exception:
        eeg, env_att, fs, attAB = helper.subject_eeg_env_ab(preproc_dir, subj_id)
        shift = int(round(1.0*fs))
        envB  = np.roll(env_att, shift)
        envA  = env_att
        use_fallback = True
        logger.warning("S%d: Using fallback envB (shifted). Replace with real env_B for proper AAD.", subj_id)

    X = eeg.astype(np.float32)
    X, _, _ = zscore_train(X)
    ds = AADDataset(X, envA, envB, attAB, fs, win_sec, hop_sec)
    tr_idx, va_idx, te_idx = split_indices(len(ds), 0.7, 0.1, seed=2000+subj_id)
    train_ds, val_ds, test_ds = subset_dataset(ds, tr_idx), subset_dataset(ds, va_idx), subset_dataset(ds, te_idx)

    # Windows-safe DataLoaders
    use_mp = (workers > 0)
    pinmem = (DEVICE.type == 'cuda')
    prefetch_kw = {'prefetch_factor': prefetch} if use_mp else {}

    train_ld = DataLoader(
        train_ds, batch_size=batch, shuffle=True, drop_last=True,
        num_workers=workers, pin_memory=pinmem,
        persistent_workers=use_mp, **prefetch_kw
    )
    val_ld = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=workers, pin_memory=pinmem,
        persistent_workers=use_mp, **prefetch_kw
    )
    test_ld = DataLoader(
        test_ds, batch_size=batch, shuffle=False,
        num_workers=workers, pin_memory=pinmem,
        persistent_workers=use_mp, **prefetch_kw
    )

    # model
    n_ch = X.shape[1]
    model = AADBaselineCNN(n_ch, eeg_emb=128, aud_emb=64,
                           eeg_hidden=128, aud_hidden=64,
                           eeg_blocks=3, aud_blocks=3, dropout=0.1).to(device)
    if compile_flag:
        try: model = torch.compile(model, mode="max-autotune")
        except Exception as e: logger.warning("torch.compile failed (continue): %s", str(e))

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, math.ceil(len(train_ld)/max(1,accum)))
    total = epochs*steps
    sched = make_sched(opt, total, warmup_steps=int(warmup_pct*total), final_lr_pct=final_lr_pct)

    use_fp16 = (amp=='fp16'); use_bf16 = (amp=='bf16')
    amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    scaler    = torch.amp.GradScaler('cuda', enabled=use_fp16)

    best_val = float('inf'); best_state=None; wait=0
    hist = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

    # per-subject CSV inside global Logs
    ensure_dir(logs_root)
    csv_path = os.path.join(logs_root, f"s{subj_id}_train_log.csv")
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','train_loss','val_loss','train_acc','val_acc','lr'])

    def _to(x): return x.to(device, non_blocking=True)

    for ep in range(1, epochs+1):
        model.train(); trL=0.0; trA=0.0; n=0
        t0=time.time(); opt.zero_grad(set_to_none=True)

        for step,(xb,a,b,yb) in enumerate(train_ld,1):
            xb=_to(xb); a=_to(a); b=_to(b); yb=_to(yb)
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                logits = model(xb,a,b)
                loss   = F.cross_entropy(logits,yb)/max(1,accum)
            if use_fp16: scaler.scale(loss).backward()
            else:        loss.backward()

            if (step % max(1,accum))==0:
                if use_fp16: scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_fp16: scaler.step(opt); scaler.update()
                else:        opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()

            pred = logits.argmax(-1)
            trA += (pred==yb).float().sum().item()
            trL += loss.item()*xb.size(0)*max(1,accum)
            n   += xb.size(0)

        tr_loss = trL/max(1,n); tr_acc = trA/max(1,n)

        # val
        model.eval(); vL=0.0; vA=0.0; vn=0
        with torch.no_grad():
            for xb,a,b,yb in val_ld:
                xb=_to(xb); a=_to(a); b=_to(b); yb=_to(yb)
                with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                    logits = model(xb,a,b)
                    loss   = F.cross_entropy(logits,yb)
                vL += loss.item()*xb.size(0)
                vA += (logits.argmax(-1)==yb).float().sum().item()
                vn += xb.size(0)
        va_loss = vL/max(1,vn); va_acc = vA/max(1,vn)
        hist['train_loss'].append(tr_loss); hist['val_loss'].append(va_loss)
        hist['train_acc'].append(tr_acc);   hist['val_acc'].append(va_acc)

        lr_now = sched.get_last_lr()[0] if hasattr(sched,'get_last_lr') else opt.param_groups[0]['lr']
        sec = time.time()-t0
        logger.info("S%02d Ep%02d | train %.4f acc=%.3f | val %.4f acc=%.3f | %.1fs",
                    subj_id, ep, tr_loss, tr_acc, va_loss, va_acc, sec)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, f"{tr_loss:.6f}", f"{va_loss:.6f}", f"{tr_acc:.6f}", f"{va_acc:.6f}", f"{lr_now:.8f}"])

        improved = (best_val - va_loss) > float(min_delta)
        if improved:
            best_val = va_loss; wait = 0
            best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save(best_state, os.path.join(subj_outdir, 'best_model.pt'))
        else:
            wait += 1
            if wait >= patience:
                logger.info("S%02d Early stopping at epoch %d (best val %.4f)", subj_id, ep, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # metrics & artifacts
    val_acc, val_auc, _, _ = evaluate_cls(model, val_ld, device, amp_dtype)
    test_acc, test_auc, te_probs, te_labs = evaluate_cls(model, test_ld, device, amp_dtype)
    logger.info("S%02d Final: Val acc=%.3f AUC=%.3f | Test acc=%.3f AUC=%.3f",
                subj_id, val_acc, val_auc, test_acc, test_auc)

    # plots under subject folder
    plot_training_curves(hist, subj_outdir)
    vis_dir = ensure_dir(os.path.join(subj_outdir, "vis"))
    plot_test_probabilities(te_probs, te_labs, vis_dir, 'test_probs.png')

    if use_fallback:
        with open(os.path.join(subj_outdir, 'README_FALLBACK.txt'), 'w') as f:
            f.write("env_B was synthesized by circular shift of env_att by 1 s. Replace with real env_B for proper AAD.\n")

    return float(val_acc), float(val_auc), float(test_acc), float(test_auc)

# -------------------- main --------------------

def main():
    out_root = args.outdir
    logger, log_path = setup_logging(out_root)
    logger.info("Log file: %s", log_path)
    logger.info("Data dir : %s", args.preproc_dir)
    logger.info("Out dir  : %s", out_root)
    logger.info("Device   : %s | torch %s | build_cuda=%s",
                args.device, torch.__version__, torch.version.cuda)

    # subjects
    if args.subjects and '-' in args.subjects:
        a,b = args.subjects.split('-'); subjects = list(range(int(a), int(b)+1))
    elif args.subjects and ',' in args.subjects:
        subjects = [int(x) for x in args.subjects.split(',') if x.strip()]
    else:
        subjects = DEFAULT_SUBJECTS

    # summary CSV at root
    summary_path = os.path.join(out_root, "summary_aad.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, 'w', newline='') as f:
            csv.writer(f).writerow(['subject','val_acc','val_auc','test_acc','test_auc'])

    # global Logs folder
    logs_root = ensure_dir(os.path.join(out_root, "Logs"))

    # run all subjects
    device = torch.device(args.device)
    for sid in subjects:
        subj_dir = ensure_dir(os.path.join(out_root, f"s{sid}"))  # lowercase s
        logger.info("=== Subject %d → %s ===", sid, subj_dir)
        vacc, vauc, tacc, tauc = train_one_subject(
            args.preproc_dir, sid, args.epochs, args.batch, args.win_sec, args.hop_sec,
            args.lr, device, subj_dir, logs_root, args.workers, args.prefetch, args.accum,
            args.patience, args.min_delta, args.warmup_pct, args.final_lr_pct,
            args.amp, args.compile, logger
        )
        with open(summary_path, 'a', newline='') as f:
            csv.writer(f).writerow([sid, vacc, vauc, tacc, tauc])
        logger.info("S%02d: Val acc=%.3f AUC=%.3f | Test acc=%.3f AUC=%.3f", sid, vacc, vauc, tacc, tauc)

    logger.info("Wrote summary: %s", summary_path)

if __name__ == "__main__":
    main()
