# train_eeg_env.py
import os, csv, argparse, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

from GraphEncoder import GraphEncoder
from TemporalStem import TemporalStem
from child_helper import (
    load_data_child_treatment,
    load_eeg_general_treatment,
    BASE_DIR,
)

# ------------------ HydroCel helpers ------------------
def make_hydrocel_info(n_ch: int, sfreq: float):
    """
    Return (info, ch_names, pos) for EGI HydroCel-129 template.
    If n_ch < 129 (e.g., 128 or after dropping channels), take the first n_ch positions.
    """
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    all_names = montage.ch_names  # 129
    ch_names = all_names[:n_ch]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage)
    pos_dict = montage.get_positions()['ch_pos']
    pos = np.stack([pos_dict[ch] for ch in ch_names])
    return info, ch_names, pos

# ------------------ Helper: z-score, windows ------------------
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

    def __len__(self): return len(self.spans)
    def __getitem__(self, i):
        a, b = self.spans[i]
        return self.X[a:b], self.y[a:b]

def split_indices(num_items, train_ratio=0.8, val_ratio=0.1, seed=12345):
    rng = np.random.default_rng(seed)
    idx = np.arange(num_items)
    rng.shuffle(idx)
    n_train = int(round(train_ratio * num_items))
    n_val = int(round(val_ratio * num_items))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
    return train_idx, val_idx, test_idx

class SubsetDS(Dataset):
    def __init__(self, base: Dataset, sel: np.ndarray):
        self.base = base
        self.sel = np.array(sel, dtype=int)
    def __len__(self): return len(self.sel)
    def __getitem__(self, i): return self.base[self.sel[i]]

# ------------------ Model ------------------
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
        """
        eeg: [B, T, N]
        bt_chunk: optional temporal chunk size (passed to GraphEncoder if supported)
        """
        H0 = self.stem(eeg)                                     # [B,T,d_stem]
        Lf = self.lift(H0)                                      # [B,T,d_lift]
        B, T, _ = H0.shape
        N = eeg.shape[-1]
        Xin = torch.cat([Lf.unsqueeze(2).expand(B, T, N, -1),   # [B,T,N,d_lift]
                         eeg.unsqueeze(-1)], dim=-1)            # [B,T,N,1] → concat → d_lift+1
        try:
            Z, S, A = self.graph(Xin, bt_chunk=bt_chunk)        # your GraphEncoder supports bt_chunk
        except TypeError:
            Z, S, A = self.graph(Xin)
        yhat = self.head(S).squeeze(-1)                         # [B,T]
        return yhat, A

# ------------------ Loss / metrics ------------------
def pearsonr_batch(yhat, y, eps=1e-8):
    yhat = yhat - yhat.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    num = (yhat * y).sum(dim=1)
    den = torch.sqrt((yhat**2).sum(dim=1) * (y**2).sum(dim=1) + eps)
    return num / (den + eps)

def evaluate(model, loader, device, bt_chunk=None, amp_dtype=None):
    model.eval()
    r_sum, loss_sum, n = 0.0, 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype,
                                    enabled=(amp_dtype is not None and torch.cuda.is_available())):
                yhat, _ = model(xb, bt_chunk=bt_chunk)
                r = pearsonr_batch(yhat, yb).mean()
                loss = 0.7 * (1 - r) + 0.3 * F.l1_loss(yhat, yb)
            r_sum += r.item() * xb.size(0)
            loss_sum += loss.item() * xb.size(0)
            n += xb.size(0)
    return (r_sum / max(1, n)), (loss_sum / max(1, n))

# ------------------ Plotting ------------------
def plot_training_curves(hist, outdir):
    os.makedirs(outdir, exist_ok=True)
    # loss
    plt.figure(figsize=(6,4))
    plt.plot(hist['train_loss'], label='train')
    plt.plot(hist['val_loss'], label='val')
    plt.xlabel('epoch'); plt.ylabel('loss (0.7·(1−r)+0.3·L1)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_curve.png'), dpi=150); plt.close()
    # r
    plt.figure(figsize=(6,4))
    plt.plot(hist['train_r'], label='train r')
    plt.plot(hist['val_r'], label='val r')
    plt.xlabel('epoch'); plt.ylabel('Pearson r')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'training_r.png'), dpi=150); plt.close()

def plot_adjacency_heatmap(A, outdir, name='A_heatmap.png', title='Blended adjacency A'):
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(6,5))
    plt.imshow(A, cmap='viridis')
    plt.colorbar(label='weight')
    plt.title(title)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, name), dpi=150); plt.close()

def plot_sensor_edges(A, info, outdir, topk=120, name='A_edges_sensors.png', title='Top edges on sensor map'):
    os.makedirs(outdir, exist_ok=True)
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
    if A0 is None: return
    os.makedirs(outdir, exist_ok=True)
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
    os.makedirs(outdir, exist_ok=True)
    W = A.copy(); np.fill_diagonal(W, 0.0)
    in_strength = W.sum(axis=0)
    fig, ax = plt.subplots(figsize=(5,4))
    mne.viz.plot_topomap(in_strength, info, axes=ax, show=False)
    ax.set_title('In-strength (col-sum of A)')
    fig.savefig(os.path.join(outdir, 'in_strength_topomap.png'), dpi=150); plt.close(fig)

def _amp_dtype(amp):
    if amp == 'fp16': return torch.float16
    if amp == 'bf16': return torch.bfloat16
    return None

# ------------------ Train one subject ------------------
def train_one_subject(eeg, env, fs, args, subj_dir, subj_name="subject"):
    """
    Train with early stopping; save per-epoch log CSV, best model, plots & adjacency in subj_dir.
    """
    os.makedirs(subj_dir, exist_ok=True)
    logs_dir = os.path.join(subj_dir, "Logs")
    os.makedirs(logs_dir, exist_ok=True)

    # z-score
    X, _, _ = zscore_train(eeg)
    y, _, _ = zscore_train(env[:, None]); y = y[:, 0]

    win = int(round(args.win_sec * fs))
    hop = int(round(args.hop_sec * fs))
    n_ch = X.shape[1]
    info, ch_names, pos = make_hydrocel_info(n_ch=n_ch, sfreq=fs)

    # Memory sanity print (rough)
    d_in = 128
    est_bytes = 4.0 * args.batch * win * n_ch * d_in
    print(f"[{subj_name}] Est. Xin per batch ~ {est_bytes/1e6:.1f} MB (win={win}, n_ch={n_ch}, d_in={d_in})")

    ds = EEGEnvDataset(X, y, win, hop)
    if len(ds) == 0:
        print(f"[{subj_name}] No windows with win={win}, hop={hop}. Reducing.")
        win = max(1, min(len(X) // 2, int(round(1.0 * fs))))
        hop = max(1, win // 2)
        ds = EEGEnvDataset(X, y, win, hop)
        if len(ds) == 0:
            raise RuntimeError(f"No windows could be formed for {subj_name}. Consider reducing --win_sec.")

    tr_idx, va_idx, te_idx = split_indices(len(ds), train_ratio=0.8, val_ratio=0.1, seed=2025)
    train_ds, val_ds, test_ds = SubsetDS(ds, tr_idx), SubsetDS(ds, va_idx), SubsetDS(ds, te_idx)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    model = ERGraphModel(n_ch, pos).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_path = os.path.join(logs_dir, f"{subj_name}_log.csv")
    with open(log_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss", "val_loss", "train_r", "val_r"])

    best_val = float('inf'); best_state = None; wait = 0
    hist = {'train_loss': [], 'val_loss': [], 'train_r': [], 'val_r': []}
    amp_dtype = _amp_dtype(args.amp)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss_sum, tr_r_sum, n_tr = 0.0, 0.0, 0
        for xb, yb in train_ld:
            xb, yb = xb.to(args.device), yb.to(args.device)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype,
                                    enabled=(amp_dtype is not None and torch.cuda.is_available())):
                yhat, _ = model(xb, bt_chunk=args.bt_chunk)
                r = pearsonr_batch(yhat, yb).mean()
                loss = 0.7 * (1 - r) + 0.3 * F.l1_loss(yhat, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss_sum += loss.item() * xb.size(0)
            tr_r_sum += r.item() * xb.size(0)
            n_tr += xb.size(0)

        train_r = tr_r_sum / max(1, n_tr)
        train_loss = tr_loss_sum / max(1, n_tr)
        val_r, val_loss = evaluate(model, val_ld, args.device, bt_chunk=args.bt_chunk, amp_dtype=amp_dtype)

        hist['train_loss'].append(train_loss); hist['val_loss'].append(val_loss)
        hist['train_r'].append(train_r);       hist['val_r'].append(val_r)

        with open(log_path, 'a', newline='') as f:
            w = csv.writer(f); w.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{train_r:.6f}", f"{val_r:.6f}"])

        print(f"[{subj_name}] Epoch {epoch:03d} | train_loss {train_loss:.4f} (r={train_r:.3f}) "
              f"| val_loss {val_loss:.4f} (r={val_r:.3f})")

        # Early stopping on val_loss
        if best_val - val_loss > float(args.min_delta):
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(subj_dir, f"{subj_name}_best.pt"))
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[{subj_name}] Early stopping at epoch {epoch}")
                break

    # plots: training curves
    plot_training_curves(hist, subj_dir)

    # Load best and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    test_r, test_loss = evaluate(model, test_ld, args.device, bt_chunk=args.bt_chunk, amp_dtype=amp_dtype)
    print(f"[{subj_name}] Final Test: loss={test_loss:.4f}, r={test_r:.4f}")

    # Extract adjacency from a validation batch and plot/save artifacts
    with torch.no_grad():
        sample_loader = val_ld if len(val_ds) > 0 else test_ld
        xb, yb = next(iter(sample_loader))
        xb = xb.to(args.device)
        with torch.amp.autocast(device_type='cuda', dtype=amp_dtype,
                                enabled=(amp_dtype is not None and torch.cuda.is_available())):
            _, A_t = model(xb, bt_chunk=args.bt_chunk)
        A = A_t.detach().cpu().numpy()
        if A.ndim == 3:  # if batched
            A = A.mean(axis=0)

    # Save adjacency numeric + plots
    np.savetxt(os.path.join(subj_dir, 'A_final.csv'), A, delimiter=',')
    plot_adjacency_heatmap(A, subj_dir, name='A_heatmap.png', title='Blended adjacency A')

    # Optional prior A0 if available
    try:
        A0 = model.graph.A0.detach().cpu().numpy()
        np.savetxt(os.path.join(subj_dir, 'A0.csv'), A0, delimiter=',')
        plot_adjacency_heatmap(A0, subj_dir, name='A0_heatmap.png', title='Initial prior A0')
    except Exception:
        A0 = None

    # Sensor edge visualizations
    plot_sensor_edges(A, info, subj_dir, topk=120, name='A_edges_sensors.png')
    plot_sensor_edges_delta(A, A0, info, subj_dir, topk=80)
    plot_in_strength_topomap(A, info, subj_dir)

    return float(best_val), float(test_loss), float(test_r)

# ------------------ CLI ------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path', type=str, required=True,
                   help="CSV with columns: eeg_file, tsv_file, audio_file")
    p.add_argument('--base_dir', type=str, default=None,
                   help="Root folder that CSV paths are relative to. Defaults to child_helper.BASE_DIR")
    p.add_argument('--preprocessing_style', type=str, choices=['child', 'general'], default="general",
                   help="Choose preprocessing pipeline: 'child' or 'general'")
    p.add_argument('--outdir', type=str, required=True,
                   help="Output directory where models, plots, summary CSV are stored")
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--win_sec', type=float, default=5.0)     # 5 s window (safe at 64 Hz)
    p.add_argument('--hop_sec', type=float, default=2.5)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--patience', type=int, default=10, help="Early stopping patience (epochs)")
    p.add_argument('--min_delta', type=float, default=1e-3, help="Minimum improvement in val_loss to reset patience")
    p.add_argument('--bt_chunk', type=int, default=256, help='Temporal chunk size for GraphEncoder')
    p.add_argument('--amp', type=str, choices=['none','fp16','bf16'], default='none', help='Mixed precision for CUDA')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    summary_path = os.path.join(args.outdir, "summary.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['row_index', 'style', 'best_val_loss', 'final_test_loss', 'final_test_r'])

    # 1) Load data once according to preprocessing_style
    if args.preprocessing_style == 'child':
        data = load_data_child_treatment(args.csv_path, base_dir=args.base_dir)
    else:  # 'general'
        base = args.base_dir or BASE_DIR
        data = load_eeg_general_treatment(args.csv_path, base_dir=base)

    eeg_list = list(data["eeg"])
    aud_list = list(data["audio"])
    fs_list  = list(data.get("sfreq", []))
    n_rows = len(eeg_list)
    if n_rows == 0:
        raise RuntimeError("No rows found after loading data. Check your CSV and base_dir.")

    # 2) Train per row actually present
    for subj_index in range(n_rows):
        print(f"\n=== Training Row {subj_index+1}/{n_rows} ({args.preprocessing_style}) ===")
        eeg = eeg_list[subj_index]
        env = aud_list[subj_index]

        if env.shape[0] != eeg.shape[0]:
            raise ValueError(f"Length mismatch on row {subj_index}: EEG {eeg.shape[0]} vs audio {env.shape[0]}")

        # Use the fs that came from preprocessing (64.0 for both styles now)
        fs = float(fs_list[subj_index]) if fs_list else 64.0

        subj_name = f"{args.preprocessing_style}_row{subj_index+1}"
        subj_dir = os.path.join(args.outdir, subj_name)

        best_val, test_loss, test_r = train_one_subject(eeg, env, fs, args, subj_dir, subj_name)

        # append to summary.csv
        with open(summary_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([subj_index+1, args.preprocessing_style, f"{best_val:.6f}", f"{test_loss:.6f}", f"{test_r:.6f}"])

        print(f"✅ Completed {subj_name}. Best model: {os.path.join(subj_dir, subj_name + '_best.pt')}")
        print(f"   Logs & plots saved under: {subj_dir}")
        print(f"   Summary updated at: {summary_path}")

if __name__ == "__main__":
    main()
