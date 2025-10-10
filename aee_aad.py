#!/usr/bin/env python3
"""
eval_aad_from_aee.py

Goal
----
Use your existing AEE models (outputs/SX/best.pt) to do AAD by correlation comparison:
- Slide a window across the full recording (win_sec, hop_sec, lag_ms).
- For each window, run the AEE model to reconstruct the audio envelope ê(t).
- Compute Pearson r(ê, env_A) and r(ê, env_B) (with the same lag).
- Predict attended = argmax{r_A, r_B}. Compare against ground truth `att_AB`.
- Save per-window CSV + per-subject summary + overall summary.

Assumptions
-----------
- Your AEE training code is in model.py and defines:
    - EEGGraphConformer
    - make_biosemi64_info
- Helper module provides (preferred):
    helper.subject_eeg_env_pair_ab(PREPROC_DIR, subj_id)
      -> eeg[T,C], env_A[T], env_B[T], fs, att_AB  # att_AB is 'A' or 'B' per sample OR per-window label can be derived
  Fallback:
    helper.subject_eeg_env_ab(PREPROC_DIR, subj_id)
      -> eeg[T,C], env_att[T], fs, att_AB
    Then we synthesize a "B" envelope by circularly shifting env_att by 1s.

- Each subject's best model checkpoint is at: {--models_dir}/S{subj}/best.pt
  If not found, that subject is skipped (as requested).

Outputs
-------
- {--outdir}/S{subj}/aad_windows.csv   (per-window correlations, preds, truth)
- {--outdir}/S{subj}/aad_summary.json  (acc, mean r_A, mean r_B, #windows)
- {--outdir}/summary_aad.csv           (per-subject accuracy + counts)
- Optional quick plots (turned on with --plots)

Notes
-----
- Normalization: by default we compute z-score from the *first trainstat_pct* of the time series
  (default 0.7, matching your AEE script vibe) and apply to the whole series.
- If att_AB is per-sample, the window label is majority vote of samples {'A','B'} inside the window.
- Ties in correlation (rare) -> prefer 'A' by default (configurable).

"""

import os, csv, json, math, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Import user's modules
import importlib
helper = importlib.import_module('helper')
model_mod = importlib.import_module('model')  # must be in PYTHONPATH or same folder

EEGGraphConformer = getattr(model_mod, 'EEGGraphConformer')
make_biosemi64_info = getattr(model_mod, 'make_biosemi64_info')

def window_indices(T, win, hop, start=0):
    spans, t = [], start
    while t + win <= T:
        spans.append((t, t + win))
        t += hop
    return np.asarray(spans, dtype=np.int64)

def pearsonr_1d(a, b, eps=1e-8):
    a = a - a.mean()
    b = b - b.mean()
    num = float((a * b).sum())
    den = math.sqrt(float((a*a).sum()) * float((b*b).sum()) + eps)
    return num / (den + eps)

def batch_predict_windows(net, eeg_win_list, device, bt_chunk=None, amp_dtype=None):
    """
    eeg_win_list: list of np arrays [T,C] (already normalized)
    Returns: list of np arrays yhat[T]
    """
    net.eval()
    y_list = []
    with torch.no_grad():
        for i in range(0, len(eeg_win_list), 32):
            batch = eeg_win_list[i:i+32]
            xb = torch.from_numpy(np.stack(batch, axis=0)).to(device)  # [B,T,C]
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                yhat, _ = net(xb, bt_chunk=bt_chunk)  # [B,T]
            y_list.extend([yhat[j].detach().cpu().numpy() for j in range(yhat.size(0))])
    return y_list

def majority_label(att_slice):
    # att_slice: array of shape [L] of 'A' or 'B' (or possibly 0/1)
    # Convert to 'A'/'B' robustly
    if att_slice.dtype.kind in ('U','S','O'):
        a = np.sum(att_slice == 'A')
        b = np.sum(att_slice == 'B')
    else:
        # assume binary {0,1} where 0->A, 1->B
        a = np.sum(att_slice == 0)
        b = np.sum(att_slice == 1)
    return 'A' if a >= b else 'B'

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def find_checkpoint(models_dir, subj):
    # Look for best.pt under models_dir/S{subj}*, prefer deeper EXACT match first
    cand1 = Path(models_dir)/f"S{subj}"/"best_model.pt"
    if cand1.exists():
        return cand1
    # Try alternative folder names user might have used (e.g., S1_FULL, S1_AEE, etc.)
    root = Path(models_dir)
    for p in root.glob(f"S{subj}/best_model.pt"):
        return p
    return None

def build_model(n_ch, pos, ckpt_path, device,
                d_stem=256, d_lift=127, d_in=128, d_model=128,
                L_graph=3, k=8, graph_heads=4, graph_dropout=0.2,
                conf_depth=2, conf_heads=4, ff_expansion=4, conf_dropout=0.2,
                kernel_size=9, bias_mode='alibi', max_rel=128, causal=True):
    net = EEGGraphConformer(
        n_ch=n_ch, pos=pos,
        d_stem=d_stem, d_lift=d_lift, d_in=d_in, d_model=d_model,
        L_graph=L_graph, k=k, graph_heads=graph_heads, graph_dropout=graph_dropout,
        conf_depth=conf_depth, conf_heads=conf_heads, ff_expansion=ff_expansion, conf_dropout=conf_dropout,
        kernel_size=kernel_size, bias_mode=bias_mode, max_rel=max_rel, causal=causal
    ).to(device)
    sd = torch.load(ckpt_path, map_location='cpu')
    net.load_state_dict(sd, strict=False)
    return net

def normalize_series(eeg, env, pct=0.7):
    """
    Train-only style normalization by default:
    - z-score EEG per channel using first pct portion
    - z-score env using first pct portion
    """
    T = len(eeg)
    cut = int(round(T * pct))
    Xm = eeg[:cut].mean(axis=0, keepdims=True)
    Xs = eeg[:cut].std(axis=0, keepdims=True) + 1e-8
    ym = env[:cut].mean(keepdims=True)
    ys = env[:cut].std(keepdims=True) + 1e-8
    Xz = (eeg - Xm) / Xs
    yz = (env - ym) / ys
    return Xz.astype(np.float32), yz.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preproc_dir', type=str, required=True)
    ap.add_argument('--models_dir', type=str, default='outputs')
    ap.add_argument('--outdir', type=str, default='outputs_aad_eval')
    ap.add_argument('--subjects', type=str, default='1-18', help="e.g., '1-18' or '1,2,5,9'")
    ap.add_argument('--win_sec', type=float, default=5.0)
    ap.add_argument('--hop_sec', type=float, default=2.5)
    ap.add_argument('--lag_ms', type=float, default=120.0)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--bt_chunk', type=int, default=256)
    ap.add_argument('--amp', type=str, choices=['none','fp16','bf16'], default='bf16')
    ap.add_argument('--trainstat_pct', type=float, default=0.7)
    ap.add_argument('--prefer_on_tie', type=str, choices=['A','B'], default='A')
    ap.add_argument('--plots', action='store_true')
    args = ap.parse_args()

    # Subjects list
    subs = []
    if '-' in args.subjects:
        a,b = args.subjects.split('-')
        subs = list(range(int(a), int(b)+1))
    else:
        subs = [int(s) for s in args.subjects.split(',') if s.strip()]

    ensure_dir(args.outdir)
    summary_csv = Path(args.outdir)/'summary_aad.csv'
    if not summary_csv.exists():
        with open(summary_csv, 'w', newline='') as f:
            csv.writer(f).writerow(['subject','n_windows','acc','mean_rA','mean_rB','ckpt'])

    use_fp16 = (args.amp == 'fp16'); use_bf16 = (args.amp == 'bf16')
    amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)

    for subj in subs:
        ckpt = find_checkpoint(args.models_dir, subj)
        if ckpt is None:
            print(f"[S{subj:02d}] No best.pt found under {args.models_dir} — skipping.")
            continue

        # --- Load data ---
        pair_ok = hasattr(helper, 'subject_eeg_env_pair_ab')
        if pair_ok:
            eeg, env_A, env_B, fs, att_AB = helper.subject_eeg_env_pair_ab(args.preproc_dir, subj)
        else:
            eeg, env_att, fs, att_AB = helper.subject_eeg_env_ab(args.preproc_dir, subj)
            shift = int(round(1.0*fs))
            env_B = np.roll(env_att, shift)
            env_A = env_att

        eeg = eeg.astype(np.float32)  # [T,C]
        T, C = eeg.shape
        lag = int(round(args.lag_ms * fs / 1000.0))
        win = int(round(args.win_sec * fs))
        hop = int(round(args.hop_sec * fs))

        # Make info/pos & model
        info, ch_names, pos = make_biosemi64_info(n_ch=C, sfreq=float(fs))
        net = build_model(
            n_ch=C, pos=pos, ckpt_path=str(ckpt), device=args.device
        )

        # Normalize using first trainstat_pct of time
        # For AAD-by-correlation we only need EEG scaling (the model consumes EEG),
        # but we'll also z-score env_A/env_B for fair r-comparison.
        Xz, _dummy = normalize_series(eeg, env_A, pct=args.trainstat_pct)
        # Independently normalize candidates (keeps *relative* correlation fair)
        _, A_z = normalize_series(env_A.reshape(-1,1), env_A, pct=args.trainstat_pct)
        _, B_z = normalize_series(env_B.reshape(-1,1), env_B, pct=args.trainstat_pct)
        A_z = A_z.astype(np.float32)
        B_z = B_z.astype(np.float32)

        # Window spans are defined on EEG (accounting for lag so y matches)
        spans = window_indices(T, win, hop, start=lag)

        # Build EEG window tensors for inference
        eeg_windows = [ Xz[a:b, :] for (a,b) in spans ]

        # Run model on windows
        yhat_list = batch_predict_windows(net, eeg_windows, args.device, bt_chunk=args.bt_chunk, amp_dtype=amp_dtype)

        # Per window: compute rA, rB, choose prediction, derive true label
        per_rows = []
        correct = 0
        rA_all, rB_all = [], []
        for (a,b), yhat in zip(spans, yhat_list):
            # predicted envelope window
            yw = yhat.astype(np.float32)

            # candidate env windows, aligned with lag (env spans are [a-lag : b-lag])
            ea = A_z[(a - lag):(b - lag)]
            eb = B_z[(a - lag):(b - lag)]

            # length guard (edge windows near start)
            L = min(len(yw), len(ea), len(eb))
            ywL, eaL, ebL = yw[:L], ea[:L], eb[:L]

            rA = pearsonr_1d(ywL, eaL)
            rB = pearsonr_1d(ywL, ebL)
            rA_all.append(rA); rB_all.append(rB)

            pred = 'A' if (rA > rB or (rA == rB and args.prefer_on_tie == 'A')) else 'B'

            # True label in the window: majority of att_AB samples
            att_slice = att_AB[(a - lag):(b - lag)]
            true_lab = majority_label(att_slice)

            ok = int(pred == true_lab)
            correct += ok

            per_rows.append([a, b, float(rA), float(rB), pred, true_lab, ok])

        nW = len(per_rows)
        acc = correct / max(1, nW)
        mean_rA = float(np.mean(rA_all)) if rA_all else 0.0
        mean_rB = float(np.mean(rB_all)) if rB_all else 0.0

        # Write outputs
        sdir = Path(args.outdir)/f"S{subj}"
        ensure_dir(sdir)
        with open(sdir/'aad_windows.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['start_idx','end_idx','rA','rB','pred','true','correct'])
            w.writerows(per_rows)

        with open(sdir/'aad_summary.json', 'w') as f:
            json.dump({
                'subject': subj,
                'n_windows': nW,
                'acc': acc,
                'mean_rA': mean_rA,
                'mean_rB': mean_rB,
                'ckpt': str(ckpt),
                'win_sec': args.win_sec,
                'hop_sec': args.hop_sec,
                'lag_ms': args.lag_ms
            }, f, indent=2)

        with open(summary_csv, 'a', newline='') as f:
            csv.writer(f).writerow([subj, nW, f"{acc:.6f}", f"{mean_rA:.6f}", f"{mean_rB:.6f}", str(ckpt)])

        print(f"[S{subj:02d}] AAD-by-correlation → acc={acc:.4f}  (nW={nW})  ckpt={ckpt}")

        # Optional quick plot
        if args.plots:
            try:
                import matplotlib.pyplot as plt
                rA_arr = np.array(rA_all); rB_arr = np.array(rB_all)
                plt.figure(figsize=(7,3))
                plt.plot(rA_arr, label='r(ê, A)')
                plt.plot(rB_arr, label='r(ê, B)')
                plt.title(f"S{subj} window-wise correlations")
                plt.xlabel("Window #"); plt.ylabel("Pearson r")
                plt.legend(); plt.tight_layout()
                plt.savefig(sdir/'aad_correlations.png', dpi=150); plt.close()
            except Exception as e:
                print(f"[S{subj:02d}] Plot skipped: {e}")

    print(f"\nWrote summary to: {summary_csv}")

if __name__ == '__main__':
    main()
