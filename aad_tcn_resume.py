#!/usr/bin/env python3
"""
resume_aad_graph_xattn.py

Continue training AAD (Graph + Cross-Attn) for a single subject from the previously saved best_model.pt,
then evaluate on the test split and regenerate artifacts (plots, adjacency heatmaps, topomaps, probability timeline).

Assumptions:
- You already trained with train_aad_graph_xattn.py, which created AAD/outputs_aad/S<subject>/best_model.pt
- This script reuses the same data split recipe (seed=2000+subject) and the same dataset windowing.
- If hyperparameters differ from the first run, you can pass them here (they default to the train script’s defaults).

Outputs (inside AAD/outputs_aad/S<subject>):
- best_model_resume.pt  (new best during resumed run)
- training_curve_resume.png, training_acc_resume.png
- A0_heatmap.png (if available), A_heatmap_resume.png, A_edges_sensors_resume.png, A_edges_delta_resume.png
- in_strength_topomap_resume.png
- vis/test_probs_resume.png
- Appends/updates summary_aad_resume.csv at outdir root
"""

import os, math, time, argparse, csv, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')
import mne

# ---------- import your original components ----------
from aad_tcn import (
    AADModel, AADDataset, evaluate_cls,
    ensure_dir, zscore_train, window_indices, maj_vote_label,
    make_biosemi64_info, split_indices, subset_dataset,
    make_warmup_cosine_scheduler,
    plot_training_curves, plot_adjacency_heatmap, plot_sensor_edges,
    plot_sensor_edges_delta, plot_in_strength_topomap, plot_test_probabilities,
)
import helper


def load_subject_data(preproc_dir, subj_id, win_sec, hop_sec):
    """
    Load EEG + envelopes using the same preference/fallback as the train script.
    Returns: ds, fs, n_ch, info, pos, use_fallback
    """
    use_fallback = False
    try:
        eeg, envA, envB, fs, attAB = helper.subject_eeg_env_ab_aad(preproc_dir, subj_id)
    except Exception:
        eeg, env_att, fs, attAB = helper.subject_eeg_env_ab(preproc_dir, subj_id)
        shift = int(round(1.0 * fs))
        envB = np.roll(env_att, shift)
        envA = env_att
        use_fallback = True
        print("WARNING: Using fallback envB (circularly shifted). Replace with real env_B for proper AAD.")

    X = eeg.astype(np.float32)
    X, _, _ = zscore_train(X)
    n_ch = X.shape[1]
    info, ch_names, pos = make_biosemi64_info(n_ch=n_ch, sfreq=fs)

    ds = AADDataset(X, envA, envB, attAB, fs, win_sec, hop_sec)
    return ds, fs, n_ch, info, pos, use_fallback


def build_loaders(ds, subj_id, batch, workers, prefetch):
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
    return train_ld, val_ld, test_ld


def train_resume_one_subject(args):
    sx = ensure_dir(os.path.join(args.outdir, f"S{args.subject}"))
    ckpt_path = args.resume_from if args.resume_from else os.path.join(sx, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Data
    ds, fs, n_ch, info, pos, use_fallback = load_subject_data(args.preproc_dir, args.subject, args.win_sec, args.hop_sec)
    train_ld, val_ld, test_ld = build_loaders(ds, args.subject, args.batch, args.workers, args.prefetch)

    # Model
    device = args.device
    model = AADModel(
        n_ch=n_ch, pos=pos, d_model=128, d_audio=64,
        L=args.blocks_graph, k=args.k,
        heads_graph=args.heads_graph, heads_xattn=args.heads_xattn,
        dropout=0.1
    ).to(device)

    # Load weights
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)

    if args.compile:
        try:
            model = torch.compile(model, mode='max-autotune')
        except Exception as e:
            print("torch.compile failed (continuing):", str(e))

    # Optim, sched, AMP
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, math.ceil(len(train_ld) / max(1, args.accum)))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_pct * total_steps)
    sched = make_warmup_cosine_scheduler(opt, total_steps, warmup_steps=warmup_steps, final_lr_pct=args.final_lr_pct)

    use_fp16 = (args.amp == 'fp16'); use_bf16 = (args.amp == 'bf16')
    amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    # Early stop over val loss (fresh)
    best_val = float('inf'); best_state = None; wait = 0
    hist = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def _to(x): return x.to(device, non_blocking=True)

    # Resume training
    print(f"Resuming training for {args.epochs} epochs…")
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss_sum, tr_acc_sum, tr_n = 0.0, 0.0, 0
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        for step, (xb, a, b, yb) in enumerate(train_ld, 1):
            xb = _to(xb); a = _to(a); b = _to(b); yb = _to(yb)
            with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                logits, _ = model(xb, a, b, bt_chunk=args.bt_chunk)
                loss = F.cross_entropy(logits, yb) / max(1, args.accum)

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step % max(1, args.accum)) == 0:
                if use_fp16: scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_fp16:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()

            pred = logits.argmax(-1)
            tr_acc_sum += (pred == yb).float().sum().item()
            tr_loss_sum += loss.item() * xb.size(0) * max(1, args.accum)
            tr_n += xb.size(0)

        tr_loss = tr_loss_sum / tr_n
        tr_acc = tr_acc_sum / tr_n

        # Validate
        model.eval()
        va_loss_sum, va_acc_sum, va_n = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, a, b, yb in val_ld:
                xb = _to(xb); a = _to(a); b = _to(b); yb = _to(yb)
                with torch.amp.autocast('cuda', enabled=(amp_dtype is not None), dtype=amp_dtype):
                    logits, _ = model(xb, a, b, bt_chunk=args.bt_chunk)
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

        # Track new best (val loss)
        improved = (best_val - va_loss) > float(args.min_delta)
        if improved:
            best_val = va_loss; wait = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(sx, 'best_model_resume.pt'))
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {ep} (best val {best_val:.4f})")
                break

    # Restore best from this resume session (if any), otherwise keep latest
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # Final metrics
    use_fp16 = (args.amp == 'fp16'); use_bf16 = (args.amp == 'bf16')
    amp_dtype = torch.float16 if use_fp16 else (torch.bfloat16 if use_bf16 else None)
    val_acc, val_auc, _, _ = evaluate_cls(model, val_ld, device, args.bt_chunk, amp_dtype)
    test_acc, test_auc, te_probs, te_labs = evaluate_cls(model, test_ld, device, args.bt_chunk, amp_dtype)
    print(f"[RESUME] Final: Val acc={val_acc:.3f} AUC={val_auc:.3f} | Test acc={test_acc:.3f} AUC={test_auc:.3f}")

    # Artifacts with _resume suffixes
    plot_training_curves(hist, sx)
    # rename the curve files with _resume suffix to avoid overwriting originals
    import shutil, os.path as osp
    for old, new in [
        ('training_curve.png', 'training_curve_resume.png'),
        ('training_acc.png', 'training_acc_resume.png'),
    ]:
        o, n = osp.join(sx, old), osp.join(sx, new)
        if osp.exists(o): shutil.move(o, n)

    # One batch to visualize adjacency
    model.eval()
    with torch.no_grad():
        try:
            xb, a, b, yb = next(iter(val_ld)) if len(val_ld) > 0 else next(iter(test_ld))
        except StopIteration:
            xb, a, b, yb = next(iter(test_ld))
        xb = xb.to(device); a = a.to(device); b = b.to(device)
        logits, A_t = model(xb, a, b, bt_chunk=args.bt_chunk)
        A = A_t.detach().cpu().numpy()

    plot_adjacency_heatmap(A, sx, name='A_heatmap_resume.png', title='Blended adjacency A (resumed)')
    # initial prior A0 (if present)
    A0 = None
    try:
        A0 = model.eeg_enc.graph.A0.detach().cpu().numpy()
        np = __import__('numpy')
        np.savetxt(os.path.join(sx, 'A0.csv'), A0, delimiter=',')
        plot_adjacency_heatmap(A0, sx, name='A0_heatmap.png', title='Initial prior A0')
    except Exception:
        pass

    info, ch_names, _pos = make_biosemi64_info(n_ch=n_ch, sfreq=fs)
    plot_sensor_edges(A, info, sx, topk=120, name='A_edges_sensors_resume.png')
    plot_sensor_edges_delta(A, A0, info, sx, topk=80)
    plot_in_strength_topomap(A, info, sx)

    # Probability timeline (test)
    vis_dir = ensure_dir(os.path.join(sx, "vis"))
    path_probs = plot_test_probabilities(te_probs, te_labs, vis_dir, fname='test_probs_resume.png')
    print(f"[viz] wrote {path_probs}")

    # Fallback notice (if any), but only add a line to an existing/append file
    if use_fallback:
        with open(os.path.join(sx, 'README_FALLBACK.txt'), 'a') as f:
            f.write("RESUME: env_B was synthesized by circular shift of env_att by 1 s. Replace with real env_B for proper AAD.\n")

    # Append to resume summary CSV
    summ_path = os.path.join(args.outdir, "summary_aad_resume.csv")
    header = ['subject', 'epochs_additional', 'val_acc', 'val_auc', 'test_acc', 'test_auc', 'ckpt_used']
    write_header = not os.path.exists(summ_path)
    with open(summ_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow([args.subject, args.epochs, float(val_acc), float(val_auc), float(test_acc), float(test_auc), ckpt_path])

    print(f"[RESUME] Subject {args.subject}: Val acc={val_acc:.3f} AUC={val_auc:.3f} | Test acc={test_acc:.3f} AUC={test_auc:.3f}")
    print(f"Wrote {summ_path}")
    return val_acc, val_auc, test_acc, test_auc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preproc_dir', type=str, default='/home/naren-root/Dataset/DATA_preproc')
    p.add_argument('--outdir', type=str, default='AAD/outputs_aad')
    p.add_argument('--subject', type=int, required=True)
    p.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint; defaults to S<subj>/best_model.pt')
    p.add_argument('--epochs', type=int, default=30)
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
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--min_delta', type=float, default=5e-4)
    p.add_argument('--warmup_pct', type=float, default=0.05)
    p.add_argument('--final_lr_pct', type=float, default=0.1)
    args = p.parse_args()

    ensure_dir(args.outdir)
    train_resume_one_subject(args)


if __name__ == '__main__':
    main()
