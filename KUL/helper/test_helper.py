# test_helper.py
# Smoke tests for helper.py on KUL AAD preprocessed data
# Dataset root (contains: preprocessed_data/SXX.mat):
DATASET_DIR = "/home/naren-root/KUL/DATA_preproc"

import os
import glob
import argparse
import numpy as np

# import your helper module (must be in PYTHONPATH or same folder)
import helper


# -------------------------- utils -------------------------- #

def pick_subject_path(dataset_dir: str, subj: str | None) -> str:
    preproc_dir = DATASET_DIR
    if not os.path.isdir(preproc_dir):
        raise FileNotFoundError(f"preprocessed_data/ not found under {dataset_dir}")
    if subj:
        path = os.path.join(preproc_dir, f"{subj}.mat")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Requested subject not found: {path}")
        return path
    cand = sorted(glob.glob(os.path.join(preproc_dir, "S*.mat")))
    if not cand:
        raise FileNotFoundError(f"No SXX.mat files found in {preproc_dir}")
    return cand[0]


def sanity_shapes(eeg, envL, envR, fs, att, meta, subj_name: str, trial_idx: int):
    print("\n=== Trial summary ===")
    print(f"Subject: {subj_name} | Trial: {trial_idx}")
    print(f"SampleRate: {fs} Hz")
    print(f"EEG shape:     {eeg.shape}  [T, C]")
    print(f"Env Left:      {envL.shape} [T, B]")
    print(f"Env Right:     {envR.shape} [T, B]")
    print(f"Attended ear:  {att}")
    print(f"Stimuli:       {meta.get('stimuli')}")
    print(f"Repetition:    {meta.get('repetition', False)}")
    T = eeg.shape[0]
    assert envL.shape[0] == T and envR.shape[0] == T, "EEG/envelopes not time-aligned"
    assert eeg.ndim == 2 and envL.ndim == 2 and envR.ndim == 2, "Unexpected dims"
    assert isinstance(fs, (int, float)), "fs must be numeric"
    assert att in ("left", "right"), "Attended label missing/inference failed"


def maybe_save_two_panel(subj_path, trial_idx, fs, eeg, bb_env, sl, out_dir="test_plots"):
    """
    Save a simple two-panel plot for the provided window slice:
      TOP: EEG (up to 8 channels, z-scored, stacked)
      BOTTOM: broadband attended envelope (z-scored)
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib not available, skipping plot. ({e})")
        return
    os.makedirs(out_dir, exist_ok=True)
    t = np.arange(sl.start, sl.stop) / fs
    X = eeg[sl, :]
    # z-score per channel for visibility
    Xz = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    envz = (bb_env[sl] - bb_env[sl].mean()) / (bb_env[sl].std() + 1e-8)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    off = 0.0
    for c in range(min(8, Xz.shape[1])):  # show up to 8 channels
        ax1.plot(t, Xz[:, c] + off, lw=1.0)
        ax1.text(t[0], off, f"C{c+1}", va='bottom', ha='left', fontsize=8, alpha=0.7)
        off += 2.5
    ax1.set_ylabel("EEG (z) stacked"); ax1.grid(True, alpha=0.2)

    ax2.plot(t, envz, lw=2.0)
    ax2.set_ylabel("Envelope (z)"); ax2.set_xlabel("Time (s)"); ax2.grid(True, alpha=0.2)

    base = os.path.splitext(os.path.basename(subj_path))[0]
    out_png = os.path.join(out_dir, f"{base}_trial{trial_idx}_win{sl.start}-{sl.stop}.png")
    fig.suptitle(f"{base} trial {trial_idx}  |  window {sl.start}:{sl.stop}  |  fs={fs} Hz")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] saved {out_png}")


# -------------------------- tests -------------------------- #

def test_low_level_iter(eeg, envL, envR, fs, win_s: float, hop_s: float | None):
    print("\n=== iter_windows (low-level, multi-band) ===")
    n_win = 0
    first_shapes = None
    for sl, X, L, R in helper.iter_windows(eeg, envL, envR, fs, win_s=win_s, hop_s=hop_s, start_s=0.0, center=False):
        if n_win == 0:
            first_shapes = (sl, X.shape, L.shape, R.shape)
        n_win += 1
    assert n_win > 0, "No windows generated — check win_s/hop_s vs trial length."
    sl, sx, sl_, sr_ = first_shapes
    print(f"First window slice: {sl.start}:{sl.stop}")
    print(f"EEG window shape:   {sx}  [Tw, C]")
    print(f"Env L/R shapes:     {sl_} / {sr_}  [Tw, B]")
    print(f"Total windows:      {n_win}")


def test_multiband_windows_api(trials, idx, win_s: float, hop_s: float | None):
    print("\n=== windows_for_trial_multiband (high-level) ===")
    n = 0
    first = None
    for X, L, R, label, meta in helper.windows_for_trial_multiband(
        trials, idx, attended='auto', fallback_attend_map=None,
        win_s=win_s, hop_s=hop_s, start_s=0.0
    ):
        if first is None:
            first = (X.shape, L.shape, R.shape, label, meta)
        n += 1
    assert n > 0, "No windows yielded from windows_for_trial_multiband."
    sx, sl, sr, lab, meta = first
    print(f"First EEG window: {sx}  [Tw, C]")
    print(f"First L/R env:    {sl} / {sr}  [Tw, B]")
    print(f"First label:      {lab}")
    print(f"Total windows:    {n}")


def test_broadband_paths(envL, envR, weights, att, eeg, fs, save_plot: bool, subj_path: str, trial_idx: int, win_s: float):
    print("\n=== Broadband conversion (T,1) ===")
    bbL, bbR = helper.to_broadband(envL, envR, weights)
    print(f"Broadband L/R shapes: {bbL.shape}, {bbR.shape}  [T]")

    # Build a target (T,1)
    target = (bbL if att == "left" else bbR)[:, None]
    print(f"Target shape (T,1): {target.shape}")
    assert target.ndim == 2 and target.shape[1] == 1, "Target is not (T,1)"

    # Optional: save a quick two-panel plot of the first window
    if save_plot:
        W = int(round(win_s * fs))
        sl = slice(0, min(W, eeg.shape[0]))
        bb_env = bbL if att == "left" else bbR
        maybe_save_two_panel(subj_path, trial_idx, fs, eeg, bb_env, sl, out_dir="test_plots")


# -------------------------- main -------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default=DATASET_DIR, help="Root dir that contains preprocessed_data/")
    ap.add_argument("--subject", default=None, help="Pick specific subject id (e.g., S01). Default: auto-pick first.")
    ap.add_argument("--trial", type=int, default=1, help="1-based trial index to test")
    ap.add_argument("--win_s", type=float, default=5.0, help="Window length seconds (e.g., 5 or 10)")
    ap.add_argument("--hop_s", type=float, default=None, help="Hop seconds (default = win_s)")
    ap.add_argument("--save_plot", action="store_true", help="Save a demo two-panel plot for the first window")
    args = ap.parse_args()

    subj_path = pick_subject_path(args.dataset_dir, args.subject)
    subj_name = os.path.splitext(os.path.basename(subj_path))[0]
    print(f"Using subject file: {subj_path}")

    # Load trials
    trials = helper.load_subject(subj_path)
    print(f"Trials loaded: {len(trials)}")

    # Pull a trial (auto label; heuristic inference if missing)
    eeg, envL, envR, fs, att, meta = helper.get_trial(
        trials, idx=args.trial, attended='auto', fallback_attend_map=None
    )
    sanity_shapes(eeg, envL, envR, fs, att, meta, subj_name, args.trial)

    # Low-level multiband iterator
    test_low_level_iter(eeg, envL, envR, fs, args.win_s, args.hop_s)

    # High-level multiband API
    test_multiband_windows_api(trials, args.trial, args.win_s, args.hop_s)

    # Broadband path (T,1) target + optional plot
    test_broadband_paths(envL, envR, meta.get("subband_weights"), att, eeg, fs, args.save_plot, subj_path, args.trial, args.win_s)

    print("\nAll tests passed ✅")


if __name__ == "__main__":
    main()
