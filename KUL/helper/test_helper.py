#!/usr/bin/env python3
"""
test_helper.py — QA/diagnostic for helper.py (KUL, DTU-style audio + 150 ms lead)

What this script checks:
  1) Loads first subject .mat from DATASET_DIR
  2) Runs helper.get_trial() with:
       - envelope_mode='audio'  (DTU/COCOHA-style gammatone pipeline)
       - envelope_keep_multiband=False (broadband, as in MATLAB co_dimaverage)
       - apply_lag_ms=150 (advance envelopes to compensate neural lag)
  3) Verifies shapes, prints metadata
  4) Iterates a few windows (1 s, hop 0.1 s)
  5) Optionally shows a quick plot of one EEG channel and the two broadband envelopes

Dependencies:
  pip install numpy scipy mne soundfile gammatone matplotlib
"""

from pathlib import Path
import sys
import numpy as np

import helper  # ensure helper.py is in the same folder or on PYTHONPATH

# ----------------------------- CONFIG ----------------------------- #
DATASET_DIR = Path("/home/naren-root/KUL/DATA_preproc")  # folder with S*.mat
AUDIO_DIR    = DATASET_DIR / "stimuli"                     # folder with WAVs referenced by `trial['stimuli']`

TRIAL_IDX = 1       # 1-based
WIN_S     = 1.0     # seconds
HOP_S     = 0.1     # seconds

# DTU-style envelope settings
ENVELOPE_MODE            = "audio"   # 'audio' for DTU/COCOHA path; 'mat' to force .mat envelopes; 'auto' to prefer .mat, else audio
KEEP_MULTIBAND           = False     # False => average to broadband (matches MATLAB co_dimaverage)
APPLY_LAG_MS             = 150       # +150 ms envelope lead (typical DTU value). 0/None to disable.
# ------------------------------------------------------------------ #


def pick_subject_mat(dataset_dir: Path) -> Path:
    """Pick the first subject .mat file (S*.mat)."""
    cand = sorted(dataset_dir.glob("S*.mat"))
    if not cand:
        print(f"[ERROR] No S*.mat files found under {dataset_dir}")
        sys.exit(1)
    return cand[0]


def main():
    if not DATASET_DIR.exists():
        print(f"[ERROR] DATASET_DIR not found: {DATASET_DIR}")
        sys.exit(1)
    if ENVELOPE_MODE in ("audio", "auto") and not AUDIO_DIR.exists():
        print(f"[WARN] AUDIO_DIR not found: {AUDIO_DIR} (audio-mode needs WAVs).")

    mat_path = pick_subject_mat(DATASET_DIR)
    print(f"Subject file: {mat_path.name}")

    trials = helper.load_subject(str(mat_path))
    print(f"Loaded trials: {len(trials)}")

    print(f"\n↪ Running get_trial(trial={TRIAL_IDX}, envelope_mode='{ENVELOPE_MODE}', "
          f"multiband={KEEP_MULTIBAND}, lag={APPLY_LAG_MS} ms)")
    eeg, envL, envR, fs, att, meta = helper.get_trial(
        trials, TRIAL_IDX,
        attended='auto',
        fallback_attend_map=None,
        envelope_mode=ENVELOPE_MODE,
        audio_dir=str(AUDIO_DIR),
        envelope_keep_multiband=KEEP_MULTIBAND,
        apply_lag_ms=APPLY_LAG_MS
    )

    # ---- Basic diagnostics ----
    print("\n=== Trial Diagnostics ===")
    print(f"Fs (EEG & envelopes): {fs:.2f} Hz")
    print(f"EEG shape: {eeg.shape}")       # [T, C]
    print(f"EnvL shape: {envL.shape}")     # [T, B] (B=1 if broadband)
    print(f"EnvR shape: {envR.shape}")     # [T, B]
    print(f"Attended ear (label): {att}")
    print(f"Stimuli (L,R): {meta.get('stimuli')}")
    print(f"Applied lag (ms): {meta.get('apply_lag_ms')}")
    print(f"Multiband: {meta.get('multiband')}")

    # ---- Windowing sanity check ----
    print(f"\nGenerating windows (win={WIN_S:.1f}s, hop={HOP_S:.1f}s) ...")
    n_win = 0
    for sl, eeg_w, envL_w, envR_w in helper.iter_windows(
            eeg, envL, envR, fs, win_s=WIN_S, hop_s=HOP_S):
        n_win += 1
        if n_win <= 3:
            print(f"  Window {n_win:02d}: slice={sl}, EEG {eeg_w.shape}, EnvL {envL_w.shape}, EnvR {envR_w.shape}")
    print(f"Total windows: {n_win}")

    # ---- Multiband iterator (same settings) ----
    print("\nChecking windows_for_trial_multiband(...)")
    it = helper.windows_for_trial_multiband(
        trials, TRIAL_IDX,
        win_s=WIN_S, hop_s=HOP_S,
        envelope_mode=ENVELOPE_MODE,
        audio_dir=str(AUDIO_DIR),
        envelope_keep_multiband=KEEP_MULTIBAND,
        apply_lag_ms=APPLY_LAG_MS
    )
    eeg_w, envL_w, envR_w, label, meta_w = next(it)
    print(f"Sample win — EEG {eeg_w.shape}, EnvL {envL_w.shape}, EnvR {envR_w.shape}, Label: {label}")
    print(f"Meta (win): {{trial_index={meta_w['trial_index']}, slice={meta_w['slice']}, fs={meta_w['fs']}}}")

    # ---- Optional quick visualization ----
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        ax[0].plot(eeg_w[:, 0], lw=0.8)
        ax[0].set_title(f"EEG (Ch1) — Trial {TRIAL_IDX} — label={label}")
        # If broadband, column 0 is the single band
        ax[1].plot(envL_w[:, 0], lw=0.8, label="Env Left (broadband)" if envL_w.shape[1] == 1 else "Env Left (band 1)")
        ax[1].plot(envR_w[:, 0], lw=0.8, label="Env Right (broadband)" if envR_w.shape[1] == 1 else "Env Right (band 1)")
        ax[1].legend(loc="upper right")
        ax[1].set_xlabel("Samples")
        ax[0].set_ylabel("EEG (µV)")
        ax[1].set_ylabel("Envelope (a.u.)")
        fig.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[WARN] Plot skipped: {e}")

    print("\n✅ helper.py DTU-style audio + lag pipeline appears to run OK.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
test_helper.py — QA/diagnostic for helper.py (KUL, DTU-style audio + 150 ms lead)

What this script checks:
  1) Loads first subject .mat from DATASET_DIR
  2) Runs helper.get_trial() with:
       - envelope_mode='audio'  (DTU/COCOHA-style gammatone pipeline)
       - envelope_keep_multiband=False (broadband, as in MATLAB co_dimaverage)
       - apply_lag_ms=150 (advance envelopes to compensate neural lag)
  3) Verifies shapes, prints metadata
  4) Iterates a few windows (1 s, hop 0.1 s)
  5) Optionally shows a quick plot of one EEG channel and the two broadband envelopes

Dependencies:
  pip install numpy scipy mne soundfile gammatone matplotlib
"""

from pathlib import Path
import sys
import numpy as np

import helper  # ensure helper.py is in the same folder or on PYTHONPATH

# ----------------------------- CONFIG ----------------------------- #
DATASET_DIR = Path("/home/naren-root/KUL/DATA_preproc")  # folder with S*.mat
AUDIO_DIR    = DATASET_DIR / "audio"                     # folder with WAVs referenced by `trial['stimuli']`

TRIAL_IDX = 1       # 1-based
WIN_S     = 1.0     # seconds
HOP_S     = 0.1     # seconds

# DTU-style envelope settings
ENVELOPE_MODE            = "audio"   # 'audio' for DTU/COCOHA path; 'mat' to force .mat envelopes; 'auto' to prefer .mat, else audio
KEEP_MULTIBAND           = False     # False => average to broadband (matches MATLAB co_dimaverage)
APPLY_LAG_MS             = 150       # +150 ms envelope lead (typical DTU value). 0/None to disable.
# ------------------------------------------------------------------ #


def pick_subject_mat(dataset_dir: Path) -> Path:
    """Pick the first subject .mat file (S*.mat)."""
    cand = sorted(dataset_dir.glob("S*.mat"))
    if not cand:
        print(f"[ERROR] No S*.mat files found under {dataset_dir}")
        sys.exit(1)
    return cand[0]


def main():
    if not DATASET_DIR.exists():
        print(f"[ERROR] DATASET_DIR not found: {DATASET_DIR}")
        sys.exit(1)
    if ENVELOPE_MODE in ("audio", "auto") and not AUDIO_DIR.exists():
        print(f"[WARN] AUDIO_DIR not found: {AUDIO_DIR} (audio-mode needs WAVs).")

    mat_path = pick_subject_mat(DATASET_DIR)
    print(f"Subject file: {mat_path.name}")

    trials = helper.load_subject(str(mat_path))
    print(f"Loaded trials: {len(trials)}")

    print(f"\n↪ Running get_trial(trial={TRIAL_IDX}, envelope_mode='{ENVELOPE_MODE}', "
          f"multiband={KEEP_MULTIBAND}, lag={APPLY_LAG_MS} ms)")
    eeg, envL, envR, fs, att, meta = helper.get_trial(
        trials, TRIAL_IDX,
        attended='auto',
        fallback_attend_map=None,
        envelope_mode=ENVELOPE_MODE,
        audio_dir=str(AUDIO_DIR),
        envelope_keep_multiband=KEEP_MULTIBAND,
        apply_lag_ms=APPLY_LAG_MS
    )

    # ---- Basic diagnostics ----
    print("\n=== Trial Diagnostics ===")
    print(f"Fs (EEG & envelopes): {fs:.2f} Hz")
    print(f"EEG shape: {eeg.shape}")       # [T, C]
    print(f"EnvL shape: {envL.shape}")     # [T, B] (B=1 if broadband)
    print(f"EnvR shape: {envR.shape}")     # [T, B]
    print(f"Attended ear (label): {att}")
    print(f"Stimuli (L,R): {meta.get('stimuli')}")
    print(f"Applied lag (ms): {meta.get('apply_lag_ms')}")
    print(f"Multiband: {meta.get('multiband')}")

    # ---- Windowing sanity check ----
    print(f"\nGenerating windows (win={WIN_S:.1f}s, hop={HOP_S:.1f}s) ...")
    n_win = 0
    for sl, eeg_w, envL_w, envR_w in helper.iter_windows(
            eeg, envL, envR, fs, win_s=WIN_S, hop_s=HOP_S):
        n_win += 1
        if n_win <= 3:
            print(f"  Window {n_win:02d}: slice={sl}, EEG {eeg_w.shape}, EnvL {envL_w.shape}, EnvR {envR_w.shape}")
    print(f"Total windows: {n_win}")

    # ---- Multiband iterator (same settings) ----
    print("\nChecking windows_for_trial_multiband(...)")
    it = helper.windows_for_trial_multiband(
        trials, TRIAL_IDX,
        win_s=WIN_S, hop_s=HOP_S,
        envelope_mode=ENVELOPE_MODE,
        audio_dir=str(AUDIO_DIR),
        envelope_keep_multiband=KEEP_MULTIBAND,
        apply_lag_ms=APPLY_LAG_MS
    )
    eeg_w, envL_w, envR_w, label, meta_w = next(it)
    print(f"Sample win — EEG {eeg_w.shape}, EnvL {envL_w.shape}, EnvR {envR_w.shape}, Label: {label}")
    print(f"Meta (win): {{trial_index={meta_w['trial_index']}, slice={meta_w['slice']}, fs={meta_w['fs']}}}")

    # ---- Optional quick visualization ----
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        ax[0].plot(eeg_w[:, 0], lw=0.8)
        ax[0].set_title(f"EEG (Ch1) — Trial {TRIAL_IDX} — label={label}")
        # If broadband, column 0 is the single band
        ax[1].plot(envL_w[:, 0], lw=0.8, label="Env Left (broadband)" if envL_w.shape[1] == 1 else "Env Left (band 1)")
        ax[1].plot(envR_w[:, 0], lw=0.8, label="Env Right (broadband)" if envR_w.shape[1] == 1 else "Env Right (band 1)")
        ax[1].legend(loc="upper right")
        ax[1].set_xlabel("Samples")
        ax[0].set_ylabel("EEG (µV)")
        ax[1].set_ylabel("Envelope (a.u.)")
        fig.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[WARN] Plot skipped: {e}")

    print("\n✅ helper.py DTU-style audio + lag pipeline appears to run OK.")


if __name__ == "__main__":
    main()
