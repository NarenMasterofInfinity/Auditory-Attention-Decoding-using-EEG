#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 14:03:42 2025

@author: naren-root
"""

#!/usr/bin/env python3
"""
convert.py — Build windowed NPZ/manifest from preprocessed KUL subjects

This script reads the *preprocessed* subject files created by your
`preprocess_kul.py` (i.e., DATA_preproc/preprocessed_data/S*.mat containing
`preproc_trials`). It then uses your `helper.py` API to:

  1) load a subject's preprocessed trials
  2) (optionally) compute audio envelopes from WAVs — but for preprocessed KUL
     we typically use the envelopes already embedded in the .mat
  3) slice into short decision windows (e.g., 1.0 s with 0.1 s hop)
  4) save compact training-ready arrays per subject:

        EEG     : [N, Tw, C]
        EnvL    : [N, Tw, B]   (B=1 for broadband)
        EnvR    : [N, Tw, B]
        Label   : [N]          ('left' or 'right')
        Meta    : JSON alongside the NPZ (trial index + slice per window)

Outputs:
  - <dst>/<subject>/windows.npz
  - <dst>/<subject>/manifest.json

Example:
  python convert.py \
      --src /home/naren-root/KUL/DATA_preproc/preprocessed_data \
      --dst /home/naren-root/KUL/DATA_preproc/converted \
      --win 1.0 --hop 0.1 \
      --mode mat \
      --broadband

Dependencies:
  pip install numpy scipy tqdm
  (and whatever helper.py requires: mne, soundfile, gammatone if using --mode audio)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from tqdm import tqdm

# We import your helper.py (must be in the same folder or PYTHONPATH)
import helper


def build_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert preprocessed KUL .mat → windowed NPZ/manifest")
    p.add_argument("--src", type=str, required=True,
                   help="Folder with preprocessed subjects: .../DATA_preproc/preprocessed_data")
    p.add_argument("--dst", type=str, required=True,
                   help="Output folder for converted NPZ per subject")
    p.add_argument("--audio-dir", type=str, default=None,
                   help="Folder with WAVs (only used when --mode audio or auto w/o mat envs)")
    p.add_argument("--mode", type=str, default="mat", choices=["mat", "audio", "auto"],
                   help="Where to read envelopes from: mat|audio|auto (default: mat)")
    p.add_argument("--win", type=float, default=1.0,
                   help="Window length in seconds (default: 1.0)")
    p.add_argument("--hop", type=float, default=0.1,
                   help="Hop length in seconds (default: 0.1)")
    p.add_argument("--start", type=float, default=0.0,
                   help="Start offset in seconds per trial (default: 0.0)")
    p.add_argument("--center", action="store_true",
                   help="Use centered windows around positions")
    p.add_argument("--multiband", action="store_true",
                   help="Keep multi-band envelopes (B=n_bands). If not set, uses broadband (B=1)")
    p.add_argument("--lag-ms", type=int, default=None,
                   help="Apply +lag ms envelope lead (e.g., 150). If omitted, uses helper default.")
    p.add_argument("--subjects", type=str, nargs="*", default=None,
                   help="Limit to specific subject basenames (e.g., S01.mat S02.mat ...). Default: all S*.mat")
    p.add_argument("--max-trials", type=int, default=None,
                   help="Limit how many trials to convert per subject (debug)")
    p.add_argument("--split-per-trial", action="store_true",
                   help="Save a separate NPZ per trial instead of a single subject file.")
    return p.parse_args()


def convert_subject(subject_mat: Path,
                    dst_root: Path,
                    audio_dir: str,
                    mode: str,
                    win_s: float,
                    hop_s: float,
                    start_s: float,
                    center: bool,
                    keep_multiband: bool,
                    lag_ms: int | None,
                    max_trials: int | None,
                    split_per_trial: bool) -> None:
    """
    Convert a single preprocessed subject file (S*.mat) into windowed arrays.
    """
    subj_name = subject_mat.stem  # e.g., S01
    out_dir = dst_root / subj_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed trials from the .mat
    trials = helper.load_subject(str(subject_mat))

    # Containers (subject-level)
    all_eeg: List[np.ndarray] = []
    all_envL: List[np.ndarray] = []
    all_envR: List[np.ndarray] = []
    all_lab: List[str] = []
    manifest: Dict[str, Any] = {"subject": subj_name, "windows": [], "fs": None}

    n_trials = len(trials) if max_trials is None else min(max_trials, len(trials))
    for t_idx in tqdm(range(1, n_trials + 1), desc=f"{subj_name}"):
        # get_trial will:
        #   - apply EEG preprocessing (already in S*.mat but helper does its own robust pass)
        #   - read envelopes (mat|audio|auto)
        #   - optionally apply +lag-ms
        eeg, envL, envR, fs, att, meta = helper.get_trial(
            trials, t_idx,
            attended='auto',
            fallback_attend_map=None,
            envelope_mode=mode,
            audio_dir=audio_dir,
            envelope_keep_multiband=keep_multiband,
            apply_lag_ms=lag_ms
        )
        if manifest["fs"] is None:
            manifest["fs"] = fs

        # Iterate windows and collect
        it = helper.iter_windows(
            eeg, envL, envR, fs,
            win_s=win_s, hop_s=hop_s, start_s=start_s, center=center
        )

        trial_eegs, trial_envLs, trial_envRs, trial_labs, trial_meta = [], [], [], [], []
        wcount = 0
        for sl, eeg_w, envL_w, envR_w in it:
            wcount += 1
            trial_eegs.append(eeg_w)
            trial_envLs.append(envL_w)
            trial_envRs.append(envR_w)
            trial_labs.append(att)
            trial_meta.append({
                "trial_index": meta.get("trial_index", t_idx),
                "slice": [sl.start, sl.stop],
                "attended": att,
                "stimuli": meta.get("stimuli"),
                "repetition": bool(meta.get("repetition", False)),
                "apply_lag_ms": meta.get("apply_lag_ms"),
                "multiband": bool(meta.get("multiband", keep_multiband)),
            })

        if split_per_trial:
            if wcount == 0:
                continue
            EEG = np.stack(trial_eegs, axis=0)
            ENVL = np.stack(trial_envLs, axis=0)
            ENVR = np.stack(trial_envRs, axis=0)
            LAB = np.array(trial_labs, dtype="U5")  # 'left'/'right'
            npz_path = out_dir / f"trial{t_idx:02d}_windows.npz"
            np.savez_compressed(npz_path, EEG=EEG, EnvL=ENVL, EnvR=ENVR, Label=LAB)
            with open(out_dir / f"trial{t_idx:02d}_manifest.json", "w") as f:
                json.dump({"subject": subj_name, "fs": fs, "windows": trial_meta}, f, indent=2)
        else:
            all_eeg.extend(trial_eegs)
            all_envL.extend(trial_envLs)
            all_envR.extend(trial_envRs)
            all_lab.extend(trial_labs)
            manifest["windows"].extend(trial_meta)

    if not split_per_trial and len(all_eeg):
        EEG = np.stack(all_eeg, axis=0)   # [N, Tw, C]
        ENVL = np.stack(all_envL, axis=0) # [N, Tw, B]
        ENVR = np.stack(all_envR, axis=0) # [N, Tw, B]
        LAB = np.array(all_lab, dtype="U5")
        np.savez_compressed(out_dir / "windows.npz", EEG=EEG, EnvL=ENVL, EnvR=ENVR, Label=LAB)
        with open(out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)


def main():
    args = build_argparser()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    if args.subjects:
        mats = [src / s for s in args.subjects]
    else:
        mats = sorted(src.glob("S*.mat"))

    if not mats:
        raise SystemExit(f"No subjects found under: {src}")

    for mat_path in mats:
        convert_subject(
            subject_mat=mat_path,
            dst_root=dst,
            audio_dir=(args.audio_dir if args.audio_dir else None),
            mode=args.mode,
            win_s=args.win,
            hop_s=args.hop,
            start_s=args.start,
            center=args.center,
            keep_multiband=args.multiband,
            lag_ms=args.lag_ms,
            max_trials=args.max_trials,
            split_per_trial=args.split_per_trial
        )

    print(f"\n✅ Done. Converted subjects are in: {dst}")


if __name__ == "__main__":
    main()
