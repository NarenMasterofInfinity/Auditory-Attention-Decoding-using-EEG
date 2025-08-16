# Helper Module — Reference

This guide explains all functions in **`helper.py`** for working with the DTU auditory attention dataset:

* Load and normalize `.mat` files
* Build attention masks (A/B) robustly
* Extract audio envelopes (Gammatone + Hilbert, with safe fallback)
* Prepare EEG + attended envelope for modeling
* Plot EEG + audio/envelopes
* Save figures and logs automatically

> **Folders used**
>
> * **`figures/`** — all PNGs saved here
> * **`Logs/`** — log file at `Logs/helper.log`

---

## Requirements

* Python 3.9+
* `numpy`, `scipy`, `matplotlib`
* `gammatone` (for true gammatone filterbank). If missing, code falls back to Hilbert + low-pass.

```bash
pip install numpy scipy matplotlib
pip install gammatone  
```

---

## Quick Start

```python
import helper

PREPROC_DIR = "the path to the preproc folder"
subj_id = 1 # can be between 1 and 18

# 1) Load subject & plot a window
D = helper.load_subject(PREPROC_DIR, subj_id, drop_exg=True)
helper.plot_window(PREPROC_DIR, subj_id, t_start=10, t_end=20, max_channels=16)

# 2) Get EEG (T×C), attended envelope (T,), fs, and 'A'/'B' per sample
eeg, env_att, fs, att_AB = helper.subject_eeg_env_ab(PREPROC_DIR, subj_id)

# 3) Visualize EEG + envelope
helper.plot_eeg_and_audio(eeg, env_att, fs, t_start=5, t_end=15, max_channels=16)

# 4) Check attention per trial
helper.check_attention_distribution(PREPROC_DIR, subj_id)
```

---

## API Reference

### `_unique_png(prefix)`

Return a unique PNG path in `figures/` using the given prefix (adds `_1`, `_2`, … if needed).

**Returns:** `Path` that does not yet exist.

### `_unwrap_mat_struct(obj)`

Recursively convert MATLAB `mat_struct` / object arrays into plain Python dicts/arrays.

**Use when:** Cleaning the output of `scipy.io.loadmat`.

### `load_mat_for_subject(preproc_dir, subj_id)`

Find and load `S{subj_id}_data_preproc.mat` (also handles zero-padded `S01_...`).

**Inputs**

* `preproc_dir`: directory containing preprocessed `.mat` files
* `subj_id`: integer id (1, 2, …)

**Returns:** `dict` with at least key `"data"` (cleaned).
**Raises:** `FileNotFoundError` if the file is missing.

### `_ensure_ch_t(mat2)`

Coerce a 2-D EEG array to shape `(n_ch, n_t)`; transpose if time is first.

**Returns:** float array `(n_ch, n_t)`.
**Raises:** `ValueError` if it cannot be made 2-D.

### `_extract_event_container(event_all)`

Pull an inner event container if events were stored inside a nested dict (tries keys like `'eeg'`, `'att'`, `'wavA'`, `'wavB'`, etc.).

### `_explode_trials_from_group(group)`

Split a single group with **3-D EEG** (channels × time × trials *in any order*) into a list of **per-trial** dicts. Also slices `wavA`, `wavB`, and `event` per trial and resamples audio to each trial length.

**Returns:** list of trial dicts with keys: `eeg`, `wavA`, `wavB`, `event`, `fsample`, `chan` (if available).

### `_iter_trials(mat)`

Return a list of per-trial dicts from a loaded `.mat`:

* handles Python lists, object arrays, and single dicts (will explode 3-D EEG to trials).

**Returns:** list of trial dicts.
**Raises:** `KeyError` if `data` missing; `TypeError` for unsupported shapes.

### `_lp_filter(y, fs, cutoff_hz, order=4)`

Simple Butterworth low-pass with `filtfilt`. Returns input if `cutoff_hz <= 0`.

### `_resample_to(y, fs_in, fs_out)`

Resample `y` from `fs_in` to `fs_out` using polyphase (`resample_poly`) if needed.

**Returns:** `(y_resampled, fs_out)` or `(y, fs_in)` if unchanged.

### `gammatone_hilbert_envelope(audio, fs, *, num_bands=32, fmin=50.0, fmax=None, compress="pow", compress_exp=0.6, aggregate="sum", lowpass_hz=8.0, target_fs=None, normalize="unit", return_bands=False)`

High-quality envelope extractor.

* If `fs ≥ 1000` and `gammatone` is available:
  gammatone filterbank → Hilbert magnitude per band → optional compression (power/log) → aggregate (sum/mean/RMS) → low-pass (e.g., 8 Hz) → optional resample to `target_fs` → normalization (`unit`/`zscore`).
* Otherwise (low rate or no gammatone):
  Hilbert magnitude → low-pass → optional resample → normalization.

**Inputs**

* `audio`: 1-D waveform
* `fs`: sample rate of `audio` (Hz)
* `target_fs`: set to EEG rate (e.g., 64) to align with EEG

**Returns:** `(env, meta)`

* `env`: 1-D envelope (possibly resampled)
* `meta["fs"]`: output sample rate
* `meta["cf"]`: center frequencies (gammatone path)
* `meta["band_envs"]`: per-band envelopes if `return_bands=True`

**Notes:** For EEG-aligned envelopes, set `target_fs` to EEG `fs`.

### `get_audio_signal(mat, stream="A")`

Concatenate `wavA` or `wavB` across all trials in a **mat dict** (from `load_mat_for_subject`). If trials report different `fs`, resample to the first valid `fs` and log it.

**Returns:** `(audio, fs)` where `audio` is 1-D.

### `plot_audio(audio, fs, t_start, t_end, also_envelope=True, num_bands=32, fmin=100.0, out_path=None, title=None, env_norm="unit", audio_norm="none", lp_hz=8.0)`

Plot raw audio (and optional envelope overlay) in a time window. Saves PNG to `figures/` and returns the envelope if computed (otherwise empty array).

**Good for:** sanity checks of segments.

### `_build_att_mask(n_t, tr)`

**Robust attention parser** for one trial. Produces a per-sample mask with:

* `1` → attended A
* `2` → attended B
* `0` → unknown (only if truly missing)

Understands:

* full-length vectors under keys like `attmask`, `att`, `attention`, … (numeric or `'A'/'B'`)
* change-point events as `(N,2)` or `(2,N)` (samples, values) — auto-handles **1-based** indices, fills head/tail
* scalar per-trial labels inside `trialinfo` (`'A'/'B'` or `1/2`)

### `load_subject(preproc_dir, subj_id, drop_exg=True)`

High-level loader that concatenates trials into continuous streams.

**Returns** a dict with:

* `fs`: sampling rate (usually the EEG rate, e.g., 64 Hz)
* `ch_names`: channel names (EXG removed if `drop_exg=True`)
* `eeg`: `(n_ch, T)`
* `envA`, `envB`: concatenated audio/envelope-like series `(T,)`
* `attmask`: `(T,)` attention samples (1=A, 2=B, 0=unknown)
* `lengths`: list of per-trial lengths (`n_t` each)
* `has_two_streams`, `path`, `subj_id`

**Notes**

* Explodes 3-D EEG groups into trials first.
* Resamples `wavA/B` inside each trial to match that trial’s EEG length.

### `_clean_to_64(eeg, ch_names=None)`

If EEG has **> 64** channels (e.g., 66 with EXG/EOG), drop likely aux channels to keep 64:

* drop names starting with `EXG/MISC/AUX/STATUS/TRIG` or containing `EOG/ECG/EMG` (if names known), else keep the first 64.

**Returns:** `(eeg64, ch64, idx64)`.

### `subject_eeg_env_ab(preproc_dir, subj_id, num_bands=32, fmin=50.0, fmax=None, lowpass_hz=8.0, normalize="unit")`

Prepare **time-major** EEG and the **attended envelope** for modeling, with per-sample `'A'/'B'` labels.

**Steps**

1. `load_subject(...)`
2. Trim to 64 channels via `_clean_to_64(...)` if needed
3. Extract envelopes for `wavA`, `wavB` using `gammatone_hilbert_envelope` (`target_fs=fs`)
4. Switch sample-wise using `attmask` (1→A, 2→B; 0→0)

**Returns**

* `eeg_TxC`: `(T, C)` (time first)
* `env_att`: `(T,)` attended envelope
* `fs`: EEG/envelope sample rate
* `att_AB`: `(T,)` array of `'A'` / `'B'` (and `'U'` if unknown exists)

### `plot_window(preproc_dir, subj_id, t_start, t_end, max_channels=None, figsize=(14, 7))`

Two-panel figure:

* **Top:** stacked, z-scored EEG (optionally limit to `max_channels`)
* **Bottom:** **Audio 1 (A)** and **Audio 2 (B)** envelopes with attention-aware styling: attended=solid, unattended=dashed. Trial boundaries marked.

**Saves:** PNG into `figures/`.

### `eeg_and_attended_envelope(mat, which_env="gammatone", num_bands=32, fmin=100.0)`

Work directly on a **mat dict**: for each trial, make envelopes for A/B, build the attention mask, and pick the attended samples; then concatenate.

**Returns:** `(eeg_all, env_all)` where `eeg_all` is `(T, C)` and `env_all` is `(T,)`.

### `check_attention_distribution(preproc_dir, subj_id)`

Prints per-subject and per-trial attention stats:

* percentage of A, B, and unknown
* trial start/end times and durations

**Use for:** validating parsing and concatenation.

### `plot_eeg_and_audio(eeg_TxC, audio, fs, t_start, t_end, num_bands=32, fs_audio=None, channel_names=None, max_channels=None, env_lowpass_hz=8.0, title=None, out_path=None, figsize=(14, 7))`

Two-panel figure:

* **Top:** stacked, z-scored EEG in the window
* **Bottom:** envelope computed from `audio` with `gammatone_hilbert_envelope`, resampled to EEG `fs`

**Inputs**

* `eeg_TxC`: `(T, C)` time-major EEG
* `audio`: raw waveform or low-rate envelope
* `fs`: EEG rate (Hz)
* `fs_audio`: audio rate (Hz), if different
* `t_start`, `t_end`: window (seconds)

**Saves:** PNG to `figures/`.
**Returns:** path to the saved PNG (string).

---

## Tips & Troubleshooting

* **66 channels?** Use `subject_eeg_env_ab(...)` (it trims to 64).
* **Unknown blips (\~0.03%)?** Usually one unlabeled sample at trial start; fixed by the robust attention parser (handles 1-based indices and pre-fills the head).
* **Make envelopes EEG-aligned:** pass `target_fs=fs` to `gammatone_hilbert_envelope`.
* **Logs:** check `Logs/helper.log` if something looks odd (e.g., event parsing, resampling).
* **No `gammatone`?** The code falls back to Hilbert + low-pass (works fine for low-rate traces).

---

### Run test_helper.py to test if everything works as expected. 

