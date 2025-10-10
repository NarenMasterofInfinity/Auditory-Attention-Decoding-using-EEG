# child_helper.py
# Lean, AAD-friendly preprocessing + TSV alignment for CHILD and GENERAL EEG.
# BASE_DIR is the root for relative CSV paths.

import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA, create_eog_epochs
import soundfile as sf
from scipy.signal import resample

# =========================
# Global base directory
# =========================
BASE_DIR = "/home/naren-root/Documents/FYP/AAD/Project/Child_eeg"

# =========================
# Small utilities
# =========================
def _to_mono(audio_np: np.ndarray) -> np.ndarray:
    """Ensure 1-D mono audio."""
    return audio_np if audio_np.ndim == 1 else np.mean(audio_np, axis=1)

def downsample_audio(audio: np.ndarray, orig_rate: float, target_rate: float, target_len: int | None = None) -> np.ndarray:
    """
    Resample audio to target_rate (or directly to target_len if provided).
    """
    if target_len is not None:
        return resample(audio, target_len)
    ratio = float(target_rate) / float(orig_rate)
    new_len = int(round(len(audio) * ratio))
    return resample(audio, new_len)

def find_flat_channels(raw: mne.io.BaseRaw, threshold: float = 1e-6) -> list[str]:
    """
    Detect truly flat/dead channels via band-limited variance.
    Assumes raw already band-limited (e.g., 0.5–30 Hz).
    """
    data = raw.get_data()
    flats = [raw.ch_names[i] for i in range(data.shape[0]) if np.std(data[i]) < threshold]
    return flats

# =========================
# Lean CHILD preprocessing
# =========================
def preprocess_eeg_child_preprocess(
    eeg_file: str,
    target_sfreq: float = 64.0,
    hp: float = 0.5,
    lp: float = 30.0,
    notch: float | None = 50.0,
    use_lof: bool = False,       # OFF by default: can over-flag in child EEG
    use_iclabel: bool = True,    # Try ICLabel if available (fallback safely)
    ica_method: str = 'fastica',
    ica_var: float = 0.99,       # variance explained for ICA
    ica_decim: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """
    Lean CHILD treatment for EGI HydroCel (.set):
      1) Load at native fs; set HydroCel montage.
      2) Band-limit at native fs (hp/lp + optional notch).
      3) Flat-channel detection only; interpolate.
      4) Average reference (post-interp).
      5) ICA: fit on 1–40 Hz copy (decimated), detect blinks via frontal surrogates (Fp1/Fp2/AF7/AF8),
         optionally auto-label with ICLabel (if installed). Apply ICA to main raw (native fs).
      6) Resample to target_sfreq (default 64 Hz).
    Returns: (cleaned_eeg [T, C], target_sfreq).
    """
    # 1) Load
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
    native_fs = float(raw.info['sfreq'])

    # 2) Montage (robust to missing labels)
    try:
        raw.set_montage("GSN-HydroCel-129", on_missing="ignore")
    except Exception:
        pass

    # 3) Band-limit at native fs (anti-alias & constrain task bandwidth)
    raw.filter(hp, lp, fir_design='firwin', phase='zero')
    if notch is not None:
        raw.notch_filter(freqs=[notch], fir_design='firwin')

    # 4) Bad-channel handling (flat only by default)
    flats = find_flat_channels(raw, threshold=1e-6)
    bads = sorted(set(flats))
    if use_lof:
        # Optional: try MNE LOF (requires scikit-learn). It may over-flag on kids—use sparingly.
        try:
            from mne.preprocessing import find_bad_channels_lof
            lof_bads = find_bad_channels_lof(raw)
            bads = sorted(set(bads + lof_bads))
        except Exception:
            pass

    if bads:
        raw.info['bads'] = bads
        raw.interpolate_bads(reset_bads=True)

    # 5) Average reference after interpolation (PREP-style order)
    raw.set_eeg_reference('average', projection=False)

    # 6) ICA on a 1–40 Hz copy at native fs (decimated for speed)
    ica_raw = raw.copy().filter(1., 40., fir_design='firwin', phase='zero')
    ica = ICA(n_components=ica_var, method=ica_method, random_state=random_state, max_iter='auto')

    try:
        ica.fit(ica_raw, decim=ica_decim)
        exclude = []

        # ICLabel support (optional)
        if use_iclabel:
            try:
                from mne_icalabel import label_components
                labels = label_components(raw, ica, method='iclabel')
                klass = labels['labels']           # list of strings
                probs = labels['y_pred_proba']     # Nx7 class probabilities
                # Common: 'eye' (blink), 'muscle', 'line_noise'
                for idx, (lab, p) in enumerate(zip(klass, probs)):
                    if (lab == 'eye' and p[labels['classes'].index('eye')] >= 0.7) \
                       or (lab == 'muscle' and p[labels['classes'].index('muscle')] >= 0.7) \
                       or (lab == 'line_noise' and p[labels['classes'].index('line_noise')] >= 0.7):
                        exclude.append(idx)
            except Exception:
                pass

        # Blink/EOG via frontal surrogates (EGI often lacks dedicated EOG)
        frontal = [ch for ch in raw.ch_names if ch in ('Fp1','Fp2','AF7','AF8','AF3','AF4')]
        if frontal:
            try:
                eog_epochs = create_eog_epochs(raw, ch_name=frontal)
                eog_inds, _ = ica.find_bads_eog(eog_epochs)
                exclude.extend(eog_inds)
            except Exception:
                # fallback: correlate components with each frontal channel
                for ch in frontal:
                    try:
                        inds, _ = ica.find_bads_eog(raw, ch_name=ch)
                        exclude.extend(inds)
                    except Exception:
                        pass

        ica.exclude = sorted(set(exclude))
        if ica.exclude:
            raw = ica.apply(raw)
    except Exception:
        # ICA failed or not helpful; continue with band-limited, avg-ref data
        pass

    # 7) Resample to target_sfreq for modeling
    if abs(native_fs - target_sfreq) > 1e-3:
        raw.resample(target_sfreq)

    # Done
    return raw.get_data().T.astype(np.float32), float(raw.info['sfreq'])

# =========================
# TSV alignment (per row)
# =========================
def _align_eeg_audio_from_row(
    eeg_data: np.ndarray,
    sfreq_eeg: float,
    audio_data: np.ndarray,
    sfreq_audio: float,
    tsv_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use TSV events (video_start/stop in EEG samples) to cut, align, and concatenate segments.
    Audio is resampled to match each EEG segment’s length at sfreq_eeg (typically 64 Hz).
    Returns (eeg_concat [T,C], audio_concat [T]).
    """
    ev = tsv_df[tsv_df['value'].isin(['video_start', 'video_stop'])]
    if len(ev) < 2 or len(ev) % 2 != 0:
        return np.empty((0, eeg_data.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)

    audio_mono = _to_mono(audio_data)
    eeg_segments, audio_segments = [], []

    for i in range(0, len(ev), 2):
        s_start = int(ev.iloc[i]['sample'])
        s_stop  = int(ev.iloc[i+1]['sample'])
        s_start = max(0, min(s_start, eeg_data.shape[0]))
        s_stop  = max(0, min(s_stop,  eeg_data.shape[0]))
        if s_stop <= s_start:
            continue

        # Times in seconds
        t_start, t_stop = s_start / sfreq_eeg, s_stop / sfreq_eeg
        a_start, a_stop = int(round(t_start * sfreq_audio)), int(round(t_stop * sfreq_audio))
        a_start = max(0, min(a_start, len(audio_mono)))
        a_stop  = max(0, min(a_stop,  len(audio_mono)))
        if a_stop <= a_start:
            continue

        eeg_seg = eeg_data[s_start:s_stop]
        aud_seg = audio_mono[a_start:a_stop]
        aud_seg = downsample_audio(aud_seg, sfreq_audio, sfreq_eeg, target_len=eeg_seg.shape[0])

        eeg_segments.append(eeg_seg.astype(np.float32))
        audio_segments.append(aud_seg.astype(np.float32))

    if not eeg_segments:
        return np.empty((0, eeg_data.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.concatenate(eeg_segments, axis=0), np.concatenate(audio_segments, axis=0)

# =========================
# CHILD loader (64 Hz out)
# =========================
def load_data_child_treatment(csv_path: str, base_dir: str | None = None) -> dict:
    """
    CHILD loader (lean pipeline, 64 Hz output):
      - CSV columns: eeg_file, tsv_file, audio_file
      - For each row: preprocess EEG (child @64 Hz), load TSV+audio, align by TSV events,
        then concatenate segments → one EEG and one audio per row.
    Returns: {"eeg": array(object) of [T_i,C], "audio": array(object) of [T_i], "sfreq": array(float)}
    """
    root = base_dir or BASE_DIR
    df = pd.read_csv(csv_path)
    eeg_all, audio_all, sf_all = [], [], []

    for _, row in df.iterrows():
        eeg_path = os.path.join(root, row['eeg_file'])
        tsv_path = os.path.join(root, row['tsv_file'])
        audio_path = os.path.join(root, row['audio_file'])

        eeg_data, sfreq_eeg = preprocess_eeg_child_preprocess(eeg_path)  # → 64.0
        tsv_df = pd.read_csv(tsv_path, sep='\t')
        audio_np, sfreq_audio = sf.read(audio_path)

        eeg_concat, aud_concat = _align_eeg_audio_from_row(eeg_data, sfreq_eeg, audio_np, sfreq_audio, tsv_df)
        eeg_all.append(eeg_concat)
        audio_all.append(aud_concat)
        sf_all.append(sfreq_eeg)

    return {
        "eeg": np.array(eeg_all, dtype=object),
        "audio": np.array(audio_all, dtype=object),
        "sfreq": np.array(sf_all, dtype=float),
    }

def load_data_child_treatment_concat(csv_path: str, base_dir: str | None = None) -> dict:
    """
    Concatenate across ALL CSV rows (after per-row alignment).
    Returns a single long pair: {"eeg": [T_total,C], "audio": [T_total], "sfreq": 64.0}
    """
    out = load_data_child_treatment(csv_path, base_dir=base_dir)
    eeg_all = out["eeg"]; aud_all = out["audio"]
    eeg_cat = np.concatenate(list(eeg_all), axis=0) if len(eeg_all) else np.empty((0, 0), dtype=np.float32)
    aud_cat = np.concatenate(list(aud_all), axis=0) if len(aud_all) else np.empty((0,), dtype=np.float32)
    return {"eeg": eeg_cat, "audio": aud_cat, "sfreq": 64.0}

# =========================
# GENERAL pipeline (64 Hz)
# =========================
def general_preprocess_eeg(eeg_file: str) -> tuple[np.ndarray, float]:
    """
    GENERAL preprocessing: (assumes arbitrary montage; we still try HydroCel)
      - Resample to 64 Hz
      - Bandpass 0.5–30 Hz (zero-phase)
      - Average reference
      - ICA (EOG/ECG if present)
    Returns: (cleaned_eeg[T,C], sfreq=64.0)
    """
    raw = mne.io.read_raw_eeglab(eeg_file, preload=True)

    # Try HydroCel montage (ignored if mismatched)
    try:
        raw.set_montage("GSN-HydroCel-129", on_missing="ignore")
    except Exception:
        pass

    raw.resample(64.0)
    raw.filter(0.5, 40., fir_design="firwin", phase='zero')
    raw.set_eeg_reference('average', projection=False)

    ica = ICA(n_components=20, random_state=42, max_iter="auto")
    try:
        ica.fit(raw)
        # EOG if present
        try:
            eog_inds, _ = ica.find_bads_eog(raw)
            if eog_inds:
                ica.exclude.extend(eog_inds)
        except Exception:
            pass
        # ECG optional (surrogate)
        try:
            ecg_inds, _ = ica.find_bads_ecg(raw, ch_name="Fz")
            if ecg_inds:
                ica.exclude.extend(ecg_inds)
        except Exception:
            pass
        raw = ica.apply(raw)
    except Exception:
        pass

    eeg_data = raw.get_data().T.astype(np.float32)
    return eeg_data, float(raw.info['sfreq'])  # 64.0

def load_eeg_general_treatment(csv_path: str, base_dir: str) -> dict:
    """
    GENERAL loader (64 Hz):
      - CSV columns: eeg_file, tsv_file, audio_file
      - For each row: general_preprocess_eeg (@64 Hz), align by TSV events, concatenate per row.
    Returns: {"eeg": array(object) of [T_i,C], "audio": array(object) of [T_i], "sfreq": array(float)}
    """
    df = pd.read_csv(csv_path)
    eeg_all, audio_all, sf_all = [], [], []

    for _, row in df.iterrows():
        eeg_path = os.path.join(base_dir, row['eeg_file'])
        tsv_path = os.path.join(base_dir, row['tsv_file'])
        audio_path = os.path.join(base_dir, row['audio_file'])

        eeg_data, sfreq_eeg = general_preprocess_eeg(eeg_path)  # → 64.0
        tsv_df = pd.read_csv(tsv_path, sep='\t')
        audio_np, sfreq_audio = sf.read(audio_path)

        eeg_concat, aud_concat = _align_eeg_audio_from_row(eeg_data, sfreq_eeg, audio_np, sfreq_audio, tsv_df)
        eeg_all.append(eeg_concat.astype(np.float32))
        audio_all.append(aud_concat.astype(np.float32))
        sf_all.append(sfreq_eeg)

    return {
        "eeg": np.array(eeg_all, dtype=object),
        "audio": np.array(audio_all, dtype=object),
        "sfreq": np.array(sf_all, dtype=float),
    }
