"""
helper.py — AAD I/O utilities for preprocessed KUL dataset (SXX.mat)

New in this version:
  - windows_for_trial_multiband(...): yields EEG and BOTH multi-band envelopes per window.

Main functions:
  - load_subject(mat_path)
  - get_trial(trials, idx, attended='auto', fallback_attend_map=None)
      -> (eeg[T,C], env_left[T,B], env_right[T,B], fs, attended_ear, meta)
  - iter_windows(eeg, env_left, env_right, fs, win_s=5.0, hop_s=None, start_s=0.0, center=False)
      -> yields (slice, eeg_win, envL_win, envR_win)
  - windows_for_trial_multiband(...):
      -> yields (X_eeg[Tw,C], envL_win[Tw,B], envR_win[Tw,B], label, meta_win)
  - (optional) make_broadband/env to_broadband for future use

Dependencies:
  pip install numpy scipy
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Iterator
import numpy as np
from scipy import io as sio

# ------------------------------ Core loaders ------------------------------ #

def load_subject(mat_path: str) -> List[Dict[str, Any]]:
    m = sio.loadmat(mat_path, simplify_cells=True)
    trials = m.get('preproc_trials')
    if trials is None:
        raise RuntimeError(f"'preproc_trials' not found in {mat_path}")
    if isinstance(trials, dict):
        trials = [trials]
    if not isinstance(trials, list):
        raise RuntimeError(f"Unexpected type for preproc_trials: {type(trials)}")
    return trials

def _normalize_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = np.asarray(eeg, dtype=float)
    if eeg.ndim != 2:
        raise ValueError("EEG must be 2D [T,C] or [C,T].")
    if eeg.shape[0] == 64 and eeg.shape[1] > 64:  # likely [C,T]
        return eeg.T
    return eeg

def _extract_attended(trial: Dict[str, Any]) -> Optional[str]:
    for key in ('attended_ear','attend','attended','attention','Attended','Attention'):
        v = trial.get(key)
        if isinstance(v, str) and v.lower() in ('left','right'):
            return v.lower()
    for key in ('is_left_attended','left_attend','attend_left'):
        v = trial.get(key)
        if isinstance(v, (bool, np.bool_)):
            return 'left' if bool(v) else 'right'
    return None

def get_trial(
    trials: List[Dict[str, Any]],
    idx: int,
    attended: str = 'auto',
    fallback_attend_map: Optional[Dict[int, str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Optional[str], Dict[str, Any]]:
    if idx < 1 or idx > len(trials):
        raise IndexError(f"Trial index {idx} out of range 1..{len(trials)}")
    tr = trials[idx - 1]

    eeg = _normalize_eeg(tr['RawData']['EegData'])
    fs = float(tr['FileHeader'].get('SampleRate', 32.0))

    env_tbc2 = np.asarray(tr['Envelope']['AudioData'], dtype=float)
    if env_tbc2.ndim != 3 or env_tbc2.shape[2] != 2:
        raise ValueError("Envelope.AudioData must be [T,B,2] (left/right).")
    env_left, env_right = env_tbc2[:, :, 0], env_tbc2[:, :, 1]

    w = tr['Envelope'].get('subband_weights')
    w = np.asarray(w, dtype=float) if w is not None else None

    stimuli = tr.get('stimuli')
    repetition = bool(tr.get('repetition', False))

    if attended == 'auto':
        att = _extract_attended(tr)
        if att is None and fallback_attend_map is not None:
            att = fallback_attend_map.get(idx)
        if att is None:
            att = _infer_attended_by_corr(eeg, env_left, env_right, fs)
    elif attended in ('left','right'):
        att = attended
    elif attended == 'none':
        att = None
    else:
        raise ValueError("attended must be 'auto','left','right','none'")

    meta = dict(stimuli=stimuli, repetition=repetition, subband_weights=w, trial_index=idx)
    return eeg, env_left, env_right, fs, att, meta

# ------------------------------ Windowing ------------------------------ #

def iter_windows(
    eeg: np.ndarray,
    env_left: np.ndarray,
    env_right: np.ndarray,
    fs: float,
    win_s: float = 5.0,
    hop_s: Optional[float] = None,
    start_s: float = 0.0,
    center: bool = False,
) -> Iterator[Tuple[slice, np.ndarray, np.ndarray, np.ndarray]]:
    eeg = np.asarray(eeg); env_left = np.asarray(env_left); env_right = np.asarray(env_right)
    assert eeg.shape[0] == env_left.shape[0] == env_right.shape[0], "EEG and envelopes must be aligned."
    T = eeg.shape[0]
    win = int(round(win_s * fs))
    hop = int(round((hop_s if hop_s is not None else win_s) * fs))
    pos0 = int(round(start_s * fs))
    if win <= 0 or hop <= 0:
        raise ValueError("win_s and hop_s must be > 0")

    pos = pos0
    while pos + win <= T:
        if center:
            half = win // 2
            sl = slice(max(0, pos - half), min(T, pos + half))
            if sl.stop - sl.start != win:
                pos += hop; continue
        else:
            sl = slice(pos, pos + win)
        yield sl, eeg[sl, :], env_left[sl, :], env_right[sl, :]
        pos += hop

def windows_for_trial_multiband(
    trials: List[Dict[str, Any]],
    idx: int,
    attended: str = 'auto',
    fallback_attend_map: Optional[Dict[int, str]] = None,
    win_s: float = 5.0,
    hop_s: Optional[float] = None,
    start_s: float = 0.0,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, str, Dict[str, Any]]]:
    """
    Yield multi-band windows for BOTH ears (for architectures that consume both).

    Returns per window:
        X_eeg:    [Tw, C]
        envL_win: [Tw, B]
        envR_win: [Tw, B]
        label:    'left' or 'right'  (attended ear)
        meta_win: { 'trial_index','slice','fs','stimuli','repetition','attended' }
    """
    eeg, envL, envR, fs, att, meta = get_trial(
        trials, idx, attended=attended, fallback_attend_map=fallback_attend_map
    )
    if att not in ('left','right'):
        raise RuntimeError("Attended label is missing and could not be inferred.")

    for sl, eeg_w, envL_w, envR_w in iter_windows(eeg, envL, envR, fs, win_s, hop_s, start_s, center=False):
        meta_win = {
            'trial_index': meta['trial_index'],
            'slice': sl,
            'fs': fs,
            'stimuli': meta.get('stimuli'),
            'repetition': meta.get('repetition', False),
            'attended': att,
        }
        yield eeg_w, envL_w, envR_w, att, meta_win

# ------------------------------ Broadband utils (optional) ------------------------------ #

def make_broadband(env_tb: np.ndarray, subband_weights: Optional[np.ndarray] = None) -> np.ndarray:
    env_tb = np.asarray(env_tb, dtype=float)
    if env_tb.ndim != 2:
        raise ValueError("env_tb must be [T,B].")
    T, B = env_tb.shape
    if subband_weights is None:
        w = np.ones((1, B), dtype=float)
    else:
        w = np.asarray(subband_weights, dtype=float)
        if w.ndim == 1:
            w = w[None, :]
        if w.shape[1] != B:
            raise ValueError(f"weights length {w.shape[1]} != bands {B}")
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    wT = np.broadcast_to(w, (T, w.shape[1]))
    return (env_tb * wT).sum(axis=1)

def to_broadband(env_left: np.ndarray, env_right: np.ndarray, weights: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    return make_broadband(env_left, weights), make_broadband(env_right, weights)

# ------------------------------ Heuristic attended inference ------------------------------ #

def _infer_attended_by_corr(
    eeg: np.ndarray,
    env_left: np.ndarray,
    env_right: np.ndarray,
    fs: float,
    win_s: float = 10.0,
    start_s: float = 5.0,
) -> str:
    T, C = eeg.shape
    s = int(round(start_s * fs))
    w = int(round(win_s * fs))
    e = min(T, s + w)
    if e <= s:
        s, e = 0, T

    def _bb(xTB):
        if xTB.ndim == 2:
            wts = np.ones((1, xTB.shape[1]), dtype=float); wts /= wts.sum()
            return (xTB * np.broadcast_to(wts, (xTB.shape[0], xTB.shape[1]))).sum(axis=1)
        return xTB

    eeg_w = eeg[s:e, :]
    bbL = _bb(env_left)[s:e]
    bbR = _bb(env_right)[s:e]

    X  = (eeg_w - eeg_w.mean(0, keepdims=True)) / (eeg_w.std(0, keepdims=True) + 1e-8)
    yL = (bbL - bbL.mean()) / (bbL.std() + 1e-8)
    yR = (bbR - bbR.mean()) / (bbR.std() + 1e-8)

    rL = np.abs((X * yL[:, None]).mean(axis=0)).sum()
    rR = np.abs((X * yR[:, None]).mean(axis=0)).sum()
    return 'left' if rL >= rR else 'right'

# ------------------------------ Tiny helper ------------------------------ #

def zscore(x: np.ndarray, axis=0, eps=1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True)
    return (x - m) / (s + eps)
