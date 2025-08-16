import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import io as sio, signal
from gammatone.filters import centre_freqs, make_erb_filters, erb_filterbank
from scipy.io.matlab.mio5_params import mat_struct
from matplotlib.lines import Line2D


# Folders
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("Logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# Logging
_logger = logging.getLogger("helper")
_logger.setLevel(logging.INFO)
if not _logger.handlers:
    fh = logging.FileHandler(LOG_DIR / "helper.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(funcName)s | %(message)s"))
    _logger.addHandler(fh)


def _unique_png(prefix):
    p = FIG_DIR / f"{prefix}.png"
    k = 1
    while p.exists():
        p = FIG_DIR / f"{prefix}_{k}.png"
        k += 1
    return p


def _unwrap_mat_struct(obj):
    if isinstance(obj, mat_struct):
        d = {}
        for name in obj._fieldnames:
            d[name] = _unwrap_mat_struct(getattr(obj, name))
        return d
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return np.array([_unwrap_mat_struct(x) for x in obj], dtype=object)
    return obj


def load_mat_for_subject(preproc_dir, subj_id):
    pre = Path(preproc_dir)
    candidates = [pre / f"S{subj_id}_data_preproc.mat", pre / f"S{subj_id:02d}_data_preproc.mat"]
    for p in candidates:
        if p.exists():
            mat_path = p
            break
    else:
        raise FileNotFoundError(f"No file for subject {subj_id} in {pre}")
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    mat = {k: _unwrap_mat_struct(v) for k, v in mat.items() if not k.startswith("__")}
    _logger.info(f"Loaded MAT for S{subj_id} from {mat_path}")
    return mat


def _ensure_ch_t(mat2):
    M = np.squeeze(np.asarray(mat2))
    if M.ndim != 2:
        raise ValueError("Could not coerce EEG slice to 2D")
    if M.shape[0] > M.shape[1]:
        M = M.T
    return M.astype(float)


def _extract_event_container(event_all):
    if isinstance(event_all, dict):
        for k in ("eeg", "EEG", "att", "attention", "wavA", "wavB"):
            if k in event_all:
                return event_all[k]
    return event_all


def _explode_trials_from_group(group):
    eeg3 = np.asarray(group.get("eeg"))
    if eeg3.ndim != 3:
        return [group]
    chan_names = group.get("chan", None)
    n_ch_hint = len(np.atleast_1d(chan_names)) if chan_names is not None else None
    axis_ch = None

    if n_ch_hint is not None:
        for ax, sz in enumerate(eeg3.shape):
            if sz == n_ch_hint:
                axis_ch = ax
                break
    if axis_ch is None:
        axis_ch = int(np.argsort(eeg3.shape)[1])
    rem = [0, 1, 2]
    rem.remove(axis_ch)
    axis_time = rem[int(np.argmax([eeg3.shape[r] for r in rem]))]
    axis_trial = [ax for ax in [0, 1, 2] if ax not in (axis_ch, axis_time)][0]
    n_trials = eeg3.shape[axis_trial]

    def _slice_wav(wav_all, i, n_t):
        if wav_all is None:
            return np.zeros(n_t, float)
        arr = np.asarray(wav_all)
        if arr.ndim == 2:
            if arr.shape[0] == n_trials:
                w = arr[i, :]
            elif arr.shape[1] == n_trials:
                w = arr[:, i]
            else:
                w = np.ravel(arr)
        elif arr.ndim == 1:
            w = arr
        elif arr.dtype == object and arr.ndim == 1 and arr.shape[0] == n_trials:
            w = np.asarray(arr[i]).ravel()
        else:
            w = np.ravel(arr)
        if w.size != n_t:
            w = signal.resample(w, n_t)
        return w.astype(float)

    def _slice_event(event_all, i):
        if event_all is None:
            return None
        container = _extract_event_container(event_all)
        if isinstance(container, (list, tuple)) and len(container) == n_trials:
            return container[i]
        arr = np.asarray(container)
        if arr.dtype == object and arr.ndim == 1 and arr.shape[0] == n_trials:
            return arr[i]
        if arr.ndim == 3 and (arr.shape[0] == n_trials or arr.shape[2] == n_trials):
            return arr[i] if arr.shape[0] == n_trials else arr[:, :, i]
        return None

    trials = []
    for i in range(n_trials):
        slicer = [slice(None)] * 3
        slicer[axis_trial] = i
        Xi = _ensure_ch_t(eeg3[tuple(slicer)])
        n_ch, n_t = Xi.shape
        A_i = _slice_wav(group.get("wavA", None), i, n_t)
        B_i = _slice_wav(group.get("wavB", None), i, n_t)
        ev_i = _slice_event(group.get("event", None), i)
        trials.append(dict(
            eeg=Xi.astype(float),
            wavA=A_i.astype(float),
            wavB=B_i.astype(float),
            event=ev_i,
            fsample=group.get("fsample", group.get("freq", None)),
            chan=chan_names
        ))
    _logger.info(f"Exploded group into {len(trials)} trials")
    return trials


def _iter_trials(mat):
    data = mat.get("data", None)
    if data is None:
        raise KeyError("Expected key 'data' in .mat file")
    if isinstance(data, list):
        return data
    if isinstance(data, np.ndarray) and data.dtype == object:
        return [x.item() if hasattr(x, "item") else x for x in data]
    if isinstance(data, dict):
        eeg = data.get("eeg", None)
        if isinstance(eeg, np.ndarray) and eeg.ndim == 3:
            return _explode_trials_from_group(data)
        return [data]
    raise TypeError("Unsupported mat['data'] container type")


def _lp_filter(y, fs, cutoff_hz, order=4):
    if not cutoff_hz or cutoff_hz <= 0:
        return y
    nyq = fs / 2.0
    cutoff = min(cutoff_hz, nyq * 0.99)
    b, a = signal.butter(order, cutoff / nyq)
    return signal.filtfilt(b, a, y)


def _resample_to(y, fs_in, fs_out):
    if fs_out is None or fs_out == fs_in:
        return y, fs_in
    from fractions import Fraction
    frac = Fraction(fs_out, fs_in).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    y2 = signal.resample_poly(y, up, down)
    return y2, fs_out


def gammatone_hilbert_envelope(audio, fs, *, num_bands=32, fmin=50.0, fmax=None,
                               compress="pow", compress_exp=0.6, aggregate="sum",
                               lowpass_hz=8.0, target_fs=None, normalize="unit",
                               return_bands=False):
    x = np.asarray(audio, dtype=float).ravel()
    if fs < 1000 or centre_freqs is None or make_erb_filters is None or erb_filterbank is None:
        env = np.abs(signal.hilbert(x))
        env = _lp_filter(env, fs, lowpass_hz)
        env, fs_env = _resample_to(env, fs, target_fs)
        if normalize == "unit":
            m = np.max(np.abs(env))
            if m > 0:
                env = env / (m + 1e-12)
        elif normalize == "zscore":
            env = (env - env.mean()) / (env.std() + 1e-12)
        meta = {"fs": fs_env, "cf": None, "band_envs": None, "aggregate": "simple", "compress": "none"}
        return env.astype(float), meta
    if fmax is None:
        fmax = min(0.45 * fs, fs / 2.0)
    cf = centre_freqs(fs, num_bands, fmin)
    cf = cf[cf <= fmax]
    if cf.size == 0:
        raise ValueError("No center frequencies within [fmin, fmax].")
    fcoefs = make_erb_filters(fs, cf)
    y = erb_filterbank(x, fcoefs)
    band_envs = np.abs(signal.hilbert(y, axis=-1))
    if compress == "pow":
        band_envs = np.power(band_envs, float(compress_exp))
    elif compress == "log":
        band_envs = np.log1p(band_envs)
    if band_envs.ndim == 1:
        agg = band_envs
    else:
        if aggregate == "sum":
            agg = band_envs.sum(axis=0)
        elif aggregate == "rms":
            agg = np.sqrt((band_envs ** 2).mean(axis=0))
        else:
            agg = band_envs.mean(axis=0)
    agg = _lp_filter(agg, fs, lowpass_hz)
    env, fs_env = _resample_to(agg, fs, target_fs)
    if normalize == "unit":
        m = np.max(np.abs(env))
        if m > 0:
            env = env / (m + 1e-12)
    elif normalize == "zscore":
        env = (env - env.mean()) / (env.std() + 1e-12)
    meta = {
        "fs": fs_env,
        "cf": cf,
        "band_envs": band_envs if return_bands else None,
        "aggregate": aggregate,
        "compress": compress
    }
    return env.astype(float), meta


def get_audio_signal(mat, stream="A"):
    assert stream.upper() in ("A", "B")
    key = "wavA" if stream.upper() == "A" else "wavB"
    trials = _iter_trials(mat)
    parts, fs_list = [], []
    for idx, tr in enumerate(trials):
        if key not in tr:
            continue
        wav = np.asarray(tr[key]).squeeze()
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        parts.append(wav.astype(float))
        fs_val = tr.get("fsample", tr.get("freq", 0.0))
        if isinstance(fs_val, dict):
            for cand in (key, key.lower(), key.upper(), "fs", "audio_fs", "eeg"):
                if cand in fs_val and np.isscalar(fs_val[cand]):
                    fs_val = fs_val[cand]
                    break
        fs_list.append(float(fs_val) if np.isscalar(fs_val) else 0.0)
    if not parts:
        raise KeyError(f"No '{key}' found in any trial")
    fs_vals = [f for f in fs_list if f > 0]
    fs = fs_vals[0] if fs_vals else 16000.0
    if any(f > 0 and abs(f - fs) > 1e-6 for f in fs_list):
        parts = [signal.resample(p, int(round(len(p) * (fs / max(f, 1e-9))))) if (f and f > 0) else p for p, f in zip(parts, fs_list)]
    audio = np.concatenate(parts).astype(float)
    _logger.info(f"Concatenated {len(parts)} trials for stream {stream}")
    return audio, fs


def plot_audio(audio, fs, t_start, t_end, also_envelope=True, num_bands=32, fmin=100.0,
               out_path=None, title=None, env_norm="unit", audio_norm="none", lp_hz=8.0):
    if t_end <= t_start:
        raise ValueError("t_end must be > t_start")
    i0 = max(0, int(np.floor(t_start * fs)))
    i1 = min(len(audio), int(np.ceil(t_end * fs)))
    seg = np.asarray(audio[i0:i1], float)
    if audio_norm == "zscore":
        seg = (seg - seg.mean()) / (seg.std() + 1e-12)
    t = np.arange(i0, i1) / fs
    env = None
    if also_envelope:
        env, _ = gammatone_hilbert_envelope(seg, fs, num_bands=num_bands, fmin=fmin,
                                            lowpass_hz=lp_hz, target_fs=fs, normalize="unit")
        if env_norm == "zscore":
            env = (env - env.mean()) / (env.std() + 1e-12)
    plt.figure(figsize=(12, 4))
    plt.plot(t, seg, lw=0.8, label="audio")
    if env is not None:
        plt.plot(t, env, lw=1.2, label="envelope (GT+Hilbert)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title or "Audio (with envelope)")
    plt.legend()
    plt.tight_layout()
    if out_path is None:
        out_path = _unique_png(f"audio_{int(t_start * 1000)}ms_{int(t_end * 1000)}ms")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    _logger.info(f"Saved plot_audio -> {out_path}")
    return env if env is not None else np.array([])


def _parse_event_segments(samples, values, n_t):
    m = np.zeros(n_t, dtype=np.uint8)
    if samples is None or values is None:
        return m

    samples = np.asarray(samples).astype(int).ravel()
    values = np.asarray(values).astype(int).ravel()
    if samples.size == 0 or values.size == 0:
        return m

    # convert 1-based to 0-based if it looks like 1..n_t
    if samples.min() >= 1 and samples.max() <= n_t:
        samples = samples - 1

    # clamp, sort
    samples = np.clip(samples, 0, n_t)
    order = np.argsort(samples)
    samples = samples[order]
    values = values[order]

    # prefill head with first value
    s0 = int(samples[0])
    v0 = int(values[0])
    if s0 > 0:
        m[:s0] = v0

    # fill segments
    cur = s0
    for j in range(1, samples.size):
        s = int(samples[j])
        if s > cur:
            m[cur:s] = int(values[j - 1])
        cur = min(s, n_t)
        if cur >= n_t:
            break

    # tail
    if cur < n_t:
        m[cur:n_t] = int(values[-1])
    return m


def _try_direct_vector(v, n_t):
    """Map a full-length vector of labels to {1(A),2(B)}."""
    v = np.asarray(v)
    if v.dtype == object and v.size == 1 and hasattr(v.item(), "__array__"):
        v = np.asarray(v.item())
    v = np.ravel(v)
    if v.size != n_t:
        return None

    if v.dtype.kind in ("U", "S", "O"):
        vu = np.char.upper(v.astype(str))
        return np.where(vu == "A", 1, np.where(vu == "B", 2, 0)).astype(np.uint8)

    vv = v.astype(int)
    uniq = np.unique(vv)
    if np.all(np.isin(uniq, [1, 2])):
        return vv.astype(np.uint8)
    if np.all(np.isin(uniq, [0, 1])):
        # Heuristic: datasets sometimes use {0,1}. Map 0->A(1), 1->B(2).
        return (vv + 1).astype(np.uint8)
    return vv.astype(np.uint8)


def _build_att_mask(n_t, tr):
    """
    Robust attention parser:
      1) direct full-length vector under many possible keys
      2) (samples, values) events in many shapes/keys
      3) trialinfo scalar ('A'/'B' or 1/2)
    Returns uint8 mask with values in {1,2} (0 if truly unknown).
    """
    # --- 1) direct per-sample vector
    for k in ("attmask", "att", "attention", "attended", "att_stream", "attended_stream"):
        if k in tr:
            m = _try_direct_vector(tr[k], n_t)
            if m is not None:
                return m

    # --- 2) events container: dict or array with two fields/dims
    evt = tr.get("event", tr.get("events", tr.get("att_events", None)))
    if isinstance(evt, dict):
        samples = None
        values = None
        # try common field names
        for sk in ("samples", "sample", "pos", "samp", "indices", "idx"):
            if sk in evt:
                samples = evt[sk]
                break
        for vk in ("values", "value", "val", "att", "attention", "label"):
            if vk in evt:
                values = evt[vk]
                break
        m = _parse_event_segments(samples, values, n_t)
        if np.any(m):
            return m

    if evt is not None:
        arr = np.asarray(evt)
        # Accept (N,2) or (2,N)
        if arr.ndim == 2 and (arr.shape[0] == 2 or arr.shape[1] == 2):
            A = arr if arr.shape[0] == 2 else arr.T
            a, b = A[0].astype(float), A[1].astype(float)
            # decide which row are indices by dynamic range
            if (a.max() - a.min()) >= (b.max() - b.min()):
                samples, values = a, b
            else:
                samples, values = b, a
            m = _parse_event_segments(samples, values, n_t)
            if np.any(m):
                return m

    # --- 3) trialinfo scalar per trial
    trialinfo = tr.get("trialinfo", None)
    if isinstance(trialinfo, dict):
        for k in ("attend", "attended", "attention"):
            if k in trialinfo:
                lbl = trialinfo[k]
                # numbers 1/2
                try:
                    val = int(lbl)
                    if val in (1, 2):
                        return np.full(n_t, val, np.uint8)
                except Exception:
                    pass
                # strings 'A'/'B'
                s = str(lbl).upper()
                if s.startswith(("A", "L")):
                    return np.full(n_t, 1, np.uint8)
                if s.startswith(("B", "R")):
                    return np.full(n_t, 2, np.uint8)

    # --- Nothing found: leave zeros but LOG what we saw
    try:
        import logging
        logging.getLogger("helper").warning(
            "Attention not found for trial; keys=%s", list(tr.keys())
        )
    except Exception:
        pass
    return np.zeros(n_t, dtype=np.uint8)


def load_subject(preproc_dir, subj_id, drop_exg=True):
    mat = load_mat_for_subject(preproc_dir, subj_id)
    trials = _iter_trials(mat)
    eeg_trials, envA_trials, envB_trials, attmask_trials, lengths = [], [], [], [], []
    ch_names, fs = None, None
    for tr in trials:
        eeg = np.asarray(tr["eeg"], float)
        if eeg.ndim != 2:
            raise ValueError("Each trial 'eeg' must be 2D after explosion")
        if eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T
        n_ch, n_t = eeg.shape
        if ch_names is None and "chan" in tr and tr["chan"] is not None:
            ch_names = [str(x) for x in np.atleast_1d(tr["chan"]).tolist()]
        if fs is None:
            fs_val = tr.get("fsample", tr.get("freq", 0.0))
            if isinstance(fs_val, dict):
                for k in ("eeg", "EEG", "fs"):
                    if k in fs_val and np.isscalar(fs_val[k]):
                        fs_val = fs_val[k]
                        break
            fs = float(fs_val) if np.isscalar(fs_val) and fs_val else 64.0
        A = np.asarray(tr.get("wavA", np.zeros(n_t)), float).squeeze()
        B = np.asarray(tr.get("wavB", np.zeros(n_t)), float).squeeze()
        if A.size != n_t:
            A = signal.resample(A, n_t)
        if B.size != n_t:
            B = signal.resample(B, n_t)
        att = _build_att_mask(n_t, tr)
        eeg_trials.append(eeg.astype(float))
        envA_trials.append(A.astype(float))
        envB_trials.append(B.astype(float))
        attmask_trials.append(att.astype(np.uint8))
        lengths.append(n_t)
    if ch_names is None:
        ch_names = [f"Ch{c+1}" for c in range(eeg_trials[0].shape[0])]
    if drop_exg:
        keep = [not nm.upper().startswith("EXG") for nm in ch_names]
        ch_names = [nm for nm, k in zip(ch_names, keep) if k]
        keep_idx = np.array(keep, bool)
        eeg_trials = [x[keep_idx, :] for x in eeg_trials]
    eeg = np.concatenate(eeg_trials, axis=1).astype(float)
    envA = np.concatenate(envA_trials).astype(float) if envA_trials else None
    envB = np.concatenate(envB_trials).astype(float) if envB_trials else None
    attmask = np.concatenate(attmask_trials).astype(np.uint8) if attmask_trials else None
    D = dict(
        fs=fs,
        ch_names=ch_names,
        eeg=eeg,
        envA=envA,
        envB=envB,
        attmask=attmask,
        lengths=lengths,
        subj_id=subj_id,
        has_two_streams=(envA is not None and envB is not None and (np.any(envA) or np.any(envB))),
        path=str(Path(preproc_dir))
    )
    _logger.info(f"Loaded subject S{subj_id}: EEG {eeg.shape}, two_streams={D['has_two_streams']}")
    return D


def _clean_to_64(eeg, ch_names=None):
    X = np.asarray(eeg, float)
    n_ch = X.shape[0]
    if n_ch <= 64:
        return X, (ch_names if ch_names is not None else [f"Ch{i+1}" for i in range(n_ch)]), np.arange(n_ch)
    if ch_names is not None:
        bad_prefixes = ("EXG", "MISC", "AUX", "STATUS", "TRIG")
        bad_contains = ("EOG", "ECG", "EMG")
        keep_idx = []
        for i, nm in enumerate(ch_names):
            nm_u = str(nm).upper()
            if any(nm_u.startswith(p) for p in bad_prefixes):
                continue
            if any(k in nm_u for k in bad_contains):
                continue
            keep_idx.append(i)
        if len(keep_idx) >= 64:
            keep_idx = keep_idx[:64]
        else:
            rest = [i for i in range(n_ch) if i not in keep_idx]
            keep_idx = (keep_idx + rest)[:64]
    else:
        keep_idx = list(range(64))
    keep_idx = np.asarray(keep_idx, int)
    X64 = X[keep_idx, :]
    ch64 = [ch_names[i] for i in keep_idx] if ch_names is not None else [f"Ch{i+1}" for i in range(64)]
    _logger.info(f"Trimmed EEG channels {n_ch}->64")
    return X64, ch64, keep_idx


def subject_eeg_env_ab(preproc_dir, subj_id, num_bands=32, fmin=50.0, fmax=None,
                       lowpass_hz=8.0, normalize="unit"):
    D = load_subject(preproc_dir, subj_id, drop_exg=True)
    fs = float(D["fs"])
    eeg = np.asarray(D["eeg"], float)
    ch_names = D.get("ch_names", None)
    eeg, ch_names64, idx64 = _clean_to_64(eeg, ch_names)
    T = eeg.shape[1]
    att = np.asarray(D["attmask"], np.uint8)
    wavA = np.asarray(D["envA"], float) if D["envA"] is not None else np.zeros(T, float)
    wavB = np.asarray(D["envB"], float) if D["envB"] is not None else np.zeros(T, float)

    def _env(x):
        env, _ = gammatone_hilbert_envelope(x, fs, num_bands=num_bands, fmin=fmin, fmax=fmax,
                                            compress="pow", compress_exp=0.6, aggregate="sum",
                                            lowpass_hz=lowpass_hz, target_fs=fs, normalize=normalize,
                                            return_bands=False)
        if env.size != T:
            env = signal.resample(env, T)
        return env.astype(float)

    envA = _env(wavA)
    envB = _env(wavB)
    env_att = np.where(att == 1, envA, np.where(att == 2, envB, 0.0)).astype(float)
    att_AB = np.where(att == 1, "A", np.where(att == 2, "B", "U")).astype("<U1")
    eeg_TxC = eeg.T
    _logger.info(f"Built attended envelope; A%={np.mean(att==1):.3f}, B%={np.mean(att==2):.3f}")
    return eeg_TxC, env_att, fs, att_AB


def plot_window(preproc_dir, subj_id, t_start, t_end, max_channels=None, figsize=(14, 7)):
    D = load_subject(preproc_dir, subj_id, drop_exg=True)
    fs, eeg, ch_names = D["fs"], D["eeg"], D["ch_names"]
    A, B, att = D["envA"], D["envB"], D["attmask"]
    T = eeg.shape[1]
    if not D["has_two_streams"]:
        _logger.info("File lacks two valid streams across all trials")
    i0 = max(0, int(np.floor(t_start * fs)))
    i1 = min(T, int(np.ceil(t_end * fs)))
    if i1 <= i0:
        raise ValueError("t_end must be > t_start within data length")
    t = np.arange(i0, i1) / fs
    X = np.asarray(eeg[:, i0:i1], float)
    Xz = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
    idxs = np.arange(Xz.shape[0])
    if max_channels is not None:
        idxs = idxs[:max_channels]
        Xz = Xz[idxs, :]
        ch_show = [ch_names[i] for i in idxs]
    else:
        ch_show = ch_names

    def _slice_and_z(env):
        if env is None or np.size(env) == 0:
            return None
        y = np.asarray(env[i0:i1], float)
        return (y - y.mean()) / (y.std() + 1e-12)

    Awin = _slice_and_z(A)
    Bwin = _slice_and_z(B)
    attw = att[i0:i1] if att is not None and att.size else np.zeros(i1 - i0, dtype=np.uint8)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    step = 5.0
    offsets = np.arange(len(idxs))[::-1] * step
    for k in range(Xz.shape[0]):
        ax_top.plot(t, Xz[k] + offsets[k], linewidth=0.6, color="black")
    ax_top.set_yticks(offsets)
    ax_top.set_yticklabels(ch_show)
    ax_top.set_ylabel("Channels (z, stacked)")
    ax_top.set_title(f"S{subj_id}: EEG + audio (attention-aware)\n{t_start:.2f}–{t_end:.2f}s")
    cum_samples = np.r_[0, np.cumsum(D["lengths"])]
    edges_sec = cum_samples / fs
    for ti, ed in enumerate(edges_sec):
        if t_start < ed < t_end:
            ax_top.axvline(ed, color="gray", ls=":", lw=1)
            ax_bot.axvline(ed, color="gray", ls=":", lw=1)
            ax_bot.text(ed, 0.95, f"T{ti:02d}", transform=ax_bot.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="gray")
    BLUE = "#1f77b4"
    ORANGE = "#ff7f0e"

    def _plot_att_segments(ax, y, attw, attended_is, color):
        if y is None or y.size == 0:
            return
        N = y.shape[0]
        changes = np.flatnonzero(np.diff(attw, prepend=attw[0]))
        seg_starts = np.r_[0, changes]
        seg_ends = np.r_[changes, N - 1] + 1
        for s, e in zip(seg_starts, seg_ends):
            att_val = attw[s]
            tt, yy = t[s:e], y[s:e]
            if att_val == attended_is:
                ax.plot(tt, yy, color=color, linewidth=2.0, linestyle="-")
            else:
                ax.plot(tt, yy, color=color, linewidth=1.2, linestyle="--")

    _plot_att_segments(ax_bot, Awin, attw, 1, BLUE)
    _plot_att_segments(ax_bot, Bwin, attw, 2, ORANGE)
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("Audio envelope (z)")

    legend_handles = [
        Line2D([0], [0], color=BLUE, lw=2.0, ls="-", label="Audio 1 (attended)"),
        Line2D([0], [0], color=BLUE, lw=1.2, ls="--", label="Audio 1 (unattended)"),
        Line2D([0], [0], color=ORANGE, lw=2.0, ls="-", label="Audio 2 (attended)"),
        Line2D([0], [0], color=ORANGE, lw=1.2, ls="--", label="Audio 2 (unattended)"),
    ]
    ax_bot.legend(handles=legend_handles, loc="upper right", ncol=2)
    a_pct = 100.0 * np.mean(attw == 1) if attw.size else 0.0
    b_pct = 100.0 * np.mean(attw == 2) if attw.size else 0.0
    _logger.info(f"Window {t_start:.2f}-{t_end:.2f}s A%={a_pct:.1f} B%={b_pct:.1f}")
    plt.tight_layout()
    out_path = _unique_png(f"S{subj_id:02d}_eeg_audio_{int(t_start * 1000)}ms_{int(t_end * 1000)}ms")
    plt.savefig(out_path, dpi=200)
    plt.close()
    _logger.info(f"Saved plot_window -> {out_path}")


def eeg_and_attended_envelope(mat, which_env="gammatone", num_bands=32, fmin=100.0):
    trials = _iter_trials(mat)
    eeg_parts, att_env_parts = [], []
    for tr in trials:
        Xi = np.asarray(tr["eeg"], float)
        if Xi.ndim != 2:
            raise ValueError("Each trial 'eeg' must be 2D after explosion")
        if Xi.shape[0] > Xi.shape[1]:
            Xi = Xi.T
        n_ch, n_t = Xi.shape
        A = np.asarray(tr.get("wavA", np.zeros(n_t)), float).squeeze()
        B = np.asarray(tr.get("wavB", np.zeros(n_t)), float).squeeze()
        if A.size != n_t:
            A = signal.resample(A, n_t)
        if B.size != n_t:
            B = signal.resample(B, n_t)
        fs = float(tr.get("fsample", tr.get("freq", 0.0)) or 64.0)
        if isinstance(tr.get("fsample", None), dict):
            fs = float(tr["fsample"].get("eeg", tr["fsample"].get("fs", 64.0)))
        if which_env == "gammatone":
            envA, _ = gammatone_hilbert_envelope(A, fs, num_bands=num_bands, fmin=fmin, target_fs=fs, normalize="unit")
            envB, _ = gammatone_hilbert_envelope(B, fs, num_bands=num_bands, fmin=fmin, target_fs=fs, normalize="unit")
        else:
            envA = np.abs(signal.hilbert(A))
            envB = np.abs(signal.hilbert(B))
        att = _build_att_mask(n_t, tr)
        y_att = np.where(att == 1, envA, 0.0) + np.where(att == 2, envB, 0.0)
        eeg_parts.append(Xi.T)
        att_env_parts.append(y_att)
    eeg_all = np.vstack(eeg_parts)
    env_all = np.concatenate(att_env_parts)
    _logger.info(f"eeg_and_attended_envelope -> EEG {eeg_all.shape}, ENV {env_all.shape}")
    return eeg_all, env_all


def check_attention_distribution(preproc_dir, subj_id):
    D = load_subject(preproc_dir, subj_id, drop_exg=True)
    fs = D["fs"]
    lens = D["lengths"]
    att = D["attmask"]
    print(f"S{subj_id}: att==1 (A) {np.mean(att==1):.2%}, att==2 (B) {np.mean(att==2):.2%}, unknown {np.mean(att==0):.2%}")
    start_sample = 0
    current_time = 0.0
    for ti, L in enumerate(lens):
        seg = att[start_sample:start_sample + L]
        start_time = current_time
        end_time = current_time + (L / fs)
        duration = L / fs
        print(f"  trial {ti:02d} | start={start_time:.2f}s | end={end_time:.2f}s | dur={duration:.2f}s | A {np.mean(seg==1):.2%}, B {np.mean(seg==2):.2%}, unknown {np.mean(seg==0):.2%}")
        start_sample += L
        current_time += duration


def plot_eeg_and_audio(eeg_TxC, audio, fs, t_start, t_end, num_bands=32, fs_audio=None,
                       channel_names=None, max_channels=None, env_lowpass_hz=8.0, title=None,
                       out_path=None, figsize=(14, 7)):
    X = np.asarray(eeg_TxC, float)
    T, C = X.shape
    if fs_audio is None:
        fs_audio = fs
    i0 = max(0, int(np.floor(t_start * fs)))
    i1 = min(T, int(np.ceil(t_end * fs)))
    if i1 <= i0:
        raise ValueError("t_end must be > t_start within EEG length")
    Xw = X[i0:i1, :]
    mu = Xw.mean(axis=0, keepdims=True)
    sd = Xw.std(axis=0, keepdims=True) + 1e-12
    Xz = (Xw - mu) / sd
    if max_channels is not None:
        keep_idx = np.arange(min(max_channels, C))
        Xz = Xz[:, keep_idx]
        ch_show = [channel_names[i] for i in keep_idx] if channel_names is not None else [f"Ch{i+1}" for i in keep_idx]
    else:
        ch_show = channel_names if channel_names is not None else [f"Ch{i+1}" for i in range(Xz.shape[1])]
    env, _ = gammatone_hilbert_envelope(np.asarray(audio, float).ravel(), float(fs_audio),
                                        num_bands=num_bands, fmin=50.0, fmax=None, compress="pow",
                                        compress_exp=0.6, aggregate="sum", lowpass_hz=env_lowpass_hz,
                                        target_fs=float(fs), normalize="unit", return_bands=False)
    if env.size != T:
        env = signal.resample(env, T)
    envw = env[i0:i1]
    t = np.arange(i0, i1) / float(fs)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    step = 5.0
    offsets = np.arange(Xz.shape[1])[::-1] * step
    for k in range(Xz.shape[1]):
        ax_top.plot(t, Xz[:, k] + offsets[k], lw=0.6, color="black")
    ax_top.set_yticks(offsets)
    ax_top.set_yticklabels(ch_show)
    ax_top.set_ylabel("EEG (z, stacked)")
    ax_top.set_title(title or f"EEG + Envelope | {t_start:.2f}–{t_end:.2f}s")
    ax_bot.plot(t, envw, lw=1.5)
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("Envelope")
    plt.tight_layout()
    if out_path is None:
        out_path = _unique_png(f"eeg_audio_{int(t_start * 1000)}ms_{int(t_end * 1000)}ms")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    _logger.info(f"Saved plot_eeg_and_audio -> {out_path}")
    return str(out_path)
