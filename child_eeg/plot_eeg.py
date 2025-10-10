# compare_preproc_diffs.py
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA, create_eog_epochs

# -----------------------------
# Helper preprocessing routines
# -----------------------------
def general_preproc_raw(raw_in):
    """General pipeline directly on a copy of the input Raw; returns a new Raw at 64 Hz."""
    raw = raw_in.copy().load_data()
    # Montage (robust)
    try:
        raw.set_montage("GSN-HydroCel-129", on_missing="ignore")
    except Exception:
        pass
    # Resample first (simple)
    raw.resample(64.0)
    # Band-limit + reference
    raw.filter(0.5, 30.0, fir_design='firwin', phase='zero')
    raw.set_eeg_reference('average', projection=False)
    # ICA (light)
    try:
        ica = ICA(n_components=20, random_state=42, max_iter='auto')
        ica.fit(raw)
        try:
            eog_inds, _ = ica.find_bads_eog(raw)
            if eog_inds: ica.exclude.extend(eog_inds)
        except Exception:
            pass
        try:
            ecg_inds, _ = ica.find_bads_ecg(raw, ch_name="Fz")
            if ecg_inds: ica.exclude.extend(ecg_inds)
        except Exception:
            pass
        raw = ica.apply(raw)
    except Exception:
        pass
    return raw

def find_flat_channels_by_std(raw, thresh=1e-6):
    data = raw.get_data(picks=mne.pick_types(raw.info, eeg=True))
    chs = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    stds = np.std(data, axis=1)
    return [ch for ch, s in zip(chs, stds) if s < thresh]

def child_preproc_raw(raw_in, hp=0.5, lp=30.0, notch=50.0, ica_var=0.99, ica_decim=3):
    """Lean child pipeline; returns a new Raw at 64 Hz."""
    raw = raw_in.copy().load_data()
    native_fs = float(raw.info['sfreq'])
    # Montage (robust)
    try:
        raw.set_montage("GSN-HydroCel-129", on_missing="ignore")
    except Exception:
        pass
    # Band-limit at native fs
    raw.filter(hp, lp, fir_design='firwin', phase='zero')
    if notch is not None:
        raw.notch_filter(freqs=[notch], fir_design='firwin')
    # Conservative bad-channel handling: flat only
    flats = find_flat_channels_by_std(raw, thresh=1e-6)
    if flats:
        raw.info['bads'] = sorted(set(flats))
        raw.interpolate_bads(reset_bads=True)
    # Average reference AFTER interpolation
    raw.set_eeg_reference('average', projection=False)
    # ICA at native fs on 1–40 Hz copy (decimated)
    try:
        ica_raw = raw.copy().filter(1., 40., fir_design='firwin', phase='zero')
        ica = ICA(n_components=ica_var, method='fastica', random_state=42, max_iter='auto')
        ica.fit(ica_raw, decim=ica_decim)
        exclude = []
        # Frontal surrogates for EOG
        frontal = [ch for ch in raw.ch_names if ch in ('Fp1','Fp2','AF7','AF8','AF3','AF4')]
        if frontal:
            try:
                eog_epochs = create_eog_epochs(raw, ch_name=frontal)
                eog_inds, _ = ica.find_bads_eog(eog_epochs)
                exclude.extend(eog_inds)
            except Exception:
                # per-channel fallback
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
        pass
    # Resample AFTER cleaning
    if abs(native_fs - 64.0) > 1e-6:
        raw.resample(64.0)
    return raw

# -----------------------------
# Metrics & plots
# -----------------------------
def band_power_welch(x, sf, f_lo, f_hi, fmax_pad=60.0):
    """
    Compute band power via Welch on short windows safely.
    x: [T, C] (or [T]) window; returns (band_power, freqs, psd_mean)
    """
    from mne.time_frequency import psd_array_welch
    x = x if x.ndim == 2 else x[:, None]  # [T, C]
    n_times = x.shape[0]

    # Choose a valid n_per_seg <= n_times (power-of-two when possible).
    if n_times < 32:
        n_per_seg = n_times
    else:
        n_per_seg = 2 ** int(np.floor(np.log2(min(n_times, 1024))))
    n_per_seg = max(16, min(n_per_seg, n_times))
    n_fft = n_per_seg
    n_overlap = n_per_seg // 2 if n_per_seg >= 32 else 0

    # Respect Nyquist
    fmax = min(fmax_pad, sf / 2.0 - 1e-6)

    psd, freqs = psd_array_welch(
        x.T, sf,
        fmin=0.0, fmax=fmax,
        n_per_seg=n_per_seg,
        n_overlap=n_overlap,
        n_fft=n_fft,
        average='mean'
    )
    psd_mean = psd.mean(axis=0)  # avg across channels

    # Integrate requested band
    lo = max(f_lo, 0.0)
    hi = min(f_hi, fmax)
    if hi <= lo or len(freqs) == 0:
        return 0.0, freqs, psd_mean
    idx = (freqs >= lo) & (freqs <= hi)
    band_pow = float(np.trapz(psd_mean[idx], freqs[idx]))
    return band_pow, freqs, psd_mean


def common_picks(raw, max_ch=8):
    """Pick the same EEG channels by name; prefer Fp1, Fp2, Cz, Pz if present."""
    preferred = ['Fp1','Fp2','Fz','Cz','Pz','Oz','T7','T8','P3','P4','C3','C4','F3','F4']
    have = [ch for ch in preferred if ch in raw.ch_names]
    if len(have) < max_ch:
        others = [ch for ch in raw.ch_names if ch not in have]
        have = have + others[:max_ch-len(have)]
    return have[:max_ch]

def extract_window(raw, t0, dur, picks):
    sf = raw.info['sfreq']
    a = int(round(t0 * sf))
    b = int(round((t0 + dur) * sf))
    a = max(0, min(a, raw.n_times-2))
    b = max(a+1, min(b, raw.n_times))
    X = raw.get_data(picks=mne.pick_channels(raw.ch_names, include=picks))[:, a:b].T  # [T, C]
    t = np.arange(a, b) / sf
    return t, X, sf

def plot_traces(ax, t, X, title, ylims=None):
    # no z-scoring: preserve amplitude differences
    C = X.shape[1]
    spacing = np.nanpercentile(np.abs(X), 95) * 2.5  # robust spacing
    for i in range(C):
        ax.plot(t, X[:, i] + i * spacing, lw=0.8)
    yticks = [i * spacing for i in range(C)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"Ch{i+1}" for i in range(C)])
    ax.set_title(title)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("Time (s)")
    if ylims:
        ax.set_ylim(*ylims)
    ax.grid(True, alpha=0.2)

def overlay_psd(ax, traces, sfs, labels, fmax=60.0):
    from mne.time_frequency import psd_array_welch
    for (X, sf, lab) in zip(traces, sfs, labels):
        if X.ndim == 1:
            X = X[:, None]  # [T,1]
        n_times = X.shape[0]
        # pick a valid segment length <= n_times (power of 2 when possible)
        if n_times < 32:
            n_per_seg = n_times
        else:
            n_per_seg = 2 ** int(np.floor(np.log2(min(n_times, 1024))))
        n_per_seg = max(16, min(n_per_seg, n_times))
        n_fft = n_per_seg
        n_overlap = n_per_seg // 2 if n_per_seg >= 32 else 0

        psd, freqs = psd_array_welch(
            X.T, sf,
            fmin=0.0, fmax=fmax,
            n_per_seg=n_per_seg,
            n_overlap=n_overlap,
            n_fft=n_fft,
            average='mean'
        )
        ax.semilogy(freqs, psd.mean(axis=0), label=lab)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("PSD overlay (avg across displayed channels)")
    ax.grid(True, alpha=0.2)
    ax.legend()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg_file", required=True, help=".set EEG file")
    ap.add_argument("--t_start", type=float, default=60.0)
    ap.add_argument("--t_dur", type=float, default=5.0)
    ap.add_argument("--max_ch", type=int, default=6)
    ap.add_argument("--outdir", default="viz_diffs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Load RAW once ----
    raw0 = mne.io.read_raw_eeglab(args.eeg_file, preload=True, verbose="ERROR")
    # Ensure EEG-only
    picks_eeg = mne.pick_types(raw0.info, eeg=True)
    raw0.pick(picks_eeg)

    # ---- Build processed versions on the SAME channel order ----
    raw_gen  = general_preproc_raw(raw0)
    raw_child = child_preproc_raw(raw0)

    # ---- Pick common channels by NAME (same order across all three) ----
    pick_names = common_picks(raw0, max_ch=args.max_ch)

    # ---- Extract identical time windows ----
    t_raw,   X_raw,   fs_raw   = extract_window(raw0,     args.t_start, args.t_dur, pick_names)
    t_gen,   X_gen,   fs_gen   = extract_window(raw_gen,  args.t_start, args.t_dur, pick_names)
    t_child, X_child, fs_child = extract_window(raw_child,args.t_start, args.t_dur, pick_names)

    # ---- Figure 1: Stacked traces (no z-score, fixed scale) ----
    fig1, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    # Fix y-lims across panels for apples-to-apples
    # Estimate a shared amplitude scale from RAW
    amp95 = np.nanpercentile(np.abs(X_raw), 95)
    ylim = (-amp95 * 6, amp95 * (args.max_ch) * 2.5 + amp95 * 2)

    plot_traces(axes[0], t_raw,   X_raw,   f"RAW (fs={fs_raw:.1f} Hz) | ch={pick_names}", ylims=None)
    plot_traces(axes[1], t_gen,   X_gen,   f"GENERAL (fs={fs_gen:.1f} Hz) | same channels", ylims=None)
    plot_traces(axes[2], t_child, X_child, f"CHILD (fs={fs_child:.1f} Hz) | same channels", ylims=None)

    plt.tight_layout()
    p1 = os.path.join(args.outdir, "01_traces_raw_gen_child.png")
    plt.savefig(p1, dpi=150); plt.close(fig1)

    # ---- Figure 2: PSD overlays on the same displayed channels ----
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    overlay_psd(ax2,
                traces=[X_raw, X_gen, X_child],
                sfs=[fs_raw, fs_gen, fs_child],
                labels=["RAW", "GENERAL", "CHILD"],
                fmax=60.0)
    p2 = os.path.join(args.outdir, "02_psd_overlay.png")
    plt.tight_layout(); plt.savefig(p2, dpi=150); plt.close(fig2)

    # ---- Figure 3: Difference traces (what each pipeline removed) for the first channel ----
    ch0 = 0  # first displayed channel
    # resample RAW channel to each fs to compute diff fairly
    raw_ch = X_raw[:, ch0]
    # Interpolate raw to general/child time bases
    raw_to_gen   = np.interp(t_gen,   t_raw,   raw_ch)
    raw_to_child = np.interp(t_child, t_raw,   raw_ch)

    diff_gen   = raw_to_gen   - X_gen[:, ch0]
    diff_child = raw_to_child - X_child[:, ch0]

    fig3, ax3 = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    ax3[0].plot(t_gen,   diff_gen,   lw=0.8); ax3[0].set_title("RAW − GENERAL (first channel)")
    ax3[1].plot(t_child, diff_child, lw=0.8); ax3[1].set_title("RAW − CHILD (first channel)")
    for ax in ax3:
        ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.2)
    p3 = os.path.join(args.outdir, "03_difference_traces.png")
    plt.tight_layout(); plt.savefig(p3, dpi=150); plt.close(fig3)

    # ---- Numeric metrics: 50 Hz and 30–45 Hz band power (avg across displayed channels) ----
    # (Use entire displayed window)
    bp_50_raw,  fR, pR  = band_power_welch(X_raw,   fs_raw,   49.0, 51.0)
    bp_50_gen,  fG, pG  = band_power_welch(X_gen,   fs_gen,   49.0, 51.0)
    bp_50_child,fC, pC  = band_power_welch(X_child, fs_child, 49.0, 51.0)

    bp_hf_raw,  _, _ = band_power_welch(X_raw,   fs_raw,   30.0, 45.0)
    bp_hf_gen,  _, _ = band_power_welch(X_gen,   fs_gen,   30.0, 45.0)
    bp_hf_child,_, _ = band_power_welch(X_child, fs_child, 30.0, 45.0)

    lines = [
        f"Channels displayed: {pick_names}",
        f"Window: {args.t_start:.2f}–{args.t_start+args.t_dur:.2f}s",
        "",
        f"50 Hz band power (avg across channels):",
        f"  RAW   : {bp_50_raw:.4e}",
        f"  GENERAL: {bp_50_gen:.4e} (Δ={bp_50_raw-bp_50_gen:+.4e})",
        f"  CHILD : {bp_50_child:.4e} (Δ={bp_50_raw-bp_50_child:+.4e})",
        "",
        f"30–45 Hz band power (avg across channels):",
        f"  RAW   : {bp_hf_raw:.4e}",
        f"  GENERAL: {bp_hf_gen:.4e} (Δ={bp_hf_raw-bp_hf_gen:+.4e})",
        f"  CHILD : {bp_hf_child:.4e} (Δ={bp_hf_raw-bp_hf_child:+.4e})",
        "",
        f"Saved:\n  {p1}\n  {p2}\n  {p3}",
    ]
    report_path = os.path.join(args.outdir, "00_metrics.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nMetrics saved to: {report_path}")

if __name__ == "__main__":
    main()
