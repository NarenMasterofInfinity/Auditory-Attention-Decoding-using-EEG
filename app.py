# streamlit_aad_with_model_render.py
"""
Streamlit AAD viewer + model inference
- Uses helper.subject_eeg_env_ab_aad(preproc_dir, subj_id)
- Loads model from subject folder best_model.pt (AADModel from aad_tcn.py)
- Load -> choose start/end & channels -> press "Render selection" to re-render plots & model predictions
- TRUE (top) and PREDICTED (bottom) plots: envA blue, envB orange; solid line when attended, dotted when not.
- Shows textual sequences (AAABBA...) for true & predicted (window-wise majority).
"""
import streamlit as st
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import io, traceback
import torch
import soundfile as sf
from scipy import signal

# import user's helper
import helper

# import model
try:
    from aad_tcn import AADModel, make_biosemi64_info
except Exception:
    AADModel = None
    make_biosemi64_info = None
    traceback.print_exc()

st.set_page_config(page_title="AAD Viewer — Render Selection", layout="wide")
st.title("AAD Viewer — choose interval → press Render")

# -------------------------
# small utilities
# -------------------------
def audio_to_wav_bytes(x, sr):
    import soundfile as sf, io
    bio = io.BytesIO()
    sf.write(bio, x, int(sr), format="WAV")
    bio.seek(0)
    return bio.read()

def window_indices(T, win, hop):
    idx = []
    t = 0
    while t + win <= T:
        idx.append((t, t + win)); t += hop
    return np.array(idx, dtype=int)

def safe_att_to_AB(arr):
    """Normalize attAB array to 'A'/'B' strings"""
    arr = np.asarray(arr)
    if arr.dtype.kind in ('U','S','O'):
        # already strings maybe; ensure uppercase 'A'/'B'
        out = np.array([str(x).upper()[0] if len(str(x))>0 else 'A' for x in arr])
    else:
        # numeric: assume 1 -> A, 0 -> B (anything >=0.5 -> A)
        out = np.array(['A' if float(x) >= 0.5 else 'B' for x in arr])
    out = np.where(np.isin(out, ['A','B']), out, 'A')
    return out

# -------------------------
# Left panel: controls
# -------------------------
col_ctrl, col_main = st.columns([1,3])
with col_ctrl:
    st.markdown("### Controls")
    base = Path("AAD/outputs_aad")
    if not base.exists():
        st.error(f"Folder {base} not found. Run from repo root or change path.")
        st.stop()

    subjects = sorted([p for p in base.iterdir() if p.is_dir() and p.name.lower().startswith("s")])
    subj_names = [p.name for p in subjects]
    sel = st.selectbox("Select subject folder", subj_names)
    subj_path = base / sel

    preproc_dir = st.text_input("preproc_dir (for helper)", value=str("/home/naren-root/Dataset/DATA_preproc"))
    try:
        subj_id = int(''.join([c for c in sel if c.isdigit()]))
    except Exception:
        subj_id = None

    st.markdown("---")
    st.write("Model checkpoint (expected):")
    pt_path = subj_path / "best_model.pt"


    st.markdown("---")
    win_sec = st.number_input("Model window (s)", value=5.0, min_value=0.5, step=0.5)
    hop_sec = st.number_input("Model hop (s)", value=2.5, min_value=0.1, step=0.1)
    device_opt = st.selectbox("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []), index=0)

    st.markdown("---")
    show_raw_audio = st.checkbox("Enable raw audio playback (search subj folder)", value=False)

    load_btn = st.button("Load data (via helper)")

# -------------------------
# Session state storage for loaded data
# -------------------------
if "loaded" not in st.session_state:
    st.session_state.loaded = False
    st.session_state.eeg = None
    st.session_state.envA = None
    st.session_state.envB = None
    st.session_state.fs = None
    st.session_state.attAB = None
    st.session_state.n_chan = None

# -------------------------
# Load data when user presses Load
# -------------------------
if load_btn:
    try:
        eeg, envA, envB, fs, attAB = helper.subject_eeg_env_ab_aad(preproc_dir, subj_id)
    except Exception as e:
        st.error(f"helper.subject_eeg_env_ab_aad(...) failed: {e}")
        st.stop()
    eeg = np.asarray(eeg, dtype=np.float32)
    envA = np.asarray(envA, dtype=np.float32)
    envB = np.asarray(envB, dtype=np.float32)
    fs = float(fs)
    attAB = safe_att_to_AB(attAB)

    # align lengths
    T = min(len(envA), len(envB), eeg.shape[0], len(attAB))
    envA = envA[:T]; envB = envB[:T]; eeg = eeg[:T]; attAB = attAB[:T]
    if eeg.ndim == 1:
        eeg = eeg[:, None]

    st.session_state.eeg = eeg
    st.session_state.envA = envA
    st.session_state.envB = envB
    st.session_state.fs = fs
    st.session_state.attAB = attAB
    st.session_state.n_chan = eeg.shape[1]
    st.session_state.loaded = True
    st.success(f"Loaded EEG shape {eeg.shape}, env len {len(envA)}, fs={fs}")

# -------------------------
# If not loaded, show message
# -------------------------
with col_main:
    if not st.session_state.loaded:
        st.info("Load data to continue (press 'Load data').")
        st.stop()

    # display total duration and time interval controls
    eeg = st.session_state.eeg
    envA = st.session_state.envA
    envB = st.session_state.envB
    attAB = st.session_state.attAB
    fs = st.session_state.fs
    n_chan = st.session_state.n_chan
    T = len(envA)
    total_sec = T / fs
    st.markdown(f"**Total duration:** {total_sec:.2f} s — channels: {n_chan}")

    # Start/end sliders (explicit render button required)
    start_sec = st.number_input("Start time (s)", min_value=0.0, max_value=max(0.0, total_sec - 0.01), value=0.0, step=0.01, format="%.2f")
    end_sec = st.number_input("End time (s)", min_value=0.01, max_value=total_sec, value=min(total_sec, 10.0), step=0.01, format="%.2f")
    if end_sec <= start_sec:
        st.error("End must be greater than start.")
        st.stop()

    # channel selection
    st.markdown("Channel selection")
    n_plot = st.number_input("Number of channels to plot (first N)", min_value=1, max_value=n_chan, value=min(4, n_chan), step=1)
    explicit = st.multiselect("Or pick exact channels (overrides N)", [f"ch{i}" for i in range(n_chan)], default=[])
    if len(explicit) > 0:
        chosen_idxs = [int(s.replace("ch","")) for s in explicit]
    else:
        chosen_idxs = list(range(int(n_plot)))

    st.write(f"Will plot channels: {chosen_idxs}")

    # Render selection button (explicit re-render)
    render_btn = st.button("Render selection")

    # Optional: show raw audio files for playback
    if show_raw_audio:
        aud_files = sorted(list(subj_path.glob("**/*.wav")) + list(subj_path.glob("**/*.flac")) + list(subj_path.glob("**/*.mp3")))
        if aud_files:
            st.markdown("Raw audio files found (top 3):")
            for a in aud_files[:3]:
                st.write(f"- {a.name}")
                try:
                    data, sr = sf.read(str(a))
                    if data.ndim > 1: data = data.mean(axis=1)
                    st.audio(audio_to_wav_bytes(data, sr), format="audio/wav")
                except Exception as e:
                    st.write(f" playback failed: {e}")

    # Only proceed when user presses render
    if not render_btn:
        st.info("Adjust start/end or channels and press **Render selection** to (re)draw and run model.")
        st.stop()

    # slice selected interval
    s_idx = int(round(start_sec * fs)); e_idx = int(round(end_sec * fs))
    s_idx = max(0, s_idx); e_idx = min(T, e_idx)
    if e_idx <= s_idx:
        st.error("Selected interval too short.")
        st.stop()

    eeg_sel = eeg[s_idx:e_idx, :]
    envA_sel = envA[s_idx:e_idx]
    envB_sel = envB[s_idx:e_idx]
    attAB_sel = attAB[s_idx:e_idx]
    T_sel = len(envA_sel)
    t = np.arange(T_sel) / fs + start_sec

    # Plot TRUE panel (top): envA/envB with solid if attended (from attAB_sel), dotted otherwise
        # -----------------------
    # Plot EEG (selected) + TRUE panel (envA/envB) below it
    # -----------------------
    # eeg_sel: [T_sel, C], envA_sel/envB_sel: [T_sel], t: time axis
    fig_true, (ax_eeg, ax_true) = plt.subplots(2,1, figsize=(12,5), sharex=True,
                                               gridspec_kw={'height_ratios':[1,0.6]})

    # --- EEG plotting (normalize + small vertical offsets so multiple channels are visible) ---
    # chosen_idxs should be defined earlier (from channel selection)
    if len(chosen_idxs) == 0:
        # plot mean of channels
        eeg_mean = np.mean(eeg_sel, axis=1)
        down = max(1, int(fs // 50))  # downsample for plotting density
        ax_eeg.plot(t[::down], eeg_mean[::down], color='black', linewidth=0.7)
        ax_eeg.set_ylabel("EEG (mean)")
    else:
        chs = chosen_idxs
        # compute stds and a sensible offset
        ch_stds = np.std(eeg_sel[:, chs], axis=0)
        max_std = np.max(ch_stds) if ch_stds.size>0 else 1.0
        offset = max_std * 4.0  # spacing between channels
        down = max(1, int(fs // 50))
        for i, ch in enumerate(chs):
            trace = eeg_sel[:, ch]
            # normalize by channel std to make plots comparable
            norm = (trace - np.mean(trace)) / (np.std(trace) + 1e-8)
            ax_eeg.plot(t[::down], (norm[::down] + i*offset), linewidth=0.8, label=f"ch{ch}")
        ax_eeg.set_ylabel("EEG (normalized + offset)")
        # limit number of legend columns to avoid overflow
        ax_eeg.legend(fontsize='small', ncol=min(4, len(chs)))
    ax_eeg.grid(alpha=0.2)

    # --- TRUE envelopes (use same t axis). Solid line where att=='A' or 'B', dotted otherwise ---
    # create masked arrays so solid/dashed segments appear correctly
    envA_solid = np.where(attAB_sel == 'A', envA_sel, np.nan)
    envA_dash  = np.where(attAB_sel != 'A', envA_sel, np.nan)
    envB_solid = np.where(attAB_sel == 'B', envB_sel, np.nan)
    envB_dash  = np.where(attAB_sel != 'B', envB_sel, np.nan)

    ax_true.plot(t, envA_dash, color='tab:blue', linestyle=':', label='envA (not attended)')
    ax_true.plot(t, envA_solid, color='tab:blue', linestyle='-', linewidth=1.2, label='envA (attended)')
    ax_true.plot(t, envB_dash, color='tab:orange', linestyle=':', label='envB (not attended)')
    ax_true.plot(t, envB_solid, color='tab:orange', linestyle='-', linewidth=1.2, label='envB (attended)')

    ax_true.set_title("TRUE attend (per-sample) — blue=A, orange=B — solid = attended")
    ax_true.set_ylabel("Envelope")
    ax_true.grid(alpha=0.2)
    ax_true.legend(fontsize='small', ncol=2)

    st.pyplot(fig_true)


    # Run model on selected interval (sliding windows) to get predicted labels
    if AADModel is None:
        st.warning("AADModel not available (could not import). Predictions disabled.")
        # Still display true textual sequence
        st.markdown("**True window-wise sequence (no predictions)**")
        st.text("Model not available.")
    else:
        # load checkpoint
        if not pt_path.exists():
            st.error(f"Checkpoint {pt_path} not found. Cannot run model.")
        else:
            device = torch.device(device_opt)
            n_ch = eeg_sel.shape[1]
            # create model with plausible params (match training if needed)
            # --- Build a valid `pos` array (shape [n_ch, 3], dtype float32) ---
            pos = None
            if 'make_biosemi64_info' in globals() and make_biosemi64_info is not None:
                try:
                    # try calling with arguments (some implementations accept n_ch, sfreq)
                    info, ch_names, pos = make_biosemi64_info(n_ch=n_ch, sfreq=fs)
                except TypeError:
                    try:
                        info, ch_names, pos = make_biosemi64_info()
                    except Exception:
                        pos = None
                except Exception:
                    pos = None

            # Fallback: place channels evenly on a circle (x,y,0)
            if pos is None:
                theta = np.linspace(0, 2 * np.pi, n_ch, endpoint=False)
                x = np.cos(theta)
                y = np.sin(theta)
                z = np.zeros_like(x)
                pos = np.stack([x, y, z], axis=1).astype(np.float32)

            # Ensure pos has shape (n_ch, 3)
            pos = np.asarray(pos, dtype=np.float32)
            if pos.ndim == 2 and pos.shape[0] == n_ch:
                if pos.shape[1] == 2:
                    pos = np.concatenate([pos, np.zeros((n_ch, 1), dtype=np.float32)], axis=1)
            elif pos.shape != (n_ch, 3):
                # fallback to zeros if shape is weird
                pos = np.zeros((n_ch, 3), dtype=np.float32)

            # --- Instantiate model with pos ---
            model = AADModel(n_ch=n_ch, pos=pos, d_model=128, d_audio=64, L=3, k=8, heads_graph=4, heads_xattn=4, dropout=0.1)
            model.to(device)

            model.to(device)
            try:
                sd = torch.load(str(pt_path), map_location='cpu')
                if isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values()):
                    model.load_state_dict(sd, strict=True)
                else:
                    if 'model' in sd and isinstance(sd['model'], dict):
                        model.load_state_dict(sd['model'], strict=True)
                    elif 'state_dict' in sd:
                        model.load_state_dict(sd['state_dict'], strict=False)
                    else:
                        model.load_state_dict(sd, strict=False)
            except Exception as e:
                st.error(f"Failed to load checkpoint: {e}")
                st.stop()

            model.eval()
            win = int(round(win_sec * fs))
            hop = int(round(hop_sec * fs))
            spans = window_indices(T_sel, win, hop)
            if spans.shape[0] == 0:
                st.error("Selected interval shorter than model window. Increase interval or reduce window size.")
                st.stop()

            probs_list = []
            with torch.no_grad():
                for (a,b) in spans:
                    eeg_win = eeg_sel[a:b]         # [W, C]
                    envA_win = envA_sel[a:b]
                    envB_win = envB_sel[a:b]
                    xb = torch.from_numpy(eeg_win[None,:,:]).to(device)        # [1, W, C]
                    a_t = torch.from_numpy(envA_win[None,:]).to(device)        # [1, W]
                    b_t = torch.from_numpy(envB_win[None,:]).to(device)
                    try:
                        logits, _ = model(xb, a_t, b_t)
                        prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # [2] idx0->B idx1->A
                    except Exception as e:
                        st.error(f"Model forward failed on a window: {e}")
                        prob = np.array([0.5,0.5])
                    probs_list.append(prob)
            probs_all = np.stack(probs_list, axis=0)  # [n_windows, 2]
            pA = probs_all[:,1]
            pred_labels = (pA >= 0.5).astype(int)  # 1->A, 0->B

            # Build predicted per-sample label array by assigning each window's label to its span (last-wins on overlap)
            pred_sample = np.full((T_sel,), fill_value='B', dtype='<U1')
            for i,(a,b) in enumerate(spans):
                ch = 'A' if pred_labels[i]==1 else 'B'
                pred_sample[a:b] = ch

            # Build true window-wise majority labels for text sequence
            true_win_labels = []
            for (a,b) in spans:
                seg = attAB_sel[a:b]
                a_count = np.sum(seg == 'A')
                lab = 'A' if a_count >= ((b-a)/2.0) else 'B'
                true_win_labels.append(lab)
            true_seq = ''.join(true_win_labels)

            pred_seq = ''.join(['A' if x==1 else 'B' for x in pred_labels])

            # Plot PREDICTED panel (bottom): envA/envB colored with style from pred_sample
            fig_pred, ax_pred = plt.subplots(1,1, figsize=(12,2.8))
            predA_solid = np.where(pred_sample == 'A', envA_sel, np.nan)
            predA_dash  = np.where(pred_sample != 'A', envA_sel, np.nan)
            predB_solid = np.where(pred_sample == 'B', envB_sel, np.nan)
            predB_dash  = np.where(pred_sample != 'B', envB_sel, np.nan)

            ax_pred.plot(t, predA_dash, color='tab:blue', linestyle=':', label='envA (predicted not attended)')
            ax_pred.plot(t, predA_solid, color='tab:blue', linestyle='-', linewidth=1.2, label='envA (predicted attended)')
            ax_pred.plot(t, predB_dash, color='tab:orange', linestyle=':', label='envB (predicted not attended)')
            ax_pred.plot(t, predB_solid, color='tab:orange', linestyle='-', linewidth=1.2, label='envB (predicted attended)')

            ax_pred.set_title("PREDICTED attend (window-wise mapped to samples) — solid = model says attended")
            ax_pred.set_ylabel("Envelope")
            ax_pred.grid(alpha=0.2)
            ax_pred.legend(fontsize='small', ncol=2)
            st.pyplot(fig_pred)

            # Show textual sequences
            st.markdown("### Window-wise sequences (majority per model window)")
            st.write(f"**True:**      {true_seq}")
            st.write(f"**Predicted:** {pred_seq}")

            # Window-wise accuracy
            true_bin = np.array([1 if x=='A' else 0 for x in true_win_labels])
            pred_bin = pred_labels
            acc = (true_bin == pred_bin).mean() if len(true_bin)>0 else float('nan')
            st.write(f"Window-wise accuracy on selection: **{acc*100:.2f}%**  (n_windows={len(true_bin)})")

            # Also show P(att=A) timeline as small plot
            fig_p, axp = plt.subplots(1,1, figsize=(12,2))
            axp.plot(np.arange(len(pA)), pA, label='P(att=A)', color='tab:green')
            axp.set_xlabel("window index")
            axp.set_ylabel("P(att=A)")
            axp.grid(alpha=0.2)
            st.pyplot(fig_p)
