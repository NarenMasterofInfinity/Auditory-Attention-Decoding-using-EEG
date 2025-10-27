# app.py — EEG ↔ Audio Demo (AAD & AEE) — interactive plots + AEE Pearson r per window
from __future__ import annotations
from pathlib import Path
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Plotly for interactive charts
import plotly.express as px
import plotly.graph_objects as go

# Optional deps
try:
    from scipy import signal
except Exception:
    signal = None
try:
    import soundfile as sf
except Exception:
    sf = None

# ==== Helpers ====
# AAD: helper.subject_eeg_env_ab_aad(preproc_dir, subj_id) -> (eeg, envA, envB, fs, attAB)
# AEE: helper.subject_eeg_env_ab(preproc_dir, subj_id) -> (eeg, env, fs, attAB)
import helper

# ==== AAD model (optional) ====
AADModel = None
make_biosemi64_info_aad = None
try:
    from aad_tcn import AADModel, make_biosemi64_info as make_biosemi64_info_aad
except Exception:
    pass

# ==== AEE model (exact import as requested) ====
ERGraphModel = None
make_biosemi64_info_aee = None
try:
    from TestingGraphMemEfficient import ERGraphModel, make_biosemi64_info as make_biosemi64_info_aee
except Exception:
    pass

# -----------------------------------------------------------------------------------
# Streamlit page
# -----------------------------------------------------------------------------------
st.set_page_config(page_title="EEG ↔ Audio — AAD & AEE", layout="wide")
st.title("EEG ↔ Audio — Auditory Attention (AAD) & Envelope (AEE)")

# -----------------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------------
def natural_key(s: str):
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s)))

def window_indices(T: int, win: int, hop: int) -> np.ndarray:
    idx, t = [], 0
    while t + win <= T:
        idx.append((t, t + win))
        t += hop
    return np.array(idx, dtype=int)

def safe_att_to_AB(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.kind in ("U", "S", "O"):
        out = np.array([str(x).upper()[0] if len(str(x)) > 0 else "A" for x in arr])
    else:
        out = np.array(["A" if float(x) >= 0.5 else "B" for x in arr])
    return np.where(np.isin(out, ["A", "B"]), out, "A")

def list_window_folders(base: Path) -> list[str]:
    if not base.exists(): return []
    wins = [p.name for p in base.iterdir() if p.is_dir()]
    prefer = ["outputs_aad_5s", "outputs_aad_1s", "outputs_aad_0_1s"]
    return [w for w in prefer if w in wins] + [w for w in sorted(wins) if w not in prefer]

def list_subjects(folder: Path) -> list[str]:
    if not folder.exists(): return []
    subs = [p.name for p in folder.iterdir() if p.is_dir() and p.name.lower().startswith("s")]
    return sorted(subs, key=natural_key)

def read_aad_summary(win_dir: Path):
    for name in ("summary.csv", "summary_aad.csv", "summary_results.csv"):
        p = win_dir / name
        if p.exists():
            try:
                df = pd.read_csv(p)
            except Exception:
                try:
                    df = pd.read_csv(p, sep="\t")
                except Exception:
                    return None
            cols = {c.lower(): c for c in df.columns}
            subj = next((cols[k] for k in ["subject","subj","sid","id","participant"] if k in cols), None)
            if subj is None:
                df = df.copy(); df["subject"] = [f"S{str(i+1).zfill(2)}" for i in range(len(df))]; subj="subject"
            acc = next((cols[k] for k in ["accuracy","acc","test_acc","val_acc","score"] if k in cols), None)
            if acc is None:
                num = df.select_dtypes(include=[np.number]).columns.tolist()
                if subj in num: num.remove(subj)
                if not num: return None
                acc = num[0]
            out = df[[subj, acc]].copy(); out.columns = ["subject","accuracy"]
            return out
    return None

def audio_to_wav_bytes(x: np.ndarray, sr: float) -> bytes:
    import soundfile as _sf, io as _io
    bio = _io.BytesIO()
    _sf.write(bio, x.astype(np.float32), int(sr), format="WAV")
    bio.seek(0)
    return bio.read()

def zscore(x, axis=0, eps=1e-8):
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True) + eps
    return (x - m) / s

def pearson_r(a: np.ndarray, b: np.ndarray, eps=1e-8) -> float:
    a = a - a.mean(); b = b - b.mean()
    num = (a*b).sum()
    den = np.sqrt((a*a).sum() * (b*b).sum()) + eps
    return float(num/den)

# -----------------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------------
if "aad_loaded" not in st.session_state:
    st.session_state.aad_loaded = False
    st.session_state.eeg = None
    st.session_state.envA = None
    st.session_state.envB = None
    st.session_state.fs = None
    st.session_state.attAB = None
    st.session_state.n_chan = None

if "aee_loaded" not in st.session_state:
    st.session_state.aee_loaded = False
    st.session_state.eeg_aee = None
    st.session_state.env_true = None
    st.session_state.fs_aee = None
    st.session_state.attAB_aee = None
    st.session_state.n_chan_aee = None

# -----------------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------------
tab_aad, tab_aee = st.tabs(["AAD", "AEE"])

# ===================================================================================
# AAD
# ===================================================================================
with tab_aad:
    st.subheader("Auditory Attention Decoding (AAD)")
    t_demo, t_results = st.tabs(["Demo", "Results"])

    # ---------------- DEMO ----------------
    with t_demo:
        c_left, c_right = st.columns([1,3])

        with c_left:
            base = Path(st.text_input("AAD base folder", value=str(Path("AAD").as_posix()), key="aad_base"))
            windows = list_window_folders(base)
            win_name = st.selectbox("Window folder", windows if windows else ["<none>"], key="aad_win")
            subj_list = list_subjects(base / win_name) if windows else []
            subj_name = st.selectbox("Subject", subj_list if subj_list else ["<none>"], key="aad_subj")
            preproc = st.text_input("preproc_dir (helper)", value=str("/home/naren-root/Dataset/DATA_preproc"), key="aad_preproc")
            try:
                subj_id = int("".join([c for c in subj_name if c.isdigit()])) if subj_name and subj_name!="<none>" else None
            except Exception:
                subj_id = None
            ckpt_path = (base / win_name / subj_name / "best_model.pt") if subj_name and subj_name!="<none>" else None
            st.caption("Checkpoint path (AAD)"); st.code(str(ckpt_path) if ckpt_path else "<invalid>", language="text")

            st.markdown("---")
            win_sec = st.number_input("Model window (s)", value=5.0, min_value=0.5, step=0.5, key="aad_winsec")
            hop_sec = st.number_input("Model hop (s)", value=2.5, min_value=0.1, step=0.1, key="aad_hopsec")
            device_opt = st.selectbox("Device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []), index=0, key="aad_dev")
            show_audio = st.checkbox("Show raw audio if found", value=False, key="aad_audio")
            load_btn = st.button("Load data (AAD)", key="aad_load")

        if load_btn:
            if subj_id is None:
                st.error("Could not parse subject id.")
            else:
                try:
                    eeg, envA, envB, fs, attAB = helper.subject_eeg_env_ab_aad(preproc, subj_id)
                except Exception as e:
                    st.error(f"Load failed: {e}"); st.stop()
                eeg = np.asarray(eeg, dtype=np.float32)
                envA = np.asarray(envA, dtype=np.float32)
                envB = np.asarray(envB, dtype=np.float32)
                fs = float(fs); attAB = safe_att_to_AB(attAB)
                T = min(len(envA), len(envB), eeg.shape[0], len(attAB))
                envA, envB, eeg, attAB = envA[:T], envB[:T], eeg[:T], attAB[:T]
                if eeg.ndim==1: eeg = eeg[:, None]
                st.session_state.update(dict(
                    aad_loaded=True, eeg=eeg, envA=envA, envB=envB, fs=fs, attAB=attAB, n_chan=eeg.shape[1]
                ))
                st.success(f"Loaded EEG {eeg.shape}, envelopes {len(envA)}, fs={fs}")

        with c_right:
            if not st.session_state.aad_loaded:
                st.info("Load data to continue.")
            else:
                eeg = st.session_state.eeg; envA = st.session_state.envA; envB = st.session_state.envB
                attAB = st.session_state.attAB; fs = st.session_state.fs; n_chan = st.session_state.n_chan
                total_sec = len(envA)/fs
                st.markdown(f"**Total duration:** {total_sec:.2f} s — **Channels:** {n_chan}")

                c1,c2 = st.columns(2)
                with c1:
                    start_sec = st.number_input("Start (s)", 0.0, max(0.0,total_sec-0.01), 0.0, 0.01, format="%.2f", key="aad_start")
                with c2:
                    end_sec = st.number_input("End (s)", 0.01, total_sec, min(10.0,total_sec), 0.01, format="%.2f", key="aad_end")
                if end_sec <= start_sec: st.error("End must be greater than start."); st.stop()

                n_plot = st.number_input("Plot first N channels", 1, n_chan, min(6,n_chan), 1, key="aad_nplot")
                explicit = st.multiselect("Or pick channels", [f"ch{i}" for i in range(n_chan)], default=[], key="aad_pick")
                ch_idx = [int(s[2:]) for s in explicit] if explicit else list(range(int(n_plot)))

                render_btn = st.button("Render & Predict (AAD)", key="aad_render")

                # Interactive EEG stack (z-scored) + TRUE envelopes
                if render_btn:
                    s_idx = int(round(start_sec*fs)); e_idx = int(round(end_sec*fs))
                    s_idx = max(0,s_idx); e_idx = min(len(envA), e_idx)
                    if e_idx <= s_idx: st.error("Interval too short."); st.stop()
                    eeg_sel = eeg[s_idx:e_idx,:]; envA_sel = envA[s_idx:e_idx]; envB_sel = envB[s_idx:e_idx]; attAB_sel = attAB[s_idx:e_idx]
                    T_sel = len(envA_sel); t = np.arange(T_sel)/fs + start_sec

                    # EEG stack (interactive)
                    ds = max(1, int(fs//200))
                    show = {}
                    for i, ch in enumerate(ch_idx):
                        show[f"ch{ch}"] = zscore(eeg_sel[::ds, ch], axis=0) + i*6.0
                    df_eeg = pd.DataFrame(show, index=np.round(t[::ds], 3))
                    fig_eeg = px.line(df_eeg, labels={"index":"Time (s)", "value":"EEG (z, stacked)"}, title="EEG selection (stacked z-score)")
                    st.plotly_chart(fig_eeg, use_container_width=True, theme="streamlit")

                    # TRUE envelopes (solid when attended per-sample)
                    envA_att = np.where(attAB_sel=="A", envA_sel, np.nan)
                    envA_not = np.where(attAB_sel!="A", envA_sel, np.nan)
                    envB_att = np.where(attAB_sel=="B", envB_sel, np.nan)
                    envB_not = np.where(attAB_sel!="B", envB_sel, np.nan)
                    fig_true = go.Figure()
                    fig_true.add_trace(go.Scatter(x=t, y=envA_att, name="envA (att)", line=dict(width=2)))
                    fig_true.add_trace(go.Scatter(x=t, y=envA_not, name="envA (not)", line=dict(dash="dot")))
                    fig_true.add_trace(go.Scatter(x=t, y=envB_att, name="envB (att)", line=dict(width=2, color="#F18F01")))
                    fig_true.add_trace(go.Scatter(x=t, y=envB_not, name="envB (not)", line=dict(dash="dot", color="#F18F01")))
                    fig_true.update_layout(title="TRUE attended envelopes", xaxis_title="Time (s)", yaxis_title="Envelope", legend_title="")
                    st.plotly_chart(fig_true, use_container_width=True, theme="streamlit")

                    # Predictions
                    if AADModel is None or ckpt_path is None or not ckpt_path.exists():
                        st.warning("AAD prediction unavailable (missing model or checkpoint).")
                    else:
                        device = torch.device(st.session_state.get("aad_dev","cpu"))
                        n_ch = eeg_sel.shape[1]
                        pos = None
                        if make_biosemi64_info_aad is not None:
                            try:
                                _, _, pos = make_biosemi64_info_aad(n_ch=n_ch, sfreq=fs)
                            except Exception:
                                pos = None
                        if pos is None:
                            theta = np.linspace(0, 2*np.pi, n_ch, endpoint=False)
                            pos = np.stack([np.cos(theta), np.sin(theta), np.zeros(n_ch)], axis=1).astype(np.float32)

                        model = AADModel(n_ch=n_ch, pos=pos, d_model=128, d_audio=64, L=3, k=8,
                                         heads_graph=4, heads_xattn=4, dropout=0.1).to(device)
                        # load
                        try:
                            sd = torch.load(str(ckpt_path), map_location="cpu")
                            if isinstance(sd, dict) and all(isinstance(v, torch.Tensor) for v in sd.values()):
                                model.load_state_dict(sd, strict=True)
                            elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
                                model.load_state_dict(sd["model"], strict=True)
                            elif isinstance(sd, dict) and "state_dict" in sd:
                                model.load_state_dict(sd["state_dict"], strict=False)
                            else:
                                model.load_state_dict(sd, strict=False)
                        except Exception as e:
                            st.error(f"Failed to load AAD weights: {e}"); st.stop()

                        model.eval()
                        win = int(round(win_sec*fs)); hop = int(round(hop_sec*fs))
                        spans = window_indices(T_sel, win, hop)
                        if spans.shape[0]==0: st.error("Interval shorter than model window."); st.stop()

                        probs=[]
                        with torch.no_grad():
                            for (a,b) in spans:
                                xb = torch.from_numpy(eeg_sel[a:b][None,:,:]).to(device)
                                a_t = torch.from_numpy(envA_sel[a:b][None,:]).to(device)
                                b_t = torch.from_numpy(envB_sel[a:b][None,:]).to(device)
                                try:
                                    logits, _ = model(xb, a_t, b_t)
                                    p = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                                except Exception:
                                    p = np.array([0.5,0.5])
                                probs.append(p)
                        probs = np.stack(probs,0); pA = probs[:,1]; pred_labels = (pA>=0.5).astype(int)
                        pred_sample = np.full((T_sel,), "B", dtype="<U1")
                        for i,(a,b) in enumerate(spans): pred_sample[a:b] = "A" if pred_labels[i]==1 else "B"
                        true_win = ["A" if np.sum(attAB_sel[a:b]=="A") >= (b-a)/2 else "B" for (a,b) in spans]
                        true_seq = "".join(true_win); pred_seq = "".join(["A" if x==1 else "B" for x in pred_labels])

                        # Predicted envelope highlights (interactive)
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(x=t, y=np.where(pred_sample=="A", envA_sel, np.nan),
                                                      name="envA (pred att)", line=dict(width=2)))
                        fig_pred.add_trace(go.Scatter(x=t, y=np.where(pred_sample=="B", envB_sel, np.nan),
                                                      name="envB (pred att)", line=dict(width=2, color="#F18F01")))
                        fig_pred.update_layout(title="Predicted attended envelopes (solid = attended)",
                                               xaxis_title="Time (s)", yaxis_title="Envelope", legend_title="")
                        st.plotly_chart(fig_pred, use_container_width=True, theme="streamlit")

                        st.markdown("### Window-wise sequences")
                        st.code(f"TRUE:      {true_seq}\nPREDICTED: {pred_seq}")
                        acc = (np.array([1 if x=="A" else 0 for x in true_win]) == pred_labels).mean() if len(true_win)>0 else float("nan")
                        st.success(f"Window-wise accuracy on selection: {acc*100:.2f}%  (n_windows={len(true_win)})")

                        # Probability timeline (interactive)
                        fig_p = px.line(x=np.arange(len(pA)), y=pA, labels={"x":"Window index", "y":"P(att=A)"},
                                        title="P(att=A) per window")
                        fig_p.update_yaxes(range=[0,1])
                        st.plotly_chart(fig_p, use_container_width=True, theme="streamlit")

                # Optional audio previews
                if show_audio and sf is not None and subj_name and subj_name!="<none>":
                    subj_path = base / win_name / subj_name
                    if subj_path.exists():
                        files = list(subj_path.glob("*.wav")) + list(subj_path.glob("*.flac")) + list(subj_path.glob("*.mp3"))
                        for a in files[:2]:
                            try:
                                data, sr = sf.read(str(a))
                                if data.ndim>1: data=data.mean(1)
                                st.caption(a.name); st.audio(audio_to_wav_bytes(data, sr))
                            except Exception: pass

    # ---------------- RESULTS ----------------
    with t_results:
        base = Path(st.text_input("AAD base folder (results)", value=str(Path("AAD").as_posix()), key="aad_res_base"))
        windows = list_window_folders(base)
        if not windows: st.info("No outputs_aad_* folders found."); st.stop()
        picked = st.multiselect("Include windows", options=windows, default=windows, key="aad_res_pick")
        if not picked: st.warning("Pick at least one window."); st.stop()

        rows, missing = [], []
        for w in picked:
            df = read_aad_summary(base / w)
            if df is None or df.empty: missing.append(w); continue
            def norm(s): s=str(s); d="".join([c for c in s if c.isdigit()]); return f"S{d.zfill(2)}" if d else s
            df = df.copy(); df["subject"] = df["subject"].apply(norm)
            for _, r in df.iterrows():
                try:
                    val = float(r["accuracy"])
                except Exception:
                    continue
                if 0.0 <= val <= 1.0: val *= 100.0
                rows.append({"window": w, "subject": r["subject"], "accuracy": val})
        if not rows:
            st.error("No valid summaries parsed.")
            if missing: st.caption(f"Missing/invalid: {missing}")
            st.stop()

        df_all = pd.DataFrame(rows).sort_values(["window","subject"], key=lambda s: s.map(natural_key))

        # Per-window interactive bar + median line
        for w in picked:
            wdf = df_all[df_all["window"]==w]
            if wdf.empty: continue
            fig = px.bar(wdf, x="subject", y="accuracy", title=f"Per-subject accuracy — {w}",
                         labels={"accuracy":"Accuracy (%)"})
            fig.update_yaxes(range=[0,100])
            fig.add_hline(y=float(wdf["accuracy"].median()), line_dash="dash", line_color="#F18F01")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        # Across-window interactive box plot
        fig_box = px.box(df_all, x="window", y="accuracy", points="outliers", color="window",
                         title="Distribution across subjects per window", labels={"accuracy":"Accuracy (%)"})
        fig_box.update_yaxes(range=[0,100])
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True, theme="streamlit")

# ===================================================================================
# AEE
# ===================================================================================
with tab_aee:
    st.subheader("Auditory Envelope Extraction (AEE)")
    t_demo, t_results = st.tabs(["Demo", "Results"])

    # ---------------- RESULTS ----------------
    with t_results:
        base = Path(st.text_input("AEE base folder (results)", value=str(Path("outputs").as_posix()), key="aee_res_base"))
        summ = base / "summary_pearsonr_best.csv"
        if not summ.exists():
            st.info("Expected outputs/summary_pearsonr_best.csv with columns: subject, val_r, test_r.")
        else:
            try:
                df = pd.read_csv(summ)
            except Exception:
                try: df = pd.read_csv(summ, sep="\t")
                except Exception as e: st.error(f"Failed to read summary: {e}"); df=None
            if df is not None and not df.empty:
                def norm(s): s=str(s); d="".join([c for c in s if c.isdigit()]); return f"S{d.zfill(2)}" if d else s
                if "subject" in df.columns: df["subject"]=df["subject"].apply(norm)
                else: df.insert(0, "subject", [f"S{str(i+1).zfill(2)}" for i in range(len(df))])
                ycol = "test_r" if "test_r" in df.columns else df.columns[-1]
                df[ycol] = pd.to_numeric(df[ycol], errors="coerce").clip(-1,1)
                df = df.sort_values("subject", key=lambda s: s.map(natural_key))
                fig = px.bar(df, x="subject", y=ycol, title="AEE: Test Pearson r by subject",
                             labels={ycol:"Pearson r (test)"})
                fig.update_yaxes(range=[-0.1, 1.0])
                fig.add_hline(y=float(np.nanmedian(df[ycol].values)), line_dash="dash", line_color="#F18F01")
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # ---------------- DEMO ----------------
    with t_demo:
        c_left, c_right = st.columns([1,3])
        with c_left:
            base = Path(st.text_input("AEE base folder", value=str(Path("outputs").as_posix()), key="aee_base"))
            subj_list = list_subjects(base)
            subj_name = st.selectbox("Subject", subj_list if subj_list else ["<none>"], key="aee_subj")
            preproc = st.text_input("preproc_dir (helper)", value=str("/home/naren-root/Dataset/DATA_preproc"), key="aee_preproc")
            try:
                subj_id = int("".join([c for c in subj_name if c.isdigit()])) if subj_name and subj_name!="<none>" else None
            except Exception:
                subj_id = None
            ckpt_path = (base / subj_name / "best_model.pt") if subj_name and subj_name!="<none>" else None
            st.caption("Model path (AEE)"); st.code(str(ckpt_path) if ckpt_path else "<not found>", language="text")

            device_opt = st.selectbox("Device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []), index=0, key="aee_dev")
            st.caption("Model hyperparameters (fixed): L=3, heads=4, k=8, d_model=128")

            # Windowing for Pearson r per-window
            win_sec_aee = st.number_input("Eval window (s)", value=5.0, min_value=0.5, step=0.5, key="aee_winsec")
            hop_sec_aee = st.number_input("Eval hop (s)", value=2.5, min_value=0.1, step=0.1, key="aee_hopsec")

            load_btn = st.button("Load data (AEE)", key="aee_load")

        if load_btn:
            if subj_id is None:
                st.error("Could not parse subject id.")
            else:
                try:
                    eeg, env_true, fs, attAB = helper.subject_eeg_env_ab(preproc, subj_id)
                except Exception as e:
                    st.error(f"Load failed: {e}"); st.stop()
                eeg = np.asarray(eeg, dtype=np.float32); env_true = np.asarray(env_true, dtype=np.float32); fs=float(fs)
                attAB = safe_att_to_AB(attAB)
                T = min(len(env_true), len(attAB), eeg.shape[0])
                eeg, env_true, attAB = eeg[:T], env_true[:T], attAB[:T]
                if eeg.ndim==1: eeg = eeg[:,None]
                st.session_state.update(dict(
                    aee_loaded=True, eeg_aee=eeg, env_true=env_true, fs_aee=fs, attAB_aee=attAB, n_chan_aee=eeg.shape[1]
                ))
                st.success(f"Loaded EEG {eeg.shape}, env len {len(env_true)}, fs={fs}")

        with c_right:
            if not st.session_state.aee_loaded:
                st.info("Load data to continue.")
            else:
                eeg = st.session_state.eeg_aee; env_true = st.session_state.env_true; fs = st.session_state.fs_aee; n_chan = st.session_state.n_chan_aee
                total_sec = len(env_true)/fs
                st.markdown(f"**Total duration:** {total_sec:.2f} s — **Channels:** {n_chan}")

                c1,c2 = st.columns(2)
                with c1:
                    start_sec = st.number_input("Start (s)", 0.0, max(0.0,total_sec-0.01), 0.0, 0.01, format="%.2f", key="aee_start")
                with c2:
                    end_sec = st.number_input("End (s)", 0.01, total_sec, min(10.0,total_sec), 0.01, format="%.2f", key="aee_end")
                if end_sec <= start_sec: st.error("End must be greater than start."); st.stop()

                n_plot = st.number_input("Plot first N channels", 1, n_chan, min(6,n_chan), 1, key="aee_nplot")
                render_btn = st.button("Render & Predict (AEE)", key="aee_render")

                if render_btn:
                    if ERGraphModel is None or make_biosemi64_info_aee is None:
                        st.error("Could not import ERGraphModel/make_biosemi64_info from TestingGraphMemEfficient."); st.stop()
                    if ckpt_path is None or not ckpt_path.exists():
                        st.error("best_model.pt not found under outputs/SX."); st.stop()

                    s_idx = int(round(start_sec*fs)); e_idx = int(round(end_sec*fs))
                    s_idx=max(0,s_idx); e_idx=min(len(env_true), e_idx)
                    if e_idx<=s_idx: st.error("Interval too short."); st.stop()
                    eeg_sel = eeg[s_idx:e_idx,:]; env_sel = env_true[s_idx:e_idx]
                    T_sel = len(env_sel); t = np.arange(T_sel)/fs + start_sec

                    # Positions from *TestingGraphMemEfficient* (as requested)
                    try:
                        _, _, pos = make_biosemi64_info_aee()  # alias from TestingGraphMemEfficient
                    except Exception:
                        # fallback: simple circular layout
                        n_ch = eeg_sel.shape[1]
                        theta = np.linspace(0, 2*np.pi, n_ch, endpoint=False)
                        pos = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1).astype(np.float32)
                    n_ch = eeg_sel.shape[1]

                    device = torch.device(st.session_state.get("aee_dev","cpu"))
                    model = ERGraphModel(
                        n_ch=n_ch, pos=pos, d_stem=256, d_lift=127, d_in=128, d_model=128,
                        L=3, k=8, heads=4, dropout=0.1, causal=True
                    ).to(device)

                    # strict load exactly like your snippet
                    try:
                        sd = torch.load(str(ckpt_path), map_location=device)
                        model.load_state_dict(sd)
                    except Exception as e:
                        st.error(f"Failed to load AEE weights: {e}"); st.stop()

                    model.eval()
                    with torch.no_grad():
                        xb = torch.from_numpy(eeg_sel[None,:,:]).to(device)
                        yout = model(xb)
                        yhat = yout[0] if isinstance(yout,(tuple,list)) else yout
                        pred = yhat.detach().cpu().numpy().reshape(-1)[:T_sel]

                    # Interactive EEG stack
                    ds = max(1, int(fs//200))
                    traces = {}
                    for i in range(int(n_plot)):
                        traces[f"ch{i}"] = zscore(eeg_sel[::ds, i], axis=0) + i*6.0
                    df_eeg = pd.DataFrame(traces, index=np.round(t[::ds], 3))
                    fig_eeg = px.line(df_eeg, labels={"index":"Time (s)", "value":"EEG (z, stacked)"},
                                      title="EEG selection (stacked z-score)")
                    st.plotly_chart(fig_eeg, use_container_width=True, theme="streamlit")

                    # True vs Pred envelope (interactive)
                    # Rescale pred visually to match true envelope dynamics
                    y = env_sel; yh = pred
                    y_z = (y - y.mean())/(y.std()+1e-8); yh_z = (yh - yh.mean())/(yh.std()+1e-8)
                    yh_vis = yh_z * (y.std()+1e-8) + y.mean()

                    fig_env = go.Figure()
                    fig_env.add_trace(go.Scatter(x=t, y=y, name="True envelope", line=dict(width=2)))
                    fig_env.add_trace(go.Scatter(x=t, y=yh_vis, name="Pred envelope", line=dict(width=2, color="#F18F01")))
                    fig_env.update_layout(title="AEE — True vs Predicted Envelope",
                                          xaxis_title="Time (s)", yaxis_title="Envelope", legend_title="")
                    st.plotly_chart(fig_env, use_container_width=True, theme="streamlit")

                    # # -------- Pearson r per window (AEE) --------
                    # win = int(round(win_sec_aee*fs)); hop = int(round(hop_sec_aee*fs))
                    # spans = window_indices(T_sel, win, hop)
                    # if spans.size == 0:
                    #     st.warning("Selected interval shorter than eval window; no windowed Pearson r computed.")
                    # else:
                    #     r_vals = []
                    #     for (a,b) in spans:
                    #         r_vals.append(pearson_r(y[a:b], yh[a:b]))
                    #     r_vals = np.array(r_vals, dtype=float)
                    #     overall_r = pearson_r(y, yh)

                    #     # Interactive bar of r per window
                    #     fig_r = px.bar(x=np.arange(len(r_vals)), y=r_vals, labels={"x":"Window index", "y":"Pearson r"},
                    #                    title=f"AEE window-wise Pearson r (win={win_sec_aee}s, hop={hop_sec_aee}s)")
                    #     fig_r.add_hline(y=float(np.nanmedian(r_vals)), line_dash="dash", line_color="#F18F01")
                    #     fig_r.update_yaxes(range=[-1.0, 1.0])
                    #     st.plotly_chart(fig_r, use_container_width=True, theme="streamlit")

                    #     st.success(f"Overall Pearson r on selection: {overall_r:.3f}   |   "
                    #                f"Median window r: {np.nanmedian(r_vals):.3f}  (n_windows={len(r_vals)})")
