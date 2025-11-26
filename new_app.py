# # app.py — Child tab (runs without importing matplotlib)
# # Flow:
# #   1) Before "Load data": preproc dir, Helper textbox, Subject dropdown, Load button.
# #   2) After "Load data": Start, End, Channels (enforce 128), Render button.
# #   3) On Render: three plots (via st.line_chart), no matplotlib required.

# import sys, types
# from pathlib import Path
# import numpy as np
# import streamlit as st
# import torch
# import pandas as pd

# st.set_page_config(page_title="Child", layout="wide")
# st.title("Child")

# # -------------------------
# # Safe matplotlib stub (prevents DLL issues when child_eeg.model imports it)
# # -------------------------
# def _stub_matplotlib():
#     if "matplotlib" in sys.modules:
#         return
#     mpl = types.ModuleType("matplotlib")
#     def _noop(*a, **k): pass
#     mpl.use = _noop
#     sys.modules["matplotlib"] = mpl
#     # minimal pyplot and colors if indirectly imported
#     plt = types.ModuleType("matplotlib.pyplot")
#     for name in ["figure","subplots","imshow","plot","scatter","bar","colorbar",
#                  "tight_layout","savefig","close","title","xlabel","ylabel",
#                  "xlim","ylim","axis","show"]:
#         setattr(plt, name, _noop)
#     colors = types.ModuleType("matplotlib.colors")
#     class Colormap: pass
#     colors.Colormap = Colormap
#     sys.modules["matplotlib.pyplot"] = plt
#     sys.modules["matplotlib.colors"] = colors

# # -------------------------
# # Helpers
# # -------------------------
# def zscore(x, eps=1e-8):
#     x = np.asarray(x, dtype=float)
#     return (x - x.mean()) / (x.std() + eps)

# def sliding_windows(T, win, hop):
#     idx = []
#     t = 0
#     while t + win <= T:
#         idx.append((t, t + win))
#         t += hop
#     return np.asarray(idx, dtype=int)

# def detect_subjects(preproc_dir: Path):
#     """
#     Try to detect subjects as subfolders in preproc_dir or its parent.
#     If none found, return a default list of 18 subjects.
#     """
#     subs = []
#     if preproc_dir.is_dir():
#         subs = [p.name for p in preproc_dir.iterdir() if p.is_dir()]
#     if not subs and preproc_dir.parent.is_dir():
#         subs = [p.name for p in preproc_dir.parent.iterdir() if p.is_dir()]
#     # simple filter: plausible subject-like names
#     subs = [s for s in subs if s.lower().startswith(("s","sub","child","subject"))]
#     if not subs:
#         subs = [f"s{str(i).zfill(2)}" for i in range(1, 19)]  # 18 subjects default
#     subs = sorted(subs)
#     return subs

# def try_load_child_data():
#     """
#     Try to load child EEG/audio via child_eeg helper. If not available, return a mock.
#     Returns: (eeg[T,C], audio[T], fs)
#     """
#     try:
#         from child_eeg.child_helper import load_data_child_treatment
#         base_dir = Path("child_eeg")
#         csv_path = base_dir / "sub_1.csv"
#         if csv_path.exists():
#             data = load_data_child_treatment(str(csv_path), base_dir=str(base_dir))
#             if len(data.get("eeg", [])) > 0:
#                 eeg0 = data["eeg"][0].astype(np.float32)
#                 fs0 = float(data["sfreq"][0]) if len(data.get("sfreq", [])) else 64.0
#                 audio0 = data.get("audio", [None])[0]
#                 if audio0 is None or len(audio0) == 0:
#                     audio0 = np.abs(eeg0.mean(axis=1)).astype(np.float32)
#                 return eeg0, audio0, fs0
#     except Exception:
#         pass

#     # Fallback mock
#     fs = 64.0
#     seconds = 30.0
#     T = int(seconds * fs)
#     C = 128
#     t = np.arange(T) / fs
#     basis = (
#         10e-6 * np.sin(2 * np.pi * 6 * t) +
#         6e-6  * np.sin(2 * np.pi * 10 * t) +
#         3e-6  * np.random.randn(T)
#     )
#     eeg = np.repeat(basis[:, None], C, axis=1) * (0.9 + 0.2*np.random.rand(C))
#     audio = np.abs(basis).astype(np.float32)
#     return eeg.astype(np.float32), audio, fs

# def try_load_child_model(preproc_dir: Path, n_ch: int, fs: float):
#     """
#     Load your ERGraphModel from child_eeg/model.py and the best checkpoint in preproc_dir (if present).
#     Uses matplotlib stub to avoid DLL issues.
#     """
#     try:
#         _stub_matplotlib()
#         from child_eeg.model import ERGraphModel, make_hydrocel_info
#         # find best model file
#         cand = None
#         if preproc_dir.is_dir():
#             for pat in ["*_best.pt", "best_model.pt", "*best*.pt"]:
#                 hits = sorted(preproc_dir.glob(pat))
#                 if hits:
#                     cand = hits[-1]; break
#         if cand is None and preproc_dir.parent.is_dir():
#             for pat in ["*_best.pt", "best_model.pt", "*best*.pt"]:
#                 hits = sorted(preproc_dir.parent.glob(pat))
#                 if hits:
#                     cand = hits[-1]; break

#         info, ch_names, pos3d = make_hydrocel_info(n_ch=n_ch, sfreq=fs)
#         pos2 = pos3d[:, :2]
#         model = ERGraphModel(
#             n_ch=n_ch, pos=pos2,
#             d_stem=64, d_lift=63, d_in=64, d_model=64,
#             L=2, k=8, heads=2, dropout=0.1, causal=True
#         )
#         if cand is not None and cand.exists():
#             state = torch.load(str(cand), map_location="cpu")
#             if isinstance(state, dict) and "model" in state:
#                 model.load_state_dict(state["model"], strict=False)
#             else:
#                 model.load_state_dict(state, strict=False)
#         model.eval()
#         return model, cand.name if cand else None
#     except Exception:
#         return None, None

# # -------------------------
# # Session state
# # -------------------------
# if "child_loaded" not in st.session_state:
#     st.session_state.child_loaded = False
#     st.session_state.child_fs = 64.0
#     st.session_state.child_T = 0
#     st.session_state.child_detected_channels = 128
#     st.session_state.child_selected_channels = list(range(128))
#     st.session_state.child_eeg = None
#     st.session_state.child_audio = None
#     st.session_state.child_model = None
#     st.session_state.child_model_file = None

# # -------------------------
# # BEFORE LOAD: preproc dir, helper, subject, load
# # -------------------------
# colL, _ = st.columns([1, 2], gap="large")

# with colL:
#     preproc_dir_str = st.text_input(
#         "preproc dir",
#         value=r"F:\Auditory-Attention-Decoding-using-EEG\child_eeg\outputs_child_1\child_row3",
#         key="child_preproc_dir",
#     )
#     helper_text = st.text_input("Helper", value="", key="child_helper_text")

#     # subjects (auto-detect + count)
#     subjects = detect_subjects(Path(preproc_dir_str))
#     subj = st.selectbox("Subject", subjects, key="child_subject")
#     st.caption(f"Detected {len(subjects)} subjects.")

#     load_btn = st.button("Load data", key="child_load_btn")

# if load_btn:
#     eeg, audio, fs = try_load_child_data()
#     st.session_state.child_eeg = eeg
#     st.session_state.child_audio = audio
#     st.session_state.child_fs = float(fs)
#     st.session_state.child_T = int(eeg.shape[0]) if eeg is not None else 0
#     st.session_state.child_detected_channels = int(eeg.shape[1]) if (eeg is not None and eeg.ndim == 2) else 128

#     model, model_file = try_load_child_model(Path(preproc_dir_str),
#                                              n_ch=st.session_state.child_detected_channels,
#                                              fs=st.session_state.child_fs)
#     st.session_state.child_model = model
#     st.session_state.child_model_file = model_file
#     st.session_state.child_loaded = True

# # -------------------------
# # AFTER LOAD: Start/End/Channels + Render
# # -------------------------
# if st.session_state.child_loaded:
#     st.divider()
#     c1, c2 = st.columns([1, 2], gap="large")

#     with c1:
#         fs = st.session_state.child_fs
#         T = st.session_state.child_T
#         total_sec = (T / fs) if fs > 0 else 0.0

#         start_s = st.number_input("Start (s)", 0.0, float(max(0.01, total_sec)), value=0.0, step=0.01, key="child_start_s")
#         end_s   = st.number_input("End (s)",   0.01, float(max(0.01, total_sec)), value=min(10.0, total_sec), step=0.01, key="child_end_s")
#         if end_s <= start_s:
#             st.warning("End must be > start.")

#         n_ch = st.number_input("n_channels", 1, 256, value=st.session_state.child_detected_channels, step=1, key="child_n_channels")
#         if n_ch != 128:
#             st.warning("For CHILD, please use 128 channels.")

#         ch_opts = [f"ch{i}" for i in range(int(n_ch))]
#         default_sel = [f"ch{i}" for i in st.session_state.child_selected_channels if i < int(n_ch)]
#         picked = st.multiselect("Pick channels", ch_opts, default=default_sel, key="child_channel_picker")
#         if picked:
#             st.session_state.child_selected_channels = [int(s.replace("ch","")) for s in picked]
#         else:
#             st.session_state.child_selected_channels = list(range(min(int(n_ch), 128)))

#         render_btn = st.button("Render selection", key="child_render_btn")

#     # -------------------------
#     # On Render: three plots (no matplotlib; using Streamlit charts)
#     # -------------------------
#     if render_btn:
#         eeg = st.session_state.child_eeg
#         audio_true = st.session_state.child_audio
#         fs = st.session_state.child_fs
#         sel = st.session_state.child_selected_channels

#         if eeg is None or eeg.ndim != 2:
#             st.error("No EEG available.")
#         else:
#             a = int(max(0, min(st.session_state.child_T-1, start_s * fs)))
#             b = int(max(a+1, min(st.session_state.child_T,   end_s * fs)))
#             seg = eeg[a:b, :]                 # [W, C]
#             t = np.arange(seg.shape[0]) / fs

#             seg_sel = seg[:, sel] if len(sel) > 0 else seg
#             mean_eeg = seg_sel.mean(axis=1)

#             # (1) EEG mean over time
#             df1 = pd.DataFrame({"t_s": t, "EEG_mean": mean_eeg})
#             st.line_chart(df1.set_index("t_s"))

#             # Predict envelope with model if loaded, else proxy
#             model = st.session_state.child_model
#             if model is not None:
#                 with torch.no_grad():
#                     xb = torch.from_numpy(seg.astype(np.float32))[None, :, :]     # [1, W, C]
#                     try:
#                         yhat, _ = model(xb, bt_chunk=512)                          # ERGraphModel -> [B,T]
#                         pred_env = yhat.cpu().numpy()[0]
#                     except Exception:
#                         pred_env = np.abs(mean_eeg)
#             else:
#                 pred_env = np.abs(mean_eeg)

#             # (2) True vs Pred envelope (z-scored overlay)
#             env_true = audio_true[a:b]
#             L = min(len(env_true), len(pred_env))
#             env_true, env_pred = zscore(env_true[:L]), zscore(pred_env[:L])
#             tt = t[:L]
#             df2 = pd.DataFrame({"t_s": tt, "True_env": env_true, "Pred_env": env_pred}).set_index("t_s")
#             st.line_chart(df2)

#             # (3) Sliding Pearson r
#             win_sec, hop_sec = 2.0, 1.0
#             win, hop = int(win_sec * fs), int(hop_sec * fs)
#             spans = sliding_windows(L, win, hop) if win > 0 and hop > 0 else np.empty((0,2), int)
#             rs = []
#             for s0, s1 in spans:
#                 x = env_true[s0:s1]; y = env_pred[s0:s1]
#                 rs.append(np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan)
#             df3 = pd.DataFrame({"win_idx": np.arange(len(rs)), "r": rs}).set_index("win_idx")
#             st.line_chart(df3)

#             # Show which model file was used (if any)
#             if st.session_state.child_model_file:
#                 st.caption(f"Model: {st.session_state.child_model_file}")





# # app.py — Child tab (full app, fixed Streamlit state usage; no matplotlib import)
# # Flow:
# #   BEFORE "Load data":
# #       - Child base folder
# #       - Window folder (dropdown)
# #       - Subject (dropdown)
# #       - preproc_dir (helper)
# #       - Checkpoint path (Child) [auto]
# #       - Model window (s) / Model hop (s)
# #       - Device (cpu/cuda), Show raw audio
# #       - Load data (Child)
# #   AFTER "Load data":
# #       - Start/End
# #       - n_channels (enforce 128) + channel picker
# #       - Render selection
# #   Render → 3 plots (EEG mean, True vs Pred envelope, Sliding r timeline)

# # Notes:
# # - No matplotlib import: plots use Streamlit charts.
# # - Model import from child_eeg/model.py is protected by a matplotlib stub.
# # - If child_eeg helper/model isn't available, it falls back to a mock so UI still works.

# import sys, types
# from pathlib import Path
# import numpy as np
# import streamlit as st
# import torch
# import pandas as pd

# st.set_page_config(page_title="Child", layout="wide")
# st.title("Child")

# # -------------------------
# # Safe matplotlib stub (prevents DLL issues when child_eeg.model imports it)
# # -------------------------
# def _stub_matplotlib():
#     if "matplotlib" in sys.modules:
#         return
#     mpl = types.ModuleType("matplotlib")
#     def _noop(*a, **k): pass
#     mpl.use = _noop
#     sys.modules["matplotlib"] = mpl
#     # minimal pyplot and colors if indirectly imported
#     plt = types.ModuleType("matplotlib.pyplot")
#     for name in ["figure","subplots","imshow","plot","scatter","bar","colorbar",
#                  "tight_layout","savefig","close","title","xlabel","ylabel",
#                  "xlim","ylim","axis","show"]:
#         setattr(plt, name, _noop)
#     colors = types.ModuleType("matplotlib.colors")
#     class Colormap: pass
#     colors.Colormap = Colormap
#     sys.modules["matplotlib.pyplot"] = plt
#     sys.modules["matplotlib.colors"] = colors

# # -------------------------
# # Helpers
# # -------------------------
# def zscore(x, eps=1e-8):
#     x = np.asarray(x, dtype=float)
#     s = x.std()
#     return (x - x.mean()) / (s + eps if s != 0 else eps)

# def sliding_windows(T, win, hop):
#     idx, t = [], 0
#     while t + win <= T:
#         idx.append((t, t + win))
#         t += hop
#     return np.asarray(idx, dtype=int)

# def find_window_folders(base: Path):
#     if not base.exists() or not base.is_dir():
#         return []
#     return sorted([p.name for p in base.iterdir()
#                    if p.is_dir() and p.name.lower().startswith("outputs_child")])

# def find_subjects(window_dir: Path):
#     if not window_dir.exists() or not window_dir.is_dir():
#         return []
#     subs = [p.name for p in window_dir.iterdir() if p.is_dir()]
#     subs = [s for s in subs if s.lower().startswith(("child","s","sub","subject"))]
#     return sorted(subs)

# def compute_checkpoint_path(base_dir: Path, window_folder: str, subject: str):
#     default_ckpt = base_dir / window_folder / subject / "best_model.pt"
#     if default_ckpt.exists():
#         return default_ckpt
#     cand_dir = base_dir / window_folder / subject
#     if cand_dir.exists():
#         for pat in ["*_best.pt", "*best*.pt"]:
#             hits = sorted(cand_dir.glob(pat))
#             if hits:
#                 return hits[-1]
#     return default_ckpt

# def try_load_child_data():
#     """
#     Try to load child EEG/audio via child_eeg helper. If not available, return a mock.
#     Returns: (eeg[T,C], audio[T], fs)
#     """
#     try:
#         from child_eeg.child_helper import load_data_child_treatment
#         base_dir = Path("child_eeg")
#         csv_path = base_dir / "sub_1.csv"
#         if csv_path.exists():
#             data = load_data_child_treatment(str(csv_path), base_dir=str(base_dir))
#             if len(data.get("eeg", [])) > 0:
#                 eeg0 = data["eeg"][0].astype(np.float32)
#                 fs0 = float(data["sfreq"][0]) if len(data.get("sfreq", [])) else 64.0
#                 audio0 = data.get("audio", [None])[0]
#                 if audio0 is None or len(audio0) == 0:
#                     audio0 = np.abs(eeg0.mean(axis=1)).astype(np.float32)
#                 return eeg0, audio0, fs0
#     except Exception:
#         pass
#     # fallback mock
#     fs = 64.0
#     T  = int(30.0 * fs)
#     C  = 128
#     t  = np.arange(T) / fs
#     basis = (10e-6*np.sin(2*np.pi*6*t) + 6e-6*np.sin(2*np.pi*10*t) + 3e-6*np.random.randn(T))
#     eeg = np.repeat(basis[:, None], C, axis=1) * (0.9 + 0.2*np.random.rand(C))
#     audio = np.abs(basis).astype(np.float32)
#     return eeg.astype(np.float32), audio, fs

# def try_load_child_model(ckpt_dir: Path, n_ch: int, fs: float, device: str):
#     """
#     Load ERGraphModel from child_eeg/model.py with the best checkpoint under ckpt_dir (subject folder),
#     or its parent, and move to selected device.
#     """
#     try:
#         _stub_matplotlib()
#         from child_eeg.model import ERGraphModel, make_hydrocel_info
#         cand = None
#         if ckpt_dir.is_dir():
#             for pat in ["*_best.pt", "best_model.pt", "*best*.pt"]:
#                 hits = sorted(ckpt_dir.glob(pat))
#                 if hits:
#                     cand = hits[-1]; break
#         if cand is None and ckpt_dir.parent.is_dir():
#             for pat in ["*_best.pt", "best_model.pt", "*best*.pt"]:
#                 hits = sorted(ckpt_dir.parent.glob(pat))
#                 if hits:
#                     cand = hits[-1]; break
#         info, ch_names, pos3d = make_hydrocel_info(n_ch=n_ch, sfreq=fs)
#         pos2 = pos3d[:, :2]
#         model = ERGraphModel(n_ch=n_ch, pos=pos2, d_stem=64, d_lift=63,
#                              d_in=64, d_model=64, L=2, k=8, heads=2,
#                              dropout=0.1, causal=True)
#         if cand is not None and cand.exists():
#             state = torch.load(str(cand), map_location="cpu")
#             if isinstance(state, dict) and "model" in state:
#                 model.load_state_dict(state["model"], strict=False)
#             else:
#                 model.load_state_dict(state, strict=False)
#         model.to(device).eval()
#         return model, (cand.name if cand else None)
#     except Exception:
#         return None, None

# # -------------------------
# # Session defaults
# # -------------------------
# st.session_state.setdefault("child_loaded", False)
# st.session_state.setdefault("child_fs", 64.0)
# st.session_state.setdefault("child_T", 0)
# st.session_state.setdefault("child_detected_channels", 128)
# st.session_state.setdefault("child_selected_channels", list(range(128)))
# st.session_state.setdefault("child_eeg", None)
# st.session_state.setdefault("child_audio", None)
# st.session_state.setdefault("child_model", None)
# st.session_state.setdefault("child_model_file", None)

# # -------------------------
# # BEFORE LOAD (AAD-like controls for Child)
# # -------------------------
# cL, cR = st.columns([1, 2], gap="large")

# with cL:
#     # Child base folder
#     base_default = Path("child_eeg")
#     child_base_folder = st.text_input("Child base folder",
#                                       value=str(base_default),
#                                       key="child_base_folder")
#     base_path = Path(child_base_folder)

#     # Window folder dropdown (e.g., outputs_child_1)
#     window_folders = find_window_folders(base_path) or ["outputs_child_1"]
#     win_folder = st.selectbox("Window folder", window_folders, key="child_win_folder_sel")

#     # Subject dropdown (e.g., child_row3)
#     subj_list = find_subjects(base_path / win_folder) or ["child_row3"]
#     subj = st.selectbox("Subject", subj_list, key="child_subject_sel")

#     # preproc_dir (helper) – your requested default
#     preproc_dir_str = st.text_input(
#         "preproc_dir (helper)",
#         value=r"F:\Auditory-Attention-Decoding-using-EEG\child_eeg\outputs_child_1\child_row3",
#         key="child_preproc_dir_in",
#     )

#     # Checkpoint path display (auto)
#     ckpt_path = compute_checkpoint_path(base_path, win_folder, subj)
#     st.text_input("Checkpoint path (Child)", value=str(ckpt_path), key="child_ckpt_display")

#     st.markdown("---")
#     # Model window/hop (use distinct widget keys; don't write to same state keys)
#     model_win = st.number_input("Model window (s)", value=5.0, min_value=0.5, step=0.5, key="child_model_win_input")
#     model_hop = st.number_input("Model hop (s)", value=2.5, min_value=0.1, step=0.1, key="child_model_hop_input")
#     device_opts = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
#     device_sel = st.selectbox("Device", device_opts, index=0, key="child_device_sel")
#     show_audio = st.checkbox("Show raw audio if found", value=False, key="child_show_audio_chk")

#     load_btn = st.button("Load data (Child)", key="child_load_btn")

# if load_btn:
#     # Load EEG/audio
#     eeg, audio, fs = try_load_child_data()
#     st.session_state["child_eeg"] = eeg
#     st.session_state["child_audio"] = audio
#     st.session_state["child_fs"] = float(fs)
#     st.session_state["child_T"] = int(eeg.shape[0]) if eeg is not None else 0
#     st.session_state["child_detected_channels"] = int(eeg.shape[1]) if (eeg is not None and eeg.ndim == 2) else 128

#     # Load model using selected subject folder as checkpoint directory
#     model, model_file = try_load_child_model(Path(child_base_folder) / win_folder / subj,
#                                              n_ch=st.session_state["child_detected_channels"],
#                                              fs=st.session_state["child_fs"],
#                                              device=device_sel)
#     st.session_state["child_model"] = model
#     st.session_state["child_model_file"] = model_file
#     st.session_state["child_loaded"] = True

#     # Persist user-chosen model params under separate keys
#     st.session_state["child_model_win_val"] = float(model_win)
#     st.session_state["child_model_hop_val"] = float(model_hop)
#     st.session_state["child_device_val"] = device_sel
#     st.session_state["child_show_audio_val"] = bool(show_audio)

# # -------------------------
# # AFTER LOAD: Start/End/Channels + Render
# # -------------------------
# if st.session_state["child_loaded"]:
#     st.divider()
#     c1, c2 = st.columns([1, 2], gap="large")

#     with c1:
#         fs = st.session_state["child_fs"]
#         T  = st.session_state["child_T"]
#         total_sec = (T / fs) if fs > 0 else 0.0

#         start_s = st.number_input("Start (s)", 0.0, float(max(0.01, total_sec)),
#                                   value=0.0, step=0.01, key="child_start_s_in")
#         end_s   = st.number_input("End (s)",   0.01, float(max(0.01, total_sec)),
#                                   value=min(10.0, total_sec), step=0.01, key="child_end_s_in")
#         if end_s <= start_s:
#             st.warning("End must be > start.")

#         n_ch = st.number_input("n_channels", 1, 256,
#                                value=st.session_state["child_detected_channels"], step=1, key="child_nch_in")
#         if n_ch != 128:
#             st.warning("For CHILD, please use 128 channels.")

#         ch_opts = [f"ch{i}" for i in range(int(n_ch))]
#         default_sel = [f"ch{i}" for i in st.session_state["child_selected_channels"] if i < int(n_ch)]
#         picked = st.multiselect("Pick channels", ch_opts, default=default_sel, key="child_chpick_in")
#         if picked:
#             st.session_state["child_selected_channels"] = [int(s.replace("ch","")) for s in picked]
#         else:
#             st.session_state["child_selected_channels"] = list(range(min(int(n_ch), 128)))

#         render_btn = st.button("Render selection", key="child_render_btn")

#     # -------------------------
#     # On Render: three plots (Streamlit charts; no matplotlib)
#     # -------------------------
#     if render_btn:
#         eeg = st.session_state["child_eeg"]
#         audio_true = st.session_state["child_audio"]
#         fs = st.session_state["child_fs"]
#         sel = st.session_state["child_selected_channels"]
#         model_win = float(st.session_state.get("child_model_win_val", 5.0))
#         model_hop = float(st.session_state.get("child_model_hop_val", 2.5))
#         show_audio = bool(st.session_state.get("child_show_audio_val", False))

#         if eeg is None or eeg.ndim != 2:
#             st.error("No EEG available.")
#         else:
#             a = int(max(0, min(T-1, start_s * fs)))
#             b = int(max(a+1, min(T,   end_s * fs)))
#             seg = eeg[a:b, :]                 # [W, C]
#             t = np.arange(seg.shape[0]) / fs

#             seg_sel = seg[:, sel] if len(sel) > 0 else seg
#             mean_eeg = seg_sel.mean(axis=1)

#             # Optional: play raw audio if requested and available
#             if show_audio and audio_true is not None and len(audio_true) > 0:
#                 try:
#                     import soundfile as sf, io
#                     bio = io.BytesIO()
#                     sf.write(bio, audio_true.astype(np.float32), int(fs), format="WAV")
#                     st.audio(bio.getvalue(), sample_rate=int(fs))
#                 except Exception:
#                     pass

#             # (1) EEG mean over time
#             df1 = pd.DataFrame({"t_s": t, "EEG_mean": mean_eeg}).set_index("t_s")
#             st.line_chart(df1)

#             # Predict envelope with model if loaded; else proxy
#             model = st.session_state["child_model"]
#             if model is not None:
#                 with torch.no_grad():
#                     xb = torch.from_numpy(seg.astype(np.float32))[None, :, :]  # [1, W, C]
#                     try:
#                         yhat, _ = model(xb, bt_chunk=512)
#                         pred_env = yhat.cpu().numpy()[0]
#                     except Exception:
#                         pred_env = np.abs(mean_eeg)
#             else:
#                 pred_env = np.abs(mean_eeg)

#             # (2) True vs Pred envelope (z-scored overlay)
#             env_true = audio_true[a:b]
#             L = min(len(env_true), len(pred_env))
#             env_true = zscore(env_true[:L]); env_pred = zscore(pred_env[:L])
#             tt = t[:L]
#             df2 = pd.DataFrame({"t_s": tt, "True_env": env_true, "Pred_env": env_pred}).set_index("t_s")
#             st.line_chart(df2)

#             # (3) Sliding Pearson r using chosen window/hop
#             win = int(model_win * fs); hop = int(model_hop * fs)
#             spans = sliding_windows(L, win, hop) if win > 0 and hop > 0 else np.empty((0,2), int)
#             rs = []
#             for s0, s1 in spans:
#                 x = env_true[s0:s1]; y = env_pred[s0:s1]
#                 rs.append(np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan)
#             df3 = pd.DataFrame({"win_idx": np.arange(len(rs)), "r": rs}).set_index("win_idx")
#             st.line_chart(df3)

#             # Show which model file was used (if any)
#             if st.session_state["child_model_file"]:
#                 st.caption(f"Model: {st.session_state['child_model_file']}")







# app.py — EEG ↔ Audio Demo (AAD & AEE) — interactive plots + Child AEE pred
# ---------------------------------------------------------------------------------
# Tabs:
#   1) AAD (demo + results)
#   2) AEE (demo + results)
#   3) Child (demo; predict envelope; up to 128 channels; no CSV row index)
# ---------------------------------------------------------------------------------

from __future__ import annotations
import sys, types

# ------------------------------------------------------------
# SAFE MATPLOTLIB STUB — must run BEFORE importing helper/model
# ------------------------------------------------------------
if "matplotlib" not in sys.modules:
    mpl = types.ModuleType("matplotlib")
    def _noop(*a, **k): pass
    mpl.use = _noop

    plt = types.ModuleType("matplotlib.pyplot")
    for name in ["figure","subplots","imshow","plot","scatter","bar",
                 "colorbar","tight_layout","savefig","close",
                 "title","xlabel","ylabel","xlim","ylim","axis","show"]:
        setattr(plt, name, _noop)

    colors = types.ModuleType("matplotlib.colors")
    class Colormap: pass
    colors.Colormap = Colormap

    lines = types.ModuleType("matplotlib.lines")
    class Line2D:
        def __init__(self, *a, **k): pass
    lines.Line2D = Line2D

    sys.modules["matplotlib"] = mpl
    sys.modules["matplotlib.pyplot"] = plt
    sys.modules["matplotlib.colors"] = colors
    sys.modules["matplotlib.lines"] = lines

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from pathlib import Path
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import torch
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

# ---- Namespaced key helper (avoid duplicate keys anywhere) ----
def k(ns: str, name: str) -> str:
    return f"{ns}:{name}"

# ==== Helpers (provide these in your repo) ====
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

# --- Child repo wiring (before any use) ---
import os
CHILD_ROOT = Path(r"F:\Auditory-Attention-Decoding-using-EEG\child_eeg")
for p in [CHILD_ROOT, CHILD_ROOT / "child_helper"]:
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

load_data_child_treatment = None
ChildERGraphModel = None
make_hydrocel_info = None

# Child helper function
try:
    from child_helper import load_data_child_treatment as _ldct
    load_data_child_treatment = _ldct
except Exception:
    try:
        from child_helper.child_helper import load_data_child_treatment as _ldct
        load_data_child_treatment = _ldct
    except Exception:
        pass

# Child model + montage util
try:
    from model import ERGraphModel as _ChildERGraphModel, make_hydrocel_info as _make_hydrocel_info
    ChildERGraphModel = _ChildERGraphModel
    make_hydrocel_info = _make_hydrocel_info
except Exception:
    try:
        from child_eeg.model import ERGraphModel as _ChildERGraphModel, make_hydrocel_info as _make_hydrocel_info
        ChildERGraphModel = _ChildERGraphModel
        make_hydrocel_info = _make_hydrocel_info
    except Exception:
        pass

# -----------------------------------------------------------------------------------
# Streamlit page
# -----------------------------------------------------------------------------------
st.set_page_config(page_title="EEG ↔ Audio — AAD & AEE", layout="wide")
st.set_option("client.showErrorDetails", True)
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

if "child_loaded" not in st.session_state:
    st.session_state.child_loaded = False
    st.session_state.child_eeg = None
    st.session_state.child_fs = None
    st.session_state.child_n_chan = None
    st.session_state.child_ckpt_path = ""

# -----------------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------------
tab_aad, tab_aee, tab_child = st.tabs(["AAD", "AEE", "Child"])

# ===================================================================================
# AAD
# ===================================================================================
with tab_aad:
    st.subheader("Auditory Attention Decoding (AAD)")
    t_demo, t_results = st.tabs(["Demo", "Results"])

    # ---------------- DEMO ----------------
    with t_demo:
        NSL, NSR = "aad_demo_left", "aad_demo_right"
        c_left, c_right = st.columns([1,3])

        with c_left:
            base = Path(st.text_input("AAD base folder", value=str(Path("AAD").as_posix()), key=k(NSL,"base")))
            windows = list_window_folders(base)
            win_name = st.selectbox("Window folder", windows if windows else ["<none>"], key=k(NSL,"win"))
            subj_list = list_subjects(base / win_name) if windows else []
            subj_name = st.selectbox("Subject", subj_list if subj_list else ["<none>"], key=k(NSL,"subj"))
            preproc = st.text_input("preproc_dir (helper)", value=str("/home/naren-root/Dataset/DATA_preproc"), key=k(NSL,"preproc"))

            try:
                subj_id = int("".join([c for c in subj_name if c.isdigit()])) if subj_name and subj_name!="<none>" else None
            except Exception:
                subj_id = None

            ckpt_path = (base / win_name / subj_name / "best_model.pt") if subj_name and subj_name!="<none>" else None
            st.caption("Checkpoint path (AAD)")
            st.code(str(ckpt_path) if ckpt_path else "<invalid>", language="text")

            st.markdown("---")
            win_sec = st.number_input("Model window (s)", value=5.0, min_value=0.5, step=0.5, key=k(NSL,"winsec"))
            hop_sec = st.number_input("Model hop (s)", value=2.5, min_value=0.1, step=0.1, key=k(NSL,"hopsec"))
            device_opt = st.selectbox("Device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []), index=0, key=k(NSL,"dev"))
            show_audio = st.checkbox("Show raw audio if found", value=False, key=k(NSL,"show_audio"))
            load_btn = st.button("Load data (AAD)", key=k(NSL,"load"))

        with c_right:
            if load_btn:
                if subj_id is None:
                    st.error("Could not parse subject id.")
                else:
                    try:
                        eeg, envA, envB, fs, attAB = helper.subject_eeg_env_ab_aad(preproc, subj_id)
                    except Exception as e:
                        st.error(f"Load failed: {e}")
                        st.stop()
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

            # You can add AAD plotting/pred here if desired. For now we keep demo minimal.

    # ---------------- RESULTS ----------------
    with t_results:
        NS = "aad_results"
        base = Path(st.text_input("AAD base folder (results)", value=str(Path("AAD").as_posix()), key=k(NS,"base")))
        windows = list_window_folders(base)
        if not windows:
            st.info("No outputs_aad_* folders found.")
            st.stop()
        picked = st.multiselect("Include windows", options=windows, default=windows, key=k(NS,"pick"))
        if not picked:
            st.warning("Pick at least one window.")
            st.stop()

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

        for w in picked:
            wdf = df_all[df_all["window"]==w]
            if wdf.empty: continue
            fig = px.bar(wdf, x="subject", y="accuracy", title=f"Per-subject accuracy — {w}",
                         labels={"accuracy":"Accuracy (%)"})
            fig.update_yaxes(range=[0,100])
            fig.add_hline(y=float(wdf["accuracy"].median()), line_dash="dash")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

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
        NS = "aee_results"
        base = Path(st.text_input("AEE base folder (results)", value=str(Path("outputs").as_posix()), key=k(NS,"base")))
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
                fig.add_hline(y=float(np.nanmedian(df[ycol].values)), line_dash="dash")
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    # ---------------- DEMO ----------------
    with t_demo:
        NSL, NSR = "aee_demo_left", "aee_demo_right"
        c_left, c_right = st.columns([1,3])

        with c_left:
            base = Path(st.text_input("AEE base folder", value=str(Path("outputs").as_posix()), key=k(NSL,"base")))
            subj_list = list_subjects(base)
            subj_name = st.selectbox("Subject", subj_list if subj_list else ["<none>"], key=k(NSL,"subj"))
            preproc = st.text_input("preproc_dir (helper)", value=str("/home/naren-root/Dataset/DATA_preproc"), key=k(NSL,"preproc"))
            try:
                subj_id = int("".join([c for c in subj_name if c.isdigit()])) if subj_name and subj_name!="<none>" else None
            except Exception:
                subj_id = None
            ckpt_path = (base / subj_name / "best_model.pt") if subj_name and subj_name!="<none>" else None
            st.caption("Model path (AEE)")
            st.code(str(ckpt_path) if ckpt_path else "<not found>", language="text")

            device_opt = st.selectbox("Device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []), index=0, key=k(NSL,"dev"))
            st.caption("Model hyperparameters (fixed): L=3, heads=4, k=8, d_model=128")
            win_sec_aee = st.number_input("Eval window (s)", value=5.0, min_value=0.5, step=0.5, key=k(NSL,"win"))
            hop_sec_aee = st.number_input("Eval hop (s)", value=2.5, min_value=0.1, step=0.1, key=k(NSL,"hop"))

            load_btn = st.button("Load data (AEE)", key=k(NSL,"load"))

        with c_right:
            if load_btn:
                if subj_id is None:
                    st.error("Could not parse subject id.")
                else:
                    try:
                        eeg, env_true, fs, attAB = helper.subject_eeg_env_ab(preproc, subj_id)
                    except Exception as e:
                        st.error(f"Load failed: {e}")
                        st.stop()
                    eeg = np.asarray(eeg, dtype=np.float32); env_true = np.asarray(env_true, dtype=np.float32); fs=float(fs)
                    attAB = safe_att_to_AB(attAB)
                    T = min(len(env_true), len(attAB), eeg.shape[0])
                    eeg, env_true, attAB = eeg[:T], env_true[:T], attAB[:T]
                    if eeg.ndim==1: eeg = eeg[:,None]
                    st.session_state.update(dict(
                        aee_loaded=True, eeg_aee=eeg, env_true=env_true, fs_aee=fs, attAB_aee=attAB, n_chan_aee=eeg.shape[1]
                    ))
                    st.success(f"Loaded EEG {eeg.shape}, env len {len(env_true)}, fs={fs}")

            if not st.session_state.aee_loaded:
                st.info("Load data to continue.")
            else:
                eeg = st.session_state.eeg_aee; env_true = st.session_state.env_true; fs = st.session_state.fs_aee; n_chan = st.session_state.n_chan_aee
                total_sec = len(env_true)/fs
                st.markdown(f"**Total duration:** {total_sec:.2f} s — **Channels:** {n_chan}")

                c1,c2 = st.columns(2)
                with c1:
                    start_sec = st.number_input("Start (s)", 0.0, max(0.0,total_sec-0.01), 0.0, 0.01, format="%.2f", key=k(NSR,"start"))
                with c2:
                    end_sec = st.number_input("End (s)", 0.01, total_sec, min(10.0,total_sec), 0.01, format="%.2f", key=k(NSR,"end"))
                if end_sec <= start_sec:
                    st.error("End must be greater than start.")
                    st.stop()

                n_plot = st.number_input("Plot first N channels", 1, n_chan, min(6,n_chan), 1, key=k(NSR,"nplot"))
                render_btn = st.button("Render & Predict (AEE)", key=k(NSR,"render"))

                if render_btn:
                    if ERGraphModel is None or make_biosemi64_info_aee is None:
                        st.error("Could not import ERGraphModel/make_biosemi64_info from TestingGraphMemEfficient.")
                        st.stop()
                    if ckpt_path is None or not ckpt_path.exists():
                        st.error("best_model.pt not found under outputs/SX.")
                        st.stop()

                    s_idx = max(0, int(round(start_sec*fs))); e_idx = min(len(env_true), int(round(end_sec*fs)))
                    if e_idx<=s_idx:
                        st.error("Interval too short.")
                        st.stop()
                    eeg_sel = eeg[s_idx:e_idx,:]; env_sel = env_true[s_idx:e_idx]
                    T_sel = len(env_sel); t = np.arange(T_sel)/fs + start_sec

                    try:
                        _, _, pos = make_biosemi64_info_aee()
                    except Exception:
                        n_ch = eeg_sel.shape[1]
                        theta = np.linspace(0, 2*np.pi, n_ch, endpoint=False)
                        pos = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1).astype(np.float32)
                    n_ch = eeg_sel.shape[1]

                    device = torch.device(st.session_state.get(k(NSL,"dev"),"cpu"))
                    model = ERGraphModel(
                        n_ch=n_ch, pos=pos, d_stem=256, d_lift=127, d_in=128, d_model=128,
                        L=3, k=8, heads=4, dropout=0.1, causal=True
                    ).to(device)

                    try:
                        sd = torch.load(str(ckpt_path), map_location=device)
                        model.load_state_dict(sd)
                    except Exception as e:
                        st.error(f"Failed to load AEE weights: {e}")
                        st.stop()

                    model.eval()
                    with torch.no_grad():
                        xb = torch.from_numpy(eeg_sel[None,:,:]).to(device)
                        yout = model(xb)
                        yhat = yout[0] if isinstance(yout,(tuple,list)) else yout
                        pred = yhat.detach().cpu().numpy().reshape(-1)[:T_sel]

                    ds = max(1, int(fs//200))
                    traces = {f"ch{i}": zscore(eeg_sel[::ds, i], axis=0) + i*6.0 for i in range(int(n_plot))}
                    df_eeg = pd.DataFrame(traces, index=np.round(t[::ds], 3))
                    fig_eeg = px.line(df_eeg, labels={"index":"Time (s)", "value":"EEG (z, stacked)"},
                                      title="EEG selection (stacked z-score)")
                    st.plotly_chart(fig_eeg, use_container_width=True, theme="streamlit")

                    y = env_sel; yh = pred
                    y_z = (y - y.mean())/(y.std()+1e-8); yh_z = (yh - yh.mean())/(yh.std()+1e-8)
                    yh_vis = yh_z * (y.std()+1e-8) + y.mean()

                    fig_env = go.Figure()
                    fig_env.add_trace(go.Scatter(x=t, y=y, name="True envelope", line=dict(width=2)))
                    fig_env.add_trace(go.Scatter(x=t, y=yh_vis, name="Pred envelope", line=dict(width=2)))
                    fig_env.update_layout(title="AEE — True vs Predicted Envelope",
                                          xaxis_title="Time (s)", yaxis_title="Envelope")
                    st.plotly_chart(fig_env, use_container_width=True, theme="streamlit")

# ===================================================================================
# CHILD (AEE-style demo; envelope is predicted only)
# ===================================================================================
with tab_child:
    st.subheader("Child EEG — Envelope Extraction (predicted)")

    t_demo, = st.tabs(["Demo"])
    with t_demo:
        NSL, NSR = "child_demo_left", "child_demo_right"
        c_left, c_right = st.columns([1, 3])

        # ---------------- LEFT: inputs ----------------
        with c_left:
            child_base_default = r"F:\Auditory-Attention-Decoding-using-EEG\child_eeg"
            child_ckpt_default = r"F:\Auditory-Attention-Decoding-using-EEG\child_eeg\outputs_child_1\child_row3\child_row3_best.pt"

            ckpt_path_child_str = st.text_input("Model path (child)", value=child_ckpt_default, key=k(NSL,"ckpt"))
            st.session_state["child_ckpt_path"] = ckpt_path_child_str

            child_base = Path(st.text_input("Base folder (child)", value=child_base_default, key=k(NSL,"base")))
            child_csv  = Path(st.text_input("CSV (child)", value=str(child_base / "sub_1.csv"), key=k(NSL,"csv")))

            device_opt_child = st.selectbox("Device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
                                            index=0, key=k(NSL,"dev"))

            win_sec_child = st.number_input("Eval window (s)", value=5.0, min_value=0.5, step=0.5, key=k(NSL,"win"))
            hop_sec_child = st.number_input("Eval hop (s)", value=2.5, min_value=0.1, step=0.1, key=k(NSL,"hop"))

            load_btn_child = st.button("Load data (Child)", key=k(NSL,"load"))

        # ---------------- LOAD ----------------
        if load_btn_child:
            if not callable(load_data_child_treatment):
                st.error(
                    "Child loader not found. Ensure one of these exists:\n"
                    " - child_eeg\\child_helper\\child_helper.py with load_data_child_treatment(...)\n"
                    " - child_eeg\\child_helper.py with the same function\n"
                    "Also confirm __init__.py inside child_helper/ if it’s a folder."
                )
            else:
                try:
                    data = load_data_child_treatment(str(child_csv), base_dir=str(child_base))
                    eeg_list = data.get("eeg", None)
                    sfreqs   = data.get("sfreq", None)

                    if isinstance(eeg_list, np.ndarray):
                        seq = list(eeg_list)
                    elif isinstance(eeg_list, (list, tuple)):
                        seq = list(eeg_list)
                    else:
                        seq = []

                    eeg_np = None
                    for arr in seq:
                        if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.size > 0:
                            eeg_np = arr.astype(np.float32, copy=False)
                            break

                    if eeg_np is None:
                        st.error("No non-empty EEG rows returned from child helper; check CSV/base paths.")
                    else:
                        fs_ch = 64.0
                        if isinstance(sfreqs, (list, tuple, np.ndarray)) and len(sfreqs) > 0:
                            try:
                                fs_ch = float(sfreqs[0])
                            except Exception:
                                pass

                        st.session_state.update(dict(
                            child_loaded=True,
                            child_eeg=eeg_np,            # [T, C]
                            child_fs=fs_ch,              # float
                            child_n_chan=int(eeg_np.shape[1])
                        ))
                        st.success(f"Loaded child EEG {eeg_np.shape}, fs={fs_ch} Hz")

                except Exception as e:
                    st.exception(e)

        # ---------- cached model loader (matches checkpoint dims) ----------
        @st.cache_resource(show_spinner=False)
        def _load_child_model_matched(ckpt_path: Path, n_ch: int, pos2d: np.ndarray, device: torch.device):
            """
            Build Child ERGraphModel to MATCH the checkpoint shapes, then load weights.
            Falls back to filtering mismatched keys if anything still disagrees.
            """
            if ChildERGraphModel is None:
                raise RuntimeError("ChildERGraphModel not importable. Check child_eeg/model.py on sys.path.")
            if not ckpt_path or not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            sd_raw = torch.load(str(ckpt_path), map_location="cpu")
            sd = sd_raw.get("state_dict", sd_raw) if isinstance(sd_raw, dict) else sd_raw

            def shp(key, default=None):
                t = sd.get(key, None)
                return tuple(t.shape) if isinstance(t, torch.Tensor) else default

            stem_pw = shp("stem.pw.conv.weight")         # (d_stem, d_in+1, 1)
            lift_w  = shp("lift.weight")                 # (d_lift, d_stem)
            proj_in = shp("graph.proj_in.weight")        # (d_model, d_in)
            gat_a   = shp("graph.blocks.0.gat.a")        # (heads, head_dim)

            d_stem  = int(stem_pw[0]) if stem_pw else 256
            d_lift  = int(lift_w[0])  if lift_w  else 127
            d_model = int(proj_in[0]) if proj_in else 128
            d_in    = int(proj_in[1]) if proj_in else d_model
            heads   = int(gat_a[0])   if gat_a   else 4

            model = ChildERGraphModel(
                n_ch=n_ch, pos=pos2d,
                d_stem=d_stem, d_lift=d_lift, d_in=d_in, d_model=d_model,
                L=2, k=8, heads=heads, dropout=0.1, causal=True
            ).to(device)

            try:
                model.load_state_dict(sd, strict=True)
            except RuntimeError:
                cur = model.state_dict()
                compat = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
                skipped = sorted(set(sd.keys()) - set(compat.keys()))
                if not compat:
                    raise
                model.load_state_dict(compat, strict=False)
                st.warning(
                    "Loaded checkpoint with filtered keys. Skipped:\n" +
                    "\n".join(skipped[:20]) + ("… (truncated)" if len(skipped) > 20 else "")
                )
            model.eval()
            return model, dict(d_stem=d_stem, d_lift=d_lift, d_in=d_in, d_model=d_model, heads=heads)

        # ---------------- RIGHT: plots + inference ----------------
        with c_right:
            if not st.session_state.get("child_loaded", False):
                st.info("Load child EEG to continue.")
            else:
                eeg_np = st.session_state["child_eeg"]    # [T, C]
                fs     = float(st.session_state["child_fs"])
                n_chan = int(st.session_state["child_n_chan"])

                total_sec = eeg_np.shape[0] / max(fs, 1.0)
                st.markdown(f"**Total duration:** {total_sec:.2f} s — **Channels:** {n_chan}")

                c1, c2 = st.columns(2)
                with c1:
                    start_sec = st.number_input("Start (s)", 0.0, max(0.0, total_sec - 0.01),
                                                0.0, 0.01, format="%.2f", key=k(NSR,"start"))
                with c2:
                    end_sec   = st.number_input("End (s)", 0.01, total_sec,
                                                min(10.0, total_sec), 0.01, format="%.2f", key=k(NSR,"end"))

                if end_sec <= start_sec:
                    st.error("End must be greater than start.")
                    st.stop()

                nplot_max = min(128, n_chan)
                n_plot = st.number_input("Plot first N channels", 1, nplot_max, min(6, nplot_max), 1, key=k(NSR,"nplot"))

                render_btn_child = st.button("Render & Predict (Child)", key=k(NSR,"render"))

                if render_btn_child:
                    with st.status("Running child inference…", expanded=True) as status:
                        try:
                            s_idx = max(0, int(round(start_sec * fs)))
                            e_idx = min(eeg_np.shape[0], int(round(end_sec * fs)))
                            if e_idx <= s_idx:
                                st.error("Empty interval after rounding. Increase End(s) or decrease Start(s).")
                                status.update(label="Stopped (empty interval).", state="error")
                                st.stop()

                            eeg_sel = np.ascontiguousarray(eeg_np[s_idx:e_idx, :])  # [T_sel, C]
                            T_sel, C = eeg_sel.shape
                            t = (np.arange(T_sel) / max(fs, 1.0)) + start_sec

                            st.write(f"Slice: T_sel={T_sel}, C={C}, fs={fs}, idx=({s_idx}, {e_idx})")

                            try:
                                info, ch_names, pos3d = make_hydrocel_info(n_ch=C, sfreq=fs)
                                pos2d = pos3d[:, :2].astype(np.float32)
                            except Exception:
                                theta = np.linspace(0, 2*np.pi, C, endpoint=False)
                                pos2d = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)

                            dev_key = st.session_state.get(k(NSL,"dev"), "cpu")
                            if dev_key == "cuda" and not torch.cuda.is_available():
                                st.warning("CUDA not available; falling back to CPU.")
                                dev_key = "cpu"
                            device = torch.device(dev_key)
                            st.write(f"Device: {device}")

                            ckpt_str = st.session_state.get("child_ckpt_path") or ""
                            ckpt_path_fixed = Path(ckpt_str.strip()).resolve() if ckpt_str else None
                            if ckpt_path_fixed and "using- EEG" in str(ckpt_path_fixed):
                                ckpt_path_fixed = Path(str(ckpt_path_fixed).replace("using- EEG", "using-EEG"))
                            if not ckpt_path_fixed or not ckpt_path_fixed.exists():
                                st.error(f"Model path does not exist:\n{ckpt_path_fixed}")
                                status.update(label="Stopped (checkpoint missing).", state="error")
                                st.stop()

                            model_child, dims = _load_child_model_matched(ckpt_path_fixed, C, pos2d, device)
                            st.write(f"Model dims: {dims}")

                            with torch.no_grad():
                                xb = torch.from_numpy(eeg_sel)[None, :, :].to(device)  # [1, T_sel, C]
                                yhat = model_child(xb)
                                if isinstance(yhat, (tuple, list)):
                                    yhat = yhat[0]
                                pred = yhat.detach().cpu().numpy().reshape(-1)[:T_sel]

                            ds = max(1, int(fs // 200))
                            show_n = int(min(n_plot, 128, C))
                            traces = {
                                f"ch{i}": (eeg_sel[::ds, i] - eeg_sel[::ds, i].mean()) /
                                          (eeg_sel[::ds, i].std() + 1e-8) + i*6.0
                                for i in range(show_n)
                            }
                            df_eeg = pd.DataFrame(traces, index=np.round(t[::ds], 3))
                            fig_eeg = px.line(
                                df_eeg,
                                labels={"index": "Time (s)", "value": "EEG (z, stacked)"},
                                title=f"Child EEG selection — first {show_n} channels"
                            )
                            st.plotly_chart(fig_eeg, use_container_width=True, theme="streamlit")

                            fig_env = go.Figure()
                            fig_env.add_trace(go.Scatter(x=t, y=pred, name="Pred envelope", line=dict(width=2)))
                            fig_env.update_layout(
                                title="Child — Predicted Envelope",
                                xaxis_title="Time (s)",
                                yaxis_title="Envelope",
                                legend_title=""
                            )
                            st.plotly_chart(fig_env, use_container_width=True, theme="streamlit")

                            status.update(label="Done.", state="complete")

                        except Exception as e:
                            st.exception(e)
                            status.update(label="Failed.", state="error")
