import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import helper

# ---------------------------------------------------------------------
# Config via environment variables
# ---------------------------------------------------------------------
PREPROC_DIR = "/home/naren-root/Documents/FYP/AAD/Notebooks/Dataset/DATA_preproc/"
SUBJ_ID = 1

FIG_DIR = Path("figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("Logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)

PASS = "\u2705"  # checkmark
FAIL = "\u274c"  # cross


def _ok(msg):
    print(f"{PASS} {msg}")


def _bad(msg):
    print(f"{FAIL} {msg}")


# ---------------------------------------------------------------------
# 0) Smoke test: logging exists
# ---------------------------------------------------------------------
print("=== Smoke: logging setup ===")
try:
    log_path = LOG_DIR / "helper.log"
    # touch by importing helper (handlers already configured there)
    with open(log_path, "a") as f:
        f.write("")
    _ok(f"Log file present: {log_path}")
except Exception as e:
    _bad(f"Logging init failed: {e}")


# ---------------------------------------------------------------------
# 1) Envelope extractor on synthetic audio
# ---------------------------------------------------------------------
print("\n=== Testing gammatone_hilbert_envelope (synthetic) ===")
try:
    fs = 16000
    dur = 1.0
    t = np.linspace(0, dur, int(fs*dur), endpoint=False)
    x = 0.6*np.sin(2*np.pi*440*t) + 0.2*np.sin(2*np.pi*220*t)
    env, meta = helper.gammatone_hilbert_envelope(x, fs, num_bands=32, fmin=50.0,
                                                  lowpass_hz=8.0, target_fs=None,
                                                  normalize="unit", return_bands=False)
    assert env.ndim == 1 and env.size == x.size
    assert np.isfinite(env).all()
    assert np.max(env) <= 1.0 + 1e-6
    helper.plot_audio(x, fs, 0.0, 1.0, also_envelope=True, title="Synthetic Audio + Env")
    _ok(f"Envelope shape {env.shape}, meta fs={meta['fs']}")
except Exception as e:
    _bad(f"gammatone_hilbert_envelope failed: {e}")


# ---------------------------------------------------------------------
# 2) Dummy MAT to test get_audio_signal + eeg_and_attended_envelope
# ---------------------------------------------------------------------
print("\n=== Testing with dummy MAT ===")
try:
    n_t = 1000
    n_ch = 32
    trial1 = {
        "eeg": np.random.randn(n_ch, n_t),
        "wavA": np.random.randn(n_t),
        "wavB": np.random.randn(n_t),
        "fsample": 16000,
        # attention: 0..499 -> A(1), 500..999 -> B(2)
        "event": np.array([[0, 500], [1, 2]], dtype=float),
    }
    trial2 = {
        "eeg": np.random.randn(n_ch, n_t),
        "wavA": np.random.randn(n_t),
        "wavB": np.random.randn(n_t),
        "fsample": 16000,
        "event": np.array([[0, 500], [2, 1]], dtype=float),
    }
    dummy_mat = {"data": [trial1, trial2]}

    # get_audio_signal
    aA, fsA = helper.get_audio_signal(dummy_mat, stream="A")
    aB, fsB = helper.get_audio_signal(dummy_mat, stream="B")
    assert aA.ndim == 1 and aB.ndim == 1
    assert aA.size == 2*n_t and aB.size == 2*n_t
    _ok(f"get_audio_signal OK (A:{aA.shape}, B:{aB.shape}, fsA={fsA}, fsB={fsB})")

    # eeg_and_attended_envelope
    eeg_all, env_all = helper.eeg_and_attended_envelope(dummy_mat)
    assert eeg_all.shape[0] == 2*n_t and eeg_all.shape[1] in (n_ch,)
    assert env_all.shape == (2*n_t,)
    _ok(f"eeg_and_attended_envelope OK (EEG:{eeg_all.shape}, ENV:{env_all.shape})")
except Exception as e:
    _bad(f"Dummy MAT tests failed: {e}")


# ---------------------------------------------------------------------
# 3) Subject-based tests (require real .mat). These will be skipped if missing
# ---------------------------------------------------------------------
have_real = False
mat = None
print("\n=== Testing load_mat_for_subject / load_subject (real data) ===")
try:
    if PREPROC_DIR and Path(PREPROC_DIR).exists():
        mat = helper.load_mat_for_subject(PREPROC_DIR, SUBJ_ID)
        assert isinstance(mat, dict) and "data" in mat
        _ok(f"Loaded MAT keys: {list(mat.keys())}")
        have_real = True
    else:
        _bad("DTU_PREPROC_DIR not set or path missing; skipping real-data tests.")
except Exception as e:
    _bad(f"load_mat_for_subject failed: {e}")

if have_real:
    # load_subject
    try:
        D = helper.load_subject(PREPROC_DIR, SUBJ_ID, drop_exg=True)
        eeg = D["eeg"]; fs = float(D["fs"])
        assert eeg.ndim == 2 and eeg.shape[0] <= 66
        assert D["attmask"].ndim == 1 and D["attmask"].size == eeg.shape[1]
        _ok(f"load_subject OK (EEG:{eeg.shape}, fs:{fs})")
    except Exception as e:
        _bad(f"load_subject failed: {e}")

    # subject_eeg_env_ab
    try:
        eeg_TxC, env_att, fs_subj, att_AB = helper.subject_eeg_env_ab(PREPROC_DIR, SUBJ_ID)
        # print(att_AB)
        assert eeg_TxC.ndim == 2 and env_att.ndim == 1
        assert eeg_TxC.shape[0] == env_att.size
        assert att_AB.shape[0] == env_att.size
        _ok(f"subject_eeg_env_ab OK (EEG:{eeg_TxC.shape}, ENV:{env_att.shape}, fs:{fs_subj})")
    except Exception as e:
        _bad(f"subject_eeg_env_ab failed: {e}")

    # plot_window
    try:
        helper.plot_window(PREPROC_DIR, SUBJ_ID, t_start=0.0, t_end=10.0, max_channels=16)
        pngs = sorted(FIG_DIR.glob("S*_eeg_audio_*ms_*ms.png"))
        assert len(pngs) > 0
        _ok(f"plot_window saved: {pngs[-1]}")
    except Exception as e:
        _bad(f"plot_window failed: {e}")

    # check_attention_distribution (prints stats)
    try:
        helper.check_attention_distribution(PREPROC_DIR, SUBJ_ID)
        _ok("check_attention_distribution printed per-trial stats")
    except Exception as e:
        _bad(f"check_attention_distribution failed: {e}")

# ---------------------------------------------------------------------
# 4) plot_eeg_and_audio using synthetic or real data
# ---------------------------------------------------------------------
print("\n=== Testing plot_eeg_and_audio ===")
try:
    if have_real:
        # use real subject outputs
        eeg_TxC, env_att, fs_subj, _att_AB = helper.subject_eeg_env_ab(PREPROC_DIR, SUBJ_ID)
        png = helper.plot_eeg_and_audio(eeg_TxC, env_att, fs_subj, t_start=5.0, t_end=15.0,
                                        num_bands=32, fs_audio=fs_subj, max_channels=16,
                                        title="Real Subject Window")
    else:
        # synthetic fallback
        fs_eeg = 64
        T = 10*fs_eeg
        eeg_TxC = np.random.randn(T, 8)
        # create a slow-varying envelope-like signal
        t = np.arange(T)/float(fs_eeg)
        audio = 0.5 + 0.4*np.sin(2*np.pi*1.0*t)
        png = helper.plot_eeg_and_audio(eeg_TxC, audio, fs_eeg, t_start=1.0, t_end=9.0,
                                        num_bands=16, fs_audio=fs_eeg, max_channels=8,
                                        title="Synthetic EEG + Envelope")
    assert Path(png).exists()
    _ok(f"plot_eeg_and_audio saved: {png}")
except Exception as e:
    _bad(f"plot_eeg_and_audio failed: {e}")


# ---------------------------------------------------------------------
# 5) get_audio_signal (real data)
# ---------------------------------------------------------------------
if have_real:
    print("\n=== Testing get_audio_signal (real MAT) ===")
    try:
        aA, fsA = helper.get_audio_signal(mat, stream="A")
        aB, fsB = helper.get_audio_signal(mat, stream="B")
        assert aA.ndim == 1 and aB.ndim == 1 and fsA > 0 and fsB > 0
        _ok(f"get_audio_signal(real) OK (A:{aA.shape}, B:{aB.shape}, fsA={fsA}, fsB={fsB})")
    except Exception as e:
        _bad(f"get_audio_signal(real) failed: {e}")


# ---------------------------------------------------------------------
# 6) Wrap up
# ---------------------------------------------------------------------
print("\n=== Done ===")
plt.close('all')
