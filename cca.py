import os
import csv
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal

import helper

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
FIG_DIR = Path("Figures"); FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("Logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = Path("results"); RES_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = RES_DIR / "cca_results.csv"

_logger = logging.getLogger("cca_batch"); _logger.setLevel(logging.INFO)
if not _logger.handlers:
    fh = logging.FileHandler(LOG_DIR / "cca_batch.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(funcName)s | %(message)s"))
    _logger.addHandler(fh)


def _pearsonr(y, yhat):
    y = np.asarray(y, float).ravel()
    yhat = np.asarray(yhat, float).ravel()
    y = y - y.mean(); yhat = yhat - yhat.mean()
    num = np.dot(y, yhat)
    den = np.sqrt((y**2).sum() * (yhat**2).sum()) + 1e-12
    return float(num / den)


def _zscore_train(X):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-12
    return (X - mu) / sd, mu, sd


def _zscore_apply(X, mu, sd):
    return (X - mu) / (sd + 1e-12)


def _make_lagged(X_TxC, lags_samp):
    """Create a lagged EEG design matrix.
    X_TxC: EEG with shape (T, C).
    lags_samp: 1-D array of non-negative integer lags (in samples). 0 means no lag.

    Returns X_lagged (T - max_lag, C * L), and the slice indices used to align y.
    """
    X = np.asarray(X_TxC, float)
    T, C = X.shape
    lags_samp = np.asarray(lags_samp, int)
    assert np.all(lags_samp >= 0), "lags must be >= 0"
    L = lags_samp.size
    maxlag = int(lags_samp.max()) if L else 0
    if L == 0:
        return X, slice(0, T)
    # Build block matrix
    rows = T - maxlag
    Xlag = np.zeros((rows, C * L), dtype=float)
    for i, Ls in enumerate(lags_samp):
        Xi = X[maxlag - Ls : T - Ls, :]  # shift upwards
        Xlag[:, i*C:(i+1)*C] = Xi
    return Xlag, slice(maxlag, T)


def _unique_png(prefix):
    p = FIG_DIR / f"{prefix}.png"; k = 1
    while p.exists():
        p = FIG_DIR / f"{prefix}_{k}.png"; k += 1
    return p


def plot_reconstruction(y_true, y_pred, fs, t_start=0.0, duration=20.0, title=None, out_path=None):
    """Plot true vs predicted envelopes in a window."""
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    T = min(y_true.size, y_pred.size)
    i0 = max(0, int(np.floor(t_start * fs)))
    i1 = min(T, int(np.ceil((t_start + duration) * fs)))
    if i1 <= i0:
        i0, i1 = 0, min(T, int(10*fs))
    t = np.arange(i0, i1) / float(fs)
    yt = y_true[i0:i1]; yp = y_pred[i0:i1]
    # z-score for display
    yt = (yt - yt.mean()) / (yt.std() + 1e-12)
    yp = (yp - yp.mean()) / (yp.std() + 1e-12)
    plt.figure(figsize=(12, 4))
    plt.plot(t, yt, lw=1.2, label="True env")
    plt.plot(t, yp, lw=1.2, label="CCA recon")
    plt.xlabel("Time (s)"); plt.ylabel("z")
    plt.title(title or "Envelope Reconstruction (CCA)")
    plt.legend(); plt.tight_layout()
    if out_path is None:
        out_path = _unique_png("cca_envelope_reconstruction")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200); plt.close()
    _logger.info(f"Saved reconstruction plot -> {out_path}")
    return str(out_path)


# ---------------------------------------------------------------------
# CCA (1D Y reduces to multiple correlation / OLS)
# ---------------------------------------------------------------------

def fit_cca_1d(X_train, y_train):
    """Fit CCA for 1-D target (equivalent to multiple correlation / OLS).
    Returns a model dict with beta and z-score params.
    """
    X_train = np.asarray(X_train, float)
    y_train = np.asarray(y_train, float).ravel()

    Xz, muX, sdX = _zscore_train(X_train)
    yz, muy, sdy = _zscore_train(y_train[:, None])
    yz = yz.ravel()

    # OLS / CCA weight for 1-D Y
    # beta minimizes ||Xz beta - yz||^2
    XtX = Xz.T @ Xz + 1e-6 * np.eye(Xz.shape[1])
    Xty = Xz.T @ yz
    beta = np.linalg.solve(XtX, Xty)

    # train correlation
    yhat_z = Xz @ beta
    r_train = _pearsonr(yz, yhat_z)

    return {
        'beta': beta,
        'muX': muX,
        'sdX': sdX,
        'muy': muy.ravel()[0],
        'sdy': sdy.ravel()[0],
        'r_train': r_train,
    }


def predict_cca_1d(model, X):
    X = np.asarray(X, float)
    Xz = _zscore_apply(X, model['muX'], model['sdX'])
    yhat_z = Xz @ model['beta']
    # back to original scale
    yhat = yhat_z * model['sdy'] + model['muy']
    return yhat


# ---------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------

def run_cca_for_subject(preproc_dir, subj_id, train_ratio=0.8, use_lags=True, max_lag_ms=250, lag_step_ms=None, plot_secs=20.0):
    """
    Load EEG + attended envelope via helper, build (optional) lagged EEG, split 80/20,
    fit 1-D CCA (OLS), report Pearson r on train/test, and save a reconstruction plot.
    """
    # 1) Load
    eeg_TxC, env_att, fs, att_AB = helper.subject_eeg_env_ab(preproc_dir, subj_id)
    T, C = eeg_TxC.shape
    _logger.info(f"Loaded subject S{subj_id}: EEG {eeg_TxC.shape}, fs={fs}")

    # 2) Lags
    if use_lags:
        if lag_step_ms is None:
            # default: 1-sample step
            lags_samp = np.arange(0, int(round(max_lag_ms * 1e-3 * fs)) + 1, 1, dtype=int)
        else:
            step = max(1, int(round(lag_step_ms * 1e-3 * fs)))
            lags_samp = np.arange(0, int(round(max_lag_ms * 1e-3 * fs)) + 1, step, dtype=int)
    else:
        lags_samp = np.array([0], dtype=int)

    X_lag, ys = _make_lagged(eeg_TxC, lags_samp)
    y = env_att[ys]
    T2 = X_lag.shape[0]

    # 3) Train/Test split (time-contiguous)
    n_train = int(train_ratio * T2)
    idx_train = slice(0, n_train)
    idx_test  = slice(n_train, T2)

    Xtr, ytr = X_lag[idx_train, :], y[idx_train]
    Xte, yte = X_lag[idx_test, :], y[idx_test]

    # 4) Fit and evaluate
    model = fit_cca_1d(Xtr, ytr)
    yhat_tr = predict_cca_1d(model, Xtr)
    yhat_te = predict_cca_1d(model, Xte)

    r_tr = _pearsonr(ytr, yhat_tr)
    r_te = _pearsonr(yte, yhat_te)

    _logger.info(f"CCA (lags={list(lags_samp)[:5]}... total {lags_samp.size}) -> r_train={r_tr:.4f}, r_test={r_te:.4f}")

    # 5) Plot a window from the test set
    test_start_sec = (n_train / fs)
    plot_path = plot_reconstruction(y, np.r_[yhat_tr, yhat_te], fs,
                                    t_start=test_start_sec,
                                    duration=plot_secs,
                                    title=f"S{subj_id} CCA recon | r_test={r_te:.3f}")

    return {
        'r_train': r_tr,
        'r_test': r_te,
        'plot_path': plot_path,
        'lags_samp': lags_samp,
        'fs': fs,
        'n_train': n_train,
        'T_used': T2,
    }




# ---------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------

def run_batch(preproc_dir,
              subjects=range(1, 19),
              train_ratio=0.8,
              use_lags=True,
              max_lag_ms=250,
              lag_step_ms=None,
              plot_secs=20.0,
              csv_path=CSV_PATH):
    headers = [
        "subject",
        "r_train",
        "r_test",
        "fs",
        "lags_count",
        "max_lag_ms",
        "lag_step_ms",
        "n_train_samples",
        "n_total_samples",
        "train_seconds",
        "total_seconds",
        "figure_path",
        "status",
        "error",
    ]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for sid in subjects:
            status = "ok"; err = ""; fig_final = ""; rtr = rte = fs = 0.0
            lcount = 0; ntr = nall = 0
            try:
                _logger.info(f"Running CCA for subject S{sid:02d}")
                res = run_cca_for_subject(preproc_dir, sid,
                                          train_ratio=train_ratio,
                                          use_lags=use_lags,
                                          max_lag_ms=max_lag_ms,
                                          lag_step_ms=lag_step_ms,
                                          plot_secs=plot_secs)
                rtr = float(res.get('r_train', 0.0))
                rte = float(res.get('r_test', 0.0))
                fs  = float(res.get('fs', 0.0))
                lags = res.get('lags_samp', [])
                lcount = len(lags) if lags is not None else 0
                ntr = int(res.get('n_train', 0))
                nall = int(res.get('T_used', 0))

                # Rename/standardize figure name (one per subject)
                plot_path = Path(res.get('plot_path', ''))
                tgt = FIG_DIR / f"cca_S{sid:02d}_reconstruction.png"
                try:
                    if plot_path and plot_path.exists():
                        if tgt.exists():
                            tgt.unlink()
                        plot_path.replace(tgt)
                        fig_final = str(tgt)
                    else:
                        fig_final = ""
                except Exception as e_move:
                    _logger.warning(f"Rename figure failed for S{sid:02d}: {e_move}")
                    fig_final = str(plot_path)

            except Exception as e:
                status = "error"
                err = str(e)
                _logger.exception(f"Subject S{sid:02d} failed")

            train_sec = ntr / fs if fs else 0.0
            total_sec = nall / fs if fs else 0.0

            w.writerow([
                sid, rtr, rte, fs, lcount, max_lag_ms, (lag_step_ms if lag_step_ms else 1),
                ntr, nall, train_sec, total_sec, fig_final, status, err
            ])
            f.flush()

    return str(csv_path)


if __name__ == "__main__":
    PREPROC_DIR = "/home/naren-root/Documents/FYP/AAD/Notebooks/Dataset/DATA_preproc"
    

    START = 1
    END   = 18

    LAG_MAX  = 250
    LAG_STEP = None

    csv_out = run_batch(PREPROC_DIR,
                        subjects=range(START, END+1),
                        train_ratio=0.8,
                        use_lags=True,
                        max_lag_ms=LAG_MAX,
                        lag_step_ms=LAG_STEP,
                        plot_secs=20.0)
    print("Saved results:", csv_out)
