# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Ridge
# from mne.decoding import ReceptiveField

# # -------------------------------
# # 1. Simulate stimulus and EEG data
# # -------------------------------
# np.random.seed(42)

# sfreq = 100.  # Sampling frequency (Hz)
# n_times = 1000  # Number of time samples
# n_channels = 32  # EEG channels

# # Stimulus: 1D feature (e.g., audio envelope)
# stim = np.random.randn(n_times, 1)

# # Simulate EEG with some lagged response
# time_lags = np.arange(-0.1, 0.4, 1/sfreq)  # -100 ms to 400 ms
# true_filter = np.sin(2 * np.pi * 5 * time_lags)[:, np.newaxis]  # example TRF kernel
# eeg = np.convolve(stim[:, 0], true_filter[:, 0], mode="same")
# eeg = eeg[:, np.newaxis] + 0.05 * np.random.randn(n_times, 1)  # add noise
# eeg = np.tile(eeg, (1, n_channels))  # replicate across channels

# print("Stimulus shape:", stim.shape)
# print("EEG shape:", eeg.shape)

# # -------------------------------
# # 2. Define TRF model
# # -------------------------------
# tmin, tmax = -0.1, 0.4  # TRF window: -100 ms to +400 ms
# alphas = [1, 10, 100]   # Regularization values

# rf = ReceptiveField(
#     tmin=tmin,
#     tmax=tmax,
#     sfreq=sfreq,
#     feature_names=["stimulus"],
#     estimator=Ridge(alpha=1.0)
# )


# # -------------------------------
# # 3. Train model
# # -------------------------------
# # Train on first 70% of data, test on last 30%
# n_train = int(0.7 * n_times)

# rf.fit(stim[:n_train], eeg[:n_train])

# # -------------------------------
# # 4. Predict on test data
# # -------------------------------
# pred = rf.predict(stim[n_train:])

# print("Prediction shape:", pred.shape)

# # -------------------------------
# # 5. Evaluate performance
# # -------------------------------
# from sklearn.metrics import r2_score

# for ch in range(3):  # show first 3 channels
#     score = r2_score(eeg[n_train:, ch], pred[:, ch])
#     print(f"Channel {ch}: R^2 = {score:.3f}")

# # -------------------------------
# # 6. Plot TRF weights
# # -------------------------------
# times = rf.delays_ / sfreq  # convert lags to seconds

# plt.figure(figsize=(10, 6))
# for ch in range(3):
#     plt.plot(times, rf.coef_[ch, 0, :], label=f"Channel {ch}")
# plt.axvline(0, color="k", linestyle="--", label="Stimulus onset")
# plt.xlabel("Time lag (s)")
# plt.ylabel("TRF weight")
# plt.title("Estimated Temporal Response Function (TRF)")
# plt.legend()
# plt.show()

# # -------------------------------
# # 7. Compare predicted vs actual EEG (one channel)
# # -------------------------------
# plt.figure(figsize=(12, 4))
# plt.plot(eeg[n_train:, 0], label="True EEG (Ch 0)")
# plt.plot(pred[:, 0], label="Predicted EEG (Ch 0)", alpha=0.7)
# plt.title("Prediction vs True EEG (Channel 0)")
# plt.xlabel("Time samples")
# plt.ylabel("EEG signal")
# plt.legend()
# plt.show()


import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
from mne.decoding import ReceptiveField
import helper
from pathlib import Path

# -------------------------------
# CONFIG
# -------------------------------
PREPROC_DIR = r"F:\fyp"   # <-- change if needed
subjects = range(1, 19)   # 18 subjects
tmin, tmax = -0.1, 0.4    # TRF window in seconds
alpha = 1.0               # Ridge regularization

results = []

# -------------------------------
# LOOP OVER SUBJECTS
# -------------------------------
for subj_id in subjects:
    print(f"\n=== Subject {subj_id} ===")

    # Load EEG & attended envelope
    eeg, env_att, fs, att_AB = helper.subject_eeg_env_ab(PREPROC_DIR, subj_id)

    # Shapes
    print("EEG shape:", eeg.shape, "Envelope shape:", env_att.shape)

    # Train/test split (80/20)
    n_samples = len(env_att)
    n_train = int(0.8 * n_samples)

    X_train, X_test = eeg[:n_train], eeg[n_train:]
    y_train, y_test = env_att[:n_train], env_att[n_train:]

    # Define TRF (backward: EEG -> stimulus envelope)
    rf = ReceptiveField(
        tmin=tmin,
        tmax=tmax,
        sfreq=fs,
        feature_names=[f"EEG{ch}" for ch in range(eeg.shape[1])],
        estimator=Ridge(alpha=alpha),
        scoring="r2",
    )

    # Fit
    rf.fit(X_train, y_train)

    # Predict
    y_pred = rf.predict(X_test).ravel()

    # Pearson correlation
    corr, _ = pearsonr(y_test, y_pred)
    print(f"Pearson correlation: {corr:.3f}")

    results.append({"subject": subj_id, "pearson_corr": corr})

# -------------------------------
# SAVE RESULTS TO CSV
# -------------------------------
df = pd.DataFrame(results)
out_path = Path(PREPROC_DIR) / "results_trf.csv"
df.to_csv(out_path, index=False)

print("\n✅ Done! Results saved to:", out_path)