import os
import h5py
import numpy as np

# === global constant ===
DATASET_DIR = "/home/naren-root/KUL/processed"  # <-- change to your actual processed folder path

def load_subject(subject_id):
    """
    Load all trials for a subject from processed .h5

    Args:
        subject_id (str): e.g. "S1"

    Returns:
        dict with keys:
          'eeg'         -> np.ndarray (T_total, C)
          'env_left'    -> np.ndarray (T_total, B)
          'env_right'   -> np.ndarray (T_total, B)
          'attended'    -> np.ndarray (T_total, B)
          'fs'          -> int
          'attended_ear_labels' -> list[str] per trial
    """
    file_path = os.path.join(DATASET_DIR, f"{subject_id}.h5")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")

    eeg_all, envL_all, envR_all, att_all = [], [], [], []
    ears = []

    with h5py.File(file_path, "r") as h5:
        fs = int(h5["meta/target_fs"][()])
        trials = list(h5["trials"].keys())
        trials.sort()
        for tr in trials:
            tg = h5[f"trials/{tr}"]
            eeg = np.array(tg["eeg"])
            envL = np.array(tg["env_left"])
            envR = np.array(tg["env_right"])
            ear = tg["attended_ear"][()].decode()

            if ear.upper() == "L" or ear.upper() == "A":
                att = envL
            elif ear.upper() == "R" or ear.upper() == "B":
                att = envR
            else:
                # unknown, fill zeros same shape
                att = np.zeros_like(envL)

            eeg_all.append(eeg)
            envL_all.append(envL)
            envR_all.append(envR)
            att_all.append(att)
            ears.append(ear)

    eeg_cat  = np.concatenate(eeg_all,  axis=0)
    envL_cat = np.concatenate(envL_all, axis=0)
    envR_cat = np.concatenate(envR_all, axis=0)
    att_cat  = np.concatenate(att_all,  axis=0)

    return {
        "eeg": eeg_cat,
        "env_left": envL_cat,
        "env_right": envR_cat,
        "attended": att_cat,
        "fs": fs,
        "attended_ear_labels": ears,
    }

# === quick test ===
if __name__ == "__main__":
    data = load_subject("S1")
    for k,v in data.items():
        if isinstance(v, np.ndarray):
            print(k, v.shape)
        else:
            print(k, v)
