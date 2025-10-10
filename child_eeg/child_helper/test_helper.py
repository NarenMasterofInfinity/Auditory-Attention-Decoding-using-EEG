"""
tester.py

Validation script for EEG preprocessing and data loading functions.
It tests both CHILD and GENERAL EEG pipelines:
    - preprocess_eeg_child_preprocess()
    - load_data_child_treatment()
    - general_preprocess_eeg()
    - load_eeg_general_treatment()
"""

import os
import numpy as np
import traceback

# === Import your modules ===
from child_helper import preprocess_eeg_child_preprocess, load_data_child_treatment, general_preprocess_eeg, load_eeg_general_treatment



# -------------------- CONFIG --------------------
CHILD_BASE_DIR =  "/home/naren-root/Documents/FYP/AAD/Project/Child_eeg"
GENERAL_BASE_DIR =  "/home/naren-root/Documents/FYP/AAD/Project/Child_eeg"

CHILD_CSV = "../sub_1.csv"
GENERAL_CSV = "../sub_1.csv"


# -------------------- Helper --------------------
def print_section(title):
    print("\n" + "="*80)
    print(f"🔹 {title}")
    print("="*80 + "\n")


# -------------------- 1️⃣ Test: Child EEG Preprocessing --------------------
try:
    print_section("TESTING: preprocess_eeg_child_preprocess()")

    test_eeg_file = os.path.join(CHILD_BASE_DIR, os.listdir(CHILD_BASE_DIR)[0])
    if not test_eeg_file.endswith(".set"):
        test_eeg_file = [f for f in os.listdir(CHILD_BASE_DIR) if f.endswith(".set")][0]
        test_eeg_file = os.path.join(CHILD_BASE_DIR, test_eeg_file)

    eeg_data, sfreq = preprocess_eeg_child_preprocess(test_eeg_file)
    print(f"✅ Child EEG preprocessing successful → shape={eeg_data.shape}, sfreq={sfreq} Hz")

except Exception as e:
    print("❌ Child EEG preprocessing failed.")
    traceback.print_exc()


# -------------------- 2️⃣ Test: Child EEG+Audio Loader --------------------
try:
    print_section("TESTING: load_data_child_treatment()")

    data_child = load_data_child_treatment(CHILD_CSV)
    print(f"✅ Child loader successful → EEG subjects: {len(data_child['eeg'])}")
    print(f"   First EEG shape: {data_child['eeg'][0].shape}")
    print(f"   First Audio shape: {data_child['audio'][0].shape}")

except Exception as e:
    print("❌ Child loader failed.")
    traceback.print_exc()


# -------------------- 3️⃣ Test: General EEG Preprocessing --------------------
try:
    print_section("TESTING: general_preprocess_eeg()")

    test_eeg_file = os.path.join(GENERAL_BASE_DIR, os.listdir(GENERAL_BASE_DIR)[0])
    if not test_eeg_file.endswith(".set"):
        test_eeg_file = [f for f in os.listdir(GENERAL_BASE_DIR) if f.endswith(".set")][0]
        test_eeg_file = os.path.join(GENERAL_BASE_DIR, test_eeg_file)

    eeg_data, sfreq = general_preprocess_eeg(test_eeg_file)
    print(f"✅ General EEG preprocessing successful → shape={eeg_data.shape}, sfreq={sfreq} Hz")

except Exception as e:
    print("❌ General EEG preprocessing failed.")
    traceback.print_exc()


# -------------------- 4️⃣ Test: General EEG+Audio Loader --------------------
try:
    print_section("TESTING: load_eeg_general_treatment()")

    data_general = load_eeg_general_treatment(GENERAL_CSV, base_dir=GENERAL_BASE_DIR)
    print(f"✅ General loader successful → EEG subjects: {len(data_general['eeg'])}")
    print(f"   First EEG shape: {data_general['eeg'][0].shape}")
    print(f"   First Audio shape: {data_general['audio'][0].shape}")

except Exception as e:
    print("❌ General loader failed.")
    traceback.print_exc()


# -------------------- DONE --------------------
print("\n" + "="*80)
print("✅ TESTING COMPLETE")
print("="*80)
