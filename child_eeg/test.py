import mne

raw = mne.io.read_raw_eeglab("../../Child_eeg/sub-NDARAC904DMU_task-DiaryOfAWimpyKid_eeg.set", preload=True)
print("Sampling rate:", raw.info['sfreq'])

try:
    raw.filter(0.5, 40., fir_design='firwin', phase='zero')
    print("Filter applied successfully")
except Exception as e:
    print("Filter skipped:", e)
