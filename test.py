import scipy.io

# Load the .mat file
mat = scipy.io.loadmat("D:\FYP\DATA_preproc\S1_data_preproc.mat", struct_as_record=False, squeeze_me=True)
data = mat["data"]

# Access fields inside 'data'
print("Fields available:", data._fieldnames)

# Example: Access sampling frequency
print("fsample:", data.fsample)

# Example: Check EEG shape/data
print("EEG type:", type(data.eeg))
try:
    print("EEG shape:", data.eeg.shape)
except AttributeError:
    print("EEG might be a list or struct:", data.eeg)


# Inspect fsample
print("fsample fields:", dir(data.fsample))

# If it has __dict__, check inside
print("fsample as dict:", data.fsample.__dict__)
