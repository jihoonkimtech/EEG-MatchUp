import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# EEG Feature Extractor (FAA, OAA, OASI)
# ----------------------------------------------------------

# --- File path setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
target_folder = os.path.join(script_dir, "../pre-processed/")

REACTION_DELAY_SECONDS = 0.5  

testcase_name = "challenge" 
# ----------------------------------------------------------

# Changed underscore to hyphen to match the pre-processor's output file
file_name = f'pre-processed_{testcase_name}-epo.fif'
input_file = os.path.join(target_folder, file_name)

print(f"--- Starting Feature Extraction: '{testcase_name}' ---")

# Load preprocessed EPOCHED data
try:
    epochs = mne.read_epochs(input_file, preload=True)
    print(f"Loaded {len(epochs)} epochs from: {input_file}")
    print(f"Original epoch time range: {epochs.tmin}s to {epochs.tmax}s")

    # REACTION_DELAY
    epochs.crop(tmin=REACTION_DELAY_SECONDS)
    print(f"Cropped epochs for delay: New range {epochs.tmin}s to {epochs.tmax}s")

except FileNotFoundError:
    raise FileNotFoundError(f"*** ERROR: cannot find file ***\n{input_file}")
except ValueError:
    raise ValueError(f"*** ERROR: {input_file}은 Epoch 파일이 아닌 것 같습니다.***")

print(f"Channels: {epochs.ch_names}")

# Compute Power Spectral Density (PSD)
spectrum = epochs.compute_psd(method='welch', fmin=1, fmax=45, n_fft=128)

# PSD data extract, calculate average
# spectrum.get_data() is return (n_epochs, n_channels, n_freqs) 3D array
psd_data_all_epochs = spectrum.get_data()

# all epochs(axis=0)'s average -> (n_channels, n_freqs) 2D array
psd = np.mean(psd_data_all_epochs, axis=0) 

freqs = spectrum.freqs

# Find index for alpha range (8-13 Hz)
alpha_idx = np.where((freqs >= 8) & (freqs <= 13))[0]

# Mean alpha power per channel
alpha_power = np.mean(psd[:, alpha_idx], axis=1)
alpha_dict = dict(zip(epochs.ch_names, alpha_power))

print("\nAlpha power by channel:")
for ch, pwr in alpha_dict.items():
    print(f"  {ch}: {pwr:.4f}")

# Calculate EEG indices
def safe_log(val):
    if pd.isna(val):
        return np.nan
    return np.log(val + 1e-10)

def get_if_exists(ch):
    return alpha_dict[ch] if ch in alpha_dict else np.nan

F3, F4, O1, O2 = [get_if_exists(x) for x in ['F3', 'F4', 'O1', 'O2']]

# Calculate standard metrics
FAA = safe_log(F4) - safe_log(F3)
OAA = safe_log(O2) - safe_log(O1)
OASI = (O1 + O2) / 2

metrics = {
    'FAA': FAA,
    'OAA': OAA,
    'OASI': OASI
}

print("\n--- Computed metrics ---")
for k, v in metrics.items():
    print(f"{k}: {v:.5f}")

# Save metrics as CSV
csv_path = os.path.join(script_dir, f"EEG_features_{testcase_name}.csv")
pd.DataFrame([metrics]).to_csv(csv_path, index=False)
print(f"Feature CSV saved to: {csv_path}")

# Simple bar plot summary
plt.figure(figsize=(6, 4))
colors = ['tomato', 'royalblue', 'mediumseagreen']
plt.bar(metrics.keys(), metrics.values(), color=colors)
plt.title(f"EEG Feature Summary ({testcase_name})")
plt.ylabel("Value (log-scaled differences)")

plt.ylim(-2.5, 2.5)

plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

img_path = os.path.join(script_dir, f"EEG_features_{testcase_name}.png")
plt.savefig(img_path)
print(f"Feature bar plot saved to: {img_path}")

print("\nAll feature extraction completed successfully.")
plt.show()