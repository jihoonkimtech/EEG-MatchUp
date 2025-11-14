import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# EEG Feature Extractor for alpha-based indices
# Works with preprocessed .fif files (F3, F4, O1, O2)

# File path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
target_folder = os.path.join(script_dir, "../pre-processed/")


testcase_name = "info"
file_name = f'pre-processed_{testcase_name}.fif'
input_file = os.path.join(target_folder, file_name)

print(f"--- Starting Feature Extraction: '{testcase_name}' ---")

# Load preprocessed EEG data
try:
    raw = mne.io.read_raw_fif(input_file, preload=True)
    print(f"Loaded file: {input_file}")
except FileNotFoundError:
    raise FileNotFoundError(f"*** ERROR: cannot find file ***\n{input_file}")

print(f"Channels: {raw.ch_names}")

# Compute Power Spectral Density (PSD)
# Use Welch method for stationary PSD estimation
spectrum = raw.compute_psd(method='welch', fmin=1, fmax=45, n_fft=2048)

psd = spectrum.get_data()
freqs = spectrum.freqs

# Find index for alpha range (8-13 Hz)
alpha_idx = np.where((freqs >= 8) & (freqs <= 13))[0]

# Mean alpha power per channel
alpha_power = np.mean(psd[:, alpha_idx], axis=1)
alpha_dict = dict(zip(raw.ch_names, alpha_power))
print("\nAlpha power by channel:")
for ch, pwr in alpha_dict.items():
    print(f"  {ch}: {pwr:.4f}")

# Calculate EEG indices
def safe_log(val):
    """avoid log(0) by adding epsilon"""
    return np.log(val + 1e-10)

def get_if_exists(ch):
    """return alpha power if channel exists"""
    return alpha_dict[ch] if ch in alpha_dict else np.nan

F3, F4, O1, O2 = [get_if_exists(x) for x in ['F3','F4','O1','O2']]

FAA = safe_log(F4) - safe_log(F3)
OAA = safe_log(O2) - safe_log(O1)
OASI = (O1 + O2)/2  # baseline unavailable (treated as raw alpha power mean)
metrics = {'FAA': FAA, 'OAA': OAA, 'OASI': OASI}

# Optional parietal alpha balance (if P3/P4 exist)
if 'P3' in alpha_dict and 'P4' in alpha_dict:
    P3, P4 = alpha_dict['P3'], alpha_dict['P4']
    PAB = safe_log(P4) - safe_log(P3)
    metrics['PAB'] = PAB

print("\n--- Computed metrics ---")
for k, v in metrics.items():
    print(f"{k}: {v:.5f}")

# Save metrics as CSV
csv_path = os.path.join(script_dir, f"EEG_features_{testcase_name}.csv")
pd.DataFrame([metrics]).to_csv(csv_path, index=False)
print(f"\nFeature CSV saved to: {csv_path}")

# Simple bar plot summary
plt.figure(figsize=(6,4))
plt.bar(metrics.keys(), metrics.values(), color=['tomato','royalblue','mediumseagreen','orange'])
plt.title(f"EEG Feature Summary ({testcase_name})")
plt.ylabel("Value (log-scaled differences)")
plt.ylim(-0.5, 2.5)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

img_path = os.path.join(script_dir, f"EEG_features_{testcase_name}.png")
plt.savefig(img_path)
print(f"Feature bar plot saved to: {img_path}")

plt.show()

print("\nAll feature extraction completed successfully.")
