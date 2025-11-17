import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# EEG Feature Extractor (FAA, OASI, FAP, TBR) & PSD Plots
# ----------------------------------------------------------

# --- File path setup ---
# (assume script is run from its directory)
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
target_folder = os.path.join(script_dir, "../pre-processed/")

REACTION_DELAY_SECONDS = 0.5  

testcase_name = "120s" 
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

# define key channels (using O2 as in the original script)
KEY_CHANNELS_F = ['F3', 'F4']
KEY_CHANNELS_O = ['O1', 'O2'] 
KEY_CHANNELS_ALL = KEY_CHANNELS_F + KEY_CHANNELS_O

print(f"Channels found: {epochs.ch_names}")

# check for missing channels
missing_channels = [ch for ch in KEY_CHANNELS_ALL if ch not in epochs.ch_names]
if missing_channels:
    print(f"*** WARNING: Missing key channels: {missing_channels} ***")
    print("Metrics involving these channels will be 'nan'.")


# Compute Power Spectral Density (PSD)
spectrum = epochs.compute_psd(method='welch', fmin=1, fmax=45, n_fft=128)

# --- NEW PLOT 1: Topomaps for Theta, Alpha, Beta ---
try:
    bands = {'Theta': (4.0, 8.0), 'Alpha': (8.0, 13.0), 'Beta': (13.0, 30.0)}
    topo_fig = spectrum.plot_topomap(bands=bands, ch_type='eeg', show=False, vlim=(None, None))
    
    topo_img_path = os.path.join(script_dir, f"EEG_topo_plot_{testcase_name}.png")
    topo_fig.savefig(topo_img_path)
    print(f"Topomap plot saved to: {topo_img_path}")
    plt.close(topo_fig) 

except Exception as e:
    print(f"Could not generate topomap plot. Error: {e}")
# ---------------------------------------------------


# PSD data extract, calculate average
psd_data_all_epochs = spectrum.get_data()
psd = np.mean(psd_data_all_epochs, axis=0) 
freqs = spectrum.freqs

# --- NEW PLOT 2: PSD Curves for Key Channels ---
try:
    plt.figure(figsize=(10, 6))
    channels_to_plot = [ch for ch in KEY_CHANNELS_ALL if ch in epochs.ch_names]
    
    for ch in channels_to_plot:
        ch_idx = epochs.ch_names.index(ch)
        plt.plot(freqs, 10 * np.log10(psd[ch_idx, :]), label=ch)
        
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.title(f'PSD Curves for Key Channels ({testcase_name})')
    plt.legend()
    plt.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    
    psd_curve_img_path = os.path.join(script_dir, f"EEG_psd_curves_{testcase_name}.png")
    plt.savefig(psd_curve_img_path)
    print(f"PSD curve plot saved to: {psd_curve_img_path}")
    plt.close()

except Exception as e:
    print(f"Could not generate PSD curve plot. Error: {e}")
# ---------------------------------------------------


# Find indices for frequency bands
theta_idx = np.where((freqs >= 4) & (freqs <= 8))[0]
alpha_idx = np.where((freqs >= 8) & (freqs <= 13))[0]
beta_idx = np.where((freqs >= 13) & (freqs <= 30))[0]

# Mean power per channel for each band
theta_power = np.mean(psd[:, theta_idx], axis=1)
alpha_power = np.mean(psd[:, alpha_idx], axis=1)
beta_power = np.mean(psd[:, beta_idx], axis=1)

# Create dictionaries for easy access
theta_dict = dict(zip(epochs.ch_names, theta_power))
alpha_dict = dict(zip(epochs.ch_names, alpha_power))
beta_dict = dict(zip(epochs.ch_names, beta_power))

print("\nAlpha power by channel:")
for ch, pwr in alpha_dict.items():
    print(f"  {ch}: {pwr:.4f}")

# Helper functions
def safe_log(val):
    if pd.isna(val) or val <= 0:
        return np.nan
    return np.log(val)

def get_power(ch, power_dict):
    return power_dict[ch] if ch in power_dict else np.nan

# Get power values for key channels from all bands
F3_alpha, F4_alpha = [get_power(x, alpha_dict) for x in KEY_CHANNELS_F]
O1_alpha, O2_alpha = [get_power(x, alpha_dict) for x in KEY_CHANNELS_O]

F3_theta, F4_theta = [get_power(x, theta_dict) for x in KEY_CHANNELS_F]
F3_beta, F4_beta = [get_power(x, beta_dict) for x in KEY_CHANNELS_F]

# Calculate standard metrics
FAA = safe_log(F4_alpha) - safe_log(F3_alpha)
FAP = (F3_alpha + F4_alpha) / 2
OASI = (O1_alpha + O2_alpha) / 2

# Calculate NEW metric: TBR (Theta/Beta Ratio)
frontal_theta_avg = (F3_theta + F4_theta) / 2
frontal_beta_avg = (F3_beta + F4_beta) / 2
TBR = frontal_theta_avg / (frontal_beta_avg + 1e-10) 

# Updated metrics dictionary
metrics = {
    'FAA': FAA,
    'FAP': FAP,
    'OASI': OASI,
    'TBR': TBR
}

print("\n--- Computed metrics ---")
for k, v in metrics.items():
    print(f"{k}: {v:.5f}")

# Save metrics as CSV
csv_path = os.path.join(script_dir, f"EEG_features_{testcase_name}.csv")
pd.DataFrame([metrics]).to_csv(csv_path, index=False)
print(f"Feature CSV saved to: {csv_path}")

# --- MODIFIED: Summary bar plot (with value labels) ---
plt.figure(figsize=(7, 4)) 
colors = ['tomato', 'royalblue', 'mediumseagreen', 'orange']

# store keys and values
metric_keys = list(metrics.keys())
metric_values = list(metrics.values())

# create the bar plot and store the container
bars = plt.bar(metric_keys, metric_values, color=colors)

# --- NEW: add text labels on bars ---
# this automatically adds labels on top of each bar
# 'fmt' formats the number (3 decimal places)
plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=10)
# ------------------------------------

plt.title(f"EEG Feature Summary ({testcase_name})")
plt.ylabel("Value") 

# adjust y-axis limits to give space for labels
ymin, ymax = plt.ylim()
# add 10% padding to the top (or a minimum of 0.1)
padding = max(abs(ymax) * 0.1, 0.1) 
plt.ylim(ymin, ymax + padding) 

plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()

img_path = os.path.join(script_dir, f"EEG_features_{testcase_name}.png")
plt.savefig(img_path)
print(f"Feature bar plot saved to: {img_path}")
plt.close() 

print("\nAll feature extraction completed successfully.")