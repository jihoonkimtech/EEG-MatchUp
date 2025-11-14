import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import annotate_muscle_zscore

# Specify the path to the base folder.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Example: ../raw/base/ or ../raw/info/
target_folder = os.path.join(script_dir, "../raw/challenge/")

# Extract folder name (prefix)
folder_name = os.path.basename(os.path.normpath(target_folder))
print(f"--- Starting Stage 2 Processing: '{folder_name}' ---")

# Automatically construct the path to the .fif file generated in Stage 1
input_filename = f"combined_{folder_name}_raw.fif"
input_file_path = os.path.join(target_folder, input_filename)

# Load the file
try:
    raw = mne.io.read_raw_fif(input_file_path, preload=True)
    print(f"Successfully loaded file: {input_file_path}")
except FileNotFoundError:
    print(f"*** ERROR: File not found! ***")
    print(f"Please check the path: {input_file_path}")
    exit()

# Extract events (rising edge) from Ch6
print("Extracting events from Ch6...")
try:
    raw.set_channel_types({'Ch6': 'misc'})
    # min_duration: Ignore signals shorter than 0.01s (10ms) to prevent noise
    events = mne.find_events(raw, stim_channel='Ch6', output='onset', min_duration=0.01)
    print(f"Found {len(events)} 'Stimulus_Playback' events from Ch6.")

    # 6. Convert to MNE Annotations object
    onsets = events[:, 0] / raw.info['sfreq']  # Convert samples to seconds
    durations = np.zeros(len(onsets))          # Point events (0s duration)
    descriptions = ['Stimulus_Playback'] * len(onsets)

    annotations = mne.Annotations(onsets, durations, descriptions,
                                orig_time=raw.info['meas_date'])
    # Set annotations on the raw object
    raw.set_annotations(annotations)

except ValueError as e:
    print(f"Error processing 'Ch6' (may be missing or already processed): {e}")
    # Continue even if Ch6 is missing

# Drop unnecessary channels (Ch1, Ch6)
channels_to_drop = []
if 'Ch1' in raw.ch_names:
    channels_to_drop.append('Ch1')
if 'Ch6' in raw.ch_names:
    channels_to_drop.append('Ch6')

if channels_to_drop:
    raw.drop_channels(channels_to_drop)
    print(f"Channels dropped: {channels_to_drop}")


# Rename channels (e.g., Ch2 -> F3)
rename_map = {
    'Ch2': 'F3',
    'Ch3': 'F4',
    'Ch4': 'O1',
    'Ch5': 'O2'
}
# rename_channels only works if the key is present in current ch_names
raw.rename_channels(rename_map)
print(f"Channels renamed. Final channels: {raw.ch_names}")

# 60Hz notch + 0.5~35Hz band pass
print("\nApplying filters...")
raw.notch_filter(freqs=[60])
raw.filter(l_freq=1, h_freq=45)
print("Filtering complete (1–45 Hz, notch 60 Hz)")

# Average Reference
print("\nApplying average reference...")
raw.set_eeg_reference(ref_channels='average')

# Artifact
print("\nAnnotating muscle and blink artifacts...")
annot_musc, scores = annotate_muscle_zscore(raw, ch_type='eeg', threshold=4.0)
raw.set_annotations(annot_musc)
print(f"→ {len(annot_musc)} potential artifact segments annotated.")




# --- Save Results ---

# Save processed .fif file (Location: target_folder)
fif_save_path = os.path.join(script_dir, f"pre-processed_{folder_name}.fif")
raw.save(fif_save_path, overwrite=True)
print(f"Final .fif file saved: {fif_save_path}")

# Save PSD plot 
print("Generating and saving PSD plot...")
psd_fig = raw.plot_psd(fmin=1, fmax=45, average=True, show=False)
psd_title = f"PSD (Processed) for Testcase:{folder_name} (F3, F4, O1, O2)"
psd_fig.suptitle(psd_title, fontsize=16)
psd_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
psd_save_path = os.path.join(script_dir, f"pre-processed_{folder_name}_psd_plot.png")
psd_fig.savefig(psd_save_path)
print(f"PSD plot saved: {psd_save_path}")

# Save colored signal plot (4 channels)
print("Generating and saving colored signal plot...")
data, times = raw.get_data(return_times=True)
ch_names = raw.info['ch_names']
n_channels = len(ch_names)
colors = ['C0', 'C1', 'C2', 'C3'] # Colors for 4 channels

fig, axes = plt.subplots(n_channels, 1, figsize=(15, 8), sharex=True)
if n_channels == 1:
    axes = [axes]

for i in range(n_channels):
    axes[i].plot(times, data[i], color=colors[i], linewidth=0.7)
    axes[i].set_ylabel(ch_names[i], rotation=0,
                       horizontalalignment='right',
                       verticalalignment='center',
                       fontweight='bold', fontsize=10)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['left'].set_visible(False)
    axes[i].set_yticks([])

axes[-1].set_xlabel('Time (s)')
fig.suptitle(f'EEG Signals (Processed) for Testcase:{folder_name} (F3, F4, O1, O2)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

eeg_save_path = os.path.join(script_dir, f"pre-processed_{folder_name}_signals_plot.png")
fig.savefig(eeg_save_path)
print(f"EEG signal plot saved: {eeg_save_path}")

# Display plots 
print("\n--- All processing complete. Displaying plots. ---")
print("Close all plot windows to exit the script.")
plt.show()