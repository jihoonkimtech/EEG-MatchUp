import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import annotate_muscle_zscore

# Specify the path to the base folder.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Example: ../raw/base/ or ../raw/info/
target_folder = os.path.join(script_dir, "../raw/120s/")

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
    # this finds all rising edges (e.g., to 4 or 5)
    events = mne.find_events(raw, stim_channel='Ch6', output='onset', min_duration=0.01)
    print(f"Found {len(events)} 'Stimulus_Playback' events from Ch6.")

    # 6. Convert to MNE Annotations object
    onsets = events[:, 0] / raw.info['sfreq']  # Convert samples to seconds
    durations = np.zeros(len(onsets))          # Point events (0s duration)
    # this maps all found event IDs (4, 5, etc.) to one description
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
print("Filtering complete (1–45DHz, notch 60DHz)")

# Average Reference
print("\nApplying average reference...")
raw.set_eeg_reference(ref_channels='average')

# Artifact
print("\nAnnotating muscle and blink artifacts...")
annot_musc, scores = annotate_muscle_zscore(raw, ch_type='eeg', threshold=4.0)
raw.set_annotations(annot_musc)
print(f"→ {len(annot_musc)} potential artifact segments annotated.")


# --- Save Results (Original Raw) ---

# Save processed .fif file (Location: target_folder)
fif_save_path = os.path.join(script_dir, f"pre-processed_{folder_name}_epo.fif")
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


# --- (MODIFIED) Epoching Data ---
print("\n--- Starting Epoching ---")

# define epoch parameters
tmin, tmax = -0.2, 0.8  # 200ms before stimulus, 800ms after

# We will use the 'Stimulus_Playback' annotations we created earlier.
# This approach correctly handles all trigger IDs (e.g., 4 or 5)
# by mapping them all to a single new event ID (e.g., 1).

# define a new mapping for our event
event_id_desc = 'Stimulus_Playback'
event_id_code = 1 # we will map the description to the integer 1
event_mapping = {event_id_desc: event_id_code}

try:
    # extract events from the annotations
    # this finds all 'Stimulus_Playback' annotations we made
    # (which came from Ch6 triggers 4, 5, etc.)
    # and gives them the new ID '1'
    events_from_annot, event_dict = mne.events_from_annotations(
        raw, event_id=event_mapping)
    
    print(f"Extracted {len(events_from_annot)} events from annotations.")
    print(f"Mapped '{event_id_desc}' to ID {event_id_code}.")

    # create epochs
    # preload=True loads data into memory
    # 'reject_by_annotation=True' (default) automatically drops epochs
    # that overlap with 'BAD_muscle' annotations
    epochs = mne.Epochs(raw,
                        events=events_from_annot, # use the new events array
                        event_id=event_dict,      # use the new event_dict
                        tmin=tmin,
                        tmax=tmax,
                        baseline=(None, 0),  # baseline from tmin to 0
                        preload=True)

    print(f"Created {len(epochs)} epochs.")
    # epochs.drop_log shows which epochs were dropped and why
    print(f"Dropped {len(epochs.drop_log)} epochs due to muscle/artifact annotations.")

    # create evoked (average)
    evoked = epochs.average()
    print("Averaged epochs to create Evoked response.")

    # --- (NEW) Save Epochs and Evoked Results ---

    # Save processed Epochs .fif file
    # suffix '-epo' indicates epochs file
    epochs_save_path = os.path.join(script_dir, f"pre-processed_{folder_name}-epo.fif")
    epochs.save(epochs_save_path, overwrite=True)
    print(f"Epochs .fif file saved: {epochs_save_path}")

    # Save Evoked plot (average of epochs)
    print("Generating and saving Evoked plot...")
    # .plot() returns a Figure object
    evoked_fig = evoked.plot(show=False, spatial_colors=True, gfp=True)
    evoked_title = f'Evoked Response for Testcase:{folder_name} (F3, F4, O1, O2)'
    evoked_fig.suptitle(evoked_title, fontsize=16)
    evoked_fig.tight_layout(rect=[0, 0.03, 1, 0.9]) # adjust rect for title
    eeg_save_path = os.path.join(script_dir, f"pre-processed_{folder_name}_evoked_plot.png")
    evoked_fig.savefig(eeg_save_path)
    print(f"Evoked plot saved: {eeg_save_path}")

    # Save an epochs image plot (heatmap)
    print("Generating and saving Epochs image plot...")
    # 'combine='gfp'' shows the global field power
    # plot_image returns a list of figs, one per event_id
    epochs_img_fig = epochs.plot_image(combine='gfp', show=False)
    img_title = f'Epochs Image (GFP) for Testcase:{folder_name}'
    epochs_img_fig[0].suptitle(img_title, fontsize=16)
    epochs_img_fig[0].tight_layout(rect=[0, 0.03, 1, 0.95])
    img_save_path = os.path.join(script_dir, f"pre-processed_{folder_name}_epochs_image_plot.png")
    epochs_img_fig[0].savefig(img_save_path)
    print(f"Epochs image plot saved: {img_save_path}")

except ValueError:
    # this happens if no 'Stimulus_Playback' annotations were found
    print("No 'Stimulus_Playback' annotations found. Skipping epoching.")


# Display plots
print("\n--- All processing complete. Displaying plots. ---")
print("Close all plot windows to exit the script.")
plt.show()