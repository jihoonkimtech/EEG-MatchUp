import mne
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# --- 0. SETTINGS (Modify here) ---
# ------------------------------------------
# process this subject
SUBJECT = 'P01' 

# channels to pick for analysis
CHANNELS_TO_PICK = ['F3', 'F4', 'O1', 'O2']

# source data folders
oDATA_FOLDER = './eeg/'
oMETA_FILE = './meta/Stimuli_Meta.v1.csv' # script and eeg folder's parent

# output folder
oOUTPUT_FOLDER = './preprocessed/'
# ------------------------------------------

# --- 1. Set paths (relative to script) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(script_dir, oDATA_FOLDER)
META_FILE = os.path.join(script_dir, oMETA_FILE)
OUTPUT_FOLDER = os.path.join(script_dir, oOUTPUT_FOLDER)
# ------------------------------------------


# --- 2. Create output folder ---
# create the output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Output folder created at: {OUTPUT_FOLDER}")


# --- 3. Load Stimuli Metadata ---
try:
    # load song names from the meta file
    meta_df = pd.read_csv(META_FILE, usecols=['id', 'song'])
    # create a mapping dictionary, e.g., {11: 'Chim Chim Cheree...'}
    song_name_map = meta_df.set_index('id')['song'].to_dict()
    print(f"Successfully loaded song names from {META_FILE}")
except FileNotFoundError:
    print(f"Error: Could not find '{META_FILE}'.")
    print("Ensure meta file is in the same folder as this script.")
    song_name_map = {} # proceed with an empty map
except Exception as e:
    print(f"Error reading meta file: {e}")
    song_name_map = {}


# --- 4. Define file paths ---
raw_fname = os.path.join(DATA_FOLDER, f'{SUBJECT}-raw.fif')
ica_fname = os.path.join(DATA_FOLDER, f'{SUBJECT}-100p_64c-ica.fif')

# all outputs will go to the 'preprocessed' folder
output_raw_plot_before = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-plot-01-raw_before_ica.png')
output_ica_plot = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-plot-02-ica_components.png')
output_raw_plot_after = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-plot-03-raw_after_ica.png')
# [NEW PLOT] add path for the custom colored plot
output_custom_plot = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-plot-04-custom_signal.png')
# [RENUMBERED]
output_epochs_plot = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-plot-05-epochs.png')
output_erp_plot = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-plot-06-erp_average.png')
output_heatmap_plot = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-plot-07-epochs_heatmap.png')
output_epochs_fname = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-preprocessed-epo.fif')


# --- 5. Load data and ICA ---
print(f"Loading raw data from {raw_fname}...")
raw = mne.io.read_raw_fif(raw_fname, preload=True)

print(f"Loading ICA solution from {ica_fname}...")
ica = mne.preprocessing.read_ica(ica_fname) 


# --- 6. [PLOT 1] Plot raw data before ICA ---
print("Plotting raw data before ICA...")
try:
    fig = raw.plot(n_channels=64, duration=10, show=False)
    fig.savefig(output_raw_plot_before, dpi=300)
    plt.close(fig) # close figure to save memory
except Exception as e:
    print(f"Warning: Could not plot raw data. Error: {e}")


# --- 7. [PLOT 2] Plot ICA components ---
print(f"Plotting ICA components... (Excluding: {ica.exclude})")
try:
    # plot the component properties
    fig = ica.plot_components(show=False)
    fig[0].savefig(output_ica_plot, dpi=300) # plot_components returns a list of figures
    plt.close(fig[0])
except Exception as e:
    print(f"Warning: Could not plot ICA components. Error: {e}")


# --- 8. Apply ICA (remove artifacts) ---
print(f"Applying ICA... Excluding components: {ica.exclude}")
ica.apply(raw)


# --- 9. Resample and Filter ---
# (A) resample to 600Hz
print(f"Resampling data from {raw.info['sfreq']}Hz to 600Hz...")
raw.resample(600)

# (B) apply 60Hz notch filter
print("Applying 60Hz Notch filter...")
raw.notch_filter(freqs=60)

# (C) apply 1-45Hz bandpass filter
print("Applying 1-45Hz Bandpass filter...")
raw.filter(l_freq=1.0, h_freq=45.0)


# --- 10. [PLOT 3] Plot cleaned data (MNE default) ---
print("Plotting cleaned, filtered, and resampled data...")
try:
    fig = raw.plot(n_channels=64, duration=10, show=False)
    fig.savefig(output_raw_plot_after, dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Warning: Could not plot cleaned raw data. Error: {e}")


# --- 11. Define Events (Perception -> Resting, Imagination -> 3 Emotions) ---

# create the new mapping based on user request
# ('Excited', 'Majestic', 'Calm')
emotion_mapping = {
    # 'Excited'
    2: 'Excited', 12: 'Excited',      # Take me out to the ballgame
    3: 'Excited', 13: 'Excited',      # Jingle Bells
    
    # 'Majestic'
    21: 'Majestic',               # Emperor Waltz
    23: 'Majestic',               # Star Wars Theme
    
    # 'Calm'
    22: 'Calm',               # Harry Potter Theme
    24: 'Calm',               # Eine Kleine Nachtmusik
    
    # IDs 1, 4, 11, 14 (Mary, Chim Chim) will be skipped for Imagination
}

print("Mapping 'Perception' (xx1) to 'Resting' (Baseline)...")
print("Mapping 'Imagination' (xx4) to 'Excited', 'Majestic', 'Calm'...")

event_id_map = {}

# loop 1 to 29 (covers 1-24 from meta file)
for stimulus_id in range(1, 30): 
    
    # (A) 'Perception' (condition=1) -> 'Resting'
    # ----------------------------------------------------
    event_id_p = (stimulus_id * 10) + 1 # 1 = perception
    
    song_name = song_name_map.get(stimulus_id) 
    
    if song_name is None:
        # skip if song ID is not in meta file (e.g., 5-10, 15-20)
        continue 
    
    # map all perception events to 'Resting'
    label_p = f'Resting/{song_name}' 
    event_id_map[label_p] = event_id_p

    # (B) 'Imagination' (condition=4) -> 3 Groups
    # ----------------------------------------------------
    event_id_i = (stimulus_id * 10) + 4 # 4 = imagination
    
    # get the group ('Excited', 'Majestic', 'Calm')
    emotion_group = emotion_mapping.get(stimulus_id)
    
    if emotion_group: # if a group was defined
        # map to the specific group
        label_i = f'{emotion_group}/{song_name}' 
        event_id_map[label_i] = event_id_i
    else:
        # discard if not in the map (e.g., ID 1, 4, 11, 14)
        pass
# -----------------------------------------------------------------


# --- 12. Channel Selection and Validation ---

# (A) use the channel list defined in settings
channels_to_pick = list(CHANNELS_TO_PICK)

# (B) validate that the channels exist in the raw file
missing_channels = [ch for ch in channels_to_pick if ch not in raw.ch_names]
if missing_channels:
    print(f"Warning: Requested channels not in raw file: {missing_channels}")
    print(f"    (Found channels example: {raw.ch_names[:5]}...)")
    # proceed with only the channels that were found
    channels_to_pick = [ch for ch in channels_to_pick if ch in raw.ch_names]
    print(f"    -> Proceeding with channels: {channels_to_pick}")
else:
    print(f"Found all requested channels: {channels_to_pick}")


# --- 13. [PLOT 4] Custom Colored Signal Plot (User Request) ---
# (This plots the first 20 seconds of the *continuous* cleaned data)
print("Generating and saving colored signal plot...")
if channels_to_pick:
    try:
        # get only 20 seconds of data for the selected channels
        data, times = raw.get_data(picks=channels_to_pick, duration=20, return_times=True)
        ch_names = channels_to_pick
        n_channels = len(ch_names)
        
        # define colors for channels
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        colors = colors[:n_channels] # select only needed colors

        # dynamic height: 2 inches per channel
        fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2 * n_channels), sharex=True)
        if n_channels == 1:
            axes = [axes] # ensure axes is iterable

        # plot each channel on its own subplot
        for i in range(n_channels):
            axes[i].plot(times, data[i], color=colors[i], linewidth=0.7)
            axes[i].set_ylabel(ch_names[i], rotation=0,
                               horizontalalignment='right',
                               verticalalignment='center',
                               fontweight='bold', fontsize=10)
            # styling
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['left'].set_visible(False)
            axes[i].set_yticks([])

        axes[-1].set_xlabel('Time (s) (First 20s)') # 
        fig.suptitle(f'EEG Signals (Processed) for {SUBJECT} ({", ".join(ch_names)})', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        fig.savefig(output_custom_plot)
        plt.close(fig) # free memory
        print(f"EEG signal plot saved: {output_custom_plot}")
    except Exception as e:
        print(f"Warning: Could not generate custom signal plot. Error: {e}")
else:
    print("Skipping custom signal plot: no channels selected or found.")


# --- 14. Epoching ---
try:
    events = mne.find_events(raw, shortest_event=1)
    event_dict = event_id_map
    print(f"Found {len(events)} events total.")
except Exception as e:
    print(f"Error finding events: {e}")
    events = []
    event_dict = {}

if len(events) > 0 and channels_to_pick:
    # define epoch timings
    tmin = -0.5
    tmax = 8.0   
    baseline = (tmin, 0) # use pre-stimulus period as baseline

    print(f"Creating epochs from {tmin}s to {tmax}s... (sfreq={raw.info['sfreq']}Hz)")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        on_missing='warn', # ignore events not in our map (e.g., condition 2, 3)
        baseline=baseline,
        preload=True,
        reject=None,
        picks=channels_to_pick 
    )
    
    # --- 15. Save final file (PRIORITY) ---
    # (Save the file *before* attempting memory-intensive plots)
    print("\n--- Generated Epochs (Filtered, Resampled, Specific Channels) ---")
    print(epochs) 
    print(f"Epochs channel names: {epochs.ch_names}")
    
    # this file contains all groups: Resting, Excited, Majestic, Calm
    epochs.save(output_epochs_fname, overwrite=True)
    print(f"\nSuccessfully saved ALL epochs to: {output_epochs_fname}")

    # --- 15. [PLOT 5, 6, 7] Plot Epochs, ERP, and Heatmap ---
    print("\n--- Generated Epochs (Filtered, Resampled, Specific Channels) ---")
    print(epochs) 
    print(f"Epochs channel names: {epochs.ch_names}")

    print("Plotting generated epochs, average ERP, and heatmap...")
    try:
        # [PLOT 5] plot a snippet of the epochs (MNE default style)
        fig_epochs = epochs.plot(n_epochs=5, n_channels=4, show=False)
        fig_epochs.savefig(output_epochs_plot, dpi=300)
        plt.close(fig_epochs)
        
        
        # [PLOT 6] plot the average of all epochs (ERP) - CUSTOM STYLE
        print("Generating custom ERP average plot (stacked channels)...")
        try:
            # get the average data (ERP)
            evoked = epochs.average()
            data = evoked.get_data() 
            times = evoked.times
            ch_names = evoked.ch_names
            n_channels = len(ch_names)
            
            # define colors (as per your template)
            colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
            colors = colors[:n_channels] # select only needed colors

            # dynamic height: 2 inches per channel
            fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2 * n_channels), sharex=True)
            if n_channels == 1:
                axes = [axes] # ensure axes is iterable

            # plot each channel on its own subplot
            for i in range(n_channels):
                axes[i].plot(times, data[i], color=colors[i], linewidth=0.7)
                axes[i].set_ylabel(ch_names[i], rotation=0,
                                   horizontalalignment='right',
                                   verticalalignment='center',
                                   fontweight='bold', fontsize=10)
                # styling (as per your template)
                axes[i].spines['top'].set_visible(False)
                axes[i].spines['right'].set_visible(False)
                axes[i].spines['left'].set_visible(False)
                axes[i].set_yticks([])
                
                # add a vertical line at 0s (stimulus onset) for ERP
                axes[i].axvline(0, color='black', linestyle='--', linewidth=0.5)
                # add a horizontal line at 0mv
                axes[i].axhline(0, color='black', linestyle='-', linewidth=0.3)


            axes[-1].set_xlabel('Time (s)')
            fig.suptitle(f'ERP Average ({SUBJECT}) - ({", ".join(ch_names)})', fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # save to the original output_erp_plot path
            fig.savefig(output_erp_plot) 
            plt.close(fig) # free memory
            print(f"Custom ERP plot saved: {output_erp_plot}")
        
        except Exception as e_erp:
            print(f"Warning: Could not plot custom ERP. Error: {e_erp}")

        # [PLOT 7] plot an image heatmap of all trials (MNE default style)
        print("Generating epochs heatmap (plot_image)...")
        fig_heatmap = epochs.plot_image(
            picks=channels_to_pick, 
            combine='mean',      # average the 4 channels
            show=False,
            cmap='viridis'       
        )
        fig_heatmap[0].savefig(output_heatmap_plot, dpi=300)
        plt.close(fig_heatmap[0])

    except Exception as e:
        print(f"Warning: Could not plot epochs. Error: {e}")