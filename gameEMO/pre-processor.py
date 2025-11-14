import os
import mne
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt  # <--- [추가] 플로팅을 위해 임포트

# User Configuration
TESTCASE = "G4-Control"

# input folder path containing .mat files (relative to this script)
INPUT_FOLDER_NAME = f"./eeg/{TESTCASE}"

# output folder name to save Epochs(.fif) files (relative to this script)
# analyzer.py looks for ../pre-processed/, so we set this to "pre-processed".
OUTPUT_FOLDER_NAME = f"./pre-processed/{TESTCASE}"

# (e.g., 250, 256, 500, 1000...)
# This will be applied to ALL .mat files.
SFREQ = 600  # <--- Enter the correct sampling frequency here.

# scaling factor to convert data to Volts (MNE's default unit)
# if data is in Microvolts (µV), use 1e-6
# if data is already in Volts (V), use 1.0
# if data is in Millivolts (mV), use 1e-3
DATA_SCALING_FACTOR = 1e-6  # (µV -> V)

# duration (in seconds) for splitting data into epochs
EPOCH_DURATION_SEC = 20.0

# list of channel names to extract from the .mat files
# this list will be applied to all files in the folder
CHANNELS_TO_EXTRACT = [
    'F3', 'F4', 'O1', 'O2'
]
# ----------------------------------------------------


def process_single_mat_file(mat_path, testcase_name, output_folder):
    """
    Loads, pre-processes, epochs a single .mat file, and saves it as .fif.
    """
    print(f"\n--- 1. Processing: {testcase_name} ---")
    
    try:
        # 1-1. Load .mat file
        mat_data = loadmat(mat_path)

        data_list = []
        ch_names_found = []
        
        # 1-2. Extract specified channels
        for ch_name in CHANNELS_TO_EXTRACT:
            if ch_name in mat_data:
                # data is (n_samples, 1), squeeze to (n_samples,)
                ch_data = mat_data[ch_name].squeeze()
                data_list.append(ch_data)
                ch_names_found.append(ch_name)
            else:
                print(f"Warning: Channel '{ch_name}' not found in {testcase_name}.")

        if not data_list:
            print("*** ERROR: No valid EEG channels found. Skipping file. ***")
            return False

        # 1-3. Stack and scale data
        # stack channels into (n_channels, n_samples) array
        eeg_data = np.stack(data_list, axis=0) * DATA_SCALING_FACTOR
        n_channels, n_samples = eeg_data.shape
        print(f"Found {n_channels} channels, {n_samples} samples.")

        # --- 2. Create MNE Raw Object ---
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names=ch_names_found, sfreq=SFREQ, ch_types=ch_types)
        raw = mne.io.RawArray(eeg_data, info, verbose=False)
        
        # set standard 10-20 montage (optional but good for plotting)
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, on_missing='warn')
        except Exception as e:
            print(f"Warning: Failed to set montage: {e}")

        # --- 3. Apply Pre-processing (similar to pre-processor.py) ---
        # 60Hz notch + 1~45Hz band pass
        raw.notch_filter(freqs=[60], verbose=False)
        raw.filter(l_freq=1, h_freq=45, verbose=False)
        
        # average Reference
        raw.set_eeg_reference(ref_channels='average', verbose=False)
        print("Filtering (1-45Hz, 60Hz notch) and Average reference applied.")

        # --- [추가] 4. Save Plots (from raw object) ---

        # --- 4A. Save PSD plot ---
        print("Generating and saving PSD plot...")
        # use fmax=45 to match the filter
        psd_fig = raw.plot_psd(fmin=1, fmax=45, average=True, show=False)
        psd_title = f"PSD (Processed) for Testcase:{testcase_name} ({', '.join(ch_names_found)})"
        psd_fig.suptitle(psd_title, fontsize=16)
        psd_fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        psd_save_path = os.path.join(output_folder, f"pre-processed_{testcase_name}_psd_plot.png")
        psd_fig.savefig(psd_save_path)
        plt.close(psd_fig) # free memory
        print(f"PSD plot saved: {psd_save_path}")

        # --- 4B. Save colored signal plot ---
        print("Generating and saving colored signal plot...")
        data, times = raw.get_data(return_times=True)
        ch_names = raw.info['ch_names']
        n_channels = len(ch_names)
        
        # define colors for channels
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        colors = colors[:n_channels] # select only needed colors

        # dynamic height: 2 inches per channel
        fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2 * n_channels), sharex=True)
        if n_channels == 1:
            axes = [axes] # ensure axes is iterable

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
        fig.suptitle(f'EEG Signals (Processed) for Testcase:{testcase_name} ({", ".join(ch_names)})', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        eeg_save_path = os.path.join(output_folder, f"pre-processed_{testcase_name}_signals_plot.png")
        fig.savefig(eeg_save_path)
        plt.close(fig) # free memory
        print(f"EEG signal plot saved: {eeg_save_path}")

        # --- 5. Create Fixed-Length Epochs ---
        # (This section was formerly step 4)
        epochs = mne.make_fixed_length_epochs(
            raw, 
            duration=EPOCH_DURATION_SEC, 
            preload=True, 
            verbose=False
        )
        print(f"Created {len(epochs)} epochs of {EPOCH_DURATION_SEC}s duration.")

        # --- 6. Save Processed Epochs ---
        # (This section was formerly step 5)
        # use the '-epo.fif' suffix so analyzer.py can find it
        output_filename = f"pre-processed_{testcase_name}-epo.fif"
        output_filepath = os.path.join(output_folder, output_filename)
        
        epochs.save(output_filepath, overwrite=True, verbose=False)
        print(f"Saved: {output_filepath}")
        return True

    except Exception as e:
        print(f"*** ERROR processing {testcase_name}: {e} ***")
        return False


def batch_convert_mats():
    """
    Scans INPUT_FOLDER_NAME for all .mat files and processes them.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # construct full paths for input and output folders
    input_folder = os.path.join(script_dir, INPUT_FOLDER_NAME)
    output_folder = os.path.join(script_dir, OUTPUT_FOLDER_NAME)

    print(f"--- 1. Starting Batch Conversion ---")
    print(f"Scanning for .mat files in: {input_folder}")
    print(f"Saving .fif files to:      {output_folder}")

    if not os.path.isdir(input_folder):
        print(f"\n*** ERROR: Input folder not found! ***")
        print(f"Please check the path: {input_folder}")
        print("Verify the 'INPUT_FOLDER_NAME' variable is set correctly.")
        return

    # create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    found_files = 0
    processed_files = 0

    # iterate over all files in the input directory
    for filename in os.listdir(input_folder):
        if filename.endswith('.mat'):
            found_files += 1
            
            # get the filename without the .mat extension (e.g., "S20G1AllChannels")
            testcase_name = os.path.splitext(filename)[0]
            mat_file_path = os.path.join(input_folder, filename)
            
            # process the single file
            success = process_single_mat_file(mat_file_path, testcase_name, output_folder)
            
            if success:
                processed_files += 1

    print("\nComplete!")
    print(f"Found {found_files} .mat files.")
    print(f"Successfully processed and saved {processed_files} files.")


if __name__ == "__main__":
    batch_convert_mats()