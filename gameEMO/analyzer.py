import os
import glob
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# EEG Feature Extractor (FAA, OAA, OASI, FAP, TBR) & PSD Plots
# ----------------------------------------------------------

# User Configuration
TESTCASE = "G1-TrainSim"

# --- [NEW] Define bands ---
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)
ALL_BANDS = {'Theta': THETA_BAND, 'Alpha': ALPHA_BAND, 'Beta': BETA_BAND}
KEY_CHANNELS = ['F3', 'F4', 'O1', 'O2']
# ---------------------------

# input folder path containing .mat files (relative to this script)
INPUT_FOLDER_NAME = f"./pre-processed/{TESTCASE}"
OUTPUT_FOLDER_NAME = f"./analyze/{TESTCASE}"
CSV_FOLDER_NAME = f"./analyze/Summary/"


# script_dir = os.path.dirname(os.path.abspath(__file__)) # (이미 상단에 있음)

target_folder = os.path.join(script_dir, INPUT_FOLDER_NAME)
output_folder = os.path.join(script_dir, OUTPUT_FOLDER_NAME)
csv_folder = os.path.join(script_dir, CSV_FOLDER_NAME)

# --- [NEW] Create output subfolders ---
topo_output_folder = os.path.join(output_folder, 'topomaps')
psd_output_folder = os.path.join(output_folder, 'psd_curves')
os.makedirs(output_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)
os.makedirs(topo_output_folder, exist_ok=True)
os.makedirs(psd_output_folder, exist_ok=True)
# ------------------------------------

REACTION_DELAY_SECONDS = 0.5
# ----------------------------------------------------------

print(f"--- Starting Batch Feature Extraction ---")
print(f"Scanning for epoch files in: {target_folder}\n")
print(f"(Saving Topomaps to: {topo_output_folder})")
print(f"(Saving PSD Curves to: {psd_output_folder})\n")


epoch_files = glob.glob(os.path.join(target_folder, "*-epo.fif"))

if not epoch_files:
    print(f"*** ERROR: No '*-epo.fif' files found in {target_folder} ***")
    exit()

all_metrics = []

# --- [MODIFIED] Define plot metrics and colors for consistency ---
metric_keys = ['FAA', 'FAP', 'OAA', 'OASI', 'TBR'] # Add TBR
plot_colors = ['tomato', 'royalblue', 'mediumseagreen', 'orange', 'purple'] # Add color for TBR


for file_path in epoch_files:
    filename = os.path.basename(file_path)
    testcase_name = filename.replace("pre-processed_", "").replace("-epo.fif", "")
    
    print(f"--- Processing: '{testcase_name}' ---")

    try:
        # Load preprocessed EPOCHED data
        epochs = mne.read_epochs(file_path, preload=True, verbose=False)
        print(f"Loaded {len(epochs)} epochs from: {os.path.basename(file_path)}")
        
        # start from 0.5s
        epochs.crop(tmin=REACTION_DELAY_SECONDS, verbose=False)
        print(f"Cropped epochs: New range {epochs.tmin}s to {epochs.tmax}s")

        # --- 2. Compute PSD ---
        n_fft = 128
        
        spectrum = epochs.compute_psd(method='welch', fmin=1, fmax=45, n_fft=n_fft, verbose=False)

        # --- [NEW PLOT 1: Topomap] ---
        try:
            topo_fig = spectrum.plot_topomap(bands=ALL_BANDS, ch_type='eeg', show=False, vlim=(None, None))
            topo_fig.suptitle(f'Topomap - {testcase_name}', fontsize=14)
            topo_img_path = os.path.join(topo_output_folder, f"EEG_topo_{testcase_name}.png")
            topo_fig.savefig(topo_img_path)
            plt.close(topo_fig)
        except Exception as e:
            print(f"    Warning: Could not save topomap for {testcase_name}: {e}")
        # -------------------------------

        psd = np.mean(spectrum.get_data(), axis=0) 
        freqs = spectrum.freqs

        # --- [NEW PLOT 2: PSD Curve] ---
        try:
            plt.figure(figsize=(10, 6))
            channels_to_plot = [ch for ch in KEY_CHANNELS if ch in epochs.ch_names]
            for ch in channels_to_plot:
                ch_idx = epochs.ch_names.index(ch)
                plt.plot(freqs, 10 * np.log10(psd[ch_idx, :]), label=ch)
            
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density (dB/Hz)')
            plt.title(f'PSD Curves - {testcase_name}')
            plt.legend()
            plt.grid(alpha=0.5, linestyle='--')
            plt.tight_layout()
            
            psd_curve_img_path = os.path.join(psd_output_folder, f"EEG_psd_curve_{testcase_name}.png")
            plt.savefig(psd_curve_img_path)
            plt.close()
        except Exception as e:
            print(f"    Warning: Could not save PSD curve for {testcase_name}: {e}")
        # -------------------------------


        # --- 3. Calculate EEG indices ---
        
        # Find indices
        theta_idx = np.where((freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1]))[0]
        alpha_idx = np.where((freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1]))[0]
        beta_idx = np.where((freqs >= BETA_BAND[0]) & (freqs <= BETA_BAND[1]))[0]

        # Get mean power per band
        theta_power = np.mean(psd[:, theta_idx], axis=1)
        alpha_power = np.mean(psd[:, alpha_idx], axis=1)
        beta_power = np.mean(psd[:, beta_idx], axis=1)

        # Create dicts
        theta_dict = dict(zip(epochs.ch_names, theta_power))
        alpha_dict = dict(zip(epochs.ch_names, alpha_power))
        beta_dict = dict(zip(epochs.ch_names, beta_power))


        def safe_log(val):
            if pd.isna(val) or val <= 0: # check non-positive
                return np.nan
            return np.log(val) # remove 1e-10

        def get_power(ch, power_dict): # Renamed
            return power_dict[ch] if ch in power_dict else np.nan

        # Get alpha powers
        F3_alpha, F4_alpha = [get_power(ch, alpha_dict) for ch in ['F3', 'F4']]
        O1_alpha, O2_alpha = [get_power(ch, alpha_dict) for ch in ['O1', 'O2']]

        # Get theta/beta powers
        F3_theta, F4_theta = [get_power(ch, theta_dict) for ch in ['F3', 'F4']]
        F3_beta, F4_beta = [get_power(ch, beta_dict) for ch in ['F3', 'F4']]

        # Calculate original metrics
        FAA = safe_log(F4_alpha) - safe_log(F3_alpha)
        OAA = safe_log(O2_alpha) - safe_log(O1_alpha)
        OASI = (O1_alpha + O2_alpha) / 2
        FAP = (F3_alpha + F4_alpha) / 2
        
        # --- [NEW] Calculate TBR ---
        frontal_theta_avg = (F3_theta + F4_theta) / 2
        frontal_beta_avg = (F3_beta + F4_beta) / 2
        TBR = frontal_theta_avg / (frontal_beta_avg + 1e-10) # add epsilon
        # ---------------------------

        metrics = {
            'testcase': testcase_name,
            'FAA': FAA,
            'FAP': FAP,
            'OAA': OAA,
            'OASI': OASI,
            'TBR': TBR # Add TBR
        }
        
        all_metrics.append(metrics)
        
        # --- [MODIFIED] Plot and save summary for THIS file ---
        plt.figure(figsize=(7, 4))
        
        current_values = [metrics[key] for key in metric_keys] 
        
        bars = plt.bar(metric_keys, current_values, color=plot_colors)
        plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=9) # Add labels
        
        plt.title(f"EEG Feature Summary ({testcase_name})")
        plt.ylabel("Value")
        # plt.ylim(-10.0, 10.0) # Removed for auto-scaling
        
        # adjust y-axis limits to give space for labels
        ymin, ymax = plt.ylim()
        padding = max(abs(ymax) * 0.1, abs(ymin) * 0.1, 0.5) 
        plt.ylim(ymin - padding, ymax + padding) 
        
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()

        img_path = os.path.join(output_folder, f"EEG_features_{testcase_name}.png")
        plt.savefig(img_path)
        plt.close()
        # ----------------------------------------------------
        
        print(f"*** Successfully processed {testcase_name} ***")
        print(f"*** Individual plots saved to {output_folder} ***\n")


    except Exception as e:
        print(f"*** ERROR processing {testcase_name}: {e} ***\n")

# --- 4. Save All Results & Average Plot ---
if all_metrics:
    df = pd.DataFrame(all_metrics)
    
    # Save the CSV with individual results (as before)
    csv_path = os.path.join(csv_folder, f"EEG_features_{TESTCASE}_ALL.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"--- All processing complete ---")
    print(f"Successfully processed {len(all_metrics)} files.")
    print(f"Individual results saved to: {csv_path}")
    
    # --- [MODIFIED] Plot and save AVERAGE summary plot ---
    
    # Calculate averages
    average_metrics = df[metric_keys].mean()

    plt.figure(figsize=(7, 4.5)) 
    
    bars = plt.bar(average_metrics.index, average_metrics.values, color=plot_colors)
    
    plt.title(f"Average EEG ({TESTCASE}) - {len(all_metrics)} files")
    plt.ylabel("Average Value")
    
    plt.grid(alpha=0.3, linestyle='--')
    
    # Add data labels on top of bars (using bar_label for simplicity)
    plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)

    # adjust y-axis limits to give space for labels
    ymin, ymax = plt.ylim()
    padding = max(abs(ymax) * 0.1, abs(ymin) * 0.1, 0.5) 
    plt.ylim(ymin - padding, ymax + padding) 
    
    plt.tight_layout()

    img_path_avg = os.path.join(csv_folder, f"EEG_features_{TESTCASE}_AVERAGE_plot.png")
    plt.savefig(img_path_avg)
    plt.close()
    
    print(f"Average summary plot saved: {img_path_avg}")
    
    # --- [MODIFIED] Create and save combined CSV (individual + average) ---
    
    avg_row = {'testcase': f'AVERAGE ({len(all_metrics)} files)'}
    avg_row.update(average_metrics.to_dict())
    
    avg_df = pd.DataFrame([avg_row])
    
    df_combined = pd.concat([df, avg_df], ignore_index=True)
    
    csv_path_combined = os.path.join(csv_folder, f"EEG_features_{TESTCASE}_ALL_with_AVG.csv")
    df_combined.to_csv(csv_path_combined, index=False)
    # -------------------------------------------------------------------
    
    print(f"Combined (Individual + AVG) results saved to: {csv_path_combined}")

else:
    print("--- No files processed. Exiting. ---")