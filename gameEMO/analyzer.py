import os
import glob
import sys
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# EEG Feature Extractor (FAA, OAA, OASI, FAP, TBR) & PSD Plots
# ----------------------------------------------------------

# User Configuration
TESTCASE = "G4-Control"
# --- [NEW] Check if TESTCASE is provided as command-line argument ---
if len(sys.argv) > 1:
    TESTCASE = sys.argv[1]
    print(f"[Info] Using TESTCASE from command line: {TESTCASE}")
# ---------------------------------------------------------------------

# --- [NEW] Define bands ---
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)
ALL_BANDS = {'Theta': THETA_BAND, 'Alpha': ALPHA_BAND, 'Beta': BETA_BAND}
KEY_CHANNELS = ['F3', 'F4', 'O1', 'O2']
PSD_FMIN = 1.0
PSD_FMAX = 45.0
# ---------------------------

# input folder path containing .mat files (relative to this script)
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
INPUT_FOLDER_NAME = os.path.join(script_dir, "pre-processed", TESTCASE)
OUTPUT_FOLDER_NAME = os.path.join(script_dir, "analyze", TESTCASE)
CSV_FOLDER_NAME = os.path.join(script_dir, "analyze", "Summary")


# --- [MODIFIED] Create output subfolders ---
topo_output_folder = os.path.join(OUTPUT_FOLDER_NAME, 'topomaps')
psd_output_folder = os.path.join(OUTPUT_FOLDER_NAME, 'psd_curves')
# --- [NEW] Create summary PSD folder ---
summary_psd_folder = os.path.join(CSV_FOLDER_NAME, 'summary_psd_curves')
# ------------------------------------

os.makedirs(OUTPUT_FOLDER_NAME, exist_ok=True)
os.makedirs(CSV_FOLDER_NAME, exist_ok=True)
os.makedirs(topo_output_folder, exist_ok=True)
os.makedirs(psd_output_folder, exist_ok=True)
# --- [NEW] ---
os.makedirs(summary_psd_folder, exist_ok=True)
# ------------------------------------


REACTION_DELAY_SECONDS = 0.5
# ----------------------------------------------------------

print(f"--- Starting Batch Feature Extraction ---")
print(f"Testcase: {TESTCASE}")
print(f"Scanning for epoch files in: {INPUT_FOLDER_NAME}")
print(f"(Saving Individual PSD Curves to: {psd_output_folder})")
print(f"(Saving Summary PSD Curve to: {summary_psd_folder})\n")


epoch_files = glob.glob(os.path.join(INPUT_FOLDER_NAME, "*-epo.fif"))

if not epoch_files:
    print(f"*** ERROR: No '*-epo.fif' files found in {INPUT_FOLDER_NAME} ***")
    sys.exit(1) # [MODIFIED] Use sys.exit

all_metrics = []
# --- [NEW] For unified PSD plot ---
all_psds_list = []
all_freqs = None
ch_names_last = None # To store channel names
# ----------------------------------

# --- [MODIFIED] Define plot metrics and colors for consistency ---
metric_keys = ['FAA', 'FAP', 'OASI', 'TBR'] # OAA Removed, OASI Restored
plot_colors = ['tomato', 'royalblue', 'orange', 'purple'] # mediumseagreen Removed, orange Restored


for file_path in epoch_files:
    filename = os.path.basename(file_path)
    testcase_name = filename.replace("pre-processed_", "").replace("-epo.fif", "")
    
    print(f"--- Processing: '{testcase_name}' ---")

    try:
        # Load preprocessed EPOCHED data
        epochs = mne.read_epochs(file_path, preload=True, verbose=False)
        print(f"Loaded {len(epochs)} epochs from: {os.path.basename(file_path)}")
        
        # Store channel names (for final unified plot)
        ch_names_last = epochs.ch_names
        
        # start from 0.5s
        epochs.crop(tmin=REACTION_DELAY_SECONDS, verbose=False)
        print(f"Cropped epochs: New range {epochs.tmin}s to {epochs.tmax}s")

        # --- 2. Compute PSD ---
        n_fft = 128
        
        spectrum = epochs.compute_psd(
            method='welch', 
            fmin=PSD_FMIN, 
            fmax=PSD_FMAX, 
            n_fft=n_fft, 
            verbose=False,
            n_jobs=-1
        )

        # --- [NEW PLOT 1: Topomap] (Still attempted, but warnings are OK) ---
        try:
            topo_fig = spectrum.plot_topomap(bands=ALL_BANDS, ch_type='eeg', show=False, vlim=(None, None))
            topo_fig.suptitle(f'Topomap - {testcase_name}', fontsize=14)
            topo_img_path = os.path.join(topo_output_folder, f"EEG_topo_{testcase_name}.png")
            topo_fig.savefig(topo_img_path)
            plt.close(topo_fig)
        except Exception as e:
            # (We expect errors here for 4 channels, so just print a minimal warning)
            if 'cocircular' in str(e):
                print(f"    Note: Topomap skipped (expected for low channel count).")
            else:
                print(f"    Warning: Could not save topomap for {testcase_name}: {e}")
        # -------------------------------

        psds = spectrum.get_data() # (n_epochs, n_channels, n_freqs)
        psd_avg_file = np.mean(psds, axis=0) # (n_channels, n_freqs)
        freqs = spectrum.freqs

        # --- [NEW] Store data for unified plot ---
        all_psds_list.append(psd_avg_file)
        if all_freqs is None:
            all_freqs = freqs
        # ---------------------------------------


        # --- [NEW PLOT 2: PSD Curve] (Individual) ---
        try:
            plt.figure(figsize=(10, 6))
            channels_to_plot = [ch for ch in KEY_CHANNELS if ch in epochs.ch_names]
            for ch in channels_to_plot:
                ch_idx = epochs.ch_names.index(ch)
                plt.plot(freqs, 10 * np.log10(psd_avg_file[ch_idx, :]), label=ch)
            
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
        theta_power = np.mean(psd_avg_file[:, theta_idx], axis=1)
        alpha_power = np.mean(psd_avg_file[:, alpha_idx], axis=1)
        beta_power = np.mean(psd_avg_file[:, beta_idx], axis=1)

        # Create dicts
        theta_dict = dict(zip(epochs.ch_names, theta_power))
        alpha_dict = dict(zip(epochs.ch_names, alpha_power))
        beta_dict = dict(zip(epochs.ch_names, beta_power))


        def safe_log(val):
            if pd.isna(val) or val <= 0: # check non-positive
                return np.nan
            return np.log(val)

        def get_power(ch, power_dict):
            return power_dict[ch] if ch in power_dict else np.nan

        # Get alpha powers
        F3_alpha, F4_alpha = [get_power(ch, alpha_dict) for ch in ['F3', 'F4']]
        O1_alpha, O2_alpha = [get_power(ch, alpha_dict) for ch in ['O1', 'O2']]

        # Get theta/beta powers
        F3_theta, F4_theta = [get_power(ch, theta_dict) for ch in ['F3', 'F4']]
        F3_beta, F4_beta = [get_power(ch, beta_dict) for ch in ['F3', 'F4']]

        # Calculate original metrics (All are calculated for CSV)
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
            'OAA': OAA,   # Still saved in CSV
            'OASI': OASI, # Saved in CSV & Plotted
            'TBR': TBR
        }
        
        all_metrics.append(metrics)
        
        # --- [MODIFIED] Plot and save summary for THIS file (FAA, FAP, OASI, TBR) ---
        plt.figure(figsize=(7, 4))
        
        # Note: We use 'metric_keys' defined outside the loop
        current_values = [metrics[key] for key in metric_keys] 
        
        bars = plt.bar(metric_keys, current_values, color=plot_colors)
        plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        
        plt.title(f"EEG Feature Summary ({testcase_name})")
        plt.ylabel("Value")
        
        # adjust y-axis limits to give space for labels
        ymin, ymax = plt.ylim()
        padding = max(abs(ymax) * 0.1, abs(ymin) * 0.1, 0.5) 
        plt.ylim(ymin - padding, ymax + padding) 
        
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()

        img_path = os.path.join(OUTPUT_FOLDER_NAME, f"EEG_features_{testcase_name}.png")
        plt.savefig(img_path)
        plt.close()
        # ----------------------------------------------------
        
        print(f"*** Successfully processed {testcase_name} ***")
        print(f"*** Individual plots saved to {OUTPUT_FOLDER_NAME} ***\n")


    except Exception as e:
        print(f"*** ERROR processing {testcase_name}: {e} ***\n")

# --- 4. Save All Results & Average Plot ---
if all_metrics:
    df = pd.DataFrame(all_metrics)
    
    # Save the CSV with individual results (as before)
    csv_path = os.path.join(CSV_FOLDER_NAME, f"EEG_features_{TESTCASE}_ALL.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"--- All processing complete ---")
    print(f"Successfully processed {len(all_metrics)} files.")
    print(f"Individual results saved to: {csv_path}")
    
    # --- [MODIFIED] Plot and save AVERAGE summary plot (FAA, FAP, OASI, TBR) ---
    
    # Calculate averages (for plotting keys only)
    average_metrics = df[metric_keys].mean()

    plt.figure(figsize=(7, 4.5)) 
    
    bars = plt.bar(average_metrics.index, average_metrics.values, color=plot_colors)
    
    plt.title(f"Average EEG ({TESTCASE}) - {len(all_metrics)} files")
    plt.ylabel("Average Value")
    
    plt.grid(alpha=0.3, linestyle='--')
    
    plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)

    # adjust y-axis limits to give space for labels
    ymin, ymax = plt.ylim()
    padding = max(abs(ymax) * 0.1, abs(ymin) * 0.1, 0.5) 
    plt.ylim(ymin - padding, ymax + padding) 
    
    plt.tight_layout()

    img_path_avg = os.path.join(CSV_FOLDER_NAME, f"EEG_features_{TESTCASE}_AVERAGE_plot.png")
    plt.savefig(img_path_avg)
    plt.close()
    
    print(f"Average summary plot saved: {img_path_avg}")
    
    # --- [MODIFIED] Create and save combined CSV (individual + average) ---
    
    # Calculate averages for ALL metrics for the CSV
    all_average_metrics = df[['FAA', 'FAP', 'OAA', 'OASI', 'TBR']].mean()
    
    avg_row = {'testcase': f'AVERAGE ({len(all_metrics)} files)'}
    avg_row.update(all_average_metrics.to_dict())
    
    avg_df = pd.DataFrame([avg_row])
    
    df_combined = pd.concat([df, avg_df], ignore_index=True)
    
    csv_path_combined = os.path.join(CSV_FOLDER_NAME, f"EEG_features_{TESTCASE}_ALL_with_AVG.csv")
    df_combined.to_csv(csv_path_combined, index=False)
    # -------------------------------------------------------------------
    
    print(f"Combined (Individual + AVG) results saved to: {csv_path_combined}")


    # --- [NEW] Plot and save UNIFIED PSD CURVE plot ---
    if all_psds_list and all_freqs is not None and ch_names_last is not None:
        print(f"Generating Unified PSD Curve plot...")
        try:
            # (A) Calculate the average PSD across all files
            unified_psd_avg = np.mean(all_psds_list, axis=0) # (n_channels, n_freqs)
            
            # (B) Plot unified PSD Curve
            plt.figure(figsize=(10, 6))
            channels_to_plot = [ch for ch in KEY_CHANNELS if ch in ch_names_last]
            for ch in channels_to_plot:
                ch_idx = ch_names_last.index(ch)
                plt.plot(all_freqs, 10 * np.log10(unified_psd_avg[ch_idx, :]), label=ch)
            
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density (dB/Hz)')
            plt.title(f'PSD Curves - AVERAGE ({TESTCASE}) - {len(all_psds_list)} files')
            plt.legend()
            plt.grid(alpha=0.5, linestyle='--')
            plt.tight_layout()
            
            psd_curve_img_path = os.path.join(summary_psd_folder, f"EEG_psd_curve_{TESTCASE}_AVERAGE.png")
            plt.savefig(psd_curve_img_path)
            plt.close()
            print(f"  Saved unified PSD curve plot to: {psd_curve_img_path}")

        except Exception as e:
            print(f"    Warning: Could not generate unified PSD curve plot. Error: {e}")
    # -----------------------------------------------

else:
    print("--- No files processed. Exiting. ---")

print("\n--- Analysis complete. ---")