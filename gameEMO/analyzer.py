import os
import glob
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# EEG Feature Extractor (FAA, OAA, OASI, FAP)
# ----------------------------------------------------------

# User Configuration
TESTCASE = "G1-TrainSim"

# input folder path containing .mat files (relative to this script)
INPUT_FOLDER_NAME = f"./pre-processed/{TESTCASE}"
OUTPUT_FOLDER_NAME = f"./analyze/{TESTCASE}"
CSV_FOLDER_NAME = f"./analyze/Summary/"


script_dir = os.path.dirname(os.path.abspath(__file__))

target_folder = os.path.join(script_dir, INPUT_FOLDER_NAME)

output_folder = os.path.join(script_dir, OUTPUT_FOLDER_NAME)

csv_folder = os.path.join(script_dir, CSV_FOLDER_NAME)

# --- Create output folder if it doesn't exist ---
os.makedirs(output_folder, exist_ok=True)

REACTION_DELAY_SECONDS = 0.5
# ----------------------------------------------------------

print(f"--- Starting Batch Feature Extraction ---")
print(f"Scanning for epoch files in: {target_folder}\n")

epoch_files = glob.glob(os.path.join(target_folder, "*-epo.fif"))

if not epoch_files:
    print(f"*** ERROR: No '*-epo.fif' files found in {target_folder} ***")
    exit()

all_metrics = []

# --- Define plot metrics and colors for consistency ---
metric_keys = ['FAA', 'FAP', 'OAA', 'OASI']
plot_colors = ['tomato', 'royalblue', 'mediumseagreen', 'orange']


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

        psd = np.mean(spectrum.get_data(), axis=0) 
        freqs = spectrum.freqs

        alpha_idx = np.where((freqs >= 8) & (freqs <= 13))[0]
        alpha_power = np.mean(psd[:, alpha_idx], axis=1)
        alpha_dict = dict(zip(epochs.ch_names, alpha_power))

        # --- 3. Calculate EEG indices ---
        def safe_log(val):
            if pd.isna(val):
                return np.nan
            return np.log(val + 1e-10)

        def get_if_exists(ch):
            return alpha_dict[ch] if ch in alpha_dict else np.nan

        F3, F4, O1, O2 = [get_if_exists(ch) for ch in ['F3', 'F4', 'O1', 'O2']]

        FAA = safe_log(F4) - safe_log(F3)
        OAA = safe_log(O2) - safe_log(O1)
        OASI = (O1 + O2) / 2
        FAP = (F3 + F4) / 2
        # ------------------------------------------------------------

        metrics = {
            'testcase': testcase_name,
            'FAA': FAA,
            'FAP': FAP,
            'OAA': OAA,
            'OASI': OASI
        }
        
        all_metrics.append(metrics)
        
        # --- Plot and save summary for THIS file ---
        plt.figure(figsize=(7, 4))
        
        current_values = [metrics[key] for key in metric_keys] 
        
        plt.bar(metric_keys, current_values, color=plot_colors)
        plt.title(f"EEG Feature Summary ({testcase_name})")
        plt.ylabel("Value")
        plt.ylim(-10.0, 10.0) 
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()

        img_path = os.path.join(output_folder, f"EEG_features_{testcase_name}.png")
        plt.savefig(img_path)
        plt.close()
        
        print(f"*** Successfully processed {testcase_name} ***")
        print(f"*** Individual plot saved to {img_path} ***\n")


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
    
    # --- Plot and save AVERAGE summary plot ---
    
    # Calculate averages
    average_metrics = df[metric_keys].mean()

    plt.figure(figsize=(7, 4.5)) # slightly taller for labels
    
    # store bars to add labels
    bars = plt.bar(average_metrics.index, average_metrics.values, color=plot_colors)
    
    # Add testcase name to title
    plt.title(f"Average EEG ({TESTCASE}) - {len(all_metrics)} files")
    plt.ylabel("Average Value")
    
    # Adjust y-limit for labels
    plt.ylim(-10.0, 11.5) 
    plt.grid(alpha=0.3, linestyle='--')
    
    # Add data labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        # format label to 3 decimal places, center it
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, 
                 f'{yval:.3f}', 
                 va='bottom' if yval >= 0 else 'top', # position based on value
                 ha='center', fontsize=9)

    plt.tight_layout()

    img_path_avg = os.path.join(csv_folder, f"EEG_features_{TESTCASE}_AVERAGE_plot.png")
    plt.savefig(img_path_avg)
    plt.close()
    
    print(f"Average summary plot saved: {img_path_avg}")
    
    # --- Create and save combined CSV (individual + average) ---
    
    # create a new dictionary for the average row
    avg_row = {'testcase': f'AVERAGE ({len(all_metrics)} files)'}
    avg_row.update(average_metrics.to_dict())
    
    # convert dictionary to a DataFrame
    avg_df = pd.DataFrame([avg_row])
    
    # concatenate original data with the new average row
    df_combined = pd.concat([df, avg_df], ignore_index=True)
    
    # define new CSV path and save
    csv_path_combined = os.path.join(csv_folder, f"EEG_features_{TESTCASE}_ALL_with_AVG.csv")
    df_combined.to_csv(csv_path_combined, index=False)
    
    print(f"Combined (Individual + AVG) results saved to: {csv_path_combined}")

else:
    print("--- No files processed. Exiting. ---")