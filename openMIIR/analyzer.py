import os
import sys
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SUBJECT = 'P01' 
PREPROCESSED_FOLDER = 'preprocessed'
ANALYSIS_FOLDER = 'analyze'

# --- [NEW] Define all frequency bands ---
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)
ALL_BANDS = {'Theta': THETA_BAND, 'Alpha': ALPHA_BAND, 'Beta': BETA_BAND}
PSD_FMIN = 1.0
PSD_FMAX = 45.0
# ----------------------------------------


def calculate_asymmetry(psds, freqs, ch_left, ch_right, ch_names, band):
    """
    Calculates asymmetry and average log power for a specific band.
    psds: (n_channels, n_freqs) - average psd
    """
    try:
        # find the frequency indices corresponding to the band
        freq_mask = (freqs >= band[0]) & (freqs <= band[1])
        
        # find the channel indices
        ch_left_idx = [ch_names.index(ch) for ch in ch_left]
        ch_right_idx = [ch_names.index(ch) for ch in ch_right]
        
        # (1) calculate mean power "in the band"
        psds_band = psds[:, freq_mask]
        power_left = psds_band[ch_left_idx, :].mean(axis=1).mean()
        power_right = psds_band[ch_right_idx, :].mean(axis=1).mean()
        
        epsilon = 1e-10 # prevent log(0)
        
        # (2) log-transform the average power
        log_power_left = np.log(power_left + epsilon)
        log_power_right = np.log(power_right + epsilon)
        
        # (3) Asymmetry = ln(Right) - ln(Left)
        asymmetry = log_power_right - log_power_left
        
        # (4) Average Power = (ln(Right) + ln(Left)) / 2
        avg_log_power = (log_power_right + log_power_left) / 2.0
        
        return asymmetry, avg_log_power
        
    except Exception as e:
        print(f"    Error in calculate_asymmetry: {e}")
        return np.nan, np.nan

def calculate_tbr(psds, freqs, ch_f_left, ch_f_right, ch_names, theta_band, beta_band):
    """
    Calculates Theta/Beta Ratio (TBR) for frontal channels.
    psds: (n_channels, n_freqs) - average psd
    """
    try:
        # find frequency indices
        theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
        beta_mask = (freqs >= beta_band[0]) & (freqs <= beta_band[1])
        
        # find channel indices
        ch_f_left_idx = [ch_names.index(ch) for ch in ch_f_left]
        ch_f_right_idx = [ch_names.index(ch) for ch in ch_f_right]
        
        # get power for each channel in each band
        theta_left = psds[ch_f_left_idx, :][:, theta_mask].mean(axis=1).mean()
        theta_right = psds[ch_f_right_idx, :][:, theta_mask].mean(axis=1).mean()
        beta_left = psds[ch_f_left_idx, :][:, beta_mask].mean(axis=1).mean()
        beta_right = psds[ch_f_right_idx, :][:, beta_mask].mean(axis=1).mean()
        
        # average frontal power
        frontal_theta_avg = (theta_left + theta_right) / 2.0
        frontal_beta_avg = (beta_left + beta_right) / 2.0
        
        # calculate TBR
        TBR = frontal_theta_avg / (frontal_beta_avg + 1e-10) # add epsilon
        return TBR

    except Exception as e:
        print(f"    Error in calculate_tbr: {e}")
        return np.nan
# ----------------------------------------


# --- 1. 경로 및 폴더 설정 ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
preprocessed_dir = os.path.join(script_dir, PREPROCESSED_FOLDER)
analysis_dir = os.path.join(script_dir, ANALYSIS_FOLDER)

# --- [NEW] Create subfolders for plots ---
analysis_topo_dir = os.path.join(analysis_dir, 'topomaps')
analysis_psd_dir = os.path.join(analysis_dir, 'summary_psd_curves')
analysis_psd_dir_per_song = os.path.join(analysis_dir, 'psd_curves')
os.makedirs(analysis_dir, exist_ok=True)
os.makedirs(analysis_topo_dir, exist_ok=True)
os.makedirs(analysis_psd_dir, exist_ok=True)
print(f"Analysis output folder created at: ./{ANALYSIS_FOLDER}/")
print(f"  (saving topomaps to ./topomaps/)")
print(f"  (saving PSD curves to ./summary_psd_curves/)")
# -----------------------------------------

input_epochs_file = os.path.join(preprocessed_dir, f'{SUBJECT}-preprocessed-epo.fif')


# --- 2. 통합 Epochs 파일 로드 ---
print(f"Loading combined epochs file: {input_epochs_file}")
try:
    epochs = mne.read_epochs(input_epochs_file, preload=True)
except FileNotFoundError:
    print(f"Error: File not found. Run pre-processor.py first.")
    print(f"    (Expected file at: {input_epochs_file})")
    sys.exit(1) 

expected_channels = {'F3', 'F4', 'O1', 'O2'}
if not expected_channels.issubset(set(epochs.ch_names)):
    print(f"Error: Epochs file missing required channels! Got: {epochs.ch_names}, Expected: {expected_channels}")
    sys.exit(1)

song_tags = list(epochs.event_id.keys())
print(f"Loaded epochs with {len(song_tags)} event types (songs).")


# --- 3. 음악별 비대칭성 분석 ---
FRONTAL_LEFT = ['F3']
FRONTAL_RIGHT = ['F4']
OCCIPITAL_LEFT = ['O1']
OCCIPITAL_RIGHT = ['O2']
KEY_CHANNELS = ['F3', 'F4', 'O1', 'O2']

all_results = []
print(f"\n--- Starting Per-Song Analysis (Bands: T={THETA_BAND}, A={ALPHA_BAND}, B={BETA_BAND} Hz) ---")

# --- [NEW] 3.5. 전체 통합 PSD 계산 및 플롯 ---
print(f"\n--- Calculating Unified PSD for ALL {len(epochs)} epochs ---")
try:
    # (A) Calculate PSD for ALL epochs combined
    spectrum_all = epochs.compute_psd(
        method='welch',
        fmin=PSD_FMIN,
        fmax=PSD_FMAX,
        picks=epochs.ch_names,
        n_jobs=-1,
        n_fft=128 
    )
    psds_all_avg = spectrum_all.get_data().mean(axis=0) # (n_channels, n_freqs)
    freqs_all = spectrum_all.freqs
    ch_names_all = epochs.ch_names
    
    # (B) Plot unified PSD Curve
    plt.figure(figsize=(10, 6))
    channels_to_plot = [ch for ch in KEY_CHANNELS if ch in ch_names_all]
    for ch in channels_to_plot:
        ch_idx = ch_names_all.index(ch)
        plt.plot(freqs_all, 10 * np.log10(psds_all_avg[ch_idx, :]), label=ch)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.title(f'PSD Curves - ALL MUSIC ({SUBJECT})') # Unified Title
    plt.legend()
    plt.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    
    psd_curve_img_path = os.path.join(analysis_psd_dir, f"{SUBJECT}_ALL_MUSIC_psd_curve.png")
    plt.savefig(psd_curve_img_path)
    plt.close()
    print(f"  Saved unified PSD curve plot to: {psd_curve_img_path}")

except Exception as e:
    print(f"    Warning: Could not generate unified PSD curve plot. Error: {e}")
# -----------------------------------------------


all_results = []
print(f"\n--- Starting Per-Song Analysis (Bands: T={THETA_BAND}, A={ALPHA_BAND}, B={BETA_BAND} Hz) ---")

for song_tag in song_tags:
    print(f"Processing: {song_tag}")
    
    try:
        song_epochs = epochs[song_tag]
        if len(song_epochs) == 0:
            print("  Skipping (no epochs found for this tag).")
            continue
    except KeyError:
        print(f"  Skipping (KeyError): {song_tag}")
        continue
    
    # clean song name for file paths
    song_name = song_tag.split('/', 1)[-1].replace(' ', '_').replace('/', '_')
        
    print(f"  Calculating PSD for {len(song_epochs)} epochs...")
    try:
        # (A) Calculate PSD for a WIDE range (1-45Hz)
        spectrum = song_epochs.compute_psd(
            method='welch',
            fmin=PSD_FMIN,
            fmax=PSD_FMAX,
            picks=epochs.ch_names,
            n_jobs=-1,
            n_fft=128 # n_fft must be <= n_times
        )
        psds = spectrum.get_data() # (n_epochs, n_channels, n_freqs)
        freqs = spectrum.freqs
        
        # average PSD across epochs
        psds_avg = psds.mean(axis=0) # (n_channels, n_freqs)
        ch_names = song_epochs.ch_names

        # --- [NEW PLOT 1: Per-Song Topomap] ---
        try:
            topo_fig = spectrum.plot_topomap(bands=ALL_BANDS, ch_type='eeg', show=False, vlim=(None, None))
            topo_fig.suptitle(f'Topomap - {song_name}', fontsize=14)
            topo_img_path = os.path.join(analysis_topo_dir, f"{SUBJECT}_{song_name}_topo.png")
            topo_fig.savefig(topo_img_path)
            plt.close(topo_fig)
        except Exception as e:
            print(f"    Warning: Could not generate topomap plot for {song_name}. Error: {e}")
        # ----------------------------------------

        # --- [NEW PLOT 2: Per-Song PSD Curve] ---
        try:
            plt.figure(figsize=(10, 6))
            channels_to_plot = [ch for ch in KEY_CHANNELS if ch in ch_names]
            for ch in channels_to_plot:
                ch_idx = ch_names.index(ch)
                plt.plot(freqs, 10 * np.log10(psds_avg[ch_idx, :]), label=ch)
            
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density (dB/Hz)')
            plt.title(f'PSD Curves - {song_name}')
            plt.legend()
            plt.grid(alpha=0.5, linestyle='--')
            plt.tight_layout()
            
            psd_curve_img_path = os.path.join(analysis_psd_dir_per_song, f"{SUBJECT}_{song_name}_psd_curve.png")
            plt.savefig(psd_curve_img_path)
            plt.close()
        except Exception as e:
            print(f"    Warning: Could not generate PSD curve plot for {song_name}. Error: {e}")
        # ----------------------------------------

        # (B) Calculate Alpha Asymmetry metrics
        FAA, FAP = calculate_asymmetry(psds_avg, freqs, FRONTAL_LEFT, FRONTAL_RIGHT, ch_names, ALPHA_BAND)
        OAA, OASI = calculate_asymmetry(psds_avg, freqs, OCCIPITAL_LEFT, OCCIPITAL_RIGHT, ch_names, ALPHA_BAND)

        # (C) Calculate TBR metric
        TBR = calculate_tbr(psds_avg, freqs, FRONTAL_LEFT, FRONTAL_RIGHT, ch_names, THETA_BAND, BETA_BAND)

        print(f"  Metrics: FAA={FAA:.3f}, FAP={FAP:.3f}, OAA={OAA:.3f}, OASI={OASI:.3f}, TBR={TBR:.3f}")

        # (D) Store results
        metrics = {
            'song_name': song_name,
            'testcase': song_tag, 
            'FAA': FAA, 'FAP': FAP, 'OASI': OASI, 'TBR': TBR
        }
        all_results.append(metrics)

    except Exception as e:
        print(f"    Error calculating PSD/Metrics for {song_name}: {e}")


# --- 4. 최종 결과 저장 (CSV 및 3개의 분리된 플롯) ---
if all_results:
    output_csv_file = os.path.join(analysis_dir, f'{SUBJECT}-all_metrics.csv')
    print(f"\n--- Saving all results to: {output_csv_file} ---")
    
    # (A) DataFrame 생성 및 CSV 저장
    results_df = pd.DataFrame(all_results)
    # sort by song_name for consistent plotting
    results_df = results_df.sort_values(by='song_name').reset_index(drop=True)
    results_df.to_csv(output_csv_file, index=False)
    print(results_df.head())

    # --- [플롯 1] 비대칭 (FAA, OAA) Subplots 생성 ---
    print("Generating Asymmetry (FAA, OAA) plot...")
    try:
        plot_metric_keys = ['FAA']
        num_songs = len(results_df)
        ncols = 4
        nrows = int(np.ceil(num_songs / ncols))
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                                 figsize=(ncols * 4, nrows * 3.5), 
                                 squeeze=False)
        ax_flat = axes.flatten()

        for i, (index, row) in enumerate(results_df.iterrows()):
            ax = ax_flat[i]
            song_name = row['song_name']
            metrics = row[plot_metric_keys]
            
            bars = ax.bar(metrics.index, metrics.values, color=['tomato', 'royalblue'])
            
            ax.set_title(song_name, fontsize=10)
            ax.set_ylabel("Asymmetry Value (ln(R)-ln(L))")
            ax.grid(alpha=0.3, linestyle='--')
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_ylim(-0.05, 0.05) # slightly wider range
            
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval, 
                         f'{yval:.3f}', 
                         va='bottom' if yval >= 0 else 'top',
                         ha='center', fontsize=9)

        for j in range(i + 1, len(ax_flat)):
            ax_flat[j].axis('off')

        fig.suptitle(f"Per-Song Asymmetry Metrics ({SUBJECT}) - Alpha Band", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        img_path = os.path.join(analysis_dir, f"{SUBJECT}-PLOT_1_Asymmetry (FAA).png")
        plt.savefig(img_path)
        plt.close(fig)
        print(f"  Saved Asymmetry plot to: {img_path}")
        
    except Exception as e:
        print(f"    Warning: Could not generate Asymmetry plot. Error: {e}")


    # --- [플롯 2] 로그 파워 (FAP, OASI) Subplots 생성 ---
    print("Generating Log Power (FAP, OASI) plot...")
    try:
        plot_metric_keys = ['FAP', 'OASI'] # 파워만
        
        num_songs = len(results_df)
        ncols = 4
        nrows = int(np.ceil(num_songs / ncols))
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                                 figsize=(ncols * 4, nrows * 3.5), 
                                 squeeze=False)
        ax_flat = axes.flatten()
        
        # find global min/max for FAP/OASI to set a consistent y-axis
        global_min = results_df[plot_metric_keys].min().min()
        global_max = results_df[plot_metric_keys].max().max()
        padding = (global_max - global_min) * 0.1
        ylim = (global_min - padding, global_max + padding)

        for i, (index, row) in enumerate(results_df.iterrows()):
            ax = ax_flat[i]
            song_name = row['song_name']
            metrics = row[plot_metric_keys]
            
            bars = ax.bar(metrics.index, metrics.values, color=['mediumseagreen', 'orange'])
            
            ax.set_title(song_name, fontsize=10)
            ax.set_ylabel("Avg Log Power Value ((ln(R)+ln(L))/2)")
            ax.grid(alpha=0.3, linestyle='--')
            
            ax.set_ylim(ylim) # set consistent y-axis
            
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval, 
                         f'{yval:.3f}', 
                         va='bottom', ha='center', fontsize=9)

        for j in range(i + 1, len(ax_flat)):
            ax_flat[j].axis('off')

        fig.suptitle(f"Per-Song Log Power Metrics ({SUBJECT}) - Alpha Band", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        img_path = os.path.join(analysis_dir, f"{SUBJECT}-PLOT_2_LogPower (FAP,OASI).png")
        plt.savefig(img_path)
        plt.close(fig)
        print(f"  Saved Log Power plot to: {img_path}")
        
    except Exception as e:
        print(f"    Warning: Could not generate Log Power plot. Error: {e}")

    # --- [NEW PLOT 3: TBR (Theta/Beta Ratio)] ---
    print("Generating TBR (Theta/Beta Ratio) plot...")
    try:
        plot_metric_keys = ['TBR'] # TBR
        
        num_songs = len(results_df)
        ncols = 4
        nrows = int(np.ceil(num_songs / ncols))
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, 
                                 figsize=(ncols * 4, nrows * 3.5), 
                                 squeeze=False)
        ax_flat = axes.flatten()
        
        # find global min/max for TBR
        global_min = results_df[plot_metric_keys].min().min()
        global_max = results_df[plot_metric_keys].max().max()
        padding = (global_max - global_min) * 0.1
        #ylim = (global_min - padding, global_max + padding + 0.1) # add extra top padding for label

        for i, (index, row) in enumerate(results_df.iterrows()):
            ax = ax_flat[i]
            song_name = row['song_name']
            metrics = row[plot_metric_keys]
            
            bars = ax.bar(metrics.index, metrics.values, color=['purple'])
            
            ax.set_title(song_name, fontsize=10)
            ax.set_ylabel("Frontal TBR (Theta/Beta)")
            ax.grid(alpha=0.3, linestyle='--')
            
            ax.set_ylim(-0.1, 0.1) # set consistent y-axis
            
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval, 
                         f'{yval:.3f}', 
                         va='bottom', ha='center', fontsize=9)

        for j in range(i + 1, len(ax_flat)):
            ax_flat[j].axis('off')

        fig.suptitle(f"Per-Song Frontal TBR ({SUBJECT})", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        img_path = os.path.join(analysis_dir, f"{SUBJECT}-PLOT_3_TBR.png")
        plt.savefig(img_path)
        plt.close(fig)
        print(f"  Saved TBR plot to: {img_path}")
        
    except Exception as e:
        print(f"    Warning: Could not generate TBR plot. Error: {e}")
        
else:
    print("\n--- No results to save. ---")

print(f"\n--- Analysis complete. Results saved in ./{ANALYSIS_FOLDER}/ ---")