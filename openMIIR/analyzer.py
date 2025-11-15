import os
import sys
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SUBJECT = 'P14' 
PREPROCESSED_FOLDER = 'preprocessed'
ANALYSIS_FOLDER = 'analyze'
ALPHA_BAND = (8.0, 13.0)


def calculate_asymmetry(psds, freqs, ch_left, ch_right, ch_names, band):
    try:
        # find the frequency indices corresponding to the band
        freq_mask = (freqs >= band[0]) & (freqs <= band[1])
        
        # find the channel indices
        ch_left_idx = [ch_names.index(ch) for ch in ch_left]
        ch_right_idx = [ch_names.index(ch) for ch in ch_right]
        
        # (1) "원본 파워"의 평균을 계산
        psds_band = psds[:, freq_mask]
        power_left = psds_band[ch_left_idx, :].mean(axis=1).mean()
        power_right = psds_band[ch_right_idx, :].mean(axis=1).mean()
        
        epsilon = 1e-10 # 0 방지
        
        #     두 채널의 "로그 파워"를 먼저 계산
        log_power_left = np.log(power_left + epsilon)
        log_power_right = np.log(power_right + epsilon)
        
        # (3) Asymmetry = ln(Right) - ln(Left)
        asymmetry = log_power_right - log_power_left
        
        # (4) Average Power = (ln(Right) + ln(Left)) / 2
        avg_log_power = (log_power_right + log_power_left) / 2.0
        
        return asymmetry, avg_log_power # 🔺🔺🔺 [수정됨] 🔺🔺🔺
        
    except Exception as e:
        print(f"    Error in calculate_asymmetry: {e}")
        return np.nan, np.nan
# ----------------------------------------


# --- 1. 경로 및 폴더 설정 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
preprocessed_dir = os.path.join(script_dir, PREPROCESSED_FOLDER)
analysis_dir = os.path.join(script_dir, ANALYSIS_FOLDER)

os.makedirs(analysis_dir, exist_ok=True)
print(f"Analysis output folder created at: ./{ANALYSIS_FOLDER}/")

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
if set(epochs.ch_names) != expected_channels:
    print(f"Error: Epochs file missing required channels! Got: {epochs.ch_names}")
    sys.exit(1)

song_tags = list(epochs.event_id.keys())
print(f"Loaded epochs with {len(song_tags)} event types (songs).")


# --- 3. 음악별 비대칭성 분석 ---
FRONTAL_LEFT = ['F3']
FRONTAL_RIGHT = ['F4']
OCCIPITAL_LEFT = ['O1']
OCCIPITAL_RIGHT = ['O2']

all_results = []
print(f"\n--- Starting Asymmetry Analysis (Alpha Band: {ALPHA_BAND} Hz) ---")

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
        
    print(f"  Calculating PSD for {len(song_epochs)} epochs...")
    try:
        # (A) MNE 1.0+ 버전 호환 PSD 계산
        spectrum = song_epochs.compute_psd(
            method='welch',
            fmin=ALPHA_BAND[0],
            fmax=ALPHA_BAND[1],
            picks=epochs.ch_names,
            n_jobs=-1
        )
        psds = spectrum.get_data() 
        freqs = spectrum.freqs
        
        psds_avg = psds.mean(axis=0) 
        
        # (B) 메트릭 계산
        ch_names = song_epochs.ch_names
        FAA, FAP = calculate_asymmetry(psds_avg, freqs, FRONTAL_LEFT, FRONTAL_RIGHT, ch_names, ALPHA_BAND)
        OAA, OASI = calculate_asymmetry(psds_avg, freqs, OCCIPITAL_LEFT, OCCIPITAL_RIGHT, ch_names, ALPHA_BAND)

        print(f"  Metrics: FAA={FAA:.3f}, FAP={FAP:.3f}, OAA={OAA:.3f}, OASI={OASI:.3f}")

        # (C) 결과 저장
        song_name = song_tag.split('/', 1)[1] # 'Perception/SongName' -> 'SongName'
        metrics = {
            'song_name': song_name,
            'testcase': song_tag, 
            'FAA': FAA, 'FAP': FAP, 'OAA': OAA, 'OASI': OASI
        }
        all_results.append(metrics)

    except Exception as e:
        print(f"    Error calculating PSD/Metrics: {e}")


# --- 4. 최종 결과 저장 (CSV 및 2개의 분리된 플롯) ---
if all_results:
    output_csv_file = os.path.join(analysis_dir, f'{SUBJECT}-asymmetry_metrics.csv')
    print(f"\n--- Saving all results to: {output_csv_file} ---")
    
    # (A) DataFrame 생성 및 CSV 저장
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_csv_file, index=False)
    print(results_df.head()) # (사용자님이 보여주신 -22.9가 나오는 부분)

    # --- [플롯 1] 비대칭 (FAA, OAA) Subplots 생성 ---
    print("Generating Asymmetry (FAA, OAA) plot...")
    try:
        plot_metric_keys = ['FAA', 'OAA'] # 비대칭만
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
            
            bars = ax.bar(metrics.index, metrics.values)
            
            ax.set_title(song_name, fontsize=10)
            ax.set_ylabel("Asymmetry Value (ln(R)-ln(L))")
            ax.grid(alpha=0.3, linestyle='--')
            ax.axhline(0, color='black', linewidth=0.8)
            ax.set_ylim(-0.1, 0.1)
            
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
        
        img_path = os.path.join(analysis_dir, f"{SUBJECT}-PLOT_1_Asymmetry (FAA,OAA).png")
        plt.savefig(img_path)
        plt.close(fig)
        print(f"  Saved Asymmetry plot to: {img_path}")
        
    except Exception as e:
        print(f"    Warning: Could not generate Asymmetry plot. Error: {e}")


    # --- [플롯 2] 로그 파워 (FAP, OASI) Subplots 생성 ---
    print("Generating Log Power (FAP, OASI) plot...")
    try:
        plot_metric_keys = ['FAP', 'OASI'] # 파워만
        
        # (그리드 설정은 위와 동일)
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
            
            bars = ax.bar(metrics.index, metrics.values)
            
            ax.set_title(song_name, fontsize=10)
            ax.set_ylabel("Avg Log Power Value ((ln(R)+ln(L))/2)")
            ax.grid(alpha=0.3, linestyle='--')
            
            # 파워 플롯의 Y축 (-22.9 근처)
            ax.set_ylim(-22.5, -23.5) 
            
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
        
else:
    print("\n--- No results to save. ---")

print(f"\n--- Analysis complete. Results saved in ./{ANALYSIS_FOLDER}/ ---")