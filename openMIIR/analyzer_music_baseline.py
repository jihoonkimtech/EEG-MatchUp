import os
import sys
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- [중요] 한글 경로 오류 해결 코드 ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
mpl_config_dir = os.path.join(script_dir, ".mpl_config_music_base") # cache folder
os.environ['MPLCONFIGDIR'] = mpl_config_dir
os.makedirs(mpl_config_dir, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
# ----------------------------------------

# --- (1) 사용자 설정 ---
SUBJECT = 'P14' 
PREPROCESSED_FOLDER = 'preprocessed'
ANALYSIS_FOLDER = 'analyze'

# (중요) Baseline 으로 사용할 이벤트 태그 이름을 정확히 입력하세요.
BASELINE_TAG = 'T0' # 또는 'Rest' 등
# ---------------------------

# --- (2) 경로 설정 ---
preprocessed_dir = os.path.join(script_dir, PREPROCESSED_FOLDER)
analysis_dir = os.path.join(script_dir, ANALYSIS_FOLDER)
os.makedirs(analysis_dir, exist_ok=True)

# 출력될 파일 이름 접두사
output_prefix = f"{SUBJECT}_Baseline_AVG"

# --- (3) 주파수 밴드 정의 ---
THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)
BETA_BAND = (13.0, 30.0)
ALL_BANDS = {'Theta': THETA_BAND, 'Alpha': ALPHA_BAND, 'Beta': BETA_BAND}
KEY_CHANNELS = ['F3', 'F4', 'O1', 'O2']

print(f"--- Starting Baseline Analysis for '{SUBJECT}' ---")
print(f"Targeting Baseline Tag: '{BASELINE_TAG}'")

# --- (4) Epoch 파일 로드 ---
input_epochs_file = os.path.join(preprocessed_dir, f'{SUBJECT}-preprocessed-epo.fif')
try:
    epochs = mne.read_epochs(input_epochs_file, preload=True, verbose=False)
    print(f"Loaded epochs file: {input_epochs_file}")
except FileNotFoundError:
    print(f"*** ERROR: File not found: {input_epochs_file} ***")
    sys.exit(1)

# --- (5) Baseline 에포크만 선택 ---
try:
    baseline_epochs = epochs[BASELINE_TAG]
    print(f"Found {len(baseline_epochs)} epochs matching tag '{BASELINE_TAG}'.")
except KeyError:
    print(f"*** ERROR: Tag '{BASELINE_TAG}' not found in epoch events. ***")
    print(f"Available tags are: {list(epochs.event_id.keys())}")
    sys.exit(1)

# --- (6) PSD 계산 ---
try:
    spectrum = baseline_epochs.compute_psd(
        method='welch', fmin=1, fmax=45, n_fft=128, verbose=False
    )
    
    # --- [PLOT 1: Topomap] ---
    topo_fig = spectrum.plot_topomap(bands=ALL_BANDS, ch_type='eeg', show=False, vlim=(None, None))
    topo_fig.suptitle(f'Baseline Average Topomap ({SUBJECT} - {BASELINE_TAG})', fontsize=14)
    topo_img_path = os.path.join(analysis_dir, f"{output_prefix}_topo.png")
    topo_fig.savefig(topo_img_path)
    plt.close(topo_fig)
    print(f"Baseline Topomap saved to: {topo_img_path}")

    # (중요) 모든 Baseline 에포크의 PSD를 평균
    psd_avg = np.mean(spectrum.get_data(), axis=0) # (n_channels, n_freqs)
    freqs = spectrum.freqs

    # --- [PLOT 2: PSD Curve] ---
    plt.figure(figsize=(10, 6))
    channels_to_plot = [ch for ch in KEY_CHANNELS if ch in epochs.ch_names]
    for ch in channels_to_plot:
        ch_idx = epochs.ch_names.index(ch)
        plt.plot(freqs, 10 * np.log10(psd_avg[ch_idx, :]), label=ch)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.title(f'Baseline Average PSD Curves ({SUBJECT} - {BASELINE_TAG})')
    plt.legend()
    plt.grid(alpha=0.5, linestyle='--')
    plt.tight_layout()
    
    psd_curve_img_path = os.path.join(analysis_dir, f"{output_prefix}_psd_curve.png")
    plt.savefig(psd_curve_img_path)
    plt.close()
    print(f"Baseline PSD Curve plot saved to: {psd_curve_img_path}")

except Exception as e:
    print(f"*** ERROR during PSD/Plot generation: {e} ***")
    sys.exit(1)

# --- (7) 지표 계산 ---
try:
    # Find indices
    theta_idx = np.where((freqs >= THETA_BAND[0]) & (freqs <= THETA_BAND[1]))[0]
    alpha_idx = np.where((freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1]))[0]
    beta_idx = np.where((freqs >= BETA_BAND[0]) & (freqs <= BETA_BAND[1]))[0]

    # Get mean power per band (psd_avg 사용)
    theta_power = np.mean(psd_avg[:, theta_idx], axis=1)
    alpha_power = np.mean(psd_avg[:, alpha_idx], axis=1)
    beta_power = np.mean(psd_avg[:, beta_idx], axis=1)

    theta_dict = dict(zip(epochs.ch_names, theta_power))
    alpha_dict = dict(zip(epochs.ch_names, alpha_power))
    beta_dict = dict(zip(epochs.ch_names, beta_power))

    def safe_log(val):
        if pd.isna(val) or val <= 0: return np.nan
        return np.log(val)

    def get_power(ch, power_dict):
        return power_dict[ch] if ch in power_dict else np.nan

    # Get alpha powers
    F3_alpha, F4_alpha = [get_power(ch, alpha_dict) for ch in ['F3', 'F4']]
    O1_alpha, O2_alpha = [get_power(ch, alpha_dict) for ch in ['O1', 'O2']]

    # Get theta/beta powers
    F3_theta, F4_theta = [get_power(ch, theta_dict) for ch in ['F3', 'F4']]
    F3_beta, F4_beta = [get_power(ch, beta_dict) for ch in ['F3', 'F4']]

    # Calculate metrics
    FAA = safe_log(F4_alpha) - safe_log(F3_alpha)
    OAA = safe_log(O2_alpha) - safe_log(O1_alpha) # (Music/Game 에는 OAA 유지)
    OASI = (O1_alpha + O2_alpha) / 2
    FAP = (F3_alpha + F4_alpha) / 2
    
    frontal_theta_avg = (F3_theta + F4_theta) / 2
    frontal_beta_avg = (F3_beta + F4_beta) / 2
    TBR = frontal_theta_avg / (frontal_beta_avg + 1e-10)

    metrics = {
        'testcase': f'{SUBJECT}_Baseline_AVG',
        'FAA': FAA, 'FAP': FAP, 'OAA': OAA, 'OASI': OASI, 'TBR': TBR
    }
    
    print("\n--- Computed Baseline Metrics ---")
    for k, v in metrics.items():
        if k != 'testcase':
            print(f"{k}: {v:.5f}")

    # --- (8) CSV 저장 ---
    csv_path = os.path.join(analysis_dir, f"{output_prefix}_features.csv")
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    print(f"\nBaseline Metrics CSV saved to: {csv_path}")

    # --- (9) [PLOT 3: Bar Plot] ---
    metric_keys = ['FAA', 'FAP', 'OAA', 'OASI', 'TBR']
    metric_values = [metrics[k] for k in metric_keys]
    colors = ['tomato', 'royalblue', 'mediumseagreen', 'orange', 'purple']
    
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(metric_keys, metric_values, color=colors)
    plt.bar_label(bars, fmt='%.3f', padding=3, fontsize=10)
    
    plt.title(f'Baseline Average Metrics ({SUBJECT} - {BASELINE_TAG})')
    plt.ylabel("Value")
    
    ymin, ymax = plt.ylim()
    padding = max(abs(ymax) * 0.1, abs(ymin) * 0.1, 0.5) 
    plt.ylim(ymin - padding, ymax + padding) 
    
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    img_path = os.path.join(analysis_dir, f"{output_prefix}_bar_plot.png")
    plt.savefig(img_path)
    plt.close()
    print(f"Baseline Bar plot saved to: {img_path}")

except Exception as e:
    print(f"*** ERROR during Metric/CSV generation: {e} ***")

print("\n--- Baseline analysis complete. ---")