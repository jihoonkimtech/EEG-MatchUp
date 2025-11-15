import mne
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# process this subject
SUBJECT = 'P14' 

# source data folders
oDATA_FOLDER = './eeg/'
oMETA_FILE = './meta/Stimuli_Meta.v1.csv' # script and eeg folder's parent

# output folder
oOUTPUT_FOLDER = './preprocessed/'

script_dir = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(script_dir, oDATA_FOLDER)
META_FILE = os.path.join(script_dir, oMETA_FILE)
OUTPUT_FOLDER = os.path.join(script_dir, oOUTPUT_FOLDER)
# ------------------------------------------

# --- 1. 출력 폴더 생성 ---
# create the output directory if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"Output folder created at: ./{OUTPUT_FOLDER}/")


# --- 2. Stimuli 메타데이터 로드 ---
# (Using the 'xx1' = Perception logic we found)
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


# --- 3. 파일 경로 정의 ---
raw_fname = os.path.join(DATA_FOLDER, f'{SUBJECT}-raw.fif')
ica_fname = os.path.join(DATA_FOLDER, f'{SUBJECT}-100p_64c-ica.fif')

# all outputs will go to the 'preprocessed' folder
output_raw_plot_before = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-plot-01-raw_before_ica.png')
output_ica_plot = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-plot-02-ica_components.png')
output_raw_plot_after = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-plot-03-raw_after_ica.png')
output_epochs_plot = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-plot-04-epochs.png')
output_erp_plot = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-plot-05-erp_average.png')
output_epochs_fname = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-preprocessed-epo.fif')
output_heatmap_plot = os.path.join(OUTPUT_FOLDER, f'{SUBJECT}-plot-06-epochs_heatmap.png')


# --- 4. 데이터 및 ICA 로드 ---
print(f"Loading raw data from {raw_fname}...")
raw = mne.io.read_raw_fif(raw_fname, preload=True)

print(f"Loading ICA solution from {ica_fname}...")
ica = mne.preprocessing.read_ica(ica_fname) 


# --- 5. [PLOT 1] ICA 적용 전 원본 데이터 플롯 ---
print("Plotting raw data before ICA...")
try:
    fig = raw.plot(n_channels=64, duration=10, show=False)
    fig.savefig(output_raw_plot_before, dpi=300)
    plt.close(fig) # close figure to save memory
except Exception as e:
    print(f"Warning: Could not plot raw data. Error: {e}")


# --- 6. [PLOT 2] ICA 컴포넌트 플롯 ---
print(f"Plotting ICA components... (Excluding: {ica.exclude})")
try:
    # plot the component properties
    fig = ica.plot_components(show=False)
    fig[0].savefig(output_ica_plot, dpi=300) # plot_components returns a list of figures
    plt.close(fig[0])
except Exception as e:
    print(f"Warning: Could not plot ICA components. Error: {e}")


# --- 7. ICA 적용 (노이즈 제거) ---
print(f"Applying ICA... Excluding components: {ica.exclude}")
ica.apply(raw)


# --- 8. 리샘플링 및 필터링 (비교 데이터 사양 맞춤) ---
# (A) 리샘플링: 원본(e.g., 512Hz) -> 600Hz
print(f"Resampling data from {raw.info['sfreq']}Hz to 600Hz...")
raw.resample(600)

# (B) 노치 필터: 60Hz 라인 노이즈 제거
print("Applying 60Hz Notch filter...")
raw.notch_filter(freqs=60)

# (C) 대역통과 필터: 1Hz ~ 45Hz
print("Applying 1-45Hz Bandpass filter...")
raw.filter(l_freq=1.0, h_freq=45.0)
# --------------------------------------


# --- 8. [PLOT 3] ICA 적용 후 클린 데이터 플롯 ---
print("Plotting cleaned data after ICA...")
try:
    fig = raw.plot(n_channels=64, duration=10, show=False)
    fig.savefig(output_raw_plot_after, dpi=300)
    plt.close(fig)
except Exception as e:
    print(f"Warning: Could not plot cleaned raw data. Error: {e}")


# --- 9. 이벤트 정의 (Perception 'xx1'만) ---
print("Finding events and mapping 'Perception' (xx1) IDs...")
event_id_map = {}

# iterate from 1 to 30 (covers 1-24 from meta file)
for stimulus_id in range(1, 30): 
    
    # 'Perception' (condition=1) event ID
    # e.g., stimulus_id 11 -> event_id 111
    event_id = (stimulus_id * 10) + 1 # 1 = perception
    
    # get the actual song name from the map
    song_name = song_name_map.get(stimulus_id, f'UnknownStimulus_{stimulus_id}')
    
    # create hierarchical tag (e.g., 'Perception/Emperor Waltz')
    label = f'Perception/{song_name}'
    
    event_id_map[label] = event_id


# --- 10. Epoching ---
channels_to_pick = ['F3', 'F4', 'O1', 'O2']
print(f"Creating epochs for channels: {channels_to_pick}...")

try:
    events = mne.find_events(raw, shortest_event=1)
    event_dict = event_id_map
    
    print(f"Found {len(events)} events total.")
    # print(f"Found unique event codes: {set(events[:, 2])}") # (uncomment for debug)

    missing_channels = [ch for ch in channels_to_pick if ch not in raw.ch_names]
    if missing_channels:
        print(f"경고: 요청한 채널 중 일부가 raw 파일에 없습니다: {missing_channels}")
        print(f"    (파일에 있는 채널 예시: {raw.ch_names[:5]}...)")
        # 없는 채널을 제외하고 진행
        channels_to_pick = [ch for ch in channels_to_pick if ch in raw.ch_names]
        print(f"    -> 실제 추출될 채널: {channels_to_pick}")
    else:
        print(f"요청한 모든 채널 {channels_to_pick}을(를) 찾았습니다.")
    # ---------------------------------------------------

except Exception as e:
    print(f"Error finding events: {e}")
    events = []
    event_dict = {}

if len(events) > 0 and channels_to_pick:
    tmin = 1.0 
    tmax = 20.0  
    baseline = None

    print(f"Creating epochs from {tmin}s to {tmax}s...")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        on_missing='warn', 
        baseline=baseline,
        preload=True,
        reject=None,
        picks=channels_to_pick 
    )

    print("\n--- Generated Epochs (F3, F4, O1, O2 only) ---")
    print(epochs) 
    print(f"Epochs channel names: {epochs.ch_names}")

    # --- 12. 📊 [PLOT 4, 5, 6] Epochs, ERP, 및 히트맵 플롯 ---
    print("\n--- Generated Epochs (Filtered, Resampled, Specific Channels) ---")
    print(epochs) 
    print(f"Epochs channel names: {epochs.ch_names}")

    print("Plotting generated epochs, average ERP, and heatmap...")
    try:
        # [PLOT 4] Epochs snippet
        fig_epochs = epochs.plot(n_epochs=5, n_channels=4, show=False)
        fig_epochs.savefig(output_epochs_plot, dpi=300)
        plt.close(fig_epochs)
        
        # [PLOT 5] Average ERP
        fig_erp = epochs.average().plot(show=False)
        fig_erp.savefig(output_erp_plot, dpi=300)
        plt.close(fig_erp)

        # [PLOT 6] Epochs Heatmap (plot_image)
        print("Generating epochs heatmap (plot_image)...")
        fig_heatmap = epochs.plot_image(
            picks=channels_to_pick, # ['F3', 'F4', 'O1', 'O2']
            combine='mean',
            show=False,
            cmap='viridis'
        )
        fig_heatmap[0].savefig(output_heatmap_plot, dpi=300)
        plt.close(fig_heatmap[0])

    except Exception as e:
        print(f"Warning: Could not plot epochs. Error: {e}")


    # --- save ---
    epochs.save(output_epochs_fname, overwrite=True)
    print(f"\n✅ Successfully saved 'Perception'-only (4 channels) epochs to: {output_epochs_fname}")
else:
    if not channels_to_pick:
        print("\nError: 지정한 채널이 raw 파일에 하나도 없습니다. Epochs를 생성할 수 없습니다.")
    else:
        print("\nNo events were found. Epochs could not be created or saved.")

print(f"\n--- Preprocessing for {SUBJECT} complete. ---")