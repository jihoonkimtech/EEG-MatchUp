import os
import glob
from scipy.io import loadmat
import numpy as np
import mne
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

target_folder = os.path.join(script_dir, "../raw/info/")
folder_name = os.path.basename(os.path.normpath(target_folder))

file_list = glob.glob(os.path.join(target_folder, "*.mat"))
file_list.sort()

print(f"--- 총 {len(file_list)}개의 파일을 찾았습니다 ---")

raw_list = []
chan_names = ["Ch" + str(i) for i in range(1, 7)]

for mat_path in file_list:
    print(f"======{os.path.basename(mat_path)}======")
    
    mat = loadmat(mat_path)
    
    # load data
    fs = int(mat['SR\x00'][0][0])  # sampling frequency
    data = mat['y']               # numpy array
    
    
    # check keys
    print(f"키 : {mat.keys()}")

    meta_keys = ['__header__', '__version__', '__globals__']
    for key in meta_keys:
            print(f"{key} : {mat[key]}")

    # check struct
    print(f"데이터 형태 : {type(mat['y']), mat['y'].shape}")

    # load data
    fs = int(mat['SR\x00'][0][0])           # sampling frequency
    data = mat['y']                         # numpy array

    print(f"Fs = {fs}Hz")


    # Origin 3D shape: (6, 1, 32766)
    if data.ndim == 3:
        data = data.squeeze()  # 2D shape after squeeze: (6, 32766)

    # shape fix (n_channels, n_samples)
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    n_channels, n_samples = data.shape
    
    # create MNE raw
    info = mne.create_info(ch_names=chan_names, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    raw_list.append(raw)
    print(f"=====================================================")
    
# combine all raw
if not raw_list:
    print("처리할 파일이 없습니다.")
else:
    print(f"\n--- {len(raw_list)}개의 파일을 하나로 통합합니다 ---")
    raw_combined = mne.concatenate_raws(raw_list)


    print(f"통합된 데이터 정보: {raw_combined.info}")
    
    save_path = os.path.join(target_folder, f"combined_{folder_name}_raw.fif")
    
    # .fif save
    raw_combined.save(save_path, overwrite=True)
    print(f"\n--- 합본 데이터 저장 ---")
    print(f"파일 위치: {save_path}")
    print("---------------------------------")

    # create plot
    print("플롯 생성 중...")
    raw_combined.plot(n_channels=n_channels, scalings='auto',
                      title="EEG Signals (All Files Combined)", show=True)
    
    # get data and time
    data, times = raw_combined.get_data(return_times=True)
    ch_names = raw_combined.info['ch_names']
    
    # define colors for 6 channels
    # (e.g., 'C0'=blue, 'C1'=orange, 'C2'=green...)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    
    # create 6 vertically stacked subplots
    # sharex=True means zooming one plot zooms all
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, 10), sharex=True)
    
    if n_channels == 1: # just in case
        axes = [axes]

    for i in range(n_channels):
        # plot each channel on its own axis with its own color
        axes[i].plot(times, data[i], color=colors[i], linewidth=0.7)
        
        # add channel name to the y-axis
        # 'rotation=0' makes it horizontal
        axes[i].set_ylabel(ch_names[i], rotation=0, 
                           horizontalalignment='right', 
                           verticalalignment='center',
                           fontweight='bold', fontsize=10)
        
        # styling: remove top/right/left borders
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['left'].set_visible(False)
        axes[i].set_yticks([]) # remove y-axis numbers

    # set title and x-axis label only on the last plot
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('EEG Signals (Colored & Spaced per Channel)', fontsize=16)
    
    eeg_save_path = os.path.join(target_folder, f"combined_{folder_name}_sig_plot_bf_pp.png")
    fig.savefig(eeg_save_path)

    psd_fig = raw_combined.plot_psd(fmin=1, fmax=45, average=True, show=True)
    
    psd_save_path = os.path.join(target_folder, f"combined_{folder_name}_psd_plot_bf_pp.png")
    psd_fig.savefig(psd_save_path)
    
    # adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    print("\n--- 모든 처리 완료 ---")

    