import os
import mne
import numpy as np

def scan_fif_files():

    # Finds all .fif files in the script's directory and prints their info.

    # get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_folder = os.path.join(script_dir, "./eeg/")
    print(f"Scanning for .fif files in: {target_folder}\n")

    found_files = False
    
    # suppress MNE's default info messages during the scan
    mne.set_log_level('WARNING') 

    # loop through all files in the directory
    for filename in os.listdir(target_folder):
        # check if it's a .fif file
        if filename.endswith('.fif'):
            found_files = True
            filepath = os.path.join(script_dir, filename)
            print(f"--- File: {filename} ---")

            # Try reading as a Raw file
            try:
                # preload=False makes it load instantly
                raw = mne.io.read_raw_fif(filepath, preload=False)
                info = raw.info
                duration = raw.n_times / info['sfreq']
                n_annots = len(raw.annotations)
                
                print("  [Type: Raw data file]")
                print(f"    • Channels: {info['nchan']}")
                print(f"    • Channel Names: {info['ch_names']}")
                print(f"    • Sampling Freq: {info['sfreq']:.2f} Hz")
                print(f"    • Duration: {duration:.2f} seconds")
                print(f"    • Annotations: {n_annots} found")

            except ValueError:
                # If it's not Raw, try reading as Epochs
                try:
                    # preload=False makes it load instantly
                    epochs = mne.read_epochs(filepath, preload=False)
                    info = epochs.info
                    n_epochs = len(epochs)
                    n_times = len(epochs.times)
                    
                    print("  [Type: Epochs data file]")
                    print(f"    • Channels: {info['nchan']}")
                    print(f"    • Channel Names: {info['ch_names']}")
                    print(f"    • Sampling Freq: {info['sfreq']:.2f} Hz")
                    print(f"    • Number of Epochs: {n_epochs}")
                    print(f"    • Epoch Length: {n_times} samples ({epochs.tmin:.2f}s to {epochs.tmax:.2f}s)")
                    print(f"    • Event IDs: {epochs.event_id}")

                except Exception as e_epoch:
                    print(f"  *** ERROR: Could not read file (neither Raw nor Epochs). ***")
                    print(f"    {e_epoch}")
            
            except Exception as e_raw:
                # Handle other loading errors
                print(f"  *** ERROR: Could not read Raw file. ***")
                print(f"    {e_raw}")
            
            print("-" * (len(filename) + 6))

    if not found_files:
        print("No .fif files found in this directory.")

if __name__ == "__main__":
    scan_fif_files()