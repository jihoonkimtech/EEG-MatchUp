import os
import numpy as np
from scipy.io import loadmat
import h5py # Newer v7.3 .mat files

def scan_mat_files():
    # Finds all .mat files in the script's directory and prints their contents.
    # get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_folder = os.path.join(script_dir, "./eeg/G1-TrainSim")

    print(f"Scanning for .mat files in: {target_folder}\n")

    found_files = False

    # loop through all files in the directory
    for filename in os.listdir(target_folder):
        # check if it's a .mat file
        if filename.endswith('.mat'):
            found_files = True
            filepath = os.path.join(target_folder, filename)
            print(f"--- File: {filename} ---")

            try:
                # Try loading with scipy.io.loadmat
                data = loadmat(filepath)
                print("  [Type: Standard MATLAB file (loaded with scipy)]")

                # loop through all variables (keys) in the file
                for key, value in data.items():
                    # skip metadata keys
                    if key.startswith('__'):
                        continue

                    print(f"\n  ➤ Key: '{key}'")
                    print(f"    • Type: {type(value)}")

                    # check if it's a numpy array (most data is)
                    if isinstance(value, np.ndarray):
                        print(f"    • Shape: {value.shape}")
                        
                        # check if it's a MATLAB Struct
                        # This will show sub-fields like 'eeg.data', 'eeg.chan_locs'
                        if value.dtype.names:
                            print(f"    • Struct Fields: {value.dtype.names}")

                print("-" * (len(filename) + 6))

            except NotImplementedError:
                # If scipy fails, try h5py (for v7.3 files)
                print("  [Type: HDF5 (v7.3) MATLAB file (loaded with h5py)]")
                try:
                    with h5py.File(filepath, 'r') as f:
                        for key in f.keys():
                            print(f"\n  ➤ Key: '{key}'")
                            data_obj = f[key]
                            print(f"    • Type: {type(data_obj)}")
                            if hasattr(data_obj, 'shape'):
                                print(f"    • Shape: {data_obj.shape}")
                    print("-" * (len(filename) + 6))
                except Exception as e_h5:
                    print(f"  *** ERROR: Could not read HDF5 file. {e_h5} ***")
            
            except Exception as e:
                # Handle other loading errors
                print(f"  *** ERROR: Could not load file. {e} ***")

    if not found_files:
        print("No .mat files found in this directory.")

if __name__ == "__main__":
    scan_mat_files()