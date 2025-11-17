import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# EEG Feature Comparator (FOR AVERAGES)
# Reads all '..._ALL_with_AVG.csv' files in the directory.
# Finds the 'AVERAGE' row in each file.
# Creates a comparison plot of these averages.

def create_average_comparison_plot():
    print("--- Starting Average Feature Comparison ---")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    smry_folder = os.path.join(script_dir, "./analyze/Summary/")
    
    search_pattern = os.path.join(smry_folder, "EEG_features_*_ALL_with_AVG.csv")
    csv_files = glob.glob(search_pattern)
    # --------------------
    
    if not csv_files:
        print(f"*** ERROR: No '..._ALL_with_AVG.csv' files found in directory. ***")
        print(f"Search pattern was: {search_pattern}")
        return

    print(f"Found {len(csv_files)} files to compare:")
    
    all_average_data = []
    
    # read each CSV and combine them
    for f_path in csv_files:
        
        file_name = os.path.basename(f_path) 
            
        # --- [MODIFIED] ---
        # Extract the test case group name, e.g., "G1-TrainSim"
        # from "EEG_features_G1-TrainSim_ALL_with_AVG.csv"
        try:
            testcase_group_name = file_name.replace("EEG_features_", "").replace("_ALL_with_AVG.csv", "")
        except Exception:
            # Fallback if parsing fails
            testcase_group_name = file_name.replace(".csv", "")
        # --------------------

        try:
            # read the full CSV (which includes individual and average rows)
            df_full = pd.read_csv(f_path)
            
            # find the 'AVERAGE' row
            df_avg = df_full[df_full['testcase'].str.startswith('AVERAGE', na=False)]
            
            if df_avg.empty:
                print(f"  - WARNING: No 'AVERAGE' row found in {file_name}. Skipping.")
                continue
                
            # take the first average row found and make a copy
            avg_data_row = df_avg.head(1).copy()
            
            # set the 'testcase' index to our group name (e.g., 'G1-TrainSim')
            avg_data_row['testcase'] = testcase_group_name
            
            all_average_data.append(avg_data_row)
            print(f"  - Read AVERAGE data from '{file_name}' (Group: {testcase_group_name})")

        except Exception as e:
            print(f"  - *** ERROR reading {file_name}: {e} ***")


    if not all_average_data:
        print("\n*** ERROR: No average data could be extracted from any files. Exiting. ***")
        return

    # combine all average dataframes into one
    combined_df = pd.concat(all_average_data, ignore_index=True)
    combined_df.set_index('testcase', inplace=True)
    
    # --- [MODIFIED] ---
    # Re-enable sorting with the *correct* names from the image.
    # This fixes the source of the previous IndexError.
    desired_order = ['G1-TrainSim', 'G2-Adventure', 'G3-Horror', 'G4-Control']
    
    # Get the testcases we found, in the desired order
    available_testcases = [c for c in desired_order if c in combined_df.index]
    
    # Get any *other* testcases found that weren't in the list
    other_testcases = [c for c in combined_df.index if c not in available_testcases]
    
    # Combine them: ordered first, then any others at the end
    final_order = available_testcases + other_testcases
    
    combined_df = combined_df.loc[final_order]
    # ------------------
    
    print("\n--- Combined Average Data ---")
    print(combined_df)
    print("-----------------------------\n")
    
    # Check if dataframe is empty *after* all processing, before plotting
    if combined_df.empty:
        print("*** ERROR: Combined DataFrame is empty. Nothing to plot. ***")
        print("This might happen if 'desired_order' list doesn't match file names.")
        return

    # Save combined data to CSV
    output_csv_filename = os.path.join(smry_folder, "EEG_features_AVERAGE_COMPARISON.csv")
    try:
        combined_df.to_csv(output_csv_filename)
        print(f"Combined average data saved to: {output_csv_filename}\n")
    except Exception as e:
        print(f"*** ERROR: Could not save combined CSV file. ***\n{e}\n")


    # Create the grouped bar plot
    ax = combined_df.plot(
        kind='bar', 
        figsize=(12, 7), 
        width=0.8, 
        edgecolor='black'
    )
    
    # get min/max values for each metric
    metric_columns = [col for col in ['FAA', 'FAP', 'OASI', 'TBR'] if col in combined_df.columns]
    
    ax = combined_df[metric_columns].plot(
        kind='bar', 
        figsize=(12, 7), 
        width=0.8, 
        edgecolor='black'
    )

    min_vals = combined_df[metric_columns].min()
    max_vals = combined_df[metric_columns].max()

    # iterate through each metric's container
    for i, container in enumerate(ax.containers):
        metric_name = metric_columns[i]
        min_val = min_vals[metric_name]
        max_val = max_vals[metric_name]

        # iterate through each bar in that container
        for bar in container:
            height = bar.get_height()
            
            if pd.isna(height):
                continue
                
            label_text = f"{height:.4f}"
            
            # default style
            label_color = 'black'
            label_weight = 'normal'

            # check for max/min *for this metric*
            if np.isclose(height, max_val):
                label_color = 'green'
                label_weight = 'bold'
            elif np.isclose(height, min_val):
                label_color = 'red'
                label_weight = 'bold'

            # set vertical offset
            v_offset = 3 if height >= 0 else -3
            vertical_align = 'bottom' if height >= 0 else 'top'

            # place the text label
            ax.annotate(
                label_text,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, v_offset), # 3 points offset
                textcoords="offset points",
                ha='center',
                va=vertical_align,
                color=label_color,
                weight=label_weight,
                fontsize=8 
            )
    
    plt.title('EEG Average Feature Comparison by Test Case Group', fontsize=16)
    plt.ylabel('Average Value', fontsize=12)
    plt.xlabel('Test Case Group', fontsize=12)
    
    plt.ylim(-1.0, 11.5) 
    
    # --- [MODIFIED] ---
    # Rotate labels slightly so they don't overlap
    plt.xticks(rotation=15, ha='right', fontsize=10)
    # ------------------
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout() 
    
    # save the final comparison image
    output_filename = os.path.join(smry_folder, "EEG_features_AVERAGE_COMPARISON.png")
    plt.savefig(output_filename)
    
    print(f"Average comparison plot saved to: {output_filename}")


# run the function
if __name__ == "__main__":
    create_average_comparison_plot()