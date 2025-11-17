import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# EEG Feature Comparator
# Reads all 'EEG_features_...csv' files in the directory
# Creates a comparison plot AND saves a combined CSV.

def create_comparison_plot():
    print("--- Starting Feature Comparison ---")
    
    # find all feature CSV files in the current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = glob.glob(os.path.join(script_dir, "EEG_features_*.csv"))
    
    if not csv_files:
        print("*** ERROR: No 'EEG_features_...csv' files found. ***")
        print("Please run analyzer.py for each testcase first.")
        return

    print(f"Found {len(csv_files)} files to compare:")
    
    all_data = []
    
    # read each CSV and combine them
    for f in csv_files:
        # extract testcase name from filename
        file_name = os.path.basename(f)
        if file_name.startswith("EEG_features_COMBINED"):
            continue
            
        testcase_name = file_name.replace("EEG_features_", "").replace(".csv", "")
        
        print(f"  - Reading '{file_name}' (testcase: {testcase_name})")
        
        df = pd.read_csv(f)
        df['testcase'] = testcase_name
        all_data.append(df)

    # combine all dataframes into one
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.set_index('testcase', inplace=True)
    
    # ensure correct order (optional, but good for display)
    desired_order = ['base', 'info', 'challenge', 'highlight', '120s']
    available_testcases = [c for c in desired_order if c in combined_df.index]
    combined_df = combined_df.loc[available_testcases]
    
    print("\n--- Combined Data ---")
    print(combined_df)
    print("---------------------\n")

    # Save combined data to CSV
    output_csv_filename = os.path.join(script_dir, "EEG_features_COMBINED_data.csv")
    try:
        combined_df.to_csv(output_csv_filename)
        print(f"Combined data saved to: {output_csv_filename}\n")
    except Exception as e:
        print(f"*** ERROR: Could not save combined CSV file. ***\n{e}\n")
    # End of new section

    # Create the grouped bar plot
    ax = combined_df.plot(
        kind='bar', 
        figsize=(12, 7), 
        width=0.8, 
        edgecolor='black'
    )
    
    # get min/max values for each metric
    min_vals = combined_df.min()
    max_vals = combined_df.max()
    metrics = combined_df.columns # e.g., ['FAA', 'OAA', 'OASI']

    # iterate through each metric's container
    for i, container in enumerate(ax.containers):
        metric_name = metrics[i]
        min_val = min_vals[metric_name]
        max_val = max_vals[metric_name]

        # iterate through each bar in that container
        for bar in container:
            height = bar.get_height()
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
    
    plt.title('EEG Feature Comparison by testcase', fontsize=16)
    plt.ylabel('Value', fontsize=12)
    plt.xlabel('testcase', fontsize=12)
    
    # set fixed Y-axis (slightly taller for labels)
    plt.ylim(-0.5, 11.5)
    
    # rotate x-axis labels to be horizontal
    plt.xticks(rotation=0, fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout() 
    
    # save the final comparison image
    output_filename = os.path.join(script_dir, "EEG_features_COMPARISON.png")
    plt.savefig(output_filename)
    
    print(f"Comparison plot saved to: {output_filename}")
    plt.show() 

# run the function
if __name__ == "__main__":
    create_comparison_plot()