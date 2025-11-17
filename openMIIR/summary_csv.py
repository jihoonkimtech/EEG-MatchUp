import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob 

ANALYSIS_FOLDER = 'analyze'


def create_average_comparison_plot():
    
    # --- 1. 경로 설정 및 모든 CSV 파일 로드 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_dir = os.path.join(script_dir, ANALYSIS_FOLDER)

    # find all subject-specific csv files
    csv_pattern = os.path.join(analysis_dir, '*-all_metrics.csv')
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"Error: No '*-asymmetry_metrics.csv' files found in {analysis_dir}")
        print("    (Run analyzer.py for at least one subject first)")
        return

    # load all csvs into one dataframe
    all_dfs = []
    subjects_found = []
    for f in csv_files:
        try:
            subject_name = os.path.basename(f).split('-')[0]
            subjects_found.append(subject_name)
            
            df = pd.read_csv(f)
            df['subject'] = subject_name 
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}. Error: {e}")

    if not all_dfs:
        print("Error: No valid data found in CSV files.")
        return

    # (A) combine all dataframes
    all_data_df = pd.concat(all_dfs, ignore_index=True)
    num_subjects = len(set(subjects_found))
    print(f"Loaded {len(all_data_df)} total song entries from {len(csv_files)} files ({num_subjects} subjects).")


    # --- 2. 음악별 통계 계산 ---
    print("Calculating averages by song name (across all subjects)...")
    
    metric_columns = ['FAA', 'FAP', 'OASI', 'TBR']
    
    # (B) group by song name, calculate mean for all 4 metrics
    summary_df = all_data_df.groupby('song_name')[metric_columns].mean()
    
    # (C) sort alphabetically by song name
    summary_df = summary_df.sort_index()

    # (D) save the summary csv
    summary_csv_file = os.path.join(analysis_dir, 'ALL_SUBJECTS_summary_metrics.csv')
    summary_df.to_csv(summary_csv_file)
    print(f"All-subject summary CSV saved to: {summary_csv_file}")
    print(summary_df) 

    
    # --- 3. [PLOT 1] 비대칭 (FAA, OAA) 플롯 생성 ---
    print("Generating Asymmetry (FAA, OAA) plot...")
    try:
        plot_metrics_asymm = ['FAA']
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # (A) plot only FAA and OAA
        summary_df[plot_metrics_asymm].plot(kind='bar', ax=ax, width=0.8)
        
        # (B) find min/max for label styling (only for these 2 metrics)
        min_vals = summary_df[plot_metrics_asymm].min()
        max_vals = summary_df[plot_metrics_asymm].max()

        # (C) iterate containers (FAA, OAA)
        for i, container in enumerate(ax.containers):
            metric_name = plot_metrics_asymm[i]
            min_val = min_vals[metric_name]
            max_val = max_vals[metric_name]

            # (D) apply styling (copied from user)
            for bar in container:
                height = bar.get_height()
                if pd.isna(height): continue
                label_text = f"{height:.3f}"
                label_color = 'black'
                label_weight = 'normal'

                if np.isclose(height, max_val):
                    label_color = 'green'
                    label_weight = 'bold'
                elif np.isclose(height, min_val):
                    label_color = 'red'
                    label_weight = 'bold'

                v_offset = 3 if height >= 0 else -3
                vertical_align = 'bottom' if height >= 0 else 'top'

                ax.annotate(
                    label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, v_offset), textcoords="offset points",
                    ha='center', va=vertical_align,
                    color=label_color, weight=label_weight, fontsize=8 
                )
        
        # (E) set title and labels
        plt.title(f'EEG Asymmetry Feature Comparison (All Subjects, n={num_subjects})', fontsize=16)
        plt.ylabel('Asymmetry Value (ln(R)-ln(L))', fontsize=12)
        plt.xlabel('Song Name', fontsize=12)
        
        plt.ylim(-0.03, 0.03) 
        
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout() 
        
        output_filename = os.path.join(analysis_dir, "ALL_SUBJECTS_PLOT_1_Asymmetry.png")
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"Asymmetry summary plot saved to: {output_filename}")
        
    except Exception as e:
        print(f"    Warning: Could not generate Asymmetry plot. Error: {e}")

        
    # --- 4. [PLOT 2] 로그 파워 (FAP, OASI) 플롯 생성 ---
    print("Generating Log Power (FAP, OASI) plot...")
    try:
        plot_metrics_power = ['FAP', 'OASI']
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # (A) plot only FAP and OASI
        summary_df[plot_metrics_power].plot(kind='bar', ax=ax, width=0.8)
        
        # (B) find min/max for label styling (only for these 2 metrics)
        min_vals = summary_df[plot_metrics_power].min()
        max_vals = summary_df[plot_metrics_power].max()

        # (C) iterate containers (FAP, OASI)
        for i, container in enumerate(ax.containers):
            metric_name = plot_metrics_power[i]
            min_val = min_vals[metric_name]
            max_val = max_vals[metric_name]

            # (D) apply styling (copied from user)
            for bar in container:
                height = bar.get_height()
                if pd.isna(height): continue
                label_text = f"{height:.3f}"
                label_color = 'black'
                label_weight = 'normal'

                if np.isclose(height, max_val):
                    label_color = 'green'
                    label_weight = 'bold'
                elif np.isclose(height, min_val):
                    label_color = 'red'
                    label_weight = 'bold'

                v_offset = 3 # all values are negative, so offset is fixed
                vertical_align = 'bottom' # 

                ax.annotate(
                    label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, v_offset), textcoords="offset points",
                    ha='center', va=vertical_align,
                    color=label_color, weight=label_weight, fontsize=8 
                )
        
        # (E) set title and labels
        plt.title(f'EEG Log Power Feature Comparison (All Subjects, n={num_subjects})', fontsize=16)
        plt.ylabel('Avg Log Power ((ln(R)+ln(L))/2)', fontsize=12)
        plt.xlabel('Song Name', fontsize=12)
        
        plt.ylim(-22.5, -23.5) 
        
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout() 
        
        output_filename = os.path.join(analysis_dir, "ALL_SUBJECTS_PLOT_2_LogPower.png")
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"Log Power summary plot saved to: {output_filename}")
        
    except Exception as e:
        print(f"    Warning: Could not generate Log Power plot. Error: {e}")

    # TBR
    print("Generating Log Power (FAP, OASI) plot...")
    try:
        plot_metrics_power = ['TBR']
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # (A) plot only FAP and OASI
        summary_df[plot_metrics_power].plot(kind='bar', ax=ax, width=0.8)
        
        # (B) find min/max for label styling (only for these 2 metrics)
        min_vals = summary_df[plot_metrics_power].min()
        max_vals = summary_df[plot_metrics_power].max()

        # (C) iterate containers (FAP, OASI)
        for i, container in enumerate(ax.containers):
            metric_name = plot_metrics_power[i]
            min_val = min_vals[metric_name]
            max_val = max_vals[metric_name]

            # (D) apply styling (copied from user)
            for bar in container:
                height = bar.get_height()
                if pd.isna(height): continue
                label_text = f"{height:.3f}"
                label_color = 'black'
                label_weight = 'normal'

                if np.isclose(height, max_val):
                    label_color = 'green'
                    label_weight = 'bold'
                elif np.isclose(height, min_val):
                    label_color = 'red'
                    label_weight = 'bold'

                v_offset = 3 # all values are negative, so offset is fixed
                vertical_align = 'bottom' # 

                ax.annotate(
                    label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, v_offset), textcoords="offset points",
                    ha='center', va=vertical_align,
                    color=label_color, weight=label_weight, fontsize=8 
                )
        
        # (E) set title and labels
        plt.title(f'EEG TBR Feature Comparison (All Subjects, n={num_subjects})', fontsize=16)
        plt.ylabel('Avg TBR', fontsize=12)
        plt.xlabel('Song Name', fontsize=12)
        
        plt.ylim(-0.02, 0.05) 
        
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout() 
        
        output_filename = os.path.join(analysis_dir, "ALL_SUBJECTS_PLOT_3_TBR.png")
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"Log Power summary plot saved to: {output_filename}")
        
    except Exception as e:
        print(f"    Warning: Could not generate Log Power plot. Error: {e}")


# run the function
if __name__ == "__main__":
    create_average_comparison_plot()