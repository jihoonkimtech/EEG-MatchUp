import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob 

# [수정] 그룹 분석 결과를 담은 폴더를 바라보도록 변경
ANALYSIS_FOLDER = 'analyze' 


def create_average_comparison_plot():
    
    # --- 1. 경로 설정 및 모든 CSV 파일 로드 ---
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    analysis_dir = os.path.join(script_dir, ANALYSIS_FOLDER)

    # [수정] 그룹 CSV 파일을 찾도록 패턴 변경 (예: P01-GROUP_all_metrics.csv)
    csv_pattern = os.path.join(analysis_dir, '*-GROUP_all_metrics.csv')
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"Error: No '*-GROUP_all_metrics.csv' files found in {analysis_dir}")
        print("    (Run the GROUP-based analyzer.py for at least one subject first)")
        return

    # load all csvs into one dataframe
    all_dfs = []
    subjects_found = []
    for f in csv_files:
        try:
            # P01-GROUP_all_metrics.csv -> P01
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
    print(f"Loaded {len(all_data_df)} total group entries from {len(csv_files)} files ({num_subjects} subjects).")


    # --- 2. [수정] 그룹별 통계 계산 ---
    print("Calculating averages by group name (across all subjects)...")
    
    metric_columns = ['FAA', 'FAP', 'OASI', 'TBR']
    
    # (B) [수정] 'song_name' 대신 'group_name' 으로 그룹화
    summary_df = all_data_df.groupby('group_name')[metric_columns].mean()
    
    # (C) [수정] 'Resting'을 맨 앞으로 오도록 정렬
    all_groups = summary_df.index
    # define the desired order
    desired_order = ['Resting', 'Calm', 'Excited', 'Majestic']
    # filter desired_order to only include groups that actually exist
    final_order = [g for g in desired_order if g in all_groups]
    # add any other groups (that might exist) to the end, sorted
    final_order.extend(sorted([g for g in all_groups if g not in desired_order]))
    
    summary_df = summary_df.reindex(final_order)
    print("  Reordered groups to:", final_order)


    # (D) [수정] 요약 CSV 파일 이름 변경
    summary_csv_file = os.path.join(analysis_dir, 'ALL_SUBJECTS_GROUP_summary_metrics.csv')
    summary_df.to_csv(summary_csv_file)
    print(f"All-subject group summary CSV saved to: {summary_csv_file}")
    print(summary_df) 

    
    # --- 3. [PLOT 1] [수정] 비대칭 (FAA) 및 TBR 통합 플롯 ---
    print("Generating Asymmetry (FAA) & TBR plot...")
    try:
        # [수정] FAA와 TBR을 함께 플로팅
        plot_metrics_combo = ['FAA', 'TBR']
        
        # [수정] 그룹 4개만 표시하므로 Figure 크기 축소
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # (A) plot only FAA and TBR
        summary_df[plot_metrics_combo].plot(kind='bar', ax=ax, width=0.8)
        
        # (B) find min/max for label styling
        min_vals = summary_df[plot_metrics_combo].min()
        max_vals = summary_df[plot_metrics_combo].max()

        # (C) iterate containers (FAA, TBR)
        for i, container in enumerate(ax.containers):
            metric_name = plot_metrics_combo[i]
            min_val = min_vals[metric_name]
            max_val = max_vals[metric_name]

            # (D) apply styling
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
                    color=label_color, weight=label_weight, fontsize=9
                )
        
        # (E) set title and labels
        # [수정] 제목/레이블 변경
        plt.title(f'EEG Asymmetry (FAA) & TBR Comparison (All Subjects, n={num_subjects}) - By Group', fontsize=16)
        plt.ylabel('Metric Value', fontsize=12)
        plt.xlabel('Emotion Group', fontsize=12)
        
        plt.xticks(rotation=0, ha='center', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.axhline(0, color='black', linewidth=0.5) # add zero line
        plt.tight_layout() 
        
        # [수정] 출력 파일명 변경
        output_filename = os.path.join(analysis_dir, "ALL_SUBJECTS_PLOT_1_GROUP_FAA_TBR.png")
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"FAA & TBR summary plot saved to: {output_filename}")
        
    except Exception as e:
        print(f"    Warning: Could not generate FAA/TBR plot. Error: {e}")

        
    # --- 4. [PLOT 2] 로그 파워 (FAP, OASI) 플롯 생성 ---
    print("Generating Log Power (FAP, OASI) plot...")
    try:
        plot_metrics_power = ['FAP', 'OASI']
        
        # [수정] Figure 크기 축소
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # (A) plot only FAP and OASI
        summary_df[plot_metrics_power].plot(kind='bar', ax=ax, width=0.8)
        
        # (B) find min/max for label styling
        min_vals = summary_df[plot_metrics_power].min()
        max_vals = summary_df[plot_metrics_power].max()

        # (C) iterate containers (FAP, OASI)
        for i, container in enumerate(ax.containers):
            metric_name = plot_metrics_power[i]
            min_val = min_vals[metric_name]
            max_val = max_vals[metric_name]

            # (D) apply styling
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
                    color=label_color, weight=label_weight, fontsize=9
                )
        
        # (E) set title and labels
        plt.title(f'EEG Log Power Feature Comparison (All Subjects, n={num_subjects}) - By Group', fontsize=16)
        plt.ylabel('Avg Log Power ((ln(R)+ln(L))/2)', fontsize=12)
        plt.xlabel('Emotion Group', fontsize=12)
        
        plt.xticks(rotation=0, ha='center', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Metrics', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout() 
        
        # [수정] 출력 파일명 변경 (넘버링 유지)
        output_filename = os.path.join(analysis_dir, "ALL_SUBJECTS_PLOT_2_GROUP_LogPower.png")
        plt.savefig(output_filename)
        plt.close(fig)
        print(f"Log Power summary plot saved to: {output_filename}")
        
    except Exception as e:
        print(f"    Warning: Could not generate Log Power plot. Error: {e}")

    # --- 5. [삭제] PLOT 3 (TBR 단독)은 PLOT 1로 통합되어 삭제 ---
    print("TBR plot is merged into PLOT 1.")


# run the function
if __name__ == "__main__":
    create_average_comparison_plot()