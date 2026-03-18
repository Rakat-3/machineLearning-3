import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
import os

def generate_energy_plot():
    print("Generating Energy Distribution plot...")
    data_path = "data/processed/sonar_data_unified.parquet"
    parquet_file = pq.ParquetFile(data_path)
    
    # Identify signal columns from first batch
    batch = parquet_file.read_row_group(0).to_pandas()
    sig_cols = sorted([c for c in batch.columns if str(c).isdigit()], key=int)[-50000:]
    
    # Collect stats for a variety of rows
    all_stats = []
    # Sampling 100 rows from different row groups
    for i in range(min(10, parquet_file.num_row_groups)):
        batch = parquet_file.read_row_group(i).to_pandas()
        # Randomly sample 20 rows from this batch if possible
        n = min(len(batch), 20)
        sample = batch.sample(n)
        
        # Calculate stats for these rows
        sigs = sample[sig_cols].values.astype(float)
        stds = np.std(sigs, axis=1)
        means = np.mean(sigs, axis=1)
        
        for idx in range(n):
            all_stats.append({
                'movement_type': sample.iloc[idx]['movement_type'],
                'label': sample.iloc[idx]['label'],
                'std': stds[idx],
                'mean': means[idx]
            })
            
    df_stats = pd.DataFrame(all_stats)
    
    plt.figure(figsize=(10, 6))
    # Use hue for movement_type and x for simple categories
    sns.boxplot(data=df_stats, x='movement_type', y='std', palette='Set2', hue='movement_type', legend=False)
    plt.title('Signal Standard Deviation (Noise Level) by Category', fontsize=14)
    plt.ylabel('Standard Deviation')
    plt.xlabel('Movement Type')
    plt.grid(True, axis='y', alpha=0.3)
    
    os.makedirs('analysis/outputs', exist_ok=True)
    plt.savefig('analysis/outputs/energy_distribution.png')
    print("Plot saved to analysis/outputs/energy_distribution.png")

if __name__ == "__main__":
    generate_energy_plot()
