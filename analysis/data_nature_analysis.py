import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
from scipy.fft import fft, fftfreq
import os

def analyze_data_nature():
    print("Starting Deep Data Analysis...")
    data_path = "data/processed/sonar_data_unified.parquet"
    
    # 1. Smart Sampling: Find one of each class
    parquet_file = pq.ParquetFile(data_path)
    
    moving_row = None
    static_row = None
    
    # Iterate through row groups to find examples
    for i in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(i).to_pandas()
        if moving_row is None and not batch[batch['label'] == 1].empty:
            moving_row = batch[batch['label'] == 1].iloc[0]
            print(f"Found Moving sample in row group {i}")
        if static_row is None and not batch[batch['label'] == 0].empty:
            static_row = batch[batch['label'] == 0].iloc[0]
            print(f"Found Stationary sample in row group {i}")
        if moving_row is not None and static_row is not None:
            break
            
    # Identify signal columns
    sig_cols = sorted([c for c in batch.columns if str(c).isdigit()], key=int)
    sig_cols = sig_cols[-50000:] # Make sure we have the 50k points

    # --- ANALYSIS 1: Nature of the Data (Stats of the two types) ---
    print("\n--- [1] Nature Comparison ---")
    m_sig = moving_row[sig_cols].values.astype(float)
    s_sig = static_row[sig_cols].values.astype(float)
    
    print(f"Moving Signal - Mean: {np.mean(m_sig):.4f}, Std: {np.std(m_sig):.4f}, Energy: {np.sum(m_sig**2):.4f}")
    print(f"Stationary Signal - Mean: {np.mean(s_sig):.4f}, Std: {np.std(s_sig):.4f}, Energy: {np.sum(s_sig**2):.4f}")

    # --- ANALYSIS 3: Signal Visualization ---
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(m_sig, label='Moving Object (Sonar Pulse)', color='#2e59a8', alpha=0.8)
    plt.title('Nature: Sonar Reflection from Moving Object', fontsize=12, fontweight='bold')
    plt.ylabel('Signal Intensity')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(s_sig, label='Stationary (Background)', color='#7f8c8d', alpha=0.8)
    plt.title('Nature: Background Sensor Noise (No Object)', fontsize=12, fontweight='bold')
    plt.ylabel('Signal Intensity')
    plt.xlabel('Time Points (1 to 50,000)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/outputs/signal_nature.png')

    # --- ANALYSIS 4: Autocorrelation (Justification for Transformer) ---
    # Shows long-term patterns vs random noise
    def autocorr(x, lags=2000):
        x = x - np.mean(x)
        return [1.0] + [np.corrcoef(x[i:], x[:-i])[0, 1] for i in range(1, lags)]

    print("\nCalculating Autocorrelation (Justifying Transformer)...")
    plt.figure(figsize=(10, 5))
    lags = 1000
    m_corr = autocorr(m_sig, lags)
    s_corr = autocorr(s_sig, lags)
    
    plt.plot(m_corr, color='#2e59a8', label='Moving (Patterned)')
    plt.plot(s_corr, color='#7f8c8d', label='Stationary (Random)', alpha=0.6)
    plt.title('Data Dependency Analysis: Autocorrelation', fontsize=12, fontweight='bold')
    plt.xlabel('Lag (Distance between points)')
    plt.ylabel('Correlation Strength')
    plt.legend()
    plt.savefig('analysis/outputs/data_dependency.png')
    
    print("\nAnalysis complete. Visuals saved to analysis/outputs/")

if __name__ == "__main__":
    analyze_data_nature()
