import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq
from scipy import signal
import os

def generate_extra_visuals():
    print("Generating Extra Dataset Visualizations...")
    data_path = "data/processed/sonar_data_unified.parquet"
    parquet_file = pq.ParquetFile(data_path)
    
    # 1. Load a larger sample for distribution analysis
    df = parquet_file.read_row_group(0).to_pandas() # Get a batch
    sig_cols = sorted([c for c in df.columns if str(c).isdigit()], key=int)[-50000:]
    
    # --- VISUAL 1: Distance Decay Analysis ---
    # Show how signal strength changes with distance (Moving objects)
    print("Plotting Distance Decay...")
    plt.figure(figsize=(12, 6))
    
    dist_samples = {}
    # Search for specific distances in metadata
    for i in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(i).to_pandas()
        for d in ['70m', '100m', '170m']:
            if d not in dist_samples:
                match = batch[batch['source_file'].str.contains(d, na=False)]
                if not match.empty:
                    dist_samples[d] = match.iloc[0][sig_cols].values.astype(float)
        if len(dist_samples) == 3: break

    colors = {'70m': '#27ae60', '100m': '#f39c12', '170m': '#c0392b'}
    for dist, sig_data in dist_samples.items():
        # Zoom into a specific window where echoes are active (e.g., points 20k to 35k)
        plt.plot(sig_data[20000:35000], label=f'Moving Object at {dist}', color=colors[dist], alpha=0.8)
    
    plt.title('Signal Signature Variation by Distance (Zoomed Echo Window)', fontsize=14)
    plt.xlabel('Sampling Points (Relative 20k-35k)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('analysis/outputs/distance_decay.png')

    # --- VISUAL 2: Spectrogram (Time-Frequency Analysis) ---
    # This is how the model "sees" frequency shifts over time
    print("Generating Spectrogram...")
    sample_sig = dist_samples.get('70m', df.iloc[0][sig_cols].values.astype(float))
    
    f, t, Sxx = signal.spectrogram(sample_sig, fs=50000) # Assuming 50kHz
    
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='magma')
    plt.title('Sonar Signal Spectrogram (Time-Frequency Representation)', fontsize=14)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Intensity [dB]')
    plt.savefig('analysis/outputs/spectrogram.png')

    # --- VISUAL 3: Energy Distribution by Movement Type ---
    # Statistical spread of signal power
    print("Analyzing Energy Distribution...")
    # Load rows from multiple groups to get variety
    rows = []
    for i in range(min(5, parquet_file.num_row_groups)):
        rows.append(parquet_file.read_row_group(i).to_pandas())
    df_dist = pd.concat(rows)
    
    # Calculate Energy (Sum of Squares)
    df_dist['signal_energy'] = np.sum(df_dist[sig_cols].values**2, axis=1)
    df_dist['signal_std'] = np.std(df_dist[sig_cols].values, axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_dist, x='movement_type', y='signal_std', palette='Set2')
    plt.title('Signal Variance (Std Dev) Spread by Movement Category', fontsize=14)
    plt.ylabel('Standard Deviation')
    plt.xlabel('Category')
    plt.savefig('analysis/outputs/energy_distribution.png')

    print("\nExtra visuals saved in analysis/outputs/")

if __name__ == "__main__":
    generate_extra_visuals()
