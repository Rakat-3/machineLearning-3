import torch
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, precision_score, f1_score
from torch.utils.data import DataLoader
import pyarrow.parquet as pq
import ast
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from movement_model import SonarTransformer, SonarDataset

def deep_analysis():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = "data/processed/sonar_data_unified.parquet"
    weights_path = 'models/movement_detector.pth'
    params_path = 'models/best_params.txt'
    scaler_path = 'models/scaler.joblib'

    # 1. Load Everything
    with open(params_path, "r") as f:
        params = ast.literal_eval(f.read())
    
    model = SonarTransformer(
        patch_size=params['patch_size'],
        d_model=params['d_model'],
        nhead=params['nhead'],
        num_layers=params['num_layers']
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    scaler = joblib.load(scaler_path)
    
    # 2. Load Full Dataset for slicing
    table = pq.read_table(data_path)
    df = table.to_pandas()
    
    signal_cols = [c for c in df.columns if str(c).isdigit()]
    signal_cols = sorted(signal_cols, key=lambda x: int(x))
    
    X = df[signal_cols[-50000:]].values
    y = df['label'].values
    
    # Metadata for slicing
    df['distance'] = df['source_file'].str.extract(r'(\d+m)')
    df['distance'] = df['distance'].fillna('Unknown')
    
    # 3. Batch Evaluation
    X_scaled = scaler.transform(X)
    dataset = SonarDataset(X_scaled, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    predictions = []
    probabilities = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probabilities.extend(outputs.cpu().numpy())
            predictions.extend((outputs > 0.5).cpu().numpy())
    
    df['pred'] = predictions
    df['prob'] = probabilities
    df['is_correct'] = (df['pred'] == df['label'])

    # --- ANALYSIS 1: By Movement Type ---
    print("\n[Analysis 1] Performance by Direction:")
    type_stats = df.groupby('movement_type').agg({
        'is_correct': 'mean',
        'prob': ['mean', 'std']
    })
    type_stats.columns = ['Accuracy', 'Confidence_Mean', 'Confidence_Std']
    print(type_stats)

    # --- ANALYSIS 2: By Distance ---
    print("\n[Analysis 2] Performance by Distance:")
    dist_stats = df.groupby('distance').agg({
        'is_correct': 'mean',
        'prob': 'mean'
    })
    dist_stats.columns = ['Accuracy', 'Avg_Probability']
    print(dist_stats)

    # --- ANALYSIS 3: Edge Case Discovery (High Confidence Mistakes) ---
    print("\n[Analysis 3] High-Confidence Mistakes (Edge Cases):")
    # Model was sure (prob > 0.9 or < 0.1) but wrong
    edge_cases = df[((df['prob'] > 0.9) | (df['prob'] < 0.1)) & (df['is_correct'] == False)]
    print(f"Found {len(edge_cases)} instances where model was very confident but wrong.")
    if not edge_cases.empty:
        print(edge_cases[['movement_type', 'distance', 'label', 'prob']].head())

    # --- VISUALIZATION: Accuracy Over Distance ---
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='distance', y='is_correct', hue='movement_type', palette='viridis')
    plt.axhline(0.95, color='red', linestyle='--', label='95% Threshold')
    plt.title('Accuracy Consistency Across Distances & Types')
    plt.ylim(0.9, 1.01)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('analysis/detailed_accuracy_analysis.png')
    print("\nAnalysis chart saved to: analysis/detailed_accuracy_analysis.png")

if __name__ == "__main__":
    deep_analysis()
