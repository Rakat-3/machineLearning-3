import torch
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
import pyarrow.parquet as pq
import ast
import sys

# Ensure we can import the model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from movement_model import SonarTransformer, SonarDataset

def generate_report():
    print("--- Starting Extended Model Performance Analysis ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = "data/processed/sonar_data_unified.parquet"
    weights_path = 'models/movement_detector.pth'
    params_path = 'models/best_params.txt'
    scaler_path = 'models/scaler.joblib'
    
    output_dir = 'analysis/outputs'
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Model & Meta
    print("--- Loading model and parameters ---")
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
    
    # 2. Load and Sample Data for evaluation (1000 samples for speed/viz)
    print("--- Loading dataset ---")
    df = pq.read_table(data_path).to_pandas()
    
    # Robust sampling
    df_1 = df[df['label'] == 1].sample(n=min(len(df[df['label']==1]), 1000), random_state=42)
    df_0 = df[df['label'] == 0].sample(n=min(len(df[df['label']==0]), 1000), random_state=42)
    df_sample = pd.concat([df_1, df_0]).sample(frac=1).reset_index(drop=True)
    
    sig_cols = sorted([c for c in df_sample.columns if str(c).isdigit()], key=int)[-50000:]
    X = scaler.transform(df_sample[sig_cols].values)
    y_true = df_sample['label'].values
    
    print("--- Running inference ---")
    dataset = SonarDataset(X, y_true)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    probs = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            out = model(batch_x)
            probs.extend(out.cpu().numpy().flatten())
    
    y_prob = np.array(probs)
    y_pred = (y_prob > 0.5).astype(int)
    
    # --- PLOT 1: Confusion Matrix ---
    print("--- Generating Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Static', 'Moving'], yticklabels=['Static', 'Moving'])
    plt.title('Confusion Matrix: Movement Detection', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{output_dir}/confusion_matrix.png')

    # --- PLOT 2: Performance Metrics (P, R, F1, Accuracy) ---
    print("--- Generating Metrics Comparison ---")
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    acc = (y_pred == y_true).mean()
    
    metrics = {
        'Precision': p * 100,
        'Recall': r * 100,
        'F1-Score': f1 * 100,
        'Accuracy': acc * 100
    }
    
    plt.figure(figsize=(10, 6))
    colors = ['#2ECC71', '#E67E22', '#9B59B6', '#3498DB']
    bars = plt.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.85, width=0.6)
    plt.ylim(90, 102)
    plt.title('Core Performance Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Score (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.2f}%', ha='center', fontweight='bold')
    
    plt.savefig(f'{output_dir}/metrics_summary.png')

    # --- PLOT 3: "Performance over Time" (Proxy: Distance Trend) ---
    print("--- Generating Time/Distance Trend ---")
    df_sample['correct'] = (y_pred == y_true)
    
    # Improved regex to find distance (e.g., 100m, 170m)
    def get_dist(fname):
        import re
        m = re.search(r'(\d+m)', str(fname))
        if m: return int(m.group(1).replace('m', ''))
        return 0
        
    df_sample['dist_val'] = df_sample['source_file'].apply(get_dist)
    dist_perf = df_sample.groupby('dist_val')['correct'].mean() * 100
    # Remove 0-distance (failures)
    dist_perf = dist_perf[dist_perf.index > 0]
    
    plt.figure(figsize=(10, 6))
    plt.plot(dist_perf.index, dist_perf.values, marker='s', markersize=8, color='#E74C3C', lw=3, label='Detection Accuracy')
    plt.fill_between(dist_perf.index, dist_perf.values - 1, dist_perf.values + 1, color='#E74C3C', alpha=0.1)
    plt.title('Model Performance vs. Object Distance (Pulse Time)', fontsize=14, fontweight='bold')
    plt.xlabel('Distance (Meters) - Correlates with Signal Time-of-Flight')
    plt.ylabel('Accuracy (%)')
    plt.ylim(90, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{output_dir}/performance_vs_time.png')

    # --- PLOT 4: Confidence Spread (Violin Plot) ---
    print("--- Generating Confidence Distribution ---")
    plt.figure(figsize=(10, 6))
    eval_df = pd.DataFrame({'Target': y_true, 'Confidence': y_prob})
    eval_df['Target'] = eval_df['Target'].map({1: 'Moving', 0: 'Static'})
    
    sns.violinplot(data=eval_df, x='Target', y='Confidence', palette='muted', inner='quartile')
    plt.title('Model Confidence Distribution by Class', fontsize=14, fontweight='bold')
    plt.savefig(f'{output_dir}/confidence_spread.png')

    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    generate_report()
