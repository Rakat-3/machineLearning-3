import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_final_metrics():
    print("Generating Final Metric Visuals...")
    output_dir = 'analysis/outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Data from evaluation results
    data = {
        'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
        'Score': [99.3, 99.9, 99.6, 99.6]
    }
    df = pd.DataFrame(data)
    
    # --- PLOT 1: Performance Bar Chart ---
    plt.figure(figsize=(10, 6))
    custom_palette = ['#2ECC71', '#3498DB', '#9B59B6', '#F1C40F']
    ax = sns.barplot(x='Metric', y='Score', data=df, palette=custom_palette, hue='Metric', legend=False)
    plt.ylim(95, 100.5)
    plt.title('Final Model Performance Verification', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Score (%)', fontsize=12)
    plt.xlabel('Evaluation Category', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points',
                       fontsize=12, fontweight='bold')
                       
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_metrics.png')

    # --- PLOT 2: Movement Type Breakdown (P/R/F1 per group) ---
    # Representative data based on deep_analysis.py output
    groups = ['Horizontal', 'Perpendicular', 'Steady']
    # Accuracy per group
    accs = [99.8, 99.1, 99.9]
    f1s = [99.7, 98.9, 99.9]
    
    x = np.arange(len(groups))
    width = 0.35
    
    fig, ax2 = plt.subplots(figsize=(10, 6))
    rects1 = ax2.bar(x - width/2, accs, width, label='Accuracy', color='#3498DB', alpha=0.8)
    rects2 = ax2.bar(x + width/2, f1s, width, label='F1-Score', color='#E67E22', alpha=0.8)
    
    ax2.set_ylabel('Score (%)')
    ax2.set_title('Performance Consistency by Movement Type', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups)
    ax2.set_ylim(95, 100.5)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/movement_performance.png')
    
    print(f"Visuals saved to {output_dir}")

if __name__ == "__main__":
    plot_final_metrics()
