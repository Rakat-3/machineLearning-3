import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Load Data
# Main LLM Method (Already measured)
llm_acc = 99.6
llm_latency = 5.2 # ms

# Traditional Transformer (Just measured)
with open('transformer_comparison/comparison_results.txt', 'r') as f:
    data = f.read().split(',')
    trad_acc = float(data[0])
    trad_latency = float(data[2])

# 2. Data Preparation
labels = ['LLM Method (PatchTST)', 'Traditional Transformer']
accuracies = [llm_acc, trad_acc]
latencies = [llm_latency, trad_latency]

# Create Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: Accuracy ---
colors = ['#4CAF50', '#FF5722']
bars1 = ax1.bar(labels, accuracies, color=colors, alpha=0.8, width=0.6)
ax1.set_title('Detection Accuracy (%)', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 110)
ax1.set_ylabel('Accuracy (%)')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# --- Plot 2: Latency (Lower is Better) ---
bars2 = ax2.bar(labels, latencies, color=['#2196F3', '#9C27B0'], alpha=0.8, width=0.6)
ax2.set_title('Inference Latency (Milliseconds)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Latency (ms)')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.2f}ms', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('analysis/outputs/llm_vs_transformer_plot.png')
print("Comparison plot saved to: analysis/outputs/llm_vs_transformer_plot.png")
