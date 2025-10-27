#!/usr/bin/env python3
"""
SmolVLA Dataset Test - With Visualization (English Version)
Outputs:
1. Inference performance bar chart
2. Action trajectory comparison
3. State and action distribution
"""

import sys
sys.path.insert(0, '/root/smolvla_project/lerobot/src')
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No GUI backend
import matplotlib.pyplot as plt
from pathlib import Path
import pyarrow.parquet as pq
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
import time

# Use default English fonts
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("\n" + "="*60)
print("SmolVLA Visualization Test")
print("="*60 + "\n")

device = "cuda"

# 1. Load data
print("[1/5] Loading data...")
df = pq.read_table(Path("/root/smolvla_project/datasets/lerobot_pusht/data/chunk-000/file-000.parquet")).to_pandas()
print(f"   OK: {len(df)} samples")

# 2. Load model
print(f"\n[2/5] Loading model...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).float().eval()
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
print(f"   OK: Model loaded")

# 3. Batch inference
print(f"\n[3/5] Batch inference (20 samples)...")
num_samples = 20
times = []
pred_actions = []
true_actions = []

for i in range(num_samples):
    # Prepare input
    state = torch.tensor(df.iloc[i]['observation.state'], dtype=torch.float32)
    if len(state) < 14:
        state = torch.cat([state, torch.zeros(14-len(state))])
    
    img = torch.rand(1,3,256,256).to(device)
    tokens = tokenizer("Push the block", return_tensors="pt")
    
    obs = {
        "observation.images.camera1": img,
        "observation.images.camera2": img.clone(),
        "observation.images.camera3": img.clone(),
        "observation.state": state.unsqueeze(0).to(device),
        "observation.language.tokens": tokens['input_ids'].to(device),
        "observation.language.attention_mask": tokens['attention_mask'].to(device).bool(),
    }
    
    # Inference
    start = time.time()
    with torch.no_grad():
        action = policy.select_action(obs)
    t = (time.time()-start)*1000
    times.append(t)
    
    # Save results
    pred_actions.append(action.cpu().numpy().flatten()[:2])
    true_action = df.iloc[i]['action']
    true_actions.append(np.array(true_action))
    
    if (i+1) % 5 == 0:
        print(f"   Progress: {i+1}/{num_samples}")

print(f"   OK: Inference completed")

# 4. Create visualization
print(f"\n[4/5] Generating visualization...")

fig = plt.figure(figsize=(16, 10))

# Subplot 1: Inference time distribution
ax1 = plt.subplot(2, 3, 1)
times_array = np.array(times)
ax1.bar(range(len(times)), times_array, color='skyblue', edgecolor='navy')
ax1.axhline(y=times_array.mean(), color='r', linestyle='--', 
            label=f"Average: {times_array.mean():.1f}ms")
ax1.set_xlabel('Sample Index', fontsize=10)
ax1.set_ylabel('Inference Time (ms)', fontsize=10)
ax1.set_title('Inference Time Distribution', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Performance statistics
ax2 = plt.subplot(2, 3, 2)
stats_text = f"""Performance Statistics

Average Time: {times_array.mean():.2f} ms
Median: {np.median(times_array):.2f} ms
Min: {times_array.min():.2f} ms
Max: {times_array.max():.2f} ms
Std Dev: {times_array.std():.2f} ms

Inference Freq: {1000/times_array.mean():.2f} Hz
Total Samples: {num_samples}

GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB
"""
ax2.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax2.axis('off')

# Subplot 3: X-axis action comparison
ax3 = plt.subplot(2, 3, 3)
pred_x = [p[0] for p in pred_actions]
true_x = [t[0] for t in true_actions]
x_indices = range(len(pred_x))
ax3.plot(x_indices, true_x, 'o-', label='True Action', color='green', alpha=0.7)
ax3.plot(x_indices, pred_x, 's-', label='Predicted Action', color='red', alpha=0.7)
ax3.set_xlabel('Sample Index', fontsize=10)
ax3.set_ylabel('X-axis Action Value', fontsize=10)
ax3.set_title('X-axis Action Comparison', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: Y-axis action comparison
ax4 = plt.subplot(2, 3, 4)
pred_y = [p[1] for p in pred_actions]
true_y = [t[1] for t in true_actions]
ax4.plot(x_indices, true_y, 'o-', label='True Action', color='blue', alpha=0.7)
ax4.plot(x_indices, pred_y, 's-', label='Predicted Action', color='orange', alpha=0.7)
ax4.set_xlabel('Sample Index', fontsize=10)
ax4.set_ylabel('Y-axis Action Value', fontsize=10)
ax4.set_title('Y-axis Action Comparison', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Subplot 5: Action scatter plot
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(true_x, true_y, c='green', label='True Action', alpha=0.6, s=100, edgecolors='black')
ax5.scatter(pred_x, pred_y, c='red', label='Predicted Action', alpha=0.6, s=100, marker='s', edgecolors='black')
ax5.set_xlabel('X-axis', fontsize=10)
ax5.set_ylabel('Y-axis', fontsize=10)
ax5.set_title('Action Space Distribution', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Subplot 6: Inference time boxplot
ax6 = plt.subplot(2, 3, 6)
ax6.boxplot(times_array, vert=True)
ax6.set_ylabel('Inference Time (ms)', fontsize=10)
ax6.set_title('Inference Time Boxplot', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle('SmolVLA Dataset Test - Visualization Report', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
output_file = '/root/smolvla_project/visualization_result.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"   OK: Visualization saved to {output_file}")

# 5. Summary
print(f"\n[5/5] Test Summary")
print("="*60)
print(f"Samples: {num_samples}")
print(f"Average inference time: {times_array.mean():.2f} ms")
print(f"Inference frequency: {1000/times_array.mean():.2f} Hz")
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"\nVisualization saved: visualization_result.png")
print("="*60 + "\n")
