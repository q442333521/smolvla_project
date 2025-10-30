#!/usr/bin/env python3
"""
SmolVLA Dataset Test - With Visualization
"""

import sys
sys.path.insert(0, '/root/lerobot_project/lerobot/src')
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
import time
import torch.nn.functional as F

# 使用默认英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("\n" + "="*60)
print("SmolVLA Visualization Test")
print("="*60 + "\n")

device = "cuda"

# 1. 加载数据
print("[1/5] Loading data...")
dataset = LeRobotDataset("lerobot/pusht")
print(f"   OK: {len(dataset)} samples")

# 取前500个样本用于测试
test_samples = min(500, len(dataset))
print(f"   Using first {test_samples} samples for testing")

# 2. 加载模型
print(f"\n[2/5] Loading model...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).float().eval()
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
print("   OK")

# 检查模型期望的图像尺寸
expected_size = (256, 256)  # 根据错误信息
print(f"   Model expects images of size: {expected_size}")

# 3. 推理测试
print("\n[3/5] Running inference...")
times = []
predictions = []
ground_truths = []

with torch.no_grad():
    for i in range(test_samples):
        sample = dataset[i]
        
        # 准备输入 - LeRobotDataset 返回的图像是 [C, H, W] 格式，值域 [0, 255]
        image = sample['observation.image'].unsqueeze(0).to(device).float() / 255.0
        
        # 调整图像大小到模型期望的尺寸
        image = F.interpolate(image, size=expected_size, mode='bilinear', align_corners=False)
        
        # 模型期望 3 个相机输入，我们将同一图像复制 3 次
        batch = {
            'observation.images.camera1': image,
            'observation.images.camera2': image,
            'observation.images.camera3': image,
            'text': [sample['task']]
        }
        
        # 推理计时
        start = time.time()
        output = policy.select_action(batch)
        times.append(time.time() - start)
        
        # 保存预测和真实值
        pred_action = output[0].cpu().numpy()
        true_action = sample['action'].cpu().numpy()
        
        predictions.append(pred_action)
        ground_truths.append(true_action)
        
        if (i + 1) % 100 == 0:
            print(f"   Progress: {i + 1}/{test_samples}")

predictions = np.array(predictions)
ground_truths = np.array(ground_truths)

print(f"   Avg inference time: {np.mean(times)*1000:.2f}ms")

# 4. 计算指标
print("\n[4/5] Computing metrics...")
mse = np.mean((predictions - ground_truths) ** 2, axis=0)
mae = np.mean(np.abs(predictions - ground_truths), axis=0)

print(f"   MSE: {mse.mean():.4f}")
print(f"   MAE: {mae.mean():.4f}")

# 5. 生成可视化
print("\n[5/5] Generating visualization...")

fig = plt.figure(figsize=(20, 12))

# 子图1: 推理性能统计
ax1 = plt.subplot(2, 3, 1)
inference_stats = {
    'Min': np.min(times) * 1000,
    'Mean': np.mean(times) * 1000,
    'Median': np.median(times) * 1000,
    'Max': np.max(times) * 1000,
    'P95': np.percentile(times, 95) * 1000
}
bars = ax1.bar(range(len(inference_stats)), list(inference_stats.values()), color='skyblue', edgecolor='navy', alpha=0.7)
ax1.set_xticks(range(len(inference_stats)))
ax1.set_xticklabels(list(inference_stats.keys()), rotation=45)
ax1.set_ylabel('Time (ms)', fontsize=12)
ax1.set_title('Inference Performance Statistics', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')

# 子图2: 动作轨迹对比（x轴）
ax2 = plt.subplot(2, 3, 2)
sample_indices = range(min(100, len(predictions)))
ax2.plot(sample_indices, predictions[:len(sample_indices), 0], 'b-', label='Prediction', alpha=0.7, linewidth=2)
ax2.plot(sample_indices, ground_truths[:len(sample_indices), 0], 'r--', label='Ground Truth', alpha=0.7, linewidth=2)
ax2.set_xlabel('Step', fontsize=12)
ax2.set_ylabel('Action (X-axis)', fontsize=12)
ax2.set_title('Action Trajectory - X Dimension', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# 子图3: 动作轨迹对比（y轴）
ax3 = plt.subplot(2, 3, 3)
ax3.plot(sample_indices, predictions[:len(sample_indices), 1], 'g-', label='Prediction', alpha=0.7, linewidth=2)
ax3.plot(sample_indices, ground_truths[:len(sample_indices), 1], 'm--', label='Ground Truth', alpha=0.7, linewidth=2)
ax3.set_xlabel('Step', fontsize=12)
ax3.set_ylabel('Action (Y-axis)', fontsize=12)
ax3.set_title('Action Trajectory - Y Dimension', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# 子图4: 误差分布（各维度MSE）
ax4 = plt.subplot(2, 3, 4)
dims = [f'Dim {i}' for i in range(len(mse))]
bars = ax4.bar(dims, mse, color='coral', edgecolor='darkred', alpha=0.7)
ax4.set_ylabel('MSE', fontsize=12)
ax4.set_title('MSE by Dimension', fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# 子图5: 预测vs真实散点图（第一维）
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(ground_truths[:, 0], predictions[:, 0], alpha=0.5, s=20, c='blue')
lims = [min(ground_truths[:, 0].min(), predictions[:, 0].min()),
        max(ground_truths[:, 0].max(), predictions[:, 0].max())]
ax5.plot(lims, lims, 'r--', alpha=0.5, linewidth=2)
ax5.set_xlabel('Ground Truth', fontsize=12)
ax5.set_ylabel('Prediction', fontsize=12)
ax5.set_title('Prediction vs Ground Truth (Dim 0)', fontsize=14, fontweight='bold')
ax5.grid(alpha=0.3)

# 子图6: 预测vs真实散点图（第二维）
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(ground_truths[:, 1], predictions[:, 1], alpha=0.5, s=20, c='green')
lims = [min(ground_truths[:, 1].min(), predictions[:, 1].min()),
        max(ground_truths[:, 1].max(), predictions[:, 1].max())]
ax6.plot(lims, lims, 'r--', alpha=0.5, linewidth=2)
ax6.set_xlabel('Ground Truth', fontsize=12)
ax6.set_ylabel('Prediction', fontsize=12)
ax6.set_title('Prediction vs Ground Truth (Dim 1)', fontsize=14, fontweight='bold')
ax6.grid(alpha=0.3)

plt.tight_layout()

# 保存图像
output_file = '/root/lerobot_project/visualization_result.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"   Saved: {output_file}")

print("\n" + "="*60)
print("Test Complete!")
print("="*60 + "\n")
