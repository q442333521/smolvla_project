#!/usr/bin/env python3
"""
SmolVLA Dataset Test - 修复尺度匹配问题
包含反归一化逻辑，正确计算评估指标
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
print("SmolVLA Visualization Test - 修复版")
print("="*60 + "\n")

device = "cuda"

# 1. 加载数据集
print("[1/6] Loading data...")
dataset = LeRobotDataset("lerobot/pusht")
print(f"   OK: {len(dataset)} samples")

# 2. 计算动作统计信息（用于反归一化）
print("\n[2/6] Computing action statistics...")
print("   Sampling 1000 actions for statistics...")
actions_for_stats = []
for i in range(min(1000, len(dataset))):
    actions_for_stats.append(dataset[i]['action'].cpu().numpy())
actions_for_stats = np.array(actions_for_stats)

action_mean = actions_for_stats.mean(axis=0)
action_std = actions_for_stats.std(axis=0)
action_min = actions_for_stats.min(axis=0)
action_max = actions_for_stats.max(axis=0)

print(f"   Mean: [{action_mean[0]:.2f}, {action_mean[1]:.2f}]")
print(f"   Std:  [{action_std[0]:.2f}, {action_std[1]:.2f}]")
print(f"   Range: [{action_min[0]:.0f}, {action_max[0]:.0f}] x [{action_min[1]:.0f}, {action_max[1]:.0f}]")

# 3. 加载模型
print(f"\n[3/6] Loading model...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).float().eval()
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
expected_size = (256, 256)
print("   OK")

# 4. 推理测试
test_samples = min(100, len(dataset))
print(f"\n[4/6] Running inference on {test_samples} samples...")

times = []
predictions_normalized = []  # 归一化的预测值
predictions_denormalized = []  # 反归一化的预测值
ground_truths = []

with torch.no_grad():
    for i in range(test_samples):
        sample = dataset[i]
        
        # 准备输入
        image = sample['observation.image'].unsqueeze(0).to(device).float() / 255.0
        image = F.interpolate(image, size=expected_size, mode='bilinear', align_corners=False)
        
        text = sample['task']
        text_tokens = tokenizer(text, return_tensors="pt")
        
        batch = {
            'observation.images.camera1': image,
            'observation.images.camera2': image,
            'observation.images.camera3': image,
            'observation.state': sample['observation.state'].unsqueeze(0).to(device),
            'observation.language.tokens': text_tokens['input_ids'].to(device),
            'observation.language.attention_mask': text_tokens['attention_mask'].to(device).bool()
        }
        
        # 推理计时
        start = time.time()
        output = policy.select_action(batch)
        times.append(time.time() - start)
        
        # 保存归一化的预测值
        pred_norm = output[0, :2].cpu().numpy()
        predictions_normalized.append(pred_norm)
        
        # 反归一化预测值到原始尺度
        pred_denorm = pred_norm * action_std + action_mean
        predictions_denormalized.append(pred_denorm)
        
        # 保存真实值
        true_action = sample['action'].cpu().numpy()
        ground_truths.append(true_action)
        
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i + 1}/{test_samples}")

predictions_normalized = np.array(predictions_normalized)
predictions_denormalized = np.array(predictions_denormalized)
ground_truths = np.array(ground_truths)

print(f"   Avg inference time: {np.mean(times)*1000:.2f}ms")

# 5. 计算指标（基于反归一化后的值）
print("\n[5/6] Computing metrics...")

# 使用反归一化后的预测值计算指标
mse = np.mean((predictions_denormalized - ground_truths) ** 2, axis=0)
mae = np.mean(np.abs(predictions_denormalized - ground_truths), axis=0)
max_error = np.max(np.abs(predictions_denormalized - ground_truths), axis=0)

print(f"   MSE: {mse.mean():.4f} (Dim0: {mse[0]:.4f}, Dim1: {mse[1]:.4f})")
print(f"   MAE: {mae.mean():.4f} (Dim0: {mae[0]:.4f}, Dim1: {mae[1]:.4f})")
print(f"   Max Error: {max_error.mean():.4f} (Dim0: {max_error[0]:.4f}, Dim1: {max_error[1]:.4f})")

# 打印数值范围验证
print(f"\n   数值范围验证:")
print(f"   预测值(归一化): [{predictions_normalized.min():.4f}, {predictions_normalized.max():.4f}]")
print(f"   预测值(反归一化): [{predictions_denormalized.min():.1f}, {predictions_denormalized.max():.1f}]")
print(f"   真实值: [{ground_truths.min():.1f}, {ground_truths.max():.1f}]")

# 6. 生成可视化
print("\n[6/6] Generating visualization...")

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

# 子图2: 动作轨迹对比（x轴）- 使用反归一化的值
ax2 = plt.subplot(2, 3, 2)
sample_indices = range(len(predictions_denormalized))
ax2.plot(sample_indices, predictions_denormalized[:, 0], 'b-', label='Prediction (Denorm)', alpha=0.7, linewidth=2)
ax2.plot(sample_indices, ground_truths[:, 0], 'r--', label='Ground Truth', alpha=0.7, linewidth=2)
ax2.set_xlabel('Step', fontsize=12)
ax2.set_ylabel('Action (X-axis, pixels)', fontsize=12)
ax2.set_title('Action Trajectory - X Dimension', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# 子图3: 动作轨迹对比（y轴）- 使用反归一化的值
ax3 = plt.subplot(2, 3, 3)
ax3.plot(sample_indices, predictions_denormalized[:, 1], 'g-', label='Prediction (Denorm)', alpha=0.7, linewidth=2)
ax3.plot(sample_indices, ground_truths[:, 1], 'm--', label='Ground Truth', alpha=0.7, linewidth=2)
ax3.set_xlabel('Step', fontsize=12)
ax3.set_ylabel('Action (Y-axis, pixels)', fontsize=12)
ax3.set_title('Action Trajectory - Y Dimension', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# 子图4: 误差分布（各维度MSE）
ax4 = plt.subplot(2, 3, 4)
dims = [f'Dim {i}' for i in range(len(mse))]
bars = ax4.bar(dims, mse, color='coral', edgecolor='darkred', alpha=0.7)
ax4.set_ylabel('MSE (pixels^2)', fontsize=12)
ax4.set_title('MSE by Dimension', fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=9)

# 子图5: 预测vs真实散点图（第一维）
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(ground_truths[:, 0], predictions_denormalized[:, 0], alpha=0.5, s=20, c='blue')
lims = [min(ground_truths[:, 0].min(), predictions_denormalized[:, 0].min()),
        max(ground_truths[:, 0].max(), predictions_denormalized[:, 0].max())]
ax5.plot(lims, lims, 'r--', alpha=0.5, linewidth=2, label='Perfect Prediction')
ax5.set_xlabel('Ground Truth (pixels)', fontsize=12)
ax5.set_ylabel('Prediction (pixels)', fontsize=12)
ax5.set_title('Prediction vs Ground Truth (Dim 0)', fontsize=14, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)

# 子图6: 预测vs真实散点图（第二维）
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(ground_truths[:, 1], predictions_denormalized[:, 1], alpha=0.5, s=20, c='green')
lims = [min(ground_truths[:, 1].min(), predictions_denormalized[:, 1].min()),
        max(ground_truths[:, 1].max(), predictions_denormalized[:, 1].max())]
ax6.plot(lims, lims, 'r--', alpha=0.5, linewidth=2, label='Perfect Prediction')
ax6.set_xlabel('Ground Truth (pixels)', fontsize=12)
ax6.set_ylabel('Prediction (pixels)', fontsize=12)
ax6.set_title('Prediction vs Ground Truth (Dim 1)', fontsize=14, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3)

plt.tight_layout()

# 保存图像
output_file = '/root/lerobot_project/visualization_result_fixed.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"   Saved: {output_file}")

# 保存详细结果
print("\n" + "="*60)
print("详细对比（前10个样本）:")
print("="*60)
for i in range(min(10, test_samples)):
    pred_norm = predictions_normalized[i]
    pred_denorm = predictions_denormalized[i]
    true = ground_truths[i]
    error = np.abs(pred_denorm - true)
    print(f"\n样本 {i}:")
    print(f"  预测(归一化): [{pred_norm[0]:7.4f}, {pred_norm[1]:7.4f}]")
    print(f"  预测(反归一化): [{pred_denorm[0]:7.2f}, {pred_denorm[1]:7.2f}]")
    print(f"  真实值:      [{true[0]:7.2f}, {true[1]:7.2f}]")
    print(f"  误差(像素):  [{error[0]:7.2f}, {error[1]:7.2f}]")

print("\n" + "="*60)
print("Test Complete!")
print("="*60 + "\n")
