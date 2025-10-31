"""
SmolVLA SO100 Pick-Place 数据集测试脚本 - 带反归一化
测试 SmolVLA 在其训练数据集上的性能
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

def main():
    print("=" * 60)
    print("SmolVLA SO100 Pick-Place 测试 - 带反归一化")
    print("=" * 60)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # ========== 1. 加载数据集 ==========
    print("\n[1/7] 加载数据集...")
    dataset_name = "lerobot/svla_so100_pickplace"
    dataset = LeRobotDataset(dataset_name)
    print(f"   数据集: {dataset_name}")
    print(f"   总样本数: {len(dataset)}")
    print(f"   动作维度: 6 (SO100 6-DOF)")
    
    # 检查数据集结构
    sample = dataset[0]
    print(f"   相机: {[k for k in sample.keys() if 'images' in k]}")
    print(f"   图像尺寸: {sample['observation.images.top'].shape}")
    
    # ========== 2. 获取归一化统计信息 ==========
    print("\n[2/7] 获取归一化统计信息...")
    action_stats = dataset.meta.stats['action']
    action_mean = action_stats['mean']
    action_std = action_stats['std']
    
    print(f"   动作均值: {action_mean}")
    print(f"   动作标准差: {action_std}")
    print(f"   动作范围: [{action_stats['min']}] 到 [{action_stats['max']}]")
    
    # ========== 3. 加载模型 ==========
    print("\n[3/7] 加载模型...")
    model_name = "lerobot/smolvla_base"
    policy = SmolVLAPolicy.from_pretrained(model_name).to(device)
    policy.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    print("   模型加载完成")
    
    # ========== 4. 推理测试 ==========
    print("\n[4/7] 运行推理测试...")
    num_samples = 100
    
    inference_times = []
    predictions_norm = []  # 归一化的预测
    predictions_denorm = []  # 反归一化的预测
    ground_truths = []
    
    # 任务描述
    task_text = "Pick up the cube and place it in the box."
    
    print(f"   测试样本: {num_samples}")
    print(f"   任务描述: {task_text}")
    
    with torch.no_grad():
        for i in range(num_samples):
            if (i + 1) % 50 == 0:
                print(f"   进度: {i+1}/{num_samples}")
            
            sample = dataset[i]
            
            # 准备图像输入 - SO100 有 top 和 wrist 两个相机
            top_image = sample['observation.images.top'].unsqueeze(0).to(device)
            wrist_image = sample['observation.images.wrist'].unsqueeze(0).to(device)
            
            # 调整图像尺寸到 256x256
            top_image = F.interpolate(top_image, size=(256, 256), mode='bilinear', align_corners=False)
            wrist_image = F.interpolate(wrist_image, size=(256, 256), mode='bilinear', align_corners=False)
            
            # 准备状态输入
            state = sample['observation.state'].unsqueeze(0).to(device)
            
            # 准备语言输入
            text_tokens = tokenizer(task_text, return_tensors="pt")
            lang_tokens = text_tokens['input_ids'].to(device)
            lang_attention_mask = text_tokens['attention_mask'].to(device).bool()
            
            # 构建批次 - SmolVLA 需要 3 个相机输入
            batch = {
                'observation.images.camera1': top_image,
                'observation.images.camera2': wrist_image,
                'observation.images.camera3': wrist_image,
                'observation.state': state,
                'observation.language.tokens': lang_tokens,
                'observation.language.attention_mask': lang_attention_mask
            }
            
            # 推理
            start_time = time.time()
            output = policy.select_action(batch)
            inference_time = (time.time() - start_time) * 1000
            
            inference_times.append(inference_time)
            
            # 提取预测动作（归一化值）
            pred_action_norm = output[0, :6].cpu().numpy()
            predictions_norm.append(pred_action_norm)
            
            # 反归一化
            pred_action_denorm = pred_action_norm * action_std + action_mean
            predictions_denorm.append(pred_action_denorm)
            
            # 真实值（已经是原始尺度）
            true_action = sample['action'].cpu().numpy()
            ground_truths.append(true_action)
    
    print(f"   平均推理时间: {np.mean(inference_times):.2f}ms")
    
    # ========== 5. 计算指标 ==========
    print("\n[5/7] 计算性能指标...")
    
    predictions_norm = np.array(predictions_norm)
    predictions_denorm = np.array(predictions_denorm)
    ground_truths = np.array(ground_truths)
    
    # 反归一化后的指标
    errors = predictions_denorm - ground_truths
    mse = np.mean(errors ** 2)
    mae = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))
    
    # 每个关节的指标
    mse_per_joint = np.mean(errors ** 2, axis=0)
    mae_per_joint = np.mean(np.abs(errors), axis=0)
    
    print(f"   整体 MSE: {mse:.4f}")
    print(f"   整体 MAE: {mae:.4f}°")
    print(f"   最大误差: {max_error:.4f}°")
    print(f"\n   各关节 MAE:")
    joint_names = ['Shoulder Pan', 'Shoulder Lift', 'Elbow', 'Wrist Flex', 'Wrist Roll', 'Gripper']
    for i, (name, mae_val) in enumerate(zip(joint_names, mae_per_joint)):
        print(f"     关节{i} ({name}): {mae_val:.4f}°")
    
    # 数值范围验证
    print(f"\n   数值范围验证:")
    print(f"   预测值(归一化): [{predictions_norm.min():.2f}, {predictions_norm.max():.2f}]")
    print(f"   预测值(反归一化): [{predictions_denorm.min():.2f}, {predictions_denorm.max():.2f}]")
    print(f"   真实值: [{ground_truths.min():.2f}, {ground_truths.max():.2f}]")
    
    # ========== 6. 生成可视化 ==========
    print("\n[6/7] 生成可视化...")
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('SmolVLA SO100 Pick-Place 测试结果 (反归一化)', fontsize=16, fontweight='bold')
    
    # 1. 推理性能统计
    ax = axes[0, 0]
    stats = [
        np.min(inference_times),
        np.mean(inference_times),
        np.median(inference_times),
        np.percentile(inference_times, 95),
        np.max(inference_times)
    ]
    labels = ['Min', 'Mean', 'Median', 'P95', 'Max']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#e67e22']
    bars = ax.bar(labels, stats, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('时间 (ms)', fontsize=10)
    ax.set_title('推理性能统计', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar, stat in zip(bars, stats):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{stat:.1f}ms', ha='center', va='bottom', fontsize=9)
    
    # 2. 各关节 MAE 分布
    ax = axes[0, 1]
    x_pos = np.arange(len(joint_names))
    bars = ax.bar(x_pos, mae_per_joint, color='skyblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('关节', fontsize=10)
    ax.set_ylabel('平均绝对误差 (度)', fontsize=10)
    ax.set_title('各关节 MAE', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'J{i}' for i in range(6)], rotation=0)
    ax.grid(axis='y', alpha=0.3)
    for bar, mae_val in zip(bars, mae_per_joint):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae_val:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 3. 各关节 MSE 分布
    ax = axes[0, 2]
    bars = ax.bar(x_pos, mse_per_joint, color='lightcoral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('关节', fontsize=10)
    ax.set_ylabel('均方误差', fontsize=10)
    ax.set_title('各关节 MSE', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'J{i}' for i in range(6)], rotation=0)
    ax.grid(axis='y', alpha=0.3)
    
    # 4-9. 每个关节的预测 vs 真实值轨迹
    for joint_idx in range(6):
        row = 1 + joint_idx // 3
        col = joint_idx % 3
        ax = axes[row, col]
        
        sample_range = range(min(50, num_samples))
        ax.plot(sample_range, ground_truths[sample_range, joint_idx], 
                'b-', label='真实值', linewidth=2, alpha=0.7)
        ax.plot(sample_range, predictions_denorm[sample_range, joint_idx], 
                'r--', label='预测值', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('样本索引', fontsize=9)
        ax.set_ylabel('关节角度 (度)', fontsize=9)
        ax.set_title(f'{joint_names[joint_idx]} (MAE={mae_per_joint[joint_idx]:.2f}°)', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = '/root/lerobot_project/so100_pickplace_result_fixed.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   可视化已保存: {output_path}")
    
    # ========== 7. 打印详细示例 ==========
    print("\n[7/7] 详细示例（前10个样本）:")
    print("=" * 100)
    
    for i in range(min(10, num_samples)):
        print(f"\n样本 {i}:")
        print(f"  预测值(归一化):   {predictions_norm[i]}")
        print(f"  预测值(反归一化): {predictions_denorm[i]}")
        print(f"  真实值:          {ground_truths[i]}")
        print(f"  误差:            {errors[i]}")
        print(f"  绝对误差:        {np.abs(errors[i])}")
    
    # ========== 总结 ==========
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    print(f"\n关键指标:")
    print(f"  - 平均推理时间: {np.mean(inference_times):.2f}ms")
    print(f"  - 整体 MAE: {mae:.4f}°")
    print(f"  - 整体 MSE: {mse:.4f}")
    print(f"  - 最大误差: {max_error:.4f}°")
    
    # 性能评估
    print(f"\n性能评估:")
    if mae < 5.0:
        print(f"  ✅ 优秀 (MAE < 5°) - 符合预期!")
    elif mae < 10.0:
        print(f"  ✅ 良好 (MAE < 10°) - 可接受")
    elif mae < 20.0:
        print(f"  ⚠️  一般 (MAE < 20°) - 需要改进")
    else:
        print(f"  ❌ 较差 (MAE >= 20°) - 存在问题")
    
    print(f"\n可视化报告: {output_path}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
