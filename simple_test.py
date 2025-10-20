#!/usr/bin/env python3
"""
简单的 SmolVLA 测试脚本
测试已下载的数据集和模型推理
"""

import sys
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
import time

# 添加 lerobot 路径
sys.path.insert(0, '/root/smolvla_project/lerobot/src')

print("\n" + "="*60)
print("SmolVLA 测试脚本")
print("="*60 + "\n")

# ============================================================
# 1. 加载数据集
# ============================================================
print("步骤 1/3: 加载数据集")
print("-"*60)

dataset_name = "lerobot/pusht"  # 使用最小的数据集测试
print(f"加载: {dataset_name}")

try:
    dataset = load_dataset(dataset_name, split="train")
    print(f"✅ 数据集加载成功")
    print(f"   样本数: {len(dataset)}")
    
    # 查看第一个样本
    sample = dataset[0]
    print(f"   字段: {list(sample.keys())}")
    
except Exception as e:
    print(f"❌ 数据集加载失败: {e}")
    exit(1)

# ============================================================
# 2. 加载 SmolVLA 模型
# ============================================================
print(f"\n步骤 2/3: 加载 SmolVLA 模型")
print("-"*60)

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    
    # ✅ 修复后的正确加载方式
    print("加载模型...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy = policy.to(device).float()
    policy.eval()
    
    print(f"✅ 模型加载成功")
    print(f"   数据类型: {next(policy.parameters()).dtype}")
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# 3. 测试推理
# ============================================================
print(f"\n步骤 3/3: 测试推理")
print("-"*60)

try:
    # 准备输入数据
    sample = dataset[0]
    
    # 查找图像字段
    image_key = None
    for key in sample.keys():
        if 'image' in key.lower():
            image_key = key
            break
    
    # 查找状态字段
    state_key = None
    for key in sample.keys():
        if 'state' in key.lower():
            state_key = key
            break
    
    # 构造观察
    obs = {}
    
    if image_key:
        image_data = sample[image_key]
        if isinstance(image_data, np.ndarray):
            obs["image"] = Image.fromarray(image_data)
        else:
            obs["image"] = image_data
        print(f"图像字段: {image_key}")
    else:
        obs["image"] = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        print(f"图像字段: 使用虚拟数据")
    
    if state_key:
        state_data = sample[state_key]
        if isinstance(state_data, np.ndarray):
            obs["state"] = torch.from_numpy(state_data).float()
        else:
            obs["state"] = state_data
        print(f"状态字段: {state_key}, shape={obs['state'].shape}")
    else:
        obs["state"] = torch.randn(7)
        print(f"状态字段: 使用虚拟数据, shape={obs['state'].shape}")
    
    obs["state"] = obs["state"].to(device)
    
    # 执行推理
    print(f"\n执行推理...")
    start_time = time.time()
    
    with torch.no_grad():
        action = policy.select_action(
            observation=obs,
            instruction="Pick up the object"
        )
    
    inference_time = (time.time() - start_time) * 1000
    
    print(f"✅ 推理成功!")
    print(f"   输出形状: {action.shape}")
    print(f"   动作维度: {action.shape[-1]}")
    print(f"   动作范围: [{action.min().item():.3f}, {action.max().item():.3f}]")
    print(f"   推理时间: {inference_time:.1f} ms")
    
    if action.shape[-1] == 6:
        print(f"   ℹ️  输出为 6 维（正常行为）")
    
except Exception as e:
    print(f"❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# 总结
# ============================================================
print(f"\n" + "="*60)
print("✅ 所有测试通过!")
print("="*60)
print(f"\n测试结果:")
print(f"  数据集: ✅")
print(f"  模型加载: ✅")
print(f"  推理: ✅")
print(f"\n💡 提示: 可以修改 dataset_name 测试其他数据集")
print("")

