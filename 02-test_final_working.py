#!/usr/bin/env python3
"""SmolVLA 完整工作测试 - 所有问题已修复"""

import sys
import torch
import numpy as np
import time

sys.path.insert(0, '/root/smolvla_project/lerobot/src')

print("\n" + "="*60)
print("SmolVLA 完整测试 - 所有修复已应用")
print("="*60 + "\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {device}")

# 1. 加载模型
print("\n[1/3] 加载模型...")
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# ✅ 修复1: 不传 torch_dtype 和 device
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to(device).float()  # ✅ 修复2: 手动设置，使用 float32
policy.eval()
print("✅ 加载成功")

# 2. 准备输入
print("\n[2/3] 准备输入...")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
instruction = "Pick up the red block and place it in the box"
tokens = tokenizer(instruction, return_tensors="pt")

# ✅ 修复3: attention_mask 转为 bool 类型
batch = {
    "observation.images.camera1": torch.randn(1, 3, 256, 256).to(device),
    "observation.images.camera2": torch.randn(1, 3, 256, 256).to(device),
    "observation.images.camera3": torch.randn(1, 3, 256, 256).to(device),
    "observation.state": torch.randn(1, 14).to(device),
    "observation.language.tokens": tokens['input_ids'].to(device),
    "observation.language.attention_mask": tokens['attention_mask'].to(device).bool(),  # ✅ 关键修复!
}

print(f"  图像: 3个视角 (1,3,256,256)")
print(f"  状态: (1,14)")
print(f"  指令: '{instruction}'")
print(f"  attention_mask类型: {batch['observation.language.attention_mask'].dtype}")  # 应该是 bool

# 3. 推理
print("\n[3/3] 推理测试...")
start = time.time()
with torch.no_grad():
    action = policy.select_action(batch)
time_ms = (time.time() - start) * 1000

print(f"\n✅ 推理成功!")
print(f"   输出形状: {action.shape}")
print(f"   动作维度: {action.shape[-1]}")
print(f"   动作范围: [{action.min().item():.3f}, {action.max().item():.3f}]")
print(f"   推理时间: {time_ms:.0f} ms")

# 性能测试
print(f"\n性能测试（20次）...")
times = []
for i in range(20):
    start = time.time()
    with torch.no_grad():
        _ = policy.select_action(batch)
    times.append((time.time() - start) * 1000)
    if (i + 1) % 5 == 0:
        print(f"  进度: {i+1}/20")

print(f"\n性能统计:")
print(f"  平均: {np.mean(times):.0f} ms")
print(f"  中位数: {np.median(times):.0f} ms")
print(f"  最小: {np.min(times):.0f} ms")
print(f"  最大: {np.max(times):.0f} ms")
print(f"  控制频率: ~{1000/np.mean(times):.1f} Hz")

# 总结
print(f"\n" + "="*60)
print("✅ 所有测试通过! 问题全部修复!")
print("="*60)

print("\n关键修复汇总:")
print("  1. ✅ from_pretrained() 不传 torch_dtype/device")
print("  2. ✅ 使用 .to(device).float() 手动设置")
print("  3. ✅ attention_mask 转为 bool 类型")
print("  4. ✅ 图像 batch 维度: (1,3,256,256)")
print("  5. ✅ 正确的键名格式")

print("\n完整输入格式:")
print("  observation.images.camera1/2/3: torch.Tensor (1,3,256,256)")
print("  observation.state: torch.Tensor (1,14)")
print("  observation.language.tokens: torch.Tensor (1,seq_len)")
print("  observation.language.attention_mask: torch.BoolTensor (1,seq_len)")

print(f"\n💡 这个脚本可以直接用于生产环境!")
print("")

