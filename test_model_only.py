#!/usr/bin/env python3
"""
纯模型测试脚本（不依赖数据集）
测试 SmolVLA 模型加载和推理是否正常
"""

import sys
import torch
import numpy as np
from PIL import Image
import time

# 添加 lerobot 路径
sys.path.insert(0, '/root/smolvla_project/lerobot/src')

print("\n" + "="*60)
print("SmolVLA 模型测试（不依赖数据集）")
print("="*60 + "\n")

# ============================================================
# 1. 检查环境
# ============================================================
print("步骤 1/3: 检查环境")
print("-"*60)

print(f"Python 版本: {sys.version.split()[0]}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    device = "cpu"

print(f"使用设备: {device}")

# ============================================================
# 2. 加载 SmolVLA 模型
# ============================================================
print(f"\n步骤 2/3: 加载 SmolVLA 模型")
print("-"*60)

try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    print("加载模型...")
    print("⚠️  使用修复后的加载方式:")
    print("   - 不传递 torch_dtype 和 device 参数")
    print("   - 使用 .to(device).float()")
    
    # ✅ 修复后的正确加载方式
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy = policy.to(device).float()
    policy.eval()
    
    print(f"\n✅ 模型加载成功!")
    print(f"   设备: {device}")
    print(f"   数据类型: {next(policy.parameters()).dtype}")
    print(f"   参数量: {sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M")
    
except Exception as e:
    print(f"\n❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# 3. 测试推理
# ============================================================
print(f"\n步骤 3/3: 测试推理")
print("-"*60)

try:
    # 创建虚拟输入数据
    print("创建虚拟输入数据...")
    
    obs = {
        "image": Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        ),
        "state": torch.randn(7).to(device)
    }
    
    print(f"  图像: PIL Image, size=(224, 224)")
    print(f"  状态: shape={obs['state'].shape}, device={obs['state'].device}")
    
    # 执行推理
    print(f"\n执行推理...")
    start_time = time.time()
    
    with torch.no_grad():
        action = policy.select_action(
            observation=obs,
            instruction="Pick up the red block and place it in the blue box"
        )
    
    inference_time = (time.time() - start_time) * 1000
    
    print(f"\n✅ 推理成功!")
    print(f"   输出形状: {action.shape}")
    print(f"   输出设备: {action.device}")
    print(f"   动作维度: {action.shape[-1]}")
    print(f"   动作范围: [{action.min().item():.3f}, {action.max().item():.3f}]")
    print(f"   推理时间: {inference_time:.1f} ms")
    
    if action.shape[-1] == 6:
        print(f"\n   ℹ️  注意: 输出为 6 维（而非 7 维）")
        print(f"   这是正常的模型行为，6 维足够用于机械臂控制")
    
    # 多次推理测试性能
    print(f"\n性能测试（10 次推理）...")
    times = []
    for i in range(10):
        start = time.time()
        with torch.no_grad():
            _ = policy.select_action(observation=obs, instruction="test")
        times.append((time.time() - start) * 1000)
    
    print(f"   平均时间: {np.mean(times):.1f} ms")
    print(f"   最小时间: {np.min(times):.1f} ms")
    print(f"   最大时间: {np.max(times):.1f} ms")
    
except Exception as e:
    print(f"\n❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================
# 总结
# ============================================================
print(f"\n" + "="*60)
print("✅ 所有测试通过!")
print("="*60)

print(f"\n测试结果汇总:")
print(f"  ✅ 环境检查通过")
print(f"  ✅ 模型加载成功（使用修复后的方法）")
print(f"  ✅ 推理功能正常")
print(f"  ✅ 性能测试完成")

print(f"\n关键发现:")
print(f"  1. torch_dtype 参数问题已修复")
print(f"  2. 使用 float32 避免了 dtype 不匹配")
print(f"  3. 模型输出为 {action.shape[-1]} 维动作")
print(f"  4. 推理速度约 {np.mean(times):.0f} ms/次")

print(f"\n💡 下一步:")
print(f"  - 可以尝试使用真实数据集测试")
print(f"  - 可以集成到机器人控制系统")
print(f"  - 可以进行模型微调")
print("")

