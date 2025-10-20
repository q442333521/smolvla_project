"""
SmolVLA 修复版本 - 解决 attention_mask 类型问题
"""

import torch
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
import time

print("🚀 SmolVLA 推理测试 (修复版)\n")

# 加载模型
print("加载模型...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to("cuda").half().eval()
print(f"✅ 模型加载 ({sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M参数)\n")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
print("✅ Tokenizer加载\n")

# 准备输入
image = torch.rand(1, 3, 256, 256).cuda().half()
state = torch.randn(1, 7).cuda().half()
instruction = "pick up the red cube"

# Token化
tokens = tokenizer(
    instruction,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=77
)

# 关键修复：将 attention_mask 转换为布尔类型
observation = {
    "observation.images.camera1": image,
    "observation.state": state,
    "observation.language.tokens": tokens["input_ids"].cuda(),
    "observation.language.attention_mask": tokens["attention_mask"].cuda().bool(),  # ← 修复：转为bool!
}

print(f"输入：")
print(f"  图像: {image.shape}, dtype: {image.dtype}")
print(f"  状态: {state.shape}, dtype: {state.dtype}")
print(f"  指令: '{instruction}'")
print(f"  tokens: {tokens['input_ids'].shape}")
print(f"  attention_mask: {tokens['attention_mask'].shape}, dtype: bool ✅\n")

# 推理
print("推理中...")
start = time.time()

with torch.no_grad():
    actions = policy.select_action(observation)

t = (time.time() - start) * 1000

print(f"✅ 推理成功！")
print(f"   时间: {t:.1f}ms")
print(f"   输出: {actions.shape}")
print(f"   范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]\n")

# 速度测试
print("速度测试（10次）...")
times = []
for i in range(10):
    obs = {
        "observation.images.camera1": torch.rand(1, 3, 256, 256).cuda().half(),
        "observation.state": torch.randn(1, 7).cuda().half(),
        "observation.language.tokens": tokens["input_ids"].cuda(),
        "observation.language.attention_mask": tokens["attention_mask"].cuda().bool(),  # 修复
    }
    
    start = time.time()
    with torch.no_grad():
        _ = policy.select_action(obs)
    times.append((time.time() - start) * 1000)
    
    if (i + 1) % 5 == 0:
        print(f"  完成 {i+1}/10...")

times = np.array(times)
print(f"\n✅ 平均: {times.mean():.1f}ms, 频率: {1000/times.mean():.1f}Hz")
print(f"   中位数: {np.median(times):.1f}ms")
print(f"   标准差: {times.std():.1f}ms\n")

# 显存
mem = torch.cuda.memory_allocated() / 1024**3
print(f"💾 显存: {mem:.2f}GB\n")

# 总结
print("=" * 60)
print("📊 性能总结")
print("=" * 60)
print(f"推理速度: {times.mean():.1f}ms", end="")
if times.mean() < 150:
    print(" ✅ 达标 (<150ms)")
else:
    print(f" ⚠️  未达标 (目标<150ms)")

print(f"推理频率: {1000/times.mean():.1f}Hz", end="")
if 1000/times.mean() > 7:
    print(" ✅ 达标 (>7Hz)")
else:
    print(" ⚠️  未达标 (目标>7Hz)")

print(f"显存占用: {mem:.2f}GB", end="")
if mem < 8:
    print(" ✅ 达标 (<8GB)")
else:
    print(" ⚠️  未达标 (目标<8GB)")

print("\n性能等级: ", end="")
if times.mean() < 100 and mem < 6:
    print("优秀 (A)")
elif times.mean() < 150 and mem < 8:
    print("良好 (B)")
elif times.mean() < 200:
    print("及格 (C)")
else:
    print("需要优化 (D)")

if times.mean() < 150 and mem < 8:
    print("\n🎉 所有指标达标！可以进入下一阶段")
    print("\n📚 下一步:")
    print("  1. 运行模拟环境测试")
    print("  2. 性能基准测试")
    print("  3. ROS2集成准备")
else:
    print("\n继续优化或直接进入下一阶段")

print("\n" + "=" * 60)
