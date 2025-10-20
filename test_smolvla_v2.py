"""
SmolVLA 完整修复版本 V2
修复问题：
1. attention_mask 类型 (Long -> bool)
2. dtype 不匹配 (Float -> Half)
"""

import torch
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
import time

print("🚀 SmolVLA 推理测试 V2 (完整修复版)\n")

# 加载模型（使用 float32 避免精度问题）
print("加载模型...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to("cuda").eval()  # 先不用 half()
print(f"✅ 模型加载 ({sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M参数)")
print(f"   模型数据类型: {next(policy.parameters()).dtype}\n")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
print("✅ Tokenizer加载\n")

# 准备输入（与模型数据类型一致）
dtype = next(policy.parameters()).dtype
device = next(policy.parameters()).device

image = torch.rand(1, 3, 256, 256, dtype=dtype, device=device)
state = torch.randn(1, 7, dtype=dtype, device=device)
instruction = "pick up the red cube"

# Token化
tokens = tokenizer(
    instruction,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=77
)

# 准备observation（关键：数据类型要一致）
observation = {
    "observation.images.camera1": image,
    "observation.state": state,
    "observation.language.tokens": tokens["input_ids"].to(device),
    "observation.language.attention_mask": tokens["attention_mask"].to(device).bool(),  # bool类型
}

print(f"输入配置：")
print(f"  图像: {image.shape}, dtype: {image.dtype}")
print(f"  状态: {state.shape}, dtype: {state.dtype}")
print(f"  指令: '{instruction}'")
print(f"  tokens: {tokens['input_ids'].shape}")
print(f"  attention_mask: {tokens['attention_mask'].shape}, dtype: bool ✅\n")

# 第一次推理（预热 + 编译）
print("🔥 预热推理（可能需要1-2分钟，请耐心等待）...")
start = time.time()

with torch.no_grad():
    actions = policy.select_action(observation)

t = (time.time() - start) * 1000

print(f"✅ 推理成功！")
print(f"   首次时间: {t:.1f}ms (包含编译)")
print(f"   输出形状: {actions.shape}")
print(f"   输出范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
print(f"   输出dtype: {actions.dtype}\n")

# 验证输出
if torch.isnan(actions).any():
    print("⚠️  警告: 输出包含 NaN!")
elif torch.isinf(actions).any():
    print("⚠️  警告: 输出包含 Inf!")
else:
    print("✅ 输出数值正常\n")

# 速度测试
print("⏱️  速度测试（10次）...")
times = []
for i in range(10):
    obs = {
        "observation.images.camera1": torch.rand(1, 3, 256, 256, dtype=dtype, device=device),
        "observation.state": torch.randn(1, 7, dtype=dtype, device=device),
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].to(device).bool(),
    }
    
    start = time.time()
    with torch.no_grad():
        _ = policy.select_action(obs)
    times.append((time.time() - start) * 1000)
    
    if (i + 1) % 5 == 0:
        print(f"  完成 {i+1}/10 (平均: {np.mean(times):.1f}ms)")

times = np.array(times)
print(f"\n✅ 统计结果:")
print(f"   平均时间: {times.mean():.1f}ms")
print(f"   中位数: {np.median(times):.1f}ms")
print(f"   最小值: {times.min():.1f}ms")
print(f"   最大值: {times.max():.1f}ms")
print(f"   标准差: {times.std():.1f}ms")
print(f"   推理频率: {1000/times.mean():.1f}Hz\n")

# 显存使用
mem_allocated = torch.cuda.memory_allocated() / 1024**3
mem_reserved = torch.cuda.memory_reserved() / 1024**3
mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

print(f"💾 显存使用:")
print(f"   已分配: {mem_allocated:.2f}GB")
print(f"   已预留: {mem_reserved:.2f}GB")
print(f"   总显存: {mem_total:.2f}GB")
print(f"   使用率: {(mem_allocated / mem_total) * 100:.1f}%\n")

# 性能总结
print("=" * 60)
print("📊 性能总结与验收")
print("=" * 60)

# 推理速度检查
speed_ok = times.mean() < 150
print(f"1. 推理速度: {times.mean():.1f}ms", end="")
if speed_ok:
    print(" ✅ 达标 (<150ms)")
else:
    print(f" ⚠️  未达标 (目标<150ms, 差 {times.mean()-150:.1f}ms)")

# 推理频率检查
freq_ok = 1000/times.mean() > 7
print(f"2. 推理频率: {1000/times.mean():.1f}Hz", end="")
if freq_ok:
    print(" ✅ 达标 (>7Hz)")
else:
    print(f" ⚠️  未达标 (目标>7Hz)")

# 显存检查
mem_ok = mem_allocated < 8
print(f"3. 显存占用: {mem_allocated:.2f}GB", end="")
if mem_ok:
    print(" ✅ 达标 (<8GB)")
else:
    print(f" ⚠️  未达标 (目标<8GB)")

# 稳定性检查
stability_ok = times.std() < 50
print(f"4. 稳定性: σ={times.std():.1f}ms", end="")
if stability_ok:
    print(" ✅ 达标 (<50ms)")
else:
    print(" ⚠️  不稳定")

print()

# 性能等级
all_ok = speed_ok and freq_ok and mem_ok and stability_ok
if times.mean() < 100 and mem_allocated < 6:
    grade = "优秀 (A)"
elif times.mean() < 150 and mem_allocated < 8:
    grade = "良好 (B)"
elif times.mean() < 200:
    grade = "及格 (C)"
else:
    grade = "需要优化 (D)"

print(f"🏆 性能等级: {grade}")

# 验收结论
print("\n" + "=" * 60)
if all_ok:
    print("🎉 验收通过！所有指标达标")
    print("\n📚 下一步行动:")
    print("  1️⃣  运行模拟环境测试 (test_simulation.py)")
    print("  2️⃣  运行性能基准测试 (benchmark.py)")
    print("  3️⃣  填写项目进度清单")
    print("  4️⃣  准备进入 ROS2 集成阶段")
else:
    print("⚠️  部分指标未达标，但可以继续")
    print("\n💡 建议:")
    if not speed_ok:
        print("  - 尝试使用 FP16: policy.half()")
        print("  - 使用 torch.compile() 优化")
    if not mem_ok:
        print("  - 使用 FP16 减少显存")
        print("  - 减小 batch size")
    if not stability_ok:
        print("  - 检查 GPU 温度和频率")
        print("  - 关闭其他GPU占用程序")

print("\n✅ 测试完成！")
print("=" * 60)
