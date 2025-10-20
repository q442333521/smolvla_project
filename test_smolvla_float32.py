"""
SmolVLA 测试 - 使用 Float32 (显存充足版本)
"""

import torch
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
import time

print("🚀 SmolVLA 推理测试 (Float32 完整版)\n")

# 加载模型 - 强制使用 float32
print("加载模型（使用 Float32）...")
policy = SmolVLAPolicy.from_pretrained(
    "lerobot/smolvla_base",
    torch_dtype=torch.float32  # 强制使用 float32
)
policy = policy.to("cuda").eval()

# 确保所有参数都是 float32
policy = policy.float()

print(f"✅ 模型加载 ({sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M参数)")
print(f"   数据类型: {next(policy.parameters()).dtype}\n")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
print("✅ Tokenizer加载\n")

# 准备输入
image = torch.rand(1, 3, 256, 256).cuda()  # float32
state = torch.randn(1, 7).cuda()  # float32
instruction = "pick up the red cube"

# Token化
tokens = tokenizer(
    instruction,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=77
)

# 准备observation
observation = {
    "observation.images.camera1": image,
    "observation.state": state,
    "observation.language.tokens": tokens["input_ids"].cuda(),
    "observation.language.attention_mask": tokens["attention_mask"].cuda().bool(),
}

print(f"输入配置：")
print(f"  图像: {image.shape}, dtype: {image.dtype}")
print(f"  状态: {state.shape}, dtype: {state.dtype}")
print(f"  指令: '{instruction}'")
print(f"  attention_mask: dtype bool ✅\n")

# 第一次推理（预热）
print("🔥 预热推理（首次可能需要1-2分钟）...")
start = time.time()

with torch.no_grad():
    actions = policy.select_action(observation)

t = (time.time() - start)

print(f"✅ 推理成功！")
print(f"   首次时间: {t:.2f}秒 ({t*1000:.1f}ms)")
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
        "observation.images.camera1": torch.rand(1, 3, 256, 256).cuda(),
        "observation.state": torch.randn(1, 7).cuda(),
        "observation.language.tokens": tokens["input_ids"].cuda(),
        "observation.language.attention_mask": tokens["attention_mask"].cuda().bool(),
    }
    
    start = time.time()
    with torch.no_grad():
        _ = policy.select_action(obs)
    times.append((time.time() - start) * 1000)
    
    if (i + 1) % 5 == 0:
        print(f"  完成 {i+1}/10 (当前平均: {np.mean(times):.1f}ms)")

times = np.array(times)
print(f"\n✅ 统计结果:")
print(f"   平均时间: {times.mean():.1f}ms")
print(f"   中位数: {np.median(times):.1f}ms")
print(f"   最小值: {times.min():.1f}ms")
print(f"   最大值: {times.max():.1f}ms")
print(f"   标准差: {times.std():.1f}ms")
print(f"   推理频率: {1000/times.mean():.2f}Hz\n")

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
    print(f" ⚠️  未达标 (目标<150ms)")

# 推理频率检查
freq_ok = 1000/times.mean() > 7
print(f"2. 推理频率: {1000/times.mean():.2f}Hz", end="")
if freq_ok:
    print(" ✅ 达标 (>7Hz)")
else:
    print(f" ⚠️  未达标 (目标>7Hz)")

# 显存检查（float32 可能需要更多显存，目标 <12GB）
mem_ok = mem_allocated < 12
print(f"3. 显存占用: {mem_allocated:.2f}GB", end="")
if mem_ok:
    print(" ✅ 达标 (<12GB for Float32)")
else:
    print(f" ⚠️  超标 (目标<12GB)")

# 稳定性检查
stability_ok = times.std() < 50
print(f"4. 稳定性: σ={times.std():.1f}ms", end="")
if stability_ok:
    print(" ✅ 达标")
else:
    print(" ⚠️  不稳定")

print()

# 性能等级
all_ok = speed_ok and freq_ok and mem_ok and stability_ok
if times.mean() < 100 and mem_allocated < 8:
    grade = "优秀 (A)"
elif times.mean() < 150 and mem_allocated < 12:
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
    print("  1️⃣  运行模拟环境测试")
    print("  2️⃣  运行性能基准测试")
    print("  3️⃣  填写项目进度清单")
    print("  4️⃣  准备进入 ROS2 集成阶段")
else:
    print("⚠️  部分指标未达标，建议:")
    if not mem_ok:
        print("  - 显存超标，可以尝试 FP16")
    if not speed_ok:
        print("  - 速度较慢，正常情况（Float32 比 FP16 慢）")

print("\n✅ 测试完成！")
print("=" * 60)
