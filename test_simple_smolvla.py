"""
SmolVLA 最简单的推理测试
手动准备所有必需的输入
"""

import torch
import numpy as np
from PIL import Image
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
import time

print("🚀 开始 SmolVLA 简单推理测试\n")

# 步骤1：加载模型
print("=" * 60)
print("步骤1：加载模型")
print("=" * 60)

policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to("cuda").half().eval()

print(f"✅ 模型加载成功")
print(f"   参数量: {sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M")
print(f"   设备: {next(policy.parameters()).device}")

# 步骤2：加载tokenizer
print("\n" + "=" * 60)
print("步骤2：加载Tokenizer")
print("=" * 60)

# 从config中获取VLM模型名称
vlm_model_name = policy.config.vlm_model_name
print(f"   VLM模型: {vlm_model_name}")

tokenizer = AutoTokenizer.from_pretrained(vlm_model_name)
print("✅ Tokenizer加载成功")

# 步骤3：准备输入数据
print("\n" + "=" * 60)
print("步骤3：准备输入数据")
print("=" * 60)

# 图像：(batch, channel, height, width)
image = torch.rand(1, 3, 256, 256).cuda().half()
print(f"   图像形状: {image.shape}")

# 状态：(batch, state_dim)
state = torch.randn(1, 7).cuda().half()
print(f"   状态形状: {state.shape}")

# 语言指令
instruction = "pick up the red cube"
print(f"   指令: '{instruction}'")

# Token化指令
tokens = tokenizer(
    instruction,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=77  # 默认长度
)
print(f"   Token形状: {tokens['input_ids'].shape}")

# 准备observation字典
observation = {
    "observation.images.camera1": image,
    "observation.state": state,
    "observation.language.tokens": tokens["input_ids"].cuda(),
}

print("✅ 输入数据准备完成")

# 步骤4：推理
print("\n" + "=" * 60)
print("步骤4：执行推理")
print("=" * 60)

start_time = time.time()

with torch.no_grad():
    actions = policy.select_action(observation)

inference_time = time.time() - start_time

print(f"✅ 推理成功！")
print(f"   推理时间: {inference_time * 1000:.2f}ms")
print(f"   输出形状: {actions.shape}")
print(f"   动作范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")

# 步骤5：多次推理测试速度
print("\n" + "=" * 60)
print("步骤5：速度测试（10次）")
print("=" * 60)

times = []
for i in range(10):
    image = torch.rand(1, 3, 256, 256).cuda().half()
    state = torch.randn(1, 7).cuda().half()
    
    observation = {
        "observation.images.camera1": image,
        "observation.state": state,
        "observation.language.tokens": tokens["input_ids"].cuda(),
    }
    
    start = time.time()
    with torch.no_grad():
        _ = policy.select_action(observation)
    times.append(time.time() - start)

times = np.array(times) * 1000  # 转为ms

print(f"✅ 速度测试完成")
print(f"   平均: {times.mean():.2f}ms")
print(f"   最小: {times.min():.2f}ms")
print(f"   最大: {times.max():.2f}ms")
print(f"   推理频率: {1000 / times.mean():.2f} Hz")

# 性能评级
if times.mean() < 100:
    grade = "⭐⭐⭐⭐⭐ 优秀 (A)"
elif times.mean() < 150:
    grade = "⭐⭐⭐⭐ 良好 (B)"
elif times.mean() < 200:
    grade = "⭐⭐⭐ 及格 (C)"
else:
    grade = "⭐⭐ 需优化 (D)"
print(f"   性能等级: {grade}")

# 步骤6：显存检查
print("\n" + "=" * 60)
print("步骤6：显存检查")
print("=" * 60)

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
total = torch.cuda.get_device_properties(0).total_memory / 1024**3

print(f"   已分配: {allocated:.2f} GB")
print(f"   已预留: {reserved:.2f} GB")
print(f"   总显存: {total:.2f} GB")
print(f"   使用率: {(allocated / total) * 100:.1f}%")

if allocated < 8.0:
    print("   等级: ✅ 优秀 (<8GB)")
elif allocated < 12.0:
    print("   等级: ⚠️  良好 (8-12GB)")
else:
    print("   等级: ❌ 偏高 (>12GB)")

# 最终总结
print("\n" + "🎉" * 30)
print("测试完成总结")
print("🎉" * 30)

print(f"\n📊 性能指标：")
print(f"   ✅ 推理速度: {times.mean():.2f}ms ({grade})")
print(f"   ✅ 显存占用: {allocated:.2f}GB")
print(f"   ✅ 推理频率: {1000/times.mean():.2f}Hz")

print(f"\n✅ 验收标准：")
达标项 = 0
总项 = 2

if times.mean() < 150:
    print(f"   ✅ 推理时间 < 150ms")
    达标项 += 1
else:
    print(f"   ❌ 推理时间 {times.mean():.2f}ms (目标<150ms)")

if allocated < 8.0:
    print(f"   ✅ 显存占用 < 8GB")
    达标项 += 1
else:
    print(f"   ⚠️  显存占用 {allocated:.2f}GB (目标<8GB)")

print(f"\n🎯 达标率: {达标项}/{总项} ({达标项/总项*100:.0f}%)")

if 达标项 == 总项:
    print("\n🎉 恭喜！所有指标达标，可以进入下一阶段！")
    print("📚 下一步：")
    print("   1. 运行模拟环境测试")
    print("   2. 填写项目进度清单")
    print("   3. 准备ROS2集成")
else:
    print("\n⚠️  部分指标未达标，但仍可继续")
    print("💡 优化建议：")
    if times.mean() >= 150:
        print("   - 使用torch.compile()加速")
        print("   - 减小图像分辨率")
    if allocated >= 8.0:
        print("   - 使用更低精度（INT8）")
        print("   - 减小batch size")
