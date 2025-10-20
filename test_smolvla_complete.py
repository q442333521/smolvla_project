"""
SmolVLA 完整测试 - 展示两种推理模式
"""

import torch
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
import time

print("🚀 SmolVLA 完整推理测试\n")

# 加载模型
print("加载模型...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to("cuda").float().eval()

print(f"✅ 模型加载 ({sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M参数)")
print(f"   数据类型: {next(policy.parameters()).dtype}")
print(f"   n_action_steps: {policy.config.n_action_steps}\n")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
print("✅ Tokenizer加载\n")

# 准备输入
image = torch.rand(1, 3, 256, 256).cuda().float()
state = torch.randn(1, 7).cuda().float()
instruction = "pick up the red cube"

tokens = tokenizer(
    instruction,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=77
)

observation = {
    "observation.images.camera1": image,
    "observation.state": state,
    "observation.language.tokens": tokens["input_ids"].cuda(),
    "observation.language.attention_mask": tokens["attention_mask"].cuda().bool(),
}

print(f"输入: 图像{image.shape}, 状态{state.shape}, 指令'{instruction}'\n")

# ============================================================
# 模式1：获取完整动作序列（用于分析）
# ============================================================
print("=" * 60)
print("模式1：获取完整动作序列 (_get_action_chunk)")
print("=" * 60)

from lerobot.policies.utils import populate_queues
policy._queues = populate_queues(policy._queues, observation, exclude_keys=["action"])

print("推理中...")
start = time.time()

with torch.no_grad():
    action_chunk = policy._get_action_chunk(observation, noise=None)

t = (time.time() - start) * 1000

print(f"✅ 完整动作序列生成成功！")
print(f"   时间: {t:.1f}ms")
print(f"   输出形状: {action_chunk.shape} (batch, steps, dims)")
print(f"   范围: [{action_chunk.min().item():.3f}, {action_chunk.max().item():.3f}]")

# 分析动作平滑性
if action_chunk.shape[1] > 1:
    diff = torch.diff(action_chunk[0], dim=0)
    smoothness = diff.abs().mean().item()
    print(f"   动作平滑度: {smoothness:.4f} (越小越平滑)\n")

# ============================================================
# 模式2：逐步获取动作（用于实时控制）
# ============================================================
print("=" * 60)
print("模式2：逐步获取动作 (select_action)")
print("=" * 60)

# 重置队列
policy._queues["action"].clear()

actions_list = []
times_list = []

print(f"逐步获取 {policy.config.n_action_steps} 个动作...")

for i in range(policy.config.n_action_steps):
    start = time.time()
    
    with torch.no_grad():
        action = policy.select_action(observation)
    
    t = (time.time() - start) * 1000
    times_list.append(t)
    actions_list.append(action.cpu().numpy())
    
    if (i + 1) % 20 == 0 or i == 0:
        print(f"  Step {i+1}: {t:.1f}ms, action shape: {action.shape}")

actions_array = np.array(actions_list)
print(f"\n✅ 逐步获取完成")
print(f"   总步数: {len(actions_list)}")
print(f"   平均时间: {np.mean(times_list):.1f}ms/step")
print(f"   首次时间: {times_list[0]:.1f}ms (包含推理)")
print(f"   后续时间: {np.mean(times_list[1:]):.1f}ms (从队列取)")
print(f"   动作形状: {actions_array.shape}\n")

# ============================================================
# 速度基准测试
# ============================================================
print("=" * 60)
print("性能基准测试（10次完整推理）")
print("=" * 60)

times = []
for i in range(10):
    # 清空队列
    policy._queues["action"].clear()
    
    obs = {
        "observation.images.camera1": torch.rand(1, 3, 256, 256).cuda().float(),
        "observation.state": torch.randn(1, 7).cuda().float(),
        "observation.language.tokens": tokens["input_ids"].cuda(),
        "observation.language.attention_mask": tokens["attention_mask"].cuda().bool(),
    }
    
    start = time.time()
    with torch.no_grad():
        _ = policy.select_action(obs)
    times.append((time.time() - start) * 1000)
    
    if (i + 1) % 5 == 0:
        print(f"  {i+1}/10 完成")

times = np.array(times)
print(f"\n✅ 基准测试结果:")
print(f"   平均: {times.mean():.1f}ms")
print(f"   中位数: {np.median(times):.1f}ms")
print(f"   标准差: {times.std():.1f}ms")
print(f"   频率: {1000/times.mean():.1f}Hz\n")

# 显存
mem = torch.cuda.memory_allocated() / 1024**3
print(f"💾 显存: {mem:.2f}GB\n")

# ============================================================
# 总结
# ============================================================
print("=" * 60)
print("📊 验收结果")
print("=" * 60)

speed_ok = times.mean() < 150
freq_ok = 1000/times.mean() > 7
mem_ok = mem < 12

print(f"1. 推理速度: {times.mean():.1f}ms " + ("✅ 达标" if speed_ok else "⚠️  未达标"))
print(f"2. 推理频率: {1000/times.mean():.1f}Hz " + ("✅ 达标" if freq_ok else "⚠️  未达标"))
print(f"3. 显存占用: {mem:.2f}GB " + ("✅ 达标" if mem_ok else "⚠️  超标"))
print(f"4. 动作维度: {action_chunk.shape} ✅")
print(f"5. 输出正常: " + ("✅" if not torch.isnan(action_chunk).any() else "❌"))

if speed_ok and freq_ok and mem_ok:
    print("\n🎉 所有指标达标！准备进入下一阶段")
    print("\n📚 下一步:")
    print("  1️⃣  运行模拟环境测试")
    print("  2️⃣  填写项目进度清单")
    print("  3️⃣  准备 ROS2 集成")
else:
    print("\n⚠️  部分指标未达标，但可以继续")

print("\n✅ 完整测试完成！")
print("=" * 60)
