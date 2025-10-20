"""
SmolVLA 简单测试 - Float32
"""

import torch
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoTokenizer
import time

print("🚀 SmolVLA 推理测试\n")

# 加载模型
print("加载模型...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = policy.to("cuda").float().eval()  # 转换为 float32

print(f"✅ 模型加载 ({sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M参数)")
print(f"   数据类型: {next(policy.parameters()).dtype}\n")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
print("✅ Tokenizer加载\n")

# 准备输入
image = torch.rand(1, 3, 256, 256).cuda().float()
state = torch.randn(1, 7).cuda().float()
instruction = "pick up the red cube"

# Token化
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

# 推理
print("🔥 推理中（首次较慢，请耐心等待）...")
start = time.time()

with torch.no_grad():
    actions = policy.select_action(observation)

t = (time.time() - start)

print(f"\n✅ 推理成功！")
print(f"   时间: {t:.2f}秒 ({t*1000:.1f}ms)")
print(f"   输出: {actions.shape}")
print(f"   范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]\n")

# 速度测试
print("速度测试（10次）...")
times = []
for i in range(10):
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
print(f"\n✅ 平均: {times.mean():.1f}ms, 频率: {1000/times.mean():.1f}Hz")

# 显存
mem = torch.cuda.memory_allocated() / 1024**3
print(f"💾 显存: {mem:.2f}GB\n")

# 总结
print("=" * 60)
print("📊 结果")
print("=" * 60)
print(f"推理速度: {times.mean():.1f}ms " + ("✅" if times.mean() < 150 else "⚠️"))
print(f"显存占用: {mem:.2f}GB " + ("✅" if mem < 12 else "⚠️"))
print("\n🎉 测试完成！")
