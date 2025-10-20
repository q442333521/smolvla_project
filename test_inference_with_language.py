"""
SmolVLA 完整推理测试（包含语言指令）
"""

import torch
import numpy as np
from PIL import Image
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import SmolVLAProcessor
from torchvision import transforms
import time

def test_model_and_processor_loading():
    """测试1：加载模型和处理器"""
    print("=" * 60)
    print("测试1：加载SmolVLA模型和处理器")
    print("=" * 60)
    
    try:
        print("正在加载模型...")
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        policy = policy.to("cuda").half().eval()
        
        print("正在加载处理器...")
        processor = SmolVLAProcessor.from_pretrained("lerobot/smolvla_base")
        
        print("✅ 加载成功")
        print(f"   模型参数: {sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M")
        print(f"   模型设备: {next(policy.parameters()).device}")
        print(f"   模型精度: {next(policy.parameters()).dtype}")
        
        return policy, processor
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_with_processor(policy, processor):
    """测试2：使用processor处理输入"""
    print("\n" + "=" * 60)
    print("测试2：使用Processor处理输入")
    print("=" * 60)
    
    try:
        # 创建PIL图像
        pil_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        
        # 语言指令
        instruction = "pick up the red cube"
        
        print(f"   图像大小: {pil_image.size}")
        print(f"   指令: '{instruction}'")
        
        # 使用processor处理
        print("   处理输入...")
        processed = processor(
            images=[pil_image],
            text=[instruction]
        )
        
        print(f"   处理后的键: {processed.keys()}")
        
        # 移到GPU并转换类型
        observation = {}
        for key, value in processed.items():
            if isinstance(value, torch.Tensor):
                observation[key] = value.cuda().half()
            else:
                observation[key] = value
        
        # 添加状态（如果需要）
        observation["observation.state"] = torch.randn(1, 7).cuda().half()
        
        print("   开始推理...")
        start_time = time.time()
        
        with torch.no_grad():
            actions = policy.select_action(observation)
        
        inference_time = time.time() - start_time
        
        print(f"✅ 推理成功！")
        print(f"   推理时间: {inference_time * 1000:.2f}ms")
        print(f"   输出形状: {actions.shape}")
        print(f"   动作范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
        
        return actions
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_manual_preparation(policy, processor):
    """测试3：手动准备输入（不使用processor）"""
    print("\n" + "=" * 60)
    print("测试3：手动准备输入")
    print("=" * 60)
    
    try:
        # 创建图像tensor
        image = torch.rand(1, 3, 256, 256).cuda().half()
        state = torch.randn(1, 7).cuda().half()
        
        # 手动创建language tokens（使用tokenizer）
        instruction = "move to target position"
        
        # 使用processor的tokenizer
        tokens = processor.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # 准备observation
        observation = {
            "observation.images.camera1": image,
            "observation.state": state,
            "observation.language.tokens": tokens["input_ids"].cuda(),
        }
        
        print(f"   图像形状: {image.shape}")
        print(f"   状态形状: {state.shape}")
        print(f"   Token形状: {tokens['input_ids'].shape}")
        print(f"   指令: '{instruction}'")
        
        # 推理
        start_time = time.time()
        
        with torch.no_grad():
            actions = policy.select_action(observation)
        
        inference_time = time.time() - start_time
        
        print(f"✅ 推理成功")
        print(f"   推理时间: {inference_time * 1000:.2f}ms")
        
        return actions
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_speed(policy, processor, num_iterations=10):
    """测试4：推理速度"""
    print("\n" + "=" * 60)
    print(f"测试4：推理速度（{num_iterations}次）")
    print("=" * 60)
    
    times = []
    instructions = [
        "pick up the red cube",
        "place object in box",
        "move to target position",
        "grasp the bottle",
        "push the button"
    ]
    
    for i in range(num_iterations):
        # 创建输入
        image = torch.rand(1, 3, 256, 256).cuda().half()
        state = torch.randn(1, 7).cuda().half()
        instruction = instructions[i % len(instructions)]
        
        tokens = processor.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        observation = {
            "observation.images.camera1": image,
            "observation.state": state,
            "observation.language.tokens": tokens["input_ids"].cuda(),
        }
        
        # 推理
        start = time.time()
        with torch.no_grad():
            _ = policy.select_action(observation)
        times.append(time.time() - start)
        
        if (i + 1) % 5 == 0:
            print(f"   进度: {i+1}/{num_iterations}")
    
    times = np.array(times) * 1000
    
    print(f"\n✅ 速度测试完成")
    print(f"   平均: {times.mean():.2f}ms")
    print(f"   中位数: {np.median(times):.2f}ms")
    print(f"   最小: {times.min():.2f}ms")
    print(f"   最大: {times.max():.2f}ms")
    print(f"   推理频率: {1000 / times.mean():.2f} Hz")
    
    # 性能评估
    if times.mean() < 100:
        print("   性能等级: ⭐⭐⭐⭐⭐ 优秀 (A)")
    elif times.mean() < 150:
        print("   性能等级: ⭐⭐⭐⭐ 良好 (B)")
    elif times.mean() < 200:
        print("   性能等级: ⭐⭐⭐ 及格 (C)")
    else:
        print("   性能等级: ⭐⭐ 需要优化 (D)")
    
    return times

def test_gpu_memory():
    """测试5：显存使用"""
    print("\n" + "=" * 60)
    print("测试5：GPU显存使用")
    print("=" * 60)
    
    if torch.cuda.is_available():
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

def main():
    """主测试流程"""
    print("\n" + "🎯" * 30)
    print("SmolVLA 完整推理测试（包含语言指令）")
    print("🎯" * 30 + "\n")
    
    try:
        # 测试1: 加载
        policy, processor = test_model_and_processor_loading()
        
        # 测试2: 使用processor
        test_with_processor(policy, processor)
        
        # 测试3: 手动准备
        test_manual_preparation(policy, processor)
        
        # 测试4: 速度
        test_speed(policy, processor, num_iterations=10)
        
        # 测试5: 显存
        test_gpu_memory()
        
        print("\n" + "=" * 60)
        print("🎉🎉🎉 所有测试完成！")
        print("=" * 60)
        print("\n📊 性能总结：")
        print("- 如果推理时间 < 150ms 且显存 < 8GB → 性能达标 ✅")
        print("- 可以进入下一阶段：模拟环境测试")
        print("\n📚 下一步：")
        print("1. 查看文档：01-smolvla本地稳定复现.md 第四章")
        print("2. 运行模拟环境测试")
        print("3. 填写项目进度清单")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ 测试失败")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        print("\n💡 常见问题：")
        print("1. 模型下载失败 → 使用 HF 镜像")
        print("2. CUDA不可用 → 检查 nvidia-smi")
        print("3. 导入失败 → 重新 pip install -e .")

if __name__ == "__main__":
    main()
