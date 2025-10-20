"""
SmolVLA 本地推理测试脚本（最终修复版）
关键：需要 batch 维度！
"""

import torch
import numpy as np
from PIL import Image
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from torchvision import transforms
import time

def test_model_loading():
    """测试1：模型加载"""
    print("=" * 60)
    print("测试1：加载SmolVLA预训练模型")
    print("=" * 60)
    
    try:
        print("正在下载模型...")
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
        
        # 移到GPU并转换精度
        policy = policy.to("cuda")
        policy = policy.half()
        policy.eval()
        
        print("✅ 模型加载成功")
        print(f"   模型设备: {next(policy.parameters()).device}")
        print(f"   模型精度: {next(policy.parameters()).dtype}")
        
        # 打印模型期望的输入
        print(f"   期望的相机: {list(policy.config.input_features.keys())}")
        
        # 统计参数量
        total_params = sum(p.numel() for p in policy.parameters())
        print(f"   总参数量: {total_params / 1e6:.2f}M")
        
        return policy
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_dummy_inference(policy):
    """测试2：虚拟数据推理（正确的维度）"""
    print("\n" + "=" * 60)
    print("测试2：虚拟数据推理")
    print("=" * 60)
    
    try:
        # 关键：需要 (batch, channel, height, width) 格式！
        # batch=1, channel=3, height=256, width=256
        dummy_image = torch.rand(1, 3, 256, 256).cuda().half()  # ← 注意这里是 4 维！
        dummy_state = torch.randn(1, 7).cuda().half()  # ← state 也要 batch 维度
        
        # 使用模型期望的键名
        observation = {
            "observation.images.camera1": dummy_image,
            "observation.state": dummy_state,
        }
        
        print(f"   图像形状: {dummy_image.shape} (batch, c, h, w)")
        print(f"   状态维度: {dummy_state.shape} (batch, state_dim)")
        
        # 推理
        print("   开始推理...")
        start_time = time.time()
        
        with torch.no_grad():
            actions = policy.select_action(observation)
        
        inference_time = time.time() - start_time
        
        print(f"✅ 推理成功！")
        print(f"   推理时间: {inference_time * 1000:.2f}ms")
        print(f"   输出形状: {actions.shape}")
        print(f"   动作范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
        
        # 检查输出
        if torch.isnan(actions).any():
            print("⚠️  警告: 输出包含NaN")
        if torch.isinf(actions).any():
            print("⚠️  警告: 输出包含Inf")
            
        return actions
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_pil_to_tensor(policy):
    """测试3：从PIL图像推理"""
    print("\n" + "=" * 60)
    print("测试3：PIL图像转换推理")
    print("=" * 60)
    
    try:
        # 创建PIL图像
        pil_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        
        # 转换为tensor并添加batch维度
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # 输出 (3, 256, 256)
        ])
        
        image_tensor = transform(pil_image)  # (3, 256, 256)
        image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度 → (1, 3, 256, 256)
        image_tensor = image_tensor.cuda().half()
        
        dummy_state = torch.randn(1, 7).cuda().half()
        
        observation = {
            "observation.images.camera1": image_tensor,
            "observation.state": dummy_state,
        }
        
        print(f"   PIL图像: {pil_image.size}")
        print(f"   转换后: {image_tensor.shape}")
        
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

def test_batch_inference(policy):
    """测试4：批量推理"""
    print("\n" + "=" * 60)
    print("测试4：批量推理（batch_size=4）")
    print("=" * 60)
    
    try:
        batch_size = 4
        
        # 创建批量数据
        images = torch.rand(batch_size, 3, 256, 256).cuda().half()
        states = torch.randn(batch_size, 7).cuda().half()
        
        observation = {
            "observation.images.camera1": images,
            "observation.state": states,
        }
        
        print(f"   Batch size: {batch_size}")
        print(f"   图像形状: {images.shape}")
        print(f"   状态形状: {states.shape}")
        
        # 推理
        start_time = time.time()
        
        with torch.no_grad():
            actions = policy.select_action(observation)
        
        inference_time = time.time() - start_time
        
        print(f"✅ 推理成功")
        print(f"   总时间: {inference_time * 1000:.2f}ms")
        print(f"   平均时间: {inference_time * 1000 / batch_size:.2f}ms/sample")
        print(f"   输出形状: {actions.shape}")
        
        return actions
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_speed(policy, num_iterations=20):
    """测试5：推理速度"""
    print("\n" + "=" * 60)
    print(f"测试5：推理速度（{num_iterations}次）")
    print("=" * 60)
    
    times = []
    
    for i in range(num_iterations):
        # 创建随机输入
        image = torch.rand(1, 3, 256, 256).cuda().half()
        state = torch.randn(1, 7).cuda().half()
        
        observation = {
            "observation.images.camera1": image,
            "observation.state": state,
        }
        
        # 推理
        start = time.time()
        with torch.no_grad():
            _ = policy.select_action(observation)
        times.append(time.time() - start)
        
        if (i + 1) % 5 == 0:
            print(f"   进度: {i+1}/{num_iterations}")
    
    times = np.array(times) * 1000  # 转换为ms
    
    print(f"\n✅ 速度测试完成")
    print(f"   平均: {times.mean():.2f}ms")
    print(f"   中位数: {np.median(times):.2f}ms")
    print(f"   最小: {times.min():.2f}ms")
    print(f"   最大: {times.max():.2f}ms")
    print(f"   标准差: {times.std():.2f}ms")
    print(f"   推理频率: {1000 / times.mean():.2f} Hz")
    
    return times

def test_gpu_memory():
    """测试6：显存使用"""
    print("\n" + "=" * 60)
    print("测试6：GPU显存使用")
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
            print("✅ 显存使用正常 (<8GB)")
        elif allocated < 12.0:
            print("⚠️  显存使用偏高 (8-12GB)")
        else:
            print("❌ 显存使用过高 (>12GB)")

def main():
    """主测试流程"""
    print("\n" + "🚀" * 30)
    print("SmolVLA 本地推理完整测试（最终版）")
    print("🚀" * 30 + "\n")
    
    try:
        # 测试1: 加载模型
        policy = test_model_loading()
        
        # 测试2: 基础推理
        test_dummy_inference(policy)
        
        # 测试3: PIL图像
        test_pil_to_tensor(policy)
        
        # 测试4: 批量推理
        test_batch_inference(policy)
        
        # 测试5: 速度测试
        test_speed(policy, num_iterations=20)
        
        # 测试6: 显存
        test_gpu_memory()
        
        print("\n" + "=" * 60)
        print("✅✅✅ 所有测试完成！SmolVLA本地推理稳定运行 ✅✅✅")
        print("=" * 60)
        print("\n下一步：")
        print("1. 查看性能是否达标（推理<150ms，显存<8GB）")
        print("2. 如果达标，可以进入模拟环境测试阶段")
        print("3. 参考文档：01-smolvla本地稳定复现.md")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败")
        print("=" * 60)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
