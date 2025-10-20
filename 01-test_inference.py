"""
SmolVLA 本地推理测试脚本 v2（完全修复版）
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
        
        # 打印模型期望的相机配置
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

def pil_to_tensor(pil_image, size=(256, 256)):
    """将PIL图像转换为模型需要的tensor格式"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),  # 转为 (C, H, W) 格式，范围 [0, 1]
    ])
    return transform(pil_image)

def test_dummy_inference_simple(policy):
    """测试2：虚拟数据推理（简单版）"""
    print("\n" + "=" * 60)
    print("测试2：虚拟数据推理（Tensor输入）")
    print("=" * 60)
    
    try:
        # 方法1：直接创建tensor（最简单）
        dummy_image = torch.rand(3, 256, 256).cuda().half()
        dummy_state = torch.randn(7).cuda().half()
        
        # 使用模型期望的键名
        observation = {
            "observation.images.camera1": dummy_image,
            "observation.state": dummy_state,
        }
        
        print(f"   图像形状: {dummy_image.shape}")
        print(f"   状态维度: {dummy_state.shape}")
        
        # 推理
        print("   开始推理...")
        start_time = time.time()
        
        with torch.no_grad():
            actions = policy.select_action(observation)
        
        inference_time = time.time() - start_time
        
        print(f"✅ 推理成功")
        print(f"   推理时间: {inference_time * 1000:.2f}ms")
        print(f"   输出形状: {actions.shape}")
        print(f"   动作范围: [{actions.min().item():.3f}, {actions.max().item():.3f}]")
        
        return actions
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_pil_image_inference(policy):
    """测试3：PIL图像推理"""
    print("\n" + "=" * 60)
    print("测试3：PIL图像推理")
    print("=" * 60)
    
    try:
        # 创建PIL图像
        pil_image = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        
        # 转换为tensor
        image_tensor = pil_to_tensor(pil_image).cuda().half()
        dummy_state = torch.randn(7).cuda().half()
        
        observation = {
            "observation.images.camera1": image_tensor,
            "observation.state": dummy_state,
        }
        
        print(f"   PIL图像尺寸: {pil_image.size}")
        print(f"   转换后形状: {image_tensor.shape}")
        
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

def test_multiple_cameras(policy):
    """测试4：多相机输入"""
    print("\n" + "=" * 60)
    print("测试4：多相机输入")
    print("=" * 60)
    
    try:
        # 创建多个相机的图像
        camera1 = torch.rand(3, 256, 256).cuda().half()
        camera2 = torch.rand(3, 256, 256).cuda().half()
        camera3 = torch.rand(3, 256, 256).cuda().half()
        dummy_state = torch.randn(7).cuda().half()
        
        observation = {
            "observation.images.camera1": camera1,
            "observation.images.camera2": camera2,
            "observation.images.camera3": camera3,
            "observation.state": dummy_state,
        }
        
        print(f"   相机数量: 3")
        print(f"   每个相机形状: {camera1.shape}")
        
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
            print("✅ 显存使用正常")
        else:
            print("⚠️  显存使用较高")

def main():
    """主测试流程"""
    print("\n" + "🚀" * 30)
    print("SmolVLA 本地推理完整测试 v2")
    print("🚀" * 30 + "\n")
    
    try:
        # 测试1: 加载模型
        policy = test_model_loading()
        
        # 测试2: Tensor输入推理
        test_dummy_inference_simple(policy)
        
        # 测试3: PIL图像推理
        test_pil_image_inference(policy)
        
        # 测试4: 多相机
        test_multiple_cameras(policy)
        
        # 测试5: 显存
        test_gpu_memory()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()