"""
快速测试模型加载是否正常
"""
import torch
import sys

print("="*60)
print("测试 SmolVLA 模型加载 (修复后)")
print("="*60)

try:
    # 添加路径
    sys.path.insert(0, '/root/smolvla_project/lerobot/src')
    
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    
    print("\n1. 测试错误的加载方式 (应该失败)...")
    try:
        policy_wrong = SmolVLAPolicy.from_pretrained(
            "lerobot/smolvla_base",
            torch_dtype=torch.float16,
            device="cuda"
        )
        print("   ❌ 意外成功 - 这不应该发生!")
    except TypeError as e:
        print(f"   ✅ 预期的错误: {e}")
    
    print("\n2. 测试正确的加载方式...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 正确方式: 不传递 torch_dtype 和 device
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    
    # 然后手动设置
    policy = policy.to(device).float()
    policy.eval()
    
    print(f"   ✅ 模型加载成功!")
    print(f"   - Device: {device}")
    print(f"   - Dtype: {next(policy.parameters()).dtype}")
    
    print("\n3. 测试推理...")
    from PIL import Image
    import numpy as np
    
    # 创建虚拟输入
    obs = {
        "image": Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)),
        "state": torch.randn(7).to(device)
    }
    
    with torch.no_grad():
        action = policy.select_action(
            observation=obs,
            instruction="test"
        )
    
    print(f"   ✅ 推理成功!")
    print(f"   - 输出形状: {action.shape}")
    print(f"   - 输出范围: [{action.min():.3f}, {action.max():.3f}]")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过! 修复有效!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
