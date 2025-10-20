"""
LeRobot数据集快速测试脚本（简化版）
适用于快速验证数据集是否可用
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path

def quick_test_pusht():
    """
    快速测试pusht数据集（最小的LeRobot数据集）
    """
    print("\n" + "="*60)
    print("快速测试: lerobot/pusht 数据集")
    print("="*60)
    
    try:
        from datasets import load_dataset
        
        # 1. 加载数据集
        print("\n[1/4] 加载数据集...")
        dataset = load_dataset("lerobot/pusht", split="train")
        print(f"✅ 加载成功! 共 {len(dataset)} 个样本")
        
        # 2. 检查数据结构
        print("\n[2/4] 检查数据结构...")
        sample = dataset[0]
        print("数据集字段:")
        for key in sample.keys():
            value = sample[key]
            if hasattr(value, 'shape'):
                print(f"  - {key}: {value.shape}")
            else:
                print(f"  - {key}: {type(value)}")
        
        # 3. 测试数据访问
        print("\n[3/4] 测试数据访问...")
        
        # 查找图像和状态
        image_keys = [k for k in sample.keys() if 'image' in k.lower()]
        state_keys = [k for k in sample.keys() if 'state' in k.lower() or 'qpos' in k.lower()]
        action_keys = [k for k in sample.keys() if 'action' in k.lower()]
        
        print(f"图像字段: {image_keys}")
        print(f"状态字段: {state_keys}")
        print(f"动作字段: {action_keys}")
        
        if image_keys:
            img = sample[image_keys[0]]
            print(f"  图像类型: {type(img)}, 形状: {np.array(img).shape if hasattr(img, 'shape') or isinstance(img, np.ndarray) else 'PIL Image'}")
        
        if action_keys:
            action = sample[action_keys[0]]
            print(f"  动作形状: {action.shape if hasattr(action, 'shape') else len(action)}")
        
        # 4. 测试SmolVLA兼容性
        print("\n[4/4] 测试SmolVLA兼容性...")
        
        try:
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            
            print("  加载SmolVLA模型...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            policy = SmolVLAPolicy.from_pretrained(
                "lerobot/smolvla_base",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device=device
            )
            policy.eval()
            
            # 准备观测数据
            obs = {}
            
            # 处理图像
            if image_keys:
                img_data = sample[image_keys[0]]
                if isinstance(img_data, np.ndarray):
                    obs["image"] = Image.fromarray(img_data)
                else:
                    obs["image"] = img_data
            else:
                obs["image"] = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            
            # 处理状态
            if state_keys:
                state_data = sample[state_keys[0]]
                if isinstance(state_data, np.ndarray):
                    obs["state"] = torch.from_numpy(state_data).float().to(device)
                else:
                    obs["state"] = torch.tensor(state_data).float().to(device)
            else:
                obs["state"] = torch.randn(7).to(device)
            
            # 执行推理
            with torch.no_grad():
                action = policy.select_action(obs, "push the block to the target")
            
            print(f"  ✅ 推理成功!")
            print(f"     输出形状: {action.shape}")
            print(f"     数值范围: [{action.min():.3f}, {action.max():.3f}]")
            
        except ImportError:
            print("  ⚠️  SmolVLA未安装，跳过兼容性测试")
        except Exception as e:
            print(f"  ❌ 兼容性测试失败: {e}")
        
        print("\n" + "="*60)
        print("✅ 快速测试完成!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_usage_examples():
    """
    显示数据集使用示例
    """
    print("\n" + "="*60)
    print("LeRobot数据集使用示例")
    print("="*60)
    
    examples = """
# 示例1: 加载数据集
from datasets import load_dataset
dataset = load_dataset("lerobot/pusht", split="train")
print(f"数据集大小: {len(dataset)}")

# 示例2: 访问单个样本
sample = dataset[0]
image = sample['observation.image']
action = sample['action']

# 示例3: 遍历数据集
for i in range(10):
    sample = dataset[i]
    # 处理样本...

# 示例4: 与SmolVLA一起使用
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

obs = {
    "image": sample['observation.image'],
    "state": torch.tensor(sample['observation.state'])
}
action = policy.select_action(obs, "your instruction here")

# 示例5: 训练SmolVLA
# 使用LeRobot训练脚本
python -m lerobot.scripts.train \\
    --dataset.repo_id=lerobot/pusht \\
    --policy.type=smolvla \\
    --output_dir=outputs/train/smolvla_pusht \\
    --policy.device=cuda

# 示例6: 下载多个数据集
datasets_to_download = [
    "lerobot/pusht",
    "lerobot/aloha_sim_insertion_human",
    "lerobot/aloha_sim_transfer_cube_human",
]

for dataset_name in datasets_to_download:
    dataset = load_dataset(dataset_name, split="train")
    print(f"{dataset_name}: {len(dataset)} samples")
"""
    
    print(examples)


def list_available_datasets():
    """
    列出可用的LeRobot数据集
    """
    print("\n" + "="*60)
    print("推荐的LeRobot数据集（按大小排序）")
    print("="*60)
    
    datasets = [
        {
            "name": "lerobot/pusht",
            "size": "~200 episodes, 25K frames",
            "description": "推动T形方块到目标位置",
            "difficulty": "⭐ 简单",
            "download_time": "1-2分钟"
        },
        {
            "name": "lerobot/aloha_sim_insertion_human",
            "size": "~50 episodes, 25K frames",
            "description": "ALOHA机器人插入任务",
            "difficulty": "⭐⭐ 中等",
            "download_time": "2-5分钟"
        },
        {
            "name": "lerobot/aloha_sim_transfer_cube_human",
            "size": "~50 episodes, 20K frames",
            "description": "ALOHA机器人转移立方体",
            "difficulty": "⭐⭐ 中等",
            "download_time": "2-5分钟"
        },
        {
            "name": "lerobot/xarm_lift_medium",
            "size": "~800 episodes, 20K frames",
            "description": "XArm提升任务",
            "difficulty": "⭐⭐ 中等",
            "download_time": "3-5分钟"
        },
        {
            "name": "lerobot/metaworld_mt50",
            "size": "~2500 episodes, 200K+ frames",
            "description": "MetaWorld 50个多任务",
            "difficulty": "⭐⭐⭐ 困难",
            "download_time": "10-20分钟"
        },
    ]
    
    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds['name']}")
        print(f"   大小: {ds['size']}")
        print(f"   描述: {ds['description']}")
        print(f"   难度: {ds['difficulty']}")
        print(f"   预计下载时间: {ds['download_time']}")
    
    print("\n" + "="*60)
    print("💡 提示:")
    print("  - 建议从 lerobot/pusht 开始测试")
    print("  - 数据集会自动缓存到 ~/.cache/huggingface/datasets/")
    print("  - 完整数据集列表: https://huggingface.co/lerobot")
    print("="*60)


def main():
    """
    主函数
    """
    print("\n" + "🤖"*30)
    print("LeRobot数据集快速测试工具")
    print("🤖"*30)
    
    # 显示可用数据集
    list_available_datasets()
    
    # 询问是否开始测试
    print("\n" + "="*60)
    response = input("是否开始测试 lerobot/pusht 数据集? (y/n): ").strip().lower()
    
    if response == 'y':
        success = quick_test_pusht()
        
        if success:
            print("\n✅ 测试成功! 你可以:")
            print("  1. 查看更多使用示例: 运行 show_usage_examples()")
            print("  2. 运行完整测试: python download_and_test_dataset.py")
            print("  3. 开始训练: python -m lerobot.scripts.train --dataset.repo_id=lerobot/pusht ...")
        else:
            print("\n❌ 测试失败，请检查:")
            print("  1. 网络连接是否正常")
            print("  2. 是否安装了所有依赖: pip install datasets huggingface_hub")
            print("  3. 是否有足够的磁盘空间")
    else:
        print("\n跳过测试。稍后可以手动运行:")
        print("  python quick_test_dataset.py")
    
    # 显示使用示例
    print("\n" + "="*60)
    response = input("是否显示使用示例? (y/n): ").strip().lower()
    if response == 'y':
        show_usage_examples()
    
    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
