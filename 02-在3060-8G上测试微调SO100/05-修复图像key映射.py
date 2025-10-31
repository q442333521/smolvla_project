#!/usr/bin/env python3
"""
修复图像key映射的wrapper
将SO100的图像keys映射到SmolVLA期望的格式
"""

import sys
sys.path.insert(0, "/root/lerobot_project/lerobot/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset

class ImageKeyRemapDataset(Dataset):
    """包装数据集，重映射图像keys"""
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        # 重映射图像keys
        if "observation.images.top" in sample:
            sample["observation.images.camera1"] = sample.pop("observation.images.top")
        
        if "observation.images.wrist" in sample:
            sample["observation.images.camera2"] = sample["observation.images.wrist"]
            sample["observation.images.camera3"] = sample.pop("observation.images.wrist")
        
        return sample
    
    @property
    def num_episodes(self):
        return self.base_dataset.num_episodes
    
    @property  
    def meta(self):
        return self.base_dataset.meta

# 测试
if __name__ == "__main__":
    print("加载数据集...")
    base_ds = LeRobotDataset(
        repo_id="lerobot/svla_so100_pickplace",
        root="/tmp/lerobot_datasets"
    )
    
    print("创建映射数据集...")
    mapped_ds = ImageKeyRemapDataset(base_ds)
    
    print(f"✅ 数据集准备完成: {len(mapped_ds)} 个样本")
    
    # 测试一个样本
    sample = mapped_ds[0]
    print(f"\n✅ 映射后的keys:")
    for key in sorted(sample.keys()):
        if 'image' in key.lower():
            print(f"  - {key}: {sample[key].shape}")
