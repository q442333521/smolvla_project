#!/usr/bin/env python3
"""
简单的 LeRobot 数据集下载程序
用法: python simple_download.py
"""

import sys
import torch
from datasets import load_dataset
from pathlib import Path

# 添加 lerobot 路径
sys.path.insert(0, '/root/smolvla_project/lerobot/src')

print("="*60)
print("LeRobot 数据集下载程序")
print("="*60)

# 要下载的数据集列表
DATASETS = [
    "lerobot/pusht",
    "lerobot/aloha_sim_insertion_human",
]

def download_dataset(dataset_name):
    """下载单个数据集"""
    print(f"\n{'='*60}")
    print(f"下载: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        print("正在下载...")
        dataset = load_dataset(dataset_name, split="train")
        print(f"✅ 成功! 样本数: {len(dataset)}")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False

def main():
    """主函数"""
    print(f"\n将下载 {len(DATASETS)} 个数据集")
    print(f"保存位置: ~/.cache/huggingface/datasets/\n")
    
    results = {}
    
    for dataset_name in DATASETS:
        success = download_dataset(dataset_name)
        results[dataset_name] = "✅ 成功" if success else "❌ 失败"
    
    # 打印总结
    print("\n" + "="*60)
    print("下载总结")
    print("="*60)
    for name, status in results.items():
        print(f"{name}: {status}")
    
    print("\n✅ 完成!")

if __name__ == "__main__":
    main()
