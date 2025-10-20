# LeRobot数据集下载和测试指南

本目录包含用于下载和测试LeRobot Community Datasets的脚本。

## 📁 文件说明

### 1. `quick_test_dataset.py` - 快速测试脚本（推荐新手使用）
- **用途**: 快速测试单个数据集（pusht）
- **运行时间**: 2-5分钟
- **适合场景**: 
  - 第一次使用LeRobot数据集
  - 快速验证环境是否正确
  - 检查数据集结构

### 2. `download_and_test_dataset.py` - 完整测试脚本
- **用途**: 下载多个数据集并进行完整测试
- **运行时间**: 10-30分钟（取决于数据集数量）
- **适合场景**:
  - 需要下载多个数据集
  - 完整的兼容性测试
  - 生成可视化报告

## 🚀 快速开始

### 步骤1: 准备环境

```bash
# 在WSL2中，确保已激活smolvla环境
conda activate smolvla

# 安装额外依赖（如果还没安装）
pip install datasets huggingface_hub matplotlib
```

### 步骤2: 运行快速测试

```bash
# 复制脚本到WSL2（如果在Windows侧）
# 方法1: 直接在WSL2中下载
cd ~/smolvla_project
wget <脚本URL> 或手动复制

# 方法2: 从Windows复制到WSL2
# 假设脚本在 D:/quick_test_dataset.py
cp /mnt/d/quick_test_dataset.py ~/smolvla_project/

# 运行快速测试
python quick_test_dataset.py
```

**预期输出**:
```
🤖🤖🤖...
LeRobot数据集快速测试工具
🤖🤖🤖...

推荐的LeRobot数据集（按大小排序）
===========================================================

1. lerobot/pusht
   大小: ~200 episodes, 25K frames
   描述: 推动T形方块到目标位置
   ...

是否开始测试 lerobot/pusht 数据集? (y/n): y

============================================================
快速测试: lerobot/pusht 数据集
============================================================

[1/4] 加载数据集...
✅ 加载成功! 共 25650 个样本

[2/4] 检查数据结构...
...

[4/4] 测试SmolVLA兼容性...
  ✅ 推理成功!
     输出形状: torch.Size([100, 2])
     数值范围: [-0.523, 0.487]

✅ 快速测试完成!
```

### 步骤3: 运行完整测试（可选）

```bash
python download_and_test_dataset.py
```

这将：
1. 下载 `lerobot/pusht` 和 `lerobot/aloha_sim_insertion_human`
2. 测试数据集结构
3. 测试SmolVLA兼容性
4. 生成可视化图像
5. 保存测试报告到 `dataset_test_results.json`

## 📊 可用的LeRobot数据集

### 推荐用于SmolVLA测试的数据集

| 数据集 | 大小 | 任务类型 | 下载时间 | 推荐度 |
|-------|------|---------|----------|--------|
| `lerobot/pusht` | 25K frames | 推动任务 | 1-2分钟 | ⭐⭐⭐⭐⭐ |
| `lerobot/aloha_sim_insertion_human` | 25K frames | 插入任务 | 2-5分钟 | ⭐⭐⭐⭐ |
| `lerobot/aloha_sim_transfer_cube_human` | 20K frames | 转移任务 | 2-5分钟 | ⭐⭐⭐⭐ |
| `lerobot/xarm_lift_medium` | 20K frames | 提升任务 | 3-5分钟 | ⭐⭐⭐ |
| `lerobot/metaworld_mt50` | 200K+ frames | 多任务 | 10-20分钟 | ⭐⭐ |

### 查看所有数据集
访问: https://huggingface.co/lerobot

## 💡 使用示例

### 示例1: 简单加载数据集

```python
from datasets import load_dataset

# 加载数据集（自动下载并缓存）
dataset = load_dataset("lerobot/pusht", split="train")

# 查看第一个样本
sample = dataset[0]
print(sample.keys())  # 查看所有字段

# 访问图像和动作
image = sample['observation.image']
action = sample['action']
```

### 示例2: 与SmolVLA一起使用

```python
import torch
from PIL import Image
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("lerobot/pusht", split="train")
sample = dataset[0]

# 加载SmolVLA
policy = SmolVLAPolicy.from_pretrained(
    "lerobot/smolvla_base",
    torch_dtype=torch.float16,
    device="cuda"
)

# 准备观测
obs = {
    "image": sample['observation.image'],  # PIL Image
    "state": torch.tensor(sample['observation.state']).cuda()
}

# 推理
action = policy.select_action(obs, "push the block to target")
print(f"Action: {action.shape}")  # [100, 2]
```

### 示例3: 训练SmolVLA

```bash
python -m lerobot.scripts.train \
    --dataset.repo_id=lerobot/pusht \
    --policy.type=smolvla \
    --output_dir=outputs/train/smolvla_pusht \
    --job_name=smolvla_pusht \
    --policy.device=cuda \
    --train.num_epochs=100 \
    --train.batch_size=8 \
    --wandb.enable=true
```

### 示例4: 下载多个数据集

```python
from datasets import load_dataset

datasets_to_download = [
    "lerobot/pusht",
    "lerobot/aloha_sim_insertion_human",
    "lerobot/aloha_sim_transfer_cube_human",
]

for dataset_name in datasets_to_download:
    print(f"\n下载: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    print(f"✅ {dataset_name}: {len(dataset)} samples")
```

## 🔧 故障排除

### 问题1: 下载速度慢

**解决方案**:
```bash
# 使用HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com
pip install hf-mirror
```

### 问题2: 显存不足

**解决方案**:
```python
# 使用更小的batch size
--train.batch_size=2

# 使用FP16
policy = SmolVLAPolicy.from_pretrained(
    "lerobot/smolvla_base",
    torch_dtype=torch.float16
)
```

### 问题3: 数据集加载失败

**解决方案**:
```bash
# 清除缓存重新下载
rm -rf ~/.cache/huggingface/datasets/lerobot*

# 重新运行测试脚本
python quick_test_dataset.py
```

### 问题4: 找不到某个字段

**检查数据集结构**:
```python
from datasets import load_dataset
dataset = load_dataset("lerobot/pusht", split="train")
sample = dataset[0]

# 打印所有字段
for key in sample.keys():
    print(f"{key}: {type(sample[key])}")
```

## 📝 测试结果解读

### 成功标志
- ✅ 数据集下载成功
- ✅ 数据结构验证通过
- ✅ SmolVLA推理成功
- ✅ 输出动作形状正确（通常是 [100, action_dim]）

### 警告标志
- ⚠️  找不到某些字段但有替代字段
- ⚠️  使用虚拟数据进行测试

### 失败标志
- ❌ 下载失败 - 检查网络
- ❌ 导入失败 - 检查依赖安装
- ❌ 推理失败 - 检查数据格式

## 📚 相关资源

- **LeRobot GitHub**: https://github.com/huggingface/lerobot
- **LeRobot数据集**: https://huggingface.co/lerobot
- **数据集可视化工具**: https://huggingface.co/spaces/lerobot/visualize_dataset
- **SmolVLA论文**: https://huggingface.co/papers/2506.01844

## 🎯 下一步

完成数据集测试后，你可以:

1. **继续本地复现测试** - 返回 `test_inference.py`
2. **开始训练SmolVLA** - 使用训练脚本
3. **进入ROS2集成** - 按照项目计划

## ❓ 常见问题

**Q: 数据集会存储在哪里？**
A: 默认缓存在 `~/.cache/huggingface/datasets/`

**Q: 需要多少磁盘空间？**
A: 
- pusht: ~500MB
- aloha系列: ~1-2GB each
- metaworld: ~5GB

**Q: 可以离线使用吗？**
A: 下载后会自动缓存，第二次加载不需要网络

**Q: 如何删除下载的数据集？**
A: `rm -rf ~/.cache/huggingface/datasets/lerobot*`

---

**最后更新**: 2025-10-20
**作者**: SmolVLA项目组
