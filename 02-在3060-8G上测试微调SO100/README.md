# SmolVLA微调 - RTX 3060 8GB优化

## 📂 文件说明

- `00-启动微调-简化版.md` - 使用官方脚本的详细说明
- `01-微调脚本-8G优化版.py` - 自定义脚本（未完成，有bug）
- `04-立即开始微调.sh` - **推荐使用**的启动脚本
- `README.md` - 本文件

## 🚀 快速开始（3种方式）

### 方式1: 一键启动（推荐）

```bash
cd /root/lerobot_project/02-在3060-8G上测试微调SO100
./04-立即开始微调.sh
```

### 方式2: 后台运行

```bash
nohup ./04-立即开始微调.sh > train.log 2>&1 &
tail -f train.log
```

### 方式3: 自定义参数

```bash
cd /root/lerobot_project
source smolvla_env/bin/activate

python lerobot/src/lerobot/scripts/lerobot_train.py \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=lerobot/svla_so100_pickplace \
    --batch_size=4 \
    --steps=20000 \
    --output_dir=./02-在3060-8G上测试微调SO100/outputs \
    --policy.device=cuda
```

## ⚙️ 配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 4 | 8GB显存优化 |
| gradient_accumulation | 8 | 有效batch=32 |
| steps | 20000 | 完整训练 |
| save_freq | 2000 | 每2000步保存 |
| 预计时间 | 2-3小时 | RTX 3060 Ti |

## 📊 预期结果

训练前（Base模型）：
- MAE: ~30°
- 成功率: 10-20%

训练后（微调模型）：
- MAE: ~10-15° ⬇️ **降低50%**
- 成功率: 60-70% ⬆️ **提升4-6倍**

## 💾 输出文件

训练完成后，文件保存在：
```
./outputs/
├── checkpoints/         # 检查点（每2000步）
├── final_model/         # 最终模型
└── logs/                # 训练日志
```

## 🎯 下一步

微调完成后：
1. 测试微调模型性能
2. 与Base模型对比MAE
3. 准备Parol6数据集
4. 在Parol6上继续微调

## ⚠️ 注意事项

1. **显存不足**：降低batch_size到2
2. **训练中断**：会自动保存检查点，可恢复
3. **Parol6应用**：必须在Parol6数据上重新微调
