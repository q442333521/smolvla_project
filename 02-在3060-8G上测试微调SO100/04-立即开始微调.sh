#!/bin/bash
# SmolVLA微调 - 立即开始脚本

cd /root/lerobot_project
source smolvla_env/bin/activate

# 使用时间戳创建唯一输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/root/lerobot_project/02-在3060-8G上测试微调SO100/outputs_${TIMESTAMP}"

echo "==============================================="
echo "🚀 开始SmolVLA微调 - SO100数据集"
echo "==============================================="
echo "配置:"
echo "  - 模型类型: smolvla"
echo "  - 预训练模型: lerobot/smolvla_base"
echo "  - 数据集: lerobot/svla_so100_pickplace"  
echo "  - 批量大小: 4"
echo "  - 训练步数: 20000"
echo "  - 输出目录: ${OUTPUT_DIR}"
echo "==============================================="
echo ""

# 使用官方训练脚本（正确的参数）
python lerobot/src/lerobot/scripts/lerobot_train.py \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.repo_id=local/smolvla_so100_finetuned \
    --dataset.repo_id=lerobot/svla_so100_pickplace \
    --batch_size=4 \
    --steps=20000 \
    --output_dir=${OUTPUT_DIR} \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    --save_freq=2000 \
    --log_freq=100
