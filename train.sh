#!/bin/bash
# MODS Training Script
# Usage: ./train.sh [dataset] [epochs] [batch_size]

# 默认参数
DATASET=${1:-mosi}
EPOCHS=${2:-30}
BATCH_SIZE=${3:-32}

# 激活conda环境
source /home/anaconda/etc/profile.d/conda.sh
conda activate ITHP

# 切换到脚本所在目录
cd "$(dirname "$0")"

echo "=============================================="
echo "MODS Training - DeBERTa + GDC + MSelector + PCCA"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Checkpoint Dir: checkpoints/"
echo "=============================================="

# 创建checkpoint目录
mkdir -p checkpoints

# 运行训练
python train_mods.py \
    --dataset $DATASET \
    --n_epochs $EPOCHS \
    --train_batch_size $BATCH_SIZE \
    --learning_rate 3e-5 \
    --hidden_dim 128 \
    --num_pcca_layers 3 \
    --num_gcn_layers 2 \
    --num_routing 3 \
    --dropout_prob 0.1 \
    --weight_decay 1e-3 \
    --checkpoint_dir checkpoints \
    --seed 128

echo "=============================================="
echo "Training Complete!"
echo "Checkpoint saved to: checkpoints/mods_${DATASET}_best.pt"
echo "=============================================="
