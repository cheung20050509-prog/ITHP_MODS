#!/bin/bash
# MODS Training Script
# Usage: ./run.sh [dataset] [epochs] [batch_size]

# 默认参数
DATASET=${1:-mosi}
EPOCHS=${2:-30}
BATCH_SIZE=${3:-32}

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ITHP

# 切换到脚本所在目录
cd "$(dirname "$0")"

echo "=============================================="
echo "MODS Training - DeBERTa + GDC + MSelector + PCCA"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "=============================================="

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
    --seed 128

echo "=============================================="
echo "Training Complete!"
echo "=============================================="
