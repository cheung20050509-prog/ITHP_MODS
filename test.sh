#!/bin/bash
# MODS Testing Script
# Usage: ./test.sh [dataset] [checkpoint_path]

# 默认参数
DATASET=${1:-mosi}
CHECKPOINT=${2:-"checkpoints/mods_${DATASET}_best.pt"}

# 激活conda环境
source /home/anaconda/etc/profile.d/conda.sh
conda activate ITHP

# 切换到脚本所在目录
cd "$(dirname "$0")"

echo "=============================================="
echo "MODS Testing"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Checkpoint: $CHECKPOINT"
echo "=============================================="

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Please train the model first using: ./train.sh $DATASET"
    exit 1
fi

# 运行测试
python test_mods.py \
    --dataset $DATASET \
    --checkpoint $CHECKPOINT \
    --test_batch_size 128 \
    --hidden_dim 128 \
    --num_pcca_layers 3 \
    --num_gcn_layers 2 \
    --num_routing 3

echo "=============================================="
echo "Testing Complete!"
echo "=============================================="
