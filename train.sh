#!/bin/bash
# MODS Training Script
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON="${PYTHON:-/root/autodl-tmp/anaconda3/envs/ITHP/bin/python}"
DATASET=${1:-mosi}
EPOCHS=${2:-30}
BATCH_SIZE=${3:-32}

mkdir -p checkpoints logs

nohup "$PYTHON" -u train_mods.py \
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
    --alpha_nce 0.1 \
    --checkpoint_dir checkpoints \
    --seed 128 \
    > logs/train_${DATASET}_fixed.log 2>&1 &

echo "Training started in background. PID: $!"
echo "tail -f logs/train_${DATASET}_fixed.log"
