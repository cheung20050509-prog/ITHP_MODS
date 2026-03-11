"""
MODS 测试脚本
加载训练好的 checkpoint 并在测试集上评估
"""

import argparse
import os
import pickle
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from transformers import DebertaV2Tokenizer
from deberta_MODS import MODS_DeBertaForSequenceClassification
import global_configs
from global_configs import DEVICE

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint文件路径")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint目录")

# MODS 特有参数 (需要与训练时一致)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--num_gcn_layers", type=int, default=2)
parser.add_argument("--num_routing", type=int, default=3)
parser.add_argument("--num_pcca_layers", type=int, default=3)
parser.add_argument("--num_attention_heads", type=int, default=4)
parser.add_argument("--dropout_prob", type=float, default=0.1)

args = parser.parse_args()

global_configs.set_dataset_config(args.dataset)
ACOUSTIC_DIM, VISUAL_DIM, TEXT_DIM = (
    global_configs.ACOUSTIC_DIM, 
    global_configs.VISUAL_DIM,
    global_configs.TEXT_DIM
)


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_to_features(examples, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        (words, visual, acoustic), label_id, segment = example

        tokens, inversions = [], []
        for idx, word in enumerate(words):
            tokenized = tokenizer.tokenize(word)
            tokens.extend(tokenized)
            inversions.extend([idx] * len(tokenized))

        assert len(tokens) == len(inversions)

        aligned_visual = []
        aligned_audio = []

        for inv_idx in inversions:
            aligned_visual.append(visual[inv_idx, :])
            aligned_audio.append(acoustic[inv_idx, :])

        visual = np.array(aligned_visual)
        acoustic = np.array(aligned_audio)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            acoustic = acoustic[: max_seq_length - 2]
            visual = visual[: max_seq_length - 2]

        input_ids, visual, acoustic, input_mask, segment_ids = prepare_deberta_input(
            tokens, visual, acoustic, tokenizer
        )

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
            )
        )
    return features


def prepare_deberta_input(tokens, visual, acoustic, tokenizer):
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    tokens = [CLS] + tokens + [SEP]

    acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))
    visual_zero = np.zeros((1, VISUAL_DIM))
    visual = np.concatenate((visual_zero, visual, visual_zero))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    pad_length = args.max_seq_length - len(input_ids)

    acoustic_padding = np.zeros((pad_length, ACOUSTIC_DIM))
    acoustic = np.concatenate((acoustic, acoustic_padding))

    visual_padding = np.zeros((pad_length, VISUAL_DIM))
    visual = np.concatenate((visual, visual_padding))

    padding = [0] * pad_length

    input_ids += padding
    input_mask += padding
    segment_ids += padding

    return input_ids, visual, acoustic, input_mask, segment_ids


def get_tokenizer(model):
    return DebertaV2Tokenizer.from_pretrained(model)


def get_appropriate_dataset(data):
    tokenizer = get_tokenizer(args.model)
    features = convert_to_features(data, args.max_seq_length, tokenizer)
    
    all_input_ids = torch.tensor(np.array([f.input_ids for f in features]), dtype=torch.long)
    all_visual = torch.tensor(np.array([f.visual for f in features]), dtype=torch.float)
    all_acoustic = torch.tensor(np.array([f.acoustic for f in features]), dtype=torch.float)
    all_label_ids = torch.tensor(np.array([f.label_id for f in features]), dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_visual, all_acoustic, all_label_ids)
    return dataset


def load_test_data():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)
    
    test_data = data["test"]
    test_dataset = get_appropriate_dataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    
    return test_dataloader


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []
    all_weights = []
    all_primary_idx = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

            logits, weights, primary_idx, _ = model(input_ids, visual_norm, acoustic_norm)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()
            weights = weights.detach().cpu().numpy()
            primary_idx = primary_idx.detach().cpu().numpy()

            preds.extend(np.squeeze(logits).tolist())
            labels.extend(np.squeeze(label_ids).tolist())
            all_weights.extend(weights.tolist())
            all_primary_idx.extend(primary_idx.tolist())

    preds = np.array(preds)
    labels = np.array(labels)
    all_weights = np.array(all_weights)
    all_primary_idx = np.array(all_primary_idx)

    return preds, labels, all_weights, all_primary_idx


def compute_metrics(preds, y_test, weights, primary_idx, use_zero=False):
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds_nz = preds[non_zeros]
    y_test_nz = y_test[non_zeros]

    mae = np.mean(np.absolute(preds_nz - y_test_nz))
    corr = np.corrcoef(preds_nz, y_test_nz)[0][1]

    # Acc-2 (binary)
    preds_binary = preds_nz >= 0
    y_test_binary = y_test_nz >= 0
    f_score = f1_score(y_test_binary, preds_binary, average="weighted")
    acc2 = accuracy_score(y_test_binary, preds_binary)

    # Acc-7 (7-class)
    preds_7 = np.clip(np.round(preds_nz), -3, 3).astype(int) + 3
    y_test_7 = np.clip(np.round(y_test_nz), -3, 3).astype(int) + 3
    acc7 = accuracy_score(y_test_7, preds_7)

    # 统计主模态选择分布
    weights_mean = np.mean(weights, axis=0)
    
    # 统计各模态被选为主模态的比例
    modality_names = ['Acoustic', 'Language', 'Visual']
    primary_counts = np.bincount(primary_idx.astype(int), minlength=3)
    primary_ratios = primary_counts / len(primary_idx)

    return {
        'acc2': acc2,
        'acc7': acc7,
        'mae': mae,
        'corr': corr,
        'f1': f_score,
        'weights_mean': weights_mean,
        'primary_ratios': primary_ratios,
        'modality_names': modality_names
    }


def load_model(checkpoint_path):
    """加载训练好的模型"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # 从 checkpoint 获取训练时的参数
    saved_args = checkpoint.get('args', args)
    
    # 更新当前 args
    if hasattr(saved_args, 'hidden_dim'):
        args.hidden_dim = saved_args.hidden_dim
    if hasattr(saved_args, 'num_pcca_layers'):
        args.num_pcca_layers = saved_args.num_pcca_layers
    if hasattr(saved_args, 'num_gcn_layers'):
        args.num_gcn_layers = saved_args.num_gcn_layers
    if hasattr(saved_args, 'num_routing'):
        args.num_routing = saved_args.num_routing
    
    # 创建模型
    model = MODS_DeBertaForSequenceClassification.from_pretrained(
        args.model, multimodal_config=args, num_labels=1,
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"  Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Training valid_loss: {checkpoint.get('valid_loss', 'unknown'):.4f}")
    
    if 'test_results' in checkpoint and checkpoint['test_results']:
        acc2, acc7, mae, corr, f1 = checkpoint['test_results']
        print(f"  Training best test results: Acc-2={acc2*100:.2f}%, Acc-7={acc7*100:.2f}%")
    
    return model


def main():
    print("=" * 60)
    print("MODS Testing")
    print("=" * 60)
    
    # 确定 checkpoint 路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"mods_{args.dataset}_best.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first or specify --checkpoint path")
        return
    
    # 加载模型
    model = load_model(checkpoint_path)
    
    # 加载测试数据
    print(f"\nLoading test data for {args.dataset}...")
    test_dataloader = load_test_data()
    print(f"  Test samples: {len(test_dataloader.dataset)}")
    
    # 运行测试
    print("\nRunning evaluation...")
    preds, labels, weights, primary_idx = test_epoch(model, test_dataloader)
    
    # 计算指标
    metrics = compute_metrics(preds, labels, weights, primary_idx)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"  Acc-2: {metrics['acc2']*100:.2f}%")
    print(f"  Acc-7: {metrics['acc7']*100:.2f}%")
    print(f"  MAE:   {metrics['mae']:.4f}")
    print(f"  Corr:  {metrics['corr']:.4f}")
    print(f"  F1:    {metrics['f1']:.4f}")
    
    print("\nModality Analysis:")
    print(f"  Mean weights (A/L/V): {metrics['weights_mean'][0]:.3f}/{metrics['weights_mean'][1]:.3f}/{metrics['weights_mean'][2]:.3f}")
    print(f"  Primary modality selection ratios:")
    for i, name in enumerate(metrics['modality_names']):
        print(f"    {name}: {metrics['primary_ratios'][i]*100:.1f}%")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
