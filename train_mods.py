"""
MODS 训练脚本
基于 AAAI 2026 论文的实现
"""

import argparse
import os
import random
import pickle
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch.nn import MSELoss

from transformers import get_linear_schedule_with_warmup, DebertaV2Tokenizer
from torch.optim import AdamW
from deberta_MODS import MODS_DeBertaForSequenceClassification
import global_configs
from global_configs import DEVICE

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base")
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=32)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=30)
parser.add_argument("--dropout_prob", type=float, default=0.1)
parser.add_argument("--learning_rate", type=float, default=3e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=128)
parser.add_argument("--weight_decay", type=float, default=1e-3)

# MODS 特有参数
parser.add_argument("--hidden_dim", type=int, default=128, help="统一隐藏维度")
parser.add_argument("--num_gcn_layers", type=int, default=2, help="GDC中GCN层数")
parser.add_argument("--num_routing", type=int, default=3, help="胶囊网络动态路由迭代次数")
parser.add_argument("--num_pcca_layers", type=int, default=3, help="PCCA层数")
parser.add_argument("--num_attention_heads", type=int, default=4, help="注意力头数")
parser.add_argument("--alpha_nce", type=float, default=0.1, help="InfoNCE损失系数")
parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint保存目录")

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


def set_up_data_loader():
    with open(f"datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    train_data = data["train"]
    dev_data = data["dev"]
    test_data = data["test"]

    train_dataset = get_appropriate_dataset(train_data)
    dev_dataset = get_appropriate_dataset(dev_data)
    test_dataset = get_appropriate_dataset(test_data)

    num_train_optimization_steps = (
        int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_step)
        * args.n_epochs
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.dev_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    return train_dataloader, dev_dataloader, test_dataloader, num_train_optimization_steps


def set_random_seed(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed: {seed}")


def prep_for_training(num_train_optimization_steps: int):
    model = MODS_DeBertaForSequenceClassification.from_pretrained(
        args.model, multimodal_config=args, num_labels=1,
    )
    model.to(DEVICE)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_proportion * num_train_optimization_steps),
        num_training_steps=num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def compute_infonce_loss(nce_extras, temperature=0.07):
    """
    InfoNCE loss (Eq.20-22).
    Measures how well h_p can predict each unimodal vector h_m via reverse projection F_m.
    Temperature scaling ensures stable gradients with small batch sizes.
    """
    h_p = nce_extras['h_p']      # [B, D]
    modality_keys = [('h_a', 'F_a'), ('h_l', 'F_l'), ('h_v', 'F_v')]
    
    total_nce = h_p.new_tensor(0.0)
    for h_key, f_key in modality_keys:
        h_m = nce_extras[h_key]          # [B, D]
        F_m = nce_extras[f_key]          # nn.Linear
        
        h_m_norm = F.normalize(h_m, dim=-1)
        f_hp_norm = F.normalize(F_m(h_p), dim=-1)
        
        # cosine similarity with temperature scaling
        sim_matrix = torch.mm(h_m_norm, f_hp_norm.t()) / temperature  # [B, B]
        
        labels = torch.arange(h_p.size(0), device=h_p.device)
        total_nce = total_nce + F.cross_entropy(sim_matrix, labels)
    
    return total_nce / 3.0  # average over 3 modalities


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    
    for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)

        visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
        acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)
        
        logits, weights, primary_idx, nce_extras = model(input_ids, visual_norm, acoustic_norm)
        
        # L_reg: MAE loss (Eq.24)
        loss_reg = F.l1_loss(logits.view(-1), label_ids.view(-1))
        
        # L_NCE: InfoNCE loss (Eq.22)
        loss_nce = compute_infonce_loss(nce_extras) if nce_extras is not None else 0.0
        
        # L_task = L_reg + alpha * L_NCE (Eq.25)
        loss = loss_reg + args.alpha_nce * loss_nce

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()
        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader):
    model.eval()
    dev_loss = 0
    nb_dev_steps = 0
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Validation")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, visual, acoustic, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            visual_norm = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
            acoustic_norm = (acoustic - acoustic.min()) / (acoustic.max() - acoustic.min() + 1e-8)

            logits, weights, primary_idx, _ = model(input_ids, visual_norm, acoustic_norm)
            
            loss = F.l1_loss(logits.view(-1), label_ids.view(-1))

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []
    all_weights = []

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

            preds.extend(np.squeeze(logits).tolist())
            labels.extend(np.squeeze(label_ids).tolist())
            all_weights.extend(weights.tolist())

    preds = np.array(preds)
    labels = np.array(labels)
    all_weights = np.array(all_weights)

    return preds, labels, all_weights


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):
    preds, y_test, weights = test_epoch(model, test_dataloader)
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
    print(f"  Modality weights (a/l/v): {weights_mean[0]:.3f}/{weights_mean[1]:.3f}/{weights_mean[2]:.3f}")

    return acc2, acc7, mae, corr, f_score


def train(model, train_dataloader, validation_dataloader, test_data_loader, optimizer, scheduler):
    best_valid_loss = float('inf')
    best_test_results = None
    
    # 创建 checkpoint 目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f"mods_{args.dataset}_best.pt")
    
    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, validation_dataloader)

        print(f"\nEpoch {epoch_i + 1}/{args.n_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        
        # 每5个epoch或最后一个epoch进行测试
        if (epoch_i + 1) % 5 == 0 or epoch_i == args.n_epochs - 1:
            test_acc2, test_acc7, test_mae, test_corr, test_f1 = test_score_model(
                model, test_data_loader
            )
            print(f"  Test: Acc-2={test_acc2*100:.2f}%, Acc-7={test_acc7*100:.2f}%, "
                  f"MAE={test_mae:.4f}, Corr={test_corr:.4f}, F1={test_f1:.4f}")
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_test_results = (test_acc2, test_acc7, test_mae, test_corr, test_f1)
                # 保存最佳模型
                torch.save({
                    'epoch': epoch_i + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                    'test_results': best_test_results,
                    'args': args,
                }, checkpoint_path)
                print(f"  >> Best model saved to {checkpoint_path}")
    
    print("\n" + "="*60)
    print("Final Best Results:")
    if best_test_results:
        acc2, acc7, mae, corr, f1 = best_test_results
        print(f"  Acc-2: {acc2*100:.2f}%")
        print(f"  Acc-7: {acc7*100:.2f}%")
        print(f"  MAE: {mae:.4f}")
        print(f"  Corr: {corr:.4f}")
        print(f"  F1: {f1:.4f}")
    
    return best_test_results


def main():
    print("="*60)
    print("MODS Training Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  PCCA layers: {args.num_pcca_layers}")
    print(f"  GCN layers: {args.num_gcn_layers}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.train_batch_size}")
    print(f"  Epochs: {args.n_epochs}")
    print("="*60)
    
    set_random_seed(args.seed)
    
    train_data_loader, dev_data_loader, test_data_loader, num_train_optimization_steps = set_up_data_loader()
    
    model, optimizer, scheduler = prep_for_training(num_train_optimization_steps)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*60)
    
    train(model, train_data_loader, dev_data_loader, test_data_loader, optimizer, scheduler)


if __name__ == '__main__':
    main()
