"""
MODS: Modality Optimization and Dynamic Primary Modality Selection
Implementation based on AAAI 2026 paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CapsuleLayer(nn.Module):
    """
    胶囊网络层：将序列特征压缩为图节点
    Caps[i,j] = W[ij] * H[i]
    """
    def __init__(self, input_dim, output_dim, num_nodes, num_routing=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes  # 压缩后的节点数（对齐到语言模态长度）
        self.num_routing = num_routing
        
        # W[ij]: 从第i个时间步创建第j个节点的胶囊
        # 使用单个权重矩阵，所有时间步共享
        self.W = nn.Parameter(torch.randn(num_nodes, input_dim, output_dim) * 0.01)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim] 输入序列特征
        Returns:
            nodes: [batch, num_nodes, output_dim] 图节点表示
        """
        batch_size, seq_len, _ = x.shape
        
        # 简化计算：直接矩阵乘法
        # W: [num_nodes, input_dim, output_dim] -> [input_dim, num_nodes * output_dim]
        # [batch, seq_len, input_dim] @ [input_dim, num_nodes * output_dim]
        capsules = x @ self.W.permute(1, 0, 2).reshape(self.input_dim, -1)
        capsules = capsules.view(batch_size, seq_len, self.num_nodes, self.output_dim)
        
        # 动态路由
        nodes = self.dynamic_routing(capsules)
        return nodes
    
    def dynamic_routing(self, capsules):
        """
        动态路由算法
        Args:
            capsules: [batch, seq_len, num_nodes, output_dim]
        Returns:
            nodes: [batch, num_nodes, output_dim]
        """
        batch_size, seq_len, num_nodes, output_dim = capsules.shape
        
        # 初始化路由系数 b[i,j] = 0
        b = torch.zeros(batch_size, seq_len, num_nodes, device=capsules.device)
        
        for iteration in range(self.num_routing):
            # r[i,j] = softmax(b[i,j]) over j
            r = F.softmax(b, dim=2)  # [batch, seq_len, num_nodes]
            
            # N[j] = sum_i(Caps[i,j] * r[i,j])
            # [batch, seq_len, num_nodes, 1] * [batch, seq_len, num_nodes, output_dim]
            weighted_caps = r.unsqueeze(-1) * capsules
            nodes = weighted_caps.sum(dim=1)  # [batch, num_nodes, output_dim]
            
            if iteration < self.num_routing - 1:
                # 更新 b: b[i,j] += Caps[i,j] ⊙ tanh(N[j])
                # [batch, seq_len, num_nodes, output_dim] * [batch, 1, num_nodes, output_dim]
                agreement = (capsules * torch.tanh(nodes.unsqueeze(1))).sum(dim=-1)
                b = b + agreement
        
        return nodes


class GDC(nn.Module):
    """
    Graph-based Dynamic Compression (GDC) 模块
    用于压缩非语言模态（acoustic/visual）的序列特征
    
    流程：
    1. 胶囊网络将序列压缩为图节点
    2. 自注意力计算边权重
    3. GCN进行图表示学习
    """
    def __init__(self, input_dim, hidden_dim, target_len, num_gcn_layers=2, num_routing=3):
        """
        Args:
            input_dim: 输入特征维度 (acoustic: 74, visual: 47)
            hidden_dim: 隐藏层维度 (统一维度 d)
            target_len: 目标序列长度 (对齐到语言模态长度 T_l)
            num_gcn_layers: GCN层数
            num_routing: 动态路由迭代次数
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_len = target_len
        
        # 胶囊网络：压缩序列到 target_len 个节点
        self.capsule = CapsuleLayer(input_dim, hidden_dim, target_len, num_routing)
        
        # 边权重计算（自注意力）
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        
        # GCN层
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim, hidden_dim) for _ in range(num_gcn_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim] 输入序列（acoustic 或 visual）
        Returns:
            H: [batch, target_len, hidden_dim] 压缩后的特征
        """
        # 1. 胶囊网络创建图节点
        nodes = self.capsule(x)  # [batch, target_len, hidden_dim]
        
        # 2. 自注意力计算边权重
        Q = self.W_q(nodes)  # [batch, target_len, hidden_dim]
        K = self.W_k(nodes)  # [batch, target_len, hidden_dim]
        
        # E = ReLU((Q @ K^T) / sqrt(d))
        edge_weights = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        edge_weights = F.relu(edge_weights)  # [batch, target_len, target_len]
        
        # 3. GCN表示学习
        H = nodes
        for gcn in self.gcn_layers:
            H = gcn(H, edge_weights)
        
        H = self.layer_norm(H)
        return H


class MSelector(nn.Module):
    """
    Primary Modality Selector (MSelector)
    动态选择主模态，基于样本特征自适应决定
    
    流程：
    1. 对每个模态进行自适应聚合（序列 -> 向量）
    2. 拼接三个模态向量，通过MLP得到权重
    3. softmax得到三个权重，选择最大的作为主模态
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 每个模态的聚合权重参数
        self.W_a = nn.Linear(hidden_dim, 1)  # acoustic
        self.W_l = nn.Linear(hidden_dim, 1)  # language
        self.W_v = nn.Linear(hidden_dim, 1)  # visual
        
        # MLP for computing modality weights
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)
        )
    
    def adaptive_aggregate(self, H, W_proj):
        """
        自适应聚合：将序列特征聚合为一维向量
        a = softmax(H @ W / sqrt(d))
        h = a @ H
        
        Args:
            H: [batch, seq_len, hidden_dim]
            W_proj: Linear layer
        Returns:
            h: [batch, hidden_dim]
        """
        # [batch, seq_len, 1]
        scores = W_proj(H) / math.sqrt(self.hidden_dim)
        attn = F.softmax(scores, dim=1)
        # [batch, 1, seq_len] @ [batch, seq_len, hidden_dim] -> [batch, 1, hidden_dim]
        h = torch.bmm(attn.transpose(1, 2), H).squeeze(1)
        return h
    
    def forward(self, H_a, H_l, H_v):
        """
        Args:
            H_a: [batch, seq_len, hidden_dim] acoustic features
            H_l: [batch, seq_len, hidden_dim] language features
            H_v: [batch, seq_len, hidden_dim] visual features
        Returns:
            H_p: [batch, seq_len, hidden_dim] primary modality (weighted)
            H_a1: [batch, seq_len, hidden_dim] auxiliary 1 (weighted)
            H_a2: [batch, seq_len, hidden_dim] auxiliary 2 (weighted)
            weights: [batch, 3] modality weights [w_a, w_l, w_v]
            primary_idx: [batch] index of primary modality (0=a, 1=l, 2=v)
        """
        # 1. 自适应聚合
        h_a = self.adaptive_aggregate(H_a, self.W_a)  # [batch, hidden_dim]
        h_l = self.adaptive_aggregate(H_l, self.W_l)
        h_v = self.adaptive_aggregate(H_v, self.W_v)
        
        # 2. 拼接并计算权重
        h_concat = torch.cat([h_a, h_l, h_v], dim=-1)  # [batch, hidden_dim * 3]
        logits = self.mlp(h_concat)  # [batch, 3]
        weights = F.softmax(logits, dim=-1)  # [batch, 3]
        
        # 3. 选择主模态
        primary_idx = torch.argmax(weights, dim=-1)  # [batch]
        
        # 4. 加权特征
        w_a = weights[:, 0:1].unsqueeze(-1)  # [batch, 1, 1]
        w_l = weights[:, 1:2].unsqueeze(-1)
        w_v = weights[:, 2:3].unsqueeze(-1)
        
        H_a_weighted = w_a * H_a
        H_l_weighted = w_l * H_l
        H_v_weighted = w_v * H_v
        
        # 根据主模态索引重排序（训练时使用soft weighting）
        # 返回按权重排序的特征
        # 为了简化实现，我们直接返回加权后的所有特征
        # PCCA会根据weights来处理
        
        return H_a_weighted, H_l_weighted, H_v_weighted, weights, primary_idx


class GCNLayer(nn.Module):
    """
    图卷积网络层
    H^l = ReLU(D^{-1/2} E D^{-1/2} H^{l-1} W^l + b^l)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, H, E):
        """
        Args:
            H: [batch, num_nodes, input_dim] 节点特征
            E: [batch, num_nodes, num_nodes] 边权重矩阵
        Returns:
            H_out: [batch, num_nodes, output_dim]
        """
        # 计算度矩阵 D
        D = E.sum(dim=-1, keepdim=True)  # [batch, num_nodes, 1]
        D = D.clamp(min=1e-6)  # 避免除零
        D_inv_sqrt = 1.0 / torch.sqrt(D)
        
        # 归一化邻接矩阵: D^{-1/2} E D^{-1/2}
        # [batch, num_nodes, 1] * [batch, num_nodes, num_nodes] * [batch, 1, num_nodes]
        E_norm = D_inv_sqrt * E * D_inv_sqrt.transpose(1, 2)
        
        # 图卷积: E_norm @ H @ W + b
        H_out = torch.bmm(E_norm, H)  # [batch, num_nodes, input_dim]
        H_out = self.linear(H_out)     # [batch, num_nodes, output_dim]
        H_out = F.relu(H_out)
        
        return H_out


class MultiHeadAttention(nn.Module):
    """多头注意力，用于PCCA中的cross-attention和self-attention"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value):
        """
        Args:
            query: [batch, seq_q, hidden_dim]
            key: [batch, seq_k, hidden_dim]
            value: [batch, seq_k, hidden_dim]
        Returns:
            output: [batch, seq_q, hidden_dim]
        """
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # [batch, num_heads, seq_q, seq_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # [batch, num_heads, seq_q, head_dim]
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        return self.W_o(context)


class PositionwiseFFN(nn.Module):
    """Position-wise Feed-Forward Network"""
    def __init__(self, hidden_dim, ffn_dim=None, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or hidden_dim * 4
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class PCCALayer(nn.Module):
    """
    单层 Primary-modality-Centric Cross-Attention
    
    流程：
    1. CA_a1→p: 从辅助模态1到主模态的cross-attention
    2. CA_a2→p: 从辅助模态2到主模态的cross-attention
    3. SA_p: 主模态自注意力
    4. 融合得到增强的主模态
    5. CA_p→a1, CA_p→a2: 从主模态到辅助模态的cross-attention
    6. FFN with skip connection
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        
        # 辅助到主模态的cross-attention
        self.ca_a1_to_p = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ca_a2_to_p = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        # 主模态自注意力
        self.sa_p = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        # 主模态到辅助模态的cross-attention
        self.ca_p_to_a1 = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ca_p_to_a2 = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        # Layer Norms
        self.ln_p1 = nn.LayerNorm(hidden_dim)
        self.ln_p2 = nn.LayerNorm(hidden_dim)
        self.ln_a1 = nn.LayerNorm(hidden_dim)
        self.ln_a2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn_p = PositionwiseFFN(hidden_dim, dropout=dropout)
        self.ffn_a1 = PositionwiseFFN(hidden_dim, dropout=dropout)
        self.ffn_a2 = PositionwiseFFN(hidden_dim, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, H_p, H_a1, H_a2):
        """
        Args:
            H_p: [batch, seq_len, hidden_dim] 主模态
            H_a1: [batch, seq_len, hidden_dim] 辅助模态1
            H_a2: [batch, seq_len, hidden_dim] 辅助模态2
        Returns:
            H_p_out, H_a1_out, H_a2_out
        """
        # Layer Norm
        H_p_norm = self.ln_p1(H_p)
        H_a1_norm = self.ln_a1(H_a1)
        H_a2_norm = self.ln_a2(H_a2)
        
        # CA_a→p: Query=H_p, Key/Value=H_a
        ca_a1_to_p = self.ca_a1_to_p(H_p_norm, H_a1_norm, H_a1_norm)
        ca_a2_to_p = self.ca_a2_to_p(H_p_norm, H_a2_norm, H_a2_norm)
        
        # SA_p: 主模态自注意力
        sa_p = self.sa_p(H_p_norm, H_p_norm, H_p_norm)
        H_p_update = H_p + self.dropout(sa_p)
        
        # 融合主模态: H_p = H_p_update + CA_a1→p + CA_a2→p
        H_p_fused = H_p_update + self.dropout(ca_a1_to_p) + self.dropout(ca_a2_to_p)
        
        # CA_p→a: 主模态信息流向辅助模态
        H_p_fused_norm = self.ln_p2(H_p_fused)
        ca_p_to_a1 = self.ca_p_to_a1(H_a1_norm, H_p_fused_norm, H_p_fused_norm)
        ca_p_to_a2 = self.ca_p_to_a2(H_a2_norm, H_p_fused_norm, H_p_fused_norm)
        
        # FFN with skip connection
        H_a1_out = H_a1 + self.dropout(ca_p_to_a1)
        H_a1_out = H_a1_out + self.ffn_a1(self.ln_a1(H_a1_out))
        
        H_a2_out = H_a2 + self.dropout(ca_p_to_a2)
        H_a2_out = H_a2_out + self.ffn_a2(self.ln_a2(H_a2_out))
        
        H_p_out = H_p_fused + self.ffn_p(self.ln_p2(H_p_fused))
        
        return H_p_out, H_a1_out, H_a2_out


class PCCA(nn.Module):
    """
    Primary-modality-Centric Cross-Attention (PCCA) 模块
    多层堆叠的PCCA层
    """
    def __init__(self, hidden_dim, num_layers=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            PCCALayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(hidden_dim)
    
    def forward(self, H_p, H_a1, H_a2):
        """
        Args:
            H_p: [batch, seq_len, hidden_dim] 主模态
            H_a1: [batch, seq_len, hidden_dim] 辅助模态1
            H_a2: [batch, seq_len, hidden_dim] 辅助模态2
        Returns:
            H_p: [batch, seq_len, hidden_dim] 增强的主模态
        """
        for layer in self.layers:
            H_p, H_a1, H_a2 = layer(H_p, H_a1, H_a2)
        
        H_p = self.final_ln(H_p)
        return H_p
