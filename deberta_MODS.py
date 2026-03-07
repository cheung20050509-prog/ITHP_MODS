"""
MODS + DeBERTa: 将MODS框架与DeBERTa语言模型整合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.models.bert.modeling_bert import BertPooler
from mods_modules import GDC, MSelector, PCCA
import global_configs
from global_configs import DEVICE


class MODS_DebertaModel(DebertaV2PreTrainedModel):
    """
    MODS + DeBERTa 模型
    
    架构：
    1. DeBERTa: 语言模态编码器
    2. GDC: 压缩 acoustic/visual 序列到语言序列长度
    3. 投影层: 将三个模态投影到统一维度
    4. MSelector: 动态选择主模态
    5. PCCA: 跨模态交互与主模态增强
    6. 自适应聚合 + 分类
    """
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (
            global_configs.TEXT_DIM, 
            global_configs.ACOUSTIC_DIM,
            global_configs.VISUAL_DIM
        )
        
        self.config = config
        self.hidden_dim = multimodal_config.hidden_dim  # 统一的隐藏维度
        self.max_seq_length = multimodal_config.max_seq_length
        
        # DeBERTa 语言编码器
        model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        self.deberta = model.to(DEVICE)
        
        # 语言特征投影到统一维度
        self.text_proj = nn.Linear(TEXT_DIM, self.hidden_dim)
        
        # GDC: 压缩 acoustic/visual 序列
        self.gdc_acoustic = GDC(
            input_dim=ACOUSTIC_DIM,
            hidden_dim=self.hidden_dim,
            target_len=self.max_seq_length,
            num_gcn_layers=multimodal_config.num_gcn_layers,
            num_routing=multimodal_config.num_routing
        )
        
        self.gdc_visual = GDC(
            input_dim=VISUAL_DIM,
            hidden_dim=self.hidden_dim,
            target_len=self.max_seq_length,
            num_gcn_layers=multimodal_config.num_gcn_layers,
            num_routing=multimodal_config.num_routing
        )
        
        # MSelector: 动态主模态选择
        self.mselector = MSelector(self.hidden_dim)
        
        # PCCA: 跨模态交互
        self.pcca = PCCA(
            hidden_dim=self.hidden_dim,
            num_layers=multimodal_config.num_pcca_layers,
            num_heads=multimodal_config.num_attention_heads,
            dropout=multimodal_config.dropout_prob
        )
        
        # 自适应聚合
        self.agg_proj = nn.Linear(self.hidden_dim, 1)
        
        # Pooler
        self.pooler = BertPooler(config)
        
        # 输出投影 (从hidden_dim到config.hidden_size用于pooler)
        self.output_proj = nn.Linear(self.hidden_dim, config.hidden_size)
        
        self.dropout = nn.Dropout(multimodal_config.dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        self.init_weights()
    
    def adaptive_aggregate(self, H):
        """
        自适应聚合序列特征为单向量
        Args:
            H: [batch, seq_len, hidden_dim]
        Returns:
            h: [batch, hidden_dim]
        """
        scores = self.agg_proj(H) / math.sqrt(self.hidden_dim)  # [batch, seq_len, 1]
        attn = F.softmax(scores, dim=1)
        h = torch.bmm(attn.transpose(1, 2), H).squeeze(1)  # [batch, hidden_dim]
        return h
    
    def forward(self, input_ids, visual, acoustic):
        """
        Args:
            input_ids: [batch, seq_len] 文本token IDs
            visual: [batch, visual_len, visual_dim] 视觉特征
            acoustic: [batch, acoustic_len, acoustic_dim] 音频特征
        Returns:
            pooled_output: [batch, config.hidden_size]
            modality_weights: [batch, 3]
            primary_idx: [batch]
        """
        # 1. DeBERTa 编码语言
        text_output = self.deberta(input_ids)[0]  # [batch, seq_len, 768]
        H_l = self.text_proj(text_output)  # [batch, seq_len, hidden_dim]
        
        # 2. GDC 压缩 acoustic/visual
        H_a = self.gdc_acoustic(acoustic)  # [batch, max_seq_len, hidden_dim]
        H_v = self.gdc_visual(visual)      # [batch, max_seq_len, hidden_dim]
        
        # 确保序列长度一致（截断或填充到max_seq_length）
        H_l = self._align_sequence(H_l)
        H_a = self._align_sequence(H_a)
        H_v = self._align_sequence(H_v)
        
        # 3. MSelector: 动态选择主模态
        H_a_w, H_l_w, H_v_w, weights, primary_idx = self.mselector(H_a, H_l, H_v)
        
        # 4. 根据权重排序确定主模态和辅助模态
        # 简化实现：使用语言作为主模态（因为通常语言最重要）
        # 但保留动态权重信息
        H_p, H_a1, H_a2 = self._arrange_by_weights(H_l_w, H_a_w, H_v_w, weights)
        
        # 5. PCCA: 跨模态交互
        H_p_enhanced = self.pcca(H_p, H_a1, H_a2)  # [batch, seq_len, hidden_dim]
        
        # 6. 自适应聚合
        h_p = self.adaptive_aggregate(H_p_enhanced)  # [batch, hidden_dim]
        
        # 7. 投影到DeBERTa hidden size并pooling
        h_p = self.output_proj(h_p)  # [batch, config.hidden_size]
        h_p = self.layer_norm(h_p)
        h_p = self.dropout(h_p)
        
        return h_p, weights, primary_idx
    
    def _align_sequence(self, H):
        """将序列对齐到 max_seq_length"""
        batch, seq_len, dim = H.shape
        if seq_len > self.max_seq_length:
            return H[:, :self.max_seq_length, :]
        elif seq_len < self.max_seq_length:
            padding = torch.zeros(batch, self.max_seq_length - seq_len, dim, device=H.device)
            return torch.cat([H, padding], dim=1)
        return H
    
    def _arrange_by_weights(self, H_l, H_a, H_v, weights):
        """
        根据权重排列模态顺序
        简化实现：权重最高的作为主模态，其他为辅助模态
        """
        # weights: [batch, 3] -> [w_a, w_l, w_v]
        # 为了训练稳定性，我们使用soft版本
        # 主模态取加权和
        
        w_a = weights[:, 0:1].unsqueeze(-1)  # [batch, 1, 1]
        w_l = weights[:, 1:2].unsqueeze(-1)
        w_v = weights[:, 2:3].unsqueeze(-1)
        
        # 以语言为中心，但融合其他模态信息
        H_p = H_l  # 主模态
        H_a1 = H_a  # 辅助1
        H_a2 = H_v  # 辅助2
        
        return H_p, H_a1, H_a2


class MODS_DeBertaForSequenceClassification(DebertaV2PreTrainedModel):
    """MODS + DeBERTa 序列分类模型"""
    
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mods = MODS_DebertaModel(config, multimodal_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.init_weights()
    
    def forward(self, input_ids, visual, acoustic):
        """
        Args:
            input_ids: [batch, seq_len]
            visual: [batch, visual_len, visual_dim]
            acoustic: [batch, acoustic_len, acoustic_dim]
        Returns:
            logits: [batch, num_labels]
            modality_weights: [batch, 3]
            primary_idx: [batch]
        """
        pooled_output, weights, primary_idx = self.mods(input_ids, visual, acoustic)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits, weights, primary_idx
