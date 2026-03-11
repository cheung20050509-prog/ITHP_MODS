"""
MODS + DeBERTa: 将MODS框架与DeBERTa语言模型整合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2PreTrainedModel, DebertaV2Model
from mods_modules import GDC, MSelector, PCCA
import global_configs
from global_configs import DEVICE


class MODS_DebertaModel(DebertaV2PreTrainedModel):
    """
    MODS + DeBERTa: GDC -> MSelector -> PCCA -> prediction
    with InfoNCE reverse projection for training stability.
    """
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM = (
            global_configs.TEXT_DIM, 
            global_configs.ACOUSTIC_DIM,
            global_configs.VISUAL_DIM
        )
        
        self.config = config
        self.hidden_dim = multimodal_config.hidden_dim
        self.max_seq_length = multimodal_config.max_seq_length
        
        # DeBERTa language encoder
        model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        self.deberta = model.to(DEVICE)
        
        self.text_proj = nn.Linear(TEXT_DIM, self.hidden_dim)
        
        # GDC for acoustic / visual compression
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
        
        self.mselector = MSelector(self.hidden_dim)
        
        self.pcca = PCCA(
            hidden_dim=self.hidden_dim,
            num_layers=multimodal_config.num_pcca_layers,
            num_heads=multimodal_config.num_attention_heads,
            dropout=multimodal_config.dropout_prob
        )
        
        self.agg_proj = nn.Linear(self.hidden_dim, 1)
        self.output_proj = nn.Linear(self.hidden_dim, config.hidden_size)
        self.dropout = nn.Dropout(multimodal_config.dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # InfoNCE reverse projection heads F_m (Eq.20): h_p -> h_m prediction
        self.reverse_proj_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.reverse_proj_l = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.reverse_proj_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.init_weights()
    
    def adaptive_aggregate(self, H):
        """Aggregate sequence features into a single vector (Eq.7-8)."""
        scores = self.agg_proj(H) / math.sqrt(self.hidden_dim)
        attn = F.softmax(scores, dim=1)
        h = torch.bmm(attn.transpose(1, 2), H).squeeze(1)
        return h
    
    def forward(self, input_ids, visual, acoustic):
        """
        Returns:
            h_out: [B, config.hidden_size] for classification
            weights: [B, 3] modality weights [w_a, w_l, w_v]
            primary_idx: [B]
            nce_extras: dict with h_p_raw, h_a, h_l, h_v for InfoNCE (None at eval)
        """
        # 1. encode
        text_output = self.deberta(input_ids)[0]
        H_l = self.text_proj(text_output)
        H_a = self.gdc_acoustic(acoustic)
        H_v = self.gdc_visual(visual)
        
        H_l = self._align_sequence(H_l)
        H_a = self._align_sequence(H_a)
        H_v = self._align_sequence(H_v)
        
        # 2. MSelector: weighted features + per-sample primary_idx
        H_a_w, H_l_w, H_v_w, weights, primary_idx = self.mselector(H_a, H_l, H_v)
        
        # 3. Dynamic per-sample routing (Eq.11-12)
        H_p, H_a1, H_a2 = self._route_by_primary(H_a_w, H_l_w, H_v_w, weights, primary_idx)
        
        # 4. PCCA
        H_p_enhanced = self.pcca(H_p, H_a1, H_a2)
        
        # 5. aggregate
        h_p_raw = self.adaptive_aggregate(H_p_enhanced)
        
        # 6. project to output space
        h_out = self.output_proj(h_p_raw)
        h_out = self.layer_norm(h_out)
        h_out = self.dropout(h_out)
        
        # 7. collect unimodal vectors for InfoNCE (only useful during training)
        nce_extras = None
        if self.training:
            h_a = self.mselector.adaptive_aggregate(H_a, self.mselector.W_a)
            h_l = self.mselector.adaptive_aggregate(H_l, self.mselector.W_l)
            h_v = self.mselector.adaptive_aggregate(H_v, self.mselector.W_v)
            nce_extras = {
                'h_p': h_p_raw,         # [B, hidden_dim]
                'h_a': h_a,             # [B, hidden_dim]
                'h_l': h_l,             # [B, hidden_dim]
                'h_v': h_v,             # [B, hidden_dim]
                'F_a': self.reverse_proj_a,
                'F_l': self.reverse_proj_l,
                'F_v': self.reverse_proj_v,
            }
        
        return h_out, weights, primary_idx, nce_extras
    
    def _align_sequence(self, H):
        batch, seq_len, dim = H.shape
        if seq_len > self.max_seq_length:
            return H[:, :self.max_seq_length, :]
        elif seq_len < self.max_seq_length:
            padding = torch.zeros(batch, self.max_seq_length - seq_len, dim, device=H.device)
            return torch.cat([H, padding], dim=1)
        return H
    
    def _route_by_primary(self, H_a_w, H_l_w, H_v_w, weights, primary_idx):
        """
        Per-sample dynamic routing (Eq.11).
        primary_idx[i] in {0=a, 1=l, 2=v} selects the primary modality for sample i.
        The two remaining modalities are ordered by descending weight as a1, a2.
        """
        B = primary_idx.size(0)
        # [B, 3, T, D]
        all_mod = torch.stack([H_a_w, H_l_w, H_v_w], dim=1)
        
        idx_p = primary_idx  # [B]
        H_p = all_mod[torch.arange(B, device=idx_p.device), idx_p]  # [B, T, D]
        
        # mask out primary to find a1, a2 among remaining two
        mask = torch.ones(B, 3, device=weights.device, dtype=torch.bool)
        mask[torch.arange(B), idx_p] = False
        
        remaining_weights = weights.masked_select(mask).view(B, 2)
        remaining_indices = torch.arange(3, device=idx_p.device).unsqueeze(0).expand(B, -1)
        remaining_indices = remaining_indices.masked_select(mask).view(B, 2)
        
        # a1 = higher weight auxiliary, a2 = lower weight auxiliary
        order = remaining_weights.argsort(dim=1, descending=True)
        sorted_indices = remaining_indices.gather(1, order)
        
        idx_a1 = sorted_indices[:, 0]
        idx_a2 = sorted_indices[:, 1]
        
        H_a1 = all_mod[torch.arange(B, device=idx_p.device), idx_a1]
        H_a2 = all_mod[torch.arange(B, device=idx_p.device), idx_a2]
        
        return H_p, H_a1, H_a2


class MODS_DeBertaForSequenceClassification(DebertaV2PreTrainedModel):
    """MODS + DeBERTa sequence classification / regression."""
    
    def __init__(self, config, multimodal_config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.mods = MODS_DebertaModel(config, multimodal_config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.init_weights()
    
    def forward(self, input_ids, visual, acoustic):
        """
        Returns:
            logits: [B, num_labels]
            weights: [B, 3]
            primary_idx: [B]
            nce_extras: dict (training only) or None
        """
        pooled_output, weights, primary_idx, nce_extras = self.mods(input_ids, visual, acoustic)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits, weights, primary_idx, nce_extras
