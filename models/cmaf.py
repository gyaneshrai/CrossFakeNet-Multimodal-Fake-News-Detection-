"""
models/cmaf.py
Cross-Modal Attention Fusion (CMAF) module.
Used by image+comment and video pipelines.
"""

import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):
    """
    Fuses text, visual, metadata, and social features via
    bidirectional cross-attention and contradiction-aware pooling.
    """

    def __init__(self, text_dim=1024, visual_dim=1024,
                 meta_dim=8, social_dim=8, hidden=256):
        super().__init__()
        self.text_proj   = nn.Linear(text_dim, hidden)
        self.vis_proj    = nn.Linear(visual_dim, hidden)
        self.meta_proj   = nn.Linear(meta_dim, 32)
        self.social_proj = nn.Linear(social_dim, 32)

        self.cross_attn  = nn.MultiheadAttention(hidden, num_heads=8,
                                                  batch_first=True,
                                                  dropout=0.1)
        self.layer_norm  = nn.LayerNorm(hidden)

        # Scalar contradiction score
        self.contradiction = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, 1),     nn.Sigmoid()
        )

        total_dim = hidden * 2 + 32 + 32 + 1
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),  nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, text_feat, vis_feat, meta_feat, social_feat):
        t = self.text_proj(text_feat).unsqueeze(1)
        v = self.vis_proj(vis_feat).unsqueeze(1)

        attn_v, _ = self.cross_attn(v, t, t)
        attn_t, _ = self.cross_attn(t, v, v)

        t_fused = self.layer_norm(t + attn_t).squeeze(1)
        v_fused = self.layer_norm(v + attn_v).squeeze(1)

        c = self.contradiction(torch.abs(t_fused - v_fused))
        m = self.meta_proj(meta_feat)
        s = self.social_proj(social_feat)

        z = torch.cat([t_fused, v_fused, m, s, c], dim=-1)
        return self.classifier(z)
