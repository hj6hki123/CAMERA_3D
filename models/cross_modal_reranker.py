# -*- coding: utf-8 -*-
import torch, torch.nn as nn


class CrossModalReranker(nn.Module):
    def __init__(self, txt_dim=768, vis_dim=512, heads=8, L=2):
        super().__init__()
        self.map_vis = nn.Linear(vis_dim, txt_dim)
        self.blocks  = nn.ModuleList([
            nn.TransformerEncoderLayer(txt_dim, heads, dim_feedforward=txt_dim*4, batch_first=True)
            for _ in range(L)])
        self.scorer = nn.Linear(txt_dim, 1)    # <-- 統一名稱 scorer

    def forward(self, txt_tok, vis_tok):        # txt_tok (B,Tt,D) vis_tok (B,Tv,Dv)
        vis = self.map_vis(vis_tok)             # match dim
        seq = torch.cat([txt_tok, vis], 1)
        for blk in self.blocks:
            seq = blk(seq)
        return self.scorer(seq[:,0]).squeeze(-1)   # (B,)

