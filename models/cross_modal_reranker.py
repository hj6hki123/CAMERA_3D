# -*- coding: utf-8 -*-
import torch, torch.nn as nn


# class CrossModalReranker(nn.Module):
#     def __init__(self, txt_dim=768, vis_dim=512, heads=8, L=2):
#         super().__init__()
#         self.map_vis = nn.Linear(vis_dim, txt_dim)
#         self.blocks  = nn.ModuleList([
#             nn.TransformerEncoderLayer(txt_dim, heads, dim_feedforward=txt_dim*4, batch_first=True)
#             for _ in range(L)])
#         self.scorer = nn.Linear(txt_dim, 1)    # <-- 統一名稱 scorer

#     def forward(self, txt_tok, vis_tok):        # txt_tok (B,Tt,D) vis_tok (B,Tv,Dv)
#         vis = self.map_vis(vis_tok)             # match dim
#         seq = torch.cat([txt_tok, vis], 1)
#         for blk in self.blocks:
#             seq = blk(seq)
#         return self.scorer(seq[:,0]).squeeze(-1)   # (B,)

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Sinusoidal 位置編碼（同 Transformer 原論文）
    pos_dim: Embedding 維度 (d_model)
    max_len: 序列最大長度
    """
    def __init__(self, pos_dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, pos_dim)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, pos_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / pos_dim))
        # pe[pos, 2i]   = sin(pos / 10000^(2i/d_model))
        # pe[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, L, d_model)
        回傳 x + pos_encoding[:, :L, :]
        """
        return x + self.pe[:, : x.size(1), :]


class CrossModalReranker(nn.Module):
    def __init__(self,
                 txt_dim: int = 768,
                 vis_dim: int = 512,
                 heads: int = 8,
                 num_layers: int = 2,
                 txt_max_len: int = 64,
                 vis_max_len: int = 64):
        """
        txt_dim, vis_dim: 文字／視覺 token 維度
        heads: Transformer 多頭注意力頭數
        num_layers: Transformer Encoder Layer 層數
        txt_max_len: 文字 token 序列最大長度
        vis_max_len: 視覺 token 序列最大長度
        """

        super().__init__()

        # 1. CLS token（Learnable）
        #    用來表示整段跨模態序列的全域向量，最後取 CLS 做分數
        self.cls_token = nn.Parameter(torch.randn(1, 1, txt_dim))  # (1, 1, txt_dim)

        # 2. 視覺 token 映射：Linear(vis_dim → txt_dim) + LayerNorm
        self.map_vis = nn.Sequential(
            nn.Linear(vis_dim, txt_dim),
            nn.LayerNorm(txt_dim)
        )

        # 3. 類別 (Type Embedding)：0 表示文字，1 表示視覺
        #    一共兩個類別 embedding
        self.type_embeddings = nn.Embedding(2, txt_dim)  # (num_types=2, txt_dim)

        # 4. 位置編碼：sinusoidal
        #    序列總長度 = 1 (CLS) + txt_max_len + vis_max_len
        self.pos_encoding = PositionalEncoding(txt_dim, max_len=1 + txt_max_len + vis_max_len)

        # 5. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=txt_dim,
            nhead=heads,
            dim_feedforward=txt_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.blocks = nn.ModuleList([encoder_layer for _ in range(num_layers)])

        # 6. 最後取 CLS token 後做線性打分
        self.scorer = nn.Sequential(
            nn.LayerNorm(txt_dim),
            nn.Dropout(0.1),
            nn.Linear(txt_dim, 1)
        )

        # Save for mask construction
        self.txt_max_len = txt_max_len
        self.vis_max_len = vis_max_len

    def forward(self,
                txt_tok: torch.Tensor,
                vis_tok: torch.Tensor,
                txt_padding_mask: torch.Tensor = None,
                vis_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        txt_tok:   (B, Tt, txt_dim)     - 已經是文字 token embedding
        vis_tok:   (B, Tv, vis_dim)     - 原始視覺 token embedding
        txt_padding_mask: (B, Tt)       - True 代表 PAD，不計算注意力 (optional)
        vis_padding_mask: (B, Tv)       - True 代表 PAD，不計算注意力 (optional)

        回傳：score (B,)
        """

        B, Tt, _ = txt_tok.shape
        _, Tv, _ = vis_tok.shape

        # 1. 插入 CLS token 到所有 batch
        #    cls_tokens_expand: (B, 1, txt_dim)
        cls_tokens_expand = self.cls_token.expand(B, -1, -1)

        # 2. 處理文字 token
        #    先假設上層傳進來的 txt_tok 已經是 (B, Tt, txt_dim)，裡面含有 pos encoding 嗎？
        #    為了安全，我們重新把位置編碼 + 類別編碼都加進來。
        #    類別 embedding: 文字類別 idx = 0
        text_type_ids = torch.zeros(B, Tt, dtype=torch.long, device=txt_tok.device)  # (B, Tt)
        text_type_emb = self.type_embeddings(text_type_ids)                         # (B, Tt, txt_dim)

        # 3. 處理視覺 token：先 map_vis 再加 type embedding
        #    map_vis_vis: (B, Tv, txt_dim)
        mapped_vis = self.map_vis(vis_tok)  # (B, Tv, txt_dim)
        vis_type_ids = torch.ones(B, Tv, dtype=torch.long, device=txt_tok.device)   # (B, Tv)
        vis_type_emb = self.type_embeddings(vis_type_ids)                           # (B, Tv, txt_dim)

        # 4. 把原始 txt_tok + 類別 emb、視覺 mapped tok + 類別 emb 做相加
        txt_combined = txt_tok + text_type_emb             # (B, Tt, txt_dim)
        vis_combined = mapped_vis + vis_type_emb           # (B, Tv, txt_dim)

        # 5. 把序列串接：CLS + txt_combined + vis_combined
        #    最後整個序列長度 = 1 + Tt + Tv
        seq = torch.cat([cls_tokens_expand, txt_combined, vis_combined], dim=1)  # (B, 1+Tt+Tv, txt_dim)

        # 6. 加上位置編碼
        seq = self.pos_encoding(seq)  # (B, 1+Tt+Tv, txt_dim)

        # 7. 構造 attention mask / key_padding_mask
        #    TransformerEncoderLayer() 中的 src_key_padding_mask 形狀是 (B, S)，
        #    其中 True 表示對應位置應該被遮罩（不計算注意力）。
        #
        #    需要把 (B, Tt) 和 (B, Tv) 先組成 (B, 1+Tt+Tv)：
        #    - CLS token 一定要看見所有 token，所以對應位置填 False（不遮罩）
        #    - 文字 token 直接複製 txt_padding_mask
        #    - 視覺 token 直接複製 vis_padding_mask
        if txt_padding_mask is None:
            txt_padding_mask = torch.zeros(B, Tt, dtype=torch.bool, device=txt_tok.device)
        if vis_padding_mask is None:
            vis_padding_mask = torch.zeros(B, Tv, dtype=torch.bool, device=txt_tok.device)

        # 拼接：CLS mask (all False) + txt_padding_mask + vis_padding_mask
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=txt_tok.device)
        combined_padding_mask = torch.cat([cls_mask, txt_padding_mask, vis_padding_mask], dim=1)  # (B, 1+Tt+Tv)

        # 8. 依序通過每層 TransformerEncoderLayer
        #    注意一定要把 src_key_padding_mask 傳下去
        for layer in self.blocks:
            seq = layer(src=seq, src_key_padding_mask=combined_padding_mask)  # (B, 1+Tt+Tv, txt_dim)

        # 9. 取序列第 0 個位置 (CLS token) 作為整段融合後的摘要向量
        cls_final = seq[:, 0, :]  # (B, txt_dim)

        # 10. 最後做分數打分
        scores = self.scorer(cls_final).squeeze(-1)  # (B,)
        return scores
