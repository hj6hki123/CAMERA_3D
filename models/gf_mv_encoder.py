# models/gf_mv_encoder.py
import math, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights

class PosEnc(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        pos = torch.arange(n).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.)/d))
        pe  = torch.zeros(n, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # (B, T, D)
        return x + self.pe[:, :x.size(1)]

class Block(nn.Module):
    def __init__(self, d=512, h=8, p=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d, d*4), nn.GELU(),
            nn.Dropout(p), nn.Linear(d*4, d)
        )

    def forward(self, x):
        # 自注意力
        attn_out, attn_w = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=True)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_w

class GFMVEncoder(nn.Module):
    def __init__(self,
                 num_views=12,
                 dim=512,
                 vis_layers=2,
                 fusion_layers=2,
                 heads=8):
        """
        num_views      : 每個樣本的視角數量
        dim            : token 維度
        vis_layers     : 純視覺自注意力層數
        fusion_layers  : 語言門控自注意力層數
        heads          : 多頭注意力頭數
        """
        super().__init__()
        self.dim, self.V = dim, num_views

        # 2D 特徵 extractor
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.feat2d = nn.Sequential(*list(vgg.features.children()))
        self.map2d  = nn.Linear(512*7*7, dim)

        # learnable CLS token
        self.cls_tok = nn.Parameter(torch.randn(1, 1, dim))

        # Gate token 投影
        self.gate_proj = nn.Linear(dim, dim)

        # 純視覺階段的 PosEnc + Blocks
        self.pos_vis    = PosEnc(dim, 1 + num_views)
        self.vis_blocks = nn.ModuleList([Block(dim, heads) for _ in range(vis_layers)])

        # 語言門控階段的 PosEnc + Blocks
        # 序列長度 = CLS + Gate + V
        self.pos_fuse     = PosEnc(dim, 1 + 1 + num_views)
        self.fusion_blocks = nn.ModuleList([Block(dim, heads) for _ in range(fusion_layers)])

        # 池化與輸出投影
        self.pool      = nn.AdaptiveMaxPool1d(1)
        self.out_proj  = nn.Linear(dim, dim)

    def forward(self, imgs, txt_vec=None):
        """
        imgs   : (B, V, C, H, W)
        txt_vec: (B, dim)  或 None
        回傳:
          global_vec      : (B, dim)
          all_attn_maps   : {
            'vis': [...],    # 每層純視覺 attn weights
            'fuse': [...]    # 每層融合 attn weights
          }
          final_tok_seq   : (B, 1+1+V, dim)
        """
        B, V, C, H, W = imgs.shape

        # 1) extract 2D features per view
        f = self.feat2d(imgs.view(B*V, C, H, W))                      # (B*V, 512,7,7)
        f = self.map2d(f.flatten(1)).view(B, V, self.dim)            # (B, V, dim)

        # ────── 純視覺階段 ──────
        # 組序列：CLS + V
        cls_expand = self.cls_tok.expand(B, -1, -1)                  # (B,1,dim)
        vis_seq = torch.cat([cls_expand, f], dim=1)                  # (B,1+V,dim)
        vis_seq = self.pos_vis(vis_seq)

        vis_attn_maps = []
        for blk in self.vis_blocks:
            vis_seq, attn_w = blk(vis_seq)
            vis_attn_maps.append(attn_w)                             # (B, heads, T_vis, T_vis)

        # ────── 語言門控階段 ──────
        # 準備 Gate token
        if txt_vec is None:
            gate = torch.zeros(B, 1, self.dim, device=imgs.device)
        else:
            gate = self.gate_proj(txt_vec).unsqueeze(1)              # (B,1,dim)

        # 組序列：CLS(from vis_seq)+ Gate + V(from vis_seq)
        # vis_seq[:,0] 是第一階段 CLS，vis_seq[:,1:] 是各 view tokens
        cls_vis = vis_seq[:, :1]                                     # (B,1,dim)
        views   = vis_seq[:, 1:]                                     # (B,V,dim)
        fuse_seq = torch.cat([cls_vis, gate, views], dim=1)          # (B,1+1+V,dim)
        fuse_seq = self.pos_fuse(fuse_seq)

        fuse_attn_maps = []
        for blk in self.fusion_blocks:
            fuse_seq, attn_w = blk(fuse_seq)
            fuse_attn_maps.append(attn_w)                            # (B, heads, T_fuse, T_fuse)

        # ────── 全域向量與輸出 ──────
        # fuse_seq: (B, T, dim) → (B, dim, T) → pool → (B, dim)
        global_vec = self.pool(fuse_seq.transpose(1,2)).squeeze(-1)
        global_vec = self.out_proj(global_vec)                       # (B, dim)

        return global_vec, {'vis': vis_attn_maps, 'fuse': fuse_attn_maps}, fuse_seq
