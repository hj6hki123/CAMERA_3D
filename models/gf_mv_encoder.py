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
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):  # (B, T, D)
        return x + self.pe[:, :x.size(1)]

class Block(nn.Module):
    def __init__(self, d=512, h=8, p=.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        # 啟用 need_weights=True
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d, d*4), nn.GELU(),
            nn.Dropout(p), nn.Linear(d*4, d)
        )

    def forward(self, x):
        # ln + self-attn，並取回 attn_weights
        attn_out, attn_weights = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=True)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_weights   # attn_weights shape = (B, num_heads, T, T)

class GFMVEncoder(nn.Module):
    def __init__(self, num_views=12, dim=512, heads=8, layers=3):
        super().__init__()
        self.dim, self.V = dim, num_views

        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.feat2d = nn.Sequential(*list(vgg.features.children()))
        self.map2d  = nn.Linear(512*7*7, dim)

        self.cls_tok   = nn.Parameter(torch.randn(1, 1, dim))
        self.gate_proj = nn.Linear(512, dim)
        self.pos       = PosEnc(dim, num_views + 2)  # CLS + GATE + V
        # 用 Block 來保留每層 attn weights
        self.blocks    = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.pool      = nn.AdaptiveMaxPool1d(1)
        self.out_proj  = nn.Linear(dim, 512)

    def forward(self, imgs, txt_vec=None):
        """
        imgs: (B, V, C, H, W)
        txt_vec: (B, 512) or None
        回傳：global_vec: (B,512), 所有層的視覺 attn weights list, tok_seq: (B, T, 512)
        """
        B, V, C, H, W = imgs.shape
        # 1) extract 2D feature
        f = self.feat2d(imgs.view(B*V, C, H, W))                       # (B*V, 512, 7,7)
        f = self.map2d(f.flatten(1)).view(B, V, -1)                    # (B, V, 512)

        # 2) 準備 CLS + Gate + 每張 view 的 token
        if txt_vec is None:
            txt_vec = torch.zeros(B, 512, device=imgs.device)
        gate = self.gate_proj(txt_vec).unsqueeze(1)  # (B,1,512)
        # tok: (B, 1 + 1 + V, 512) ＝ CLS + Gate + V×feat
        tok = torch.cat([self.cls_tok.expand(B, -1, -1), gate, f], dim=1)
        tok = self.pos(tok)

        # 3) 逐層 Block，每層都回傳 attn weights
        all_attn_maps = []  # list of length = layers，每個 element 形狀 (B, heads, T, T)
        for blk in self.blocks:
            tok, attn_w = blk(tok)
            all_attn_maps.append(attn_w)

        # 4) global pooling
        global_vec = self.pool(tok.transpose(1,2)).squeeze(-1)  # (B,dim)
        global_vec = self.out_proj(global_vec)                 # (B,512)

        return global_vec, all_attn_maps, tok  # tok for potential logging；all_attn_maps 用來畫 heatmap
