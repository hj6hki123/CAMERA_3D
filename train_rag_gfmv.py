#!/usr/bin/env python
import os, json, random, math, faiss, torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.utils.data import Dataset, DataLoader
from   torchvision import transforms
from   PIL import Image

from CAMERA_3D.models.rag_text_encoder  import RAGTextEncoder     # noqa
from CAMERA_3D.models.gf_mv_encoder     import GFMVEncoder        # noqa
from semantic_memory   import SemanticMemory     # noqa

# ════════════════════════════════════════════════════════════════════════
# 0.  DATASET  (jsonl => caption + V renders)
# ════════════════════════════════════════════════════════════════════════
class MultiViewObjaverse(Dataset):
    """每筆樣本: { "caption": str, "render_paths": [img0 … imgV-1] }"""

    def __init__(self, jsonl_path: str, num_views: int = 12, image_size: int = 224):
        self.recs = [json.loads(l) for l in open(jsonl_path, "r", encoding="utf-8")]
        self.num_views = num_views
        self.tr = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()   # range 0-1
        ])

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        rec   = self.recs[idx]
        paths = rec["render_paths"][: self.num_views]
        imgs  = [self.tr(Image.open(p).convert("RGB")) for p in paths]
        imgs  = torch.stack(imgs)                # (V,C,H,W)
        return rec["caption"], imgs

# ════════════════════════════════════════════════════════════════════════
# 1.  LOSSES
# ════════════════════════════════════════════════════════════════════════

def info_nce_loss(vis: torch.Tensor, txt: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """vis/txt both (B,512) and already on same device"""
    vis = F.normalize(vis, dim=1)
    txt = F.normalize(txt, dim=1)
    logits = vis @ txt.T / tau                     # (B,B)
    targets = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, targets)


def ranknet_loss(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    """pos (B,1)  neg (B,K)"""
    return F.softplus(neg - pos).mean()

# ════════════════════════════════════════════════════════════════════════
# 2.  RERANKER (可換成 cross-attention 版本)
# ════════════════════════════════════════════════════════════════════════
class SimpleReranker(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, q: torch.Tensor, v: torch.Tensor):
        """q,v (N,512) – returns (N,1) score"""
        return self.ffn(torch.cat([q, v], dim=1))

# ════════════════════════════════════════════════════════════════════════
# 3.  TRAINING UTILITIES
# ════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def build_visual_index(dataloader: DataLoader, vis_enc: GFMVEncoder) -> faiss.IndexFlatIP:
    """離線將整個 3-D 資料集 encode → FAISS 內積索引 (cosine)"""
    vecs = []
    for _, imgs in dataloader:
        vis_vec, _ = vis_enc(imgs.cuda(), None)   # (B,512)
        vecs.append(F.normalize(vis_vec, dim=1).cpu())
    vecs = torch.cat(vecs).numpy()
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, vecs

# ════════════════════════════════════════════════════════════════════════
# 4.  主訓練函式
# ════════════════════════════════════════════════════════════════════════
class CFG:
    # dataset
    train_jsonl = "train_meta.jsonl"
    batch_size  = 8
    num_views   = 12
    # stage-1
    lr1     = 2e-4
    epochs1 = 8
    tau     = 0.07
    # stage-2
    L       = 50
    lr2     = 1e-4
    epochs2 = 5
    # misc
    topk_ctx = 4
    ckpt_stage1 = "enc_stage1.pth"
    ckpt_rerank = "rerank_stage2.pth"
    num_workers  = 4


def train_stage1(cfg: CFG):
    ds = MultiViewObjaverse(cfg.train_jsonl, cfg.num_views)
    dl = DataLoader(ds, cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)

    txt_enc = RAGTextEncoder(top_k=cfg.topk_ctx).cuda()
    vis_enc = GFMVEncoder(num_views=cfg.num_views).cuda()
    opt     = torch.optim.AdamW(list(txt_enc.parameters()) + list(vis_enc.parameters()), lr=cfg.lr1)

    for epoch in range(cfg.epochs1):
        for caps, imgs in dl:
            imgs = imgs.cuda()
            txt_feat = txt_enc(list(caps))               # (B,512)
            vis_feat, _ = vis_enc(imgs, txt_feat)        # (B,512)
            loss = info_nce_loss(vis_feat, txt_feat, cfg.tau)

            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[Stage1 E{epoch}] InfoNCE={loss.item():.4f}")

    torch.save({"txt": txt_enc.state_dict(), "vis": vis_enc.state_dict()}, cfg.ckpt_stage1)
    print("✓ Stage-1 encoders saved →", cfg.ckpt_stage1)
    return txt_enc, vis_enc


def train_stage2(cfg: CFG, txt_enc: RAGTextEncoder, vis_enc: GFMVEncoder):
    # 凍結 encoder（先精排）
    txt_enc.eval(); vis_enc.eval()
    for p in txt_enc.parameters(): p.requires_grad = False
    for p in vis_enc.parameters(): p.requires_grad = False

    rerank = SimpleReranker().cuda()
    opt    = torch.optim.AdamW(rerank.parameters(), lr=cfg.lr2)

    base_ds = MultiViewObjaverse(cfg.train_jsonl, cfg.num_views)
    base_dl = DataLoader(base_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    index, all_vis_np = build_visual_index(base_dl, vis_enc)

    # 重新用 shuffle loader 取 query
    q_dl = DataLoader(base_ds, cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)

    for epoch in range(cfg.epochs2):
        for caps, _ in q_dl:
            q_vec = txt_enc(list(caps))                  # (B,512) 已歸一
            sims, idx = index.search(q_vec.cpu().numpy(), cfg.L)

            # 候選向量 & reshape
            can   = torch.from_numpy(all_vis_np[idx.reshape(-1)]).view(q_vec.size(0), cfg.L, -1).cuda()
            can   = F.normalize(can, dim=-1)

            pos_s = rerank(q_vec, can[:, 0, :])          # (B,1)
            q_tile = q_vec.unsqueeze(1).expand(-1, cfg.L-1, -1)  # (B,L-1,512)
            neg_in = torch.cat([q_tile, can[:, 1:, :]], dim=-1).reshape(-1, 1024)
            neg_s  = rerank.ffn(neg_in).view(-1, cfg.L-1)       # (B,L-1)

            loss = ranknet_loss(pos_s, neg_s)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"[Stage2 E{epoch}] RankNet={loss.item():.4f}")

    torch.save(rerank.state_dict(), cfg.ckpt_rerank)
    print("✓ Stage-2 reranker saved →", cfg.ckpt_rerank)

# ════════════════════════════════════════════════════════════════════════
# 5.  main
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = CFG()
    #  Stage-1
    txt_enc, vis_enc = train_stage1(cfg)
    #  Stage-2
    train_stage2(cfg, txt_enc, vis_enc)
    print("✔ 全流程訓練完成")
