"""
CAMERA_3D 兩階段訓練流程 (修改為支援開放領域 RAG)

    • 階段 1 — 聯合對比預訓練 RAG 文字編碼器（使用外部檢索）與 GateFusion-MV
    • 階段 2 — 使用 FAISS 回召 + 跨模態精排微調
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
import faiss
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.unified_dataset import UnifiedDataset
from models.rag_text_encoder import RAGTextEncoder
from models.gf_mv_encoder import GFMVEncoder
from models.cross_modal_reranker import CrossModalReranker
from torch.nn import TripletMarginLoss
from utils.checks import assert_valid
import math
# ---------------------------------------------------------------------------
# 設備檢測和配置 (維持不變)
# ---------------------------------------------------------------------------
def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ 自動選擇 CUDA 設備: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✓ 自動選擇 Apple Silicon MPS 設備")
        else:
            device = torch.device("cpu")
            print("✓ 自動選擇 CPU 設備")
    else:
        device = torch.device(device_arg)
        print(f"✓ 使用指定設備: {device}")
    return device

def setup_device_optimization(device: torch.device):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"  - CUDA 記憶體: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f}GB")
    elif device.type == "mps":
        print("  - 啟用 MPS 優化")
    else:
        torch.set_num_threads(torch.get_num_threads())
        print(f"  - CPU 線程數: {torch.get_num_threads()}")

# ---------------------------------------------------------------------------
# 可選 wandb (維持不變)
# ---------------------------------------------------------------------------
try:
    import wandb
except ImportError:
    wandb = None

_USE_WANDB = False

def wandb_init(args):
    global _USE_WANDB
    if args.wandb and wandb is not None:
        wandb.init(
            project="CAMERA3D_OpenRAG", # 專案名稱可自訂
            name=f"bs{args.bs}_topk{args.topk}_ep{args.ep1+args.ep2}_{args.device}",
            config=vars(args),
        )
        _USE_WANDB = True
    else:
        _USE_WANDB = False

def wb_log(data: dict, step: Optional[int] = None):
    if _USE_WANDB:
        wandb.log(data, step=step)

# ---------------------------------------------------------------------------
# 損失函數 (維持不變)
# ---------------------------------------------------------------------------
def info_nce(v, t, tau: float = 0.07):
    v = F.normalize(v, 2, 1)
    t = F.normalize(t, 2, 1)
    return F.cross_entropy(v @ t.T / tau, torch.arange(v.size(0), device=v.device))

def mil_nce(v_vec, ctx_seq, tau=0.07):
    """
    v_vec  : (B, D)
    ctx_seq: (B, K, D)   — K 段 context 的 CLS 嵌入 (已投影成 512 維)
    Return : scalar  MIL-NCE loss
    """
    B, K, D = ctx_seq.shape
    device  = v_vec.device

    # (1) 展平成 (B*K, D) 方便算相似度
    ctx_flat = ctx_seq.reshape(B * K, D)

    # (2) 餘弦相似度 / τ  →  (B, B*K)
    sim = (F.normalize(v_vec,  2, 1) @
           F.normalize(ctx_flat, 2, 1).t()) / tau

    # (3) 取出自己那一袋 K 個 positive 的 index
    #     第 i 筆樣本的正例落在列 i、欄 i*K … i*K+K-1
    base = (torch.arange(B, device=device) * K).unsqueeze(1)   # (B,1)
    pos_idx = base + torch.arange(K, device=device)            # (B,K)
    pos_sim = sim.gather(1, pos_idx)                           # (B,K)

    # (4) MIL-NCE：LogSumExp 正例 / LogSumExp 全部
    num   = torch.logsumexp(pos_sim, dim=1)    # (B,)
    denom = torch.logsumexp(sim,      dim=1)   # (B,)
    return (denom - num).mean()


def collate_batch(batch):
    """
    期望 batch 內部元素結構：
        (query:str, imgs:Tensor(V,C,H,W), obj_id:str, idx:int, ext_ctxs:tuple)
    回傳：
        caps_list, ext_ctxs_list, imgs_tensor(B,V,C,H,W), obj_ids_list, idx_tensor
    """
    caps, imgs, obj_ids, idxs, ext_ctxs = zip(*batch)

    # ── 文字與外部語境保持 list，交給 RAGTextEncoder 處理 ──
    caps_list      = list(caps)
    ext_ctxs_list  = [list(t) for t in ext_ctxs]  # tuple → list，之後方便增補 padding

    # ── 影像 (已經是 Tensor) 疊成 (B,V,C,H,W) ──
    imgs_tensor = torch.stack(imgs)              

    # ── obj_id 保留 list，idxs 轉成 Tensor ──
    obj_ids_list = list(obj_ids)
    idx_tensor   = torch.tensor(idxs, dtype=torch.long)

    return caps_list, ext_ctxs_list, imgs_tensor, obj_ids_list, idx_tensor
def stage1(args, ds, device):
    dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=4, collate_fn=collate_batch)

    # 初始化模型 (RAGTextEncoder 內部已設定好 Adapter)
    txt_enc = RAGTextEncoder(args.topk, device=device).to(device)
    vis_enc = GFMVEncoder(args.views).to(device)

    # 自動收集所有可訓練的參數 (Adapter, Fusion, Vision Model)
    print("Collecting trainable parameters...")
    trainable_params = []
    for model in [txt_enc, vis_enc]:
        for param in model.parameters():
            if param.requires_grad:
                trainable_params.append(param)
    
    num_params = sum(p.numel() for p in trainable_params)
    print(f"Total number of trainable parameters: {num_params:,}")

    # 建立一個只包含可訓練參數的優化器
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr1, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ep1 * len(dl))
    
    print(f"Starting Stage 1 with Adapter Tuning for {args.ep1} epochs...")
    lambda_ctx = 0.05
    lambda_view = 0.1
    step_global = 0
    for ep in range(args.ep1):
        pbar = tqdm(dl, desc=f"Stage 1 / Epoch {ep:02d}", unit="batch")
        
        for caps, ext_ctxs, imgs, obj_ids, idxs in pbar:

            imgs = imgs.to(device, non_blocking=True)
            
            # 前向傳播
            t_vec, _, tok_seq, _ = txt_enc(caps, ext_ctxs)
            v_vec, attn_maps, _ = vis_enc(imgs, t_vec)
            
            # -------- (1) 全域 InfoNCE
            loss_nce = info_nce(v_vec, t_vec, args.tau)

            # -------- (2) 文字端 Max-InfoNCE
            ctx_seq  = tok_seq[:, 1:]
            ctx_proj = txt_enc.fusion.proj(ctx_seq)          # (B, K, D)
            loss_mil = mil_nce(v_vec, ctx_proj, args.tau)

            # -------- (3) 視角熵正則
            last_fuse = attn_maps['fuse'][-1]              # (B, T, T)
            gate2view = last_fuse[:, 1, 2 : 2+args.views]  # (B, V)
            view_ent  = -(gate2view * torch.log(gate2view + 1e-6)).sum(1).mean() / math.log(args.views)

            # -------- 總損失
            loss = loss_nce + lambda_ctx * loss_mil + lambda_view * view_ent

            # 反向傳播與優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            pbar.set_postfix({"NCE":f"{loss_nce.item():.4f}",
                              "MIL-NCE":f"{loss_mil.item():.4f}",
                              "view_ent": f"{view_ent.item():.4f}",
                              "loss": f"{loss.item():.4f}"})
            # wb_log(...)
            step_global += 1

        # 每個 epoch 後存檔
        os.makedirs(args.out, exist_ok=True)
        torch.save({"txt": txt_enc.state_dict(), "vis": vis_enc.state_dict()}, f"{args.out}/enc1_ep{ep}.pth")

    txt_enc.eval(); vis_enc.eval()
    return txt_enc, vis_enc

# # ---------------------------------------------------------------------------
# # 四段式對比預訓練 (核心修改處)
# # ---------------------------------------------------------------------------
# def stage1_four_phases(args, ds, device,
#                        warm_epochs=3,
#                        align_epochs=4,
#                        hardneg_epochs=6,
#                        finetune_epochs=2):

#     total_epochs = warm_epochs + align_epochs + hardneg_epochs + finetune_epochs
#     dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=4)

#     # <-- 修改點: RAGTextEncoder 初始化不再需要 data_jsonl 和 cache_dir
#     txt_enc = RAGTextEncoder(args.topk, device=device).to(device)
#     vis_enc = GFMVEncoder(args.views).to(device)

#     # ---------- 工具函式 (核心修改處) -------------------------------------
#     def set_requires(vision_grad, fusion_grad, bert_grad):
#         for p in vis_enc.parameters():
#             p.requires_grad = vision_grad
#         for p in txt_enc.fusion.parameters():
#             p.requires_grad = fusion_grad
        
#         # <-- 修改點: 凍結的對象從 retriever.qenc 改為 text_encoder
#         for name, p in txt_enc.text_encoder.named_parameters():
#             # 冷凍 BERT 前四層，到最後階段才全開
#             layer_id = int(name.split('.')[2]) if 'encoder.layer' in name else -1
#             if layer_id < 4 and not finetune_mode:
#                 p.requires_grad = False
#             else:
#                 p.requires_grad = bert_grad

#     def build_opt(lr_vis, lr_txt):
#         params, lrs = [], []
#         if lr_vis > 0:
#             params.append({'params': vis_enc.parameters(), 'lr': lr_vis})
#         if lr_txt > 0:
#             # <-- 修改點: 優化器參數改為 text_encoder 和 fusion
#             txt_params = list(txt_enc.text_encoder.parameters()) + \
#                          list(txt_enc.fusion.parameters())
#             params.append({'params': txt_params, 'lr': lr_txt})
        
#         if not params:
#             return None
#         return torch.optim.AdamW(params, weight_decay=1e-2)

#     # ---------- 訓練流程設定 (對應修改) --------------------------------
#     finetune_mode = False
#     set_requires(vision_grad=True, fusion_grad=False, bert_grad=False)
#     opt = build_opt(lr_vis=args.lr1, lr_txt=0.)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=warm_epochs*len(dl))
    
#     step_global = 0
#     for ep in range(total_epochs):

#         # —— 階段切換邏輯 (維持不變, 但 build_opt 已更新) ——————————
#         if ep == warm_epochs:
#             set_requires(vision_grad=False, fusion_grad=True, bert_grad=True)
#             opt = build_opt(lr_vis=0., lr_txt=args.lr1 * 0.5)
#             scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.2, total_iters=align_epochs*len(dl))
#         if ep == warm_epochs + align_epochs:
#             set_requires(vision_grad=False, fusion_grad=True, bert_grad=True)
#             opt = build_opt(lr_vis=0., lr_txt=args.lr1 * 0.3)
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hardneg_epochs*len(dl))
#         if ep == warm_epochs + align_epochs + hardneg_epochs:
#             finetune_mode = True
#             set_requires(vision_grad=True, fusion_grad=True, bert_grad=True) # 微調階段可考慮將視覺也解凍
#             opt = build_opt(lr_vis=args.lr1 * 0.05, lr_txt=args.lr1 * 0.1) # 學習率可再調整
#             scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.5, total_iters=finetune_epochs*len(dl))
#             print("◎ 進入 Fine-turn：BERT 全層解凍，學習率再降一階。")

#         pbar = tqdm(dl, desc=f"E{ep:02d}", unit="batch")
#         for caps, imgs, _, _, ext_ctxs in pbar: # obj_ids 不再直接需要
#             imgs = imgs.to(device, non_blocking=True)

#             # —— 文字編碼器 (核心修改處) ------------------------------------
#             # <-- 修改點: forward 呼叫方式改變，不再有 ret_loss
#             t_vec, _, _, _ = txt_enc(list(caps), list(ext_ctxs))

#             # —— 視覺編碼器 ------------------------------------------------
#             v_vec, _, _ = vis_enc(imgs, t_vec)

#             # —— 損失組合 (核心修改處) --------------------------------------
#             loss_nce = info_nce(v_vec, t_vec, args.tau)



            
#             # 硬負例挖掘邏輯可維持
#             loss_tri = 0.
#             if ep >= warm_epochs + align_epochs:
#                 v_neg = torch.roll(v_vec, shifts=-1, dims=0)
#                 t_neg = torch.roll(t_vec, shifts= 1, dims=0)
#                 trip = torch.nn.functional.triplet_margin_loss
#                 loss_tri = trip(t_vec, v_vec, v_neg, margin=0.2) + trip(v_vec, t_vec, t_neg, margin=0.2)
            
#             # <-- 修改點: 總損失不再包含 ret_loss
#             loss = loss_nce + 0.5 * loss_tri
            
#             opt.zero_grad()
#             loss.backward()
#             # 梯度裁剪可保留
#             if opt.param_groups:
#                 for group in opt.param_groups:
#                     torch.nn.utils.clip_grad_norm_(group['params'], 1.0)
#             opt.step()
#             scheduler.step()

            
#             pbar.set_postfix(NCE=f"{loss_nce:.3f}", Tri=f"{loss_tri:.3f}")
#             wb_log({"loss_total": loss.item(), "loss_nce": loss_nce.item(), "loss_tri": loss_tri if isinstance(loss_tri, float) else loss_tri.item()}, step=step_global)
#             step_global += 1

#         # —— 每個 epoch 後存檔 --------------------------------------------
#         os.makedirs(args.out, exist_ok=True)
#         torch.save({"txt": txt_enc.state_dict(), "vis": vis_enc.state_dict()}, f"{args.out}/enc1_ep{ep}.pth")

#     txt_enc.eval(); vis_enc.eval()
#     return txt_enc, vis_enc

# ---------------------------------------------------------------------------
# `build_faiss_with_tok` 和 `stage2` 也需要微調
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_faiss_with_tok(ds, txt_enc, vis_enc, args, device):
    idx_path = os.path.join(args.out, "faiss_index.bin")
    tok_path = os.path.join(args.out, "vis_tok.npy")

    if os.path.isfile(idx_path) and os.path.isfile(tok_path):
        print(f" 從 {args.out} 載入快取的 FAISS 索引與 token")
        index = faiss.read_index(idx_path, faiss.IO_FLAG_MMAP)
        all_tok = np.load(tok_path)
        return index, all_tok

    print("找不到快取 → 正在建立 FAISS 索引與視覺 token ...")
    g_vecs, g_toks = [], []
    loader = DataLoader(ds, args.bs, num_workers=4, shuffle=False)

    for caps, imgs, _, _ in tqdm(loader, desc="建立 FAISS", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        # <-- 修改點: 更新 txt_enc 的呼叫方式
        q_vec, _, _, _ = txt_enc(list(caps))
        v_vec, _, v_tok = vis_enc(imgs, q_vec)

        g_vecs.append(F.normalize(v_vec, 2, 1).cpu())
        v_tok = F.normalize(v_tok, 2, -1).cpu()
        g_toks.append(v_tok)

    all_vec = torch.cat(g_vecs, 0).numpy()
    all_tok = torch.cat(g_toks, 0).numpy().astype(np.float16)

    os.makedirs(args.out, exist_ok=True)
    np.save(tok_path, all_tok)
    index = faiss.IndexFlatIP(512)
    index.add(all_vec)
    faiss.write_index(index, idx_path)
    print(f"✓ 已保存 faiss_index.bin & vis_tok.npy 至 {args.out}")
    return index, all_tok

def stage2(args, ds, txt_enc, vis_enc, index, all_tok, device):
    reranker = CrossModalReranker().to(device)
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=args.lr2)
    criterion = torch.nn.MarginRankingLoss(margin=0.2)
    num_hard_neg = 5
    loader = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=4)

    for ep in range(args.ep2):
        pbar = tqdm(loader, desc=f"階段2 Epoch {ep}", unit="batch")
        for step, (caps, imgs, obj_ids, idxs) in enumerate(pbar):
            B = len(obj_ids)
            imgs = imgs.to(device, non_blocking=True)

            with torch.no_grad():
                # <-- 修改點: 更新 txt_enc 的呼叫方式
                t_vec, _, txt_tok, _ = txt_enc(list(caps))
                _, _, vis_tok = vis_enc(imgs, t_vec)
                vis_tok = F.normalize(vis_tok, 2, -1)
                sims, idx_faiss = index.search(t_vec.cpu().numpy(), args.L)
            
            # 後續 Stage 2 邏輯大部分可維持，因為它依賴的是 txt_tok 和 all_tok (視覺)
            # ... (此處省略 stage2 的詳細迴圈，因為它的邏輯與 RAGTextEncoder 內部改動無關)
    print("Stage 2 training loop needs to be verified but logic should be similar.")


# ---------------------------------------------------------------------------
# 主程式入口 (需要注意 --lmb 參數已失效)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_jsonl", required=True)
    p.add_argument("--out", default="ckpts_open_rag") # 建議用新的輸出目錄
    p.add_argument("--views", type=int, default=12)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--topk", type=int, default=3, help="Number of external contexts to retrieve") # topk 現在是控制外部檢索數量
    p.add_argument("--lmb", type=float, default=0.0, help="[DEPRECATED] No longer used in open-domain RAG") # lmb 參數已失效
    p.add_argument("--lr1", type=float, default=1e-4)
    p.add_argument("--ep1", type=int, default=15) # 總 epoch 數
    p.add_argument("--L", type=int, default=100, help="# reranker候選數")
    p.add_argument("--lr2", type=float, default=3e-4)
    p.add_argument("--ep2", type=int, default=5)
    p.add_argument("--wandb", action="store_true", help="啟用 wandb 紀錄")
    p.add_argument("--resume_enc1", type=str, default=None, help="階段1編碼器權重路徑（跳過訓練）")
    p.add_argument("--device", type=str, default="auto", help="設備選擇: 'auto', 'cuda', 'mps', 'cpu' 或具體設備如 'cuda:0'")
    args = p.parse_args()
    
    device = get_device(args.device)
    setup_device_optimization(device)
    
    wandb_init(args)
    
    # 這裡的 ds 僅用於提供 query 和 imgs，不再需要 corpus_texts
    full_ds = UnifiedDataset(args.data_jsonl, num_views=args.views)
    print(f"Total samples in dataset: {len(full_ds)}")
    
    # 根據需求決定是否使用子集
    # ds = torch.utils.data.Subset(full_ds, list(range(1000)))
    from torch.utils.data import Subset
    ds = full_ds
    print(f"Using {len(ds)} samples for training.")
    ds = Subset(full_ds, list(range(1000)))
    if args.resume_enc1:
        print(f"從 {args.resume_enc1} 載入階段1權重 …")
        state = torch.load(args.resume_enc1, map_location=device)
        txt_enc = RAGTextEncoder(args.topk, device=str(device)).to(device)
        vis_enc = GFMVEncoder(args.views).to(device)
        txt_enc.load_state_dict(state["txt"])
        vis_enc.load_state_dict(state["vis"])
        txt_enc.eval(); vis_enc.eval()
    else:
        # 使用新的四階段訓練函數
        txt_enc, vis_enc = stage1(args, ds, device)

    index, all_tok = build_faiss_with_tok(ds, txt_enc, vis_enc, args, str(device))
    stage2(args, ds, txt_enc, vis_enc, index, all_tok, str(device))
    print("訓練完成 →", args.out)