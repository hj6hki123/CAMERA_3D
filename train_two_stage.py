#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CAMERA_3D 兩階段訓練流程

    • 階段 1 — 聯合對比預訓練 RAG 文字編碼器與 GateFusion-MV
    • 階段 2 — 使用 FAISS 回召 + 跨模態精排微調

可選的 Weights & Biases (wandb) 日誌紀錄
---------------------------------
安裝 wandb 且帶入 `--wandb` 參數即可啟用日誌，否則所有 wandb 呼叫皆為空操作。
"""
import os
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
from utils.checks import (assert_valid,
                          check_gradients,
                          stable_rank_loss,
                          xavier_init)

# ---------------------------------------------------------------------------
# 可選 wandb (延遲匯入 + 優雅降級)
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
            project="CAMERA3D",
            name=f"bs{args.bs}_topk{args.topk}_ep{args.ep1+args.ep2}",
            config=vars(args),
        )
        _USE_WANDB = True
    else:
        _USE_WANDB = False


def wb_log(data: dict, step: Optional[int] = None):
    if _USE_WANDB:
        wandb.log(data, step=step)

# ---------------------------------------------------------------------------
# 視覺化：將相似度熱力圖紀錄到 wandb
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image

def log_sim_heatmap(v_vec, t_vec, step, tag="stage1/sim_matrix", max_show=16):
    if not _USE_WANDB:
        return
    with torch.no_grad():
        v = F.normalize(v_vec, 2, 1)[:max_show]
        t = F.normalize(t_vec, 2, 1)[:max_show]
        sim = (v @ t.T).cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(sim, vmin=-1, vmax=1, cmap="viridis")
    ax.set_title(f"SimMatrix @ step {step}")
    ax.set_xlabel("text_idx"); ax.set_ylabel("vis_idx")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    buf = io.BytesIO()
    fig.tight_layout(); fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    wandb.log({tag: wandb.Image(img)}, step=step)

def log_gradients(models: dict, top_n: int = 5, prefix: str = "grad"):
    grad_stats = {}
    for mod_name, module in models.items():
        for name, param in module.named_parameters():
            if param.grad is not None:
                key = f"{mod_name}.{name}"
                grad_stats[key] = param.grad.norm().item()
    if not grad_stats:
        print(" 無梯度資料。")
        return

    # 排序
    sorted_stats = sorted(grad_stats.items(), key=lambda x: x[1])
    print(f"=== {prefix} 範圍（最低 {top_n}） ===")
    for k, v in sorted_stats[:top_n]:
        print(f"{k:60s}: {v:.4e}")
    print(f"=== {prefix} 範圍（最高 {top_n}） ===")
    for k, v in sorted_stats[-top_n:]:
        print(f"{k:60s}: {v:.4e}")




# ---------------------------------------------------------------------------
# 損失函數
# ---------------------------------------------------------------------------
def info_nce(v, t, tau: float = 0.07):
    v = F.normalize(v, 2, 1)
    t = F.normalize(t, 2, 1)
    return F.cross_entropy(v @ t.T / tau, torch.arange(v.size(0), device=v.device))

def ranknet(pos, neg):
    return F.softplus(neg - pos).mean()

# ---------------------------------------------------------------------------
# 階段 1 – 對比預訓練
# ---------------------------------------------------------------------------
def stage1(args, ds):
    print("階段1開始…")
    dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=4)
    txt_enc = RAGTextEncoder(args.data_jsonl, args.topk).cuda()
    # # 凍結 BERT（只訓練 GateFusion-MV） 
    # for p in txt_enc.retriever.qenc.parameters():
    #     p.requires_grad = False
    vis_enc = GFMVEncoder(args.views).cuda()
    
    ## TODO: 嘗試調整不同學習率
    bert_low_lr = 1e-5 
    
    param_groups = [
        {"params": txt_enc.retriever.qenc.parameters(), "lr": bert_low_lr},
        {"params": txt_enc.fusion.parameters(), "lr": args.lr1},
        {"params": vis_enc.parameters(), "lr": args.lr1},
        ]
    
    opt = torch.optim.AdamW(
        param_groups, lr=args.lr1 ,weight_decay=1e-2,
    )

    for ep in range(args.ep1):
        pbar = tqdm(dl, desc=f"階段1 Epoch {ep}", unit="batch")
        for step, (cap, imgs, obj_id, idx) in enumerate(pbar):
            imgs = imgs.cuda(non_blocking=True)
            # 1) Text fusion → 拿到四項：vec, retriever loss, tok, fusion_attn_maps
            t_vec, ret_loss, txt_tok, fusion_attn_maps = txt_enc(
                q_list=list(cap),
                obj_ids=list(obj_id),
                return_loss=True
            )
            # 2) Visual (GateFusion) → global vec, vis_attn_maps, tok_vis
            v_vec, vis_attn_maps, vis_tok = vis_enc(imgs, t_vec)

            # 計算 loss：InfoNCE + retriever loss
            loss = info_nce(v_vec, t_vec, args.tau) + args.lmb * ret_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            wb_log({"stage1/loss": loss.item()}, step=ep * len(dl) + step)
            # 每 1000 steps 紀錄一次 SimHeatmap
            if _USE_WANDB and (step % 1000 == 0):
                log_sim_heatmap(v_vec.detach(), t_vec.detach(), step=ep * len(dl) + step)

                # 3.a 紀錄 FusionBlock 第一層第一個 head 的 attention
                # fusion_attn_maps[0] shape=(B, heads, T, T)
                # 取 batch=0, head=0 的那張 (T×T) heatmap
                fa = fusion_attn_maps[0][0].detach().cpu().numpy()  # (T, T)
                fig, ax = plt.subplots(figsize=(4,4))
                im = ax.imshow(fa, vmin=0, vmax=1, cmap="viridis")
                ax.set_title(f"Fusion Attn L0 H0 @ step {step}")
                fig.colorbar(im, fraction=0.04, pad=0.03)
                buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png")
                plt.close(fig); buf.seek(0)
                wandb.log({f"stage1/fusion_attn": wandb.Image(Image.open(buf))},
                          step=ep * len(dl) + step)

                # 3.b 紀錄 GateFusion 第一層第一個 head 的 attention
                # vis_attn_maps[0] shape=(B, heads, T, T)
                va = vis_attn_maps[0][0].detach().cpu().numpy()  # (T, T)
                fig, ax = plt.subplots(figsize=(4,4))
                im = ax.imshow(va, vmin=0, vmax=1, cmap="viridis")
                ax.set_title(f"GateFusion Attn L0 H0 @ step {step}")
                fig.colorbar(im, fraction=0.04, pad=0.03)
                buf = io.BytesIO(); fig.tight_layout(); fig.savefig(buf, format="png")
                plt.close(fig); buf.seek(0)
                wandb.log({f"stage1/vis_attn": wandb.Image(Image.open(buf))},
                          step=ep * len(dl) + step)

        # 每個 epoch 結束 存一次 encoder weights
        os.makedirs(args.out, exist_ok=True)
        torch.save({"txt": txt_enc.state_dict(), "vis": vis_enc.state_dict()},
                   f"{args.out}/enc1.pth")

    return txt_enc.eval(), vis_enc.eval()

@torch.no_grad()
def build_faiss_with_tok(ds, txt_enc, vis_enc, args):
    """
    回傳：
      - index: Faiss IndexFlatIP(512)，保存所有攤平成樣本的全局向量
      - all_tok: (N, T_vis, 512) numpy 陣列，保存每筆攤平成樣本的多視角 token 序列
    """
    idx_path = os.path.join(args.out, "faiss_index.bin")
    tok_path = os.path.join(args.out, "vis_tok.npy")

    if os.path.isfile(idx_path) and os.path.isfile(tok_path):
        print(f"✓ 從 {args.out} 載入快取的 FAISS 索引與 token")
        index   = faiss.read_index(idx_path, faiss.IO_FLAG_MMAP)
        all_tok = np.load(tok_path)
        return index, all_tok

    print("✗ 找不到快取 → 正在建立 FAISS 索引與 token ...")
    g_vecs, g_toks = [], []
    loader = DataLoader(ds, args.bs, num_workers=4, shuffle=False)

    for caps, imgs, _, _ in tqdm(loader, desc="建立 FAISS", unit="batch"):
        imgs = imgs.cuda(non_blocking=True)
        # 1) 文字編碼器：
        #q_vec, _, _ = txt_enc(caps, [None]*len(caps))
        q_vec, _, _, _ = txt_enc(list(caps), obj_ids=None)

        # 2) 多視角視覺編碼器：
        v_vec,_, v_tok = vis_enc(imgs, q_vec)   # v_vec: (B, 512), v_tok: (B, T_vis, 512)

        g_vecs.append(F.normalize(v_vec, 2, 1).cpu())
        v_tok = F.normalize(v_tok, 2, -1).cpu()
        g_toks.append(v_tok)

    all_vec = torch.cat(g_vecs, 0).numpy()                      # (N, 512)
    all_tok = torch.cat(g_toks, 0).numpy().astype(np.float16)   # (N, T_vis, 512)

    os.makedirs(args.out, exist_ok=True)
    np.save(tok_path, all_tok)

    index = faiss.IndexFlatIP(512)
    index.add(all_vec)
    faiss.write_index(index, idx_path)
    print(f"✓ 已保存 faiss_index.bin & vis_tok.npy 至 {args.out}")
    return index, all_tok


def stage2(args, ds, txt_enc, vis_enc, index, all_tok):
    """
    階段2：粗排(FAISS top-L)→精排（reranker）微調
    ds.obj2idx 已是攤平後 self.items 的映射
    index 返回攤平後的全局索引
    all_tok 與攤平序號一一對應
    """
    reranker = CrossModalReranker().cuda()
    # 若已有預訓練 reranker，可直接載入：
    # reranker.load_state_dict(torch.load(os.path.join(args.out, "rerank.pth")))
    optimizer = torch.optim.AdamW(reranker.parameters(), lr=args.lr2)
    criterion = torch.nn.MarginRankingLoss(margin=0.2)

    num_hard_neg = 5
    loader = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=4)

    for ep in range(args.ep2):
        pbar = tqdm(loader, desc=f"階段2 Epoch {ep}", unit="batch")
        for step, (caps, imgs, obj_ids, idxs) in enumerate(pbar):
            B = len(obj_ids)
            imgs = imgs.cuda(non_blocking=True)

            # ——— 1) 用 txt_enc & vis_enc 取得本批次文字向量 t_vec、文字 token txt_tok、以及視覺 token vis_tok ———
            with torch.no_grad():
                t_vec, _, txt_tok, _ = txt_enc(
                    q_list=list(caps),
                    obj_ids=list(obj_ids),
                    return_loss=False
                )
                _, vis_attn_maps, vis_tok = vis_enc(imgs, t_vec)   # vis_tok: (B, T_vis, 512)
                vis_tok = F.normalize(vis_tok, 2, -1)

                sims, idx_faiss = index.search(t_vec.cpu().numpy(), args.L)
                # idx_faiss：形狀為 (B, L)，存攤平成的全局索引 0…N-1

            # ——— 2) 挑選正例與硬負例，並計算 reranker 分數 ———
            s_pos_list, s_neg_list = [], []
            for b in range(B):
                gt_flat_idx = idxs[b].item()
                all_pos_for_this_obj = ds.obj2idx[obj_ids[b]]  # list of all flat-indices for this obj_id

                # 取得 FAISS top-L 的候選清單
                cand = idx_faiss[b].tolist()
                # 確保 ground truth 第一
                if gt_flat_idx in cand:
                    cand.remove(gt_flat_idx)
                    cand = [gt_flat_idx] + cand
                else:
                    cand = [gt_flat_idx] + cand[:-1]

                # 從 cand[1:] 過濾掉同一 obj_id 的「其他正例」，剩下才是真正負例候選
                neg_cands = [x for x in cand[1:] if x not in all_pos_for_this_obj]
                hard_negs = neg_cands[:num_hard_neg]  # 取前 num_hard_neg 個硬負例

                # 文字 token：txt_tok[b] shape = (1+top_k, 768)，
                # expand 成 (num_hard_neg, 1+top_k, 768) 以便一次送入 reranker
                t_tok_b = txt_tok[b : b+1]                            # shape = (1, 1+top_k, 768)
                t_tok_expand = t_tok_b.expand(num_hard_neg, -1, -1)   # shape = (num_hard_neg, 1+top_k, 768)

                # 視覺 token：從 all_tok 擷取 pos 與 neg
                pos_vtok = torch.from_numpy(all_tok[gt_flat_idx]).float().cuda()       # (T_vis, 512)
                neg_vtok = torch.from_numpy(all_tok[hard_negs]).float().cuda()          # (num_hard_neg, T_vis, 512)
                pos_vtok_expand = pos_vtok.unsqueeze(0).expand(num_hard_neg, -1, -1)     # (num_hard_neg, T_vis, 512)

                # 一次把正例與多個硬負例丟給 reranker 拿分數
                x_pos = reranker(t_tok_expand, pos_vtok_expand)  # shape = (num_hard_neg,)
                x_neg = reranker(t_tok_expand, neg_vtok)         # shape = (num_hard_neg,)

                s_pos_list.append(x_pos)
                s_neg_list.append(x_neg)

            # ——— 3) 組裝 margin ranking loss ———
            # s_pos_list: length=B，每元素 shape=(num_hard_neg,)  (正例分數 replicated)
            pos_tensor = torch.stack([x[0] for x in s_pos_list])   # shape = (B,)
            # s_neg_list: length=B，每元素 shape=(num_hard_neg,)
            neg_tensor = torch.stack(s_neg_list)                    # shape = (B, num_hard_neg)

            target = torch.ones_like(neg_tensor)  # shape=(B, num_hard_neg) 全 1
            loss = criterion(pos_tensor.unsqueeze(1), neg_tensor, target)

            # ——— 4) 反向傳播與優化更新 ———
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(epoch=ep, loss=loss.item())

            # ====== W&B Logging ======
            # 1) Stage2 每 batch 都 log loss
            wb_log({"stage2/loss": loss.item()},
                   step=(args.ep1 + ep) * len(loader) + step)

            # 2) 每隔 1000 steps log 一張「相似度矩陣 heatmap」
            if _USE_WANDB and (step % 1000 == 0):
                # 重新 forward 一筆（batch[0]）去拿 global v_vec
                with torch.no_grad():
                    t_v, _, _, _ = txt_enc([caps[0]], [obj_ids[0]])
                    v_v, _, _ = vis_enc(imgs[0:1].cuda(), t_v)
                    v_norm = F.normalize(v_v, 2, 1)[:16]
                    t_norm = F.normalize(t_v, 2, 1)[:16]
                    sim2 = (v_norm @ t_norm.T).cpu().numpy()  # (<=16, <=16)

                fig, ax = plt.subplots(figsize=(4,4))
                im = ax.imshow(sim2, vmin=-1, vmax=1, cmap="viridis")
                ax.set_title(f"stage2 sim @ ep{ep} step{step}")
                fig.colorbar(im, fraction=0.04, pad=0.03)
                buf = io.BytesIO()
                fig.tight_layout()
                fig.savefig(buf, format="png")
                plt.close(fig)
                buf.seek(0)
                wandb.log({f"stage2/sim_matrix":
                       wandb.Image(Image.open(buf))},
                       step=(args.ep1 + ep) * len(loader) + step)

            # # 3) 每隔 2000 steps log 一組「Retrieval 範例 Carousel」
            # if _USE_WANDB and (step % 2000 == 0):
            #     b0 = 0  # 只示範 batch 中的第 0 筆
            #     query_text = caps[b0]
            #     # FAISS top-5 OIDs
            #     faiss_idxs = idx_faiss[b0][:5].tolist()
            #     faiss_oids = [ds.items[i]["obj_id"] for i in faiss_idxs]

                # Rerank top-5 OIDs
                # scores = reranker(txt_tok[b0:b0+1].expand(args.L, -1, -1),
                #                   torch.from_numpy(all_tok[idx_faiss[b0]]).float().cuda())
                # order = torch.argsort(scores, descending=True).cpu().numpy()
                # rerank_oids = [ds.items[idx_faiss[b0][i]]["obj_id"] for i in order[:5]]

                # images_to_log = []
                # for oid in faiss_oids:
                #     img_path = os.path.join(ds.base_view_dir, oid, "000.png")
                #     images_to_log.append(wandb.Image(img_path, caption=f"FAISS: {oid}"))
                # for oid in rerank_oids:
                #     img_path = os.path.join(ds.base_view_dir, oid, "000.png")
                #     images_to_log.append(wandb.Image(img_path, caption=f"RERANK: {oid}"))

                # wandb.log({
                #     "stage2/query": query_text,
                #     "stage2/retrieval_examples": images_to_log
                # }, step=(args.ep1 + ep) * len(loader) + step)

        # 每個 epoch 結束存檔 reranker 權重
        torch.save(reranker.state_dict(), os.path.join(args.out, "rerank.pth"))
# ---------------------------------------------------------------------------
# 命令列參數解析
# ---------------------------------------------------------------------------
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data_jsonl", required=True)
    p.add_argument("--out", default="ckpts")
    p.add_argument("--views", type=int, default=12)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--lmb", type=float, default=0.1)
    p.add_argument("--lr1", type=float, default=2e-4)
    p.add_argument("--ep1", type=int, default=8)
    p.add_argument("--L", type=int, default=100, help="# reranker候選數")
    p.add_argument("--lr2", type=float, default=3e-4)
    p.add_argument("--ep2", type=int, default=5)
    p.add_argument("--wandb", action="store_true", help="啟用 wandb 紀錄")
    p.add_argument("--resume_enc1", type=str, default=None,
               help="階段1編碼器權重路徑（跳過訓練）")
    return p.parse_args()

if __name__ == "__main__":
    args = parse()
    wandb_init(args)
    ds = UnifiedDataset(args.data_jsonl, num_views=args.views)
    if args.resume_enc1:  # 載入階段1模型
        print(f"從 {args.resume_enc1} 載入階段1權重 …")
        state = torch.load(args.resume_enc1, map_location="cuda")
        txt_enc = RAGTextEncoder(args.data_jsonl, args.topk).cuda()
        vis_enc = GFMVEncoder(args.views).cuda()
        txt_enc.load_state_dict(state["txt"])
        vis_enc.load_state_dict(state["vis"])
        txt_enc.eval(); vis_enc.eval()
    else:
        txt_enc, vis_enc = stage1(args, ds)

    index, all_tok = build_faiss_with_tok(ds, txt_enc, vis_enc, args)
    stage2(args, ds, txt_enc, vis_enc, index, all_tok)
    print("訓練完成 →", args.out)
