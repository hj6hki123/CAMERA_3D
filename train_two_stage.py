"""CAMERA_3D 兩階段訓練流程

    • 階段 1 — 聯合對比預訓練 RAG 文字編碼器與 GateFusion-MV
    • 階段 2 — 使用 FAISS 回召 + 跨模態精排微調

可選的 Weights & Biases (wandb) 日誌紀錄
---------------------------------
安裝 wandb 且帶入 `--wandb` 參數即可啟用日誌，否則所有 wandb 呼叫皆為空操作。
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
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

# ---------------------------------------------------------------------------
# 設備檢測和配置
# ---------------------------------------------------------------------------
def get_device(device_arg: str = "auto") -> torch.device:
    """
    根據參數和硬件可用性選擇最佳設備
    
    Args:
        device_arg: 'auto', 'cuda', 'mps', 'cpu' 或具體設備如 'cuda:0'
    
    Returns:
        torch.device: 選定的設備
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f" 自動選擇 CUDA 設備: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print(" 自動選擇 Apple Silicon MPS 設備")
        else:
            device = torch.device("cpu")
            print(" 自動選擇 CPU 設備")
    else:
        device = torch.device(device_arg)
        print(f" 使用指定設備: {device}")
    
    return device

def setup_device_optimization(device: torch.device):
    """根據設備類型進行相應優化設置"""
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
# 可選 wandb (延遲匯入)
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
# 損失函數
# ---------------------------------------------------------------------------
def info_nce(v, t, tau: float = 0.07):
    v = F.normalize(v, 2, 1)
    t = F.normalize(t, 2, 1)
    return F.cross_entropy(v @ t.T / tau, torch.arange(v.size(0), device=v.device))

def mil_nce(v_vec, ctx_seq, tau=0.07):
    """
    v_vec : (B, 512)          Gate-Fusion 影像全域向量
    ctx_seq: (B, K, 512)      Cross-Fusion 投影後的 K 段 context CLS
    """
    B, K, D = ctx_seq.shape
    ctx_flat = ctx_seq.reshape(B * K, D)                    # (B*K,512)

    sim = (F.normalize(v_vec,  2, 1) @                      # (B,512)
           F.normalize(ctx_flat, 2, 1).t()) / tau           # (B, B*K)

    # 取自己袋內 K 個 positive 的 index
    base   = (torch.arange(B, device=v_vec.device) * K).unsqueeze(1)  # (B,1)
    pos_id = base + torch.arange(K, device=v_vec.device)              # (B,K)
    pos_sim= sim.gather(1, pos_id)                                    # (B,K)

    num   = torch.logsumexp(pos_sim, dim=1)           # (B,)
    denom = torch.logsumexp(sim,     dim=1)           # (B,)
    return (denom - num).mean()

def stage1(args, ds, device):
    dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=4)

    # ── 1. 建模 ────────────────────────────────────────────────────────────
    txt_enc = RAGTextEncoder(args.data_jsonl,
                             args.topk,
                             device=device,
                             cache_dir=args.out).to(device)
    vis_enc = GFMVEncoder(args.views).to(device)

    # ── 2. 參數凍結策略 ───────────────────────────────────────────────────
    # memory side (kenc) 永遠凍結
    for p in txt_enc.retriever.kenc.parameters():
        p.requires_grad = False

    # qenc → 只開啟 adapter 層
    for n, p in txt_enc.retriever.qenc.named_parameters():
        p.requires_grad = "adapters" in n          # 其餘權重凍結

    # Cross-Fusion 與 Gate-Fusion 全開
    pg_qenc  = [p for p in txt_enc.retriever.qenc.parameters() if p.requires_grad]
    pg_xfus  = list(txt_enc.fusion.parameters())          # Cross-Fusion
    pg_gate  = list(vis_enc.parameters())                 # Gate-Fusion-MV

    # ── 3. optimizer / scheduler ──────────────────────────────────────────
    opt = torch.optim.AdamW(
        [{"params": pg_qenc, "lr": args.lr1 * 0.2},   # adapter 用小一點 LR
         {"params": pg_xfus, "lr": args.lr1},
         {"params": pg_gate, "lr": args.lr1}],
        weight_decay=1e-2
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.ep1 * len(dl)
    )

    # ── 3-a)  印出本次真正可訓練參數量 ────────────────────────────────────
    def count(p_list): return sum(p.numel() for p in p_list)
    trainable = [p for p in txt_enc.parameters() if p.requires_grad] + \
                [p for p in vis_enc.parameters() if p.requires_grad]
    print(f"  Trainable parameters this run: {count(trainable):,}")
    print(f"   • q-enc  adapter : {count(pg_qenc):,}")
    print(f"   • Cross-Fusion   : {count(pg_xfus):,}")
    print(f"   • Gate-Fusion-MV : {count(pg_gate):,}")

    # ── 4. loss 權重 ─────────────────────────────────────────────────────
    λ_nll, λ_ctx, λ_view = 0.02, 0.05, 0.10

    # ── 5. 進入訓練迴圈 ──────────────────────────────────────────────────
    for ep in range(args.ep1):
        pbar = tqdm(dl, desc=f"Stage-1  Epoch {ep}", unit="batch")

        for caps, imgs, obj_ids, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)

            # 5-a) Text branch
            t_vec, ret_loss, tok_seq, _ = txt_enc(
                q_list=list(caps),
                obj_ids=list(obj_ids),
                return_loss=True
            )

            # 5-b) Vision branch
            v_vec, attn_maps, _ = vis_enc(imgs, t_vec)

            # 5-c) losses ------------------------------------------------
            loss_nce = info_nce(v_vec, t_vec, args.tau)            # global
            ctx_seq  = txt_enc.fusion.proj(tok_seq[:, 1:])         # (B,K,512)
            loss_mil = mil_nce(v_vec, ctx_seq, args.tau)           # MIL-NCE

            # gate-to-view entropy  (head-avg → (B,T,T))
            last = attn_maps['fuse'][-1].mean(1)                   # (B,T,T)
            gate2v = last[:, 1, 2:2+args.views]                    # (B,V)
            view_ent = -(gate2v * torch.log(gate2v + 1e-6)).sum(1)
            view_ent = view_ent.mean() / math.log(args.views)

            loss = (loss_nce +
                    λ_ctx  * loss_mil +
                    λ_view * view_ent +
                    λ_nll  * ret_loss)

            # 5-d) backward ---------------------------------------------
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step()

            pbar.set_postfix(NCE =f"{loss_nce:.3f}",
                             MIL =f"{loss_mil:.3f}",
                             NLL =f"{ret_loss:.3f}",
                             Ent =f"{view_ent:.3f}",
                             Σ   =f"{loss:.3f}")

            wb_log({"loss/NCE":       loss_nce.item(),
                    "loss/MIL":       loss_mil.item(),
                    "loss/NLL":       ret_loss.item(),
                    "loss/view_ent":  view_ent.item(),
                    "loss/total":     loss.item()},
                   step=ep * len(dl))

        # 5-e) checkpoint ----------------------------------------------
        os.makedirs(args.out, exist_ok=True)
        torch.save({"txt": txt_enc.state_dict(),
                    "vis": vis_enc.state_dict()},
                   f"{args.out}/enc1_ep{ep}.pth")

    txt_enc.eval(); vis_enc.eval()
    return txt_enc, vis_enc

@torch.no_grad()
def build_faiss_with_tok(ds, txt_enc, vis_enc, args, device):
    """
    回傳：
      - index: Faiss IndexFlatIP(512)，保存所有攤平成樣本的全局向量
      - all_tok: (N, T_vis, 512) numpy 陣列，保存每筆攤平成樣本的多視角 token 序列
    """
    idx_path = os.path.join(args.out, "faiss_index.bin")
    tok_path = os.path.join(args.out, "vis_tok.npy")

    if os.path.isfile(idx_path) and os.path.isfile(tok_path):
        print(f" 從 {args.out} 載入快取的 FAISS 索引與 token")
        index   = faiss.read_index(idx_path, faiss.IO_FLAG_MMAP)
        all_tok = np.load(tok_path)
        return index, all_tok

    print(" 找不到快取 → 正在建立 FAISS 索引與 token ...")
    g_vecs, g_toks = [], []
    loader = DataLoader(ds, args.bs, num_workers=4, shuffle=False)

    for caps, imgs, _, _ in tqdm(loader, desc="建立 FAISS", unit="batch"):
        imgs = imgs.to(device, non_blocking=True)
        # 1) 文字編碼器：
        q_vec, _, _, _ = txt_enc(list(caps), obj_ids=None)

        # 2) 多視角視覺編碼器：
        v_vec, _, v_tok = vis_enc(imgs, q_vec)   # v_vec: (B, 512), v_tok: (B, T_vis, 512)

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
    print(f" 已保存 faiss_index.bin & vis_tok.npy 至 {args.out}")
    return index, all_tok
# ---------------------------------------------------------------------------
# 階段 1 – 交錯凍結 / 解凍 訓練
# ---------------------------------------------------------------------------
def stage1_alt(args, ds, device, phase_len: int = 2):
    dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=4)

    txt_enc = RAGTextEncoder(args.data_jsonl, args.topk,
                             device=device, cache_dir=args.out).to(device)
    vis_enc = GFMVEncoder(args.views).to(device)

    def set_requires(vision_on: bool):
        for p in vis_enc.parameters():
            p.requires_grad = vision_on
        for p in txt_enc.retriever.qenc.parameters():
            p.requires_grad = not vision_on
        for p in txt_enc.fusion.parameters():
            p.requires_grad = not vision_on

    def build_opt(vision_on: bool):
        if vision_on:
            params = vis_enc.parameters()
            lr     = args.lr1
        else:
            params = list(txt_enc.retriever.qenc.parameters()) + \
                     list(txt_enc.fusion.parameters())
            lr     = args.lr1 * 0.5
        return torch.optim.AdamW(params, lr=lr)

    # 初始階段先訓練視覺
    set_requires(vision_on=True)
    opt = build_opt(vision_on=True)
    print("◎ 起手式：先訓練 Gate-Fusion 視覺路徑。")

    for ep in range(args.ep1):
        # 每到 phase_len ×2 就重設回視覺；phase_len ×1~2 為視覺，×3~4 為文字
        cycle_pos   = ep % (phase_len * 2)
        vision_on   = cycle_pos < phase_len
        if (cycle_pos == 0 and ep != 0) or (cycle_pos == phase_len):
            # 進入新子階段時切換 require_grad 並重建 optimizer
            set_requires(vision_on)
            opt = build_opt(vision_on)
            tag = "視覺" if vision_on else "文字"
            print(f"◎ epoch {ep} 切換至 {tag} 訓練階段。")

        pbar = tqdm(dl, desc=f"E{ep:02d} {'V' if vision_on else 'T'}", unit="batch")
        for step, (caps, imgs, obj_ids, _) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)

            t_vec, ret_loss, _, _ = txt_enc(
                q_list=list(caps),
                obj_ids=list(obj_ids),
                return_loss=True
            )
            v_vec, _, _ = vis_enc(imgs, t_vec)

            loss = info_nce(v_vec, t_vec, args.tau) + args.lmb * ret_loss
            opt.zero_grad(); loss.backward(); opt.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            wb_log({"stage1/loss": loss.item()}, step=ep * len(dl) + step)

        # 每輪都存 checkpoint 方便中斷續訓
        os.makedirs(args.out, exist_ok=True)
        torch.save({"txt": txt_enc.state_dict(),
                    "vis": vis_enc.state_dict()},
                   f"{args.out}/enc1_ep{ep}.pth")

    txt_enc.eval(); vis_enc.eval()
    return txt_enc, vis_enc

def stage2(args, ds, txt_enc, vis_enc, index, all_tok, device):
    """
    階段2：粗排(FAISS top-L)→精排（reranker）微調
    """
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

            # ——— 2) 挑選正例與硬負例，並計算 reranker 分數 ———
            s_pos_list, s_neg_list = [], []
            for b in range(B):
                gt_flat_idx = idxs[b].item()
                all_pos_for_this_obj = ds.obj2idx[obj_ids[b]]

                cand = idx_faiss[b].tolist()
                if gt_flat_idx in cand:
                    cand.remove(gt_flat_idx)
                    cand = [gt_flat_idx] + cand
                else:
                    cand = [gt_flat_idx] + cand[:-1]

                neg_cands = [x for x in cand[1:] if x not in all_pos_for_this_obj]
                hard_negs = neg_cands[:num_hard_neg]

                t_tok_b = txt_tok[b : b+1]
                t_tok_expand = t_tok_b.expand(num_hard_neg, -1, -1)

                pos_vtok = torch.from_numpy(all_tok[gt_flat_idx]).float().to(device)
                neg_vtok = torch.from_numpy(all_tok[hard_negs]).float().to(device)
                pos_vtok_expand = pos_vtok.unsqueeze(0).expand(num_hard_neg, -1, -1)

                x_pos = reranker(t_tok_expand, pos_vtok_expand)
                x_neg = reranker(t_tok_expand, neg_vtok)

                s_pos_list.append(x_pos)
                s_neg_list.append(x_neg)

            # ——— 3) 組裝 margin ranking loss ———
            pos_tensor = torch.stack([x[0] for x in s_pos_list])
            neg_tensor = torch.stack(s_neg_list)

            target = torch.ones_like(neg_tensor)
            loss = criterion(pos_tensor.unsqueeze(1), neg_tensor, target)

            # ——— 4) 反向傳播與優化更新 ———
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(epoch=ep, loss=loss.item())
            wb_log({"stage2/loss": loss.item()},
                   step=(args.ep1 + ep) * len(loader) + step)

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
    p.add_argument("--lmb", type=float, default=0.02)
    p.add_argument("--lr1", type=float, default=1e-4)
    p.add_argument("--ep1", type=int, default=8)
    p.add_argument("--L", type=int, default=100, help="# reranker候選數")
    p.add_argument("--lr2", type=float, default=3e-4)
    p.add_argument("--ep2", type=int, default=5)
    p.add_argument("--wandb", action="store_true", help="啟用 wandb 紀錄")
    p.add_argument("--resume_enc1", type=str, default=None,
               help="階段1編碼器權重路徑（跳過訓練）")
    # 新增設備參數
    p.add_argument("--device", type=str, default="auto", 
                   help="設備選擇: 'auto', 'cuda', 'mps', 'cpu' 或具體設備如 'cuda:0'")
    return p.parse_args()

if __name__ == "__main__":
    args = parse()
    
    # 設備配置
    device = get_device(args.device)
    setup_device_optimization(device)
    device = str(device)  # 轉為字串以便後續使用
    
    wandb_init(args)
    # ds = UnifiedDataset(args.data_jsonl, num_views=args.views)
    from torch.utils.data import Subset

    full_ds = UnifiedDataset(args.data_jsonl, num_views=args.views)
    # ds = full_ds
    print(len(full_ds))
    # 只取前 1000 筆
    ds = Subset(full_ds, list(range(1000)))

    print(len(ds))
    
    if args.resume_enc1:  # 載入階段1模型
        print(f"從 {args.resume_enc1} 載入階段1權重 …")
        state = torch.load(args.resume_enc1, map_location=device)
        txt_enc = RAGTextEncoder(args.data_jsonl, args.topk, device=device).to(device)
        vis_enc = GFMVEncoder(args.views).to(device)
        txt_enc.load_state_dict(state["txt"])
        vis_enc.load_state_dict(state["vis"])
        txt_enc.eval(); vis_enc.eval()
    else:
        txt_enc, vis_enc = stage1(args, ds, device)

    index, all_tok = build_faiss_with_tok(ds, txt_enc, vis_enc, args, device)
    stage2(args, ds, txt_enc, vis_enc, index, all_tok, device)
    print("訓練完成 →", args.out)