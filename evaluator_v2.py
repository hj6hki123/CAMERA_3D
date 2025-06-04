#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate CAMERA_3D (baseline  + optional rerank)

範例：
  python evaluator_v2.py \
      --data datasets/unified_data.jsonl \
      --ckpt ckpts_fix \
      --bs   32 \
      --views 12 \
      --topk  4
--out 可省略；若未給自動設為 --ckpt。
"""
import os, time, argparse, psutil, torch, faiss
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets.unified_dataset      import UnifiedDataset
from models.rag_text_encoder       import RAGTextEncoder
from models.gf_mv_encoder          import GFMVEncoder
from models.cross_modal_reranker   import CrossModalReranker
from train_two_stage               import build_faiss_with_tok

# ---------------- helper metrics ----------------
def dcg(rel):
    rel = np.asarray(rel)
    log = np.log2(np.arange(2, rel.size + 2))
    return (rel / log).sum()

def ndcg(rel, k):
    idcg = dcg(sorted(rel, reverse=True)[:k])
    return dcg(rel[:k]) / idcg if idcg else 0.

def apk(rel, k):
    rel = np.asarray(rel[:k])
    if rel.sum() == 0: return 0.
    prec = rel.cumsum() / (np.arange(rel.size) + 1)
    return (prec * rel).sum() / rel.sum()

# ---------------- evaluator core ----------------
def evaluate(args):
    ds  = UnifiedDataset(args.data, num_views=args.views)
    ks  = (1, 5, 10)
    dl  = DataLoader(ds, args.bs, num_workers=4)

    # ---- Load encoders ----------------------------------------------------
    ck  = torch.load(os.path.join(args.ckpt, "enc1.pth"), map_location="cuda")
    txt = RAGTextEncoder(args.data, top_k=args.topk).cuda().eval()
    txt.load_state_dict(ck["txt"])
    vis = GFMVEncoder(args.views).cuda().eval()
    vis.load_state_dict(ck["vis"])

    # ---- Build / Load FAISS & tokens --------------------------------------
    L = max(ks[-1], 50)
    print(f"[INFO] Build / load FAISS index (top-L={L}) from {args.out}")
    index, all_tok = build_faiss_with_tok(ds, txt, vis, args)   # 會自動讀取 cache

    # ---- Reranker (若存在) -------------------------------------------------
    rerank_path = os.path.join(args.ckpt, "rerank.pth")
    use_rerank  = os.path.exists(rerank_path)
    if use_rerank:
        print("[INFO] Rerank enabled →", rerank_path)
        reranker = CrossModalReranker().cuda().eval()
        reranker.load_state_dict(torch.load(rerank_path))
    else:
        print("[INFO] 只計算 baseline（FAISS）")

    # ---- Metric accumulators ----------------------------------------------
    hit_b = {k:0 for k in ks};  APb=[]; NDb=[]
    hit_r = {k:0 for k in ks};  APr=[]; NDr=[]
    tot, gpu_mem, cpu_mem = 0, 0, 0
    t0 = time.time()

    with torch.no_grad():
        for q, _, obj_ids in tqdm(dl, desc="Eval"):
            gpu0 = torch.cuda.memory_allocated()/1e6
            cpu0 = psutil.Process(os.getpid()).memory_info().rss/1e6

            q_vec, _, txt_tok = txt(list(q), list(obj_ids))
            _, idx = index.search(q_vec.cpu().numpy(), L)

            gpu_mem += torch.cuda.memory_allocated()/1e6 - gpu0
            cpu_mem += psutil.Process(os.getpid()).memory_info().rss/1e6 - cpu0

            for b, oid in enumerate(obj_ids):
                tot += 1

                # ---------- baseline ----------
                base_ids = [ds.items[j]["obj_id"] for j in idx[b][:ks[-1]]]
                rel = [1 if oid == rid else 0 for rid in base_ids]
                for k in ks: hit_b[k] += int(any(rel[:k]))
                APb.append(apk(rel,10));  NDb.append(ndcg(rel,10))

                # ---------- rerank ------------
                if use_rerank:
                    txt_t = txt_tok[b:b+1].expand(L,-1,-1)
                    vis_t = torch.from_numpy(all_tok[idx[b]]).float().cuda()
                    scores = reranker(txt_t, vis_t)
                    order  = torch.argsort(scores, descending=True).cpu().numpy()
                    rer_ids= [ds.items[idx[b][j]]["obj_id"] for j in order[:ks[-1]]]
                    rel_r  = [1 if oid == rid else 0 for rid in rer_ids]
                    for k in ks: hit_r[k] += int(any(rel_r[:k]))
                    APr.append(apk(rel_r,10)); NDr.append(ndcg(rel_r,10))

    dur = time.time() - t0

    # ---------------- Report ----------------
    print("\n===========  Evaluation  ===========")
    print(":: Baseline (FAISS) ::")
    for k in ks: print(f"Recall@{k:<2}: {hit_b[k]/tot:6.3%}")
    print(f"mAP@10  : {np.mean(APb):6.3%}")
    print(f"NDCG@10 : {np.mean(NDb):6.3%}")
    print(f"QPS     : {tot/dur:6.1f}  query/s")
    print(f"Avg GPU : {gpu_mem/tot:6.1f} MB | Avg CPU : {cpu_mem/tot:6.1f} MB")

    if use_rerank:
        print("\n:: Rerank (Cross-Modal) ::")
        for k in ks: print(f"Recall@{k:<2}: {hit_r[k]/tot:6.3%}")
        print(f"mAP@10  : {np.mean(APr):6.3%}")
        print(f"NDCG@10 : {np.mean(NDr):6.3%}")
    print("====================================\n")

# ---------------- CLI ----------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--data",  required=True, help="unified_data.jsonl")
    pa.add_argument("--ckpt",  required=True, help="checkpoint directory")
    pa.add_argument("--bs",    type=int, default=32)
    pa.add_argument("--views", type=int, default=12)
    pa.add_argument("--topk",  type=int, default=4)
    pa.add_argument("--out",   default=None, help="dir for faiss_index.bin / vis_tok.npy")
    args = pa.parse_args()
    if args.out is None:
        args.out = args.ckpt        # 預設與 ckpt 同資料夾
    evaluate(args)
