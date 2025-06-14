# test_stage1_analysis.py

import os
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import ndcg_score, average_precision_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# 請自行調整以下 import path
from datasets.unified_dataset import UnifiedDataset
from models.rag_text_encoder import RAGTextEncoder
from models.gf_mv_encoder import GFMVEncoder

sns.set(style="whitegrid", font="Arial")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_jsonl", required=True,
                   help="unified_data.jsonl 路徑")
    p.add_argument("--cache_dir", default="ckpts",
                   help="Stage1 權重、cache 存放資料夾")
    p.add_argument("--val_size", type=int, default=1000,
                   help="驗證集樣本數 (從頭取)")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--topk", type=int, default=5,
                   help="Retriever top-k")
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_models(args, device):
    # 載入 Stage1 訓練好的權重
    cmap = torch.load(os.path.join(args.cache_dir, "enc1.pth"),
                      map_location=device)
    txt_enc = RAGTextEncoder(
        unified_jsonl=args.data_jsonl,
        top_k=args.topk,
        device=args.device,
        cache_dir=args.cache_dir
    ).to(device)
    txt_enc.load_state_dict(cmap["txt"])
    txt_enc.eval()

    vis_enc = GFMVEncoder(num_views=12).to(device)
    vis_enc.load_state_dict(cmap["vis"])
    vis_enc.eval()
    return txt_enc, vis_enc


def compute_retrieval_metrics(retriever, q_list, obj_ids, topk):
    """
    用 DenseRetriever 直接算 Recall@k、NDCG@k、mAP
    """
    # sims: (B, N_text) ; idx: (B,topk)
    q_vec = retriever.query_encode(q_list)
    sims = q_vec @ F.normalize(retriever.memory_vec, 2, -1).T
    idx = sims.topk(topk, dim=-1).indices.cpu().numpy()
    # 相關度 binary relevance list
    y_true = []
    y_score = []
    recall_counts = np.zeros(len(q_list))
    ndcgs = []
    aps = []
    for i, oid in enumerate(obj_ids):
        # 找出 corpus 中所有屬於這個 oid 的 text index
        pos_idxs = np.where(np.array(retriever.obj_ids) == oid)[0]
        # relevance vector
        rel = np.zeros(sims.size(1), dtype=int)
        rel[pos_idxs] = 1
        # ranked scores & relevance
        scores = sims[i].cpu().numpy()
        # Recall@k
        retrieved = idx[i]
        recall_counts[i] = int(np.any(rel[retrieved] == 1))
        # NDCG@k
        ndcgs.append(
            ndcg_score([rel], [scores], k=topk)
        )
        # AP
        aps.append(
            average_precision_score(rel, scores)
        )
    recall_at_k = recall_counts.mean()
    mean_ndcg = np.mean(ndcgs)
    mean_ap = np.mean(aps)
    return recall_at_k, mean_ndcg, mean_ap


def gather_attention_stats(txt_enc, loader, device):
    """
    收集所有樣本的 fusion_attn_maps（list of layers）
    並計算 CLS→各 context 平均 attention
    回傳 DataFrame: cols = [sample, layer, head, ctx_idx, attn_value, is_gt]
    """
    records = []
    for caps, imgs, obj_ids, idxs in tqdm(loader, desc="收集 attention"):
        # 只要 text fusion
        vec, ret_loss, tok_seq, fusion_attn_maps = txt_enc(
            q_list=list(caps),
            obj_ids=list(obj_ids),
            return_loss=True
        )
        B = len(caps)
        T = tok_seq.size(1)  # =1+topk
        for l, attn in enumerate(fusion_attn_maps):
            # attn: list of length layers; each (B, heads, T, T)
            arr = attn.detach().cpu().numpy()
            for b in range(B):
                for h in range(arr.shape[1]):
                    # CLS token index = 0
                    cls2ctx = arr[b, h, 0, :]  # shape=(T,)
                    for c in range(T):
                        records.append({
                            "sample": idxs[b].item(),
                            "layer": l,
                            "head": h,
                            "ctx_idx": c,
                            "attn": cls2ctx[c],
                            "is_gt": (c == 0)  # 因為 RAGTextEncoder tok_seq 排序: [CLS, q, ctx1...], 0 對應 CLS, 1 是 query token, 後面才是 contexts
                        })
    df = pd.DataFrame(records)
    # 標記 contexts 真偽: 以 c >=2 當候選 context，並對照 obj_ids 判斷 is_gt
    # 這裡假設 idxs 方式與 RAGTextEncoder forward 一致
    return df


def plot_attention_distribution(df, out_dir):
    """
    畫出各 layer/head CLS→context 的盒鬚圖
    """
    os.makedirs(out_dir, exist_ok=True)
    # 只畫 context 部分（ctx_idx>=2）
    df_ctx = df[df["ctx_idx"] >= 2]
    g = sns.catplot(
        data=df_ctx,
        x="layer", y="attn",
        col="head",
        kind="box",
        col_wrap=4,
        sharey=False
    )
    g.fig.suptitle("各 layer/head CLS→contexts Attention 分佈", y=1.02)
    plt.tight_layout()
    g.savefig(f"{out_dir}/attention_boxplot.png")
    plt.close()


def ablation_layer(txt_enc, val_loader, vis_enc, args):
    """
    Layer Ablation：依序把某一層 fusion.blocks 換成 Identity
    計算 InfoNCE loss 平均值，觀察訓練對比效果變化
    """
    results = []
    base_blocks = txt_enc.fusion.blocks
    for abl in range(len(base_blocks)):
        # 深複製模型
        m = torch.clone(txt_enc.state_dict())
        enc = RAGTextEncoder(
            unified_jsonl=args.data_jsonl,
            top_k=args.topk,
            device=args.device,
            cache_dir=args.cache_dir
        ).to(args.device)
        enc.load_state_dict(m)
        # 把第 abl 層換成 identity
        enc.fusion.blocks[abl] = torch.nn.Identity()
        enc.eval()

        # 計算整個驗證集的 InfoNCE loss
        tot_loss = 0.0
        cnt = 0
        for caps, imgs, obj_ids, _ in val_loader:
            t_vec, ret_loss, _, _ = enc(
                q_list=list(caps), obj_ids=list(obj_ids), return_loss=True
            )
            v_vec, _, _ = vis_enc(imgs.to(args.device), t_vec)
            loss = F.cross_entropy(
                F.normalize(v_vec,2,1) @ F.normalize(t_vec,2,1).T / args.tau,
                torch.arange(t_vec.size(0), device=t_vec.device)
            ) + args.tau * ret_loss  # 同 Stage1 結合 loss
            tot_loss += loss.item() * t_vec.size(0)
            cnt += t_vec.size(0)
        results.append({"abl_layer": abl,
                        "info_nce_loss": tot_loss / cnt})
    df = pd.DataFrame(results)
    df.to_csv(f"{args.cache_dir}/ablation_layer.csv", index=False)
    return df


def ablation_head_stub():
    """
    Head Ablation 需要在 MultiheadAttention 層面做細節修改，
    當前僅列出步驟，實作請參考 PyTorch multi_head_attention_forward：
    1. 在每個 FusionBlock.attn 中，拆解各頭輸出
    2. 對指定 head 置零後再 concat & out_proj
    3. 重跑 InfoNCE loss
    """
    pass


def qualitative_heatmap(txt_enc, vis_enc, dataset, args, num_examples=5):
    """
    隨機挑幾組正確 vs 錯誤檢索案例，畫 CLS→contexts attention heatmap
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    os.makedirs(f"{args.cache_dir}/heatmaps", exist_ok=True)
    used = 0
    for caps, imgs, obj_ids, idxs in loader:
        vec, ret_loss, tok_seq, fusion_attn_maps = txt_enc(
            q_list=list(caps), obj_ids=list(obj_ids), return_loss=True
        )
        # 取第一層第一個頭的 attention
        attn = fusion_attn_maps[0][0,0]  # (T,T)
        # CLS→contexts row
        cls2ctx = attn[0]  # shape=(T,)
        labels = ["CLS"] + ["q"] + [f"ctx{i}" for i in range(1, cls2ctx.shape[0]-1)]
        plt.figure(figsize=(6,4))
        sns.heatmap(cls2ctx[None,:], annot=True, cmap="Reds",
                    xticklabels=labels, yticklabels=[])
        plt.title(f"Example {idxs.item()} Layer0 Head0 CLS→contexts")
        plt.savefig(f"{args.cache_dir}/heatmaps/heat_{idxs.item()}.png")
        plt.close()
        used += 1
        if used >= num_examples: break


def embedding_tsne(txt_enc, dataset, args, num_samples=500):
    """
    抽樣部分樣本，取其 CLS 向量 + first view global vec，
    用 t-SNE 投影並用 obj_ids 分群
    """
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=True)
    all_feats, all_labels = [], []
    cnt = 0
    for caps, imgs, obj_ids, _ in loader:
        t_vec, _, _, _ = txt_enc(
            q_list=list(caps), obj_ids=list(obj_ids), return_loss=False
        )
        v_vec, _, _ = vis_enc(imgs.to(args.device), t_vec)
        for i, oid in enumerate(obj_ids):
            all_feats.append(torch.cat([t_vec[i], v_vec[i]]).cpu().numpy())
            all_labels.append(oid)
        cnt += len(obj_ids)
        if cnt >= num_samples: break

    feats = np.stack(all_feats)  # (M, 1024)
    tsne = TSNE(n_components=2, random_state=42)
    X2 = tsne.fit_transform(feats)
    df = pd.DataFrame({
        "x": X2[:,0], "y": X2[:,1], "obj_id": all_labels
    })
    plt.figure(figsize=(6,6))
    sns.scatterplot(data=df, x="x", y="y", hue="obj_id", legend=False, s=10)
    plt.title("t-SNE of CLS(text)||Global(vis) embeddings")
    plt.savefig(f"{args.cache_dir}/tsne.png")
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    device = torch.device(args.device)

    # 1. 載入資料
    full_ds = UnifiedDataset(args.data_jsonl, num_views=12)
    val_ds  = Subset(full_ds, list(range(args.val_size)))
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=4)

    # 2. 載入模型
    txt_enc, vis_enc = load_models(args, device)

    # 3. 基線檢索性能
    recall, ndcg, mAP = compute_retrieval_metrics(
        txt_enc.retriever,
        [caps for caps,_,_,_ in val_loader.dataset],
        [oid for _,_,oid,_ in val_loader.dataset],
        args.topk
    )
    print(f"Baseline Recall@{args.topk}: {recall:.4f}, NDCG@{args.topk}: {ndcg:.4f}, mAP: {mAP:.4f}")

    # 4. 注意力統計
    df_attn = gather_attention_stats(txt_enc, val_loader, device)
    plot_attention_distribution(df_attn, args.cache_dir)

    # 5. 消融實驗
    df_abl = ablation_layer(txt_enc, val_loader, vis_enc, args)
    print("Layer ablation results:\n", df_abl)

    # 6. 定性案例
    qualitative_heatmap(txt_enc, vis_enc, val_ds, args)

    # 7. Embedding 投影
    embedding_tsne(txt_enc, val_ds, args)

    print("所有分析完成，結果請見", args.cache_dir)


if __name__ == "__main__":
    main()
