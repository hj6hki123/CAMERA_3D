# models/dense_retriever.py
# -*- coding: utf-8 -*-
"""
DenseRetriever:DPR 風格可微檢索器
 • GPU/CPU 初始化 + tqdm
 • memory_vec.pt 離線快取
"""
import json, os, torch, torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from collections import Counter
import re


class BigruEncoder(torch.nn.Module):
    def __init__(self,vocab_size, embedffing_dim=300 , hidden_size=384, num_layers=2, dropout=0.2):
        super.__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedffing_dim, padding_idx=0) ## 文字嵌入向量
        self.bigru = torch.nn.GRU(embedffing_dim,  
                                hidden_size,
                                num_layers=num_layers,
                                batch_first=True,
                                bidirectional=True,
                                dropout=dropout if num_layers > 1 else 0)## 雙向GRU
        self.dropout = torch.nn.Dropout(dropout) ## dropout
        self.output_proj = torch.nn.Linear(hidden_size * 2 , 768) ## output projection (hidden_size * 2 為雙向GRU的輸出維度)

                                
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # (B , L , embedding_dim)
        x = self.dropout(x)  
        
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output , hidden = self.bigru(x)  # (num_layers*2, B, hidden_dim)
        
        forward_hidden = hidden[-2] ## 最後一層前向
        backward_hidden = hidden[-1] ## 最後一層後向
        
        ## 合併
        combine = torch.cat((forward_hidden, backward_hidden), dim=1)  # (B, hidden_size * 2)
        output = self.output_proj(combine)  # (B, 768)
        
        class OutPut: ## 用來模擬 bert 輸出
            def __init__(self, output):
                self.last_hidden_state = output.unsqueeze(1)  # (B, 1, 768)
        
        return OutPut(output)  

class DenseRetriever(torch.nn.Module):
    def __init__(self,
                 corpus_jsonl: str,
                 dim:    int   = 768,
                 device: str   = "cpu",   # "cpu" | "cuda" | "cuda:1"
                 batch:  int   = 32,
                 cache:  bool  = True):
        super().__init__()
        self.device = torch.device(device)

        # ------- BERT encoder & tokenizer -----------------------------------
        self.tok  = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.qenc = AutoModel.from_pretrained("bert-base-uncased").to(self.device)
        self.kenc = AutoModel.from_pretrained("bert-base-uncased").to(self.device)

        # ------- 讀取 corpus -------------------------------------------------
        corpus      = [json.loads(l) for l in open(corpus_jsonl) if l.strip()]
        self.texts  = [c["text"] for c in corpus]
        self.obj_ids= [c.get("obj_id") or c.get("id") for c in corpus]
        
        cache_path = corpus_jsonl + ".pt"
        if cache and os.path.exists(cache_path):
            vecs = torch.load(cache_path, map_location="cpu")
            if vecs.size(0) == len(self.texts):
                print(f"Loaded cached memory_vec ({cache_path})")
                self.memory_vec = torch.nn.Parameter(vecs)
                return
            else:
                print("! cache size mismatch,重算向量…")

        # ------- 首次批次編碼 (CPU or GPU) ----------------------------------
        vecs = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            for i in tqdm(range(0, len(self.texts), batch),
                          desc="Init Retriever", unit="batch"):
                enc = self.tok(self.texts[i:i+batch],
                               return_tensors="pt",
                               padding=True, truncation=True).to(self.device)
                v = self.kenc(**enc).last_hidden_state[:, 0]   # (b,768)
                vecs.append(F.normalize(v, 2, -1).cpu())
        vecs = torch.cat(vecs)                               # (N,768)
        if cache:
            torch.save(vecs, cache_path)
            print(f"Saved memory_vec cache → {cache_path}")
        self.memory_vec = torch.nn.Parameter(vecs)

    # --------------------------------------------------------------------- #
    #                          推論期  API                                   #
    # --------------------------------------------------------------------- #
    def query_encode(self, q_list):                          # (B,768)
        enc = self.tok(q_list, return_tensors="pt",
                       padding=True, truncation=True).to(self.qenc.device)
        return self.qenc(**enc).last_hidden_state[:, 0]

    def pos_index_from_obj(self, obj_ids):                   # list[str]
        mapping = {o: i for i, o in enumerate(self.obj_ids)}
        return torch.tensor([mapping[o] for o in obj_ids],
                            device=self.memory_vec.device)

    def forward(self, q_list, obj_ids=None, topk=4):
        """
        q_list : List[str]   查詢句
        obj_ids: List[str]   同 batch 真實 obj_id,用來計算 NLL
        topk   : int         返回前 k 個語境
        -------------------------------------------------------------
        return:
          q_vec   (B,768)
          sims    (B,N)      內積相似度 (可用於 loss)
          idx     (B,topk)   top-k 索引
          ctx     List[List[str]]  檢索到的 caption
          loss    ()         retriever NLL;若 obj_ids=None 為 0
        """
        q_vec = self.query_encode(q_list)                    # (B,768)
        sims  = q_vec @ F.normalize(self.memory_vec, 2, -1).T
        idx   = sims.topk(topk, -1).indices                  # (B,topk)
        ctx   = [[self.texts[j] for j in row] for row in idx.cpu()]

        if obj_ids is not None:
            loss = F.cross_entropy(sims, self.pos_index_from_obj(obj_ids))
            return q_vec, sims, idx, ctx, loss
        return q_vec, sims, idx, ctx, torch.tensor(0., device=q_vec.device)


# 舊介面兼容（外部若直接調用 retriever_nll）
def retriever_nll(logits: torch.Tensor, pos_idx: torch.Tensor):
    return torch.nn.functional.cross_entropy(logits, pos_idx)
