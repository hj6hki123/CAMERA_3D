import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig

from adapters import BertAdapterModel
# from models.external_retriever import ExternalRetriever

class FusionBlock(nn.Module):
    def __init__(self, d=768, h=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=True)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_weights

class CrossFusion(nn.Module):
    def __init__(self, L=2, d=768, h=8):
        super().__init__()
        self.blocks = nn.ModuleList([FusionBlock(d, h) for _ in range(L)])
        self.proj   = nn.Linear(d, 512)

    def forward(self, seq):
        all_attn_maps = []
        for blk in self.blocks:
            seq, attn_w = blk(seq)
            all_attn_maps.append(attn_w)
        cls_vec = self.proj(seq[:, 0])
        return cls_vec, all_attn_maps

class RAGTextEncoder(nn.Module):
    def __init__(self, top_k=3, device="cuda"):
        super().__init__()
        self.top_k  = top_k
        self.device = torch.device(device)

        # 載入 BERT-Adapter
        self.text_encoder = BertAdapterModel.from_pretrained("bert-base-uncased")
        try:
            self.text_encoder.add_adapter("fusion_adapter")
        except ValueError as e:
            if "already been added" not in str(e):
                raise
        self.text_encoder.train_adapter("fusion_adapter")
        self.text_encoder.to(self.device)

        # 預設以 config.hidden_size 為初始值，之後若偵測不符會動態重建
        self.hidden_dim = self.text_encoder.config.hidden_size
        self.fusion     = CrossFusion(d=self.hidden_dim).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def forward(self, q_list: list[str], ctx_list: list):
        B = len(q_list)
        flat_texts = []
        for q, ctxts in zip(q_list, ctx_list):
            c = list(ctxts) if hasattr(ctxts, "__iter__") else []
            if len(c) < self.top_k:
                c.extend([""] * (self.top_k - len(c)))
            flat_texts.extend([q] + c[: self.top_k])

        if not flat_texts:
            z = torch.zeros(B, 512, device=self.device)
            seq = torch.zeros(B, 1 + self.top_k, self.hidden_dim, device=self.device)
            return z, torch.tensor(0.0, device=self.device), seq, []

        tok_inputs = self.tokenizer(
            flat_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=256
        ).to(self.device)

        # 啟動 adapter、編碼
        prev_adpt = self.text_encoder.active_adapters
        try:
            self.text_encoder.set_active_adapters("fusion_adapter")
            base_out = self.text_encoder.base_model(
                **tok_inputs, return_dict=True
            )
        finally:
            self.text_encoder.set_active_adapters(prev_adpt)

        # 取 [CLS] 向量
        cls_vecs = base_out.last_hidden_state[:, 0].contiguous()
        cur_dim  = cls_vecs.size(-1)

        # 若 hidden_dim 變了，立即重建 Cross-Fusion
        if cur_dim != self.hidden_dim:
            print(f"[CrossFusion] 重建：{self.hidden_dim} → {cur_dim}")
            self.hidden_dim = cur_dim
            self.fusion     = CrossFusion(d=self.hidden_dim).to(self.device)

        tok_seq = cls_vecs.reshape(B, 1 + self.top_k, self.hidden_dim)
        vec, attn_maps = self.fusion(tok_seq)

        return vec, torch.tensor(0.0, device=self.device), tok_seq, attn_maps

# # ▼▼▼ 核心修改：改造 RAGTextEncoder ▼▼▼
# class RAGTextEncoder(nn.Module):
#     def __init__(self, top_k=3, device="cuda"):
#         """
#         初始化 RAG 文字編碼器。
#         這個版本使用 ExternalRetriever 從外部知識庫（維基百科）檢索資訊。
#         """
#         super().__init__()
#         self.top_k = top_k
#         self.device = torch.device(device)

#         # 1. 初始化外部檢索器
#         #    注意：請將 User-Agent 字串換成您自己的應用名稱和聯繫方式
#         # <-- 修改點：不再使用 DenseRetriever
#         self.retriever = ExternalRetriever()

#         # 2. 初始化用於文本編碼的 BERT 模型和 Tokenizer
#         #    這個 BERT 用來將 query 和檢索到的 contexts 轉換成 CrossFusion 能接受的向量序列
#         # <-- 新增：獨立的文本編碼器
#         self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#         self.text_encoder = AutoModel.from_pretrained("bert-base-uncased").to(self.device)

#         # 3. 複用您原有的 CrossFusion 模組
#         self.fusion = CrossFusion().to(self.device)

#     def forward(self, q_list: list[str]):
#         """
#         新的前向傳播流程，從即時檢索外部語義開始。
#         """
#         B = len(q_list)
        
#         # <-- 核心流程重構 -->
#         # 步驟 1: 為批次中的每個查詢句，從外部進行即時檢索
#         all_contexts = [self.retriever.retrieve(q, top_k=self.top_k) for q in q_list]

#         # 步驟 2: 準備送入 BERT 編碼器的序列
#         flat_texts = []
#         for i in range(B):
#             query = q_list[i]
#             contexts = all_contexts[i]
            
#             # 為了保持序列長度一致，如果檢索到的 context 不足 top_k，用空字串填充
#             if len(contexts) < self.top_k:
#                 contexts.extend([''] * (self.top_k - len(contexts)))

#             flat_texts.extend([query] + contexts[:self.top_k])

#         # 步驟 3: 使用 BERT 進行編碼
#         tok_inputs = self.tokenizer(
#             flat_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
#         ).to(self.device)
        
#         encoded_vecs = self.text_encoder(**tok_inputs).last_hidden_state[:, 0]
        
#         tok_seq = encoded_vecs.view(B, 1 + self.top_k, -1)

#         # 步驟 4: 使用 CrossFusion 進行語意融合
#         vec, fusion_attn_maps = self.fusion(tok_seq)

#         # <-- 返回值改變：不再有 ret_loss，因為檢索器不可訓練 -->
#         return vec, torch.tensor(0., device=vec.device), tok_seq, fusion_attn_maps