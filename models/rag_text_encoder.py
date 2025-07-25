import json, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from models.dense_retriever import DenseRetriever
from pathlib import Path

class FusionBlock(nn.Module):
    def __init__(self, d=768, h=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        # 啟用 need_weights=True
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff   = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))

    def forward(self, x):
        # ln + self-attn，取回 attn weights
        attn_out, attn_weights = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=True,average_attn_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_weights  # attn_weights shape = (B, heads, T, T)

class CrossFusion(nn.Module):
    def __init__(self, L=3, d=768, h=8):
        super().__init__()
        self.blocks = nn.ModuleList([FusionBlock(d, h) for _ in range(L)])
        self.proj   = nn.Linear(d, 512)

    def forward(self, seq):
        all_attn_maps = []  # collect each layer's attn weights
        for blk in self.blocks:
            seq, attn_w = blk(seq)
            all_attn_maps.append(attn_w)  # shape = (B, heads, T, T)
        cls_vec = self.proj(seq[:, 0])   # use CLS token 作為輸出
        return cls_vec, all_attn_maps

class RAGTextEncoder(nn.Module):
    def __init__(self, unified_jsonl, top_k=6, device="cuda", cache_dir: str = None):
        super().__init__()
        self.top_k = top_k

        # 從 unified_jsonl 抽出 corpus → 存到 temp_corpus.jsonl
        corpus = []
        for line in open(unified_jsonl, encoding="utf-8"):
            item = json.loads(line)
            oid  = item["obj_id"]
            for t in item.get("corpus_texts", []):
                corpus.append({"text": t, "obj_id": oid})
                
        if cache_dir:
            base = Path(cache_dir)
        else:
            base = Path(unified_jsonl).parent
        base.mkdir(parents=True, exist_ok=True)
        temp_path = base / "temp_corpus.jsonl"
        
        if not temp_path.exists():
            with open(temp_path, "w") as f:
                for c in corpus:
                    f.write(json.dumps(c) + "\n")

        # DenseRetriever - 使用傳入的設備參數
        self.retriever = DenseRetriever(str(temp_path), device=device, batch=24)
        self.fusion    = CrossFusion()

    def forward(self, q_list, obj_ids=None, return_loss=False):
        """
        回傳：
          q_vec: text embedding (B,512)
          ret_loss: retriever 的 NLL loss
          tok: token sequence for reranker ((B, 1+top_k, 768))
          fusion_attn_maps: list of length L，每層 shape=(B, heads, T, T)
        """
        # 1) 用 DenseRetriever 拿 q_vec, top-k context, retriever loss
        q_vec, _, _, ctx, ret_loss = self.retriever(q_list, obj_ids, self.top_k)

        B = len(q_list)
        # 2) 把 query + top-k context 串成一個長序列，一起做 cross-fusion
        flat = [t for i in range(B) for t in ([q_list[i]] + ctx[i])]
        tok_inputs = self.retriever.tok(
            flat, return_tensors="pt", padding=True, truncation=True
        ).to(q_vec.device)

        #  切到 retr_adapter
        prev = self.retriever.qenc.active_adapters
        self.retriever.qenc.set_active_adapters("retr_adapter")

        #  只跑 backbone → 不會回傳 MaskedLMOutput
        out = self.retriever.qenc.base_model(**tok_inputs, return_dict=True)
        tok_encoded = out.last_hidden_state[:, 0]          # (B*(1+K),768)

        #  還原先前 adapter 狀態
        self.retriever.qenc.set_active_adapters(prev)

        tok_encoded = tok_encoded.view(B, 1 + self.top_k, 768)

        # 3) CrossFusion
        vec, fusion_attn_maps = self.fusion(tok_encoded)  # vec: (B,512)

        if return_loss:
            return vec, ret_loss, tok_encoded, fusion_attn_maps
        else:
            return vec, torch.tensor(0., device=vec.device), tok_encoded, fusion_attn_maps