"""
線上可微 SemanticMemory
‧ _vec(txt) 同步用 BERT 取 CLS
‧ retrieve(query) 回傳 [ctx_texts] , [sim_scores Tensor]
"""

import json, torch, faiss
from transformers import AutoTokenizer, AutoModel

class SemanticMemory:
    def __init__(self, idx="sem_mem.index", meta="sem_mem.meta",
                 model="bert-base-uncased", top_k=4):
        self.index = faiss.read_index(idx)
        self.meta  = json.load(open(meta))
        self.top_k = top_k
        self.tok   = AutoTokenizer.from_pretrained(model)
        self.enc   = AutoModel.from_pretrained(model).cuda()

    @torch.no_grad()
    def _vec(self, txt:str):
        t = self.tok(txt, return_tensors="pt",
                     truncation=True,padding=True).to("cuda")
        v = self.enc(**t).last_hidden_state[:,0]
        v = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
        return v

    def retrieve(self, query:str):
        D,I = self.index.search(self._vec(query), self.top_k)
        return [self.meta[i]["text"] for i in I[0]], torch.tensor(D[0])
