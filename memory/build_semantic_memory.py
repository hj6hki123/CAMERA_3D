# build_semantic_memory.py
import json, torch, faiss
from transformers import AutoTokenizer, AutoModel

# 讀剛才生成的 corpus
corpus = json.load(open("semantic_corpus.json"))
texts  = [c["text"] for c in corpus]

tok  = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").cuda().eval()

def encode(lst, bs=32):
    vecs = []
    for i in range(0, len(lst), bs):
        batch = tok(lst[i:i+bs], return_tensors="pt",
                    padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            v = bert(**batch).last_hidden_state[:,0]      # CLS
        vecs.append(torch.nn.functional.normalize(v, dim=-1).cpu())
    return torch.cat(vecs).numpy()

vecs = encode(texts)                                     # (N,768)

# index = faiss.index_factory(768, "FlatIP")               # cosine
index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
faiss.write_index(index, "sem_mem.index")
json.dump(corpus, open("sem_mem.meta", "w"))
print("FAISS index + meta written.")