# attribute_select.py
# -----------------------------------------------------------
# 功能：自動從多句 caption 中
#   1. 移除樣板句
#   2. 切句並清洗
#   3. 以「顏色、材質、形狀、風格」四大屬性為優先
#      貪婪挑句，兼顧資訊多樣性（餘弦相似度去重）
#   4. 把最能代表整體語意的中心句設為 query
#      其餘經過屬性-greedy 篩選的句子寫入 corpus_texts
#
# 使用：
#   python attribute_select.py --inp unified_data.jsonl --out selected.jsonl
#
# -----------------------------------------------------------
import json, re, argparse
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk; nltk.download("punkt", quiet=True)
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# ───── 1. 句子向量模型 ─────
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ───── 2. Prototype 提示（可自由增刪）─────
PROTO_RAW = {
    "color": [
        "The object is red.", "The object is blue.",
        "The object is green.", "It is black."
    ],
    "material": [
        "It is made of wood.", "It is made of metal.",
        "It is made of plastic.", "It is made of glass."
    ],
    "shape": [
        "It has a round shape.", "It is rectangular.",
        "It is square.", "It is oval."
    ],
    "style": [
        "It is in modern style.", "It looks vintage.",
        "It has an industrial style.", "It is minimalist."
    ]
}
# 預先編碼 prototype
PROTO_VEC = {k: MODEL.encode(v, normalize_embeddings=True)
             for k, v in PROTO_RAW.items()}

# ───── 3. 樣板句正規化規則 ─────
BOILERPLATE = [
    r"^The main object in the image is [^\.]*\.",
    r"^The image features [^\.]*\.",
    r"^It has [^\.]*\."
]
def strip_boiler(text: str) -> str:
    for p in BOILERPLATE:
        text = re.sub(p, "", text, flags=re.I).strip()
    return text

# ───── 4. 句子切分與清洗 ─────
def split_and_clean(text: str, max_sent: int = 3) -> list[str]:
    sents = sent_tokenize(strip_boiler(text))
    return [s.strip() for s in sents[:max_sent] if s.strip()]

# ───── 5. Prototype 向量比對取屬性 ─────
def attr_from_proto(sent: str,
                    thresh: float = 0.45) -> set[str]:
    vec = MODEL.encode(sent, normalize_embeddings=True)
    found = set()
    for attr, bank in PROTO_VEC.items():
        if (bank @ vec).max() > thresh:
            found.add(attr)
    return found

# ───── 6. greedy-attribute 篩選 ─────
def greedy_attribute_filter(sents: list[str],
                            vecs: np.ndarray,
                            thresh: float = 0.85,
                            max_keep: int = 10) -> list[int]:
    covered: set[str] = set()
    kept: list[int]   = []
    order = np.argsort([-len(s) for s in sents])  # 字元較多優先

    for idx in order:
        if len(kept) >= max_keep:
            break
        attrs = attr_from_proto(sents[idx])
        # 先以補屬性為主
        if not attrs.issubset(covered):
            kept.append(idx)
            covered |= attrs
            continue
        # 屬性皆已覆蓋→做去重判斷
        if not kept:
            kept.append(idx); continue
        sim_max = (vecs[kept] @ vecs[idx]).max()
        if sim_max < thresh:
            kept.append(idx)
    return kept


# ───── 7. 主流程─────
def build_selected(inp: str,
                   out: str,
                   thresh_sim: float = 0.85,
                   max_keep: int = 10,
                   thresh_topic: float = 0.5) -> None: # <-- 新增 thresh_topic 參數
    """
    主流程：讀取、清洗、篩選並寫入新的 JSONL 檔案。
    新增了 thresh_topic 參數來控制主題一致性過濾的嚴格程度。
    """
    with open(inp, encoding="utf-8") as fin,\
         open(out, "w", encoding="utf-8") as fout:

        # 使用 tqdm 來顯示進度
        from tqdm import tqdm
        num_lines = sum(1 for line in open(inp, encoding="utf-8"))
        fin.seek(0)

        for line in tqdm(fin, total=num_lines, desc="Processing records"):
            rec = json.loads(line)
            oid = rec["obj_id"]
            caps: list[str] = rec.get("queries", [])
            
            sents: list[str] = []
            for cap in caps:
                sents.extend(split_and_clean(cap))
            
            # 先對所有句子做一次基礎去重
            sents = sorted(list(set(sents)), key=sents.index)
            
            if not sents:
                continue

            # 向量化
            vecs = MODEL.encode(sents, normalize_embeddings=True)
            
            # 選出代表性 query
            centroid = normalize(vecs.mean(0, keepdims=True))[0]
            rep_idx  = int(np.argmax(vecs @ centroid))
            query_sent = sents[rep_idx]
            query_vec = vecs[rep_idx]

            # ==========================================================
            # ==                  核心修改：主題一致性過濾              ==
            # ==========================================================
            
            # 1. 計算所有候選句子與代表性 query 的語意相似度
            all_sims = vecs @ query_vec.T

            # 2. 篩選出主題相關的句子 (相似度需大於門檻，且不能是 query 自己)
            on_topic_indices = []
            for i in range(len(sents)):
                if i != rep_idx and all_sims[i] > thresh_topic:
                    on_topic_indices.append(i)

            # 3. 如果沒有任何主題相關的句子，corpus_texts 為空
            if not on_topic_indices:
                keep_idx = []
            else:
                # 4. 只從「主題相關」的句子中，再用 greedy-attribute 策略篩選
                on_topic_sents = [sents[i] for i in on_topic_indices]
                on_topic_vecs = vecs[on_topic_indices]
                
                # greedy_attribute_filter 返回的是 on_topic_sents 這個子列表中的索引
                kept_indices_in_subset = greedy_attribute_filter(
                    on_topic_sents,
                    on_topic_vecs,
                    thresh_sim,
                    max_keep
                )
                # 將子列表的索引映射回原始 sents 列表中的全域索引
                keep_idx = [on_topic_indices[i] for i in kept_indices_in_subset]
            
            # ==========================================================
            
            out_rec = {
                "obj_id":      oid,
                "query":       query_sent,
                "corpus_texts": [sents[i] for i in keep_idx]
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    print(f"已完成屬性化清洗 → {out}")
# ───── 8. CLI 介面 ─────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/home/klooom/cheng/3d_retrival/CAMERA_3D/datasets/unified_data_cleaned.jsonl",
                    help="輸入 jsonl 檔，每行含 queries 陣列")
    ap.add_argument("--out", default="selected_data.jsonl",
                    help="輸出 jsonl 檔")
    ap.add_argument("--sim", type=float, default=0.85,
                    help="相似度去重門檻")
    ap.add_argument("--top", type=int,   default=10,
                    help="每物件最多保留句數 (不含代表句)")
    args = ap.parse_args()

    build_selected(args.inp, args.out, args.sim, args.top)
