import json

raw = [json.loads(l) for l in open("datasets/unified_data_clean.jsonl", encoding="utf-8") if l.strip()]
new_records = []
for rec in raw:
    oid = rec["obj_id"]
    queries = rec.get("queries", [])        # 你的格式里可能是 "query" 或 "queries" 列表
    corpus = rec.get("corpus_texts", [])
    if not corpus:
        # 如果 corpus_texts 为空，就把第一个 query 当成 fallback caption
        # （也可以把所有 queries 都加进 corpus，至少保证不空）
        fallback = queries[0] if len(queries) > 0 else ""
        corpus = [fallback]
    new_records.append({
        "obj_id": oid,
        "queries": queries,
        "corpus_texts": corpus
    })

# 把 new_records 重写回 unified_data.jsonl，或者直接在内存里把它传给后续 Pipeline：
with open("datasets/unified_data.cleaned.jsonl", "w", encoding="utf-8") as fout:
    for rec in new_records:
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")