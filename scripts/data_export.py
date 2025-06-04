import json, pathlib
src = "datasets/unified_data.jsonl"
dst = "datasets/semantic_corpus_from_unified.jsonl"

with open(src) as f, open(dst,"w") as g:
    for line in f:
        item = json.loads(line)
        oid  = item["obj_id"]
        for t in item.get("corpus_texts", []):
            g.write(json.dumps({"text": t, "obj_id": oid}) + "\n")

print(" 轉出", dst)
