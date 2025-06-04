#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将每行一个 query 的 unified_data.jsonl，按 obj_id 合并为“每个 obj_id 一行”。
新的结构里，所有同一个 obj_id 下的 query 都存在 "queries" list 里，
views/corpus_texts 等字段保持一致（假设它们对同一 obj_id 是相同的）。
"""

import json
from collections import defaultdict

# 1. 读入原始文件
in_path = "datasets/unified_data.jsonl"
out_path = "datasets/unified_data_grouped.jsonl"

# 按 obj_id 分组: 
#   group_data[obj_id] = { "views": [...], "corpus_texts": [...], "queries": [ ... ] }
group_data = {}

with open(in_path, "r", encoding="utf-8") as fin:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)

        oid = item["obj_id"]
        q   = item["query"]
        views = item.get("views", [])
        corpus = item.get("corpus_texts", [])

        # 如果第一次见到这个 obj_id，就初始化
        if oid not in group_data:
            group_data[oid] = {
                "views": views.copy(),
                "corpus_texts": corpus.copy(),
                "queries": [q]
            }
        else:
            # 如果已经有这一条 obj_id，只需把 query 加到列表里
            # 理论上 views/corpus_texts 对同一个 obj_id 应该是相同的，
            # 如果不一样，你可以自行决定要不要去重或者报错
            group_data[oid]["queries"].append(q)

# 2. 把分组之后的数据写回到一个新的 JSONL
with open(out_path, "w", encoding="utf-8") as fout:
    for oid, data in group_data.items():
        new_rec = {
            "obj_id": oid,
            "queries": data["queries"],
            "views": data["views"],
            "corpus_texts": data["corpus_texts"],
        }
        fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")

print(f"合并完成 → 新的文件: {out_path}")
