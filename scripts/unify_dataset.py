import json, os
base = '/home/klooom/cheng/3d_retrival/CAMERA_3D/datasets'
# 1. 讀 model_paths.json
with open(f"{base}/model_paths.json") as f:
    model_paths = json.load(f)
obj_id_to_views = {}
for path in model_paths:
    obj_id = os.path.basename(path).split(".")[0]
    views_dir = f"/home/klooom/cheng/3d_retrival/objaverse-rendering/views/{obj_id}"
    view_imgs = sorted([os.path.join(views_dir, v) for v in os.listdir(views_dir) if v.endswith(".png")])
    obj_id_to_views[obj_id] = view_imgs

# 2. 讀 semantic_corpus.json
with open(f"{base}/semantic_corpus.json") as f:
    corpus_data = json.load(f)
obj_id_to_corpus = {}
for item in corpus_data:
    obj_id_to_corpus.setdefault(item["id"], []).append(item["text"])

# 3. 整合 llava captions
with open(f"{base}/objaverse_mv_llava_captions.jsonl") as f:
    lines = f.readlines()

with open(f"{base}/unified_data.jsonl", "w") as out_f:
    for line in lines:
        item = json.loads(line)
        obj_id = item["obj_id"]
        entry = {
            "query": item["caption"],
            "obj_id": obj_id,
            "views": obj_id_to_views.get(obj_id, []),
            "corpus_texts": obj_id_to_corpus.get(obj_id, [])
        }
        json.dump(entry, out_f)
        out_f.write("\n")
