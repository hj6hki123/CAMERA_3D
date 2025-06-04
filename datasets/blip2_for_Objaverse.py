# blip2_caption_objaverse.py
# =============================================================
# 1) 套件
# =============================================================
import os, json, random, torch, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import open_clip                     # pip install open_clip_torch

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

# =============================================================
# 2) 下載 / 載入模型
# =============================================================
BLIP_ID = "Salesforce/blip2-flan-t5-xl"         # 或 blip2-opt-2.7b
CLIP_ID = "ViT-L-14"

print(" Loading BLIP-2 ...")
blip_proc = Blip2Processor.from_pretrained(BLIP_ID)
blip      = Blip2ForConditionalGeneration.from_pretrained(
               BLIP_ID, torch_dtype=dtype, low_cpu_mem_usage=True
           ).to(device)

print(" Loading CLIP ...")
clip_model, _, clip_proc = open_clip.create_model_and_transforms(CLIP_ID, pretrained="openai")
clip_model = clip_model.to(device).eval()

# =============================================================
# 3) 產生 + 排序 caption
# =============================================================
@torch.no_grad()
def top_caption(img: Image.Image, num_cap=10, max_tokens=60):
    captions = []
    for _ in range(num_cap):
        inputs = blip_proc(
            images=img,
            text="Describe the image in detail.Please ignore the background",
            return_tensors="pt"
        ).to(device, dtype)

        out = blip.generate(**inputs,
                            max_new_tokens=max_tokens,
                            do_sample=False)
        cap = blip_proc.decode(out[0], skip_special_tokens=True).strip()
        captions.append(cap)

    # --- CLIP 圖片向量 ---
    img_emb = clip_model.encode_image(
        clip_proc(img).unsqueeze(0).to(device)
    )
    img_emb = F.normalize(img_emb, dim=-1)

    best_cap, best_score = None, -1
    for cap in captions:
        txt_tokens = open_clip.tokenize([cap]).to(device)
        txt_emb = clip_model.encode_text(txt_tokens)
        txt_emb = F.normalize(txt_emb, dim=-1)
        score = (img_emb @ txt_emb.T).item()
        if score > best_score:
            best_score, best_cap = score, cap

    return best_cap

# =============================================================
# 4) 主流程:遍歷 Objaverse 多視角資料夾
# =============================================================
def caption_dataset(root_dir: str, out_jsonl="objaverse_blip2.jsonl",
                    views=12, num_cap=10):
    fout = open(out_jsonl, "w", encoding="utf-8")
    obj_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    for obj_id in tqdm(obj_ids, desc="objects"):
        obj_path = os.path.join(root_dir, obj_id)
        for vid in range(views):
            img_file = os.path.join(obj_path, f"{vid:03d}.png")
            if not os.path.exists(img_file):
                print(f" miss {img_file}")
                continue
            img = Image.open(img_file).convert("RGB")

            cap = top_caption(img, num_cap=num_cap)
            record = dict(obj_id=obj_id, view=f"{vid:03d}", caption=cap)
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    fout.close()
    print(f" Done. Saved to {out_jsonl}")

# =============================================================
# 5) 執行
# =============================================================
if __name__ == "__main__":
    DATA_ROOT = "/home/klooom/cheng/3d_retrival/objaverse-rendering/views"   # 路徑
    caption_dataset(DATA_ROOT,
                    out_jsonl="objaverse_mv_blip2_top1_1.jsonl",
                    views=12, num_cap=10)
