import os, json, random, torch, math
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm

# ---------- 0) 模型初始化 ----------
model_id   = "llava-hf/llava-1.5-7b-hf"
processor  = AutoProcessor.from_pretrained(model_id)
llava      = LlavaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
).cuda()

# ---------- 1)  prompt 模板 ----------
prompt = (
    "Focus only on the main object in the image."
    "Describe its SHAPE, major PARTS, MATERIAL or texture."
    "If possible, describe the DESIGN STYLE or era it represents (e.g., Baroque, minimalist, industrial)."
    "Ignore background, lighting, camera or scene details."
)

def build_chat(prompt):
    conv = [
        {"role":"user",
         "content":[
             {"type":"text","text":prompt},
             {"type":"image"}
         ]}
    ]
    return processor.apply_chat_template(conv, add_generation_prompt=True)

# ---------- 2) 單張圖片 → caption ----------
@torch.no_grad()
def caption_one(img: Image.Image, prompt_txt: str):
    chat_prompt = build_chat(prompt_txt)
    inputs = processor(images=img, text=chat_prompt, return_tensors="pt").to("cuda", torch.float16)
    out_ids = llava.generate(**inputs, max_new_tokens=120, do_sample=False)[0]
    caption = processor.decode(out_ids, skip_special_tokens=True)
    caption = caption.split("ASSISTANT:")[-1].strip()  
    return caption

# ---------- 3) 遍歷資料夾 ----------
def process_dataset(root_dir, out_json="objaverse_captions.jsonl"):
    fout = open(out_json, "w")
    for obj_id in tqdm(sorted(os.listdir(root_dir))):
        obj_path = os.path.join(root_dir, obj_id)
        if not os.path.isdir(obj_path):
            continue
        view_files = sorted([f for f in os.listdir(obj_path) if f.endswith(".png")])
        for vfile in view_files:
            img_path = os.path.join(obj_path, vfile)
            img = Image.open(img_path).convert("RGB")
            prompt_txt = prompt  
            cap = caption_one(img, prompt_txt)
            record = {
                "obj_id": obj_id,
                "view":   vfile,
                "prompt": prompt_txt,
                "caption": cap
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    fout.close()

# ---------- 4) 執行 ----------
if __name__ == "__main__":
    DATA_ROOT = "/home/klooom/cheng/3d_retrival/objaverse-rendering/views"  # 替換成你的資料夾
    process_dataset(DATA_ROOT, "objaverse_mv_llava_captions.jsonl")
