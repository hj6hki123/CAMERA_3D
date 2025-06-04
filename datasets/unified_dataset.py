# -*- coding: utf-8 -*-
import os
import random
import json
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict

'''
// 每行一個 JSON 物件
{
  "obj_id": "0005ea6ae6944ebd90a4c8de52ecfb2f",
  "queries": [
    "圖片裡有一張木製桌子 ...",
    "深棕色矩形的存在 ...",
    "...",
    "..."
  ],
  "corpus_texts": [
    "晚餐桌。",
    "木製",
    "晚餐",
    "棕色",
  ]
}
'''

class UnifiedDataset(Dataset):
    def __init__(self,
                 jsonl_path: str,
                 view_dir: str = "/home/klooom/cheng/3d_retrival/objaverse-rendering/views",
                 image_size: int = 224,
                 num_views: int = 12):
        super().__init__()

        # 1) 先讀取原始 JSONL，每行包含一個 obj_id 與多個 queries 列表
        raw_items = [json.loads(l) for l in open(jsonl_path, encoding="utf-8") if l.strip()]

        # 2) 將每筆紀錄的 queries 攤平成多條單獨紀錄
        self.items = []
        for rec in raw_items:
            oid = rec["obj_id"]
            for q in rec["queries"]:
                self.items.append({
                    "obj_id": oid,
                    "query":  q,
                    "corpus_texts": rec.get("corpus_texts", [])
                })

        # 3) 在「攤平的 self.items」上，建立 obj2idx 字典：將同一 obj_id 出現的所有索引存入列表
        self.obj2idx = defaultdict(list)
        for i, rec in enumerate(self.items):
            self.obj2idx[rec["obj_id"]].append(i)
        # 例如：self.obj2idx["0005ea6ae6944ebd90a4c8de52ecfb2f"] = [0, 1, 2, … 11]

        # 4) 設定 __getitem__ 使用的多視角圖片路徑與 transform 等參數
        self.base_view_dir = view_dir
        self.num_views = num_views
        self.tr = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        obj_id = rec["obj_id"]
        query = rec["query"]

        # 取得該 obj_id 的視角資料夾（有 V 張 png 圖），再挑選前 num_views 張
        view_folder = os.path.join(self.base_view_dir, obj_id)
        all_pngs = sorted(glob.glob(os.path.join(view_folder, "*.png")))
        if len(all_pngs) == 0:
            raise RuntimeError(f"{obj_id} 在資料夾 {view_folder} 中找不到任何 png 檔案")

        selected = all_pngs[:self.num_views]
        imgs = [self.tr(Image.open(p).convert("RGB")) for p in selected]
        # 如果圖片數不足 num_views，就複製最後一張補齊
        while len(imgs) < self.num_views:
            imgs.append(imgs[-1].clone())
        imgs = torch.stack(imgs, dim=0)  # (V, C, H, W)

        # 多回傳一個 idx —— 該筆樣本在 self.items 攤平列表的索引
        return query, imgs.float(), obj_id, idx

    def get_flatten_index(self, obj_id, query):
        """
        若業務上需要根據 (obj_id, query) 找到對應 idx，可用此方法：
        """
        for i, rec in enumerate(self.items):
            if rec["obj_id"] == obj_id and rec["query"] == query:
                return i
        raise ValueError(f"找不到 obj_id={obj_id} 且 query={query} 對應的 idx")
