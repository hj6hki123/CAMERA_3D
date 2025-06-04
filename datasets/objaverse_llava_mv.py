"""
依你 jsonl & 圖片路徑 (*.png) 實作,多視角 → Tensor(V,C,H,W)
"""
import json, os, random
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ObjaverseLlavaMV(Dataset):
    def __init__(self, caption_jsonl:str, img_root:str,
                 num_views:int=12, image_size:int=224,
                 caption_strategy:str="random"):
        self.V   = num_views
        self.tr  = transforms.Compose([transforms.Resize((image_size,image_size)),
                                       transforms.ToTensor()])
        self.mode= caption_strategy
        buckets = defaultdict(list)
        with open(caption_jsonl) as f:
            for line in f:
                j = json.loads(line)
                buckets[j["obj_id"]].append((j["view"],j["caption"]))
        self.objs=[]
        for oid,lst in buckets.items():
            lst.sort(key=lambda x:x[0])
            if len(lst)>=num_views and \
               all(os.path.isfile(os.path.join(img_root,oid,v)) for v,_ in lst[:num_views]):
                self.objs.append((oid,lst[:num_views]))
        assert self.objs,"no valid objects"

        self.img_root=img_root
    def __len__(self): return len(self.objs)

    def __getitem__(self,idx):
        oid,lst=self.objs[idx]
        caps=[c for _,c in lst]
        if self.mode=="first": cap=caps[0]
        elif self.mode=="concat": cap=" ".join(caps)
        else: cap=random.choice(caps)

        imgs=[]
        for v,_ in lst:
            p=os.path.join(self.img_root,oid,v)
            imgs.append(self.tr(Image.open(p).convert("RGB")))
        return cap, torch.stack(imgs)     # (V,C,H,W)
