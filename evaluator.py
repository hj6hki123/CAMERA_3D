#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate trained CAMERA model – Recall@1/5/10  +  QPS
用法:  python evaluator.py  --data unified_data.jsonl  --corpus semantic_corpus.json  --ckpt ckpts
"""
import time, argparse, faiss, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.unified_dataset import UnifiedDataset
from models.rag_text_encoder  import RAGTextEncoder
from models.gf_mv_encoder     import GFMVEncoder
from train_two_stage          import build_faiss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ckpt",   required=True)
    args = ap.parse_args()

    ds   = UnifiedDataset(args.data)
    Kset = (1,5,10)

    # load encoders ----------------------------------------------------------
    ck   = torch.load(f"{args.ckpt}/enc1.pth", map_location="cuda")
    txt  = RAGTextEncoder(args.corpus).cuda().eval(); txt.load_state_dict(ck["txt"])
    vis  = GFMVEncoder().cuda().eval();            vis.load_state_dict(ck["vis"])

    # build faiss ------------------------------------------------------------
    index, all_v = build_faiss(ds, vis, argparse.Namespace(bs=32))

    # loop -------------------------------------------------------------------
    hit = {k:0 for k in Kset};  tot = 0;  t0 = time.time()
    dl  = DataLoader(ds, 32, num_workers=4)
    for q,_, oid in dl:
        tv,_,_ = txt(list(q), list(oid))
        sims, idx = index.search(tv.cpu().numpy(), max(Kset))
        for i,o in enumerate(oid):
            tot += 1
            for k in Kset:
                if o in [ds.items[j]['obj_id'] for j in idx[i,:k]]:
                    hit[k] += 1

    dur = time.time()-t0
    for k in Kset:
        print(f"Recall@{k}: {hit[k]/tot:.3f}")
    print(f"QPS: {tot/dur:.2f}")

if __name__ == "__main__":
    main()
