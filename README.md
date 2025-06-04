# CAMERA_3D

CAMERA_3D 是一個針對多視角 3D 物件檢索的實驗性專案。專案將文字描述與多張渲染圖像進行對齊，並透過兩階段流程訓練：首先預訓練跨模態編碼器，再使用 reranker 微調。

## 主要內容

- **datasets/** : 各式資料集定義，例如 `UnifiedDataset` 會讀取每個物件的文字與多視角圖片。
- **models/**   : 文字與視覺編碼器（RAGTextEncoder、GFMVEncoder）、CrossModalReranker 等模組。
- **train_two_stage.py** : 主要訓練腳本，包含兩階段流程與 wandb 紀錄。
- **evaluator_v2.py**    : 以 Recall、mAP、NDCG 等指標評估模型，亦支援載入 reranker。

## 環境需求

- Python 3.8 以上
- PyTorch 與 torchvision
- faiss
- transformers
- 其他依賴：numpy、tqdm、psutil、matplotlib

## 訓練範例

```bash
python train_two_stage.py \
    --data_jsonl datasets/unified_data.jsonl \
    --out ckpts \
    --views 12 \
    --bs 8 \
    --topk 4
```

上述指令會輸出 `enc1.pth`、`rerank.pth` 等權重檔案到指定資料夾。

## 評估範例

```bash
python evaluator_v2.py \
    --data datasets/unified_data.jsonl \
    --ckpt ckpts \
    --bs 32 \
    --views 12 \
    --topk 4
```

執行後將列印 Recall@K、mAP@10、NDCG@10 以及平均資源使用量等統計。

## 版權

本專案僅供研究用途，請依原始作者授權條款使用。
