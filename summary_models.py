# -*- coding: utf-8 -*-
import os, json, tempfile, torch, argparse
from torchinfo import summary

# 匯入各模型
from models.cross_modal_reranker import CrossModalReranker
from models.gf_mv_encoder import GFMVEncoder
from models.dense_retriever import DenseRetriever
from models.rag_text_encoder import RAGTextEncoder

# ---------------------------------------------------------------------------
# 設備檢測和配置
# ---------------------------------------------------------------------------
def get_device(device_arg: str = "auto") -> torch.device:
    """
    根據參數和硬件可用性選擇最佳設備
    
    Args:
        device_arg: 'auto', 'cuda', 'mps', 'cpu' 或具體設備如 'cuda:0'
    
    Returns:
        torch.device: 選定的設備
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ 自動選擇 CUDA 設備: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✓ 自動選擇 Apple Silicon MPS 設備")
        else:
            device = torch.device("cpu")
            print("✓ 自動選擇 CPU 設備")
    else:
        device = torch.device(device_arg)
        print(f"✓ 使用指定設備: {device}")
    
    return device

def save_text(fname, text):
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)

def summarize_reranker(output_dir, device, args):
    """使用訓練時的預設參數初始化 CrossModalReranker"""
    # 使用訓練代碼中的預設參數
    model = CrossModalReranker(
        txt_dim=768,      # 訓練時預設
        vis_dim=512,      # 訓練時預設  
        heads=8,          # 訓練時預設
        num_layers=2,     # 訓練時預設
        txt_max_len=args.txt_max_len,  # 可調整
        vis_max_len=args.vis_max_len   # 可調整
    ).to(device)
    model.eval()
    
    buf = []
    buf.append(f"=== CrossModalReranker 結構 (設備: {device}) ===\n")
    buf.append(str(model) + "\n\n")
    
    # 參數統計
    buf.append("=== CrossModalReranker 參數統計 ===\n")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buf.append(f"總參數數量: {total_params:,}\n")
    buf.append(f"可訓練參數: {trainable_params:,}\n\n")
    
    # 手動測試 forward
    buf.append("=== 測試 forward ===\n")
    try:
        with torch.no_grad():
            # 使用與訓練時一致的輸入尺寸
            txt_input = torch.randn(1, args.txt_max_len, 768).to(device)
            vis_input = torch.randn(1, args.vis_max_len, 512).to(device)
            scores = model(txt_input, vis_input)
            buf.append(f"文字輸入 shape: {txt_input.shape}\n")
            buf.append(f"視覺輸入 shape: {vis_input.shape}\n")
            buf.append(f"輸出分數 shape: {scores.shape}\n")
            buf.append(f"分數值: {scores.item():.4f}\n")
    except Exception as e:
        buf.append(f"forward 測試失敗: {e}\n")
        
    save_text(os.path.join(output_dir, "summary_reranker.txt"), "".join(buf))
    print(f"✓ CrossModalReranker 摘要已保存 (設備: {device})")

def summarize_gfmv(output_dir, device, args):
    """使用與訓練時完全一致的參數初始化 GFMVEncoder"""
    # 使用訓練代碼中傳入的 args.views 參數
    model = GFMVEncoder(
        num_views=args.views      # 與 train.py 中的 args.views 一致                 # 訓練時預設
    ).to(device)
    model.eval()
    
    buf = []
    buf.append(f"=== GFMVEncoder 結構（精簡版，設備: {device}） ===\n")
    buf.append(str(model) + "\n\n")
    
    # 參數統計
    buf.append("=== GFMVEncoder 參數統計 ===\n")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buf.append(f"總參數數量: {total_params:,}\n")
    buf.append(f"可訓練參數: {trainable_params:,}\n\n")
    
    # 手動測試 forward
    buf.append("=== 測試 forward ===\n")
    try:
        # 使用與訓練時一致的輸入格式
        # GFMVEncoder 的文本輸入應該是 128 維（模型內部維度）
        img_input = torch.randn(1, args.views, 3, 224, 224).to(device)
        txt_input = torch.randn(1, 512).to(device)  # 使用模型內部維度 128
        
        with torch.no_grad():
            global_vec, attn_maps, final_tok = model(img_input, txt_input)
            buf.append(f"輸入圖像 shape: {img_input.shape}\n")
            buf.append(f"輸入文字 shape: {txt_input.shape}\n")
            buf.append(f"輸出全域向量 shape: {global_vec.shape}\n")
            buf.append(f"最終 token 序列 shape: {final_tok.shape}\n")
            buf.append(f"視覺注意力層數: {len(attn_maps['vis'])}\n")
            buf.append(f"融合注意力層數: {len(attn_maps['fuse'])}\n")
            if len(attn_maps['vis']) > 0:
                buf.append(f"視覺注意力 shape (第1層): {attn_maps['vis'][0].shape}\n")
            if len(attn_maps['fuse']) > 0:
                buf.append(f"融合注意力 shape (第1層): {attn_maps['fuse'][0].shape}\n")
                
    except Exception as e:
        buf.append(f"forward 測試失敗: {e}\n")
        # 添加更詳細的調試信息
        buf.append(f"模型參數: views={args.views}, dim=128\n")
        buf.append(f"錯誤詳情: {str(e)}\n")
        
    save_text(os.path.join(output_dir, "summary_gfmv.txt"), "".join(buf))
    print(f"✓ GFMVEncoder 摘要已保存 (設備: {device})")

def summarize_dense_retriever(output_dir, device, args):
    """使用與訓練時一致的參數初始化 DenseRetriever"""
    # 建立一個超小 corpus JSONL，確保有足夠的樣本供檢索
    tmp = os.path.join(output_dir, "tmp_corpus.jsonl")
    samples = [
        {"text": "這是測試一句話。", "obj_id": "id1"},
        {"text": "另一個簡短文字", "obj_id": "id2"},
        {"text": "第三個測試文本", "obj_id": "id3"},
        {"text": "第四個測試文本", "obj_id": "id4"},
        {"text": "第五個測試文本", "obj_id": "id5"}  # 增加更多樣本以避免 topk 超出範圍
    ]
    with open(tmp, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    buf = []
    buf.append(f"=== DenseRetriever 結構 (設備: {device}) ===\n")
    try:
        # 使用訓練時的參數配置，但降低 batch size 以適應小數據集
        model = DenseRetriever(
            corpus_jsonl=tmp, 
            device=str(device), 
            batch=min(args.bs, len(samples)),  # 避免 batch size 超過樣本數
            cache=False
        )
        buf.append(str(model) + "\n\n")
        
        # 檢查 memory_vec 設備位置
        buf.append(f"memory_vec 設備: {model.memory_vec.device}\n")
        buf.append(f"qenc 設備: {next(model.qenc.parameters()).device}\n\n")
        
        # 參數統計
        buf.append("=== DenseRetriever 參數統計 ===\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        buf.append(f"總參數數量: {total_params:,}\n")
        buf.append(f"可訓練參數: {trainable_params:,}\n")
        buf.append(f"memory_vec shape: {model.memory_vec.shape}\n\n")
            
        # 測試簡單 forward
        buf.append("=== 測試 forward ===\n")
        try:
            model = model.to(device)
            q_list = ["測試查詢"]
            # 確保 topk 不超過可用樣本數
            effective_topk = min(args.topk, len(samples))
            q_vec, sims, idx, ctx, loss = model(q_list, obj_ids=["id1"], topk=effective_topk)
            buf.append(f"forward 成功: q_vec shape={q_vec.shape}, loss={loss.item():.4f}\n")
            buf.append(f"有效 topk: {effective_topk}\n")
            buf.append(f"檢索到的文本: {ctx[0] if ctx else 'None'}\n")
        except Exception as e:
            buf.append(f"forward 測試失敗: {e}\n")
            buf.append(f"可用樣本數: {len(samples)}, 請求 topk: {args.topk}\n")
            
    except Exception as e:
        buf.append(f"DenseRetriever 初始化失敗: {e}\n")
    
    save_text(os.path.join(output_dir, "summary_dense_retriever.txt"), "".join(buf))
    print(f"✓ DenseRetriever 摘要已保存 (設備: {device})")
    
    # 清理臨時文件
    try: 
        os.remove(tmp)
        cache_file = tmp + ".pt"
        if os.path.exists(cache_file):
            os.remove(cache_file)
    except: 
        pass

def summarize_rag_text_encoder(output_dir, device, args):
    """使用與訓練時一致的參數初始化 RAGTextEncoder"""
    # 建立一個非常小的 unified_dummy.jsonl（模擬 args.data_jsonl 格式）
    tmp = os.path.join(output_dir, "unified_dummy.jsonl")
    # 增加更多樣本以避免 topk 超出範圍
    samples = [
        {"obj_id": "id1", "corpus_texts": ["測試語意一"], "queries": ["測試問句"]},
        {"obj_id": "id2", "corpus_texts": ["測試語意二"], "queries": ["另一問句"]},
        {"obj_id": "id3", "corpus_texts": ["測試語意三"], "queries": ["第三問句"]},
        {"obj_id": "id4", "corpus_texts": ["測試語意四"], "queries": ["第四問句"]},
        {"obj_id": "id5", "corpus_texts": ["測試語意五"], "queries": ["第五問句"]}
    ]
    with open(tmp, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    buf = []
    buf.append(f"=== RAGTextEncoder 結構 (設備: {device}) ===\n")
    try:
        # 使用與訓練時完全一致的參數
        effective_topk = min(args.topk, len(samples))  # 確保 topk 不超過樣本數
        model = RAGTextEncoder(
            unified_jsonl=tmp,      # 模擬 args.data_jsonl
            top_k=effective_topk,   # 使用安全的 topk 值
            device=str(device)      # 與訓練時的設備配置一致
        )
        model = model.to(device)
        model.eval()
        buf.append(str(model) + "\n\n")
        
        # 列出子模組
        buf.append("-> retriever.qenc:\n" + str(model.retriever.qenc) + "\n\n")
        buf.append("-> fusion:\n" + str(model.fusion) + "\n\n")
        
        # 參數統計
        buf.append("=== RAGTextEncoder 參數統計 ===\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        buf.append(f"總參數數量: {total_params:,}\n")
        buf.append(f"可訓練參數: {trainable_params:,}\n")
        buf.append(f"memory_vec shape: {model.retriever.memory_vec.shape}\n")
        buf.append(f"有效 top_k: {effective_topk}\n\n")
            
        # 測試 forward
        buf.append("=== 測試 forward ===\n")
        try:
            q_list = ["測試"]
            vec, loss, tok, attn = model(q_list, obj_ids=["id1"], return_loss=True)
            buf.append(f"查詢: {q_list[0]}\n")
            buf.append(f"輸出向量 shape: {vec.shape}\n")
            buf.append(f"retriever loss: {loss.item():.4f}\n")
            buf.append(f"token 序列 shape: {tok.shape}\n")
            buf.append(f"融合注意力層數: {len(attn)}\n")
            if len(attn) > 0:
                buf.append(f"第1層注意力 shape: {attn[0].shape}\n")
        except Exception as e:
            buf.append(f"forward 測試失敗: {e}\n")
            buf.append(f"可用樣本數: {len(samples)}, 原始 topk: {args.topk}, 有效 topk: {effective_topk}\n")
            
    except Exception as e:
        buf.append(f"RAGTextEncoder 初始化失敗: {e}\n")
    
    save_text(os.path.join(output_dir, "summary_rag_text_encoder.txt"), "".join(buf))
    print(f"✓ RAGTextEncoder 摘要已保存 (設備: {device})")
    
    # 清理臨時文件
    try: 
        os.remove(tmp)
        temp_corpus = os.path.join(output_dir, "temp_corpus.jsonl")
        if os.path.exists(temp_corpus):
            os.remove(temp_corpus)
        cache_file = temp_corpus + ".pt"
        if os.path.exists(cache_file):
            os.remove(cache_file)
    except: 
        pass

def main():
    parser = argparse.ArgumentParser(description="生成所有模型的結構摘要")
    # 設備和輸出參數
    parser.add_argument("--device", type=str, default="auto",
                       help="設備選擇: 'auto', 'cuda', 'mps', 'cpu' 或具體設備如 'cuda:0'")
    parser.add_argument("--output_dir", type=str, default="model_summaries",
                       help="輸出目錄")
    
    # 訓練參數（與 train.py 保持一致）
    parser.add_argument("--views", type=int, default=12,
                       help="多視角數量（與訓練時一致）")
    parser.add_argument("--bs", type=int, default=8,
                       help="批次大小（與訓練時一致）")
    parser.add_argument("--topk", type=int, default=4,
                       help="檢索 top-k（與訓練時一致）")
    
    # CrossModalReranker 特定參數
    parser.add_argument("--txt_max_len", type=int, default=10,
                       help="文字最大長度")
    parser.add_argument("--vis_max_len", type=int, default=12,
                       help="視覺最大長度")
    
    args = parser.parse_args()
    
    # 設備配置
    device = get_device(args.device)
    
    # 設備優化設置
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"  - CUDA 記憶體: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f}GB")
    elif device.type == "mps":
        print("  - 啟用 MPS 優化")
    else:
        torch.set_num_threads(torch.get_num_threads())
        print(f"  - CPU 線程數: {torch.get_num_threads()}")
    
    # 創建輸出目錄
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n開始生成模型摘要，使用設備: {device}")
    print("=" * 50)
    print(f"參數配置:")
    print(f"  - views: {args.views}")
    print(f"  - batch_size: {args.bs}")
    print(f"  - topk: {args.topk}")
    print(f"  - txt_max_len: {args.txt_max_len}")
    print(f"  - vis_max_len: {args.vis_max_len}")
    print("=" * 50)
    
    try:
        summarize_reranker(output_dir, device, args)
        summarize_gfmv(output_dir, device, args)
        summarize_dense_retriever(output_dir, device, args)
        summarize_rag_text_encoder(output_dir, device, args)
        
        print("=" * 50)
        print(f"✓ 所有模型摘要已成功生成，保存至: {output_dir}")
        
        # 列出生成的文件
        files = [f for f in os.listdir(output_dir) if f.startswith("summary_")]
        print(f"生成的文件: {', '.join(files)}")
        
    except Exception as e:
        print(f"❌ 生成摘要時發生錯誤: {e}")
        raise

if __name__ == "__main__":
    main()