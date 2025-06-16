# build_augmented_dataset.py

import json
from tqdm import tqdm
from models.external_retriever import ExternalRetriever # 確保可以導入

def build_augmented_dataset(original_data_path: str, output_path: str, top_k: int = 3):
    """
    讀取原始資料集，為每個查詢檢索外部上下文，並寫入一個新的增強資料集。
    """
    # 初始化外部檢索器
    # 請替換成您自己的 User-Agent
    retriever = ExternalRetriever(lang='en')
    
    # 讀取原始資料
    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data = [json.loads(line) for line in f]
        
    print(f"Loaded {len(original_data)} records from {original_data_path}")
    
    # 打開輸出檔案，準備寫入
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(original_data, desc="Augmenting dataset"):
            query = item.get('query')
            if not query:
                continue
            
            # 進行外部檢索
            retrieved_contexts = retriever.retrieve(query, top_k=top_k)
            
            # 建立新的紀錄，包含原始資訊和檢索到的上下文
            new_record = item.copy()
            new_record['external_contexts'] = retrieved_contexts
            
            # 將新紀錄寫入檔案
            f_out.write(json.dumps(new_record, ensure_ascii=False) + '\n')
            
    print(f"\n✓ Successfully created augmented dataset at: {output_path}")

if __name__ == '__main__':
    # 您清洗好的、語意一致的資料集
    INPUT_PATH = r"C:\Users\klooom\Desktop\code\CAMERA_3D\datasets\selected_data.jsonl" 
    # 我們要生成的、新的增強資料集
    OUTPUT_PATH = "datasets/augmented_data.jsonl"
    
    build_augmented_dataset(INPUT_PATH, OUTPUT_PATH, top_k=3)