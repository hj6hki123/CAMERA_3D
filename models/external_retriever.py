# external_retriever.py (修正後完整可執行版)
import wikipedia
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk
import warnings

# 忽略來自 wikipedia 套件的特定 BeautifulSoup 警告
warnings.filterwarnings("ignore", category=UserWarning, module='wikipedia')

# 確保 NLTK 的句子分割模型已下載
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK's 'punkt' model for sentence tokenization...")
    nltk.download('punkt', quiet=True)

class ExternalRetriever:
    """
    一個從維基百科檢索、解析並重排序(Re-rank)外部語義的工具。
    (修正版：使用 'wikipedia' 套件進行搜尋)
    """
    def __init__(self, lang: str = 'en', model_name: str = 'all-MiniLM-L6-v2'):
        """
        初始化檢索器。
        
        Args:
            lang (str): 維基百科的語言版本，預設為 'en' (英文)。
            model_name (str): 用於重排序的 SentenceTransformer 模型名稱。
        """
        # <-- CHANGED: 設定 wikipedia 套件的語言
        wikipedia.set_lang(lang)
        
        # 初始化用於語意相似度計算的重排序模型
        self.rerank_model = SentenceTransformer(model_name)
        print(f"ExternalRetriever initialized with model '{model_name}' for '{lang}' Wikipedia.")

    def search(self, query: str, num_results: int = 5) -> list[str]:
        """
        步驟1: 初步搜尋。從維基百科找出可能相關的頁面標題。
        
        Returns:
            list[str]: 一個包含維基百科頁面標題的列表。
        """
        try:
            # <-- CHANGED: 使用 wikipedia.search()，它直接返回標題列表
            page_titles = wikipedia.search(query, results=num_results)
            return page_titles
        except Exception as e:
            print(f"An error occurred during Wikipedia search for '{query}': {e}")
            return []

    def parse(self, page_title: str, chunk_size: int = 4, overlap: int = 1) -> list[str]:
        """
        步驟2: 解析與切塊。將單一維基百科頁面的長文本切成較短的語意段落。

        Args:
            page_title (str): 維基百科頁面標題 (來自 search 方法的返回結果)。
        
        Returns:
            list[str]: 乾淨的文字片段列表。
        """
        try:
            # <-- CHANGED: 使用 wikipedia.page() 獲取頁面內容
            page = wikipedia.page(page_title, auto_suggest=False, redirect=True)
            text_content = page.content
        except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError, KeyError) as e:
            # 處理找不到頁面或消歧義頁面的情況
            print(f" -> Skipping page '{page_title}' due to: {type(e).__name__}")
            return []
        
        # 使用 NLTK 將全文切成句子
        sentences = sent_tokenize(text_content)
        if not sentences:
            return []

        # 將句子分組成塊 (chunks)
        chunks = []
        stride = max(1, chunk_size - overlap) # 確保步長至少為 1
        for i in range(0, len(sentences), stride):
            chunk = " ".join(sentences[i:i + chunk_size])
            if len(chunk) > 50 and any(c.isalpha() for c in chunk):
                chunks.append(chunk)
        return chunks

    def rerank(self, query: str, chunks: list[str], top_k: int = 4) -> list[str]:
        """
        步驟3: 語意重排序。從所有文字片段中，找出與原始查詢最相關的 top_k 個。
        """
        if not chunks:
            return []

        query_vec = self.rerank_model.encode(query, normalize_embeddings=True)
        chunk_vecs = self.rerank_model.encode(chunks, normalize_embeddings=True)
        
        sims = chunk_vecs @ query_vec
        
        # 降序排列索引
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        return [chunks[i] for i in top_indices]

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        """
        將所有步驟串聯起來的完整檢索流程。
        """
        print(f"Retrieving external knowledge for query: '{query}'")
        
        page_titles = self.search(query)
        if not page_titles:
            print(" -> No relevant pages found on Wikipedia.")
            return []
        print(f" -> Found {len(page_titles)} potential pages: {page_titles}")
        
        all_chunks = []
        for title in page_titles:
            all_chunks.extend(self.parse(title))
        
        if not all_chunks:
            print(" -> Could not parse any meaningful content from the pages.")
            return []
        print(f" -> Parsed into {len(all_chunks)} text chunks.")

        ranked_chunks = self.rerank(query, all_chunks, top_k)
        print(f" -> Reranked and selected top {len(ranked_chunks)} chunks.")
        
        return ranked_chunks

# ==========================================================
# ==                      使用範例                       ==
# ==========================================================
if __name__ == '__main__':
    # 初始化檢索器
    ext_retriever = ExternalRetriever(lang='en')
    
    print("\n" + "="*50)
    
    # 測試一個查詢
    test_query = "a minimalist wooden dining chair"
    retrieved_contexts = ext_retriever.retrieve(test_query, top_k=3)
    
    print("\n" + "="*50)
    print(f"Final retrieved contexts for '{test_query}':\n")
    if retrieved_contexts:
        for i, context in enumerate(retrieved_contexts, 1):
            print(f"[{i}] {context}\n")
    else:
        print("No context was retrieved.")