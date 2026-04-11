import os
import json
import chromadb
from google import genai
from google.genai import types
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

# ================= 配置区 =================
client_ai = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

PROJECT_ROOT = Path(r"c:\Users\a2231\Desktop\RAG")
DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "multimodal_papers"
QUERY_MODEL = "gemini-embedding-2-preview"

# ================= 初始化 =================
client_db = chromadb.PersistentClient(path=str(DB_PATH))
collection = client_db.get_or_create_collection(name=COLLECTION_NAME)

def query_rag(query_text, n_results=3):
    """
    对向量库进行语义搜索
    """
    print(f"\n>>> 正在搜索: '{query_text}'")
    
    # 1. 将用户的查询语句转化为向量 (task_type 使用 RETRIEVAL_QUERY)
    response = client_ai.models.embed_content(
        model=QUERY_MODEL,
        contents=query_text,
        config=types.EmbedContentConfig(task_type='RETRIEVAL_QUERY')
    )
    
    query_vector = response.embeddings[0].values
    
    # 2. 在 ChromaDB 中进行检索
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    
    # 3. 打印结果
    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        content = results['documents'][0][i]
        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]
        
        print(f"\n--- 匹配结果 {i+1} (ID: {doc_id}, 距离: {distance:.4f}) ---")
        print(f"内容摘要: {content[:1000]}...")
        print(f"来源: {metadata.get('filename')} (年份: {metadata.get('year')})")
        
        # 展示图片路径
        img_paths_json = metadata.get("image_paths_json")
        if img_paths_json:
            paths = json.loads(img_paths_json)
            print(f"关联图片 ({len(paths)}张):")
            for p in paths:
                print(f"  [图片] {p}")

if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("错误：未找到 GOOGLE_API_KEY 环境变量！")
    else:
        test_query = "哪篇文章使用了折线图来表示敏感度分析的结果?"
        query_rag(test_query)
