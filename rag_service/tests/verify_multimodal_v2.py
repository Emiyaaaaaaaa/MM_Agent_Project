import os
import json
import time
from pathlib import Path
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ================= 配置区 =================
client_ai = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
PROJECT_ROOT = Path(r"c:\Users\a2231\Desktop\RAG")
DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "multimodal_papers"
QUERY_MODEL = "gemini-embedding-2-preview"

# 定义测试用的特定图片路径 (2025 F 题，政策分类图)
TEST_IMAGE_PATH = PROJECT_ROOT / "RAG_Chunks_Gemini" / "F" / "2025" / "F" / "2508764" / "2508764" / "_page_12_Figure_4.jpeg"

# 初始化 ChromaDB
client_db = chromadb.PersistentClient(path=str(DB_PATH))
collection = client_db.get_collection(name=COLLECTION_NAME)

def get_embedding(text=None, image_path=None):
    """
    生成多模态查询向量
    """
    contents = []
    if text:
        contents.append(text)
    if image_path and Path(image_path).exists():
        p = Path(image_path)
        img_bytes = p.read_bytes()
        mime_type = "image/jpeg" if p.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
    
    if not contents:
        return None

    response = client_ai.models.embed_content(
        model=QUERY_MODEL,
        contents=contents,
        config=types.EmbedContentConfig(task_type='RETRIEVAL_QUERY')
    )
    return response.embeddings[0].values

def run_multimodal_showcase():
    print(f"{'='*80}")
    print(f" 多模态检索能力深度验证 (V2) ")
    print(f"{'='*80}")
    
    if not TEST_IMAGE_PATH.exists():
        print(f"[!] 错误: 找不到测试图片 {TEST_IMAGE_PATH}")
        return

    # 测试方案
    scenarios = [
        {
            "name": "1. 纯文本检索 (关键词匹配)",
            "text": "Cybersecurity policy classification framework Figure 10",
            "image": None
        },
        {
            "name": "2. 纯图片检索 (以图搜文)",
            "text": None,
            "image": TEST_IMAGE_PATH
        },
        {
            "name": "3. 图文融合检索 (模糊语义 + 视觉特征)",
            "text": "Explain the policy distribution shown in this diagram.",
            "image": TEST_IMAGE_PATH
        }
    ]

    for sc in scenarios:
        print(f"\n>>> 场景: {sc['name']}")
        if sc['text']: print(f"    文本输入: '{sc['text']}'")
        if sc['image']: print(f"    图片输入: {Path(sc['image']).name}")
        
        # 获取查询向量
        query_v = get_embedding(text=sc['text'], image_path=sc['image'])
        
        # 检索 Top 1
        results = collection.query(query_embeddings=[query_v], n_results=1)
        
        dist = results['distances'][0][0]
        doc = results['documents'][0][0]
        meta = results['metadatas'][0][0]
        
        print(f"    [结果] 匹配 ID: {results['ids'][0][0]}")
        print(f"    [结果] 相似度距离: {dist:.4f}")
        print(f"    [内容摘要]: {doc[:200]}...")
        print(f"    [命中来源]: {meta.get('filename')} (Question {meta.get('question')})")
        
        if dist < 0.5:
            print("    ✅ 验证结论：检索结果高度相关！")
        else:
            print("    ⚠️ 验证结论：相关度较低，请检查库数据。")

if __name__ == "__main__":
    run_multimodal_showcase()
