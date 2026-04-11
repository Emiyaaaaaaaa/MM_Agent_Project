import os
import json
import chromadb
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

import sys
import io

# 强制使用 UTF-8 输出，解决 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ================= 配置 =================
client_ai = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
PROJECT_ROOT = Path(r"c:\Users\a2231\Desktop\RAG")
DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "multimodal_papers"
QUERY_MODEL = "gemini-embedding-2-preview"

def audit_multimodal():
    print(f"{'='*60}")
    print(f" 多模态 Embedding 专项审计报告")
    print(f"{'='*60}")
    
    try:
        client_db = chromadb.PersistentClient(path=str(DB_PATH))
        collection = client_db.get_collection(name=COLLECTION_NAME)
        
        # 1. 统计总体情况
        all_data = collection.get(include=['metadatas', 'documents'])
        metadatas = all_data['metadatas']
        
        multimodal_chunks = []
        for i, meta in enumerate(metadatas):
            if meta.get("image_paths_json"):
                multimodal_chunks.append({
                    "id": all_data['ids'][i],
                    "images": json.loads(meta.get("image_paths_json")),
                    "content": all_data['documents'][i]
                })
        
        print(f"📊 总分块数: {len(metadatas)}")
        print(f"🖼️ 多模态（含图表）分块数: {len(multimodal_chunks)}")
        print(f"📈 多模态占比: {(len(multimodal_chunks)/len(metadatas)*100):.2f}%")
        
        if not multimodal_chunks:
            print("\n[!] 警告：未发现任何多模态分块。请检查 index_to_chroma.py 是否正确运行。")
            return

        # 2. 物理文件校验
        print(f"\n🔍 抽样物理文件校验 (Top 5):")
        verified_count = 0
        for mc in multimodal_chunks[:5]:
            exists_all = True
            for img_path in mc['images']:
                p = Path(img_path)
                if not p.exists():
                    print(f"  [X] 丢失: {p.name}")
                    exists_all = False
                else:
                    print(f"  [OK] 存在: {p.name} ({p.stat().st_size / 1024:.1f} KB)")
            if exists_all: verified_count += 1
            
        # 3. 视觉语义检索测试
        print(f"\n🎯 视觉语义检索测试 (Visual Semantic Search):")
        # 尝试通过“图表内容”进行检索
        test_queries = [
            "A diagram showing the cybersecurity policy classification framework.",
            "Visual data summary of cybercrime trends in different countries.",
            "Flowchart of the model implementation and data processing."
        ]
        
        for q in test_queries:
            print(f"\n   Q: '{q}'")
            response = client_ai.models.embed_content(
                model=QUERY_MODEL,
                contents=q,
                config=types.EmbedContentConfig(task_type='RETRIEVAL_QUERY')
            )
            v = response.embeddings[0].values
            res = collection.query(query_embeddings=[v], n_results=1)
            
            best_id = res['ids'][0][0]
            best_dist = res['distances'][0][0]
            best_meta = res['metadatas'][0][0]
            has_img = "YES" if best_meta.get("image_paths_json") else "NO"
            
            print(f"   > 命中 ID: {best_id} | 距离: {best_dist:.4f} | 含图表: {has_img}")
            if has_img == "YES":
                print(f"   > 图片路径: {best_meta.get('image_paths_json')[:80]}...")

    except Exception as e:
        print(f"审计过程出错: {e}")

if __name__ == "__main__":
    audit_multimodal()
