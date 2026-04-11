import os
import json
import time
from pathlib import Path
from tqdm import tqdm
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

# ================= 配置区 =================
# 初始化 Google GenAI 客户端
client_ai = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# 路径配置
PROJECT_ROOT = Path(r"c:\Users\a2231\Desktop\RAG")
CHUNKS_DIR = PROJECT_ROOT / "RAG_Chunks_Gemini"
DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "multimodal_papers"
EMBEDDING_MODEL = "gemini-embedding-2-preview" # 新 SDK 中通常不带 models/ 前缀，或者会自动处理

# ================= 初始化 ChromaDB =================
client_db = chromadb.PersistentClient(path=str(DB_PATH))
collection = client_db.get_or_create_collection(name=COLLECTION_NAME)

def get_multimodal_embedding(text, image_paths):
    """
    使用新版 google-genai SDK 获取多模态向量
    """
    # 构造内容列表
    contents = [text]
    for img_path in image_paths:
        p = Path(img_path)
        if p.exists():
            img_bytes = p.read_bytes()
            mime_type = "image/jpeg" if p.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
            # 使用新版 SDK 的 Part 构造方式
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
    
    # 指数退避重试逻辑
    for attempt in range(5):
        try:
            response = client_ai.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(task_type='RETRIEVAL_DOCUMENT')
            )
            
            # 获取向量并展平
            embedding = response.embeddings[0].values
            return embedding
        except Exception as e:
            if "429" in str(e) or "ResourceExhausted" in str(e):
                wait = 2 ** attempt + 1
                print(f"\n[!] 速率限制。等待 {wait} 秒后重试...")
                time.sleep(wait)
            else:
                print(f"\n[!] Embedding 报错: {e}")
                return None
    return None

def run_indexing():
    json_files = list(CHUNKS_DIR.rglob("*_chunks.json"))
    print(f">>> 准备处理 {len(json_files)} 个 JSON 文件...")

    for file_path in tqdm(json_files, desc="总进度"):
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
            
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            images = chunk.get("images", [])
            raw_metadata = chunk.get("metadata", {})
            
            valid_img_paths = [img["local_path"] for img in images if Path(img["local_path"]).exists()]
            vector = get_multimodal_embedding(content, valid_img_paths)
            if not vector:
                continue
            
            processed_metadata = {}
            for k, v in raw_metadata.items():
                if isinstance(v, list):
                    processed_metadata[k] = ",".join(map(str, v))
                else:
                    processed_metadata[k] = v
            
            if valid_img_paths:
                processed_metadata["image_paths_json"] = json.dumps(valid_img_paths)
            
            doc_id = f"{file_path.stem}_{i}"
            collection.upsert(
                ids=[doc_id],
                embeddings=[vector],
                metadatas=[processed_metadata],
                documents=[content]
            )

if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("错误：未找到 GOOGLE_API_KEY 环境变量！")
    else:
        run_indexing()
        print("\n>>> 索引任务圆满完成！数据已持久化至:", DB_PATH)
