import os
import json
from pathlib import Path
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 加载父目录中的 .env
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

class RAGService:
    """
    多模态 RAG 检索服务引擎。
    负责连接 ChromaDB 向量数据库，并利用 Gemini Embedding 模型执行语义搜索。
    支持动态 API Key 注入以实现多租户隔离。
    """
    def __init__(self):
        # 基础路径与配置初始化
        self.project_root = BASE_DIR
        self.db_path = self.project_root / "rag_service" / "chroma_db"
        self.collection_name = "multimodal_papers"
        self.embedding_model = "gemini-embedding-2-preview"

        # 初始化持久化客户端。如果是首次运行，会自动创建数据库目录。
        try:
            self.client_db = chromadb.PersistentClient(path=str(self.db_path))
            self.collection = self.client_db.get_or_create_collection(name=self.collection_name)
        except Exception as e:
            print(f"[RAG Init Error] 无法初始化 ChromaDB: {e}")
            self.collection = None
        
        # 默认回退客户端：从环境变量加载
        self._default_api_key = os.environ.get("GOOGLE_API_KEY")
        self._default_client = genai.Client(api_key=self._default_api_key) if self._default_api_key else None

    def _get_client(self, api_key: str = None) -> genai.Client:
        """
        获取指定用户的 GenAI 客户端或使用全局默认客户端。
        :param api_key: 任务专属的可选 API Key
        """
        if api_key:
            return genai.Client(api_key=api_key)
        return self._default_client

    def search(self, query_text: str, n_results: int = 3, api_key: str = None) -> list:
        """
        执行向量搜索。
        1. 使用 Gemini 生成 query 的嵌入向量 (Embedding)。
        2. 在 ChromaDB 中进行余弦相似度匹配。
        3. 还原元数据并解析关联的视觉资产路径。
        """
        if self.collection is None:
            print("[RAG Error] ChromaDB 未就绪，无法检索。")
            return []

        client = self._get_client(api_key)
        if not client:
            print("[RAG Error] 未提供有效的 API Key，跳过检索环节。")
            return []

        try:
            # 1. 生成查询向量 (Task Type: RETRIEVAL_QUERY)
            response = client.models.embed_content(
                model=self.embedding_model,
                contents=query_text,
                config=types.EmbedContentConfig(task_type='RETRIEVAL_QUERY')
            )
            query_vector = response.embeddings[0].values

            # 2. 从向量库检索 Top-K 结果
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results
            )

            # 3. 结果结构化封装
            formatted_results = []
            if not results or not results['ids'] or len(results['ids'][0]) == 0:
                return []

            for i in range(len(results['ids'][0])):
                meta = results['metadatas'][0][i]
                # 解析元数据中存储的图像路径 JSON
                img_paths = json.loads(meta.get("image_paths_json", "[]"))
                
                formatted_results.append({
                    "id": results['ids'][0][i],
                    "distance": results['distances'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": meta,
                    "images": img_paths
                })
            
            return formatted_results
        except Exception as e:
            print(f"[RAG Search Runtime Error] 检索过程异常: {e}")
            return []

# 全局单例引擎
rag_engine = RAGService()
