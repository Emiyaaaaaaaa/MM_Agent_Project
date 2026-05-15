import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
import chromadb
from google import genai
from google.genai import types
from backend.services.serialization import extract_text_content

BASE_DIR = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger("mm-agent")

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
        self.embed_timeout_seconds = float(os.environ.get("MM_AGENT_RAG_EMBED_TIMEOUT_SECONDS", "45"))

        # 初始化持久化客户端。如果是首次运行，会自动创建数据库目录。
        try:
            self.client_db = chromadb.PersistentClient(path=str(self.db_path))
            self.collection = self.client_db.get_or_create_collection(name=self.collection_name)
        except Exception as e:
            print(f"[RAG Init Error] 无法初始化 ChromaDB: {e}")
            self.collection = None
        
    def _get_client(self, api_key: str = None) -> genai.Client:
        """
        获取指定用户的 GenAI 客户端或使用全局默认客户端。
        :param api_key: 任务专属的可选 API Key
        """
        if api_key:
            return genai.Client(api_key=api_key)
        return None

    def _embed_content_with_timeout(self, client: genai.Client, normalized_query: str):
        """避免 embed_content 在网络异常时阻塞数分钟，导致 WS 空闲超时。"""
        def _call():
            return client.models.embed_content(
                model=self.embedding_model,
                contents=normalized_query,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            fut = pool.submit(_call)
            return fut.result(timeout=self.embed_timeout_seconds)
        finally:
            # 超时后不得 wait=True，否则仍会卡在未结束的 HTTP 线程上
            pool.shutdown(wait=False)

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

        normalized_query = extract_text_content(query_text).strip()
        if not normalized_query:
            print("[RAG Error] 检索入参为空，跳过检索。")
            return []

        client = self._get_client(api_key)
        if not client:
            print("[RAG Error] 未提供有效的 API Key，跳过检索环节。")
            return []

        try:
            t0 = time.monotonic()
            logger.info(
                "RAG embed_content start model=%s query_len=%s timeout_s=%s",
                self.embedding_model,
                len(normalized_query),
                self.embed_timeout_seconds,
            )
            # 1. 生成查询向量 (Task Type: RETRIEVAL_QUERY)，带独立超时
            try:
                response = self._embed_content_with_timeout(client, normalized_query)
            except FuturesTimeout:
                logger.warning(
                    "RAG embed_content timeout after %ss (network or API slow); skip RAG hits",
                    self.embed_timeout_seconds,
                )
                return []
            query_vector = response.embeddings[0].values
            t1 = time.monotonic()
            logger.info("RAG embed_content done elapsed_ms=%s", int((t1 - t0) * 1000))

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
            
            logger.info("RAG chroma query done hits=%s elapsed_total_ms=%s", len(formatted_results), int((time.monotonic() - t0) * 1000))
            return formatted_results
        except Exception as e:
            logger.warning("RAG search failed: %s", e)
            return []

# 全局单例引擎
rag_engine = RAGService()
