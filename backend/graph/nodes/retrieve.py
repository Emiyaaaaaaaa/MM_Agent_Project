import json
from langchain_core.messages import HumanMessage
from backend.services.rag_engine import rag_engine
from backend.graph.state import AgentState
from backend.services.serialization import extract_text_content, make_stage

def retrieve_node(state: AgentState):
    """
    知识检索节点 (Retrieve)：负责在 ChromaDB 向量数据库中执行精准搜索。
    该节点会将检索到的 Top-K 文档片段（含年份、来源、图片关联）注入 context 字段。
    """
    shared_mem = state.get("shared_memory", {})
    # 获取动态 API Key
    api_key = shared_mem.get("api_key")
    
    # 提取最后一条提问作为 RAG 查询词
    messages = state.get("messages", [])
    last_message = extract_text_content(messages[-1]) if messages else ""
    
    # 执行 RAG 检索 (传入动态 api_key)
    search_results = rag_engine.search(last_message, n_results=5, api_key=api_key)
    
    # 格式化检索结果流
    context_parts = []
    for i, res in enumerate(search_results):
        meta = res.get('metadata', {})
        source_info = f"[论文来源: {meta.get('filename', 'Unknown')}, 年份: {meta.get('year', 'N/A')}]"
        context_parts.append(f"--- 知识参考 {i+1} {source_info} ---\n{res['content']}")
        
        # 记录关联的视觉素材路径
        if res['images']:
            img_list = ", ".join(res['images'])
            context_parts.append(f"(视觉参考路径: {img_list})")

    # 封装备选 Context 供回答节点 (Respond) 调用
    full_context = "\n\n".join(context_parts)
    
    print(f"[Node: Retrieve] 向量检索成功，捕获 {len(search_results)} 个知识分块。")
    
    return {
        "context": full_context,
        "stage": make_stage("retrieve_completed", "Knowledge Retrieval Completed"),
    }
