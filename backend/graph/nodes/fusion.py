import os
import json
from langchain_core.messages import HumanMessage
from google import genai
from google.genai import types
from backend.services.rag_engine import rag_engine
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress

class FusionNode:
    """
    全系统知识融合节点 (Knowledge Fusion Expert):
    1. 负责接收全工作流中的问题/上下文搜索需求。
    2. 多线程/异步调用局域 RAG 数据库 (获取 O 奖论文资源)。
    3. 调用原生 Gemini 的 Google Search Grounding 能力获取外网最新事实。
    4. 【动态智能加权】：由 LLM 视当前问题的侧重点（客观数据用 Web vs 学术方法用 RAG），进行知识的裁剪合并。
    """
    def __init__(self):
        self.system_prompt = (
            "你是一个『多模态数据与知识融合专家』(Knowledge Fusion Expert)。"
            "你会收到两部分知识：【本地 O 奖论文参考库】和【自带的 Google Search 实时公网检索库】。"
            "\n\n【动态加权融合法则】：\n"
            "1. **学术框架 vs 实时物理世界**：\n"
            "   - 若用户需求涉及真实的现实客观数据（如：巴拿马水库最近的流量、气象卫星数据观测、地缘政治新闻、宏观经济报表等时效性参数），**必须赋予【Google Search 网络结果】绝对的高权重**，并以网络数据填补 RAG 的空白。\n"
            "   - 若用户需求涉及数学建模体系构建、算法推导、特殊分布函数的应用、论文排版格式或往年评审偏好等学术向内容，**必须赋予【本地 O 奖论文参考库】绝对的高权重**，摒弃网络搜索中的业余“野路子”与低质博客代码。\n"
            "   - 若两者均有涉及，请用网络客观数据填充本地学术方程里的参数变量，达成完美融合。\n"
            "2. **去伪存真**：根据当前的提问场景，精准地滤除不相干或权重极低的矛盾噪点。\n"
            "3. **成果产出**：输出一份结构化、数据详实且兼具学术底蕴的《综合事实与学术背景简报》(Synthesized Background Context)。\n"
            "4. **安全与版权红线**：绝不可在最终的简报中暴露本地参考库的具体来源标记（如论文年份、具体参赛队号、O 奖等级等字眼），用“行业前沿常识”、“学术先例”等通用词汇打码代替。"
        )

    async def __call__(self, state: AgentState):
        shared_mem = state.get("shared_memory", {})
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-2.5-flash"
        
        messages = state.get("messages", [])
        if not messages:
            return {"context": "No query provided."}
            
        # 提取最后一条消息作为意图载体
        last_message = messages[-1].content
        
        await broadcast_progress("Retrieve(Fusion)", "正在启动双轨检索引擎 (Local RAG + Web Search)...", 20)
        
        # 1. 发兵本地 ChromaDB
        search_results = rag_engine.search(last_message, n_results=3, api_key=api_key)
        
        rag_context_parts = []
        for i, res in enumerate(search_results):
            rag_context_parts.append(f"--- 内部高价值学术片段 (RAG) {i+1} ---\n{res['content']}")
        rag_context = "\n".join(rag_context_parts) if rag_context_parts else "未在本地发现高度匹配的学术先例。"

        # 2. 调动 Gemini 原生搜索并执行 LLM “动态加权融合”
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")

        fusion_query = (
            f"【用户原始探索意图】：\n{last_message}\n\n"
            f"【系统为你准备的本地 RAG 学术参考（供加权判断使用）】：\n{rag_context}\n\n"
            f"请立刻使用外置 Google Search 工具打通公网客观数据，并根据【动态加权融合法则】输出一份终版综合简报。"
        )

        try:
            await broadcast_progress("Retrieve(Fusion)", f"检索完成，已触发 {model_id} 进行跨域知识的加权与去伪存真审查...", 60)
            
            # 使用原生 GenAI 客户端以启用 tools Grounding 特性
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_id,
                contents=fusion_query,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.2, # 融合任务需降低幻觉
                    tools=[{"google_search": {}}], # 原生开启谷歌搜索接地
                )
            )
            fused_context = response.text
            
        except Exception as e:
            # 安全降级策略：如果没开启搜索权限或网络阻断，回退到原始 RAG
            print(f"[Fusion Error] Web Grounding Failed, fallback to strict RAG. {e}")
            fused_context = f"[系统警告: Google Search Grounding 拒绝或失败，退回防断层模式]\n【本地学术参考回退】：\n{rag_context}"
            
        await broadcast_progress("Retrieve(Fusion)", "知识域交叉加权融合完毕，已挂载至工作流上下文。", 100)
        
        return {
            "context": fused_context,
            "current_stage": "Web Grounding & Dynamic Fusion Completed"
        }

fusion_node = FusionNode()
