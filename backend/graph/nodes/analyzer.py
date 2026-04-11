import os
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress

class AnalyzerNode:
    """
    审题分析专家节点 (Analyzer)：
    负责对原始赛题进行深度解构，提取核心问题、建立初步假设并定义关键变量。
    集成 RAG 增强逻辑：自动吸收往年优秀论文的分析套路。
    版权保护：确保分析过程中不泄露参考源的队号与年份。
    """
    def __init__(self):
        self.system_prompt = (
            "你是一个资深的 MCM/ICM 建模竞赛『审题分析专家』。"
            "你的任务是仔细阅读题目内容，并产出包含以下模块的结构化分析报告：\n\n"
            "1. **核心问题提取**：精炼概括待解决的子任务。\n"
            "2. **模型合理假设**：提出基于物理或逻辑的简化假设及其理由。\n"
            "3. **关键变量定义**：明确自变量、因变量及主要参数的物理/数学含义。\n"
            "4. **初步模型建议**：指出适用该问题的数学工具（如微分方程、博弈论、模拟退火等）。"
            "\n\n【核心指令】\n"
            "- **RAG 参考增强**：对话上下文中可能包含由 RAG 检索到的往年优秀论文片段，请深度参考这些资料的分析逻辑与变量建模风格，使其更具学术深度。\n"
            "- **版权保护约束**：在生成报告时，严禁主动提及参考资料的具体队号、年份或获奖等级（如：‘参考自25年X队O奖论文’）。应使用‘以往优秀论文实践’或‘学术惯例’等通用措辞代称，保护数据隐私。"
        )

    async def __call__(self, state: AgentState):
        """执行全文本审题分析逻辑"""
        shared_mem = state.get("shared_memory", {})
        doc_content = shared_mem.get("raw_document_content", "（暂无题目原文，请上传文件）")
        feedback = state.get("human_feedback", "")
        
        # 1. 广播进度：开始重读文档
        await broadcast_progress("Analysis", "正在扫描文档全文并提取核心赛题信息...", 10)
        
        prompt_parts = [
            ("system", self.system_prompt),
            ("system", f"【原始题目原文】\n{doc_content[:3000]}...")
        ]
        
        # 如果有 RAG 背景信息，将其显式作为参考资料注入
        context = state.get("context")
        if context:
            prompt_parts.append(("system", f"【RAG 检索参考资料（严禁直接引用队号/年份）：】\n{context}"))

        messages = state.get("messages", [])
        if not messages:
            messages = [HumanMessage(content="请开始审题分析。")]
        
        if feedback:
            prompt_parts.append(("system", f"【修正建议】\n{feedback}"))
            
        prompt_parts.append(MessagesPlaceholder(variable_name="messages"))
        
        # 1.5. 动态模型实例化 (从任务配置加载)
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-2.5-flash"
        
        if not api_key:
            print("[Warning] No API key found in shared_memory. Falling back to environment variable, but this is discouraged.")
            api_key = os.environ.get("GOOGLE_API_KEY")

        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0.1,
            google_api_key=api_key
        )
        
        # 2. 广播进度：进入深度推理
        await broadcast_progress("Analysis", f"正在使用 {model_id} 构建数学假设与变量映射框架...", 40)
        
        prompt = ChatPromptTemplate.from_messages(prompt_parts)
        chain = prompt | llm
        
        response = await chain.ainvoke({"messages": messages})
        
        # 3. 广播进度：完成分析报告
        await broadcast_progress("Analysis", "审题报告构建完毕，正在固化分析成果...", 90)
        
        new_memory = shared_mem.copy()
        new_memory["analysis_report"] = response.content
        new_memory["analysis"] = response.content 
        
        await broadcast_progress("Analysis", "分析已就绪，已进入审批环节。", 100)
        
        return {
            "messages": [response],
            "shared_memory": new_memory,
            "status": "PENDING",
            "draft": response.content,
            "current_stage": "Problem Analysis Completed"
        }

analyzer_node = AnalyzerNode()
