import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress
from backend.services.serialization import extract_text_content, json_safe, ensure_blocks, make_stage, messages_for_llm

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
            "- **文风强制要求**：语言和文字风格要符合各个获奖论文严谨冷静的范式，不要使用对表达论文内容来说不必要的比喻以及其他修辞，也不要使用过于抽象的合成词语或者自造词语"
            "- **版权保护约束**：在生成报告时，严禁主动提及参考资料的具体队号、年份或获奖等级（如：‘参考自25年X队O奖论文’）。应使用‘以往优秀论文实践’或‘学术惯例’等通用措辞代称，保护数据隐私。"
            "- **Fusion 强约束**：若输入包含 Fusion 指南，必须在输出中显式体现其风格与结构要点，不可忽略。"
        )

    async def __call__(self, state: AgentState):
        """执行全文本审题分析逻辑"""
        shared_mem = state.get("shared_memory", {})
        doc_content = extract_text_content(shared_mem.get("raw_document_content", "")).strip()
        if not doc_content:
            for msg in reversed(state.get("messages", []) or []):
                msg_type = getattr(msg, "type", None)
                if msg_type is None and isinstance(msg, dict):
                    msg_type = msg.get("role")
                if msg_type in {"human", "user"}:
                    doc_content = extract_text_content(msg).strip()
                    if doc_content:
                        break
        if not doc_content:
            doc_content = "（暂无题目原文，请上传文件）"
        feedback = state.get("human_feedback", "")
        
        # 1. 广播进度：开始重读文档
        await broadcast_progress("Analysis", "正在扫描文档全文并提取核心赛题信息...", 10)
        
        # 题目原文与 RAG 资料转义
        safe_doc = doc_content[:3000].replace("{", "{{").replace("}", "}}")
        prompt_parts = [
            ("system", self.system_prompt),
            ("system", f"【原始题目原文】\n{safe_doc}...")
        ]
        
        # 如果有 RAG 背景信息，将其显式作为参考资料注入
        context = state.get("context")
        if context:
            safe_context = extract_text_content(context)[:1800].replace("{", "{{").replace("}", "}}")
            prompt_parts.append(("system", f"【RAG 检索参考资料（严禁直接引用队号/年份）：】\n{safe_context}"))
        fusion_guidance = extract_text_content(shared_mem.get("fusion_guidance", "")) or extract_text_content(context)
        if fusion_guidance:
            safe_fusion = fusion_guidance[:1800].replace("{", "{{").replace("}", "}}")
            prompt_parts.append(
                (
                    "system",
                    "【Fusion 强约束指南（CRITICAL）】\n"
                    f"{safe_fusion}\n\n"
                    "输出要求：请在报告末尾增加“Fusion映射清单”，列出至少 3 条你实际采用的融合要点。"
                )
            )

        messages = messages_for_llm(
            state.get("messages", []) or [],
            max_messages=4,
            max_chars_per_message=1200,
        )
        if not messages:
            messages = [HumanMessage(content="请开始审题分析。")]
        
        if feedback:
            prompt_parts.append(("system", f"【修正建议】\n{feedback}"))
            
        prompt_parts.append(MessagesPlaceholder(variable_name="messages"))
        
        # 1.5. 动态模型实例化 (从任务配置加载)
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-3.1-flash-lite"
        
        if not api_key:
            return {
                "status": "REJECTED",
                "human_feedback": "缺少任务 API Key，Analysis 节点无法执行。",
                "stage": make_stage("analysis_failed_missing_api_key", "Analysis Failed: Missing Task API Key"),
            }

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
        analysis_text = extract_text_content(response)
        
        # 3. 广播进度：完成分析报告
        await broadcast_progress("Analysis", "审题报告构建完毕，正在固化分析成果...", 90)
        
        new_memory = shared_mem.copy()
        new_memory["analysis_report"] = analysis_text
        new_memory["analysis"] = analysis_text
        new_memory["analysis_raw_content"] = json_safe(getattr(response, "content", None))
        
        await broadcast_progress("Analysis", "分析已就绪，已进入审批环节。", 100)
        
        return {
            "messages": [{"role": "ai", "content": ensure_blocks(analysis_text)}],
            "shared_memory": new_memory,
            "status": "APPROVED",
            "draft": ensure_blocks(analysis_text),
            "stage": make_stage("analysis_completed", "Problem Analysis Completed"),
        }

analyzer_node = AnalyzerNode()
