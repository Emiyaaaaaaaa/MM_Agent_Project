import os
import json
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress

class WriterNode:
    """
    学术写作专家节点 (Writer)：
    整合建模、仿真及审查的所有产出，生成符合 MCM/ICM 规范的学术论文。
    - 具备续写保护功能：防止长论文生成时因 Token 溢出导致的上下文断裂。
    - RAG 对标：模仿往年优秀论文的专业文风与图例说明。
    """
    def __init__(self):
        self.system_prompt = (
            "你是一个 MCM/ICM 建模竞赛特刊主编，擅长撰写符合学术规范的 O 奖论文。"
            "你的任务是根据审题、建模和代码逻辑，生成 LaTeX/Markdown 论文正文。\n\n"
            "写作准则：\n"
            "1. **学术结构**：必须包含 Abstract, Introduction, Methods, Results, Conclusion。\n"
            "2. **图文嵌入**：将 Latex 公式、代码核心逻辑及图表结论自然融合。\n"
            "3. **续写感知**：若指令提示为‘续写’，请紧接前文逻辑，严禁重复输出已完成章节。\n\n"
            "【大师级指令】\n"
            "- **对标 O 奖文风**：参考 RAG 背景中优秀论文的措辞风格、摘要深度以及图表说明的专业度。确保生成的论文看起来像是出自经验丰富的建模团队之手。\n"
            "- **版权屏蔽约束**：严禁在论文正文中主动披露参考资料的具体队号、年份或获奖等级（如：‘参考自25年X队’）。应使用‘以往优秀实践’或‘学术惯例’等通用措辞代称。"
        )

    async def __call__(self, state: AgentState):
        """执行学术论文生成 (支持 Token 截断自动保护)"""
        shared_mem = state.get("shared_memory", {})
        existing_draft = shared_mem.get("paper_draft", "")
        
        # 1. 广播进度：开始聚合上下文
        await broadcast_progress("Writer", "正在汇总建模背景、公式推导与仿真数据...", 15)
        
        context = (
            f"【1. 背景分析】\n{shared_mem.get('analysis_report', '')[:500]}...\n\n"
            f"【2. 数学推导】\n{shared_mem.get('mathematical_model', '')[:1000]}...\n\n"
            f"【3. 运行审计】\n{shared_mem.get('review_report', '')[:500]}...\n\n"
        )
        
        continuation_hint = ""
        if existing_draft:
            last_snippet = existing_draft[-500:]
            continuation_hint = f"\n\n【续写入口】\n前序输出已截断，以下是结尾部分：\n...{last_snippet}\n请无视重复，直接从缺失处继续完成。"
            await broadcast_progress("Writer", "检测到既有草稿，正在执行断点续写衔接...", 30)

        prompt_parts = [
            ("system", self.system_prompt),
            ("system", f"建模全背景：\n{context}")
        ]
        
        alignment_record = shared_mem.get("alignment_record", {})
        if alignment_record:
            align_prompt = (
                f"【全局防漂移强制对齐约束 (CRITICAL)】\n"
                f"在撰写公式与代码解读时：\n"
                f"1. 强制使用以下全局符号体系，保证整篇论文变量名绝对不割裂：\n{json.dumps(alignment_record.get('symbols', []), ensure_ascii=False, indent=2)}\n"
                f"2. 在前提假设段落必须涵括以下全局预设：\n{json.dumps(alignment_record.get('assumptions', []), ensure_ascii=False, indent=2)}\n"
                f"3. 摘要和结论中须突出回应以下核心优化目标：\n{json.dumps(alignment_record.get('objectives', []), ensure_ascii=False, indent=2)}"
            )
            prompt_parts.append(("system", align_prompt))
            
        prompt_parts.extend([
            MessagesPlaceholder(variable_name="messages"),
            ("system", continuation_hint if continuation_hint else "请开始撰写论文全文。"),
            ("system", "提示：若篇幅触及限制请自动停止，系统将记录草稿并引导续写。")
        ])
        
        prompt = ChatPromptTemplate.from_messages(prompt_parts)
        
        # 1.5. 动态模型实例化 (从任务配置加载)
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-2.5-flash"
        
        if not api_key:
            print("[Warning] No API key found in shared_memory. Falling back to environment variable.")
            api_key = os.environ.get("GOOGLE_API_KEY")

        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0.7,
            google_api_key=api_key
        )
        
        # 将 RAG context 注入写作背景中 (如果存在)
        rag_context = state.get("context", "")
        if rag_context:
            context = f"【RAG 优秀论文参考标准（严禁提及队号/年份）】\n{rag_context[:3000]}\n\n" + context
        
        # 2. 广播进度：进入生成阶段
        await broadcast_progress("Writer", f"正在使用 {model_id} 进行学术创作 (Token 流已开启)...", 50)
        
        chain = prompt | llm
        response = await chain.ainvoke({"messages": state.get("messages", [])})
        
        finish_reason = response.response_metadata.get("finish_reason", "")
        new_draft = existing_draft + response.content
        new_memory = shared_mem.copy()
        new_memory["paper_draft"] = new_draft
        
        if finish_reason in ["length", "MAX_TOKENS"]:
            status = "PENDING"
            await broadcast_progress("Writer", "检测到内容超限，已保存阶段性草稿。请回复‘继续’。", 100)
        else:
            status = "APPROVED" 
            await broadcast_progress("Writer", "学术论文全篇初稿生成完毕。", 100)
            
        return {
            "messages": [response],
            "shared_memory": new_memory,
            "status": status,
            "draft": response.content,
            "current_stage": "Paper Finishing" if status == "APPROVED" else "Paper Truncated"
        }

writer_node = WriterNode()
