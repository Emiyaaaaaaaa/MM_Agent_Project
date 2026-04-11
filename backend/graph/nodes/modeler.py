import os
import json
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress

class ModelerNode:
    """
    数学建模专家节点 (Modeler)：负责将定性分析转化为定量的数学表述。
    该节点专注于算法选型、符号定义以及严格的数学公式推导。
    """
    def __init__(self):
        # 建模节点建议使用更强的推理能力
        self.system_prompt = (
            "你是一个顶级的 MCM/ICM 建模专家，擅长从定性分析转化为定量的数学模型。"
            "你的任务是根据审题报告 (Analysis Report)，深入推导具体的数学公式并选择最优算法。"
            "\n输出要求（使用 LaTeX 格式）：\n"
            "1. **算法选型与依据**：详细说明为什么选择该算法（如：时间序列、评价模型、神经网络），分析其对本题的适配性。\n"
            "2. **符号系统建立**：列出模型中涉及的所有数学符号及其物理/实际含义（表格形式）。\n"
            "3. **数学公式推导**：给出详细的推导过程，包括目标函数 (Objective Function)、约束条件 (Constraints) 或微分方程组 (ODEs)。\n"
            "4. **求解路线图**：描述具体的计算方案，为程序员节点 (Coder Node) 提供伪代码或数值求解思路。"
            "\n\n【重要指令】\n"
            "- **模仿优秀论文**：参考 RAG 资料中获奖论文的公式推导逻辑严密性和变量抽象能力。确保你的模型不仅能工作，而且具备学术美感与逻辑自洽性。\n"
            "- **版权屏蔽约束**：严禁在生成内容中出现具体参考论文的队号、年份或 O 奖等标识。使用‘学术通用模型’或‘经典建模处理’等中性词汇代称。"
        )

    async def __call__(self, state: AgentState):
        """执行数学建模与公式推导逻辑"""
        shared_mem = state.get("shared_memory", {})
        analysis_report = shared_mem.get("analysis_report", "（暂无审题报告）")
        feedback = state.get("human_feedback", "")
        
        # 1. 广播进度
        await broadcast_progress("Modeling", "正在基于审题报告提取关键变量与物理关系...", 20)
        
        # 2. 动态模型实例化 (从任务配置加载)
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-2.5-flash"
        
        if not api_key:
            print("[Warning] No API key found in shared_memory. Falling back to environment variable.")
            api_key = os.environ.get("GOOGLE_API_KEY")

        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0.2, # 稍高一点以提供更多建模视角
            google_api_key=api_key
        )
        
        # 构造上下文
        prompt_parts = [
            ("system", self.system_prompt),
            ("system", f"【深度分析背景】\n{analysis_report[:2000]}...")
        ]
        
        if feedback:
            prompt_parts.append(("system", f"【用户最新指导建议】\n{feedback}"))
            
        alignment_record = shared_mem.get("alignment_record", {})
        if alignment_record:
            align_prompt = (
                f"【全局防漂移强制对齐约束 (CRITICAL)】\n"
                f"1. 强制使用以下全局符号体系，严禁擅自造词改名：\n{json.dumps(alignment_record.get('symbols', []), ensure_ascii=False, indent=2)}\n"
                f"2. 严禁违背以下全局预设简化假设：\n{json.dumps(alignment_record.get('assumptions', []), ensure_ascii=False, indent=2)}\n"
                f"3. 你的推导必须紧密朝向以下核心优化目标：\n{json.dumps(alignment_record.get('objectives', []), ensure_ascii=False, indent=2)}"
            )
            prompt_parts.append(("system", align_prompt))
            
        prompt_parts.append(MessagesPlaceholder(variable_name="messages"))
        
        # 3. 广播进度
        await broadcast_progress("Modeling", f"正在使用 {model_id} 进行 LaTeX 公式推导与算法选型...", 60)
        
        prompt = ChatPromptTemplate.from_messages(prompt_parts)
        chain = prompt | llm
        
        response = await chain.ainvoke({"messages": state.get("messages", [])})
        
        # 4. 广播进度
        await broadcast_progress("Modeling", "数学模型推导完毕，已生成 LaTeX 配套文档。", 100)
        
        new_memory = shared_mem.copy()
        new_memory["mathematical_model"] = response.content
        
        return {
            "messages": [response],
            "shared_memory": new_memory,
            "status": "PENDING",
            "draft": response.content,
            "current_stage": "Mathematical Modeling Completed"
        }

# 单例提供
modeler_node = ModelerNode()
