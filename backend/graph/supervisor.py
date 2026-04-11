import os
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from backend.graph.state import AgentState

# 加载环境变量
load_dotenv()

# 定义意图识别的输出结构
class RouteResponse(BaseModel):
    """决定下一步要跳转的环节"""
    next: Literal["Fusion", "Respond", "Read", "Coding", "Analysis", "Modeling", "Review", "Writing", "Export"] = Field(
        description="选择下一步：'Analysis' (审题分析)；'Modeling' (数学推导)；'Coding' (Python实现)；'Review' (结果审查)；'Writing' (全篇写作)；'Export' (导出本地文件)；'Read' (读取)；'Fusion' (双轨知识融合检索)；'Respond' (直接回答)。"
    )

# 初始化 Supervisor 节点
class SupervisorNode:
    """
    智能主管节点 (Supervisor)：Agent 的大脑。
    负责意图识别、工作流调度以及根据当前建模进度自主决策下一步路径。
    """
    def __init__(self):
        # 高级调度指令：确立“意图优先 > 进度辅助”的动态路由准则
        self.system_prompt = (
            "你是一个 MCM/ICM 建模竞赛智能体的主管 (Supervisor)。"
            "你的核心原则是：**尊重用户意图 > 遵循建模流程**。\n\n"
            "【路由决策优先级】\n"
            "1. **意图锚定 (Highest)**：如果用户明确要求执行某项任务（例如“帮我编程”、“写论文”），请直接跳至该节点。\n"
            "2. **情报探测 (Critical)**：如果用户提出有关客观现实数据（如河流流量、国家经济参数）或寻求以往的建模经验思路，**务必首先指向 'Fusion' 节点**以触发公网与本地 RAG 数据库的加权知识检索。\n"
            "3. **进度补偿 (Secondary)**：如果用户指令模糊（如“下一步做什么？”、“继续”），请参考以下逻辑链的下一环。\n\n"
            "【进度感知指南】\n"
            "- Read (如果刚读完题) -> Analysis\n"
            "- Analysis (已完成) -> Modeling (建议)\n"
            "- Modeling (已完成) -> Coding (建议)\n"
            "- Coding (已完成) -> Review (自动建议)"
        )

    def __call__(self, state: AgentState):
        """意图识别与动态决策逻辑实现"""
        shared_mem = state.get("shared_memory", {})
        
        # 处理文件读取的特殊冷启动逻辑
        if shared_mem.get("input_file_path") and not shared_mem.get("raw_document_content"):
            return {"next": "Read"}

        messages = state["messages"]
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.type == "human":
                last_user_msg = msg.content
                break
        
        # 进度上下文提取
        completed_stages = [k for k in shared_mem.keys() if k.endswith("_report") or k in ["mathematical_model", "generated_code", "paper_draft"]]
        progress_info = f"【当前进度】: {completed_stages}"

        # 构造增强提示词
        content_preview = shared_mem.get("raw_document_content", "")[:3000]
        context_context = f"{progress_info}\n【文档预览】: {content_preview[:500]}..."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("system", context_context),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "请决策下一步跳转的目标节点。")
        ])
        
        # 动态模型实例化 (从任务配置加载)
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-2.5-flash"
        
        if not api_key:
            print("[Warning] No API key found in shared_memory for Supervisor. Falling back to environment variable.")
            api_key = os.environ.get("GOOGLE_API_KEY")

        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0,
            google_api_key=api_key
        )
        router = llm.with_structured_output(RouteResponse)
        
        # 执行路由决策
        chain = prompt | router
        result = chain.invoke({"messages": messages})
        
        print(f"[Node: Supervisor] 决策流转至 -> {result.next}")
        
        return {
            "next": result.next,
            "status": "PENDING", # 重置状态，激活下一节点的 HITL 中断
            "current_stage": f"Routing to {result.next} Phase"
        }

# 单例
supervisor_node = SupervisorNode()
