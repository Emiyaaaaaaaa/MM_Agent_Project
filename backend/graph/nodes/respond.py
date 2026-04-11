import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from backend.graph.state import AgentState

class RespondNode:
    """
    专家咨询节点 (Respond)：负责基于检索到的 RAG 知识背景，为用户提供专业的建模咨询建议。
    它是 Agent 的“门面”，用于解释复杂的模型概念或提供竞赛指导。
    """
    def __init__(self):
        # 官方顾问指令
        self.system_prompt = (
            "你是一个专业的 MCM/ICM（美国大学生数学建模竞赛）官方高级顾问。"
            "你的任务是根据提供的参考资料和对话历史，提供专业、严谨且具有启发性的回答。\n\n"
            "回复原则：\n"
            "1. **事实优先**：优先基于参考资料（RAG Context）回答，确保学术准确性。\n"
            "2. **图表关联**：如果资料中包含图片路径，请在回复中解析其可能展示的模型趋势。\n"
            "3. **启发性引导**：若资料不足，请根据建模常识给出方法论层面的建议。\n\n"
            "【版权约束】\n"
            "- **信息脱敏**：严禁在回答中主动提及参考资料的具体队号、年份或获奖等级。引导用户关注方法论本身，而非特定论文的出处。"
        )

    async def __call__(self, state: AgentState):
        """执行专业回复生成逻辑"""
        shared_mem = state.get("shared_memory", {})
        messages = state["messages"]
        context = state.get("context", "（无参考资料）")
        feedback = state.get("human_feedback", "")
        
        # 动态模型实例化 (从任务配置加载)
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
        
        # 构造知识驱动提示词
        prompt_parts = [
            ("system", self.system_prompt),
            ("system", f"【RAG 参考背景】\n{context}")
        ]
        
        # 处理纠偏反馈
        if feedback:
            prompt_parts.append(("system", f"【人类修正建议】\n{feedback}"))
            
        prompt_parts.append(MessagesPlaceholder(variable_name="messages"))
        
        prompt = ChatPromptTemplate.from_messages(prompt_parts)
        chain = prompt | llm
        response = await chain.ainvoke({"messages": messages})
        
        # 存入草稿以供审批流展示
        return {
            "messages": [response],
            "status": "PENDING", # 进入 HITL 流程
            "draft": response.content,
            "current_stage": "Consultation Response Prepared"
        }

# 单例封装与 LangGraph 包装函数
respond_node_instance = RespondNode()

async def respond_node(state: AgentState):
    """LangGraph 调度包装"""
    print("[Node: Respond] 正在基于知识库生成专业解答...")
    return await respond_node_instance(state)
