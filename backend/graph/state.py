from typing import Annotated, Sequence, TypedDict, Literal, Dict, Any, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class StageState(TypedDict):
    id: str
    label: str
    meta: Dict[str, Any]


class AgentState(TypedDict):
    """
    Agent 全局状态定义。
    它是所有节点之间共享的‘黑板’，记录了对话历史、中间产物、审批状态以及任务进度。
    """
    # LangGraph 核心：对话消息序列 (支持自动合并新消息)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # 动态路由标记：由 Supervisor 决定下一个执行节点
    next: str
    
    # 知识检索缓存：存储从 RAG 引擎获取的相关背景资料
    context: str
    
    # HITL (人机协作) 审批字段
    status: Literal["PENDING", "APPROVED", "REJECTED"] # 执行状态：待定/通过/驳回
    human_feedback: str                             # 用户反馈/修正建议
    draft: List[Dict[str, Any]]                     # 节点中间产物（结构化内容块）
    
    # 全局共享记忆体：持久化存储各阶段的核心产物 (Report, Python Code, Latex, etc.)
    shared_memory: Dict[str, Any]
    
    # 阶段描述符：结构化阶段信息
    stage: StageState
