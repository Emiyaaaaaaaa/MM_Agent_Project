from langgraph.graph import StateGraph, START, END
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from backend.graph.state import AgentState
from backend.graph.supervisor import supervisor_node
from backend.graph.nodes.fusion import fusion_node
from backend.graph.nodes.respond import respond_node
from backend.graph.nodes.reader import reader_node
from backend.graph.nodes.coder import coder_node
from backend.graph.nodes.analyzer import analyzer_node
from backend.graph.nodes.modeler import modeler_node
from backend.graph.nodes.reviewer import reviewer_node
from backend.graph.nodes.writer import writer_node
from backend.graph.nodes.exporter import exporter_node
from backend.graph.nodes.alignment import alignment_node

def build_graph():
    """
    状态图编排逻辑：构建 MCM/ICM 建模竞赛全生命周期工作流。
    该图集成了动态路由、自动流水线以及 HITL (人机审批) 机制。
    """
    # 1. 定义有状态图
    workflow = StateGraph(AgentState)
    
    # 2. 注入核心功能节点
    workflow.add_node("Reader", reader_node)       # 文件解析
    workflow.add_node("Supervisor", supervisor_node) # 智能意图路由
    workflow.add_node("Fusion", fusion_node)       # 动态加权双轨知识检索
    workflow.add_node("Respond", respond_node)     # 用户直接对话
    workflow.add_node("Coder", coder_node)         # 建模仿真与可视化
    workflow.add_node("Analysis", analyzer_node)   # 赛题深度分析
    workflow.add_node("Modeling", modeler_node)     # 数学公式推导
    workflow.add_node("Review", reviewer_node)     # 多模态逻辑审查
    workflow.add_node("Writing", writer_node)       # 学术文稿写作
    workflow.add_node("Export", exporter_node)     # 物理资产打包导出
    workflow.add_node("Alignment", alignment_node) # 记忆防漂移中间件
    
    # 3. 配置静态连线
    workflow.add_edge(START, "Supervisor")          # 起始即主管
    workflow.add_edge("Reader", "Supervisor")        # 读取后回流
    workflow.add_edge("Fusion", "Respond")           # 融合情报探测后由后端梳理回答
    
    # 4. 主管节点的动态决策路由
    def route_from_supervisor(state: AgentState):
        """基于 LLM 决策的目标节点跳转"""
        return state["next"]
    
    workflow.add_conditional_edges(
        "Supervisor",
        route_from_supervisor,
        {
            "Fusion": "Fusion",
            "Respond": "Respond",
            "Read": "Reader",
            "Coding": "Coder",
            "Analysis": "Analysis",
            "Modeling": "Modeling",
            "Review": "Review",
            "Writing": "Writing",
            "Export": "Export"
        }
    )
    
    # 5. HITL 审批流路由架构
    def route_after_node(state: AgentState):
        """
        审批流转规则：
        - APPROVED (通过): 任务完结，回流至 Supervisor 寻求下一阶段自动建议。
        - REJECTED (驳回): 原节点原地修正 (Loopback)。
        - PENDING (等待): 触发中断，停止执行等待用户输入。
        """
        status = state.get("status", "PENDING")
        if status == "APPROVED":
            return "Supervisor"
        elif status == "REJECTED":
            return "RE-EXECUTE"
        return "Supervisor"
            
    # 为关键作业节点注册审批策略
    workflow.add_conditional_edges("Respond", lambda s: "END" if s.get("status")=="APPROVED" else "Respond", {"END": END, "Respond": "Respond"})
    
    # 核心推理节点完成后，审批通过的均流向防漂移抽提中间件
    workflow.add_conditional_edges("Analysis", route_after_node, {"Supervisor": "Alignment", "RE-EXECUTE": "Analysis"})
    workflow.add_conditional_edges("Modeling", route_after_node, {"Supervisor": "Alignment", "RE-EXECUTE": "Modeling"})
    workflow.add_conditional_edges("Coder", route_after_node, {"Supervisor": "Alignment", "RE-EXECUTE": "Coder"})
    workflow.add_conditional_edges("Review", route_after_node, {"Supervisor": "Alignment", "RE-EXECUTE": "Review"})
    
    # 中间件抽提完成后自动导向 Supervisor
    workflow.add_edge("Alignment", "Supervisor")
    
    workflow.add_conditional_edges("Writing", route_after_node, {"Supervisor": "Supervisor", "RE-EXECUTE": "Writing"})
    workflow.add_conditional_edges("Export", lambda s: "END" if s.get("status")=="APPROVED" else "Export", {"END": END, "Export": "Export"})
    
    # 6. 持久化与中断节点编译
    # 建立 SQLite 连接以支持跨会话状态持久化 (无限期长对话)
    conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=["Respond", "Coder", "Analysis", "Modeling", "Review", "Writing", "Export"]
    )
    return app

# 单例编译后的图形对象 (供 FastAPI 调用)
app = build_graph()
