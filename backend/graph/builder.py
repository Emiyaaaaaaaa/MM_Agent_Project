from langgraph.graph import StateGraph, START, END
from backend.config import GRAPH_ROUTING_CONFIG
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


def create_workflow():
    workflow = StateGraph(AgentState)
    routing_config = GRAPH_ROUTING_CONFIG
    supervisor_route_map = dict(routing_config.get("supervisor_route_map", {}))
    start_node = str(routing_config.get("start_node", "Supervisor"))

    workflow.add_node("Reader", reader_node)
    workflow.add_node("Supervisor", supervisor_node)
    workflow.add_node("Fusion", fusion_node)
    workflow.add_node("Respond", respond_node)
    workflow.add_node("Coder", coder_node)
    workflow.add_node("Analysis", analyzer_node)
    workflow.add_node("Modeling", modeler_node)
    workflow.add_node("Review", reviewer_node)
    workflow.add_node("Writing", writer_node)
    workflow.add_node("Export", exporter_node)
    workflow.add_node("Alignment", alignment_node)

    workflow.add_edge(START, start_node)
    for edge in routing_config.get("fixed_edges", []):
        if not isinstance(edge, list) or len(edge) != 2:
            continue
        workflow.add_edge(edge[0], edge[1])

    def route_from_supervisor(state: AgentState):
        return state["next"]

    conditional_map = {
        route_key: (END if target == "END" else target)
        for route_key, target in supervisor_route_map.items()
    }
    workflow.add_conditional_edges(
        "Supervisor",
        route_from_supervisor,
        conditional_map,
    )

    def route_after_node(state: AgentState):
        rule_config = routing_config.get("route_after_node", {})
        status = state.get("status", "PENDING")
        rejected_status = str(rule_config.get("rejected_status", "REJECTED"))
        reexecute_label = str(rule_config.get("reexecute_label", "RE-EXECUTE"))
        supervisor_label = str(rule_config.get("supervisor_label", "Supervisor"))
        supervisor_statuses = set(rule_config.get("supervisor_statuses", ["APPROVED", "PENDING"]))
        if status == rejected_status:
            return reexecute_label
        if status in supervisor_statuses:
            return supervisor_label
        return supervisor_label

    for node_name, mapping in routing_config.get("conditional_nodes", {}).items():
        workflow.add_conditional_edges(node_name, route_after_node, mapping)

    return workflow


def compile_graph(checkpointer):
    workflow = create_workflow()
    interrupt_after = GRAPH_ROUTING_CONFIG.get("interrupt_after", [])
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
    )
