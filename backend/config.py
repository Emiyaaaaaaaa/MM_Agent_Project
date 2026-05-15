import os
from typing import Dict, List


def get_cors_origins() -> List[str]:
    raw = os.environ.get("MM_AGENT_CORS_ORIGINS", "")
    if raw.strip():
        return [item.strip() for item in raw.split(",") if item.strip()]
    return ["http://localhost:5173", "http://127.0.0.1:5173"]


def get_secret_key() -> str:
    return os.environ.get("MM_AGENT_SECRET_KEY", "mm-agent-dev-secret")


# Supervisor 统一路由配置
SUPERVISOR_ROUTING_CONFIG: Dict[str, object] = {
    "fusion_rag_precheck_top_k": 1,
    "allowed_next_nodes": [
        "END",
        "Fusion",
        "Respond",
        "Read",
        "Coding",
        "Analysis",
        "Modeling",
        "Review",
        "Writing",
        "Export",
    ],
    "end_fallback_node": "Respond",
    "end_stage_ids": [
        "export_completed",
    ],
    "paper_finished_stage_id": "paper_finished",
    "approved_stage_transitions": {
        "analysis_completed": "Modeling",
        "modeling_completed": "Coding",
        "coding_completed": "Review",
        "review_completed": "Writing",
    },
    "approved_failure_stage_transitions": {
        "modeling_failed_empty_output": "Respond",
        "modeling_failed_missing_api_key": "Respond",
    },
    "workflow_completed_stage": {"id": "workflow_completed", "label": "Workflow Completed"},
    "export_routing_stage": {"id": "routing_to_export", "label": "Routing to Export"},
    "fusion_routing_stage": {"id": "routing_to_fusion", "label": "Enforcing data retrieval (Fusion)"},
    "missing_api_key_stage": {"id": "missing_task_api_key", "label": "Missing Task API Key"},
    "routing_stage_template": {"id_prefix": "routing_to_", "label_template": "Routing to {next}"},
    "stage_memory_keys": {
        "Analysis": "analysis_report",
        "Modeling": "mathematical_model",
        "Coding": "generated_code",
        "Review": "review_report",
        "Writing": "paper_draft",
        "Export": "paper_draft",
    },
    "node_stage_ids": {
        "Analysis": "analysis_completed",
        "Modeling": "modeling_completed",
        "Coding": "coding_completed",
        "Review": "review_completed",
        "Writing": "paper_finished",
        "Export": "export_completed",
        "Respond": "respond_prepared",
    },
}


# 目标节点前置依赖链（按顺序满足，否则回退到首个缺失对应阶段）
SUPERVISOR_GUARD_RULES: Dict[str, List[str]] = {
    "Modeling": ["analysis_report"],
    "Coding": ["analysis_report", "mathematical_model"],
    "Review": ["analysis_report", "mathematical_model", "generated_code"],
    "Writing": ["analysis_report", "mathematical_model", "generated_code", "review_report"],
    "Export": ["paper_draft"],
}


GRAPH_ROUTING_CONFIG: Dict[str, object] = {
    "start_node": "Supervisor",
    "fixed_edges": [
        ["Reader", "Supervisor"],
        ["Fusion", "Supervisor"],
        ["Alignment", "Supervisor"],
    ],
    "supervisor_route_map": {
        "END": "END",
        "Fusion": "Fusion",
        "Respond": "Respond",
        "Read": "Reader",
        "Coding": "Coder",
        "Analysis": "Analysis",
        "Modeling": "Modeling",
        "Review": "Review",
        "Writing": "Writing",
        "Export": "Export",
    },
    "route_after_node": {
        "rejected_status": "REJECTED",
        "supervisor_label": "Supervisor",
        "reexecute_label": "RE-EXECUTE",
        "supervisor_statuses": ["APPROVED", "PENDING"],
    },
    "conditional_nodes": {
        "Respond": {"Supervisor": "Supervisor", "RE-EXECUTE": "Respond"},
        "Analysis": {"Supervisor": "Alignment", "RE-EXECUTE": "Analysis"},
        "Modeling": {"Supervisor": "Alignment", "RE-EXECUTE": "Modeling"},
        "Coder": {"Supervisor": "Alignment", "RE-EXECUTE": "Coder"},
        "Review": {"Supervisor": "Alignment", "RE-EXECUTE": "Review"},
        "Writing": {"Supervisor": "Supervisor", "RE-EXECUTE": "Writing"},
        "Export": {"Supervisor": "Supervisor", "RE-EXECUTE": "Export"},
    },
    # 关闭 HITL 打断，工作流自动连续执行到终态
    "interrupt_after": [],
}
