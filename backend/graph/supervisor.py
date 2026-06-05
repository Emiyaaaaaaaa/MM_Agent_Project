import json
import logging
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field, field_validator

from backend.graph.state import AgentState
from backend.graph.supervisor_decision import (
    SupervisorDecision,
    apply_decision_to_memory,
    build_progress_snapshot,
    normalize_node,
)
from backend.config import SUPERVISOR_ROUTING_CONFIG, SUPERVISOR_GUARD_RULES
from backend.services.rag_engine import rag_engine
from backend.services.serialization import extract_text_content, get_stage_id, make_stage, messages_for_llm

logger = logging.getLogger("mm-agent")


class RouteResponse(BaseModel):
    """冷启动/无用户控制请求时的简化路由（兼容）。"""

    next: str = Field(description="选择下一步目标节点。")

    @field_validator("next")
    @classmethod
    def validate_next(cls, value: str) -> str:
        normalized = normalize_node(value)
        allowed = set(SUPERVISOR_ROUTING_CONFIG.get("allowed_next_nodes", []))
        if normalized not in allowed:
            raise ValueError(f"Invalid next node: {value}")
        return normalized


class SupervisorNode:
    """
    智能主管：节点完成后确定性推进；用户发消息时由 LLM 阅读请求并决策
    （续跑 / 跳转 / 指定阶段重做 / 答疑 / Fusion / 结束等）。
    """

    def __init__(self):
        self.routing_config = SUPERVISOR_ROUTING_CONFIG
        self.guard_rules = SUPERVISOR_GUARD_RULES
        self.stage_to_memory_key = {
            mem_key: stage for stage, mem_key in self.routing_config.get("stage_memory_keys", {}).items()
        }
        self.stage_to_memory_key["paper_draft"] = "Writing"
        self.stage_to_memory_key["fusion_guidance"] = "Fusion"

        self.auto_pipeline_prompt = (
            "你是 MCM/ICM 工作流主管。根据对话与进度选择下一节点。\n"
            "流程: Analysis -> Modeling -> Writing(大纲) -> Coding -> Review -> Writing(正文) -> Export。\n"
            "缺前置产物时不得越级。纯咨询用 Respond。"
        )

        self.user_control_prompt = (
            "你是 MCM/ICM 建模工作台的主管 (Supervisor)。用户通过消息指挥流程，你必须读懂其意图并给出结构化决策。\n\n"
            "【可选意图 intent】\n"
            "- continue_pipeline: 按当前进度继续下一未完成阶段（如用户说「继续」「接着做」）\n"
            "- rerun_stage: 重做某一阶段，并清除该阶段及之后产物（如「重新建模」「重做仿真」「重写论文大纲」）\n"
            "- jump_to_stage: 跳转到指定阶段执行（若缺前置产物，target 改为应先补做的阶段）\n"
            "- consult: 仅回答问题，不跑流水线（Respond）\n"
            "- fusion_first: 需要先 Fusion 检索（Respond）\n"
            "- read_file: 需要先读取上传文件（Read）\n"
            "- end_task: 结束任务（END）\n\n"
            "【节点说明】\n"
            "- Analysis 审题 | Modeling 建模 | Writing 论文(分 outline 规划 与 draft 正文，writing_phase 区分)\n"
            "- Coding 仿真与图表 | Review 审查 | Export 导出 PDF/ZIP\n"
            "- Fusion 知识融合 | Respond 直接回复 | Read 读文件\n\n"
            "【Writing 子阶段】\n"
            "- outline: 仅在 Modeling 完成后、尚无 paper_outline 时；或用户明确要求「重新规划大纲/图表计划」\n"
            "- draft: 已有大纲且 Review 完成后写正文；或用户要求「重写论文正文」\n"
            "- auto: 由系统根据进度判断\n\n"
            "【重做规则】\n"
            "用户要求重做某阶段时，intent=rerun_stage，target_node=该阶段，writing_phase 按需填写。\n"
            "重做 Modeling 会清除模型及之后所有产物；重做 Coding 清除代码/图表及之后；依此类推。\n\n"
            "【Fusion Contract 优先规则】\n"
            "- 若用户要求“补图/图不够/高优先级图缺失”，优先决策到 Coding（rerun_stage 或 jump_to_stage）。\n"
            "- 若用户要求“篇幅不足/缺标题摘要/章节不完整/结果解读不足”，优先决策到 Writing.draft。\n"
            "- 若用户要求“重排章节结构/重做论文规划”，优先决策到 Writing.outline。\n"
            "- 所有阶段决策必须尽量遵守 fusion_contract 的质量门槛（图表下限、正文最小长度、必备章节等）。\n\n"
            "【原则】尊重用户指令优先；用户仅咨询时不要强行进入 Analysis。\n"
            "【继续流水线硬性规则】intent=continue_pipeline 时，必须按【当前进度 JSON】中的 suggested_next 执行；"
            "若 has_fusion=false 或 fusion 未完成，禁止进入 Analysis/Modeling/Writing/Coding。"
        )

    def _make_routing_stage(self, next_node: str, label: Optional[str] = None) -> dict:
        template = self.routing_config.get("routing_stage_template", {})
        id_prefix = str(template.get("id_prefix", "routing_to_"))
        label_template = str(template.get("label_template", "Routing to {next}"))
        return make_stage(
            stage_id=f"{id_prefix}{next_node.lower()}",
            label=label or label_template.format(next=next_node),
        )

    def _artifact_ready(self, shared_mem: dict, key: str) -> bool:
        if key == "paper_outline":
            outline = shared_mem.get("paper_outline")
            return isinstance(outline, dict) and bool(outline.get("sections"))
        if key == "fusion_guidance":
            return self._fusion_artifact_ready(shared_mem)
        return bool(extract_text_content(shared_mem.get(key, "")).strip())

    def _fusion_artifact_ready(self, shared_mem: dict) -> bool:
        """Fusion 完成：需有效融合简报或 contract/审计落盘，错误占位文本不算。"""
        guidance = extract_text_content(shared_mem.get("fusion_guidance", "")).strip()
        if guidance.startswith("[系统错误]") or guidance.startswith("[系统警告"):
            return False
        contract = shared_mem.get("fusion_contract")
        if isinstance(contract, dict) and contract.get("chart_contract"):
            return True
        meta = shared_mem.get("fusion_audit_meta")
        if isinstance(meta, dict) and meta.get("saved"):
            return True
        if guidance and ("[STYLE_GUIDE]" in guidance or "CHART_PLAN_JSON" in guidance):
            return True
        return False

    def _has_paper_outline(self, shared_mem: dict) -> bool:
        outline = shared_mem.get("paper_outline")
        return isinstance(outline, dict) and bool(outline.get("sections"))

    def _progress_fallback_next(self, shared_mem: dict) -> str:
        if not self._fusion_artifact_ready(shared_mem):
            logger.info("Supervisor fallback -> Fusion (fusion not completed)")
            return "Fusion"
        if not self._artifact_ready(shared_mem, "analysis_report"):
            return "Analysis"
        if not self._artifact_ready(shared_mem, "mathematical_model"):
            return "Modeling"
        if not self._has_paper_outline(shared_mem):
            return "Writing"
        if not self._artifact_ready(shared_mem, "generated_code"):
            return "Coding"
        if not self._artifact_ready(shared_mem, "review_report"):
            return "Review"
        if not self._artifact_ready(shared_mem, "paper_draft"):
            return "Writing"
        return "Export"

    def _apply_dependency_guard(self, chosen_next: str, shared_mem: dict, *, strict: bool) -> str:
        if not strict:
            return chosen_next
        required_keys = self.guard_rules.get(chosen_next, [])
        for key in required_keys:
            if key == "paper_outline":
                if not self._has_paper_outline(shared_mem):
                    return self.stage_to_memory_key.get(key, "Writing")
            elif key == "fusion_guidance":
                if not self._fusion_artifact_ready(shared_mem):
                    return "Fusion"
            elif not shared_mem.get(key):
                return self.stage_to_memory_key.get(key, "Analysis")
        return chosen_next

    def _route_after_node_completion(self, state: AgentState, shared_mem: dict, status: str, stage_id: str) -> Optional[Dict[str, Any]]:
        """节点 APPROVED / PENDING 后的确定性推进，不经用户 LLM。"""
        if status == "APPROVED":
            approved_failure_stage_transitions = self.routing_config.get("approved_failure_stage_transitions", {})
            last_failure_stage = str(shared_mem.get("last_failure_stage_id", "") or "")
            if last_failure_stage:
                next_node = approved_failure_stage_transitions.get(last_failure_stage)
                if next_node:
                    patched_memory = shared_mem.copy()
                    patched_memory.pop("last_failure_stage_id", None)
                    return {
                        "next": next_node,
                        "shared_memory": patched_memory,
                        "status": "PENDING",
                        "stage": self._make_routing_stage(next_node),
                    }

            end_stage_ids = self.routing_config.get("end_stage_ids", [])
            if stage_id in end_stage_ids:
                return {
                    "next": "END",
                    "shared_memory": shared_mem,
                    "status": "APPROVED",
                    "stage": self.routing_config.get(
                        "workflow_completed_stage",
                        make_stage("workflow_completed", "Workflow Completed"),
                    ),
                }
            if stage_id == "outline_completed":
                return {
                    "next": "Coding",
                    "shared_memory": shared_mem,
                    "status": "PENDING",
                    "stage": self._make_routing_stage("Coding"),
                }
            paper_finished_stage_id = str(self.routing_config.get("paper_finished_stage_id", "paper_finished"))
            if stage_id == paper_finished_stage_id:
                return {
                    "next": "Export",
                    "shared_memory": shared_mem,
                    "status": "PENDING",
                    "stage": self.routing_config.get(
                        "export_routing_stage",
                        make_stage("routing_to_export", "Routing to Export"),
                    ),
                }
            approved_stage_transitions = self.routing_config.get("approved_stage_transitions", {})
            next_node = approved_stage_transitions.get(stage_id)
            if next_node:
                return {
                    "next": next_node,
                    "shared_memory": shared_mem,
                    "status": "PENDING",
                    "stage": self._make_routing_stage(next_node),
                }
            if stage_id == "alignment_updated":
                next_node = self._progress_fallback_next(shared_mem)
                logger.info("Supervisor deterministic route after alignment -> %s", next_node)
                return {
                    "next": next_node,
                    "shared_memory": shared_mem,
                    "status": "PENDING",
                    "stage": self._make_routing_stage(next_node),
                }

        approved_stage_transitions = self.routing_config.get("approved_stage_transitions", {})
        pending_next = approved_stage_transitions.get(stage_id)
        if status == "PENDING" and pending_next:
            logger.info("Supervisor auto-advance pending stage_id=%s -> %s", stage_id, pending_next)
            return {
                "next": pending_next,
                "shared_memory": shared_mem,
                "status": "PENDING",
                "stage": self._make_routing_stage(pending_next),
            }
        if stage_id == "alignment_updated" or str(stage_id).startswith("alignment_"):
            next_node = self._progress_fallback_next(shared_mem)
            logger.info("Supervisor deterministic route after alignment stage=%s -> %s", stage_id, next_node)
            return {
                "next": next_node,
                "shared_memory": shared_mem,
                "status": "PENDING",
                "stage": self._make_routing_stage(next_node),
            }
        if self._artifact_ready(shared_mem, "analysis_report") and stage_id in {
            "analysis_completed",
            "routing_to_analysis",
        }:
            next_node = self._progress_fallback_next(shared_mem)
            if next_node != "Analysis":
                logger.info("Supervisor skip repeat Analysis -> %s", next_node)
                return {
                    "next": next_node,
                    "shared_memory": shared_mem,
                    "status": "PENDING",
                    "stage": self._make_routing_stage(next_node),
                }
        return None

    def _invoke_user_decision_llm(
        self,
        *,
        shared_mem: dict,
        messages: list,
        user_request: str,
        stage_id: str,
        suggested_next: str,
    ) -> Optional[SupervisorDecision]:
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-3.1-flash-lite"
        if not api_key:
            return None

        progress = build_progress_snapshot(shared_mem, stage_id)
        progress["suggested_next"] = suggested_next
        fusion_contract = shared_mem.get("fusion_contract")
        fusion_contract_brief: Dict[str, Any] = {}
        if isinstance(fusion_contract, dict):
            chart_contract = fusion_contract.get("chart_contract") if isinstance(fusion_contract.get("chart_contract"), dict) else {}
            quality_gates = fusion_contract.get("quality_gates") if isinstance(fusion_contract.get("quality_gates"), dict) else {}
            fusion_contract_brief = {
                "required_chart_count": chart_contract.get("required_chart_count"),
                "must_include_high_priority": chart_contract.get("must_include_high_priority"),
                "draft_min_chars": quality_gates.get("draft_min_chars"),
                "min_figure_refs": quality_gates.get("min_figure_refs"),
                "mandatory_sections": quality_gates.get("mandatory_sections"),
            }
        safe_progress = json.dumps(progress, ensure_ascii=False).replace("{", "{{").replace("}", "}}")
        safe_contract = json.dumps(fusion_contract_brief, ensure_ascii=False).replace("{", "{{").replace("}", "}}")
        safe_request = (user_request or "").replace("{", "{{").replace("}", "}}")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.user_control_prompt),
                ("system", f"【当前进度 JSON】\n{safe_progress}"),
                ("system", f"【Fusion Contract 关键门槛】\n{safe_contract}"),
                ("system", f"【用户最新指令】\n{safe_request}"),
                MessagesPlaceholder(variable_name="messages"),
                ("system", "请输出结构化决策（intent/target_node/writing_phase/user_instruction/reasoning）。"),
            ]
        )
        llm = ChatGoogleGenerativeAI(model=model_id, temperature=0, google_api_key=api_key)
        chain = prompt | llm.with_structured_output(SupervisorDecision)
        try:
            result = chain.invoke({
                "messages": messages_for_llm(
                    messages,
                    max_messages=6,
                    max_chars_per_message=1000,
                )
            })
            if isinstance(result, SupervisorDecision):
                return result
        except Exception:
            logger.exception("Supervisor user decision LLM failed")
        return None

    def _resolve_target_from_decision(
        self,
        decision: SupervisorDecision,
        shared_mem: dict,
    ) -> str:
        intent = decision.intent
        target = decision.target_node

        if intent == "continue_pipeline":
            return self._progress_fallback_next(shared_mem)
        if intent == "end_task":
            return "END"
        if intent == "consult":
            return "Respond"
        if intent == "fusion_first":
            return "Fusion"
        if intent == "read_file":
            return "Read"
        if intent in {"rerun_stage", "jump_to_stage"}:
            return target
        return target or self._progress_fallback_next(shared_mem)

    def _route_by_user_control(
        self,
        state: AgentState,
        shared_mem: dict,
        user_request: str,
        stage_id: str,
    ) -> Dict[str, Any]:
        messages = state.get("messages") or []
        suggested_next = self._progress_fallback_next(shared_mem)

        decision = self._invoke_user_decision_llm(
            shared_mem=shared_mem,
            messages=messages,
            user_request=user_request,
            stage_id=stage_id,
            suggested_next=suggested_next,
        )

        if decision is None:
            chosen_next = suggested_next
            new_memory = shared_mem.copy()
            new_memory.pop("user_control_request", None)
            logger.warning("Supervisor user decision fallback -> %s", chosen_next)
            return {
                "next": chosen_next,
                "shared_memory": new_memory,
                "status": "PENDING",
                "stage": self._make_routing_stage(chosen_next),
            }

        new_memory = apply_decision_to_memory(shared_mem, decision)
        new_memory.pop("user_control_request", None)

        chosen_next = self._resolve_target_from_decision(decision, new_memory)
        if not self._fusion_artifact_ready(new_memory) and chosen_next not in {
            "Fusion",
            "Read",
            "Respond",
            "END",
        }:
            logger.info(
                "Supervisor force Fusion before %s (fusion not completed, llm_target=%s)",
                chosen_next,
                decision.target_node,
            )
            chosen_next = "Fusion"
        strict_guard = decision.intent not in {"rerun_stage", "jump_to_stage"}
        chosen_next = self._apply_dependency_guard(chosen_next, new_memory, strict=strict_guard)

        if chosen_next == "END":
            end_stage_ids = self.routing_config.get("end_stage_ids", [])
            if stage_id not in end_stage_ids:
                chosen_next = str(self.routing_config.get("end_fallback_node", "Respond"))

        instruction = (decision.user_instruction or "").strip()
        logger.info(
            "Supervisor user decision intent=%s target=%s resolved=%s reason=%s",
            decision.intent,
            decision.target_node,
            chosen_next,
            (decision.reasoning or "")[:120],
        )

        routing_label = decision.reasoning or f"User-directed: {chosen_next}"
        return {
            "next": chosen_next,
            "shared_memory": new_memory,
            "status": "PENDING",
            "human_feedback": instruction,
            "stage": self._make_routing_stage(chosen_next, label=routing_label[:80]),
        }

    def _route_cold_start_llm(self, state: AgentState, shared_mem: dict, messages: list, progress_info: str) -> Dict[str, Any]:
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-3.1-flash-lite"
        if not api_key:
            return {
                "next": "Respond",
                "status": "REJECTED",
                "human_feedback": "缺少任务 API Key，请在任务配置中提供后重试。",
                "stage": self.routing_config.get(
                    "missing_api_key_stage",
                    make_stage("missing_task_api_key", "Missing Task API Key"),
                ),
            }

        content_preview = extract_text_content(shared_mem.get("raw_document_content", ""))[:500]
        safe_preview = content_preview.replace("{", "{{").replace("}", "}}")
        context_context = f"{progress_info}\n【文档预览】: {safe_preview}..."

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.auto_pipeline_prompt),
                ("system", context_context),
                MessagesPlaceholder(variable_name="messages"),
                ("system", "请决策下一步跳转的目标节点。"),
            ]
        )
        llm = ChatGoogleGenerativeAI(model=model_id, temperature=0, google_api_key=api_key)
        result = (prompt | llm.with_structured_output(RouteResponse)).invoke(
            {
                "messages": messages_for_llm(
                    messages,
                    max_messages=6,
                    max_chars_per_message=1000,
                )
            }
        )

        chosen_next = getattr(result, "next", None) if result else None
        if not chosen_next:
            chosen_next = self._progress_fallback_next(shared_mem)
        chosen_next = self._apply_dependency_guard(chosen_next, shared_mem, strict=True)

        if chosen_next == "END":
            chosen_next = str(self.routing_config.get("end_fallback_node", "Respond"))

        return {
            "next": chosen_next,
            "shared_memory": shared_mem.copy(),
            "status": "PENDING",
            "stage": self._make_routing_stage(chosen_next),
        }

    def __call__(self, state: AgentState):
        shared_mem = dict(state.get("shared_memory", {}) or {})
        status = state.get("status", "PENDING")
        stage_id = get_stage_id(state.get("stage"))
        logger.info("Supervisor enter status=%s stage_id=%s", status, stage_id)

        auto_route = self._route_after_node_completion(state, shared_mem, status, stage_id)
        if auto_route is not None:
            return auto_route

        if shared_mem.get("input_file_path") and not shared_mem.get("raw_document_content"):
            return {"next": "Read", "shared_memory": shared_mem, "status": "PENDING", "stage": self._make_routing_stage("Read")}

        user_request = str(shared_mem.pop("user_control_request", "") or "").strip()
        if user_request:
            return self._route_by_user_control(state, shared_mem, user_request, stage_id)

        messages = state.get("messages") or []
        last_user_msg = ""
        for msg in reversed(messages):
            msg_type = getattr(msg, "type", None)
            if msg_type is None and isinstance(msg, dict):
                msg_type = msg.get("role")
            if msg_type in {"human", "user"}:
                last_user_msg = extract_text_content(msg).strip()
                break

        if last_user_msg:
            return self._route_by_user_control(state, shared_mem, last_user_msg, stage_id)

        completed_stages = [
            k
            for k in shared_mem.keys()
            if k.endswith("_report") or k in ["mathematical_model", "generated_code", "paper_draft", "paper_outline"]
        ]
        progress_info = f"【当前进度】: {completed_stages}"

        api_key = shared_mem.get("api_key")
        precheck_top_k = int(self.routing_config.get("fusion_rag_precheck_top_k", 1))
        forced_fusion_for = str(shared_mem.get("forced_fusion_for_message", "") or "")
        if api_key and last_user_msg:
            rag_hits = rag_engine.search(last_user_msg, n_results=precheck_top_k, api_key=api_key)
            if rag_hits and forced_fusion_for != last_user_msg:
                return {
                    "next": "Fusion",
                    "shared_memory": {**shared_mem, "forced_fusion_for_message": last_user_msg},
                    "status": "PENDING",
                    "stage": self.routing_config.get(
                        "fusion_routing_stage",
                        make_stage("routing_to_fusion", "Enforcing data retrieval (Fusion)"),
                    ),
                }

        return self._route_cold_start_llm(state, shared_mem, messages, progress_info)


supervisor_node = SupervisorNode()
