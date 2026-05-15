import re
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field, field_validator
from backend.graph.state import AgentState
from backend.config import SUPERVISOR_ROUTING_CONFIG, SUPERVISOR_GUARD_RULES
from backend.services.rag_engine import rag_engine
from backend.services.serialization import extract_text_content, get_stage_id, make_stage

logger = logging.getLogger("mm-agent")

# 定义意图识别的输出结构
class RouteResponse(BaseModel):
    """决定下一步要跳转的环节"""
    next: str = Field(description="选择下一步目标节点。")

    @field_validator("next")
    @classmethod
    def validate_next(cls, value: str) -> str:
        normalized = (value or "").strip()
        alias_map = {
            "end": "END",
            "fusion": "Fusion",
            "respond": "Respond",
            "response": "Respond",
            "read": "Read",
            "reader": "Read",
            "coding": "Coding",
            "coder": "Coding",
            "analysis": "Analysis",
            "analyze": "Analysis",
            "modeling": "Modeling",
            "modeler": "Modeling",
            "review": "Review",
            "reviewer": "Review",
            "writing": "Writing",
            "writer": "Writing",
            "export": "Export",
        }
        normalized = alias_map.get(normalized.lower(), normalized)
        allowed = set(SUPERVISOR_ROUTING_CONFIG.get("allowed_next_nodes", []))
        if normalized not in allowed:
            raise ValueError(f"Invalid next node: {value}")
        return normalized

# 初始化 Supervisor 节点
class SupervisorNode:
    """
    智能主管节点 (Supervisor)：Agent 的大脑。
    负责意图识别、工作流调度以及根据当前建模进度自主决策下一步路径。
    """
    def __init__(self):
        self.routing_config = SUPERVISOR_ROUTING_CONFIG
        self.guard_rules = SUPERVISOR_GUARD_RULES
        self.stage_to_memory_key = {
            mem_key: stage for stage, mem_key in self.routing_config.get("stage_memory_keys", {}).items()
        }
        self.stage_to_memory_key["paper_draft"] = "Writing"
        # 高级调度指令：确立“意图优先 > 进度辅助”的动态路由准则
        self.system_prompt = (
            "你是一个 MCM/ICM 建模竞赛智能体的主管 (Supervisor)。"
            "你的核心原则是：**尊重用户意图 > 遵循建模流程**。\n\n"
            "【任务流水线引导】\n"
            "1. **撰写论文/全流程任务**：当用户要求“撰写论文”、“准备 F 题/A 题”或执行类似完整流程时，你的目标是**依次**经过：Analysis (审题) -> Modeling (建模) -> Coding (编程) -> Review (审查) -> Writing (写作) -> Export (导出)。\n"
            "2. **严禁越级**：在没有分析报告的情况下严禁直接写论文。如果没有读取文件，请先 Read。\n"
            "3. **咨询切换**：如果用户只是单纯询问某个概念或寻求建议，可以使用 Respond 节点直接回答。\n\n"
            "【进度感知指南】\n"
            "- 如果当前存在 context (检索到的资料) 但未开始分析 -> Analysis\n"
            "- Analysis (已完成) -> Modeling\n"
            "- Modeling (已完成) -> Coding\n"
            "- Coding (已完成) -> Review\n"
            "- Review (已完成) -> Writing"
        )

    def _make_routing_stage(self, next_node: str) -> dict:
        template = self.routing_config.get("routing_stage_template", {})
        id_prefix = str(template.get("id_prefix", "routing_to_"))
        label_template = str(template.get("label_template", "Routing to {next}"))
        return make_stage(
            stage_id=f"{id_prefix}{next_node.lower()}",
            label=label_template.format(next=next_node),
        )

    def _progress_fallback_next(self, shared_mem: dict) -> str:
        if not shared_mem.get("analysis_report"):
            return "Analysis"
        if not shared_mem.get("mathematical_model"):
            return "Modeling"
        if not shared_mem.get("generated_code"):
            return "Coding"
        if not shared_mem.get("review_report"):
            return "Review"
        if not shared_mem.get("paper_draft"):
            return "Writing"
        return "Export"

    def __call__(self, state: AgentState):
        """意图识别与动态决策逻辑实现"""
        shared_mem = state.get("shared_memory", {})
        status = state.get("status", "PENDING")
        stage_id = get_stage_id(state.get("stage"))
        msg_list = state.get("messages") or []
        logger.info(
            "Supervisor enter status=%s stage_id=%s messages=%s",
            status,
            stage_id,
            len(msg_list),
        )

        # 统一终止与后续节点调度：所有节点回到 Supervisor 后在此集中决策
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
                    "stage": self.routing_config.get("workflow_completed_stage", make_stage("workflow_completed", "Workflow Completed")),
                }
            paper_finished_stage_id = str(self.routing_config.get("paper_finished_stage_id", "paper_finished"))
            if stage_id == paper_finished_stage_id:
                return {
                    "next": "Export",
                    "shared_memory": shared_mem,
                    "status": "PENDING",
                    "stage": self.routing_config.get("export_routing_stage", make_stage("routing_to_export", "Routing to Export")),
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
        
        # 处理文件读取的特殊冷启动逻辑
        if shared_mem.get("input_file_path") and not shared_mem.get("raw_document_content"):
            return {"next": "Read"}

        messages = state["messages"]
        last_user_msg = ""
        for msg in reversed(messages):
            msg_type = getattr(msg, "type", None)
            if msg_type is None and isinstance(msg, dict):
                msg_type = msg.get("role")
            if msg_type in {"human", "user"}:
                last_user_msg = extract_text_content(msg).strip()
                break
        
        # 进度上下文提取
        completed_stages = [k for k in shared_mem.keys() if k.endswith("_report") or k in ["mathematical_model", "generated_code", "paper_draft"]]
        progress_info = f"【当前进度】: {completed_stages}"

        # 知识库优先：先做 RAG 预检，只要命中就强制先 Fusion（同一条用户消息仅触发一次）
        precheck_top_k = int(self.routing_config.get("fusion_rag_precheck_top_k", 1))
        forced_fusion_for = str(shared_mem.get("forced_fusion_for_message", "") or "")
        api_key = shared_mem.get("api_key")
        rag_hits = []
        if api_key and last_user_msg:
            logger.info(
                "Supervisor RAG precheck start (embed+chroma) message_len=%s top_k=%s",
                len(last_user_msg),
                precheck_top_k,
            )
            rag_hits = rag_engine.search(last_user_msg, n_results=precheck_top_k, api_key=api_key)
            logger.info("Supervisor RAG precheck done hits=%s", len(rag_hits))
        elif not last_user_msg:
            logger.warning("Supervisor RAG precheck skipped: empty last_user_msg")
        if rag_hits and forced_fusion_for != last_user_msg:
            logger.info(
                "Supervisor RAG precheck hit=%s, force Fusion first (message_len=%s)",
                len(rag_hits),
                len(last_user_msg),
            )
            return {
                "next": "Fusion",
                "shared_memory": {**shared_mem, "forced_fusion_for_message": last_user_msg},
                "stage": self.routing_config.get("fusion_routing_stage", make_stage("routing_to_fusion", "Enforcing data retrieval (Fusion)")),
            }

        new_memory = shared_mem.copy()

        # 构造增强提示词 (转义花括号以防干扰 LangChain)
        content_preview = extract_text_content(shared_mem.get("raw_document_content", ""))[:3000]
        safe_preview = content_preview[:500].replace("{", "{{").replace("}", "}}")
        context_context = f"{progress_info}\n【文档预览】: {safe_preview}..."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("system", context_context),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "请决策下一步跳转的目标节点。")
        ])
        
        # 动态模型实例化 (从任务配置加载)
        model_id = shared_mem.get("model_id") or "gemini-2.5-flash-lite"
        
        if not api_key:
            return {
                "next": "Respond",
                "status": "REJECTED",
                "human_feedback": "缺少任务 API Key，请在任务配置中提供后重试。",
                "stage": self.routing_config.get("missing_api_key_stage", make_stage("missing_task_api_key", "Missing Task API Key")),
            }

        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0,
            google_api_key=api_key
        )
        router = llm.with_structured_output(RouteResponse)
        
        # 执行路由决策
        chain = prompt | router
        route_model = shared_mem.get("model_id") or "gemini-2.5-flash-lite"
        logger.info("Supervisor LLM route invoke start model_id=%s", route_model)
        result = chain.invoke({"messages": messages})
        logger.info("Supervisor LLM route invoke done")
        allowed_next_nodes = set(self.routing_config.get("allowed_next_nodes", []))
        llm_route_fallback_node = str(self.routing_config.get("llm_route_fallback_node", "Respond"))
        progress_fallback_node = self._progress_fallback_next(shared_mem)

        chosen_next = None
        if result is None:
            logger.warning(
                "Supervisor route result is None, fallback to progress=%s then default=%s",
                progress_fallback_node,
                llm_route_fallback_node,
            )
        elif hasattr(result, "next"):
            chosen_next = getattr(result, "next")
        elif isinstance(result, dict):
            chosen_next = result.get("next")
        else:
            logger.warning(
                "Supervisor route result has unexpected type=%s, fallback to progress=%s then default=%s",
                type(result),
                progress_fallback_node,
                llm_route_fallback_node,
            )

        if not chosen_next or (allowed_next_nodes and chosen_next not in allowed_next_nodes):
            fallback_candidate = progress_fallback_node
            if allowed_next_nodes and fallback_candidate not in allowed_next_nodes:
                fallback_candidate = llm_route_fallback_node
            logger.warning("Supervisor route invalid next=%s, fallback to %s", chosen_next, fallback_candidate)
            chosen_next = fallback_candidate

        # 防越级门控：即便模型误判，也强制保持可恢复的阶段顺序
        required_keys = self.guard_rules.get(chosen_next, [])
        for key in required_keys:
            if not shared_mem.get(key):
                chosen_next = self.stage_to_memory_key.get(key, "Analysis")
                break

        if chosen_next == "END":
            end_stage_ids = self.routing_config.get("end_stage_ids", [])
            if stage_id not in end_stage_ids:
                chosen_next = str(self.routing_config.get("end_fallback_node", "Respond"))

        logger.info("Supervisor route decided -> %s", chosen_next)

        return {
            "next": chosen_next,
            "shared_memory": new_memory,
            "status": "PENDING", 
            "stage": self._make_routing_stage(chosen_next),
        }

# 单例
supervisor_node = SupervisorNode()
