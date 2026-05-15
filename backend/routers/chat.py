import asyncio
import json
import logging
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from langgraph.errors import GraphInterrupt
from backend.db import crud, database
from backend.schemas.api import ChatMessage
from backend.services.bus import message_bus
from backend.services.serialization import (
    blocks_effectively_empty,
    blocks_to_text,
    ensure_blocks,
    extract_final_response,
    extract_text_content,
    get_stage_id,
    make_stage,
    message_to_text,
    resolve_final_blocks,
    summary_from_blocks,
)
from backend.config import SUPERVISOR_ROUTING_CONFIG


logger = logging.getLogger("mm-agent")
router = APIRouter(tags=["chat"])
STATE_UPDATE_TIMEOUT_SECONDS = 20
STREAM_IDLE_TIMEOUT_SECONDS = 90


def _classify_runtime_error(exc: Exception) -> Dict[str, Any]:
    message = str(exc or "")
    lower_msg = message.lower()
    exc_name = type(exc).__name__.lower()
    network_markers = [
        "end of tcp stream",
        "serviceunavailable",
        "unavailable",
        "grpc_status:14",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "timed out",
        "timeout",
        "failed to connect",
        "handshaker shutdown",
        "tcp handshaker",
    ]
    is_network = any(marker in lower_msg for marker in network_markers) or "serviceunavailable" in exc_name
    if is_network:
        return {
            "code": "NETWORK_UPSTREAM_UNAVAILABLE",
            "category": "network",
            "retriable": True,
            "user_message": "网络连接异常或上游模型服务暂不可用，请稍后重试。",
            "detail": message[:600],
        }
    return {
        "code": "RUNTIME_ERROR",
        "category": "runtime",
        "retriable": False,
        "user_message": "执行中断，已保存当前生成内容。",
        "detail": message[:600],
    }


def _structured_error_payload(
    *,
    user_message: str,
    code: str,
    category: str = "network",
    retriable: bool = True,
    detail: str = "",
) -> Dict[str, Any]:
    return {
        "blocks": ensure_blocks(user_message),
        "error": {
            "code": code,
            "category": category,
            "retriable": bool(retriable),
            "detail": (detail or "")[:600],
        },
    }


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, message: Dict[str, Any], websocket: WebSocket):
        try:
            event_type = message.get("type", "UNKNOWN")
            node = message.get("node")
            payload = message.get("payload")
            payload_len = len(str(payload)) if payload is not None else 0
            stage_id = ""
            if isinstance(payload, dict):
                stage_id = get_stage_id(payload.get("stage"))
            log_fn = logger.debug if event_type in {"TOKEN", "STATUS"} else logger.info
            log_fn(
                "WS outbound: client=%s type=%s node=%s stage=%s payload_len=%s",
                getattr(websocket, "client", None),
                event_type,
                node,
                stage_id,
                payload_len,
            )
            await websocket.send_json(message)
        except Exception:
            self.disconnect(websocket)


manager = ConnectionManager()


def _normalize_feedback(feedback: str) -> Dict[str, str]:
    text = (feedback or "").strip()
    lower = text.lower()
    if lower in {"approve", "approved", "continue", "yes", "继续", "同意"}:
        return {"status": "APPROVED", "human_feedback": ""}
    if lower.startswith("reject"):
        return {"status": "REJECTED", "human_feedback": text}
    if lower.startswith("拒绝"):
        return {"status": "REJECTED", "human_feedback": text}
    return {"status": "REJECTED", "human_feedback": text}


def _build_intermediate_payload(state_values: Dict[str, Any], interrupted_node: str) -> Dict[str, Any]:
    shared_memory = state_values.get("shared_memory", {}) or {}
    draft = state_values.get("draft", []) or []
    stage = state_values.get("stage") or make_stage("unknown_stage", "Unknown Stage")
    node_to_key = {
        "Analysis": "analysis_report",
        "Modeling": "mathematical_model",
        "Coder": "generated_code",
        "Review": "review_report",
        "Writing": "paper_draft",
        "Export": "draft",
        "Respond": "draft",
    }
    key = node_to_key.get(interrupted_node)
    source = draft
    if key and key != "draft":
        source = shared_memory.get(key, draft) or draft

    source_blocks = ensure_blocks(source)
    summary = summary_from_blocks(source_blocks, max_len=180)
    # Reviewer 节点需要完整展示审查结论，便于用户判断是否回流，不做预览截断。
    if interrupted_node == "Review":
        preview_blocks = source_blocks
    else:
        preview_blocks = ensure_blocks(summary_from_blocks(source_blocks, max_len=600))
    return {
        "node": interrupted_node,
        "summary_blocks": ensure_blocks(summary or f"{interrupted_node} 阶段已完成，等待确认。"),
        "preview_blocks": preview_blocks or ensure_blocks("暂无可展示内容。"),
        "ask_continue": True,
        "stage": stage,
    }


def _is_terminal_state(final_state: Dict[str, Any]) -> bool:
    if not isinstance(final_state, dict):
        return False
    if final_state.get("next") == "END":
        return True

    status = final_state.get("status")
    stage_id = get_stage_id(final_state.get("stage"))
    if status != "APPROVED":
        return False

    end_stage_ids = SUPERVISOR_ROUTING_CONFIG.get("end_stage_ids", [])
    return stage_id in end_stage_ids


def _guess_intermediate_node(final_state: Dict[str, Any]) -> str:
    next_node = final_state.get("next")
    if isinstance(next_node, str) and next_node:
        return next_node

    stage_id = get_stage_id(final_state.get("stage"))
    if "analysis" in stage_id:
        return "Analysis"
    if "modeling" in stage_id:
        return "Modeling"
    if "coding" in stage_id or "coder" in stage_id:
        return "Coder"
    if "review" in stage_id:
        return "Review"
    if "paper" in stage_id or "writing" in stage_id:
        return "Writing"
    if "export" in stage_id or "asset" in stage_id:
        return "Export"
    if "respond" in stage_id or "consultation" in stage_id:
        return "Respond"
    return "Unknown"


def _fallback_blocks_from_state(final_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(final_state, dict):
        return []
    draft = final_state.get("draft")
    if draft:
        return ensure_blocks(draft)
    shared_memory = final_state.get("shared_memory", {}) or {}
    for key in ["paper_draft", "review_report", "generated_code", "mathematical_model", "analysis_report"]:
        value = shared_memory.get(key)
        if value:
            return ensure_blocks(value)
    return []


def _get_graph_app_from_state(app_like) -> Any:
    runtime = getattr(app_like.state, "graph_runtime", None)
    if runtime is None:
        raise RuntimeError("Graph runtime missing on app.state")
    return runtime.get_graph()


def _build_user_inputs(user_text: str, has_feedback: bool):
    if has_feedback:
        return None
    return {"messages": [{"role": "user", "content": ensure_blocks(user_text)}]}


@router.websocket("/ws/chat/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    logger.info("WS connected: client_id=%s", client_id)
    await manager.connect(websocket)
    sub_queue = await message_bus.subscribe()

    async def bus_listener():
        try:
            while True:
                msg = await sub_queue.get()
                await manager.send_json(msg, websocket)
        except asyncio.CancelledError:
            pass

    listener_task = asyncio.create_task(bus_listener())

    try:
        graph_app = _get_graph_app_from_state(websocket.app)
        while True:
            data = await websocket.receive_text()
            try:
                msg_data = json.loads(data)
            except json.JSONDecodeError:
                await manager.send_json({"type": "ERROR", "payload": {"blocks": ensure_blocks("格式非法")}}, websocket)
                continue

            user_message = msg_data.get("message", "")
            thread_id = msg_data.get("thread_id", client_id)
            human_feedback = msg_data.get("human_feedback")
            logger.info("WS inbound: thread_id=%s has_feedback=%s message_len=%s", thread_id, bool(human_feedback), len(user_message))

            with database.SessionLocal() as db:
                task = crud.get_task(db, thread_id)
                if not task:
                    await manager.send_json({"type": "ERROR", "payload": {"blocks": ensure_blocks("Task ID 无效")}}, websocket)
                    continue
                task_config = crud.get_task_runtime_config(db, thread_id)
                if user_message:
                    crud.create_message(db, task_id=thread_id, role="user", content=ensure_blocks(user_message))

            config = {"configurable": {"thread_id": thread_id}}
            inputs = _build_user_inputs(user_message, bool(human_feedback))

            logger.info("WS state update start: thread_id=%s has_feedback=%s", thread_id, bool(human_feedback))
            try:
                if human_feedback:
                    await asyncio.wait_for(
                        graph_app.aupdate_state(config, _normalize_feedback(human_feedback)),
                        timeout=STATE_UPDATE_TIMEOUT_SECONDS,
                    )
                else:
                    await asyncio.wait_for(
                        graph_app.aupdate_state(config, {"shared_memory": task_config or {}}),
                        timeout=STATE_UPDATE_TIMEOUT_SECONDS,
                    )
            except asyncio.TimeoutError:
                logger.error("WS state update timeout: thread_id=%s timeout=%ss", thread_id, STATE_UPDATE_TIMEOUT_SECONDS)
                await manager.send_json(
                    {
                        "type": "ERROR",
                        "payload": _structured_error_payload(
                            user_message="状态同步超时（常见于网络波动或服务繁忙），请稍后重试。",
                            code="WS_STATE_UPDATE_TIMEOUT",
                            category="network",
                            retriable=True,
                            detail=f"timeout={STATE_UPDATE_TIMEOUT_SECONDS}s",
                        ),
                    },
                    websocket,
                )
                continue
            logger.info("WS state update done: thread_id=%s", thread_id)

            ai_buffer: List[str] = []
            ai_saved = False

            try:
                logger.info("WS stream start: thread_id=%s", thread_id)
                event_stream = graph_app.astream_events(inputs, config=config, version="v2")
                logger.info(
                    "WS stream polling: first graph event may block until sync nodes "
                    "(e.g. Supervisor RAG embed / LLM route) yield; thread_id=%s",
                    thread_id,
                )
                while True:
                    try:
                        event = await asyncio.wait_for(
                            anext(event_stream),
                            timeout=STREAM_IDLE_TIMEOUT_SECONDS,
                        )
                    except StopAsyncIteration:
                        break
                    except asyncio.TimeoutError:
                        logger.error("WS stream idle timeout: thread_id=%s timeout=%ss", thread_id, STREAM_IDLE_TIMEOUT_SECONDS)
                        await manager.send_json(
                            {
                                "type": "ERROR",
                                "payload": _structured_error_payload(
                                    user_message=(
                                        "执行长时间无新事件（常见于网络不稳定或上游模型阻塞/重试中），"
                                        "请稍后重试；若任务较重可适当延长等待或缩小范围。"
                                    ),
                                    code="WS_STREAM_IDLE_TIMEOUT",
                                    category="network",
                                    retriable=True,
                                    detail=f"idle_no_event_for_{STREAM_IDLE_TIMEOUT_SECONDS}s",
                                ),
                            },
                            websocket,
                        )
                        break
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        content = message_to_text(event.get("data", {}).get("chunk"))
                        if content:
                            ai_buffer.append(content)
                            await manager.send_json(
                                {
                                    "type": "TOKEN",
                                    "node": event.get("metadata", {}).get("langgraph_node", "Unknown"),
                                    "payload": {"blocks": ensure_blocks(content)},
                                },
                                websocket,
                            )
                    elif kind == "on_chain_start":
                        chain_name = event.get("name", "")
                        if chain_name in ["Reader", "Analysis", "Modeling", "Coder", "Review", "Writing", "Export", "Supervisor"]:
                            await manager.send_json(
                                {
                                    "type": "STATUS",
                                    "node": chain_name,
                                    "status": "running",
                                    "payload": {
                                        "blocks": ensure_blocks(f"Started: {chain_name}"),
                                        "stage": make_stage(f"running_{chain_name.lower()}", f"Started: {chain_name}"),
                                    },
                                },
                                websocket,
                            )
                    elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                        final_state = event.get("data", {}).get("output", {})
                        final_blocks = resolve_final_blocks(final_state)
                        if blocks_effectively_empty(final_blocks) and ai_buffer:
                            final_blocks = ensure_blocks("".join(ai_buffer))
                        if blocks_effectively_empty(final_blocks):
                            final_blocks = _fallback_blocks_from_state(final_state)
                        final_text = blocks_to_text(final_blocks)
                        if _is_terminal_state(final_state):
                            with database.SessionLocal() as db:
                                if final_blocks and not blocks_effectively_empty(final_blocks):
                                    crud.create_message(db, task_id=thread_id, role="ai", content=final_blocks)
                                    ai_saved = True

                            await manager.send_json(
                                {
                                    "type": "FINAL",
                                    "payload": {
                                        "blocks": final_blocks,
                                        "paper_blocks": ensure_blocks((final_state.get("shared_memory", {}) or {}).get("paper_draft", "")),
                                        "stage": final_state.get("stage") or make_stage("task_completed", "Task Completed"),
                                    },
                                },
                                websocket,
                            )
                            logger.info(
                                "WS final sent: thread_id=%s response_len=%s stage=%s",
                                thread_id,
                                len(final_text),
                                get_stage_id(final_state.get("stage")),
                            )
                        else:
                            interrupted_node = _guess_intermediate_node(final_state)
                            await manager.send_json(
                                {
                                    "type": "INTERMEDIATE",
                                    "payload": _build_intermediate_payload(final_state, interrupted_node),
                                },
                                websocket,
                            )
                            logger.info(
                                "WS intermediate sent: thread_id=%s node=%s stage=%s",
                                thread_id,
                                interrupted_node,
                                get_stage_id(final_state.get("stage")),
                            )
            except GraphInterrupt:
                snapshot = await graph_app.aget_state(config)
                values = getattr(snapshot, "values", {}) or {}
                next_nodes = list(getattr(snapshot, "next", ()) or ())
                interrupted_node = next_nodes[0] if next_nodes else values.get("next", "Unknown")
                await manager.send_json(
                    {
                        "type": "INTERMEDIATE",
                        "payload": _build_intermediate_payload(values, interrupted_node),
                    },
                    websocket,
                )
            except Exception as stream_exc:
                logger.exception("WS stream failed: thread_id=%s", thread_id)
                err_info = _classify_runtime_error(stream_exc)
                if ai_buffer and not ai_saved:
                    partial_text = "".join(ai_buffer)
                    with database.SessionLocal() as db:
                        partial_blocks = ensure_blocks(partial_text)
                        crud.create_message(db, task_id=thread_id, role="ai", content=partial_blocks)
                    await manager.send_json(
                        {
                            "type": "FINAL",
                            "payload": {
                                "blocks": ensure_blocks(partial_text),
                                "stage": make_stage("interrupted_with_error", "Interrupted with error"),
                            },
                        },
                        websocket,
                    )
                await manager.send_json(
                    {
                        "type": "ERROR",
                        "payload": _structured_error_payload(
                            user_message=str(err_info.get("user_message", "执行中断，已保存当前生成内容。")),
                            code=str(err_info.get("code", "RUNTIME_ERROR")),
                            category=str(err_info.get("category", "runtime")),
                            retriable=bool(err_info.get("retriable", False)),
                            detail=str(err_info.get("detail", "") or ""),
                        ),
                    },
                    websocket,
                )
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WS disconnected: client_id=%s", client_id)
    finally:
        listener_task.cancel()
        message_bus.unsubscribe(sub_queue)


@router.post("/api/v1/chat")
async def chat_with_agent(item: ChatMessage, request: Request):
    try:
        graph_app = _get_graph_app_from_state(request.app)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    config = {"configurable": {"thread_id": item.thread_id}}
    inputs = _build_user_inputs(item.message, bool(item.human_feedback))
    res = await graph_app.ainvoke(inputs, config=config)
    return {"status": "success", "agent_response_blocks": resolve_final_blocks(res), "agent_response_text": extract_final_response(res)}
