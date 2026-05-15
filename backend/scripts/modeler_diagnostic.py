import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from unittest.mock import patch

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from backend.db import crud, database
from backend.graph.nodes.modeler import ModelerNode
from backend.services.serialization import get_stage_id


@dataclass
class FakeResponse:
    content: Any
    response_metadata: Dict[str, Any]


def _build_fake_runnable(response_factory: Callable[[], Any]):
    def _invoke(_inputs: Dict[str, Any]) -> Any:
        result = response_factory()
        if isinstance(result, Exception):
            raise result
        return result

    return RunnableLambda(_invoke)


def _build_state(api_key: str, analysis_report: str = "这是一份审题报告，包含变量、目标和约束。") -> Dict[str, Any]:
    return {
        "messages": [HumanMessage(content="请完成该题建模")],
        "human_feedback": "",
        "shared_memory": {
            "api_key": api_key,
            "model_id": "gemini-2.5-flash-lite",
            "analysis_report": analysis_report,
        },
    }


async def run_case(case_name: str, response_factory: Callable[[], Any]) -> Dict[str, Any]:
    node = ModelerNode()
    state = _build_state(api_key="FAKE_KEY")
    progress_events: List[Dict[str, Any]] = []

    async def fake_broadcast_progress(node_name: str, message: str, progress: int):
        progress_events.append({"node": node_name, "message": message, "progress": progress})

    with patch(
        "backend.graph.nodes.modeler.ChatGoogleGenerativeAI",
        side_effect=lambda **_kwargs: _build_fake_runnable(response_factory),
    ):
        with patch("backend.graph.nodes.modeler.broadcast_progress", side_effect=fake_broadcast_progress):
            try:
                output = await node(state)
                return {
                    "case": case_name,
                    "ok": True,
                    "output": output,
                    "progress_events": progress_events,
                }
            except Exception as exc:
                return {
                    "case": case_name,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "progress_events": progress_events,
                }


async def run_missing_key_case() -> Dict[str, Any]:
    node = ModelerNode()
    state = _build_state(api_key="")
    progress_events: List[Dict[str, Any]] = []

    async def fake_broadcast_progress(node_name: str, message: str, progress: int):
        progress_events.append({"node": node_name, "message": message, "progress": progress})

    with patch("backend.graph.nodes.modeler.broadcast_progress", side_effect=fake_broadcast_progress):
        output = await node(state)
        return {
            "case": "missing_api_key",
            "ok": True,
            "output": output,
            "progress_events": progress_events,
        }


async def run_real_case(api_key: str, model_id: str, analysis_report: str, user_prompt: str) -> Dict[str, Any]:
    node = ModelerNode()
    state = {
        "messages": [HumanMessage(content=user_prompt)],
        "human_feedback": "",
        "shared_memory": {
            "api_key": api_key,
            "model_id": model_id,
            "analysis_report": analysis_report,
        },
    }

    progress_events: List[Dict[str, Any]] = []

    async def fake_broadcast_progress(node_name: str, message: str, progress: int):
        progress_events.append({"node": node_name, "message": message, "progress": progress})

    with patch("backend.graph.nodes.modeler.broadcast_progress", side_effect=fake_broadcast_progress):
        try:
            output = await node(state)
            return {
                "case": "real_model_call",
                "ok": True,
                "output": output,
                "progress_events": progress_events,
            }
        except Exception as exc:
            return {
                "case": "real_model_call",
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "progress_events": progress_events,
            }


def load_task_runtime_config(task_id: str) -> Dict[str, str]:
    with database.SessionLocal() as db:
        runtime_cfg = crud.get_task_runtime_config(db, task_id) or {}
        if not runtime_cfg.get("api_key"):
            raise ValueError(f"task_id={task_id} 未找到可用 api_key，请在任务配置中检查。")

        # 仅做辅助，不强依赖历史消息存在
        messages = crud.get_task_messages(db, task_id)
        last_user_message = ""
        for msg in reversed(messages):
            if msg.role == "user" and (msg.content or "").strip():
                last_user_message = msg.content.strip()
                break

        return {
            "api_key": runtime_cfg.get("api_key", ""),
            "model_id": runtime_cfg.get("model_id") or "gemini-2.5-flash-lite",
            "last_user_message": last_user_message,
        }


def _shorten(value: Any, max_len: int = 300) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...(truncated)"


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for k, v in value.items():
            key_lower = str(k).lower()
            if "api_key" in key_lower or "token" in key_lower or "secret" in key_lower:
                redacted[k] = "***REDACTED***"
            else:
                redacted[k] = _redact_secrets(v)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def print_report(results: List[Dict[str, Any]]) -> None:
    print("\n===== MODELER NODE DIAGNOSTIC REPORT =====")
    for item in results:
        print(f"\n[CASE] {item['case']}")
        print(f"  ok: {item['ok']}")
        print(f"  progress_events: {len(item.get('progress_events', []))}")
        if item["ok"]:
            output = _redact_secrets(item.get("output", {}))
            stage = get_stage_id(output.get("stage"))
            status = output.get("status", "")
            draft_len = len(output.get("draft", "") or "")
            has_model = bool((output.get("shared_memory", {}) or {}).get("mathematical_model"))
            has_raw_modeling = "modeling_raw_content" in (output.get("shared_memory", {}) or {})
            print(f"  status: {status}")
            print(f"  stage: {stage}")
            print(f"  draft_len: {draft_len}")
            print(f"  has_mathematical_model: {has_model}")
            print(f"  has_modeling_raw_content: {has_raw_modeling}")
            print(f"  output_preview: {_shorten(output, 260)}")
        else:
            print(f"  error_type: {item.get('error_type')}")
            print(f"  error: {item.get('error')}")

    # 结论规则
    list_case = next((r for r in results if r["case"] == "mock_content_list"), None)
    exception_case = next((r for r in results if r["case"] == "mock_llm_exception"), None)
    if list_case and list_case.get("ok"):
        output = list_case.get("output", {})
        shared_memory = output.get("shared_memory", {}) or {}
        if get_stage_id(output.get("stage")) == "modeling_failed_empty_output":
            print("\n[KEY FINDING] response.content 为 list 时，当前实现仍会被判定为空输出，需要继续修复。")
        elif output.get("status") == "PENDING" and shared_memory.get("modeling_raw_content") is not None:
            print("\n[KEY FINDING] response.content 为 list 时，已可成功提取文本并保留原结构数据。")
    if exception_case and not exception_case.get("ok"):
        print("[KEY FINDING] LLM 调用异常未在 Modeler 内部兜底，异常会直接抛出到上游。")
    print("===== END OF REPORT =====\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated diagnostic for backend.graph.nodes.modeler.ModelerNode")
    parser.add_argument("--run-real", action="store_true", help="Run one real online call in addition to mock tests")
    parser.add_argument("--task-id", type=str, default="", help="Load api_key/model_id from task runtime config")
    parser.add_argument("--api-key", type=str, default="", help="API key for real call")
    parser.add_argument("--model-id", type=str, default="gemini-2.5-flash-lite", help="Model id for real call")
    parser.add_argument(
        "--analysis-report",
        type=str,
        default="目标：对给定数据建立预测模型，给出变量定义、目标函数、约束条件与可计算流程。",
        help="Analysis report text used in real call",
    )
    parser.add_argument("--user-prompt", type=str, default="", help="User message for real call")
    args = parser.parse_args()

    tests = [
        run_missing_key_case(),
        run_case(
            "mock_content_empty_string",
            lambda: FakeResponse(content="", response_metadata={"finish_reason": "STOP"}),
        ),
        run_case(
            "mock_content_list",
            lambda: FakeResponse(
                content=[{"type": "text", "text": "这里其实有建模内容，但 content 是 list 结构。"}],
                response_metadata={"finish_reason": "STOP"},
            ),
        ),
        run_case(
            "mock_content_string",
            lambda: FakeResponse(content="这是建模正文，包含公式 $J(\\theta)$。", response_metadata={"finish_reason": "STOP"}),
        ),
        run_case("mock_llm_exception", lambda: RuntimeError("simulated llm transport error")),
    ]
    results = await asyncio.gather(*tests)

    if args.run_real:
        resolved_api_key = args.api_key
        resolved_model_id = args.model_id
        resolved_user_prompt = args.user_prompt
        if args.task_id:
            cfg = load_task_runtime_config(args.task_id)
            resolved_api_key = cfg["api_key"]
            resolved_model_id = cfg["model_id"]
            if not args.user_prompt and cfg["last_user_message"]:
                resolved_user_prompt = cfg["last_user_message"]

        if not resolved_api_key:
            print("[WARN] --run-real 已指定，但未提供 --api-key 且 --task-id 不可用，跳过真实调用。")
        else:
            real_result = await run_real_case(
                api_key=resolved_api_key,
                model_id=resolved_model_id,
                analysis_report=args.analysis_report,
                user_prompt=resolved_user_prompt or "请给出可执行的建模方案与公式推导。",
            )
            results.append(real_result)

    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())
