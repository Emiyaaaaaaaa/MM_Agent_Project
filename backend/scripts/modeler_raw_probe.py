import argparse
import asyncio
import json
from typing import Any, Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.db import crud, database
from backend.services.serialization import extract_text_content


SYSTEM_PROMPT = (
    "你是一个顶级的 MCM/ICM 建模专家，擅长从定性分析转化为定量的数学模型。"
    "你的任务是根据审题报告 (Analysis Report)，深入推导具体的数学公式并选择最优算法。"
    "\n输出要求（使用 LaTeX 格式）：\n"
    "1. 算法选型与依据。\n"
    "2. 符号系统建立。\n"
    "3. 数学公式推导。\n"
    "4. 求解路线图。"
)


def _extract_candidate_summaries(response: Any) -> Dict[str, Any]:
    response_metadata = getattr(response, "response_metadata", {}) or {}
    additional_kwargs = getattr(response, "additional_kwargs", {}) or {}

    candidates = additional_kwargs.get("candidates")
    if not isinstance(candidates, list):
        candidates = response_metadata.get("candidates")
    if not isinstance(candidates, list):
        candidates = []

    summaries: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates[:5]):
        if isinstance(cand, dict):
            content = cand.get("content", {}) or {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
            summaries.append(
                {
                    "idx": idx,
                    "finish_reason": cand.get("finish_reason"),
                    "parts_count": len(parts),
                    "parts_keys_preview": [sorted(p.keys()) for p in parts[:3] if isinstance(p, dict)],
                }
            )
        else:
            summaries.append({"idx": idx, "type": type(cand).__name__})

    prompt_feedback = response_metadata.get("prompt_feedback", {}) or {}
    block_reason = ""
    if isinstance(prompt_feedback, dict):
        block_reason = str(prompt_feedback.get("block_reason", "") or "")

    return {
        "candidate_count": len(candidates),
        "candidate_summaries": summaries,
        "prompt_feedback_block_reason": block_reason,
        "response_metadata_keys": list(response_metadata.keys()),
        "additional_kwargs_keys": list(additional_kwargs.keys()),
    }


def _load_task_runtime(task_id: str) -> Dict[str, str]:
    with database.SessionLocal() as db:
        cfg = crud.get_task_runtime_config(db, task_id)
        if not cfg or not cfg.get("api_key"):
            raise ValueError(f"task_id={task_id} 未找到可用 api_key")
        messages = crud.get_task_messages(db, task_id)
        last_user = ""
        for msg in reversed(messages):
            if msg.role == "user" and (msg.content or "").strip():
                last_user = extract_text_content(msg.content).strip()
                break
        return {
            "api_key": cfg.get("api_key", ""),
            "model_id": cfg.get("model_id") or "gemini-2.5-flash-lite",
            "last_user": last_user,
        }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Probe raw model response around Modeler behavior")
    parser.add_argument("--task-id", required=True, help="Task id used to load api_key/model_id")
    parser.add_argument(
        "--analysis-report",
        default="请基于题意建立变量、目标函数、约束条件，并给出可执行求解路线。",
        help="Analysis report text injected into prompt",
    )
    parser.add_argument("--user-prompt", default="", help="Optional user prompt override")
    args = parser.parse_args()

    runtime = _load_task_runtime(args.task_id)
    user_prompt = args.user_prompt or runtime["last_user"] or "请给出完整数学建模推导。"
    safe_analysis = args.analysis_report[:2000].replace("{", "{{").replace("}", "}}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("system", f"【深度分析背景】\n{safe_analysis}..."),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    llm = ChatGoogleGenerativeAI(
        model=runtime["model_id"],
        temperature=0.2,
        google_api_key=runtime["api_key"],
    )
    chain = prompt | llm
    response = await chain.ainvoke({"messages": [HumanMessage(content=user_prompt)]})

    raw_content = getattr(response, "content", None)
    text = extract_text_content(raw_content).strip()
    diagnostics = _extract_candidate_summaries(response)

    print("===== MODELER RAW PROBE =====")
    print(f"model_id: {runtime['model_id']}")
    print(f"user_prompt_len: {len(user_prompt)}")
    print(f"content_type: {type(raw_content).__name__}")
    print(f"content_size: {len(raw_content) if isinstance(raw_content, (str, list, tuple, dict)) else 0}")
    print(f"text_len: {len(text)}")
    print(f"text_preview: {text[:300]}")
    print(f"diagnostics: {json.dumps(diagnostics, ensure_ascii=False)}")
    print("===== END =====")


if __name__ == "__main__":
    asyncio.run(main())
