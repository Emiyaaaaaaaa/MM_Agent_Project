import json
from typing import Any, Dict, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def make_text_block(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text or ""}


def normalize_block(block: Dict[str, Any]) -> Dict[str, Any]:
    block_type = str(block.get("type", "text")).strip().lower() or "text"
    if block_type == "text":
        return {"type": "text", "text": str(block.get("text", block.get("content", "")))}
    if block_type == "markdown":
        return {"type": "markdown", "text": str(block.get("text", block.get("content", "")))}
    if block_type == "code":
        return {
            "type": "code",
            "code": str(block.get("code", block.get("text", ""))),
            "language": str(block.get("language", "text")),
        }
    if block_type == "math":
        return {"type": "math", "latex": str(block.get("latex", block.get("text", "")))}
    if block_type == "image":
        return {"type": "image", "url": str(block.get("url", "")), "alt": str(block.get("alt", ""))}
    if block_type == "table":
        headers = block.get("headers", [])
        rows = block.get("rows", [])
        return {"type": "table", "headers": json_safe(headers), "rows": json_safe(rows)}
    # 未知块类型不再回调 extract_text_content，避免 normalize -> blocks_to_text -> ensure -> normalize 递归。
    try:
        safe_text = json.dumps(json_safe(block), ensure_ascii=False)
    except Exception:
        safe_text = str(block)
    return {"type": "text", "text": safe_text}


def ensure_blocks(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        return [make_text_block(value)]
    if isinstance(value, list):
        if value and all(isinstance(item, dict) and "type" in item for item in value):
            return [normalize_block(item) for item in value]
        parts: List[Dict[str, Any]] = []
        for item in value:
            parts.extend(ensure_blocks(item))
        return parts
    if isinstance(value, dict):
        if "type" in value:
            return [normalize_block(value)]
        if "content" in value:
            return ensure_blocks(value.get("content"))
        if "text" in value:
            return [make_text_block(str(value.get("text", "")))]
        # 普通 dict（非 content/text/type）直接序列化为文本，避免 ensure -> extract -> blocks_to_text -> ensure 递归。
        try:
            safe_text = json.dumps(json_safe(value), ensure_ascii=False)
        except Exception:
            safe_text = str(value)
        return [make_text_block(safe_text)]
    content = getattr(value, "content", None)
    if content is not None:
        return ensure_blocks(content)
    return [make_text_block(str(value))]


def blocks_to_text(blocks: Any) -> str:
    parsed_blocks = ensure_blocks(blocks)
    parts: List[str] = []
    for block in parsed_blocks:
        block_type = block.get("type")
        if block_type in {"text", "markdown"}:
            parts.append(str(block.get("text", "")))
        elif block_type == "code":
            parts.append(str(block.get("code", "")))
        elif block_type == "math":
            parts.append(str(block.get("latex", "")))
        elif block_type == "image":
            alt = str(block.get("alt", ""))
            url = str(block.get("url", ""))
            parts.append(alt or url)
        elif block_type == "table":
            parts.append(str(block.get("rows", "")))
    return "\n".join(part for part in parts if part).strip()


def blocks_effectively_empty(blocks: Any) -> bool:
    """用于判断最终展示块是否仅有空白（避免把 [ {type:text,text:''} ] 当成有内容）。"""
    return not bool(blocks_to_text(blocks).strip())


def extract_text_content(value: Any) -> str:
    return blocks_to_text(value)


def summary_from_blocks(value: Any, max_len: int = 180) -> str:
    text = blocks_to_text(value)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    content = getattr(value, "content", None)
    if content is not None:
        return json_safe(content)
    return str(value)


def make_stage(stage_id: str, label: str = "", meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "id": stage_id,
        "label": label or stage_id,
        "meta": meta or {},
    }


def get_stage_id(stage: Any) -> str:
    if isinstance(stage, dict):
        return str(stage.get("id", ""))
    if isinstance(stage, str):
        return stage
    return ""


def message_to_text(message_like: Any) -> str:
    return blocks_to_text(message_like)


def _clip_text(text: str, max_chars: int | None = None) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 20:
        return text[:max_chars]
    head = max_chars // 2
    tail = max_chars - head - 15
    return f"{text[:head]}\n...[omitted]...\n{text[-tail:]}"


def messages_for_llm(
    messages: Sequence[Any],
    *,
    max_messages: int | None = None,
    max_chars_per_message: int | None = None,
) -> List[BaseMessage]:
    """
    将图状态中的 messages 转为 Gemini/OpenAI 可接受的纯文本 LangChain 消息。
    避免 markdown/code/image 等结构化块触发 Unrecognized message part type。
    """
    if not messages:
        return []
    selected_messages = list(messages)
    if max_messages is not None and max_messages > 0:
        selected_messages = selected_messages[-max_messages:]
    out: List[BaseMessage] = []
    for msg in selected_messages:
        if isinstance(msg, HumanMessage):
            text = _clip_text(extract_text_content(msg.content).strip(), max_chars_per_message) or " "
            out.append(HumanMessage(content=text))
            continue
        if isinstance(msg, AIMessage):
            text = _clip_text(extract_text_content(msg.content).strip(), max_chars_per_message) or " "
            out.append(AIMessage(content=text))
            continue

        role = ""
        if isinstance(msg, dict):
            role = str(msg.get("role", "") or "")
            raw_content = msg.get("content", "")
        else:
            msg_type = getattr(msg, "type", None)
            if msg_type in {"human", "user"}:
                role = "user"
            elif msg_type in {"ai", "assistant"}:
                role = "ai"
            else:
                role = str(msg_type or "ai")
            raw_content = getattr(msg, "content", msg)

        text = _clip_text(extract_text_content(raw_content).strip(), max_chars_per_message) or " "
        if role in {"user", "human"}:
            out.append(HumanMessage(content=text))
        else:
            out.append(AIMessage(content=text))
    return out


def extract_final_blocks(final_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = final_state.get("messages", [])
    if not messages:
        return []
    last = messages[-1]
    if isinstance(last, dict):
        return ensure_blocks(last.get("content", []))
    return ensure_blocks(last)


def resolve_final_blocks(final_state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    合并最终展示块：优先最后一条 AI 消息；若仅有空块则回退到 draft / shared_memory 产物。
    解决 Export 后 messages 仍为 Writer 空占位、而真实说明在 draft 的情况。
    """
    blocks = extract_final_blocks(final_state)
    if not blocks_effectively_empty(blocks):
        return blocks
    draft = final_state.get("draft")
    if draft and not blocks_effectively_empty(draft):
        return ensure_blocks(draft)
    shared_memory = final_state.get("shared_memory", {}) or {}
    for key in ["paper_draft", "review_report", "generated_code", "mathematical_model", "analysis_report"]:
        value = shared_memory.get(key)
        if value and not blocks_effectively_empty(value):
            return ensure_blocks(value)
    return blocks


def extract_final_response(final_state: Dict[str, Any]) -> str:
    return blocks_to_text(resolve_final_blocks(final_state))
