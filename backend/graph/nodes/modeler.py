import json
import asyncio
import logging
from typing import Any, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress
from backend.services.serialization import extract_text_content, json_safe, ensure_blocks, make_stage

logger = logging.getLogger("mm-agent")


def _raw_response_diagnostics(response: Any) -> Dict[str, Any]:
    response_metadata = getattr(response, "response_metadata", {}) or {}
    additional_kwargs = getattr(response, "additional_kwargs", {}) or {}

    candidates = additional_kwargs.get("candidates")
    if not isinstance(candidates, list):
        candidates = response_metadata.get("candidates")
    if not isinstance(candidates, list):
        candidates = []

    candidate_summaries: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates[:3]):
        if isinstance(cand, dict):
            content = cand.get("content", {}) or {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
            part_keys = []
            for part in parts[:5]:
                if isinstance(part, dict):
                    part_keys.append(sorted(part.keys()))
                else:
                    part_keys.append([type(part).__name__])
            candidate_summaries.append(
                {
                    "idx": idx,
                    "finish_reason": cand.get("finish_reason"),
                    "safety_ratings_count": len(cand.get("safety_ratings", []) or []),
                    "parts_count": len(parts),
                    "parts_keys_preview": part_keys,
                }
            )
        else:
            candidate_summaries.append({"idx": idx, "type": type(cand).__name__})

    prompt_feedback = response_metadata.get("prompt_feedback", {}) or {}
    block_reason = ""
    if isinstance(prompt_feedback, dict):
        block_reason = str(prompt_feedback.get("block_reason", "") or "")

    return {
        "candidate_count": len(candidates),
        "candidate_summaries": candidate_summaries,
        "prompt_feedback_block_reason": block_reason,
        "additional_kwargs_keys": list(additional_kwargs.keys()),
    }


def _build_clean_modeler_messages(state: AgentState) -> List[HumanMessage]:
    raw_messages = list(state.get("messages", []) or [])
    last_user_text = ""
    message_shapes: List[Dict[str, Any]] = []
    for msg in raw_messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content")
        else:
            role = getattr(msg, "type", "")
            content = getattr(msg, "content", None)
        message_shapes.append(
            {
                "role": str(role),
                "content_type": type(content).__name__,
                "content_len": len(content) if isinstance(content, (str, list, tuple, dict)) else 0,
            }
        )
        if role in {"human", "user"}:
            text = extract_text_content(msg).strip()
            if text:
                last_user_text = text
    logger.info(
        "Modeling input diagnostics original_message_count=%s message_shapes=%s clean_user_len=%s",
        len(raw_messages),
        json.dumps(message_shapes[-8:], ensure_ascii=False),
        len(last_user_text),
    )
    if not last_user_text:
        last_user_text = "请基于审题报告完成数学建模，输出算法选型、符号表、公式推导与求解路线。"
    return [HumanMessage(content=last_user_text)]

class ModelerNode:
    """
    数学建模专家节点 (Modeler)：负责将定性分析转化为定量的数学表述。
    该节点专注于算法选型、符号定义以及严格的数学公式推导。
    """
    def __init__(self):
        # 建模节点建议使用更强的推理能力
        self.system_prompt = (
            "你是一个顶级的 MCM/ICM 建模专家，擅长从定性分析转化为定量的数学模型。"
            "你的任务是根据审题报告 (Analysis Report)，深入推导具体的数学公式并选择最优算法。"
            "\n输出要求（使用 LaTeX 格式）：\n"
            "1. **算法选型与依据**：详细说明为什么选择该算法（如：时间序列、评价模型、神经网络），分析其对本题的适配性。\n"
            "2. **符号系统建立**：列出模型中涉及的所有数学符号及其物理/实际含义（表格形式）。\n"
            "3. **数学公式推导**：给出详细的推导过程，包括目标函数 (Objective Function)、约束条件 (Constraints) 或微分方程组 (ODEs)。\n"
            "4. **求解路线图**：描述具体的计算方案，为程序员节点 (Coder Node) 提供伪代码或数值求解思路。"
            "- **版权屏蔽约束**：严禁在生成内容中出现具体参考论文的队号、年份或 O 奖等标识。使用‘学术通用模型’或‘经典建模处理’等中性词汇代称。\n"
            "- **文风强制要求**：语言和文字风格要符合各个获奖论文严谨冷静的范式，不要使用对表达论文内容来说不必要的比喻以及其他修辞，也不要使用过于抽象的合成词语或者自造词语\n"
            "- **Fusion 强约束**：若输入包含 Fusion 指南，必须体现在算法选型、变量定义和结果验证设计中。"
        )

    async def __call__(self, state: AgentState):
        """执行数学建模与公式推导逻辑"""
        shared_mem = state.get("shared_memory", {})
        analysis_report = extract_text_content(shared_mem.get("analysis_report", "（暂无审题报告）"))
        feedback = state.get("human_feedback", "")
        
        # 1. 广播进度
        await broadcast_progress("Modeling", "正在基于审题报告提取关键变量与物理关系...", 20)
        
        # 2. 动态模型实例化 (从任务配置加载)
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-2.5-flash-lite"
        
        if not api_key:
            return {
                "status": "REJECTED",
                "human_feedback": "缺少任务 API Key，Modeling 节点无法执行。",
                "stage": make_stage("modeling_failed_missing_api_key", "Modeling Failed: Missing Task API Key"),
            }

        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0.2, 
            google_api_key=api_key
        )
        
        # 构造上下文 (转义花括号以防 LaTeX/JSON 干扰 LangChain)
        safe_analysis = analysis_report[:2000].replace("{", "{{").replace("}", "}}")
        prompt_parts = [
            ("system", self.system_prompt),
            ("system", f"【深度分析背景】\n{safe_analysis}...")
        ]
        fusion_guidance = extract_text_content(shared_mem.get("fusion_guidance", "")) or extract_text_content(state.get("context", ""))
        if fusion_guidance:
            safe_fusion = fusion_guidance[:3800].replace("{", "{{").replace("}", "}}")
            prompt_parts.append(
                (
                    "system",
                    "【Fusion 建模约束（CRITICAL）】\n"
                    f"{safe_fusion}\n\n"
                    "输出要求：在末尾增加“Fusion建模映射”，至少列 3 条你实际采用的融合建模先验。"
                )
            )
        
        if feedback:
            prompt_parts.append(("system", f"【用户最新指导建议】\n{feedback}"))
            
        alignment_record = shared_mem.get("alignment_record", {})
        if alignment_record:
            align_data = json.dumps(alignment_record, ensure_ascii=False, indent=2).replace("{", "{{").replace("}", "}}")
            align_prompt = (
                f"【全局防漂移强制对齐约束 (CRITICAL)】\n"
                f"{align_data}"
            )
            prompt_parts.append(("system", align_prompt))
            
        prompt_parts.append(MessagesPlaceholder(variable_name="messages"))
        
        # 3. 广播进度
        await broadcast_progress("Modeling", f"正在使用 {model_id} 进行 LaTeX 公式推导与算法选型...", 60)
        
        prompt = ChatPromptTemplate.from_messages(prompt_parts)
        chain = prompt | llm
        clean_messages = _build_clean_modeler_messages(state)
        
        response = await chain.ainvoke({"messages": clean_messages})
        response_metadata = getattr(response, "response_metadata", {}) or {}
        finish_reason = response_metadata.get("finish_reason", "")
        safety_ratings = response_metadata.get("safety_ratings", [])
        prompt_feedback = response_metadata.get("prompt_feedback", {})
        raw_content = getattr(response, "content", None)
        content_type = type(raw_content).__name__
        content_size = len(raw_content) if isinstance(raw_content, (str, list, tuple, dict)) else 0
        modeling_text = extract_text_content(raw_content).strip()
        metadata_preview = json.dumps(response_metadata, ensure_ascii=False)[:1200]
        if isinstance(raw_content, str):
            content_preview = raw_content[:200]
        elif isinstance(raw_content, list):
            content_preview = json.dumps(raw_content[:2], ensure_ascii=False)[:200]
        elif isinstance(raw_content, dict):
            content_preview = json.dumps(raw_content, ensure_ascii=False)[:200]
        else:
            content_preview = str(raw_content)[:200]
        logger.info(
            "Modeling output length=%s model_id=%s finish_reason=%s content_type=%s content_size=%s metadata_keys=%s",
            len(modeling_text),
            model_id,
            finish_reason,
            content_type,
            content_size,
            list(response_metadata.keys()),
        )
        logger.info(
            "Modeling diagnostics model_id=%s finish_reason=%s safety_ratings=%s prompt_feedback=%s metadata_preview=%s content_preview=%s",
            model_id,
            finish_reason,
            safety_ratings,
            prompt_feedback,
            metadata_preview,
            content_preview,
        )
        raw_diag = _raw_response_diagnostics(response)
        logger.info(
            "Modeling raw-response diagnostics model_id=%s candidate_count=%s block_reason=%s additional_keys=%s candidate_summaries=%s",
            model_id,
            raw_diag.get("candidate_count"),
            raw_diag.get("prompt_feedback_block_reason", ""),
            raw_diag.get("additional_kwargs_keys", []),
            json.dumps(raw_diag.get("candidate_summaries", []), ensure_ascii=False)[:1500],
        )
        # 空输出兜底：先同模型重试，再降级到稳定模型重试一次
        if not modeling_text:
            logger.warning("Modeling empty output on first attempt, retrying with same model=%s", model_id)
            retry_prompt = ChatPromptTemplate.from_messages(
                prompt_parts
                + [
                    (
                        "system",
                        "上一次输出为空。请务必直接输出完整建模内容（算法选型、符号表、公式推导、求解路线），不要返回空字符串。",
                    )
                ]
            )
            retry_chain = retry_prompt | ChatGoogleGenerativeAI(
                model=model_id,
                temperature=0.1,
                google_api_key=api_key,
            )
            retry_response = await retry_chain.ainvoke({"messages": clean_messages})
            retry_raw = getattr(retry_response, "content", None)
            retry_text = extract_text_content(retry_raw).strip()
            if retry_text:
                response = retry_response
                raw_content = retry_raw
                modeling_text = retry_text
                response_metadata = getattr(retry_response, "response_metadata", {}) or {}
                logger.info("Modeling retry succeeded with model=%s length=%s", model_id, len(modeling_text))
            else:
                fallback_model_id = "gemini-2.5-flash-lite"
                if model_id != fallback_model_id:
                    logger.warning(
                        "Modeling retry still empty on model=%s, fallback to model=%s",
                        model_id,
                        fallback_model_id,
                    )
                    fallback_chain = retry_prompt | ChatGoogleGenerativeAI(
                        model=fallback_model_id,
                        temperature=0.1,
                        google_api_key=api_key,
                    )
                    fallback_response = await fallback_chain.ainvoke({"messages": clean_messages})
                    fallback_raw = getattr(fallback_response, "content", None)
                    fallback_text = extract_text_content(fallback_raw).strip()
                    if fallback_text:
                        response = fallback_response
                        raw_content = fallback_raw
                        modeling_text = fallback_text
                        response_metadata = getattr(fallback_response, "response_metadata", {}) or {}
                        logger.info(
                            "Modeling fallback succeeded with model=%s length=%s",
                            fallback_model_id,
                            len(modeling_text),
                        )
        if not modeling_text:
            failed_memory = shared_mem.copy()
            failed_memory["last_failure_stage_id"] = "modeling_failed_empty_output"
            await broadcast_progress("Modeling", "建模节点未生成有效内容，请重试或切换模型。", 100)
            return {
                "shared_memory": failed_memory,
                "status": "REJECTED",
                "human_feedback": "Modeling 节点空输出，请重试或切换模型后继续。",
                "stage": make_stage("modeling_failed_empty_output", "Modeling Failed: Empty Output"),
            }
        
        # 4. 广播进度
        await broadcast_progress("Modeling", "数学模型推导完毕，已生成 LaTeX 配套文档。", 100)
        
        new_memory = shared_mem.copy()
        new_memory.pop("last_failure_stage", None)
        new_memory.pop("last_failure_stage_id", None)
        new_memory["mathematical_model"] = modeling_text
        new_memory["modeling_raw_content"] = json_safe(raw_content)
        new_memory["modeling_response_metadata"] = json_safe(response_metadata)
        
        return {
            "messages": [{"role": "ai", "content": ensure_blocks(modeling_text)}],
            "shared_memory": new_memory,
            "status": "PENDING",
            "draft": ensure_blocks(modeling_text),
            "stage": make_stage("modeling_completed", "Mathematical Modeling Completed"),
        }

# 单例提供
modeler_node = ModelerNode()
