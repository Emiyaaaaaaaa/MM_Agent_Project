import json
from typing import Any, Dict, List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress
from backend.services.serialization import ensure_blocks, extract_text_content, json_safe, make_stage


class WriterNode:
    """
    学术写作专家节点（分块写作版）：
    - 按章节独立生成正文，而非一次性生成后追加扩写。
    - 每个章节设置最小字数约束，避免短章导致总文篇幅不足。
    - 每个章节都使用统一上游上下文（Analysis/Modeling/Coder/Review/Artifacts）。
    """

    def __init__(self):
        self.system_prompt = (
            "你是 MCM/ICM 建模论文写作专家。你的任务是基于建模流程产物生成工程化、可复核的论文正文。\n"
            "写作约束：\n"
            "1) 使用严谨、专业、可验证的技术表达，不使用修辞化语言。\n"
            "2) 每章必须引用上游分析、建模、仿真和审查信息，禁止脱离上下文泛写。\n"
            "3) 不输出无关说明，不输出提示词，不输出章节外内容。\n"
            "4) 严禁披露参考资料中的具体队号、年份或获奖等级。\n"
        )
        self.section_specs: List[Tuple[str, int, str]] = [
            ("Abstract", 350, "概述问题、方法、核心结果与结论，给出量化结论摘要。"),
            ("Introduction", 500, "说明问题背景、研究目标、技术路线和本文贡献。"),
            ("Assumptions and Problem Restatement", 450, "给出建模假设、变量定义、问题重述与约束条件。"),
            ("Notation and Data Preparation", 350, "定义符号体系、数据预处理、指标构建与归一化流程。"),
            ("Methodology", 1100, "详细说明模型构建、目标函数、约束、求解流程和算法设计。"),
            ("Results and Validation", 950, "基于仿真结果展开定量分析，引用图表并解释关键现象。"),
            ("Sensitivity and Robustness Analysis", 550, "开展敏感性、稳健性与误差来源分析。"),
            ("Discussion and Limitations", 400, "讨论模型适用边界、局限和可改进方向。"),
            ("Conclusion", 300, "总结方法有效性、主要发现和实际应用建议。"),
        ]
        self.max_section_retry = 2

    def _collect_image_blocks_from_manifest(self, manifest_like: Any) -> List[Dict[str, Any]]:
        manifest = manifest_like if isinstance(manifest_like, list) else []
        image_blocks: List[Dict[str, Any]] = []
        for item in manifest:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", "")).lower()
            url = str(item.get("url", "")).strip()
            if kind != "image" or not url:
                continue
            filename = str(item.get("filename", "")).strip() or "result_image"
            image_blocks.append({"type": "image", "url": url, "alt": filename})
            image_blocks.append({"type": "text", "text": f"图注：{filename}"})
        return image_blocks

    def _collect_export_refs_from_manifest(self, manifest_like: Any) -> List[str]:
        manifest = manifest_like if isinstance(manifest_like, list) else []
        refs: List[str] = []
        for item in manifest:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", "")).lower()
            if kind == "image":
                continue
            filename = str(item.get("filename", "")).strip()
            url = str(item.get("url", "")).strip()
            if filename and url:
                refs.append(f"- {filename}: {url}")
        return refs

    def _build_context(self, state: AgentState, shared_mem: Dict[str, Any]) -> str:
        analysis = extract_text_content(shared_mem.get("analysis_report", ""))[:2600]
        model = extract_text_content(shared_mem.get("mathematical_model", ""))[:3600]
        review = extract_text_content(shared_mem.get("review_report", ""))[:2200]
        code = extract_text_content(shared_mem.get("generated_code", ""))[:1800]
        logs = extract_text_content(shared_mem.get("execution_logs", ""))[:1600]
        rag_context = extract_text_content(state.get("context", ""))[:2600]
        fusion_guidance = extract_text_content(shared_mem.get("fusion_guidance", ""))[:4200]
        fusion_chart_plan = shared_mem.get("fusion_chart_plan", [])
        manifest = shared_mem.get("artifacts_manifest", [])
        image_names = ", ".join(
            [
                str((item or {}).get("filename", ""))
                for item in (manifest if isinstance(manifest, list) else [])
                if isinstance(item, dict) and str(item.get("kind", "")).lower() == "image"
            ]
        )[:1200]
        export_refs = "\n".join(self._collect_export_refs_from_manifest(manifest))[:1800]
        chart_plan_text = ""
        if isinstance(fusion_chart_plan, list) and fusion_chart_plan:
            try:
                chart_plan_text = json.dumps(fusion_chart_plan, ensure_ascii=False, indent=2)[:2600]
            except Exception:
                chart_plan_text = str(fusion_chart_plan)[:2600]

        context = (
            f"【Analysis 输出】\n{analysis or '暂无'}\n\n"
            f"【Modeling 输出】\n{model or '暂无'}\n\n"
            f"【Coder 代码摘要】\n{code or '暂无'}\n\n"
            f"【执行日志摘要】\n{logs or '暂无'}\n\n"
            f"【Review 输出】\n{review or '暂无'}\n\n"
            f"【图表资产】\n{image_names or '暂无图表资产'}\n\n"
            f"【非图像导出资产】\n{export_refs or '暂无导出资产'}\n\n"
            f"【Fusion 图表计划（CHART_PLAN）】\n{chart_plan_text or '暂无图表计划'}\n\n"
            f"【Fusion 强约束指南】\n{fusion_guidance or rag_context or '暂无'}\n\n"
            f"【RAG 参考上下文】\n{rag_context or '暂无'}\n"
        )
        return context.replace("{", "{{").replace("}", "}}")

    def _normalize_section_text(self, section_title: str, text: str) -> str:
        body = (text or "").strip()
        if not body:
            return f"## {section_title}\n\n（本节生成失败，待补充）"
        lower = body.lower()
        if lower.startswith(f"## {section_title.lower()}") or lower.startswith(f"# {section_title.lower()}"):
            return body
        return f"## {section_title}\n\n{body}"

    async def __call__(self, state: AgentState):
        shared_mem = state.get("shared_memory", {}) or {}
        artifacts_manifest = shared_mem.get("artifacts_manifest", [])
        image_blocks = self._collect_image_blocks_from_manifest(artifacts_manifest)
        global_context = self._build_context(state, shared_mem)

        await broadcast_progress("Writer", "正在构建分章节写作上下文...", 10)

        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-2.5-flash-lite"
        if not api_key:
            return {
                "status": "REJECTED",
                "human_feedback": "缺少任务 API Key，Writer 节点无法执行。",
                "stage": make_stage("writer_failed_missing_api_key", "Writer Failed: Missing Task API Key"),
            }

        llm = ChatGoogleGenerativeAI(model=model_id, temperature=0.5, google_api_key=api_key)

        prompt_parts = [
            ("system", self.system_prompt),
            ("system", f"【全局上游上下文】\n{global_context}"),
        ]
        alignment_record = shared_mem.get("alignment_record", {})
        if alignment_record:
            align_data = json.dumps(alignment_record, ensure_ascii=False, indent=2).replace("{", "{{").replace("}", "}}")
            prompt_parts.append(("system", f"【全局防漂移约束】\n{align_data}"))
        prompt_parts.append(MessagesPlaceholder(variable_name="messages"))

        prompt = ChatPromptTemplate.from_messages(prompt_parts)
        chain = prompt | llm

        base_messages = list(state.get("messages", []))
        section_blocks: List[Dict[str, Any]] = []
        section_stats: Dict[str, Any] = {}
        final_response_metadata: Dict[str, Any] = {}
        final_raw_content: Any = None

        total_sections = len(self.section_specs)
        for idx, (section_title, min_chars, section_goal) in enumerate(self.section_specs, start=1):
            progress = min(90, 10 + int((idx - 1) * (70 / max(total_sections, 1))))
            await broadcast_progress("Writer", f"正在撰写章节 {idx}/{total_sections}: {section_title}", progress)

            best_text = ""
            best_len = 0
            attempt_meta: Dict[str, Any] = {}
            for attempt in range(1, self.max_section_retry + 1):
                section_prompt = (
                    f"请仅输出章节《{section_title}》正文。\n"
                    f"章节目标：{section_goal}\n"
                    f"最小字数要求：不少于 {min_chars} 字。\n"
                    "输出格式要求：以 `## 章节名` 开头，随后给出完整章节内容。\n"
                    "必须引用并使用上游上下文信息，不得泛化写作。\n"
                    "强制要求：本章末尾增加“Fusion对齐说明”小段，至少列出 2 条本章采用的 Fusion 要点。\n"
                    "若本章涉及方法或结果，必须增加“图表映射说明”：说明本章对应的图表计划项（chart_id）及其支撑结论。"
                )
                section_messages = base_messages + [HumanMessage(content=section_prompt)]
                response = await chain.ainvoke({"messages": section_messages})
                text = self._normalize_section_text(section_title, extract_text_content(response))
                text_len = len(text)
                if text_len > best_len:
                    best_len = text_len
                    best_text = text
                    attempt_meta = getattr(response, "response_metadata", {}) or {}
                    final_raw_content = getattr(response, "content", None)
                if text_len >= min_chars:
                    break

            section_blocks.append({"type": "markdown", "text": best_text})
            section_stats[section_title] = {"min_chars": min_chars, "actual_chars": best_len}
            base_messages.append(HumanMessage(content=f"已完成章节：{section_title}"))
            base_messages.append(HumanMessage(content=best_text[:1500]))
            final_response_metadata = attempt_meta or final_response_metadata

            if section_title == "Results and Validation" and image_blocks:
                section_blocks.append({"type": "markdown", "text": "## Figures\n以下图表为 Coder 节点产出，已纳入论文正文资产。"})
                section_blocks.extend(image_blocks)

        paper_text = "\n\n".join([extract_text_content(block) for block in section_blocks]).strip()
        new_memory = shared_mem.copy()
        new_memory["paper_draft"] = section_blocks
        new_memory["writer_raw_content"] = json_safe(final_raw_content)
        new_memory["writer_response_metadata"] = json_safe(final_response_metadata)
        new_memory["writer_stats"] = {
            "section_stats": section_stats,
            "final_chars": len(paper_text),
            "mode": "sectional_generation",
        }

        await broadcast_progress("Writer", "分章节论文草稿生成完成（已并入图表资产）。", 100)
        return {
            "messages": [{"role": "ai", "content": ensure_blocks({"type": "markdown", "text": paper_text})}],
            "shared_memory": new_memory,
            "status": "APPROVED",
            "draft": section_blocks,
            "stage": make_stage("paper_finished", "Paper Finishing"),
        }


writer_node = WriterNode()
