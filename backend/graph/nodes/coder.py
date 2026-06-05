import re
import json
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress
from backend.services.sandbox import execute_python_code
from backend.services.serialization import extract_text_content, json_safe, ensure_blocks, make_stage, messages_for_llm
from backend.services.multimodal import collect_multimodal_images, image_meta_summary
from backend.services.artifacts import (
    plot_url,
    export_url,
    task_plot_dir,
    task_export_dir,
    sanitize_task_segment,
)

class CoderNode:
    """
    程序员与仿真专家节点
    - 基于模型推导结果生成代码
    - 使用 Sandbox 自动执行验证并自治调试 (Self-Debugging)
    """
    def __init__(self):
        self.system_prompt = (
            "你是一个顶级 MCM/ICM 建模竞赛『程序员』，擅长算法实现与数据可视化。\n"
            "你的任务是生成高质量、可运行且具备学术美感的 Python 代码。\n\n"
            "核心准则：\n"
            "1. **自主可视化**：只要涉及数据分布、模型拟合或结果对比，必须自主决定绘制图表。\n"
            "2. **科学审美**：图表使用 `seaborn` 风格，必须包含标题、标签与图例。\n"
            "3. **动态产物输出**：根据前序节点结果动态决定需要产出的实验资产；"
            "可输出多算法代码、多张图、多份结果文件，而不是固定单图。\n"
            "4. **产物落盘规范**：必须使用沙箱注入的 `MM_AGENT_PLOT_DIR` / `MM_AGENT_EXPORT_DIR` 保存文件；"
            "也可使用 `backend/static/plots/<task_id>/` 与 `backend/static/exports/<task_id>/`（任务 ID 见上下文）。\n"
            "5. **图表规划驱动**：若输入包含 CHART_PLAN，则必须先按计划决定输出图表，而不是固定单图。\n"
            "   - 必须尝试覆盖高优先级图表项；无法覆盖时给出替代图并说明原因。\n"
            "   - 对于流程/思路相关任务，必须输出流程图或思路网络图（可用 matplotlib/networkx）。\n"
            "6. **技术栈**：优先使用 numpy, pandas, matplotlib, seaborn, pulp。除非绝对必要，不要依赖 scipy。\n"
            "5. **输出格式**：将唯一要执行的代码块用 ```python 和 ``` 包裹起来。\n\n"
            "【特别指令】\n"
            "- **代码学术性**：参考 RAG 背景中优秀论文展示的数据处理逻辑与算法实现深度。代码注释应当体现建模逻辑，而不仅仅是代码功能。\n"
            "- **沙箱兼容**：若出现 `No module named scipy`，请立即改写为仅用 numpy/pandas 的等价实现；`rankdata` 请用 pandas 的 `Series.rank()` 替代。\n"
            "- **文风强制要求**：语言和文字风格要符合各个获奖论文严谨冷静的范式，不要使用对表达论文内容来说不必要的比喻以及其他修辞，也不要使用过于抽象的合成词语或者自造词语"
        )

    def _chart_plan_text(self, chart_plan: Any) -> str:
        if not isinstance(chart_plan, list) or not chart_plan:
            return "[]"
        try:
            return json.dumps(chart_plan, ensure_ascii=False, indent=2)
        except Exception:
            return "[]"

    def _clip_middle(self, text: Any, max_chars: int) -> str:
        value = extract_text_content(text).strip()
        if max_chars <= 0 or len(value) <= max_chars:
            return value
        head = max_chars // 2
        tail = max_chars - head - 20
        return f"{value[:head]}\n...[omitted]...\n{value[-tail:]}"

    def _task_id(self, shared_mem: Dict[str, Any]) -> str:
        return str(shared_mem.get("task_id") or "").strip()

    def _artifact_roots(self, task_id: str) -> Dict[str, Path]:
        if task_id:
            return {
                "plot": task_plot_dir(task_id),
                "export": task_export_dir(task_id),
            }
        base = Path(os.path.abspath(os.getcwd()))
        return {
            "plot": base / "backend" / "static" / "plots",
            "export": base / "backend" / "static" / "exports",
        }

    def _snapshot_artifacts(self, task_id: str = "") -> Dict[str, Dict[str, Any]]:
        snap: Dict[str, Dict[str, Any]] = {}
        for _kind, root in self._artifact_roots(task_id).items():
            if not root.exists():
                continue
            for file_path in root.rglob("*"):
                if not file_path.is_file():
                    continue
                try:
                    stat = file_path.stat()
                except OSError:
                    continue
                key = str(file_path.resolve())
                snap[key] = {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}
        return snap

    def _build_artifact_manifest(
        self,
        before: Dict[str, Dict[str, Any]],
        after: Dict[str, Dict[str, Any]],
        task_id: str = "",
    ) -> List[Dict[str, Any]]:
        roots = self._artifact_roots(task_id)
        seg = sanitize_task_segment(task_id) if task_id else ""
        manifest: List[Dict[str, Any]] = []
        for abs_path, after_meta in after.items():
            before_meta = before.get(abs_path)
            if before_meta == after_meta:
                continue
            path_obj = Path(abs_path)
            kind = "file"
            url = ""
            try:
                rel = path_obj.relative_to(roots["plot"]).as_posix()
                kind = "image"
                url = plot_url(task_id, rel) if task_id else f"/plots/{rel}"
            except ValueError:
                try:
                    rel = path_obj.relative_to(roots["export"]).as_posix()
                    kind = "export"
                    url = export_url(task_id, rel) if task_id else f"/exports/{rel}"
                except ValueError:
                    pass
            if kind == "file" and path_obj.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}:
                kind = "image"
            manifest.append(
                {
                    "kind": kind,
                    "path": str(path_obj),
                    "filename": path_obj.name,
                    "size": int(after_meta.get("size", 0)),
                    "updated_at_ns": int(after_meta.get("mtime_ns", 0)),
                    "url": url,
                }
            )
        manifest.sort(key=lambda item: int(item.get("updated_at_ns", 0)), reverse=True)
        return manifest

    def _extract_code(self, text: str) -> str:
        """从 LLM 输出中提取第一个 Python 代码块"""
        pattern = r"```python(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _chart_id_from_manifest_item(self, item: Dict[str, Any]) -> str:
        filename = str(item.get("filename", "")).strip().lower()
        if not filename:
            return ""
        return filename.rsplit(".", 1)[0]

    def _positive_int(self, value: Any, fallback: int = 0) -> int:
        if isinstance(value, bool):
            return fallback
        if isinstance(value, (int, float)):
            parsed = int(value)
            return parsed if parsed > 0 else fallback
        if isinstance(value, str):
            match = re.search(r"\d+", value.strip().replace(",", ""))
            if match:
                parsed = int(match.group(0))
                return parsed if parsed > 0 else fallback
        return fallback

    def _required_chart_lower_bound(
        self,
        chart_plan: Any,
        chart_weight_plan: Any,
        fusion_contract: Any = None,
        fallback_min_required: int = 3,
    ) -> int:
        if isinstance(fusion_contract, dict):
            chart_contract = fusion_contract.get("chart_contract")
            if isinstance(chart_contract, dict):
                parsed = self._positive_int(chart_contract.get("required_chart_count"), 0)
                if parsed:
                    return parsed
            quality = fusion_contract.get("quality_gates")
            if isinstance(quality, dict):
                gate = quality.get("chart_coverage_gate")
                if isinstance(gate, dict):
                    parsed = self._positive_int(gate.get("required_chart_count"), 0)
                    if parsed:
                        return parsed
                else:
                    parsed = self._positive_int(gate, 0)
                    if parsed:
                        return parsed
        if isinstance(chart_weight_plan, dict):
            parsed = self._positive_int(chart_weight_plan.get("required_chart_count"), 0)
            if parsed:
                return parsed
        plan = chart_plan if isinstance(chart_plan, list) else []
        return max(fallback_min_required, len(plan) if plan else fallback_min_required)

    def _check_chart_coverage(
        self,
        chart_plan: Any,
        chart_weight_plan: Any,
        fusion_contract: Any,
        manifest: List[Dict[str, Any]],
        min_required: int = 3,
    ) -> Dict[str, Any]:
        plan = chart_plan if isinstance(chart_plan, list) else []
        plan_ids = [str(it.get("id", "")).strip().lower() for it in plan if isinstance(it, dict) and it.get("id")]
        high_ids = [
            str(it.get("id", "")).strip().lower()
            for it in plan
            if isinstance(it, dict) and str(it.get("priority", "")).strip().lower() == "high" and it.get("id")
        ]
        produced_ids = {
            self._chart_id_from_manifest_item(item)
            for item in manifest
            if isinstance(item, dict) and str(item.get("kind", "")).lower() == "image"
        }
        produced_ids.discard("")
        missing_high = [cid for cid in high_ids if cid not in produced_ids]
        min_target = self._required_chart_lower_bound(plan, chart_weight_plan, fusion_contract, min_required)
        enough_count = len(produced_ids) >= min_target
        must_cover_high = True
        if isinstance(fusion_contract, dict):
            cc = fusion_contract.get("chart_contract")
            if isinstance(cc, dict):
                must_cover_high = bool(cc.get("must_include_high_priority", True))
            qg = fusion_contract.get("quality_gates")
            if isinstance(qg, dict):
                gate = qg.get("chart_coverage_gate")
                if isinstance(gate, dict):
                    must_cover_high = bool(gate.get("must_include_high_priority", must_cover_high))
        ok = enough_count and ((len(missing_high) == 0) if must_cover_high else True)
        return {
            "ok": ok,
            "produced_count": len(produced_ids),
            "required_min": min_target,
            "missing_high": missing_high,
            "produced_ids": sorted(list(produced_ids))[:20],
            "must_cover_high_priority": must_cover_high,
        }

    async def __call__(self, state: AgentState):
        """执行代码生成逻辑实现"""
        shared_mem = state.get("shared_memory", {})
        task_id = self._task_id(shared_mem)
        doc_content = extract_text_content(shared_mem.get("raw_document_content", ""))
        analysis = extract_text_content(shared_mem.get("analysis_report", ""))
        rag_context = extract_text_content(state.get("context", ""))
        fusion_guidance = extract_text_content(shared_mem.get("fusion_guidance", ""))
        fusion_style_contract = shared_mem.get("fusion_style_contract", {})
        fusion_contract = shared_mem.get("fusion_contract", {})
        fusion_figure_style_learning = shared_mem.get("fusion_figure_style_learning", {})
        fusion_chart_plan = (
            shared_mem.get("writer_chart_plan")
            or shared_mem.get("fusion_chart_plan", [])
        )
        fusion_chart_weight_plan = shared_mem.get("fusion_chart_weight_plan", {})
        paper_outline = shared_mem.get("paper_outline", {})
        feedback = state.get("human_feedback", "")
        
        await broadcast_progress("Coder", "正在准备仿真环境与算法指令...", 10)
        
        # 1.5. 动态模型实例化 (从任务配置加载)
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-3.1-flash-lite"
        
        if not api_key:
            return {
                "status": "REJECTED",
                "human_feedback": "缺少任务 API Key，Coder 节点无法执行。",
                "stage": make_stage("coder_failed_missing_api_key", "Coder Failed: Missing Task API Key"),
            }

        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0.2,
            google_api_key=api_key
        )
        
        # 构造执行背景 (转义花括号以防 LaTeX/代码 干扰 LangChain)
        safe_doc = doc_content[:800].replace("{", "{{").replace("}", "}}")
        safe_analysis = analysis[:800].replace("{", "{{").replace("}", "}}")
        context = f"【赛题】\n{safe_doc}...\n\n【模型思路】\n{safe_analysis}..."
        if task_id:
            seg = sanitize_task_segment(task_id)
            context += (
                f"\n\n【任务 ID / 产物目录】\n"
                f"task_id={task_id}\n"
                f"plots=backend/static/plots/{seg}/\n"
                f"exports=backend/static/exports/{seg}/\n"
                f"沙箱变量: MM_AGENT_PLOT_DIR, MM_AGENT_EXPORT_DIR\n"
            )
        if fusion_guidance and not (isinstance(fusion_contract, dict) and fusion_contract):
            safe_fusion = fusion_guidance[:1400].replace("{", "{{").replace("}", "}}")
            context += f"\n\n【Fusion 强约束】\n{safe_fusion}"
        if isinstance(fusion_style_contract, dict) and fusion_style_contract:
            try:
                style_text = json.dumps(fusion_style_contract, ensure_ascii=False, indent=2)[:1400]
                style_text = style_text.replace("{", "{{").replace("}", "}}")
                context += f"\n\n【Style Contract（必须遵守）】\n{style_text}\n"
            except Exception:
                pass
        if isinstance(fusion_contract, dict) and fusion_contract:
            try:
                contract_brief = {
                    "task_requirements": fusion_contract.get("task_requirements", {}),
                    "chart_contract": fusion_contract.get("chart_contract", {}),
                    "quality_gates": fusion_contract.get("quality_gates", {}),
                }
                contract_text = json.dumps(contract_brief, ensure_ascii=False, indent=2)[:1800]
                contract_text = contract_text.replace("{", "{{").replace("}", "}}")
                context += f"\n\n【Fusion Contract（硬约束，必须执行）】\n{contract_text}\n"
            except Exception:
                pass
        if isinstance(fusion_figure_style_learning, dict) and fusion_figure_style_learning:
            try:
                figure_style_text = json.dumps(fusion_figure_style_learning, ensure_ascii=False, indent=2)[:1200]
                figure_style_text = figure_style_text.replace("{", "{{").replace("}", "}}")
                context += (
                    "\n\n【RAG 论文图表风格学习（必须迁移到本任务图表）】\n"
                    f"{figure_style_text}\n"
                    "执行要求：图表类型、配色、caption、坐标轴标签、图例与正文解释要参考这些可迁移规则。"
                )
            except Exception:
                pass
        if fusion_chart_plan:
            chart_plan_text = self._chart_plan_text(fusion_chart_plan).replace("{", "{{").replace("}", "}}")
            context += (
                "\n\n【CHART_PLAN（Writer 大纲审定，必须优先执行）】\n"
                f"{chart_plan_text}\n"
                "执行要求：高优先级图表优先；每个 chart_id 保存为 MM_AGENT_PLOT_DIR 下 <chart_id>.png。"
            )
        if isinstance(paper_outline, dict) and paper_outline.get("sections"):
            try:
                outline_preview = json.dumps(paper_outline.get("sections", [])[:8], ensure_ascii=False, indent=2)
                outline_preview = outline_preview[:1200].replace("{", "{{").replace("}", "}}")
                context += f"\n\n【论文大纲（章节与 figure_ids 对齐）】\n{outline_preview}\n"
            except Exception:
                pass
        if rag_context:
            safe_rag = rag_context[:1000].replace("{", "{{").replace("}", "}}")
            context += f"\n\n【RAG 参考代码风格资料】\n{safe_rag}"
        vision_parts, vision_meta = collect_multimodal_images(
            shared_mem,
            sources=("rag", "blocks", "artifacts"),
            max_images=6,
        )
        if vision_meta:
            context += (
                "\n\n【已附加可直接观察的参考图片】\n"
                f"{image_meta_summary(vision_meta).replace('{', '{{').replace('}', '}}')}\n"
                "执行要求：直接观察附图的布局、caption 风格、图例、坐标轴、配色和视觉层次；"
                "将可迁移设计用于本任务图表，但不得照搬与本题无关的数据结论。"
            )
        
        prompt_parts = [
            ("system", self.system_prompt),
            ("system", f"【当前执行上下文】\n{context}")
        ]
        if feedback:
            prompt_parts.append(("system", f"【用户最新反馈建议】\n{feedback}"))
            
        alignment_record = shared_mem.get("alignment_record", {})
        if alignment_record:
            align_data = json.dumps(alignment_record, ensure_ascii=False, indent=2).replace("{", "{{").replace("}", "}}")
            align_prompt = (
                f"【全局防漂移强制对齐约束 (CRITICAL)】\n"
                f"{align_data}"
            )
            prompt_parts.append(("system", align_prompt))
            
        prompt_parts.append(MessagesPlaceholder(variable_name="messages"))
        
        MAX_RETRIES = 3
        current_attempt = 0
        success = False
        final_code = ""
        final_response_text = ""
        final_raw_content = None
        execution_logs = {}
        manifest: List[Dict[str, Any]] = []
        before_snapshot = self._snapshot_artifacts(task_id)
        
        # 对话历史转为纯文本消息，避免 markdown 块导致 Gemini 报错
        current_messages = messages_for_llm(
            state.get("messages", []) or [],
            max_messages=4,
            max_chars_per_message=1200,
        )
        
        while current_attempt < MAX_RETRIES and not success:
            current_attempt += 1
            progress_val = 20 + (current_attempt - 1) * 20
            
            await broadcast_progress(
                "Coder", 
                f"正在使用 {model_id} 构思算法实现... (尝试 {current_attempt}/{MAX_RETRIES})", 
                progress_val
            )
            
            prompt = ChatPromptTemplate.from_messages(prompt_parts)
            chain = prompt | llm
            
            # TODO: Token 流监听会在第一次 ainvoke 时由 main.py 正常拦截
            # 后续的重试属于内部调用，流可能无法直接推送给前端，但状态会广播
            attempt_messages = list(current_messages)
            if vision_parts:
                attempt_messages.append(
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": (
                                    "请结合这些参考图片生成可执行 Python 代码。重点迁移图表布局、"
                                    "坐标轴标注、图例组织、配色与 caption 风格；代码输出仍必须只有一个 python code block。"
                                ),
                            },
                            *vision_parts,
                        ]
                    )
                )
            response = await chain.ainvoke({"messages": attempt_messages})
            final_raw_content = getattr(response, "content", None)
            generated_text = extract_text_content(response)
            final_response_text = generated_text
            
            extracted_code = self._extract_code(generated_text)
            
            if not extracted_code:
                # 没提出代码，强制中断自纠错，可能只是在解释步骤
                final_code = generated_text
                execution_logs = {"stdout": "", "stderr": "No python block found.", "success": True}
                break
                
            final_code = extracted_code
            await broadcast_progress("Coder", "提取代码完毕，准备扔进沙箱自动执行...", progress_val + 10)
            
            # 执行沙箱
            exec_result = execute_python_code(extracted_code, task_id=task_id or None)
            execution_logs = exec_result
            
            if exec_result["success"]:
                success = True
                after_snapshot = self._snapshot_artifacts(task_id)
                manifest = self._build_artifact_manifest(before_snapshot, after_snapshot, task_id)
                if not manifest:
                    fallback_plot = self._artifact_roots(task_id)["plot"] / "output.png"
                    if fallback_plot.exists():
                        manifest = [
                            {
                                "kind": "image",
                                "path": str(fallback_plot),
                                "filename": fallback_plot.name,
                                "size": int(fallback_plot.stat().st_size),
                                "updated_at_ns": int(fallback_plot.stat().st_mtime_ns),
                                "url": plot_url(task_id, fallback_plot.name) if task_id else "/plots/output.png",
                            }
                        ]
                artifact_urls = [str(item.get("url", "")) for item in manifest if item.get("url")]
                image_urls = [str(item.get("url", "")) for item in manifest if item.get("kind") == "image" and item.get("url")]
                primary_url = image_urls[0] if image_urls else (artifact_urls[0] if artifact_urls else None)
                await broadcast_progress(
                    "Coder",
                    "沙箱执行成功，已根据任务上下文动态产出实验资产。",
                    progress_val + 15,
                    artifact_url=primary_url,
                    artifact_urls=artifact_urls or None,
                    artifact_manifest=manifest or None,
                )

                coverage = self._check_chart_coverage(
                    fusion_chart_plan,
                    fusion_chart_weight_plan,
                    fusion_contract,
                    manifest,
                    min_required=3,
                )
                if not coverage.get("ok"):
                    success = False
                    miss = coverage.get("missing_high", [])
                    miss_text = ", ".join(miss) if miss else "none"
                    await broadcast_progress(
                        "Coder",
                        (
                            "图表覆盖不足，触发自动补图重试："
                            f"当前 {coverage.get('produced_count')} 张，至少 {coverage.get('required_min')} 张；"
                            f"缺少高优先级: {miss_text}"
                        ),
                        min(progress_val + 18, 95),
                    )
                    fix_feedback = (
                        "上一轮代码已运行，但图表覆盖不足。\n"
                        f"- Produced chart ids: {coverage.get('produced_ids')}\n"
                        f"- Missing high-priority chart ids: {miss}\n"
                        f"- At least {coverage.get('required_min')} unique chart images are required.\n"
                        "请输出修订后的完整 Python 代码，补齐缺失图表并保留已有有效图。"
                    )
                    current_messages.append(AIMessage(content=self._clip_middle(response, 1200) or " "))
                    current_messages.append(HumanMessage(content=fix_feedback))
            else:
                stderr_preview = exec_result["stderr"][:1000]
                await broadcast_progress("Coder", "沙箱执行失败！正在提取 Traceback 自我修复...", progress_val + 15)
                
                # 构造自我纠错消息 (转义报错信息中的可能的花括号)
                safe_stderr = stderr_preview.replace("{", "{{").replace("}", "}}")
                extra_hint = ""
                if "No module named 'scipy'" in stderr_preview or 'No module named "scipy"' in stderr_preview:
                    extra_hint = (
                        "\n额外要求：当前环境缺少 scipy，请你完全移除 scipy 依赖，"
                        "用 numpy/pandas 实现同等功能（例如 rankdata -> pandas.Series.rank）。\n"
                    )
                error_feedback = (
                    f"代码执行失败，错误信息（Traceback）如下：\n"
                    f"```text\n{safe_stderr}\n```\n"
                    f"{extra_hint}"
                    f"请分析错误原因，并输出修正后的完整 Python 代码（必须包含 ```python 包裹）。"
                )
                
                # 追加到消息列表中以供下一次重试使用
                current_messages.append(AIMessage(content=self._clip_middle(response, 1200) or " "))
                current_messages.append(HumanMessage(content=error_feedback))
        
        # 3. 广播进度：扫尾工作
        await broadcast_progress("Coder", "代码自治调试环节结束，正在打包运行日志...", 95)
        
        new_memory = shared_mem.copy()
        new_memory["generated_code"] = final_code
        new_memory["execution_logs"] = execution_logs
        new_memory["coder_raw_content"] = json_safe(final_raw_content)
        if isinstance(fusion_chart_plan, list):
            new_memory["coder_chart_plan"] = json_safe(fusion_chart_plan)
        if execution_logs.get("success") and not manifest:
            after_snapshot = self._snapshot_artifacts(task_id)
            manifest = self._build_artifact_manifest(before_snapshot, after_snapshot, task_id)
        if execution_logs.get("success") and not manifest:
            fallback_plot = self._artifact_roots(task_id)["plot"] / "output.png"
            if fallback_plot.exists():
                manifest = [
                    {
                        "kind": "image",
                        "path": str(fallback_plot),
                        "filename": fallback_plot.name,
                        "size": int(fallback_plot.stat().st_size),
                        "updated_at_ns": int(fallback_plot.stat().st_mtime_ns),
                        "url": plot_url(task_id, fallback_plot.name) if task_id else "/plots/output.png",
                    }
                ]
        artifact_urls = [str(item.get("url", "")) for item in manifest if item.get("url")]
        image_urls = [str(item.get("url", "")) for item in manifest if item.get("kind") == "image" and item.get("url")]
        artifact_url = image_urls[0] if image_urls else (artifact_urls[0] if artifact_urls else None)
        new_memory["artifacts_manifest"] = json_safe(manifest)
        
        await broadcast_progress(
            "Coder",
            "执行日志已注入共享内存，进入审计环节。",
            100,
            artifact_url=artifact_url,
            artifact_urls=artifact_urls or None,
            artifact_manifest=manifest or None,
        )
        
        return {
            "messages": [{"role": "ai", "content": ensure_blocks(final_response_text or final_code)}],
            "shared_memory": new_memory,
            "status": "APPROVED",
            "draft": ensure_blocks(final_response_text or final_code),
            "stage": make_stage("coding_completed", "Code Implementation & Sandbox Execution Completed"),
        }

coder_node = CoderNode()
