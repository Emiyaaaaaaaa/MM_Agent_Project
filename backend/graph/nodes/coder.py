import re
import json
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress
from backend.services.sandbox import execute_python_code
from backend.services.serialization import extract_text_content, json_safe, ensure_blocks, make_stage

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
            "4. **产物落盘规范**：图像保存到 `backend/static/plots/`，表格/结果保存到 `backend/static/exports/`。\n"
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

    def _artifact_roots(self) -> Dict[str, Path]:
        base = Path(os.path.abspath(os.getcwd()))
        return {
            "plot": base / "backend" / "static" / "plots",
            "export": base / "backend" / "static" / "exports",
        }

    def _snapshot_artifacts(self) -> Dict[str, Dict[str, Any]]:
        snap: Dict[str, Dict[str, Any]] = {}
        for _kind, root in self._artifact_roots().items():
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
    ) -> List[Dict[str, Any]]:
        roots = self._artifact_roots()
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
                url = f"/plots/{rel}"
            except ValueError:
                try:
                    rel = path_obj.relative_to(roots["export"]).as_posix()
                    kind = "export"
                    url = f"/exports/{rel}"
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

    async def __call__(self, state: AgentState):
        """执行代码生成逻辑实现"""
        shared_mem = state.get("shared_memory", {})
        doc_content = extract_text_content(shared_mem.get("raw_document_content", ""))
        analysis = extract_text_content(shared_mem.get("analysis_report", ""))
        rag_context = extract_text_content(state.get("context", ""))
        fusion_guidance = extract_text_content(shared_mem.get("fusion_guidance", ""))
        fusion_chart_plan = shared_mem.get("fusion_chart_plan", [])
        feedback = state.get("human_feedback", "")
        
        await broadcast_progress("Coder", "正在准备仿真环境与算法指令...", 10)
        
        # 1.5. 动态模型实例化 (从任务配置加载)
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-2.5-flash-lite"
        
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
        safe_doc = doc_content[:1000].replace("{", "{{").replace("}", "}}")
        safe_analysis = analysis[:1000].replace("{", "{{").replace("}", "}}")
        context = f"【赛题】\n{safe_doc}...\n\n【模型思路】\n{safe_analysis}..."
        if fusion_guidance:
            safe_fusion = fusion_guidance[:2600].replace("{", "{{").replace("}", "}}")
            context += f"\n\n【Fusion 强约束】\n{safe_fusion}"
        if fusion_chart_plan:
            chart_plan_text = self._chart_plan_text(fusion_chart_plan).replace("{", "{{").replace("}", "}}")
            context += (
                "\n\n【CHART_PLAN（必须优先执行）】\n"
                f"{chart_plan_text}\n"
                "执行要求：高优先级图表优先，生成文件名建议采用 <chart_id>.png。"
            )
        if rag_context:
            safe_rag = rag_context[:2000].replace("{", "{{").replace("}", "}}")
            context += f"\n\n【RAG 参考代码风格资料】\n{safe_rag}"
        
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
        before_snapshot = self._snapshot_artifacts()
        
        # 当前的对话历史拷贝
        current_messages = list(state["messages"])
        
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
            response = await chain.ainvoke({"messages": current_messages})
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
            exec_result = execute_python_code(extracted_code)
            execution_logs = exec_result
            
            if exec_result["success"]:
                success = True
                after_snapshot = self._snapshot_artifacts()
                manifest = self._build_artifact_manifest(before_snapshot, after_snapshot)
                if not manifest:
                    fallback_plot = self._artifact_roots()["plot"] / "output.png"
                    if fallback_plot.exists():
                        manifest = [
                            {
                                "kind": "image",
                                "path": str(fallback_plot),
                                "filename": fallback_plot.name,
                                "size": int(fallback_plot.stat().st_size),
                                "updated_at_ns": int(fallback_plot.stat().st_mtime_ns),
                                "url": "/plots/output.png",
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
                current_messages.append(response)
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
            after_snapshot = self._snapshot_artifacts()
            manifest = self._build_artifact_manifest(before_snapshot, after_snapshot)
        if execution_logs.get("success") and not manifest:
            fallback_plot = self._artifact_roots()["plot"] / "output.png"
            if fallback_plot.exists():
                manifest = [
                    {
                        "kind": "image",
                        "path": str(fallback_plot),
                        "filename": fallback_plot.name,
                        "size": int(fallback_plot.stat().st_size),
                        "updated_at_ns": int(fallback_plot.stat().st_mtime_ns),
                        "url": "/plots/output.png",
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
            "status": "PENDING",
            "draft": ensure_blocks(final_response_text or final_code),
            "stage": make_stage("coding_completed", "Code Implementation & Sandbox Execution Completed"),
        }

coder_node = CoderNode()
