import os
import json
import base64
from typing import Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress
from backend.services.serialization import extract_text_content, json_safe, ensure_blocks, make_stage

class ReviewerNode:
    """
    辩论式审查专家节点 (Debate-Style Reviewer)：
    包含红方(攻击)、蓝方(防守)、主审法官(裁决)的三元验证框架。
    """
    def __init__(self):
        self.critique_prompt = (
            "你是一个最严酷的 MCM/ICM 学术打假人和漏洞挖掘专家 (The Critique)。\n"
            "你的任务是毫无保留地寻找前方生成的【原始材料】（逻辑、公式、代码、输出结论、图文一致性）中的所有缺陷。\n"
            "重点打击：\n"
            "1. 数学假设不合理，或者推导出现明显跳跃。\n"
            "2. 代码逻辑错位、数值可能溢出或算法选择极度劣势。\n"
            "4. 记忆漂移(Drift)：检查代码与公式的符号是否割裂（如公式用X，代码用Y），或代码擅自添加了《全局预设假设》中不允许的考虑因素。\n"
            "不要进行任何客套和赞美，直接列出所有痛点和致命雷区，生成一份火力全开的《攻击报告》。\n"
            "【文风强制要求】：语言和文字风格要符合各个获奖论文严谨冷静的范式，不要使用对表达论文内容来说不必要的比喻以及其他修辞，也不要使用过于抽象的合成词语或者自造词语"
        )
        
        self.defender_prompt = (
            "你是一个顶尖的 MCM/ICM 建模竞赛论文原作者 (The Defender)。\n"
            "你刚刚收到了一份极其刺耳的《攻击报告》，质疑了你的模型体系、代码推导或图表展示。\n"
            "你的任务是：\n"
            "1. 对于合理的学术简化或为了模型收敛而做出的妥协，给出强有力的学术术语进行理论辩护。\n"
            "2. 澄清对方可能因为未详读原始代码而产生的误解。\n"
            "3. 如果对方指出了真正无法辩驳的硬伤（如明显且致命的代码错误、自相矛盾的数据点），果断承认失误。\n"
            "生成一份理据充分的、守护心血结晶的《答辩报告》。\n"
            "【文风强制要求】：语言和文字风格要符合各个获奖论文严谨冷静的范式，不要使用对表达论文内容来说不必要的比喻以及其他修辞，也不要使用过于抽象的合成词语或者自造词语"
        )
        
        self.judge_prompt = (
            "你是 MCM/ICM 的 O 奖评审委员会主席兼全场主法官 (The Judge)。\n"
            "你手上有三份核心文件：全链路【原始材料】及其运行反馈、【攻击者的攻击报告】、【原作者的答辩报告】。\n"
            "你的任务是进行终审裁决。\n"
            "【权威裁判准则】\n"
            "1. 一刀切去攻击者的过度苛刻和防御者的杠精扯皮心态，你只关心“这会不会导致整模跨掉或丢分严重”。\n"
            "2. 提炼真正影响模型可用性的有效缺陷，生成一份庄重、中立、直指盲证的《官方综合审查意见》。\n"
            "3. 如果存在必须修复的红牌硬伤，务必明确在结论段给出具体的“回滚与修复指令清单”（例如：建议退回 Coder 节点重写算法架构）。\n"
            "4. 版权红线底限：审查报告中绝对不允许提及任何 RAG 参考资料中的队号、年份或特定 O 奖头衔，必须统一以‘顶尖赛事通行学术标准’的宏大叙事面貌展示。\n"
            "5. 图表覆盖度核查：必须检查 `Fusion 图表计划` 与 `Coder 实际产物` 是否一致，"
            "并检查论文正文是否引用并解释了关键图表。若不一致，必须给出退回修复指令。\n"
            "【文风强制要求】：语言和文字风格要符合各个获奖论文严谨冷静的范式，不要使用对表达论文内容来说不必要的比喻以及其他修辞，也不要使用过于抽象的合成词语或者自造词语"
        )

    def _get_image_data(self, file_path, max_bytes: int = 4 * 1024 * 1024):
        """多模态图像 Base64 提取器"""
        if os.path.exists(file_path):
            try:
                if os.path.getsize(file_path) > max_bytes:
                    return None
            except OSError:
                return None
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return None

    def _positive_int(self, value: Any, fallback: int = 0) -> int:
        if isinstance(value, bool):
            return fallback
        if isinstance(value, (int, float)):
            parsed = int(value)
            return parsed if parsed > 0 else fallback
        if isinstance(value, str):
            import re

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

    def _chart_coverage_check(self, chart_plan: Any, chart_weight_plan: Any, fusion_contract: Any, manifest: Any) -> Dict[str, Any]:
        plan = chart_plan if isinstance(chart_plan, list) else []
        produced = manifest if isinstance(manifest, list) else []
        required = [str(item.get("id", "")).strip().lower() for item in plan if isinstance(item, dict) and item.get("id")]
        required_high = [
            str(item.get("id", "")).strip().lower()
            for item in plan
            if isinstance(item, dict) and str(item.get("priority", "")).strip().lower() == "high" and item.get("id")
        ]
        produced_ids = set()
        for item in produced:
            if not isinstance(item, dict) or str(item.get("kind", "")).lower() != "image":
                continue
            fname = str(item.get("filename", "")).strip().lower()
            if fname:
                produced_ids.add(fname.rsplit(".", 1)[0])
        missing_high = [cid for cid in required_high if cid not in produced_ids]
        min_required = self._required_chart_lower_bound(plan, chart_weight_plan, fusion_contract, fallback_min_required=3)
        enough_count = len(produced_ids) >= min_required
        must_cover_high = True
        if isinstance(fusion_contract, dict):
            chart_contract = fusion_contract.get("chart_contract")
            if isinstance(chart_contract, dict):
                must_cover_high = bool(chart_contract.get("must_include_high_priority", True))
            quality = fusion_contract.get("quality_gates")
            if isinstance(quality, dict):
                gate = quality.get("chart_coverage_gate")
                if isinstance(gate, dict):
                    must_cover_high = bool(gate.get("must_include_high_priority", must_cover_high))
        return {
            "ok": enough_count and (not missing_high if must_cover_high else True),
            "produced_count": len(produced_ids),
            "required_min": min_required,
            "missing_high": missing_high,
        }

    def _draft_quality_check(self, draft_text: str, fusion_contract: Any) -> Dict[str, Any]:
        text = (draft_text or "").strip()
        if not text:
            return {"ok": False, "reason": "empty_draft"}
        lowered = text.lower()
        required_heads = ["abstract", "introduction", "methodology", "results", "conclusion"]
        min_figure_refs = 3
        min_chars = 0
        if isinstance(fusion_contract, dict):
            quality = fusion_contract.get("quality_gates")
            if isinstance(quality, dict):
                mandatory = quality.get("mandatory_sections")
                if isinstance(mandatory, list) and mandatory:
                    required_heads = [str(x).strip().lower() for x in mandatory if str(x).strip()]
                min_figure_refs = max(1, self._positive_int(quality.get("min_figure_refs"), min_figure_refs))
                min_chars = max(0, self._positive_int(quality.get("draft_min_chars"), min_chars))
        miss = [h for h in required_heads if h not in lowered]
        figure_mentions = lowered.count("figure")
        if min_chars > 0 and len(text) < min_chars:
            return {"ok": False, "reason": f"draft_too_short:{len(text)}<{min_chars}"}
        if miss:
            return {"ok": False, "reason": f"missing_sections:{','.join(miss)}"}
        if figure_mentions < min_figure_refs:
            return {"ok": False, "reason": "insufficient_figure_references"}
        return {"ok": True, "reason": "ok"}

    async def __call__(self, state: AgentState):
        """执行流式多模态三元辩论审查"""
        shared_mem = state.get("shared_memory", {})
        analysis = extract_text_content(shared_mem.get("analysis_report", ""))
        model_doc = extract_text_content(shared_mem.get("mathematical_model", ""))
        code = extract_text_content(shared_mem.get("generated_code", ""))
        execution_logs = shared_mem.get("execution_logs", {})
        fusion_chart_plan = shared_mem.get("fusion_chart_plan", [])
        fusion_chart_weight_plan = shared_mem.get("fusion_chart_weight_plan", {})
        fusion_contract = shared_mem.get("fusion_contract", {})
        artifacts_manifest = shared_mem.get("artifacts_manifest", [])
        paper_draft_text = extract_text_content(shared_mem.get("paper_draft", ""))
        
        await broadcast_progress("Review", "[1/4] 资源锁定：正在抽取并组装全链路原始文稿素材视界...", 10)
        
        images = []
        max_review_images = 4
        if isinstance(artifacts_manifest, list):
            for item in artifacts_manifest:
                if len(images) >= max_review_images:
                    break
                if not isinstance(item, dict) or str(item.get("kind", "")).lower() != "image":
                    continue
                abs_path = str(item.get("path", "")).strip()
                if not abs_path:
                    continue
                plot_b64 = self._get_image_data(abs_path)
                if plot_b64:
                    images.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{plot_b64}"}})
            if images:
                await broadcast_progress("Review", f"捕获 {len(images)} 张任务图表，激活多模态联合阵列...", 15)
        if not images:
            task_id = str(shared_mem.get("task_id") or "").strip()
            from backend.services.artifacts import task_plot_dir

            plot_path = os.path.join(task_plot_dir(task_id, create=False), "output.png") if task_id else os.path.join(
                os.path.abspath(os.getcwd()), "backend", "static", "plots", "output.png"
            )
            plot_b64 = self._get_image_data(plot_path)
            if plot_b64:
                images.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{plot_b64}"}})
                await broadcast_progress("Review", "捕获物理图纸 output.png，激活多模态联合阵列...", 15)
            
        base_material_text = (
            f"以下是全部的【原始材料】产出摘要：\n\n"
            f"【1. 审题分析】\n{analysis[:700]}...\n\n"
            f"【2. 核心大模型】\n{model_doc[:700]}...\n\n"
            f"【3. 控制代码基】\n{code[:700]}...\n\n"
        )
        if execution_logs:
            stdout_data = execution_logs.get('stdout', '')
            base_material_text += f"【4. 代码标准运行时反射区】\n{stdout_data[:800]}\n\n"
        if isinstance(fusion_chart_plan, list):
            base_material_text += f"【5. Fusion 图表计划】\n{json.dumps(fusion_chart_plan, ensure_ascii=False)[:1200]}\n\n"
        if isinstance(artifacts_manifest, list):
            base_material_text += f"【6. Coder 实际图表/产物】\n{json.dumps(artifacts_manifest, ensure_ascii=False)[:1000]}\n\n"
        if paper_draft_text:
            base_material_text += f"【7. 论文正文图表引用节选】\n{paper_draft_text[:1200]}\n\n"
        
        rag_context = state.get("context", "")
        if rag_context:
            base_material_text = f"【系统锚定 RAG 知识框架下限（不可违章）】\n{rag_context[:700]}\n\n" + base_material_text
            
        alignment_record = shared_mem.get("alignment_record", {})
        if alignment_record:
            base_material_text = (
                f"【全局防漂移强制对齐约束档案 (ALIGNMENT RECORD)】\n"
                f"{json.dumps(alignment_record, ensure_ascii=False, indent=2)}\n\n"
            ) + base_material_text
            
        base_content = [{"type": "text", "text": base_material_text}] + images
        
        # 建立网络连线心跳
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-3.1-flash-lite"
        if not api_key:
            return {
                "status": "REJECTED",
                "human_feedback": "缺少任务 API Key，Review 节点无法执行。",
                "stage": make_stage("review_failed_missing_api_key", "Review Failed: Missing Task API Key"),
            }

        # 辩手模型（微高温度刺激跳跃性思维）
        debater_llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0.3, 
            google_api_key=api_key
        )
        
        # =======================
        # Round 1: Critique (红方攻击)
        # =======================
        await broadcast_progress("Review", "[2/4] 第一轮辩论：红方攻击专家 (The Critique) 正在执行毁灭性挑刺抓虫...", 30)
        critique_msg = [
            SystemMessage(content=self.critique_prompt),
            HumanMessage(content=base_content)
        ]
        critique_response = await debater_llm.ainvoke(critique_msg)
        critique_report = extract_text_content(critique_response)
        
        # =======================
        # Round 2: Defender (蓝方防守)
        # =======================
        await broadcast_progress("Review", "[3/4] 第二轮辩论：蓝方原作者 (The Defender) 遭到质询，正在提取文献撰写技术答辩方案...", 60)
        defender_text = (
            f"以下是攻击者向你抛出的《毁击报告》，请依据你的【原始材料】立意进行强硬答辩：\n\n"
            f"【你的核心底座原始材料】:\n{base_material_text[:1200]}\n\n"
            f"--- 对方论点切片 ---\n{critique_report}"
        )
        defender_msg = [
            SystemMessage(content=self.defender_prompt),
            HumanMessage(content=defender_text)
        ]
        defender_response = await debater_llm.ainvoke(defender_msg)
        defense_report = extract_text_content(defender_response)
        
        # =======================
        # Round 3: Judge (主板裁决)
        # =======================
        await broadcast_progress("Review", "[4/4] 最终裁决：主裁委员会 (The Judge) 已入座，正在提炼有效争议，敲定官方结贴信...", 85)
        # 法官模型（低温确保裁决一致性与极度镇静）
        judge_llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0.1, 
            google_api_key=api_key
        )
        judge_text = (
            f"请法官查阅以下案件素材：\n\n"
            f"【第一附件: 前置原始材料】:\n{base_material_text[:1000]}\n\n"
            f"================\n"
            f"【原告提交: 攻击者的火力全开报告】:\n{critique_report[:2000]}\n\n"
            f"================\n"
            f"【被告提交: 原作者的逐条答辩报告】:\n{defense_report[:1800]}\n\n"
            f"请正式对本案全息建模结论下达《官方综合审查意见》。"
        )
        judge_msg = [
            SystemMessage(content=self.judge_prompt),
            HumanMessage(content=[{"type": "text", "text": judge_text}] + images)
        ]
        judge_response = await judge_llm.ainvoke(judge_msg)
        final_review = extract_text_content(judge_response)
        
        await broadcast_progress("Review", "辩论式逻辑审查体系运作顺利流转毕，已封印综合意见卷底。", 100)

        chart_check = self._chart_coverage_check(
            fusion_chart_plan,
            fusion_chart_weight_plan,
            fusion_contract,
            artifacts_manifest,
        )
        draft_check = self._draft_quality_check(paper_draft_text, fusion_contract)
        
        new_memory = shared_mem.copy()
        new_memory["review_report"] = final_review
        # 溯源日志防丢失
        new_memory["review_debate_logs"] = f"【The Critique Report】\n{critique_report}\n\n【The Defender Report】\n{defense_report}"
        new_memory["review_raw_content"] = {
            "critique": json_safe(getattr(critique_response, "content", None)),
            "defense": json_safe(getattr(defender_response, "content", None)),
            "judge": json_safe(getattr(judge_response, "content", None)),
        }

        if not chart_check["ok"]:
            failure_note = (
                "Review Gate: chart coverage is insufficient.\n"
                f"- produced={chart_check['produced_count']}, required>={chart_check['required_min']}\n"
                f"- missing high-priority charts={chart_check['missing_high']}\n"
                "Action: rerun Coder to complete planned figures before final export."
            )
            new_memory["last_failure_stage_id"] = "review_failed_chart_coverage"
            new_memory["review_report"] = failure_note
            return {
                "messages": [{"role": "ai", "content": ensure_blocks(failure_note)}],
                "shared_memory": new_memory,
                "status": "APPROVED",
                "draft": ensure_blocks(failure_note),
                "stage": make_stage("review_failed_chart_coverage", "Review Gate Failed: Chart Coverage"),
            }

        if not draft_check["ok"]:
            failure_note = (
                "Review Gate: draft completeness is insufficient.\n"
                f"- reason={draft_check['reason']}\n"
                "Action: rerun Writing (draft phase) to complete missing sections and figure interpretation."
            )
            new_memory["last_failure_stage_id"] = "review_failed_draft_quality"
            new_memory["review_report"] = failure_note
            return {
                "messages": [{"role": "ai", "content": ensure_blocks(failure_note)}],
                "shared_memory": new_memory,
                "status": "APPROVED",
                "draft": ensure_blocks(failure_note),
                "stage": make_stage("review_failed_draft_quality", "Review Gate Failed: Draft Quality"),
            }
        
        return {
            "messages": [{"role": "ai", "content": ensure_blocks(final_review)}],
            "shared_memory": new_memory,
            "status": "APPROVED",
            "draft": ensure_blocks(final_review),
            "stage": make_stage("review_completed", "Debate-Style Review Completed"),
        }

reviewer_node = ReviewerNode()
