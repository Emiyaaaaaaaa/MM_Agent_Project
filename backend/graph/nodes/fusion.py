import json
import logging
import math
import mimetypes
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage
from google import genai
from google.genai import types
from backend.services.rag_engine import rag_engine
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress
from backend.services.serialization import extract_text_content, make_stage, json_safe
from backend.services.fusion_audit import save_fusion_audit

logger = logging.getLogger("mm-agent")

class FusionNode:
    """
    全系统知识融合节点 (Knowledge Fusion Expert):
    1. 负责接收全工作流中的问题/上下文搜索需求。
    2. 多线程/异步调用局域 RAG 数据库 (获取 O 奖论文资源)。
    3. 调用原生 Gemini 的 Google Search Grounding 能力获取外网最新事实。
    4. 【动态智能加权】：由 LLM 视当前问题的侧重点（客观数据用 Web vs 学术方法用 RAG），进行知识的裁剪合并。
    """
    def __init__(self):
        self.system_prompt = (
            "你是一个『多模态数据与知识融合专家』(Knowledge Fusion Expert)。"
            "你会收到两部分知识：【本地 O 奖论文参考库】和【自带的 Google Search 实时公网检索库】。"
            "\n\n【动态加权融合法则】：\n"
            "1. **学术框架 vs 实时物理世界**：\n"
            "   - 若用户需求涉及真实的现实客观数据（如：巴拿马水库最近的流量、气象卫星数据观测、地缘政治新闻、宏观经济报表等时效性参数），**必须赋予【Google Search 网络结果】绝对的高权重**，并以网络数据填补 RAG 的空白。\n"
            "   - 若用户需求涉及数学建模体系构建、算法推导、特殊分布函数的应用、论文排版格式或往年评审偏好等学术向内容，**必须赋予【本地 O 奖论文参考库】绝对的高权重**，摒弃网络搜索中的业余“野路子”与低质博客代码。\n"
            "   - 若两者均有涉及，请用网络客观数据填充本地学术方程里的参数变量，达成完美融合。\n"
            "2. **去伪存真**：根据当前的提问场景，精准地滤除不相干或权重极低的矛盾噪点。\n"
            "3. **成果产出**：输出一份结构化、数据详实且兼具学术底蕴的《综合事实与学术背景简报》(Synthesized Background Context)。\n"
            "3.1 **论文结构与行文方式显式学习（必须）**：你必须先阅读本地 RAG 中相关论文片段，"
            "识别不同优秀论文的章节标题、章节顺序、每章内部组织方式、公式/算法/图表嵌入方式、结果解释方式与语气特征。"
            "不同论文结构可能不一致，此时必须由你总结差异并给出适合当前题目的综合推荐。"
            "在正文简报后追加 `PAPER_STRUCTURE_LEARNING_JSON`，JSON 对象必须包含：\n"
            "   - observed_structures: 数组，每项 {source_hint, headings, section_flow, writing_patterns, figure_table_usage, evidence_style, limitations_style}\n"
            "   - common_patterns: 归纳不同论文共同采用的结构和行文套路\n"
            "   - divergent_patterns: 说明不同论文结构差异及适用条件\n"
            "   - recommended_sections: 为当前题目综合出的章节标题数组\n"
            "   - recommended_section_templates: 对每个推荐章节给出内部段落结构/公式图表证据安排\n"
            "   - recommended_writing_style: 当前论文应采用的语气、段落长度、图表解释、数学表达密度\n"
            "   注意：不要暴露队号、奖项级别、具体论文身份；source_hint 只能写 masked_reference_1 这类泛化标识。\n"
            "3.2 **RAG 论文图表显式学习（必须）**：如果输入中包含论文图表图片或图表 caption，必须观察其视觉结构和正文解释方式。"
            "在正文简报后追加 `FIGURE_STYLE_LEARNING_JSON`，JSON 对象必须包含：\n"
            "   - observed_figures: 数组，每项 {source_hint, figure_type, visual_layout, caption_pattern, axis_labeling_style, color_style, text_explanation_pattern, transferable_design_rule}\n"
            "   - common_chart_patterns: 归纳相关论文常用图表类型、排版和解释套路\n"
            "   - divergent_chart_patterns: 说明不同图表风格差异及适用条件\n"
            "   - recommended_chart_style: 当前题目应采用的图表视觉风格、配色、caption、正文解释规则\n"
            "   - recommended_chart_plan_adjustments: 对 CHART_PLAN_JSON 的图表类型、优先级、章节绑定或视觉设计建议\n"
            "   注意：图表学习必须来自 RAG 论文图片/caption/图文关系，不要只泛泛说要画图。\n"
            "3.3 **强约束输出格式（必须严格遵守）**：必须输出以下 5 个一级小节，且每节至少 3 条要点：\n"
            "   - [STYLE_GUIDE]：优秀论文的语言风格与论证模式\n"
            "   - [MODELING_PRIORS]：可复用的建模结构、变量设定、目标函数/约束套路\n"
            "   - [RESULTS_NARRATIVE]：结果章节应如何解释图表与数值，如何做对比与验证\n"
            "   - [RISK_AND_LIMITATIONS]：稳健性、敏感性、边界条件与局限性写法\n"
            "   - [WRITING_BLUEPRINT]：论文章节级写作蓝图（每章应覆盖的关键点）\n"
            "   严禁输出空节、严禁只给泛泛建议。\n"
            "3.4 **图表规划输出（必须）**：在文末追加 `CHART_PLAN_JSON`，输出 JSON 数组。每项字段必须包含：\n"
            "   - id: 图表唯一标识（英文下划线）\n"
            "   - title: 图表标题\n"
            "   - chart_type: 例如 flowchart/network/line/bar/scatter/heatmap/radar/stacked_area/table\n"
            "   - purpose: 该图用于证明什么结论\n"
            "   - required_inputs: 依赖的上游信息键（如 analysis_report/mathematical_model/execution_logs）\n"
            "   - section_hint: 建议放在论文哪个章节（如 Model Establishment / Model Solution and Results / Sensitivity Analysis 等）\n"
            "   - priority: high/medium/low\n"
            "   - fallback_if_missing: 数据不足时的替代图\n"
            "   只要题目涉及方法流程与决策链，必须包含至少 1 个流程图或思路网络图计划项。\n"
            "3.5 **图表权重输出（必须）**：在 `CHART_PLAN_JSON` 后追加 `CHART_WEIGHT_PLAN_JSON`，结构示例：\n"
            "   {\n"
            "     \"weights\": [\n"
            "       {\"id\":\"method_flowchart\",\"priority\":\"high\",\"weight\":0.95,\"reason\":\"...\"}\n"
            "     ],\n"
            "     \"required_chart_count\": 7\n"
            "   }\n"
            "   规则：weight ∈ [0.3,1.0]，且 high >= medium >= low；required_chart_count 建议为 ceil(sum(weights))。\n"
            "3.6 **统一硬约束协议输出（必须）**：在文末追加 `FUSION_CONTRACT_JSON`，输出 JSON 对象，必须包含并仅按以下语义组织：\n"
            "   - task_requirements: {problem_type, primary_objective, key_constraints, success_criteria}\n"
            "   - writing_contract: {target_language, tone, required_sections, section_focus, section_templates, writing_style_rules, forbidden_patterns}\n"
            "   - chart_contract: {required_chart_count, must_include_high_priority, chart_plan, chart_weight_plan, figure_style_learning, chart_style_rules}\n"
            "   - evidence_contract: {core_claim_policy, required_evidence_types, results_requirements}\n"
            "   - quality_gates: {draft_min_chars, min_figure_refs, mandatory_sections, chart_coverage_gate}\n"
            "   注意：chart_contract 中的 required_chart_count 必须与 CHART_WEIGHT_PLAN_JSON 一致或更严格；"
            "quality_gates.chart_coverage_gate.required_chart_count 也必须一致或更严格。\n"
            "4. **文风强制要求**：语言和文字风格要符合各个获奖论文严谨冷静的范式，不要使用对表达论文内容来说不必要的比喻以及其他修辞，也不要使用过于抽象的合成词语或者自造词语\n"
            "5. **安全与版权红线**：绝不可在最终的简报中暴露本地参考库的具体来源标记（如论文年份、具体参赛队号、O 奖等级等字眼），用“行业前沿常识”、“学术先例”等通用词汇打码代替。"
        )

    def _default_chart_plan(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "method_flowchart",
                "title": "Method Workflow Diagram",
                "chart_type": "flowchart",
                "purpose": "展示从问题定义到求解与验证的完整流程链路",
                "required_inputs": ["analysis_report", "mathematical_model"],
                "section_hint": "Model Establishment",
                "priority": "high",
                "fallback_if_missing": "改为简化流程网络图",
            },
            {
                "id": "indicator_relation_network",
                "title": "Indicator Relation Network",
                "chart_type": "network",
                "purpose": "展示指标或变量之间的依赖关系与影响路径",
                "required_inputs": ["analysis_report", "mathematical_model"],
                "section_hint": "Model Establishment",
                "priority": "medium",
                "fallback_if_missing": "改为变量关系矩阵热力图",
            },
            {
                "id": "result_comparison_chart",
                "title": "Result Comparison",
                "chart_type": "bar",
                "purpose": "对比核心方案或核心指标的结果差异",
                "required_inputs": ["execution_logs", "generated_code"],
                "section_hint": "Model Solution and Results",
                "priority": "high",
                "fallback_if_missing": "改为表格汇总",
            },
            {
                "id": "sensitivity_curve",
                "title": "Sensitivity Analysis Curve",
                "chart_type": "line",
                "purpose": "验证关键参数变化对目标结果的影响",
                "required_inputs": ["mathematical_model", "execution_logs"],
                "section_hint": "Sensitivity Analysis",
                "priority": "medium",
                "fallback_if_missing": "改为区间柱状图",
            },
            {
                "id": "robustness_boxplot",
                "title": "Robustness Distribution",
                "chart_type": "box",
                "purpose": "展示多场景下结果分布及稳定性",
                "required_inputs": ["execution_logs"],
                "section_hint": "Sensitivity Analysis",
                "priority": "medium",
                "fallback_if_missing": "改为误差条形图",
            },
            {
                "id": "ablation_comparison",
                "title": "Ablation Comparison",
                "chart_type": "bar",
                "purpose": "比较关键模块/策略移除前后的性能变化",
                "required_inputs": ["execution_logs", "generated_code"],
                "section_hint": "Model Solution and Results",
                "priority": "high",
                "fallback_if_missing": "改为对照表格",
            },
        ]

    def _extract_section_bullets(self, text: str, tag: str) -> List[str]:
        pattern = rf"\[{tag}\](.*?)(?=\[(?:STYLE_GUIDE|MODELING_PRIORS|RESULTS_NARRATIVE|RISK_AND_LIMITATIONS|WRITING_BLUEPRINT)\]|PAPER_STRUCTURE_LEARNING_JSON|FIGURE_STYLE_LEARNING_JSON|CHART_PLAN_JSON|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if not match:
            return []
        body = match.group(1)
        lines = [ln.strip(" -\t\r") for ln in body.splitlines() if ln.strip()]
        return [ln for ln in lines if len(ln) > 4][:8]

    def _extract_marked_json(self, text: str, marker: str) -> Optional[Any]:
        """Extract the first balanced JSON value after a named marker.

        LLMs often wrap the JSON in ```json fences. Regex-only extraction is
        brittle for nested objects, so scan braces/brackets with string-state.
        """
        if not text:
            return None
        match = re.search(re.escape(marker), text, re.IGNORECASE)
        if not match:
            return None
        tail = text[match.end():].lstrip(" \t\r\n:：")
        if tail.startswith("```"):
            tail = re.sub(r"^```(?:json)?\s*", "", tail, count=1, flags=re.IGNORECASE)

        start = -1
        for idx, ch in enumerate(tail):
            if ch in "{[":
                start = idx
                break
            if ch == "`" and tail[idx: idx + 3] == "```":
                return None
        if start < 0:
            return None

        opening = tail[start]
        closing = "}" if opening == "{" else "]"
        stack: List[str] = []
        in_string = False
        escape = False
        for idx in range(start, len(tail)):
            ch = tail[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if not stack or ch != stack[-1]:
                    return None
                stack.pop()
                if not stack:
                    raw = tail[start: idx + 1]
                    try:
                        return json.loads(raw)
                    except Exception:
                        return None
        return None

    def _positive_int(self, value: Any, fallback: int = 0) -> int:
        if isinstance(value, bool):
            return fallback
        if isinstance(value, (int, float)):
            parsed = int(value)
            return parsed if parsed > 0 else fallback
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            match = re.search(r"\d+", cleaned)
            if match:
                parsed = int(match.group(0))
                return parsed if parsed > 0 else fallback
        return fallback

    def _sections_from_writing_blueprint(self, fused_context: str) -> List[str]:
        bullets = self._extract_section_bullets(fused_context, "WRITING_BLUEPRINT")
        sections: List[str] = []
        for item in bullets:
            match = re.search(r"\*\*([^*]{2,80})\*\*\s*[:：]", item)
            if not match:
                match = re.search(r"^\d+[\).]\s*([^:：]{2,80})[:：]", item)
            title = match.group(1).strip() if match else ""
            if title and title not in sections:
                sections.append(title)
        return sections

    def _build_rag_queries(self, user_query: str) -> List[str]:
        text = extract_text_content(user_query).strip()
        queries = [text] if text else []

        year_match = re.search(r"(20\d{2})", text)
        question_match = re.search(r"(?:problem|question)\s*([A-Za-z])|([A-Za-z])\s*题", text, re.IGNORECASE)
        year = year_match.group(1) if year_match else ""
        question = (question_match.group(1) or question_match.group(2)).upper() if question_match else ""

        if year and question:
            queries.extend(
                [
                    f"{year} MCM ICM Problem {question} official problem requirements modeling constraints",
                    f"{year} ICM Problem {question} award paper mathematical modeling structure assumptions sensitivity analysis",
                    f"{year} ICM Problem {question} paper table of contents section headings writing style results discussion",
                    f"{year} ICM Problem {question} award paper figures charts tables captions visual style",
                ]
            )
        if question == "F" or "cyber" in text.lower() or "网络" in text:
            queries.extend(
                [
                    "Cyber Strong national cybersecurity policy effectiveness GCI DEA clustering model",
                    "Global Cybersecurity Index policy effectiveness pattern recognition sensitivity analysis",
                ]
            )
        queries.append("MCM ICM award paper figures charts tables captions visualization style result explanation")
        queries.append("MCM ICM award paper table of contents section structure abstract model solution sensitivity strengths weaknesses writing style")

        deduped: List[str] = []
        for query in queries:
            query = query.strip()
            if query and query not in deduped:
                deduped.append(query)
        return deduped[:6]

    def _conversation_text_for_target(self, messages: List[Any], shared_mem: Optional[Dict[str, Any]] = None) -> str:
        parts: List[str] = []
        for msg in messages[-8:]:
            text = extract_text_content(msg).strip()
            if text:
                parts.append(text)
        if isinstance(shared_mem, dict):
            raw_doc = extract_text_content(shared_mem.get("raw_document_content", "")).strip()
            if raw_doc:
                parts.append(f"[Uploaded Problem Document]\n{raw_doc[:3500]}")
        return "\n".join(parts)[:6000]

    def _default_search_target_profile(self, user_query: str) -> Dict[str, Any]:
        text = extract_text_content(user_query).strip()
        year, question = self._extract_target_year_question(text)
        lower = text.lower()
        keywords: List[str] = []
        for kw in ["cybersecurity", "policy", "network security", "ranking", "sensitivity"]:
            if kw in lower:
                keywords.append(kw)
        return {
            "year": year,
            "question": question,
            "keywords": keywords,
        }

    def _normalize_search_target_profile(self, raw: Any, fallback_query: str) -> Dict[str, Any]:
        fallback = self._default_search_target_profile(fallback_query)
        if not isinstance(raw, dict):
            return fallback
        year = str(raw.get("year", "")).strip()
        if not re.fullmatch(r"20\d{2}", year):
            year = fallback.get("year", "")
        question = str(raw.get("question", "")).strip().upper()
        if question not in {"A", "B", "C", "D", "E", "F"}:
            question = fallback.get("question", "")
        kws = raw.get("keywords")
        if not isinstance(kws, list):
            kws = raw.get("topic_keywords")
        keywords: List[str] = []
        if isinstance(kws, list):
            for item in kws:
                text = str(item).strip()
                if text:
                    keywords.append(text)
        keywords = list(dict.fromkeys(keywords))[:10]
        return {
            "year": year,
            "question": question,
            "keywords": keywords,
        }

    def _extract_search_target_profile(
        self,
        user_query: str,
        conversation_text: str,
        api_key: str,
        model_id: str,
    ) -> Dict[str, Any]:
        fallback = self._default_search_target_profile(user_query)
        if not api_key:
            return fallback
        target_prompt = (
            "You are an extraction assistant. Read the conversation and extract retrieval constraints.\n"
            "Return strict JSON only after marker SEARCH_TARGET_JSON with this schema:\n"
            "{\n"
            '  "year": "2023 or empty",\n'
            '  "question": "A-F or empty",\n'
            '  "keywords": ["..."]\n'
            "}\n"
            "Rules:\n"
            "- If user asks a specific MCM/ICM year/problem letter, fill them exactly.\n"
            "- If uncertain, keep empty instead of guessing.\n"
            "- keywords should include domain words useful for RAG retrieval.\n"
            "- Official contest rules/guidelines are injected by the system by default; do not include a flag for that.\n\n"
            f"[Conversation]\n{conversation_text}\n\n"
            "SEARCH_TARGET_JSON:"
        )
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_id,
                contents=[types.Part.from_text(text=target_prompt)],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            parsed = self._extract_marked_json(getattr(response, "text", "") or "", "SEARCH_TARGET_JSON")
            return self._normalize_search_target_profile(parsed, user_query)
        except Exception:
            return fallback

    def _extract_target_year_question(self, user_query: str) -> tuple[str, str]:
        text = extract_text_content(user_query).strip()
        if not text:
            return "", ""
        year_match = re.search(r"(20\d{2})", text)
        year = year_match.group(1) if year_match else ""

        # 支持 “2023F题”“2023 f 题”“Problem F”“Question F” 等写法
        compact_match = re.search(r"20\d{2}\s*[-_/ ]*\s*([A-Fa-f])\s*题?", text, re.IGNORECASE)
        explicit_match = re.search(r"(?:problem|question)\s*([A-Fa-f])", text, re.IGNORECASE)
        zh_match = re.search(r"([A-Fa-f])\s*题", text, re.IGNORECASE)
        letter = ""
        if compact_match:
            letter = compact_match.group(1)
        elif explicit_match:
            letter = explicit_match.group(1)
        elif zh_match:
            letter = zh_match.group(1)
        question = letter.upper() if letter else ""
        return year, question

    def _hit_meta(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        meta = hit.get("metadata")
        return meta if isinstance(meta, dict) else {}

    def _hit_year(self, hit: Dict[str, Any]) -> str:
        meta = self._hit_meta(hit)
        year = str(meta.get("year", "")).strip()
        return year if re.fullmatch(r"20\d{2}", year) else ""

    def _hit_question(self, hit: Dict[str, Any]) -> str:
        meta = self._hit_meta(hit)
        q = str(meta.get("question", "")).strip().upper()
        return q if q in {"A", "B", "C", "D", "E", "F"} else ""

    def _hit_year_int(self, hit: Dict[str, Any]) -> int:
        year = self._hit_year(hit)
        return int(year) if year else 0

    def _official_rule_filename_rank(self, hit: Dict[str, Any]) -> int:
        meta = self._hit_meta(hit)
        filename = str(meta.get("filename", "")).strip().lower()
        header = str(meta.get("chunk_header", "")).strip().lower()
        haystack = f"{filename} {header}"
        official_names = [
            "contest_ai_policy",
            "mcm-icm_subprocess",
            "mcm-icm_summary",
        ]
        for idx, name in enumerate(official_names):
            if name in haystack:
                return idx
        return 99

    def _is_named_official_rule_hit(self, hit: Dict[str, Any]) -> bool:
        return self._official_rule_filename_rank(hit) < 99

    def _is_official_background_hit(self, hit: Dict[str, Any]) -> bool:
        meta = self._hit_meta(hit)
        category = str(meta.get("category", "")).strip().lower()
        source = str(meta.get("source", "")).strip().lower()
        filename = str(meta.get("filename", "")).strip().lower()
        header = str(meta.get("chunk_header", "")).strip().lower()
        return (
            self._is_named_official_rule_hit(hit)
            or
            category in {"problem", "guideline"}
            or "official problem" in source
            or "rules and guidelines" in source
            or "summary" in filename
            or "guideline" in filename
            or "rule" in filename
            or "submission" in filename
            or "summary" in header
            or "guideline" in header
            or "rule" in header
        )

    def _official_background_queries(self, year: str) -> List[str]:
        queries = [
            "Contest_AI_Policy.pdf MCM ICM AI policy official rules latest",
            "MCM-ICM_SubProcess.pdf MCM ICM submission process official rules latest",
            "MCM-ICM_Summary.docx MCM ICM summary sheet official format latest",
            "COMAP MCM ICM contest rules and guidelines paper format summary sheet",
            "MCM ICM official problem description submission format requirements",
        ]
        if year:
            queries.insert(0, f"{year} Contest_AI_Policy.pdf MCM ICM AI policy official rules")
            queries.insert(1, f"{year} MCM-ICM_SubProcess.pdf MCM ICM submission process official rules")
            queries.insert(2, f"{year} MCM-ICM_Summary.docx MCM ICM summary sheet official format")
            queries.insert(0, f"{year} COMAP MCM ICM contest rules guidelines summary sheet submission requirements")
            queries.insert(1, f"{year} COMAP MCM ICM official problem description format requirements")
        return queries

    def _rerank_hits_for_target(
        self,
        hits: List[Dict[str, Any]],
        year: str,
        question: str,
    ) -> List[Dict[str, Any]]:
        if not hits:
            return []

        official_hits = [h for h in hits if self._is_official_background_hit(h)]
        regular_hits = [h for h in hits if not self._is_official_background_hit(h)]
        original_regular = list(regular_hits)

        if question:
            same_question = [h for h in regular_hits if self._hit_question(h) == question]
            if same_question:
                regular_hits = same_question
        if year:
            same_year = [h for h in regular_hits if self._hit_year(h) == year]
            if same_year:
                regular_hits = same_year

        if len(regular_hits) < 4:
            seen_ids = {str(h.get("id", "")) for h in regular_hits}
            for hit in original_regular:
                hid = str(hit.get("id", ""))
                if hid and hid in seen_ids:
                    continue
                regular_hits.append(hit)
                if hid:
                    seen_ids.add(hid)
                if len(regular_hits) >= 6:
                    break

        named_rule_hits = [h for h in official_hits if self._is_named_official_rule_hit(h)]
        other_official_hits = [h for h in official_hits if h not in named_rule_hits]
        named_rule_hits.sort(key=lambda h: (self._official_rule_filename_rank(h), -self._hit_year_int(h)))
        if year:
            same_year_official = [h for h in other_official_hits if self._hit_year(h) == year]
            fallback_official = [h for h in other_official_hits if h not in same_year_official]
            other_official_hits = same_year_official + sorted(
                fallback_official,
                key=lambda h: -self._hit_year_int(h),
            )
        else:
            other_official_hits = sorted(other_official_hits, key=lambda h: -self._hit_year_int(h))
        official_hits = named_rule_hits[:3] + other_official_hits

        final_hits: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for hit in official_hits[:4] + regular_hits[:6]:
            hid = str(hit.get("id", "")).strip()
            if hid and hid in seen:
                continue
            if hid:
                seen.add(hid)
            final_hits.append(hit)
        return final_hits[:8]

    def _default_figure_style_learning(self) -> Dict[str, Any]:
        return {
            "observed_figures": [],
            "common_chart_patterns": [
                "Use figures to support a specific modeling claim rather than as decoration.",
                "Introduce each figure before display and interpret the quantitative implication immediately after display.",
                "Prefer readable labels, restrained colors, and captions that state the analytical purpose.",
            ],
            "divergent_chart_patterns": [
                "Flowcharts are useful for model construction or decision pipelines.",
                "Heatmaps and radar charts are useful for multi-indicator comparison.",
                "Line charts and bar charts are useful for sensitivity, ranking, and scenario comparison.",
            ],
            "recommended_chart_style": {
                "visual_style": "clean academic style with high contrast, clear axis labels, and concise legends",
                "caption_pattern": "Figure N. What is measured, under which scenario, and what conclusion it supports.",
                "text_explanation_pattern": "Before figure: state why it is needed. After figure: interpret trend, outlier, ranking, or robustness implication.",
                "color_style": "limited palette, consistent colors across related charts, avoid decorative gradients",
            },
            "recommended_chart_plan_adjustments": [],
        }

    def _extract_figure_style_learning(self, fused_context: str) -> Dict[str, Any]:
        parsed = self._extract_marked_json(fused_context, "FIGURE_STYLE_LEARNING_JSON")
        if not isinstance(parsed, dict):
            return self._default_figure_style_learning()
        fallback = self._default_figure_style_learning()
        for key in ["observed_figures", "common_chart_patterns", "divergent_chart_patterns", "recommended_chart_plan_adjustments"]:
            if not isinstance(parsed.get(key), list):
                parsed[key] = fallback[key]
        if not isinstance(parsed.get("recommended_chart_style"), dict):
            parsed["recommended_chart_style"] = fallback["recommended_chart_style"]
        return parsed

    def _resolve_rag_image_path(self, raw_path: Any) -> Optional[Path]:
        if not raw_path:
            return None
        path_text = str(raw_path).strip()
        if not path_text:
            return None
        candidates = []
        p = Path(path_text)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(Path(rag_engine.project_root) / p)
            candidates.append(Path(rag_engine.project_root).parent / p)
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue
        return None

    def _collect_rag_image_inputs(
        self,
        search_results: List[Dict[str, Any]],
        max_images: int = 6,
    ) -> tuple[List[Any], List[Dict[str, Any]]]:
        parts: List[Any] = []
        meta: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for hit_idx, hit in enumerate(search_results or [], start=1):
            images = hit.get("images") if isinstance(hit, dict) else []
            if not isinstance(images, list):
                continue
            for image_idx, raw_path in enumerate(images, start=1):
                resolved = self._resolve_rag_image_path(raw_path)
                if not resolved:
                    continue
                key = str(resolved.resolve())
                if key in seen:
                    continue
                seen.add(key)
                mime_type = mimetypes.guess_type(str(resolved))[0] or "image/png"
                if not mime_type.startswith("image/"):
                    continue
                try:
                    data = resolved.read_bytes()
                except OSError:
                    continue
                if not data or len(data) > 8 * 1024 * 1024:
                    continue
                label = f"masked_reference_{hit_idx}_figure_{image_idx}"
                parts.append(types.Part.from_text(text=f"\n[Attached RAG Figure: {label}]\n"))
                parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
                meta.append(
                    {
                        "label": label,
                        "hit_id": hit.get("id"),
                        "query": hit.get("query"),
                        "path": str(resolved),
                        "mime_type": mime_type,
                    }
                )
                if len(meta) >= max_images:
                    return parts, meta
        return parts, meta

    def _default_paper_structure_learning(self) -> Dict[str, Any]:
        recommended_sections = [
            "Summary",
            "Problem Restatement",
            "Assumptions and Notations",
            "Problem Analysis",
            "Model Establishment",
            "Model Solution and Results",
            "Sensitivity Analysis",
            "Model Evaluation",
            "Conclusion",
        ]
        return {
            "observed_structures": [],
            "common_patterns": [
                "Begin with a compact summary that states the problem, method, main results, and conclusions.",
                "Move from problem restatement and assumptions into model construction before numerical solution.",
                "Use sensitivity analysis and model evaluation to justify robustness and limitations.",
            ],
            "divergent_patterns": [
                "Some papers split data preprocessing or validation into independent sections when data work is central.",
                "Policy-oriented ICM papers may replace a generic conclusion with targeted recommendations.",
            ],
            "recommended_sections": recommended_sections,
            "recommended_section_templates": {
                "Summary": ["problem context", "model strategy", "key results", "robustness", "recommendations"],
                "Problem Restatement": ["original task translation", "subproblem decomposition", "success criteria"],
                "Assumptions and Notations": ["assumptions with justification", "symbols and units", "model boundaries"],
                "Problem Analysis": ["mechanism analysis", "data and constraints", "modeling route"],
                "Model Establishment": ["variables", "objective functions", "constraints", "algorithm or workflow"],
                "Model Solution and Results": ["solution process", "parameter settings", "numerical outputs", "interpretation"],
                "Sensitivity Analysis": ["key parameters", "perturbation design", "ranking/output changes", "robustness conclusion"],
                "Model Evaluation": ["strengths", "weaknesses", "applicability", "improvements"],
                "Conclusion": ["answers to subproblems", "policy/practical implications", "final recommendations"],
            },
            "recommended_writing_style": {
                "tone": "formal, concise, evidence-driven mathematical modeling prose",
                "paragraph_pattern": "claim -> model/evidence -> quantitative interpretation -> implication",
                "figure_table_usage": "Every key figure should be introduced before display and interpreted after display.",
                "math_density": "Use equations where they clarify objectives, constraints, or scoring logic; avoid decorative formulas.",
            },
        }

    def _extract_paper_structure_learning(self, fused_context: str) -> Dict[str, Any]:
        parsed = self._extract_marked_json(fused_context, "PAPER_STRUCTURE_LEARNING_JSON")
        if not isinstance(parsed, dict):
            return self._default_paper_structure_learning()
        fallback = self._default_paper_structure_learning()
        recommended_sections = parsed.get("recommended_sections")
        if not isinstance(recommended_sections, list) or not any(str(x).strip() for x in recommended_sections):
            parsed["recommended_sections"] = fallback["recommended_sections"]
        else:
            parsed["recommended_sections"] = [str(x).strip() for x in recommended_sections if str(x).strip()]
        templates = parsed.get("recommended_section_templates")
        if not isinstance(templates, dict) or not templates:
            parsed["recommended_section_templates"] = fallback["recommended_section_templates"]
        style = parsed.get("recommended_writing_style")
        if not isinstance(style, dict) or not style:
            parsed["recommended_writing_style"] = fallback["recommended_writing_style"]
        for key in ["observed_structures", "common_patterns", "divergent_patterns"]:
            if not isinstance(parsed.get(key), list):
                parsed[key] = fallback[key]
        return parsed

    def _is_informative_rag_hit(self, hit: Dict[str, Any]) -> bool:
        content = extract_text_content(hit.get("content", "")).strip()
        if len(content) < 240:
            return False
        low_info_markers = [
            "team control number",
            "this document is an",
            "mathematical contest in modeling",
        ]
        lowered = content.lower()
        if len(content) < 500 and sum(marker in lowered for marker in low_info_markers) >= 2:
            return False
        return True

    def _search_rag_multi(
        self,
        user_query: str,
        api_key: str,
        target_profile: Optional[Dict[str, Any]] = None,
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        profile = target_profile if isinstance(target_profile, dict) else {}
        target_year = str(profile.get("year", "")).strip()
        target_question = str(profile.get("question", "")).strip().upper()
        keywords = profile.get("keywords")
        if not isinstance(keywords, list):
            keywords = profile.get("topic_keywords")
        keywords = [str(x).strip() for x in keywords if str(x).strip()] if isinstance(keywords, list) else []
        queries = self._build_rag_queries(user_query)
        if target_year and target_question:
            queries.insert(0, f"{target_year} MCM ICM Problem {target_question}")
        if keywords:
            queries.append("MCM ICM " + " ".join(keywords[:8]))
        queries.extend(self._official_background_queries(target_year))
        deduped_queries: List[str] = []
        for query in queries:
            q = str(query).strip()
            if q and q not in deduped_queries:
                deduped_queries.append(q)
        queries = deduped_queries[:10]
        informative: Dict[str, Dict[str, Any]] = {}
        fallback: Dict[str, Dict[str, Any]] = {}
        for query in queries:
            for hit in rag_engine.search(query, n_results=5, api_key=api_key):
                if not isinstance(hit, dict):
                    continue
                hid = str(hit.get("id", "")).strip()
                if not hid:
                    continue
                enriched = dict(hit)
                enriched["query"] = query
                if self._is_informative_rag_hit(enriched):
                    informative.setdefault(hid, enriched)
                else:
                    fallback.setdefault(hid, enriched)

        selected = list(informative.values())
        if len(selected) < 5:
            for hit in fallback.values():
                if str(hit.get("id", "")) not in {str(x.get("id", "")) for x in selected}:
                    selected.append(hit)
                if len(selected) >= 5:
                    break
        selected = self._rerank_hits_for_target(selected, target_year, target_question)
        return selected[:8], queries

    def _rag_hit_context_text(self, hit: Dict[str, Any], index: int) -> str:
        content = extract_text_content(hit.get("content", "")).strip()
        meta = self._hit_meta(hit)
        source = str(meta.get("source", "")).strip()
        year = str(meta.get("year", "")).strip()
        question = str(meta.get("question", "")).strip()
        category = str(meta.get("category", "")).strip()
        header = str(meta.get("chunk_header", "")).strip()
        limit = 2800 if self._is_official_background_hit(hit) else 3400
        if len(content) > limit:
            content = content[:limit] + "\n...[RAG chunk truncated]..."
        meta_line = f"source={source}, year={year}, question={question}, category={category}, header={header}"
        return f"--- 内部高价值学术片段 (RAG) {index} ---\n[{meta_line}]\n{content}"

    def _build_style_contract(
        self,
        fused_context: str,
        chart_weight_plan: Dict[str, Any],
        paper_structure_learning: Dict[str, Any],
        figure_style_learning: Dict[str, Any],
    ) -> Dict[str, Any]:
        style_guide = self._extract_section_bullets(fused_context, "STYLE_GUIDE")
        modeling_priors = self._extract_section_bullets(fused_context, "MODELING_PRIORS")
        results_narrative = self._extract_section_bullets(fused_context, "RESULTS_NARRATIVE")
        risk_limits = self._extract_section_bullets(fused_context, "RISK_AND_LIMITATIONS")
        writing_blueprint = self._extract_section_bullets(fused_context, "WRITING_BLUEPRINT")
        blueprint_sections = self._sections_from_writing_blueprint(fused_context)
        learned_sections = paper_structure_learning.get("recommended_sections")
        learned_sections = [str(s).strip() for s in learned_sections if str(s).strip()] if isinstance(learned_sections, list) else []

        # 风格合同：给 Writer/Coder 的硬约束摘要
        return {
            "target_language": "English",
            "tone": "formal_technical",
            "evidence_policy": "every core claim must be supported by model/log/chart evidence",
            "required_sections": learned_sections or blueprint_sections or [
                "Summary",
                "Problem Restatement",
                "Assumptions and Notations",
                "Problem Analysis",
                "Model Establishment",
                "Model Solution and Results",
                "Sensitivity Analysis",
                "Model Evaluation",
                "Conclusion",
            ],
            "style_guide": style_guide,
            "modeling_priors": modeling_priors,
            "results_narrative": results_narrative,
            "risk_and_limitations": risk_limits,
            "writing_blueprint": writing_blueprint,
            "paper_structure_learning": paper_structure_learning,
            "figure_style_learning": figure_style_learning,
            "section_templates": paper_structure_learning.get("recommended_section_templates", {}),
            "writing_style_rules": paper_structure_learning.get("recommended_writing_style", {}),
            "chart_style_rules": figure_style_learning.get("recommended_chart_style", {}),
            "forbidden_patterns": [
                "marketing-like adjectives without evidence",
                "metaphorical narrative unrelated to modeling",
                "unbounded claims without quantitative support",
            ],
            "min_chart_count": int(chart_weight_plan.get("required_chart_count", 0) or 0),
            "chart_weight_plan": chart_weight_plan,
        }

    def _extract_chart_plan(self, fused_context: str) -> List[Dict[str, Any]]:
        text = fused_context or ""
        marked = self._extract_marked_json(text, "CHART_PLAN_JSON")
        if isinstance(marked, list):
            candidates = [marked]
        else:
            candidates = []
        # 优先匹配 CHART_PLAN_JSON 标记后的 JSON 数组
        marker_match = re.search(r"CHART_PLAN_JSON\s*[:：]?\s*(\[[\s\S]*?\])", text, re.IGNORECASE)
        if marker_match:
            candidates.append(marker_match.group(1))
        # 其次匹配第一个 JSON 数组代码块
        block_match = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
        if block_match:
            candidates.append(block_match.group(1))
        for raw in candidates:
            try:
                parsed = raw if isinstance(raw, list) else json.loads(raw)
                if isinstance(parsed, list):
                    normalized: List[Dict[str, Any]] = []
                    for item in parsed:
                        if not isinstance(item, dict):
                            continue
                        if not item.get("id") or not item.get("chart_type"):
                            continue
                        normalized.append(
                            {
                                "id": str(item.get("id", "")).strip(),
                                "title": str(item.get("title", item.get("id", ""))).strip(),
                                "chart_type": str(item.get("chart_type", "")).strip(),
                                "purpose": str(item.get("purpose", "")).strip(),
                                "required_inputs": item.get("required_inputs", []),
                                "section_hint": str(item.get("section_hint", "")).strip(),
                                "priority": str(item.get("priority", "medium")).strip().lower(),
                                "fallback_if_missing": str(item.get("fallback_if_missing", "")).strip(),
                            }
                        )
                    if normalized:
                        return normalized
            except Exception:
                continue
        return self._default_chart_plan()

    def _extract_chart_weight_plan(
        self,
        fused_context: str,
        chart_plan: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        text = fused_context or ""
        marked = self._extract_marked_json(text, "CHART_WEIGHT_PLAN_JSON")
        marker_match = re.search(
            r"CHART_WEIGHT_PLAN_JSON\s*[:：]?\s*(\{[\s\S]*?\})",
            text,
            re.IGNORECASE,
        )
        candidates: List[str] = []
        if isinstance(marked, dict):
            candidates.append(json.dumps(marked, ensure_ascii=False))
        if marker_match:
            candidates.append(marker_match.group(1))
        block_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if block_match:
            candidates.append(block_match.group(1))

        parsed_obj: Dict[str, Any] = {}
        for raw in candidates:
            try:
                loaded = json.loads(raw)
            except Exception:
                continue
            if isinstance(loaded, dict):
                parsed_obj = loaded
                break

        weights_raw = parsed_obj.get("weights") if isinstance(parsed_obj, dict) else None
        weights: List[Dict[str, Any]] = []
        priority_map: Dict[str, str] = {}
        for cp in chart_plan:
            if not isinstance(cp, dict):
                continue
            cid = str(cp.get("id", "")).strip()
            pri = str(cp.get("priority", "medium")).strip().lower()
            if cid:
                priority_map[cid] = pri if pri in {"high", "medium", "low"} else "medium"

        if isinstance(weights_raw, list):
            for item in weights_raw:
                if not isinstance(item, dict):
                    continue
                cid = str(item.get("id", "")).strip()
                if not cid:
                    continue
                pri = str(item.get("priority", priority_map.get(cid, "medium"))).strip().lower()
                if pri not in {"high", "medium", "low"}:
                    pri = priority_map.get(cid, "medium")
                try:
                    w = float(item.get("weight", 0.7))
                except Exception:
                    w = 0.7
                w = max(0.3, min(1.0, w))
                weights.append(
                    {
                        "id": cid,
                        "priority": pri,
                        "weight": w,
                        "reason": str(item.get("reason", "")).strip(),
                    }
                )

        # 保证每个计划图都有权重
        fallback_weight = {"high": 1.0, "medium": 0.7, "low": 0.4}
        by_id = {str(w.get("id", "")): w for w in weights}
        for cp in chart_plan:
            if not isinstance(cp, dict):
                continue
            cid = str(cp.get("id", "")).strip()
            if not cid or cid in by_id:
                continue
            pri = str(cp.get("priority", "medium")).strip().lower()
            if pri not in {"high", "medium", "low"}:
                pri = "medium"
            weights.append(
                {
                    "id": cid,
                    "priority": pri,
                    "weight": fallback_weight.get(pri, 0.7),
                    "reason": "fallback_by_priority",
                }
            )

        # 单调约束：high >= medium >= low
        high_vals = [float(w["weight"]) for w in weights if w.get("priority") == "high"]
        med_vals = [float(w["weight"]) for w in weights if w.get("priority") == "medium"]
        low_vals = [float(w["weight"]) for w in weights if w.get("priority") == "low"]
        high_floor = max(high_vals) if high_vals else 1.0
        med_floor = max(med_vals) if med_vals else 0.7
        low_floor = max(low_vals) if low_vals else 0.4
        med_floor = min(med_floor, high_floor)
        low_floor = min(low_floor, med_floor)
        for w in weights:
            pri = str(w.get("priority", "medium")).lower()
            if pri == "high":
                w["weight"] = max(float(w.get("weight", 1.0)), high_floor)
            elif pri == "medium":
                w["weight"] = min(max(float(w.get("weight", 0.7)), low_floor), high_floor)
            else:
                w["weight"] = min(float(w.get("weight", 0.4)), med_floor)
            w["weight"] = max(0.3, min(1.0, float(w["weight"])))

        required = int(math.ceil(sum(float(w.get("weight", 0.0)) for w in weights))) if weights else 0
        parsed_required = self._positive_int(
            parsed_obj.get("required_chart_count") if isinstance(parsed_obj, dict) else None,
            0,
        )
        if parsed_required:
            required = max(required, parsed_required)
        required = max(required, len([w for w in weights if str(w.get("priority", "")).lower() == "high"]))

        return {
            "weights": weights,
            "required_chart_count": required,
        }

    def _extract_fusion_contract_raw(self, fused_context: str) -> Dict[str, Any]:
        text = fused_context or ""
        marked = self._extract_marked_json(text, "FUSION_CONTRACT_JSON")
        if isinstance(marked, dict) and "writing_contract" in marked:
            return marked
        marker_match = re.search(
            r"FUSION_CONTRACT_JSON\s*[:：]?\s*(\{[\s\S]*\})",
            text,
            re.IGNORECASE,
        )
        candidates: List[str] = []
        if marker_match:
            candidates.append(marker_match.group(1))
        block_match = re.search(r"```json\s*(\{[\s\S]*\})\s*```", text, re.IGNORECASE)
        if block_match:
            candidates.append(block_match.group(1))

        for raw in candidates:
            try:
                loaded = json.loads(raw)
                if isinstance(loaded, dict) and "writing_contract" in loaded:
                    return loaded
            except Exception:
                continue
        return {}

    def _build_fusion_contract(
        self,
        fused_context: str,
        chart_plan: List[Dict[str, Any]],
        chart_weight_plan: Dict[str, Any],
        style_contract: Dict[str, Any],
        paper_structure_learning: Dict[str, Any],
        figure_style_learning: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw = self._extract_fusion_contract_raw(fused_context)
        learned_required_sections = paper_structure_learning.get("recommended_sections")
        learned_required_sections = (
            [str(s).strip() for s in learned_required_sections if str(s).strip()]
            if isinstance(learned_required_sections, list)
            else []
        )
        required_sections = learned_required_sections or style_contract.get("required_sections") or [
            "Summary",
            "Problem Restatement",
            "Assumptions and Notations",
            "Problem Analysis",
            "Model Establishment",
            "Model Solution and Results",
            "Sensitivity Analysis",
            "Model Evaluation",
            "Conclusion",
        ]
        required_sections = [str(s).strip() for s in required_sections if str(s).strip()]
        if not required_sections:
            required_sections = ["Summary", "Problem Restatement", "Model Establishment", "Model Solution and Results", "Conclusion"]

        required_chart_count = self._positive_int(chart_weight_plan.get("required_chart_count"), 0)
        high_priority_count = len(
            [
                c
                for c in (chart_plan or [])
                if isinstance(c, dict) and str(c.get("priority", "")).strip().lower() == "high"
            ]
        )
        required_chart_count = max(required_chart_count, high_priority_count, 1)

        task_requirements = raw.get("task_requirements") if isinstance(raw.get("task_requirements"), dict) else {}
        writing_contract = raw.get("writing_contract") if isinstance(raw.get("writing_contract"), dict) else {}
        chart_contract = raw.get("chart_contract") if isinstance(raw.get("chart_contract"), dict) else {}
        evidence_contract = raw.get("evidence_contract") if isinstance(raw.get("evidence_contract"), dict) else {}
        quality_gates = raw.get("quality_gates") if isinstance(raw.get("quality_gates"), dict) else {}

        normalized = {
            "task_requirements": {
                "problem_type": str(task_requirements.get("problem_type", "general_modeling")).strip() or "general_modeling",
                "primary_objective": str(task_requirements.get("primary_objective", "")).strip(),
                "key_constraints": task_requirements.get("key_constraints", []),
                "success_criteria": task_requirements.get("success_criteria", []),
            },
            "writing_contract": {
                "target_language": str(
                    writing_contract.get("target_language", style_contract.get("target_language", "English"))
                ).strip()
                or "English",
                "tone": str(writing_contract.get("tone", style_contract.get("tone", "formal_technical"))).strip()
                or "formal_technical",
                "required_sections": required_sections or writing_contract.get("required_sections", required_sections),
                "section_focus": writing_contract.get("section_focus", {}),
                "section_templates": writing_contract.get(
                    "section_templates",
                    paper_structure_learning.get("recommended_section_templates", {}),
                ),
                "writing_style_rules": writing_contract.get(
                    "writing_style_rules",
                    paper_structure_learning.get("recommended_writing_style", {}),
                ),
                "paper_structure_learning": paper_structure_learning,
                "forbidden_patterns": writing_contract.get(
                    "forbidden_patterns",
                    style_contract.get("forbidden_patterns", []),
                ),
            },
            "chart_contract": {
                "required_chart_count": self._positive_int(
                    chart_contract.get("required_chart_count"),
                    required_chart_count,
                ),
                "must_include_high_priority": bool(chart_contract.get("must_include_high_priority", True)),
                "chart_plan": chart_contract.get("chart_plan", chart_plan),
                "chart_weight_plan": chart_contract.get("chart_weight_plan", chart_weight_plan),
                "figure_style_learning": figure_style_learning,
                "chart_style_rules": chart_contract.get(
                    "chart_style_rules",
                    figure_style_learning.get("recommended_chart_style", {}),
                ),
            },
            "evidence_contract": {
                "core_claim_policy": str(
                    evidence_contract.get(
                        "core_claim_policy",
                        style_contract.get("evidence_policy", "every core claim must be supported by model/log/chart evidence"),
                    )
                ).strip()
                or "every core claim must be supported by model/log/chart evidence",
                "required_evidence_types": evidence_contract.get(
                    "required_evidence_types",
                    ["model", "code", "chart", "execution_log"],
                ),
                "results_requirements": evidence_contract.get(
                    "results_requirements",
                    ["comparison", "sensitivity_or_robustness", "limitations"],
                ),
            },
            "quality_gates": {
                "draft_min_chars": self._positive_int(quality_gates.get("draft_min_chars"), 12000),
                "min_figure_refs": self._positive_int(quality_gates.get("min_figure_refs"), 4),
                "mandatory_sections": quality_gates.get("mandatory_sections", required_sections),
                "chart_coverage_gate": quality_gates.get(
                    "chart_coverage_gate",
                    {
                        "required_chart_count": required_chart_count,
                        "must_include_high_priority": True,
                    },
                ),
            },
        }

        # 统一关键下限，不允许比 chart_weight_plan 更松
        normalized["chart_contract"]["required_chart_count"] = max(
            int(normalized["chart_contract"]["required_chart_count"] or 0),
            required_chart_count,
        )
        gate = normalized["quality_gates"]["chart_coverage_gate"]
        gate_scalar_required = self._positive_int(gate, 0)
        if not isinstance(gate, dict):
            gate = {}
        gate_required = self._positive_int(gate.get("required_chart_count"), gate_scalar_required or required_chart_count)
        gate["required_chart_count"] = max(gate_required, required_chart_count)
        gate["must_include_high_priority"] = bool(gate.get("must_include_high_priority", True))
        normalized["quality_gates"]["chart_coverage_gate"] = gate

        # required_sections / mandatory_sections 归一化
        ws = normalized["writing_contract"].get("required_sections", required_sections)
        ws = [str(s).strip() for s in (ws if isinstance(ws, list) else required_sections) if str(s).strip()]
        ws = ws or required_sections

        ms = normalized["quality_gates"].get("mandatory_sections", required_sections)
        ms = [str(s).strip() for s in (ms if isinstance(ms, list) else required_sections) if str(s).strip()]
        ms = ms or required_sections
        for title in ms:
            if title not in ws:
                ws.append(title)
        normalized["writing_contract"]["required_sections"] = ws
        normalized["quality_gates"]["mandatory_sections"] = ms

        focus = normalized["writing_contract"].get("section_focus")
        if isinstance(focus, str) and focus.strip():
            normalized["writing_contract"]["section_focus"] = {"global": focus.strip()}
        elif not isinstance(focus, dict):
            normalized["writing_contract"]["section_focus"] = {}

        # 修正 chart_contract 内嵌计划，保证与主计划一致
        normalized["chart_contract"]["chart_plan"] = chart_plan
        normalized["chart_contract"]["chart_weight_plan"] = chart_weight_plan

        return normalized

    async def __call__(self, state: AgentState):
        logger.info("Fusion node entered")
        shared_mem = state.get("shared_memory", {})
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-3.1-flash-lite"
        
        messages = state.get("messages", [])
        if not messages:
            return {"context": "No query provided.", "stage": make_stage("fusion_skipped_no_query", "Fusion skipped: no query")}
            
        # 提取最后一条消息作为意图载体
        last_message = extract_text_content(messages[-1])
        conversation_text = self._conversation_text_for_target(messages, shared_mem)
        target_profile = self._extract_search_target_profile(
            user_query=last_message,
            conversation_text=conversation_text or last_message,
            api_key=api_key,
            model_id=model_id,
        )
        
        await broadcast_progress("Retrieve(Fusion)", "正在启动双轨检索引擎 (Local RAG + Web Search)...", 20)
        
        # 1. 发兵本地 ChromaDB
        rag_engine_ready = rag_engine.collection is not None
        rag_queries = self._build_rag_queries(last_message)
        search_results: List[Dict[str, Any]] = []
        if api_key:
            search_results, rag_queries = self._search_rag_multi(
                last_message,
                api_key=api_key,
                target_profile=target_profile,
            )
        
        rag_context_parts = []
        for i, res in enumerate(search_results):
            rag_context_parts.append(self._rag_hit_context_text(res, i + 1))
        rag_context = "\n".join(rag_context_parts) if rag_context_parts else "未在本地发现高度匹配的学术先例。"
        rag_image_parts, rag_image_meta = self._collect_rag_image_inputs(search_results)

        web_grounding = "skipped"
        web_error = ""
        fusion_query = ""

        # 2. 调动 Gemini 原生搜索并执行 LLM “动态加权融合”
        if not api_key:
            task_id = str(shared_mem.get("task_id") or "").strip()
            audit_meta = save_fusion_audit(
                task_id=task_id,
                query_text="\n".join(rag_queries),
                rag_hits=search_results,
                rag_assembled_context=rag_context,
                rag_engine_ready=rag_engine_ready,
                fusion_guidance="[系统错误] 缺少任务 API Key，Fusion 节点无法执行联网融合检索。",
                fusion_contract={},
                chart_plan=[],
                chart_weight_plan={},
                style_contract={},
                web_grounding="failed_missing_api_key",
                fusion_query="",
                web_error="missing_task_api_key",
            )
            fail_mem = shared_mem.copy()
            fail_mem["fusion_rag_context"] = rag_context
            fail_mem["fusion_rag_hits"] = json_safe(search_results)
            fail_mem["fusion_audit_meta"] = json_safe(audit_meta)
            return {
                "context": "[系统错误] 缺少任务 API Key，Fusion 节点无法执行联网融合检索。",
                "shared_memory": fail_mem,
                "stage": make_stage("fusion_failed_missing_api_key", "Fusion Failed: Missing Task API Key"),
            }

        fusion_query = (
            f"【用户原始探索意图】：\n{last_message}\n\n"
            f"【系统为你准备的本地 RAG 学术参考（供加权判断使用）】：\n{rag_context}\n\n"
            f"【已附加的 RAG 论文图表图片（供视觉风格学习）】：\n"
            f"{json.dumps(rag_image_meta, ensure_ascii=False, indent=2) if rag_image_meta else '未发现可读取的关联图表图片。'}\n\n"
            f"请立刻使用外置 Google Search 工具打通公网客观数据，并根据【动态加权融合法则】输出一份终版综合简报。"
        )
        web_grounding = "pending"

        try:
            await broadcast_progress("Retrieve(Fusion)", f"检索完成，已触发 {model_id} 进行跨域知识的加权与去伪存真审查...", 60)
            
            # 使用原生 GenAI 客户端以启用 tools Grounding 特性
            client = genai.Client(api_key=api_key)
            logger.info("Fusion generate_content start model_id=%s (grounding+search may block long)", model_id)
            contents = [types.Part.from_text(text=fusion_query)] + rag_image_parts
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.2, # 融合任务需降低幻觉
                    tools=[{"google_search": {}}], # 原生开启谷歌搜索接地
                )
            )
            fused_context = response.text
            web_grounding = "success"
            logger.info("Fusion generate_content done text_len=%s", len(fused_context or ""))
            
        except Exception as e:
            # 安全降级策略：如果没开启搜索权限或网络阻断，回退到原始 RAG
            web_grounding = "fallback_rag_only"
            web_error = str(e)
            print(f"[Fusion Error] Web Grounding Failed, fallback to strict RAG. {e}")
            fused_context = (
                "[系统警告: Google Search Grounding 拒绝或失败，退回防断层模式]\n"
                "[STYLE_GUIDE]\n- 使用严谨技术写法，不做修辞化表达。\n"
                "- 结论必须给出可复核量化依据。\n"
                "- 章节论证遵循问题定义-方法-结果-验证闭环。\n\n"
                "[MODELING_PRIORS]\n- 变量定义、目标函数与约束条件需闭环对齐。\n"
                "- 给出符号体系并保持前后一致。\n"
                "- 模型假设要对应验证与局限性分析。\n\n"
                "[RESULTS_NARRATIVE]\n- 每张图/表需对应解释其结论意义。\n"
                "- 结果比较应给出基准或对照。\n"
                "- 需要包含误差来源与稳健性说明。\n\n"
                "[RISK_AND_LIMITATIONS]\n- 明确数据质量与外推边界。\n"
                "- 给出敏感性分析口径。\n"
                "- 说明模型适用场景与失效条件。\n\n"
                "[WRITING_BLUEPRINT]\n- Abstract: 问题、方法、结果、贡献。\n"
                "- Methods: 假设、变量、公式、求解流程。\n"
                "- Results/Discussion: 图表解释、验证、局限与改进。\n\n"
                f"【本地学术参考回退】\n{rag_context}"
            )
            
        await broadcast_progress("Retrieve(Fusion)", "知识域交叉加权融合完毕，已挂载至工作流上下文。", 100)
        new_memory: Dict[str, Any] = shared_mem.copy()
        chart_plan = self._extract_chart_plan(fused_context)
        chart_weight_plan = self._extract_chart_weight_plan(fused_context, chart_plan)
        paper_structure_learning = self._extract_paper_structure_learning(fused_context)
        figure_style_learning = self._extract_figure_style_learning(fused_context)
        style_contract = self._build_style_contract(
            fused_context,
            chart_weight_plan,
            paper_structure_learning,
            figure_style_learning,
        )
        fusion_contract = self._build_fusion_contract(
            fused_context=fused_context,
            chart_plan=chart_plan,
            chart_weight_plan=chart_weight_plan,
            style_contract=style_contract,
            paper_structure_learning=paper_structure_learning,
            figure_style_learning=figure_style_learning,
        )
        new_memory["fusion_guidance"] = fused_context
        new_memory["fusion_rag_context"] = rag_context
        new_memory["fusion_raw_content"] = json_safe(fused_context)
        new_memory["fusion_chart_plan"] = json_safe(chart_plan)
        new_memory["fusion_chart_weight_plan"] = json_safe(chart_weight_plan)
        new_memory["fusion_style_contract"] = json_safe(style_contract)
        new_memory["fusion_contract"] = json_safe(fusion_contract)
        new_memory["fusion_paper_structure_learning"] = json_safe(paper_structure_learning)
        new_memory["fusion_figure_style_learning"] = json_safe(figure_style_learning)
        new_memory["fusion_rag_image_meta"] = json_safe(rag_image_meta)
        new_memory["fusion_search_target_profile"] = json_safe(target_profile)

        task_id = str(shared_mem.get("task_id") or "").strip()
        if not task_id:
            logger.warning("Fusion audit not saved: shared_memory.task_id is empty")
        audit_meta = save_fusion_audit(
            task_id=task_id,
            query_text="\n".join(rag_queries),
            rag_hits=search_results,
            rag_assembled_context=rag_context,
            rag_engine_ready=rag_engine_ready,
            fusion_guidance=fused_context or "",
            fusion_contract=fusion_contract,
            chart_plan=chart_plan,
            chart_weight_plan=chart_weight_plan,
            style_contract=style_contract,
            paper_structure_learning=paper_structure_learning,
            figure_style_learning=figure_style_learning,
            rag_image_meta=rag_image_meta,
            web_grounding=web_grounding,
            fusion_query=fusion_query,
            web_error=web_error,
        )
        new_memory["fusion_rag_hits"] = json_safe(search_results)
        new_memory["fusion_audit_meta"] = json_safe(audit_meta)
        
        return {
            "context": fused_context,
            "shared_memory": new_memory,
            "stage": make_stage("fusion_completed", "Web Grounding & Dynamic Fusion Completed"),
        }

fusion_node = FusionNode()
