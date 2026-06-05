import json
import re
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress
from backend.services.serialization import (
    ensure_blocks,
    extract_text_content,
    json_safe,
    make_stage,
    messages_for_llm,
)
from backend.services.multimodal import collect_multimodal_images, image_meta_summary

_FUSION_INTERNAL_SECTIONS = (
    "STYLE_GUIDE",
    "MODELING_PRIORS",
    "RESULTS_NARRATIVE",
    "RISK_AND_LIMITATIONS",
    "WRITING_BLUEPRINT",
)

# 仅在完全缺少 Fusion 合同时用于数学建模论文兜底篇幅权重
_SECTION_CHAR_WEIGHTS: Dict[str, float] = {
    "title": 0.03,
    "abstract": 0.10,
    "problem_restatement": 0.10,
    "assumptions": 0.09,
    "problem_analysis": 0.12,
    "model_establishment": 0.22,
    "model_solution": 0.18,
    "sensitivity": 0.08,
    "model_evaluation": 0.07,
    "conclusion": 0.04,
}


class WriterNode:
    """
    重构版 Writer：
    1) outline：产出严格论文结构 + 图表绑定关系；
    2) draft：按章节最小篇幅与质量约束生成正文，并将图表插入对应章节。
    """

    def __init__(self):
        self.outline_system_prompt = (
            "You are an award-level MCM/ICM paper architect.\n"
            "Generate only structured planning JSON for a full paper, not prose.\n"
            "Mandatory output language: English (unless user explicitly requested another language).\n"
            "The plan must include title, abstract, and all major sections with enforceable minimum lengths.\n"
            "Figure IDs must map to chart plan IDs and section hints."
        )
        self.draft_system_prompt = (
            "You are an award-level MCM/ICM technical writer.\n"
            "Write rigorous, evidence-grounded paper sections from upstream artifacts.\n"
            "Mandatory output language: English (unless user explicitly requested another language).\n"
            "No orchestration metadata, no Fusion/RAG mention, no placeholder-style weak prose.\n"
            "Use natural mathematical modeling paper prose. Do not use template labels such as "
            "Objective, Method/Evidence, Quantitative Findings, or Concluding Implication unless "
            "the actual section title requires them."
        )
        self.max_section_retry = 3
        # 仅当 shared_memory 中不存在 fusion_contract / fusion_style_contract 时启用：
        # 兜底结构必须仍符合数学建模论文，而不是通用 research paper。
        self._fallback_sections: List[Dict[str, Any]] = [
            {
                "id": "title",
                "title": "Title",
                "min_chars": 80,
                "goal": "Provide a concise mathematical modeling paper title that reflects the core problem and method.",
                "figure_ids": [],
            },
            {
                "id": "abstract",
                "title": "Abstract",
                "min_chars": 900,
                "goal": "Summarize the modeling problem, assumptions, model strategy, solution process, key results, sensitivity findings, and conclusions.",
                "figure_ids": [],
            },
            {
                "id": "problem_restatement",
                "title": "Problem Restatement",
                "min_chars": 900,
                "goal": "Restate the original problem in precise modeling language and identify the subproblems to be solved.",
                "figure_ids": [],
            },
            {
                "id": "assumptions_notation",
                "title": "Assumptions and Notations",
                "min_chars": 900,
                "goal": "List justified assumptions, define symbols, variables, parameters, units, and model boundaries.",
                "figure_ids": [],
            },
            {
                "id": "problem_analysis",
                "title": "Problem Analysis",
                "min_chars": 1100,
                "goal": "Analyze the problem mechanism, data conditions, constraints, evaluation criteria, and modeling strategy for each subproblem.",
                "figure_ids": [],
            },
            {
                "id": "model_establishment",
                "title": "Model Establishment",
                "min_chars": 2200,
                "goal": "Build the mathematical model with variables, objective functions, constraints, equations, algorithms, and solution workflow.",
                "figure_ids": [],
            },
            {
                "id": "model_solution_results",
                "title": "Model Solution and Results",
                "min_chars": 1900,
                "goal": "Explain the solution method, parameter settings, computation process, numerical results, and interpretation for each subproblem.",
                "figure_ids": [],
            },
            {
                "id": "sensitivity_robustness",
                "title": "Sensitivity Analysis",
                "min_chars": 900,
                "goal": "Test how key parameters, assumptions, and perturbations affect the model outputs and conclusions.",
                "figure_ids": [],
            },
            {
                "id": "model_evaluation",
                "title": "Model Evaluation",
                "min_chars": 800,
                "goal": "Evaluate strengths, weaknesses, applicability, limitations, and possible improvements of the model.",
                "figure_ids": [],
            },
            {
                "id": "conclusion",
                "title": "Conclusion",
                "min_chars": 600,
                "goal": "Summarize final answers to the subproblems, practical implications, and concise recommendations.",
                "figure_ids": [],
            },
        ]

    def _artifact_ready(self, shared_mem: Dict[str, Any], key: str) -> bool:
        return bool(extract_text_content(shared_mem.get(key, "")).strip())

    def _detect_phase(self, shared_mem: Dict[str, Any]) -> str:
        forced = str(shared_mem.get("writer_force_phase", "") or "").strip().lower()
        if forced in {"outline", "draft"}:
            return forced
        outline = shared_mem.get("paper_outline")
        if not isinstance(outline, dict) or not outline.get("sections"):
            return "outline"
        if self._artifact_ready(shared_mem, "review_report") or self._artifact_ready(shared_mem, "generated_code"):
            return "draft"
        return "outline"

    def _target_language(self, state: AgentState) -> str:
        msgs = messages_for_llm(
            state.get("messages", []) or [],
            max_messages=1,
            max_chars_per_message=1000,
        )
        text = extract_text_content(msgs[-1].content) if msgs else ""
        lowered = text.lower()
        if any(x in lowered for x in ["中文", "chinese", "简体", "繁體"]):
            return "Chinese"
        return "English"

    def _extract_fusion_internal_brief(self, fusion_guidance: str) -> str:
        text = (fusion_guidance or "").strip()
        if not text:
            return ""
        parts: List[str] = []
        for tag in _FUSION_INTERNAL_SECTIONS:
            pattern = rf"\[{tag}\](.*?)(?=\[(?:{'|'.join(_FUSION_INTERNAL_SECTIONS)})\]|CHART_PLAN_JSON|$)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                body = match.group(1).strip()[:1200]
                if body:
                    parts.append(f"[{tag}]\n{body}")
        return "\n\n".join(parts)[:5000]

    def _parse_json_block(self, text: str, marker: str) -> Optional[Any]:
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
                    try:
                        return json.loads(tail[start: idx + 1])
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
            match = re.search(r"\d+", value.strip().replace(",", ""))
            if match:
                parsed = int(match.group(0))
                return parsed if parsed > 0 else fallback
        return fallback

    def _normalize_chart_plan(self, raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        plan: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("id", "")).strip()
            if not cid:
                continue
            plan.append(
                {
                    "id": cid,
                    "title": str(item.get("title", cid)).strip(),
                    "chart_type": str(item.get("chart_type", "bar")).strip(),
                    "purpose": str(item.get("purpose", "")).strip(),
                    "section_hint": str(item.get("section_hint", "Model Solution and Results")).strip(),
                    "priority": str(item.get("priority", "medium")).strip().lower(),
                    "filename": str(item.get("filename", f"{cid}.png")).strip(),
                }
            )
        return plan

    def _fusion_contract(self, shared_mem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        contract = shared_mem.get("fusion_contract")
        return contract if isinstance(contract, dict) and contract else None

    def _contract_draft_min_chars(self, shared_mem: Dict[str, Any], fallback: int = 7000) -> int:
        contract = self._fusion_contract(shared_mem)
        if not contract:
            return fallback
        quality = contract.get("quality_gates")
        if isinstance(quality, dict):
            parsed = self._positive_int(quality.get("draft_min_chars"), 0)
            if parsed:
                return parsed
        return fallback

    def _contract_required_section_titles(self, shared_mem: Dict[str, Any]) -> List[str]:
        contract = self._fusion_contract(shared_mem)
        titles: List[str] = []
        if contract:
            writing = contract.get("writing_contract")
            if isinstance(writing, dict):
                raw = writing.get("required_sections")
                if isinstance(raw, list):
                    titles.extend([str(x).strip() for x in raw if str(x).strip()])
            quality = contract.get("quality_gates")
            if not titles and isinstance(quality, dict):
                raw = quality.get("mandatory_sections")
                if isinstance(raw, list):
                    titles.extend([str(x).strip() for x in raw if str(x).strip()])
        if not titles:
            style = shared_mem.get("fusion_style_contract")
            if isinstance(style, dict):
                raw = style.get("required_sections")
                if isinstance(raw, list):
                    titles.extend([str(x).strip() for x in raw if str(x).strip()])
        normalized: List[str] = []
        for title in titles:
            if title.lower() == "title":
                continue
            if title not in normalized:
                normalized.append(title)
        if not any(t.lower() == "title" for t in normalized):
            normalized.insert(0, "Title")
        return normalized

    def _section_id_from_title(self, title: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", title.strip().lower()).strip("_") or "section"

    def _section_weight_key(self, title: str) -> str:
        t = title.lower()
        if "title" == t:
            return "title"
        if "abstract" in t or t == "summary":
            return "abstract"
        if "problem restatement" in t or "restate" in t or "background" in t:
            return "problem_restatement"
        if "assumption" in t or "notation" in t:
            return "assumptions"
        if "data preprocessing" in t or "preprocessing" in t:
            return "problem_analysis"
        if "problem analysis" in t or "pattern analysis" in t or "analysis" in t:
            return "problem_analysis"
        if "model establishment" in t or "model construction" in t or "model building" in t:
            return "model_establishment"
        if "method" in t or "equation" in t or "algorithm" in t:
            return "model_establishment"
        if "solution" in t or "solving" in t or "result" in t or "validation" in t:
            return "model_solution"
        if "sensitivity" in t or "robust" in t:
            return "sensitivity"
        if "evaluation" in t or "strength" in t or "weakness" in t or "limitation" in t:
            return "model_evaluation"
        if "intro" in t:
            return "problem_restatement"
        if "policy recommendation" in t or "recommendation" in t:
            return "conclusion"
        if "conclusion" in t:
            return "conclusion"
        return "problem_analysis"

    def _template_for_section(self, title: str, section_templates: Dict[str, Any]) -> str:
        if not isinstance(section_templates, dict):
            return ""
        title_l = title.strip().lower()
        for key, val in section_templates.items():
            key_l = str(key).strip().lower()
            if not key_l:
                continue
            if key_l in title_l or title_l in key_l:
                if isinstance(val, list):
                    return " -> ".join(str(x).strip() for x in val if str(x).strip())
                if isinstance(val, dict):
                    parts: List[str] = []
                    for sub_key, sub_val in val.items():
                        if isinstance(sub_val, list):
                            rendered = ", ".join(str(x).strip() for x in sub_val if str(x).strip())
                        else:
                            rendered = str(sub_val).strip()
                        if rendered:
                            parts.append(f"{sub_key}: {rendered}")
                    return "; ".join(parts)
                return str(val).strip()
        return ""

    def _default_goal_for_section(
        self,
        title: str,
        section_focus: Dict[str, Any],
        task_requirements: Dict[str, Any],
        section_templates: Optional[Dict[str, Any]] = None,
    ) -> str:
        focus = ""
        if isinstance(section_focus, dict):
            for key, val in section_focus.items():
                if str(key).strip().lower() in title.lower() or title.lower() in str(key).strip().lower():
                    focus = str(val).strip()
                    break
            if not focus:
                focus = str(section_focus.get("global", "")).strip()
        if focus:
            return focus
        objective = ""
        if isinstance(task_requirements, dict):
            objective = str(task_requirements.get("primary_objective", "")).strip()
        templates = {
            "title": "Provide a concrete technical title aligned with the modeling task.",
            "abstract": "Summarize problem, method, quantitative findings, and contributions.",
            "problem_restatement": "Restate the problem, subproblems, objectives, and constraints in modeling language.",
            "assumptions": "State assumptions, notation, variables, parameters, units, and model boundaries.",
            "problem_analysis": "Analyze mechanisms, data conditions, constraints, and the modeling strategy.",
            "model_establishment": "Detail model formulation, objective functions, constraints, equations, and algorithms.",
            "model_solution": "Present solution process, numerical results, validation evidence, and interpretation.",
            "sensitivity": "Analyze sensitivity, robustness, uncertainty, and parameter impacts.",
            "model_evaluation": "Discuss model strengths, weaknesses, applicability, limitations, and improvements.",
            "conclusion": "Conclude answers to subproblems and provide actionable recommendations.",
        }
        base = templates.get(self._section_weight_key(title), "Develop this section with evidence-backed analysis.")
        template = self._template_for_section(title, section_templates or {})
        if template:
            base = f"{base} Follow learned section structure: {template}."
        if objective:
            return f"{base} Primary task objective: {objective}"
        return base

    def _allocate_section_min_chars(self, titles: List[str], total_min_chars: int) -> Dict[str, int]:
        if not titles:
            return {}
        weights: Dict[str, float] = {}
        for title in titles:
            key = self._section_weight_key(title)
            weights[title] = _SECTION_CHAR_WEIGHTS.get(key, 0.08)
        title_min = 0
        content_titles = [t for t in titles if self._section_weight_key(t) != "title"]
        if len(content_titles) != len(titles):
            title_min = max(80, min(200, int(total_min_chars * 0.03)))
        content_target = max(0, total_min_chars - title_min)
        weight_sum = sum(weights[t] for t in content_titles) or 1.0
        allocated: Dict[str, int] = {}
        for title in titles:
            if self._section_weight_key(title) == "title":
                allocated[title] = title_min
                continue
            share = weights[title] / weight_sum
            allocated[title] = max(300, int(content_target * share))
        gap = total_min_chars - sum(allocated.values())
        if gap > 0 and content_titles:
            heaviest = max(content_titles, key=lambda t: weights.get(t, 0.0))
            allocated[heaviest] = allocated.get(heaviest, 0) + gap
        return allocated

    def _chart_plan_from_fusion(self, shared_mem: Dict[str, Any]) -> List[Dict[str, Any]]:
        contract = self._fusion_contract(shared_mem)
        if contract:
            chart_contract = contract.get("chart_contract")
            if isinstance(chart_contract, dict):
                raw = chart_contract.get("chart_plan")
                plan = self._normalize_chart_plan(raw)
                if plan:
                    return plan
        return self._normalize_chart_plan(shared_mem.get("fusion_chart_plan", []))

    def _bind_figures_to_sections(self, sections: List[Dict[str, Any]], chart_plan: List[Dict[str, Any]]) -> None:
        by_hint: Dict[str, List[str]] = {}
        for item in chart_plan:
            hint = str(item.get("section_hint", "")).strip().lower()
            cid = str(item.get("id", "")).strip()
            if hint and cid:
                by_hint.setdefault(hint, []).append(cid)
        for sec in sections:
            if sec.get("figure_ids"):
                continue
            title_l = str(sec.get("title", "")).lower()
            mapped: List[str] = []
            for hint, ids in by_hint.items():
                if hint in title_l or title_l in hint:
                    mapped.extend(ids)
            sec["figure_ids"] = list(dict.fromkeys(mapped))

    def _sections_from_fusion_contract(
        self,
        shared_mem: Dict[str, Any],
        chart_plan: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        titles = self._contract_required_section_titles(shared_mem)
        if not titles:
            return [dict(s) for s in self._fallback_sections]

        contract = self._fusion_contract(shared_mem) or {}
        writing = contract.get("writing_contract") if isinstance(contract.get("writing_contract"), dict) else {}
        section_focus = writing.get("section_focus") if isinstance(writing.get("section_focus"), dict) else {}
        section_templates = writing.get("section_templates") if isinstance(writing.get("section_templates"), dict) else {}
        task_requirements = contract.get("task_requirements") if isinstance(contract.get("task_requirements"), dict) else {}
        min_chars_map = self._allocate_section_min_chars(titles, self._contract_draft_min_chars(shared_mem))

        sections: List[Dict[str, Any]] = []
        for title in titles:
            sections.append(
                {
                    "id": self._section_id_from_title(title),
                    "title": title,
                    "min_chars": int(min_chars_map.get(title, 800)),
                    "goal": self._default_goal_for_section(title, section_focus, task_requirements, section_templates),
                    "figure_ids": [],
                }
            )
        self._bind_figures_to_sections(sections, chart_plan)
        return sections

    def _merge_outline_with_contract(
        self,
        llm_sections: List[Dict[str, Any]],
        contract_sections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """以 Fusion 章节合同为主，LLM 仅可补充 goal/figure_ids，不得删减合同章节或降低 min_chars。"""
        contract_by_title = {str(s["title"]).strip().lower(): s for s in contract_sections}
        llm_by_title = {str(s.get("title", "")).strip().lower(): s for s in llm_sections if s.get("title")}

        merged: List[Dict[str, Any]] = []
        for base in contract_sections:
            title = str(base["title"])
            key = title.lower()
            llm_sec = llm_by_title.get(key, {})
            min_chars = max(
                self._positive_int(base.get("min_chars"), 800),
                self._positive_int(llm_sec.get("min_chars"), 0),
            )
            goal = str(llm_sec.get("goal", "")).strip() or str(base.get("goal", "")).strip()
            fig_ids = llm_sec.get("figure_ids") if isinstance(llm_sec.get("figure_ids"), list) and llm_sec.get("figure_ids") else base.get("figure_ids", [])
            merged.append(
                {
                    "id": str(base.get("id") or self._section_id_from_title(title)),
                    "title": title,
                    "min_chars": min_chars,
                    "goal": goal,
                    "figure_ids": [str(x).strip() for x in (fig_ids or []) if str(x).strip()],
                }
            )
        return merged

    def _normalize_outline(
        self,
        raw: Any,
        chart_plan: List[Dict[str, Any]],
        shared_mem: Dict[str, Any],
    ) -> Dict[str, Any]:
        contract_sections = self._sections_from_fusion_contract(shared_mem, chart_plan)

        llm_sections: List[Dict[str, Any]] = []
        if isinstance(raw, dict) and isinstance(raw.get("sections"), list):
            for sec in raw["sections"]:
                if not isinstance(sec, dict):
                    continue
                title = str(sec.get("title", "")).strip()
                if not title:
                    continue
                fig_ids = sec.get("figure_ids") or []
                if not isinstance(fig_ids, list):
                    fig_ids = []
                llm_sections.append(
                    {
                        "id": str(sec.get("id", self._section_id_from_title(title))).strip(),
                        "title": title,
                        "min_chars": max(200, self._positive_int(sec.get("min_chars"), 800)),
                        "goal": str(sec.get("goal", "")).strip(),
                        "figure_ids": [str(x).strip() for x in fig_ids if str(x).strip()],
                    }
                )

        if self._fusion_contract(shared_mem) or self._contract_required_section_titles(shared_mem):
            sections = self._merge_outline_with_contract(llm_sections, contract_sections)
        elif llm_sections:
            sections = llm_sections
        else:
            sections = contract_sections if contract_sections else [dict(s) for s in self._fallback_sections]

        self._bind_figures_to_sections(sections, chart_plan)
        return {"sections": sections, "outline_source": "fusion_contract" if self._fusion_contract(shared_mem) else "fallback"}

    def _style_contract_text(self, shared_mem: Dict[str, Any], max_len: int = 5000) -> str:
        fusion_contract = shared_mem.get("fusion_contract")
        if isinstance(fusion_contract, dict) and fusion_contract:
            try:
                return json.dumps(fusion_contract, ensure_ascii=False, indent=2)[:max_len]
            except Exception:
                pass
        contract = shared_mem.get("fusion_style_contract")
        if not isinstance(contract, dict):
            return "N/A"
        try:
            return json.dumps(contract, ensure_ascii=False, indent=2)[:max_len]
        except Exception:
            return "N/A"

    def _paper_structure_learning_text(self, shared_mem: Dict[str, Any], max_len: int = 5000) -> str:
        learning = shared_mem.get("fusion_paper_structure_learning")
        if not isinstance(learning, dict) or not learning:
            contract = self._fusion_contract(shared_mem) or {}
            writing = contract.get("writing_contract") if isinstance(contract.get("writing_contract"), dict) else {}
            learning = writing.get("paper_structure_learning") if isinstance(writing.get("paper_structure_learning"), dict) else {}
        if not isinstance(learning, dict) or not learning:
            return "N/A"
        try:
            return json.dumps(learning, ensure_ascii=False, indent=2)[:max_len]
        except Exception:
            return "N/A"

    def _figure_style_learning_text(self, shared_mem: Dict[str, Any], max_len: int = 4000) -> str:
        learning = shared_mem.get("fusion_figure_style_learning")
        if not isinstance(learning, dict) or not learning:
            contract = self._fusion_contract(shared_mem) or {}
            chart_contract = contract.get("chart_contract") if isinstance(contract.get("chart_contract"), dict) else {}
            learning = chart_contract.get("figure_style_learning") if isinstance(chart_contract.get("figure_style_learning"), dict) else {}
        if not isinstance(learning, dict) or not learning:
            return "N/A"
        try:
            return json.dumps(learning, ensure_ascii=False, indent=2)[:max_len]
        except Exception:
            return "N/A"

    def _contract_required_chart_count(self, shared_mem: Dict[str, Any], fallback: int = 6) -> int:
        fusion_contract = shared_mem.get("fusion_contract")
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
        chart_weight_plan = shared_mem.get("fusion_chart_weight_plan")
        if isinstance(chart_weight_plan, dict):
            parsed = self._positive_int(chart_weight_plan.get("required_chart_count"), 0)
            if parsed:
                return parsed
        return fallback

    def _build_outline_context(self, state: AgentState, shared_mem: Dict[str, Any], lang: str) -> str:
        analysis = extract_text_content(shared_mem.get("analysis_report", ""))[:5000]
        model = extract_text_content(shared_mem.get("mathematical_model", ""))[:7000]
        doc = extract_text_content(shared_mem.get("raw_document_content", ""))[:2200]
        fusion_brief = self._extract_fusion_internal_brief(
            extract_text_content(shared_mem.get("fusion_guidance", ""))
        )
        return (
            f"[Target Language]\n{lang}\n\n"
            f"[Problem Summary]\n{doc or 'N/A'}\n\n"
            f"[Analysis]\n{analysis or 'N/A'}\n\n"
            f"[Modeling]\n{model or 'N/A'}\n\n"
            f"[Style Contract]\n{self._style_contract_text(shared_mem)}\n\n"
            f"[Learned Paper Structure and Writing Patterns]\n{self._paper_structure_learning_text(shared_mem)}\n\n"
            f"[Learned Figure Style and Explanation Patterns]\n{self._figure_style_learning_text(shared_mem)}\n\n"
            f"[Internal Fusion Reference - do not copy literally]\n{fusion_brief or 'N/A'}\n"
        ).replace("{", "{{").replace("}", "}}")

    def _build_draft_context(self, shared_mem: Dict[str, Any], section: Dict[str, Any], lang: str) -> str:
        return (
            f"[Target Language]\n{lang}\n\n"
            f"[Analysis]\n{extract_text_content(shared_mem.get('analysis_report', ''))[:1800]}\n\n"
            f"[Modeling]\n{extract_text_content(shared_mem.get('mathematical_model', ''))[:2200]}\n\n"
            f"[Coder Code Summary]\n{extract_text_content(shared_mem.get('generated_code', ''))[:1000]}\n\n"
            f"[Execution Logs]\n{extract_text_content(shared_mem.get('execution_logs', ''))[:800]}\n\n"
            f"[Review]\n{extract_text_content(shared_mem.get('review_report', ''))[:1000]}\n\n"
            f"[Section]\n{section.get('title', '')}\n"
            f"[Section Goal]\n{section.get('goal', '')}\n"
            f"[Planned Figure IDs]\n{', '.join(section.get('figure_ids') or []) or 'none'}\n"
            f"[Style Contract]\n{self._style_contract_text(shared_mem, max_len=1800)}\n"
            f"[Learned Paper Structure and Writing Patterns]\n{self._paper_structure_learning_text(shared_mem, max_len=2200)}\n"
            f"[Learned Figure Style and Explanation Patterns]\n{self._figure_style_learning_text(shared_mem, max_len=1400)}\n"
        ).replace("{", "{{").replace("}", "}}")

    def _section_kind(self, section_title: str) -> str:
        title = (section_title or "").strip().lower()
        if title == "title":
            return "title"
        if "abstract" in title or title == "summary":
            return "abstract"
        return "body"

    def _draft_request_for_section(
        self,
        title: str,
        min_chars: int,
        lang: str,
        ctx: str,
    ) -> str:
        kind = self._section_kind(title)
        if kind == "title":
            return (
                f"[Section Context]\n{ctx}\n\n"
                f"Write only the final paper title in {lang}.\n"
                "Rules:\n"
                "- Return a single concise title line only.\n"
                "- Do not write Objective, Method/Evidence, quantitative findings, or explanatory paragraphs.\n"
                "- Do not use markdown heading markers."
            )
        if kind == "abstract":
            return (
                f"[Section Context]\n{ctx}\n\n"
                f"Write the paper abstract/summary in {lang}.\n"
                f"Minimum length: {min_chars} characters.\n"
                "Rules:\n"
                "- Use the heading `## Abstract` or `## Summary` matching the section title.\n"
                "- Write cohesive abstract prose covering problem, assumptions, model, solution, key results, sensitivity, and conclusions.\n"
                "- If section images are attached, summarize only their visible quantitative role; do not invent unseen chart content.\n"
                "- Do not split it into Objective/Method/Evidence/Quantitative Findings labels."
            )
        return (
            f"[Section Context]\n{ctx}\n\n"
            f"Write section '{title}' in {lang}.\n"
            f"Minimum length: {min_chars} characters.\n"
            "Rules:\n"
            "- Use heading format `## Section Title`.\n"
            "- Write coherent mathematical modeling paper paragraphs, equations, explanations, and evidence as appropriate.\n"
            "- Cover the section goal, model/evidence basis, quantitative interpretation, and implications naturally.\n"
            "- If section images are attached, explain visible trends, axes, legends, contrasts, and visual relationships directly; avoid generic 'Figure shows...' filler.\n"
            "- Ensure any caption-style prose is consistent with the attached image content. The final image blocks will be inserted separately by the system.\n"
            "- Do not use boilerplate labels like Objective, Method/Evidence, Quantitative Findings, or Concluding Implication."
        )

    def _section_quality_ok(self, section: Dict[str, Any], text: str) -> bool:
        title = str(section.get("title", ""))
        min_chars = self._positive_int(section.get("min_chars"), 800)
        body = (text or "").strip()
        kind = self._section_kind(title)
        if kind == "title":
            return 10 <= len(body.replace("#", "").strip()) <= 180 and "\n\n" not in body
        if len(body) < min_chars:
            return False
        lowered = body.lower()
        banned_template_labels = [
            "**objective**",
            "**method/evidence**",
            "**quantitative findings**",
            "**concluding implication**",
        ]
        if any(label in lowered for label in banned_template_labels):
            return False
        if kind == "abstract":
            return all(tok in lowered for tok in ["problem", "model"]) and (
                "result" in lowered or "finding" in lowered
            )
        title_specific = {
            "Model Establishment": ["equation", "constraint"],
            "Model Construction": ["equation", "constraint"],
            "Model Solution": ["result", "parameter"],
            "Results": ["comparison", "metric"],
            "Sensitivity": ["sensitivity", "parameter"],
        }
        for key, tokens in title_specific.items():
            if key.lower() in title.lower():
                return all(tok in lowered for tok in tokens[:2])
        return True

    def _manifest_index(self, manifest_like: Any, chart_plan: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        manifest = manifest_like if isinstance(manifest_like, list) else []
        idx: Dict[str, Dict[str, Any]] = {}
        filename_map = {
            str(c.get("filename", f"{c.get('id', '')}.png")).strip().lower(): str(c.get("id", "")).strip().lower()
            for c in chart_plan
            if isinstance(c, dict) and c.get("id")
        }
        for item in manifest:
            if not isinstance(item, dict) or str(item.get("kind", "")).lower() != "image":
                continue
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            fname = str(item.get("filename", "")).strip().lower()
            stem = fname.rsplit(".", 1)[0] if fname else ""
            cid = filename_map.get(fname) or filename_map.get(stem) or stem
            if cid:
                idx[cid] = item
        return idx

    def _fallback_section_images(self, section_title: str, chart_plan: List[Dict[str, Any]]) -> List[str]:
        title_l = section_title.lower()
        out: List[str] = []
        for c in chart_plan:
            if not isinstance(c, dict):
                continue
            hint = str(c.get("section_hint", "")).strip().lower()
            cid = str(c.get("id", "")).strip().lower()
            if cid and hint and (hint in title_l or title_l in hint):
                out.append(cid)
        return list(dict.fromkeys(out))

    def _figure_blocks_for_section(
        self,
        section: Dict[str, Any],
        manifest_index: Dict[str, Dict[str, Any]],
        chart_plan: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        planned = [str(x).strip().lower() for x in (section.get("figure_ids") or []) if str(x).strip()]
        if not planned:
            planned = self._fallback_section_images(str(section.get("title", "")), chart_plan)
        if not planned:
            return blocks
        title_by_id = {str(c.get("id", "")).strip().lower(): str(c.get("title", "")).strip() for c in chart_plan if isinstance(c, dict)}
        seen: set[str] = set()
        figure_no = 1
        for cid in planned:
            item = manifest_index.get(cid)
            if not item:
                blocks.append(
                    {
                        "type": "text",
                        "text": f"[Figure Missing] Planned chart '{cid}' was not generated by Coder.",
                    }
                )
                continue
            url = str(item.get("url", "")).strip()
            if not url or url in seen:
                continue
            seen.add(url)
            caption = title_by_id.get(cid) or str(item.get("filename", cid))
            blocks.append({"type": "image", "url": url, "alt": caption})
            blocks.append({"type": "text", "text": f"Figure {figure_no}. {caption}"})
            figure_no += 1
        return blocks

    def _normalize_section_text(self, section_title: str, text: str) -> str:
        body = (text or "").strip()
        if not body:
            return f"## {section_title}\n\n(Section generation failed. Please regenerate.)"
        if self._section_kind(section_title) == "title":
            title_line = re.sub(r"^#+\s*", "", body.splitlines()[0]).strip()
            return f"# {title_line}"
        lower = body.lower()
        if lower.startswith(f"## {section_title.lower()}") or lower.startswith(f"# {section_title.lower()}"):
            return body
        return f"## {section_title}\n\n{body}"

    def _ensure_chart_plan_count(
        self,
        chart_plan: List[Dict[str, Any]],
        shared_mem: Dict[str, Any],
        required_chart_count: int,
    ) -> List[Dict[str, Any]]:
        if len(chart_plan) >= required_chart_count:
            return chart_plan
        base = self._chart_plan_from_fusion(shared_mem)
        if base:
            merged = {str(c.get("id", "")).strip(): c for c in chart_plan if isinstance(c, dict) and c.get("id")}
            for item in base:
                cid = str(item.get("id", "")).strip()
                if cid and cid not in merged:
                    merged[cid] = item
            chart_plan = list(merged.values())
        if len(chart_plan) >= required_chart_count:
            return chart_plan
        idx = 1
        while len(chart_plan) < required_chart_count:
            chart_plan.append(
                {
                    "id": f"fusion_extra_chart_{idx}",
                    "title": f"Supplementary Chart {idx}",
                    "chart_type": "line",
                    "purpose": "Additional evidence required by fusion contract.",
                    "section_hint": "Model Solution and Results",
                    "priority": "medium",
                    "filename": f"fusion_extra_chart_{idx}.png",
                }
            )
            idx += 1
        return chart_plan

    async def _run_outline_phase(self, state: AgentState, shared_mem: Dict[str, Any], llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
        lang = self._target_language(state)
        required_chart_count = self._contract_required_chart_count(shared_mem, fallback=6)
        draft_min_chars = self._contract_draft_min_chars(shared_mem)
        contract_sections = self._sections_from_fusion_contract(shared_mem, self._chart_plan_from_fusion(shared_mem))
        section_titles = [str(s.get("title", "")) for s in contract_sections]
        await broadcast_progress("Writer", "正在按 Fusion 合同重构论文蓝图（章节/篇幅/图表）...", 15)
        context = self._build_outline_context(state, shared_mem, lang)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.outline_system_prompt),
                ("system", f"[Upstream Context]\n{context}"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        outline_request = (
            "Output two JSON payloads:\n"
            "1) PAPER_OUTLINE_JSON:\n"
            "{\"sections\":[{\"id\":\"...\",\"title\":\"...\",\"min_chars\":1200,\"goal\":\"...\",\"figure_ids\":[\"chart_id\"]}]}\n"
            "2) CHART_PLAN_JSON: [{\"id\":\"...\",\"title\":\"...\",\"chart_type\":\"...\",\"purpose\":\"...\",\"section_hint\":\"...\",\"priority\":\"high|medium|low\",\"filename\":\"...\"}]\n"
            "Hard rules:\n"
            f"- Section titles MUST exactly match this Fusion list (same order): {section_titles}\n"
            f"- Total draft minimum characters target: {draft_min_chars} (allocate per section, do not undercut contract floors)\n"
            f"- Chart plan minimum count: {required_chart_count}; prefer Fusion chart_plan IDs when provided in Style Contract\n"
            f"- Language for labels/goals: {lang}\n"
            "- Do NOT invent a different paper structure."
        )
        messages = messages_for_llm(
            state.get("messages", []) or [],
            max_messages=4,
            max_chars_per_message=1200,
        ) + [HumanMessage(content=outline_request)]
        response = await (prompt | llm).ainvoke({"messages": messages})
        raw = extract_text_content(response)

        outline_raw = self._parse_json_block(raw, "PAPER_OUTLINE_JSON")
        chart_raw = self._parse_json_block(raw, "CHART_PLAN_JSON")
        if chart_raw is None:
            chart_raw = self._chart_plan_from_fusion(shared_mem)
        chart_plan = self._ensure_chart_plan_count(self._normalize_chart_plan(chart_raw), shared_mem, required_chart_count)

        paper_outline = self._normalize_outline(outline_raw, chart_plan, shared_mem)
        summary = ["## Paper Outline Completed", "", "### Sections"]
        for sec in paper_outline["sections"]:
            summary.append(
                f"- **{sec['title']}** (min_chars={sec['min_chars']}, figures={', '.join(sec.get('figure_ids') or []) or 'none'})"
            )
        summary.append("")
        summary.append("### Chart Plan")
        for c in chart_plan:
            summary.append(f"- `{c['id']}` -> {c['title']} ({c['chart_type']}, {c['section_hint']})")
        summary_text = "\n".join(summary)

        new_memory = shared_mem.copy()
        new_memory["paper_outline"] = json_safe(paper_outline)
        new_memory["writer_chart_plan"] = json_safe(chart_plan)
        new_memory["fusion_chart_plan"] = json_safe(chart_plan)
        new_memory["writer_phase"] = "outline_done"
        new_memory["writer_outline_source"] = paper_outline.get("outline_source", "fusion_contract")

        await broadcast_progress("Writer", "论文蓝图完成（已对齐 Fusion 合同），进入图表产出阶段。", 100)
        return {
            "messages": [{"role": "ai", "content": ensure_blocks({"type": "markdown", "text": summary_text})}],
            "shared_memory": new_memory,
            "status": "APPROVED",
            "draft": ensure_blocks({"type": "markdown", "text": summary_text}),
            "stage": make_stage("outline_completed", "Paper Outline & Chart Plan Completed"),
        }

    async def _run_draft_phase(self, state: AgentState, shared_mem: Dict[str, Any], llm: ChatGoogleGenerativeAI) -> Dict[str, Any]:
        lang = self._target_language(state)
        outline = shared_mem.get("paper_outline") or {}
        sections = outline.get("sections") if isinstance(outline, dict) else None
        chart_plan = self._normalize_chart_plan(
            shared_mem.get("writer_chart_plan") or self._chart_plan_from_fusion(shared_mem)
        )
        if not isinstance(sections, list) or not sections:
            sections = self._sections_from_fusion_contract(shared_mem, chart_plan)
        elif self._fusion_contract(shared_mem):
            sections = self._merge_outline_with_contract(
                sections,
                self._sections_from_fusion_contract(shared_mem, chart_plan),
            )
        manifest_index = self._manifest_index(shared_mem.get("artifacts_manifest", []), chart_plan)

        await broadcast_progress("Writer", "正在生成完整论文正文（含标题、摘要、章节与图表锚点）...", 10)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.draft_system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        chain = prompt | llm
        base_messages = messages_for_llm(
            state.get("messages", []) or [],
            max_messages=2,
            max_chars_per_message=1000,
        )
        blocks: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {}
        total = len(sections)

        for idx, section in enumerate(sections, start=1):
            title = str(section.get("title", f"Section {idx}"))
            min_chars = self._positive_int(section.get("min_chars"), 1000)
            progress = min(94, 10 + int((idx - 1) * (82 / max(total, 1))))
            await broadcast_progress("Writer", f"Writing {idx}/{total}: {title}", progress)

            best_text = ""
            best_len = 0
            ctx = self._build_draft_context(shared_mem, section, lang)
            planned_figures = [
                str(x).strip().lower()
                for x in (section.get("figure_ids") or [])
                if str(x).strip()
            ]
            if not planned_figures:
                planned_figures = self._fallback_section_images(title, chart_plan)
            vision_parts, vision_meta = collect_multimodal_images(
                shared_mem,
                sources=("artifacts",),
                chart_ids=planned_figures,
                max_images=6,
            )
            if vision_meta:
                ctx += (
                    "\n\n[Attached Section Figure Images]\n"
                    f"{image_meta_summary(vision_meta)}\n"
                    "Use the attached images to write concrete figure interpretation for this section. "
                    "Do not describe chart elements that are not visible."
                )
            for _ in range(self.max_section_retry):
                req = self._draft_request_for_section(title, min_chars, lang, ctx)
                section_message = HumanMessage(content=req)
                if vision_parts:
                    section_message = HumanMessage(
                        content=[{"type": "text", "text": req}, *vision_parts]
                    )
                resp = await chain.ainvoke({"messages": base_messages + [section_message]})
                text = self._normalize_section_text(title, extract_text_content(resp))
                if len(text) > best_len:
                    best_text = text
                    best_len = len(text)
                if self._section_quality_ok(section, best_text):
                    break

            blocks.append({"type": "markdown", "text": best_text})
            fig_blocks = self._figure_blocks_for_section(section, manifest_index, chart_plan)
            blocks.extend(fig_blocks)
            stats[title] = {
                "min_chars": min_chars,
                "actual_chars": best_len,
                "figure_ids": section.get("figure_ids") or [],
                "figures_inserted": len([b for b in fig_blocks if b.get("type") == "image"]),
            }
            base_messages.append(HumanMessage(content=f"Section done: {title}"))
            base_messages.append(HumanMessage(content=best_text[:500]))
            if len(base_messages) > 8:
                base_messages = base_messages[:2] + base_messages[-6:]

        paper_text = "\n\n".join(extract_text_content(b) for b in blocks).strip()
        new_memory = shared_mem.copy()
        new_memory["paper_draft"] = blocks
        new_memory["writer_phase"] = "draft_done"
        new_memory["writer_stats"] = {
            "mode": "fusion_contract_driven_writer",
            "outline_source": shared_mem.get("writer_outline_source", "unknown"),
            "draft_min_chars_target": self._contract_draft_min_chars(shared_mem),
            "final_chars": len(paper_text),
            "section_stats": stats,
        }

        await broadcast_progress("Writer", "论文正文生成完成，已执行结构与图表绑定。", 100)
        return {
            "messages": [{"role": "ai", "content": ensure_blocks({"type": "markdown", "text": paper_text})}],
            "shared_memory": new_memory,
            "status": "APPROVED",
            "draft": blocks,
            "stage": make_stage("paper_finished", "Paper Finishing"),
        }

    async def __call__(self, state: AgentState):
        shared_mem = state.get("shared_memory", {}) or {}
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-3.1-flash-lite"
        if not api_key:
            return {
                "status": "REJECTED",
                "human_feedback": "缺少任务 API Key，Writer 节点无法执行。",
                "stage": make_stage("writer_failed_missing_api_key", "Writer Failed: Missing Task API Key"),
            }

        llm = ChatGoogleGenerativeAI(model=model_id, temperature=0.35, google_api_key=api_key)
        phase = self._detect_phase(shared_mem)
        if phase == "outline":
            if not self._artifact_ready(shared_mem, "mathematical_model"):
                return {
                    "status": "REJECTED",
                    "human_feedback": "缺少 Modeling 产物，无法规划论文结构。",
                    "stage": make_stage("writer_failed_missing_model", "Writer Outline: Missing Model"),
                }
            return await self._run_outline_phase(state, shared_mem, llm)
        return await self._run_draft_phase(state, shared_mem, llm)


writer_node = WriterNode()
