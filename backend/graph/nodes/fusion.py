import json
import logging
import re
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage
from google import genai
from google.genai import types
from backend.services.rag_engine import rag_engine
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress
from backend.services.serialization import extract_text_content, make_stage, json_safe

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
            "3.1 **强约束输出格式（必须严格遵守）**：必须输出以下 5 个一级小节，且每节至少 3 条要点：\n"
            "   - [STYLE_GUIDE]：优秀论文的语言风格与论证模式\n"
            "   - [MODELING_PRIORS]：可复用的建模结构、变量设定、目标函数/约束套路\n"
            "   - [RESULTS_NARRATIVE]：结果章节应如何解释图表与数值，如何做对比与验证\n"
            "   - [RISK_AND_LIMITATIONS]：稳健性、敏感性、边界条件与局限性写法\n"
            "   - [WRITING_BLUEPRINT]：论文章节级写作蓝图（每章应覆盖的关键点）\n"
            "   严禁输出空节、严禁只给泛泛建议。\n"
            "3.2 **图表规划输出（必须）**：在文末追加 `CHART_PLAN_JSON`，输出 JSON 数组。每项字段必须包含：\n"
            "   - id: 图表唯一标识（英文下划线）\n"
            "   - title: 图表标题\n"
            "   - chart_type: 例如 flowchart/network/line/bar/scatter/heatmap/radar/stacked_area/table\n"
            "   - purpose: 该图用于证明什么结论\n"
            "   - required_inputs: 依赖的上游信息键（如 analysis_report/mathematical_model/execution_logs）\n"
            "   - section_hint: 建议放在论文哪个章节（Methodology/Results 等）\n"
            "   - priority: high/medium/low\n"
            "   - fallback_if_missing: 数据不足时的替代图\n"
            "   只要题目涉及方法流程与决策链，必须包含至少 1 个流程图或思路网络图计划项。\n"
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
                "section_hint": "Methodology",
                "priority": "high",
                "fallback_if_missing": "改为简化流程网络图",
            },
            {
                "id": "indicator_relation_network",
                "title": "Indicator Relation Network",
                "chart_type": "network",
                "purpose": "展示指标或变量之间的依赖关系与影响路径",
                "required_inputs": ["analysis_report", "mathematical_model"],
                "section_hint": "Methodology",
                "priority": "medium",
                "fallback_if_missing": "改为变量关系矩阵热力图",
            },
            {
                "id": "result_comparison_chart",
                "title": "Result Comparison",
                "chart_type": "bar",
                "purpose": "对比核心方案或核心指标的结果差异",
                "required_inputs": ["execution_logs", "generated_code"],
                "section_hint": "Results",
                "priority": "high",
                "fallback_if_missing": "改为表格汇总",
            },
            {
                "id": "sensitivity_curve",
                "title": "Sensitivity Analysis Curve",
                "chart_type": "line",
                "purpose": "验证关键参数变化对目标结果的影响",
                "required_inputs": ["mathematical_model", "execution_logs"],
                "section_hint": "Sensitivity and Robustness Analysis",
                "priority": "medium",
                "fallback_if_missing": "改为区间柱状图",
            },
        ]

    def _extract_chart_plan(self, fused_context: str) -> List[Dict[str, Any]]:
        text = fused_context or ""
        # 优先匹配 CHART_PLAN_JSON 标记后的 JSON 数组
        marker_match = re.search(r"CHART_PLAN_JSON\s*[:：]?\s*(\[[\s\S]*?\])", text, re.IGNORECASE)
        candidates = []
        if marker_match:
            candidates.append(marker_match.group(1))
        # 其次匹配第一个 JSON 数组代码块
        block_match = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
        if block_match:
            candidates.append(block_match.group(1))
        for raw in candidates:
            try:
                parsed = json.loads(raw)
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

    async def __call__(self, state: AgentState):
        logger.info("Fusion node entered")
        shared_mem = state.get("shared_memory", {})
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-2.5-flash"
        
        messages = state.get("messages", [])
        if not messages:
            return {"context": "No query provided.", "stage": make_stage("fusion_skipped_no_query", "Fusion skipped: no query")}
            
        # 提取最后一条消息作为意图载体
        last_message = extract_text_content(messages[-1])
        
        await broadcast_progress("Retrieve(Fusion)", "正在启动双轨检索引擎 (Local RAG + Web Search)...", 20)
        
        # 1. 发兵本地 ChromaDB
        search_results = rag_engine.search(last_message, n_results=3, api_key=api_key)
        
        rag_context_parts = []
        for i, res in enumerate(search_results):
            rag_context_parts.append(f"--- 内部高价值学术片段 (RAG) {i+1} ---\n{res['content']}")
        rag_context = "\n".join(rag_context_parts) if rag_context_parts else "未在本地发现高度匹配的学术先例。"

        # 2. 调动 Gemini 原生搜索并执行 LLM “动态加权融合”
        if not api_key:
            return {
                "context": "[系统错误] 缺少任务 API Key，Fusion 节点无法执行联网融合检索。",
                "stage": make_stage("fusion_failed_missing_api_key", "Fusion Failed: Missing Task API Key"),
            }

        fusion_query = (
            f"【用户原始探索意图】：\n{last_message}\n\n"
            f"【系统为你准备的本地 RAG 学术参考（供加权判断使用）】：\n{rag_context}\n\n"
            f"请立刻使用外置 Google Search 工具打通公网客观数据，并根据【动态加权融合法则】输出一份终版综合简报。"
        )

        try:
            await broadcast_progress("Retrieve(Fusion)", f"检索完成，已触发 {model_id} 进行跨域知识的加权与去伪存真审查...", 60)
            
            # 使用原生 GenAI 客户端以启用 tools Grounding 特性
            client = genai.Client(api_key=api_key)
            logger.info("Fusion generate_content start model_id=%s (grounding+search may block long)", model_id)
            response = client.models.generate_content(
                model=model_id,
                contents=fusion_query,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    temperature=0.2, # 融合任务需降低幻觉
                    tools=[{"google_search": {}}], # 原生开启谷歌搜索接地
                )
            )
            fused_context = response.text
            logger.info("Fusion generate_content done text_len=%s", len(fused_context or ""))
            
        except Exception as e:
            # 安全降级策略：如果没开启搜索权限或网络阻断，回退到原始 RAG
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
        new_memory["fusion_guidance"] = fused_context
        new_memory["fusion_rag_context"] = rag_context
        new_memory["fusion_raw_content"] = json_safe(fused_context)
        new_memory["fusion_chart_plan"] = json_safe(chart_plan)
        
        return {
            "context": fused_context,
            "shared_memory": new_memory,
            "stage": make_stage("fusion_completed", "Web Grounding & Dynamic Fusion Completed"),
        }

fusion_node = FusionNode()
