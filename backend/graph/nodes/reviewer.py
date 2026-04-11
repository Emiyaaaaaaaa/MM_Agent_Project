import os
import json
import base64
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress

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
            "3. 运行输出日志与文本结论相违背，或图不达意（坐标轴、标题严重缺失）。\n"
            "4. 记忆漂移(Drift)：检查代码与公式的符号是否割裂（如公式用X，代码用Y），或代码擅自添加了《全局预设假设》中不允许的考虑因素。\n"
            "不要进行任何客套和赞美，直接列出所有痛点和致命雷区，生成一份火力全开的《攻击报告》。"
        )
        
        self.defender_prompt = (
            "你是一个顶尖的 MCM/ICM 建模竞赛论文原作者 (The Defender)。\n"
            "你刚刚收到了一份极其刺耳的《攻击报告》，质疑了你的模型体系、代码推导或图表展示。\n"
            "你的任务是：\n"
            "1. 对于合理的学术简化或为了模型收敛而做出的妥协，给出强有力的学术术语进行理论辩护。\n"
            "2. 澄清对方可能因为未详读原始代码而产生的误解。\n"
            "3. 如果对方指出了真正无法辩驳的硬伤（如明显且致命的代码错误、自相矛盾的数据点），果断承认失误。\n"
            "生成一份理据充分的、守护心血结晶的《答辩报告》。"
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
        )

    def _get_image_data(self, file_path):
        """多模态图像 Base64 提取器"""
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return None

    async def __call__(self, state: AgentState):
        """执行流式多模态三元辩论审查"""
        shared_mem = state.get("shared_memory", {})
        analysis = shared_mem.get("analysis_report", "")
        model_doc = shared_mem.get("mathematical_model", "")
        code = shared_mem.get("generated_code", "")
        execution_logs = shared_mem.get("execution_logs", {})
        
        await broadcast_progress("Review", "[1/4] 资源锁定：正在抽取并组装全链路原始文稿素材视界...", 10)
        
        # 将生成的 Plot 纳入审查眼帘
        base_dir = os.path.abspath(os.getcwd())
        plot_path = os.path.join(base_dir, "backend", "static", "plots", "output.png")
        
        images = []
        plot_b64 = self._get_image_data(plot_path)
        if plot_b64:
            images.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{plot_b64}"}})
            await broadcast_progress("Review", "捕获物理图纸 output.png，激活多模态联合阵列...", 15)
            
        base_material_text = (
            f"以下是全部的【原始材料】产出摘要：\n\n"
            f"【1. 审题分析】\n{analysis[:1000]}...\n\n"
            f"【2. 核心大模型】\n{model_doc[:1000]}...\n\n"
            f"【3. 控制代码基】\n{code[:1000]}...\n\n"
        )
        if execution_logs:
            stdout_data = execution_logs.get('stdout', '')
            base_material_text += f"【4. 代码标准运行时反射区】\n{stdout_data[:1500]}\n\n"
        
        rag_context = state.get("context", "")
        if rag_context:
            base_material_text = f"【系统锚定 RAG 知识框架下限（不可违章）】\n{rag_context[:1000]}\n\n" + base_material_text
            
        alignment_record = shared_mem.get("alignment_record", {})
        if alignment_record:
            base_material_text = (
                f"【全局防漂移强制对齐约束档案 (ALIGNMENT RECORD)】\n"
                f"{json.dumps(alignment_record, ensure_ascii=False, indent=2)}\n\n"
            ) + base_material_text
            
        base_content = [{"type": "text", "text": base_material_text}] + images
        
        # 建立网络连线心跳
        api_key = shared_mem.get("api_key") or os.environ.get("GOOGLE_API_KEY")
        model_id = shared_mem.get("model_id") or "gemini-2.5-flash"

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
        critique_report = critique_response.content
        
        # =======================
        # Round 2: Defender (蓝方防守)
        # =======================
        await broadcast_progress("Review", "[3/4] 第二轮辩论：蓝方原作者 (The Defender) 遭到质询，正在提取文献撰写技术答辩方案...", 60)
        defender_text = (
            f"以下是攻击者向你抛出的《毁击报告》，请依据你的【原始材料】立意进行强硬答辩：\n\n"
            f"【你的核心底座原始材料】:\n{base_material_text[:2000]}\n\n"
            f"--- 对方论点切片 ---\n{critique_report}"
        )
        defender_msg = [
            SystemMessage(content=self.defender_prompt),
            HumanMessage(content=[{"type": "text", "text": defender_text}] + images)
        ]
        defender_response = await debater_llm.ainvoke(defender_msg)
        defense_report = defender_response.content
        
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
            f"【第一附件: 前置原始材料】:\n{base_material_text[:1500]}\n\n"
            f"================\n"
            f"【原告提交: 攻击者的火力全开报告】:\n{critique_report}\n\n"
            f"================\n"
            f"【被告提交: 原作者的逐条答辩报告】:\n{defense_report}\n\n"
            f"请正式对本案全息建模结论下达《官方综合审查意见》。"
        )
        judge_msg = [
            SystemMessage(content=self.judge_prompt),
            HumanMessage(content=[{"type": "text", "text": judge_text}] + images)
        ]
        judge_response = await judge_llm.ainvoke(judge_msg)
        final_review = judge_response.content
        
        await broadcast_progress("Review", "辩论式逻辑审查体系运作顺利流转毕，已封印综合意见卷底。", 100)
        
        new_memory = shared_mem.copy()
        new_memory["review_report"] = final_review
        # 溯源日志防丢失
        new_memory["review_debate_logs"] = f"【The Critique Report】\n{critique_report}\n\n【The Defender Report】\n{defense_report}"
        
        return {
            "messages": [judge_response], 
            "shared_memory": new_memory,
            "status": "PENDING", # 此处中断交与业务人类复裁
            "draft": final_review,
            "current_stage": "Debate-Style Review Completed"
        }

reviewer_node = ReviewerNode()
