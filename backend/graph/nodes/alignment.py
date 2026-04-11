import os
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress

class SymbolEntry(BaseModel):
    symbol: str = Field(description="变量名或数学符号 (如: x_i, item_count, T_max)")
    meaning: str = Field(description="该变量的实际物理或业务含义")
    unit: Optional[str] = Field(description="单位 (若有，如 kg, m/s, USD)")

class AlignmentRecord(BaseModel):
    symbols: List[SymbolEntry] = Field(description="本次发言中定义或提取的所有关键数学符号及代码变量名")
    assumptions: List[str] = Field(description="本次发言中提及的所有关键『简化假设』或『不可逾越的物理限制』")
    objectives: List[str] = Field(description="本次发言中定调的『核心优化目标』或『核心待求解方程约束』")

class MemoryAlignmentNode:
    """
    全链路记忆对齐与防漂移护盾 (Memory Alignment Engine)
    中间件节点：
    1. 在 Analysis/Modeling 节点完成后被挂载执行。
    2. 使用 LLM 从它们的输出报告中精准抽提符号表 (Symbol Table)、核心假设 (Assumptions) 和目标 (Objectives)。
    3. 将增量数据与共享内存中的历史对齐本 (alignment_record) 合并持久化。
    """
    def __init__(self):
        self.system_prompt = (
            "你是一个 MCM/ICM 建模竞赛的『状态对齐护盾提取器』(Memory Registry Extracter)。\n"
            "你的任务是从上游节点（如模型推导、程序代码或审计报告）刚刚输出的文本中，精准剥离出以下三大元信息：\n"
            "1. **数学变量与符号 (Symbols)**：所有新定义的带角标的数学变量、核心常数，以及刚声明的代码变量名。\n"
            "2. **简化假设 (Assumptions)**：专家或法官做出的任何为了模型可计算性而妥协的现实假设，或刚刚放宽的假设限制。\n"
            "3. **核心目标 (Objectives)**：模型致力于求解的核心优化指标或方程式。\n\n"
            "只需忠实于前文提取，如果前文没有任何新的变量、假设或目标产生，请全部存入空列表，绝不自行编造！"
        )

    async def __call__(self, state: AgentState):
        shared_mem = state.get("shared_memory", {})
        
        # 解除硬编码：直接提取对话历史中最近一条 AI 产出的文本
        source_text = ""
        current_node_tag = "Upstream Node"
        
        messages = state.get("messages", [])
        if messages:
            for msg in reversed(messages):
                if msg.type == "ai" and msg.content:
                    source_text = str(msg.content)
                    current_node_tag = "Latest AI Output"
                    break
            
        if not source_text:
            return {"current_stage": "Alignment Pipeline Bypassed (No Source)"}
            
        await broadcast_progress("Alignment", f"[{current_node_tag}] 溯源结束，启动全局『记忆防漂移』对齐拦截扫描表单...", 20)
        
        api_key = shared_mem.get("api_key") or os.environ.get("GOOGLE_API_KEY")
        model_id = shared_mem.get("model_id") or "gemini-2.5-flash"

        # 使用低温度提取模型，使用结构化输出
        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0, 
            google_api_key=api_key
        ).with_structured_output(AlignmentRecord)
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"请提取下述刚刚生成的文本，建立全局防漂移对齐档案：\n{source_text[:4000]}")
        ]
        
        await broadcast_progress("Alignment", f"利用 LLM {model_id} 进行符号表、预设假设的结构化析出化约...", 60)
        
        try:
            extraction = await llm.ainvoke(messages)
            
            # --- 历史账本 Merge 策略 ---
            existing_record = shared_mem.get("alignment_record", {"symbols": [], "assumptions": [], "objectives": []})
            
            # 由于可能重复经历 Analysis / Modeling 回滚修正循环，我们将当前抽取结果**强制超覆(Override)或智能拼接**
            # 这里为简便与防旧历史污染，当由 Modeling 触发时，允许部分保留 Analysis 的，但若发现同名 Symbol，理应替换。
            # 为了实现轻量化，采取直接累加密集去重策略（依赖字典化）
            
            sym_dict = {item.get('symbol'): item for item in existing_record.get('symbols', [])}
            # 更新/增加符号
            for s in extraction.symbols:
                sym_dict[s.symbol] = {"symbol": s.symbol, "meaning": s.meaning, "unit": s.unit}
                
            assumptions_set = set(existing_record.get('assumptions', []))
            assumptions_set.update(extraction.assumptions)
            
            objectives_set = set(existing_record.get('objectives', []))
            objectives_set.update(extraction.objectives)
            
            new_record = {
                "symbols": list(sym_dict.values()),
                "assumptions": list(assumptions_set),
                "objectives": list(objectives_set)
            }
            
            new_memory = shared_mem.copy()
            new_memory["alignment_record"] = new_record
            
            await broadcast_progress("Alignment", f"提取完毕！已向共享字典中硬装载 {len(new_record['symbols'])} 个防漂移符号护盾。", 100)
            
            return {
                "shared_memory": new_memory,
                "current_stage": f"Alignment Middleware Updated (via {current_node_tag})"
            }
            
        except Exception as e:
            print(f"[Alignment Error] 结构化析出时遭遇失败, {e}")
            await broadcast_progress("Alignment", f"对齐日志抽取失败，跳过本次存档...", 100)
            return {"current_stage": "Alignment Failed"}

alignment_node = MemoryAlignmentNode()
