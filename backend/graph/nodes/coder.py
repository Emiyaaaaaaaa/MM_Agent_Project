import os
import re
import json
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress
from backend.services.sandbox import execute_python_code

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
            "3. **静态保存**：必须将图表保存至 `backend/static/plots/output.png`。\n"
            "4. **技术栈**：优先使用 numpy, pandas, scipy, matplotlib, seaborn, pulp 等。\n"
            "5. **输出格式**：将唯一要执行的代码块用 ```python 和 ``` 包裹起来。\n\n"
            "【特别指令】\n"
            "- **代码学术性**：参考 RAG 背景中优秀论文展示的数据处理逻辑与算法实现深度。代码注释应当体现建模逻辑，而不仅仅是代码功能。\n"
            "- **版权保护约束**：严禁在生成内容或注释中提及参考资料的具体队号、年份或获奖等级。保护数据隐私。"
        )

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
        doc_content = shared_mem.get("raw_document_content", "")
        analysis = shared_mem.get("analysis_report", "")
        feedback = state.get("human_feedback", "")
        
        await broadcast_progress("Coder", "正在准备仿真环境与算法指令...", 10)
        
        # 1.5. 动态模型实例化 (从任务配置加载)
        api_key = shared_mem.get("api_key")
        model_id = shared_mem.get("model_id") or "gemini-2.5-flash"
        
        if not api_key:
            print("[Warning] No API key found in shared_memory. Falling back to environment variable.")
            api_key = os.environ.get("GOOGLE_API_KEY")

        llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0.2,
            google_api_key=api_key
        )
        
        # 构造执行背景 (如果存在 RAG context, 注入到 prompt)
        rag_context = state.get("context", "")
        context = f"【赛题】\n{doc_content[:1000]}...\n\n【模型思路】\n{analysis[:1000]}..."
        if rag_context:
            context += f"\n\n【RAG 参考代码风格资料】\n{rag_context[:2000]}"
        
        prompt_parts = [
            ("system", self.system_prompt),
            ("system", f"【当前执行上下文】\n{context}")
        ]
        if feedback:
            prompt_parts.append(("system", f"【用户最新反馈建议】\n{feedback}"))
            
        alignment_record = shared_mem.get("alignment_record", {})
        if alignment_record:
            align_prompt = (
                f"【全局防漂移强制对齐约束 (CRITICAL)】\n"
                f"1. 强制使用以下全局符号体系，严禁在代码中定义与数学推导意义冲突的变量名：\n{json.dumps(alignment_record.get('symbols', []), ensure_ascii=False, indent=2)}\n"
                f"2. 严禁违背以下全局预设简化假设：\n{json.dumps(alignment_record.get('assumptions', []), ensure_ascii=False, indent=2)}\n"
                f"3. 核心计算算法与迭代循环必须紧密朝向以下核心优化目标：\n{json.dumps(alignment_record.get('objectives', []), ensure_ascii=False, indent=2)}"
            )
            prompt_parts.append(("system", align_prompt))
            
        prompt_parts.append(MessagesPlaceholder(variable_name="messages"))
        
        MAX_RETRIES = 3
        current_attempt = 0
        success = False
        final_code = ""
        final_response = None
        execution_logs = {}
        
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
            final_response = response
            
            generated_text = response.content
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
                await broadcast_progress("Coder", "沙箱执行成功，图表及计算结果已生成！", progress_val + 15)
            else:
                stderr_preview = exec_result["stderr"][:1000]
                await broadcast_progress("Coder", "沙箱执行失败！正在提取 Traceback 自我修复...", progress_val + 15)
                
                # 构造自我纠错消息
                error_feedback = (
                    f"代码执行失败，错误信息（Traceback）如下：\n"
                    f"```text\n{stderr_preview}\n```\n"
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
        
        await broadcast_progress("Coder", "执行日志已注入共享内存，进入审计环节。", 100)
        
        return {
            "messages": [final_response],
            "shared_memory": new_memory,
            "status": "PENDING",
            "draft": final_response.content if final_response else final_code,
            "current_stage": "Code Implementation & Sandbox Execution Completed"
        }

coder_node = CoderNode()
