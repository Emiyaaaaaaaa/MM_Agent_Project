import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.builder import mcm_graph

load_dotenv()

def test_modeler_flow():
    print(">>> 正在启动『数学建模』节点集成测试...")
    
    config = {"configurable": {"thread_id": "modeler_test_thread"}}
    
    # 模拟审题已完成的状态
    initial_input = {
        "messages": [HumanMessage(content="根据审题报告，请给出具体的数学模型和公式推导。")],
        "shared_memory": {
            "analysis_report": (
                "1. 核心问题：预测未来 20 年人口趋势并考虑资源瓶颈。\n"
                "2. 假设：环境承载力 K 为常数，人口增长率 r 随资源减少而递减。\n"
                "3. 变量：P(t) 为 t 时刻人口，K 为承载力，r 为内在增长率。"
            )
        }
    }
    
    print("\n--- [阶段 1：自动路由与公式推导] ---")
    for event in mcm_graph.stream(initial_input, config, stream_mode="values"):
        if event.get("draft"):
            model_draft = event.get("draft")
            print(f"✅ 已生成数学模型 (长度: {len(model_draft)}):")
            print("-" * 30)
            
            # 检查关键数学元素
            if "$" in model_draft or "\\lambda" in model_draft or "f(t)" in model_draft:
                print("  [OK] 检测到 LaTeX 符号或公式。")
            if "目标函数" in model_draft or "约束" in model_draft or "方程" in model_draft:
                print("  [OK] 包含核心建模模块。")
            break

    # 模拟用户反馈：要求增加离散化逻辑
    print("\n--- [阶段 2：人工反馈修正] ---")
    feedback = "目前的微分方程是连续的，请增加离散化差分方程的形式，方便后续编程实现。"
    mcm_graph.update_state(config, {"status": "REJECTED", "human_feedback": feedback})
    
    for event in mcm_graph.stream(None, config, stream_mode="values"):
        if event.get("draft"):
            updated_model = event.get("draft")
            if "t+1" in updated_model or "Delta" in updated_model or "差分" in updated_model:
                print("✅ [成功] AI 已根据反馈补充了离散化模型公式。")
            break

    print("\n✅ 数学建模节点测试圆满完成！")

if __name__ == "__main__":
    test_modeler_flow()
