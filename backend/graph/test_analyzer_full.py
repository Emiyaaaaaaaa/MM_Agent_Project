import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.builder import mcm_graph

load_dotenv()

def test_analyzer_flow():
    print(">>> 正在启动『审题分析』节点全链路测试...")
    
    config = {"configurable": {"thread_id": "analyzer_test_thread"}}
    
    # 模拟已读取题目的状态
    initial_input = {
        "messages": [HumanMessage(content="请详细分析一下这个题目，提取核心问题并设定模型假设。")],
        "shared_memory": {
            "raw_document_content": (
                "2025 MCM Problem F: Sustainable Business and Resource Management. "
                "The task is to analyze how a corporation can balance profitability with resource conservation. "
                "Consider a scenario where water usage is a critical factor. The goal is to provide a 5-year strategy..."
            )
        }
    }
    
    print("\n--- [测试：执行审题分析] ---")
    for event in mcm_graph.stream(initial_input, config, stream_mode="values"):
        if event.get("draft"):
            report = event.get("draft")
            print(f"✅ 已拦截到分析报告草案 (字数: {len(report)}):")
            print("-" * 30)
            # 检查关键模块是否存在
            if "**核心问题" in report: print("  [OK] 包含核心问题提取")
            if "**模型合理假设" in report: print("  [OK] 包含模型假设")
            if "**关键变量" in report: print("  [OK] 包含变量定义")
            break

    # 模拟用户反馈与重写
    print("\n--- [测试：反馈修正] ---")
    feedback = "请额外增加一条关于『政府政策补贴』的社会学假设。"
    mcm_graph.update_state(config, {"status": "REJECTED", "human_feedback": feedback})
    
    for event in mcm_graph.stream(None, config, stream_mode="values"):
        if event.get("draft"):
            new_report = event.get("draft")
            if "政府" in new_report or "补贴" in new_report:
                print("✅ [成功] AI 已在更新后的报告中加入政府补贴假设。")
            break

    print("\n✅ 审题分析节点 HITL 测试圆满成功！")

if __name__ == "__main__":
    test_analyzer_flow()
