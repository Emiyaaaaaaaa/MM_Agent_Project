import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.builder import build_graph

load_dotenv()

def test_autonomous_coordination():
    print(">>> 正在启动『全链路联动』专项测试...")
    app = build_graph()
    
    # 模拟状态：分析已完成且被通过
    state = {
        "messages": [HumanMessage(content="这是赛题文件，请分析。")],
        "shared_memory": {
            "analysis_report": "问题 1 要求建立人口预测模型...",
            "raw_document_content": "2025 MCM Problem F: Sustainable Population..."
        },
        "status": "APPROVED",
        "next": "Analysis" # 假设刚从 Analysis 结束
    }
    
    # 执行图
    # 预期流转：Analysis (Approved) -> Supervisor -> Modeling (Pending)
    print("\n--- [启动自动流转测试] ---")
    
    # 使用 invoke，它会运行直到遇到 interrupt (即下一个节点的 PENDING 状态)
    config = {"configurable": {"thread_id": "test_coord_001"}}
    result = app.invoke(state, config=config)
    
    print(f"\n✅ 自动化流转结果:")
    print(f"  - 当前节点 (next): {result.get('next')}")
    print(f"  - 审批状态 (status): {result.get('status')}")
    print(f"  - 阶段标记 (current_stage): {result.get('current_stage')}")
    
    if result.get("next") == "Modeling":
        print("\n✅ [成功] 主管节点监测到 Analysis 已完成，并成功自动导向了 Modeling！")
    else:
        print(f"\n❌ [失败] 期望导向 Modeling，但实际导向了 {result.get('next')}")

if __name__ == "__main__":
    test_autonomous_coordination()
