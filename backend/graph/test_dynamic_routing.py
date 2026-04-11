import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.supervisor import supervisor_node

load_dotenv()

def test_dynamic_routing():
    print(">>> 正在启动『动态意图优先』专项路由测试...")
    
    # CASE 1: 故意缺失前置，但用户明确要求编程
    print("\n--- [CASE 1: 意图优先 (跳级测试)] ---")
    state_1 = {
        "messages": [HumanMessage(content="我不需要分析，请直接帮我写一个 PSM 算法的 Python 代码。")],
        "shared_memory": {}, # 故意没有任何前置
        "status": "PENDING"
    }
    result_1 = supervisor_node(state_1)
    print(f"✅ 用户要求辅助编程，决策结果: {result_1['next']}")
    assert result_1["next"] == "Coding", "CASE 1 失败：未能识别‘意图优先’原则"

    # CASE 2: 进度反馈 (当指令模糊时)
    print("\n--- [CASE 2: 进度辅助 (顺流测试)] ---")
    state_2 = {
        "messages": [HumanMessage(content="好的，继续吧。")],
        "shared_memory": {
            "analysis_report": "分析已完成..."
        },
        "status": "APPROVED"
    }
    result_2 = supervisor_node(state_2)
    print(f"✅ 分析已完成且用户指令模糊，决策结果: {result_2['next']}")
    assert result_2["next"] == "Modeling", "CASE 2 失败：未能识别‘进度补偿’原则"

    # CASE 3: 直接回答 (闲聊/非任务)
    print("\n--- [CASE 3: 闲聊识别] ---")
    state_3 = {
        "messages": [HumanMessage(content="你今天心情怎么样？")],
        "shared_memory": {},
        "status": "APPROVED"
    }
    result_3 = supervisor_node(state_3)
    print(f"✅ 闲聊场景，决策结果: {result_3['next']}")
    assert result_3["next"] == "Respond", "CASE 3 失败：未能正确路由到对话节点"

    print("\n✅ 动态意图优先路由所有测试通过！")

if __name__ == "__main__":
    test_dynamic_routing()
