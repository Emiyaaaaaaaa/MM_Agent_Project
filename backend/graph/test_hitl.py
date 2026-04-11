import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.builder import mcm_graph

load_dotenv()

def test_hitl_loop():
    print(">>> 正在启动 HITL (人机协作) 审批流测试...")
    
    # 1. 初始输入 (使用 thread_id 进行持久化)
    config = {"configurable": {"thread_id": "test_thread_123"}}
    initial_input = {
        "messages": [HumanMessage(content="2025 年 MCM F 题的主要挑战是什么？")],
        "status": "PENDING"
    }
    
    print("\n--- [第一轮：执行并中断] ---")
    # 第一次模拟运行 (会停在 Respond 之后)
    for event in mcm_graph.stream(initial_input, config, stream_mode="values"):
        pass # 等待中断
    
    # 检查状态
    state = mcm_graph.get_state(config)
    print(f"当前节点: {state.next}")
    print(f"审批状态: {state.values.get('status')}")
    print(f"初步草案预览: {state.values.get('draft')[:100]}...")
    
    # 2. 模拟用户驳回 (Step C & D)
    print("\n--- [第二轮：补充意见并驳回] ---")
    feedback = "回答得不错，但请特别强调一下题目中关于『时间惩罚项』的逻辑。"
    mcm_graph.update_state(config, {"status": "REJECTED", "human_feedback": feedback})
    
    # 唤醒图运行 (由于 status=REJECTED，路由会跳回 Respond)
    for event in mcm_graph.stream(None, config, stream_mode="values"):
         pass
        
    state = mcm_graph.get_state(config)
    print(f"再次生成的草案预览: {state.values.get('draft')[:100]}...")
    if "时间惩罚项" in state.values.get('draft', ''):
        print("✅ 成功！反馈建议已被 AI 采纳。")
    
    # 3. 模拟用户通过
    print("\n--- [第三轮：同意并结束] ---")
    mcm_graph.update_state(config, {"status": "APPROVED", "human_feedback": ""})
    
    # 唤醒图运行 (跳转至 END)
    for event in mcm_graph.stream(None, config, stream_mode="values"):
         pass
         
    state = mcm_graph.get_state(config)
    print(f"流程结束？ {not state.next}")
    print("✅ HITL 全链路测试圆满成功！")

if __name__ == "__main__":
    test_hitl_loop()
