import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.builder import mcm_graph

load_dotenv()

def test_coder_flow():
    print(">>> 正在启动『程序员』节点测试...")
    
    config = {"configurable": {"thread_id": "coder_debug_thread"}}
    
    # 1. 第一阶段：初始编码需求
    print("\n--- [第一轮：生成基础代码] ---")
    initial_input = {
        "messages": [HumanMessage(content="请帮我写一个 Python 脚本，用核心库实现一个简单的 PSM (倾向评分匹配) 算法。要求包含数据模拟。")],
        "shared_memory": {"analysis": "用户需要实现 PSM 算法用于 2025 F 题的数据处理环节。"}
    }
    
    for event in mcm_graph.stream(initial_input, config, stream_mode="values"):
        if event.get("draft"):
            print(f"✅ 已拦截到草案预览:\n{event.get('draft')[:200]}...")
            break
            
    # 2. 第二阶段：模拟驳回并提出具体框架要求
    print("\n--- [第二轮：补充要求并重写] ---")
    feedback = "生成的代码很好，但请将数据模拟部分改为使用 pandas DataFrame 构造，并且添加注释解释一下 logit 转换。"
    mcm_graph.update_state(config, {"status": "REJECTED", "human_feedback": feedback})
    
    # 唤醒图运行
    for event in mcm_graph.stream(None, config, stream_mode="values"):
        if event.get("draft"):
            new_code = event.get("draft")
            if "DataFrame" in new_code and "logit" in new_code.lower():
                print("✅ [成功] AI 已根据反馈采纳了 pandas 和 logit 建议。")
                print(f"新草案结尾预览:\n...{new_code[-200:]}")
            break

    # 3. 最终确认
    mcm_graph.update_state(config, {"status": "APPROVED"})
    for event in mcm_graph.stream(None, config, stream_mode="values"):
        pass
    
    state = mcm_graph.get_state(config)
    print(f"\n流程结束状态: {not state.next}")
    print("✅ 程序员节点 HITL 测试圆满成功！")

if __name__ == "__main__":
    test_coder_flow()
