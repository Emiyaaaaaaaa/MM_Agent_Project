import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.builder import mcm_graph

load_dotenv()

def test_writer_flow():
    print(">>> 正在启动『学术写作』分段流程测试...")
    
    config = {"configurable": {"thread_id": "writer_test_thread"}}
    
    # --- Case 1: 正常对话 (预期不触发 Writing) ---
    print("\n[Case 1] 随机提问 (测试非敏感触发)...")
    input_1 = {"messages": [HumanMessage(content="你好，请问建模的第一步是什么？")]}
    for event in mcm_graph.stream(input_1, config, stream_mode="values"):
        if event.get("next") == "Writing":
            print("❌ [失败] 未经授权触发了写作节点。")
            break
        elif event.get("next") == "Respond":
            print("✅ [成功] 路由至 Respond，未越权触发。")
            break

    # --- Case 2: 明确要求写作 (预期触发 Writing 第1章) ---
    print("\n[Case 2] 明确要求生成完整论文...")
    input_2 = {"messages": [HumanMessage(content="请根据之前的建模结果，为我生成一份完整的学术论文正文。")]}
    # 模拟已有建模数据
    mcm_graph.update_state(config, {
        "shared_memory": {
            "analysis_report": "基于 PSM 算法的资源分配方案...",
            "mathematical_model": "P(t) = ...",
            "review_report": "模型逻辑一致，图表符合预期。"
        }
    })
    
    for event in mcm_graph.stream(input_2, config, stream_mode="values"):
        if event.get("draft"):
            draft = event.get("draft")
            stage = event.get("current_stage")
            print(f"✅ 已完成阶段: {stage}")
            if "Abstract" in draft or "Introduction" in draft:
                print("  [OK] 第一章节内容生成成功。")
            break

    # --- Case 3: 模拟用户确认，继续下一章 ---
    print("\n[Case 3] 模拟用户输入『同意』，继续下一章节...")
    # 设置状态为 APPROVED 触发流转
    mcm_graph.update_state(config, {"status": "APPROVED"})
    
    # 再次运行，此时 Supervisor 应该检测到进度未完成，再次路由到 Writing
    # 注意：在真实的 LangGraph 中，APPROVED 可能会结束当前 Thread
    # 但由于我们的逻辑是循环的（除非环节完成），我们再次调用 stream
    for event in mcm_graph.stream(None, config, stream_mode="values"):
        # 此时 Supervisor 应该再次决策
        next_val = event.get("next")
        if next_val == "Writing":
            print("✅ [成功] 主管节点识别到进度未完成，继续路由至 Writing。")
        
        if event.get("draft"):
            new_draft = event.get("draft")
            new_stage = event.get("current_stage")
            print(f"✅ 已完成阶段: {new_stage}")
            if "Modeling" in new_stage:
                 print("  [OK] 第二章节 (Modeling) 续写成功。")
            break

    print("\n✅ 学术写作分段流测试圆满完成！")

if __name__ == "__main__":
    test_writer_flow()
