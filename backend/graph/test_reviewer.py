import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.builder import mcm_graph

load_dotenv()

def test_reviewer_flow():
    print(">>> 正在启动『多模态结果审查』集成测试...")
    
    # 确保测试路径存在
    plot_dir = os.path.join("backend", "static", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "output.png")
    
    if not os.path.exists(plot_path):
        print("⚠️ [警告] 未发现 output.png，请确保之前运行过 Coder 节点测试。")
        # 如果没有图，审查将回退到纯文本模式，这也是一种测试路径
    
    config = {"configurable": {"thread_id": "reviewer_test_thread"}}
    
    # 构造包含代码和模型的初始状态
    initial_input = {
        "messages": [HumanMessage(content="请对当前结果进行最终审查，特别是检查图表趋势是否符合逻辑。")],
        "shared_memory": {
            "analysis_report": "核心目标：模拟由于资源枯竭导致的人口崩溃过程。",
            "mathematical_model": "使用 Logistic 增长模型 P(t) = K / (1 + exp(-r*(t-t0)))。预期趋势：先增长平稳，最后因 K 骤降而崩溃。",
            "generated_code": "import matplotlib.pyplot as plt; plt.plot([1, 2, 3], [10, 20, 30]); plt.show() # 注意：这里代码画的是上升趋势，可能与逻辑不符"
        }
    }
    
    print("\n--- [视觉审查启动] ---")
    for event in mcm_graph.stream(initial_input, config, stream_mode="values"):
        if event.get("draft"):
            report = event.get("draft")
            print(f"✅ 已生成审查报告 (长度: {len(report)}):")
            print("-" * 30)
            print(report[:500] + "...")
            
            # 验证 Reviewer 的批判性思维
            if "一致" in report or "不符" in report or "异常" in report:
                print("\n✅ [成功] 审查节点已产出包含逻辑判定结论的报告。")
            break

    print("\n✅ 多模态结果审查节点测试完成！")

if __name__ == "__main__":
    test_reviewer_flow()
