import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.builder import mcm_graph

load_dotenv()

def test_coder_visualization():
    print(">>> 正在启动『程序员』自主可视化测试...")
    
    config = {"configurable": {"thread_id": "viz_test_thread"}}
    
    # 模拟一个需要绘图的场景（虽然只说了预测，但逻辑上涉及趋势，Agent 应自主绘图）
    initial_input = {
        "messages": [HumanMessage(content="请帮我建立一个 Logistic 人口增长模型，预测某地区未来 20 年的人口变化。")],
        "shared_memory": {
            "analysis": "2025 年 MCM F 题背景：分析资源承载力与人口关系的动态演化。"
        }
    }
    
    print("\n--- [测试：自主判定绘图并生成] ---")
    for event in mcm_graph.stream(initial_input, config, stream_mode="values"):
        if event.get("draft"):
            code = event.get("draft")
            print(f"✅ 生成的代码长度: {len(code)}")
            
            # 验证关键绘图逻辑是否包含
            if "matplotlib" in code or "plt." in code:
                print("✅ [状态] AI 已自主识别绘图需求并加入 matplotlib 库。")
            if "backend/static/plots" in code.replace("\\", "/"):
                print("✅ [状态] AI 已指定正确的静态资源保存路径。")
            if "seaborn" in code.lower() or "style.use" in code.lower():
                print("✅ [审美] AI 已应用科学绘图风格。")
            
            # 这里我们不真正运行生成的代码（因为环境限制），但通过静态检查确认 Logic。
            break

    print("\n✅ 自主可视化逻辑验证完毕！")

if __name__ == "__main__":
    test_coder_visualization()
