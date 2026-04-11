import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.builder import mcm_graph

# 加载环境变量
load_dotenv()

def test_supervisor_decisions():
    print(">>> 正在启动 LangGraph 编排测试 (Supervisor 模式)...")
    
    test_cases = [
        "你好，你是谁？",                      # 预测结果: Respond
        "2025 年 Question F 的主要挑战是什么？",  # 预测结果: Retrieve
        "你能解释一下什么是 PSM 模型吗？",       # 预测结果: Retrieve
        "谢谢你，我的建模完成了。",             # 预测结果: Respond
    ]
    
    for query in test_cases:
        print(f"\n--- 测试问题: {query} ---")
        # 初始化状态
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "next": ""
        }
        
        # 运行图
        # 我们这里只运行 Supervisor 节点后的路径，或者完整运行
        # 为了验证决策，我们查看状态中的 next 字段
        for output in mcm_graph.stream(initial_state):
            for node_name, state_update in output.items():
                print(f"[Node Execution] {node_name}")
                if "next" in state_update:
                    print(f"  [Supervisor Decision] 下一步: {state_update['next']}")
    
if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("错误：请先设置 GOOGLE_API_KEY")
    else:
        test_supervisor_decisions()
