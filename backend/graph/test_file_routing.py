import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.builder import mcm_graph

load_dotenv()

def test_file_routing():
    print(">>> 正在启动『智能文件路由』测试...")
    config = {"configurable": {"thread_id": "routing_test_thread"}}
    
    # --- 场景 A: 上传真实题目 (2025 F) ---
    print("\n[Case A] 上传建模题目文件...")
    # 模拟 Read 节点已解析出内容
    input_a = {
        "messages": [],
        "shared_memory": {
            "raw_document_content": (
                "2025 MCM Problem F: Sustainable Business and Resource Management. "
                "Corporations must balance profit with environment. Please provide a 5-year strategy..."
            )
        }
    }
    
    for event in mcm_graph.stream(input_a, config, stream_mode="values"):
        if event.get("next") == "Analysis":
            print("✅ [成功] 识别为建模题目，成功路由至 Analysis。")
            break
        elif event.get("next") == "Respond":
            print("❌ [失败] 题目被误判为普通回复。")
            break

    # --- 场景 B: 上传非题目文件 (例如一段代码或笔记) ---
    print("\n[Case B] 上传非建模题目文件 (代码片段)...")
    input_b = {
        "messages": [],
        "shared_memory": {
            "raw_document_content": (
                "def hello_world():\n    print('This is a python script for data processing.')\n"
                "It contains some utility functions but no contest problem description."
            )
        }
    }
    
    # 使用新 thread 防止状态干扰
    config_b = {"configurable": {"thread_id": "routing_test_thread_b"}}
    for event in mcm_graph.stream(input_b, config_b, stream_mode="values"):
        next_node = event.get("next")
        if next_node == "Analysis":
            print("❌ [失败] 非题目文件被误判为建模题目。")
            break
        elif next_node in ["Respond", "Coding"]:
            print(f"✅ [成功] 识别为非题目文件，路由至 {next_node}。")
            break

    print("\n✅ 智能路由分类测试完成！")

if __name__ == "__main__":
    test_file_routing()
