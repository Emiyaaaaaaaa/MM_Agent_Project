import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.builder import mcm_graph

load_dotenv()

def test_intelligent_routing_v2():
    print(">>> 正在启动『全自动内容驱动路由 (V2)』测试...")
    
    # --- Case 1: 上传一份数据字典/说明书 (非题目) ---
    print("\n[Case 1] 上传数据说明书 (预览含字段定义)...")
    input_1 = {
        "messages": [],
        "shared_memory": {
            "raw_document_content": (
                "Dataset Overview: This file contains traffic flow data for Seattle. "
                "Columns: timestamp, vehicle_count, average_speed, weather_condition. "
                "The data is collected from sensors on I-5. Please summarize the data structure."
            )
        }
    }
    config_1 = {"configurable": {"thread_id": "smart_v2_case1"}}
    for event in mcm_graph.stream(input_1, config_1, stream_mode="values"):
        next_node = event.get("next")
        if next_node:
            print(f"📍 LLM 决策路由: {next_node}")
            if next_node in ["Respond", "Coding"]:
                print("✅ [成功] 识别为非题目资料，路由正确。")
            else:
                print("❌ [警告] 可能误判为建模题目。")
            break

    # --- Case 2: 上传真实竞赛题目 (2025 MCM F) ---
    print("\n[Case 2] 上传真实建模竞赛题目 (2025 F)...")
    input_2 = {
        "messages": [],
        "shared_memory": {
            "raw_document_content": (
                "Problem F: Sustainable Business and Resource Management. "
                "Background: Sustainability is the core of future growth. "
                "Your team is asked to develop a model that optimizes resource allocation... "
                "Requirement 1: Define a sustainability metric. Requirement 2: Forecast future scarcity..."
            )
        }
    }
    config_2 = {"configurable": {"thread_id": "smart_v2_case2"}}
    for event in mcm_graph.stream(input_2, config_2, stream_mode="values"):
        next_node = event.get("next")
        if next_node:
            print(f"📍 LLM 决策路由: {next_node}")
            if next_node == "Analysis":
                print("✅ [成功] 识别为竞赛题目，进入深度分析流。")
            else:
                print("❌ [失败] 题目未获分析。")
            break

    # --- Case 3: 上传一段算法辅助代码 ---
    print("\n[Case 3] 上传算法实现助手 (BFS/DFS 代码)...")
    input_3 = {
        "messages": [],
        "shared_memory": {
            "raw_document_content": (
                "import collections\ndef bfs(graph, root):\n    visited, queue = set(), collections.deque([root])\n"
                "    visited.add(root)\n    while queue: ... \n"
                "This is a standard template for breadth-first search."
            )
        }
    }
    config_3 = {"configurable": {"thread_id": "smart_v2_case3"}}
    for event in mcm_graph.stream(input_3, config_3, stream_mode="values"):
        next_node = event.get("next")
        if next_node:
            print(f"📍 LLM 决策路由: {next_node}")
            if next_node in ["Respond", "Coding"]:
                print("✅ [成功] 识别为辅助代码，成功跳转。")
            else:
                print("❌ [失败] 代码被误认为建模题目。")
            break

    print("\n✅ V2 版智能路由验证完成！")

if __name__ == "__main__":
    test_intelligent_routing_v2()
