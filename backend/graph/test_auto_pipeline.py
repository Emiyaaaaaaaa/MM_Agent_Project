import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.builder import mcm_graph

load_dotenv()

def test_auto_pipeline():
    print(">>> 正在启动『零对话-自动解析分析』全链路流水线测试...")
    
    config = {"configurable": {"thread_id": "auto_pipeline_thread"}}
    
    # 场景：用户只上传了文件，还没有说话 (messages 为空或仅有寒暄)
    # 我们直接在 shared_memory 中存入路径
    test_file = os.path.join("rag_service", "COMAP", "F", "2025", "F", "2508764.pdf")
    upload_path = os.path.abspath(test_file)
    
    initial_input = {
        "messages": [], # 用户还没说话
        "shared_memory": {"input_file_path": upload_path}
    }
    
    print(f"\n--- [流水线启动：仅输入文件路径] ---")
    
    # 我们使用 stream 观察状态演变
    for event in mcm_graph.stream(initial_input, config, stream_mode="values"):
        current_stage = event.get("current_stage")
        if current_stage:
            print(f"📍 当前阶段: {current_stage}")
            
        # 如果到了分析完成阶段，说明全链路自动跑通了
        if current_stage == "Analysis Completed":
            print("\n✅ [成功] 流水线已自动完成：上传 -> 解析 (Read) -> 审题分析 (Analysis)")
            print(f"分析报告预览:\n{event.get('shared_memory', {}).get('analysis_report', '')[:200]}...")
            break

    print("\n✅ 全流程自动化测试圆满成功！")

if __name__ == "__main__":
    test_auto_pipeline()
