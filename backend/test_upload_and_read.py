import os
import requests
from pathlib import Path
from langchain_core.messages import HumanMessage
from backend.graph.builder import mcm_graph

def test_full_upload_flow():
    print(">>> 正在启动『上传 - 自动解析』全链路测试...")
    
    BASE_URL = "http://localhost:8000"
    # 我们假设本地 server 已启动，或者我们直接调子程序逻辑
    # 为了简化，我们直接模拟上传后的文件路径
    
    # 1. 模拟一个合法 PDF
    test_pdf = os.path.join("rag_service", "COMAP", "F", "2025", "F", "2508764.pdf")
    print(f"\n--- [场景 1：合法文件自动解析] ---")
    
    config = {"configurable": {"thread_id": "auto_read_flow"}}
    # 模拟上传后得到的 local_path
    upload_result_path = os.path.abspath(test_pdf)
    
    initial_input = {
        "messages": [HumanMessage(content="你好")],
        "shared_memory": {"input_file_path": upload_result_path}
    }
    
    # 运行图
    for event in mcm_graph.stream(initial_input, config, stream_mode="values"):
        if "shared_memory" in event:
            content = event["shared_memory"].get("raw_document_content", "")
            if content:
                print(f"✅ [成功] 监测到文件并自动完成解析。内容预览: {content[:50]}...")
                break

    # 2. 模拟非法格式
    print(f"\n--- [场景 2：非法格式报错拦截] ---")
    bad_config = {"configurable": {"thread_id": "bad_format_flow"}}
    bad_input = {
        "messages": [HumanMessage(content="帮我读下这个文件")],
        "shared_memory": {"input_file_path": "test.exe"} # 不支持的格式
    }
    
    for event in mcm_graph.stream(bad_input, bad_config, stream_mode="values"):
        status = event.get("status")
        if status == "REJECTED":
            print(f"✅ [成功] 拦截非法格式。错误反馈: {event.get('human_feedback')}")
            break

if __name__ == "__main__":
    # 提醒：此测试需要本地能找到对应的 PDF 文件
    test_full_upload_flow()
