import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.builder import mcm_graph

load_dotenv()

def test_reader_integration():
    print(">>> 正在启动 Reader 节点集成测试...")
    
    # 测试文件路径 (使用 workspace 中的真实 PDF)
    pdf_path = os.path.join("rag_service", "COMAP", "F", "2025", "F", "2508764.pdf")
    
    config = {"configurable": {"thread_id": "reader_test_split"}}
    
    # 场景：用户提供路径并要求读取
    query = f"请帮我读取并解析这个题目文件：{pdf_path}"
    
    initial_input = {
        "messages": [HumanMessage(content=query)],
        "shared_memory": {"input_file_path": pdf_path} # 模拟前端已预填路径
    }
    
    print(f"\n--- [测试开始：读取指令] ---")
    for event in mcm_graph.stream(initial_input, config, stream_mode="values"):
        # 我们会在输出中看到节点流转
        pass
        
    state = mcm_graph.get_state(config)
    print(f"\n当前节点: {state.next}")
    
    # 检查共享记忆
    memory = state.values.get("shared_memory", {})
    content = memory.get("raw_document_content", "")
    
    if content:
        print(f"✅ 成功读取文件内容！预览前 100 字:\n{content[:100]}...")
    else:
        print("❌ 未能读取到文件内容。")
        
    print(f"当前阶段: {state.values.get('current_stage')}")

if __name__ == "__main__":
    test_reader_integration()
