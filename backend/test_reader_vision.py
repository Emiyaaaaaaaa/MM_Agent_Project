import asyncio
import os
from backend.graph.nodes.reader import reader_node

async def test_reader():
    # 模拟一个包含 PDF 的状态
    # 这里的 PDF 路径需要是一个真实存在的 PDF，或者我们在测试中临时创建一个
    # 为了演示，我们找一个 rag_service 目录下的 pdf
    
    pdf_path = r"c:\Users\a2231\Desktop\MM_Agent_Project\rag_service\COMAP\F\2025\2508764.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Test skipped: {pdf_path} not found.")
        return

    state = {
        "messages": [],
        "shared_memory": {
            "input_file_path": pdf_path,
            "model_id": "gemini-1.5-flash"
        }
    }
    
    print("Testing ReaderNode with Vision OCR...")
    result = await reader_node(state)
    
    content = result["shared_memory"].get("raw_document_content", "")
    print(f"Result Status: {result['status']}")
    print(f"Content Length: {len(content)}")
    print(f"Preview: {content[:500]}...")

if __name__ == "__main__":
    asyncio.run(test_reader())
