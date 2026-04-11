import os
import base64
from pathlib import Path
import docx
import fitz  # PyMuPDF
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress

async def parse_pdf_with_vision(pdf_path: str, llm) -> str:
    """使用 Gemini Vision 逐页渲染 PDF 并提取高保真 Markdown (含 LaTeX)"""
    try:
        doc = fitz.open(pdf_path)
        full_markdown = []
        
        for i in range(len(doc)):
            page = doc[i]
            # 渲染页面为图片 (300 DPI)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            b64_img = base64.b64encode(img_data).decode("utf-8")
            
            await broadcast_progress("Reader", f"正在对第 {i+1}/{len(doc)} 页进行多模态 OCR 视觉解析...", 30 + int(i/len(doc)*60))
            
            prompt = (
                "You are a professional academic OCR assistant. "
                "Extract all text from this page and convert it into high-fidelity Markdown. "
                "CRITICAL: For all mathematical formulas, symbols, and equations, use standard LaTeX syntax (e.g., $...$ or $$...$$). "
                "Maintain the original layout and titles. Output ONLY the Markdown content."
            )
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                    },
                ]
            )
            
            response = await llm.ainvoke([message])
            full_markdown.append(response.content.strip())
            
        return "\n\n---\n\n".join(full_markdown)
    except Exception as e:
        return f"Vision OCR Error: {str(e)}"

def read_file_content_basic(file_path: str) -> str:
    """传统文本级读取 (快速但可能丢失格式/公式)"""
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        doc = fitz.open(path)
        return "\n".join([page.get_text() for page in doc])
    elif ext in [".docx", ".doc"]:
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext in [".txt", ".md"]:
        return path.read_text(encoding="utf-8")
    return ""

async def reader_node(state: AgentState):
    """
    Reader 节点：负责将磁盘上的物理文件加载到 Agent 的共享状态空间 (State) 中。
    支持：
    1. 传统文本提取 (用于简单 DOCX/TXT)
    2. 多模态视觉 OCR (用于 PDF 复杂公式提取)
    """
    allowed_exts = {".pdf", ".docx", ".doc", ".txt", ".md"}
    shared_mem = state.get("shared_memory", {})
    file_path = shared_mem.get("input_file_path")
    
    if not file_path:
        return {"status": "REJECTED", "human_feedback": "缺少 'input_file_path'，请上传文件。"}
    
    # 动态模型实例化 (从任务配置加载)
    api_key = shared_mem.get("api_key")
    model_id = shared_mem.get("model_id") or "gemini-2.5-flash"
    
    if not api_key:
        print("[Warning] No API key found in shared_memory for Reader. Falling back to environment variable.")
        api_key = os.environ.get("GOOGLE_API_KEY")
    
    llm = ChatGoogleGenerativeAI(model=model_id, google_api_key=api_key)
    
    await broadcast_progress("Reader", f"正在解析物理文件: {os.path.basename(file_path)}...", 10)
    
    ext = Path(file_path).suffix.lower()
    if ext not in allowed_exts:
        return {"status": "REJECTED", "human_feedback": f"格式不支持 ({ext})"}

    if ext == ".pdf":
        # 强制 PDF 使用多模态视觉解析以确保公式准确性
        await broadcast_progress("Reader", "检测到 PDF，正在启动多模态高保真视觉解析引擎...", 20)
        content = await parse_pdf_with_vision(file_path, llm)
    else:
        # 其他格式采用常规读取
        content = read_file_content_basic(file_path)
    
    await broadcast_progress("Reader", "解析成功，内容已注入共享状态空间。", 90)
    
    new_memory = shared_mem.copy()
    new_memory["raw_document_content"] = content
    
    await broadcast_progress("Reader", "文件处理完毕。", 100)
    
    return {
        "shared_memory": new_memory,
        "status": "APPROVED",
        "next": "Supervisor",
        "current_stage": "File Content Fully Loaded (Multi-modal Enabled)"
    }

reader_node = reader_node  # 导出
