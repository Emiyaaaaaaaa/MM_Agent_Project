import os
import re
import json
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 配置 (Configuration) ---
INPUT_DIR = Path(r"c:\Users\a2231\Desktop\RAG\RAG_Output_Gemini")
OUTPUT_DIR = Path(r"c:\Users\a2231\Desktop\RAG\RAG_Chunks_Gemini")
CHUNK_SIZE = 1500  # 建议分块大小在 1500-2000 字符之间
CHUNK_OVERLAP = 200

# 语义锚点分类
STATIC_TOP_LEVELS = ["Summary", "Contents", "Memo", "Reference", "Appendices"]
CONTEXTUAL_KEYS = ["Introduction", "Model preparation", "Sensitivity Analysis", "Model Evaluation"]

# --- 核心算法 (Core Algorithm) ---

def expert_logic_splitter(content: str, source_metadata: Dict[str, Any]) -> List[Document]:
    """
    基于逻辑权重和语义锚点的深度分块算法
    """
    # 1. 匹配模式：优先寻找“层级编号”
    # 支持 1., 1.1, 2.1.2 等格式
    header_pattern = r'^(\d+(\.\d+)*)\s+.*|^#{1,6}\s+.*'
    
    sections = []
    lines = content.split('\n')
    current_section_lines = []
    current_header = "Root"
    current_level = 0
    for line in lines:
        clean_line = line.strip()
        
        # 识别层级：语义优先策略 (Semantic Priority Strategy)
        is_static = any(kw.lower() in clean_line.lower() for kw in STATIC_TOP_LEVELS)
        is_contextual = any(kw.lower() in clean_line.lower() for kw in CONTEXTUAL_KEYS)
        is_md_header = clean_line.startswith('#')
        
        # 剥离 Markdown 符号后尝试匹配编号 (1.1, 1.1.1 等)
        stripped_line = clean_line.lstrip('#* ').strip()
        num_match = re.match(r'^(\d+(\.\d+)*)', stripped_line)
        
        # 判定分块边界
        if (num_match or is_static or is_contextual or is_md_header) and 0 < len(clean_line) < 100:
            # 存入上一个章节
            if current_section_lines:
                sections.append(Document(
                    page_content='\n'.join(current_section_lines),
                    metadata={
                        **source_metadata,
                        "chunk_header": current_header,
                        "chunk_level": current_level
                    }
                ))
            
            # 优先级 1: 绝对语义锚点 (Summary/Memo) -> Level 1
            if is_static:
                current_level = 1
            # 优先级 2: 显式编号 (1.1, 1.1.1) -> 按照点号数量判断
            elif num_match:
                num_part = num_match.group(1)
                current_level = num_part.count('.') + 1
            # 优先级 3: Markdown 定义的层级
            elif is_md_header:
                current_level = clean_line.count('#')
            # 优先级 4: 上下文关键词默认层级
            else:
                current_level = 2
                
            current_header = stripped_line
            current_section_lines = [line]
        else:
            current_section_lines.append(line)
            
    # 存入最后一个章节
    if current_section_lines:
        sections.append(Document(
            page_content='\n'.join(current_section_lines),
            metadata={
                **source_metadata,
                "chunk_header": current_header,
                "chunk_level": current_level
            }
        ))
    return sections

def process_markdown_file(file_path: Path):
    """
    读取、解析并拆分单个 Markdown 文件
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 提取 YAML 前置元数据 (Front Matter)
    metadata = {}
    front_matter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if front_matter_match:
        try:
            metadata = yaml.safe_load(front_matter_match.group(1))
            body_content = content[front_matter_match.end():]
        except Exception:
            body_content = content
    else:
        body_content = content

    # 2. 步骤一：逻辑分块 (Semantic/Logical Splitting)
    logical_docs = expert_logic_splitter(body_content, metadata)
    
    # 3. 步骤二：细粒度字符拆分 (Recursive Splitting for over-sized chunks)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    
    final_chunks = child_splitter.split_documents(logical_docs)
    
    # 4. 迁移图片并处理引用
    final_output_chunks = []
    
    # 准备输出子目录
    rel_path = file_path.relative_to(INPUT_DIR)
    out_dir = OUTPUT_DIR / rel_path.parent / file_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    
    image_pattern = r'!\[.*?\]\((.*?)\)|\(([\w._-]+\.(?:png|jpg|jpeg|gif|webp))\)'
    
    for c in final_chunks:
        chunk_text = c.page_content
        matches = list(re.finditer(image_pattern, chunk_text))
        
        valid_images = []
        for match in matches:
            img_rel_path = match.group(1) or match.group(2)
            full_match_text = match.group(0)
            
            abs_img_path = file_path.parent / img_rel_path
            
            if abs_img_path.exists():
                dest_img_path = out_dir / abs_img_path.name
                import shutil
                if not dest_img_path.exists():
                    shutil.copy2(abs_img_path, dest_img_path)
                
                valid_images.append({
                    "local_path": str(dest_img_path.absolute()),
                    "mime_type": f"image/{abs_img_path.suffix.lstrip('.').lower()}".replace("jpg", "jpeg")
                })
            else:
                # 缺失则删除标记
                chunk_text = chunk_text.replace(full_match_text, "")

        final_output_chunks.append({
            "content": chunk_text.strip(),
            "images": valid_images,
            "metadata": c.metadata
        })
    
    # 保存分块结果
    chunk_file = out_dir / f"{file_path.stem}_chunks.json"
    
    with open(chunk_file, 'w', encoding='utf-8') as f:
        json.dump(final_output_chunks, f, ensure_ascii=False, indent=2)

def main():
    print("开始执行多模态文档分块任务...")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    md_files = [f for f in INPUT_DIR.rglob("*.md") if "commentary" not in f.parts]
    print(f"找到 {len(md_files)} 个 Markdown 文件 (已排除评委点评)。")
    
    for md_file in tqdm(md_files, desc="Processing chunks"):
        try:
            process_markdown_file(md_file)
        except Exception as e:
            print(f"文件处理失败 {md_file}: {e}")

if __name__ == "__main__":
    main()
