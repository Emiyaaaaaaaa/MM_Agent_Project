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
# 输入目录：评委点评转换后的 Markdown 文件
INPUT_DIR = Path(r"c:\Users\a2231\Desktop\RAG\RAG_Output_Gemini\commentary")
OUTPUT_DIR = Path(r"c:\Users\a2231\Desktop\RAG\RAG_Chunks_Gemini\commentary")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# 语义锚点配置
STATIC_TOP_LEVELS = ["Summary", "Introduction", "General Comments", "Problem Specific Comments", "Conclusion"]
# 识别 Team 分析和 Case Study 等上下文关键字
CONTEXTUAL_KEYS = ["Team", "Analysis", "Case Study", "Model", "Approach", "Results"]

def expert_logic_splitter(content: str, source_metadata: Dict[str, Any]) -> List[Document]:
    """
    逻辑感知的评委点评分块算法
    """
    lines = content.split('\n')
    sections = []
    current_section_lines = []
    current_header = "Root"
    current_level = 0
    
    for line in lines:
        clean_line = line.strip()
        
        # 1. 识别层级与锚点
        is_static = any(kw.lower() in clean_line.lower() for kw in STATIC_TOP_LEVELS)
        is_contextual = any(kw.lower() in clean_line.lower() for kw in CONTEXTUAL_KEYS)
        is_md_header = clean_line.startswith('#')
        
        # 剥离符号识别编号 (如 1.1, Team 1234567)
        stripped_line = clean_line.lstrip('#* ').strip()
        num_match = re.match(r'^(\d+(\.\d+)*)', stripped_line)
        team_match = re.search(r'Team\s*(\d{7})', stripped_line, re.IGNORECASE)
        
        # 2. 判定分块边界 (标题特征：短且符合语义或编号)
        if (num_match or team_match or is_static or is_contextual or is_md_header) and 0 < len(clean_line) < 120:
            if current_section_lines:
                sections.append(Document(
                    page_content='\n'.join(current_section_lines),
                    metadata={
                        **source_metadata,
                        "chunk_header": current_header,
                        "chunk_level": current_level
                    }
                ))
            
            # 层级逻辑
            if is_static:
                current_level = 1
            elif team_match:
                current_level = 2 # Team 分析通常是二级
            elif num_match:
                current_level = num_match.group(1).count('.') + 1
            elif is_md_header:
                current_level = clean_line.count('#')
            else:
                current_level = 2
                
            current_header = stripped_line
            current_section_lines = [line]
        else:
            current_section_lines.append(line)
            
    if current_section_lines:
        sections.append(Document(
            page_content='\n'.join(current_section_lines),
            metadata={**source_metadata, "chunk_header": current_header, "chunk_level": current_level}
        ))
    return sections

def process_markdown_file(file_path: Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取 YAML
    metadata = {}
    front_matter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    if front_matter_match:
        try:
            metadata = yaml.safe_load(front_matter_match.group(1))
            body_content = content[front_matter_match.end():]
        except:
            body_content = content
    else:
        body_content = content

    # 分块
    logical_docs = expert_logic_splitter(body_content, metadata)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    final_chunks = child_splitter.split_documents(logical_docs)
    
    # 4. 迁移图片并处理引用
    final_output_chunks = []
    
    # 准备输出子目录 (嵌套结构以便于管理图片)
    rel_path = file_path.relative_to(INPUT_DIR)
    out_dir = OUTPUT_DIR / rel_path.parent / file_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 匹配标准 Markdown 图片和 可能出现的 (image.png) 格式
    standard_pattern = r'!\[.*?\]\((.*?)\)'
    alt_pattern = r'\(([\w._-]+\.(?:png|jpg|jpeg|gif|webp))\)'
    
    image_pattern = r'!\[.*?\]\((.*?)\)|\(([\w._-]+\.(?:png|jpg|jpeg|gif|webp))\)'
    
    for c in final_chunks:
        chunk_text = c.page_content
        matches = list(re.finditer(image_pattern, chunk_text))
        
        valid_images = []
        # 为了避免替换冲突，我们从后往前处理或者使用特殊的占位符转换
        # 这里采用简单的处理逻辑，因为图片标记通常是独立的
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
    if not INPUT_DIR.exists():
        print(f"错误: 找不到输入目录: {INPUT_DIR}")
        return
    
    md_files = list(INPUT_DIR.rglob("*.md"))
    print(f"开始对 {len(md_files)} 个点评文件进行多模态分块...")
    
    for md_file in tqdm(md_files):
        try:
            process_markdown_file(md_file)
        except Exception as e:
            print(f"处理失败 {md_file}: {e}")

if __name__ == "__main__":
    main()
