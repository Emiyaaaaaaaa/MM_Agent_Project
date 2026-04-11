import os
import re
import base64
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe_images_in_md(md_path, llm):
    """解析 Markdown 中的图片并注入 AI 描述"""
    content = md_path.read_text(encoding="utf-8")
    
    # 查找 ![] (path) 模式
    # 我们之前的发现是 ![](_page_x_Picture_y.jpeg)
    img_pattern = r"!\[\]\((.*?)\)"
    matches = re.findall(img_pattern, content)
    
    if not matches:
        return False

    modified = False
    new_content = content
    
    for img_rel_path in matches:
        # 检查是否已经描述过
        if f"> [AI Description for {img_rel_path}]:" in new_content:
            continue
            
        img_abs_path = md_path.parent / img_rel_path
        if not img_abs_path.exists():
            print(f"Warning: Image {img_abs_path} not found.")
            continue
            
        print(f"Describing: {img_abs_path.name} in {md_path.name}")
        
        try:
            b64_image = encode_image(img_abs_path)
            prompt = (
                "Please provide a technical and detailed description of this chart, table, or figure. "
                "Include key data points, trends, and the core purpose of this visual. "
                "Ensure the description is concise but informative for a RAG indexing system."
            )
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                ]
            )
            
            response = llm.invoke([message])
            description = response.content.strip()
            
            # 在图片引用下方插入描述
            insertion = f"\n> [AI Description for {img_rel_path}]: {description}\n"
            search_str = f"![]({img_rel_path})"
            new_content = new_content.replace(search_str, search_str + insertion)
            modified = True
            
        except Exception as e:
            print(f"Error describing {img_rel_path}: {e}")
            
    if modified:
        md_path.write_text(new_content, encoding="utf-8")
        return True
    return False

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment.")
        return
        
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    
    rag_root = Path(r"c:\Users\a2231\Desktop\MM_Agent_Project\rag_service\RAG_Output_Gemini")
    
    md_files = list(rag_root.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files.")
    
    success_count = 0
    for md_file in md_files:
        if describe_images_in_md(md_file, llm):
            success_count += 1
            
    print(f"Done! Updated {success_count} files.")

if __name__ == "__main__":
    main()
