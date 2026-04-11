import subprocess
import os
import json
import re
from pathlib import Path
import yaml
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

# --- 配置 (Configuration) ---
BASE_DIR = Path(r"c:\Users\a2231\Desktop\RAG\COMAP")
OUTPUT_BASE_DIR = Path(r"c:\Users\a2231\Desktop\RAG\RAG_Output_Gemini")
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

def setup_commentary_metadata(md_path, filename):
    """
    评委点评元数据注入与深度清洗逻辑
    """
    if not md_path.exists(): return
        
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. 基础清理 (移除水印等中文残留)
    content = re.sub(r"[\u4e00-\u9fff]+", "", content) 
    
    # 2. 深度清洗：跨行块级剔除
    junk_sections = [
        r"About The Author(?:s)?",
        r"Guest Editorial",
        r"Acknowledgments",
        r"Editorial Board",
        r"Editor's Note",
        r"Publisher'S Editorial",
        r"In Memoriam",
        r"Contributors",
        r"About Doug Faires",
        r"Doug Faires Award Winners",
        r"Modeling Remains A Uniquely Human Endeavor", # AI Warning section
        r"Disclaimer"
    ]
    
    for section in junk_sections:
        # 匹配从关键词行开始，直至下一个标题、连续换行符或文件结尾
        pattern = rf"(?ism)^(?:#+|\*\*|\|)?\s*{section}.*?(?=\n#+ |\n\n\n|\Z)"
        content = re.sub(pattern, "", content)

    # 3. 版权信息块清理 (增强匹配)
    content = re.sub(r"(?is)The UMAP Journal.*?Permission to make digital or hard copies.*?from COMAP\.", "", content)
    content = re.sub(r"(?is)©Copyright \d+ by COMAP, Inc\..*?All rights reserved\.", "", content)
    content = re.sub(r"(?im)^.*Photo credit:.*$", "", content) # 清理图片来源文字

    # 4. 页眉页脚/底注/页码清理
    content = re.sub(r"(?im)^Vol\. \d+, No\. \d+ \d+ Table of Contents.*$", "", content)
    content = re.sub(r"(?im)^The UMAP Journal \d+ \(\d+\) \(\d+\) \d+–\d+\..*$", "", content)
    content = re.sub(r"(?m)^.*?\b\d{3}\b\s+Editor's Note:.*$", "", content) # 清理目录条目
    
    # 7. 额外清理：图片占位符
    content = re.sub(r"!\[\d+_(?:image|Image)_\d+\.(?:png|Png|jpg|jpeg)\]", "", content)
    content = re.sub(r"\[\d+_(?:image|Image)_\d+\.(?:png|Png|jpg|jpeg)\]", "", content) # 捕获非标准格式的图片引用

    # 5. 语义提取：自动关联 Team ID
    linked_teams = sorted(list(set(re.findall(r"\b\d{7}\b", content))))

    # 6. 构造元数据
    journal_name = filename.split('.')[0].replace('_', ' ')
    metadata = {
        "source": "COMAP MCM/ICM Judges' Commentary",
        "year": "2025",
        "journal": journal_name,
        "filename": filename,
        "category": "Commentary",
        "linked_teams": linked_teams
    }

    metadata_yaml = yaml.dump(metadata, allow_unicode=True, sort_keys=False)
    
    # 构建最终内容
    header_title = f"Judges' Commentary - {journal_name}"
    description = f"This is a professional commentary from the UMAP Journal, providing expert analysis on the COMAP competition."
    
    # 确保内容不为空
    clean_body = content.strip()
    if not clean_body:
        clean_body = "Content removed due to deep cleaning (irrelevant meta-data)."
        
    final_content = f"---\n{metadata_yaml}---\n\n# {header_title}\n*{description}*\n\n{clean_body}"
    
    # 最终空行压缩
    final_content = re.sub(r"\n{3,}", "\n\n", final_content)
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(final_content)

def main():
    print("启动评委点评转换程序...")
    
    prob_dir = BASE_DIR / "problems and standard"
    if not prob_dir.exists():
        print(f"❌ 错误: 找不到目录 {prob_dir}")
        return

    pdf_files = list(prob_dir.glob("UMAP_*.pdf"))
    print(f"找到 {len(pdf_files)} 个期刊文件等待处理。")

    for pdf_path in pdf_files:
        out_rel_dir = OUTPUT_BASE_DIR / "commentary" / pdf_path.stem
        
        # 结果定位
        md_file = out_rel_dir / pdf_path.stem / f"{pdf_path.stem}.md"
        md_alt = out_rel_dir / f"{pdf_path.stem}.md"

        target_md = None
        if md_file.exists(): target_md = md_file
        elif md_alt.exists(): target_md = md_alt

        # 如果已有 MD，执行强力二次清洗
        if target_md:
            print(f"正在更新现有 Markdown 文件: {target_md.name}")
            setup_commentary_metadata(target_md, pdf_path.name)
            print(f"处理完成: {pdf_path.stem}")
            continue

        print(f"正在转换: {pdf_path.name}")
        cmd = ["marker_single", str(pdf_path), str(out_rel_dir), "--batch_multiplier", "1"]
        try:
            subprocess.run(cmd, check=True, text=True, shell=True)
            if md_file.exists():
                setup_commentary_metadata(md_file, pdf_path.name)
                print(f"处理成功: {pdf_path.stem}")
            elif md_alt.exists():
                setup_commentary_metadata(md_alt, pdf_path.name)
                print(f"处理成功: {pdf_path.stem}")
        except Exception as e:
            print(f"❌ 处理失败: {e}")

if __name__ == "__main__":
    main()
