import subprocess
import os
import sys
from pathlib import Path

def setup_rag_metadata(md_path, year, question, filename):
    if not md_path.exists():
        return
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 避免重复写入 metadata
    if content.startswith("---"):
        return
    metadata_block = f"---\nsource: \"2024 MCM/ICM Outstanding Paper\"\nyear: {year}\nquestion: {question}\nfilename: {filename}\n---\n\n# 2024 MCM/ICM Mathematical Contest in Modeling - Question {question}\n*This document is an outstanding (O-Award) paper from the 2024 MCM/ICM.*\n\n"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(metadata_block + content)

def main():
    GEMINI_API_KEY = "AIzaSyDm3_D6Re05IiOj27SflfnN1vKfTaZ8Jq0"
    base_dir = Path("/content/drive/MyDrive/marker_project/paper")
    output_base_dir = Path("/content/drive/MyDrive/marker_project/output")

    output_base_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    questions = ['A', 'B', 'C', 'D', 'E', 'F']
    all_tasks = []
    for q in questions:
        q_dir = base_dir / q
        if q_dir.exists():
            for p in q_dir.glob("*.pdf"): 
                all_tasks.append((p, q))

    print(f"Found {len(all_tasks)} PDFs. Starting processing (Resume mode supported)...\n", flush=True)

    for i, (pdf_path, q) in enumerate(all_tasks, 1):
        pdf_name_stem = pdf_path.stem
        q_out_dir = output_base_dir / q
        expected_md_file = q_out_dir / pdf_name_stem / f"{pdf_name_stem}.md"

        # --- 断点续传逻辑: 如果文件已存在则跳过 ---
        if expected_md_file.exists():
            print(f"[{i}/{len(all_tasks)}] Skipping {pdf_path.name} (Already processed)", flush=True)
            continue

        print(f"[{i}/{len(all_tasks)}] Processing {pdf_path.name}...", flush=True)
        q_out_dir.mkdir(parents=True, exist_ok=True)
        
        # 尝试多种命令格式以确保兼容性
        cmd = ["marker_single", str(pdf_path), "--output_dir", str(q_out_dir), "--use_llm"]
        
        try:
            # 使用 subprocess.run 并捕获输出，以便在“恢复执行”时能看到最后发生了什么
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                if expected_md_file.exists():
                    setup_rag_metadata(expected_md_file, 2024, q, pdf_path.name)
                    print(f"  ✅ Success: {pdf_name_stem}", flush=True)
            else:
                print(f"  ❌ Failed: {pdf_path.name}\nError log: {result.stderr}", flush=True)
        except Exception as e:
            print(f"  ⚠️ Exception during {pdf_path.name}: {e}", flush=True)

if __name__ == "__main__":
    main()