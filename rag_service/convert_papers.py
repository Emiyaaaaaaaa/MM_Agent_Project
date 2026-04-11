import subprocess
import os
from pathlib import Path
import re
from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

def setup_rag_metadata(md_path, year, question, filename, category):
    """
    Injects YAML frontmatter, refines metadata, and removes COMAP watermarks.
    Now supports three categories: Award, Problem, and Guideline.
    """
    if not md_path.exists():
        return
        
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Strip existing YAML frontmatter if present (for migration)
    if content.startswith("---"):
        end_meta = content.find("---", 3)
        if end_meta != -1:
            content = content[end_meta+3:].lstrip()

    # 2. Strip existing custom headers (for migration)
    content = re.sub(r"^# COMAP MCM/ICM.*?\n\*.*?\*\n\n", "", content, flags=re.DOTALL)

    # 3. Clean COMAP specific ad watermarks
    patterns = [r"关注", r"数学模型", r"获取", r"更多资讯", r"更多资源", r"★取", r"庆取"]
    for p in patterns:
        regex = r"\s*".join([re.escape(char) + r"(?:\s*<[^>]+>\s*)*" for char in p])
        content = re.sub(regex, "", content)
    content = re.sub(r"[\u4e00-\u9fff]+", "", content)
    for _ in range(2):
        content = re.sub(r"<([a-z1-6]+)[^>]*>\s*</\1>", "", content)

    # 4. Map category to source description
    source_map = {
        "O": "Outstanding Award Paper",
        "F": "Finalist Award Paper",
        "Problem": "Official Problem Description",
        "Guideline": "Contest Rules and Guidelines"
    }
    source_type = source_map.get(category, "Document")
    
    header_title = f"COMAP MCM/ICM Mathematical Contest in Modeling - {year}"
    if question and question != "N/A":
        header_title += f" Question {question}"
    
    # Description logic
    if category in ["O", "F"]:
        description = f"This document is an {source_type} from the {year} COMAP MCM/ICM."
    elif category == "Problem":
        description = f"This is the official description for {header_title.split('-')[-1].strip()}."
    else:
        description = f"This is an official contest rule or guideline document for the {year} COMAP MCM/ICM."

    # 5. Build metadata block
    metadata_block = f"""---
source: "COMAP MCM/ICM {source_type}"
year: {year}
question: "{question if (question and question != 'N/A') else 'N/A'}"
filename: "{filename}"
category: "{category}"
---

# {header_title}
*{description}*

"""
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(metadata_block + content.strip())

def migrate_existing_metadata(output_base_dir):
    """
    1. Moves old style directories [Year]/[Question] to O/[Year]/[Question]
    2. Consolidates everything into the new structure
    3. Re-applies metadata injection
    """
    print("Starting directory migration and metadata alignment...")
    
    # --- PHASE 1: Move Directories (Year -> O/Year) ---
    # Old structure: RAG_Output_Gemini/{year}/{question}
    # New structure: RAG_Output_Gemini/O/{year}/{question}
    for year_dir in output_base_dir.iterdir():
        if year_dir.is_dir() and year_dir.name.isdigit():
            target_award_root = output_base_dir / "O"
            target_award_root.mkdir(exist_ok=True)
            
            target_year_dir = target_award_root / year_dir.name
            print(f"Moving {year_dir.name} to O/{year_dir.name}...")
            
            # If target exists, merge contents; if not, rename/move
            if target_year_dir.exists():
                for q_dir in year_dir.iterdir():
                    if q_dir.is_dir():
                        dest_q = target_year_dir / q_dir.name
                        if dest_q.exists():
                             # If question dir already exists in "O", move individual files
                             for sub in q_dir.iterdir():
                                 new_sub = dest_q / sub.name
                                 if not new_sub.exists():
                                     sub.rename(new_sub)
                        else:
                            q_dir.rename(dest_q)
                # Clean up empty year dir
                try: year_dir.rmdir() 
                except: pass
            else:
                year_dir.rename(target_year_dir)

    # --- PHASE 2: Refresh Metadata ---
    # Award Papers (O and F)
    for award in ["O", "F"]:
        award_root = output_base_dir / award
        if not award_root.exists(): continue
        for yr_dir in award_root.iterdir():
            if not yr_dir.is_dir() or not yr_dir.name.isdigit(): continue
            for q_dir in yr_dir.iterdir():
                if not q_dir.is_dir(): continue
                for paper_dir in q_dir.iterdir():
                    if not paper_dir.is_dir(): continue
                    md_file = paper_dir / f"{paper_dir.name}.md"
                    if md_file.exists():
                        setup_rag_metadata(md_file, yr_dir.name, q_dir.name, f"{paper_dir.name}.pdf", award)

    # Problems and Guidelines
    prob_root = output_base_dir / "problems and standard"
    if prob_root.exists():
        for yr_dir in prob_root.iterdir():
            if not yr_dir.is_dir() or not yr_dir.name.isdigit(): continue
            for pdf_dir in yr_dir.iterdir():
                if not pdf_dir.is_dir(): continue
                md_file = pdf_dir / f"{pdf_dir.name}.md"
                if md_file.exists():
                    filename = f"{pdf_dir.name}.pdf"
                    category = "Problem" if "Problem" in filename else "Guideline"
                    q_match = re.search(r"Problem_([A-F])", filename)
                    question = q_match.group(1) if q_match else "N/A"
                    setup_rag_metadata(md_file, yr_dir.name, question, filename, category)
    print("Migration and Refresh complete.\n")

def main():
    # --- USER CONFIGURATION ---
    GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
    # --------------------------

    base_dir = Path(r"c:\Users\a2231\Desktop\RAG\COMAP")
    output_base_dir = Path(r"c:\Users\a2231\Desktop\RAG\RAG_Output_Gemini")
    
    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist.")
        return

    # Trigger migration always to fix structure
    if output_base_dir.exists():
        migrate_existing_metadata(output_base_dir)

    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    
    pdf_tasks = []

    # 1. Discover Papers: COMAP/O or F / [Year] -> [Optional Question] / [PDF]
    for award in ["O", "F"]:
        award_dir = base_dir / award
        if not award_dir.exists(): continue
        
        for year_dir in award_dir.iterdir():
            if not year_dir.is_dir() or not year_dir.name.isdigit(): continue
            # Recursive scan for PDFs in award/year folders
            for pdf_path in year_dir.rglob("*.pdf"):
                # Determine question: if parent folder is A-F, use it, else N/A
                parent_name = pdf_path.parent.name
                question = parent_name if parent_name in ["A", "B", "C", "D", "E", "F"] else "N/A"
                
                # Dynamic output path: Mirror parent folders back to Year level
                rel_out = Path(award) / year_dir.name / pdf_path.relative_to(year_dir).parent
                
                pdf_tasks.append({
                    "path": pdf_path,
                    "year": year_dir.name,
                    "question": question,
                    "category": award,
                    "rel_out": rel_out
                })

    # 2. Discover Problems/Guidelines: COMAP/problems and standard / [Year] (Recursive for Data Addendums)
    prob_dir = base_dir / "problems and standard"
    if prob_dir.exists():
        for year_dir in prob_dir.iterdir():
            if not year_dir.is_dir() or not year_dir.name.isdigit(): continue
            # rglob to catch PDFs inside data folders (like 2024_Problem_D_Addendum.pdf)
            for pdf_path in year_dir.rglob("*.pdf"):
                filename = pdf_path.name
                category = "Problem" if "Problem" in filename else "Guideline"
                q_match = re.search(r"Problem_([A-F])", filename)
                question = q_match.group(1) if q_match else "N/A"
                
                # Mirror any subfolder structure (like Data folders) inside the output
                rel_out = Path("problems and standard") / year_dir.name / pdf_path.relative_to(year_dir).parent
                
                pdf_tasks.append({
                    "path": pdf_path,
                    "year": year_dir.name,
                    "question": question,
                    "category": category,
                    "rel_out": rel_out
                })

    total_files = len(pdf_tasks)
    print(f"Found {total_files} PDFs to process.\n")
    
    for i, task in enumerate(pdf_tasks, 1):
        pdf_path = task["path"]
        q_out_dir = output_base_dir / task["rel_out"]
        q_out_dir.mkdir(parents=True, exist_ok=True)
        
        md_file = q_out_dir / pdf_path.stem / f"{pdf_path.stem}.md"

        if md_file.exists():
            print(f"[{i}/{total_files}] Skipping {pdf_path.name} (Exists - Metadata Refreshed)")
            continue

        print(f"[{i}/{total_files}] Converting {pdf_path.name}...")
        cmd = [
            "marker_single", 
            str(pdf_path), 
            "--output_dir", str(q_out_dir), 
            "--use_llm"
        ]
        
        try:
            subprocess.run(cmd, check=True, text=True, shell=True)
            if md_file.exists():
                setup_rag_metadata(md_file, task["year"], task["question"], pdf_path.name, task["category"])
                print(f"   Success: {pdf_path.name}")
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    main()
