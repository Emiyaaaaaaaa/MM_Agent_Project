import os
import zipfile
import shutil
from backend.graph.state import AgentState
from backend.services.bus import broadcast_progress

class ExporterNode:
    # ... (header same)
    def __init__(self):
        self.base_dir = os.path.abspath(os.getcwd())
        self.export_dir = os.path.join(self.base_dir, "backend", "static", "exports")
        self.plots_dir = os.path.join(self.base_dir, "backend", "static", "plots")
        os.makedirs(self.export_dir, exist_ok=True)

    def _create_latex_wrapper(self, content):
        """学术 LaTeX 模板封装"""
        template = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath, amssymb}
\usepackage{graphicx}

\title{MCM/ICM Research Paper}
\author{MM-Agent Digital Orchestration}
\date{\today}

\begin{document}
\maketitle
"""
        template += content
        template += r"\n\end{document}"
        return template

    async def __call__(self, state: AgentState):
        """执行物理导出流程"""
        shared_mem = state.get("shared_memory", {})
        draft = shared_mem.get("paper_draft", "暂无内容。")
        
        # 1. 广播进度：开始文件转换
        await broadcast_progress("Exporter", "正在生成 Markdown 与 LaTeX 物理文件...", 20)
        
        md_path = os.path.join(self.export_dir, "thesis.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(draft)
        
        tex_path = os.path.join(self.export_dir, "thesis.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(self._create_latex_wrapper(draft))
            
        # 2. 广播进度：开始全量打包
        await broadcast_progress("Exporter", "正在打包全量资产 (ZIP)...", 60)
        
        zip_path = os.path.join(self.export_dir, "final_submission.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(md_path, arcname="thesis.md")
            zipf.write(tex_path, arcname="thesis.tex")
            
            # 添加生成的图表素材
            plot_file = os.path.join(self.plots_dir, "output.png")
            if os.path.exists(plot_file):
                zipf.write(plot_file, arcname="plots/output.png")
            
        await broadcast_progress("Exporter", "资产固化完成，ZIP 已就绪。", 100)
        
        report = (
            "✅ 成果物理导出成功！您可以直接下载以下资源：\n"
            f"- [Markdown 论文原文](/static/exports/thesis.md)\n"
            f"- [LaTeX 专业版本](/static/exports/thesis.tex)\n"
            f"- [全量压缩包 (含图表)](/static/exports/final_submission.zip)\n"
        )

        return {
            "status": "APPROVED",
            "draft": report,
            "current_stage": "Physical Asset固化完成"
        }

exporter_node = ExporterNode()
