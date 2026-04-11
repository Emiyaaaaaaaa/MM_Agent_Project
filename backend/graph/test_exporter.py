import os
from backend.graph.nodes.exporter import exporter_node

def test_exporter_node():
    print(">>> 正在启动『物理导出』专项功能测试...")
    
    # 获取根目录
    base_dir = os.path.abspath(os.getcwd())
    export_dir = os.path.join(base_dir, "backend", "static", "exports")
    plots_dir = os.path.join(base_dir, "backend", "static", "plots")
    
    # 模拟输入状态
    state = {
        "messages": [],
        "shared_memory": {
            "paper_draft": "# 这是一个测试论文正文\n\n## 1. 建模背景\n由于资源枯竭导致的崩溃模型...",
            "analysis_report": "测试背景..."
        }
    }
    
    # 确保有一个 mock 图片进行 zip 测试
    os.makedirs(plots_dir, exist_ok=True)
    mock_plot_path = os.path.join(plots_dir, "output.png")
    if not os.path.exists(mock_plot_path):
        with open(mock_plot_path, "wb") as f:
            f.write(b"MOCK IMAGE DATA")
        print("  - [Debug] 已创建 mock 图片进行 ZIP 测试。")

    # 执行导出
    result = exporter_node(state)
    
    # 验证文件是否存在
    files_to_check = ["thesis.md", "thesis.tex", "final_submission.zip"]
    success = True
    for f in files_to_check:
        full_path = os.path.join(export_dir, f)
        if os.path.exists(full_path):
            print(f"✅ 文件存在: {f} (大小: {os.path.getsize(full_path)} bytes)")
        else:
            print(f"❌ 文件缺失: {f}")
            success = False
            
    if success:
        print("\n✅ 所有格式物理导出成功！")
        print(f"消息回显:\n{result['draft']}")
    else:
        print("\n❌ 物理导出部分失败。")

if __name__ == "__main__":
    test_exporter_node()
