import os
from dotenv import load_dotenv
# 显式加载根目录的 .env
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(base_dir, "..", ".env"))

from langchain_core.messages import HumanMessage
from backend.graph.nodes.writer import writer_node

# 验证 Key
if not os.environ.get("GOOGLE_API_KEY"):
    print("❌ [错误] GOOGLE_API_KEY 未找到，请检查 .env 文件。")
else:
    print(f"✅ GOOGLE_API_KEY 已检测到 (长度: {len(os.environ.get('GOOGLE_API_KEY'))})")

def test_passive_writer():
    print(">>> 正在启动『被动式超限处理』专项测试...")
    
    # 模拟初始状态
    state = {
        "messages": [HumanMessage(content="请帮我写一篇关于人口动态模型的论文全文。")],
        "shared_memory": {
            "analysis_report": "研究重点：资源受限下的人口崩溃模型。",
            "mathematical_model": "P(t+1) = P(t) + r*P(t)*(1-P(t)/K)",
            "review_report": "逻辑自洽，图表正常。",
            "paper_draft": "" # 初始为空
        },
        "status": "APPROVED"
    }
    
    # 第一次尝试生成 (模拟触发截断)
    print("\n--- [第一次输出尝试] ---")
    result_1 = writer_node(state)
    
    draft_1 = result_1["shared_memory"].get("paper_draft", "")
    status_1 = result_1["status"]
    print(f"✅ 生成长度: {len(draft_1)} 字符")
    print(f"✅ 节点状态: {status_1} (如果是 PENDING 说明检测到了截断)")

    # 模拟用户指示“继续”
    if status_1 == "PENDING":
        print("\n--- [第二次续写尝试] ---")
        state_2 = state.copy()
        state_2["shared_memory"] = result_1["shared_memory"]
        state_2["messages"].append(HumanMessage(content="继续"))
        
        result_2 = writer_node(state_2)
        draft_2 = result_2["shared_memory"].get("paper_draft", "")
        print(f"✅ 续写后总长度: {len(draft_2)} 字符")
        if len(draft_2) > len(draft_1):
            print("✅ [成功] 论文已实现顺滑续写。")
            
    print("\n✅ 被动式超限处理测试完成！")

if __name__ == "__main__":
    test_passive_writer()
