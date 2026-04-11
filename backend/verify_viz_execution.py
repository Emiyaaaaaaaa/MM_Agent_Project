import os
import subprocess
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from backend.graph.builder import mcm_graph

load_dotenv()

def execute_generated_code():
    print(">>> 正在调用『程序员』节点并执行生成的代码...")
    
    config = {"configurable": {"thread_id": "real_execution_thread"}}
    
    # 构造一个具体的绘图任务
    initial_input = {
        "messages": [HumanMessage(content="请帮我建立一个 Logistic 人口增长模型，预测未来 20 年的人口趋势，并绘制精美的折线图。数据模拟如下：当前人口 1000 万，K 值 5000 万，r=0.1。")],
        "shared_memory": {"analysis": "2025 MCM F 题环境承载力分析。"}
    }
    
    # 1. 获取 Coder 节点生成的代码
    code = ""
    for event in mcm_graph.stream(initial_input, config, stream_mode="values"):
        if event.get("draft"):
            code = event.get("draft")
            break
            
    if not code:
        print("❌ 未能获取到生成的代码。")
        return

    # 2. 深度清理逻辑：使用正则精准提取 python 代码块
    import re
    code_match = re.search(r"```python\n(.*?)\n```", code, re.DOTALL)
    if code_match:
        clean_code = code_match.group(1)
    else:
        # 如果没搜到标记，则尝试直接清理掉可能的开头结尾
        clean_code = code.replace("```python", "").replace("```", "").strip()
    
    prefix = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "import os\n"
    )
    # 移除脚本中可能存在的 plt.show() 以免干扰
    executable_code = prefix + clean_code.replace("plt.show()", "# plt.show()")
    
    # 3. 将代码写入临时文件并执行
    temp_script = "tmp_viz_run.py"
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(executable_code)
        
    print(f">>> 正在执行生成的脚本: {temp_script}...")
    try:
        # 确保 backend/static/plots 存在
        os.makedirs(os.path.join('backend', 'static', 'plots'), exist_ok=True)
        
        # 使用 sys.executable 确保子进程使用与当前相同的虚拟环境
        import sys
        result = subprocess.run([sys.executable, temp_script], capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ 脚本执行成功！")
            plot_path = os.path.join('backend', 'static', 'plots', 'output.png')
            if os.path.exists(plot_path):
                print(f"✅ 图表已成功生成至: {plot_path}")
            else:
                print("⚠️ 脚本运行成功，但未检测到 output.png，请检查脚本内部保存路径。")
        else:
            print(f"❌ 脚本执行失败:\n{result.stderr}")
    except Exception as e:
        print(f"❌ 执行出错: {e}")
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)

if __name__ == "__main__":
    execute_generated_code()
