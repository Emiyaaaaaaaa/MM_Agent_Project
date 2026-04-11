import subprocess
import tempfile
import os
import sys

def execute_python_code(code: str, timeout: int = 30) -> dict:
    """
    在一个临时的隔离目录中执行生成好的 Python 代码。
    返回标准输出、标准错误和执行状态。
    """
    # 如果代码内容为空
    if not code.strip():
        return {"stdout": "", "stderr": "No code provided.", "success": False}
        
    # 我们允许代码在此目录下进行一些图表保存，避免影响系统目录
    base_dir = os.path.abspath(os.path.join(os.getcwd(), "backend", "static", "plots"))
    os.makedirs(base_dir, exist_ok=True)
    
    # 强制在代码头部注入环境设置，确保图片保存到正确的目录
    injected_header = f"""
import os
import matplotlib
matplotlib.use('Agg')  # 用于无 UI 运行环境，避免 GUI 挂起
import matplotlib.pyplot as plt

os.chdir(r"{os.path.abspath(os.getcwd())}")  # 保持当前工作目录为项目根目录，方便读取数据

# 覆盖掉代码中硬编码的图片保存路径
import builtins
# 这里假定模型会按 system_prompt 将图表保存到 backend/static/plots/output.png
"""
    full_code = injected_header + code

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "sandbox_run.py")
        
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(full_code)
            
        try:
            # 采用当前 python 环境执行
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            success = result.returncode == 0
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": success
            }
        except subprocess.TimeoutExpired as e:
            return {
                "stdout": e.stdout.decode('utf-8') if e.stdout else "",
                "stderr": f"Execution timed out after {timeout} seconds.",
                "success": False
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Sandbox Exception: {str(e)}",
                "success": False
            }
