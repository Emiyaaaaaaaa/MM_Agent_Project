import subprocess
import tempfile
import os
import sys
import re
from typing import Any, Dict, Optional, Set

_ALLOWED_MODULE_PACKAGE_MAP: Dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "pulp": "pulp",
}
_INSTALLED_CACHE: Set[str] = set()


def _extract_missing_module(stderr_text: str) -> Optional[str]:
    if not stderr_text:
        return None
    match = re.search(r"No module named ['\"]([a-zA-Z0-9_\.]+)['\"]", stderr_text)
    if not match:
        return None
    return match.group(1).split(".")[0]


def _install_allowed_package(module_name: str, timeout: int = 60) -> Dict[str, Any]:
    package = _ALLOWED_MODULE_PACKAGE_MAP.get(module_name)
    if not package:
        return {"installed": False, "reason": f"module_not_allowed:{module_name}", "stdout": "", "stderr": ""}
    if package in _INSTALLED_CACHE:
        return {"installed": True, "reason": "cache_hit", "stdout": "", "stderr": ""}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            _INSTALLED_CACHE.add(package)
            return {"installed": True, "reason": "installed", "stdout": result.stdout, "stderr": result.stderr}
        return {"installed": False, "reason": "pip_failed", "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired as exc:
        return {
            "installed": False,
            "reason": "pip_timeout",
            "stdout": exc.stdout.decode("utf-8") if exc.stdout else "",
            "stderr": exc.stderr.decode("utf-8") if exc.stderr else "",
        }
    except Exception as exc:
        return {"installed": False, "reason": f"pip_exception:{type(exc).__name__}", "stdout": "", "stderr": str(exc)}


def execute_python_code(code: str, timeout: int = 30, auto_install_missing: bool = True, _retry_on_install: bool = True) -> dict:
    """
    在一个临时的隔离目录中执行生成好的 Python 代码。
    返回标准输出、标准错误和执行状态。
    """
    # 如果代码内容为空
    if not code.strip():
        return {"stdout": "", "stderr": "No code provided.", "success": False, "auto_install": None}
        
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
            response = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": success,
                "auto_install": None,
            }
            if (not success) and auto_install_missing and _retry_on_install:
                missing_module = _extract_missing_module(result.stderr or "")
                if missing_module:
                    install_info = _install_allowed_package(missing_module)
                    response["auto_install"] = {
                        "missing_module": missing_module,
                        "package": _ALLOWED_MODULE_PACKAGE_MAP.get(missing_module),
                        **install_info,
                    }
                    if install_info.get("installed"):
                        retry_result = execute_python_code(
                            code=code,
                            timeout=timeout,
                            auto_install_missing=auto_install_missing,
                            _retry_on_install=False,
                        )
                        retry_result["auto_install"] = response["auto_install"]
                        return retry_result
            return response
        except subprocess.TimeoutExpired as e:
            return {
                "stdout": e.stdout.decode('utf-8') if e.stdout else "",
                "stderr": f"Execution timed out after {timeout} seconds.",
                "success": False,
                "auto_install": None,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"Sandbox Exception: {str(e)}",
                "success": False,
                "auto_install": None,
            }
