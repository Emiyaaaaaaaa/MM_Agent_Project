import requests
import json

BASE_URL = "http://localhost:8000"

def test_full_flow():
    print("=== 启动端到端业务流测试 ===")
    
    # 1. 注册/登录用户 (基于 Email)
    email = "researcher_001@example.com"
    user_data = {"email": email, "username": "建模领航员"}
    response = requests.post(f"{BASE_URL}/api/v1/users", json=user_data)
    if response.status_code != 200:
        print(f"[ERROR] 用户注册失败 (Status: {response.status_code}): {response.text}")
        return
    user = response.json()
    user_id = user["id"]
    print(f"[Step 1] 用户注册成功: {user['username']} (ID: {user_id})")

    # 2. 创建一个特定建模任务 (指定模型与 API Key)
    # 这里的 api_key 设置为 None 则后端会回落至环境变量
    task_data = {
        "user_id": user_id,
        "title": "2026年MCM A题仿真任务",
        "api_key": None, 
        "model_id": "gemini-2.5-flash"
    }
    response = requests.post(f"{BASE_URL}/api/v1/tasks", json=task_data)
    task = response.json()
    task_id = task["id"]
    print(f"[Step 2] 任务创建成功: {task['title']} (Task/Thread ID: {task_id})")
    print(f"        绑定模型: {task['model_id']}")

    # 3. 获取用户任务列表校验
    response = requests.get(f"{BASE_URL}/api/v1/users/{user_id}/tasks")
    tasks = response.json()
    print(f"[Step 3] 用户当前拥有任务数: {len(tasks)}")
    
    print("\n" + "="*50)
    print(f"验证通过！您现在可以使用以下 Task ID 运行 WebSocket 客户端：")
    print(f"Task ID: {task_id}")
    print("="*50)

if __name__ == "__main__":
    # 请确保后端服务已启动: poetry run python -m backend.main
    try:
        test_full_flow()
    except Exception as e:
        print(f"测试失败: {e}")
