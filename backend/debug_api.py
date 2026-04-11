import sys
import os
# 确保可以找到 backend 模块
sys.path.append(os.getcwd())

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def debug_user_creation():
    print("=== 开始原地 API 调试 ===")
    try:
        response = client.post("/api/v1/users", json={
            "email": "debug_test@example.com",
            "username": "Debugger"
        })
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.json()}")
    except Exception as e:
        print(f"Caught Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_user_creation()
