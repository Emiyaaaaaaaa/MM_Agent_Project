from fastapi.testclient import TestClient
from backend.main import app
from backend.db import crud, database


client = TestClient(app)


def test_auth_task_and_message_contract():
    register_res = client.post(
        "/api/v1/auth/register",
        json={"email": "contract_user@example.com", "password": "demo123456"},
    )
    assert register_res.status_code in (200, 409)

    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": "contract_user@example.com", "password": "demo123456"},
    )
    assert login_res.status_code == 200
    user = login_res.json()
    assert "id" in user

    task_res = client.post(
        "/api/v1/tasks",
        json={
            "user_id": user["id"],
            "title": "contract_test_task",
            "api_key": "fake_key_for_test",
            "model_id": "gemini-2.5-flash-lite",
        },
    )
    assert task_res.status_code in (200, 400)
    if task_res.status_code == 200:
        task_id = task_res.json()["id"]
    else:
        task_id = f"{user['id'][:6]}_contract_test_task"

    with database.SessionLocal() as db:
        crud.create_message(db, task_id=task_id, role="user", content="hello")
        crud.create_message(db, task_id=task_id, role="ai", content="world")

    msg_res = client.get(f"/api/v1/tasks/{task_id}/messages")
    assert msg_res.status_code == 200
    payload = msg_res.json()
    assert isinstance(payload, list)
    assert payload[-1]["role"] in ("user", "ai")
    assert isinstance(payload[-1]["content"], str)
