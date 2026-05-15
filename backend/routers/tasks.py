from typing import List
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from backend.db import crud, database
from backend.schemas.api import TaskCreate, TaskResponse
from backend.services.checkpoint_store import clear_thread_checkpoints


router = APIRouter(prefix="/api/v1", tags=["tasks"])


@router.post("/tasks", response_model=TaskResponse)
def create_task(task: TaskCreate, db: Session = Depends(database.get_db)):
    db_user = crud.get_user(db, user_id=task.user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    created = crud.create_task(
        db,
        user_id=task.user_id,
        title=task.title,
        api_key=task.api_key,
        model_id=task.model_id or "gemini-2.5-flash-lite",
    )
    if not created:
        raise HTTPException(status_code=400, detail="Task already exists")
    return created


@router.get("/users/{user_id}/tasks", response_model=List[TaskResponse])
def get_user_tasks(user_id: str, db: Session = Depends(database.get_db)):
    return crud.get_user_tasks(db, user_id=user_id)


@router.delete("/users/{user_id}/tasks/{task_id}")
def delete_task(user_id: str, task_id: str, db: Session = Depends(database.get_db)):
    ok = crud.delete_task_for_user(db, user_id=user_id, task_id=task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "success", "message": "Task deleted"}


@router.get("/tasks/{task_id}/messages")
def get_task_messages(task_id: str, db: Session = Depends(database.get_db)):
    task = crud.get_task(db, task_id=task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return [crud.serialize_message(item) for item in crud.get_task_messages(db, task_id)]


@router.post("/users/{user_id}/tasks/{task_id}/restart")
def restart_task(
    user_id: str,
    task_id: str,
    request: Request,
    db: Session = Depends(database.get_db),
):
    task = crud.get_task(db, task_id=task_id)
    if not task or task.user_id != user_id:
        raise HTTPException(status_code=404, detail="Task not found")

    cleared_messages = crud.clear_task_messages(db, task_id)

    runtime = getattr(request.app.state, "graph_runtime", None)
    checkpoint_path = getattr(runtime, "checkpoint_path", None)
    if not checkpoint_path:
        raise HTTPException(status_code=503, detail="Graph runtime not initialized")

    cleared_checkpoints = clear_thread_checkpoints(checkpoint_path, task_id)
    return {
        "status": "success",
        "message": "Task restarted",
        "cleared_messages": cleared_messages,
        "cleared_checkpoints": cleared_checkpoints,
    }
