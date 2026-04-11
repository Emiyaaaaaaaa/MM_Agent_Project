from sqlalchemy.orm import Session
from . import models
import uuid

def get_user(db: Session, user_id: str):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, email: str, username: str = None):
    db_user = models.User(email=email, username=username)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def create_task(db: Session, user_id: str, title: str, api_key: str = None, model_id: str = "gemini-2.5-flash"):
    db_task = models.Task(
        id=str(uuid.uuid4()),
        user_id=user_id,
        title=title,
        api_key=api_key,
        model_id=model_id
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

def get_user_tasks(db: Session, user_id: str):
    return db.query(models.Task).filter(models.Task.user_id == user_id).all()

def get_task(db: Session, task_id: str):
    return db.query(models.Task).filter(models.Task.id == task_id).first()

def update_task(db: Session, task_id: str, **kwargs):
    db_task = get_task(db, task_id)
    if db_task:
        for key, value in kwargs.items():
            setattr(db_task, key, value)
        db.commit()
        db.refresh(db_task)
    return db_task
