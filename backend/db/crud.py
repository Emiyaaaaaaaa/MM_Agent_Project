import base64
import hashlib
import re
import json
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet, InvalidToken
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from . import models
from backend.config import get_secret_key
from backend.services.serialization import ensure_blocks, extract_text_content

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _derive_secret_key() -> bytes:
    # 用统一 secret 派生稳定密钥，避免 API Key 明文入库
    raw = get_secret_key()
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


_FERNET = Fernet(_derive_secret_key())


def encrypt_api_key(api_key: Optional[str]) -> Optional[str]:
    if not api_key:
        return None
    return _FERNET.encrypt(api_key.encode("utf-8")).decode("utf-8")


def decrypt_api_key(encrypted_api_key: Optional[str]) -> Optional[str]:
    if not encrypted_api_key:
        return None
    try:
        return _FERNET.decrypt(encrypted_api_key.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        # 兼容历史明文数据，后续更新任务时会重新加密
        return encrypted_api_key


def get_password_hash(password: str):
    pwd_bytes = password.encode("utf-8")
    h = hashlib.sha256(pwd_bytes).hexdigest()
    return pwd_context.hash(h)


def verify_password(plain_password: str, hashed_password: str):
    pwd_bytes = plain_password.encode("utf-8")
    h = hashlib.sha256(pwd_bytes).hexdigest()
    return pwd_context.verify(h, hashed_password)


def get_user(db: Session, user_id: str):
    return db.query(models.User).filter(models.User.id == user_id, models.User.is_deleted.is_(False)).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email, models.User.is_deleted.is_(False)).first()


def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_user(db: Session, email: str, password: str, username: str = None):
    hashed_pwd = get_password_hash(password)
    db_user = models.User(email=email, hashed_password=hashed_pwd, username=username)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def create_task(db: Session, user_id: str, title: str, api_key: str = None, model_id: str = "gemini-3.1-pro-preview"):
    safe_title = re.sub(r"[^a-zA-Z0-9_]", "_", title.lower()).strip("_")
    if not safe_title:
        safe_title = "unnamed_task"
    safe_title = safe_title[:40]
    user_short = user_id[:6]
    final_id = f"{user_short}_{safe_title}"

    existing = get_task(db, final_id)
    if existing:
        return None

    db_task = models.Task(
        id=final_id,
        user_id=user_id,
        title=title,
        api_key=encrypt_api_key(api_key),
        model_id=model_id,
        status="active",
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task


def get_user_tasks(db: Session, user_id: str):
    return db.query(models.Task).filter(models.Task.user_id == user_id, models.Task.is_deleted.is_(False)).all()


def get_task(db: Session, task_id: str):
    return db.query(models.Task).filter(models.Task.id == task_id, models.Task.is_deleted.is_(False)).first()


def delete_task_for_user(db: Session, user_id: str, task_id: str) -> bool:
    task = db.query(models.Task).filter(
        models.Task.id == task_id,
        models.Task.user_id == user_id,
        models.Task.is_deleted.is_(False),
    ).first()
    if not task:
        return False
    task.is_deleted = True
    task.deleted_at = datetime.utcnow()
    task.updated_at = datetime.utcnow()
    db.query(models.Message).filter(
        models.Message.task_id == task_id,
        models.Message.is_deleted.is_(False),
    ).update(
        {
            models.Message.is_deleted: True,
            models.Message.deleted_at: datetime.utcnow(),
            models.Message.updated_at: datetime.utcnow(),
        },
        synchronize_session=False,
    )
    db.commit()
    return True


def get_task_runtime_config(db: Session, task_id: str) -> Optional[Dict[str, str]]:
    task = get_task(db, task_id)
    if not task:
        return None
    return {
        "api_key": decrypt_api_key(task.api_key),
        "model_id": task.model_id,
    }


def update_task(db: Session, task_id: str, **kwargs):
    db_task = get_task(db, task_id)
    if db_task:
        for key, value in kwargs.items():
            if key == "api_key":
                value = encrypt_api_key(value)
            setattr(db_task, key, value)
        db_task.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_task)
    return db_task


def get_task_messages(db: Session, task_id: str):
    return db.query(models.Message).filter(
        models.Message.task_id == task_id,
        models.Message.is_deleted.is_(False),
    ).order_by(models.Message.created_at.asc()).all()


def clear_task_messages(db: Session, task_id: str) -> int:
    deleted = db.query(models.Message).filter(
        models.Message.task_id == task_id,
        models.Message.is_deleted.is_(False),
    ).update(
        {
            models.Message.is_deleted: True,
            models.Message.deleted_at: datetime.utcnow(),
            models.Message.updated_at: datetime.utcnow(),
        },
        synchronize_session=False,
    )
    db.commit()
    return deleted


def create_message(db: Session, task_id: str, role: str, content: Any, node: str = None):
    content_blocks = ensure_blocks(content)
    db_msg = models.Message(
        id=str(uuid.uuid4()),
        task_id=task_id,
        role=role,
        content=json.dumps(content_blocks, ensure_ascii=False),
        node=node,
        is_deleted=False,
    )
    db.add(db_msg)
    db.commit()
    db.refresh(db_msg)
    return db_msg


def serialize_message(msg: models.Message) -> Dict[str, Any]:
    parsed_blocks = ensure_blocks(msg.content)
    try:
        parsed_blocks = ensure_blocks(json.loads(msg.content or "[]"))
    except Exception:
        parsed_blocks = ensure_blocks(msg.content)
    return {
        "id": msg.id,
        "task_id": msg.task_id,
        "role": msg.role,
        "content_blocks": parsed_blocks,
        "content_text": extract_text_content(parsed_blocks),
        "node": msg.node,
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
    }
