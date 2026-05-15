from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from sqlalchemy import text

# 数据库文件路径 (重命名以强制重置表结构)
SQLALCHEMY_DATABASE_URL = "sqlite:///./mcm_agent.db"

# 创建引擎
# check_same_thread=False 仅适用于 SQLite
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 基类，供各模型继承
Base = declarative_base()


def ensure_sqlite_compat_columns() -> None:
    """
    兼容迁移：为现有 SQLite 库补齐标准字段，避免老库启动时报 no such column。
    仅做增量补列，不做破坏性变更。
    """
    if not SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
        return

    table_columns = {
        "users": {
            "updated_at": "DATETIME",
            "is_deleted": "BOOLEAN DEFAULT 0",
            "deleted_at": "DATETIME",
        },
        "tasks": {
            "updated_at": "DATETIME",
            "is_deleted": "BOOLEAN DEFAULT 0",
            "deleted_at": "DATETIME",
        },
        "messages": {
            "updated_at": "DATETIME",
            "is_deleted": "BOOLEAN DEFAULT 0",
            "deleted_at": "DATETIME",
        },
    }

    with engine.begin() as conn:
        for table, col_defs in table_columns.items():
            existing = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
            existing_names = {row[1] for row in existing}
            for col_name, col_type in col_defs.items():
                if col_name in existing_names:
                    continue
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"))

        index_defs = {
            "users": {
                "ix_users_created_at": "created_at",
            },
            "tasks": {
                "ix_tasks_created_at": "created_at",
                "ix_tasks_user_id": "user_id",
            },
            "messages": {
                "ix_messages_created_at": "created_at",
                "ix_messages_task_id": "task_id",
            },
        }
        for table, indexes in index_defs.items():
            existing_indexes = conn.execute(text(f"PRAGMA index_list({table})")).fetchall()
            existing_index_names = {row[1] for row in existing_indexes}
            for index_name, column_name in indexes.items():
                if index_name in existing_index_names:
                    continue
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({column_name})"))

def get_db():
    """提供数据库会话的依赖项"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
