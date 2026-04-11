from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# 数据库文件路径
SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"

# 创建引擎
# check_same_thread=False 仅适用于 SQLite
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 基类，供各模型继承
Base = declarative_base()

def get_db():
    """提供数据库会话的依赖项"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
