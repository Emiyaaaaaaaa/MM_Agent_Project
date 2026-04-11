import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import shutil
import uuid
import datetime
from sqlalchemy.orm import Session
from backend.services.rag_engine import rag_engine
from backend.db import database, models, crud

# 初始化数据库表
models.Base.metadata.create_all(bind=database.engine)

# 初始化 FastAPI
app = FastAPI(title="MCM/ICM Multi-modal RAG Agent")

# 基础目录配置
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "backend" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 配置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境请限制
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有请求头
)

# ================= 文件上传接口 =================
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}

@app.post("/api/v1/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    接收用户上传的题目或参考资料，保存至本地并返回路径
    """
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"不支持的文件格式: {file_ext}。目前支持: PDF, Word, TXT, MD。"
        )

    # 生成唯一文件名防止冲突
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = UPLOAD_DIR / unique_filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")
        
    return {
        "message": "文件上传成功",
        "filename": file.filename,
        "local_path": str(file_path),
        "status": "success"
    }

# ================= 静态资源映射 =================
# 将 RAG_Chunks_Gemini 目录映射为 /static，以便前端访问图片
# 假设图片存储在 rag_service/data_chunks/ (原 RAG_Chunks_Gemini)
RAG_ASSETS_DIR = Path(__file__).resolve().parent.parent / "rag_service" / "RAG_Chunks_Gemini"

if RAG_ASSETS_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(RAG_ASSETS_DIR)), name="static")
    print(f"[*] Static assets mounted from: {RAG_ASSETS_DIR}")
else:
    print(f"[!] Warning: RAG assets directory not found at {RAG_ASSETS_DIR}")

# 将生成的图表目录映射为 /plots
PLOT_STATIC_PATH = BASE_DIR / "backend" / "static" / "plots"
PLOT_STATIC_PATH.mkdir(parents=True, exist_ok=True)
app.mount("/plots", StaticFiles(directory=str(PLOT_STATIC_PATH)), name="plots")

# 将论文导出目录映射为 /exports
EXPORT_STATIC_PATH = BASE_DIR / "backend" / "static" / "exports"
EXPORT_STATIC_PATH.mkdir(parents=True, exist_ok=True)
app.mount("/exports", StaticFiles(directory=str(EXPORT_STATIC_PATH)), name="exports")

# ================= 模型定义 =================
class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 3

class SearchResult(BaseModel):
    id: str
    content: str
    distance: float
    filename: str
    year: int
    doc_type: str
    image_urls: List[str]

# ================= 路由定义 =================
@app.get("/")
def read_root():
    return {"status": "online", "message": "MCM Agent RAG Backend is running."}

@app.post("/api/v1/search", response_model=List[SearchResult])
def search_rag(item: SearchQuery):
    """
    RAG 检索接口
    """
    results = rag_engine.search(item.query, n_results=item.top_k)
    
    if not results:
        return []

    formatted_response = []
    for res in results:
        # 将本地图片路径转换为可访问的 URL
        # 原路径类似: c:\...\RAG_Chunks_Gemini\F\2025\F\...\pic.jpg
        # 这里需要根据 app.mount 的逻辑进行相对化处理
        # 简单处理：提取相对于 RAG_ASSETS_DIR 的部分
        image_urls = []
        for local_path in res['images']:
            try:
                rel_path = Path(local_path).relative_to(RAG_ASSETS_DIR)
                # 转换 Windows 路径分隔符为 URL 分隔符
                url_path = f"/static/{rel_path.as_posix()}"
                image_urls.append(url_path)
            except Exception as e:
                print(f"Error converting path {local_path}: {e}")
        
        formatted_response.append(SearchResult(
            id=res['id'],
            content=res['content'],
            distance=res['distance'],
            filename=res['metadata'].get("filename", "Unknown"),
            year=int(res['metadata'].get("year", 0)),
            doc_type=res['metadata'].get("source", "Unknown"),
            image_urls=image_urls
        ))
    
    return formatted_response

# ================= 数据库模型 (Schemas) =================
class UserCreate(BaseModel):
    email: EmailStr
    username: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    email: str
    username: Optional[str]
    created_at: datetime.datetime

    class Config:
        from_attributes = True

class TaskCreate(BaseModel):
    user_id: str
    title: str
    api_key: Optional[str] = None
    model_id: Optional[str] = "gemini-2.5-flash"

class TaskResponse(BaseModel):
    id: str
    user_id: str
    title: str
    status: str
    model_id: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True

# ================= 业务路由 (Business Routes) =================
@app.post("/api/v1/users", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(database.get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        return db_user
    return crud.create_user(db, email=user.email, username=user.username)

@app.post("/api/v1/tasks", response_model=TaskResponse)
def create_task(task: TaskCreate, db: Session = Depends(database.get_db)):
    db_user = crud.get_user(db, user_id=task.user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return crud.create_task(
        db, 
        user_id=task.user_id, 
        title=task.title, 
        api_key=task.api_key, 
        model_id=task.model_id
    )

@app.get("/api/v1/users/{user_id}/tasks", response_model=List[TaskResponse])
def get_user_tasks(user_id: str, db: Session = Depends(database.get_db)):
    return crud.get_user_tasks(db, user_id=user_id)

# ================= WebSocket 连接管理 =================
from fastapi import WebSocket, WebSocketDisconnect
from backend.services.bus import message_bus

class ConnectionManager:
    """
    WebSocket 连接管理器，负责追踪、单播、广播以及连接的优雅清理。
    """
    def __init__(self):
        # 存储当前活跃的连接
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """
        接受并注册一个新的 WebSocket 连接。
        """
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """
        注销一个已断开的连接。
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, message: dict, websocket: WebSocket):
        """
        向特定连接发送 JSON 消息，包含异常处理确保单个连接失败不影响全局。
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending message to client: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """
        向所有活跃连接广播消息。
        """
        for connection in self.active_connections:
            await self.send_json(message, connection)

manager = ConnectionManager()

from backend.graph.builder import app as graph_app
from langchain_core.messages import HumanMessage
import json
import asyncio

class ChatMessage(BaseModel):
    message: str
    thread_id: Optional[str] = "default_user"
    human_feedback: Optional[str] = None

@app.websocket("/ws/chat/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket 核心交互节点，支持 Token 流、进度广播及上下文自动注入。
    """
    await manager.connect(websocket)
    
    # 获取专属进度订阅队列，确保多客户端并发消息隔离
    sub_queue = await message_bus.subscribe()
    
    # 异步子任务：将内部总线进度推送到 WebSocket
    async def bus_listener():
        try:
            while True:
                msg = await sub_queue.get()
                await manager.send_json(msg, websocket)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Bus Listener Error: {e}")

    listener_task = asyncio.create_task(bus_listener())

    try:
        while True:
            # 接收前端指令
            data = await websocket.receive_text()
            try:
                msg_data = json.loads(data)
                user_message = msg_data.get("message", "")
                thread_id = msg_data.get("thread_id", client_id)
                human_feedback = msg_data.get("human_feedback", None)
            except json.JSONDecodeError:
                await manager.send_json({"type": "ERROR", "payload": "格式非法"}, websocket)
                continue

            # 1. 验证任务合法性与动态配置提取
            # 使用 SessionLocal 上下文管理器，确保长耗时 Agent 运行期间不霸占连接
            with database.SessionLocal() as db:
                db_task = crud.get_task(db, task_id=thread_id)
                if not db_task:
                    await manager.send_json({"type": "ERROR", "payload": "Task ID 无效"}, websocket)
                    continue
                
                # 注入模型配置
                task_config = {
                    "api_key": db_task.api_key,
                    "model_id": db_task.model_id
                }
            
            # --- 此时 DB 会话已关闭，数据库连接已回池 ---

            config = {"configurable": {"thread_id": thread_id}}
            inputs = {"messages": [HumanMessage(content=user_message)]} if not human_feedback else None
            
            if human_feedback:
                graph_app.update_state(config, {"human_feedback": human_feedback})
            else:
                graph_app.update_state(config, {"shared_memory": task_config})

            # 2. 执行状态机流式推理
            async for event in graph_app.astream_events(inputs, config=config, version="v2"):
                kind = event["event"]
                
                # Token 增量流式处理
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        await manager.send_json({
                            "type": "TOKEN",
                            "node": event["metadata"].get("langgraph_node", "Unknown"),
                            "payload": content
                        }, websocket)

                # 逻辑节点启动状态推送
                elif kind == "on_chain_start":
                    if event["name"] in ["Reader", "Analysis", "Modeling", "Coder", "Review", "Writing", "Export", "Supervisor"]:
                        await manager.send_json({
                            "type": "STATUS",
                            "node": event["name"],
                            "payload": f"Started: {event['name']}"
                        }, websocket)

                # 最终链路结果整理
                elif kind == "on_chain_end":
                    if event["name"] == "LangGraph":
                        final_state = event["data"].get("output", {})
                        if final_state:
                            res_msg = final_state["messages"][-1].content if "messages" in final_state else ""
                            await manager.send_json({
                                "type": "FINAL",
                                "payload": {
                                    "response": res_msg,
                                    "current_stage": final_state.get("current_stage", "Task Completed")
                                }
                            }, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    finally:
        # 清理资源：停掉进度监听器并注销总线订阅
        listener_task.cancel()
        message_bus.unsubscribe(sub_queue)

@app.post("/api/v1/chat")
async def chat_with_agent(item: ChatMessage):
    config = {"configurable": {"thread_id": item.thread_id}}
    res = await graph_app.ainvoke({"messages": [HumanMessage(content=item.message)]} if not item.human_feedback else None, config=config)
    return {"status": "success", "agent_response": res["messages"][-1].content}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
