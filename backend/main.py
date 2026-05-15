
import logging
import uuid
from time import perf_counter
from pathlib import Path
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.db import models, database
from backend.config import get_cors_origins
from backend.services.graph_runtime import GraphRuntime
from backend.routers import (
    health_router,
    upload_router,
    search_router,
    auth_router,
    tasks_router,
    chat_router,
)

models.Base.metadata.create_all(bind=database.engine)
database.ensure_sqlite_compat_columns()

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE_PATH = LOG_DIR / "backend.debug.log"


class ConsoleNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "WS outbound:" in message and ("type=TOKEN" in message or "type=STATUS" in message):
            return False
        return True


def _configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    console_handler.addFilter(ConsoleNoiseFilter())

    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s:%(lineno)s - %(message)s")
    )

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    for noisy_logger in ["uvicorn.access", "httpx", "httpcore", "watchfiles.main"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


_configure_logging()
logger = logging.getLogger("mm-agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph_runtime = GraphRuntime(checkpoint_path="checkpoints.db")
    await app.state.graph_runtime.startup()
    logger.info("Graph runtime initialized.")
    try:
        yield
    finally:
        await app.state.graph_runtime.shutdown()
        logger.info("Graph runtime shutdown complete.")


app = FastAPI(title="MCM/ICM Multi-modal RAG Agent", lifespan=lifespan)


@app.middleware("http")
async def http_request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:8]
    start = perf_counter()
    path = request.url.path
    method = request.method
    client = request.client.host if request.client else "unknown"
    logger.debug("HTTP request start: id=%s method=%s path=%s client=%s", request_id, method, path, client)
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = int((perf_counter() - start) * 1000)
        logger.exception("HTTP request failed: id=%s method=%s path=%s elapsed_ms=%s", request_id, method, path, elapsed_ms)
        raise
    elapsed_ms = int((perf_counter() - start) * 1000)
    response.headers["x-request-id"] = request_id
    logger.info(
        "HTTP request done: id=%s method=%s path=%s status=%s elapsed_ms=%s",
        request_id,
        method,
        path,
        response.status_code,
        elapsed_ms,
    )
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAG_ASSETS_DIR = BASE_DIR / "rag_service" / "RAG_Chunks_Gemini"
if RAG_ASSETS_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(RAG_ASSETS_DIR)), name="static")

PLOT_STATIC_PATH = BASE_DIR / "backend" / "static" / "plots"
PLOT_STATIC_PATH.mkdir(parents=True, exist_ok=True)
app.mount("/plots", StaticFiles(directory=str(PLOT_STATIC_PATH)), name="plots")

EXPORT_STATIC_PATH = BASE_DIR / "backend" / "static" / "exports"
EXPORT_STATIC_PATH.mkdir(parents=True, exist_ok=True)
app.mount("/exports", StaticFiles(directory=str(EXPORT_STATIC_PATH)), name="exports")

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(search_router)
app.include_router(auth_router)
app.include_router(tasks_router)
app.include_router(chat_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
