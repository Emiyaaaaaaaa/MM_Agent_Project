from .health import router as health_router
from .upload import router as upload_router
from .search import router as search_router
from .auth import router as auth_router
from .tasks import router as tasks_router
from .chat import router as chat_router

__all__ = [
    "health_router",
    "upload_router",
    "search_router",
    "auth_router",
    "tasks_router",
    "chat_router",
]
