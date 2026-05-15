from fastapi import APIRouter


router = APIRouter()


@router.get("/")
def read_root():
    return {"status": "online", "message": "MCM Agent RAG Backend is running."}
