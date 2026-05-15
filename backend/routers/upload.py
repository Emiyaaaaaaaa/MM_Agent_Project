import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile


router = APIRouter(prefix="/api/v1", tags=["upload"])
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md"}
BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "backend" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {file_ext}")

    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = UPLOAD_DIR / unique_filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {exc}") from exc

    return {
        "message": "文件上传成功",
        "filename": file.filename,
        "local_path": str(file_path),
        "status": "success",
    }
