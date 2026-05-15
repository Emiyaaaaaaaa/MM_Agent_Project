from pathlib import Path
from typing import List
from fastapi import APIRouter
from backend.schemas.api import SearchQuery, SearchResult
from backend.services.rag_engine import rag_engine


router = APIRouter(prefix="/api/v1", tags=["search"])
BASE_DIR = Path(__file__).resolve().parents[2]
RAG_ASSETS_DIR = BASE_DIR / "rag_service" / "RAG_Chunks_Gemini"


@router.post("/search", response_model=List[SearchResult])
def search_rag(item: SearchQuery):
    results = rag_engine.search(item.query, n_results=item.top_k)
    if not results:
        return []

    formatted_response = []
    for res in results:
        image_urls = []
        for local_path in res["images"]:
            try:
                rel_path = Path(local_path).relative_to(RAG_ASSETS_DIR)
                image_urls.append(f"/static/{rel_path.as_posix()}")
            except Exception:
                continue
        formatted_response.append(
            SearchResult(
                id=res["id"],
                content=res["content"],
                distance=res["distance"],
                filename=res["metadata"].get("filename", "Unknown"),
                year=int(res["metadata"].get("year", 0)),
                doc_type=res["metadata"].get("source", "Unknown"),
                image_urls=image_urls,
            )
        )
    return formatted_response
