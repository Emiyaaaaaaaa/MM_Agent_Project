import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, EmailStr, Field


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


class AuthPayload(BaseModel):
    email: EmailStr
    password: str


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
    model_id: Optional[str] = "gemini-2.5-flash-lite"


class TaskResponse(BaseModel):
    id: str
    user_id: str
    title: str
    status: str
    model_id: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class ChatMessage(BaseModel):
    message: str
    thread_id: Optional[str] = "default_user"
    human_feedback: Optional[str] = None


class StagePayload(BaseModel):
    id: str
    label: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class ContentBlock(BaseModel):
    type: Literal["text", "markdown", "code", "math", "image", "table"]
    text: Optional[str] = None
    code: Optional[str] = None
    language: Optional[str] = None
    latex: Optional[str] = None
    url: Optional[str] = None
    alt: Optional[str] = None
    headers: Optional[List[str]] = None
    rows: Optional[List[List[str]]] = None


class StructuredMessage(BaseModel):
    role: Literal["user", "ai", "system"]
    content_blocks: List[ContentBlock]
    node: Optional[str] = None
    created_at: Optional[datetime.datetime] = None


class FinalEventPayload(BaseModel):
    blocks: List[ContentBlock]
    stage: StagePayload


class IntermediateEventPayload(BaseModel):
    node: str
    summary_blocks: List[ContentBlock]
    preview_blocks: List[ContentBlock]
    ask_continue: bool = True
    stage: StagePayload
