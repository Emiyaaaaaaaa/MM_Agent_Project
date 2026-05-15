from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.db import crud, database
from backend.schemas.api import AuthPayload, UserResponse


router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse)
def register(payload: AuthPayload, db: Session = Depends(database.get_db)):
    user = crud.get_user_by_email(db, payload.email)
    if user:
        raise HTTPException(status_code=409, detail="邮箱已注册")
    return crud.create_user(db, email=payload.email, password=payload.password)


@router.post("/login", response_model=UserResponse)
def login(payload: AuthPayload, db: Session = Depends(database.get_db)):
    user = crud.authenticate_user(db, payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    return user
