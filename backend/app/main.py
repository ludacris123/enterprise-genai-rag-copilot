from __future__ import annotations
import shutil
from pathlib import Path
from typing import Annotated
from uuid import uuid4
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from .rag import RAGService

class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    chroma_path: str = "./data/chroma"
    cors_origins: str = "http://localhost:5173"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    session_id: str = Field(default_factory=lambda: str(uuid4()))

class Source(BaseModel):
    filename: str
    page: int | None = None
    excerpt: str
    score: float | None = None

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[Source]

settings = Settings()
app = FastAPI(title="Enterprise RAG Copilot", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in settings.cors_origins.split(",")], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
rag = RAGService(settings.chroma_path, settings.openai_model)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "configured": bool(settings.openai_api_key)}

@app.post("/api/documents")
async def upload_document(file: Annotated[UploadFile, File()]) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(415, "Upload a PDF, TXT, or Markdown file.")
    target = UPLOAD_DIR / f"{uuid4()}{suffix}"
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    try:
        chunks = rag.ingest(target, file.filename or target.name)
        return {"filename": file.filename, "chunks_indexed": chunks}
    finally:
        target.unlink(missing_ok=True)

@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not settings.openai_api_key:
        raise HTTPException(503, "OPENAI_API_KEY is not configured.")
    answer, sources = rag.answer(payload.question, payload.session_id)
    return ChatResponse(answer=answer, session_id=payload.session_id, sources=sources)

@app.delete("/api/documents")
def clear_documents() -> dict:
    rag.clear()
    return {"status": "cleared"}
