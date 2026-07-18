"""FastAPI entrypoint exposing document ingestion and chat endpoints."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.ingestion import ingest_files, load_vectorstore
from app.models import ChatRequest, ChatResponse, HealthResponse, IngestResponse
from app.rag_chain import answer_question, reset_session

app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation chatbot that answers questions over your own documents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store = load_vectorstore()
    doc_count = store.index.ntotal if store is not None else 0
    return HealthResponse(status="ok", documents_indexed=doc_count)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile]) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        saved_paths = []
        for upload in files:
            dest = Path(tmp_dir) / upload.filename
            with dest.open("wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved_paths.append(str(dest))

        try:
            names, chunk_count = ingest_files(saved_paths)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(files_indexed=names, chunks_added=chunk_count)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    answer, sources = answer_question(request.question, request.session_id)
    return ChatResponse(answer=answer, sources=sources)


@app.post("/chat/reset")
def chat_reset(session_id: str = "default") -> dict:
    reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
