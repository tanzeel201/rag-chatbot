"""Request and response schemas for the API."""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question.")
    session_id: str = Field(default="default", description="Conversation/session identifier.")


class SourceChunk(BaseModel):
    content: str
    source: str
    page: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class IngestResponse(BaseModel):
    files_indexed: list[str]
    chunks_added: int


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
