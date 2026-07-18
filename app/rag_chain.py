"""Retrieval-Augmented Generation chain: retrieve relevant chunks, then ask the LLM."""
from __future__ import annotations

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.ingestion import load_vectorstore
from app.models import SourceChunk

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided "
    "context from the user's documents. If the answer isn't contained in the "
    "context, say you don't know instead of guessing. Keep answers concise and "
    "cite which source each fact comes from when relevant."
)

# Simple in-memory per-session chat history. Swap for Redis/DB in production.
_session_histories: dict[str, BaseChatMessageHistory] = {}


def _get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_histories:
        _session_histories[session_id] = InMemoryChatMessageHistory()
    return _session_histories[session_id]


def _build_context(chunks) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")
        label = f"[{i}] {source}" + (f" (page {page + 1})" if page is not None else "")
        parts.append(f"{label}\n{chunk.page_content}")
    return "\n\n".join(parts)


def answer_question(question: str, session_id: str = "default") -> tuple[str, list[SourceChunk]]:
    """Retrieve relevant chunks and generate a grounded answer with citations."""
    settings = get_settings()
    store = load_vectorstore()

    if store is None:
        return (
            "No documents have been indexed yet. Upload documents via /ingest first.",
            [],
        )

    retriever = store.as_retriever(search_kwargs={"k": settings.retriever_k})
    retrieved_docs = retriever.invoke(question)
    context = _build_context(retrieved_docs)

    history = _get_history(session_id)
    llm = ChatOpenAI(model=settings.chat_model, api_key=settings.openai_api_key, temperature=0.2)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *history.messages,
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    response = llm.invoke(messages)
    answer = response.content

    history.add_user_message(question)
    history.add_ai_message(answer)

    sources = [
        SourceChunk(
            content=doc.page_content[:300],
            source=doc.metadata.get("source", "unknown"),
            page=doc.metadata.get("page"),
        )
        for doc in retrieved_docs
    ]
    return answer, sources


def reset_session(session_id: str) -> None:
    _session_histories.pop(session_id, None)
