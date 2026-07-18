"""Document ingestion: load raw files, split into chunks, and index them in FAISS."""
from __future__ import annotations

import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

SUPPORTED_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
}


def _load_file(path: Path):
    loader_cls = SUPPORTED_LOADERS.get(path.suffix.lower())
    if loader_cls is None:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. Supported: {list(SUPPORTED_LOADERS)}"
        )
    return loader_cls(str(path)).load()


def _get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def _index_path() -> Path:
    settings = get_settings()
    path = Path(settings.vectorstore_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_vectorstore() -> FAISS | None:
    """Load a previously persisted FAISS index, if one exists."""
    index_dir = _index_path()
    if not (index_dir / "index.faiss").exists():
        return None
    return FAISS.load_local(
        str(index_dir), _get_embeddings(), allow_dangerous_deserialization=True
    )


def ingest_files(file_paths: list[str]) -> tuple[list[str], int]:
    """Load, chunk, embed, and persist the given files. Returns (filenames, chunk_count)."""
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    all_chunks = []
    indexed_names = []
    for file_path in file_paths:
        path = Path(file_path)
        docs = _load_file(path)
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["source"] = path.name
        all_chunks.extend(chunks)
        indexed_names.append(path.name)

    if not all_chunks:
        return indexed_names, 0

    embeddings = _get_embeddings()
    store = load_vectorstore()
    if store is None:
        store = FAISS.from_documents(all_chunks, embeddings)
    else:
        store.add_documents(all_chunks)

    store.save_local(str(_index_path()))
    return indexed_names, len(all_chunks)


def ingest_directory(directory: str) -> tuple[list[str], int]:
    """Ingest every supported file found in a directory."""
    paths = [
        str(p) for p in Path(directory).rglob("*") if p.suffix.lower() in SUPPORTED_LOADERS
    ]
    if not paths:
        return [], 0
    return ingest_files(paths)
