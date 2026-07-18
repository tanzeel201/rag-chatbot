# RAG Chatbot — Document Q&A

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded
in your own documents. Upload PDFs, text files, or Markdown, and ask
questions in natural language — answers are generated from the relevant
passages retrieved from your files, with source citations, not from the
model's memory alone.

![status](https://img.shields.io/badge/status-active-brightgreen)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-informational)

## How it works

```
                 ┌────────────┐        ┌──────────────┐
  documents ───▶ │  Chunking  │ ─────▶ │  Embeddings  │ ─────▶  FAISS index
                 └────────────┘        └──────────────┘

  question ─────▶ embed question ─────▶ similarity search ─────▶ top-k chunks
                                                                       │
                                                                       ▼
                                                          LLM (grounded prompt)
                                                                       │
                                                                       ▼
                                                        answer + cited sources
```

1. **Ingestion** — PDF / TXT / Markdown files are loaded and split into
   overlapping chunks (`RecursiveCharacterTextSplitter`).
2. **Embedding & indexing** — each chunk is embedded and stored in a local
   FAISS vector index, persisted to disk so it survives restarts.
3. **Retrieval** — an incoming question is embedded and matched against the
   index to pull back the most relevant chunks.
4. **Generation** — the chunks are injected into the LLM prompt as context.
   The model is instructed to answer only from that context and to say when
   it doesn't know, which keeps answers grounded and reduces hallucination.
5. **Citations** — every response returns the source file (and page number,
   for PDFs) each part of the answer came from.

## Tech stack

- **FastAPI** — REST API (`/ingest`, `/chat`, `/health`)
- **LangChain** — document loaders, text splitting, RAG orchestration
- **FAISS** — local vector similarity search
- **OpenAI API** — embeddings + chat completion (swappable for any LangChain-
  supported provider)
- Minimal vanilla JS/HTML frontend for demoing the chatbot in the browser

## Project structure

```
rag-chatbot/
├── app/
│   ├── main.py          # FastAPI routes
│   ├── ingestion.py      # load, chunk, embed, persist documents
│   ├── rag_chain.py       # retrieval + grounded generation + chat history
│   ├── models.py           # request/response schemas
│   └── config.py            # settings from environment variables
├── scripts/
│   └── ingest.py               # CLI to bulk-index a folder of documents
├── static/
│   └── index.html                 # demo chat UI
├── data/sample_docs/
│   └── about_rag.txt                 # sample file to try the demo with
├── tests/
│   └── test_api.py                      # API smoke tests
├── Dockerfile
└── requirements.txt
```

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/tanzeel201/rag-chatbot.git
cd rag-chatbot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY
```

### 3. Index some documents

Use the included sample file, or point it at your own folder:

```bash
python scripts/ingest.py data/sample_docs
```

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000** for the demo chat UI, or
**http://localhost:8000/docs** for interactive API docs.

### 5. Ask a question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How does this project implement RAG?"}'
```

## Run with Docker

```bash
docker build -t rag-chatbot .
docker run -p 8000:8000 --env-file .env rag-chatbot
```

## API reference

| Method | Endpoint       | Description                                   |
|--------|----------------|------------------------------------------------|
| GET    | `/health`      | Service status + number of chunks indexed      |
| POST   | `/ingest`      | Upload one or more files to index (multipart)   |
| POST   | `/chat`        | Ask a question, returns answer + source chunks  |
| POST   | `/chat/reset`  | Clear conversation history for a session        |

## Running tests

```bash
pytest
```

## Possible extensions

- Swap FAISS for a managed vector database (Pinecone, Weaviate, pgvector)
- Add per-user authentication and multi-tenant document isolation
- Stream answers token-by-token over SSE/WebSockets
- Add a feedback endpoint to flag incorrect answers and improve retrieval

