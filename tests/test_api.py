"""Basic smoke tests for the API. Run with: pytest"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["documents_indexed"], int)


def test_chat_without_indexed_documents_returns_guidance():
    response = client.post("/chat", json={"question": "What is this project about?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert isinstance(body["sources"], list)


def test_ingest_rejects_empty_upload():
    response = client.post("/ingest", files=[])
    assert response.status_code in (400, 422)
