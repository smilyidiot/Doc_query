"""API tests for DocQuery (index is built on startup)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager triggers the startup index build
        yield c


def test_health_reports_index(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] > 0


def test_sources_listed(client):
    docs = client.get("/api/sources").json()["documents"]
    assert "rag.md" in docs and "fastapi.md" in docs


def test_query_returns_answer_with_sources(client):
    r = client.post("/api/query", json={"question": "What is RAG?", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"]
    assert body["sources"][0]["source"] == "rag.md"
    assert body["confidence"] > 0


def test_query_validates_input(client):
    assert client.post("/api/query", json={"question": "", "k": 3}).status_code == 422
    assert client.post("/api/query", json={"question": "x", "k": 99}).status_code == 422
