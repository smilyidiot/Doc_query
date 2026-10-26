"""Tests for chunking and TF-IDF retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.index import VectorIndex, chunk_text, load_documents

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


@pytest.fixture(scope="module")
def index():
    return VectorIndex().build(load_documents(DOCS))


def test_chunking_overlap():
    text = " ".join(str(i) for i in range(300))
    chunks = chunk_text(text, words=120, overlap=30)
    assert len(chunks) >= 3
    # consecutive chunks should share the overlap region
    assert chunks[0].split()[-1] in chunks[1].split()


def test_index_builds(index):
    assert index.size > 0


def test_retrieval_is_relevant(index):
    hits = index.query("What is retrieval augmented generation?", k=3)
    assert hits and hits[0].source == "rag.md"


def test_retrieval_routes_to_right_doc(index):
    assert index.query("How does FastAPI validate requests?", k=1)[0].source == "fastapi.md"
    assert index.query("How do I group and resample tabular data?", k=1)[0].source == "pandas.md"


def test_no_match_returns_empty(index):
    assert index.query("xyzzy quux nonsense zzzz", k=3) == []
