"""FastAPI service for DocQuery.

The document index is built once at startup from ``data/docs`` and reused
across requests.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__, qa
from .index import VectorIndex, load_documents
from .models import Answer, QueryRequest

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"

index = VectorIndex()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the document index once when the service starts.
    docs = load_documents(DOCS_DIR)
    if docs:
        index.build(docs)
    yield


app = FastAPI(title="DocQuery API", version=__version__,
              description="Retrieval-augmented document Q&A with citations.",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "chunks_indexed": index.size}


@app.get("/api/sources")
def sources() -> dict:
    names = sorted({c.source for c in index.chunks})
    return {"documents": names, "chunks_indexed": index.size}


@app.post("/api/query", response_model=Answer)
def query(req: QueryRequest) -> dict:
    return qa.answer(index, req.question, k=req.k)
