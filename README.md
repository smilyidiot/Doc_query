# DocQuery

A small **retrieval-augmented (RAG-style) document Q&A** engine in Python. Point
it at a folder of Markdown/text documents; it chunks them, builds a vector
index, and answers questions from the most relevant passages **with source
citations** — over a **FastAPI** API and a **Typer CLI**.

Built with the help of AI coding tools (Claude Code) and reviewed/tested by hand.

## How it works

```
docs/*.md ──► chunk (overlapping windows) ──► embed ──► vector index
question ──► embed ──► cosine similarity ──► top-k chunks ──► answer + citations
```

- **Chunking** with overlap keeps retrieval focused (`index.py`).
- **Embedding/retrieval** uses **scikit-learn** TF-IDF (unigrams + bigrams) with
  cosine similarity. The embedder is behind a small interface
  (`fit_transform` / `transform`), so a dense model (sentence-transformers) or a
  vector DB (Chroma, FAISS) is a drop-in replacement.
- **Answering** (`qa.py`) is extractive by default — it returns the best passage
  and cites sources — with a `generator` hook to plug in an LLM for generative
  answers over the retrieved context.

## Tech stack

Python · **scikit-learn** · NumPy · **FastAPI** + Pydantic · **Typer** CLI · **pytest**.

## Run it

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CLI
python -m app.cli ask "What is retrieval-augmented generation?"
python -m app.cli ask "How does FastAPI validate requests?" --k 2
python -m app.cli sources

# API (docs at http://localhost:8000/docs)
uvicorn app.main:app --reload --port 8000

# tests
pytest -q
```

## API

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Liveness + number of indexed chunks |
| `GET /api/sources` | List indexed documents |
| `POST /api/query` | `{ "question": "...", "k": 3 }` → answer + cited sources |

Add your own knowledge base by dropping `.md`/`.txt` files into `backend/data/docs/`.
