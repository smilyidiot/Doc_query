"""Document loading, chunking and the TF-IDF vector index.

The retrieval seam is deliberately small: :class:`VectorIndex` depends only on
an *embedder* exposing ``fit_transform`` / ``transform``. The default
:class:`TfidfEmbedder` uses scikit-learn; swapping in sentence-transformers or
a Chroma-backed embedder is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    id: int
    source: str
    text: str


@dataclass
class Hit:
    source: str
    score: float
    text: str


class Embedder(Protocol):
    def fit_transform(self, texts: list[str]): ...
    def transform(self, texts: list[str]): ...


class TfidfEmbedder:
    """Sparse TF-IDF embedder (unigrams + bigrams, English stop-words)."""

    def __init__(self) -> None:
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)

    def fit_transform(self, texts: list[str]):
        return self._vec.fit_transform(texts)

    def transform(self, texts: list[str]):
        return self._vec.transform(texts)


def load_documents(folder: str | Path) -> list[dict]:
    """Read every ``.md`` / ``.txt`` file in ``folder`` into ``{source, text}``."""
    folder = Path(folder)
    docs = []
    for path in sorted(folder.glob("**/*")):
        if path.suffix.lower() in {".md", ".txt"} and path.is_file():
            docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs


def chunk_text(text: str, words: int = 120, overlap: int = 30) -> list[str]:
    """Split text into overlapping word windows so retrieval stays focused."""
    tokens = text.split()
    if not tokens:
        return []
    step = max(words - overlap, 1)
    chunks = [" ".join(tokens[i : i + words]) for i in range(0, len(tokens), step)]
    return [c for c in chunks if c.strip()]


class VectorIndex:
    """An in-memory vector index over document chunks."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder: Embedder = embedder or TfidfEmbedder()
        self.chunks: list[Chunk] = []
        self._matrix = None

    @property
    def size(self) -> int:
        return len(self.chunks)

    def build(self, documents: list[dict], words: int = 120, overlap: int = 30) -> "VectorIndex":
        chunks: list[Chunk] = []
        for doc in documents:
            for piece in chunk_text(doc["text"], words=words, overlap=overlap):
                chunks.append(Chunk(id=len(chunks), source=doc["source"], text=piece))
        if not chunks:
            raise ValueError("No document chunks to index.")
        self.chunks = chunks
        self._matrix = self.embedder.fit_transform([c.text for c in chunks])
        return self

    def query(self, question: str, k: int = 3, min_score: float = 0.01) -> list[Hit]:
        if self._matrix is None:
            raise RuntimeError("Index has not been built yet.")
        qv = self.embedder.transform([question])
        sims = cosine_similarity(qv, self._matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [
            Hit(source=self.chunks[i].source, score=round(float(sims[i]), 4), text=self.chunks[i].text)
            for i in order
            if sims[i] >= min_score
        ]
