"""Turn retrieved chunks into an answer with citations.

The default is **extractive**: it returns the best-matching passage(s) as the
answer context and lists the sources. A generative step (LLM over the retrieved
context) is a drop-in — see :func:`answer`'s ``generator`` hook.
"""

from __future__ import annotations

from typing import Callable

from .index import Hit, VectorIndex


def answer(
    index: VectorIndex,
    question: str,
    k: int = 3,
    generator: Callable[[str, list[Hit]], str] | None = None,
) -> dict:
    """Retrieve the top ``k`` chunks and build an answer payload.

    If ``generator`` is provided (e.g. an LLM call), it produces the prose
    answer from the question + retrieved context; otherwise the top passage is
    returned extractively.
    """
    hits = index.query(question, k=k)
    if not hits:
        return {"question": question, "answer": "No relevant information found in the indexed documents.",
                "sources": [], "confidence": 0.0}

    if generator is not None:
        text = generator(question, hits)
    else:
        text = hits[0].text

    return {
        "question": question,
        "answer": text,
        "confidence": hits[0].score,
        "sources": [
            {"source": h.source, "score": h.score, "snippet": _snippet(h.text)} for h in hits
        ],
    }


def _snippet(text: str, words: int = 40) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")
