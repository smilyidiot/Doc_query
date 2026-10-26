"""DocQuery — a small retrieval-augmented (RAG-style) document Q&A engine.

Documents are chunked and indexed into a vector space (default: scikit-learn
TF-IDF). A question is embedded with the same model and answered from the
most similar chunks, with source citations. The embedder is pluggable so a
dense model (sentence-transformers) or a vector DB (Chroma) can be dropped in.
"""

__version__ = "1.0.0"
