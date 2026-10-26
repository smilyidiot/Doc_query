# Retrieval-Augmented Generation (RAG)

Retrieval-augmented generation is a technique that improves language-model
answers by grounding them in external documents instead of relying only on the
model's internal weights. A RAG pipeline has two stages. First, an indexing
stage splits source documents into chunks, converts each chunk into a vector
embedding, and stores those vectors in a vector database such as Chroma,
FAISS, or Pinecone. Second, at query time the user's question is embedded with
the same model, the most similar chunks are retrieved by cosine similarity, and
those chunks are passed to the language model as context so it can generate a
grounded answer with citations.

RAG reduces hallucination because the model answers from retrieved evidence
rather than guessing. It also keeps answers current: updating the knowledge
base does not require retraining the model. Common building blocks include a
chunking strategy with overlap, an embedding model, a vector store, a retriever
that returns the top-k passages, and an optional re-ranking step.
