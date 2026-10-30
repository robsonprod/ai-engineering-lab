"""Find the chunks that can answer a question, or nothing.

"Nothing" is a feature: if no chunk is close enough, the assistant refuses before
spending a single token on the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


@dataclass(frozen=True)
class Retrieved:
    document: Document
    relevance: float  # 0..1, higher = closer in meaning


class Retriever:
    def __init__(self, store: VectorStore, *, k: int, min_relevance: float) -> None:
        self._store = store
        self.k = k
        self.min_relevance = min_relevance

    def search(self, question: str) -> list[Retrieved]:
        """All top-k results with their score, before the threshold (for debugging)."""
        # Raw scores differ per store: pgvector returns a *distance* (lower = closer), the
        # in-memory store a *similarity*. Relevance scores normalize both to 0..1.
        pairs = self._store.similarity_search_with_relevance_scores(question, k=self.k)
        return [Retrieved(doc, score) for doc, score in pairs]

    def retrieve(self, question: str) -> list[Retrieved]:
        return [r for r in self.search(question) if r.relevance >= self.min_relevance]
