"""Offline stand-ins for the embedding model and the vector store."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

STOPWORDS = {
    "a", "an", "and", "the", "of", "in", "on", "with", "for", "to", "is", "has", "he", "his",
    "what", "does", "did", "how", "which", "who", "it", "at", "by", "or", "as", "from", "do",
    "any", "be", "are", "was", "that", "this", "about", "there", "years",
}  # fmt: skip


class KeywordEmbeddings(Embeddings):
    """Bag of words hashed into a fixed-size vector (a poor man's embedding).

    Texts that share words end up close; texts with no words in common score 0.
    A real model also puts synonyms close ("message broker" ~ "Kafka"); this one cannot,
    which is exactly why production uses a trained model.
    """

    def __init__(self, size: int = 4096) -> None:
        self.size = size

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.size
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word in STOPWORDS or len(word) < 2:
                continue
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self.size] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class CosineInMemoryStore(InMemoryVectorStore):
    """The in-memory store already returns cosine similarity, but does not declare it.

    pgvector returns cosine *distance* instead. Declaring the relevance function per store
    is what lets the Retriever use one threshold for both.
    """

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return lambda similarity: similarity
