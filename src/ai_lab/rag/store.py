"""Wiring of the real infrastructure: Voyage embeddings and pgvector.

Kept apart from the RAG logic so tests can plug an in-memory store and fake embeddings
(Java analogy: the @Configuration class that builds the beans).
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_voyageai import VoyageAIEmbeddings

from ai_lab.config import Settings


def make_embeddings(settings: Settings) -> Embeddings:
    key = settings.voyage_api_key
    # Without a key in .env, the Voyage client reads VOYAGE_API_KEY from the environment.
    extra = {"voyage_api_key": key} if key else {}
    return VoyageAIEmbeddings(
        model=settings.lab_embedding_model,
        output_dimension=settings.lab_embedding_dim,
        batch_size=64,
        **extra,
    )


def make_pg_store(
    settings: Settings, embeddings: Embeddings, *, recreate: bool = False
) -> PGVectorStore:
    engine = PGEngine.from_connection_string(url=settings.lab_pg_url)
    if recreate:
        engine.init_vectorstore_table(
            table_name=settings.lab_pg_table,
            vector_size=settings.lab_embedding_dim,
            overwrite_existing=True,
        )
    return PGVectorStore.create_sync(
        engine=engine, table_name=settings.lab_pg_table, embedding_service=embeddings
    )


def ingest(store: PGVectorStore, chunks: list[Document]) -> int:
    store.add_documents(chunks, ids=[c.id for c in chunks if c.id])
    return len(chunks)
