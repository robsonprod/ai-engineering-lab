"""The same retrieval, against a real Postgres + pgvector started by Testcontainers.

Run with `uv run pytest -m integration` (needs Docker). Skipped automatically without it.
Why not trust the in-memory store alone: pgvector returns *distance*, not similarity, and
upserts by id. Those two behaviors only show up on the real database.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from ai_lab.config import Settings
from ai_lab.rag import Retriever, load_corpus
from ai_lab.rag.store import ingest, make_pg_store
from tests.rag_fakes import KeywordEmbeddings

pytestmark = pytest.mark.integration

CORPUS = Path(__file__).parent.parent / "data" / "resume"
DIM = 1024


def _docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def settings() -> Iterator[Settings]:
    if not _docker_available():
        pytest.skip("Docker is not running")
    import psycopg
    from testcontainers.postgres import PostgresContainer

    # asyncpg: PGVectorStore runs async inside, and async psycopg cannot use the
    # default event loop on Windows (ProactorEventLoop). asyncpg works everywhere.
    with PostgresContainer("pgvector/pgvector:pg17", driver="asyncpg") as pg:
        url = pg.get_connection_url()
        with psycopg.connect(url.replace("+asyncpg", "")) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        yield Settings(lab_pg_url=url, lab_embedding_dim=DIM, lab_pg_table="chunks_it")


def test_ingest_and_retrieve_on_pgvector(settings: Settings) -> None:
    embeddings = KeywordEmbeddings(size=DIM)
    store = make_pg_store(settings, embeddings, recreate=True)
    chunks = load_corpus(CORPUS)

    ingest(store, chunks)
    results = Retriever(store, k=3, min_relevance=0.1).retrieve("DynamoDB single-table GSI")

    assert results[0].document.metadata["breadcrumb"].startswith("Projects > balance-consistency")
    # Cosine distance was converted to relevance: 0..1, higher is better.
    assert 0.1 <= results[0].relevance <= 1.0


def test_reingesting_the_same_chunks_does_not_duplicate(settings: Settings) -> None:
    embeddings = KeywordEmbeddings(size=DIM)
    store = make_pg_store(settings, embeddings, recreate=True)
    chunks = load_corpus(CORPUS)

    ingest(store, chunks)
    ingest(store, chunks)  # same deterministic ids -> upsert, not insert

    assert len(store.get_by_ids([c.id for c in chunks if c.id])) == len(chunks)
    results = Retriever(store, k=10, min_relevance=0.0).search("Kafka")
    ids = [r.document.id for r in results]
    assert len(ids) == len(set(ids))
