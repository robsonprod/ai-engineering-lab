from pathlib import Path

import pytest

from ai_lab.rag import Retriever, load_corpus
from tests.rag_fakes import CosineInMemoryStore, KeywordEmbeddings

CORPUS = Path(__file__).parent.parent / "data" / "resume"


@pytest.fixture(scope="module")
def store() -> CosineInMemoryStore:
    s = CosineInMemoryStore(KeywordEmbeddings())
    chunks = load_corpus(CORPUS)
    s.add_documents(chunks, ids=[c.id for c in chunks if c.id])
    return s


def test_finds_the_chunk_that_mentions_the_topic(store: CosineInMemoryStore) -> None:
    retriever = Retriever(store, k=4, min_relevance=0.1)

    results = retriever.retrieve("DynamoDB single-table design GSI")

    assert results, "expected at least one relevant chunk"
    assert results[0].document.metadata["breadcrumb"].startswith("Projects > balance-consistency")


def test_results_come_sorted_by_relevance(store: CosineInMemoryStore) -> None:
    results = Retriever(store, k=4, min_relevance=0.0).search("Kafka Cassandra partitioning")

    scores = [r.relevance for r in results]
    assert scores == sorted(scores, reverse=True)


def test_out_of_scope_question_retrieves_nothing(store: CosineInMemoryStore) -> None:
    retriever = Retriever(store, k=4, min_relevance=0.1)

    assert retriever.retrieve("favorite lasagna recipe grandmother") == []


def test_threshold_is_what_separates_evidence_from_noise(store: CosineInMemoryStore) -> None:
    question = "Spanish language"
    loose = Retriever(store, k=4, min_relevance=0.0).retrieve(question)
    strict = Retriever(store, k=4, min_relevance=0.99).retrieve(question)

    assert len(loose) == 4  # k always returns something, relevant or not
    assert strict == []
