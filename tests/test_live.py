"""Real API calls. Skipped unless you run `uv run pytest --live`.

Estimated cost: 3 calls on claude-opus-5-5 with effort=low, about US$ 0.05-0.10 in total,
plus a few thousand Voyage tokens (inside the free tier). Needs ANTHROPIC_API_KEY and
VOYAGE_API_KEY.
"""

from pathlib import Path

import pytest

from ai_lab.cli import build_analyzer
from ai_lab.config import Settings

ROOT = Path(__file__).parent.parent


@pytest.mark.live
def test_second_call_reads_the_resume_from_cache() -> None:
    analyzer = build_analyzer(Settings(), ROOT / "data" / "resume" / "resume.md")
    posting = (ROOT / "data" / "samples" / "senior-ai-engineer.md").read_text(encoding="utf-8")

    first = analyzer.analyze(posting)
    second = analyzer.analyze(posting)

    assert 0 <= first.output.match_score <= 100
    assert sum(u.cache_write_5m_tokens for u in first.usages) > 0
    assert sum(u.cache_read_tokens for u in second.usages) > 0
    assert second.cost.cache_read > 0


@pytest.mark.live
def test_rag_answers_with_citations_and_refuses_out_of_scope() -> None:
    from ai_lab.cli import build_llm
    from ai_lab.rag import Outcome, ResumeAssistant, Retriever, load_corpus
    from ai_lab.rag.store import make_embeddings
    from tests.rag_fakes import CosineInMemoryStore

    settings = Settings()
    store = CosineInMemoryStore(make_embeddings(settings))  # real Voyage, no database needed
    chunks = load_corpus(ROOT / "data" / "resume")
    store.add_documents(chunks, ids=[c.id for c in chunks if c.id])
    retriever = Retriever(store, k=settings.lab_rag_k, min_relevance=settings.lab_rag_min_relevance)
    assistant = ResumeAssistant(retriever, build_llm(settings))

    kafka = assistant.ask("Has he worked with event streaming?")
    food = assistant.ask("What is his favorite food?")

    assert kafka.outcome is Outcome.ANSWERED
    assert kafka.citations
    assert food.outcome is not Outcome.ANSWERED
