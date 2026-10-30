from decimal import Decimal

from langchain_core.documents import Document

from ai_lab.llm import LlmClient
from ai_lab.rag import (
    INSTRUCTIONS,
    OUT_OF_SCOPE,
    REFUSAL_TEXT,
    Outcome,
    ResumeAssistant,
    Retrieved,
    build_content,
)
from tests.conftest import FakeAnthropic, citation, cited_message

KAFKA = Retrieved(
    Document(
        page_content="Experience > Senior Engineer\n- Architects systems with Apache Kafka.",
        metadata={"source": "resume.md", "breadcrumb": "Experience > Senior Engineer"},
    ),
    relevance=0.71,
)


class StubRetriever:
    def __init__(self, results: list[Retrieved]) -> None:
        self.results = results

    def retrieve(self, question: str) -> list[Retrieved]:
        return self.results


def assistant(fake: FakeAnthropic, results: list[Retrieved]) -> ResumeAssistant:
    llm = LlmClient(
        fake,
        model="claude-opus-5-5",
        effort="low",
        max_tokens=2000,
        max_cost_per_call_usd=Decimal("0.50"),
    )
    return ResumeAssistant(StubRetriever(results), llm)  # type: ignore[arg-type]


def test_retrieved_chunks_go_as_search_results_with_citations_enabled() -> None:
    content = build_content("Has he used Kafka?", [KAFKA])

    result, question = content
    assert result["type"] == "search_result"
    assert result["citations"] == {"enabled": True}
    assert result["source"] == "resume.md#Experience > Senior Engineer"
    assert result["content"][0]["text"].endswith("Apache Kafka.")
    # Evidence first, question last.
    assert question["text"] == "<question>\nHas he used Kafka?\n</question>"


def test_answer_keeps_the_citations_returned_by_the_api(fake_sdk: FakeAnthropic) -> None:
    src = "resume.md#Experience > Senior Engineer"
    fake_sdk.messages.responses.append(
        cited_message(
            ("Yes. ", []),
            ("He architects distributed systems with Apache Kafka.", [citation(src, "Kafka")]),
            usage={"input_tokens": 1200, "output_tokens": 300},
        )
    )

    answer = assistant(fake_sdk, [KAFKA]).ask("Has he used Kafka?")

    assert answer.outcome is Outcome.ANSWERED
    assert answer.text == "Yes. He architects distributed systems with Apache Kafka."
    assert [(c.source, c.cited_text) for c in answer.citations] == [(src, "Kafka")]
    assert answer.cost.total == Decimal("0.0108")  # 1200 x $4 + 300 x $20, per MTok
    req = fake_sdk.messages.calls[0]
    assert req["system"][0]["text"] == INSTRUCTIONS


def test_no_relevant_chunk_refuses_without_calling_the_llm(fake_sdk: FakeAnthropic) -> None:
    answer = assistant(fake_sdk, []).ask("What is his favorite food?")

    assert answer.outcome is Outcome.NO_RELEVANT_CONTEXT
    assert answer.text == REFUSAL_TEXT
    assert fake_sdk.messages.calls == []
    assert answer.cost.total == 0


def test_model_saying_out_of_scope_becomes_a_refusal(fake_sdk: FakeAnthropic) -> None:
    fake_sdk.messages.responses.append(cited_message((OUT_OF_SCOPE, [])))

    answer = assistant(fake_sdk, [KAFKA]).ask("Is he married?")

    assert answer.outcome is Outcome.OUT_OF_SCOPE
    assert answer.text == REFUSAL_TEXT


def test_answer_without_any_citation_is_discarded(fake_sdk: FakeAnthropic) -> None:
    """No evidence, no answer: this is the grounding rule, enforced in code."""
    fake_sdk.messages.responses.append(cited_message(("He has 15 years of Kafka.", [])))

    answer = assistant(fake_sdk, [KAFKA]).ask("How long with Kafka?")

    assert answer.outcome is Outcome.NO_CITATIONS
    assert answer.text == REFUSAL_TEXT
    assert answer.usages  # the call still cost money, and that is reported


def test_repeated_citations_are_listed_once(fake_sdk: FakeAnthropic) -> None:
    src = "resume.md#Experience > Senior Engineer"
    fake_sdk.messages.responses.append(
        cited_message(
            ("Kafka. ", [citation(src, "Apache Kafka")]),
            ("Again Kafka.", [citation(src, "Apache Kafka")]),
        )
    )

    answer = assistant(fake_sdk, [KAFKA]).ask("Kafka?")

    assert len(answer.citations) == 1
