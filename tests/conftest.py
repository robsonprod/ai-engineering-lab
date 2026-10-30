"""Test doubles for the Anthropic SDK.

The fake builds real SDK types (BetaMessage) and runs the SDK's own parser, so a test
fails if the SDK changes the shape we depend on. Only the HTTP call is faked.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from anthropic.lib._parse._response import parse_beta_response
from anthropic.types.beta import BetaMessage


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--live", action="store_true", help="run tests that call the real API")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="calls the real API; use --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


def make_message(
    *,
    text: str | None,
    model: str = "claude-opus-5-5",
    stop_reason: str = "end_turn",
    stop_details: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
) -> BetaMessage:
    content = [{"type": "text", "text": text}] if text is not None else []
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": usage or {"input_tokens": 100, "output_tokens": 50},
        }
    )


def cited_message(*parts: tuple[str, list[dict[str, Any]]], **kwargs: Any) -> BetaMessage:
    """A RAG answer: text blocks, each with its own citations (as the API returns them)."""
    msg = make_message(text=None, **kwargs)
    content = [{"type": "text", "text": text, "citations": cites or None} for text, cites in parts]
    return BetaMessage.model_validate({**msg.model_dump(), "content": content})


def citation(source: str, cited_text: str, index: int = 0) -> dict[str, Any]:
    return {
        "type": "search_result_location",
        "source": source,
        "title": source.split("#")[-1],
        "cited_text": cited_text,
        "search_result_index": index,
        "start_block_index": 0,
        "end_block_index": 1,
    }


class FakeMessages:
    def __init__(self) -> None:
        self.responses: list[BetaMessage] = []
        self.calls: list[dict[str, Any]] = []
        self.count_calls: list[dict[str, Any]] = []
        self.input_tokens = 3000

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        return parse_beta_response(response=response, output_format=kwargs["output_format"])

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)

    def count_tokens(self, **kwargs: Any) -> Any:
        self.count_calls.append(kwargs)
        return SimpleNamespace(input_tokens=self.input_tokens)


class FakeAnthropic:
    def __init__(self) -> None:
        self.messages = FakeMessages()
        self.beta = SimpleNamespace(messages=self.messages)


@pytest.fixture
def fake_sdk() -> FakeAnthropic:
    return FakeAnthropic()


def fit_json(**overrides: Any) -> str:
    data: dict[str, Any] = {
        "role_title": "Senior AI Engineer",
        "seniority_required": "senior",
        "match_score": 82,
        "matched": [{"skill": "Kafka", "evidence": "Architects systems with Apache Kafka"}],
        "missing": ["Offline evals"],
        "summary": "Strong back-end and AWS fit; LLM evals not evidenced yet.",
    }
    data.update(overrides)
    return json.dumps(data)
