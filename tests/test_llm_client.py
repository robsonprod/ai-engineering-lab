from decimal import Decimal

import pytest

from ai_lab.fit import JobFit
from ai_lab.llm import BudgetExceededError, InvalidOutputError, LlmClient, RefusalError
from ai_lab.llm.client import FALLBACK_BETA
from tests.conftest import FakeAnthropic, fit_json, make_message


def client(fake: FakeAnthropic, limit: str = "0.50") -> LlmClient:
    return LlmClient(
        fake,
        model="claude-opus-5-5",
        effort="low",
        max_tokens=8000,
        max_cost_per_call_usd=Decimal(limit),
    )


def ask(llm: LlmClient) -> object:
    return llm.structured(
        instructions="rules", context="big stable resume", question="posting", schema=JobFit
    )


def test_request_puts_cache_breakpoint_after_stable_content(fake_sdk: FakeAnthropic) -> None:
    fake_sdk.messages.responses.append(make_message(text=fit_json()))

    ask(client(fake_sdk))

    req = fake_sdk.messages.calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"] == {"effort": "low"}
    assert req["system"][0] == {"type": "text", "text": "rules"}
    assert req["system"][1]["cache_control"] == {"type": "ephemeral"}
    # The variable part goes after the breakpoint, in the user turn.
    assert req["messages"] == [{"role": "user", "content": "posting"}]


def test_fallback_on_refusal_is_enabled(fake_sdk: FakeAnthropic) -> None:
    fake_sdk.messages.responses.append(make_message(text=fit_json()))

    ask(client(fake_sdk))

    req = fake_sdk.messages.calls[0]
    assert req["betas"] == [FALLBACK_BETA]
    assert req["fallbacks"] == "default"


def test_returns_typed_output_with_cost(fake_sdk: FakeAnthropic) -> None:
    fake_sdk.messages.responses.append(
        make_message(text=fit_json(), usage={"input_tokens": 1000, "output_tokens": 500})
    )

    result = client(fake_sdk).structured(instructions="r", context="c", question="q", schema=JobFit)

    assert result.output.match_score == 82
    assert result.served_by == "claude-opus-5-5"
    assert result.cost.total == Decimal("0.014")


def test_refusal_is_an_error_not_an_empty_answer(fake_sdk: FakeAnthropic) -> None:
    fake_sdk.messages.responses.append(
        make_message(
            text=None,
            stop_reason="refusal",
            stop_details={"type": "refusal", "category": "cyber", "explanation": "declined"},
        )
    )

    with pytest.raises(RefusalError) as err:
        ask(client(fake_sdk))
    assert err.value.category == "cyber"


def test_business_rule_violation_is_invalid_output(fake_sdk: FakeAnthropic) -> None:
    fake_sdk.messages.responses.append(make_message(text=fit_json(match_score=150)))

    with pytest.raises(InvalidOutputError):
        ask(client(fake_sdk))


def test_json_cut_by_max_tokens_is_invalid_output(fake_sdk: FakeAnthropic) -> None:
    truncated = fit_json()[:40]
    fake_sdk.messages.responses.append(make_message(text=truncated, stop_reason="max_tokens"))

    with pytest.raises(InvalidOutputError):
        ask(client(fake_sdk))


def test_budget_guard_blocks_expensive_calls(fake_sdk: FakeAnthropic) -> None:
    # 3000 x $4 + 8000 x $20 per MTok = $0.172. Output dominates: max_tokens is the ceiling.
    with pytest.raises(BudgetExceededError):
        client(fake_sdk, limit="0.10").check_budget(3_000)
    assert client(fake_sdk, limit="0.20").check_budget(3_000) == Decimal("0.172")
