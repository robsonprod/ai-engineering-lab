from decimal import Decimal

import pytest

from ai_lab.llm.pricing import (
    TokenUsage,
    UnknownModelPriceError,
    cost_of,
    total_cost,
    usages_from_response,
    worst_case_cost,
)
from tests.conftest import make_message

OPUS = "claude-opus-5-5"


def test_input_and_output_have_different_prices() -> None:
    cost = cost_of(TokenUsage(OPUS, input_tokens=1_000, output_tokens=500))

    assert cost.input == Decimal("0.004")  # 1000 x $4/MTok
    assert cost.output == Decimal("0.01")  # 500 x $20/MTok: output is 5x more expensive
    assert cost.total == Decimal("0.014")


def test_cache_write_costs_more_and_cache_read_costs_much_less() -> None:
    write = cost_of(TokenUsage(OPUS, cache_write_5m_tokens=1_000_000))
    read = cost_of(TokenUsage(OPUS, cache_read_tokens=1_000_000))

    assert write.total == Decimal("5")  # 1.25x input
    assert read.total == Decimal("0.20")  # 0.05x input on Opus 5.5


def test_cache_pays_off_from_the_second_call() -> None:
    context = 3_000
    without_cache = total_cost([TokenUsage(OPUS, input_tokens=context)] * 2)
    with_cache = total_cost(
        [
            TokenUsage(OPUS, cache_write_5m_tokens=context),
            TokenUsage(OPUS, cache_read_tokens=context),
        ]
    )

    assert with_cache.total < without_cache.total
    assert without_cache.total == Decimal("0.024")
    assert with_cache.total == Decimal("0.0156")


def test_unknown_model_fails_instead_of_costing_zero() -> None:
    with pytest.raises(UnknownModelPriceError):
        cost_of(TokenUsage("claude-imaginary-9"))


def test_worst_case_uses_full_max_tokens() -> None:
    # 3000 input x $4 + 8000 output x $20, per million
    assert worst_case_cost(OPUS, 3_000, 8_000) == Decimal("0.172")


def test_usage_without_fallback_belongs_to_response_model() -> None:
    msg = make_message(
        text="{}",
        usage={
            "input_tokens": 10,
            "output_tokens": 20,
            "cache_read_input_tokens": 3000,
            "cache_creation_input_tokens": 0,
        },
    )

    [usage] = usages_from_response(msg)

    assert usage == TokenUsage(OPUS, input_tokens=10, output_tokens=20, cache_read_tokens=3000)


def test_fallback_bills_each_model_at_its_own_price() -> None:
    msg = make_message(
        text="{}",
        model="claude-opus-4-8",
        usage={
            "input_tokens": 2000,
            "output_tokens": 300,
            "iterations": [
                {
                    "type": "message",
                    "model": OPUS,
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                {
                    "type": "fallback_message",
                    "model": "claude-opus-4-8",
                    "input_tokens": 1000,
                    "output_tokens": 200,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            ],
        },
    )

    usages = usages_from_response(msg)

    assert [u.model for u in usages] == [OPUS, "claude-opus-4-8"]
    # Opus 5.5: 1000x4 + 100x20 = 6000 ; Opus 4.8: 1000x5 + 200x25 = 10000 (per MTok)
    assert total_cost(usages).total == Decimal("0.016")
