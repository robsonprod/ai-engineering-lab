"""Token usage and cost per call.

Money uses Decimal (Java: BigDecimal). Floats would accumulate rounding errors
when summing thousands of eval calls.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

PER_MTOK = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens. Cache writes are a multiple of the input price."""

    input: Decimal
    output: Decimal
    cache_read: Decimal

    @property
    def cache_write_5m(self) -> Decimal:
        return self.input * Decimal("1.25")

    @property
    def cache_write_1h(self) -> Decimal:
        return self.input * 2


# Anthropic first-party prices, checked on 2026-10-23. Update here when they change.
PRICES: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(Decimal("4"), Decimal("20"), Decimal("0.20")),
    "claude-sonnet-5-5": ModelPrice(Decimal("2"), Decimal("10"), Decimal("0.20")),
    # Haiku 5.5: price valid for prompts up to 100K tokens. Cache read assumed at 0.1x input.
    "claude-haiku-5-5": ModelPrice(Decimal("0.10"), Decimal("0.50"), Decimal("0.01")),
    # Fallback target for some refusals when `fallbacks: "default"` is on.
    "claude-opus-4-8": ModelPrice(Decimal("5"), Decimal("25"), Decimal("0.50")),
    "claude-fable-5-1": ModelPrice(Decimal("10"), Decimal("50"), Decimal("0.25")),
}


class UnknownModelPriceError(KeyError):
    """Fail loudly: a silent $0 cost would hide real spend."""


def price_of(model: str) -> ModelPrice:
    try:
        return PRICES[model]
    except KeyError:
        raise UnknownModelPriceError(f"No price registered for model '{model}'") from None


@dataclass(frozen=True)
class TokenUsage:
    """Usage of one model inside one API call (a call can span two models on fallback)."""

    model: str
    input_tokens: int = 0  # uncached input, full price
    output_tokens: int = 0  # includes thinking tokens, even when thinking is not displayed
    cache_read_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cache_read_tokens
            + self.cache_write_5m_tokens
            + self.cache_write_1h_tokens
        )


@dataclass(frozen=True)
class Cost:
    input: Decimal = Decimal(0)
    output: Decimal = Decimal(0)
    cache_read: Decimal = Decimal(0)
    cache_write: Decimal = Decimal(0)

    @property
    def total(self) -> Decimal:
        return self.input + self.output + self.cache_read + self.cache_write

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            self.input + other.input,
            self.output + other.output,
            self.cache_read + other.cache_read,
            self.cache_write + other.cache_write,
        )


def cost_of(usage: TokenUsage) -> Cost:
    p = price_of(usage.model)
    return Cost(
        input=usage.input_tokens * p.input / PER_MTOK,
        output=usage.output_tokens * p.output / PER_MTOK,
        cache_read=usage.cache_read_tokens * p.cache_read / PER_MTOK,
        cache_write=(
            usage.cache_write_5m_tokens * p.cache_write_5m
            + usage.cache_write_1h_tokens * p.cache_write_1h
        )
        / PER_MTOK,
    )


def total_cost(usages: Iterable[TokenUsage]) -> Cost:
    return sum((cost_of(u) for u in usages), Cost())


def worst_case_cost(model: str, input_tokens: int, max_tokens: int) -> Decimal:
    """Upper bound before calling: no cache hit and the model uses all of max_tokens."""
    p = price_of(model)
    return (input_tokens * p.input + max_tokens * p.output) / PER_MTOK


def usages_from_response(response: Any) -> list[TokenUsage]:
    """Normalize the SDK usage object into one TokenUsage per model that ran.

    With server-side fallback, `usage.iterations` lists each model separately, and each
    one is billed at its own price. Without it, the top-level usage belongs to
    `response.model`.
    """
    usage = response.usage
    iterations = [
        it for it in (getattr(usage, "iterations", None) or []) if getattr(it, "model", None)
    ]
    if iterations:
        return [_to_token_usage(it, it.model) for it in iterations]
    return [_to_token_usage(usage, response.model)]


def _to_token_usage(u: Any, model: str) -> TokenUsage:
    created = u.cache_creation_input_tokens or 0
    breakdown = getattr(u, "cache_creation", None)
    if breakdown is not None:
        write_5m = breakdown.ephemeral_5m_input_tokens or 0
        write_1h = breakdown.ephemeral_1h_input_tokens or 0
    else:
        write_5m, write_1h = created, 0
    return TokenUsage(
        model=model,
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_read_tokens=u.cache_read_input_tokens or 0,
        cache_write_5m_tokens=write_5m,
        cache_write_1h_tokens=write_1h,
    )
