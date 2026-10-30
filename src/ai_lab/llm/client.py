"""Thin wrapper over the Anthropic SDK: structured output, prompt caching, cost and errors.

Why a wrapper at all? The same reason you put a Port in front of an HTTP client in
hexagonal architecture: one place that knows the provider's quirks (fallback beta,
refusals, cache layout), and a fake that tests can swap in without network.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import pydantic

from ai_lab.llm.pricing import (
    Cost,
    TokenUsage,
    total_cost,
    usages_from_response,
    worst_case_cost,
)

# On a safety refusal, the API re-runs the request on a model chosen by refusal category.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLike(Protocol):
    """Structural type (Java: an interface the SDK satisfies without declaring it)."""

    @property
    def beta(self) -> Any: ...


class LlmError(Exception):
    pass


class RefusalError(LlmError):
    def __init__(self, category: str | None, explanation: str | None) -> None:
        super().__init__(f"Model refused (category={category}): {explanation}")
        self.category = category
        self.explanation = explanation


class InvalidOutputError(LlmError):
    """The JSON did not pass the Pydantic model (business rules or truncated output)."""


class BudgetExceededError(LlmError):
    pass


@dataclass(frozen=True)
class LlmResult[T]:
    output: T
    served_by: str  # the model that actually answered (may differ on fallback)
    usages: list[TokenUsage]
    request_id: str | None

    @property
    def cost(self) -> Cost:
        return total_cost(self.usages)


class LlmClient:
    def __init__(
        self,
        client: AnthropicLike,
        *,
        model: str,
        effort: str,
        max_tokens: int,
        max_cost_per_call_usd: Decimal,
    ) -> None:
        self._client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_cost_per_call_usd = max_cost_per_call_usd

    @staticmethod
    def build_system(instructions: str, context: str) -> list[dict[str, Any]]:
        """Stable content first, cache breakpoint on the last stable block.

        Caching is a prefix match: anything that changes per request (the question)
        must come after the breakpoint, or every call writes a new cache entry.
        """
        return [
            {"type": "text", "text": instructions},
            {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
        ]

    def count_input_tokens(self, *, instructions: str, context: str, question: str) -> int:
        result = self._client.beta.messages.count_tokens(
            model=self.model,
            system=self.build_system(instructions, context),
            messages=[{"role": "user", "content": question}],
        )
        return int(result.input_tokens)

    def check_budget(self, input_tokens: int) -> Decimal:
        worst = worst_case_cost(self.model, input_tokens, self.max_tokens)
        if worst > self.max_cost_per_call_usd:
            raise BudgetExceededError(
                f"Worst-case cost ${worst:.4f} exceeds limit ${self.max_cost_per_call_usd}"
            )
        return worst

    def structured[T: pydantic.BaseModel](
        self, *, instructions: str, context: str, question: str, schema: type[T]
    ) -> LlmResult[T]:
        try:
            response = self._client.beta.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.build_system(instructions, context),
                messages=[{"role": "user", "content": question}],
                output_format=schema,
                output_config={"effort": self.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except pydantic.ValidationError as e:
            # parse() validates inside the SDK, before we can read stop_reason. So a JSON cut
            # by max_tokens and a JSON that breaks a business rule both land here.
            raise InvalidOutputError(str(e)) from e

        _raise_on_refusal(response)

        parsed = response.parsed_output
        if parsed is None:
            raise InvalidOutputError(f"No structured output (stop_reason={response.stop_reason})")

        return LlmResult(
            output=parsed,
            served_by=response.model,
            usages=usages_from_response(response),
            request_id=getattr(response, "_request_id", None),
        )

    def generate(self, *, system: str, content: list[dict[str, Any]]) -> LlmResult[Any]:
        """Free-form answer with content blocks (e.g. search results with citations).

        Citations and structured output cannot be combined (the API returns 400), so RAG
        answers come back as text blocks and the caller reads their citations.
        """
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        _raise_on_refusal(response)
        return LlmResult(
            output=response,
            served_by=response.model,
            usages=usages_from_response(response),
            request_id=getattr(response, "_request_id", None),
        )


def _raise_on_refusal(response: Any) -> None:
    if response.stop_reason == "refusal":
        details = response.stop_details
        raise RefusalError(
            getattr(details, "category", None), getattr(details, "explanation", None)
        )
