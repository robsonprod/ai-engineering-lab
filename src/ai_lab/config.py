"""Lab settings, read from environment variables or a local `.env` file.

Java analogy: a Spring `@ConfigurationProperties` class. Pydantic validates types on load,
so a typo like `LAB_EFFORT=hihg` fails at startup instead of at the first API call.
"""

from decimal import Decimal
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # SecretStr hides the value in repr/logs: print(settings) shows '**********'.
    anthropic_api_key: SecretStr | None = None

    lab_model: str = "claude-opus-5-5"
    # Opus 5.5 always thinks; effort is the only knob for how much. "low" fits short extraction.
    lab_effort: Effort = "low"
    # max_tokens is also the cost ceiling of a call: output price x max_tokens.
    lab_max_tokens: int = 8000
    # Refuse to call the API when the worst-case cost of one call exceeds this.
    lab_max_cost_per_call_usd: Decimal = Decimal("0.50")

    # --- Stage 2: RAG ---
    voyage_api_key: SecretStr | None = None
    lab_embedding_model: str = "voyage-4-lite"
    # Smaller vectors = less storage and faster search; 512 is plenty for a short resume.
    lab_embedding_dim: int = 512
    lab_pg_url: str = "postgresql+asyncpg://lab:lab@localhost:5435/ai_lab"
    lab_pg_table: str = "resume_chunks"
    lab_rag_k: int = 4
    # Below this relevance (0..1) a chunk is not considered evidence. Calibrated by evals (stage 3).
    lab_rag_min_relevance: float = 0.40
