# ai-engineering-lab

A hands-on AI engineering lab in Python, built one step at a time. Each step solves one concrete
problem, ships with tests, and records the decision in `docs/` (problem → solution → trade-offs →
discarded alternative).

> Status: steps 1 and 2 of 6 done. Final README (diagram, eval results, CI badge) lands with the
> last step.

## Roadmap

| # | Step | Problem it solves | Status |
|---|---|---|---|
| 1 | [LLM fundamentals](docs/etapa-01-fundamentos-llm.md) | Free-text answers, invisible cost, repeated context, silent failures | ✅ |
| 2 | [RAG with citations](docs/etapa-02-rag.md) | Answer questions about my resume with citations, refuse out-of-scope | ✅ |
| 3 | Evals in pytest/CI | Detect quality regressions when prompts or chunking change | ⏳ |
| 4 | LangGraph agent + MCP | Banking actions with human-in-the-loop and amount limits | ⏳ |
| 5 | Guardrails & observability | Prompt injection (incl. via retrieved content), cost limits, tracing | ⏳ |
| 6 | AWS deployment design | Run step 2 serverless under US$ 10/month (design only) | ⏳ |

## Quick start

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker (for pgvector).

```bash
uv sync
uv run pytest            # offline; pgvector tests start their own container via Testcontainers
cp .env.example .env     # then set ANTHROPIC_API_KEY and VOYAGE_API_KEY locally (never committed)

# Step 1: typed output, caching and cost per call
uv run ai-lab estimate data/samples/senior-ai-engineer.md   # token count only, free
uv run ai-lab fit data/samples/senior-ai-engineer.md --repeat 2

# Step 2: RAG over the resume
docker compose -f docker/docker-compose.yml up -d           # pgvector on port 5435
uv run ai-lab ingest
uv run ai-lab ask "Has he worked with event streaming?" --show-context
```

Stack of step 2: LangChain (splitters, `Embeddings`/`VectorStore` interfaces), Voyage
`voyage-4-lite` embeddings, Postgres + pgvector, Claude with native `search_result` citations.
An answer without a citation is discarded in code, not just discouraged in the prompt.

Data: `data/resume/resume.md` is my public resume, anonymized (no employer or client names).
Job postings in `data/samples/` are fictional.
