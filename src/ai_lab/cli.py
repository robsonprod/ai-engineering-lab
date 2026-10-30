"""Command line for the lab.

Stage 1
    uv run ai-lab estimate data/samples/senior-ai-engineer.md   # counts tokens, no generation
    uv run ai-lab fit data/samples/senior-ai-engineer.md --repeat 2   # shows the cache hit
Stage 2
    uv run ai-lab ingest                                  # chunks + embeddings into pgvector
    uv run ai-lab ask "Has he worked with Kafka?"         # answer with citations
    uv run ai-lab ask "What is his favorite food?" --show-context   # refusal + retrieval debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anthropic

from ai_lab.config import Settings
from ai_lab.fit import FitAnalyzer, JobFit, load_resume
from ai_lab.llm import LlmClient, LlmError, LlmResult
from ai_lab.llm.pricing import Cost, TokenUsage

DEFAULT_RESUME = Path("data/resume/resume.md")
DEFAULT_CORPUS = Path("data/resume")


def build_llm(settings: Settings) -> LlmClient:
    key = settings.anthropic_api_key
    # No key in .env: the SDK falls back to ANTHROPIC_API_KEY or an `ant auth login` profile.
    sdk = anthropic.Anthropic(api_key=key.get_secret_value()) if key else anthropic.Anthropic()
    return LlmClient(
        sdk,
        model=settings.lab_model,
        effort=settings.lab_effort,
        max_tokens=settings.lab_max_tokens,
        max_cost_per_call_usd=settings.lab_max_cost_per_call_usd,
    )


def build_analyzer(settings: Settings, resume_path: Path) -> FitAnalyzer:
    return FitAnalyzer(build_llm(settings), load_resume(resume_path))


def print_usage(usages: list[TokenUsage], cost: Cost) -> None:
    for u in usages:
        print(
            f"tokens [{u.model}] input={u.input_tokens} cache_write={u.cache_write_5m_tokens}"
            f" cache_read={u.cache_read_tokens} output={u.output_tokens}"
        )
    print(
        f"cost USD total={cost.total:.6f} (input={cost.input:.6f}"
        f" cache_write={cost.cache_write:.6f} cache_read={cost.cache_read:.6f}"
        f" output={cost.output:.6f})"
    )


def print_result(result: LlmResult[JobFit], call: int) -> None:
    fit = result.output
    print(f"\n=== call {call} · served by {result.served_by} · request {result.request_id}")
    print(f"{fit.role_title} ({fit.seniority_required}) -> match {fit.match_score}/100")
    print("matched:", ", ".join(m.skill for m in fit.matched) or "-")
    print("missing:", ", ".join(fit.missing) or "-")
    print("summary:", fit.summary)
    print_usage(result.usages, result.cost)


def run_fit(args: argparse.Namespace, settings: Settings) -> None:
    analyzer = build_analyzer(settings, args.resume)
    posting = args.posting.read_text(encoding="utf-8")
    tokens, worst = analyzer.estimate(posting)
    print(f"model={settings.lab_model} effort={settings.lab_effort} input_tokens={tokens}")
    print(f"worst-case cost of one call: ${worst} (limit ${settings.lab_max_cost_per_call_usd})")
    if args.command == "fit":
        for i in range(1, args.repeat + 1):
            print_result(analyzer.analyze(posting), i)


def run_ingest(args: argparse.Namespace, settings: Settings) -> None:
    # Imported here so stage 1 commands do not need the RAG dependencies loaded.
    from ai_lab.rag import load_corpus
    from ai_lab.rag.store import ingest, make_embeddings, make_pg_store

    chunks = load_corpus(args.corpus)
    store = make_pg_store(settings, make_embeddings(settings), recreate=True)
    print(f"ingested {ingest(store, chunks)} chunks into table '{settings.lab_pg_table}'")


def run_ask(args: argparse.Namespace, settings: Settings) -> None:
    from ai_lab.rag import ResumeAssistant, Retriever
    from ai_lab.rag.store import make_embeddings, make_pg_store

    store = make_pg_store(settings, make_embeddings(settings))
    retriever = Retriever(store, k=settings.lab_rag_k, min_relevance=settings.lab_rag_min_relevance)

    if args.show_context:
        print(f"retrieval (min relevance {settings.lab_rag_min_relevance}):")
        for r in retriever.search(args.question):
            mark = "use " if r.relevance >= retriever.min_relevance else "skip"
            print(f"  [{mark}] {r.relevance:.3f}  {r.document.metadata['breadcrumb']}")

    answer = ResumeAssistant(retriever, build_llm(settings)).ask(args.question)
    print(f"\n[{answer.outcome}] {answer.text}")
    for i, c in enumerate(answer.citations, 1):
        print(f'  [{i}] {c.title}: "{c.cited_text.strip()[:160]}"')
    if answer.usages:
        print_usage(answer.usages, answer.cost)
    else:
        print("cost USD total=0 (refused before calling the LLM)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai-lab")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("estimate", "fit"):
        p = sub.add_parser(name, help="stage 1: job fit analysis")
        p.add_argument("posting", type=Path)
        p.add_argument("--resume", type=Path, default=DEFAULT_RESUME)
        if name == "fit":
            p.add_argument("--repeat", type=int, default=1)
    p = sub.add_parser("ingest", help="stage 2: index the resume into pgvector")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p = sub.add_parser("ask", help="stage 2: ask a question about the resume")
    p.add_argument("question")
    p.add_argument("--show-context", action="store_true")
    args = parser.parse_args(argv)

    settings = Settings()
    handlers = {"estimate": run_fit, "fit": run_fit, "ingest": run_ingest, "ask": run_ask}
    try:
        handlers[args.command](args, settings)
    except (LlmError, anthropic.APIError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
