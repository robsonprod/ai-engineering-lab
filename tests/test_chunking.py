from pathlib import Path

from ai_lab.rag import chunk_markdown, load_corpus
from ai_lab.rag.chunking import MAX_CHARS

CORPUS = Path(__file__).parent.parent / "data" / "resume"

MD = """\
# Title

## Experience

### Job A (2020 - 2021)
- Built Kafka pipelines.

### Job B (2018 - 2020)
- Led code reviews.

## Skills
Java, Python
"""


def test_each_job_becomes_its_own_chunk_with_its_header() -> None:
    chunks = chunk_markdown(MD, source="cv.md")

    crumbs = [c.metadata["breadcrumb"] for c in chunks]
    assert "Experience > Job A (2020 - 2021)" in crumbs
    assert "Experience > Job B (2018 - 2020)" in crumbs
    job_b = next(c for c in chunks if c.metadata["breadcrumb"].endswith("(2018 - 2020)"))
    # The header travels with the text: "Led code reviews" alone would not say where.
    assert job_b.page_content.startswith("Experience > Job B")
    assert "Kafka" not in job_b.page_content


def test_ids_are_stable_across_runs_and_change_with_content() -> None:
    first = [c.id for c in chunk_markdown(MD, source="cv.md")]
    again = [c.id for c in chunk_markdown(MD, source="cv.md")]
    edited = [c.id for c in chunk_markdown(MD.replace("Kafka", "RabbitMQ"), source="cv.md")]

    assert first == again
    assert len(set(first)) == len(first)
    assert first != edited


def test_long_sections_are_split_under_the_limit() -> None:
    long_md = "## Notes\n" + ("This sentence is about distributed systems. " * 100)

    chunks = chunk_markdown(long_md, source="n.md")

    assert len(chunks) > 1
    assert all(len(c.page_content) <= MAX_CHARS + len("Notes\n") for c in chunks)


def test_real_resume_has_one_chunk_per_job_and_project() -> None:
    chunks = load_corpus(CORPUS)
    crumbs = [c.metadata["breadcrumb"] for c in chunks]

    assert sum(c.startswith("Experience > ") for c in crumbs) == 6
    assert sum(c.startswith("Projects > ") for c in crumbs) == 4
    assert all(c.metadata["source"] == "resume.md" for c in chunks)
