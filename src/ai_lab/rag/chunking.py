"""Turn markdown documents into chunks that make sense on their own.

Analogy: indexing a book. You index by chapter and section, not "every 500 characters",
or an entry would start in the middle of a sentence about another job.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS = [("#", "title"), ("##", "section"), ("###", "entry")]
# A resume entry is ~300-900 characters. Longer sections are split with some overlap so a
# sentence cut at the border still appears whole in one of the two chunks.
MAX_CHARS = 1200
OVERLAP = 150

# Fixed namespace: the same chunk gets the same id on every run (uuid5 is deterministic).
_NAMESPACE = uuid.UUID("6f1b2a57-3c1e-4a8e-9d0c-2b7e5f0a9c11")


def breadcrumb(metadata: dict[str, str]) -> str:
    parts = [metadata[k] for k in ("section", "entry") if metadata.get(k)]
    return " > ".join(parts) or metadata.get("title", "")


def chunk_markdown(text: str, source: str) -> list[Document]:
    by_header = MarkdownHeaderTextSplitter(HEADERS, strip_headers=True).split_text(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHARS, chunk_overlap=OVERLAP)

    chunks: list[Document] = []
    for section in by_header:
        crumb = breadcrumb(section.metadata)
        for i, piece in enumerate(splitter.split_text(section.page_content)):
            # Contextual header: "Experience > Senior Software Engineer — ..." travels with the
            # text. Without it, a chunk saying "Led code reviews" has no idea which job it was.
            content = f"{crumb}\n{piece}" if crumb else piece
            digest = hashlib.sha256(content.encode()).hexdigest()[:16]
            metadata = {
                **section.metadata,
                "source": source,
                "breadcrumb": crumb,
                "part": i,
                "content_hash": digest,
            }
            chunk_id = str(uuid.uuid5(_NAMESPACE, f"{source}|{crumb}|{i}|{digest}"))
            chunks.append(Document(id=chunk_id, page_content=content, metadata=metadata))
    return chunks


def load_corpus(folder: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(folder.glob("*.md")):
        docs += chunk_markdown(path.read_text(encoding="utf-8"), source=path.name)
    return docs
