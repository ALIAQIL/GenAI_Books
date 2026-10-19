"""Safe, predictable input and output helpers."""

from __future__ import annotations

import re
from pathlib import Path

from .models import GeneratedArtifact, SourceDocument


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def read_source(path: Path) -> SourceDocument:
    source = path.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")
    if source.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        pages = PdfReader(str(source)).pages
        text = "\n\n".join(page.extract_text() or "" for page in pages).strip()
    else:
        text = source.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Source file contains no extractable text: {source}")
    return SourceDocument(path=source, text=text)


def write_artifact(
    output_dir: Path,
    filename: str,
    content: str,
    operation: str,
    book: str,
) -> GeneratedArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return GeneratedArtifact(path=path, operation=operation, book=book)
