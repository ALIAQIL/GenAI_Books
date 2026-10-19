"""Small domain types shared by CLI and workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SummaryStyle = Literal["flash", "chapters", "executive"]
TranslationLanguage = Literal["fr", "ar"]


@dataclass(frozen=True)
class GeneratedArtifact:
    """A generated file and the source operation that produced it."""

    path: Path
    operation: str
    book: str


@dataclass(frozen=True)
class SourceDocument:
    """Extracted text and metadata from a source file."""

    path: Path
    text: str

    @property
    def book_name(self) -> str:
        return self.path.stem
