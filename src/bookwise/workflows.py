"""Core generation workflows."""

from __future__ import annotations

from pathlib import Path

from .models import GeneratedArtifact, SummaryStyle, TranslationLanguage
from .prompts import QA_PROMPT, SUMMARY_PROMPTS, TRANSLATION_PROMPT
from .provider import GeminiGenerator, TextGenerator
from .storage import read_source, slugify, write_artifact

LANGUAGE_NAMES = {"fr": "French", "ar": "Arabic"}


def _generator(generator: TextGenerator | None) -> TextGenerator:
    return generator or GeminiGenerator()


def summarize(
    source_path: Path,
    output_root: Path,
    style: SummaryStyle = "flash",
    generator: TextGenerator | None = None,
    document_type: str = "document",
) -> GeneratedArtifact:
    if style not in SUMMARY_PROMPTS:
        raise ValueError(f"Unsupported summary style: {style}")
    document = read_source(source_path)
    result = _generator(generator).generate(
        SUMMARY_PROMPTS[style].format(document_type=document_type, text=document.text)
    )
    return write_artifact(
        output_root / slugify(document.book_name),
        f"{style}.md",
        result,
        "summary",
        document.book_name,
    )


def translate(
    source_path: Path,
    output_root: Path,
    language: TranslationLanguage,
    generator: TextGenerator | None = None,
) -> GeneratedArtifact:
    if language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported language: {language}")
    document = read_source(source_path)
    result = _generator(generator).generate(
        TRANSLATION_PROMPT.format(language_name=LANGUAGE_NAMES[language], text=document.text)
    )
    return write_artifact(
        output_root / slugify(document.book_name),
        f"translation_{language}.md",
        result,
        "translation",
        document.book_name,
    )


def generate_qa(
    source_path: Path,
    output_root: Path,
    generator: TextGenerator | None = None,
    document_type: str = "document",
) -> GeneratedArtifact:
    document = read_source(source_path)
    result = _generator(generator).generate(
        QA_PROMPT.format(document_type=document_type, text=document.text)
    )
    return write_artifact(
        output_root / slugify(document.book_name),
        "qa.md",
        result,
        "qa",
        document.book_name,
    )
