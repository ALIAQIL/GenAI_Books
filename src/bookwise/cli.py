"""Command-line interface for the Bookwise document intelligence toolkit."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import ConfigurationError
from .rag import answer_question
from .speech import text_to_speech
from .workflows import generate_qa, summarize, translate

LOGGER = logging.getLogger("bookwise")


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gemini-powered document intelligence workflows")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = _common_parser()
    commands = parser.add_subparsers(dest="command", required=True)

    summary = commands.add_parser("summarize", help="Generate a summary from a document")
    summary.add_argument("source", type=Path)
    summary.add_argument("--style", choices=["flash", "chapters", "executive"], default="flash")
    summary.add_argument(
        "--type",
        dest="document_type",
        default="document",
        help="Document type used to tailor the prompt, e.g. report, paper, manual, or book",
    )
    summary.add_argument("--output-root", type=Path, default=Path("outputs"))

    translation = commands.add_parser("translate", help="Translate an existing summary")
    translation.add_argument("source", type=Path)
    translation.add_argument("--language", choices=["fr", "ar"], required=True)
    translation.add_argument("--output-root", type=Path, default=Path("outputs"))

    qa = commands.add_parser("qa", help="Generate question-and-answer pairs from a document")
    qa.add_argument("source", type=Path)
    qa.add_argument("--type", dest="document_type", default="document")
    qa.add_argument("--output-root", type=Path, default=Path("outputs"))

    rag = commands.add_parser("ask", help="Ask a question across a document directory")
    rag.add_argument("source_dir", type=Path)
    rag.add_argument("--question", required=True)
    rag.add_argument("--top-k", type=int, default=5)

    speech = commands.add_parser("speak", help="Convert Markdown/text to audio")
    speech.add_argument("source", type=Path)
    speech.add_argument("--output", type=Path, required=True)
    speech.add_argument("--language", default="en")
    speech.add_argument("--play", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        if args.command == "summarize":
            artifact = summarize(
                args.source,
                args.output_root,
                args.style,
                document_type=args.document_type,
            )
            print(f"Created {artifact.path}")
        elif args.command == "translate":
            artifact = translate(args.source, args.output_root, args.language)
            print(f"Created {artifact.path}")
        elif args.command == "qa":
            artifact = generate_qa(
                args.source,
                args.output_root,
                document_type=args.document_type,
            )
            print(f"Created {artifact.path}")
        elif args.command == "ask":
            print(answer_question(args.source_dir, args.question, top_k=args.top_k))
        elif args.command == "speak":
            print(f"Created {text_to_speech(args.source, args.output, args.language, args.play)}")
        return 0
    except (ConfigurationError, FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2
