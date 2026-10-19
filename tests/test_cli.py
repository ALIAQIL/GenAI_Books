from bookwise.cli import build_parser


def test_cli_parser_supports_summary_options():
    args = build_parser().parse_args(["summarize", "book.pdf", "--style", "executive"])
    assert args.command == "summarize"
    assert args.style == "executive"


def test_cli_parser_supports_rag_question():
    args = build_parser().parse_args(["ask", "data/books", "--question", "What is grit?"])
    assert args.source_dir.name == "books"
    assert args.question == "What is grit?"


def test_cli_parser_supports_general_document_types():
    args = build_parser().parse_args(
        ["summarize", "report.pdf", "--type", "research paper", "--style", "executive"]
    )
    assert args.document_type == "research paper"
