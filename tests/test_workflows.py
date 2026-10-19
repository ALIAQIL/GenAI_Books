from pathlib import Path

from bookwise.rag import _chunks
from bookwise.workflows import generate_qa, summarize, translate


class FakeGenerator:
    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "generated result"


def test_summary_workflow_is_offline_testable(tmp_path: Path):
    source = tmp_path / "Atomic Habits.txt"
    source.write_text("Small actions compound over time.", encoding="utf-8")
    generator = FakeGenerator()

    artifact = summarize(
        source,
        tmp_path / "outputs",
        "flash",
        generator,
        document_type="report",
    )

    assert artifact.path == tmp_path / "outputs" / "atomic_habits" / "flash.md"
    assert artifact.path.read_text(encoding="utf-8") == "generated result\n"
    assert "Small actions compound" in generator.prompts[0]
    assert "report" in generator.prompts[0]


def test_translation_and_qa_use_book_specific_outputs(tmp_path: Path):
    source = tmp_path / "book.txt"
    source.write_text("Important content.", encoding="utf-8")
    generator = FakeGenerator()

    translation = translate(source, tmp_path / "outputs", "fr", generator)
    qa = generate_qa(source, tmp_path / "outputs", generator)

    assert translation.path.name == "translation_fr.md"
    assert qa.path.name == "qa.md"
    assert len(generator.prompts) == 2


def test_rag_chunking_creates_overlapping_chunks():
    chunks = _chunks("abcdefghij", size=6, overlap=2)
    assert chunks == ["abcdef", "efghij", "ij"]
