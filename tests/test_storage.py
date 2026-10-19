from pathlib import Path

import pytest

from bookwise.storage import read_source, slugify, write_artifact


def test_slugify_creates_stable_book_names():
    assert slugify("Atomic Habits: An Easy Way") == "atomic_habits_an_easy_way"


def test_read_source_reads_utf8_text(tmp_path: Path):
    source = tmp_path / "book.txt"
    source.write_text("A useful idea.", encoding="utf-8")
    document = read_source(source)
    assert document.text == "A useful idea."
    assert document.book_name == "book"


def test_read_source_rejects_empty_files(tmp_path: Path):
    source = tmp_path / "empty.txt"
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no extractable text"):
        read_source(source)


def test_write_artifact_creates_parent_directory(tmp_path: Path):
    artifact = write_artifact(tmp_path / "book", "summary.md", "Done", "summary", "book")
    assert artifact.path.read_text(encoding="utf-8") == "Done\n"
