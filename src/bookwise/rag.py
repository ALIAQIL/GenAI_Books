"""Retrieval-augmented question answering over a document directory."""

from __future__ import annotations

from pathlib import Path

from .config import Settings
from .prompts import RAG_PROMPT
from .provider import GeminiGenerator, TextGenerator
from .storage import read_source

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def _chunks(text: str, size: int = 1800, overlap: int = 250) -> list[str]:
    """Split long documents into overlapping chunks for better retrieval."""

    if overlap >= size:
        raise ValueError("Chunk overlap must be smaller than chunk size")
    step = size - overlap
    return [text[start : start + size] for start in range(0, len(text), step)]


def answer_question(
    source_dir: Path,
    question: str,
    settings: Settings | None = None,
    generator: TextGenerator | None = None,
    top_k: int = 5,
) -> str:
    """Retrieve relevant chunks from supported documents and answer with Gemini."""

    if not question.strip():
        raise ValueError("Question cannot be empty")
    sources = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not sources:
        raise FileNotFoundError(f"No PDF, Markdown, or text files found in {source_dir}")

    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    documents = [
        Document(page_content=chunk, metadata={"source": path.name})
        for path in sources
        for chunk in _chunks(read_source(path).text)
    ]
    active_settings = settings or Settings.from_environment()
    active_settings.require_api_key()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=active_settings.embedding_model,
        google_api_key=active_settings.api_key,
    )
    database = FAISS.from_documents(documents, embeddings)
    matches = database.similarity_search(question, k=top_k)
    context = "\n\n".join(match.page_content for match in matches)
    return (generator or GeminiGenerator(active_settings)).generate(
        RAG_PROMPT.format(context=context, question=question)
    )
