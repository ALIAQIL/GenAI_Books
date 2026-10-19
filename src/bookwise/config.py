"""Runtime configuration and project paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(RuntimeError):
    """Raised when the application cannot start with the supplied configuration."""


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    api_key: str | None = None
    model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    temperature: float = 0.3

    @classmethod
    def from_environment(cls) -> Settings:
        """Build settings without prompting or making a network request."""

        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        raw_temperature = os.getenv("BOOKWISE_TEMPERATURE", "0.3")
        try:
            temperature = float(raw_temperature)
        except ValueError as exc:
            raise ConfigurationError("BOOKWISE_TEMPERATURE must be a number") from exc

        return cls(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model=os.getenv("BOOKWISE_MODEL", "gemini-2.5-flash"),
            embedding_model=os.getenv("BOOKWISE_EMBEDDING_MODEL", "gemini-embedding-001"),
            temperature=temperature,
        )

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ConfigurationError(
                "GOOGLE_API_KEY is required. Copy .env.example to .env or export the variable."
            )
        return self.api_key


@dataclass(frozen=True)
class ProjectPaths:
    """Conventional locations for source books, examples, and generated results."""

    root: Path
    books: Path
    examples: Path
    outputs: Path
    indexes: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> ProjectPaths:
        project_root = (root or Path.cwd()).resolve()
        return cls(
            root=project_root,
            books=project_root / "data" / "books",
            examples=project_root / "data" / "examples",
            outputs=project_root / "outputs",
            indexes=project_root / "data" / "indexes",
        )

    def output_dir(self, book_name: str) -> Path:
        path = self.outputs / book_name
        path.mkdir(parents=True, exist_ok=True)
        return path
