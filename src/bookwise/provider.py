"""LLM provider boundary. The boundary makes workflows easy to test offline."""

from __future__ import annotations

from typing import Any, Protocol

from .config import Settings


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate text for a prompt."""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in content
        )
    return str(content)


class GeminiGenerator:
    """Lazy Gemini-backed implementation of :class:`TextGenerator`."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_environment()
        self._model: Any | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self.settings.require_api_key()
            self._model = ChatGoogleGenerativeAI(
                model=self.settings.model,
                temperature=self.settings.temperature,
                google_api_key=self.settings.api_key,
            )
        return self._model

    def generate(self, prompt: str) -> str:
        response = self._get_model().invoke(prompt)
        return _content_to_text(response.content)
