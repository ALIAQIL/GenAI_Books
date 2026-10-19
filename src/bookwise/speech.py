"""Text-to-speech support with optional local playback."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


def text_to_speech(
    source_path: Path,
    output_path: Path,
    language: str = "en",
    play: bool = False,
) -> Path:
    from gtts import gTTS

    text = source_path.read_text(encoding="utf-8")
    clean_text = re.sub(r"[*#]", "", text).strip()
    if not clean_text:
        raise ValueError(f"Source file contains no speakable text: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gTTS(text=clean_text, lang=language, slow=False).save(str(output_path))
    if play:
        player = shutil.which("mpg123")
        if not player:
            raise RuntimeError("Playback requested but mpg123 is not installed")
        subprocess.run([player, str(output_path)], check=True)
    return output_path
