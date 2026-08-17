"""OpenAI hosted Whisper."""
from __future__ import annotations

from pathlib import Path

from app.providers import ProviderError, register, require_credential
from app.providers.stt import STT, Segment, Transcript


@register("stt", "openai_whisper")
class OpenAIWhisper(STT):
    """OpenAI hosted Whisper."""

    def __init__(self, model: str = "whisper-1"):
        self._key = require_credential("OPENAI_API_KEY", "openai_whisper")
        self._model = model

    def transcribe(self, wav: Path, language: str | None = None) -> Transcript:
        import httpx
        with open(wav, "rb") as f:
            resp = httpx.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self._key}"},
                data={"model": self._model, "response_format": "verbose_json",
                      "timestamp_granularities[]": "segment",
                      **({"language": language} if language else {})},
                files={"file": (wav.name, f, "audio/wav")}, timeout=300)
        resp.raise_for_status()
        body = resp.json()
        segments = [Segment(s["start"], s["end"], s["text"].strip())
                    for s in body.get("segments", []) if s.get("text", "").strip()]
        if not segments:
            raise ProviderError("no speech detected in audio")
        return Transcript(body.get("language", language or "en"), segments)
