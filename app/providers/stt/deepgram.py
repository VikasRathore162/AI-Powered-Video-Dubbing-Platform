"""Deepgram."""
from __future__ import annotations

from pathlib import Path

from app.providers import ProviderError, register, require_credential
from app.providers.stt import STT, Segment, Transcript


@register("stt", "deepgram")
class Deepgram(STT):
    def __init__(self, model: str = "nova-2"):
        self._key = require_credential("DEEPGRAM_API_KEY", "deepgram")
        self._model = model

    def transcribe(self, wav: Path, language: str | None = None) -> Transcript:
        import httpx
        params = {"model": self._model, "utterances": "true", "punctuate": "true",
                  "detect_language": "false" if language else "true"}
        if language:
            params["language"] = language
        resp = httpx.post("https://api.deepgram.com/v1/listen", params=params,
                          headers={"Authorization": f"Token {self._key}",
                                   "Content-Type": "audio/wav"},
                          content=wav.read_bytes(), timeout=300)
        resp.raise_for_status()
        body = resp.json()
        segments = [Segment(u["start"], u["end"], u["transcript"].strip())
                    for u in body.get("results", {}).get("utterances", [])
                    if u.get("transcript", "").strip()]
        if not segments:
            raise ProviderError("no speech detected in audio")
        detected = (body.get("results", {}).get("channels", [{}])[0]
                    .get("detected_language", language or "en"))
        return Transcript(detected, segments)
