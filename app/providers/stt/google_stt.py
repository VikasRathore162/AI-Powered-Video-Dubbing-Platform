"""Google Cloud Speech-to-Text v2."""
from __future__ import annotations

from pathlib import Path

from app.providers import (ProviderError, ProviderTransientError, register,
                           require_credential)
from app.providers.stt import STT, Segment, Transcript


@register("stt", "google_stt")
class GoogleSTT(STT):
    """Google Cloud Speech-to-Text v2, inline recognize. The API caps inline
    audio (~1 min / 10MB); longer media needs the batchRecognize + GCS flow."""

    def __init__(self, model: str = "long", location: str = "global",
                 max_inline_bytes: int = 10 * 1024 * 1024):
        self._key = require_credential("GOOGLE_API_KEY", "google_stt")
        self._project = require_credential("GOOGLE_CLOUD_PROJECT", "google_stt")
        self._model, self._location = model, location
        self._max_inline = max_inline_bytes

    def transcribe(self, wav: Path, language: str | None = None) -> Transcript:
        import base64

        import httpx
        data = wav.read_bytes()
        if len(data) > self._max_inline:
            raise ProviderError(
                f"google_stt inline recognize supports up to {self._max_inline} "
                f"bytes; use batchRecognize with a GCS URI for longer audio")
        resp = httpx.post(
            f"https://speech.googleapis.com/v2/projects/{self._project}"
            f"/locations/{self._location}/recognizers/_:recognize",
            params={"key": self._key},
            json={"config": {"autoDecodingConfig": {}, "model": self._model,
                             "languageCodes": [language or "auto"],
                             "features": {"enableWordTimeOffsets": True}},
                  "content": base64.b64encode(data).decode()}, timeout=600)
        if resp.status_code >= 500:
            raise ProviderTransientError(f"google_stt {resp.status_code}")
        resp.raise_for_status()
        results = resp.json().get("results", [])

        def secs(v) -> float:      # protobuf durations arrive as "1.200s"
            return float(str(v).rstrip("s")) if v is not None else 0.0

        segments = []
        for r in results:
            alt = (r.get("alternatives") or [{}])[0]
            text = (alt.get("transcript") or "").strip()
            if not text:
                continue
            words = alt.get("words") or []
            start = secs(words[0].get("startOffset")) if words else 0.0
            end = secs(words[-1].get("endOffset")) if words else start + 1.0
            segments.append(Segment(start, end, text))
        if not segments:
            raise ProviderError("no speech detected in audio")
        detected = next((r.get("languageCode") for r in results
                         if r.get("languageCode")), language or "en")
        return Transcript(detected, segments)
