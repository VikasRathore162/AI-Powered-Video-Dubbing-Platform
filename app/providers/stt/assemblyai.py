"""AssemblyAI."""
from __future__ import annotations

from pathlib import Path

from app.providers import (ProviderError, ProviderTransientError, register,
                           require_credential)
from app.providers.stt import STT, Segment, Transcript


@register("stt", "assemblyai")
class AssemblyAI(STT):
    """Returns speaker-labelled utterances, so it carries diarization too."""

    def __init__(self, speaker_labels: bool = True, poll_sec: float = 3.0,
                 timeout_sec: float = 1800):
        self._key = require_credential("ASSEMBLYAI_API_KEY", "assemblyai")
        self._speaker_labels = speaker_labels
        self._poll, self._timeout = poll_sec, timeout_sec

    def transcribe(self, wav: Path, language: str | None = None) -> Transcript:
        import time

        import httpx
        headers = {"authorization": self._key}
        with httpx.Client(base_url="https://api.assemblyai.com", timeout=300) as c:
            up = c.post("/v2/upload", headers=headers, content=wav.read_bytes())
            up.raise_for_status()
            body = {"audio_url": up.json()["upload_url"],
                    "speaker_labels": self._speaker_labels,
                    "language_detection": language is None}
            if language:
                body["language_code"] = language
            job = c.post("/v2/transcript", headers=headers, json=body)
            job.raise_for_status()
            tid = job.json()["id"]

            deadline = time.monotonic() + self._timeout
            while time.monotonic() < deadline:
                data = c.get(f"/v2/transcript/{tid}", headers=headers).json()
                if data["status"] == "completed":
                    break
                if data["status"] == "error":
                    raise ProviderError(f"assemblyai failed: {data.get('error')}")
                time.sleep(self._poll)
            else:
                raise ProviderTransientError("assemblyai transcription timed out")

        segments = [Segment(u["start"] / 1000, u["end"] / 1000, u["text"].strip())
                    for u in (data.get("utterances") or []) if u.get("text", "").strip()]
        if not segments:
            raise ProviderError("no speech detected in audio")
        return Transcript(data.get("language_code", language or "en"), segments)
