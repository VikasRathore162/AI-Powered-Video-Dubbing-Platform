"""ElevenLabs."""
from __future__ import annotations

from pathlib import Path

from app.providers import ProviderTransientError, register, require_credential
from app.providers.tts import TTS, Voice


@register("tts", "elevenlabs")
class ElevenLabs(TTS):
    def __init__(self, model: str = "eleven_multilingual_v2"):
        self._key = require_credential("ELEVENLABS_API_KEY", "elevenlabs")
        self._model = model

    def synthesize(self, text: str, voice: str, out: Path) -> None:
        import httpx
        out.parent.mkdir(parents=True, exist_ok=True)
        resp = httpx.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}",
                          headers={"xi-api-key": self._key},
                          json={"text": text, "model_id": self._model}, timeout=120)
        if resp.status_code >= 500:
            raise ProviderTransientError(f"elevenlabs {resp.status_code}")
        resp.raise_for_status()
        out.write_bytes(resp.content)

    def voices(self, language: str) -> list[Voice]:
        import httpx
        resp = httpx.get("https://api.elevenlabs.io/v1/voices",
                         headers={"xi-api-key": self._key}, timeout=60)
        resp.raise_for_status()
        return [Voice(v["voice_id"]) for v in resp.json().get("voices", [])]
