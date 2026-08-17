"""Azure Speech. Speaks SSML, so this is the provider that carries emotional
tone via mstts:express-as styles."""
from __future__ import annotations

from pathlib import Path

from app.providers import (ProviderError, ProviderTransientError, register,
                           require_credential, with_retries)
from app.providers.tts import TTS, Voice


@register("tts", "azure")
class AzureSpeech(TTS):
    """Azure Speech. Speaks SSML, so this is the provider that can carry
    EMOTIONAL TONE: set options.style to an express-as style ("cheerful",
    "sad", "excited", ...) with an optional style_degree."""

    def __init__(self, style: str | None = None, style_degree: float = 1.0,
                 rate: str = "0%", audio_format: str = "audio-24khz-48kbitrate-mono-mp3"):
        self._key = require_credential("AZURE_SPEECH_KEY", "azure tts")
        self._region = require_credential("AZURE_SPEECH_REGION", "azure tts")
        self._style, self._style_degree = style, style_degree
        self._rate, self._format = rate, audio_format

    def _url(self, path: str) -> str:
        return f"https://{self._region}.tts.speech.microsoft.com/cognitiveservices/{path}"

    def synthesize(self, text: str, voice: str, out: Path) -> None:
        from xml.sax.saxutils import escape

        import httpx
        out.parent.mkdir(parents=True, exist_ok=True)
        lang = "-".join(voice.split("-")[:2]) if "-" in voice else "en-US"
        body = f'<prosody rate="{self._rate}">{escape(text)}</prosody>'
        if self._style:
            body = (f'<mstts:express-as style="{self._style}" '
                    f'styledegree="{self._style_degree}">{body}</mstts:express-as>')
        ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
                f'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{lang}">'
                f'<voice name="{voice}">{body}</voice></speak>')

        def once():
            resp = httpx.post(self._url("v1"),
                              headers={"Ocp-Apim-Subscription-Key": self._key,
                                       "Content-Type": "application/ssml+xml",
                                       "X-Microsoft-OutputFormat": self._format},
                              content=ssml.encode(), timeout=120)
            if resp.status_code >= 500 or resp.status_code == 429:
                raise ProviderTransientError(f"azure tts {resp.status_code}")
            resp.raise_for_status()
            out.write_bytes(resp.content)

        with_retries(once, retries=3, backoff=1.0)

    def voices(self, language: str) -> list[Voice]:
        import httpx
        resp = httpx.get(self._url("voices/list"),
                         headers={"Ocp-Apim-Subscription-Key": self._key}, timeout=60)
        resp.raise_for_status()
        lang = language.split("-")[0].lower()
        matches = [Voice(v["ShortName"], "M" if v.get("Gender") == "Male" else "F",
                         v.get("Locale"))
                   for v in resp.json() if v.get("Locale", "").lower().startswith(lang)]
        if not matches:
            raise ProviderError(f"no azure voice for language '{language}'")
        return matches
