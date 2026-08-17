"""Coqui XTTS-v2 voice cloning: offline, ~2GB model, slow on CPU."""
from __future__ import annotations

from pathlib import Path

from app.providers import ProviderConfigError, register, require_credential
from app.providers.tts import TTS, Voice


@register("tts", "xtts")
class CoquiXTTS(TTS):
    """Coqui XTTS-v2 voice cloning: offline, ~2GB model, slow on CPU.
    `voice` is a reference wav of the speaker."""

    def __init__(self, language: str = "en"):
        require_credential("COQUI_TOS_AGREED", "xtts (CPML licence acceptance)")
        try:
            from TTS.api import TTS as CoquiAPI     # the py3.12-compatible fork
        except ImportError as e:
            raise ProviderConfigError("xtts requires `pip install coqui-tts`") from e
        self._tts = CoquiAPI("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
        self._language = language

    def synthesize(self, text: str, voice: str, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        self._tts.tts_to_file(text=text, speaker_wav=voice,
                              language=self._language, file_path=str(out))

    def voices(self, language: str) -> list[Voice]:
        return []       # cloning: voices are reference wavs, not a catalog
