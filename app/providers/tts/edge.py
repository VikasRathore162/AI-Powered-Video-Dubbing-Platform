"""edge-tts: free Microsoft neural voices, no API key. The default."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from app.obs import get_logger
from app.providers import ProviderError, ProviderTransientError, register, with_retries
from app.providers.tts import CATALOG, TTS, Voice

log = get_logger("tts")


@register("tts", "edge")
class EdgeTTS(TTS):
    def __init__(self, rate: str = "+0%", volume: str = "+0%",
                 voice_cache: str = "~/.cache/dubbing/edge_voices.json"):
        self._rate, self._volume = rate, volume
        self._cache = Path(os.path.expanduser(voice_cache))

    def synthesize(self, text: str, voice: str, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)

        def once():
            try:
                asyncio.run(self._speak(text, voice, out))
            except (ConnectionError, TimeoutError, OSError):
                raise
            except Exception as e:      # aiohttp / edge-tts: it's a network service
                raise ProviderTransientError(f"edge-tts failed: {e}") from e
            if not out.exists() or out.stat().st_size == 0:
                raise ProviderTransientError("edge-tts produced empty output")

        with_retries(once, retries=3, backoff=1.0)

    async def _speak(self, text: str, voice: str, out: Path) -> None:
        import edge_tts
        await edge_tts.Communicate(text, voice, rate=self._rate,
                                   volume=self._volume).save(str(out))

    def voices(self, language: str) -> list[Voice]:
        """Curated voices first, then everything else the service offers.

        The curated few are the good, gender-balanced picks, so a normal two-speaker
        job is cast from them deterministically. The tail matters when a recording
        has several speakers of the same gender — edge-tts publishes ~20 per gender
        for the major languages, so nobody has to be dubbed by the wrong one. If the
        catalogue can't be fetched the curated list still works, so this never turns
        a network blip into a failed job.
        """
        lang = language.split("-")[0].lower()
        curated = CATALOG.get(lang, [])
        try:
            extra = self._published(lang)
        except Exception as e:                      # offline, or the list API moved
            log.warning("voice_catalogue_unavailable", language=lang, error=str(e))
            extra = []
        seen = {v.id for v in curated}
        voices = curated + [v for v in extra if v.id not in seen]
        if not voices:
            raise ProviderError(f"no edge-tts voice for language '{language}'")
        return voices

    def _published(self, lang: str) -> list[Voice]:
        if not self._cache.exists():
            import edge_tts
            self._cache.parent.mkdir(parents=True, exist_ok=True)
            self._cache.write_text(json.dumps(asyncio.run(edge_tts.list_voices())))
        return [Voice(v["ShortName"], "M" if v.get("Gender") == "Male" else "F",
                      v.get("Locale"))
                for v in json.loads(self._cache.read_text())
                if v.get("Locale", "").lower().startswith(lang)]
