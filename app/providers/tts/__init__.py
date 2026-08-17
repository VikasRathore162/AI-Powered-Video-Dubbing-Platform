"""Voice generation providers, plus the voice catalog and speaker->voice casting.

Each speaker gets a distinct, gender-matched voice and keeps it (the mapping is
persisted on the speaker row), which is how speaker identity survives dubbing
without cloning. Cloning providers (xtts, openvoice) take a reference wav as the
`voice` argument instead.

Keep this file free of provider imports — the registry loads
app.providers.tts.<name> only when that provider is selected.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Voice:
    id: str
    gender: str | None = None
    locale: str | None = None


class TTS(ABC):
    @abstractmethod
    def synthesize(self, text: str, voice: str, out: Path) -> None: ...

    @abstractmethod
    def voices(self, language: str) -> list[Voice]: ...
# Curated, gender-alternating voices for the common languages. Anything else
# falls back to querying the provider's own voice list.
CATALOG: dict[str, list[Voice]] = {
    "en": [Voice("en-US-GuyNeural", "M"), Voice("en-US-JennyNeural", "F"),
           Voice("en-GB-RyanNeural", "M"), Voice("en-GB-SoniaNeural", "F")],
    "es": [Voice("es-ES-AlvaroNeural", "M"), Voice("es-ES-ElviraNeural", "F"),
           Voice("es-MX-JorgeNeural", "M"), Voice("es-MX-DaliaNeural", "F")],
    "fr": [Voice("fr-FR-HenriNeural", "M"), Voice("fr-FR-DeniseNeural", "F"),
           Voice("fr-CA-AntoineNeural", "M"), Voice("fr-CA-SylvieNeural", "F")],
    "de": [Voice("de-DE-ConradNeural", "M"), Voice("de-DE-KatjaNeural", "F")],
    "it": [Voice("it-IT-DiegoNeural", "M"), Voice("it-IT-ElsaNeural", "F")],
    "pt": [Voice("pt-BR-AntonioNeural", "M"), Voice("pt-BR-FranciscaNeural", "F")],
    "hi": [Voice("hi-IN-MadhurNeural", "M"), Voice("hi-IN-SwaraNeural", "F")],
    "ja": [Voice("ja-JP-KeitaNeural", "M"), Voice("ja-JP-NanamiNeural", "F")],
    "ko": [Voice("ko-KR-InJoonNeural", "M"), Voice("ko-KR-SunHiNeural", "F")],
    "zh": [Voice("zh-CN-YunxiNeural", "M"), Voice("zh-CN-XiaoxiaoNeural", "F")],
    "ar": [Voice("ar-SA-HamedNeural", "M"), Voice("ar-SA-ZariyahNeural", "F")],
    "ru": [Voice("ru-RU-DmitryNeural", "M"), Voice("ru-RU-SvetlanaNeural", "F")],
    "nl": [Voice("nl-NL-MaartenNeural", "M"), Voice("nl-NL-ColetteNeural", "F")],
}


def assign_voices(genders: list[str | None], catalog: list[Voice]) -> list[str]:
    """Map speakers (ordered by first appearance) to voices.

    Gender match wins over uniqueness: an unused matching voice first, then a
    *reused* matching voice, and only if the language offers none of that gender
    does it fall back to another. Hearing a man dub a woman is more jarring than
    two minor speakers sharing a voice — and it happened on a real interview
    before this rule existed. Deterministic, so a retry re-casts identically.
    """
    if not catalog:
        raise ValueError("empty voice catalog")
    used: set[str] = set()
    out = []
    for i, gender in enumerate(genders):
        matching = [v for v in catalog
                    if gender is None or v.gender is None or v.gender == gender]
        pool = matching or catalog
        pick = next((v for v in pool if v.id not in used), None) or pool[i % len(pool)]
        used.add(pick.id)
        out.append(pick.id)
    return out
