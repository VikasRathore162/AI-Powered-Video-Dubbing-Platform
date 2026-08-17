"""Translation providers, plus the shared base for the LLM-backed ones.

Keep this file free of provider imports — the registry loads
app.providers.translation.<name> only when that provider is selected.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

from app.providers import ProviderError


class Translator(ABC):
    @abstractmethod
    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]: ...
class _LLMTranslator(Translator):
    """Shared prompt and validation for the LLM-backed translators. They differ
    only in how the request is made, so only `_complete` is implemented below."""

    def _prompt(self, texts: list[str], src: str, tgt: str) -> str:
        return (f"Translate this dubbing transcript from {src} to {tgt}. It is a "
                f"conversation between speakers: preserve tone, register and "
                f"conversational flow, and keep each line close to the source "
                f"length because it is dubbed over a fixed time slot. Reply with "
                f"ONLY a JSON array of the translated lines, same length and "
                f"order.\n" + json.dumps(texts, ensure_ascii=False))

    def _complete(self, prompt: str) -> str:
        raise NotImplementedError

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        raw = self._complete(self._prompt(texts, src, tgt)).strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```")
        out = json.loads(raw)
        if isinstance(out, dict):          # some models wrap the array
            out = next((v for v in out.values() if isinstance(v, list)), [])
        if not isinstance(out, list) or len(out) != len(texts):
            raise ProviderError(
                f"{type(self).__name__} returned {len(out)} lines for {len(texts)} inputs")
        return [str(t) for t in out]
