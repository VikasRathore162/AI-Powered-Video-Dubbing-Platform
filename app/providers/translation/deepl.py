"""DeepL."""
from __future__ import annotations

from app.providers import ProviderTransientError, register, require_credential
from app.providers.translation import Translator


@register("translation", "deepl")
class DeepL(Translator):
    def __init__(self, endpoint: str = "https://api-free.deepl.com"):
        self._key = require_credential("DEEPL_API_KEY", "deepl")
        self._endpoint = endpoint

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        import httpx
        resp = httpx.post(f"{self._endpoint}/v2/translate",
                          headers={"Authorization": f"DeepL-Auth-Key {self._key}"},
                          json={"text": texts, "source_lang": src.upper(),
                                "target_lang": tgt.upper()}, timeout=120)
        if resp.status_code >= 500:
            raise ProviderTransientError(f"deepl {resp.status_code}")
        resp.raise_for_status()
        return [t["text"] for t in resp.json()["translations"]]
