"""Google Gemini."""
from __future__ import annotations

from app.config import credential
from app.providers import ProviderConfigError, ProviderTransientError, register
from app.providers.translation import _LLMTranslator


@register("translation", "gemini")
class Gemini(_LLMTranslator):
    def __init__(self, model: str = "gemini-2.0-flash"):
        # either name works; GEMINI_API_KEY wins when both are set
        self._key = credential("GEMINI_API_KEY") or credential("GOOGLE_API_KEY")
        if not self._key:
            raise ProviderConfigError(
                "gemini translation requires GEMINI_API_KEY: set the environment "
                "variable or mount it as a secret at $SECRETS_DIR/gemini_api_key")
        self._model = model

    def _complete(self, prompt: str) -> str:
        import httpx
        resp = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent",
            params={"key": self._key},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"temperature": 0.2,
                                       "responseMimeType": "application/json"}},
            timeout=180)
        if resp.status_code >= 500:
            raise ProviderTransientError(f"gemini {resp.status_code}")
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
