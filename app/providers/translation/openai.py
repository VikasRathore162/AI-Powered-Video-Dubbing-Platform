"""OpenAI chat completions."""
from __future__ import annotations

from app.providers import ProviderTransientError, register, require_credential
from app.providers.translation import _LLMTranslator


@register("translation", "openai")
class OpenAI(_LLMTranslator):
    def __init__(self, model: str = "gpt-4o-mini"):
        self._key = require_credential("OPENAI_API_KEY", "openai translation")
        self._model = model

    def _complete(self, prompt: str) -> str:
        import httpx
        resp = httpx.post("https://api.openai.com/v1/chat/completions",
                          headers={"Authorization": f"Bearer {self._key}"},
                          json={"model": self._model, "temperature": 0.2,
                                "messages": [{"role": "user", "content": prompt}]},
                          timeout=180)
        if resp.status_code >= 500:
            raise ProviderTransientError(f"openai {resp.status_code}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
