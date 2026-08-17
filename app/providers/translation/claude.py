"""Anthropic Claude."""
from __future__ import annotations

from app.providers import (ProviderConfigError, ProviderError, ProviderTransientError,
                           register, require_credential)
from app.providers.translation import _LLMTranslator


@register("translation", "claude")
class Claude(_LLMTranslator):
    def __init__(self, model: str = "claude-opus-5", max_tokens: int = 16000):
        self._key = require_credential("ANTHROPIC_API_KEY", "claude translation")
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ProviderConfigError("claude requires `pip install anthropic`") from e
        self._model, self._max_tokens = model, max_tokens

    def _complete(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self._key)
        try:
            resp = client.messages.create(
                model=self._model, max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}])
        except (anthropic.RateLimitError, anthropic.APIConnectionError) as e:
            raise ProviderTransientError(f"claude: {e}") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise ProviderTransientError(f"claude {e.status_code}") from e
            raise ProviderError(f"claude {e.status_code}: {e.message}") from e
        if resp.stop_reason == "refusal":
            raise ProviderError("claude declined to translate this content")
        return next((b.text for b in resp.content if b.type == "text"), "")
