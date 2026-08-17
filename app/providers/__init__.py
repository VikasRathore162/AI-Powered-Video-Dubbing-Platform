"""Provider registry — selection, fallback and retry.

This file holds no interfaces and no implementations. Each capability owns its
own folder: the interface lives in that folder's `__init__.py`, and every
implementation is a separate file beside it:

    app/providers/stt/__init__.py           the STT interface + its data types
    app/providers/stt/faster_whisper.py     providers.stt.name = faster_whisper
    app/providers/stt/deepgram.py           providers.stt.name = deepgram
    app/providers/translation/claude.py     providers.translation.name = claude

The file name IS the configured name, and the registry imports that module the
first time it is asked for — so adding a vendor is: write one file, name it
after the vendor, subclass the interface, @register it, declare its credential.
Nothing else changes, and switching vendors stays a config edit:

    providers:
      stt: {name: deepgram}
"""
from __future__ import annotations

import pkgutil
import time
from importlib import import_module
from typing import Any, Callable

from app.config import credential, get_settings
from app.obs import get_logger

log = get_logger("providers")


class ProviderError(Exception):
    """Permanent provider failure."""


class ProviderTransientError(ProviderError):
    """Retryable failure (network hiccup, 5xx, timeout)."""


class ProviderConfigError(ProviderError):
    """Provider selected but unusable (missing credential or package)."""


def require_credential(name: str, provider: str) -> str:
    """Fail at construction naming what is missing, so a misconfigured provider
    is caught at startup rather than deep inside a job."""
    value = credential(name)
    if not value:
        raise ProviderConfigError(
            f"{provider} requires {name}: set the environment variable "
            f"or mount it as a secret at $SECRETS_DIR/{name.lower()}")
    return value


# The capability folders. Each one owns its interface (app/providers/<kind>/
# __init__.py) and holds one file per implementation.
KINDS = ("stt", "diarization", "translation", "tts", "storage")

_REGISTRY: dict[str, dict[str, type]] = {k: {} for k in KINDS}
_INSTANCES: dict[tuple, Any] = {}


def register(kind: str, name: str):
    def deco(cls):
        _REGISTRY[kind][name] = cls
        return cls
    return deco


def available(kind: str) -> list[str]:
    """Every provider offered for `kind`: one per file in app/providers/<kind>/,
    plus anything registered at runtime (the test fakes)."""
    pkg = import_module(f"{__name__}.{kind}")
    files = {m.name for m in pkgutil.iter_modules(pkg.__path__)
             if not m.name.startswith("_")}
    return sorted(files | set(_REGISTRY[kind]))


def clear_instances() -> None:
    """Test hook — providers are memoized so heavy models load once per worker."""
    _INSTANCES.clear()


def _load(kind: str, name: str) -> type:
    """Adding a provider means adding app/providers/<kind>/<name>.py — nothing
    else. The module is imported the first time it is selected, so an unused
    provider never costs its dependencies (torch, boto3, an SDK) at startup."""
    if name in _REGISTRY[kind]:          # already imported, or a test fake
        return _REGISTRY[kind][name]
    if name not in available(kind):
        raise ProviderConfigError(
            f"unknown {kind} provider '{name}'; available: {available(kind)}")
    import_module(f"{__name__}.{kind}.{name}")   # runs @register
    return _REGISTRY[kind][name]


def _build(kind: str, name: str, options: dict) -> Any:
    key = (kind, name, tuple(sorted(options.items(), key=str)))
    if key not in _INSTANCES:
        _INSTANCES[key] = _load(kind, name)(**options)
    return _INSTANCES[key]


def get_provider(kind: str) -> Any:
    """The configured provider for `kind`, memoized per process."""
    cfg = getattr(get_settings().providers, kind)
    primary = _build(kind, cfg.name, cfg.options)
    if not cfg.fallbacks:
        return primary
    return Fallback(kind, primary, cfg.fallbacks)


def with_retries(fn: Callable, retries: int = 3, backoff: float = 1.0):
    for attempt in range(retries + 1):
        try:
            return fn()
        except TRANSIENT as e:
            if attempt == retries:
                raise
            wait = backoff * (2 ** attempt)
            log.warning("provider_retry", error=str(e), attempt=attempt + 1, wait=wait)
            time.sleep(wait)


class Fallback:
    """Delegates to the primary; on failure tries each fallback in turn,
    instantiating it only then so a credentialed fallback can't break a healthy
    primary just by being listed."""

    def __init__(self, kind: str, primary, fallback_names: list[str]):
        self._kind, self._primary, self._names = kind, primary, fallback_names

    def __getattr__(self, attr):
        def call(*args, **kwargs):
            last: Exception | None = None
            for provider in [self._primary, *self._names]:
                try:
                    if isinstance(provider, str):
                        provider = _build(self._kind, provider, {})
                    return getattr(provider, attr)(*args, **kwargs)
                except Exception as e:  # noqa: BLE001 - fall through the chain
                    log.warning("provider_fallback", kind=self._kind,
                                method=attr, error=str(e))
                    last = e
            raise last  # type: ignore[misc]
        return call


def _transient_types() -> tuple:
    """Network errors worth retrying. httpx errors subclass neither
    ConnectionError nor OSError, so name them explicitly."""
    types = [ProviderTransientError, ConnectionError, TimeoutError, OSError]
    try:
        import httpx
        types.append(httpx.TransportError)
    except ImportError:
        pass
    return tuple(types)


TRANSIENT = _transient_types()
