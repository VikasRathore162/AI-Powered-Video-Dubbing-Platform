"""The provider abstraction: registry, selection, fail-fast, fallback chain."""
from __future__ import annotations

import sys
import types

import pytest

from app.providers import (ProviderConfigError, ProviderTransientError, available,
                           clear_instances, get_provider, register, with_retries)
from app.providers.translation import Translator


@register("translation", "always_fails")
class AlwaysFails(Translator):
    calls = 0

    def translate(self, texts, src, tgt):
        AlwaysFails.calls += 1
        raise RuntimeError("primary is down")


@register("translation", "needs_credentials")
class NeedsCredentials(Translator):
    built = 0

    def __init__(self):
        NeedsCredentials.built += 1
        raise ProviderConfigError("missing API key")

    def translate(self, texts, src, tgt):
        raise AssertionError("unreachable")


def select(settings_env, kind: str, name: str, **extra):
    settings_env(**{f"PROVIDERS__{kind.upper()}__NAME": name,
                    f"PROVIDERS__{kind.upper()}__OPTIONS": "{}", **extra})
    return get_provider(kind)


def test_every_provider_named_in_the_brief_is_selectable():
    assert {"faster_whisper", "openai_whisper", "deepgram", "assemblyai",
            "google_stt"} <= set(available("stt"))
    assert {"argos", "openai", "gemini", "claude", "deepl"} <= set(available("translation"))
    assert {"edge", "elevenlabs", "azure", "xtts", "openvoice"} <= set(available("tts"))
    assert {"ecapa_cluster"} <= set(available("diarization"))
    assert {"local", "s3"} <= set(available("storage"))


def test_an_unknown_provider_names_the_alternatives(settings_env):
    with pytest.raises(ProviderConfigError) as e:
        select(settings_env, "translation", "nope")
    assert "available" in str(e.value)


@pytest.mark.parametrize("kind,name,expected", [
    ("translation", "openai", "OPENAI_API_KEY"),
    ("translation", "claude", "ANTHROPIC_API_KEY"),
    ("translation", "gemini", "GEMINI_API_KEY"),
    ("translation", "deepl", "DEEPL_API_KEY"),
    ("stt", "openai_whisper", "OPENAI_API_KEY"),
    ("stt", "deepgram", "DEEPGRAM_API_KEY"),
    ("stt", "assemblyai", "ASSEMBLYAI_API_KEY"),
    ("stt", "google_stt", "GOOGLE_API_KEY"),
    ("tts", "elevenlabs", "ELEVENLABS_API_KEY"),
    ("tts", "azure", "AZURE_SPEECH_KEY"),
    ("tts", "xtts", "COQUI_TOS_AGREED"),
    ("tts", "openvoice", "OPENVOICE_CKPT"),
    ("storage", "s3", "bucket"),
])
def test_a_missing_credential_fails_at_startup_naming_it(settings_env, kind, name, expected):
    """Not deep inside a job, hours later."""
    with pytest.raises(ProviderConfigError) as e:
        select(settings_env, kind, name)
    assert expected in str(e.value)


def test_the_fallback_chain_takes_over_when_the_primary_fails(settings_env):
    AlwaysFails.calls = 0
    provider = select(settings_env, "translation", "always_fails",
                      PROVIDERS__TRANSLATION__FALLBACKS='["fake"]')
    assert provider.translate(["hola"], "en", "es") == ["[es] hola"]
    assert AlwaysFails.calls == 1               # the primary really was tried


def test_fallbacks_are_only_built_when_needed(settings_env):
    """A credentialed fallback must not break a healthy primary by being listed."""
    NeedsCredentials.built = 0
    provider = select(settings_env, "translation", "fake",
                      PROVIDERS__TRANSLATION__FALLBACKS='["needs_credentials"]')
    assert provider.translate(["x"], "en", "es") == ["[es] x"]
    assert NeedsCredentials.built == 0


def test_retries_recover_then_give_up():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ProviderTransientError("network blip")
        return "ok"

    assert with_retries(flaky, retries=3, backoff=0) == "ok"
    assert attempts["n"] == 3

    with pytest.raises(ProviderTransientError):
        with_retries(lambda: (_ for _ in ()).throw(
            ProviderTransientError("still down")), retries=2, backoff=0)


def test_http_transport_errors_count_as_transient():
    import httpx

    from app.providers import TRANSIENT
    assert issubclass(httpx.ConnectError, TRANSIENT)
    assert issubclass(httpx.ReadTimeout, TRANSIENT)


def test_s3_save_accepts_a_file_written_via_write_path(tmp_path, monkeypatch):
    """The normal flow is write_path() -> write -> save(). On S3 the write path
    IS the cache path, so save() must not copy the file onto itself. Found by
    running a real MinIO job, not by reading the code."""
    uploaded = []
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(
        client=lambda *a, **kw: types.SimpleNamespace(
            upload_file=lambda src, bucket, key: uploaded.append(key))))

    from app.providers.storage.s3 import S3Storage
    storage = S3Storage(bucket="b", cache_dir=str(tmp_path / "cache"))
    p = storage.write_path("jobs/abc/source.mp4")
    p.write_bytes(b"data")
    storage.save("jobs/abc/source.mp4", p)      # must not raise SameFileError

    assert uploaded == ["jobs/abc/source.mp4"]
    assert p.read_bytes() == b"data"


def test_write_path_never_fetches(tmp_path):
    """Producers write keys that do not exist yet; path() would try to download."""
    from app.providers.storage.local import LocalStorage
    storage = LocalStorage(root=str(tmp_path))
    p = storage.write_path("jobs/abc/source.mp4")
    assert p.parent.exists() and not p.exists()
    p.write_bytes(b"data")
    storage.save("jobs/abc/source.mp4", p)
    assert storage.exists("jobs/abc/source.mp4")


def test_storage_keys_cannot_escape_the_root(tmp_path):
    from app.providers.storage.local import LocalStorage
    with pytest.raises(ValueError):
        LocalStorage(root=str(tmp_path)).path("../../etc/passwd")


def test_every_provider_file_imports_and_is_named_after_itself():
    """The registry maps a config name to app/providers/<kind>/<name>.py, so a file
    that fails to import, or registers under a different name, is unreachable —
    and the rest of the suite never notices because it runs on fakes.
    """
    import pkgutil
    from importlib import import_module

    from app.providers import KINDS, _REGISTRY

    checked = 0
    for kind in KINDS:
        pkg = import_module(f"app.providers.{kind}")
        files = sorted(m.name for m in pkgutil.iter_modules(pkg.__path__)
                       if not m.name.startswith("_"))
        for name in files:
            import_module(f"app.providers.{kind}.{name}")
            assert name in _REGISTRY[kind], (
                f"app/providers/{kind}/{name}.py must @register('{kind}', '{name}') — "
                f"the file name is the name used in config")
            checked += 1
    assert checked >= 18, f"expected every provider to be checked, saw {checked}"
