"""Configuration: the four sources the brief asks for, and their precedence.

  environment variables > .env > mounted secrets > cloud config > YAML file

Credentials may come from any of those except the YAML file, which is committed.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from app.config import Settings, fetch_cloud_config, get_settings
from app.providers import ProviderConfigError, clear_instances, get_provider


def stub_ssm(monkeypatch, params: dict[str, str], pages: int = 1):
    """A fake SSM client whose get_parameters_by_path can paginate."""
    items = [{"Name": k, "Value": v} for k, v in params.items()]
    chunks = [items[i::pages] for i in range(pages)] if pages > 1 else [items]
    calls = {"n": 0}

    def get_parameters_by_path(**_):
        i = calls["n"] % len(chunks)        # cycle: a test may fetch twice
        calls["n"] += 1
        out = {"Parameters": chunks[i]}
        if i + 1 < len(chunks):
            out["NextToken"] = f"tok{i}"
        return out

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(
        client=lambda *a, **kw: types.SimpleNamespace(
            get_parameters_by_path=get_parameters_by_path)))
    return calls


# --- the basics -----------------------------------------------------------

def test_defaults():
    settings = get_settings()
    assert settings.limits.max_duration_sec == 600
    assert settings.providers.storage.name == "local"


def test_environment_overrides_with_nesting(settings_env):
    settings_env(LIMITS__MAX_DURATION_SEC="120", PROCESSING__MAX_TEMPO="2.0",
                 SECURITY__API_KEY="sekrit")
    settings = get_settings()
    assert settings.limits.max_duration_sec == 120
    assert settings.processing.max_tempo == 2.0
    assert settings.security.api_key == "sekrit"


def test_yaml_file_is_read_and_env_still_wins(tmp_path, monkeypatch):
    cfg = tmp_path / "custom.yaml"
    cfg.write_text("limits:\n  max_upload_mb: 7\nlog_level: DEBUG\n")
    monkeypatch.setenv("CONFIG_FILE", str(cfg))
    assert Settings().limits.max_upload_mb == 7
    assert Settings().log_level == "DEBUG"
    monkeypatch.setenv("LIMITS__MAX_UPLOAD_MB", "99")
    assert Settings().limits.max_upload_mb == 99


# --- credentials ----------------------------------------------------------

def test_credential_from_environment(settings_env):
    settings_env(ELEVENLABS_API_KEY="env-key", PROVIDERS__TTS__NAME="elevenlabs",
                 PROVIDERS__TTS__OPTIONS="{}")
    assert get_provider("tts")._key == "env-key"


def test_credential_from_a_mounted_secret(tmp_path, monkeypatch):
    """A Docker/K8s secret mounted as a file, with no env var in sight."""
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "elevenlabs_api_key").write_text("secret-from-file\n")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("SECRETS_DIR", str(secrets))
    monkeypatch.setenv("PROVIDERS__TTS__NAME", "elevenlabs")
    monkeypatch.setenv("PROVIDERS__TTS__OPTIONS", "{}")
    get_settings.cache_clear()
    clear_instances()
    assert get_provider("tts")._key == "secret-from-file"


def test_environment_beats_a_mounted_secret(tmp_path, monkeypatch):
    secrets = tmp_path / "secrets"
    secrets.mkdir()
    (secrets / "elevenlabs_api_key").write_text("from-file")
    monkeypatch.setenv("SECRETS_DIR", str(secrets))
    monkeypatch.setenv("ELEVENLABS_API_KEY", "from-env")
    monkeypatch.setenv("PROVIDERS__TTS__NAME", "elevenlabs")
    monkeypatch.setenv("PROVIDERS__TTS__OPTIONS", "{}")
    get_settings.cache_clear()
    clear_instances()
    assert get_provider("tts")._key == "from-env"


def test_a_credential_in_a_config_file_is_ignored(tmp_path, monkeypatch):
    """Config files are committed, so they must not be able to supply secrets —
    while still selecting the provider, which is what they are for."""
    cfg = tmp_path / "leaky.yaml"
    cfg.write_text("elevenlabs_api_key: should-not-be-used\n"
                   "providers:\n  tts:\n    name: elevenlabs\n    options: {}\n")
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setenv("CONFIG_FILE", str(cfg))
    monkeypatch.setenv("SECRETS_DIR", str(tmp_path / "absent"))
    for k in ("PROVIDERS__TTS__NAME", "PROVIDERS__TTS__OPTIONS"):
        monkeypatch.delenv(k, raising=False)
    get_settings.cache_clear()
    clear_instances()

    assert get_settings().providers.tts.name == "elevenlabs"    # selection: yes
    with pytest.raises(ProviderConfigError) as e:               # credential: no
        get_provider("tts")
    assert "ELEVENLABS_API_KEY" in str(e.value)


def test_credentials_stay_out_of_logs(settings_env):
    settings_env(ELEVENLABS_API_KEY="super-secret-value")
    settings = get_settings()
    assert "super-secret-value" not in repr(settings)
    assert "super-secret-value" not in str(settings.model_dump())
    assert settings.elevenlabs_api_key.get_secret_value() == "super-secret-value"


# --- cloud configuration service -----------------------------------------

def test_ssm_strips_the_prefix_and_paginates(monkeypatch):
    stub_ssm(monkeypatch, {"/dubbing/log_level": "DEBUG",
                           "/dubbing/limits__max_duration_sec": "120"})
    assert fetch_cloud_config("aws-ssm://dubbing/") == {
        "log_level": "DEBUG", "limits__max_duration_sec": "120"}

    calls = stub_ssm(monkeypatch, {f"/dubbing/k{i}": str(i) for i in range(6)}, pages=3)
    assert len(fetch_cloud_config("aws-ssm://dubbing/")) == 6 and calls["n"] == 3


def test_ssm_nested_paths_become_nested_keys(monkeypatch):
    stub_ssm(monkeypatch, {"/dubbing/providers/tts/name": "edge"})
    assert fetch_cloud_config("aws-ssm://dubbing/") == {"providers__tts__name": "edge"}


def test_secrets_manager_reads_a_json_object(monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(
        client=lambda *a, **kw: types.SimpleNamespace(
            get_secret_value=lambda SecretId: {"SecretString": json.dumps(
                {"log_level": "WARNING", "limits__max_upload_mb": 42})})))
    assert fetch_cloud_config("aws-secrets://dubbing/app") == {
        "log_level": "WARNING", "limits__max_upload_mb": "42"}


def test_a_broken_cloud_source_fails_startup_rather_than_silently(monkeypatch):
    with pytest.raises(RuntimeError):
        fetch_cloud_config("just-a-string")
    with pytest.raises(RuntimeError):
        fetch_cloud_config("azure-appconfig://x")

    def boom(**_):
        raise ConnectionError("endpoint unreachable")
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(
        client=lambda *a, **kw: types.SimpleNamespace(get_parameters_by_path=boom)))
    with pytest.raises(RuntimeError):
        fetch_cloud_config("aws-ssm://dubbing/")


def test_settings_load_from_cloud_config(monkeypatch, tmp_path):
    """Same nesting rules as env vars — and a cloud service may carry secrets."""
    stub_ssm(monkeypatch, {"/dubbing/limits__max_duration_sec": "321",
                           "/dubbing/providers__tts__name": "elevenlabs",
                           "/dubbing/providers__tts__options": "{}",
                           "/dubbing/elevenlabs_api_key": "key-from-parameter-store"})
    monkeypatch.setenv("CLOUD_CONFIG", "aws-ssm://dubbing/")
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "absent.yaml"))
    for k in ("ELEVENLABS_API_KEY", "PROVIDERS__TTS__NAME",
              "PROVIDERS__TTS__OPTIONS", "LIMITS__MAX_DURATION_SEC"):
        monkeypatch.delenv(k, raising=False)
    get_settings.cache_clear()

    settings = Settings()
    assert settings.limits.max_duration_sec == 321
    assert settings.providers.tts.name == "elevenlabs"
    assert settings.elevenlabs_api_key.get_secret_value() == "key-from-parameter-store"


def test_cloud_config_beats_the_file_but_not_the_environment(monkeypatch, tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("log_level: INFO\nlimits:\n  max_upload_mb: 10\n")
    stub_ssm(monkeypatch, {"/dubbing/log_level": "DEBUG"})
    monkeypatch.setenv("CLOUD_CONFIG", "aws-ssm://dubbing/")
    monkeypatch.setenv("CONFIG_FILE", str(cfg))
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    get_settings.cache_clear()

    settings = Settings()
    assert settings.log_level == "DEBUG"            # cloud beats the file
    assert settings.limits.max_upload_mb == 10      # file still fills the gaps

    monkeypatch.setenv("LOG_LEVEL", "ERROR")
    assert Settings().log_level == "ERROR"          # env beats everything


def test_no_cloud_config_is_a_no_op(monkeypatch, tmp_path):
    monkeypatch.delenv("CLOUD_CONFIG", raising=False)
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "absent.yaml"))
    get_settings.cache_clear()
    assert Settings().log_level == "INFO"
