"""Settings, from all four sources the brief asks for.

Priority, highest first:
  environment variables > .env > mounted secrets > cloud config service > YAML file

Keys use `__` nesting everywhere: LIMITS__MAX_DURATION_SEC, PROVIDERS__STT__NAME.
Credentials are deliberately NOT readable from the YAML file (it is committed);
they come from env vars, mounted secrets, or a cloud config service.
"""
from __future__ import annotations

import json
import os
import warnings
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (BaseSettings, EnvSettingsSource,
                               PydanticBaseSettingsSource, SettingsConfigDict,
                               YamlConfigSettingsSource)


class Limits(BaseModel):
    max_upload_mb: int = 500
    max_duration_sec: float = 600.0
    allowed_formats: list[str] = ["mp4", "mov", "avi", "mkv"]
    max_concurrent_uploads: int = 4
    max_concurrent_jobs: int = 2
    max_queue_length: int = 20


class Processing(BaseModel):
    stage_timeouts: dict[str, int] = {
        "probe": 60, "extract_audio": 120, "transcribe": 1800, "diarize": 600,
        "translate": 1800, "synthesize": 1200, "assemble": 600}
    task_max_retries: int = 3
    retry_backoff_sec: float = 5.0
    max_tempo: float = 1.5          # most we speed a dub up to fit its slot
    gap_guard_sec: float = 0.1      # silence kept before the next segment
    duck_volume: float = 0.15       # original audio level under the dub
    tts_concurrency: int = 4
    cancel_poll_every: int = 5      # poll the cancel flag every N segments
    fault_inject_stage: str | None = None   # test hook: "translate" or "translate:fr"


class ProviderCfg(BaseModel):
    name: str
    options: dict = {}
    fallbacks: list[str] = []


class Providers(BaseModel):
    stt: ProviderCfg = ProviderCfg(
        name="faster_whisper", options={"model": "small", "compute_type": "int8"})
    diarization: ProviderCfg = ProviderCfg(
        name="ecapa_cluster",
        options={"distance_threshold": 0.55, "min_speakers": 1, "max_speakers": 8})
    translation: ProviderCfg = ProviderCfg(name="argos")
    tts: ProviderCfg = ProviderCfg(name="edge")
    storage: ProviderCfg = ProviderCfg(name="local", options={"root": "./data"})


class Security(BaseModel):
    api_key: str | None = None      # when set, X-API-Key is required
    rate_limit: str = "10/minute"


def credential_field(default=None):
    """Marks a setting as a provider credential: kept out of logs and refused
    from the YAML file. One marker per field, so the list can't drift."""
    return Field(default, json_schema_extra={"credential": True})


class CredentialFreeYaml(YamlConfigSettingsSource):
    """Keeps config files safe to commit: a credential written there is dropped
    with a warning instead of silently becoming live configuration."""

    def __call__(self) -> dict:
        data = super().__call__()
        leaked = CREDENTIALS & set(data)
        for key in leaked:
            data.pop(key)
        if leaked:
            warnings.warn(f"ignoring credential(s) {sorted(leaked)} in "
                          f"{self.yaml_file_path}: use env vars or secrets",
                          stacklevel=2)
        return data


class CloudConfig(EnvSettingsSource):
    """A cloud configuration service as a settings source. Subclassing the env
    source means `__` nesting and JSON parsing behave identically to env vars."""

    def __init__(self, settings_cls, values: dict[str, str]):
        self._values = {k.lower(): v for k, v in values.items()}
        super().__init__(settings_cls, case_sensitive=False,
                         env_nested_delimiter="__")

    def _load_env_vars(self) -> dict[str, str]:
        return self._values


def fetch_cloud_config(url: str) -> dict[str, str]:
    """Read settings from AWS SSM Parameter Store or Secrets Manager.

        aws-ssm://dubbing/          every parameter under /dubbing/
        aws-secrets://dubbing/app   one secret holding a JSON object

    Unlike a config file, these may carry credentials — that is what SecureString
    and Secrets Manager are for. Raises so a bad source fails at startup rather
    than silently running on defaults. AWS_ENDPOINT_URL points at LocalStack.
    """
    backend, _, path = url.partition("://")
    if not path or backend not in ("aws-ssm", "aws-secrets"):
        raise RuntimeError(f"CLOUD_CONFIG must be aws-ssm://<path> or "
                           f"aws-secrets://<secret>, got '{url}'")
    try:
        import boto3
        endpoint = os.environ.get("AWS_ENDPOINT_URL") or None
        if backend == "aws-ssm":
            prefix = "/" + path.strip("/") + "/"
            client = boto3.client("ssm", endpoint_url=endpoint)
            values, token = {}, None
            while True:
                page = client.get_parameters_by_path(
                    Path=prefix, Recursive=True, WithDecryption=True,
                    **({"NextToken": token} if token else {}))
                for p in page.get("Parameters", []):
                    values[p["Name"][len(prefix):].strip("/").replace("/", "__")] = p["Value"]
                token = page.get("NextToken")
                if not token:
                    return values
        client = boto3.client("secretsmanager", endpoint_url=endpoint)
        raw = client.get_secret_value(SecretId=path.strip("/"))["SecretString"]
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("secret must contain a JSON object of settings")
        return {k: v if isinstance(v, str) else json.dumps(v) for k, v in parsed.items()}
    except Exception as e:
        raise RuntimeError(f"could not read {url}: {e}") from e


def _secrets_dir() -> str | None:
    d = os.environ.get("SECRETS_DIR", "/run/secrets")
    return d if Path(d).is_dir() else None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__", env_file=".env", extra="ignore")

    limits: Limits = Limits()
    processing: Processing = Processing()
    providers: Providers = Providers()
    security: Security = Security()

    db_url: str = "sqlite:///./data/app.db"
    # any Celery broker: redis://, sqs://, amqp:// (RabbitMQ)
    broker_url: str = "redis://localhost:6379/0"
    # SQS needs visibility_timeout > the longest stage, or it redelivers mid-run
    broker_transport_options: dict = {}
    result_backend: str = ""        # not needed: job state lives in the DB
    log_level: str = "INFO"

    # Provider credentials. SecretStr keeps them out of logs and tracebacks;
    # credential_field() is what puts them in CREDENTIALS below.
    openai_api_key: SecretStr | None = credential_field()
    anthropic_api_key: SecretStr | None = credential_field()
    gemini_api_key: SecretStr | None = credential_field()
    google_api_key: SecretStr | None = credential_field()
    google_cloud_project: str | None = credential_field()
    deepgram_api_key: SecretStr | None = credential_field()
    assemblyai_api_key: SecretStr | None = credential_field()
    deepl_api_key: SecretStr | None = credential_field()
    elevenlabs_api_key: SecretStr | None = credential_field()
    azure_speech_key: SecretStr | None = credential_field()
    azure_speech_region: str | None = credential_field()
    coqui_tos_agreed: bool = credential_field(False)
    openvoice_ckpt: str | None = credential_field()

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings,
                                   dotenv_settings, file_secret_settings):
        sources: list[PydanticBaseSettingsSource] = [
            init_settings, env_settings, dotenv_settings, file_secret_settings]
        cloud = os.environ.get("CLOUD_CONFIG", "").strip()
        if cloud:
            sources.append(CloudConfig(settings_cls, fetch_cloud_config(cloud)))
        yaml_file = os.environ.get("CONFIG_FILE", "config.yaml")
        if Path(yaml_file).exists():
            sources.append(CredentialFreeYaml(settings_cls, yaml_file=yaml_file))
        return tuple(sources)


# Derived from the fields themselves, so adding a credential is one line.
CREDENTIALS = frozenset(name for name, f in Settings.model_fields.items()
                        if (f.json_schema_extra or {}).get("credential"))


@lru_cache
def get_settings() -> Settings:
    # secrets_dir is resolved per call so SECRETS_DIR is honoured at runtime,
    # not frozen at import
    return Settings(_secrets_dir=_secrets_dir())


def credential(name: str) -> str | None:
    """A provider credential from env, .env, a mounted secret, or cloud config.
    (`require_credential` in app.providers wraps this with a clear failure.)"""
    value = getattr(get_settings(), name.lower(), None)
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    return str(value) if value not in (None, "", False) else None
