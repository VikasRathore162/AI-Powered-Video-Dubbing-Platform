# Configuration Guide

Configuration is layered (highest priority first):

1. **Environment variables** — nested keys via `__`: `LIMITS__MAX_DURATION_SEC=300`
2. **`.env` file** (same syntax; see `.env.example`)
3. **Secrets management** — files under `$SECRETS_DIR` (default `/run/secrets`)
4. **Cloud configuration services** — `CLOUD_CONFIG=aws-ssm://...` (see below)
5. **YAML config file** — path from `CONFIG_FILE` (default `config.yaml`)
6. Built-in defaults

All four mechanisms named in requirement 9 are implemented and tested.

No value is hardcoded: limits, timeouts, retries, model/provider choices, queue and
DB URLs are all configurable without code changes. Secrets (API keys) come from env
vars / a secrets manager — never from YAML.

> **Two merge rules, and the difference bites.**
>
> *Between sources*, values deep-merge key by key: with the config file setting
> `PROVIDERS__STT__OPTIONS={"model":"small","compute_type":"int8"}`, an env var of
> `{"model":"base"}` overrides only `model` and leaves `compute_type` alone. Usually
> what you want — but when you switch to a provider taking *different* options
> (local storage's `root` vs S3's `bucket`), define that provider in one place only
> or the old keys merge in and reach the constructor. This is why `config.yaml`
> carries no storage block: compose defines it.
>
> *Against the built-in defaults* there is no merge at all. A map you set in any
> source **replaces** the default map outright. Set two entries of `stage_timeouts`
> and the other five stages don't fall back to 60s and 600s — they get the 1800s
> catch-all. Override a map whole, or leave it alone.

## Processing limits (`limits`)

| key | default | meaning |
|---|---|---|
| `max_upload_mb` | 500 | reject larger uploads (413) |
| `max_duration_sec` | 600 | reject longer videos (422) |
| `allowed_formats` | mp4,mov,avi,mkv | extension whitelist |
| `max_concurrent_uploads` | 4 | parallel upload streams (429 beyond) |
| `max_concurrent_jobs` | 2 | jobs processing simultaneously |
| `max_queue_length` | 20 | queued+processing cap (429 beyond) |

## Pipeline behavior (`processing`)

| key | default | meaning |
|---|---|---|
| `stage_timeouts.<stage>` | per stage | Celery soft time limit seconds |
| `task_max_retries` | 3 | automatic retries for transient errors |
| `retry_backoff_sec` | 5 | exponential backoff base |
| `max_tempo` | 1.5 | max speed-up when fitting dub into original timing |
| `gap_guard_sec` | 0.1 | silence kept before the next segment when spilling |
| `duck_volume` | 0.15 | original audio level under the dub (0 = replace) |
| `tts_concurrency` | 4 | parallel TTS synth calls |
| `fault_inject_stage` | null | test hook: stage that fails on first attempt |

## AI providers (`providers`)

Each of `stt`, `diarization`, `translation`, `tts`, `storage` takes
`{name, options, fallbacks}`. Available names:

| kind | free default | alternatives (just add the credential) |
|---|---|---|
| stt | `faster_whisper` (`options: {model: small\|base\|medium, compute_type: int8}`) | `openai_whisper` (OPENAI_API_KEY), `deepgram` (DEEPGRAM_API_KEY), `assemblyai` (ASSEMBLYAI_API_KEY), `google_stt` (GOOGLE_API_KEY + GOOGLE_CLOUD_PROJECT) |
| diarization | `ecapa_cluster` (`distance_threshold`, `min/max_speakers`) | — the brief names no diarization service |
| translation | `argos` | `claude` (ANTHROPIC_API_KEY), `openai` (OPENAI_API_KEY), `gemini` (GEMINI_API_KEY), `deepl` (DEEPL_API_KEY) |
| tts | `edge` | `azure` (AZURE_SPEECH_KEY + AZURE_SPEECH_REGION), `elevenlabs` (ELEVENLABS_API_KEY), `xtts` (COQUI_TOS_AGREED=1 + `pip install coqui-tts`), `openvoice` (OPENVOICE_CKPT) |
| storage | `local` (`options: {root: ./data}`) | `s3` / MinIO (`options: {bucket, endpoint_url}`, boto3 creds) |

Every provider named in the assignment brief is implemented and selectable. Each one
validates its credential at construction, so a missing key fails immediately with a
message naming the variable rather than deep inside a job (`tests/test_providers.py`
asserts this for every one of them).

### Adding your own provider

The name in config is the file name under `app/providers/<kind>/`, so adding a vendor
is one new file and no edits anywhere else:

```python
# app/providers/translation/mistral.py
from app.providers import Translator, register, require_credential
from app.providers.translation import _LLMTranslator     # shared prompt + JSON parsing


@register("translation", "mistral")
class Mistral(_LLMTranslator):
    def __init__(self, model: str = "mistral-large-latest"):
        self._key = require_credential("MISTRAL_API_KEY", "mistral")
        self._model = model

    def _complete(self, prompt: str) -> str:
        ...        # one HTTP call, return the text
```

Then `PROVIDERS__TRANSLATION__NAME=mistral` (or the YAML equivalent) and it is live.
The module is imported only when selected, so its dependencies cost nothing when it
is not.

The one place a new credential must be declared is `app/config.py` — a single line
beside the other credential fields, so it flows through the same config layers as
every other secret:

```python
mistral_api_key: SecretStr | None = credential_field()
```

`credential_field()` is the marker: `CREDENTIALS` is derived from the fields carrying
it, so there is no second list to keep in sync. The field is what `credential()` reads
(env → `.env` → mounted secret → cloud config), and the marker is what keeps the value
out of logs and refuses it from a committed config file.

## Where credentials live — and why

The brief separates the two concerns, so the implementation does too:

> *Req 9:* "These values should be configurable using: Environment variables ·
> Configuration files · **Secrets management** · Cloud configuration services"
> *Req 10:* "**Model selection** should be configurable through **configuration files
> or environment variables**."

| Thing | Where it goes | Why |
|---|---|---|
| **Model / provider selection** (`providers.stt.name`, options) | config file **or** env var | Req 10 names both; not sensitive, and belongs in version control |
| **Credentials** (API keys, tokens) | env var, `.env`, **or mounted secret** — never a config file | Config files are committed; secrets management is the Req 9 mechanism for these |

Resolution order for a credential is **environment variable → `.env` → mounted
secret file**. All three are exercised by tests; the mounted-secret path is also
verified live against a container.

```bash
# 1. environment variable
ELEVENLABS_API_KEY=sk-... docker compose up -d

# 2. mounted secret — the file name is the lowercased variable name
echo -n "sk-..." > ./secrets/elevenlabs_api_key
docker compose run -v "$PWD/secrets:/run/secrets:ro" tools ...
# override the directory with SECRETS_DIR; defaults to /run/secrets
```

Putting a credential in `config/*.yaml` does **not** work by design: the YAML source
drops credential keys and emits a warning, so a key committed by accident never
becomes live configuration. Credentials are typed `SecretStr`, so they don't appear
in logs, tracebacks, or a settings dump.

## Cloud configuration services

The fourth Req 9 mechanism is implemented as a settings source. Point `CLOUD_CONFIG`
at a service and the app reads its settings from there at startup:

```bash
CLOUD_CONFIG=aws-ssm://dubbing/          # every parameter under /dubbing/
CLOUD_CONFIG=aws-secrets://dubbing/app   # one secret holding a JSON object
```

Keys use the same names and `__` nesting as environment variables, so
`/dubbing/limits__max_duration_sec` sets `limits.max_duration_sec` and
`/dubbing/providers__tts__name` selects the TTS provider. Nested SSM paths work too:
`/dubbing/providers/tts/name` means the same thing.

**Unlike a config file, a cloud config service may carry credentials** — that is what
SSM `SecureString` and Secrets Manager are for; values are decrypted on read.

Full precedence, highest first:

1. Environment variables (an operator can always override on the spot)
2. `.env`
3. Mounted secrets (`$SECRETS_DIR`, default `/run/secrets`)
4. **Cloud configuration service** (`CLOUD_CONFIG`)
5. YAML config file (`CONFIG_FILE`) — selection and limits only, never credentials
6. Built-in defaults

A misconfigured or unreachable source raises at startup rather than silently falling
back to defaults. Try it locally against LocalStack:

```bash
CC="docker compose -f docker-compose.yml -f docker-compose.cloudconfig.yml"
$CC up -d --wait localstack
$CC run --rm tools python scripts/seed_cloud_config.py
$CC up -d --wait api worker      # now configured from Parameter Store
```

Adding another backend (Azure App Configuration, GCP Secret Manager, Consul) means one
fetch function in `app/cloudconfig.py` registered in `FETCHERS` — no other code changes.

### Credential per provider

| Provider | Credential(s) |
|---|---|
| `openai_whisper`, `openai` | `OPENAI_API_KEY` |
| `claude` | `ANTHROPIC_API_KEY` |
| `gemini` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
| `deepgram` | `DEEPGRAM_API_KEY` |
| `assemblyai` | `ASSEMBLYAI_API_KEY` |
| `google_stt` | `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT` |
| `deepl` | `DEEPL_API_KEY` |
| `elevenlabs` | `ELEVENLABS_API_KEY` |
| `azure` | `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` |
| `xtts` | `COQUI_TOS_AGREED` (license acceptance) |
| `openvoice` | `OPENVOICE_CKPT` (checkpoint path) |
| `s3` | standard AWS chain (`AWS_*`, IAM role, or instance profile) |

**Emotional tone** (requirement 5, "where possible") is available on the `azure`
provider, which speaks SSML: set `options.style` to an Azure express-as style and
optionally `style_degree`.

```yaml
providers:
  tts:
    name: azure
    options: {style: cheerful, style_degree: 1.2}
```

## Queue backend

Redis is the default. AWS SQS is a config profile, not a code path:

```bash
export APP_UID=$(id -u)
docker compose -f docker-compose.yml -f docker-compose.sqs.yml up -d --wait api worker
```

That overlay runs ElasticMQ locally and points `broker_url` at `sqs://local:local@sqs:9324`
(see `config/sqs.yaml`). For real AWS, delete the `sqs` service, use `broker_url: sqs://`
with no host, set the region, and supply credentials via IAM role or `AWS_*` env.
`broker_transport_options.visibility_timeout` must exceed your longest stage.

RabbitMQ is `docker-compose.rabbitmq.yml` (or just `BROKER_URL=amqp://...`), and S3
storage is `docker-compose.minio.yml`. All three overlays were verified by running the
full 22-check E2E against them, not just by inspection.

Switch a model without touching code:

```bash
export PROVIDERS__STT__NAME=faster_whisper
export PROVIDERS__STT__OPTIONS='{"model": "medium", "compute_type": "int8"}'
```

Fallback chain (tries the next provider when one fails):

```yaml
providers:
  translation:
    name: openai
    fallbacks: [argos]
```

## Security (`security`)

| key | default | meaning |
|---|---|---|
| `api_key` | null | the bearer token; when set, every endpoint requires `Authorization: Bearer <token>` |
| `rate_limit` | `10/minute` | per-IP upload rate limit (slowapi syntax) |

## Infrastructure

| key | default | |
|---|---|---|
| `db_url` | `sqlite:///./data/app.db` | any SQLAlchemy URL (compose: Postgres) |
| `broker_url` / `result_backend` | `redis://localhost:6379/0` / `/1` | Celery |
| `log_level` | INFO | structlog/stdlib level |

## Model caches

Pre-download everything (~1.1GB: whisper base+small, ECAPA, Argos en↔es/fr):

```bash
docker compose run --rm tools python scripts/setup.py models
```

Cache locations are redirected by the image's env vars (`HF_HOME`, `XDG_DATA_HOME`,
`XDG_CACHE_HOME`, `SPEECHBRAIN_CACHE`) into `/data/models`, which is the bind-mounted
`./data` volume — so models download once and survive container rebuilds.

## Running with a different config

The image defaults to `CONFIG_FILE=config.yaml`. To use another file, mount it
and point `CONFIG_FILE` at it, or override individual keys with env vars:

```bash
docker compose run --rm -e LIMITS__MAX_DURATION_SEC=120 \
  -e PROVIDERS__STT__NAME=faster_whisper \
  -e 'PROVIDERS__STT__OPTIONS={"model":"base","compute_type":"int8"}' \
  tools pytest
```
