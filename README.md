# AI-Powered Video Dubbing Platform

Upload a video → automatic speaker diarization → speech recognition → translation →
per-speaker AI voices → time-synced dubbed video + transcripts + SRT subtitles.
Multiple target languages per job, REST APIs with status/retry/cancel, Celery+Redis
pipeline, config-driven providers, Docker Compose deployment.

**Everything runs in Docker** — no host Python, no venv. Build, tests, model
downloads, fixtures and verification all execute inside containers.

**Default pipeline is 100% free — no API keys required:**

| Step | Provider (default) | Alternatives (config-selectable) |
|---|---|---|
| Speech-to-text | faster-whisper (local, CPU) | OpenAI Whisper, Deepgram, AssemblyAI, Google STT |
| Diarization | ECAPA embeddings + clustering (local) | — |
| Translation | Argos Translate (local) | Claude, OpenAI GPT, Gemini, DeepL |
| Voice generation | edge-tts (free MS neural voices) | Azure Speech (SSML emotion styles), ElevenLabs, Coqui XTTS-v2, OpenVoice |
| Storage | local filesystem | **S3 / MinIO** — verified with the full E2E |
| Queue | Redis | **AWS SQS**, **RabbitMQ** — both verified with the full E2E |

Every provider listed below is implemented behind one interface — adding a
key and a config line is all it takes to switch. See [CONFIGURATION.md](CONFIGURATION.md).

The hosted ones are contract-tested against synthetic vendor payloads (request shape,
auth header, response parsing, and 5xx-vs-permanent classification — `tests/test_hosted.py`).
They have **not** been called against the live vendor APIs, because that needs paid keys.
The free defaults, S3/MinIO, SQS, RabbitMQ and SSM are all exercised for real.

Docs: [ARCHITECTURE.md](ARCHITECTURE.md) · [API.md](API.md) ·
[CONFIGURATION.md](CONFIGURATION.md) · [DESIGN_DECISIONS.md](DESIGN_DECISIONS.md) ·
[postman.json](postman.json)

**Walkthrough video:** _<paste the link here before submitting>_

## Requirements

Docker + Docker Compose. Nothing else. (Internet is needed on first run for image
builds, AI model downloads, and edge-tts voice synthesis.)

## Quick start

```bash
export APP_UID=$(id -u)          # so containers can write the ./data volume
docker compose up --build -d     # api + worker + redis + postgres

# one-time: pull AI models into the shared ./data volume (~1.1GB)
docker compose run --rm tools python scripts/setup.py models

# dub a video into Spanish and French
curl -X POST localhost:8000/api/v1/jobs \
  -F "file=@tests/two_speakers_en.mp4" \
  -F "target_languages=es,fr"

curl localhost:8000/api/v1/jobs/<job_id>              # status + per-stage progress
curl -O localhost:8000/api/v1/jobs/<job_id>/video/es  # dubbed video
curl localhost:8000/api/v1/jobs/<job_id>/subtitles/es # SRT
```

Every endpoint as a copy-paste `curl`, in the order you'd use them, is in
[curls.txt](curls.txt) — including retry, cancel, and the calls that should be rejected.

Interactive API docs: http://localhost:8000/docs · Stop: `docker compose down`

## Verification (one command, all in Docker)

```bash
./scripts/verify.sh
```

Eight phases: build → test suite on SQLite → same suite on Postgres → models +
real-model integration test → full stack up → black-box HTTP E2E (validation
rejects, stage progression, speaker detection, MP4 integrity via ffprobe + audio
actually changed, SRT parsing, audit logs, cancel mid-flight, and retry that
resumes without re-running completed stages) → **the same E2E again over an AWS SQS
broker**, proving the queue is a config choice → and once more with configuration
loaded from a cloud config service.

Individual pieces:

```bash
docker compose run --rm tools pytest                            # fast suite (fake AI, real ffmpeg)
docker compose run --rm tools env RUN_FULL_PIPELINE=1 pytest -m integration
docker compose run --rm tools python scripts/setup.py fixture   # regenerate test video
docker compose run --rm tools python scripts/e2e.py http://api:8000
# longer input: REPEAT=29 builds a ~10 minute clip (the maximum supported duration)
docker compose run --rm tools env REPEAT=12 OUT=/data/long.mp4 python scripts/setup.py fixture
```

**Verified on this machine** (8-core CPU, no GPU, whisper `small`):

| Check | Result |
|---|---|
| Test suite | **108 passed on SQLite and on Postgres** + 1 real-model integration test |
| Black-box E2E | **22/22 checks** against the compose stack — and the same 22 against an **AWS SQS** broker |
| 21s clip, 2 speakers → es + fr | 44s cold worker / 22s warm; both speakers separated, distinct gender-matched voice per language |
| 4.4 min clip, 48 turns → es | 110s on a warm worker |
| **9.6 min clip, 104 turns → es** (near the 10-min maximum) | **251s**, 2 speakers, 104 SRT cues, valid MP4 |
| 644s clip (over the limit) | rejected with 422 before any work |
| 5 jobs submitted at once | peak concurrency exactly 2 (the configured cap), all 5 done in 50s |
| **Real 5-min interview** (Royal Society, CC BY 3.0 — two people, unscripted) | **136s**, correct 2 speakers, accurate English transcript, distinct gender-matched Spanish voices, 58 SRT cues |

Worker math threads are pinned (`OMP_NUM_THREADS=2` etc. in the Dockerfile) — without
that, parallel language branches starve each other during cold model loads; see
DESIGN_DECISIONS.md. Scale it as cores ÷ worker concurrency.

## Services

| Service | Role |
|---|---|
| `api` | FastAPI (stateless) — upload, status, downloads, retry/cancel, `/metrics` |
| `worker` | Celery worker + beat — runs the pipeline, dispatches queued jobs, serves processing metrics on `:9100` |
| `redis` | default broker (no result backend: job state lives in Postgres) |
| `postgres` | job state, stages, speakers, transcripts, audit logs |
| `tools` | one-off container (`profiles: tools`) for tests/models/fixtures/E2E |

## Security

Requirement 17 marks authentication **optional**, so the API ships **open** — a fresh
`docker compose up` needs no token and the quick-start above works as written. No
default password is baked into the image, which is deliberate: a shipped credential
everyone knows is worse than none.

**Always on, no configuration needed:**

| Control | Behaviour |
|---|---|
| File validation | extension whitelist **and** a real ffprobe decode — a `.mp4` full of random bytes is rejected 422 |
| Upload size | `Content-Length` rejected before reading, plus a streaming cap for clients that lie → 413 |
| Rate limiting | per IP, `security.rate_limit` (default `10/minute`) → 429 |
| Concurrency | `max_concurrent_uploads` and `max_concurrent_jobs` gates → 429 |
| Input validation | ISO 639-1 language codes, bounded paging, storage keys that cannot escape the root |
| Secrets | credentials are `SecretStr`, never logged, and refused from committed config files |

**Turning authentication on** — one variable, no extra service:

```bash
SECURITY__API_KEY=your-secret-token docker compose up -d

curl -H "Authorization: Bearer your-secret-token" localhost:8000/api/v1/jobs
```

Every endpoint then requires `Authorization: Bearer <token>`; anything else is 401 with
`WWW-Authenticate: Bearer`. `/healthz` stays open so load balancers can probe it. The
scheme is published in OpenAPI, so http://localhost:8000/docs grows an **Authorize**
button and the token is applied to every request you try there. Comparison is
constant-time. Verified live: no header, the old `X-API-Key`, a bare token without the
scheme, and a wrong token are all rejected.

## Swapping the queue, storage or database

All three are configuration, not code. The optional services live in the same compose
file behind profiles, and each was verified by running the **same 22-check E2E**
against it:

```bash
export APP_UID=$(id -u)
# AWS SQS instead of Redis
BROKER_URL='sqs://local:local@sqs:9324' docker compose --profile sqs up -d --wait sqs api worker
# RabbitMQ
BROKER_URL='amqp://guest:guest@rabbitmq:5672//' docker compose --profile rabbitmq up -d --wait rabbitmq api worker
# S3 object storage instead of the local filesystem
STORAGE=s3 STORAGE_OPTIONS='{"bucket":"dubbing","endpoint_url":"http://minio:9000"}' \
  docker compose --profile minio up -d --wait minio api worker
# configuration (and secrets) from AWS SSM Parameter Store
CLOUD_CONFIG='aws-ssm://dubbing/' docker compose --profile localstack up -d --wait localstack api worker
# SQLite instead of Postgres — the whole stack on one file, no database container
DB_URL='sqlite:////data/app.db' docker compose up -d --wait redis api worker
```

Each stand-in (ElasticMQ, MinIO, LocalStack) swaps for the real service by
changing the URL and supplying credentials. Two things worth knowing:
cancellation works even on SQS, which has no Celery `revoke`, because cancel is
a database flag; and on SQS `visibility_timeout` must exceed your longest stage
or the message is redelivered mid-processing.

## Project layout

```
app/
  main.py        the HTTP API: upload, status, transcripts, downloads, retry, cancel
  worker.py      Celery: schedules stages, owns retry/cancel/resume/admission
  pipeline.py    the seven stages as plain functions (no Celery — directly testable)
  providers/     the swappable AI + storage layer — a folder per capability,
                 one file per provider (stt/deepgram.py, translation/claude.py …)
  models.py      database: engine, tables, audit log
  config.py      settings from env / .env / secrets / cloud config / config.yaml
  media.py       ffmpeg wrappers and SRT writing
  obs.py         structured logging and metrics
config.yaml      the one config file
docker-compose.yml   api, worker, postgres, redis (+ optional backends by profile)
scripts/         verify.sh (the gate), e2e.py (black-box test), setup.py (models/fixture)
tests/           six files, mirroring the app modules
curls.txt        every API call as a copy-paste curl
postman.json     importable collection covering every endpoint
.github/         CI: the same containers run the suite on every push (bonus)
```


## Key operational behaviors

- **Resume from failed stage**: every stage checkpoints its artifacts; `POST /retry`
  re-dispatches the pipeline and completed stages no-op through in milliseconds.
- **Cancel**: `POST /cancel` sets a flag polled at stage boundaries and inside long
  loops, plus best-effort Celery revoke.
- **Multi-language fan-out**: transcription/diarization run once; each target language
  gets its own translate→synthesize→assemble branch (parallel across workers). If some
  languages fail the job finishes `completed_with_errors` with the rest downloadable.
- **Admission control**: at most `limits.max_concurrent_jobs` process concurrently;
  the rest wait in `queued` (dispatched by a beat task every 5s).
- **Audit trail**: every state transition is recorded and served at `GET /jobs/{id}/logs`.
