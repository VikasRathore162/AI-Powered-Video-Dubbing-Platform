# Design Decisions

Decisions made deliberately, with the trade-off and the upgrade path.

## Providers

**Free/local defaults (faster-whisper, ECAPA clustering, Argos, edge-tts).**
The platform must run end-to-end on a laptop with no API keys. Paid providers
(OpenAI, Deepgram, DeepL, ElevenLabs, XTTS) are implemented behind the same
interfaces and selected purely by config — the abstraction layer is the deliverable,
the default choice is just the free path.

**Diarization = whisper segments + ECAPA embeddings + agglomerative clustering.**
The brief requires speaker diarization (Req 2) but names no service for it, so this is
the one capability with no vendor to follow. Embedding-clustering needs no account and
no token, and works well on clean 2-4 speaker audio. pyannote 3.1 would be more accurate
on overlapping speech, but it is a gated model requiring a HuggingFace token — and since
the brief never names it, shipping it would have meant carrying an unnamed, unverifiable
service. Resemblyzer was rejected too: it depends on `webrtcvad`, which has no Python
3.12 wheels and fails to build.

**edge-tts assigns each speaker a distinct, gender-matched voice rather than cloning
the original voice.** True cloning (XTTS-v2) is 3-5× real-time on CPU with a 2GB model
and a non-commercial license; edge-tts is fast, free, and covers 100+ locales. Speaker
*identity* is preserved by consistency: the voice→speaker mapping is persisted in the DB
at diarization time and reused across retries. XTTS is available as `providers.tts.name=xtts`.

**Gender heuristic = median F0 threshold (165 Hz).** ~25 lines of numpy autocorrelation;
right most of the time on clean speech, and a wrong guess only means a voice of the other
gender — the dub still works. Upgrade path: a proper classifier on the ECAPA embedding.

## Pipeline

**Data flows through DB + storage artifacts, never through Celery return values.**
Tasks carry only `(job_id, lang)`. This one rule makes checkpointing, retry-resume,
cancellation, and horizontal scaling trivial, and any worker can run any task.

**Retry = re-dispatch the full chain; completed stages no-op.** No "start from stage X"
graph surgery. Each stage checks `status == completed and artifacts exist` and returns
immediately — resuming a job that failed at synthesis costs milliseconds of skipping.

**Custom `settle` task instead of a Celery chord.** Chords require careful error
semantics (a failed header hides the body). Instead every language branch ends with
`settle`, which closes the job only when all branches are terminal — idempotent,
DB-driven, works identically in eager (test) and real mode.

**Beat-based admission control (`kick_queue` every 5s) instead of dispatch-on-upload.**
Gives an exact global cap on concurrently processing jobs regardless of worker count,
survives restarts (queued jobs are re-discovered from the DB), and doubles as the sweep
that settles jobs whose tasks were revoked before starting. Trade-off: up to 5s of
dispatch latency — irrelevant next to minutes of processing.

**Timing fit: never slow down, spill into gaps, stretch ≤1.5×, then accept overlap.**
Slowed speech sounds drunk; silence sounds natural. Overlap beyond the tempo clamp is
recorded (`overflow_sec`) and audited rather than trimming words. Lip-sync is best-effort
segment-level alignment, per the assignment.

**Original audio ducked statically (volume=0.15) under the dub, not sidechain-compressed
and not removed.** Keeps music/ambience without a stem-separation model (Demucs is too
heavy for CPU). Upgrade path: `sidechaincompress` ffmpeg filter or source separation.

## Platform

**SQLite dev / Postgres compose via SQLAlchemy; `create_all`, no Alembic.** Greenfield
project — there is no schema history to migrate. First real migration introduces Alembic.

**SQL DB as job store + Redis only as broker.** Job state must survive broker restarts
and be queryable (status lists, stage detail, audit) — that's a database job. Redis
holds only in-flight task messages.

**Rate limiting via slowapi (per-IP, in-process).** Sufficient for a single API node;
multi-node production moves this to the gateway/ingress or a Redis-backed limiter.

**Upload size enforced via Content-Length check + streamed size cap.** A chunked-encoding
client that lies about size is caught while streaming. FastAPI buffers multipart to a
spooled temp file first; a reverse proxy body limit (nginx `client_max_body_size`) is the
production front line — documented, not reimplemented.

**Metrics on two endpoints.** The API serves `/metrics` (auth-protected) for request
and upload counters; each worker serves `:9100/metrics` for the processing metrics.
See "Worker metrics needed their own endpoint" below for why that split is required
rather than incidental.

**Known minor race:** two branches finishing simultaneously can both run `settle`;
the status write is idempotent, worst case a duplicated metric increment/audit line.
A `SELECT ... FOR UPDATE` on the job row fixes it on Postgres if it ever matters.

## Testing

**Fast suite uses fake AI providers but real ffmpeg.** The expensive-to-get-wrong parts
(timing math, timeline overlay, muxing, SRT, API contracts, retry/cancel semantics) are
fully exercised in ~12s with no models loaded. Heavy imports are lazy inside provider
constructors so the fast suite never pays torch's import cost.

**Real-model integration test uses whisper `base` (not `small`) and a committed
two-speaker fixture** generated from two different edge-tts voices — deterministic,
no downloads at test time, and hard enough to prove diarization actually separates
speakers (it does: 2 clusters, correct genders).

**Everything runs in Docker, including the tests.** A `tools` service (compose
profile, never started by `up`) shares the app image and runs pytest, model
downloads, fixture generation, and the black-box E2E. One environment builds, runs,
and verifies the system — no "works on my machine" gap between dev and deployment.

**Settings sources deep-merge, key by key.** This bit twice, so it is worth stating
plainly: `PROVIDERS__STT__OPTIONS={"model":"base"}` overrides `model` and *keeps*
`compute_type` from the config file. Good for tuning one value; bad when switching to
a provider that takes different options — local storage's `root` merged into S3's
options and reached `S3Storage(root=...)` as an unexpected keyword. Two consequences:
tests force `CONFIG_FILE` to a nonexistent path so they own their configuration
outright, and `config.yaml` deliberately carries no storage block — compose defines
storage in one place. Define each switchable provider in exactly one source.

## Thread limits (found by end-to-end testing)

`OMP_NUM_THREADS` and friends are pinned to 2 in the image. torch, CTranslate2 and
BLAS each default to "use every core", so 4 Celery children on 8 cores meant up to 32
threads competing — and *concurrent cold model loads* degraded pathologically rather
than just sharing. Measured on a 2-language job with cold worker processes: the two
`translate` branches started in the same millisecond, one finished in 5.7s and the
other took **137.6s** for the same four sentences. With threads bounded, the same cold
job runs both branches in ~9.6s each and end-to-end drops from 158s to 44s (22s once
processes are warm). Rule of thumb when changing worker concurrency: threads ≈
cores ÷ concurrency.

This is why the platform pre-downloads models to a shared volume but still loads them
per worker process — the fix for the remaining cold-start cost is bounded threads, not
eager loading in every fork (which would multiply idle memory by concurrency).

## Credentials go through the config layer, not `os.environ`

The brief splits these deliberately: *model selection* is "configuration files or
environment variables" (req 10), while the configuration mechanisms include
"secrets management" (req 9). So selection lives in `config/*.yaml` and credentials
do not.

Originally each provider read `os.environ` directly. That worked for env vars but
silently defeated the other mechanism: `secrets_dir` was wired into `Settings`, yet a
mounted Docker/K8s secret could never reach a provider, because providers never
consulted `Settings` at all. Credentials are now typed `SecretStr` fields resolved by
one `require_credential()` helper, so env vars, `.env`, and mounted secrets all work
through a single path — and `SecretStr` keeps keys out of logs and tracebacks.

Two things this exposed, both fixed:

- `secrets_dir` was evaluated at class-definition time, so it froze at import and
  ignored a later `SECRETS_DIR`. It is now passed per-instantiation.
- Because credentials became `Settings` fields, the YAML source *could* supply them —
  the exact thing the design forbids. `CredentialFreeYamlSource` drops credential keys
  from YAML and warns, so a key committed by accident never becomes live config.

## All four configuration mechanisms, not three

Requirement 9 names four: environment variables, configuration files, secrets
management, cloud configuration services. The first three were straightforward; the
fourth is implemented as a `CLOUD_CONFIG` settings source reading AWS SSM Parameter
Store or Secrets Manager.

The implementation trick worth noting: the cloud source subclasses pydantic-settings'
`EnvSettingsSource` and only overrides where the values come from. Nesting (`__`),
case handling, and JSON parsing of complex fields are then identical to environment
variables for free, rather than being a second parser that drifts.

Precedence puts cloud config **above** the YAML file but **below** environment
variables: the file is the checked-in baseline, the cloud service is the deployed
truth, and an operator can still override either on the spot. A misconfigured or
unreachable source raises at startup instead of silently falling back to defaults —
a silent fallback would run production on developer defaults.

Verified against LocalStack rather than asserted: a seeded `limits__max_duration_sec`
of 300 is enforced on live uploads (a 577s video is rejected against it), and a
`SecureString` credential resolves through the same path. New backends (Azure App
Configuration, GCP Secret Manager) are one fetch function each.

## Broker portability (requirement 11)

The queue is swappable because nothing but task identifiers travels through it. Proven
rather than asserted: the same image and code pass the full 22-check E2E on **AWS SQS**
(ElasticMQ locally) with Redis idle and `DBSIZE 0`. Two things this surfaced:

- **Cancellation survives brokers without remote control.** SQS has no Celery
  `revoke` broadcast. Cancel was already a DB flag with revoke as best-effort, so it
  works unchanged — a design choice that only paid off once the broker changed.
- **`visibility_timeout` must exceed the longest stage.** SQS redelivers a message
  whose visibility window lapses; with a 30s default, a 2-minute transcribe would run
  twice concurrently. `config/sqs.yaml` sets 3600 and says why.
- No result backend is configured on that profile (`result_backend: ""`,
  `task_ignore_result`): job state is in Postgres and nothing ever reads a task result.

## A second S3 bug, found only by running it

The review caught that producers called `storage.path()` on keys that didn't exist yet
(S3 would try to *download* the file it was about to create). Fixing that introduced
`write_path()` — and on S3 the write path **is** the local cache path, so the follow-up
`save()` then asked `shutil.copyfile` to copy a file onto itself and every upload 500'd
with `SameFileError`. `LocalStorage.save` already guarded for this; `S3Storage.save`
did not. Inspection didn't catch it and neither did the local-backend tests: it took
standing up MinIO and running a real job. Both the guard and a regression test using a
stubbed boto3 client are in place, and the full E2E now passes on S3.

The lesson worth keeping: a storage abstraction is only as verified as its least-tested
implementation. `docker-compose.minio.yml` exists so that one stays exercised.

## Worker metrics needed their own endpoint

The metrics that describe processing — stage-duration histograms, job outcomes — are
incremented in Celery prefork *children*, while `/metrics` is served by the API
process. As originally shipped they were computed and never scrapable. Workers now run
`prometheus_client` in multiprocess mode (`PROMETHEUS_MULTIPROC_DIR`) and the main
worker process serves the aggregate on `:9100`. One ordering constraint the fix
depends on: in multiprocess mode a metric opens its mmap file the moment it is
*defined*, so the directory must exist before `app.metrics` is imported — creating it
in a Celery startup hook was too late and crashed the worker.

## Lip-sync drift, found by measuring instead of asserting (200 ms → 15 ms)

The E2E asserted the dub was *present* and *different from the source*, which it was —
so sync looked fine for a long time. Measuring the dubbed track's energy envelope
against the transcript timeline told a different story: every line landed **180–240 ms
late**, in both languages, on every segment. That much lag is visible against lip
movement (broadcast practice keeps audio lag under ~125 ms).

The cause was in the voices, not the timing maths: edge-tts pads its output with
~0.2 s of silence before the speech and ~0.9 s after. That padding was counted as clip
duration, so it did double damage — every line started late by its lead, *and* the
fitter saw a clip ~1.1 s longer than the actual speech and time-stretched real words to
squeeze the silence into the slot. On the test clip, 2 of 4 segments were being sped up
purely to make room for silence.

`media.trim_silence()` now strips the first and last silent run (inner pauses are kept
— they are part of the delivery) before the clip is measured or fitted. Re-measured on
the same video: mean onset offset **15 ms (es) / 20 ms (fr)**, and **0 of 4** segments
need time-stretching. The check that would have caught this originally is now a unit
test on the trim, plus the envelope measurement itself.

## What a real video exposed that the synthetic fixture could not

The committed fixture is two edge-tts voices with 0.8s gaps — clean, alternating,
studio-perfect. Running a real 5-minute Royal Society interview (two people, CC BY
3.0) through the same pipeline broke three things at once:

- **Diarization found 4 speakers where there were 2.** Short utterances ("It's lovely
  to be here") give the encoder too little signal, so each landed in its own cluster.
- **The female guest was dubbed with a male voice.** Downstream of the above: four
  clusters exhausted the two female voices in the curated catalogue, and casting fell
  through to "any unused voice", which was male.
- **43 of 58 Spanish lines were longer than their English slot.** Real conversation has
  a *median gap of 0.00s* — there is no silence to spill into, so 35 lines were
  time-compressed, many at the 1.5× clamp.

Two fixes, both driven by that evidence:

**Clusters below a floor *and* below a share of the conversation are absorbed into
their nearest real cluster** (cosine similarity on the L2-normalised embeddings). The
share test is what matters: a 4-second stray line inside a 5-minute recording is 1.4%
of the speech and is almost certainly a split turn, but hard-coding "4 seconds" would
only ever suit this one video. Result on the same input: **4 → 2 speakers**, and the
mis-split line was reattached to the speaker whose sentence it continues.

**Gender match now outranks voice uniqueness.** Casting prefers an unused matching
voice, then *reuses* a matching one, and only crosses gender if the language offers
none — plus `EdgeTTS.voices()` now appends the service's full published list (≈20 per
gender for major languages) behind the curated picks, so exhaustion is rare in the
first place. Two speakers sharing a voice is a smaller failure than a man dubbing a
woman.

The 1.5× compression is *not* a bug — it is the documented timing policy meeting a
language that expands. It is the honest limitation of segment-level dubbing without
re-timing the video.

## Fixed during review

An adversarial multi-agent review ran against the finished code; confirmed findings
were fixed and each has a regression test in `tests/test_validation.py`:

- `StorageProvider.write_path()` split from `path()` — producers were calling `path()`
  on not-yet-existing keys, which made the S3 backend try to *download* the file it was
  about to create (every upload would 500 with `providers.storage.name: s3`).
- Upload now cleans up on **any** exception (not just HTTPException) and truncates the
  filename to the column width — a 300-char filename used to orphan the uploaded file
  on Postgres with no job row referencing it.
- `get_provider()` moved before the upload-concurrency gate, so a provider config error
  can't permanently leak upload slots (4 failures → all uploads 429 until restart).
- `/metrics` is a route with the auth dependency instead of a bare mount, so an API key
  actually protects it.
- `kick_queue` claims jobs with a conditional `UPDATE ... WHERE status='queued'` and
  releases the claim if dispatch fails, instead of a non-atomic read-then-update.
- Validation hardening: cover-art streams no longer count as video (an MP3 with
  album art was a "valid video"), unknown duration is rejected instead of passing the
  10-minute check as 0.0, 3-letter language codes are rejected up front (models speak
  ISO 639-1), and `?limit=-1` / `?offset=-5` are rejected instead of reaching SQL.
- `httpx.TransportError` added to the transient-error set, so hosted providers retry
  network blips instead of failing the stage; `ValidationFailed` is explicitly
  permanent and never retried.
- Mounted secrets (`/run/secrets`) were registered as a config source but never read
  (no `secrets_dir`) and ranked below YAML. Now wired, and ordered above YAML so a
  mounted credential can't be shadowed by a checked-in config file.
- `request_id` is stored on the job and re-bound in every worker stage, so a job's
  worker-side events correlate with the HTTP request that created it. Previously
  tracing stopped at the API boundary.

**Known limitation this round exposed:** `create_all` does not alter existing tables,
so adding the `request_id` column required `docker compose down -v` against a Postgres
volume that predated it. That is the point where Alembic stops being deferrable; the
migration path is the first thing to add before this runs anywhere with real data.
