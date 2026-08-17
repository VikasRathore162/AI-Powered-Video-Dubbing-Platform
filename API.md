# API Reference

Base URL: `http://localhost:8000`. Interactive OpenAPI docs: `/docs`.
All endpoints accept/return JSON unless noted. When `security.api_key` is configured,
send `Authorization: Bearer <token>` on every request (401 otherwise; `/healthz` is
always open so load balancers can probe it). Every response carries an
`X-Request-ID` header (pass your own to correlate logs).

## POST /api/v1/jobs — upload a video

`multipart/form-data`:
| field | type | notes |
|---|---|---|
| `file` | file | mp4 / mov / avi / mkv (configurable), ≤ max size, ≤ 10 min |
| `target_languages` | string | comma-separated ISO codes: `es,fr,de` |
| `source_language` | string (optional) | hint; auto-detected when omitted |

```bash
curl -X POST localhost:8000/api/v1/jobs \
  -F "file=@video.mp4" -F "target_languages=es,fr"
```

`202` → `{"job_id": "...", "status": "queued", "target_languages": ["es","fr"]}`

Errors: `400` bad format/languages · `413` too large · `422` not decodable video /
no audio track / too long · `429` queue full or too many concurrent uploads or rate
limited · `401` bad API key.

## GET /api/v1/jobs — list jobs

Query: `status=`, `limit=`, `offset=`. Returns `{jobs: [...], total}`.

## GET /api/v1/jobs/{id} — status & progress

```json
{
  "job_id": "…", "status": "processing", "progress_pct": 55.0,
  "source_language": "en", "target_languages": ["es","fr"],
  "duration_sec": 21.4, "retry_count": 0, "error": null,
  "stages": [
    {"stage": "transcribe", "language": null, "status": "completed",
     "attempts": 1, "started_at": "…", "finished_at": "…", "error": null},
    {"stage": "translate", "language": "es", "status": "running", "...": "..."}
  ],
  "outputs": {"es": {"video": "/api/v1/jobs/…/video/es",
                      "subtitles": "/api/v1/jobs/…/subtitles/es"}},
  "transcript_url": "/api/v1/jobs/…/transcript"
}
```

Job statuses: `queued → processing → completed | completed_with_errors | failed | canceled`.

## GET /api/v1/jobs/{id}/transcript[?language=es]

Original transcript by default; translated text with `?language=<target>`.
Includes speakers (label, gender, assigned voices) and timestamped segments with
speaker attribution. `409` while not ready.

## GET /api/v1/jobs/{id}/video/{lang}

Dubbed MP4 (`video/mp4` file download). `409` while not ready.

## GET /api/v1/jobs/{id}/subtitles/{lang}

SRT text. Works for every target language **and** the source language (original
transcript timings). `409` while not ready.

## POST /api/v1/jobs/{id}/retry

Allowed from `failed`, `canceled`, `completed_with_errors`. Resets only non-completed
stages and re-queues; completed work (e.g. transcription) is not repeated.
`202` → `{"status": "queued", "retry_count": 1}` · `409` if not retryable.

## POST /api/v1/jobs/{id}/cancel

Queued jobs cancel immediately; processing jobs stop at the next stage boundary /
loop checkpoint (flag + best-effort Celery revoke). `202` · `409` if already terminal.

## GET /api/v1/jobs/{id}/logs

Audit trail: every stage transition, retry, cancellation, warning — with timestamps
and detail payloads. This is the "processing logs" output.

## Ops

- `GET /healthz` → `{"status": "ok"}`
- `GET /metrics` → Prometheus text (job counts, stage durations, active jobs, uploads)

A ready-to-import Postman collection is in [postman.json](postman.json).
