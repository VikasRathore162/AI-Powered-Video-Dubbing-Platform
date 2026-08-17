"""HTTP API: upload, status, transcripts, downloads, retry, cancel.

The API is stateless — it validates and records, and never processes media.
Everything after upload happens in the worker.
"""
from __future__ import annotations

import re
import secrets
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import structlog
from fastapi import (APIRouter, Depends, FastAPI, File, Form, Header,
                     HTTPException, Query, Request, Response, UploadFile)
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.extension import _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

from app import media
from app.config import get_settings
from app.models import (AuditLog, Base, Job, JobStage, JobStatus, Segment,
                        Speaker, StageStatus, Translation, audit, get_engine,
                        session_factory)
from app.obs import UPLOADS_TOTAL, build_registry, get_logger, setup_logging
from app.pipeline import LANGUAGE_STAGES, SHARED_STAGES, STAGE_WEIGHTS, key
from app.providers import get_provider

log = get_logger("api")
router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])
limiter = Limiter(key_func=get_remote_address)

LANGUAGE = re.compile(r"^[a-z]{2}$")     # ISO 639-1, what the models accept
CHUNK = 1024 * 1024


# --------------------------------------------------------------------------
# request plumbing
# --------------------------------------------------------------------------

def get_db():
    session = session_factory()()
    try:
        yield session
    finally:
        session.close()


# auto_error=False so a missing header reaches our check and returns 401 (not
# FastAPI's 403), and so the API stays open when no token is configured.
bearer = HTTPBearer(auto_error=False, description="Bearer token from security.api_key")


def check_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    authorization: str | None = Header(
        default=None, description="Bearer <token> — required when security.api_key "
                                  "is set. Equivalent to the Authorize button above."),
):
    """`Authorization: Bearer <token>`, compared in constant time.

    Enforced only when security.api_key is set — the brief marks authentication
    optional, so an unconfigured deployment stays open on purpose.

    The header is declared twice on purpose: `bearer` registers the OpenAPI security
    scheme (the Authorize button, one token for the whole page), and `authorization`
    makes it show up as an editable parameter on every endpoint. Same header either
    way — whichever the caller fills, `bearer` is what parses it.
    """
    expected = get_settings().security.api_key
    if not expected:
        return
    supplied = credentials.credentials if credentials else ""
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(401, "invalid or missing bearer token",
                            headers={"WWW-Authenticate": "Bearer"})


auth = Depends(check_auth)


def request_id(request: Request) -> str | None:
    """Set by the middleware; absent when a route is called outside it."""
    return getattr(request.state, "request_id", None)


class UploadGate:
    """Bounds concurrent uploads.

    A per-process counter, so it caps one API process. Multi-process deployments
    need this in Redis — which is also why the K8s-style API replica count would
    have to stay at 1 until it is.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active = 0

    def acquire(self) -> bool:
        with self._lock:
            if self._active >= get_settings().limits.max_concurrent_uploads:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)


upload_gate = UploadGate()


# --------------------------------------------------------------------------
# response shapes (these also generate the OpenAPI docs at /docs)
# --------------------------------------------------------------------------

class StageOut(BaseModel):
    stage: str
    language: str | None = None
    status: str
    attempts: int
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None


class JobOut(BaseModel):
    job_id: str
    status: str
    progress_pct: float
    source_filename: str
    source_language: str | None = None
    target_languages: list[str]
    created_at: datetime
    error: str | None = None
    duration_sec: float | None = None
    retry_count: int | None = None
    stages: list[StageOut] | None = None
    outputs: dict[str, dict[str, str]] | None = None


class JobList(BaseModel):
    jobs: list[JobOut]
    total: int


class SpeakerOut(BaseModel):
    label: str
    gender: str | None = None
    total_speech_sec: float
    segment_count: int
    voices: dict[str, str] = {}


class SegmentOut(BaseModel):
    idx: int
    start: float
    end: float
    speaker: str | None = None
    text: str


class TranscriptOut(BaseModel):
    job_id: str
    language: str
    speakers: list[SpeakerOut]
    segments: list[SegmentOut]


class LogOut(BaseModel):
    ts: datetime
    event: str
    detail: dict | None = None


class Accepted(BaseModel):
    job_id: str
    status: str
    target_languages: list[str] = []
    retry_count: int | None = None


def progress_pct(job: Job) -> float:
    if job.status in (JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_ERRORS):
        return 100.0
    languages = max(len(job.target_languages or []), 1)
    total = 0.0
    for row in job.stages:
        if row.status != StageStatus.COMPLETED:
            continue
        weight = STAGE_WEIGHTS.get(row.stage, 0)
        total += weight / languages if row.stage in LANGUAGE_STAGES else weight
    return round(min(total, 100.0), 1)


def job_out(job: Job, detail: bool = False) -> JobOut:
    out = JobOut(job_id=job.id, status=job.status, progress_pct=progress_pct(job),
                 source_filename=job.source_filename,
                 source_language=job.source_language,
                 target_languages=job.target_languages or [],
                 created_at=job.created_at, error=job.error)
    if detail:
        out.duration_sec = job.duration_sec
        out.retry_count = job.retry_count
        out.stages = [StageOut(stage=r.stage, language=r.target_language,
                               status=r.status, attempts=r.attempts,
                               started_at=r.started_at, finished_at=r.finished_at,
                               error=r.error) for r in job.stages]
        out.outputs = {
            r.target_language: {
                "video": f"/api/v1/jobs/{job.id}/video/{r.target_language}",
                "subtitles": f"/api/v1/jobs/{job.id}/subtitles/{r.target_language}"}
            for r in job.stages
            if r.stage == "assemble" and r.status == StageStatus.COMPLETED}
    return out


def _job_or_404(db, job_id: str) -> Job:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


def _stage_done(db, job_id: str, stage: str, lang: str | None = None) -> bool:
    row = (db.query(JobStage)
           .filter_by(job_id=job_id, stage=stage, target_language=lang).first())
    return bool(row and row.status == StageStatus.COMPLETED)


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------

@router.post("", status_code=202, response_model=Accepted, dependencies=[auth])
@limiter.limit(lambda: get_settings().security.rate_limit)
async def upload(request: Request, file: UploadFile = File(...),
                 target_languages: str = Form(...),
                 source_language: str | None = Form(default=None),
                 db=Depends(get_db)):
    settings = get_settings()

    languages = list(dict.fromkeys(
        l.strip().lower() for l in target_languages.split(",") if l.strip()))
    if not languages or not all(LANGUAGE.match(l) for l in languages):
        UPLOADS_TOTAL.labels(outcome="bad_language").inc()
        raise HTTPException(400, "target_languages must be comma-separated ISO 639-1 "
                                 "codes, e.g. 'es,fr'")
    if source_language and not LANGUAGE.match(source_language.strip().lower()):
        raise HTTPException(400, "source_language must be an ISO 639-1 code")

    ext = (file.filename or "").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else ""
    if ext not in settings.limits.allowed_formats:
        UPLOADS_TOTAL.labels(outcome="bad_format").inc()
        raise HTTPException(400, f"unsupported format '.{ext}'; "
                                 f"allowed: {settings.limits.allowed_formats}")

    max_bytes = settings.limits.max_upload_mb * 1024 * 1024
    declared = request.headers.get("content-length")
    if declared and int(declared) > max_bytes + 4096:       # reject before reading
        UPLOADS_TOTAL.labels(outcome="too_large").inc()
        raise HTTPException(413, f"upload exceeds {settings.limits.max_upload_mb} MB")

    queued = db.query(Job).filter(
        Job.status.in_([JobStatus.QUEUED, JobStatus.PROCESSING])).count()
    if queued >= settings.limits.max_queue_length:
        UPLOADS_TOTAL.labels(outcome="queue_full").inc()
        raise HTTPException(429, "processing queue is full, try again later")

    storage = get_provider("storage")       # before the gate: a config error must
    if not upload_gate.acquire():           # not leak a slot
        UPLOADS_TOTAL.labels(outcome="upload_limit").inc()
        raise HTTPException(429, "too many concurrent uploads")

    job_id = str(uuid.uuid4())
    source_key = f"jobs/{job_id}/source.{ext}"
    try:
        dest = storage.write_path(source_key)
        size = 0
        with open(dest, "wb") as out:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:        # a client that lied about its length
                    raise HTTPException(413, f"upload exceeds "
                                             f"{settings.limits.max_upload_mb} MB")
                out.write(chunk)
        storage.save(source_key, dest)

        try:                                # cheap check now; the pipeline re-validates
            info = media.probe(dest)
        except media.MediaError as e:
            raise HTTPException(422, f"file is not a decodable video: {e}")
        if not info["has_video"]:
            raise HTTPException(422, "file has no video stream")
        if not info["has_audio"]:
            raise HTTPException(422, "video has no audio track — nothing to dub")
        if info["duration_sec"] <= 0:
            raise HTTPException(422, "could not determine video duration")
        if info["duration_sec"] > settings.limits.max_duration_sec:
            raise HTTPException(422, f"video duration {info['duration_sec']:.0f}s "
                                     f"exceeds {settings.limits.max_duration_sec:.0f}s limit")

        # the job row and its stage rows go in the same guarded block: anything
        # that fails after the file is written must clean the file up, or it is
        # orphaned with no row referencing it
        job = Job(id=job_id, source_filename=(file.filename or "upload")[:255],
                  source_key=source_key, size_bytes=size,
                  duration_sec=info["duration_sec"], probe=info,
                  source_language=(source_language or "").strip().lower() or None,
                  target_languages=languages,
                  request_id=request_id(request))
        db.add(job)
        db.add_all([JobStage(job_id=job_id, stage=s) for s in SHARED_STAGES]
                   + [JobStage(job_id=job_id, stage=s, target_language=lang)
                      for lang in languages for s in LANGUAGE_STAGES])
        audit(db, "job_created", job_id=job_id, filename=job.source_filename,
              size_bytes=size, target_languages=languages,
              request_id=job.request_id)
        db.commit()
    except HTTPException:
        db.rollback()
        storage.delete_prefix(f"jobs/{job_id}")
        UPLOADS_TOTAL.labels(outcome="rejected").inc()
        raise
    except Exception as e:                  # disk full, DB error, missing ffprobe
        db.rollback()
        storage.delete_prefix(f"jobs/{job_id}")
        UPLOADS_TOTAL.labels(outcome="error").inc()
        log.exception("upload_failed", job_id=job_id, error=str(e))
        raise HTTPException(500, "upload failed, please retry")
    finally:
        upload_gate.release()

    UPLOADS_TOTAL.labels(outcome="accepted").inc()
    return Accepted(job_id=job_id, status=JobStatus.QUEUED, target_languages=languages)


@router.get("", response_model=JobList, dependencies=[auth])
def list_jobs(status: str | None = None, limit: int = Query(50, ge=1, le=200),
              offset: int = Query(0, ge=0), db=Depends(get_db)):
    q = db.query(Job)
    if status:
        q = q.filter(Job.status == status)
    jobs = q.order_by(Job.created_at.desc()).limit(limit).offset(offset).all()
    return JobList(jobs=[job_out(j) for j in jobs], total=q.count())


@router.get("/{job_id}", response_model=JobOut, dependencies=[auth])
def job_status(job_id: str, db=Depends(get_db)):
    return job_out(_job_or_404(db, job_id), detail=True)


@router.get("/{job_id}/transcript", response_model=TranscriptOut, dependencies=[auth])
def transcript(job_id: str, language: str | None = None, db=Depends(get_db)):
    """The original transcript, or a translation with ?language=<target>."""
    job = _job_or_404(db, job_id)
    if not _stage_done(db, job_id, "transcribe"):
        raise HTTPException(409, "transcript not ready yet")

    speakers = db.query(Speaker).filter_by(job_id=job_id).order_by(Speaker.label).all()
    segments = db.query(Segment).filter_by(job_id=job_id).order_by(Segment.idx).all()
    labels = {s.id: s.label for s in speakers}

    lang = (language or "").strip().lower()
    if lang and lang != (job.source_language or ""):
        if lang not in (job.target_languages or []):
            raise HTTPException(404, f"language '{lang}' is not a target of this job")
        texts = {t.segment_id: t.text for t in db.query(Translation)
                 .filter_by(job_id=job_id, language=lang)}
        if not texts:
            raise HTTPException(409, f"translation to '{lang}' not ready yet")
    else:
        lang = job.source_language or "unknown"
        texts = {s.id: s.text for s in segments}

    return TranscriptOut(
        job_id=job_id, language=lang,
        speakers=[SpeakerOut(label=s.label, gender=s.gender,
                             total_speech_sec=round(s.total_speech_sec, 2),
                             segment_count=s.segment_count, voices=s.voices or {})
                  for s in speakers],
        segments=[SegmentOut(idx=s.idx, start=s.start_sec, end=s.end_sec,
                             speaker=labels.get(s.speaker_id), text=texts[s.id])
                  for s in segments if s.id in texts])


@router.get("/{job_id}/video/{lang}", dependencies=[auth])
def dubbed_video(job_id: str, lang: str, db=Depends(get_db)):
    _job_or_404(db, job_id)
    lang = lang.strip().lower()
    if not _stage_done(db, job_id, "assemble", lang):
        raise HTTPException(409, f"dubbed video for '{lang}' not ready")
    storage = get_provider("storage")
    k = key(job_id, "out", f"dubbed_{lang}.mp4")
    if not storage.exists(k):
        raise HTTPException(404, "output file missing from storage")
    return FileResponse(storage.path(k), media_type="video/mp4",
                        filename=f"dubbed_{lang}.mp4")


@router.get("/{job_id}/subtitles/{lang}", response_class=PlainTextResponse,
            dependencies=[auth])
def subtitles(job_id: str, lang: str, db=Depends(get_db)):
    """SRT for a target language, or for the source language."""
    job = _job_or_404(db, job_id)
    lang = lang.strip().lower()
    storage = get_provider("storage")
    k = key(job_id, "out", f"subs_{lang}.srt")
    if not storage.exists(k):
        if lang != (job.source_language or "") and lang not in (job.target_languages or []):
            raise HTTPException(404, f"language '{lang}' is not part of this job")
        raise HTTPException(409, f"subtitles for '{lang}' not ready")
    with storage.open(k) as f:
        return PlainTextResponse(f.read().decode(),
                                 media_type="text/plain; charset=utf-8")


@router.get("/{job_id}/logs", response_model=list[LogOut], dependencies=[auth])
def job_logs(job_id: str, db=Depends(get_db)):
    """The job's processing log: every stage transition, retry and failure."""
    _job_or_404(db, job_id)
    rows = (db.query(AuditLog).filter_by(job_id=job_id)
            .order_by(AuditLog.id).limit(1000))
    return [LogOut(ts=r.ts, event=r.event, detail=r.detail) for r in rows]


@router.post("/{job_id}/retry", status_code=202, response_model=Accepted,
             dependencies=[auth])
def retry_job(job_id: str, request: Request, db=Depends(get_db)):
    """Re-queue a failed job. Completed stages are the resume point — they are
    left alone and will no-op when the pipeline runs again."""
    job = _job_or_404(db, job_id)
    if job.status not in JobStatus.RETRYABLE:
        raise HTTPException(409, f"job in state '{job.status}' cannot be retried")
    for row in db.query(JobStage).filter_by(job_id=job_id):
        if row.status != StageStatus.COMPLETED:
            row.status, row.error = StageStatus.PENDING, None
            row.started_at = row.finished_at = None
    job.status, job.cancel_requested, job.error = JobStatus.QUEUED, False, None
    job.retry_count += 1
    audit(db, "job_retry_requested", job_id=job_id, retry_count=job.retry_count,
          request_id=request_id(request))
    db.commit()
    return Accepted(job_id=job_id, status=job.status,
                    target_languages=job.target_languages or [],
                    retry_count=job.retry_count)


@router.post("/{job_id}/cancel", status_code=202, response_model=Accepted,
             dependencies=[auth])
def cancel_job(job_id: str, request: Request, db=Depends(get_db)):
    """Ask the job to stop. Queued jobs stop immediately; running ones stop at
    the next stage boundary — the flag is the mechanism, revoke is a nicety, so
    this works on brokers with no remote control (SQS)."""
    job = _job_or_404(db, job_id)
    if job.status in JobStatus.TERMINAL:
        raise HTTPException(409, f"job already {job.status}")

    job.cancel_requested = True
    if job.status == JobStatus.QUEUED:
        for row in db.query(JobStage).filter_by(job_id=job_id,
                                                status=StageStatus.PENDING):
            row.status = StageStatus.CANCELED
        job.status = JobStatus.CANCELED
    else:
        try:
            from app.worker import celery_app
            for row in db.query(JobStage).filter_by(job_id=job_id):
                if row.celery_task_id and row.status in (StageStatus.PENDING,
                                                         StageStatus.RUNNING):
                    celery_app.control.revoke(row.celery_task_id)
        except Exception as e:
            log.warning("revoke_failed", job_id=job_id, error=str(e))
    audit(db, "job_cancel_requested", job_id=job_id, status=job.status,
          request_id=request_id(request))
    db.commit()
    return Accepted(job_id=job_id, status=job.status,
                    target_languages=job.target_languages or [])


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(get_engine())   # no alembic yet — see DESIGN_DECISIONS
    log.info("startup_complete")
    yield


def create_app() -> FastAPI:
    setup_logging(get_settings().log_level)
    app = FastAPI(
        title="AI Video Dubbing Platform", version="1.0.0",
        description="Upload a video, get it dubbed into other languages with "
                    "per-speaker voices, synced audio and subtitles.",
        lifespan=lifespan)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(router)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/healthz", tags=["ops"])
    def healthz():
        return {"status": "ok"}

    # a route, not a mount, so the API key protects it too
    @app.get("/metrics", tags=["ops"], dependencies=[auth])
    def metrics():
        return Response(generate_latest(build_registry()),
                        media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
