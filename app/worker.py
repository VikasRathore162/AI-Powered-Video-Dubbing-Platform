"""Celery worker: schedules the pipeline stages and owns retry/cancel/resume.

Two rules make the rest fall out:
  * tasks carry only (stage, job_id, lang) — all data travels via the DB and
    storage, so any worker can run any task and the queue stays swappable;
  * a stage that already completed with artifacts is skipped, so "retry" is
    just re-dispatching the whole chain and finished work no-ops through.
"""
from __future__ import annotations

import os
import time
from datetime import timezone

import structlog
from celery import Celery, chain
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import worker_init

from app.config import get_settings
from app.models import (Job, JobStage, JobStatus, StageStatus, audit,
                        session_factory, utcnow)
from app.obs import (ACTIVE_JOBS, JOBS_TOTAL, STAGE_SECONDS, build_registry,
                     get_logger, multiproc_dir, setup_logging)
from app.pipeline import LANGUAGE_STAGES, SHARED_STAGES, STAGES, InvalidVideo
from app.providers import TRANSIENT

log = get_logger("worker")
_boot = get_settings()          # broker wiring only: fixed for the process
setup_logging(_boot.log_level)

celery_app = Celery("dubbing", broker=_boot.broker_url,
                    backend=_boot.result_backend or None)
celery_app.conf.update(
    task_serializer="json", accept_content=["json"], timezone="UTC",
    task_acks_late=True,            # a crashed worker's task is redelivered
    task_ignore_result=True,        # job state lives in the DB, not a backend
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    broker_transport_options=_boot.broker_transport_options,
    beat_schedule={"dispatch": {"task": "dispatch", "schedule": 5.0}},
)


class Canceled(Exception):
    """The job was canceled while this stage was running."""


def _stage_row(session, job_id: str, stage: str, lang: str | None) -> JobStage:
    """Every row exists already — upload creates one per stage per language."""
    return (session.query(JobStage)
            .filter_by(job_id=job_id, stage=stage, target_language=lang).one())


def _cancel(session, job, stage_row=None) -> None:
    if stage_row is not None and stage_row.status == StageStatus.RUNNING:
        stage_row.status, stage_row.finished_at = StageStatus.CANCELED, utcnow()
    for row in session.query(JobStage).filter_by(job_id=job.id,
                                                 status=StageStatus.PENDING):
        row.status = StageStatus.CANCELED
    if job.status not in JobStatus.TERMINAL:
        job.status = JobStatus.CANCELED
        JOBS_TOTAL.labels(status=JobStatus.CANCELED).inc()
        audit(session, "job_canceled", job_id=job.id)
    session.commit()


def _cancel_watcher(job_id: str):
    """Polls the cancel flag in its own short-lived session, so it never expires
    the running stage's session and discard its uncommitted work."""
    Session = session_factory()

    def check():
        with Session() as s:
            if s.query(Job.cancel_requested).filter(Job.id == job_id).scalar():
                raise Canceled()
    return check


@celery_app.task(bind=True, name="stage")
def run_stage(self, stage: str, job_id: str, lang: str | None = None) -> None:
    """Run one pipeline stage with checkpointing, cancellation and retries."""
    Session = session_factory()
    with Session() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        row = _stage_row(session, job_id, stage, lang)

        if job.cancel_requested:
            return _cancel(session, job, row)
        if job.status in JobStatus.TERMINAL:
            return                          # drain the rest of a dead chain
        if row.status == StageStatus.COMPLETED and row.artifacts:
            return                          # resume: already done

        structlog.contextvars.bind_contextvars(
            job_id=job_id, stage=stage, language=lang, request_id=job.request_id)
        row.status, row.started_at, row.finished_at = StageStatus.RUNNING, utcnow(), None
        row.attempts += 1
        row.error = None
        row.celery_task_id = getattr(self.request, "id", None)
        audit(session, "stage_started", job_id=job_id, stage=stage, language=lang,
              attempt=row.attempts, request_id=job.request_id)
        session.commit()

        started = time.monotonic()
        processing = get_settings().processing
        try:
            fault = processing.fault_inject_stage
            if fault in (stage, f"{stage}:{lang}") and job.retry_count == 0:
                raise RuntimeError(f"fault injected at {fault}")
            row.artifacts = STAGES[stage](session, job, lang=lang,
                                          check_cancel=_cancel_watcher(job_id))
            row.status, row.finished_at = StageStatus.COMPLETED, utcnow()
            STAGE_SECONDS.labels(stage=stage).observe(time.monotonic() - started)
            audit(session, "stage_completed", job_id=job_id, stage=stage,
                  language=lang, seconds=round(time.monotonic() - started, 2))
            session.commit()
        except Canceled:
            session.rollback()
            _cancel(session, session.get(Job, job_id),
                    _stage_row(session, job_id, stage, lang))
        except InvalidVideo as e:           # bad input: retrying cannot help
            session.rollback()
            _fail(session, job_id, stage, lang, f"invalid input: {e}")
            raise
        except TRANSIENT as e:
            session.rollback()
            attempt = self.request.retries if self.request else 0
            if attempt < processing.task_max_retries:
                audit(session, "stage_retrying", job_id=job_id, stage=stage,
                      language=lang, error=str(e), retry=attempt + 1)
                session.commit()
                raise self.retry(exc=e, countdown=(
                    processing.retry_backoff_sec * 2 ** attempt))
            _fail(session, job_id, stage, lang, f"transient failure exhausted: {e}")
            raise
        except SoftTimeLimitExceeded:
            session.rollback()
            _fail(session, job_id, stage, lang, "stage timed out")
            raise
        except Exception as e:
            session.rollback()
            _fail(session, job_id, stage, lang, str(e))
            raise
        finally:
            structlog.contextvars.unbind_contextvars(
                "job_id", "stage", "language", "request_id")


def _fail(session, job_id: str, stage: str, lang: str | None, error: str) -> None:
    job = session.get(Job, job_id)
    row = _stage_row(session, job_id, stage, lang)
    row.status, row.finished_at, row.error = StageStatus.FAILED, utcnow(), error[:2000]
    audit(session, "stage_failed", job_id=job_id, stage=stage, language=lang,
          error=error[:500])
    if stage in SHARED_STAGES:              # nothing downstream can run
        job.status, job.error = JobStatus.FAILED, f"{stage}: {error[:500]}"
        JOBS_TOTAL.labels(status=JobStatus.FAILED).inc()
    session.commit()


def _stage_sig(stage: str, job_id: str, lang: str | None = None):
    timeout = get_settings().processing.stage_timeouts.get(stage, 1800)
    return run_stage.si(stage, job_id, lang).set(
        soft_time_limit=timeout, time_limit=timeout + 30)


@celery_app.task(name="fan_out")
def fan_out(job_id: str) -> None:
    """After the shared stages, run one chain per target language. They are
    independent, so one language failing leaves the others to finish."""
    Session = session_factory()
    with Session() as session:
        job = session.get(Job, job_id)
        if job is None or job.status in JobStatus.TERMINAL:
            return
        if job.cancel_requested:
            return _cancel(session, job)
        languages = list(job.target_languages)

    for lang in languages:
        branch = chain(*[_stage_sig(s, job_id, lang) for s in LANGUAGE_STAGES],
                       settle.si(job_id))
        try:
            branch.apply_async(link_error=branch_failed.si(job_id, lang))
        except Exception:                   # eager mode raises here instead
            branch_failed(job_id, lang)


@celery_app.task(name="branch_failed")
def branch_failed(job_id: str, lang: str) -> None:
    """Mark a language's remaining stages failed, then let the job settle."""
    Session = session_factory()
    with Session() as session:
        for row in (session.query(JobStage)
                    .filter_by(job_id=job_id, target_language=lang)
                    .filter(JobStage.status.in_([StageStatus.PENDING,
                                                 StageStatus.RUNNING]))):
            row.status, row.finished_at = StageStatus.FAILED, utcnow()
            row.error = row.error or "an earlier stage in this language failed"
        session.commit()
    settle(job_id)


@celery_app.task(name="settle")
def settle(job_id: str) -> None:
    """Give the job its final status once every language branch is terminal.
    Idempotent, so every branch can call it."""
    Session = session_factory()
    with Session() as session:
        job = session.get(Job, job_id)
        if job is None or job.status != JobStatus.PROCESSING:
            return
        done, failed = [], []
        for lang in job.target_languages:
            statuses = {r.stage: r.status for r in session.query(JobStage)
                        .filter_by(job_id=job_id, target_language=lang)}
            values = [statuses.get(s) for s in LANGUAGE_STAGES]
            if all(v == StageStatus.COMPLETED for v in values):
                done.append(lang)
            elif any(v in (StageStatus.FAILED, StageStatus.CANCELED) for v in values):
                failed.append(lang)
            else:
                return                      # still in flight
        if job.cancel_requested:
            job.status = JobStatus.CANCELED
        elif failed and done:
            job.status = JobStatus.COMPLETED_WITH_ERRORS
            job.error = f"failed languages: {', '.join(failed)}"
        elif failed:
            job.status = JobStatus.FAILED
            job.error = f"all languages failed: {', '.join(failed)}"
        else:
            job.status = JobStatus.COMPLETED
        JOBS_TOTAL.labels(status=job.status).inc()
        audit(session, "job_finished", job_id=job_id, status=job.status,
              succeeded=done, failed=failed)
        session.commit()


def pipeline_for(job_id: str):
    return chain(*[_stage_sig(s, job_id) for s in SHARED_STAGES],
                 fan_out.si(job_id))


@celery_app.task(name="dispatch")
def dispatch() -> None:
    """Runs every 5s (celery beat). Starts queued jobs while under the
    concurrency cap, and cleans up after workers that died mid-stage."""
    settings = get_settings()
    Session = session_factory()
    starting: list[str] = []
    with Session() as session:
        # a stage stuck RUNNING past its timeout means the worker died: nothing
        # will ever finish it, so fail it rather than leave the job hanging
        now = utcnow()
        for row in session.query(JobStage).filter_by(status=StageStatus.RUNNING):
            limit = settings.processing.stage_timeouts.get(row.stage, 1800) + 120
            started = row.started_at
            if started is None:
                continue
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            if (now - started).total_seconds() < limit:
                continue
            row.status, row.finished_at = StageStatus.FAILED, now
            row.error = f"stage exceeded {limit}s without finishing (worker lost?)"
            job = session.get(Job, row.job_id)
            audit(session, "stage_reaped", job_id=row.job_id, stage=row.stage)
            if job and row.stage in SHARED_STAGES and job.status == JobStatus.PROCESSING:
                job.status, job.error = JobStatus.FAILED, f"{row.stage}: worker lost"
                JOBS_TOTAL.labels(status=JobStatus.FAILED).inc()
        session.commit()

        # jobs with nothing running: either a pending cancel, or branches that
        # finished without anyone settling the job
        idle = []
        for job in session.query(Job).filter_by(status=JobStatus.PROCESSING):
            if not any(r.status == StageStatus.RUNNING for r in job.stages):
                (_cancel(session, job) if job.cancel_requested else idle.append(job.id))

        processing = session.query(Job).filter_by(status=JobStatus.PROCESSING).count()
        for job_id, in (session.query(Job.id).filter_by(status=JobStatus.QUEUED)
                        .order_by(Job.created_at)
                        .limit(max(0, settings.limits.max_concurrent_jobs - processing))):
            # conditional UPDATE: only the run that flips QUEUED->PROCESSING owns
            # the job, so two schedulers can't both dispatch it
            if session.query(Job).filter(Job.id == job_id,
                                         Job.status == JobStatus.QUEUED).update(
                    {"status": JobStatus.PROCESSING}, synchronize_session=False):
                audit(session, "job_dispatched", job_id=job_id)
                starting.append(job_id)
        session.commit()
        ACTIVE_JOBS.set(processing + len(starting))

    for job_id in idle:
        settle(job_id)

    for job_id in starting:
        try:
            pipeline_for(job_id).apply_async()
        except Exception as e:
            # eager mode surfaces stage failures here (already recorded); in real
            # mode the broker was unreachable, so release the claim and retry next tick
            log.warning("dispatch_failed", job_id=job_id, error=str(e))
            if not celery_app.conf.task_always_eager:
                with Session() as session:
                    job = session.get(Job, job_id)
                    if job and job.status == JobStatus.PROCESSING and not any(
                            r.status == StageStatus.RUNNING for r in job.stages):
                        job.status = JobStatus.QUEUED
                        session.commit()


@worker_init.connect
def _serve_metrics(**_):
    """Prefork children write metrics to a shared dir; the main process serves
    the aggregate so stage timings are actually scrapable."""
    if not multiproc_dir():
        return
    try:
        from prometheus_client import start_http_server
        port = int(os.environ.get("WORKER_METRICS_PORT", "9100"))
        start_http_server(port, registry=build_registry())
        log.info("worker_metrics_serving", port=port)
    except OSError as e:
        log.warning("worker_metrics_unavailable", error=str(e))
