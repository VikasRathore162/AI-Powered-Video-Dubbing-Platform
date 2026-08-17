"""Database: engine/session, ORM models, and the audit-log helper.

Tables: jobs, job_stages, speakers, segments, translations, audit_logs.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer,
                        String, Text, UniqueConstraint, create_engine, event)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column,
                            relationship, sessionmaker)

from app.config import get_settings
from app.obs import get_logger

log = get_logger("db")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    FAILED = "failed"
    CANCELED = "canceled"

    TERMINAL = {COMPLETED, COMPLETED_WITH_ERRORS, FAILED, CANCELED}
    RETRYABLE = {FAILED, CANCELED, COMPLETED_WITH_ERRORS}


class StageStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True,
                                    default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String(32), default=JobStatus.QUEUED, index=True)
    source_filename: Mapped[str] = mapped_column(String(255))
    source_key: Mapped[str] = mapped_column(String(512))
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    duration_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    probe: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    source_language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    target_languages: Mapped[list] = mapped_column(JSON, default=list)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # request that created the job, so worker events correlate with the upload
    request_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    stages: Mapped[list["JobStage"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="JobStage.id")
    speakers: Mapped[list["Speaker"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="Speaker.label")
    segments: Mapped[list["Segment"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="Segment.idx")


class JobStage(Base):
    __tablename__ = "job_stages"
    __table_args__ = (UniqueConstraint("job_id", "stage", "target_language"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    stage: Mapped[str] = mapped_column(String(32))
    target_language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default=StageStatus.PENDING)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifacts: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    job: Mapped[Job] = relationship(back_populates="stages")


class Speaker(Base):
    __tablename__ = "speakers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    label: Mapped[str] = mapped_column(String(32))  # SPEAKER_00, by first appearance
    gender: Mapped[str | None] = mapped_column(String(1), nullable=True)  # M/F/None
    total_speech_sec: Mapped[float] = mapped_column(Float, default=0.0)
    segment_count: Mapped[int] = mapped_column(Integer, default=0)
    voices: Mapped[dict] = mapped_column(JSON, default=dict)  # {lang: voice_id}

    job: Mapped[Job] = relationship(back_populates="speakers")


class Segment(Base):
    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    speaker_id: Mapped[int | None] = mapped_column(ForeignKey("speakers.id"), nullable=True)
    idx: Mapped[int] = mapped_column(Integer)
    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    words: Mapped[list | None] = mapped_column(JSON, nullable=True)

    job: Mapped[Job] = relationship(back_populates="segments")
    speaker: Mapped[Speaker | None] = relationship()
    translations: Mapped[list["Translation"]] = relationship(
        back_populates="segment", cascade="all, delete-orphan")


class Translation(Base):
    __tablename__ = "translations"
    __table_args__ = (UniqueConstraint("segment_id", "language"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    segment_id: Mapped[int] = mapped_column(ForeignKey("segments.id"), index=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    language: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text)

    segment: Mapped[Segment] = relationship(back_populates="translations")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    event: Mapped[str] = mapped_column(String(64))
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(36), nullable=True)


def audit(session, event: str, job_id: str | None = None,
          request_id: str | None = None, **detail) -> None:
    """Record a job event in the DB (caller commits) and the structured log.
    Served back to users as the job's processing log."""
    session.add(AuditLog(job_id=job_id, event=event,
                         detail=detail or None, request_id=request_id))
    log.info(event, job_id=job_id, **detail)


_engine = None
_Session = None


def get_engine():
    """SQLite (dev, WAL) or Postgres (compose) from one db_url."""
    global _engine, _Session
    if _engine is None:
        url = get_settings().db_url
        kwargs = {}
        if url.startswith("sqlite"):
            db_path = url.replace("sqlite:///", "")
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 15}
        _engine = create_engine(url, **kwargs)
        if url.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def _pragmas(dbapi_conn, _):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=5000")
                cur.close()
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def session_factory() -> sessionmaker:
    get_engine()
    return _Session
