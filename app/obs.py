"""Observability: structured logging and Prometheus metrics.

The API serves /metrics for its own process. Celery workers are prefork, so the
processing metrics are produced in children: with PROMETHEUS_MULTIPROC_DIR set,
children write there and the worker's main process serves the aggregate.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import structlog
from prometheus_client import (REGISTRY, CollectorRegistry, Counter, Gauge,
                               Histogram, multiprocess)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout,
                        level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "app"):
    return structlog.get_logger(name)


def multiproc_dir() -> Path | None:
    d = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    return Path(d) if d else None


def _prepare_multiproc_dir() -> None:
    """Must run BEFORE any metric is defined: in multiprocess mode
    prometheus_client opens its mmap files at definition time, so a missing
    directory crashes the worker at import. Also clears a previous run's files,
    which would otherwise be double-counted."""
    d = multiproc_dir()
    if not d:
        return
    d.mkdir(parents=True, exist_ok=True)
    for stale in d.glob("*.db"):
        stale.unlink(missing_ok=True)


_prepare_multiproc_dir()

JOBS_TOTAL = Counter("dubbing_jobs_total", "Jobs by terminal status", ["status"])
UPLOADS_TOTAL = Counter("dubbing_uploads_total", "Upload attempts", ["outcome"])
STAGE_SECONDS = Histogram(
    "dubbing_stage_seconds", "Stage wall-clock seconds", ["stage"],
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800))
ACTIVE_JOBS = Gauge("dubbing_active_jobs", "Jobs currently processing",
                    multiprocess_mode="livesum")


def build_registry() -> CollectorRegistry:
    """Aggregating registry when running prefork, else the process registry."""
    d = multiproc_dir()
    if not d:
        return REGISTRY
    d.mkdir(parents=True, exist_ok=True)
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=str(d))
    return registry
