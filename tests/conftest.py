"""Test setup: fake AI providers, a real generated video, eager Celery.

The fast suite swaps the AI providers for fakes but keeps everything else real —
real ffmpeg, real timing maths, real database — so the parts that are expensive
to get wrong are genuinely exercised in seconds. RUN_FULL_PIPELINE=1 switches to
the real models for the integration test.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import wave

TMP = tempfile.mkdtemp(prefix="dubbing-tests-")
REAL = os.environ.get("RUN_FULL_PIPELINE") == "1"

# Set before importing the app: settings, the Celery app and the DB engine all
# read these at import time. CONFIG_FILE is forced away from the deployment
# config so tests own their configuration completely.
os.environ["CONFIG_FILE"] = "/nonexistent-tests-configure-by-env.yaml"
# Never inherit the deployment's DB_URL: the suite drops and recreates every
# table, so pointing it at a live database would wipe it. Postgres runs are
# opt-in through TEST_DB_URL.
os.environ["DB_URL"] = os.environ.get("TEST_DB_URL") or f"sqlite:///{TMP}/test.db"
os.environ["PROVIDERS__STORAGE__NAME"] = "local"
os.environ["PROVIDERS__STORAGE__OPTIONS"] = f'{{"root": "{TMP}/storage"}}'
os.environ["SECURITY__RATE_LIMIT"] = "10000/minute"
for kind in ("STT", "DIARIZATION", "TRANSLATION", "TTS"):
    # sources deep-merge key by key, so set both: NAME alone would still inherit
    # OPTIONS meant for a different provider
    os.environ[f"PROVIDERS__{kind}__NAME"] = "faster_whisper" if REAL else "fake"
    os.environ[f"PROVIDERS__{kind}__OPTIONS"] = "{}"
if REAL:
    os.environ["PROVIDERS__STT__OPTIONS"] = '{"model": "base", "compute_type": "int8"}'
    os.environ["PROVIDERS__DIARIZATION__NAME"] = "ecapa_cluster"
    os.environ["PROVIDERS__TRANSLATION__NAME"] = "argos"
    os.environ["PROVIDERS__TTS__NAME"] = "edge"

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.models import Base, get_engine, session_factory  # noqa: E402
from app.providers import clear_instances, register  # noqa: E402
from app.providers.diarization import Diarization, Diarizer  # noqa: E402
from app.providers.stt import STT, Segment, Transcript  # noqa: E402
from app.providers.translation import Translator  # noqa: E402
from app.providers.tts import TTS, Voice  # noqa: E402
from app.worker import celery_app  # noqa: E402

CANNED = [(0.5, 2.2, "Hello there, my name is Alex."),
          (2.9, 4.6, "Hi Alex, I am Beth, nice to meet you."),
          (5.3, 7.1, "The weather is lovely today, is it not?"),
          (7.8, 9.4, "Indeed it is, a perfect day for a walk.")]


@register("stt", "fake")
class FakeSTT(STT):
    def transcribe(self, wav, language=None) -> Transcript:
        return Transcript(language or "en",
                          [Segment(s, e, t) for s, e, t in CANNED])


@register("diarization", "fake")
class FakeDiarizer(Diarizer):
    def diarize(self, wav, spans) -> Diarization:
        return Diarization(labels=[i % 2 for i in range(len(spans))],
                           genders={0: "M", 1: "F"})


@register("translation", "fake")
class FakeTranslator(Translator):
    def translate(self, texts, src, tgt):
        return [f"[{tgt}] {t}" for t in texts]


@register("tts", "fake")
class FakeTTS(TTS):
    """Writes a sine tone whose length scales with the text, so the real timing
    maths and audio assembly are exercised without a network call."""

    def synthesize(self, text: str, voice: str, out) -> None:
        seconds = max(0.3, len(text) * 0.05)
        sr = 24000
        freq = 220.0 if voice.endswith("male") else 340.0
        t = np.arange(int(seconds * sr)) / sr
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((np.sin(2 * np.pi * freq * t) * 12000).astype(np.int16).tobytes())

    def voices(self, language: str) -> list[Voice]:
        return [Voice("fake-male", "M"), Voice("fake-female", "F")]


def make_video(path, duration: float = 10.0, audio: bool = True):
    """A real mp4 via ffmpeg: colour bars plus a sine tone (or no audio track)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    args = ["ffmpeg", "-y", "-f", "lavfi",
            "-i", f"color=c=steelblue:s=320x240:r=15:d={duration}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
                 "-c:a", "aac", "-b:a", "96k"]
    args += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-shortest", str(path)]
    subprocess.run(args, check=True, capture_output=True)
    return path


@pytest.fixture(scope="session")
def video(tmp_path_factory):
    return str(make_video(tmp_path_factory.mktemp("media") / "sample.mp4"))


@pytest.fixture(autouse=True)
def clean_state():
    """Fresh tables, empty storage and eager Celery for every test."""
    import shutil
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = False   # failures land in the DB
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    shutil.rmtree(f"{TMP}/storage/jobs", ignore_errors=True)
    yield
    get_settings.cache_clear()
    if not REAL:
        clear_instances()


@pytest.fixture
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db():
    session = session_factory()()
    yield session
    session.close()


@pytest.fixture
def run_queue():
    """Trigger the dispatcher that celery beat would normally run."""
    def go():
        from app.worker import dispatch
        dispatch.apply()
    return go


@pytest.fixture
def settings_env(monkeypatch):
    """Override settings via env for one test."""
    def apply(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
        get_settings.cache_clear()
        clear_instances()
    yield apply
    get_settings.cache_clear()


def upload(client, video, langs="es", **extra):
    with open(video, "rb") as f:
        return client.post("/api/v1/jobs",
                           files={"file": ("clip.mp4", f, "video/mp4")},
                           data={"target_languages": langs, **extra})
