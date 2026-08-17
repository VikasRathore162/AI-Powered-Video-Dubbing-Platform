"""The HTTP API: upload validation, job lifecycle, retry/cancel, security."""
from __future__ import annotations

import io
import subprocess

import pytest

from conftest import make_video, upload


# --- upload validation ----------------------------------------------------

def test_accepts_a_valid_video(client, video):
    r = upload(client, video)
    assert r.status_code == 202
    assert r.json()["status"] == "queued"
    assert r.json()["target_languages"] == ["es"]


def test_rejects_bad_extension(client):
    r = client.post("/api/v1/jobs",
                    files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
                    data={"target_languages": "es"})
    assert r.status_code == 400 and "unsupported format" in r.json()["detail"]


def test_rejects_bad_languages(client, video):
    assert upload(client, video, langs="").status_code in (400, 422)
    assert upload(client, video, langs="español!").status_code == 400
    assert upload(client, video, langs=" , ,").status_code == 400
    # 3-letter codes look plausible but the models only speak ISO 639-1
    assert upload(client, video, langs="spa").status_code == 400


def test_rejects_files_that_are_not_video(client):
    r = client.post("/api/v1/jobs",
                    files={"file": ("fake.mp4", io.BytesIO(b"not a video"), "video/mp4")},
                    data={"target_languages": "es"})
    assert r.status_code == 422


def test_rejects_video_without_audio(client, tmp_path):
    silent = make_video(tmp_path / "silent.mp4", duration=3, audio=False)
    r = upload(client, silent)
    assert r.status_code == 422 and "no audio" in r.json()["detail"]


def test_rejects_audio_with_cover_art(client, tmp_path):
    """An mp3 with album art has a video stream (disposition attached_pic) but
    is not a dubbable video."""
    audio, art, out = tmp_path / "a.wav", tmp_path / "art.png", tmp_path / "song.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=2", str(audio)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "color=c=red:s=64x64:d=1", "-frames:v", "1", str(art)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(audio), "-i", str(art),
                    "-map", "0:a", "-map", "1:v", "-c:a", "aac", "-c:v", "mjpeg",
                    "-disposition:v:0", "attached_pic", str(out)], check=True)
    r = upload(client, out)
    assert r.status_code == 422 and "no video stream" in r.json()["detail"]


def test_rejects_oversize_and_overlong(client, video, settings_env):
    settings_env(LIMITS__MAX_UPLOAD_MB="0")
    assert upload(client, video).status_code == 413
    settings_env(LIMITS__MAX_UPLOAD_MB="500", LIMITS__MAX_DURATION_SEC="5")
    r = upload(client, video)               # the fixture is 10s
    assert r.status_code == 422 and "duration" in r.json()["detail"]


def test_rejects_when_the_queue_is_full(client, video, settings_env):
    settings_env(LIMITS__MAX_QUEUE_LENGTH="1")
    assert upload(client, video).status_code == 202
    assert upload(client, video).status_code == 429


def test_rejected_upload_leaves_no_files(client):
    from app.providers import get_provider
    client.post("/api/v1/jobs",
                files={"file": ("fake.mp4", io.BytesIO(b"junk"), "video/mp4")},
                data={"target_languages": "es"})
    jobs = get_provider("storage").path("jobs")
    assert not (jobs.exists() and list(jobs.iterdir()))


def test_long_filename_is_truncated_not_fatal(client, video):
    """Postgres enforces the column width where SQLite does not."""
    from app.providers import get_provider
    with open(video, "rb") as f:
        r = client.post("/api/v1/jobs",
                        files={"file": ("x" * 300 + ".mp4", f, "video/mp4")},
                        data={"target_languages": "es"})
    assert r.status_code == 202
    assert len(client.get(f"/api/v1/jobs/{r.json()['job_id']}").json()
               ["source_filename"]) <= 255
    stored = {p.name for p in get_provider("storage").path("jobs").iterdir()}
    assert stored == {r.json()["job_id"]}       # no orphans


# --- the happy path -------------------------------------------------------

def test_full_pipeline(client, video, run_queue):
    job_id = upload(client, video).json()["job_id"]
    run_queue()

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed"
    assert job["progress_pct"] == 100.0
    assert job["source_language"] == "en"
    stages = {(s["stage"], s["language"]): s["status"] for s in job["stages"]}
    assert stages[("transcribe", None)] == "completed"
    assert stages[("assemble", "es")] == "completed"
    assert "es" in job["outputs"]

    transcript = client.get(f"/api/v1/jobs/{job_id}/transcript").json()
    assert transcript["language"] == "en"
    assert len(transcript["speakers"]) == 2
    assert {s["speaker"] for s in transcript["segments"]} == {"SPEAKER_00", "SPEAKER_01"}
    translated = client.get(f"/api/v1/jobs/{job_id}/transcript?language=es").json()
    assert all(s["text"].startswith("[es]") for s in translated["segments"])

    video_resp = client.get(f"/api/v1/jobs/{job_id}/video/es")
    assert video_resp.status_code == 200
    assert video_resp.headers["content-type"] == "video/mp4"
    assert len(video_resp.content) > 10_000
    assert "-->" in client.get(f"/api/v1/jobs/{job_id}/subtitles/es").text
    assert client.get(f"/api/v1/jobs/{job_id}/subtitles/en").status_code == 200

    events = [e["event"] for e in client.get(f"/api/v1/jobs/{job_id}/logs").json()]
    assert "job_created" in events and "job_finished" in events


def test_multiple_target_languages(client, video, run_queue):
    job_id = upload(client, video, langs="es,fr").json()["job_id"]
    run_queue()
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed"
    assert set(job["outputs"]) == {"es", "fr"}
    # every speaker keeps a voice for every language
    for speaker in client.get(f"/api/v1/jobs/{job_id}/transcript").json()["speakers"]:
        assert set(speaker["voices"]) == {"es", "fr"}


def test_outputs_are_409_until_ready(client, video):
    job_id = upload(client, video).json()["job_id"]          # queued, never run
    assert client.get(f"/api/v1/jobs/{job_id}/transcript").status_code == 409
    assert client.get(f"/api/v1/jobs/{job_id}/video/es").status_code == 409
    assert client.get(f"/api/v1/jobs/{job_id}/subtitles/es").status_code == 409


def test_listing_and_404(client, video):
    job_id = upload(client, video).json()["job_id"]
    listing = client.get("/api/v1/jobs").json()
    assert listing["total"] == 1 and listing["jobs"][0]["job_id"] == job_id
    assert client.get("/api/v1/jobs/nope").status_code == 404
    assert client.get("/api/v1/jobs?limit=-1").status_code == 422
    assert client.get("/api/v1/jobs?offset=-5").status_code == 422


# --- fault tolerance ------------------------------------------------------

def test_retry_resumes_from_the_failed_stage(client, video, run_queue, settings_env):
    settings_env(PROCESSING__FAULT_INJECT_STAGE="translate")
    job_id = upload(client, video).json()["job_id"]
    run_queue()

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "failed"
    stages = {(s["stage"], s["language"]): s for s in job["stages"]}
    assert stages[("translate", "es")]["status"] == "failed"
    assert stages[("transcribe", None)]["status"] == "completed"
    transcribe_attempts = stages[("transcribe", None)]["attempts"]

    assert client.post(f"/api/v1/jobs/{job_id}/retry").status_code == 202
    run_queue()

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed"
    stages = {(s["stage"], s["language"]): s for s in job["stages"]}
    # the expensive stage was NOT redone; only the failed one ran again
    assert stages[("transcribe", None)]["attempts"] == transcribe_attempts
    assert stages[("translate", "es")]["status"] == "completed"


def test_one_language_failing_leaves_the_others(client, video, run_queue, settings_env):
    settings_env(PROCESSING__FAULT_INJECT_STAGE="translate:fr")
    job_id = upload(client, video, langs="es,fr").json()["job_id"]
    run_queue()

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed_with_errors"
    assert "fr" in job["error"]
    assert set(job["outputs"]) == {"es"}
    assert client.get(f"/api/v1/jobs/{job_id}/video/es").status_code == 200
    assert client.get(f"/api/v1/jobs/{job_id}/video/fr").status_code == 409

    client.post(f"/api/v1/jobs/{job_id}/retry")
    run_queue()
    assert set(client.get(f"/api/v1/jobs/{job_id}").json()["outputs"]) == {"es", "fr"}


def test_shared_stage_failure_fails_the_job(client, video, run_queue, settings_env):
    settings_env(PROCESSING__FAULT_INJECT_STAGE="transcribe")
    job_id = upload(client, video).json()["job_id"]
    run_queue()
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "failed" and "transcribe" in job["error"]


def test_only_failed_jobs_can_retry(client, video):
    job_id = upload(client, video).json()["job_id"]          # queued
    assert client.post(f"/api/v1/jobs/{job_id}/retry").status_code == 409


def test_voices_survive_a_retry(client, video, run_queue, settings_env):
    """Casting is persisted at diarization, so a retry re-uses the same voices."""
    settings_env(PROCESSING__FAULT_INJECT_STAGE="assemble")
    job_id = upload(client, video).json()["job_id"]
    run_queue()
    before = {s["label"]: s["voices"]
              for s in client.get(f"/api/v1/jobs/{job_id}/transcript").json()["speakers"]}
    assert all(v.get("es") for v in before.values())

    client.post(f"/api/v1/jobs/{job_id}/retry")
    run_queue()
    after = {s["label"]: s["voices"]
             for s in client.get(f"/api/v1/jobs/{job_id}/transcript").json()["speakers"]}
    assert after == before


# --- cancel ---------------------------------------------------------------

def test_cancel_a_queued_job(client, video):
    job_id = upload(client, video).json()["job_id"]
    assert client.post(f"/api/v1/jobs/{job_id}/cancel").json()["status"] == "canceled"
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "canceled"
    assert all(s["status"] == "canceled" for s in job["stages"])


def test_cancel_is_409_once_terminal(client, video, run_queue):
    job_id = upload(client, video).json()["job_id"]
    run_queue()
    assert client.post(f"/api/v1/jobs/{job_id}/cancel").status_code == 409


def test_a_canceled_job_can_be_retried(client, video, run_queue):
    job_id = upload(client, video).json()["job_id"]
    client.post(f"/api/v1/jobs/{job_id}/cancel")
    assert client.post(f"/api/v1/jobs/{job_id}/retry").status_code == 202
    run_queue()
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"


def test_a_running_job_stops_at_the_next_stage(client, video, db, run_queue):
    from app.models import Job, JobStatus
    from app.worker import run_stage

    job_id = upload(client, video).json()["job_id"]
    db.get(Job, job_id).status = JobStatus.PROCESSING     # pretend it was dispatched
    db.commit()
    assert client.post(f"/api/v1/jobs/{job_id}/cancel").status_code == 202

    run_stage.apply(args=("probe", job_id))
    db.expire_all()
    assert db.get(Job, job_id).status == JobStatus.CANCELED


# --- security -------------------------------------------------------------

def test_rate_limit_fires(client, settings_env):
    settings_env(SECURITY__RATE_LIMIT="3/minute")
    codes = [client.post("/api/v1/jobs",
                         files={"file": ("x.txt", io.BytesIO(b"x"), "text/plain")},
                         data={"target_languages": "es"}).status_code
             for _ in range(5)]
    assert 429 in codes


def test_bearer_token_protects_everything_including_metrics(client, settings_env):
    settings_env(SECURITY__API_KEY="s3cret")
    assert client.get("/api/v1/jobs").status_code == 401
    assert client.get("/api/v1/jobs",
                      headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/v1/jobs",
                      headers={"X-API-Key": "s3cret"}).status_code == 401   # old scheme
    assert client.get("/api/v1/jobs",
                      headers={"Authorization": "s3cret"}).status_code == 401  # no scheme
    assert client.get("/api/v1/jobs",
                      headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics",
                      headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_health_and_metrics(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/metrics").status_code == 200
