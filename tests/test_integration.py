"""The real thing: actual models, actual speech, end to end.

    docker compose run --rm tools env RUN_FULL_PIPELINE=1 pytest -m integration

Needs the models downloaded (scripts/setup.py models) and internet for edge-tts.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "two_speakers_en.mp4"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_FULL_PIPELINE") != "1",
                       reason="set RUN_FULL_PIPELINE=1 to run the real models"),
    pytest.mark.skipif(not FIXTURE.exists(),
                       reason="fixture missing — run scripts/setup.py fixture"),
]


def test_real_pipeline_english_to_spanish(client, run_queue):
    with open(FIXTURE, "rb") as f:
        r = client.post("/api/v1/jobs",
                        files={"file": ("two_speakers_en.mp4", f, "video/mp4")},
                        data={"target_languages": "es"})
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    run_queue()

    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "completed", job["stages"]
    assert job["source_language"] == "en"        # detected, not told

    transcript = client.get(f"/api/v1/jobs/{job_id}/transcript").json()
    spoken = " ".join(s["text"].lower() for s in transcript["segments"])
    assert any(word in spoken for word in ("alex", "fox", "weather"))
    assert len(transcript["speakers"]) == 2      # the two voices were separated

    translated = client.get(f"/api/v1/jobs/{job_id}/transcript?language=es").json()
    spanish = " ".join(s["text"].lower() for s in translated["segments"])
    assert spanish != spoken and len(spanish) > 20

    assert len(client.get(f"/api/v1/jobs/{job_id}/video/es").content) > 50_000
    assert "-->" in client.get(f"/api/v1/jobs/{job_id}/subtitles/es").text
