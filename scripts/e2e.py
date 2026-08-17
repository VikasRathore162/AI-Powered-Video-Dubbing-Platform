"""Black-box E2E against a running stack (bare metal or docker compose).

Usage: python scripts/e2e.py [base_url]   (default http://localhost:8000)
Exercises: health, validation rejects, full dub job to completion, output
integrity (ffprobe/md5), subtitles, logs, cancel. Exits non-zero on any failure.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
FIXTURE = Path(__file__).parent.parent / "tests" / "two_speakers_en.mp4"
client = httpx.Client(base_url=BASE, timeout=60)
checks = 0


def ok(name: str, cond: bool, detail: str = ""):
    global checks
    checks += 1
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" ({detail})" if detail else ""))
    if not cond:
        sys.exit(1)


def audio_md5(path) -> str:
    return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a",
                           "-f", "md5", "-"], capture_output=True, text=True).stdout.strip()


print(f"== E2E against {BASE} ==")

ok("health", client.get("/healthz").json() == {"status": "ok"})

r = client.post("/api/v1/jobs", files={"file": ("x.txt", b"junk", "text/plain")},
                data={"target_languages": "es"})
ok("bad format rejected", r.status_code == 400, f"got {r.status_code}")

r = client.post("/api/v1/jobs", files={"file": ("x.mp4", b"junk" * 1000, "video/mp4")},
                data={"target_languages": "es"})
ok("non-video content rejected", r.status_code == 422, f"got {r.status_code}")

with open(FIXTURE, "rb") as f:
    r = client.post("/api/v1/jobs", files={"file": ("two_speakers_en.mp4", f, "video/mp4")},
                    data={"target_languages": "es"})
ok("upload accepted", r.status_code == 202, f"got {r.status_code}: {r.text[:200]}")
job_id = r.json()["job_id"]

seen = []
status = None
deadline = time.time() + 900  # first compose run may download models
while time.time() < deadline:
    d = client.get(f"/api/v1/jobs/{job_id}").json()
    status = d["status"]
    if not seen or seen[-1] != (status, d["progress_pct"]):
        seen.append((status, d["progress_pct"]))
        print(f"    {status} {d['progress_pct']}%")
    if status in ("completed", "completed_with_errors", "failed", "canceled"):
        break
    time.sleep(4)
ok("job completed", status == "completed",
   f"final={status} stages={json.dumps(d.get('stages'))[:400]}")
pcts = [p for _, p in seen]
ok("progress monotonic", pcts == sorted(pcts), str(pcts))

tr = client.get(f"/api/v1/jobs/{job_id}/transcript").json()
ok("language detected", tr["language"] == "en", tr["language"])
ok("two speakers found", len(tr["speakers"]) == 2, str(len(tr["speakers"])))
ok("segments transcribed", len(tr["segments"]) >= 3, str(len(tr["segments"])))

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "dubbed.mp4"
    out.write_bytes(client.get(f"/api/v1/jobs/{job_id}/video/es").content)
    ok("dubbed video downloads", out.stat().st_size > 50_000, f"{out.stat().st_size}B")
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.split()
    ok("valid mp4 (video+audio)", sorted(streams) == ["audio", "video"], str(streams))
    ok("audio actually dubbed", audio_md5(out) != audio_md5(FIXTURE))

srt = client.get(f"/api/v1/jobs/{job_id}/subtitles/es").text
ok("srt has cues and speakers", "-->" in srt and "SPEAKER_" in srt)
srt_src = client.get(f"/api/v1/jobs/{job_id}/subtitles/en")
ok("source-language srt", srt_src.status_code == 200)

events = [l["event"] for l in client.get(f"/api/v1/jobs/{job_id}/logs").json()]
ok("audit log complete", "job_created" in events and "job_finished" in events)

ok("completed job is not retryable",
   client.post(f"/api/v1/jobs/{job_id}/retry").status_code == 409)

# Cancel mid-flight, then retry: proves cancellation AND resume-from-checkpoint
# against the real broker (not eager mode).
with open(FIXTURE, "rb") as f:
    r = client.post("/api/v1/jobs", files={"file": ("c.mp4", f, "video/mp4")},
                    data={"target_languages": "es"})
cancel_id = r.json()["job_id"]

completed_before = {}
deadline = time.time() + 300
while time.time() < deadline:  # let some stages finish so resume has something to skip
    d = client.get(f"/api/v1/jobs/{cancel_id}").json()
    completed_before = {(s["stage"], s["language"]): s["attempts"]
                        for s in d["stages"] if s["status"] == "completed"}
    if len(completed_before) >= 2 or d["status"] in ("completed", "failed"):
        break
    time.sleep(0.5)

r = client.post(f"/api/v1/jobs/{cancel_id}/cancel")
ok("cancel accepted mid-flight", r.status_code == 202, f"got {r.status_code}")
for _ in range(60):
    s = client.get(f"/api/v1/jobs/{cancel_id}").json()["status"]
    if s in ("canceled", "completed"):
        break
    time.sleep(2)
ok("job canceled", s == "canceled", s)

r = client.post(f"/api/v1/jobs/{cancel_id}/retry")
ok("canceled job retryable", r.status_code == 202, f"got {r.status_code}")
deadline = time.time() + 300
while time.time() < deadline:
    d = client.get(f"/api/v1/jobs/{cancel_id}").json()
    if d["status"] in ("completed", "completed_with_errors", "failed"):
        break
    time.sleep(3)
ok("retried job completes", d["status"] == "completed", d["status"])
after = {(s["stage"], s["language"]): s["attempts"] for s in d["stages"]}
ok("retry resumed from checkpoints (completed stages did not re-run)",
   bool(completed_before) and all(after[k] == v for k, v in completed_before.items()),
   f"before={completed_before} after={ {k: after[k] for k in completed_before} }")
ok("dubbed output present after retry",
   client.get(f"/api/v1/jobs/{cancel_id}/video/es").status_code == 200)

print(f"== E2E OK: {checks} checks passed ==")
