"""One-off setup tasks. Run inside the container:

    docker compose run --rm tools python scripts/setup.py models   # ~1.1GB, once
    docker compose run --rm tools python scripts/setup.py fixture  # test video
    docker compose run --rm tools python scripts/setup.py cloud    # seed SSM
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

FIXTURE = Path("tests/two_speakers_en.mp4")

# two clearly different voices, so diarization has something real to separate
LINES = [
    ("en-US-GuyNeural", "Hello, my name is Alex, and I am the first speaker in this recording."),
    ("en-US-JennyNeural", "Hi there, I am Beth, the second speaker, and I do not always agree with Alex."),
    ("en-US-GuyNeural", "The quick brown fox jumps over the lazy dog near the river bank."),
    ("en-US-JennyNeural", "The weather today is sunny with a gentle breeze coming from the north."),
]


def models() -> None:
    """Pre-download everything so the first job never stalls on a download."""
    from faster_whisper import WhisperModel
    for name in ("base", "small"):          # base for tests, small in production
        print(f"  whisper {name}")
        WhisperModel(name, device="cpu", compute_type="int8")

    print("  speechbrain ECAPA")
    from speechbrain.inference.speaker import EncoderClassifier
    EncoderClassifier.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.environ.get("SPEECHBRAIN_CACHE",
                               os.path.expanduser("~/.cache/speechbrain/ecapa")))

    print("  argos language pairs")
    import argostranslate.package as pkg
    pkg.update_package_index()
    installed = {(p.from_code, p.to_code) for p in pkg.get_installed_packages()}
    wanted = {("en", "es"), ("en", "fr"), ("es", "en"), ("fr", "en")}
    for available in pkg.get_available_packages():
        pair = (available.from_code, available.to_code)
        if pair in wanted and pair not in installed:
            print(f"    {pair[0]}->{pair[1]}")
            pkg.install_from_path(available.download())
    print("models ready")


def fixture(repeat: int = 1, out: Path = FIXTURE) -> None:
    """Build the test video from TTS speech: alternating speakers, 0.8s gaps,
    a solid colour video track. REPEAT=29 makes a ~10 minute clip."""
    import tempfile
    repeat = int(os.environ.get("REPEAT", repeat))
    out = Path(os.environ.get("OUT", out))

    def run(*args):
        subprocess.run(args, check=True, capture_output=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i, (voice, text) in enumerate(LINES):
            subprocess.run([sys.executable, "-m", "edge_tts", "--voice", voice,
                            "--text", text, "--write-media", str(tmp / f"{i}.mp3")],
                           check=True, capture_output=True)
            run("ffmpeg", "-y", "-v", "error", "-i", str(tmp / f"{i}.mp3"),
                "-ar", "16000", "-ac", "1", str(tmp / f"{i}.wav"))
        run("ffmpeg", "-y", "-v", "error", "-f", "lavfi",
            "-i", "anullsrc=r=16000:cl=mono", "-t", "0.8", str(tmp / "gap.wav"))

        listing = tmp / "list.txt"
        listing.write_text("".join(
            f"file '{tmp}/{i}.wav'\nfile '{tmp}/gap.wav'\n"
            for _ in range(repeat) for i in range(len(LINES))))
        run("ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
            "-i", str(listing), "-c", "copy", str(tmp / "speech.wav"))

        out.parent.mkdir(parents=True, exist_ok=True)
        run("ffmpeg", "-y", "-v", "error", "-f", "lavfi",
            "-i", "color=c=steelblue:s=320x240:r=15", "-i", str(tmp / "speech.wav"),
            "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", str(out))
    print(f"fixture written to {out}")


def cloud() -> None:
    """Seed a cloud config service, to show settings and a secret coming from one."""
    import boto3
    prefix = os.environ.get("CLOUD_CONFIG_PREFIX", "/dubbing/")
    params = {
        "limits__max_duration_sec": ("300", "String"),
        "limits__max_concurrent_jobs": ("3", "String"),
        "processing__task_max_retries": ("5", "String"),
        "providers__tts__name": ("edge", "String"),
        "providers__tts__options": (json.dumps({}), "String"),
        # a credential — encrypted at rest, unlike a config file
        "elevenlabs_api_key": ("seeded-from-parameter-store", "SecureString"),
    }
    ssm = boto3.client("ssm", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or None)
    for name, (value, kind) in params.items():
        ssm.put_parameter(Name=f"{prefix}{name}", Value=value, Type=kind, Overwrite=True)
        print(f"  {prefix}{name} = {'***' if kind == 'SecureString' else value}")
    print(f"seeded {len(params)} parameters under {prefix}")


if __name__ == "__main__":
    tasks = {"models": models, "fixture": fixture, "cloud": cloud}
    task = sys.argv[1] if len(sys.argv) > 1 else ""
    if task not in tasks:
        sys.exit(f"usage: setup.py {{{'|'.join(tasks)}}}")
    tasks[task]()
