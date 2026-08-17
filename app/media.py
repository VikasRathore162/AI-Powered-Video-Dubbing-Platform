"""ffmpeg/ffprobe wrappers and SRT writing. Always explicit args, never shell=True."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


class MediaError(Exception):
    pass


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"{args[0]} timed out after {timeout}s") from e
    if proc.returncode != 0:
        raise MediaError(f"{args[0]} failed ({proc.returncode}): {proc.stderr[-800:]}")
    return proc


def probe(path: Path, timeout: int = 60) -> dict:
    """Distilled ffprobe: duration, container, whether it has real video/audio.
    Raises MediaError on unreadable input."""
    proc = _run(["ffprobe", "-v", "error", "-show_format", "-show_streams",
                 "-of", "json", str(path)], timeout=timeout)
    info = json.loads(proc.stdout)
    streams = info.get("streams", [])
    if not streams:
        raise MediaError("no decodable streams found")
    fmt = info.get("format", {})
    # cover art is a video stream with disposition attached_pic — an mp3 with
    # album art must not pass as a dubbable video
    video = [s for s in streams if s.get("codec_type") == "video"
             and not s.get("disposition", {}).get("attached_pic")]
    return {
        "duration_sec": float(fmt.get("duration", 0) or 0),
        "format_name": fmt.get("format_name", ""),
        "has_video": bool(video),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "video_codec": next((s.get("codec_name") for s in video), None),
        "audio_codec": next((s.get("codec_name") for s in streams
                             if s.get("codec_type") == "audio"), None),
    }


def duration_of(path: Path) -> float:
    proc = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)], timeout=60)
    return float(proc.stdout.strip())


def extract_wav(src: Path, out: Path, rate: int = 16000, timeout: int = 300) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ar", str(rate), "-ac", "1",
          "-c:a", "pcm_s16le", str(out)], timeout=timeout)


def atempo_chain(ratio: float) -> str:
    """ffmpeg's atempo accepts [0.5, 2] usefully per filter; chain for the rest."""
    if ratio <= 0:
        raise ValueError(f"tempo ratio must be positive, got {ratio}")
    parts: list[float] = []
    while ratio > 2.0:
        parts.append(2.0)
        ratio /= 2.0
    while ratio < 0.5:
        parts.append(0.5)
        ratio /= 0.5
    parts.append(round(ratio, 4))
    return ",".join(f"atempo={p}" for p in parts)


def trim_silence(src: Path, out: Path, rate: int = 24000, threshold_db: int = -50,
                 timeout: int = 120) -> None:
    """Strip leading/trailing silence from a synthesized clip.

    Neural TTS pads its output (edge-tts: ~0.2s before, ~0.9s after). Left in, that
    padding delays every dubbed line by its lead and, worse, counts as clip duration
    — so the fitter time-stretches real speech to make room for silence. Only the
    first/last silent run goes; pauses inside the sentence are kept.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    cut = f"silenceremove=start_periods=1:start_threshold={threshold_db}dB:detection=peak"
    _run(["ffmpeg", "-y", "-i", str(src), "-filter:a", f"{cut},areverse,{cut},areverse",
          "-ar", str(rate), "-ac", "1", "-c:a", "pcm_s16le", str(out)], timeout=timeout)


def fit_clip(src: Path, out: Path, tempo: float, rate: int = 24000,
             timeout: int = 120) -> None:
    """Convert a synthesized clip to mono wav, time-stretched by `tempo` (>1 = faster)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    args = ["ffmpeg", "-y", "-i", str(src)]
    if abs(tempo - 1.0) > 1e-3:
        args += ["-filter:a", atempo_chain(tempo)]
    args += ["-ar", str(rate), "-ac", "1", "-c:a", "pcm_s16le", str(out)]
    _run(args, timeout=timeout)


def mix_and_mux(video: Path, dub_wav: Path, out: Path, duck_volume: float = 0.15,
                has_source_audio: bool = True, timeout: int = 600) -> None:
    """Mux the dub over the video, original audio kept underneath and ducked."""
    out.parent.mkdir(parents=True, exist_ok=True)
    args = ["ffmpeg", "-y", "-i", str(video), "-i", str(dub_wav)]
    if has_source_audio and duck_volume > 0:
        args += ["-filter_complex",
                 f"[0:a]volume={duck_volume}[bed];"
                 f"[bed][1:a]amix=inputs=2:duration=first:normalize=0[aout]",
                 "-map", "0:v", "-map", "[aout]"]
    else:
        args += ["-map", "0:v", "-map", "1:a"]
    args += ["-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(out)]
    _run(args, timeout=timeout)


def _timestamp(seconds: float) -> str:
    ms = round(max(seconds, 0.0) * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(entries: list[dict]) -> str:
    """[{start, end, text, speaker?}] -> SRT text, speaker label prefixing the line."""
    blocks = []
    for i, e in enumerate(entries, start=1):
        start, end = float(e["start"]), float(e["end"])
        if end <= start:
            end = start + 0.5
        text = e["text"].strip()
        if e.get("speaker"):
            text = f"[{e['speaker']}] {text}"
        blocks.append(f"{i}\n{_timestamp(start)} --> {_timestamp(end)}\n{text}\n")
    return "\n".join(blocks)
