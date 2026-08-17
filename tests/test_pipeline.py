"""The pipeline itself: timing maths, subtitles, voice casting, and the real
media output produced end to end (fake AI, real ffmpeg)."""
from __future__ import annotations

import re
import subprocess

import numpy as np
import pytest

from app.media import atempo_chain, to_srt, trim_silence
from app.pipeline import fit_placements, key
from app.providers.diarization import estimate_gender
from app.providers.tts import Voice, assign_voices
from conftest import upload

MAX_TEMPO, GUARD = 1.5, 0.1


def fit(segments, durations, video_duration=60.0):
    return fit_placements(segments, durations, video_duration, MAX_TEMPO, GUARD)


# --- timing: the four branches of fitting a dub into its slot -------------

def test_short_clip_is_placed_as_is():
    [p] = fit([{"idx": 0, "start": 1.0, "end": 3.0}], {0: 1.5})
    assert p["tempo"] == 1.0 and p["fitted_duration"] == 1.5 and p["overflow_sec"] == 0


def test_never_slows_speech_down():
    [p] = fit([{"idx": 0, "start": 0.0, "end": 5.0}], {0: 1.0})
    assert p["tempo"] == 1.0        # silence is better than a drawl


def test_spills_into_the_following_gap_instead_of_speeding_up():
    segments = [{"idx": 0, "start": 0.0, "end": 2.0}, {"idx": 1, "start": 3.0, "end": 4.0}]
    p = fit(segments, {0: 2.5, 1: 0.5})[0]
    assert p["tempo"] == 1.0 and p["overflow_sec"] == 0


def test_speeds_up_within_the_clamp():
    segments = [{"idx": 0, "start": 0.0, "end": 2.0}, {"idx": 1, "start": 2.2, "end": 4.0}]
    p = fit(segments, {0: 3.0, 1: 1.0})[0]
    assert 1.4 < p["tempo"] <= MAX_TEMPO and p["overflow_sec"] == 0


def test_clamps_and_accepts_overlap_rather_than_cutting_words():
    segments = [{"idx": 0, "start": 0.0, "end": 2.0}, {"idx": 1, "start": 2.1, "end": 4.0}]
    p = fit(segments, {0: 4.0, 1: 1.0})[0]
    assert p["tempo"] == MAX_TEMPO
    assert p["fitted_duration"] == pytest.approx(4.0 / MAX_TEMPO, abs=0.01)
    assert p["overflow_sec"] > 0.5


def test_last_segment_may_spill_to_the_end_of_the_video():
    [p] = fit([{"idx": 0, "start": 8.0, "end": 9.0}], {0: 2.0}, video_duration=12.0)
    assert p["tempo"] == 1.0


def test_segments_without_a_clip_are_skipped():
    segments = [{"idx": 0, "start": 0.0, "end": 1.0}, {"idx": 1, "start": 2.0, "end": 3.0}]
    assert [p["idx"] for p in fit(segments, {1: 0.5})] == [1]


@pytest.mark.parametrize("ratio,expected", [
    (1.4, "atempo=1.4"),
    (3.5, "atempo=2.0,atempo=1.75"),        # above ffmpeg's per-filter range
    (0.3, "atempo=0.5,atempo=0.6"),
])
def test_atempo_chains_outside_the_supported_range(ratio, expected):
    assert atempo_chain(ratio) == expected


def test_atempo_rejects_nonsense():
    with pytest.raises(ValueError):
        atempo_chain(0)


def test_trim_silence_strips_tts_padding_but_keeps_inner_pauses(tmp_path):
    """TTS pads its clips; left in, every dubbed line starts late and the fitter
    stretches speech to make room for silence."""
    import soundfile as sf

    sr, tone = 24000, lambda n: 0.5 * np.sin(2 * np.pi * 440 * np.arange(n) / 24000)
    padded = np.concatenate([np.zeros(int(0.4 * sr)),      # leading pad
                             tone(int(0.3 * sr)),
                             np.zeros(int(0.2 * sr)),      # pause inside the sentence
                             tone(int(0.3 * sr)),
                             np.zeros(int(0.9 * sr))])     # trailing pad
    src, out = tmp_path / "padded.wav", tmp_path / "trimmed.wav"
    sf.write(src, padded.astype(np.float32), sr)

    trim_silence(src, out, rate=sr)
    kept, _ = sf.read(out)
    # 0.8s of speech-plus-inner-pause survives; the 1.3s of outer padding does not
    assert 0.7 < len(kept) / sr < 0.95, len(kept) / sr


# --- subtitles ------------------------------------------------------------

def test_srt_format():
    srt = to_srt([{"start": 0.5, "end": 2.25, "text": "Hello."},
                  {"start": 3.0, "end": 4.5, "text": "World.", "speaker": "SPEAKER_01"}])
    blocks = srt.strip().split("\n\n")
    assert blocks[0].split("\n")[:3] == ["1", "00:00:00,500 --> 00:00:02,250", "Hello."]
    assert blocks[1].split("\n")[2] == "[SPEAKER_01] World."


def test_srt_clamps_a_zero_length_cue_and_rolls_over_hours():
    assert "00:00:05,000 --> 00:00:05,500" in to_srt(
        [{"start": 5.0, "end": 4.0, "text": "clamped"}])
    assert "01:01:01,500 --> 01:01:02,000" in to_srt(
        [{"start": 3661.5, "end": 3662.0, "text": "hi"}])


# --- voice casting --------------------------------------------------------

CATALOG = [Voice("m1", "M"), Voice("f1", "F"), Voice("m2", "M"), Voice("f2", "F")]


def test_voices_match_gender_and_never_repeat():
    assert assign_voices(["M", "F"], CATALOG) == ["m1", "f1"]
    assert assign_voices(["F", "M"], CATALOG) == ["f1", "m1"]
    assert assign_voices([None, None], CATALOG) == ["m1", "f1"]


def test_casting_is_deterministic_so_a_retry_reproduces_it():
    assert assign_voices(["M", "F", "M"], CATALOG) == assign_voices(["M", "F", "M"], CATALOG)


def test_more_speakers_than_voices_reuses_within_gender():
    """Sharing a voice beats being dubbed by the wrong gender — found on a real
    interview where a female guest was cast with a male Spanish voice."""
    out = assign_voices(["M"] * 6, CATALOG)
    assert len(out) == 6
    assert set(out) == {"m1", "m2"}          # never falls through to f1/f2


def test_falls_back_across_gender_only_when_the_language_has_none():
    only_male = [Voice("m1", "M"), Voice("m2", "M")]
    assert assign_voices(["F", "F"], only_male) == ["m1", "m2"]


@pytest.mark.parametrize("freq,expected", [(110, "M"), (230, "F")])
def test_gender_from_pitch(freq, expected):
    sr = 16000
    tone = np.sin(2 * np.pi * freq * np.arange(sr) / sr).astype(np.float32) * 0.5
    assert estimate_gender(tone, sr) == expected


def test_gender_is_unknown_for_silence():
    assert estimate_gender(np.zeros(16000, dtype=np.float32), 16000) is None


# --- the real artifacts ---------------------------------------------------

def ffprobe(path, entries):
    return subprocess.run(["ffprobe", "-v", "error", "-show_entries", entries,
                           "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                          capture_output=True, text=True, check=True).stdout.strip()


def audio_md5(path):
    return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a",
                           "-f", "md5", "-"], capture_output=True, text=True).stdout


def test_output_is_a_real_dubbed_video(client, video, run_queue):
    from app.providers import get_provider
    job_id = upload(client, video).json()["job_id"]
    run_queue()
    assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "completed"

    out = get_provider("storage").path(key(job_id, "out", "dubbed_es.mp4"))
    assert out.exists()
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                              "stream=codec_type", "-of", "csv=p=0", str(out)],
                             capture_output=True, text=True, check=True).stdout.split()
    assert sorted(streams) == ["audio", "video"]
    src_dur = float(ffprobe(video, "format=duration"))
    assert abs(float(ffprobe(out, "format=duration")) - src_dur) / src_dur < 0.10
    assert audio_md5(out) != audio_md5(video)       # genuinely re-dubbed


def test_subtitles_parse_and_carry_speakers(client, video, run_queue):
    job_id = upload(client, video).json()["job_id"]
    run_queue()
    srt = client.get(f"/api/v1/jobs/{job_id}/subtitles/es").text

    blocks = [b for b in srt.strip().split("\n\n") if b.strip()]
    assert len(blocks) == 4
    stamp = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> "
                       r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")
    last_start, speakers = -1.0, set()
    for i, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        assert lines[0] == str(i)                   # sequential indices
        m = stamp.match(lines[1])
        assert m, lines[1]
        start = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 1000
        end = int(m[5]) * 3600 + int(m[6]) * 60 + int(m[7]) + int(m[8]) / 1000
        assert end > start and start >= last_start  # monotonic, non-empty
        last_start = start
        if (label := re.match(r"\[(\w+)\]", lines[2])):
            speakers.add(label[1])
    assert len(speakers) >= 2


def test_every_stage_leaves_its_artifact(client, video, run_queue):
    from app.providers import get_provider
    job_id = upload(client, video).json()["job_id"]
    run_queue()
    storage = get_provider("storage")
    for path in ["stages/probe.json", "audio/source_16k.wav", "stages/transcript.json",
                 "stages/diarization.json", "stages/translation_es.json",
                 "stages/synth_es.json", "out/dubbed_es.mp4", "out/subs_es.srt",
                 "out/subs_en.srt"]:
        assert storage.exists(key(job_id, *path.split("/"))), path
