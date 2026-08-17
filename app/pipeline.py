"""The dubbing pipeline, as plain functions.

Each stage reads its inputs from the database and storage, writes its outputs
the same way, and returns the artifact keys it produced. No Celery in here, so
every stage is directly callable and testable; app/worker.py is the thin shell
that schedules them and handles retry/cancel.

    probe -> extract_audio -> transcribe -> diarize -> per language:
        translate -> synthesize -> assemble
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from app import media
from app.config import get_settings
from app.models import Segment, Speaker, Translation
from app.obs import get_logger
from app.providers import get_provider
from app.providers.tts import assign_voices

log = get_logger("pipeline")

DUB_RATE = 24000        # sample rate of the assembled dub track


class InvalidVideo(Exception):
    """Corrupt or unusable upload — permanent, never retried."""


def key(job_id: str, *parts: str) -> str:
    return "/".join(["jobs", job_id, *parts])


def _write_json(storage, k: str, obj) -> None:
    storage.save_bytes(k, json.dumps(obj, ensure_ascii=False, indent=1).encode())


def _read_json(storage, k: str):
    with storage.open(k) as f:
        return json.load(f)


def probe(session, job, lang=None, check_cancel=None) -> dict:
    """Validate the upload for real (decodable, has audio and video, in limits)."""
    storage = get_provider("storage")
    try:
        info = media.probe(storage.path(job.source_key))
    except media.MediaError as e:
        raise InvalidVideo(f"unreadable or corrupt video: {e}") from e
    if not info["has_video"]:
        raise InvalidVideo("file has no video stream")
    if not info["has_audio"]:
        raise InvalidVideo("file has no audio stream — nothing to dub")
    limit = get_settings().limits.max_duration_sec
    if info["duration_sec"] <= 0:
        raise InvalidVideo("could not determine video duration")
    if info["duration_sec"] > limit:
        raise InvalidVideo(f"duration {info['duration_sec']:.0f}s exceeds "
                           f"{limit:.0f}s limit")
    job.duration_sec, job.probe = info["duration_sec"], info
    k = key(job.id, "stages", "probe.json")
    _write_json(storage, k, info)
    return {"probe": k}


def extract_audio(session, job, lang=None, check_cancel=None) -> dict:
    storage = get_provider("storage")
    k = key(job.id, "audio", "source_16k.wav")
    out = storage.write_path(k)
    media.extract_wav(storage.path(job.source_key), out)
    storage.save(k, out)
    return {"wav": k}


def transcribe(session, job, lang=None, check_cancel=None) -> dict:
    storage = get_provider("storage")
    wav = storage.path(key(job.id, "audio", "source_16k.wav"))
    result = get_provider("stt").transcribe(wav, language=job.source_language or None)
    job.source_language = result.language

    # idempotent re-run: clear children first so Postgres FKs stay happy
    session.query(Translation).filter_by(job_id=job.id).delete()
    session.query(Segment).filter_by(job_id=job.id).delete()
    for i, seg in enumerate(result.segments):
        session.add(Segment(
            job_id=job.id, idx=i, start_sec=seg.start, end_sec=seg.end, text=seg.text,
            words=[{"start": w.start, "end": w.end, "text": w.text}
                   for w in seg.words] or None))

    k = key(job.id, "stages", "transcript.json")
    _write_json(storage, k, {"language": result.language,
                             "segments": [{"idx": i, "start": s.start, "end": s.end,
                                           "text": s.text}
                                          for i, s in enumerate(result.segments)]})
    return {"transcript": k, "language": result.language,
            "segments": len(result.segments)}


def diarize(session, job, lang=None, check_cancel=None) -> dict:
    """Cluster speakers, then cast a voice per speaker per target language.

    Voice casting happens here, before the per-language fan-out, so there is a
    single writer — parallel language branches used to race on speaker.voices.
    """
    storage = get_provider("storage")
    segments = session.query(Segment).filter_by(job_id=job.id).order_by(Segment.idx).all()
    result = get_provider("diarization").diarize(
        storage.path(key(job.id, "audio", "source_16k.wav")),
        [(s.start_sec, s.end_sec) for s in segments])

    session.query(Segment).filter_by(job_id=job.id).update({"speaker_id": None})
    session.query(Speaker).filter_by(job_id=job.id).delete()
    session.flush()
    speakers = {c: Speaker(job_id=job.id, label=f"SPEAKER_{c:02d}",
                           gender=result.genders.get(c))
                for c in sorted(set(result.labels))}
    session.add_all(speakers.values())
    session.flush()

    for seg, cluster in zip(segments, result.labels):
        speaker = speakers[cluster]
        seg.speaker_id = speaker.id
        speaker.total_speech_sec += seg.end_sec - seg.start_sec
        speaker.segment_count += 1

    ordered = [speakers[c] for c in sorted(speakers)]
    tts = get_provider("tts")
    for target in job.target_languages or []:
        if all(target in (s.voices or {}) for s in ordered):
            continue                                    # retry: keep the casting
        catalog = tts.voices(target)
        for speaker, voice in zip(ordered, assign_voices(
                [s.gender for s in ordered], catalog)):
            speaker.voices = {**(speaker.voices or {}), target: voice}

    k = key(job.id, "stages", "diarization.json")
    _write_json(storage, k, {"speakers": [{"label": s.label, "gender": s.gender,
                                           "speech_sec": round(s.total_speech_sec, 2),
                                           "segments": s.segment_count,
                                           "voices": s.voices}
                                          for s in ordered],
                             "labels": result.labels})
    return {"diarization": k, "speakers": len(speakers)}


def translate(session, job, lang=None, check_cancel=None) -> dict:
    storage = get_provider("storage")
    segments = session.query(Segment).filter_by(job_id=job.id).order_by(Segment.idx).all()
    texts = [s.text for s in segments]
    translated = get_provider("translation").translate(
        texts, src=job.source_language, tgt=lang)
    if len(translated) != len(texts):
        raise RuntimeError("translation count mismatch")

    session.query(Translation).filter_by(job_id=job.id, language=lang).delete()
    for seg, text in zip(segments, translated):
        session.add(Translation(segment_id=seg.id, job_id=job.id,
                                language=lang, text=text))

    k = key(job.id, "stages", f"translation_{lang}.json")
    _write_json(storage, k, [{"idx": s.idx, "text": t}
                             for s, t in zip(segments, translated)])
    return {"translation": k}


def synthesize(session, job, lang=None, check_cancel=None) -> dict:
    """One TTS clip per segment, in the speaker's assigned voice."""
    settings = get_settings()
    storage, tts = get_provider("storage"), get_provider("tts")
    segments = session.query(Segment).filter_by(job_id=job.id).order_by(Segment.idx).all()
    speakers = session.query(Speaker).filter_by(job_id=job.id).all()
    texts = {t.segment_id: t.text for t in session.query(Translation)
             .filter_by(job_id=job.id, language=lang)}

    voices = {s.id: (s.voices or {}).get(lang) for s in speakers}
    if not all(voices.values()):        # provider changed since diarize cast them
        fresh = assign_voices([s.gender for s in speakers], tts.voices(lang))
        voices = {s.id: v for s, v in zip(speakers, fresh)}
    default_voice = next(iter(voices.values()), None)

    work = [(s.idx, texts[s.id].strip(), voices.get(s.speaker_id, default_voice))
            for s in segments if texts.get(s.id, "").strip()]

    def synth(item) -> dict:
        idx, text, voice = item
        raw_key = key(job.id, "tts", lang, f"raw_{idx:04d}.mp3")
        raw = storage.write_path(raw_key)
        tts.synthesize(text, voice, raw)
        storage.save(raw_key, raw)
        # fit against speech, not the silence the voice ships with
        clip_key = key(job.id, "tts", lang, f"clip_{idx:04d}.wav")
        clip = storage.write_path(clip_key)
        media.trim_silence(raw, clip, rate=DUB_RATE)
        storage.save(clip_key, clip)
        return {"idx": idx, "clip": clip_key, "voice": voice,
                "duration": media.duration_of(clip)}

    clips = []
    # TTS is network/subprocess bound, so overlap the calls
    with ThreadPoolExecutor(max_workers=max(1, settings.processing.tts_concurrency)) as pool:
        futures = [pool.submit(synth, item) for item in work]
        try:
            for i, fut in enumerate(as_completed(futures)):
                clips.append(fut.result())
                if check_cancel and i % settings.processing.cancel_poll_every == 0:
                    check_cancel()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)   # don't finish a dead job
            raise

    if not clips:
        raise RuntimeError(f"no synthesizable segments for language {lang}")
    clips.sort(key=lambda c: c["idx"])
    k = key(job.id, "stages", f"synth_{lang}.json")
    _write_json(storage, k, clips)
    return {"synth": k, "clips": len(clips)}


def fit_placements(segments: list[dict], durations: dict[int, float],
                   video_duration: float, max_tempo: float,
                   gap_guard: float) -> list[dict]:
    """Decide where each dubbed clip goes and how much to speed it up.

    A translation rarely takes as long to say as the original. In order:
      1. shorter than its slot -> place at the original start (silence is fine,
         slowed-down speech is not)
      2. longer -> spill into the silence before the next segment
      3. still longer -> speed up, but never past max_tempo
      4. still longer -> accept a brief overlap and record it; cutting words off
         is worse than a few hundred ms of overlap
    """
    placements = []
    for i, seg in enumerate(segments):
        duration = durations.get(seg["idx"])
        if duration is None:
            continue
        slot = max(seg["end"] - seg["start"], 0.05)
        next_start = segments[i + 1]["start"] if i + 1 < len(segments) else video_duration
        gap = max(0.0, next_start - seg["end"] - gap_guard)
        usable = slot + min(gap, max(0.0, duration - slot))
        tempo = min(max(duration / usable if usable > 0 else max_tempo, 1.0), max_tempo)
        fitted = duration / tempo
        placements.append({"idx": seg["idx"], "start": seg["start"],
                           "tempo": round(tempo, 4),
                           "fitted_duration": round(fitted, 3),
                           "overflow_sec": round(max(0.0, fitted - usable), 3)})
    return placements


def assemble(session, job, lang=None, check_cancel=None) -> dict:
    """Time-fit the clips onto one timeline, mux over the video, write SRTs."""
    import soundfile as sf

    settings = get_settings()
    storage = get_provider("storage")
    segments = session.query(Segment).filter_by(job_id=job.id).order_by(Segment.idx).all()
    clips = _read_json(storage, key(job.id, "stages", f"synth_{lang}.json"))
    clip_keys = {c["idx"]: c["clip"] for c in clips}

    video_duration = job.duration_sec or max((s.end_sec for s in segments), default=0) + 1
    placements = fit_placements(
        [{"idx": s.idx, "start": s.start_sec, "end": s.end_sec} for s in segments],
        {c["idx"]: c["duration"] for c in clips}, video_duration,
        settings.processing.max_tempo, settings.processing.gap_guard_sec)

    overflowed = [p["idx"] for p in placements if p["overflow_sec"] > 0.05]
    if overflowed:
        log.warning("assembly_overflow", job_id=job.id, lang=lang, segments=overflowed)

    timeline = np.zeros(int(video_duration * DUB_RATE) + DUB_RATE, dtype=np.int32)
    for p in placements:
        fitted = storage.write_path(key(job.id, "tts", lang, f"fit_{p['idx']:04d}.wav"))
        media.fit_clip(storage.path(clip_keys[p["idx"]]), fitted,
                       tempo=p["tempo"], rate=DUB_RATE)
        samples, _ = sf.read(fitted, dtype="int16")
        if samples.ndim > 1:
            samples = samples.mean(axis=1).astype(np.int16)
        start = int(p["start"] * DUB_RATE)
        end = start + len(samples)
        if end > len(timeline):
            timeline = np.concatenate([timeline, np.zeros(end - len(timeline), np.int32)])
        timeline[start:end] += samples.astype(np.int32)
        if check_cancel:
            check_cancel()

    dub_key = key(job.id, "tts", lang, "dub_track.wav")
    dub_path = storage.write_path(dub_key)
    sf.write(dub_path, np.clip(timeline, -32768, 32767).astype(np.int16), DUB_RATE)
    storage.save(dub_key, dub_path)

    video_key = key(job.id, "out", f"dubbed_{lang}.mp4")
    video_path = storage.write_path(video_key)
    media.mix_and_mux(storage.path(job.source_key), dub_path, video_path,
                      duck_volume=settings.processing.duck_volume,
                      has_source_audio=bool((job.probe or {}).get("has_audio", True)))
    storage.save(video_key, video_path)

    # translated subtitles follow the FITTED timing so they match the dub;
    # the source-language file keeps the original timing
    labels = {s.id: s.label for s in session.query(Speaker).filter_by(job_id=job.id)}
    texts = {t.segment_id: t.text for t in session.query(Translation)
             .filter_by(job_id=job.id, language=lang)}
    by_idx = {p["idx"]: p for p in placements}
    srt_key = key(job.id, "out", f"subs_{lang}.srt")
    storage.save_bytes(srt_key, media.to_srt(
        [{"start": by_idx[s.idx]["start"],
          "end": by_idx[s.idx]["start"] + by_idx[s.idx]["fitted_duration"],
          "text": texts[s.id], "speaker": labels.get(s.speaker_id)}
         for s in segments if s.idx in by_idx and s.id in texts]).encode())

    source_srt = key(job.id, "out", f"subs_{job.source_language}.srt")
    if not storage.exists(source_srt):
        storage.save_bytes(source_srt, media.to_srt(
            [{"start": s.start_sec, "end": s.end_sec, "text": s.text,
              "speaker": labels.get(s.speaker_id)} for s in segments]).encode())

    return {"video": video_key, "srt": srt_key, "source_srt": source_srt,
            "overflow_segments": len(overflowed)}


# The seven stages, and how they are scheduled: the shared ones run once per
# job, the language ones once per target language (see worker.fan_out).
STAGES = {"probe": probe, "extract_audio": extract_audio, "transcribe": transcribe,
          "diarize": diarize, "translate": translate, "synthesize": synthesize,
          "assemble": assemble}
SHARED_STAGES = ["probe", "extract_audio", "transcribe", "diarize"]
LANGUAGE_STAGES = ["translate", "synthesize", "assemble"]
# rough share of total work, used for the progress percentage
STAGE_WEIGHTS = {"probe": 2, "extract_audio": 3, "transcribe": 30, "diarize": 15,
                 "translate": 10, "synthesize": 25, "assemble": 15}

assert set(STAGES) == set(SHARED_STAGES) | set(LANGUAGE_STAGES) == set(STAGE_WEIGHTS)
