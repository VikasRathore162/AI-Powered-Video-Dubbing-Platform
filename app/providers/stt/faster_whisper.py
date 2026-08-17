"""faster-whisper: local CPU transcription with word timestamps. The default."""
from __future__ import annotations

from pathlib import Path

from app.obs import get_logger
from app.providers import ProviderError, register
from app.providers.stt import STT, Segment, Transcript, Word

log = get_logger("stt")


@register("stt", "faster_whisper")
class FasterWhisper(STT):
    def __init__(self, model: str = "small", compute_type: str = "int8",
                 device: str = "cpu", beam_size: int = 5):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._beam_size = beam_size
        self._name = model

    def transcribe(self, wav: Path, language: str | None = None) -> Transcript:
        segments_iter, info = self._model.transcribe(
            str(wav), language=language, word_timestamps=True,
            vad_filter=True, beam_size=self._beam_size)
        segments = []
        for seg in segments_iter:
            text = seg.text.strip()
            if text:
                segments.append(Segment(seg.start, seg.end, text,
                                        [Word(w.start, w.end, w.word.strip())
                                         for w in (seg.words or [])]))
        if not segments:
            raise ProviderError("no speech detected in audio")
        log.info("transcribed", model=self._name, language=info.language,
                 segments=len(segments))
        return Transcript(language=info.language, segments=segments)
