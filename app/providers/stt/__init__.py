"""Speech-to-text providers. One file per provider — the file name is the name
you put in `providers.stt.name`."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    segments: list[Segment]


class STT(ABC):
    @abstractmethod
    def transcribe(self, wav: Path, language: str | None = None) -> Transcript: ...

