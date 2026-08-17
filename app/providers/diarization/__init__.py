"""Speaker diarization providers, plus the shared pitch heuristic.

Keep this file free of provider imports: the registry imports
app.providers.diarization.<name> on demand, so nothing heavy loads unless
that provider is the configured one.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Diarization:
    labels: list[int]                 # cluster id per input span, order preserved
    genders: dict[int, str | None]    # cluster id -> "M" / "F" / None


class Diarizer(ABC):
    @abstractmethod
    def diarize(self, wav: Path, spans: list[tuple[float, float]]) -> Diarization:
        """Assign a speaker cluster to each (start, end) speech span."""
def estimate_gender(samples: np.ndarray, sr: int) -> str | None:
    """Median-F0 over voiced frames: <165 Hz -> M, else F.

    A deliberately crude autocorrelation pitch tracker. It only picks which voice
    to dub with, so a wrong guess costs a voice of the other gender rather than a
    broken job. Upgrade path if it misassigns: a classifier on the ECAPA embedding.
    """
    x = samples.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    frame, hop = int(0.04 * sr), int(0.02 * sr)
    lo, hi = int(sr / 400), int(sr / 60)      # 60-400 Hz plausible speech F0
    f0s = []
    for off in range(0, len(x) - frame, hop):
        w = x[off:off + frame]
        if float(np.sqrt(np.mean(w ** 2)) + 1e-9) < 0.01:
            continue                           # silence
        w = w - w.mean()
        ac = np.correlate(w, w, mode="full")[frame - 1:]
        seg = ac[lo:hi]
        if ac[0] <= 0 or len(seg) == 0:
            continue
        peak = int(np.argmax(seg)) + lo
        if ac[peak] / ac[0] < 0.3:             # weak periodicity -> unvoiced
            continue
        f0s.append(sr / peak)
    if len(f0s) < 5:
        return None
    return "M" if float(np.median(f0s)) < 165.0 else "F"
