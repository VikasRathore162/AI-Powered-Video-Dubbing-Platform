"""ECAPA speaker embeddings + agglomerative clustering. No HF token needed.
The default diarizer."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from app.obs import get_logger
from app.providers import register
from app.providers.diarization import Diarization, Diarizer, estimate_gender

log = get_logger("diarization")


@register("diarization", "ecapa_cluster")
class EcapaCluster(Diarizer):
    def __init__(self, distance_threshold: float = 0.55, min_speakers: int = 1,
                 max_speakers: int = 8, window_sec: float = 1.5,
                 min_span_sec: float = 0.4, min_cluster_sec: float = 2.0,
                 min_cluster_share: float = 0.02):
        from speechbrain.inference.speaker import EncoderClassifier  # pulls torch
        self._encoder = EncoderClassifier.from_hparams(
            "speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.environ.get("SPEECHBRAIN_CACHE",
                                   os.path.expanduser("~/.cache/speechbrain/ecapa")),
            run_opts={"device": "cpu"})
        self.distance_threshold = distance_threshold
        self.min_speakers, self.max_speakers = min_speakers, max_speakers
        self.window_sec, self.min_span_sec = window_sec, min_span_sec
        # a cluster under either bound is a clustering artifact, not a speaker
        self.min_cluster_sec = min_cluster_sec
        self.min_cluster_share = min_cluster_share

    def _embed(self, samples: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            emb = self._encoder.encode_batch(
                torch.from_numpy(samples).float().unsqueeze(0)).squeeze().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-9)

    def _absorb_tiny(self, labels, embeddings, spans) -> list[int]:
        """Fold clusters too small to be a real speaker into their nearest real one.

        A one-second interjection ("It's lovely to be here") gives the encoder very
        little to work with, so it often lands in a cluster of its own. On a real
        two-person interview that turned 2 speakers into 4 — and the extra speakers
        then consumed the gender-matched voices, so the guest got a male voice.
        Embeddings are L2-normalised, so a dot product is cosine similarity.
        """
        speech: dict[int, float] = {}
        for (start, end), lab in zip(spans, labels):
            speech[lab] = speech.get(lab, 0.0) + (end - start)
        total = sum(speech.values()) or 1.0
        # Two ways to be too small: an absolute floor for short clips, and a share
        # of the conversation. The share is what catches a single stray line in a
        # long recording, without hard-coding a duration that only suits one video.
        real = [c for c, secs in speech.items()
                if secs >= self.min_cluster_sec and secs / total >= self.min_cluster_share]
        tiny = [c for c in speech if c not in real]
        if not tiny or not real:        # everything is short: keep what clustering found
            return list(labels)

        X = np.stack(embeddings)
        centroids = {}
        for c in real:
            mean = X[[i for i, lab in enumerate(labels) if lab == c]].mean(axis=0)
            centroids[c] = mean / (np.linalg.norm(mean) + 1e-9)
        merged = [max(real, key=lambda c: float(X[i] @ centroids[c])) if lab in tiny else lab
                  for i, lab in enumerate(labels)]
        log.info("absorbed_tiny_clusters", absorbed=len(tiny),
                 before=len(speech), after=len(set(merged)))
        return merged

    def diarize(self, wav: Path, spans: list[tuple[float, float]]) -> Diarization:
        import soundfile as sf
        from sklearn.cluster import AgglomerativeClustering

        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        embeddings = []
        for start, end in spans:
            span = audio[int(start * sr):min(int(end * sr), len(audio))]
            if len(span) < int(self.min_span_sec * sr):   # too short: pad by repeat
                reps = max(1, int(np.ceil(self.min_span_sec * sr / max(len(span), 1))))
                span = np.tile(span, reps)[:int(self.min_span_sec * sr)]
            win = int(self.window_sec * sr)
            if len(span) > 2 * win:                       # long: average windows
                embs = [self._embed(span[o:o + win]) for o in range(0, len(span) - win, win)]
                emb = np.mean(embs, axis=0)
                emb = emb / (np.linalg.norm(emb) + 1e-9)
            else:
                emb = self._embed(span)
            embeddings.append(emb)

        if not embeddings:
            return Diarization(labels=[], genders={})
        if len(embeddings) == 1:
            labels = [0]
        else:
            X = np.stack(embeddings)
            labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=self.distance_threshold,
                metric="cosine", linkage="average").fit(X).labels_
            k = len(set(labels))
            if not self.min_speakers <= k <= self.max_speakers:
                labels = AgglomerativeClustering(
                    n_clusters=min(max(k, self.min_speakers), self.max_speakers),
                    metric="cosine", linkage="average").fit(X).labels_

        labels = self._absorb_tiny(labels, embeddings, spans)

        # relabel by first appearance so SPEAKER_00 speaks first
        order: dict[int, int] = {}
        labels = [order.setdefault(raw, len(order)) for raw in labels]

        genders = {}
        for cluster in sorted(set(labels)):
            parts, total = [], 0
            for (start, end), lab in zip(spans, labels):
                if lab != cluster:
                    continue
                s, e = int(start * sr), min(int(end * sr), len(audio))
                parts.append(audio[s:e])
                total += e - s
                if total > 20 * sr:            # 20s is plenty to judge pitch
                    break
            genders[cluster] = estimate_gender(np.concatenate(parts), sr) if parts else None

        log.info("diarized", spans=len(spans), speakers=len(set(labels)), genders=genders)
        return Diarization(labels=labels, genders=genders)
