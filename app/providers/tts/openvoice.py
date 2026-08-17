"""OpenVoice v2 local cloning."""
from __future__ import annotations

from pathlib import Path

from app.providers import ProviderConfigError, register, require_credential
from app.providers.tts import TTS, Voice


@register("tts", "openvoice")
class OpenVoice(TTS):
    """OpenVoice v2 local cloning. `voice` is a reference wav; needs
    `pip install myshell-openvoice` and checkpoints in OPENVOICE_CKPT."""

    def __init__(self, language: str = "EN", device: str = "cpu",
                 ckpt_dir: str | None = None):
        self._ckpt = ckpt_dir or require_credential("OPENVOICE_CKPT", "openvoice")
        try:
            from openvoice.api import ToneColorConverter  # noqa: F401
        except ImportError as e:
            raise ProviderConfigError(
                "openvoice requires `pip install myshell-openvoice`") from e
        self._language, self._device = language, device

    def synthesize(self, text: str, voice: str, out: Path) -> None:
        import torch
        from melo.api import TTS as MeloTTS
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter

        out.parent.mkdir(parents=True, exist_ok=True)
        converter = ToneColorConverter(f"{self._ckpt}/config.json", device=self._device)
        converter.load_ckpt(f"{self._ckpt}/checkpoint.pth")
        base = MeloTTS(language=self._language, device=self._device)
        tmp = out.with_suffix(".base.wav")
        base.tts_to_file(text, next(iter(base.hps.data.spk2id.values())), str(tmp))
        target_se, _ = se_extractor.get_se(voice, converter, vad=True)
        source_se = torch.load(f"{self._ckpt}/base_speakers/EN/en_default_se.pth",
                               map_location=self._device)
        converter.convert(audio_src_path=str(tmp), src_se=source_se,
                          tgt_se=target_se, output_path=str(out))
        tmp.unlink(missing_ok=True)

    def voices(self, language: str) -> list[Voice]:
        return []
