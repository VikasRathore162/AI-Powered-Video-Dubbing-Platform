"""Argos Translate: local, no key. Routes X->Y through English when both
halves are installed. The default translator."""
from __future__ import annotations

from app.obs import get_logger
from app.providers import ProviderError, ProviderTransientError, register
from app.providers.translation import Translator

log = get_logger("translation")


@register("translation", "argos")
class Argos(Translator):
    def __init__(self, auto_install: bool = True):
        self.auto_install = auto_install

    def _lookup(self, src: str, tgt: str):
        import argostranslate.translate as at
        langs = {l.code: l for l in at.get_installed_languages()}
        if src in langs and tgt in langs:
            return langs[src].get_translation(langs[tgt])
        return None

    def translate(self, texts: list[str], src: str, tgt: str) -> list[str]:
        if src == tgt:
            return list(texts)
        tr = self._lookup(src, tgt)
        if tr is None and self.auto_install:
            self._install(src, tgt)
            tr = self._lookup(src, tgt)
        if tr is None:
            raise ProviderError(
                f"no argos translation path {src}->{tgt}; run scripts/setup.py models")
        return [tr.translate(t) if t.strip() else t for t in texts]

    def _install(self, src: str, tgt: str) -> None:
        """Install the pair, or both halves of an English pivot."""
        import argostranslate.package as pkg
        try:
            pkg.update_package_index()
            available = pkg.get_available_packages()
        except Exception as e:
            raise ProviderTransientError(f"argos package index unreachable: {e}") from e
        direct = any((p.from_code, p.to_code) == (src, tgt) for p in available)
        wanted = {(src, tgt)} if direct else {(src, "en"), ("en", tgt)}
        for p in available:
            if (p.from_code, p.to_code) in wanted:
                log.info("argos_installing", pair=f"{p.from_code}->{p.to_code}")
                pkg.install_from_path(p.download())
