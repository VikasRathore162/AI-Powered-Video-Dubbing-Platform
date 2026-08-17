"""Local filesystem storage. The default."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import IO

from app.providers import register
from app.providers.storage import Storage


@register("storage", "local")
class LocalStorage(Storage):
    def __init__(self, root: str = "./data"):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if not p.is_relative_to(self.root):      # block traversal in keys
            raise ValueError(f"illegal storage key: {key}")
        return p

    def write_path(self, key: str) -> Path:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, key: str, src: Path) -> None:
        dst = self.path(key)
        if src.resolve() != dst:                 # already written in place
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

    def save_bytes(self, key: str, data: bytes) -> None:
        self.write_path(key).write_bytes(data)

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def open(self, key: str) -> IO[bytes]:
        return open(self.path(key), "rb")

    def delete_prefix(self, prefix: str) -> None:
        p = self.path(prefix)
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()
