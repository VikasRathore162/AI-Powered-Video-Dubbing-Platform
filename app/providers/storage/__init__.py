"""Artifact storage providers.

Producers call write_path(key) -> write the file -> save(key, path). On local
disk write_path is the final location and save is a no-op; on S3 it is a cache
path and save uploads. Readers call path(key), which downloads if needed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO


class Storage(ABC):
    @abstractmethod
    def path(self, key: str) -> Path:
        """Local path of an EXISTING key (downloaded if remote)."""

    @abstractmethod
    def write_path(self, key: str) -> Path:
        """Local path to write a NOT-YET-EXISTING key to; publish with save()."""

    @abstractmethod
    def save(self, key: str, src: Path) -> None: ...

    @abstractmethod
    def save_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def open(self, key: str) -> IO[bytes]: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None: ...

