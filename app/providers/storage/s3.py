"""S3 / MinIO object storage."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import IO

from app.providers import ProviderConfigError, register
from app.providers.storage import Storage


@register("storage", "s3")
class S3Storage(Storage):
    """S3 or MinIO. Keeps a local cache so ffmpeg and the models can work with
    real files; credentials come from the standard AWS chain."""

    def __init__(self, bucket: str = "", endpoint_url: str | None = None,
                 cache_dir: str = "./data/.s3cache"):
        if not bucket:
            raise ProviderConfigError("s3 storage requires options.bucket")
        try:
            import boto3
        except ImportError as e:
            raise ProviderConfigError("s3 storage requires `pip install boto3`") from e
        self.bucket = bucket
        self.client = boto3.client("s3", endpoint_url=endpoint_url)
        self.cache = Path(cache_dir).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        local = self.cache / key
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(self.bucket, key, str(local))
        return local

    def write_path(self, key: str) -> Path:
        local = self.cache / key            # never downloads: the key is new
        local.parent.mkdir(parents=True, exist_ok=True)
        return local

    def save(self, key: str, src: Path) -> None:
        self.client.upload_file(str(src), self.bucket, key)
        cached = self.cache / key
        if src.resolve() == cached.resolve():
            return                          # written straight into the cache
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, cached)

    def save_bytes(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        cached = self.write_path(key)
        cached.write_bytes(data)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def open(self, key: str) -> IO[bytes]:
        return open(self.path(key), "rb")

    def delete_prefix(self, prefix: str) -> None:
        pages = self.client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=prefix)
        for page in pages:
            for obj in page.get("Contents", []):
                self.client.delete_object(Bucket=self.bucket, Key=obj["Key"])
