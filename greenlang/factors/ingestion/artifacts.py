# -*- coding: utf-8 -*-
"""Immutable raw artifact addressing (D1): checksum + URI pointer."""

from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional


@dataclass(frozen=True)
class StoredArtifact:
    artifact_id: str
    sha256: str
    storage_uri: str
    bytes_size: int


class ArtifactStore(ABC):
    @abstractmethod
    def put_bytes(self, data: bytes, source_id: str, url: Optional[str] = None) -> StoredArtifact:
        raise NotImplementedError


class LocalArtifactStore(ArtifactStore):
    """Filesystem-backed store (S3-compatible URI scheme: file://)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, data: bytes, source_id: str, url: Optional[str] = None) -> StoredArtifact:
        h = hashlib.sha256(data).hexdigest()
        aid = str(uuid.uuid4())
        rel = Path(source_id) / h[:2] / f"{aid}.bin"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        uri = path.resolve().as_uri()
        return StoredArtifact(artifact_id=aid, sha256=h, storage_uri=uri, bytes_size=len(data))

    def put_stream(self, stream: BinaryIO, source_id: str, url: Optional[str] = None) -> StoredArtifact:
        return self.put_bytes(stream.read(), source_id, url=url)
