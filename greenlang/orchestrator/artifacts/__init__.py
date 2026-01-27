# -*- coding: utf-8 -*-
"""
Orchestrator Artifact Storage
==============================

Provides artifact storage for GLIP v1 protocol.

Available Stores:
    - S3ArtifactStore: AWS S3 / S3-compatible (primary)
    - LocalArtifactStore: Local filesystem (development)

Author: GreenLang Team
"""

from greenlang.orchestrator.artifacts.base import (
    ArtifactStore,
    ArtifactMetadata,
    ArtifactManifest,
)

# S3 store is optional (requires aioboto3)
try:
    from greenlang.orchestrator.artifacts.s3_store import S3ArtifactStore, S3StoreConfig
    S3_AVAILABLE = True
except ImportError:
    S3ArtifactStore = None
    S3StoreConfig = None
    S3_AVAILABLE = False

__all__ = [
    "ArtifactStore",
    "ArtifactMetadata",
    "ArtifactManifest",
    "S3ArtifactStore",
    "S3StoreConfig",
    "S3_AVAILABLE",
]
