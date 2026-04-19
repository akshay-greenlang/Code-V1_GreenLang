# -*- coding: utf-8 -*-
"""
GreenLang Storage Backends
==========================

This package provides storage backends for artifact management.

Available backends:
    - S3StorageClient: AWS S3 and S3-compatible storage
    - (Azure and GCS support coming soon)
"""

from greenlang.execution.core.storage.s3_client import (
    S3StorageClient,
    S3StorageConfig,
    get_s3_client,
    set_s3_client,
)

__all__ = [
    "S3StorageClient",
    "S3StorageConfig",
    "get_s3_client",
    "set_s3_client",
]
