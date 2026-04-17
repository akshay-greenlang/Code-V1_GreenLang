# -*- coding: utf-8 -*-
"""
Connector configuration (F060).

Provides ``ConnectorConfig`` dataclass for per-connector settings
loaded from environment variables.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetryPolicy:
    """Retry configuration for connector HTTP requests."""

    max_retries: int = 3
    backoff_base_sec: float = 1.0
    backoff_max_sec: float = 60.0
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504)


@dataclass
class ConnectorConfig:
    """
    Configuration for a single factor connector.

    Loaded from environment variables with prefix ``GL_FACTORS_{SOURCE_ID}_``.
    """

    source_id: str
    api_endpoint: str = ""
    license_key: str = ""
    timeout_sec: int = 30
    batch_size: int = 1000
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, source_id: str) -> ConnectorConfig:
        """
        Load config from environment variables.

        Env var pattern: ``GL_FACTORS_{SOURCE_ID_UPPER}_{FIELD}``.
        Example for IEA: ``GL_FACTORS_IEA_API_ENDPOINT``.
        """
        prefix = f"GL_FACTORS_{source_id.upper()}_"

        cfg = cls(
            source_id=source_id,
            api_endpoint=os.environ.get(f"{prefix}API_ENDPOINT", ""),
            license_key=os.environ.get(f"{prefix}LICENSE_KEY", ""),
            timeout_sec=int(os.environ.get(f"{prefix}TIMEOUT_SEC", "30")),
            batch_size=int(os.environ.get(f"{prefix}BATCH_SIZE", "1000")),
        )

        max_retries = os.environ.get(f"{prefix}MAX_RETRIES")
        if max_retries:
            cfg.retry.max_retries = int(max_retries)

        logger.debug(
            "Connector config for %s: endpoint=%s timeout=%ds batch=%d",
            source_id, cfg.api_endpoint or "(default)", cfg.timeout_sec, cfg.batch_size,
        )
        return cfg

    @property
    def has_license(self) -> bool:
        return bool(self.license_key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "api_endpoint": self.api_endpoint,
            "has_license": self.has_license,
            "timeout_sec": self.timeout_sec,
            "batch_size": self.batch_size,
            "max_retries": self.retry.max_retries,
        }
