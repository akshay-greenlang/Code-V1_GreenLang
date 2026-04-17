# -*- coding: utf-8 -*-
"""
Base connector abstraction for licensed emission factor sources (F060).

All factor connectors (IEA, ecoinvent, Electricity Maps, etc.) extend
``BaseConnector`` and implement ``fetch_metadata`` + ``fetch_factors``.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConnectorStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class ConnectorCapabilities:
    """Declarative capabilities of a connector."""

    supports_metadata_only: bool = True
    supports_batch_fetch: bool = True
    supports_incremental: bool = False
    supports_real_time: bool = False
    max_batch_size: int = 1000
    requires_license: bool = True
    license_class: str = "commercial_connector"
    typical_factor_count: int = 0


@dataclass
class ConnectorHealthResult:
    """Result of a health check."""

    status: ConnectorStatus
    latency_ms: int = 0
    message: str = ""
    checked_at: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class BaseConnector(ABC):
    """
    Abstract base class for all factor source connectors.

    Subclasses must implement:
    - ``source_id`` property
    - ``capabilities`` property
    - ``fetch_metadata()``
    - ``fetch_factors()``
    - ``health_check()``
    """

    def __init__(self, *, license_key: Optional[str] = None) -> None:
        self._license_key = license_key
        self._last_health: Optional[ConnectorHealthResult] = None
        self._request_count = 0
        self._error_count = 0

    @property
    @abstractmethod
    def source_id(self) -> str:
        """Unique source identifier (matches source_registry.yaml)."""

    @property
    @abstractmethod
    def capabilities(self) -> ConnectorCapabilities:
        """Declared capabilities of this connector."""

    @abstractmethod
    def fetch_metadata(self) -> List[Dict[str, Any]]:
        """
        Fetch factor metadata (IDs, descriptions, geographies) without values.

        Returns a list of dicts, each with at least ``factor_id``.
        Does NOT require a license key for metadata-only queries.
        """

    @abstractmethod
    def fetch_factors(
        self,
        factor_ids: List[str],
        *,
        license_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch full factor records including values.

        Args:
            factor_ids: Factor IDs to retrieve.
            license_key: Override the instance license key.

        Returns a list of dicts conforming to EmissionFactorRecord fields.
        Requires a valid license key.
        """

    @abstractmethod
    def health_check(self) -> ConnectorHealthResult:
        """
        Check API connectivity and credential validity.

        Returns a ConnectorHealthResult with status and latency.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def get_license_key(self, override: Optional[str] = None) -> str:
        """Resolve the license key (override > instance > error)."""
        key = override or self._license_key
        if not key:
            from greenlang.exceptions.connector import ConnectorAuthError
            raise ConnectorAuthError(
                f"No license key for connector {self.source_id}",
                connector_name=self.source_id,
            )
        return key

    def _track_request(self, success: bool = True) -> None:
        self._request_count += 1
        if not success:
            self._error_count += 1

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "last_health": self._last_health.status.value if self._last_health else "unknown",
        }

    def fetch_factors_batched(
        self,
        factor_ids: List[str],
        *,
        license_key: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch factors in batches, respecting the connector's max_batch_size.

        Default implementation chunks ``factor_ids`` and calls ``fetch_factors``
        for each batch. Subclasses may override for native batch APIs.
        """
        bs = batch_size or self.capabilities.max_batch_size
        results: List[Dict[str, Any]] = []
        for i in range(0, len(factor_ids), bs):
            batch = factor_ids[i: i + bs]
            logger.debug(
                "Fetching batch %d-%d of %d for %s",
                i, min(i + bs, len(factor_ids)), len(factor_ids), self.source_id,
            )
            batch_results = self.fetch_factors(batch, license_key=license_key)
            results.extend(batch_results)
        return results

    def __repr__(self) -> str:
        return f"<{type(self).__name__} source_id={self.source_id!r}>"
