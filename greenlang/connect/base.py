# -*- coding: utf-8 -*-
"""GreenLang Connect base protocol + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Type


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConnectorError(Exception):
    """Base class for Connect errors."""


class ConnectorAuthError(ConnectorError):
    """Missing or invalid credentials."""


class ConnectorDependencyError(ConnectorError):
    """A required Python client library is not installed."""


class ConnectorExtractionError(ConnectorError):
    """Extraction failed at the source-system boundary."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    tenant_id: str
    connector_id: str
    credentials: dict[str, str]
    filters: dict[str, Any] = field(default_factory=dict)
    # When True, the connector must NOT call out to any external system.
    # Callers use dry_run to validate credentials + filter shape end-to-end
    # without traffic, which is critical in CI and for design-partner demos.
    dry_run: bool = False


@dataclass
class HealthCheckResult:
    """Structured health-check verdict."""

    ok: bool
    connector_id: str
    reason: str = ""
    missing_credentials: list[str] = field(default_factory=list)
    dependency_available: bool = True


@dataclass
class ConnectorResult:
    connector_id: str
    records: list[dict[str, Any]]
    row_count: int
    checksum: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base connector
# ---------------------------------------------------------------------------


class BaseConnector(ABC):
    """Shared connector interface.

    Implementations MUST:

    - Declare ``connector_id``, ``required_credentials`` (tuple of keys),
      and ``required_python_package`` (name of the pip-installable dep,
      or ``None`` if stdlib-only).
    - Implement ``_extract_records()`` — returning a list of canonical
      dicts.  The base ``extract()`` wraps this with credential
      validation, dependency checks, dry-run short-circuit, and checksum
      calculation.
    - Implement ``_check_dependency()`` if the connector needs a special
      import path.  Default behaviour checks ``required_python_package``
      with ``importlib``.
    """

    connector_id: str = "base"
    required_credentials: tuple[str, ...] = ()
    required_python_package: Optional[str] = None

    # ------------------------------------------------------------------
    # Public (framework) entry points
    # ------------------------------------------------------------------

    async def extract(self, spec: SourceSpec) -> ConnectorResult:
        """Validate, optionally dry-run, and extract canonical records."""
        self._validate_credentials(spec.credentials)
        if not self._check_dependency():
            if not spec.dry_run:
                raise ConnectorDependencyError(
                    f"{self.connector_id}: required dependency "
                    f"'{self.required_python_package}' not installed; "
                    "run in --dry-run or install the dependency."
                )

        if spec.dry_run:
            records: list[dict[str, Any]] = []
            metadata = {
                "source_system": self.connector_id,
                "dry_run": True,
            }
        else:
            records = list(await self._extract_records(spec))
            metadata = {"source_system": self.connector_id}
        return ConnectorResult(
            connector_id=self.connector_id,
            records=records,
            row_count=len(records),
            checksum=_records_checksum(records),
            metadata=metadata,
        )

    async def healthcheck(
        self, credentials: dict[str, str]
    ) -> HealthCheckResult:
        """Return structured verdict describing credential + dep state."""
        missing = [k for k in self.required_credentials if not credentials.get(k)]
        dependency_available = self._check_dependency()
        ok = not missing and dependency_available
        reason_parts: list[str] = []
        if missing:
            reason_parts.append(f"missing credentials: {missing}")
        if not dependency_available:
            reason_parts.append(
                f"python package '{self.required_python_package}' not installed"
            )
        return HealthCheckResult(
            ok=ok,
            connector_id=self.connector_id,
            reason="; ".join(reason_parts) or "ok",
            missing_credentials=missing,
            dependency_available=dependency_available,
        )

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    async def _extract_records(self, spec: SourceSpec) -> Iterable[dict[str, Any]]:
        """Return canonical records from the source system."""

    def _check_dependency(self) -> bool:
        """Return True if the optional client library is importable."""
        if not self.required_python_package:
            return True
        try:
            import importlib

            importlib.import_module(self.required_python_package)
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_credentials(self, credentials: dict[str, str]) -> None:
        missing = [k for k in self.required_credentials if not credentials.get(k)]
        if missing:
            raise ConnectorAuthError(
                f"{self.connector_id}: missing credentials {missing}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _records_checksum(records: list[dict[str, Any]]) -> str:
    import hashlib
    import json as _json

    return hashlib.sha256(
        _json.dumps(records, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ConnectorRegistry:
    def __init__(self) -> None:
        self._connectors: dict[str, Type[BaseConnector]] = {}

    def register(self, connector_id: str, cls: Type[BaseConnector]) -> None:
        self._connectors[connector_id] = cls

    def get(self, connector_id: str) -> BaseConnector:
        cls = self._connectors.get(connector_id)
        if cls is None:
            available = ", ".join(self.available()) or "none"
            raise ValueError(
                f"Unknown connector: {connector_id} (available: {available})"
            )
        return cls()

    def available(self) -> list[str]:
        return sorted(self._connectors.keys())

    def describe(self) -> list[dict[str, Any]]:
        """Return metadata for every registered connector."""
        rows: list[dict[str, Any]] = []
        for cid in self.available():
            cls = self._connectors[cid]
            rows.append(
                {
                    "connector_id": cid,
                    "required_credentials": list(cls.required_credentials),
                    "required_python_package": cls.required_python_package,
                }
            )
        return rows


_default: Optional[ConnectorRegistry] = None


def default_registry() -> ConnectorRegistry:
    global _default
    if _default is None:
        _default = ConnectorRegistry()
    return _default
