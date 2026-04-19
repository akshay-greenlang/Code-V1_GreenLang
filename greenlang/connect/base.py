# -*- coding: utf-8 -*-
"""GreenLang Connect base protocol + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Type


@dataclass(frozen=True)
class SourceSpec:
    tenant_id: str
    connector_id: str
    credentials: dict[str, str]
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConnectorResult:
    connector_id: str
    records: list[dict[str, Any]]
    row_count: int
    checksum: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseConnector(ABC):
    """Shared connector interface.

    Implementations should ONLY transform source-system data into
    GreenLang-canonical activity records. Factor resolution + emissions
    compute happen downstream in the Scope Engine.
    """

    connector_id: str

    @abstractmethod
    async def extract(self, spec: SourceSpec) -> ConnectorResult:
        """Pull data from source system and return canonicalized records."""

    @abstractmethod
    async def healthcheck(self, credentials: dict[str, str]) -> bool:
        """Return True if credentials can authenticate against the source."""


class ConnectorRegistry:
    def __init__(self) -> None:
        self._connectors: dict[str, Type[BaseConnector]] = {}

    def register(self, connector_id: str, cls: Type[BaseConnector]) -> None:
        self._connectors[connector_id] = cls

    def get(self, connector_id: str) -> BaseConnector:
        cls = self._connectors.get(connector_id)
        if cls is None:
            raise ValueError(f"Unknown connector: {connector_id}")
        return cls()

    def available(self) -> list[str]:
        return sorted(self._connectors.keys())


_default: Optional[ConnectorRegistry] = None


def default_registry() -> ConnectorRegistry:
    global _default
    if _default is None:
        _default = ConnectorRegistry()
    return _default
