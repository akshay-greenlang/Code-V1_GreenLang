# -*- coding: utf-8 -*-
"""
GreenLang Connectors - Data Source Integration Framework
=========================================================

Connectors provide deterministic, replay-capable access to external data sources
including APIs, databases, and time-series data providers.

Key Features:
- Async-first architecture for scalability
- Deterministic replay for auditing and compliance
- Snapshot-based caching with byte-exact reproducibility
- Policy-integrated security (egress control, allowlists)
- Provenance tracking for data lineage

Architecture:
- Base connector interface (enhanced from SDK)
- Context-aware execution (record/replay/golden modes)
- Snapshot manager (reuses existing determinism infrastructure)
- Region code standards (ISO 3166-2 compliant)

Example Usage:
    from greenlang.connectors.grid.mock import GridIntensityMockConnector
    from greenlang.connectors.context import ConnectorContext, CacheMode

    # Record mode: fetch live and cache
    connector = GridIntensityMockConnector()
    ctx = ConnectorContext(mode=CacheMode.RECORD, connector_id="grid/mock")
    payload, prov = await connector.fetch(query, ctx)

    # Replay mode: use cached data only
    ctx = ConnectorContext(mode=CacheMode.REPLAY)
    payload, prov = await connector.fetch(query, ctx)
"""

from greenlang.connectors.base import (
    Connector,
    ConnectorCapabilities,
    ConnectorProvenance,
)
from greenlang.connectors.context import ConnectorContext, CacheMode
from greenlang.connectors.errors import (
    ConnectorError,
    ConnectorAuthError,
    ConnectorConfigError,
    ConnectorNetworkError,
    ConnectorReplayRequired,
    ConnectorSnapshotNotFound,
)

__all__ = [
    "Connector",
    "ConnectorCapabilities",
    "ConnectorProvenance",
    "ConnectorContext",
    "CacheMode",
    "ConnectorError",
    "ConnectorAuthError",
    "ConnectorConfigError",
    "ConnectorNetworkError",
    "ConnectorReplayRequired",
    "ConnectorSnapshotNotFound",
]
