# -*- coding: utf-8 -*-
"""
Enterprise stub: tenant-scoped licensed factor overlays (future).

Callers can register customer-supplied bundles keyed by tenant_id; resolution
merges overlay rows over the base catalog edition inside the Factors API layer.
"""

from __future__ import annotations

from typing import Dict, List


class TenantOverlayRegistry:
    """In-memory placeholder until encrypted object-store bundles land."""

    def __init__(self) -> None:
        self._paths: Dict[str, List[str]] = {}

    def register_sqlite(self, tenant_id: str, sqlite_path: str) -> None:
        self._paths.setdefault(tenant_id, []).append(sqlite_path)

    def list_paths(self, tenant_id: str) -> List[str]:
        return list(self._paths.get(tenant_id, []))
