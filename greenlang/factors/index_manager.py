# -*- coding: utf-8 -*-
"""
Index manager for the Factors catalog (F081).

Manages database index creation, monitoring, and recommendations for
the factors tables (PostgreSQL + SQLite dual support).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class IndexType(str, Enum):
    BTREE = "btree"
    GIN = "gin"
    GIST = "gist"
    HASH = "hash"
    BRIN = "brin"


class IndexStatus(str, Enum):
    ACTIVE = "active"
    PENDING = "pending"
    BUILDING = "building"
    INVALID = "invalid"
    DROPPED = "dropped"


@dataclass
class IndexDefinition:
    """Definition of a database index."""

    name: str
    table: str
    columns: List[str]
    index_type: IndexType = IndexType.BTREE
    unique: bool = False
    partial_filter: str = ""
    status: IndexStatus = IndexStatus.PENDING
    size_bytes: int = 0
    scan_count: int = 0

    def to_sql(self, dialect: str = "postgresql") -> str:
        """Generate CREATE INDEX SQL."""
        unique = "UNIQUE " if self.unique else ""
        cols = ", ".join(self.columns)
        if dialect == "postgresql" and self.index_type != IndexType.BTREE:
            using = f" USING {self.index_type.value}"
        else:
            using = ""
        where = f" WHERE {self.partial_filter}" if self.partial_filter else ""
        return f"CREATE {unique}INDEX IF NOT EXISTS {self.name} ON {self.table}{using} ({cols}){where};"


# Default indexes for the factors catalog tables
CATALOG_INDEXES: List[IndexDefinition] = [
    IndexDefinition(
        name="idx_factors_source_id",
        table="factors_catalog.factors",
        columns=["source_id"],
    ),
    IndexDefinition(
        name="idx_factors_category",
        table="factors_catalog.factors",
        columns=["category"],
    ),
    IndexDefinition(
        name="idx_factors_edition_id",
        table="factors_catalog.factors",
        columns=["edition_id"],
    ),
    IndexDefinition(
        name="idx_factors_factor_id",
        table="factors_catalog.factors",
        columns=["factor_id"],
        unique=True,
    ),
    IndexDefinition(
        name="idx_factors_geography",
        table="factors_catalog.factors",
        columns=["geography"],
    ),
    IndexDefinition(
        name="idx_factors_year",
        table="factors_catalog.factors",
        columns=["year"],
    ),
    IndexDefinition(
        name="idx_factors_status",
        table="factors_catalog.factors",
        columns=["status"],
        partial_filter="status = 'certified'",
    ),
    IndexDefinition(
        name="idx_factors_search_gin",
        table="factors_catalog.factors",
        columns=["search_vector"],
        index_type=IndexType.GIN,
    ),
    IndexDefinition(
        name="idx_factors_compound_lookup",
        table="factors_catalog.factors",
        columns=["edition_id", "source_id", "category", "geography"],
    ),
    IndexDefinition(
        name="idx_editions_status",
        table="factors_catalog.editions",
        columns=["status"],
    ),
    IndexDefinition(
        name="idx_tenant_overlays_lookup",
        table="factors_catalog.tenant_overlays",
        columns=["tenant_id", "factor_id"],
    ),
    IndexDefinition(
        name="idx_connector_audit_source",
        table="factors_catalog.connector_audit_log",
        columns=["connector_id", "created_at"],
    ),
]


@dataclass
class IndexRecommendation:
    """Recommendation to add/modify/drop an index."""

    action: str  # "create", "drop", "rebuild"
    index: IndexDefinition
    reason: str
    estimated_improvement: str = ""


class IndexManager:
    """
    Manages factor catalog indexes.

    Provides:
      - Default index set for new deployments
      - Index health monitoring
      - Usage-based recommendations (create/drop/rebuild)
      - Safe concurrent index creation
    """

    def __init__(self) -> None:
        self._indexes: Dict[str, IndexDefinition] = {}
        for idx in CATALOG_INDEXES:
            self._indexes[idx.name] = idx

    def list_indexes(self) -> List[IndexDefinition]:
        """Return all managed indexes."""
        return list(self._indexes.values())

    def get_index(self, name: str) -> Optional[IndexDefinition]:
        return self._indexes.get(name)

    def add_index(self, index: IndexDefinition) -> None:
        """Register a custom index."""
        self._indexes[index.name] = index
        logger.info("Registered index: %s on %s", index.name, index.table)

    def remove_index(self, name: str) -> Optional[IndexDefinition]:
        """Unregister an index."""
        idx = self._indexes.pop(name, None)
        if idx:
            idx.status = IndexStatus.DROPPED
        return idx

    def generate_create_sql(self, dialect: str = "postgresql") -> List[str]:
        """Generate SQL for all pending indexes."""
        return [
            idx.to_sql(dialect)
            for idx in self._indexes.values()
            if idx.status in (IndexStatus.PENDING, IndexStatus.INVALID)
        ]

    def generate_drop_sql(self) -> List[str]:
        """Generate DROP statements for all managed indexes."""
        return [
            f"DROP INDEX IF EXISTS {idx.name};"
            for idx in self._indexes.values()
        ]

    def update_stats(self, name: str, size_bytes: int = 0, scan_count: int = 0) -> None:
        """Update runtime stats for an index (from pg_stat_user_indexes)."""
        idx = self._indexes.get(name)
        if idx:
            idx.size_bytes = size_bytes
            idx.scan_count = scan_count
            idx.status = IndexStatus.ACTIVE

    def get_recommendations(self) -> List[IndexRecommendation]:
        """Analyze index usage and return optimization recommendations."""
        recs: List[IndexRecommendation] = []

        for idx in self._indexes.values():
            if idx.status == IndexStatus.INVALID:
                recs.append(IndexRecommendation(
                    action="rebuild",
                    index=idx,
                    reason=f"Index {idx.name} is invalid — needs REINDEX.",
                ))
            elif idx.status == IndexStatus.ACTIVE and idx.scan_count == 0 and idx.size_bytes > 1_000_000:
                recs.append(IndexRecommendation(
                    action="drop",
                    index=idx,
                    reason=f"Index {idx.name} uses {idx.size_bytes // 1024}KB but has 0 scans.",
                ))
        return recs

    def mark_active(self, name: str) -> None:
        """Mark index as successfully built."""
        idx = self._indexes.get(name)
        if idx:
            idx.status = IndexStatus.ACTIVE
