# -*- coding: utf-8 -*-
"""Factor catalog adapter — deterministic emission-factor lookup.

Resolves activity -> emission factor via greenlang.factors.FactorCatalogService.
Caches by lookup key (in-memory LRU; Redis layer added alongside INFRA-003 at
prod deployment time — see factor cache toggle in config.py).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from greenlang.data.emission_factor_record import Scope
from greenlang.schemas.enums import GeographicRegion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactorLookupKey:
    activity_type: str
    region: Optional[GeographicRegion]
    year: int
    scope: Scope
    methodology: Optional[str] = None

    def cache_key(self) -> tuple:
        return (
            self.activity_type,
            self.region.value if self.region else None,
            self.year,
            self.scope.value,
            self.methodology,
        )


@dataclass(frozen=True)
class ResolvedFactor:
    factor_id: str
    source: str
    vintage: int
    unit: str
    scope: Scope
    boundary: Any
    license_tag: str
    tier_required: str  # community | pro | enterprise
    raw_record: Any  # greenlang.data.emission_factor_record.EmissionFactorRecord


class FactorAdapter:
    """Resolves emission factors by lookup key with caching + tier enforcement."""

    def __init__(self, service=None, tier: str = "community", enable_cache: bool = True):
        self._service = service
        self._tier = tier
        self._enable_cache = enable_cache
        self._cache: dict[tuple, ResolvedFactor] = {}

    def resolve(self, key: FactorLookupKey, override_id: Optional[str] = None) -> ResolvedFactor:
        cache_key = (override_id, *key.cache_key())
        if self._enable_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if self._service is None:
            self._service = self._build_default_service()

        repo = getattr(self._service, "repo", None) or self._service._repo  # noqa: SLF001
        edition_id = repo.get_default_edition_id()

        if override_id:
            record = repo.get_factor(edition_id, override_id)
        else:
            geography = key.region.value if key.region else None
            records, _total = repo.list_factors(
                edition_id=edition_id,
                fuel_type=key.activity_type,
                geography=geography,
                scope=key.scope.value,
                limit=1,
            )
            # Fallback: retry without geography filter (any region)
            if not records and geography is not None:
                records, _total = repo.list_factors(
                    edition_id=edition_id,
                    fuel_type=key.activity_type,
                    scope=key.scope.value,
                    limit=1,
                )
            record = records[0] if records else None

        if record is None:
            raise ValueError(
                f"No emission factor found for lookup key={key} override={override_id}"
            )

        resolved = self._record_to_resolved(record, key)
        self._enforce_tier(resolved)
        if self._enable_cache:
            self._cache[cache_key] = resolved
        return resolved

    @staticmethod
    def _record_to_resolved(record: Any, key: FactorLookupKey) -> ResolvedFactor:
        provenance = getattr(record, "provenance", None)
        license_info = getattr(record, "license_info", None)
        source = getattr(provenance, "source_org", None) or "unknown"
        vintage = getattr(provenance, "source_year", None) or key.year
        license_tag = getattr(license_info, "license", None) or "proprietary"
        tier_required = getattr(record, "tier_required", None) or "community"
        return ResolvedFactor(
            factor_id=record.factor_id,
            source=source,
            vintage=int(vintage),
            unit=record.unit,
            scope=record.scope,
            boundary=getattr(record, "boundary", None),
            license_tag=license_tag,
            tier_required=tier_required,
            raw_record=record,
        )

    def _enforce_tier(self, resolved: ResolvedFactor) -> None:
        order = {"community": 0, "pro": 1, "enterprise": 2}
        required = order.get(resolved.tier_required, 0)
        have = order.get(self._tier, 0)
        if required > have:
            raise PermissionError(
                f"Factor {resolved.factor_id} requires tier "
                f"'{resolved.tier_required}'; client tier='{self._tier}'"
            )

    @staticmethod
    @lru_cache(maxsize=1)
    def _build_default_service():
        """Build default service with graceful fallback.

        Priority:
        1. SQLite catalog if GREENLANG_FACTORS_DB has populated editions
        2. MemoryFactorCatalogRepository wrapping built-in EmissionFactorDatabase
        """
        from greenlang.factors.catalog_repository import (
            MemoryFactorCatalogRepository,
            SqliteFactorCatalogRepository,
        )
        from greenlang.factors.service import FactorCatalogService

        db_path = Path(
            os.environ.get(
                "GREENLANG_FACTORS_DB",
                str(Path("greenlang") / "data" / "emission_factors.db"),
            )
        )
        if db_path.exists():
            sqlite_repo = SqliteFactorCatalogRepository(db_path)
            try:
                editions = sqlite_repo.list_editions(include_pending=False)
            except Exception:
                editions = []
            if editions:
                return FactorCatalogService(sqlite_repo)

        from greenlang.data.emission_factor_database import EmissionFactorDatabase

        db = EmissionFactorDatabase(enable_cache=False)
        memory_repo = MemoryFactorCatalogRepository(
            edition_id="builtin-v2", label="Built-in v2 factors", db=db
        )
        return FactorCatalogService(memory_repo)
