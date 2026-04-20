# -*- coding: utf-8 -*-
"""
Runtime wiring for the factor catalog (SQLite path vs in-memory built-ins).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from greenlang.data.emission_factor_database import EmissionFactorDatabase

logger = logging.getLogger(__name__)

from greenlang.factors.catalog_repository import (
    FactorCatalogRepository,
    MemoryFactorCatalogRepository,
    SqliteFactorCatalogRepository,
)


def _sqlite_path_from_env() -> Optional[Path]:
    raw = os.getenv("GL_FACTORS_SQLITE_PATH", "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


class FactorCatalogService:
    """Holds the active repository and default edition resolution helpers."""

    def __init__(self, repo: FactorCatalogRepository):
        self.repo = repo

    def compare_editions(self, left: str, right: str) -> Dict[str, Any]:
        """Diff factor_id sets and content_hash between two editions (A2)."""
        self.repo.resolve_edition(left)
        self.repo.resolve_edition(right)
        lm = {r["factor_id"]: r["content_hash"] for r in self.repo.list_factor_summaries(left)}
        rm = {r["factor_id"]: r["content_hash"] for r in self.repo.list_factor_summaries(right)}
        left_ids, right_ids = set(lm), set(rm)
        added = sorted(right_ids - left_ids)
        removed = sorted(left_ids - right_ids)
        changed = sorted(fid for fid in left_ids & right_ids if lm[fid] != rm[fid])
        unchanged = sorted(fid for fid in left_ids & right_ids if lm[fid] == rm[fid])
        return {
            "left_edition_id": left,
            "right_edition_id": right,
            "added_factor_ids": added,
            "removed_factor_ids": removed,
            "changed_factor_ids": changed,
            "unchanged_count": len(unchanged),
        }

    def status_summary(self, edition_id: str) -> Dict[str, Any]:
        """Aggregate factor counts by three-label coverage (Phase 5.3).

        Returns the counts in the public dashboard shape::

            {
              "edition_id": "<id>",
              "totals": {
                "certified": 245, "preview": 78,
                "connector_only": 12, "deprecated": 5, "all": 340,
              },
              "by_source": [
                {"source_id": "epa_hub", "certified": 100, "preview": 20, ...},
                ...
              ],
              "generated_at": "<iso>",
            }

        The aggregation pulls every factor in the edition once and folds
        it by ``factor_status`` and ``source_id``.  In memory-backed
        catalogs (tests / CI) this is cheap; in the SQLite-backed
        catalog we stream via ``list_factors(limit=…)`` in one pass.
        """
        from datetime import datetime, timezone

        self.repo.resolve_edition(edition_id)
        factors, _total = self.repo.list_factors(
            edition_id,
            page=1,
            limit=1_000_000,
            include_preview=True,
            include_connector=True,
        )

        totals: Dict[str, int] = {
            "certified": 0,
            "preview": 0,
            "connector_only": 0,
            "deprecated": 0,
        }
        by_source: Dict[str, Dict[str, int]] = {}
        for factor in factors:
            status = str(
                getattr(factor, "factor_status", "certified") or "certified"
            ).lower()
            if status not in totals:
                status = "certified"
            totals[status] += 1

            source_id = getattr(factor, "source_id", None) or "unknown"
            source_bucket = by_source.setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "certified": 0,
                    "preview": 0,
                    "connector_only": 0,
                    "deprecated": 0,
                    "all": 0,
                },
            )
            source_bucket[status] += 1
            source_bucket["all"] += 1

        totals["all"] = sum(totals.values())
        by_source_rows = sorted(by_source.values(), key=lambda r: r["source_id"])
        return {
            "edition_id": edition_id,
            "totals": totals,
            "by_source": by_source_rows,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def replacement_chain(self, edition_id: str, factor_id: str, max_depth: int = 32) -> List[str]:
        """Walk replacement_factor_id links (A3)."""
        chain: List[str] = []
        cur: Optional[str] = factor_id
        for _ in range(max_depth):
            if not cur or cur in chain:
                break
            chain.append(cur)
            rec = self.repo.get_factor(edition_id, cur)
            if not rec:
                break
            nxt = getattr(rec, "replacement_factor_id", None)
            if not nxt:
                break
            cur = nxt
        return chain

    @classmethod
    def from_environment(
        cls,
        emission_db: Optional[EmissionFactorDatabase] = None,
    ) -> "FactorCatalogService":
        """
        Prefer GL_FACTORS_SQLITE_PATH when the file exists and contains editions;
        otherwise use in-memory EmissionFactorDatabase adapter.
        """
        db = emission_db or EmissionFactorDatabase(enable_cache=True)
        sqlite_path = _sqlite_path_from_env()
        if sqlite_path and sqlite_path.is_file():
            try:
                probe = SqliteFactorCatalogRepository(sqlite_path)
                eid = probe.get_default_edition_id()
                if eid:
                    logger.info("Resolved factor catalog from SQLite path=%s edition=%s", sqlite_path, eid)
                    return cls(probe)
            except Exception:
                logger.warning("SQLite catalog at %s failed probe, falling back to memory", sqlite_path)
        edition = os.getenv("GL_FACTORS_BUILTIN_EDITION", "builtin-v1.0.0")
        label = os.getenv("GL_FACTORS_BUILTIN_LABEL", "GreenLang built-in v2 factors")
        logger.info("Using in-memory factor catalog edition=%s", edition)
        return cls(MemoryFactorCatalogRepository(edition, label, db))


def resolve_edition_id(
    repo: FactorCatalogRepository,
    header_edition: Optional[str],
    query_edition: Optional[str],
) -> Tuple[str, str]:
    """
    Resolve edition from X-Factors-Edition header, then ?edition=, then default.

    Returns:
        (resolved_edition_id, resolution_source) where source is header|query|default

    Raises:
        ValueError: unknown edition id when header or query pins a non-existent edition.
    """
    forced = os.getenv("GL_FACTORS_FORCE_EDITION", "").strip()
    if forced:
        logger.info("Edition forced by GL_FACTORS_FORCE_EDITION=%s", forced)
        return repo.resolve_edition(forced), "rollback_override"
    if header_edition:
        logger.debug("Edition resolved from header=%s", header_edition)
        return repo.resolve_edition(header_edition), "header"
    if query_edition:
        logger.debug("Edition resolved from query=%s", query_edition)
        return repo.resolve_edition(query_edition), "query"
    return repo.resolve_edition(None), "default"
