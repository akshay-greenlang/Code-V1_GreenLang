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
