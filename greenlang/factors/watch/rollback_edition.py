# -*- coding: utf-8 -*-
"""Hotfix / rollback edition override (U6) via GL_FACTORS_FORCE_EDITION (see service.resolve_edition_id)."""

from __future__ import annotations

from typing import Optional, Tuple

from greenlang.factors.catalog_repository import FactorCatalogRepository
from greenlang.factors.service import resolve_edition_id


def resolve_edition_with_rollback_override(
    repo: FactorCatalogRepository,
    header_edition: Optional[str],
    query_edition: Optional[str],
) -> Tuple[str, str]:
    """Delegate to resolve_edition_id (env override lives there)."""
    return resolve_edition_id(repo, header_edition, query_edition)
