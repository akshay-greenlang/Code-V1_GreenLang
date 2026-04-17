# -*- coding: utf-8 -*-
"""GreenLang Factors FY27 — catalog, editions, ETL, and repository layer."""

from __future__ import annotations

from typing import Any

__all__ = [
    "collect_inventory",
    "write_coverage_matrix",
    "EditionManifest",
    "FactorCatalogService",
    "SqliteFactorCatalogRepository",
    "MemoryFactorCatalogRepository",
]


def __getattr__(name: str) -> Any:
    if name == "collect_inventory":
        from greenlang.factors.inventory import collect_inventory

        return collect_inventory
    if name == "write_coverage_matrix":
        from greenlang.factors.inventory import write_coverage_matrix

        return write_coverage_matrix
    if name == "EditionManifest":
        from greenlang.factors.edition_manifest import EditionManifest

        return EditionManifest
    if name == "FactorCatalogService":
        from greenlang.factors.service import FactorCatalogService

        return FactorCatalogService
    if name == "SqliteFactorCatalogRepository":
        from greenlang.factors.catalog_repository import SqliteFactorCatalogRepository

        return SqliteFactorCatalogRepository
    if name == "MemoryFactorCatalogRepository":
        from greenlang.factors.catalog_repository import MemoryFactorCatalogRepository

        return MemoryFactorCatalogRepository
    raise AttributeError(name)
