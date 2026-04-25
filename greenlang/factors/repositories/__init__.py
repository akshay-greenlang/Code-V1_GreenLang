# -*- coding: utf-8 -*-
"""GreenLang Factors — typed repositories for the v0.1 alpha publish pipeline.

Wave D / TaskCreate #31 introduces a real, contract-shaped repository for
v0.1-alpha factor records. Unlike the legacy ``FactorCatalogService`` /
``SqliteFactorCatalogRepository`` pair (which expects the older
``EmissionFactorRecord`` dataclass), :class:`AlphaFactorRepository`
stores and serves records that exactly match
``config/schemas/factor_record_v0_1.schema.json`` — so the alpha API can
round-trip every required v0.1 field without ``_coerce_v0_1`` lossy
re-shaping.

This package is read-aware: the legacy repositories continue to back the
beta+ surfaces and are unaffected.
"""
from __future__ import annotations

from greenlang.factors.repositories.alpha_v0_1_repository import (
    AlphaFactorRepository,
    FactorURNAlreadyExistsError,
)

__all__ = [
    "AlphaFactorRepository",
    "FactorURNAlreadyExistsError",
]
