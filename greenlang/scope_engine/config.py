# -*- coding: utf-8 -*-
"""Scope Engine configuration."""

from __future__ import annotations

from greenlang.schemas.base import GreenLangConfig
from greenlang.scope_engine.models import ConsolidationApproach, GWPBasis


class ScopeEngineConfig(GreenLangConfig):
    default_gwp_basis: GWPBasis = GWPBasis.AR6_100YR
    default_consolidation: ConsolidationApproach = ConsolidationApproach.OPERATIONAL_CONTROL
    factor_cache_ttl_seconds: int = 3600
    max_activities_per_request: int = 10_000
    enable_provenance: bool = True
    strict_mode: bool = True
