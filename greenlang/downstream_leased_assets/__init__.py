# -*- coding: utf-8 -*-
"""
Downstream Leased Assets Agent Package - AGENT-MRV-026

GHG Protocol Scope 3, Category 13: Downstream Leased Assets.
Calculates emissions from the operation of assets OWNED by the reporting
company and LEASED TO other entities (reporter is LESSOR). This is the
mirror of Category 8 (Upstream Leased Assets) where the reporter is lessee.

Key distinction from Cat 8:
    - Reporter OWNS the assets and leases them OUT to tenants.
    - Emissions arise from tenant operations of the leased asset.
    - Requires tenant data collection, vacancy handling, allocation to tenants.
    - Operational control boundary determines inclusion.

Agent ID: GL-MRV-S3-013
Package: greenlang.downstream_leased_assets
API: /api/v1/downstream-leased-assets
DB Migration: V072
Metrics Prefix: gl_dla_
Table Prefix: gl_dla_

Supported Asset Categories:
    - Buildings (8 types, 5 climate zones, EUI benchmarks, vacancy handling)
    - Vehicles (8 types, 7 fuel types, fleet management)
    - Equipment (6 types, fuel-based and load-factor calculations)
    - IT Assets (7 types, PUE-adjusted power, data center focus)

Calculation Methods:
    - Asset-specific (metered energy data from tenants)
    - Average-data (EUI benchmarks by building type and climate zone)
    - Spend-based (EEIO factors by NAICS leasing codes)
    - Hybrid (combines multiple methods with weighted aggregation)

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

__all__ = [
    "DownstreamAssetDatabaseEngine",
    "AssetSpecificCalculatorEngine",
    "AverageDataCalculatorEngine",
    "SpendBasedCalculatorEngine",
    "HybridAggregatorEngine",
    "ComplianceCheckerEngine",
    "DownstreamLeasedAssetsPipelineEngine",
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    "PIPELINE_STAGES",
    "get_config",
]

AGENT_ID: str = "GL-MRV-S3-013"
AGENT_COMPONENT: str = "AGENT-MRV-026"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_dla_"

PIPELINE_STAGES: list = [
    "validate",
    "classify",
    "normalize",
    "resolve_efs",
    "calculate",
    "allocate",
    "aggregate",
    "compliance",
    "provenance",
    "seal",
]

# Graceful imports - each engine with try/except
try:
    from greenlang.downstream_leased_assets.downstream_asset_database import (
        DownstreamAssetDatabaseEngine,
    )
except ImportError:
    DownstreamAssetDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_leased_assets.asset_specific_calculator import (
        AssetSpecificCalculatorEngine,
    )
except ImportError:
    AssetSpecificCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_leased_assets.average_data_calculator import (
        AverageDataCalculatorEngine,
    )
except ImportError:
    AverageDataCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_leased_assets.spend_based_calculator import (
        SpendBasedCalculatorEngine,
    )
except ImportError:
    SpendBasedCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_leased_assets.hybrid_aggregator import (
        HybridAggregatorEngine,
    )
except ImportError:
    HybridAggregatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_leased_assets.compliance_checker import (
        ComplianceCheckerEngine,
    )
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.downstream_leased_assets.downstream_leased_assets_pipeline import (
        DownstreamLeasedAssetsPipelineEngine,
    )
except ImportError:
    DownstreamLeasedAssetsPipelineEngine = None  # type: ignore[assignment,misc]

# Export agent metadata from models (authoritative source)
try:
    from greenlang.downstream_leased_assets.models import (
        AGENT_ID,
        AGENT_COMPONENT,
        VERSION as MODELS_VERSION,
        TABLE_PREFIX,
    )
except ImportError:
    # Fallback metadata already defined above
    pass

# Export configuration helper
try:
    from greenlang.downstream_leased_assets.config import get_config
except ImportError:
    def get_config():  # type: ignore[misc]
        """Fallback get_config if config module is not available."""
        return None
