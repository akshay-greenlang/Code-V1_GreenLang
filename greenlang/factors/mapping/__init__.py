# -*- coding: utf-8 -*-
"""
GreenLang Factors — Mapping Layer (Phase F4).

Customers almost never start with factor-ready labels. They hand us
strings like "diesel for our Indian truck fleet" or "aluminium ingot
imports from Norway" or NAICS-221112 spend lines. The mapping layer
turns those into canonical labels the resolution engine can match.

Every sub-module returns a :class:`MappingResult` that carries:

- the canonical key(s) it matched (fuel_type, transport_mode, naics, etc.)
- a confidence score (0.0–1.0)
- a short rationale (matched pattern, fuzzy rank, etc.)
- the raw source attributes that drove the decision (for the Explain
  endpoint).

Public entry points re-exported here. Individual taxonomies live in
``fuels.py``, ``transport.py``, ``materials.py``, ``waste.py``,
``electricity_market.py``, ``classifications.py``, ``spend.py``.

Usage::

    from greenlang.factors.mapping import map_fuel, map_transport, map_spend

    fuel = map_fuel("No. 2 distillate diesel, India")
    # MappingResult(canonical='diesel', confidence=0.92, ...)

    spend = map_spend("Purchased goods — aluminum extrusions", amount_usd=42000)
    # MappingResult(canonical={'factor_family': 'material_embodied', ...}, ...)

All mapping modules are **pure** — no network calls, no database reads.
Callers that need persistence wrap the results.
"""
from __future__ import annotations

from greenlang.factors.mapping.base import (
    BaseMapping,
    MappingConfidence,
    MappingError,
    MappingResult,
    normalize_text,
)
from greenlang.factors.mapping.biogenic_sources import (
    BIOGENIC_TAXONOMY,
    BiogenicCategory,
    BiogenicSource,
    CO2AccountingTreatment,
    SustainabilityCertification,
    list_biogenic_sources,
    load_biogenic_sources,
    map_biogenic_source,
)
from greenlang.factors.mapping.circular_economy import (
    CIRCULAR_TAXONOMY,
    CircularMaterialFlow,
    CircularRoute,
    MaterialLifecycle,
    RecycledContent,
    RecycledSource,
    list_circular_materials,
    load_recycled_content_factors,
    map_circular_flow,
)
from greenlang.factors.mapping.classifications import (
    TradeCodeSystem,
    cross_map_classification,
    map_classification,
    map_trade_code,
    parse_trade_code,
)
from greenlang.factors.mapping.electricity_market import (
    ElectricityMarketCategory,
    map_electricity_market,
)
from greenlang.factors.mapping.fuels import FUEL_TAXONOMY, map_fuel
from greenlang.factors.mapping.industry_codes import (
    CodeLevel,
    IndustryCode,
    IndustryCodeSystem,
    children_of,
    count_codes,
    crosswalk_code,
    get_sector_default_ef,
    list_systems,
    load_codes,
    lookup_industry_code,
    lookup_industry_label,
    parent_of,
)
from greenlang.factors.mapping.land_use import (
    DeforestationRiskTier,
    LUCCommodity,
    LUC_COMMODITY_TAXONOMY,
    LandUseCategory,
    PermanenceClass,
    ProductionRegion,
    eudr_is_in_scope,
    get_risk_tier,
    list_luc_commodities,
    load_luc_commodity_data,
    map_land_use,
    map_luc_commodity,
)
from greenlang.factors.mapping.materials import (
    MATERIAL_TAXONOMY,
    map_material,
)
from greenlang.factors.mapping.regulatory_frameworks import (
    BUILTIN_RULES as REGULATORY_FRAMEWORK_RULES,
    FrameworkApplicability,
    FrameworkIndex,
    FrameworkScope,
    RegulatoryFramework,
    tag_factor,
    tag_factor_batch,
)
from greenlang.factors.mapping.spend import map_spend
from greenlang.factors.mapping.transport import (
    TRANSPORT_MODES,
    VEHICLE_CLASSES,
    map_transport,
)
from greenlang.factors.mapping.waste import WASTE_ROUTES, map_waste

__all__ = [
    # Base
    "BaseMapping",
    "MappingConfidence",
    "MappingError",
    "MappingResult",
    "normalize_text",
    # Fuel
    "FUEL_TAXONOMY",
    "map_fuel",
    # Transport
    "TRANSPORT_MODES",
    "VEHICLE_CLASSES",
    "map_transport",
    # Materials
    "MATERIAL_TAXONOMY",
    "map_material",
    # Waste
    "WASTE_ROUTES",
    "map_waste",
    # Electricity market
    "ElectricityMarketCategory",
    "map_electricity_market",
    # Classifications
    "map_classification",
    "cross_map_classification",
    "TradeCodeSystem",
    "map_trade_code",
    "parse_trade_code",
    # Industry codes (GAP-7)
    "IndustryCode",
    "IndustryCodeSystem",
    "CodeLevel",
    "lookup_industry_code",
    "lookup_industry_label",
    "crosswalk_code",
    "children_of",
    "parent_of",
    "get_sector_default_ef",
    "list_systems",
    "count_codes",
    "load_codes",
    # Circular economy (GAP-7)
    "CIRCULAR_TAXONOMY",
    "CircularMaterialFlow",
    "CircularRoute",
    "MaterialLifecycle",
    "RecycledContent",
    "RecycledSource",
    "map_circular_flow",
    "list_circular_materials",
    "load_recycled_content_factors",
    # Land use (GAP-7)
    "DeforestationRiskTier",
    "LUCCommodity",
    "LUC_COMMODITY_TAXONOMY",
    "LandUseCategory",
    "PermanenceClass",
    "ProductionRegion",
    "map_land_use",
    "map_luc_commodity",
    "eudr_is_in_scope",
    "get_risk_tier",
    "list_luc_commodities",
    "load_luc_commodity_data",
    # Biogenic sources (GAP-7)
    "BIOGENIC_TAXONOMY",
    "BiogenicCategory",
    "BiogenicSource",
    "CO2AccountingTreatment",
    "SustainabilityCertification",
    "map_biogenic_source",
    "list_biogenic_sources",
    "load_biogenic_sources",
    # Spend
    "map_spend",
    # Regulatory framework tagger
    "REGULATORY_FRAMEWORK_RULES",
    "FrameworkApplicability",
    "FrameworkIndex",
    "FrameworkScope",
    "RegulatoryFramework",
    "tag_factor",
    "tag_factor_batch",
]
