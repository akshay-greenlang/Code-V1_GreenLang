# -*- coding: utf-8 -*-
"""Electricity method pack (Scope 2 four-way basis).

One pack object per basis — location, market, supplier-specific, residual-mix —
so callers can route at the pack granularity rather than encoding the basis
inside the Corporate Scope 2 packs.  The Corporate Scope 2 packs in
``corporate.py`` are the commercial umbrellas; these Electricity packs are
the lower-level primitives that a Scope Engine adapter can call directly.
"""
from __future__ import annotations

from greenlang.data.canonical_v2 import (
    ElectricityBasis,
    FactorFamily,
    FormulaType,
    MethodProfile,
)
from greenlang.factors.method_packs.base import (
    BiogenicTreatment,
    BoundaryRule,
    DEFAULT_FALLBACK,
    DeprecationRule,
    MarketInstrumentTreatment,
    MethodPack,
    SelectionRule,
)

# Electricity packs are NOT standalone commercial offerings — they back
# the Corporate Scope 2 packs.  We don't register them under top-level
# MethodProfile enum values; instead, expose them as constants and allow
# direct import.  A future MethodProfile extension can surface them if a
# buyer wants supplier-specific resolution without the Scope 2 wrapper.

_DEPRECATION = DeprecationRule(max_age_days=365 * 3, grace_period_days=180)


ELECTRICITY_LOCATION = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,   # reuses Scope 2 profile
    name="Electricity — Location-Based Grid Intensity",
    description=(
        "Grid-average factors for the reporting entity's physical location. "
        "Primary key: (country, grid sub-region). Used when market-based data "
        "is unavailable or when policy requires location-based disclosure."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.GRID_INTENSITY,),
        allowed_formula_types=(FormulaType.DIRECT_FACTOR,),
        allowed_statuses=("certified",),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("2",),
        allowed_boundaries=("combustion",),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.PROHIBITED,
        include_transmission_losses=False,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=_DEPRECATION,
    reporting_labels=("GHG_Protocol_Scope2_LocationBased",),
    audit_text_template=(
        "Location-based Scope 2 grid factor for {geography} ({grid_region}). "
        "Source: {source_org} {source_year}."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.LOCATION_BASED,
    tags=("electricity", "open_core"),
)


ELECTRICITY_MARKET = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_MARKET,     # reuses Scope 2 profile
    name="Electricity — Market-Based (supplier-specific + certificates)",
    description=(
        "Supplier-specific contractual instruments (PPAs, RECs, GOs) applied "
        "per GHG Protocol Scope 2 Quality Criteria. Where no instrument exists, "
        "the AIB residual mix (EU) or equivalent applies."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.GRID_INTENSITY, FactorFamily.RESIDUAL_MIX),
        allowed_formula_types=(FormulaType.DIRECT_FACTOR, FormulaType.RESIDUAL_MIX),
        allowed_statuses=("certified", "preview"),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("2",),
        allowed_boundaries=("combustion",),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.ALLOWED,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=_DEPRECATION,
    reporting_labels=("GHG_Protocol_Scope2_MarketBased", "RE100"),
    audit_text_template=(
        "Market-based Scope 2. Supplier: {supplier}. Certificate: "
        "{certificate}. If none, residual mix: {residual_mix_factor_id}."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.MARKET_BASED,
    tags=("electricity",),
)


__all__ = ["ELECTRICITY_LOCATION", "ELECTRICITY_MARKET"]
