# -*- coding: utf-8 -*-
"""Land Sector & Removals pack (GHG Protocol LSR Standard) — FY28 stub."""
from __future__ import annotations

from greenlang.data.canonical_v2 import (
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
from greenlang.factors.method_packs.registry import register_pack


LAND_REMOVALS = MethodPack(
    profile=MethodProfile.LAND_REMOVALS,
    name="Land Sector & Removals (GHG Protocol LSR Standard)",
    description=(
        "Biogenic CO2 removals, land-use change emissions, afforestation / "
        "reforestation, soil carbon, BECCS, biochar, and durable carbon "
        "storage per the GHG Protocol Land Sector and Removals Standard. "
        "Captures permanence, reversal risk, and leakage explicitly — do NOT "
        "bolt onto generic Scope 3."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.LAND_USE_REMOVALS,),
        allowed_formula_types=(
            FormulaType.DIRECT_FACTOR,
            FormulaType.CARBON_BUDGET,
        ),
        allowed_statuses=("certified", "preview"),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("1", "3"),
        allowed_boundaries=("cradle_to_grave",),
        biogenic_treatment=BiogenicTreatment.INCLUDED,
        market_instruments=MarketInstrumentTreatment.REQUIRE_CERTIFICATE,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=DeprecationRule(max_age_days=365 * 5, grace_period_days=730),
    reporting_labels=("GHG_Protocol_LSR",),
    audit_text_template=(
        "Land / removals entry per GHG Protocol LSR. Activity: {activity}. "
        "Factor: {factor_id}. Permanence class: {permanence_class}. "
        "Reversal risk: {reversal_risk_flag}."
    ),
    pack_version="0.1.0",            # FY28 pack; schema stable, data pending
    tags=("land", "licensed"),
)


register_pack(LAND_REMOVALS)


__all__ = ["LAND_REMOVALS"]
