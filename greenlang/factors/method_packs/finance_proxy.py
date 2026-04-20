# -*- coding: utf-8 -*-
"""PCAF financed-emissions method pack — FY28 stub."""
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


FINANCE_PROXY = MethodPack(
    profile=MethodProfile.FINANCE_PROXY,
    name="Financed Emissions (PCAF)",
    description=(
        "Attribution factors for financed emissions across the PCAF asset "
        "classes (listed equity + corporate bonds, business loans + unlisted "
        "equity, project finance, commercial real estate, mortgages, motor "
        "vehicle loans, sovereign debt). Data-quality score 1–5 per PCAF "
        "Global Standard."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.FINANCE_PROXY,),
        allowed_formula_types=(FormulaType.SPEND_PROXY,),
        allowed_statuses=("certified", "preview"),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("3",),                  # Scope 3 Cat 15
        allowed_boundaries=("cradle_to_grave",),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.NOT_APPLICABLE,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=DeprecationRule(max_age_days=365 * 4, grace_period_days=365),
    reporting_labels=("PCAF", "GHG_Protocol_Scope3_Cat15"),
    audit_text_template=(
        "PCAF asset class {asset_class}. Attribution factor: {attribution_factor}. "
        "Data quality score: {pcaf_dqs}/5."
    ),
    pack_version="0.1.0",
    tags=("finance", "licensed"),
)


register_pack(FINANCE_PROXY)


__all__ = ["FINANCE_PROXY"]
