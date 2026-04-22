# -*- coding: utf-8 -*-
"""Product Carbon method pack (ISO 14067 + GHG Protocol Product Standard + PACT)."""
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


PRODUCT_CARBON = MethodPack(
    profile=MethodProfile.PRODUCT_CARBON,
    name="Product Carbon Footprint (ISO 14067 / GHG PS / PACT)",
    description=(
        "Product-level LCA aligned with ISO 14067:2018, GHG Protocol Product "
        "Standard, and the Partnership for Carbon Transparency (PACT) pathfinder "
        "data exchange schema. Supports cradle-to-gate and cradle-to-grave "
        "boundaries with explicit allocation + recycled-content flags."
    ),
    selection_rule=SelectionRule(
        allowed_families=(
            FactorFamily.MATERIAL_EMBODIED,
            FactorFamily.ENERGY_CONVERSION,
            FactorFamily.EMISSIONS,
            FactorFamily.LAND_USE_REMOVALS,
        ),
        allowed_formula_types=(FormulaType.LCA,),
        allowed_statuses=("certified", "preview"),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("3",),
        allowed_boundaries=("cradle_to_gate", "cradle_to_grave"),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.NOT_APPLICABLE,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=DeprecationRule(max_age_days=365 * 5, grace_period_days=365),
    reporting_labels=("ISO_14067", "GHG_Protocol_Product", "PACT"),
    audit_text_template=(
        "Product carbon footprint per ISO 14067. Product: {product_id}. "
        "Factor: {factor_id}. Boundary: {boundary}. Allocation: "
        "{allocation_method}. Recycled content: {recycled_content_pct}%."
    ),
    pack_version="1.0.0",
    tags=("product", "licensed"),     # Product Carbon Pack is a premium SKU
)


register_pack(PRODUCT_CARBON)


__all__ = ["PRODUCT_CARBON"]
