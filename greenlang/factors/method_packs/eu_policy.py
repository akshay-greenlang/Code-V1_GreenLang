# -*- coding: utf-8 -*-
"""EU Policy method pack — CBAM + DPP."""
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


EU_CBAM = MethodPack(
    profile=MethodProfile.EU_CBAM,
    name="EU CBAM — Carbon Border Adjustment Mechanism",
    description=(
        "Embedded emissions for CBAM-covered goods (cement, iron & steel, "
        "aluminium, fertilizers, electricity, hydrogen). Regulation (EU) "
        "2023/956; definitive regime from 1 January 2026. Declarants must use "
        "primary data from operators where possible; EU default values only as "
        "fallback with surcharge implications."
    ),
    selection_rule=SelectionRule(
        allowed_families=(
            FactorFamily.EMISSIONS,
            FactorFamily.MATERIAL_EMBODIED,
            FactorFamily.GRID_INTENSITY,
            FactorFamily.CARBON_CONTENT,
        ),
        allowed_formula_types=(
            FormulaType.COMBUSTION,
            FormulaType.LCA,
            FormulaType.DIRECT_FACTOR,
        ),
        # Only certified + verified data permitted.  Preview factors are
        # blocked because CBAM declarations go to EU authorities.
        allowed_statuses=("certified",),
        require_verification=True,
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("1", "2"),
        allowed_boundaries=("cradle_to_gate", "combustion"),
        # CBAM follows Annex III of the implementing regulation: direct +
        # indirect (where applicable), biogenic excluded from the CBAM value.
        biogenic_treatment=BiogenicTreatment.EXCLUDED,
        market_instruments=MarketInstrumentTreatment.REQUIRE_CERTIFICATE,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=DeprecationRule(max_age_days=365 * 2, grace_period_days=90),
    reporting_labels=("EU_CBAM",),
    audit_text_template=(
        "CBAM embedded emissions for CN code {cn_code}. Factor: {factor_id} "
        "(source: {source_org} {source_year}, verification: {verification_status}). "
        "If primary data unavailable, EU default values used with documented "
        "fallback per Article 4(2)."
    ),
    pack_version="1.0.0",
    tags=("eu_policy", "licensed"),  # CBAM pack may be a premium SKU
)


EU_DPP = MethodPack(
    profile=MethodProfile.EU_DPP,
    name="EU Digital Product Passport (DPP) — product data shape",
    description=(
        "Product data layout for the EU Digital Product Passport under ESPR "
        "(Regulation (EU) 2024/1781). Implementation acts define per-product "
        "requirements; this pack provides the factor-selection contract for "
        "emissions + embodied carbon fields in the DPP."
    ),
    selection_rule=SelectionRule(
        allowed_families=(
            FactorFamily.MATERIAL_EMBODIED,
            FactorFamily.EMISSIONS,
        ),
        allowed_formula_types=(FormulaType.LCA, FormulaType.DIRECT_FACTOR),
        allowed_statuses=("certified",),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("3",),
        allowed_boundaries=("cradle_to_gate", "cradle_to_grave"),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.NOT_APPLICABLE,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=DeprecationRule(max_age_days=365 * 3, grace_period_days=180),
    reporting_labels=("EU_DPP", "ESPR"),
    audit_text_template=(
        "DPP product-carbon entry for {product_id}. Factor: {factor_id} "
        "(LCA boundary: {boundary})."
    ),
    pack_version="0.1.0",            # pre-regulation; bumps when implementing acts land
    tags=("eu_policy", "product",),
)


for _pack in (EU_CBAM, EU_DPP):
    register_pack(_pack)


__all__ = ["EU_CBAM", "EU_DPP"]
