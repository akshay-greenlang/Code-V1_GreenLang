# -*- coding: utf-8 -*-
"""Product-level LCA method pack variants — PAS 2050, PEF, OEF.

These sit alongside the core :mod:`product_carbon` pack (ISO 14067 /
GHG PS / PACT).  They are variant-level registrations that callers
reach via ``get_pack("pas_2050")`` / ``get_pack("eu_pef")`` /
``get_pack("eu_oef")`` — mirroring the PCAF and LSR registration style.

Each variant ships with its own:
- selection rule + boundary (PEF adopts 16 EF indicators; PAS focuses on
  GHGs; OEF is entity-scope LCA);
- recommended impact-category set;
- GWP set alignment (PAS 2050 historically AR5; PEF/OEF AR6);
- fallback hierarchy biased to the variant's authoritative data sources
  (BSI, EC JRC EF 3.1, EF secondary datasets).

The variants do *not* redefine the umbrella ``PRODUCT_CARBON`` profile
— that one remains the default ISO 14067 / GHG PS flow.
"""
from __future__ import annotations

from typing import Dict

from greenlang.data.canonical_v2 import FactorFamily, FormulaType, MethodProfile
from greenlang.factors.method_packs.base import (
    BiogenicTreatment,
    BoundaryRule,
    DEFAULT_FALLBACK,
    DeprecationRule,
    MarketInstrumentTreatment,
    MethodPack,
    SelectionRule,
)


_PRODUCT_LCA_DEPRECATION = DeprecationRule(max_age_days=365 * 5, grace_period_days=365)


# ---------------------------------------------------------------------------
# PAS 2050:2011 — Product carbon footprint (BSI / UK)
# ---------------------------------------------------------------------------

PAS_2050 = MethodPack(
    profile=MethodProfile.PRODUCT_CARBON,
    name="Product Carbon Footprint (PAS 2050:2011, BSI)",
    description=(
        "BSI PAS 2050:2011 product life-cycle GHG assessment. Cradle-to-"
        "grave boundary by default (cradle-to-gate allowed for B2B "
        "intermediates). Biogenic CO2 excluded from the headline number "
        "per PAS 2050 §7.4 and reported as a supplementary line."
    ),
    selection_rule=SelectionRule(
        allowed_families=(
            FactorFamily.MATERIAL_EMBODIED,
            FactorFamily.ENERGY_CONVERSION,
            FactorFamily.EMISSIONS,
            FactorFamily.TRANSPORT_LANE,
            FactorFamily.WASTE_TREATMENT,
            FactorFamily.LAND_USE_REMOVALS,
        ),
        allowed_formula_types=(FormulaType.LCA, FormulaType.DIRECT_FACTOR),
        allowed_statuses=("certified", "preview"),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("3",),
        allowed_boundaries=("cradle_to_gate", "cradle_to_grave"),
        biogenic_treatment=BiogenicTreatment.EXCLUDED,
        market_instruments=MarketInstrumentTreatment.NOT_APPLICABLE,
    ),
    gwp_basis="IPCC_AR5_100",  # historical alignment; callers may override
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=_PRODUCT_LCA_DEPRECATION,
    reporting_labels=("PAS_2050", "BSI", "ISO_14067"),
    audit_text_template=(
        "Product carbon footprint per PAS 2050:2011. Product: {product_id}. "
        "Factor: {factor_id}. Boundary: {boundary}. Functional unit: "
        "{functional_unit}. Biogenic CO2 excluded from headline per §7.4."
    ),
    pack_version="1.0.0",
    tags=("product", "licensed", "pas_2050"),
)


# ---------------------------------------------------------------------------
# EU PEF — Product Environmental Footprint (EC JRC EF method)
# ---------------------------------------------------------------------------

PEF = MethodPack(
    profile=MethodProfile.PRODUCT_CARBON,
    name="EU Product Environmental Footprint (PEF, EC JRC EF 3.1)",
    description=(
        "European Commission Product Environmental Footprint method. "
        "16 impact-category assessment (GWP + 15 others); this pack "
        "surfaces the climate-change indicator but preserves the full "
        "EF 3.1 vector in `extras`. Cradle-to-grave boundary. Biogenic "
        "carbon handled per the -1/+1 convention; PEFCR category rules "
        "override defaults when present."
    ),
    selection_rule=SelectionRule(
        allowed_families=(
            FactorFamily.MATERIAL_EMBODIED,
            FactorFamily.ENERGY_CONVERSION,
            FactorFamily.EMISSIONS,
            FactorFamily.TRANSPORT_LANE,
            FactorFamily.WASTE_TREATMENT,
            FactorFamily.LAND_USE_REMOVALS,
        ),
        allowed_formula_types=(FormulaType.LCA,),
        require_verification=True,
        allowed_statuses=("certified",),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("3",),
        allowed_boundaries=("cradle_to_grave",),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.NOT_APPLICABLE,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=_PRODUCT_LCA_DEPRECATION,
    reporting_labels=("EU_PEF", "EF_3_1", "PEFCR", "ISO_14067"),
    audit_text_template=(
        "EU Product Environmental Footprint per EF 3.1. Product: "
        "{product_id}. Functional unit: {functional_unit}. Climate-"
        "change indicator factor: {factor_id}. PEFCR applied: {pefcr_id}."
    ),
    pack_version="1.0.0",
    tags=("product", "licensed", "eu_pef", "regulated"),
)


# ---------------------------------------------------------------------------
# EU OEF — Organisation Environmental Footprint
# ---------------------------------------------------------------------------

OEF = MethodPack(
    profile=MethodProfile.PRODUCT_CARBON,  # reuse the product-LCA profile umbrella
    name="EU Organisation Environmental Footprint (OEF, EC JRC)",
    description=(
        "Entity-level counterpart of PEF. Covers the organisation's "
        "portfolio of products + activities across the same 16 impact "
        "categories. Supports OEFSR sector rules when registered. This "
        "pack enables entity-level LCA reporting aligned with EU "
        "environmental-footprint guidance."
    ),
    selection_rule=SelectionRule(
        allowed_families=(
            FactorFamily.MATERIAL_EMBODIED,
            FactorFamily.ENERGY_CONVERSION,
            FactorFamily.EMISSIONS,
            FactorFamily.TRANSPORT_LANE,
            FactorFamily.WASTE_TREATMENT,
            FactorFamily.FINANCE_PROXY,
        ),
        allowed_formula_types=(FormulaType.LCA, FormulaType.SPEND_PROXY),
        require_verification=False,
        allowed_statuses=("certified", "preview"),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("1", "2", "3"),
        allowed_boundaries=("cradle_to_gate", "cradle_to_grave", "WTW", "WTT"),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.ALLOWED,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=_PRODUCT_LCA_DEPRECATION,
    reporting_labels=("EU_OEF", "EF_3_1", "OEFSR", "ESRS_E1"),
    audit_text_template=(
        "EU Organisation Environmental Footprint per EC JRC guidance. "
        "Entity: {entity_id}. Reporting year: {reporting_year}. OEFSR "
        "applied: {oefsr_id}. Factor: {factor_id}."
    ),
    pack_version="1.0.0",
    tags=("entity", "licensed", "eu_oef"),
)


# ---------------------------------------------------------------------------
# Variant registry — follows the PCAF / LSR pattern
# ---------------------------------------------------------------------------


_VARIANTS: Dict[str, MethodPack] = {
    "pas_2050": PAS_2050,
    "eu_pef": PEF,
    "eu_oef": OEF,
}


def get_product_lca_variant(key: str) -> MethodPack:
    """Look up a product-LCA variant pack by name.

    Raises ``KeyError`` when the variant is unknown; the umbrella
    :func:`greenlang.factors.method_packs.get_pack` wrapper translates
    that into ``MethodPackNotFound``.
    """
    needle = str(key).strip().lower()
    if needle not in _VARIANTS:
        raise KeyError(f"unknown product-LCA variant: {key!r}")
    return _VARIANTS[needle]


def list_product_lca_variants() -> Dict[str, MethodPack]:
    """Return the full variant map (read-only copy)."""
    return dict(_VARIANTS)


__all__ = [
    "PAS_2050",
    "PEF",
    "OEF",
    "get_product_lca_variant",
    "list_product_lca_variants",
]
