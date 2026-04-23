# -*- coding: utf-8 -*-
"""Product-level LCA method pack variants — PAS 2050, PEF, OEF.

**Wave 4-G — v0.2 promotion.**

These sit alongside the core :mod:`product_carbon` pack (ISO 14067 /
GHG PS / PACT).  They are variant-level registrations that callers
reach via ``get_pack("pas_2050")`` / ``get_pack("eu_pef")`` /
``get_pack("eu_oef")`` — mirroring the PCAF and LSR registration style.

Each variant ships with its own:
- selection rule + boundary (PEF adopts 16 EF indicators; PAS focuses on
  GHGs; OEF is entity-scope LCA);
- recommended impact-category set;
- GWP set alignment (PAS 2050 historically AR5; PEF/OEF AR6);
- functional-unit sidecar (ISO 14067-aligned);
- PCR / EPD / PEFCR references;
- strict cannot-resolve-safely contract.

The variants do *not* redefine the umbrella ``PRODUCT_CARBON`` profile
— that one remains the default ISO 14067 / GHG PS flow.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from greenlang.data.canonical_v2 import FactorFamily, FormulaType, MethodProfile
from greenlang.factors.method_packs.base import (
    BiogenicTreatment,
    BoundaryRule,
    CannotResolveAction,
    DEFAULT_FALLBACK,
    DeprecationRule,
    MarketInstrumentTreatment,
    MethodPack,
    SelectionRule,
)
from greenlang.factors.method_packs.product_carbon import (
    AllocationMethod,
    PRODUCT_EXCLUDED_ACTIVITY_CATEGORIES,
    PRODUCT_INCLUDED_ACTIVITY_CATEGORIES,
    ProductCarbonPackMetadata,
    ProductSystemBoundary,
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
        "per PAS 2050 §7.4 and reported as a supplementary line. Strict "
        "cannot-resolve-safely contract (no tier-7 global default)."
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
        included_activity_categories=PRODUCT_INCLUDED_ACTIVITY_CATEGORIES,
        excluded_activity_categories=PRODUCT_EXCLUDED_ACTIVITY_CATEGORIES,
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
    pack_version="0.2.0",
    tags=("product", "licensed", "pas_2050"),
    cannot_resolve_action=CannotResolveAction.RAISE_NO_SAFE_MATCH,
    global_default_tier_allowed=False,
    replacement_pack_id=None,
    deprecation_notice_days=180,
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
        "override defaults when present. Strict cannot-resolve-safely."
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
        included_activity_categories=PRODUCT_INCLUDED_ACTIVITY_CATEGORIES,
        excluded_activity_categories=PRODUCT_EXCLUDED_ACTIVITY_CATEGORIES,
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
    pack_version="0.2.0",
    tags=("product", "licensed", "eu_pef", "regulated"),
    cannot_resolve_action=CannotResolveAction.RAISE_NO_SAFE_MATCH,
    global_default_tier_allowed=False,
    replacement_pack_id=None,
    deprecation_notice_days=180,
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
        "environmental-footprint guidance. Strict cannot-resolve-safely."
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
        # OEF covers entity-scope so it legitimately DOES include Scope 1/2,
        # unlike the product packs; we therefore do NOT apply the product
        # exclusion list here.
        included_activity_categories=frozenset(),
        excluded_activity_categories=frozenset({"carbon_offsets"}),
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
    pack_version="0.2.0",
    tags=("entity", "licensed", "eu_oef"),
    cannot_resolve_action=CannotResolveAction.RAISE_NO_SAFE_MATCH,
    global_default_tier_allowed=False,
    replacement_pack_id=None,
    deprecation_notice_days=180,
)


# ---------------------------------------------------------------------------
# Variant registry — follows the PCAF / LSR pattern
# ---------------------------------------------------------------------------


_VARIANTS: Dict[str, MethodPack] = {
    "pas_2050": PAS_2050,
    "eu_pef": PEF,
    "eu_oef": OEF,
}


# Sidecar metadata map (MP8 scaffold). TODO(methodology-review): PCR
# / EPD / PEFCR references below are scaffolds — methodology lead must
# confirm each entry before the variant flips to `certified`.
PRODUCT_LCA_VARIANT_METADATA: Dict[str, ProductCarbonPackMetadata] = {
    "pas_2050": ProductCarbonPackMetadata(
        pack_id="pas_2050",
        standards_alignment=(
            "BSI PAS 2050:2011",
            "ISO 14067:2018",
            "GHG Protocol Product Standard 2011",
        ),
        system_boundary=ProductSystemBoundary.CRADLE_TO_GRAVE,
        allocation_method=AllocationMethod.PHYSICAL,  # §8 priority
        functional_unit_required=True,
        recycled_content_assumption=0.0,
        supplier_primary_data_share=0.0,
        pcr_reference=None,            # TODO(methodology-review)
        epd_reference=None,            # TODO(methodology-review)
        pefcr_id=None,
        pact_compatible=False,
        deprecation_notice_days=180,
    ),
    "eu_pef": ProductCarbonPackMetadata(
        pack_id="eu_pef",
        standards_alignment=(
            "EU PEF / EC JRC EF 3.1",
            "PEFCR (per category)",
            "ISO 14067:2018",
        ),
        system_boundary=ProductSystemBoundary.CRADLE_TO_GRAVE,
        allocation_method=AllocationMethod.PHYSICAL,
        functional_unit_required=True,
        recycled_content_assumption=0.0,
        supplier_primary_data_share=0.0,
        pcr_reference=None,
        epd_reference=None,
        pefcr_id=None,                 # TODO(methodology-review) per-category routing
        pact_compatible=False,
        deprecation_notice_days=180,
    ),
    "eu_oef": ProductCarbonPackMetadata(
        pack_id="eu_oef",
        standards_alignment=(
            "EU OEF / EC JRC EF 3.1",
            "OEFSR (per sector)",
            "ESRS E1 (Climate)",
        ),
        system_boundary=ProductSystemBoundary.CRADLE_TO_GATE,
        allocation_method=AllocationMethod.ECONOMIC,
        functional_unit_required=False,  # entity-scope — not strictly required
        recycled_content_assumption=0.0,
        supplier_primary_data_share=0.0,
        pcr_reference=None,
        epd_reference=None,
        pefcr_id=None,
        pact_compatible=False,
        deprecation_notice_days=180,
    ),
}


#: Per-variant changelog (MP12). Populated with v0.1.0 baseline +
#: v0.2.0 promotion entries.
PRODUCT_LCA_CHANGELOG: Dict[str, Tuple[Dict[str, Any], ...]] = {
    "pas_2050": (
        {
            "version": "0.1.0",
            "date": "2026-03-15",
            "changes": [
                "Initial PAS 2050:2011 variant (MP8 scaffold)",
                "Cradle-to-grave boundary; biogenic CO2 excluded per §7.4",
            ],
            "impact": "none (new variant)",
            "migration_notes": "N/A — variant introduction.",
        },
        {
            "version": "0.2.0",
            "date": "2026-04-23",
            "changes": [
                "Added structured standards_alignment",
                "Scaffolded included / excluded activity categories",
                "cannot_resolve_action=RAISE_NO_SAFE_MATCH",
                "deprecation_notice_days=180",
            ],
            "impact": "selection gate now denies scope1/scope2/offsets",
            "migration_notes": (
                "Back-compatible. TODO(methodology-review) PCR / EPD "
                "references before certification."
            ),
        },
    ),
    "eu_pef": (
        {
            "version": "0.1.0",
            "date": "2026-03-15",
            "changes": [
                "Initial EU PEF (EF 3.1) variant (MP8 scaffold)",
                "16 impact-category support; climate-change indicator surfaced",
            ],
            "impact": "none (new variant)",
            "migration_notes": "N/A.",
        },
        {
            "version": "0.2.0",
            "date": "2026-04-23",
            "changes": [
                "Added structured standards_alignment + PEFCR routing field",
                "Scaffolded included / excluded activity categories",
                "cannot_resolve_action=RAISE_NO_SAFE_MATCH",
                "deprecation_notice_days=180",
            ],
            "impact": "selection gate now denies scope1/scope2/offsets",
            "migration_notes": (
                "TODO(methodology-review): per-category PEFCR routing map "
                "pending PEFCR publication."
            ),
        },
    ),
    "eu_oef": (
        {
            "version": "0.1.0",
            "date": "2026-03-15",
            "changes": [
                "Initial EU OEF variant (MP8 scaffold)",
                "Entity-level LCA across 16 impact categories",
            ],
            "impact": "none (new variant)",
            "migration_notes": "N/A.",
        },
        {
            "version": "0.2.0",
            "date": "2026-04-23",
            "changes": [
                "Added structured standards_alignment",
                "cannot_resolve_action=RAISE_NO_SAFE_MATCH",
                "deprecation_notice_days=180",
            ],
            "impact": "minor (entity-scope unchanged; adds carbon_offsets deny)",
            "migration_notes": "Back-compatible.",
        },
    ),
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


def get_product_lca_metadata(key: str) -> ProductCarbonPackMetadata:
    """Retrieve per-variant product-LCA metadata."""
    needle = str(key).strip().lower()
    meta = PRODUCT_LCA_VARIANT_METADATA.get(needle)
    if meta is None:
        raise KeyError(f"no product-LCA metadata for: {key!r}")
    return meta


def list_product_lca_variants() -> Dict[str, MethodPack]:
    """Return the full variant map (read-only copy)."""
    return dict(_VARIANTS)


__all__ = [
    "PAS_2050",
    "PEF",
    "OEF",
    "PRODUCT_LCA_VARIANT_METADATA",
    "PRODUCT_LCA_CHANGELOG",
    "get_product_lca_variant",
    "get_product_lca_metadata",
    "list_product_lca_variants",
]
