# -*- coding: utf-8 -*-
"""Product Carbon method pack (ISO 14067 + GHG Protocol Product Standard + PACT).

**Wave 4-G — v0.2 promotion.**

This module closes the template-section gaps flagged in
``docs/specs/method_pack_audit.md`` §6 for the product-carbon family:

- Structured ``standards_alignment`` (GHG Protocol Product Standard,
  ISO 14067:2018, PACT Pathfinder v2.0).
- Enum-gated ``system_boundary`` + ``allocation_method``.
- Required ``functional_unit`` sidecar spec (ISO 14067:2018 §6.4.3
  mandates a declared functional unit for every product LCA).
- Structured inclusion / exclusion activity-category allow / deny sets.
- ``recycled_content_assumption`` + ``supplier_primary_data_share``
  validation bands (0-1 inclusive).
- PCR reference + EPD reference registry hooks for third-party programmes.
- PACT Pathfinder data-object compatibility flag.
- Strict ``cannot_resolve_action = RAISE_NO_SAFE_MATCH`` +
  ``global_default_tier_allowed = False`` (no tier-7 fall-through).
- v0.1.0 changelog baseline entry per MP12.

Every populated numeric threshold or enum value that is not already
pinned to a published standard paragraph carries a
``TODO(methodology-review)`` comment so the methodology lead can approve
before the pack flips from ``preview`` to ``certified``.

Source citations
----------------
- GHG Protocol, *Product Life Cycle Accounting and Reporting Standard*,
  WRI/WBCSD, 2011 (Chapter 7 — boundary setting; Chapter 9 — allocation).
- ISO 14067:2018, *Greenhouse gases — Carbon footprint of products —
  Requirements and guidelines for quantification*.
- Partnership for Carbon Transparency (PACT), *Pathfinder Framework
  v2.0*, WBCSD, 2023.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from greenlang.data.canonical_v2 import (
    FactorFamily,
    FormulaType,
    MethodProfile,
)
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
from greenlang.factors.method_packs.registry import register_pack


# ---------------------------------------------------------------------------
# Product-carbon specific enums (v0.2)
# ---------------------------------------------------------------------------


class ProductSystemBoundary(str, Enum):
    """System boundary choices per GHG Protocol Product Standard Ch. 7."""

    CRADLE_TO_GATE = "cradle_to_gate"
    CRADLE_TO_GRAVE = "cradle_to_grave"
    GATE_TO_GATE = "gate_to_gate"
    CRADLE_TO_CRADLE = "cradle_to_cradle"


class AllocationMethod(str, Enum):
    """Allocation methods per GHG Protocol Product Standard Ch. 9 +
    ISO 14067:2018 §6.4.4.2."""

    MASS = "mass"
    ECONOMIC = "economic"
    PHYSICAL = "physical"
    SYSTEM_EXPANSION = "system_expansion"
    SUBSTITUTION = "substitution"


# ---------------------------------------------------------------------------
# Functional-unit spec (ISO 14067:2018 §6.4.3 — required per standard).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FunctionalUnitSpec:
    """Normalised description of the functional unit per ISO 14067 §6.4.3.

    Every product carbon footprint MUST declare a functional unit; this
    sidecar captures the three dimensions the standard requires
    (quantified performance, temporal duration, quality level).
    """

    description: str                                  # e.g. "1 tonne of ambient-grade steel"
    quantified_magnitude: float                       # e.g. 1.0
    unit_of_measure: str                              # e.g. "t"
    duration: Optional[str] = None                    # e.g. "service lifetime 50 yr"
    reference_flow_description: Optional[str] = None  # ISO 14067 §6.4.4
    quality_level: Optional[str] = None               # e.g. "EN 10025 S355"


# ---------------------------------------------------------------------------
# Included / excluded GHG Protocol Product activity categories.
#
# Source: GHG Protocol Product Standard §7.3.1 (scope of the inventory
# boundary — upstream life-cycle activities). Scope 1 direct emissions +
# Scope 2 purchased electricity at the REPORTING entity are explicitly
# excluded from a product LCA because they are already counted in the
# corporate inventory; the product pack pulls the CRADLE-to-GATE / GRAVE
# chain from suppliers instead.
# ---------------------------------------------------------------------------


PRODUCT_INCLUDED_ACTIVITY_CATEGORIES: FrozenSet[str] = frozenset({
    "purchased_goods",
    "capital_goods",
    "manufacturing",
    "processing_of_sold_products",
    "use_phase",
    "end_of_life_treatment",
    "transportation_upstream",
    "transportation_downstream",
    "packaging",
    "waste_generated",
    # PACT Pathfinder-mapped activities (v2.0 Annex B).
    "pact_cradle_to_gate_pcf",
    "pact_product_lifecycle_pcf",
})


PRODUCT_EXCLUDED_ACTIVITY_CATEGORIES: FrozenSet[str] = frozenset({
    # These belong in the corporate inventory packs, not product LCA.
    "scope1_direct_emissions",
    "scope2_electricity",
    # Offsets NEVER belong inside a gross product footprint.
    "carbon_offsets",
})


# ---------------------------------------------------------------------------
# Per-pack sidecar metadata (MP8 scaffold — carried alongside the frozen
# MethodPack because ``base.MethodPack`` is immutable and out of scope for
# extension this wave).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductCarbonPackMetadata:
    """Additional template-schema fields for a product-carbon pack.

    Stored in :data:`PRODUCT_CARBON_METADATA` keyed by pack id. Consumed
    by the /explain renderer, the coverage endpoint, and the conformance
    test harness. Not mutated at runtime.
    """

    pack_id: str
    standards_alignment: Tuple[str, ...]
    system_boundary: ProductSystemBoundary
    allocation_method: AllocationMethod
    functional_unit_required: bool = True
    recycled_content_assumption: float = 0.0          # 0.0-1.0
    supplier_primary_data_share: float = 0.0          # 0.0-1.0
    pcr_reference: Optional[str] = None               # Product Category Rules pointer
    epd_reference: Optional[str] = None               # EPD programme pointer
    pefcr_id: Optional[str] = None                    # PEF category rule
    pact_compatible: bool = False                     # PACT Pathfinder data-object compatible
    included_activity_categories: FrozenSet[str] = field(
        default_factory=lambda: PRODUCT_INCLUDED_ACTIVITY_CATEGORIES
    )
    excluded_activity_categories: FrozenSet[str] = field(
        default_factory=lambda: PRODUCT_EXCLUDED_ACTIVITY_CATEGORIES
    )
    deprecation_notice_days: int = 180


def validate_recycled_content(value: float) -> float:
    """Clamp + validate ``recycled_content`` assumption to [0, 1].

    Raises :class:`ValueError` when the value is outside the ISO 14067 §7.4
    admissible range. Returns the validated float.
    """
    if value is None:
        return 0.0
    v = float(value)
    if not (0.0 <= v <= 1.0):
        raise ValueError(
            "recycled_content_assumption must satisfy 0<=v<=1 (got %r)" % value
        )
    return v


def validate_primary_data_share(value: float) -> float:
    """Clamp + validate ``supplier_primary_data_share`` to [0, 1]."""
    if value is None:
        return 0.0
    v = float(value)
    if not (0.0 <= v <= 1.0):
        raise ValueError(
            "supplier_primary_data_share must satisfy 0<=v<=1 (got %r)" % value
        )
    return v


def validate_pact_payload(data_object: Optional[Dict[str, Any]]) -> bool:
    """Minimal PACT Pathfinder v2.0 data-object shape check.

    The full data-object schema lives in the PACT v2.0 JSON schema
    registry; here we verify the four required top-level keys so a
    misrouted payload fails fast. TODO(methodology-review): full schema
    validation pending the PACT v2.1 normative JSON-schema finalisation.
    """
    if data_object is None:
        return False
    if not isinstance(data_object, dict):
        return False
    required_keys = {"productId", "pcf", "productDescription", "specVersion"}
    return required_keys.issubset(data_object.keys())


# ---------------------------------------------------------------------------
# PRODUCT_CARBON — umbrella (ISO 14067 / GHG PS / PACT).
# ---------------------------------------------------------------------------


PRODUCT_CARBON = MethodPack(
    profile=MethodProfile.PRODUCT_CARBON,
    name="Product Carbon Footprint (ISO 14067 / GHG PS / PACT)",
    description=(
        "Product-level LCA aligned with ISO 14067:2018, GHG Protocol Product "
        "Standard, and the Partnership for Carbon Transparency (PACT) "
        "Pathfinder Framework v2.0 data-exchange schema. Supports cradle-to-"
        "gate and cradle-to-grave boundaries with explicit allocation + "
        "recycled-content flags. Strict cannot-resolve-safely contract: "
        "RAISE_NO_SAFE_MATCH (no tier-7 global-default fall-through)."
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
        included_activity_categories=PRODUCT_INCLUDED_ACTIVITY_CATEGORIES,
        excluded_activity_categories=PRODUCT_EXCLUDED_ACTIVITY_CATEGORIES,
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
    reporting_labels=(
        "ISO_14067",
        "GHG_Protocol_Product",
        "PACT",
        "PACT_Pathfinder_v2",
        "CSRD_E1",
    ),
    audit_text_template=(
        "Product carbon footprint per ISO 14067:2018. Product: {product_id}. "
        "Factor: {factor_id}. Boundary: {boundary}. Allocation: "
        "{allocation_method}. Recycled content: {recycled_content_pct}%. "
        "Primary data share: {supplier_primary_data_share_pct}%. PCR: "
        "{pcr_reference}. EPD: {epd_reference}."
    ),
    pack_version="0.2.0",
    tags=("product", "licensed", "pact", "iso_14067"),
    cannot_resolve_action=CannotResolveAction.RAISE_NO_SAFE_MATCH,
    global_default_tier_allowed=False,
    replacement_pack_id=None,
    deprecation_notice_days=180,
)


register_pack(PRODUCT_CARBON)


# ---------------------------------------------------------------------------
# v0.2 sidecar metadata map (MP8 scaffold).
# TODO(methodology-review): values below are scaffolds; the methodology
# lead must confirm each entry before the pack flips to `certified`.
# ---------------------------------------------------------------------------


PRODUCT_CARBON_METADATA: Dict[str, ProductCarbonPackMetadata] = {
    "product_carbon": ProductCarbonPackMetadata(
        pack_id="product_carbon",
        standards_alignment=(
            "GHG Protocol Product Standard 2011",
            "ISO 14067:2018",
            "PACT Pathfinder Framework v2.0",
        ),
        system_boundary=ProductSystemBoundary.CRADLE_TO_GATE,
        # Default allocation per ISO 14067 §6.4.4.2 priority: avoid
        # allocation (system_expansion) > physical > economic. We ship
        # SYSTEM_EXPANSION as default so callers explicitly override
        # when allocation is needed. TODO(methodology-review).
        allocation_method=AllocationMethod.SYSTEM_EXPANSION,
        functional_unit_required=True,
        recycled_content_assumption=0.0,
        supplier_primary_data_share=0.0,
        pcr_reference=None,            # TODO(methodology-review)
        epd_reference=None,            # TODO(methodology-review)
        pact_compatible=True,
        deprecation_notice_days=180,
    ),
}


#: Changelog per MP12. Populated with v0.1.0 baseline + v0.2.0 promotion.
PRODUCT_CARBON_CHANGELOG: Tuple[Dict[str, Any], ...] = (
    {
        "version": "0.1.0",
        "date": "2026-03-15",
        "changes": [
            "Initial Product Carbon method pack (MP8 scaffold)",
            "ISO 14067 / GHG Product Standard / PACT baseline selection + boundary rules",
        ],
        "impact": "none (new pack)",
        "migration_notes": "N/A — pack introduction.",
    },
    {
        "version": "0.2.0",
        "date": "2026-04-23",
        "changes": [
            "Added structured standards_alignment, system_boundary enum, allocation_method enum",
            "Required FunctionalUnitSpec sidecar per ISO 14067 §6.4.3",
            "Added PACT Pathfinder v2.0 payload validation helper",
            "Scaffolded included / excluded activity-category allow / deny sets",
            "cannot_resolve_action=RAISE_NO_SAFE_MATCH; global_default_tier_allowed=False",
            "deprecation_notice_days=180 + replacement_pack_id=None",
        ],
        "impact": "selection gate now explicitly denies scope1/scope2/offsets",
        "migration_notes": (
            "No behavioural change for existing records because the deny-list "
            "targets activity categories that never belonged in product LCA. "
            "Callers must declare a FunctionalUnitSpec in new integrations. "
            "TODO(methodology-review) before promoting to `certified`."
        ),
    },
)


__all__ = [
    "PRODUCT_CARBON",
    "PRODUCT_CARBON_METADATA",
    "PRODUCT_CARBON_CHANGELOG",
    # Enums
    "ProductSystemBoundary",
    "AllocationMethod",
    # Sidecars
    "FunctionalUnitSpec",
    "ProductCarbonPackMetadata",
    # Activity-category sets
    "PRODUCT_INCLUDED_ACTIVITY_CATEGORIES",
    "PRODUCT_EXCLUDED_ACTIVITY_CATEGORIES",
    # Validators
    "validate_recycled_content",
    "validate_primary_data_share",
    "validate_pact_payload",
]
