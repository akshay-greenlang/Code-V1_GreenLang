# -*- coding: utf-8 -*-
"""Corporate Inventory method packs (GHG Protocol + IFRS S2)."""
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
    CannotResolveAction,
    DEFAULT_FALLBACK,
    DeprecationRule,
    MarketInstrumentTreatment,
    MethodPack,
    SelectionRule,
)
from greenlang.factors.method_packs.registry import register_pack

# Shared deprecation policy for corporate reporting packs — 4-year window
# captures the typical audit cycle + restatement horizon.
_DEPRECATION = DeprecationRule(max_age_days=365 * 4, grace_period_days=365)

# ---------------------------------------------------------------------------
# Structured activity-category allow / deny lists (SelectionRule additions).
#
# GHG Protocol Corporate Standard does not publish a regulation-grade
# allow-list the way CBAM does; the treatment is positive (Scope 1 = all
# direct fossil emissions). We therefore ship EMPTY inclusion sets which
# the SelectionRule interprets as "no restriction" while still reserving
# the field so CI / methodology review can populate it later. For Scope 3
# we encode the canonical 15 category tags so a caller looking for an
# out-of-scope category fails fast.  Every entry cites the underlying
# standard paragraph per docs/specs/method_pack_template.md §4.
# ---------------------------------------------------------------------------
_CORPORATE_SCOPE3_CATEGORIES: frozenset = frozenset(
    {
        # GHG Protocol Corporate Value Chain (Scope 3) Standard, Chapter 5.
        "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8",
        "3.9", "3.10", "3.11", "3.12", "3.13", "3.14", "3.15",
        # Free-text equivalents commonly seen on Excel imports.
        "purchased_goods_and_services",
        "capital_goods",
        "fuel_and_energy_related_activities",
        "upstream_transportation_and_distribution",
        "waste_generated_in_operations",
        "business_travel",
        "employee_commuting",
        "upstream_leased_assets",
        "downstream_transportation_and_distribution",
        "processing_of_sold_products",
        "use_of_sold_products",
        "end_of_life_treatment_of_sold_products",
        "downstream_leased_assets",
        "franchises",
        "investments",
    }
)


CORPORATE_SCOPE1 = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE1,
    name="Corporate Inventory — Scope 1 (GHG Protocol)",
    description=(
        "Direct emissions from sources owned or controlled by the reporting "
        "entity. Fossil CO2 only; biogenic CO2 reported separately per GHG "
        "Protocol Corporate Standard."
    ),
    selection_rule=SelectionRule(
        allowed_families=(
            FactorFamily.EMISSIONS,
            FactorFamily.HEATING_VALUE,
            FactorFamily.REFRIGERANT_GWP,
            FactorFamily.OXIDATION,
            FactorFamily.CARBON_CONTENT,
        ),
        allowed_formula_types=(FormulaType.DIRECT_FACTOR, FormulaType.COMBUSTION),
        allowed_statuses=("certified",),
        # No positive allow-list under GHG Protocol Corporate (all direct
        # fossil sources are admissible). Kept explicit so readers see the
        # §4 contract is wired, just unbounded.
        included_activity_categories=frozenset(),
        excluded_activity_categories=frozenset(),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("1",),
        allowed_boundaries=("combustion",),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.NOT_APPLICABLE,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=_DEPRECATION,
    reporting_labels=(
        "GHG_Protocol",
        "IFRS_S2",
        "ISO_14064",
        "CSRD_E1",       # EU CSRD ESRS E1 Climate
        "CA_SB253",      # California Climate Corporate Data Accountability Act
        "UK_SECR",       # UK Streamlined Energy & Carbon Reporting
        "India_BRSR",    # India Business Responsibility & Sustainability Report
        "TCFD",          # legacy TCFD disclosures (still referenced by many)
        "SBTi",          # Science Based Targets initiative
        "CDP",           # CDP climate questionnaire
    ),
    audit_text_template=(
        "Scope 1 direct emissions computed per GHG Protocol Corporate Standard. "
        "Factor: {factor_id} (source: {source_org}, vintage: {source_year}). "
        "GWP basis: {gwp_basis}. Biogenic CO2 reported separately."
    ),
    pack_version="1.0.0",
    tags=("open_core",),
    cannot_resolve_action=CannotResolveAction.RAISE_NO_SAFE_MATCH,
    global_default_tier_allowed=False,
    replacement_pack_id=None,
    deprecation_notice_days=365,
)

CORPORATE_SCOPE2_LOCATION = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_LOCATION,
    name="Corporate Inventory — Scope 2 Location-Based (GHG Protocol)",
    description=(
        "Purchased electricity, steam, heating, cooling using the location-"
        "based method. Grid-average factors; market instruments (RECs, GOs, "
        "PPAs) are explicitly disallowed under this method."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.GRID_INTENSITY, FactorFamily.ENERGY_CONVERSION),
        allowed_formula_types=(FormulaType.DIRECT_FACTOR,),
        allowed_statuses=("certified",),
        included_activity_categories=frozenset(),
        # GHG Protocol Scope 2 Guidance §5 excludes offsets from gross
        # inventories; we record it for auditor discoverability.
        excluded_activity_categories=frozenset({"carbon_offsets"}),
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
    reporting_labels=(
        "GHG_Protocol_Scope2",
        "IFRS_S2",
        "CSRD_E1",
        "CA_SB253",
        "UK_SECR",
        "India_BRSR",
        "TCFD",
        "SBTi",
        "CDP",
    ),
    audit_text_template=(
        "Scope 2 location-based emissions per GHG Protocol Scope 2 Guidance. "
        "Grid factor: {factor_id} ({geography}, {source_year}). RECs / GOs / "
        "PPAs are not applied under this method."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.LOCATION_BASED,
    tags=("open_core",),
    cannot_resolve_action=CannotResolveAction.RAISE_NO_SAFE_MATCH,
    global_default_tier_allowed=False,
    replacement_pack_id=None,
    deprecation_notice_days=365,
)

CORPORATE_SCOPE2_MARKET = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
    name="Corporate Inventory — Scope 2 Market-Based (GHG Protocol)",
    description=(
        "Purchased electricity using market-based method. Supplier-specific "
        "contracts, RECs, GOs, and residual-mix factors are allowed per GHG "
        "Protocol Scope 2 Guidance."
    ),
    selection_rule=SelectionRule(
        allowed_families=(
            FactorFamily.GRID_INTENSITY,
            FactorFamily.RESIDUAL_MIX,
            FactorFamily.ENERGY_CONVERSION,
        ),
        allowed_formula_types=(FormulaType.DIRECT_FACTOR, FormulaType.RESIDUAL_MIX),
        allowed_statuses=("certified", "preview"),
        included_activity_categories=frozenset(),
        # GHG Protocol Scope 2 Guidance §7 — carbon offsets MUST NOT
        # appear inside the market-based inventory line (they go to a
        # separate "net" block). Encoded here so an offset record leaking
        # in via a tenant overlay is rejected.
        excluded_activity_categories=frozenset({"carbon_offsets"}),
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
    reporting_labels=(
        "GHG_Protocol_Scope2",
        "IFRS_S2",
        "RE100",
        "CSRD_E1",
        "CA_SB253",
        "UK_SECR",
        "India_BRSR",
        "TCFD",
        "SBTi",
        "CDP",
    ),
    audit_text_template=(
        "Scope 2 market-based emissions per GHG Protocol. Supplier contract: "
        "{supplier}; certificate: {certificate}. When no contract applies, "
        "residual mix used: {residual_mix_factor_id}."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.MARKET_BASED,
    tags=("open_core",),
    cannot_resolve_action=CannotResolveAction.RAISE_NO_SAFE_MATCH,
    global_default_tier_allowed=False,
    replacement_pack_id=None,
    deprecation_notice_days=365,
)

CORPORATE_SCOPE3 = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE3,
    name="Corporate Inventory — Scope 3 (GHG Protocol 15 categories)",
    description=(
        "Value-chain emissions across 15 categories. Spend-based, average-"
        "data, supplier-specific, and hybrid methods allowed. Biogenic carbon "
        "reported separately. Category 11 'Use of sold products' draws on the "
        "per-record :class:`UsePhaseParameters` block (product_lifetime_years, "
        "use_phase_energy_kwh, use_phase_frequency_per_year, "
        "end_of_life_allocation_method) per GHG Protocol Scope 3 Technical "
        "Guidance §11.3."
    ),
    selection_rule=SelectionRule(
        allowed_families=(
            FactorFamily.EMISSIONS,
            FactorFamily.MATERIAL_EMBODIED,          # Cat 1, Cat 2, Cat 11 use-phase
            FactorFamily.TRANSPORT_LANE,
            FactorFamily.WASTE_TREATMENT,
            FactorFamily.FINANCE_PROXY,
            FactorFamily.ENERGY_CONVERSION,
            FactorFamily.GRID_INTENSITY,             # Cat 11 use-phase electricity draw
            FactorFamily.LAND_USE_REMOVALS,          # Cat 1 upstream land-use emissions
        ),
        allowed_formula_types=(
            FormulaType.DIRECT_FACTOR,
            FormulaType.SPEND_PROXY,
            FormulaType.LCA,
            FormulaType.TRANSPORT_CHAIN,
        ),
        allowed_statuses=("certified", "preview"),
        # GHG Protocol Scope 3 Standard Chapter 5 enumerates 15 canonical
        # categories. We admit any of them (plus the canonical string
        # slugs used by the Excel templates) and rely on downstream
        # boundary / formula-type filters to reject cross-scope leakage.
        included_activity_categories=_CORPORATE_SCOPE3_CATEGORIES,
        excluded_activity_categories=frozenset({"carbon_offsets"}),
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("3",),
        allowed_boundaries=("cradle_to_gate", "cradle_to_grave", "WTW", "WTT"),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.NOT_APPLICABLE,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=_DEPRECATION,
    reporting_labels=(
        "GHG_Protocol_Scope3",
        "IFRS_S2",
        "CSRD_E1",
        "CA_SB253",       # SB253 requires Scope 3 starting 2027
        "India_BRSR",     # BRSR Principle 6 climate touches Scope 3
        "SBTi_FLAG",      # SBTi Forest/Land/Agriculture — Cat 1 upstream
        "CDP",
        "PCAF",           # for Cat 15 financed emissions
    ),
    # The audit template renders a Cat 11-specific block when the
    # caller supplies ``cat11_use_phase`` truthy.  Resolution engine
    # populates the Cat 11 slot from ``record.use_phase`` whenever
    # ``scope3_category == "11"`` so the explainability carries the
    # product-lifetime + use-phase math back to the auditor.
    audit_text_template=(
        "Scope 3 Cat {scope3_category} per GHG Protocol Scope 3 Standard. "
        "Method: {calculation_method}. Factor: {factor_id}. Data quality: "
        "{dqs_score}/100. "
        "{cat11_use_phase_block}"
    ),
    pack_version="1.1.0",
    tags=("open_core",),
    cannot_resolve_action=CannotResolveAction.RAISE_NO_SAFE_MATCH,
    global_default_tier_allowed=False,
    replacement_pack_id=None,
    deprecation_notice_days=365,
)


# Register all four corporate packs on import.
for _pack in (
    CORPORATE_SCOPE1,
    CORPORATE_SCOPE2_LOCATION,
    CORPORATE_SCOPE2_MARKET,
    CORPORATE_SCOPE3,
):
    register_pack(_pack)


def render_cat11_use_phase_block(use_phase: object) -> str:
    """Render the Cat 11 use-phase sub-block for the Scope 3 audit template.

    Accepts a :class:`~greenlang.data.emission_factor_record.UsePhaseParameters`
    instance (or any object exposing the four product-lifetime attributes)
    and returns a human-readable string ready for substitution into
    ``CORPORATE_SCOPE3.audit_text_template`` via the ``cat11_use_phase_block``
    placeholder. Returns an empty string when ``use_phase`` is ``None`` so
    the template collapses cleanly for non-Cat-11 categories.
    """
    if use_phase is None:
        return ""
    lifetime = getattr(use_phase, "product_lifetime_years", None)
    energy = getattr(use_phase, "use_phase_energy_kwh", None)
    freq = getattr(use_phase, "use_phase_frequency_per_year", None)
    eol = getattr(use_phase, "end_of_life_allocation_method", None)
    parts = ["Cat 11 use-phase (GHG Protocol Scope 3 TG §11.3):"]
    if lifetime is not None:
        parts.append(f"lifetime={lifetime} yr")
    if energy is not None:
        parts.append(f"per-use={energy} kWh")
    if freq is not None:
        parts.append(f"freq={freq}/yr")
    if eol is not None:
        eol_value = getattr(eol, "value", eol)
        parts.append(f"EoL={eol_value}")
    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# MP3 scaffold — spec template additions (method_pack_template.md)
# ---------------------------------------------------------------------------
# The following block encodes the missing/partial template sections flagged
# in docs/specs/method_pack_audit.md for Corporate. Values live alongside the
# frozen MethodPack instances above so they can be consumed by the resolver
# and /explain without mutating the immutable base class.
#
# TODO(MP3): methodology review required - do not certify.

#: Default GWP override set allowed for Corporate packs without a methodology
#: review. AR5 retained because many legacy datasets + SBTi + CDP still
#: accept AR5 100-year values.
CORPORATE_GWP_ALLOWED_OVERRIDES: tuple = ("IPCC_AR5_100",)
CORPORATE_GWP_HORIZON_YEARS: int = 100
CORPORATE_GWP_METRIC: str = "GWP"

#: Activity-category allow/deny lists per pack id.
#: TODO(MP3): populate before certification. Empty list = no restriction.
CORPORATE_INCLUSION_ACTIVITIES: dict = {
    "corporate_scope1": [],  # TODO(MP3): methodology review required - do not certify
    "corporate_scope2_location": [],  # TODO(MP3): methodology review required - do not certify
    "corporate_scope2_market": [],  # TODO(MP3): methodology review required - do not certify
    "corporate_scope3": [],  # TODO(MP3): methodology review required - do not certify
}
CORPORATE_EXCLUSION_ACTIVITIES: dict = {
    "corporate_scope1": [],  # TODO(MP3): methodology review required - do not certify
    "corporate_scope2_location": [],  # TODO(MP3): methodology review required - do not certify
    "corporate_scope2_market": [],  # TODO(MP3): methodology review required - do not certify
    "corporate_scope3": [],  # TODO(MP3): methodology review required - do not certify
}
CORPORATE_EXCLUSION_GASES: dict = {
    # biogenic CO2 is reported separately, not excluded from the inventory;
    # kept empty by design. TODO(MP3): confirm with methodology lead.
    "corporate_scope1": [],
    "corporate_scope2_location": [],
    "corporate_scope2_market": [],
    "corporate_scope3": [],
}

#: Deprecation default — 180-day advance notice + replacement_pointer schema.
#: TODO(MP3): methodology review required - do not certify (replacement packs
#: are placeholders until v2 packs are authored).
CORPORATE_DEPRECATION_DEFAULTS: dict = {
    "advance_notice_days": 180,
    "replacement_pointer_schema": {
        "corporate_scope1": "corporate_scope1_v2",          # TODO(MP3)
        "corporate_scope2_location": "corporate_scope2_location_v2",  # TODO(MP3)
        "corporate_scope2_market": "corporate_scope2_market_v2",      # TODO(MP3)
        "corporate_scope3": "corporate_scope3_v2",          # TODO(MP3)
    },
    "webhook_fan_out": ("factors.deprecations", "factors.methodology"),
    "migration_notes": (
        "TODO(MP3): link to Methodology Review Board decision record."
    ),
}

#: Audit-text template file names (Jinja) under
#: ``greenlang/factors/method_packs/audit_texts/``. The resolver will render
#: these in /explain payloads as the MP3 scaffold graduates to v1.
CORPORATE_AUDIT_TEMPLATE_FILES: dict = {
    "corporate_scope1": "corporate.j2",
    "corporate_scope2_location": "corporate.j2",
    "corporate_scope2_market": "corporate.j2",
    "corporate_scope3": "corporate.j2",
}

#: Scaffolded BoundaryRule metadata that supplements the frozen BoundaryRule
#: on each pack. The frozen BoundaryRule does NOT carry system_boundary,
#: lca_mode, or functional_unit fields (base.py line 92-101), so we expose
#: them as a sidecar map keyed by pack id.
#:
#: TODO(MP3): methodology review required - do not certify. Sidecar is a
#: shim until base.BoundaryRule is extended in a later PR.
CORPORATE_BOUNDARY_METADATA: dict = {
    "corporate_scope1": {
        "system_boundary": "gate_to_gate",
        "lca_mode": "attributional",
        "functional_unit": None,      # not applicable for inventory packs
        "scope3_categories": (),
    },
    "corporate_scope2_location": {
        "system_boundary": "gate_to_gate",
        "lca_mode": "attributional",
        "functional_unit": "1 kWh delivered",
        "scope3_categories": (),
    },
    "corporate_scope2_market": {
        "system_boundary": "gate_to_gate",
        "lca_mode": "attributional",
        "functional_unit": "1 kWh delivered",
        "scope3_categories": (),
    },
    "corporate_scope3": {
        "system_boundary": "cradle_to_gate",
        "lca_mode": "attributional",
        "functional_unit": None,
        # Scope 3 Cat 1..15 (empty tuple = all 15 allowed).
        # TODO(MP3): confirm exclusion of Cat 15 if PCAF pack used instead.
        "scope3_categories": (),
    },
}


__all__ = [
    "CORPORATE_SCOPE1",
    "CORPORATE_SCOPE2_LOCATION",
    "CORPORATE_SCOPE2_MARKET",
    "CORPORATE_SCOPE3",
    "render_cat11_use_phase_block",
    # MP3 scaffold exports
    "CORPORATE_GWP_ALLOWED_OVERRIDES",
    "CORPORATE_GWP_HORIZON_YEARS",
    "CORPORATE_GWP_METRIC",
    "CORPORATE_INCLUSION_ACTIVITIES",
    "CORPORATE_EXCLUSION_ACTIVITIES",
    "CORPORATE_EXCLUSION_GASES",
    "CORPORATE_DEPRECATION_DEFAULTS",
    "CORPORATE_AUDIT_TEMPLATE_FILES",
    "CORPORATE_BOUNDARY_METADATA",
]
