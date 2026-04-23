# -*- coding: utf-8 -*-
"""EU Policy method packs — CBAM, DPP, Battery Regulation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

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
# CBAM CN-code allow-list — Annex I of Regulation (EU) 2023/956.
#
# The definitive-period CBAM covers six sectors, encoded below as CN-chapter /
# heading prefixes (the resolver admits factors whose ``activity_category``
# equals the sector slug OR whose ``cn_code`` starts with one of the prefixes).
# The prefix set below is verbatim from Annex I of EU 2023/956:
#
# * Cement                 — CN 2507, 2515, 2516, 2517, 2521, 2522, 2523
# * Iron & steel           — CN 7206-7229 (chapter 72) + CN 7301-7326
# * Aluminium              — CN 7601-7616
# * Fertilisers            — CN 2808, 2814, 2834, 2836, 3102, 3105 (nitrogen)
# * Hydrogen               — CN 2804 10 (heading 2804.10)
# * Electricity            — CN 2716
# ---------------------------------------------------------------------------
CBAM_CN_CODE_PREFIXES: frozenset = frozenset(
    {
        # Cement (Annex I section 1).
        "2507", "2515", "2516", "2517", "2521", "2522", "2523",
        # Iron & steel (Annex I section 2) — admit full chapter 72 plus
        # the specific 7301-7326 downstream headings.
        "72", "7301", "7302", "7303", "7304", "7305", "7306", "7307",
        "7308", "7309", "7310", "7311", "7312", "7313", "7314", "7315",
        "7316", "7317", "7318", "7319", "7320", "7321", "7322", "7323",
        "7324", "7325", "7326",
        # Aluminium (Annex I section 3).
        "7601", "7602", "7603", "7604", "7605", "7606", "7607", "7608",
        "7609", "7610", "7611", "7612", "7613", "7614", "7615", "7616",
        # Fertilisers (Annex I section 4) — nitrogenous only.
        "2808", "2814", "2834", "2836", "3102", "3105",
        # Hydrogen (Annex I section 6) — CN 2804 10.
        "280410",
        # Electricity (Annex I section 5).
        "2716",
    }
)

#: CBAM activity-category slugs matching the sectors above. The resolver
#: checks both the CN-code prefix set AND these slugs so callers can key
#: by sector name without having to know the CN digits.
CBAM_SECTOR_SLUGS: frozenset = frozenset(
    {
        "cement",
        "iron_and_steel",
        "steel",
        "aluminium",
        "aluminum",
        "fertilisers",
        "fertilizers",
        "hydrogen",
        "electricity",
    }
)

#: Combined inclusion set fed into SelectionRule.included_activity_categories.
CBAM_INCLUDED_CATEGORIES: frozenset = CBAM_CN_CODE_PREFIXES | CBAM_SECTOR_SLUGS

#: Hard-deny list. CBAM Annex II exempts specified third countries; the slug
#: is retained as a denylist entry so an accidental import of a non-CBAM
#: electricity factor cannot slip through. CBAM does NOT cover biogenic
#: emissions (Art. 7(2)) — encoded as excluded category as well.
CBAM_EXCLUDED_CATEGORIES: frozenset = frozenset(
    {
        "electricity_imports_non_cbam",   # Annex II exempt countries
        "biogenic_emissions",             # Art. 7(2): biogenic excluded
        "carbon_offsets",                 # not permitted in CBAM declaration
    }
)


#: EU Battery Regulation 2023/1542 Art. 7 — covered classes.
EU_BATTERY_INCLUDED_CATEGORIES: frozenset = frozenset(
    {
        "lmt_battery",                    # Art. 3(11) 0.025-2 kWh
        "industrial_stationary_battery",  # Art. 3(12) > 2 kWh
        "ev_battery",                     # Art. 3(14) propulsion > 2 kWh
    }
)
EU_BATTERY_EXCLUDED_CATEGORIES: frozenset = frozenset(
    {
        "portable_battery",               # Art. 3(9) — CFP not yet mandatory
        "carbon_offsets",
    }
)

#: EU DPP (ESPR) does not have a published allow-list (implementing acts
#: land per-product). Keep empty = no restriction until the Commission
#: publishes a delegated act.
EU_DPP_INCLUDED_CATEGORIES: frozenset = frozenset()
EU_DPP_EXCLUDED_CATEGORIES: frozenset = frozenset({"carbon_offsets"})


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
        included_activity_categories=CBAM_INCLUDED_CATEGORIES,
        excluded_activity_categories=CBAM_EXCLUDED_CATEGORIES,
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
    # CBAM Art. 4(2) MANDATES that every declaration either uses primary
    # operator data or documented EU default values with surcharge. Silent
    # fallback to a global default violates the regulation — keep strict.
    cannot_resolve_action=CannotResolveAction.RAISE_NO_SAFE_MATCH,
    global_default_tier_allowed=False,
    replacement_pack_id=None,
    deprecation_notice_days=180,
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
        included_activity_categories=EU_DPP_INCLUDED_CATEGORIES,
        excluded_activity_categories=EU_DPP_EXCLUDED_CATEGORIES,
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
    cannot_resolve_action=CannotResolveAction.RAISE_NO_SAFE_MATCH,
    global_default_tier_allowed=False,
    replacement_pack_id=None,
    deprecation_notice_days=180,
)


# ---------------------------------------------------------------------------
# EU Battery Regulation 2023/1542
# ---------------------------------------------------------------------------
# Regulation (EU) 2023/1542 of the European Parliament and of the Council of
# 12 July 2023 concerning batteries and waste batteries. Article 7 mandates
# carbon-footprint declarations for five battery classes on a phased
# enforcement schedule. The per-class obligations below are grounded in the
# regulation text + Annex II (CFP performance classes) + Annex III (waste
# battery categories).


class BatteryClass(str, Enum):
    """Battery categories defined in EU 2023/1542 Article 3(9)-(13).

    Each class has its own scope definition and (where specified) a
    carbon-footprint-declaration obligation under Article 7:

    * ``PORTABLE`` — portable battery (Art. 3(9)): <= 5 kg, sealed, not
      exclusively designed for industrial use. CFP declaration *not*
      mandated in 2023/1542 — Commission may extend via delegated act.
    * ``LMT`` — light means of transport (Art. 3(11)): battery capacity
      between 0.025 kWh (25 Wh) and 2 kWh inclusive, e-bikes / e-scooters
      / e-cargo-cycles. Carbon-footprint declaration mandatory from
      2028-08-18 per Article 7(1)(a) second indent.
    * ``INDUSTRIAL_STATIONARY`` — industrial battery (Art. 3(12)) with
      internal storage > 2 kWh designed for stationary / industrial use.
      CFP declaration mandatory 2024-02-18 (Article 7(1)(a) first indent)
      for > 2 kWh — phased against the level-based Annex II thresholds.
    * ``EV`` — electric-vehicle battery (Art. 3(14)): propulsion battery
      in an L / M / N category vehicle, > 2 kWh. CFP declaration
      mandatory from 2024-02-18 per Article 7(1)(a).
    * ``ELECTRIC_AVIATION`` — aviation propulsion battery. Not a named
      category in 2023/1542 but surfaces as ``electric_aviation`` in
      industry submissions; carbon-footprint thresholds follow the
      industrial-stationary / EV formulas, with declaration deferred
      pending Commission act.
    """

    PORTABLE = "portable"
    LMT = "lmt"
    INDUSTRIAL_STATIONARY = "industrial_stationary"
    EV = "ev"
    ELECTRIC_AVIATION = "electric_aviation"


@dataclass(frozen=True)
class BatteryClassThreshold:
    """Per-class carbon-footprint declaration scope per EU 2023/1542 Art 7.

    Fields derived directly from Regulation (EU) 2023/1542:

    * ``min_weight_kg`` / ``max_weight_kg`` — optional weight bounds
      (portable: <= 5 kg; others: no explicit weight cap).
    * ``min_energy_kwh`` / ``max_energy_kwh`` — Article 3 capacity
      definitions: LMT covers 0.025-2 kWh (inclusive); industrial and EV
      cover > 2 kWh.
    * ``cfp_declaration_required`` — whether Article 7(1) currently
      mandates a carbon-footprint declaration for this class.
    * ``enforcement_date_iso`` — first calendar date when placing a
      battery in this class on the EU market without a compliant CFP
      declaration is prohibited.
    * ``dpp_required`` — whether the Digital Product Passport carries
      the CFP (EV + LMT + industrial > 2 kWh: yes; portable: no in v1).
    """

    battery_class: BatteryClass
    min_weight_kg: Optional[float]
    max_weight_kg: Optional[float]
    min_energy_kwh: Optional[float]
    max_energy_kwh: Optional[float]
    cfp_declaration_required: bool
    enforcement_date_iso: Optional[str]
    dpp_required: bool
    legal_article: str


#: EU 2023/1542 class-threshold table. Dates per Article 7(1) indents.
EU_BATTERY_CLASS_THRESHOLDS: Tuple[BatteryClassThreshold, ...] = (
    BatteryClassThreshold(
        battery_class=BatteryClass.PORTABLE,
        min_weight_kg=None,
        max_weight_kg=5.0,
        min_energy_kwh=None,
        max_energy_kwh=None,
        cfp_declaration_required=False,
        enforcement_date_iso=None,
        dpp_required=False,
        legal_article="Regulation (EU) 2023/1542 Article 3(9) + Article 7 (reserved)",
    ),
    BatteryClassThreshold(
        battery_class=BatteryClass.LMT,
        min_weight_kg=None,
        max_weight_kg=None,
        min_energy_kwh=0.025,          # 25 Wh inclusive
        max_energy_kwh=2.0,            # inclusive
        cfp_declaration_required=True,
        enforcement_date_iso="2028-08-18",
        dpp_required=True,
        legal_article="Regulation (EU) 2023/1542 Article 3(11) + Article 7(1)(a) second indent",
    ),
    BatteryClassThreshold(
        battery_class=BatteryClass.INDUSTRIAL_STATIONARY,
        min_weight_kg=None,
        max_weight_kg=None,
        min_energy_kwh=2.0,            # > 2 kWh
        max_energy_kwh=None,
        cfp_declaration_required=True,
        enforcement_date_iso="2024-02-18",
        dpp_required=True,
        legal_article="Regulation (EU) 2023/1542 Article 3(12) + Article 7(1)(a) first indent",
    ),
    BatteryClassThreshold(
        battery_class=BatteryClass.EV,
        min_weight_kg=None,
        max_weight_kg=None,
        min_energy_kwh=2.0,            # > 2 kWh (typical EV pack is 40-100+ kWh)
        max_energy_kwh=None,
        cfp_declaration_required=True,
        enforcement_date_iso="2024-02-18",
        dpp_required=True,
        legal_article="Regulation (EU) 2023/1542 Article 3(14) + Article 7(1)(a)",
    ),
    BatteryClassThreshold(
        battery_class=BatteryClass.ELECTRIC_AVIATION,
        min_weight_kg=None,
        max_weight_kg=None,
        min_energy_kwh=2.0,
        max_energy_kwh=None,
        cfp_declaration_required=False,
        enforcement_date_iso=None,
        dpp_required=False,
        legal_article=(
            "Regulation (EU) 2023/1542 (declaration deferred pending "
            "Commission implementing act for aviation batteries)"
        ),
    ),
)


def classify_battery(weight_kg: Optional[float], energy_kwh: Optional[float]) -> BatteryClass:
    """Classify a battery into the EU 2023/1542 class using weight + energy.

    Raises ``ValueError`` when both ``weight_kg`` and ``energy_kwh`` are
    unknown (the regulation requires at least one dimension to class a
    battery). Follows the precedence: EV (if energy > 2 kWh AND caller
    tags mobile — default to industrial-stationary); portable (<= 5 kg
    and no energy flag); LMT (0.025-2 kWh inclusive); industrial (> 2
    kWh stationary).
    """
    if weight_kg is None and energy_kwh is None:
        raise ValueError(
            "classify_battery requires at least one of weight_kg / energy_kwh."
        )

    # LMT dominates when energy falls in the 0.025-2 kWh window (Art 3(11)).
    if energy_kwh is not None and 0.025 <= energy_kwh <= 2.0:
        return BatteryClass.LMT

    # Portable when small and no qualifying energy bucket (Art 3(9)).
    if weight_kg is not None and weight_kg <= 5.0 and (
        energy_kwh is None or energy_kwh < 0.025
    ):
        return BatteryClass.PORTABLE

    # > 2 kWh falls into industrial-stationary by default (Art 3(12)).
    if energy_kwh is not None and energy_kwh > 2.0:
        return BatteryClass.INDUSTRIAL_STATIONARY

    # Fallback: anything heavier than the 5 kg portable cap with no
    # energy disclosed is treated as industrial-stationary so the
    # declaration obligation is not silently bypassed.
    return BatteryClass.INDUSTRIAL_STATIONARY


def get_battery_threshold(battery_class: BatteryClass) -> BatteryClassThreshold:
    """Return the threshold row for a given class (raises if unknown)."""
    for row in EU_BATTERY_CLASS_THRESHOLDS:
        if row.battery_class is battery_class:
            return row
    raise ValueError(f"Unknown battery class: {battery_class}")


def cfp_declaration_required(
    weight_kg: Optional[float], energy_kwh: Optional[float]
) -> bool:
    """Return True iff EU 2023/1542 Art 7(1) requires a CFP declaration for this battery."""
    threshold = get_battery_threshold(classify_battery(weight_kg, energy_kwh))
    return threshold.cfp_declaration_required


EU_BATTERY = MethodPack(
    profile=MethodProfile.EU_DPP_BATTERY,
    name="EU Battery Regulation — Digital Product Passport (CFP declaration)",
    description=(
        "Carbon-footprint declaration under Regulation (EU) 2023/1542 "
        "Article 7 for LMT, industrial (> 2 kWh), and electric-vehicle "
        "battery classes. Delivers a DPP-compatible payload with per-kWh "
        "and per-unit carbon footprint, performance-class (Annex II), and "
        "supply-chain provenance back to active-material producers. Values "
        "must reflect the full life-cycle per the Delegated Regulation on "
        "harmonised CFP rules (Commission implementing act, in force from "
        "enforcement dates in Article 7(1))."
    ),
    selection_rule=SelectionRule(
        allowed_families=(
            FactorFamily.MATERIAL_EMBODIED,     # active materials, cell, pack
            FactorFamily.EMISSIONS,             # process emissions during cell manufacturing
            FactorFamily.GRID_INTENSITY,        # electricity for cell assembly
            FactorFamily.ENERGY_CONVERSION,     # energy carrier conversions
        ),
        allowed_formula_types=(FormulaType.LCA, FormulaType.DIRECT_FACTOR),
        # CFP declarations are public-regulatory filings: preview / draft
        # factors are prohibited.
        allowed_statuses=("certified",),
        require_verification=True,
        included_activity_categories=EU_BATTERY_INCLUDED_CATEGORIES,
        excluded_activity_categories=EU_BATTERY_EXCLUDED_CATEGORIES,
    ),
    boundary_rule=BoundaryRule(
        # Scope 1 + 2 direct facility emissions + Scope 3 cradle-to-gate
        # material inputs combined.
        allowed_scopes=("1", "2", "3"),
        allowed_boundaries=("cradle_to_gate", "cradle_to_grave"),
        # Biogenic tracked separately per harmonised CFP rules.
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.ALLOWED,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=DEFAULT_FALLBACK,
    deprecation=DeprecationRule(max_age_days=365 * 3, grace_period_days=180),
    reporting_labels=("EU_Battery_Regulation", "EU_DPP", "Article_7_CFP"),
    audit_text_template=(
        "EU Battery Regulation CFP for class {battery_class} (battery "
        "capacity {battery_energy_kwh} kWh, weight {battery_weight_kg} kg). "
        "Enforcement: Article 7(1) from {enforcement_date}. DPP entry: "
        "{dpp_id}. Factor: {factor_id} (source: {source_org} {source_year}, "
        "verification: {verification_status}). Functional unit: 1 kWh of "
        "energy delivered over service life."
    ),
    pack_version="1.0.0",
    # DPP Battery pack is a licensed premium SKU (regulated disclosure).
    tags=("eu_policy", "licensed", "battery", "dpp"),
    # EU 2023/1542 Art. 7 CFP declarations are regulated filings — a
    # silent global-default would produce a non-compliant declaration.
    cannot_resolve_action=CannotResolveAction.RAISE_NO_SAFE_MATCH,
    global_default_tier_allowed=False,
    replacement_pack_id=None,
    deprecation_notice_days=180,
)


for _pack in (EU_CBAM, EU_DPP, EU_BATTERY):
    register_pack(_pack)


# ---------------------------------------------------------------------------
# MP5 scaffold — spec template additions (method_pack_template.md)
# ---------------------------------------------------------------------------
# TODO(MP5): methodology review required - do not certify.
# Encodes the gaps flagged in docs/specs/method_pack_audit.md §4 without
# changing any numeric behaviour or altering selection thresholds.

from greenlang.factors.method_packs.base import FallbackStep as _FallbackStep


#: CBAM-specific fallback hierarchy. The default 7-tier chain allows a
#: global default (tier 7), which CBAM regulation forbids: declarants
#: must either use primary operator data or fall back to the published
#: EU default values (with surcharge implications). The chain below
#: reflects Art. 4(2) of Regulation (EU) 2023/956.
#:
#: TODO(MP5): wire into EU_CBAM.region_hierarchy on the next pack rev
#: (requires a new MethodPack instance since the dataclass is frozen).
CBAM_FALLBACK_HIERARCHY: tuple = (
    _FallbackStep(1, "customer_override", "Declarant-supplied factor overlay"),
    _FallbackStep(2, "operator_primary", "Primary data from third-country installation operator (Art. 4(1))"),
    _FallbackStep(3, "operator_secondary", "Secondary data from operator per default method (Annex IV)"),
    _FallbackStep(4, "eu_default_value", "EU default values per Art. 4(2) - surcharge applies"),
)

#: GWP basis: locked to AR6 100. CBAM implementing regulation references
#: AR6 directly, no override permitted without methodology review.
EU_CBAM_GWP_ALLOWED_OVERRIDES: tuple = ()
EU_CBAM_GWP_HORIZON_YEARS: int = 100
EU_CBAM_GWP_METRIC: str = "GWP"

#: CBAM CN-code allow-list per Annex I of Regulation (EU) 2023/956.
#: Structured activity-category encoding - LABELS ONLY at this scaffold stage.
#: TODO(MP5): populate with exact CN-8 codes; methodology review required.
EU_CBAM_INCLUSION_ACTIVITIES: list = [
    "cement",                 # Annex I section 1 - CN 2523
    "iron_and_steel",         # Annex I section 2 - CN 72, 7301-7326
    "aluminium",              # Annex I section 3 - CN 7601-7616
    "fertilisers",            # Annex I section 4 - CN 2808, 2814, 2834, 3102, 3105
    "electricity",            # Annex I section 5 - CN 2716
    "hydrogen",               # Annex I section 6 - CN 2804 10 000
    # TODO(MP5): add downstream extensions (organic chemicals, polymers)
    # pending the 2026-2030 scope expansion.
]
EU_CBAM_EXCLUSION_ACTIVITIES: list = [
    "electricity_imports_non_cbam",    # Annex II exempt third countries
    # TODO(MP5): methodology review required - do not certify
]

#: EU DPP (ESPR) inclusion/exclusion - implementing acts not yet in force
#: so lists are placeholder until product-specific DPP requirements land.
EU_DPP_INCLUSION_ACTIVITIES: list = []   # TODO(MP5): per-product DPP requirements
EU_DPP_EXCLUSION_ACTIVITIES: list = []   # TODO(MP5): methodology review required

#: EU Battery Regulation 2023/1542 inclusion/exclusion.
EU_BATTERY_INCLUSION_ACTIVITIES: list = [
    "lmt_battery",                      # Art. 3(11) - 0.025-2 kWh
    "industrial_stationary_battery",    # Art. 3(12) - >2 kWh
    "ev_battery",                       # Art. 3(14) - propulsion >2 kWh
    # TODO(MP5): add aviation battery once Commission implementing act lands
]
EU_BATTERY_EXCLUSION_ACTIVITIES: list = [
    "portable_battery",                 # Art. 3(9) - CFP not yet mandatory
    # TODO(MP5): methodology review required - do not certify
]

#: Allowed market instruments per pack id.
EU_POLICY_ALLOWED_INSTRUMENTS: dict = {
    "eu_cbam": ("operator_primary_data",),     # Art. 4(1) only
    "eu_dpp": (),                              # not applicable
    "eu_battery": ("rec", "i_rec", "go", "ppa_physical"),
}

#: Functional units (product-level packs only).
EU_POLICY_FUNCTIONAL_UNITS: dict = {
    "eu_cbam": "1 tonne of CN-coded CBAM good",
    "eu_dpp": "1 physical product unit",
    "eu_battery": "1 kWh delivered over service life",
}

#: Fallback defaults per pack.
EU_POLICY_FALLBACK_DEFAULTS: dict = {
    "eu_cbam": {
        "cannot_resolve_action": "raise_no_safe_match",
        "global_default_tier_allowed": False,
        "stale_factor_cutoff_days": 730,
    },
    "eu_dpp": {
        "cannot_resolve_action": "raise_no_safe_match",
        "global_default_tier_allowed": False,
        "stale_factor_cutoff_days": 1095,
    },
    "eu_battery": {
        "cannot_resolve_action": "raise_no_safe_match",
        "global_default_tier_allowed": False,
        "stale_factor_cutoff_days": 1095,
    },
}

#: Deprecation default — 180-day advance notice + replacement_pointer schema.
#: TODO(MP5): methodology review required - do not certify.
EU_POLICY_DEPRECATION_DEFAULTS: dict = {
    "advance_notice_days": 180,
    "replacement_pointer_schema": {
        "eu_cbam": "eu_cbam_v2",           # TODO(MP5)
        "eu_dpp": "eu_dpp_v2",             # TODO(MP5)
        "eu_battery": "eu_battery_v2",     # TODO(MP5)
    },
    "webhook_fan_out": (
        "factors.deprecations",
        "factors.methodology",
        "factors.eu_policy",
    ),
    "migration_notes": (
        "EU regulation cadence is tight. Replace packs when implementing "
        "acts / CBAM Annex III / Battery Delegated Regulation versions bump. "
        "TODO(MP5): link to Methodology Review Board decision record."
    ),
}

#: Audit-text template file names (Jinja).
EU_POLICY_AUDIT_TEMPLATE_FILES: dict = {
    "eu_cbam": "eu_policy.j2",
    "eu_dpp": "eu_policy.j2",
    "eu_battery": "eu_policy.j2",
}


__all__ = [
    "EU_CBAM",
    "EU_DPP",
    "EU_BATTERY",
    "BatteryClass",
    "BatteryClassThreshold",
    "EU_BATTERY_CLASS_THRESHOLDS",
    "classify_battery",
    "get_battery_threshold",
    "cfp_declaration_required",
    # MP5 scaffold exports
    "CBAM_FALLBACK_HIERARCHY",
    "EU_CBAM_GWP_ALLOWED_OVERRIDES",
    "EU_CBAM_GWP_HORIZON_YEARS",
    "EU_CBAM_GWP_METRIC",
    "EU_CBAM_INCLUSION_ACTIVITIES",
    "EU_CBAM_EXCLUSION_ACTIVITIES",
    "EU_DPP_INCLUSION_ACTIVITIES",
    "EU_DPP_EXCLUSION_ACTIVITIES",
    "EU_BATTERY_INCLUSION_ACTIVITIES",
    "EU_BATTERY_EXCLUSION_ACTIVITIES",
    "EU_POLICY_ALLOWED_INSTRUMENTS",
    "EU_POLICY_FUNCTIONAL_UNITS",
    "EU_POLICY_FALLBACK_DEFAULTS",
    "EU_POLICY_DEPRECATION_DEFAULTS",
    "EU_POLICY_AUDIT_TEMPLATE_FILES",
]
