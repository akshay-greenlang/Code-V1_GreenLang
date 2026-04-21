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


# ---------------------------------------------------------------------------
# GAP-10 Wave 2 — regional residual-mix pack variants
# ---------------------------------------------------------------------------
# These packs wrap the shared market-based profile with region-specific
# selection and routing logic so callers can opt into Green-e / NGA / METI
# residual-mix data without the generic EU-first assumption baked into
# ``ELECTRICITY_MARKET``.  They are NOT registered via ``register_pack``
# because that registry is keyed on :class:`MethodProfile` and all four
# variants share the same Scope 2 market profile.  Downstream routing uses
# the ``tags`` field to disambiguate.

from greenlang.factors.method_packs.base import FallbackStep


#: 4-step residual-mix fallback: supplier -> subregion -> country -> global.
#: CTO non-negotiable: customer override always wins (rank 1).
RESIDUAL_MIX_FALLBACK = (
    FallbackStep(1, "customer_override", "Tenant-supplied factor overlay"),
    FallbackStep(2, "supplier_specific", "Supplier-disclosed residual mix"),
    FallbackStep(
        3, "subregion_residual",
        "Sub-national residual mix (NERC / NEM / METI utility)",
    ),
    FallbackStep(
        4, "country_residual",
        "Country-level residual mix (AIB national, NGA national, etc.)",
    ),
    FallbackStep(
        5, "global_residual",
        "Global residual-mix default (lowest quality, last resort)",
    ),
)


ELECTRICITY_RESIDUAL_MIX_EU = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
    name="Electricity — Residual Mix (EU/AIB)",
    description=(
        "AIB European Residual Mix for Scope 2 market-based accounting. "
        "Used when no contractual instrument (PPA, GO, REC) applies. "
        "Covers EU-27 + EEA."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.RESIDUAL_MIX,),
        allowed_formula_types=(FormulaType.RESIDUAL_MIX,),
        allowed_statuses=("certified", "preview"),
        custom_filter=lambda rec: getattr(rec, "source_id", None)
        == "aib_residual_mix_eu",
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("2",),
        allowed_boundaries=("combustion",),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.ALLOWED,
        include_transmission_losses=False,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=RESIDUAL_MIX_FALLBACK,
    deprecation=_DEPRECATION,
    reporting_labels=("GHG_Protocol_Scope2_MarketBased", "CSRD_E1"),
    audit_text_template=(
        "EU residual mix ({geography}) — AIB {source_year}. "
        "Apply when no contractual instrument exists."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.RESIDUAL_MIX,
    tags=("electricity", "residual_mix", "eu", "aib"),
)


ELECTRICITY_RESIDUAL_MIX_US = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
    name="Electricity — Residual Mix (US/Canada, Green-e)",
    description=(
        "Green-e Energy Residual Mix for US + Canada. Applied when no "
        "REC, I-REC, PPA or utility green tariff applies. Primary key: "
        "(country, NERC region) for US; (country, province) for Canada."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.RESIDUAL_MIX,),
        allowed_formula_types=(FormulaType.RESIDUAL_MIX,),
        allowed_statuses=("certified", "preview"),
        custom_filter=lambda rec: getattr(rec, "source_id", None)
        == "green_e_residual_mix",
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("2",),
        allowed_boundaries=("combustion",),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.ALLOWED,
        include_transmission_losses=False,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=RESIDUAL_MIX_FALLBACK,
    deprecation=_DEPRECATION,
    reporting_labels=(
        "GHG_Protocol_Scope2_MarketBased",
        "CDP",
        "SEC_Climate",
        "IFRS_S2",
    ),
    audit_text_template=(
        "North American residual mix ({geography}/{grid_region}) — "
        "Green-e {source_year}. Restricted licence: attribution required."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.RESIDUAL_MIX,
    tags=("electricity", "residual_mix", "us", "ca", "green_e"),
)


ELECTRICITY_RESIDUAL_MIX_AU = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
    name="Electricity — Residual Mix (Australia, NGA-derived)",
    description=(
        "Residual mix derived from the Australian NGA grid-average "
        "factors by netting out LGC-surrendered renewable energy. "
        "Covers NEM (NSW, QLD, SA, TAS, VIC) + WA + NT."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.RESIDUAL_MIX,),
        allowed_formula_types=(FormulaType.RESIDUAL_MIX,),
        allowed_statuses=("certified", "preview"),
        custom_filter=lambda rec: getattr(rec, "source_id", None)
        == "australia_nga_factors",
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("2",),
        allowed_boundaries=("combustion",),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.ALLOWED,
        include_transmission_losses=False,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=RESIDUAL_MIX_FALLBACK,
    deprecation=_DEPRECATION,
    reporting_labels=(
        "GHG_Protocol_Scope2_MarketBased",
        "NGER",
        "Climate_Active",
        "IFRS_S2",
    ),
    audit_text_template=(
        "Australian residual mix ({geography}/{grid_region}) — "
        "NGA FY{source_year} with LGC netting. See explainability "
        "for derivation."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.RESIDUAL_MIX,
    tags=("electricity", "residual_mix", "au", "nga", "lgc"),
)


ELECTRICITY_RESIDUAL_MIX_JP = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
    name="Electricity — Residual Mix (Japan, METI-derived)",
    description=(
        "Residual mix derived from METI Basic (基礎排出係数) factors by "
        "netting out non-fossil certificate (J-Credit, Green Value "
        "Certificate, Non-Fossil Value Trading) surrenders. Covers the "
        "10 general electricity utility service areas."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.RESIDUAL_MIX,),
        allowed_formula_types=(FormulaType.RESIDUAL_MIX,),
        allowed_statuses=("certified", "preview"),
        custom_filter=lambda rec: getattr(rec, "source_id", None)
        == "japan_meti_electric_emission_factors",
    ),
    boundary_rule=BoundaryRule(
        allowed_scopes=("2",),
        allowed_boundaries=("combustion",),
        biogenic_treatment=BiogenicTreatment.REPORTED_SEPARATELY,
        market_instruments=MarketInstrumentTreatment.ALLOWED,
        include_transmission_losses=False,
    ),
    gwp_basis="IPCC_AR6_100",
    region_hierarchy=RESIDUAL_MIX_FALLBACK,
    deprecation=_DEPRECATION,
    reporting_labels=(
        "GHG_Protocol_Scope2_MarketBased",
        "TCFD",
        "IFRS_S2",
        "Japan_SSBJ",
    ),
    audit_text_template=(
        "Japan residual mix ({geography}/{grid_region}) — "
        "METI FY{source_year} with non-fossil certificate netting."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.RESIDUAL_MIX,
    tags=("electricity", "residual_mix", "jp", "meti", "j_credit"),
)


#: All residual-mix packs keyed by ISO-2 country code for routing.
RESIDUAL_MIX_PACKS_BY_COUNTRY: dict = {}


def _register_residual_mix_packs() -> None:
    """Build the country -> residual-mix pack routing map.

    EU/EEA countries all route to the AIB variant; US/CA, AU, JP each
    route to their dedicated pack.
    """
    global RESIDUAL_MIX_PACKS_BY_COUNTRY
    eu_countries = (
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE",
        "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT",
        "RO", "SK", "SI", "ES", "SE", "IS", "LI", "NO",
    )
    mapping = {c: ELECTRICITY_RESIDUAL_MIX_EU for c in eu_countries}
    mapping["US"] = ELECTRICITY_RESIDUAL_MIX_US
    mapping["CA"] = ELECTRICITY_RESIDUAL_MIX_US
    mapping["AU"] = ELECTRICITY_RESIDUAL_MIX_AU
    mapping["JP"] = ELECTRICITY_RESIDUAL_MIX_JP
    RESIDUAL_MIX_PACKS_BY_COUNTRY = mapping


_register_residual_mix_packs()


def get_residual_mix_pack(country: str):
    """Return the residual-mix pack registered for ``country``.

    Falls back to ``ELECTRICITY_RESIDUAL_MIX_EU`` if the country is
    unknown — the EU pack is the most mature and has the broadest
    methodological fit for any AIB-member jurisdiction.

    Examples::

        >>> get_residual_mix_pack("US") is ELECTRICITY_RESIDUAL_MIX_US
        True
        >>> get_residual_mix_pack("JP") is ELECTRICITY_RESIDUAL_MIX_JP
        True
    """
    if not country:
        return ELECTRICITY_RESIDUAL_MIX_EU
    return RESIDUAL_MIX_PACKS_BY_COUNTRY.get(
        country.upper().strip(), ELECTRICITY_RESIDUAL_MIX_EU
    )


__all__ = [
    "ELECTRICITY_LOCATION",
    "ELECTRICITY_MARKET",
    "ELECTRICITY_RESIDUAL_MIX_EU",
    "ELECTRICITY_RESIDUAL_MIX_US",
    "ELECTRICITY_RESIDUAL_MIX_AU",
    "ELECTRICITY_RESIDUAL_MIX_JP",
    "RESIDUAL_MIX_FALLBACK",
    "RESIDUAL_MIX_PACKS_BY_COUNTRY",
    "get_residual_mix_pack",
]
