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


# ---------------------------------------------------------------------------
# Regulatory-gap closure (2026-04-22) — §4 item #1, five new markets
# ---------------------------------------------------------------------------
# The §4 §10.1 gap is that residual mix was only covered for EU/US/CA/AU/JP.
# The five variants below extend coverage to Canada (provincial CER),
# UK national (DESNZ), Australia state-level (NGER), Korea (KEMCO),
# Singapore (EMA). Each pack enforces a jurisdiction filter via the
# SelectionRule custom_filter so a caller asking for Singapore residual
# mix cannot accidentally receive a factor from a different source.


def _jurisdiction_filter(
    *, source_id: str, required_country: str
) -> callable:
    """Build a custom_filter that accepts only factors with the target
    ``source_id`` AND ``geography`` matching the country code.

    Using a module-level helper keeps the closure explicit so the
    filter is picklable (avoids lambda closure-capture surprises).
    """
    required_country = required_country.upper()

    def _filter(rec) -> bool:
        rec_source = getattr(rec, "source_id", None)
        rec_geo = str(getattr(rec, "geography", "") or "").upper()
        return rec_source == source_id and rec_geo == required_country

    _filter.__name__ = f"filter_{source_id}_{required_country}"
    return _filter


ELECTRICITY_RESIDUAL_MIX_CA = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
    name="Electricity — Residual Mix (Canada, CER provincial)",
    description=(
        "Provincial residual mix for Canada derived from Canada Energy "
        "Regulator (CER) provincial electricity intensity factors with "
        "ECCC National Inventory Report surrender-adjusted accounting. "
        "Covers all 10 provinces + 3 territories. Used for Scope 2 "
        "market-based accounting when no REC / I-REC / PPA applies."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.RESIDUAL_MIX,),
        allowed_formula_types=(FormulaType.RESIDUAL_MIX,),
        allowed_statuses=("certified", "preview"),
        custom_filter=_jurisdiction_filter(
            source_id="cer_canada_residual", required_country="CA"
        ),
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
        "IFRS_S2",
        "CSA_CSDS",              # Canada Sustainability Disclosure Standards
    ),
    audit_text_template=(
        "Canada residual mix ({geography}/{grid_region}) — CER provincial "
        "intensity {source_year}. Applied under GHG Protocol Scope 2 "
        "market-based when no contractual instrument exists."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.RESIDUAL_MIX,
    tags=("electricity", "residual_mix", "ca", "cer"),
)


ELECTRICITY_RESIDUAL_MIX_UK_NATIONAL = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
    name="Electricity — Residual Mix (UK national, DESNZ/BEIS)",
    description=(
        "UK national residual mix factor derived from the DESNZ "
        "(formerly BEIS) GHG conversion factors with Ofgem Renewable "
        "Energy Guarantees of Origin (REGO) surrender netting. Single "
        "national factor covering England, Scotland, Wales, Northern "
        "Ireland under UK SECR / TPT reporting."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.RESIDUAL_MIX,),
        allowed_formula_types=(FormulaType.RESIDUAL_MIX,),
        allowed_statuses=("certified", "preview"),
        custom_filter=_jurisdiction_filter(
            source_id="beis_uk_residual", required_country="UK"
        ),
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
        "UK_SECR",
        "UK_TPT",
        "CSRD_E1",
        "IFRS_S2",
    ),
    audit_text_template=(
        "UK national residual mix — DESNZ {source_year} with REGO "
        "surrender netting. Use for Scope 2 market-based when no "
        "contractual instrument (GO, REGO, PPA) applies."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.RESIDUAL_MIX,
    tags=("electricity", "residual_mix", "uk", "desnz", "rego"),
)


ELECTRICITY_RESIDUAL_MIX_AU_STATE = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
    name="Electricity — Residual Mix (Australia state-level, NGER)",
    description=(
        "Australia state-level residual mix derived from the NGER "
        "(National Greenhouse and Energy Reporting) state-level grid "
        "factors with Large-scale Generation Certificate (LGC) surrender "
        "netting. Covers all 8 states + territories (NSW, VIC, QLD, SA, "
        "WA, TAS, NT, ACT). Finer-grained than the existing Australian "
        "national pack; activated when the caller supplies state "
        "resolution."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.RESIDUAL_MIX,),
        allowed_formula_types=(FormulaType.RESIDUAL_MIX,),
        allowed_statuses=("certified", "preview"),
        custom_filter=_jurisdiction_filter(
            source_id="nger_au_state_residual", required_country="AU"
        ),
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
        "ASRS",                   # Australian Sustainability Reporting Standards
    ),
    audit_text_template=(
        "Australia state residual mix ({geography}/{grid_region}) — "
        "NGER FY{source_year} with LGC netting."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.RESIDUAL_MIX,
    tags=("electricity", "residual_mix", "au", "nger", "state_level"),
)


ELECTRICITY_RESIDUAL_MIX_KR = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
    name="Electricity — Residual Mix (Korea, KEMCO)",
    description=(
        "Republic of Korea national electricity residual mix derived "
        "from the Korea Energy Management Corporation (KEMCO) / Korea "
        "Energy Agency factors with KEC (Korea Energy Certificate) "
        "surrender netting. Published alongside the Greenhouse Gas "
        "Inventory and Research Center (GIR) national inventory."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.RESIDUAL_MIX,),
        allowed_formula_types=(FormulaType.RESIDUAL_MIX,),
        allowed_statuses=("certified", "preview"),
        custom_filter=_jurisdiction_filter(
            source_id="kemco_korea_residual", required_country="KR"
        ),
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
        "Korea_KSSB",             # Korea Sustainability Standards Board
        "IFRS_S2",
        "CDP",
    ),
    audit_text_template=(
        "Korea national residual mix — KEMCO {source_year} with KEC "
        "surrender netting."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.RESIDUAL_MIX,
    tags=("electricity", "residual_mix", "kr", "kemco", "kec"),
)


ELECTRICITY_RESIDUAL_MIX_SG = MethodPack(
    profile=MethodProfile.CORPORATE_SCOPE2_MARKET,
    name="Electricity — Residual Mix (Singapore, EMA)",
    description=(
        "Singapore national electricity residual mix derived from the "
        "Energy Market Authority (EMA) Grid Emission Factor with REC "
        "(Renewable Energy Certificate) surrender netting. Singapore has "
        "a single-zone grid so only a national factor applies. Used for "
        "SGX Rule 711A mandatory Scope 2 disclosures from 2028."
    ),
    selection_rule=SelectionRule(
        allowed_families=(FactorFamily.RESIDUAL_MIX,),
        allowed_formula_types=(FormulaType.RESIDUAL_MIX,),
        allowed_statuses=("certified", "preview"),
        custom_filter=_jurisdiction_filter(
            source_id="ema_singapore_residual", required_country="SG"
        ),
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
        "SGX_711A",               # Singapore Exchange Rule 711A
        "IFRS_S2",
        "CDP",
    ),
    audit_text_template=(
        "Singapore national residual mix — EMA {source_year} with REC "
        "surrender netting. Single-zone grid: {geography}."
    ),
    pack_version="1.0.0",
    electricity_basis=ElectricityBasis.RESIDUAL_MIX,
    tags=("electricity", "residual_mix", "sg", "ema"),
)


#: All residual-mix packs keyed by ISO-2 country code for routing.
RESIDUAL_MIX_PACKS_BY_COUNTRY: dict = {}


def _register_residual_mix_packs() -> None:
    """Build the country -> residual-mix pack routing map.

    EU/EEA countries all route to the AIB variant; US/CA, AU, JP each
    route to their dedicated pack.  The five 2026-Q2 additions (Canada
    provincial, UK national, AU state-level, Korea, Singapore) override
    the legacy routing where more specific data is available.
    """
    global RESIDUAL_MIX_PACKS_BY_COUNTRY
    eu_countries = (
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE",
        "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT",
        "RO", "SK", "SI", "ES", "SE", "IS", "LI", "NO",
    )
    mapping = {c: ELECTRICITY_RESIDUAL_MIX_EU for c in eu_countries}
    mapping["US"] = ELECTRICITY_RESIDUAL_MIX_US
    # Canada: CER provincial is more authoritative than the Green-e
    # backfill in the legacy pack — override the routing.
    mapping["CA"] = ELECTRICITY_RESIDUAL_MIX_CA
    # Australia national legacy pack retained for callers that only
    # request country-level resolution; state-level pack is opt-in.
    mapping["AU"] = ELECTRICITY_RESIDUAL_MIX_AU
    mapping["JP"] = ELECTRICITY_RESIDUAL_MIX_JP
    mapping["UK"] = ELECTRICITY_RESIDUAL_MIX_UK_NATIONAL
    mapping["GB"] = ELECTRICITY_RESIDUAL_MIX_UK_NATIONAL
    mapping["KR"] = ELECTRICITY_RESIDUAL_MIX_KR
    mapping["SG"] = ELECTRICITY_RESIDUAL_MIX_SG
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
        >>> get_residual_mix_pack("UK") is ELECTRICITY_RESIDUAL_MIX_UK_NATIONAL
        True
        >>> get_residual_mix_pack("SG") is ELECTRICITY_RESIDUAL_MIX_SG
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
    "ELECTRICITY_RESIDUAL_MIX_CA",
    "ELECTRICITY_RESIDUAL_MIX_UK_NATIONAL",
    "ELECTRICITY_RESIDUAL_MIX_AU_STATE",
    "ELECTRICITY_RESIDUAL_MIX_KR",
    "ELECTRICITY_RESIDUAL_MIX_SG",
    "RESIDUAL_MIX_FALLBACK",
    "RESIDUAL_MIX_PACKS_BY_COUNTRY",
    "get_residual_mix_pack",
]
