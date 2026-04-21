# -*- coding: utf-8 -*-
"""Electricity market taxonomy — supplier / certificate / balancing area.

Extended in GAP-10 Wave 2 with residual-mix region taxonomies for North
America (Green-e / NERC), Australia (NEM + WA + NT) and Japan (10 METI
utility service areas) so the Electricity method packs can route residual
mix requests across all 5 covered geographies.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from greenlang.factors.mapping.base import (
    MappingConfidence,
    MappingResult,
    normalize_text,
)

logger = logging.getLogger(__name__)


class ElectricityMarketCategory(str, Enum):
    GRID_AVERAGE = "grid_average"                  # location-based default
    GRID_SUBREGION = "grid_subregion"              # eGRID, CEA regional, AIB zone
    SUPPLIER_SPECIFIC = "supplier_specific"        # utility tariff
    PPA = "ppa"                                    # power purchase agreement
    REC = "rec"                                    # Renewable Energy Certificate
    GO = "go"                                      # Guarantee of Origin
    GREEN_TARIFF = "green_tariff"                  # utility green tariff product
    RESIDUAL_MIX = "residual_mix"                  # AIB residual-mix fallback
    ONSITE_GENERATION = "onsite_generation"        # customer-owned solar / CHP


_PATTERNS = {
    ElectricityMarketCategory.PPA: [
        "ppa", "power purchase agreement", "virtual ppa", "vppa", "physical ppa",
    ],
    ElectricityMarketCategory.REC: [
        "rec", "renewable energy certificate", "i-rec", "irec", "ac-rec",
    ],
    ElectricityMarketCategory.GO: [
        "go", "guarantee of origin", "guarantees of origin", "aib go",
    ],
    ElectricityMarketCategory.GREEN_TARIFF: [
        "green tariff", "utility green tariff", "renewable tariff",
    ],
    ElectricityMarketCategory.RESIDUAL_MIX: [
        "residual mix", "eu residual mix", "aib residual",
    ],
    ElectricityMarketCategory.ONSITE_GENERATION: [
        "onsite solar", "behind the meter", "behind-the-meter", "onsite chp",
        "on-site solar", "captive generation",
    ],
    ElectricityMarketCategory.SUPPLIER_SPECIFIC: [
        "utility bill", "utility tariff", "supplier contract", "retail electricity",
    ],
    ElectricityMarketCategory.GRID_SUBREGION: [
        "egrid subregion", "egrid region", "cea region", "aib zone",
    ],
    ElectricityMarketCategory.GRID_AVERAGE: [
        "grid average", "grid electricity", "mains electricity", "location based",
    ],
}


def map_electricity_market(description: str) -> MappingResult:
    """Route an electricity line item to a market-attribution category."""
    needle = normalize_text(description)
    if not needle:
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale="empty input",
            raw_input=description,
        )

    best: Optional[ElectricityMarketCategory] = None
    best_score = 0.0
    matched_pattern: Optional[str] = None
    # First, exact substring match ordered by category preference.
    for category, patterns in _PATTERNS.items():
        for pattern in patterns:
            if pattern in needle:
                score = min(1.0, len(pattern) / len(needle) + 0.5)
                if score > best_score:
                    best = category
                    best_score = score
                    matched_pattern = pattern

    if best is None:
        # Default to grid average if the caller just said "electricity".
        if "electricity" in needle or "power" in needle:
            return MappingResult(
                canonical={
                    "category": ElectricityMarketCategory.GRID_AVERAGE.value,
                    "electricity_basis": "location_based",
                    "requires_certificate": False,
                },
                confidence=0.5,
                band=MappingConfidence.LOW,
                rationale="Defaulted to grid_average (no market-basis indicator)",
                matched_pattern="electricity",
                raw_input=description,
            )
        return MappingResult(
            canonical=None,
            confidence=0.0,
            band=MappingConfidence.UNKNOWN,
            rationale=f"No electricity-market pattern in '{description}'",
            raw_input=description,
        )

    electricity_basis = {
        ElectricityMarketCategory.GRID_AVERAGE: "location_based",
        ElectricityMarketCategory.GRID_SUBREGION: "location_based",
        ElectricityMarketCategory.SUPPLIER_SPECIFIC: "market_based",
        ElectricityMarketCategory.PPA: "market_based",
        ElectricityMarketCategory.REC: "market_based",
        ElectricityMarketCategory.GO: "market_based",
        ElectricityMarketCategory.GREEN_TARIFF: "market_based",
        ElectricityMarketCategory.RESIDUAL_MIX: "residual_mix",
        ElectricityMarketCategory.ONSITE_GENERATION: "supplier_specific",
    }[best]

    canonical: Dict[str, Any] = {
        "category": best.value,
        "electricity_basis": electricity_basis,
        "requires_certificate": best in (
            ElectricityMarketCategory.PPA,
            ElectricityMarketCategory.REC,
            ElectricityMarketCategory.GO,
            ElectricityMarketCategory.GREEN_TARIFF,
        ),
    }

    return MappingResult(
        canonical=canonical,
        confidence=best_score,
        band=MappingConfidence.from_score(best_score),
        rationale=(
            f"Matched '{matched_pattern}' → {best.value} "
            f"(basis={electricity_basis})"
        ),
        matched_pattern=matched_pattern,
        raw_input=description,
    )


# ---------------------------------------------------------------------------
# GAP-10 Wave 2 — residual-mix region taxonomy
# ---------------------------------------------------------------------------


class USNERCRegion(str, Enum):
    """US NERC regions used as the primary Green-e residual-mix key."""

    ERCOT = "ERCOT"
    FRCC = "FRCC"
    MRO = "MRO"
    NPCC = "NPCC"
    RFC = "RFC"
    SERC = "SERC"
    SPP = "SPP"
    TRE = "TRE"
    WECC = "WECC"


#: NERC parent region -> typical eGRID subregion children.  This is a
#: *routing* map (fallback ordering) — not an exhaustive NERC taxonomy.
NERC_SUBREGION_MAP: Dict[str, Tuple[str, ...]] = {
    "ERCOT": ("ERCT",),
    "FRCC":  ("FRCC",),
    "MRO":   ("MROE", "MROW"),
    "NPCC":  ("NEWE", "NYCW", "NYLI", "NYUP"),
    "RFC":   ("RFCE", "RFCM", "RFCW"),
    "SERC":  ("SRMV", "SRMW", "SRSO", "SRTV", "SRVC"),
    "SPP":   ("SPNO", "SPSO"),
    "TRE":   ("ERCT",),
    "WECC":  ("AZNM", "CAMX", "NWPP", "RMPA"),
}

#: US states -> NERC parent region (abbreviated; covers primary assignments
#: only — border/overlap states resolve to the region that serves the bulk
#: of their load).
US_STATE_TO_NERC: Dict[str, str] = {
    # WECC
    "AK": "WECC", "AZ": "WECC", "CA": "WECC", "CO": "WECC", "ID": "WECC",
    "MT": "WECC", "NM": "WECC", "NV": "WECC", "OR": "WECC", "UT": "WECC",
    "WA": "WECC", "WY": "WECC",
    # SPP
    "KS": "SPP", "NE": "SPP", "OK": "SPP",
    # ERCOT (most of TX)
    "TX": "ERCOT",
    # MRO
    "IA": "MRO", "MN": "MRO", "ND": "MRO", "SD": "MRO",
    # RFC
    "DE": "RFC", "IN": "RFC", "KY": "RFC", "MD": "RFC", "MI": "RFC",
    "NJ": "RFC", "OH": "RFC", "PA": "RFC", "VA": "RFC", "WV": "RFC",
    "WI": "RFC",
    # SERC
    "AL": "SERC", "AR": "SERC", "GA": "SERC", "LA": "SERC", "MO": "SERC",
    "MS": "SERC", "NC": "SERC", "SC": "SERC", "TN": "SERC", "IL": "SERC",
    # FRCC
    "FL": "FRCC",
    # NPCC
    "CT": "NPCC", "MA": "NPCC", "ME": "NPCC", "NH": "NPCC", "NY": "NPCC",
    "RI": "NPCC", "VT": "NPCC",
    # Non-contiguous
    "HI": "WECC",
    "DC": "RFC",
}


class CanadianProvince(str, Enum):
    AB = "AB"; BC = "BC"; MB = "MB"; NB = "NB"; NL = "NL"; NS = "NS"
    NT = "NT"; NU = "NU"; ON = "ON"; PE = "PE"; QC = "QC"; SK = "SK"
    YT = "YT"


#: Canadian province -> CER grid region identifier (Canada Energy
#: Regulator groupings used for REC-equivalent netting).
CANADA_PROVINCE_GRID: Dict[str, str] = {
    "AB": "AB-GRID", "BC": "BC-GRID", "MB": "MB-GRID",
    "NB": "NB-GRID", "NL": "NL-GRID", "NS": "NS-GRID",
    "NT": "NT-GRID", "NU": "NU-GRID", "ON": "ON-GRID",
    "PE": "PE-GRID", "QC": "QC-GRID", "SK": "SK-GRID",
    "YT": "YT-GRID",
}


class AustraliaRegion(str, Enum):
    """Australian electricity market regions."""

    # NEM regions
    NSW = "NSW"
    QLD = "QLD"
    SA = "SA"
    TAS = "TAS"
    VIC = "VIC"
    # Non-NEM
    WA = "WA"
    NT = "NT"


NEM_REGIONS: Tuple[str, ...] = ("NSW", "QLD", "SA", "TAS", "VIC")
NON_NEM_REGIONS: Tuple[str, ...] = ("WA", "NT")


class JapanUtilityArea(str, Enum):
    """10 general electricity utility service areas in Japan."""

    HOKKAIDO = "Hokkaido"
    TOHOKU = "Tohoku"
    TOKYO = "Tokyo"
    CHUBU = "Chubu"
    HOKURIKU = "Hokuriku"
    KANSAI = "Kansai"
    CHUGOKU = "Chugoku"
    SHIKOKU = "Shikoku"
    KYUSHU = "Kyushu"
    OKINAWA = "Okinawa"


#: Market-based certificate type per country (used by pack routing).
_CERTIFICATE_BY_COUNTRY: Dict[str, str] = {
    "US": "REC",
    "CA": "REC",
    "GB": "REGO",
    "AU": "LGC",
    "NZ": "NZU",
    "JP": "J-Credit",
    "IN": "I-REC",
    "BR": "I-REC",
    "ZA": "I-REC",
    # EU-27 + EEA use GO via AIB
    **{c: "GO" for c in (
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE",
        "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT",
        "RO", "SK", "SI", "ES", "SE", "IS", "LI", "NO",
    )},
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_residual_mix_region(
    country: str, subregion: Optional[str] = None
) -> Optional[str]:
    """Resolve a residual-mix region key for a (country, subregion) pair.

    Returns:
        The most specific region identifier the registered residual-mix
        catalog understands, or ``None`` if no residual mix is available
        for the country.

    Examples::

        >>> get_residual_mix_region("US", "CA")
        'WECC'
        >>> get_residual_mix_region("AU", "NSW")
        'NSW'
        >>> get_residual_mix_region("JP", "tokyo")
        'Tokyo'
        >>> get_residual_mix_region("DE") is None
        False   # falls back to EU / AIB handling
    """
    if not country:
        return None
    country = country.upper().strip()
    sub = (subregion or "").strip()

    if country == "US":
        up = sub.upper()
        if not up:
            return None
        # Exact NERC region match?
        if up in {r.value for r in USNERCRegion}:
            return up
        # eGRID subregion -> parent NERC
        for nerc, children in NERC_SUBREGION_MAP.items():
            if up in children:
                return nerc
        # US state code?
        if up in US_STATE_TO_NERC:
            return US_STATE_TO_NERC[up]
        logger.debug("Unknown US residual-mix subregion %s", sub)
        return None

    if country == "CA":
        up = sub.upper()
        if up in {p.value for p in CanadianProvince}:
            return up
        return None

    if country == "AU":
        up = sub.upper()
        if up in {r.value for r in AustraliaRegion}:
            return up
        return None

    if country == "JP":
        # Accept any casing / lower-case Japanese utility name.
        if not sub:
            return None
        lower_map = {u.value.lower(): u.value for u in JapanUtilityArea}
        return lower_map.get(sub.lower())

    # Default: EU/EEA countries are handled by the AIB residual mix,
    # which is keyed on the ISO-2 country code itself.
    if country in _CERTIFICATE_BY_COUNTRY and _CERTIFICATE_BY_COUNTRY[country] == "GO":
        return country

    return None


def get_market_certificate_type(country: str) -> Optional[str]:
    """Return the predominant market-based certificate type for a country.

    Examples::

        >>> get_market_certificate_type("US")
        'REC'
        >>> get_market_certificate_type("AU")
        'LGC'
        >>> get_market_certificate_type("JP")
        'J-Credit'
        >>> get_market_certificate_type("DE")
        'GO'
    """
    if not country:
        return None
    return _CERTIFICATE_BY_COUNTRY.get(country.upper().strip())


def list_residual_mix_countries() -> List[str]:
    """Return ISO-2 codes for countries with registered residual-mix support."""
    eu = [c for c, cert in _CERTIFICATE_BY_COUNTRY.items() if cert == "GO"]
    return sorted(set(eu + ["US", "CA", "AU", "JP"]))


__all__ = [
    "AustraliaRegion",
    "CANADA_PROVINCE_GRID",
    "CanadianProvince",
    "ElectricityMarketCategory",
    "JapanUtilityArea",
    "NEM_REGIONS",
    "NERC_SUBREGION_MAP",
    "NON_NEM_REGIONS",
    "US_STATE_TO_NERC",
    "USNERCRegion",
    "get_market_certificate_type",
    "get_residual_mix_region",
    "list_residual_mix_countries",
    "map_electricity_market",
]
