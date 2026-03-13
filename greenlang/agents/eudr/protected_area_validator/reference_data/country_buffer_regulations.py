# -*- coding: utf-8 -*-
"""
Country Buffer Zone Regulations - AGENT-EUDR-022

National buffer zone requirements for EUDR commodity-producing countries.
Specifies legally mandated buffer distances around protected areas by
country and IUCN category.

PRD Reference: Feature 3 (F3.6) - national buffer zone regulations.

Author: GreenLang Platform Team
Date: March 2026
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Country buffer zone regulations
# ---------------------------------------------------------------------------

COUNTRY_BUFFER_REGULATIONS: Dict[str, Dict[str, Any]] = {
    "BRA": {
        "country": "Brazil",
        "legal_basis": "Lei 12.651/2012 (Forest Code) Art. 49",
        "default_buffer_km": Decimal("10"),
        "buffer_by_iucn": {
            "Ia": Decimal("10"),
            "Ib": Decimal("10"),
            "II": Decimal("10"),
            "III": Decimal("5"),
            "IV": Decimal("5"),
            "V": Decimal("3"),
            "VI": Decimal("3"),
            "NR": Decimal("10"),
        },
        "notes": (
            "Brazilian Forest Code (Art. 49) requires Environmental Protection "
            "Areas (Zonas de Amortecimento) around Conservation Units. "
            "Strict protection units (IUCN Ia/Ib/II) require 10 km buffer. "
            "Sustainable use units (IUCN V/VI) require 3 km buffer."
        ),
        "enforcement_level": "medium",
    },
    "IDN": {
        "country": "Indonesia",
        "legal_basis": "PP 28/2011 (Management of Nature Reserve and Conservation Areas)",
        "default_buffer_km": Decimal("5"),
        "buffer_by_iucn": {
            "Ia": Decimal("5"),
            "Ib": Decimal("5"),
            "II": Decimal("5"),
            "III": Decimal("3"),
            "IV": Decimal("3"),
            "V": Decimal("2"),
            "VI": Decimal("2"),
            "NR": Decimal("5"),
        },
        "notes": (
            "Government Regulation PP 28/2011 mandates buffer zones around "
            "conservation areas. Additional provincial regulations may apply."
        ),
        "enforcement_level": "medium",
    },
    "COD": {
        "country": "Democratic Republic of Congo",
        "legal_basis": "Loi 14/003 du 11 fevrier 2014",
        "default_buffer_km": Decimal("10"),
        "buffer_by_iucn": {
            "Ia": Decimal("25"),
            "Ib": Decimal("25"),
            "II": Decimal("10"),
            "III": Decimal("5"),
            "IV": Decimal("5"),
            "V": Decimal("3"),
            "VI": Decimal("3"),
            "NR": Decimal("10"),
        },
        "notes": (
            "DRC conservation law mandates buffer zones around national parks "
            "and reserves. World Heritage Sites (Virunga, Kahuzi-Biega, etc.) "
            "have expanded 25 km buffers."
        ),
        "enforcement_level": "low",
    },
    "COL": {
        "country": "Colombia",
        "legal_basis": "Decreto 2372 de 2010",
        "default_buffer_km": Decimal("5"),
        "buffer_by_iucn": {
            "Ia": Decimal("10"),
            "Ib": Decimal("10"),
            "II": Decimal("5"),
            "III": Decimal("5"),
            "IV": Decimal("3"),
            "V": Decimal("2"),
            "VI": Decimal("2"),
            "NR": Decimal("5"),
        },
        "notes": (
            "Decreto 2372/2010 establishes buffer zones (Zonas Amortiguadoras) "
            "around protected areas within SINAP."
        ),
        "enforcement_level": "medium",
    },
    "PER": {
        "country": "Peru",
        "legal_basis": "DS 038-2001-AG (Reglamento de la Ley de ANP)",
        "default_buffer_km": Decimal("5"),
        "buffer_by_iucn": {
            "Ia": Decimal("10"),
            "Ib": Decimal("10"),
            "II": Decimal("5"),
            "III": Decimal("5"),
            "IV": Decimal("3"),
            "V": Decimal("2"),
            "VI": Decimal("2"),
            "NR": Decimal("5"),
        },
        "notes": (
            "Peruvian protected area regulations establish Zonas de "
            "Amortiguamiento around Natural Protected Areas managed by SERNANP."
        ),
        "enforcement_level": "medium",
    },
    "CMR": {
        "country": "Cameroon",
        "legal_basis": "Loi 94/01 du 20 janvier 1994 (Forestry Law)",
        "default_buffer_km": Decimal("5"),
        "buffer_by_iucn": {
            "Ia": Decimal("10"),
            "Ib": Decimal("10"),
            "II": Decimal("5"),
            "III": Decimal("5"),
            "IV": Decimal("3"),
            "V": Decimal("2"),
            "VI": Decimal("2"),
            "NR": Decimal("5"),
        },
        "notes": "Cameroon forestry law establishes buffer zones around national parks.",
        "enforcement_level": "low",
    },
    "CIV": {
        "country": "Cote d'Ivoire",
        "legal_basis": "Loi 2002-102 relative aux parcs nationaux et reserves naturelles",
        "default_buffer_km": Decimal("5"),
        "buffer_by_iucn": {
            "Ia": Decimal("10"),
            "Ib": Decimal("10"),
            "II": Decimal("5"),
            "III": Decimal("5"),
            "IV": Decimal("3"),
            "V": Decimal("2"),
            "VI": Decimal("2"),
            "NR": Decimal("5"),
        },
        "notes": (
            "Ivorian law establishes peripheral zones around parks and reserves. "
            "Critical for cocoa supply chain compliance."
        ),
        "enforcement_level": "low",
    },
    "GHA": {
        "country": "Ghana",
        "legal_basis": "Wildlife Conservation Regulations 1971 (LI 685)",
        "default_buffer_km": Decimal("5"),
        "buffer_by_iucn": {
            "Ia": Decimal("10"),
            "Ib": Decimal("10"),
            "II": Decimal("5"),
            "III": Decimal("5"),
            "IV": Decimal("3"),
            "V": Decimal("2"),
            "VI": Decimal("2"),
            "NR": Decimal("5"),
        },
        "notes": "Ghana wildlife regulations establish buffer zones around forest reserves.",
        "enforcement_level": "medium",
    },
    "MYS": {
        "country": "Malaysia",
        "legal_basis": "National Forestry Act 1984 (Act 313)",
        "default_buffer_km": Decimal("3"),
        "buffer_by_iucn": {
            "Ia": Decimal("5"),
            "Ib": Decimal("5"),
            "II": Decimal("5"),
            "III": Decimal("3"),
            "IV": Decimal("3"),
            "V": Decimal("2"),
            "VI": Decimal("1"),
            "NR": Decimal("3"),
        },
        "notes": "Malaysian forestry act establishes buffer zones. State-level variations apply.",
        "enforcement_level": "medium",
    },
    "ECU": {
        "country": "Ecuador",
        "legal_basis": "Ley Forestal y de Conservacion (2004)",
        "default_buffer_km": Decimal("5"),
        "buffer_by_iucn": {
            "Ia": Decimal("10"),
            "Ib": Decimal("10"),
            "II": Decimal("5"),
            "III": Decimal("5"),
            "IV": Decimal("3"),
            "V": Decimal("2"),
            "VI": Decimal("2"),
            "NR": Decimal("5"),
        },
        "notes": "Ecuador conservation law establishes buffer zones around SNAP areas.",
        "enforcement_level": "medium",
    },
}

# ---------------------------------------------------------------------------
# Default buffer (when no country-specific regulation exists)
# ---------------------------------------------------------------------------

_DEFAULT_REGULATION: Dict[str, Any] = {
    "country": "Default (no country-specific regulation)",
    "legal_basis": "EUDR Best Practice (GreenLang default)",
    "default_buffer_km": Decimal("5"),
    "buffer_by_iucn": {
        "Ia": Decimal("10"),
        "Ib": Decimal("10"),
        "II": Decimal("5"),
        "III": Decimal("5"),
        "IV": Decimal("3"),
        "V": Decimal("2"),
        "VI": Decimal("1"),
        "NR": Decimal("5"),
    },
    "notes": "Default buffer distances when no national regulation applies.",
    "enforcement_level": "none",
}


def get_national_buffer_km(
    country_iso3: str,
    iucn_category: str = "NR",
) -> Decimal:
    """Get the nationally mandated buffer zone distance.

    Args:
        country_iso3: ISO 3166-1 alpha-3 country code.
        iucn_category: IUCN category of the protected area.

    Returns:
        Buffer distance in kilometers.

    Example:
        >>> get_national_buffer_km("BRA", "II")
        Decimal('10')
        >>> get_national_buffer_km("IDN", "IV")
        Decimal('3')
        >>> get_national_buffer_km("XXX", "II")
        Decimal('5')
    """
    country_upper = country_iso3.upper()
    regulation = COUNTRY_BUFFER_REGULATIONS.get(country_upper, _DEFAULT_REGULATION)
    buffer_by_iucn = regulation.get("buffer_by_iucn", {})
    return buffer_by_iucn.get(iucn_category, regulation["default_buffer_km"])


def get_country_regulation(country_iso3: str) -> Optional[Dict[str, Any]]:
    """Get the full regulation metadata for a country.

    Args:
        country_iso3: ISO 3166-1 alpha-3 country code.

    Returns:
        Regulation dictionary or None if no country-specific regulation.
    """
    return COUNTRY_BUFFER_REGULATIONS.get(country_iso3.upper())


def has_national_regulation(country_iso3: str) -> bool:
    """Check if a country has specific buffer zone regulations.

    Args:
        country_iso3: ISO 3166-1 alpha-3 country code.

    Returns:
        True if the country has national buffer regulations.
    """
    return country_iso3.upper() in COUNTRY_BUFFER_REGULATIONS
