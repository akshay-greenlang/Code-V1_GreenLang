# -*- coding: utf-8 -*-
"""
ILO Convention 169 Ratification Data - AGENT-EUDR-021

Authoritative reference data for ILO Convention 169 (Indigenous and Tribal
Peoples Convention, 1989) ratification status. The convention has been
ratified by 24 countries as of 2026, including major EUDR commodity-
producing countries (Brazil, Colombia, Peru, Guatemala, Honduras,
Paraguay, Bolivia).

ILO 169 ratification is a key factor in determining FPIC legal
requirements and indigenous rights protection strength per country.
Countries that have ratified ILO 169 have legally binding obligations
to consult indigenous peoples on matters affecting them and to obtain
FPIC for relocation.

Data Sources:
    - ILO NORMLEX Information System (normlex.ilo.org)
    - Last verified: March 2026

Example:
    >>> from greenlang.agents.eudr.indigenous_rights_checker.reference_data.ilo_169_countries import (
    ...     ILO_169_COUNTRIES,
    ...     is_ilo_169_ratified,
    ... )
    >>> assert is_ilo_169_ratified("BR")
    >>> assert not is_ilo_169_ratified("US")
    >>> brazil = ILO_169_COUNTRIES["BR"]
    >>> print(brazil["ratification_date"])
    2002-07-25

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 Indigenous Rights Checker (GL-EUDR-IRC-021)
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# ILO Convention 169 Ratification Data
# 24 ratifying countries with ratification dates and EUDR commodity relevance
# ---------------------------------------------------------------------------

ILO_169_COUNTRIES: Dict[str, Dict[str, Any]] = {
    "AR": {
        "country_name": "Argentina",
        "ratification_date": "2000-07-03",
        "eudr_commodities": ["soya", "wood"],
        "indigenous_population_estimate": 955032,
        "key_indigenous_peoples": [
            "Mapuche", "Toba", "Guarani", "Wichi",
        ],
    },
    "BO": {
        "country_name": "Bolivia",
        "ratification_date": "1991-12-11",
        "eudr_commodities": ["soya", "wood", "coffee"],
        "indigenous_population_estimate": 4100000,
        "key_indigenous_peoples": [
            "Quechua", "Aymara", "Guarani", "Chiquitano",
        ],
    },
    "BR": {
        "country_name": "Brazil",
        "ratification_date": "2002-07-25",
        "eudr_commodities": ["cattle", "soya", "cocoa", "coffee", "wood"],
        "indigenous_population_estimate": 1693535,
        "key_indigenous_peoples": [
            "Yanomami", "Guarani", "Kayapo", "Tikuna", "Munduruku",
        ],
    },
    "CF": {
        "country_name": "Central African Republic",
        "ratification_date": "2010-08-30",
        "eudr_commodities": ["wood"],
        "indigenous_population_estimate": 60000,
        "key_indigenous_peoples": ["Aka", "Baka"],
    },
    "CL": {
        "country_name": "Chile",
        "ratification_date": "2008-09-15",
        "eudr_commodities": ["wood"],
        "indigenous_population_estimate": 2185792,
        "key_indigenous_peoples": ["Mapuche", "Aymara", "Rapa Nui"],
    },
    "CO": {
        "country_name": "Colombia",
        "ratification_date": "1991-03-07",
        "eudr_commodities": ["coffee", "palm_oil", "cocoa", "wood"],
        "indigenous_population_estimate": 1905617,
        "key_indigenous_peoples": [
            "Wayuu", "Nasa", "Embera", "Arhuaco", "Kogi",
        ],
    },
    "CR": {
        "country_name": "Costa Rica",
        "ratification_date": "1993-04-02",
        "eudr_commodities": ["coffee"],
        "indigenous_population_estimate": 104143,
        "key_indigenous_peoples": ["Bribri", "Cabecar", "Boruca"],
    },
    "DK": {
        "country_name": "Denmark",
        "ratification_date": "1996-02-22",
        "eudr_commodities": [],
        "indigenous_population_estimate": 50000,
        "key_indigenous_peoples": ["Inuit (Greenland)"],
    },
    "DO": {
        "country_name": "Dominica",
        "ratification_date": "2002-06-25",
        "eudr_commodities": [],
        "indigenous_population_estimate": 3000,
        "key_indigenous_peoples": ["Kalinago"],
    },
    "EC": {
        "country_name": "Ecuador",
        "ratification_date": "1998-05-15",
        "eudr_commodities": ["cocoa", "coffee", "palm_oil", "wood"],
        "indigenous_population_estimate": 1018176,
        "key_indigenous_peoples": [
            "Kichwa", "Shuar", "Waorani", "Achuar", "Cofan",
        ],
    },
    "FJ": {
        "country_name": "Fiji",
        "ratification_date": "1998-03-03",
        "eudr_commodities": [],
        "indigenous_population_estimate": 475739,
        "key_indigenous_peoples": ["iTaukei"],
    },
    "GT": {
        "country_name": "Guatemala",
        "ratification_date": "1996-06-05",
        "eudr_commodities": ["coffee", "palm_oil", "rubber"],
        "indigenous_population_estimate": 6500000,
        "key_indigenous_peoples": [
            "Maya Kiche", "Kaqchikel", "Mam", "Qeqchi",
        ],
    },
    "HN": {
        "country_name": "Honduras",
        "ratification_date": "1995-03-28",
        "eudr_commodities": ["coffee", "palm_oil", "wood"],
        "indigenous_population_estimate": 700000,
        "key_indigenous_peoples": [
            "Lenca", "Miskito", "Garifuna", "Maya Chorti",
        ],
    },
    "LU": {
        "country_name": "Luxembourg",
        "ratification_date": "2018-05-14",
        "eudr_commodities": [],
        "indigenous_population_estimate": 0,
        "key_indigenous_peoples": [],
    },
    "MX": {
        "country_name": "Mexico",
        "ratification_date": "1990-09-05",
        "eudr_commodities": ["coffee", "cattle", "soya", "wood"],
        "indigenous_population_estimate": 23200000,
        "key_indigenous_peoples": [
            "Nahua", "Maya", "Zapotec", "Mixtec", "Otomi",
        ],
    },
    "NP": {
        "country_name": "Nepal",
        "ratification_date": "2007-09-14",
        "eudr_commodities": [],
        "indigenous_population_estimate": 8500000,
        "key_indigenous_peoples": [
            "Tharu", "Tamang", "Newar", "Magar", "Gurung",
        ],
    },
    "NL": {
        "country_name": "Netherlands",
        "ratification_date": "1998-02-02",
        "eudr_commodities": [],
        "indigenous_population_estimate": 0,
        "key_indigenous_peoples": [],
    },
    "NI": {
        "country_name": "Nicaragua",
        "ratification_date": "2010-08-25",
        "eudr_commodities": ["coffee", "cattle"],
        "indigenous_population_estimate": 600000,
        "key_indigenous_peoples": [
            "Miskito", "Mayangna", "Rama", "Garifuna",
        ],
    },
    "NO": {
        "country_name": "Norway",
        "ratification_date": "1990-06-19",
        "eudr_commodities": [],
        "indigenous_population_estimate": 50000,
        "key_indigenous_peoples": ["Sami"],
    },
    "PY": {
        "country_name": "Paraguay",
        "ratification_date": "1993-08-10",
        "eudr_commodities": ["soya", "cattle", "wood"],
        "indigenous_population_estimate": 117150,
        "key_indigenous_peoples": [
            "Guarani", "Ayoreo", "Enxet", "Nivacle",
        ],
    },
    "PE": {
        "country_name": "Peru",
        "ratification_date": "1994-02-02",
        "eudr_commodities": ["coffee", "cocoa", "wood", "palm_oil"],
        "indigenous_population_estimate": 5900000,
        "key_indigenous_peoples": [
            "Quechua", "Aymara", "Ashaninka", "Awajun", "Shipibo",
        ],
    },
    "ES": {
        "country_name": "Spain",
        "ratification_date": "2007-02-15",
        "eudr_commodities": [],
        "indigenous_population_estimate": 0,
        "key_indigenous_peoples": [],
    },
    "VE": {
        "country_name": "Venezuela",
        "ratification_date": "2002-05-22",
        "eudr_commodities": ["cocoa", "coffee", "wood"],
        "indigenous_population_estimate": 724592,
        "key_indigenous_peoples": [
            "Wayuu", "Warao", "Pemon", "Yanomami", "Bari",
        ],
    },
    "DE": {
        "country_name": "Germany",
        "ratification_date": "2021-06-15",
        "eudr_commodities": [],
        "indigenous_population_estimate": 0,
        "key_indigenous_peoples": [],
    },
}

# ---------------------------------------------------------------------------
# Convenience set for fast lookups
# ---------------------------------------------------------------------------

ILO_169_COUNTRY_CODES: frozenset = frozenset(ILO_169_COUNTRIES.keys())


def is_ilo_169_ratified(country_code: str) -> bool:
    """Check if a country has ratified ILO Convention 169.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.

    Returns:
        True if the country has ratified ILO 169, False otherwise.

    Example:
        >>> is_ilo_169_ratified("BR")
        True
        >>> is_ilo_169_ratified("US")
        False
    """
    return country_code.upper() in ILO_169_COUNTRY_CODES


def get_ilo_169_data(country_code: str) -> Dict[str, Any]:
    """Get ILO 169 ratification data for a country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.

    Returns:
        Dictionary with ratification data, or empty dict if not ratified.

    Example:
        >>> data = get_ilo_169_data("BR")
        >>> data["ratification_date"]
        '2002-07-25'
    """
    return ILO_169_COUNTRIES.get(country_code.upper(), {})


def get_eudr_relevant_ratifiers() -> Dict[str, Dict[str, Any]]:
    """Get ILO 169 ratifiers with EUDR commodity relevance.

    Returns:
        Dictionary of countries that have ratified ILO 169 AND produce
        EUDR-regulated commodities.

    Example:
        >>> relevant = get_eudr_relevant_ratifiers()
        >>> assert "BR" in relevant
    """
    return {
        code: data
        for code, data in ILO_169_COUNTRIES.items()
        if data.get("eudr_commodities")
    }
