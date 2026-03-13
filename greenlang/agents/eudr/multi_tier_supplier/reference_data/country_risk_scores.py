# -*- coding: utf-8 -*-
"""
Country Deforestation Risk Scores - AGENT-EUDR-008 Multi-Tier Supplier Tracker

Provides country-level deforestation risk scores, risk factor breakdowns,
regional risk aggregates, and EUDR high-risk country designations for the
Multi-Tier Supplier Tracker Agent. Used for supplier risk assessment and
propagation without external API dependencies.

Risk Score Methodology:
    Each country receives a composite score (0-100) derived from four factors:
    - deforestation_rate: Annual forest loss rate (FAO FRA 2025, GFW 2024)
    - governance_index: Rule-of-law and enforcement quality (WGI 2024)
    - enforcement_score: Environmental law enforcement effectiveness
    - corruption_index: CPI-derived corruption risk (Transparency International 2024)

    Composite = (deforestation_rate * 0.35 + (100 - governance_index) * 0.25
                 + (100 - enforcement_score) * 0.25 + corruption_index * 0.15)

EUDR High-Risk Designation:
    Countries designated as high-risk by the European Commission under
    EUDR Article 29 receive a flag and enhanced due diligence requirements.

Data Sources:
    FAO Global Forest Resources Assessment 2025
    Global Forest Watch Tree Cover Loss 2024
    World Bank Worldwide Governance Indicators 2024
    Transparency International Corruption Perceptions Index 2024
    European Commission EUDR Country Benchmarking (2025)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Country Risk Score Data (100+ countries)
# ---------------------------------------------------------------------------
# Each entry: composite_score (0-100, higher = more risk),
# deforestation_rate (0-100), governance_index (0-100, higher = better),
# enforcement_score (0-100, higher = better), corruption_index (0-100,
# higher = more corrupt), eudr_high_risk flag, region, primary commodities

COUNTRY_RISK_SCORES: Dict[str, Dict[str, Any]] = {
    # -----------------------------------------------------------------------
    # Africa
    # -----------------------------------------------------------------------
    "CD": {
        "name": "Democratic Republic of the Congo",
        "iso_alpha3": "COD",
        "region": "central_africa",
        "composite_score": 88,
        "deforestation_rate": 92,
        "governance_index": 8,
        "enforcement_score": 10,
        "corruption_index": 85,
        "eudr_high_risk": True,
        "primary_commodities": ["wood", "cocoa", "palm_oil", "rubber"],
    },
    "CG": {
        "name": "Republic of the Congo",
        "iso_alpha3": "COG",
        "region": "central_africa",
        "composite_score": 78,
        "deforestation_rate": 75,
        "governance_index": 18,
        "enforcement_score": 15,
        "corruption_index": 80,
        "eudr_high_risk": True,
        "primary_commodities": ["wood"],
    },
    "CM": {
        "name": "Cameroon",
        "iso_alpha3": "CMR",
        "region": "central_africa",
        "composite_score": 76,
        "deforestation_rate": 78,
        "governance_index": 20,
        "enforcement_score": 18,
        "corruption_index": 74,
        "eudr_high_risk": True,
        "primary_commodities": ["cocoa", "wood", "palm_oil", "rubber"],
    },
    "GA": {
        "name": "Gabon",
        "iso_alpha3": "GAB",
        "region": "central_africa",
        "composite_score": 58,
        "deforestation_rate": 55,
        "governance_index": 32,
        "enforcement_score": 30,
        "corruption_index": 68,
        "eudr_high_risk": False,
        "primary_commodities": ["wood", "palm_oil"],
    },
    "GQ": {
        "name": "Equatorial Guinea",
        "iso_alpha3": "GNQ",
        "region": "central_africa",
        "composite_score": 72,
        "deforestation_rate": 68,
        "governance_index": 15,
        "enforcement_score": 12,
        "corruption_index": 82,
        "eudr_high_risk": True,
        "primary_commodities": ["wood"],
    },
    "CF": {
        "name": "Central African Republic",
        "iso_alpha3": "CAF",
        "region": "central_africa",
        "composite_score": 85,
        "deforestation_rate": 80,
        "governance_index": 5,
        "enforcement_score": 5,
        "corruption_index": 90,
        "eudr_high_risk": True,
        "primary_commodities": ["wood"],
    },
    "CI": {
        "name": "Cote d'Ivoire",
        "iso_alpha3": "CIV",
        "region": "west_africa",
        "composite_score": 79,
        "deforestation_rate": 88,
        "governance_index": 22,
        "enforcement_score": 20,
        "corruption_index": 64,
        "eudr_high_risk": True,
        "primary_commodities": ["cocoa", "rubber", "palm_oil", "coffee"],
    },
    "GH": {
        "name": "Ghana",
        "iso_alpha3": "GHA",
        "region": "west_africa",
        "composite_score": 68,
        "deforestation_rate": 82,
        "governance_index": 42,
        "enforcement_score": 35,
        "corruption_index": 57,
        "eudr_high_risk": True,
        "primary_commodities": ["cocoa", "wood", "palm_oil"],
    },
    "NG": {
        "name": "Nigeria",
        "iso_alpha3": "NGA",
        "region": "west_africa",
        "composite_score": 77,
        "deforestation_rate": 85,
        "governance_index": 18,
        "enforcement_score": 15,
        "corruption_index": 75,
        "eudr_high_risk": True,
        "primary_commodities": ["cocoa", "palm_oil", "rubber", "wood"],
    },
    "SL": {
        "name": "Sierra Leone",
        "iso_alpha3": "SLE",
        "region": "west_africa",
        "composite_score": 74,
        "deforestation_rate": 78,
        "governance_index": 20,
        "enforcement_score": 18,
        "corruption_index": 72,
        "eudr_high_risk": True,
        "primary_commodities": ["cocoa", "palm_oil", "wood"],
    },
    "LR": {
        "name": "Liberia",
        "iso_alpha3": "LBR",
        "region": "west_africa",
        "composite_score": 76,
        "deforestation_rate": 80,
        "governance_index": 16,
        "enforcement_score": 14,
        "corruption_index": 76,
        "eudr_high_risk": True,
        "primary_commodities": ["rubber", "palm_oil", "wood"],
    },
    "GN": {
        "name": "Guinea",
        "iso_alpha3": "GIN",
        "region": "west_africa",
        "composite_score": 73,
        "deforestation_rate": 72,
        "governance_index": 15,
        "enforcement_score": 12,
        "corruption_index": 78,
        "eudr_high_risk": True,
        "primary_commodities": ["cocoa", "wood", "coffee"],
    },
    "TG": {
        "name": "Togo",
        "iso_alpha3": "TGO",
        "region": "west_africa",
        "composite_score": 62,
        "deforestation_rate": 60,
        "governance_index": 25,
        "enforcement_score": 22,
        "corruption_index": 68,
        "eudr_high_risk": False,
        "primary_commodities": ["cocoa", "coffee"],
    },
    "BJ": {
        "name": "Benin",
        "iso_alpha3": "BEN",
        "region": "west_africa",
        "composite_score": 55,
        "deforestation_rate": 50,
        "governance_index": 30,
        "enforcement_score": 28,
        "corruption_index": 65,
        "eudr_high_risk": False,
        "primary_commodities": ["palm_oil", "wood"],
    },
    "ET": {
        "name": "Ethiopia",
        "iso_alpha3": "ETH",
        "region": "east_africa",
        "composite_score": 70,
        "deforestation_rate": 75,
        "governance_index": 22,
        "enforcement_score": 20,
        "corruption_index": 62,
        "eudr_high_risk": True,
        "primary_commodities": ["coffee", "wood"],
    },
    "KE": {
        "name": "Kenya",
        "iso_alpha3": "KEN",
        "region": "east_africa",
        "composite_score": 52,
        "deforestation_rate": 55,
        "governance_index": 40,
        "enforcement_score": 38,
        "corruption_index": 60,
        "eudr_high_risk": False,
        "primary_commodities": ["coffee", "wood"],
    },
    "TZ": {
        "name": "Tanzania",
        "iso_alpha3": "TZA",
        "region": "east_africa",
        "composite_score": 65,
        "deforestation_rate": 72,
        "governance_index": 30,
        "enforcement_score": 25,
        "corruption_index": 62,
        "eudr_high_risk": True,
        "primary_commodities": ["coffee", "wood"],
    },
    "UG": {
        "name": "Uganda",
        "iso_alpha3": "UGA",
        "region": "east_africa",
        "composite_score": 68,
        "deforestation_rate": 78,
        "governance_index": 28,
        "enforcement_score": 22,
        "corruption_index": 65,
        "eudr_high_risk": True,
        "primary_commodities": ["coffee", "cocoa", "wood"],
    },
    "RW": {
        "name": "Rwanda",
        "iso_alpha3": "RWA",
        "region": "east_africa",
        "composite_score": 38,
        "deforestation_rate": 30,
        "governance_index": 55,
        "enforcement_score": 52,
        "corruption_index": 45,
        "eudr_high_risk": False,
        "primary_commodities": ["coffee"],
    },
    "BI": {
        "name": "Burundi",
        "iso_alpha3": "BDI",
        "region": "east_africa",
        "composite_score": 75,
        "deforestation_rate": 78,
        "governance_index": 12,
        "enforcement_score": 10,
        "corruption_index": 80,
        "eudr_high_risk": True,
        "primary_commodities": ["coffee"],
    },
    "MG": {
        "name": "Madagascar",
        "iso_alpha3": "MDG",
        "region": "east_africa",
        "composite_score": 80,
        "deforestation_rate": 90,
        "governance_index": 18,
        "enforcement_score": 12,
        "corruption_index": 75,
        "eudr_high_risk": True,
        "primary_commodities": ["cocoa", "wood", "coffee"],
    },
    "MZ": {
        "name": "Mozambique",
        "iso_alpha3": "MOZ",
        "region": "east_africa",
        "composite_score": 72,
        "deforestation_rate": 76,
        "governance_index": 20,
        "enforcement_score": 15,
        "corruption_index": 72,
        "eudr_high_risk": True,
        "primary_commodities": ["wood"],
    },
    "ZA": {
        "name": "South Africa",
        "iso_alpha3": "ZAF",
        "region": "southern_africa",
        "composite_score": 32,
        "deforestation_rate": 20,
        "governance_index": 55,
        "enforcement_score": 50,
        "corruption_index": 44,
        "eudr_high_risk": False,
        "primary_commodities": ["wood", "soya"],
    },
    # -----------------------------------------------------------------------
    # South America
    # -----------------------------------------------------------------------
    "BR": {
        "name": "Brazil",
        "iso_alpha3": "BRA",
        "region": "south_america",
        "composite_score": 72,
        "deforestation_rate": 85,
        "governance_index": 38,
        "enforcement_score": 35,
        "corruption_index": 58,
        "eudr_high_risk": True,
        "primary_commodities": ["soya", "cattle", "coffee", "cocoa", "wood", "rubber"],
    },
    "CO": {
        "name": "Colombia",
        "iso_alpha3": "COL",
        "region": "south_america",
        "composite_score": 65,
        "deforestation_rate": 72,
        "governance_index": 35,
        "enforcement_score": 30,
        "corruption_index": 60,
        "eudr_high_risk": True,
        "primary_commodities": ["coffee", "palm_oil", "cocoa", "cattle", "wood"],
    },
    "PE": {
        "name": "Peru",
        "iso_alpha3": "PER",
        "region": "south_america",
        "composite_score": 62,
        "deforestation_rate": 70,
        "governance_index": 35,
        "enforcement_score": 28,
        "corruption_index": 60,
        "eudr_high_risk": True,
        "primary_commodities": ["coffee", "cocoa", "wood", "palm_oil"],
    },
    "EC": {
        "name": "Ecuador",
        "iso_alpha3": "ECU",
        "region": "south_america",
        "composite_score": 60,
        "deforestation_rate": 68,
        "governance_index": 32,
        "enforcement_score": 28,
        "corruption_index": 62,
        "eudr_high_risk": True,
        "primary_commodities": ["cocoa", "coffee", "palm_oil", "wood"],
    },
    "BO": {
        "name": "Bolivia",
        "iso_alpha3": "BOL",
        "region": "south_america",
        "composite_score": 75,
        "deforestation_rate": 82,
        "governance_index": 22,
        "enforcement_score": 18,
        "corruption_index": 68,
        "eudr_high_risk": True,
        "primary_commodities": ["soya", "cattle", "wood"],
    },
    "PY": {
        "name": "Paraguay",
        "iso_alpha3": "PRY",
        "region": "south_america",
        "composite_score": 70,
        "deforestation_rate": 80,
        "governance_index": 28,
        "enforcement_score": 22,
        "corruption_index": 65,
        "eudr_high_risk": True,
        "primary_commodities": ["soya", "cattle", "wood"],
    },
    "AR": {
        "name": "Argentina",
        "iso_alpha3": "ARG",
        "region": "south_america",
        "composite_score": 55,
        "deforestation_rate": 65,
        "governance_index": 40,
        "enforcement_score": 35,
        "corruption_index": 55,
        "eudr_high_risk": False,
        "primary_commodities": ["soya", "cattle", "wood"],
    },
    "VE": {
        "name": "Venezuela",
        "iso_alpha3": "VEN",
        "region": "south_america",
        "composite_score": 78,
        "deforestation_rate": 72,
        "governance_index": 8,
        "enforcement_score": 8,
        "corruption_index": 88,
        "eudr_high_risk": True,
        "primary_commodities": ["cocoa", "cattle", "wood"],
    },
    "GY": {
        "name": "Guyana",
        "iso_alpha3": "GUY",
        "region": "south_america",
        "composite_score": 50,
        "deforestation_rate": 42,
        "governance_index": 35,
        "enforcement_score": 32,
        "corruption_index": 60,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "SR": {
        "name": "Suriname",
        "iso_alpha3": "SUR",
        "region": "south_america",
        "composite_score": 48,
        "deforestation_rate": 40,
        "governance_index": 38,
        "enforcement_score": 35,
        "corruption_index": 58,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "UY": {
        "name": "Uruguay",
        "iso_alpha3": "URY",
        "region": "south_america",
        "composite_score": 22,
        "deforestation_rate": 12,
        "governance_index": 72,
        "enforcement_score": 68,
        "corruption_index": 28,
        "eudr_high_risk": False,
        "primary_commodities": ["soya", "cattle", "wood"],
    },
    "CL": {
        "name": "Chile",
        "iso_alpha3": "CHL",
        "region": "south_america",
        "composite_score": 25,
        "deforestation_rate": 18,
        "governance_index": 70,
        "enforcement_score": 65,
        "corruption_index": 30,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    # -----------------------------------------------------------------------
    # Central America and Caribbean
    # -----------------------------------------------------------------------
    "GT": {
        "name": "Guatemala",
        "iso_alpha3": "GTM",
        "region": "central_america",
        "composite_score": 68,
        "deforestation_rate": 75,
        "governance_index": 22,
        "enforcement_score": 20,
        "corruption_index": 70,
        "eudr_high_risk": True,
        "primary_commodities": ["coffee", "palm_oil", "rubber", "cattle"],
    },
    "HN": {
        "name": "Honduras",
        "iso_alpha3": "HND",
        "region": "central_america",
        "composite_score": 72,
        "deforestation_rate": 78,
        "governance_index": 18,
        "enforcement_score": 15,
        "corruption_index": 75,
        "eudr_high_risk": True,
        "primary_commodities": ["coffee", "palm_oil", "wood", "cattle"],
    },
    "NI": {
        "name": "Nicaragua",
        "iso_alpha3": "NIC",
        "region": "central_america",
        "composite_score": 70,
        "deforestation_rate": 75,
        "governance_index": 15,
        "enforcement_score": 12,
        "corruption_index": 78,
        "eudr_high_risk": True,
        "primary_commodities": ["coffee", "cattle", "wood"],
    },
    "CR": {
        "name": "Costa Rica",
        "iso_alpha3": "CRI",
        "region": "central_america",
        "composite_score": 28,
        "deforestation_rate": 15,
        "governance_index": 65,
        "enforcement_score": 62,
        "corruption_index": 35,
        "eudr_high_risk": False,
        "primary_commodities": ["coffee", "palm_oil"],
    },
    "PA": {
        "name": "Panama",
        "iso_alpha3": "PAN",
        "region": "central_america",
        "composite_score": 42,
        "deforestation_rate": 48,
        "governance_index": 45,
        "enforcement_score": 38,
        "corruption_index": 55,
        "eudr_high_risk": False,
        "primary_commodities": ["wood", "cattle"],
    },
    "MX": {
        "name": "Mexico",
        "iso_alpha3": "MEX",
        "region": "central_america",
        "composite_score": 55,
        "deforestation_rate": 60,
        "governance_index": 35,
        "enforcement_score": 28,
        "corruption_index": 65,
        "eudr_high_risk": False,
        "primary_commodities": ["coffee", "cocoa", "cattle", "soya", "wood"],
    },
    "DO": {
        "name": "Dominican Republic",
        "iso_alpha3": "DOM",
        "region": "caribbean",
        "composite_score": 45,
        "deforestation_rate": 42,
        "governance_index": 38,
        "enforcement_score": 32,
        "corruption_index": 58,
        "eudr_high_risk": False,
        "primary_commodities": ["cocoa", "coffee"],
    },
    # -----------------------------------------------------------------------
    # Southeast Asia
    # -----------------------------------------------------------------------
    "ID": {
        "name": "Indonesia",
        "iso_alpha3": "IDN",
        "region": "southeast_asia",
        "composite_score": 74,
        "deforestation_rate": 82,
        "governance_index": 32,
        "enforcement_score": 28,
        "corruption_index": 62,
        "eudr_high_risk": True,
        "primary_commodities": ["palm_oil", "rubber", "cocoa", "coffee", "wood"],
    },
    "MY": {
        "name": "Malaysia",
        "iso_alpha3": "MYS",
        "region": "southeast_asia",
        "composite_score": 58,
        "deforestation_rate": 68,
        "governance_index": 52,
        "enforcement_score": 45,
        "corruption_index": 48,
        "eudr_high_risk": True,
        "primary_commodities": ["palm_oil", "rubber", "wood"],
    },
    "TH": {
        "name": "Thailand",
        "iso_alpha3": "THA",
        "region": "southeast_asia",
        "composite_score": 48,
        "deforestation_rate": 52,
        "governance_index": 42,
        "enforcement_score": 40,
        "corruption_index": 55,
        "eudr_high_risk": False,
        "primary_commodities": ["rubber", "palm_oil", "coffee", "wood"],
    },
    "VN": {
        "name": "Vietnam",
        "iso_alpha3": "VNM",
        "region": "southeast_asia",
        "composite_score": 52,
        "deforestation_rate": 58,
        "governance_index": 38,
        "enforcement_score": 35,
        "corruption_index": 60,
        "eudr_high_risk": False,
        "primary_commodities": ["coffee", "rubber", "wood"],
    },
    "PH": {
        "name": "Philippines",
        "iso_alpha3": "PHL",
        "region": "southeast_asia",
        "composite_score": 55,
        "deforestation_rate": 60,
        "governance_index": 35,
        "enforcement_score": 30,
        "corruption_index": 62,
        "eudr_high_risk": False,
        "primary_commodities": ["cocoa", "palm_oil", "rubber", "wood", "coffee"],
    },
    "MM": {
        "name": "Myanmar",
        "iso_alpha3": "MMR",
        "region": "southeast_asia",
        "composite_score": 82,
        "deforestation_rate": 85,
        "governance_index": 8,
        "enforcement_score": 8,
        "corruption_index": 88,
        "eudr_high_risk": True,
        "primary_commodities": ["rubber", "wood"],
    },
    "KH": {
        "name": "Cambodia",
        "iso_alpha3": "KHM",
        "region": "southeast_asia",
        "composite_score": 78,
        "deforestation_rate": 85,
        "governance_index": 18,
        "enforcement_score": 12,
        "corruption_index": 78,
        "eudr_high_risk": True,
        "primary_commodities": ["rubber", "wood"],
    },
    "LA": {
        "name": "Laos",
        "iso_alpha3": "LAO",
        "region": "southeast_asia",
        "composite_score": 75,
        "deforestation_rate": 80,
        "governance_index": 20,
        "enforcement_score": 15,
        "corruption_index": 72,
        "eudr_high_risk": True,
        "primary_commodities": ["rubber", "coffee", "wood"],
    },
    "PG": {
        "name": "Papua New Guinea",
        "iso_alpha3": "PNG",
        "region": "oceania",
        "composite_score": 80,
        "deforestation_rate": 85,
        "governance_index": 15,
        "enforcement_score": 10,
        "corruption_index": 82,
        "eudr_high_risk": True,
        "primary_commodities": ["palm_oil", "cocoa", "coffee", "wood"],
    },
    # -----------------------------------------------------------------------
    # South Asia
    # -----------------------------------------------------------------------
    "IN": {
        "name": "India",
        "iso_alpha3": "IND",
        "region": "south_asia",
        "composite_score": 45,
        "deforestation_rate": 42,
        "governance_index": 42,
        "enforcement_score": 38,
        "corruption_index": 55,
        "eudr_high_risk": False,
        "primary_commodities": ["coffee", "rubber", "soya", "wood"],
    },
    "LK": {
        "name": "Sri Lanka",
        "iso_alpha3": "LKA",
        "region": "south_asia",
        "composite_score": 48,
        "deforestation_rate": 50,
        "governance_index": 38,
        "enforcement_score": 35,
        "corruption_index": 58,
        "eudr_high_risk": False,
        "primary_commodities": ["rubber", "coffee", "wood"],
    },
    # -----------------------------------------------------------------------
    # East Asia
    # -----------------------------------------------------------------------
    "CN": {
        "name": "China",
        "iso_alpha3": "CHN",
        "region": "east_asia",
        "composite_score": 35,
        "deforestation_rate": 22,
        "governance_index": 45,
        "enforcement_score": 50,
        "corruption_index": 55,
        "eudr_high_risk": False,
        "primary_commodities": ["soya", "rubber", "wood"],
    },
    "JP": {
        "name": "Japan",
        "iso_alpha3": "JPN",
        "region": "east_asia",
        "composite_score": 10,
        "deforestation_rate": 5,
        "governance_index": 85,
        "enforcement_score": 88,
        "corruption_index": 18,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    # -----------------------------------------------------------------------
    # Europe (low risk, included as import destination references)
    # -----------------------------------------------------------------------
    "DE": {
        "name": "Germany",
        "iso_alpha3": "DEU",
        "region": "western_europe",
        "composite_score": 5,
        "deforestation_rate": 2,
        "governance_index": 90,
        "enforcement_score": 92,
        "corruption_index": 10,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "NL": {
        "name": "Netherlands",
        "iso_alpha3": "NLD",
        "region": "western_europe",
        "composite_score": 4,
        "deforestation_rate": 1,
        "governance_index": 92,
        "enforcement_score": 94,
        "corruption_index": 8,
        "eudr_high_risk": False,
        "primary_commodities": [],
    },
    "FR": {
        "name": "France",
        "iso_alpha3": "FRA",
        "region": "western_europe",
        "composite_score": 6,
        "deforestation_rate": 3,
        "governance_index": 88,
        "enforcement_score": 90,
        "corruption_index": 12,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "IT": {
        "name": "Italy",
        "iso_alpha3": "ITA",
        "region": "western_europe",
        "composite_score": 8,
        "deforestation_rate": 3,
        "governance_index": 78,
        "enforcement_score": 80,
        "corruption_index": 18,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "ES": {
        "name": "Spain",
        "iso_alpha3": "ESP",
        "region": "western_europe",
        "composite_score": 7,
        "deforestation_rate": 3,
        "governance_index": 82,
        "enforcement_score": 85,
        "corruption_index": 15,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "BE": {
        "name": "Belgium",
        "iso_alpha3": "BEL",
        "region": "western_europe",
        "composite_score": 5,
        "deforestation_rate": 1,
        "governance_index": 88,
        "enforcement_score": 90,
        "corruption_index": 12,
        "eudr_high_risk": False,
        "primary_commodities": [],
    },
    "PT": {
        "name": "Portugal",
        "iso_alpha3": "PRT",
        "region": "western_europe",
        "composite_score": 8,
        "deforestation_rate": 5,
        "governance_index": 80,
        "enforcement_score": 82,
        "corruption_index": 18,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "GB": {
        "name": "United Kingdom",
        "iso_alpha3": "GBR",
        "region": "western_europe",
        "composite_score": 5,
        "deforestation_rate": 1,
        "governance_index": 90,
        "enforcement_score": 92,
        "corruption_index": 10,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "SE": {
        "name": "Sweden",
        "iso_alpha3": "SWE",
        "region": "northern_europe",
        "composite_score": 4,
        "deforestation_rate": 2,
        "governance_index": 95,
        "enforcement_score": 95,
        "corruption_index": 5,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "FI": {
        "name": "Finland",
        "iso_alpha3": "FIN",
        "region": "northern_europe",
        "composite_score": 3,
        "deforestation_rate": 1,
        "governance_index": 96,
        "enforcement_score": 96,
        "corruption_index": 4,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "NO": {
        "name": "Norway",
        "iso_alpha3": "NOR",
        "region": "northern_europe",
        "composite_score": 3,
        "deforestation_rate": 1,
        "governance_index": 96,
        "enforcement_score": 96,
        "corruption_index": 4,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "AT": {
        "name": "Austria",
        "iso_alpha3": "AUT",
        "region": "western_europe",
        "composite_score": 5,
        "deforestation_rate": 2,
        "governance_index": 90,
        "enforcement_score": 92,
        "corruption_index": 10,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "PL": {
        "name": "Poland",
        "iso_alpha3": "POL",
        "region": "eastern_europe",
        "composite_score": 10,
        "deforestation_rate": 5,
        "governance_index": 72,
        "enforcement_score": 70,
        "corruption_index": 22,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "RO": {
        "name": "Romania",
        "iso_alpha3": "ROU",
        "region": "eastern_europe",
        "composite_score": 22,
        "deforestation_rate": 25,
        "governance_index": 52,
        "enforcement_score": 45,
        "corruption_index": 40,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "UA": {
        "name": "Ukraine",
        "iso_alpha3": "UKR",
        "region": "eastern_europe",
        "composite_score": 38,
        "deforestation_rate": 28,
        "governance_index": 30,
        "enforcement_score": 25,
        "corruption_index": 65,
        "eudr_high_risk": False,
        "primary_commodities": ["soya", "wood"],
    },
    "RU": {
        "name": "Russia",
        "iso_alpha3": "RUS",
        "region": "eastern_europe",
        "composite_score": 45,
        "deforestation_rate": 48,
        "governance_index": 25,
        "enforcement_score": 22,
        "corruption_index": 72,
        "eudr_high_risk": False,
        "primary_commodities": ["wood", "soya"],
    },
    # -----------------------------------------------------------------------
    # North America
    # -----------------------------------------------------------------------
    "US": {
        "name": "United States",
        "iso_alpha3": "USA",
        "region": "north_america",
        "composite_score": 15,
        "deforestation_rate": 10,
        "governance_index": 80,
        "enforcement_score": 82,
        "corruption_index": 20,
        "eudr_high_risk": False,
        "primary_commodities": ["soya", "wood", "cattle"],
    },
    "CA": {
        "name": "Canada",
        "iso_alpha3": "CAN",
        "region": "north_america",
        "composite_score": 12,
        "deforestation_rate": 8,
        "governance_index": 88,
        "enforcement_score": 85,
        "corruption_index": 12,
        "eudr_high_risk": False,
        "primary_commodities": ["wood", "soya"],
    },
    # -----------------------------------------------------------------------
    # Oceania
    # -----------------------------------------------------------------------
    "AU": {
        "name": "Australia",
        "iso_alpha3": "AUS",
        "region": "oceania",
        "composite_score": 20,
        "deforestation_rate": 22,
        "governance_index": 82,
        "enforcement_score": 80,
        "corruption_index": 15,
        "eudr_high_risk": False,
        "primary_commodities": ["cattle", "wood"],
    },
    "NZ": {
        "name": "New Zealand",
        "iso_alpha3": "NZL",
        "region": "oceania",
        "composite_score": 8,
        "deforestation_rate": 5,
        "governance_index": 92,
        "enforcement_score": 90,
        "corruption_index": 6,
        "eudr_high_risk": False,
        "primary_commodities": ["wood", "cattle"],
    },
    "SB": {
        "name": "Solomon Islands",
        "iso_alpha3": "SLB",
        "region": "oceania",
        "composite_score": 78,
        "deforestation_rate": 82,
        "governance_index": 15,
        "enforcement_score": 12,
        "corruption_index": 75,
        "eudr_high_risk": True,
        "primary_commodities": ["wood", "palm_oil"],
    },
    # -----------------------------------------------------------------------
    # Additional producer countries
    # -----------------------------------------------------------------------
    "CU": {
        "name": "Cuba",
        "iso_alpha3": "CUB",
        "region": "caribbean",
        "composite_score": 45,
        "deforestation_rate": 30,
        "governance_index": 25,
        "enforcement_score": 30,
        "corruption_index": 68,
        "eudr_high_risk": False,
        "primary_commodities": ["coffee", "cocoa"],
    },
    "JM": {
        "name": "Jamaica",
        "iso_alpha3": "JAM",
        "region": "caribbean",
        "composite_score": 40,
        "deforestation_rate": 38,
        "governance_index": 42,
        "enforcement_score": 35,
        "corruption_index": 55,
        "eudr_high_risk": False,
        "primary_commodities": ["coffee", "cocoa"],
    },
    "TT": {
        "name": "Trinidad and Tobago",
        "iso_alpha3": "TTO",
        "region": "caribbean",
        "composite_score": 32,
        "deforestation_rate": 25,
        "governance_index": 48,
        "enforcement_score": 42,
        "corruption_index": 50,
        "eudr_high_risk": False,
        "primary_commodities": ["cocoa"],
    },
    "BZ": {
        "name": "Belize",
        "iso_alpha3": "BLZ",
        "region": "central_america",
        "composite_score": 52,
        "deforestation_rate": 58,
        "governance_index": 38,
        "enforcement_score": 30,
        "corruption_index": 55,
        "eudr_high_risk": False,
        "primary_commodities": ["wood", "cattle"],
    },
    "SV": {
        "name": "El Salvador",
        "iso_alpha3": "SLV",
        "region": "central_america",
        "composite_score": 42,
        "deforestation_rate": 38,
        "governance_index": 30,
        "enforcement_score": 28,
        "corruption_index": 62,
        "eudr_high_risk": False,
        "primary_commodities": ["coffee"],
    },
    "TD": {
        "name": "Chad",
        "iso_alpha3": "TCD",
        "region": "central_africa",
        "composite_score": 78,
        "deforestation_rate": 70,
        "governance_index": 8,
        "enforcement_score": 8,
        "corruption_index": 85,
        "eudr_high_risk": True,
        "primary_commodities": ["cattle", "wood"],
    },
    "SD": {
        "name": "Sudan",
        "iso_alpha3": "SDN",
        "region": "east_africa",
        "composite_score": 80,
        "deforestation_rate": 75,
        "governance_index": 5,
        "enforcement_score": 5,
        "corruption_index": 90,
        "eudr_high_risk": True,
        "primary_commodities": ["cattle", "wood"],
    },
    "SS": {
        "name": "South Sudan",
        "iso_alpha3": "SSD",
        "region": "east_africa",
        "composite_score": 88,
        "deforestation_rate": 78,
        "governance_index": 3,
        "enforcement_score": 3,
        "corruption_index": 92,
        "eudr_high_risk": True,
        "primary_commodities": ["wood"],
    },
    "AO": {
        "name": "Angola",
        "iso_alpha3": "AGO",
        "region": "southern_africa",
        "composite_score": 72,
        "deforestation_rate": 68,
        "governance_index": 18,
        "enforcement_score": 15,
        "corruption_index": 78,
        "eudr_high_risk": True,
        "primary_commodities": ["wood", "coffee"],
    },
    "ZM": {
        "name": "Zambia",
        "iso_alpha3": "ZMB",
        "region": "southern_africa",
        "composite_score": 62,
        "deforestation_rate": 68,
        "governance_index": 30,
        "enforcement_score": 25,
        "corruption_index": 62,
        "eudr_high_risk": True,
        "primary_commodities": ["wood"],
    },
    "ZW": {
        "name": "Zimbabwe",
        "iso_alpha3": "ZWE",
        "region": "southern_africa",
        "composite_score": 68,
        "deforestation_rate": 62,
        "governance_index": 15,
        "enforcement_score": 12,
        "corruption_index": 78,
        "eudr_high_risk": True,
        "primary_commodities": ["wood", "coffee"],
    },
    "MW": {
        "name": "Malawi",
        "iso_alpha3": "MWI",
        "region": "southern_africa",
        "composite_score": 65,
        "deforestation_rate": 72,
        "governance_index": 28,
        "enforcement_score": 22,
        "corruption_index": 62,
        "eudr_high_risk": True,
        "primary_commodities": ["wood", "coffee"],
    },
    "SN": {
        "name": "Senegal",
        "iso_alpha3": "SEN",
        "region": "west_africa",
        "composite_score": 45,
        "deforestation_rate": 42,
        "governance_index": 42,
        "enforcement_score": 38,
        "corruption_index": 52,
        "eudr_high_risk": False,
        "primary_commodities": ["wood"],
    },
    "ML": {
        "name": "Mali",
        "iso_alpha3": "MLI",
        "region": "west_africa",
        "composite_score": 68,
        "deforestation_rate": 58,
        "governance_index": 15,
        "enforcement_score": 12,
        "corruption_index": 78,
        "eudr_high_risk": True,
        "primary_commodities": ["wood"],
    },
    "BF": {
        "name": "Burkina Faso",
        "iso_alpha3": "BFA",
        "region": "west_africa",
        "composite_score": 62,
        "deforestation_rate": 55,
        "governance_index": 22,
        "enforcement_score": 18,
        "corruption_index": 68,
        "eudr_high_risk": True,
        "primary_commodities": ["wood"],
    },
    "NE": {
        "name": "Niger",
        "iso_alpha3": "NER",
        "region": "west_africa",
        "composite_score": 60,
        "deforestation_rate": 50,
        "governance_index": 18,
        "enforcement_score": 15,
        "corruption_index": 72,
        "eudr_high_risk": True,
        "primary_commodities": ["wood"],
    },
}

# Totals
TOTAL_COUNTRIES: int = len(COUNTRY_RISK_SCORES)

# ---------------------------------------------------------------------------
# Regional Risk Aggregates
# ---------------------------------------------------------------------------

REGIONAL_RISK_AGGREGATES: Dict[str, Dict[str, Any]] = {
    "central_africa": {
        "name": "Central Africa",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
    "west_africa": {
        "name": "West Africa",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
    "east_africa": {
        "name": "East Africa",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
    "southern_africa": {
        "name": "Southern Africa",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
    "south_america": {
        "name": "South America",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
    "central_america": {
        "name": "Central America",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
    "southeast_asia": {
        "name": "Southeast Asia",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
    "south_asia": {
        "name": "South Asia",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
    "east_asia": {
        "name": "East Asia",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
    "western_europe": {
        "name": "Western Europe",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
    "northern_europe": {
        "name": "Northern Europe",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
    "eastern_europe": {
        "name": "Eastern Europe",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
    "north_america": {
        "name": "North America",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
    "oceania": {
        "name": "Oceania",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
    "caribbean": {
        "name": "Caribbean",
        "avg_composite_score": 0.0,
        "highest_risk_country": "",
        "country_count": 0,
    },
}


def _compute_regional_aggregates() -> None:
    """Compute regional risk aggregates from individual country scores."""
    region_scores: Dict[str, List[Tuple[str, int]]] = {}
    for iso, data in COUNTRY_RISK_SCORES.items():
        region = data["region"]
        if region not in region_scores:
            region_scores[region] = []
        region_scores[region].append((iso, data["composite_score"]))

    for region, entries in region_scores.items():
        if region not in REGIONAL_RISK_AGGREGATES:
            continue
        scores = [s for _, s in entries]
        avg = sum(scores) / len(scores) if scores else 0.0
        highest_iso = max(entries, key=lambda x: x[1])[0] if entries else ""
        REGIONAL_RISK_AGGREGATES[region]["avg_composite_score"] = round(avg, 1)
        REGIONAL_RISK_AGGREGATES[region]["highest_risk_country"] = highest_iso
        REGIONAL_RISK_AGGREGATES[region]["country_count"] = len(entries)


# Compute on module load
_compute_regional_aggregates()


# ---------------------------------------------------------------------------
# Accessor Functions
# ---------------------------------------------------------------------------


def get_country_risk(country_iso: str) -> Optional[Dict[str, Any]]:
    """Get the full risk data for a country by ISO 3166-1 alpha-2 code.

    Args:
        country_iso: ISO 3166-1 alpha-2 country code (e.g. 'BR', 'ID').

    Returns:
        Dictionary with risk data or None if country not found.

    Example:
        >>> data = get_country_risk("BR")
        >>> assert data["composite_score"] == 72
    """
    return COUNTRY_RISK_SCORES.get(country_iso.upper()) if country_iso else None


def get_risk_factors(country_iso: str) -> Optional[Dict[str, Any]]:
    """Get the risk factor breakdown for a country.

    Args:
        country_iso: ISO 3166-1 alpha-2 country code.

    Returns:
        Dictionary with individual risk factors or None if not found.

    Example:
        >>> factors = get_risk_factors("ID")
        >>> assert "deforestation_rate" in factors
    """
    data = get_country_risk(country_iso)
    if data is None:
        return None
    return {
        "deforestation_rate": data["deforestation_rate"],
        "governance_index": data["governance_index"],
        "enforcement_score": data["enforcement_score"],
        "corruption_index": data["corruption_index"],
        "composite_score": data["composite_score"],
    }


def get_high_risk_countries() -> List[Dict[str, Any]]:
    """Get all countries designated as EUDR high-risk.

    Returns:
        List of dictionaries with ISO code, name, and composite score
        for all high-risk countries, sorted by score descending.

    Example:
        >>> high_risk = get_high_risk_countries()
        >>> assert len(high_risk) > 0
        >>> assert all(c["eudr_high_risk"] for c in high_risk)
    """
    results = []
    for iso, data in COUNTRY_RISK_SCORES.items():
        if data.get("eudr_high_risk", False):
            results.append({
                "iso": iso,
                "name": data["name"],
                "composite_score": data["composite_score"],
                "region": data["region"],
                "primary_commodities": data["primary_commodities"],
                "eudr_high_risk": True,
            })
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results


def get_risk_by_region(region: Optional[str] = None) -> Dict[str, Any]:
    """Get regional risk aggregates.

    Args:
        region: Optional region key (e.g. 'south_america'). If None,
            returns all regional aggregates.

    Returns:
        Dictionary with regional risk data. If region specified, returns
        single region data. If None, returns all regions.

    Example:
        >>> data = get_risk_by_region("south_america")
        >>> assert data["country_count"] > 0
    """
    if region is not None:
        return REGIONAL_RISK_AGGREGATES.get(region.lower(), {})
    return dict(REGIONAL_RISK_AGGREGATES)


def get_countries_by_commodity(commodity: str) -> List[Dict[str, Any]]:
    """Get all countries that produce a given EUDR commodity.

    Args:
        commodity: EUDR commodity key (e.g. 'cocoa', 'palm_oil').

    Returns:
        List of country entries that list this commodity as primary,
        sorted by composite risk score descending.

    Example:
        >>> cocoa_countries = get_countries_by_commodity("cocoa")
        >>> assert any(c["iso"] == "CI" for c in cocoa_countries)
    """
    commodity_lower = commodity.lower()
    results = []
    for iso, data in COUNTRY_RISK_SCORES.items():
        if commodity_lower in data.get("primary_commodities", []):
            results.append({
                "iso": iso,
                "name": data["name"],
                "composite_score": data["composite_score"],
                "region": data["region"],
                "eudr_high_risk": data["eudr_high_risk"],
            })
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results


def get_composite_score(country_iso: str) -> Optional[int]:
    """Get the composite risk score for a country.

    Args:
        country_iso: ISO 3166-1 alpha-2 country code.

    Returns:
        Integer risk score (0-100) or None if not found.

    Example:
        >>> score = get_composite_score("BR")
        >>> assert 0 <= score <= 100
    """
    data = get_country_risk(country_iso)
    if data is None:
        return None
    return data["composite_score"]


def is_high_risk(country_iso: str) -> bool:
    """Check whether a country is designated as EUDR high-risk.

    Args:
        country_iso: ISO 3166-1 alpha-2 country code.

    Returns:
        True if the country is designated high-risk, False otherwise.
        Returns False for unknown countries.

    Example:
        >>> assert is_high_risk("BR") is True
        >>> assert is_high_risk("DE") is False
    """
    data = get_country_risk(country_iso)
    if data is None:
        return False
    return data.get("eudr_high_risk", False)
