# -*- coding: utf-8 -*-
"""
Country Risk Database - AGENT-EUDR-016 Country Risk Evaluator

Comprehensive country risk reference data for 60+ countries aligned with
the EUDR (EU 2023/1115) benchmarking system per Article 29. Each entry
provides the EC risk classification (low/standard/high), a composite
base risk score (0-100), forest cover and deforestation statistics,
governance and enforcement indicators, and commodity production flags.

Countries are grouped by risk level:
    - High Risk (15+ countries): Brazil, Indonesia, DRC, Malaysia,
      Colombia, Myanmar, Cambodia, Papua New Guinea, Laos, Bolivia,
      Paraguay, Peru, Ghana, Cameroon, Cote d'Ivoire, etc.
    - Standard Risk (25+ countries): India, Thailand, Vietnam,
      Philippines, China, Mexico, Argentina, Ecuador, Kenya, Tanzania,
      Uganda, Ethiopia, Nigeria, Guatemala, Honduras, Costa Rica, etc.
    - Low Risk (20+ countries): EU member states, USA, Canada,
      Australia, New Zealand, Japan, South Korea, UK, Norway,
      Switzerland, etc.

Data Sources:
    - FAO Global Forest Resources Assessment 2025
    - Global Forest Watch Tree Cover Loss 2024
    - World Bank Worldwide Governance Indicators 2024
    - Transparency International CPI 2024
    - European Commission EUDR Country Benchmarking (2025 provisional)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Type alias for a single country risk record
# ---------------------------------------------------------------------------

CountryRiskRecord = Dict[str, Any]

# ---------------------------------------------------------------------------
# EUDR commodities for commodity_production flags
# ---------------------------------------------------------------------------

EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

# ---------------------------------------------------------------------------
# Data version and source metadata
# ---------------------------------------------------------------------------

DATA_VERSION: str = "2025-03"
DATA_SOURCES: List[str] = [
    "FAO Global Forest Resources Assessment 2025",
    "Global Forest Watch Tree Cover Loss 2024",
    "World Bank Worldwide Governance Indicators 2024",
    "Transparency International Corruption Perceptions Index 2024",
    "European Commission EUDR Country Benchmarking (2025 provisional)",
]

# ===========================================================================
# Country Risk Database - 60+ countries
# ===========================================================================
#
# Each record keys:
#   country_code            - ISO 3166-1 alpha-2
#   country_name            - Full English name
#   ec_risk_classification  - "low" | "standard" | "high"
#   base_risk_score         - 0-100, higher = riskier
#   forest_cover_pct        - Current forest cover %
#   annual_deforestation_rate - Annual tree cover loss % (negative = loss)
#   primary_forest_pct      - % of forest that is primary/old-growth
#   primary_deforestation_drivers - List of main drivers
#   regions_of_concern      - Sub-national high-risk regions
#   commodity_production    - Which of 7 EUDR commodities are produced
#   governance_score        - 0-100, higher = better governance
#   enforcement_score       - 0-100, higher = better enforcement
#   corruption_perception_index - 0-100, higher = less corrupt
#   region                  - Geographic region key

COUNTRY_RISK_DATABASE: Dict[str, CountryRiskRecord] = {

    # ===================================================================
    # HIGH RISK COUNTRIES (15+)
    # ===================================================================

    "BR": {
        "country_code": "BR",
        "country_name": "Brazil",
        "ec_risk_classification": "high",
        "base_risk_score": 78,
        "forest_cover_pct": 59.4,
        "annual_deforestation_rate": -0.50,
        "primary_forest_pct": 57.3,
        "primary_deforestation_drivers": [
            "Cattle ranching expansion",
            "Soy cultivation",
            "Illegal logging",
            "Land speculation",
            "Infrastructure development",
        ],
        "regions_of_concern": [
            "Amazonia Legal", "Cerrado", "Mato Grosso",
            "Para", "Rondonia", "Maranhao", "Tocantins",
        ],
        "commodity_production": [
            "cattle", "soya", "coffee", "cocoa", "wood", "rubber",
        ],
        "governance_score": 42,
        "enforcement_score": 38,
        "corruption_perception_index": 36,
        "region": "south_america",
    },

    "ID": {
        "country_code": "ID",
        "country_name": "Indonesia",
        "ec_risk_classification": "high",
        "base_risk_score": 76,
        "forest_cover_pct": 49.1,
        "annual_deforestation_rate": -0.80,
        "primary_forest_pct": 35.7,
        "primary_deforestation_drivers": [
            "Palm oil plantation expansion",
            "Pulp and paper plantations",
            "Mining",
            "Peatland drainage",
            "Smallholder agriculture",
        ],
        "regions_of_concern": [
            "Sumatra", "Kalimantan", "Papua", "Sulawesi",
            "Riau", "West Kalimantan", "Central Kalimantan",
        ],
        "commodity_production": [
            "oil_palm", "rubber", "cocoa", "coffee", "wood",
        ],
        "governance_score": 38,
        "enforcement_score": 32,
        "corruption_perception_index": 34,
        "region": "southeast_asia",
    },

    "CD": {
        "country_code": "CD",
        "country_name": "Democratic Republic of the Congo",
        "ec_risk_classification": "high",
        "base_risk_score": 88,
        "forest_cover_pct": 52.6,
        "annual_deforestation_rate": -0.80,
        "primary_forest_pct": 60.0,
        "primary_deforestation_drivers": [
            "Subsistence agriculture",
            "Charcoal production",
            "Industrial logging",
            "Mining",
            "Armed conflict",
        ],
        "regions_of_concern": [
            "Equateur", "Mai-Ndombe", "Tshuapa",
            "Orientale", "North Kivu", "South Kivu",
        ],
        "commodity_production": ["wood", "cocoa", "coffee", "oil_palm", "rubber"],
        "governance_score": 10,
        "enforcement_score": 8,
        "corruption_perception_index": 20,
        "region": "central_africa",
    },

    "MY": {
        "country_code": "MY",
        "country_name": "Malaysia",
        "ec_risk_classification": "high",
        "base_risk_score": 68,
        "forest_cover_pct": 57.7,
        "annual_deforestation_rate": -0.30,
        "primary_forest_pct": 18.4,
        "primary_deforestation_drivers": [
            "Palm oil plantation expansion",
            "Logging",
            "Agricultural conversion",
            "Infrastructure development",
        ],
        "regions_of_concern": [
            "Sabah", "Sarawak", "Peninsular Malaysia",
        ],
        "commodity_production": ["oil_palm", "rubber", "wood", "cocoa"],
        "governance_score": 55,
        "enforcement_score": 48,
        "corruption_perception_index": 50,
        "region": "southeast_asia",
    },

    "CO": {
        "country_code": "CO",
        "country_name": "Colombia",
        "ec_risk_classification": "high",
        "base_risk_score": 70,
        "forest_cover_pct": 52.7,
        "annual_deforestation_rate": -0.50,
        "primary_forest_pct": 46.2,
        "primary_deforestation_drivers": [
            "Cattle ranching",
            "Coca cultivation",
            "Agricultural frontier expansion",
            "Illegal mining",
        ],
        "regions_of_concern": [
            "Caqueta", "Meta", "Guaviare", "Putumayo",
        ],
        "commodity_production": [
            "coffee", "oil_palm", "cattle", "cocoa", "wood",
        ],
        "governance_score": 40,
        "enforcement_score": 35,
        "corruption_perception_index": 39,
        "region": "south_america",
    },

    "MM": {
        "country_code": "MM",
        "country_name": "Myanmar",
        "ec_risk_classification": "high",
        "base_risk_score": 84,
        "forest_cover_pct": 42.9,
        "annual_deforestation_rate": -1.20,
        "primary_forest_pct": 28.5,
        "primary_deforestation_drivers": [
            "Agricultural expansion",
            "Illegal logging",
            "Mining",
            "Armed conflict",
        ],
        "regions_of_concern": [
            "Kachin State", "Shan State", "Sagaing Region",
            "Tanintharyi Region",
        ],
        "commodity_production": ["rubber", "wood"],
        "governance_score": 12,
        "enforcement_score": 10,
        "corruption_perception_index": 23,
        "region": "southeast_asia",
    },

    "KH": {
        "country_code": "KH",
        "country_name": "Cambodia",
        "ec_risk_classification": "high",
        "base_risk_score": 80,
        "forest_cover_pct": 46.3,
        "annual_deforestation_rate": -1.50,
        "primary_forest_pct": 20.0,
        "primary_deforestation_drivers": [
            "Economic land concessions",
            "Illegal logging",
            "Agricultural expansion",
            "Infrastructure projects",
        ],
        "regions_of_concern": [
            "Mondulkiri", "Ratanakiri", "Stung Treng",
            "Preah Vihear",
        ],
        "commodity_production": ["rubber", "wood"],
        "governance_score": 18,
        "enforcement_score": 15,
        "corruption_perception_index": 22,
        "region": "southeast_asia",
    },

    "PG": {
        "country_code": "PG",
        "country_name": "Papua New Guinea",
        "ec_risk_classification": "high",
        "base_risk_score": 82,
        "forest_cover_pct": 74.1,
        "annual_deforestation_rate": -0.50,
        "primary_forest_pct": 65.0,
        "primary_deforestation_drivers": [
            "Industrial logging",
            "Palm oil expansion",
            "Mining",
            "Subsistence agriculture",
        ],
        "regions_of_concern": [
            "Western Province", "Gulf Province", "New Britain",
        ],
        "commodity_production": ["wood", "oil_palm", "cocoa", "coffee"],
        "governance_score": 18,
        "enforcement_score": 12,
        "corruption_perception_index": 25,
        "region": "oceania",
    },

    "LA": {
        "country_code": "LA",
        "country_name": "Laos",
        "ec_risk_classification": "high",
        "base_risk_score": 77,
        "forest_cover_pct": 58.0,
        "annual_deforestation_rate": -0.70,
        "primary_forest_pct": 15.0,
        "primary_deforestation_drivers": [
            "Rubber plantation expansion",
            "Hydropower development",
            "Agricultural conversion",
            "Illegal logging",
        ],
        "regions_of_concern": [
            "Attapeu", "Sekong", "Bolikhamxai",
        ],
        "commodity_production": ["rubber", "coffee", "wood"],
        "governance_score": 22,
        "enforcement_score": 18,
        "corruption_perception_index": 28,
        "region": "southeast_asia",
    },

    "BO": {
        "country_code": "BO",
        "country_name": "Bolivia",
        "ec_risk_classification": "high",
        "base_risk_score": 76,
        "forest_cover_pct": 50.6,
        "annual_deforestation_rate": -0.80,
        "primary_forest_pct": 42.1,
        "primary_deforestation_drivers": [
            "Soy cultivation",
            "Cattle ranching",
            "Small-scale agriculture",
            "Fire-based land clearing",
        ],
        "regions_of_concern": [
            "Santa Cruz", "Beni", "Pando", "Chiquitania",
        ],
        "commodity_production": ["soya", "cattle", "wood", "coffee", "cocoa"],
        "governance_score": 25,
        "enforcement_score": 20,
        "corruption_perception_index": 31,
        "region": "south_america",
    },

    "PY": {
        "country_code": "PY",
        "country_name": "Paraguay",
        "ec_risk_classification": "high",
        "base_risk_score": 74,
        "forest_cover_pct": 38.6,
        "annual_deforestation_rate": -1.00,
        "primary_forest_pct": 15.2,
        "primary_deforestation_drivers": [
            "Cattle ranching",
            "Soy cultivation",
            "Land clearing for agriculture",
        ],
        "regions_of_concern": [
            "Chaco region", "San Pedro", "Canindeyu",
        ],
        "commodity_production": ["cattle", "soya", "wood"],
        "governance_score": 30,
        "enforcement_score": 25,
        "corruption_perception_index": 28,
        "region": "south_america",
    },

    "PE": {
        "country_code": "PE",
        "country_name": "Peru",
        "ec_risk_classification": "high",
        "base_risk_score": 68,
        "forest_cover_pct": 57.8,
        "annual_deforestation_rate": -0.30,
        "primary_forest_pct": 57.3,
        "primary_deforestation_drivers": [
            "Small-scale agriculture",
            "Palm oil expansion",
            "Coca cultivation",
            "Mining",
            "Road construction",
        ],
        "regions_of_concern": [
            "Ucayali", "San Martin", "Loreto", "Madre de Dios",
        ],
        "commodity_production": [
            "coffee", "cocoa", "oil_palm", "wood", "cattle",
        ],
        "governance_score": 40,
        "enforcement_score": 32,
        "corruption_perception_index": 36,
        "region": "south_america",
    },

    "GH": {
        "country_code": "GH",
        "country_name": "Ghana",
        "ec_risk_classification": "high",
        "base_risk_score": 70,
        "forest_cover_pct": 36.2,
        "annual_deforestation_rate": -2.00,
        "primary_forest_pct": 5.8,
        "primary_deforestation_drivers": [
            "Cocoa farming",
            "Mining (galamsey)",
            "Logging",
            "Agricultural expansion",
        ],
        "regions_of_concern": [
            "Western Region", "Ashanti Region", "Brong-Ahafo",
        ],
        "commodity_production": ["cocoa", "wood", "oil_palm"],
        "governance_score": 48,
        "enforcement_score": 40,
        "corruption_perception_index": 43,
        "region": "west_africa",
    },

    "CM": {
        "country_code": "CM",
        "country_name": "Cameroon",
        "ec_risk_classification": "high",
        "base_risk_score": 74,
        "forest_cover_pct": 41.7,
        "annual_deforestation_rate": -0.50,
        "primary_forest_pct": 30.2,
        "primary_deforestation_drivers": [
            "Industrial agriculture",
            "Logging",
            "Subsistence agriculture",
            "Infrastructure",
        ],
        "regions_of_concern": [
            "South Region", "East Region", "Center Region",
        ],
        "commodity_production": ["wood", "rubber", "oil_palm", "cocoa", "coffee"],
        "governance_score": 22,
        "enforcement_score": 20,
        "corruption_perception_index": 26,
        "region": "central_africa",
    },

    "CI": {
        "country_code": "CI",
        "country_name": "Cote d'Ivoire",
        "ec_risk_classification": "high",
        "base_risk_score": 80,
        "forest_cover_pct": 8.9,
        "annual_deforestation_rate": -3.00,
        "primary_forest_pct": 2.0,
        "primary_deforestation_drivers": [
            "Cocoa farming expansion",
            "Illegal farming in protected areas",
            "Logging",
            "Population pressure",
        ],
        "regions_of_concern": [
            "Tai National Park surroundings", "Mont Peko",
            "Southwest region", "Western region",
        ],
        "commodity_production": ["cocoa", "rubber", "oil_palm", "wood", "coffee"],
        "governance_score": 28,
        "enforcement_score": 22,
        "corruption_perception_index": 36,
        "region": "west_africa",
    },

    "CG": {
        "country_code": "CG",
        "country_name": "Republic of the Congo",
        "ec_risk_classification": "high",
        "base_risk_score": 75,
        "forest_cover_pct": 65.4,
        "annual_deforestation_rate": -0.30,
        "primary_forest_pct": 55.1,
        "primary_deforestation_drivers": [
            "Industrial logging",
            "Subsistence agriculture",
            "Charcoal production",
            "Mining",
        ],
        "regions_of_concern": ["Northern regions", "Pool region"],
        "commodity_production": ["wood", "oil_palm", "rubber"],
        "governance_score": 20,
        "enforcement_score": 18,
        "corruption_perception_index": 21,
        "region": "central_africa",
    },

    # ===================================================================
    # STANDARD RISK COUNTRIES (25+)
    # ===================================================================

    "IN": {
        "country_code": "IN",
        "country_name": "India",
        "ec_risk_classification": "standard",
        "base_risk_score": 48,
        "forest_cover_pct": 24.1,
        "annual_deforestation_rate": 0.40,
        "primary_forest_pct": 2.2,
        "primary_deforestation_drivers": [
            "Mining",
            "Infrastructure",
            "Agricultural encroachment",
            "Plantation expansion",
        ],
        "regions_of_concern": [
            "Northeast India", "Western Ghats", "Central India",
        ],
        "commodity_production": ["rubber", "coffee", "wood", "soya"],
        "governance_score": 48,
        "enforcement_score": 42,
        "corruption_perception_index": 40,
        "region": "south_asia",
    },

    "TH": {
        "country_code": "TH",
        "country_name": "Thailand",
        "ec_risk_classification": "standard",
        "base_risk_score": 45,
        "forest_cover_pct": 32.6,
        "annual_deforestation_rate": 0.30,
        "primary_forest_pct": 15.1,
        "primary_deforestation_drivers": [
            "Agricultural encroachment",
            "Rubber plantation expansion",
            "Infrastructure",
        ],
        "regions_of_concern": ["Southern region", "Northeast"],
        "commodity_production": ["rubber", "oil_palm", "wood", "coffee"],
        "governance_score": 50,
        "enforcement_score": 45,
        "corruption_perception_index": 36,
        "region": "southeast_asia",
    },

    "VN": {
        "country_code": "VN",
        "country_name": "Vietnam",
        "ec_risk_classification": "standard",
        "base_risk_score": 50,
        "forest_cover_pct": 42.1,
        "annual_deforestation_rate": 0.50,
        "primary_forest_pct": 0.6,
        "primary_deforestation_drivers": [
            "Plantation establishment",
            "Infrastructure",
            "Agricultural conversion",
        ],
        "regions_of_concern": [
            "Central Highlands", "Tay Nguyen",
        ],
        "commodity_production": ["coffee", "rubber", "wood", "cocoa"],
        "governance_score": 42,
        "enforcement_score": 38,
        "corruption_perception_index": 41,
        "region": "southeast_asia",
    },

    "PH": {
        "country_code": "PH",
        "country_name": "Philippines",
        "ec_risk_classification": "standard",
        "base_risk_score": 52,
        "forest_cover_pct": 24.0,
        "annual_deforestation_rate": -0.50,
        "primary_forest_pct": 8.0,
        "primary_deforestation_drivers": [
            "Agricultural expansion",
            "Mining",
            "Infrastructure",
        ],
        "regions_of_concern": ["Mindanao", "Palawan"],
        "commodity_production": ["cocoa", "oil_palm", "rubber", "wood", "coffee"],
        "governance_score": 40,
        "enforcement_score": 35,
        "corruption_perception_index": 33,
        "region": "southeast_asia",
    },

    "CN": {
        "country_code": "CN",
        "country_name": "China",
        "ec_risk_classification": "standard",
        "base_risk_score": 40,
        "forest_cover_pct": 23.3,
        "annual_deforestation_rate": 1.00,
        "primary_forest_pct": 3.0,
        "primary_deforestation_drivers": [
            "Infrastructure",
            "Urbanization",
            "Agricultural conversion",
        ],
        "regions_of_concern": ["Yunnan", "Hainan"],
        "commodity_production": ["wood", "rubber", "soya"],
        "governance_score": 50,
        "enforcement_score": 52,
        "corruption_perception_index": 45,
        "region": "east_asia",
    },

    "MX": {
        "country_code": "MX",
        "country_name": "Mexico",
        "ec_risk_classification": "standard",
        "base_risk_score": 50,
        "forest_cover_pct": 33.7,
        "annual_deforestation_rate": -0.20,
        "primary_forest_pct": 21.5,
        "primary_deforestation_drivers": [
            "Cattle ranching",
            "Soy and palm oil expansion",
            "Avocado cultivation",
            "Illegal logging",
        ],
        "regions_of_concern": [
            "Chiapas", "Yucatan Peninsula", "Tabasco",
        ],
        "commodity_production": ["cattle", "wood", "oil_palm", "coffee", "cocoa"],
        "governance_score": 40,
        "enforcement_score": 32,
        "corruption_perception_index": 31,
        "region": "central_america",
    },

    "AR": {
        "country_code": "AR",
        "country_name": "Argentina",
        "ec_risk_classification": "standard",
        "base_risk_score": 55,
        "forest_cover_pct": 10.2,
        "annual_deforestation_rate": -0.80,
        "primary_forest_pct": 8.2,
        "primary_deforestation_drivers": [
            "Soy cultivation",
            "Cattle ranching",
            "Agricultural expansion",
        ],
        "regions_of_concern": [
            "Gran Chaco", "Salta", "Santiago del Estero",
        ],
        "commodity_production": ["soya", "cattle", "wood"],
        "governance_score": 42,
        "enforcement_score": 38,
        "corruption_perception_index": 38,
        "region": "south_america",
    },

    "EC": {
        "country_code": "EC",
        "country_name": "Ecuador",
        "ec_risk_classification": "standard",
        "base_risk_score": 55,
        "forest_cover_pct": 51.5,
        "annual_deforestation_rate": -0.40,
        "primary_forest_pct": 35.8,
        "primary_deforestation_drivers": [
            "Palm oil expansion",
            "Cattle ranching",
            "Mining",
            "Road construction",
        ],
        "regions_of_concern": ["Esmeraldas", "Oriente region"],
        "commodity_production": ["cocoa", "oil_palm", "coffee", "wood", "cattle"],
        "governance_score": 38,
        "enforcement_score": 32,
        "corruption_perception_index": 36,
        "region": "south_america",
    },

    "KE": {
        "country_code": "KE",
        "country_name": "Kenya",
        "ec_risk_classification": "standard",
        "base_risk_score": 48,
        "forest_cover_pct": 7.8,
        "annual_deforestation_rate": -0.30,
        "primary_forest_pct": 3.0,
        "primary_deforestation_drivers": [
            "Agricultural expansion",
            "Charcoal production",
            "Infrastructure",
        ],
        "regions_of_concern": ["Mau Forest Complex", "Western Kenya"],
        "commodity_production": ["coffee", "wood"],
        "governance_score": 42,
        "enforcement_score": 38,
        "corruption_perception_index": 32,
        "region": "east_africa",
    },

    "TZ": {
        "country_code": "TZ",
        "country_name": "Tanzania",
        "ec_risk_classification": "standard",
        "base_risk_score": 58,
        "forest_cover_pct": 50.6,
        "annual_deforestation_rate": -0.80,
        "primary_forest_pct": 18.0,
        "primary_deforestation_drivers": [
            "Agricultural expansion",
            "Charcoal production",
            "Logging",
            "Population pressure",
        ],
        "regions_of_concern": ["Tabora", "Katavi", "Lindi"],
        "commodity_production": ["coffee", "wood"],
        "governance_score": 35,
        "enforcement_score": 28,
        "corruption_perception_index": 38,
        "region": "east_africa",
    },

    "UG": {
        "country_code": "UG",
        "country_name": "Uganda",
        "ec_risk_classification": "standard",
        "base_risk_score": 58,
        "forest_cover_pct": 9.0,
        "annual_deforestation_rate": -2.60,
        "primary_forest_pct": 3.0,
        "primary_deforestation_drivers": [
            "Agricultural expansion",
            "Charcoal production",
            "Logging",
        ],
        "regions_of_concern": [
            "Albertine Rift", "Bugoma Forest", "Mabira Forest",
        ],
        "commodity_production": ["coffee", "cocoa", "wood"],
        "governance_score": 32,
        "enforcement_score": 25,
        "corruption_perception_index": 26,
        "region": "east_africa",
    },

    "ET": {
        "country_code": "ET",
        "country_name": "Ethiopia",
        "ec_risk_classification": "standard",
        "base_risk_score": 55,
        "forest_cover_pct": 15.5,
        "annual_deforestation_rate": -0.80,
        "primary_forest_pct": 3.2,
        "primary_deforestation_drivers": [
            "Agricultural expansion",
            "Coffee forest degradation",
            "Fuelwood collection",
            "Population pressure",
        ],
        "regions_of_concern": ["Oromia", "SNNPR", "Gambella"],
        "commodity_production": ["coffee", "wood"],
        "governance_score": 28,
        "enforcement_score": 22,
        "corruption_perception_index": 38,
        "region": "east_africa",
    },

    "NG": {
        "country_code": "NG",
        "country_name": "Nigeria",
        "ec_risk_classification": "standard",
        "base_risk_score": 62,
        "forest_cover_pct": 7.2,
        "annual_deforestation_rate": -3.70,
        "primary_forest_pct": 1.5,
        "primary_deforestation_drivers": [
            "Agricultural expansion",
            "Logging",
            "Fuelwood collection",
            "Population pressure",
        ],
        "regions_of_concern": [
            "Cross River", "Edo", "Ondo", "Ogun",
        ],
        "commodity_production": ["cocoa", "oil_palm", "rubber", "wood"],
        "governance_score": 22,
        "enforcement_score": 18,
        "corruption_perception_index": 24,
        "region": "west_africa",
    },

    "GT": {
        "country_code": "GT",
        "country_name": "Guatemala",
        "ec_risk_classification": "standard",
        "base_risk_score": 58,
        "forest_cover_pct": 33.0,
        "annual_deforestation_rate": -1.00,
        "primary_forest_pct": 20.1,
        "primary_deforestation_drivers": [
            "Palm oil expansion",
            "Cattle ranching",
            "Agricultural frontier",
        ],
        "regions_of_concern": ["Peten", "Izabal", "Alta Verapaz"],
        "commodity_production": ["oil_palm", "coffee", "wood", "cattle"],
        "governance_score": 28,
        "enforcement_score": 22,
        "corruption_perception_index": 24,
        "region": "central_america",
    },

    "HN": {
        "country_code": "HN",
        "country_name": "Honduras",
        "ec_risk_classification": "standard",
        "base_risk_score": 60,
        "forest_cover_pct": 41.0,
        "annual_deforestation_rate": -1.00,
        "primary_forest_pct": 12.5,
        "primary_deforestation_drivers": [
            "Cattle ranching",
            "Palm oil expansion",
            "Coffee cultivation",
            "Illegal logging",
        ],
        "regions_of_concern": ["Mosquitia", "Olancho", "Colon"],
        "commodity_production": ["coffee", "oil_palm", "wood", "cattle"],
        "governance_score": 22,
        "enforcement_score": 18,
        "corruption_perception_index": 23,
        "region": "central_america",
    },

    "CR": {
        "country_code": "CR",
        "country_name": "Costa Rica",
        "ec_risk_classification": "standard",
        "base_risk_score": 32,
        "forest_cover_pct": 54.0,
        "annual_deforestation_rate": 0.50,
        "primary_forest_pct": 12.0,
        "primary_deforestation_drivers": [
            "Pineapple cultivation",
            "Infrastructure",
        ],
        "regions_of_concern": [],
        "commodity_production": ["coffee", "oil_palm"],
        "governance_score": 65,
        "enforcement_score": 60,
        "corruption_perception_index": 54,
        "region": "central_america",
    },

    "CF": {
        "country_code": "CF",
        "country_name": "Central African Republic",
        "ec_risk_classification": "standard",
        "base_risk_score": 62,
        "forest_cover_pct": 35.6,
        "annual_deforestation_rate": -0.20,
        "primary_forest_pct": 30.0,
        "primary_deforestation_drivers": [
            "Subsistence agriculture",
            "Logging",
            "Armed conflict",
        ],
        "regions_of_concern": ["Southwest forests"],
        "commodity_production": ["wood", "coffee"],
        "governance_score": 8,
        "enforcement_score": 5,
        "corruption_perception_index": 24,
        "region": "central_africa",
    },

    "GA": {
        "country_code": "GA",
        "country_name": "Gabon",
        "ec_risk_classification": "standard",
        "base_risk_score": 45,
        "forest_cover_pct": 88.0,
        "annual_deforestation_rate": -0.08,
        "primary_forest_pct": 70.0,
        "primary_deforestation_drivers": [
            "Logging",
            "Mining",
            "Palm oil expansion",
        ],
        "regions_of_concern": ["Estuaire"],
        "commodity_production": ["wood", "oil_palm"],
        "governance_score": 35,
        "enforcement_score": 30,
        "corruption_perception_index": 31,
        "region": "central_africa",
    },

    "LR": {
        "country_code": "LR",
        "country_name": "Liberia",
        "ec_risk_classification": "standard",
        "base_risk_score": 62,
        "forest_cover_pct": 43.0,
        "annual_deforestation_rate": -0.80,
        "primary_forest_pct": 12.0,
        "primary_deforestation_drivers": [
            "Rubber plantations",
            "Palm oil expansion",
            "Logging",
            "Mining",
        ],
        "regions_of_concern": [
            "Sinoe County", "Grand Gedeh County",
        ],
        "commodity_production": ["rubber", "oil_palm", "wood", "cocoa"],
        "governance_score": 20,
        "enforcement_score": 15,
        "corruption_perception_index": 25,
        "region": "west_africa",
    },

    "SL": {
        "country_code": "SL",
        "country_name": "Sierra Leone",
        "ec_risk_classification": "standard",
        "base_risk_score": 58,
        "forest_cover_pct": 38.0,
        "annual_deforestation_rate": -0.60,
        "primary_forest_pct": 5.0,
        "primary_deforestation_drivers": [
            "Agricultural expansion",
            "Mining",
            "Logging",
        ],
        "regions_of_concern": ["Eastern Province", "Southern Province"],
        "commodity_production": ["cocoa", "oil_palm", "wood"],
        "governance_score": 22,
        "enforcement_score": 18,
        "corruption_perception_index": 34,
        "region": "west_africa",
    },

    "MZ": {
        "country_code": "MZ",
        "country_name": "Mozambique",
        "ec_risk_classification": "standard",
        "base_risk_score": 58,
        "forest_cover_pct": 48.0,
        "annual_deforestation_rate": -0.40,
        "primary_forest_pct": 15.0,
        "primary_deforestation_drivers": [
            "Agricultural expansion",
            "Charcoal production",
            "Illegal logging",
        ],
        "regions_of_concern": ["Zambezia", "Nampula", "Cabo Delgado"],
        "commodity_production": ["wood"],
        "governance_score": 25,
        "enforcement_score": 18,
        "corruption_perception_index": 26,
        "region": "east_africa",
    },

    "MG": {
        "country_code": "MG",
        "country_name": "Madagascar",
        "ec_risk_classification": "standard",
        "base_risk_score": 62,
        "forest_cover_pct": 21.4,
        "annual_deforestation_rate": -1.10,
        "primary_forest_pct": 12.0,
        "primary_deforestation_drivers": [
            "Slash-and-burn agriculture",
            "Charcoal production",
            "Illegal logging",
        ],
        "regions_of_concern": ["East coast forests", "Menabe"],
        "commodity_production": ["cocoa", "wood", "coffee"],
        "governance_score": 20,
        "enforcement_score": 15,
        "corruption_perception_index": 26,
        "region": "east_africa",
    },

    "LK": {
        "country_code": "LK",
        "country_name": "Sri Lanka",
        "ec_risk_classification": "standard",
        "base_risk_score": 45,
        "forest_cover_pct": 29.7,
        "annual_deforestation_rate": -0.20,
        "primary_forest_pct": 8.0,
        "primary_deforestation_drivers": [
            "Agricultural expansion",
            "Infrastructure",
        ],
        "regions_of_concern": ["Wet zone forests"],
        "commodity_production": ["rubber", "coffee", "wood"],
        "governance_score": 42,
        "enforcement_score": 38,
        "corruption_perception_index": 37,
        "region": "south_asia",
    },

    "BD": {
        "country_code": "BD",
        "country_name": "Bangladesh",
        "ec_risk_classification": "standard",
        "base_risk_score": 50,
        "forest_cover_pct": 11.0,
        "annual_deforestation_rate": -0.30,
        "primary_forest_pct": 2.0,
        "primary_deforestation_drivers": [
            "Population pressure",
            "Agricultural expansion",
            "Shrimp farming",
        ],
        "regions_of_concern": ["Chittagong Hill Tracts", "Sundarbans edge"],
        "commodity_production": ["wood"],
        "governance_score": 28,
        "enforcement_score": 22,
        "corruption_perception_index": 25,
        "region": "south_asia",
    },

    "RU": {
        "country_code": "RU",
        "country_name": "Russian Federation",
        "ec_risk_classification": "standard",
        "base_risk_score": 48,
        "forest_cover_pct": 49.8,
        "annual_deforestation_rate": 0.00,
        "primary_forest_pct": 26.3,
        "primary_deforestation_drivers": [
            "Logging",
            "Wildfires",
            "Mining",
            "Infrastructure",
        ],
        "regions_of_concern": [
            "Far East", "Siberia", "Arkhangelsk",
        ],
        "commodity_production": ["wood", "soya"],
        "governance_score": 28,
        "enforcement_score": 25,
        "corruption_perception_index": 28,
        "region": "eastern_europe",
    },

    # ===================================================================
    # LOW RISK COUNTRIES (20+)
    # ===================================================================

    "DE": {
        "country_code": "DE",
        "country_name": "Germany",
        "ec_risk_classification": "low",
        "base_risk_score": 8,
        "forest_cover_pct": 32.7,
        "annual_deforestation_rate": 0.10,
        "primary_forest_pct": 0.0,
        "primary_deforestation_drivers": [
            "Infrastructure",
            "Climate change impacts",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 92,
        "enforcement_score": 90,
        "corruption_perception_index": 79,
        "region": "western_europe",
    },

    "FR": {
        "country_code": "FR",
        "country_name": "France",
        "ec_risk_classification": "low",
        "base_risk_score": 10,
        "forest_cover_pct": 31.0,
        "annual_deforestation_rate": 0.80,
        "primary_forest_pct": 0.1,
        "primary_deforestation_drivers": [
            "Urbanization",
            "Infrastructure",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood", "cattle"],
        "governance_score": 88,
        "enforcement_score": 88,
        "corruption_perception_index": 71,
        "region": "western_europe",
    },

    "NL": {
        "country_code": "NL",
        "country_name": "Netherlands",
        "ec_risk_classification": "low",
        "base_risk_score": 5,
        "forest_cover_pct": 11.2,
        "annual_deforestation_rate": 0.10,
        "primary_forest_pct": 0.0,
        "primary_deforestation_drivers": ["Urbanization"],
        "regions_of_concern": [],
        "commodity_production": [],
        "governance_score": 94,
        "enforcement_score": 92,
        "corruption_perception_index": 79,
        "region": "western_europe",
    },

    "BE": {
        "country_code": "BE",
        "country_name": "Belgium",
        "ec_risk_classification": "low",
        "base_risk_score": 6,
        "forest_cover_pct": 22.8,
        "annual_deforestation_rate": 0.00,
        "primary_forest_pct": 0.0,
        "primary_deforestation_drivers": ["Urbanization"],
        "regions_of_concern": [],
        "commodity_production": [],
        "governance_score": 90,
        "enforcement_score": 88,
        "corruption_perception_index": 73,
        "region": "western_europe",
    },

    "IT": {
        "country_code": "IT",
        "country_name": "Italy",
        "ec_risk_classification": "low",
        "base_risk_score": 10,
        "forest_cover_pct": 32.0,
        "annual_deforestation_rate": 0.60,
        "primary_forest_pct": 0.5,
        "primary_deforestation_drivers": [
            "Wildfires",
            "Urbanization",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 80,
        "enforcement_score": 78,
        "corruption_perception_index": 56,
        "region": "western_europe",
    },

    "ES": {
        "country_code": "ES",
        "country_name": "Spain",
        "ec_risk_classification": "low",
        "base_risk_score": 10,
        "forest_cover_pct": 36.8,
        "annual_deforestation_rate": 0.20,
        "primary_forest_pct": 0.2,
        "primary_deforestation_drivers": [
            "Wildfires",
            "Urbanization",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 82,
        "enforcement_score": 82,
        "corruption_perception_index": 60,
        "region": "western_europe",
    },

    "PL": {
        "country_code": "PL",
        "country_name": "Poland",
        "ec_risk_classification": "low",
        "base_risk_score": 12,
        "forest_cover_pct": 30.8,
        "annual_deforestation_rate": 0.10,
        "primary_forest_pct": 0.5,
        "primary_deforestation_drivers": [
            "Infrastructure",
            "Managed forestry",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 74,
        "enforcement_score": 72,
        "corruption_perception_index": 55,
        "region": "eastern_europe",
    },

    "SE": {
        "country_code": "SE",
        "country_name": "Sweden",
        "ec_risk_classification": "low",
        "base_risk_score": 5,
        "forest_cover_pct": 68.7,
        "annual_deforestation_rate": 0.10,
        "primary_forest_pct": 2.3,
        "primary_deforestation_drivers": [
            "Managed forestry rotation",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 96,
        "enforcement_score": 95,
        "corruption_perception_index": 83,
        "region": "northern_europe",
    },

    "FI": {
        "country_code": "FI",
        "country_name": "Finland",
        "ec_risk_classification": "low",
        "base_risk_score": 5,
        "forest_cover_pct": 73.1,
        "annual_deforestation_rate": 0.00,
        "primary_forest_pct": 0.5,
        "primary_deforestation_drivers": [
            "Managed forestry",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 96,
        "enforcement_score": 96,
        "corruption_perception_index": 87,
        "region": "northern_europe",
    },

    "AT": {
        "country_code": "AT",
        "country_name": "Austria",
        "ec_risk_classification": "low",
        "base_risk_score": 6,
        "forest_cover_pct": 47.3,
        "annual_deforestation_rate": 0.10,
        "primary_forest_pct": 0.0,
        "primary_deforestation_drivers": [
            "Infrastructure",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 92,
        "enforcement_score": 90,
        "corruption_perception_index": 71,
        "region": "western_europe",
    },

    "PT": {
        "country_code": "PT",
        "country_name": "Portugal",
        "ec_risk_classification": "low",
        "base_risk_score": 12,
        "forest_cover_pct": 36.1,
        "annual_deforestation_rate": -0.10,
        "primary_forest_pct": 0.5,
        "primary_deforestation_drivers": [
            "Wildfires",
            "Eucalyptus expansion",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 82,
        "enforcement_score": 80,
        "corruption_perception_index": 62,
        "region": "western_europe",
    },

    "IE": {
        "country_code": "IE",
        "country_name": "Ireland",
        "ec_risk_classification": "low",
        "base_risk_score": 5,
        "forest_cover_pct": 11.0,
        "annual_deforestation_rate": 1.00,
        "primary_forest_pct": 0.0,
        "primary_deforestation_drivers": [],
        "regions_of_concern": [],
        "commodity_production": ["cattle", "wood"],
        "governance_score": 90,
        "enforcement_score": 88,
        "corruption_perception_index": 77,
        "region": "western_europe",
    },

    "DK": {
        "country_code": "DK",
        "country_name": "Denmark",
        "ec_risk_classification": "low",
        "base_risk_score": 4,
        "forest_cover_pct": 14.6,
        "annual_deforestation_rate": 0.50,
        "primary_forest_pct": 0.0,
        "primary_deforestation_drivers": [],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 96,
        "enforcement_score": 95,
        "corruption_perception_index": 90,
        "region": "northern_europe",
    },

    "US": {
        "country_code": "US",
        "country_name": "United States",
        "ec_risk_classification": "low",
        "base_risk_score": 15,
        "forest_cover_pct": 33.9,
        "annual_deforestation_rate": 0.00,
        "primary_forest_pct": 7.5,
        "primary_deforestation_drivers": [
            "Urbanization",
            "Wildfires",
            "Energy development",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood", "soya", "cattle"],
        "governance_score": 82,
        "enforcement_score": 82,
        "corruption_perception_index": 69,
        "region": "north_america",
    },

    "CA": {
        "country_code": "CA",
        "country_name": "Canada",
        "ec_risk_classification": "low",
        "base_risk_score": 12,
        "forest_cover_pct": 38.7,
        "annual_deforestation_rate": 0.00,
        "primary_forest_pct": 25.5,
        "primary_deforestation_drivers": [
            "Wildfires",
            "Oil sands development",
            "Mining",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood", "soya", "cattle"],
        "governance_score": 90,
        "enforcement_score": 88,
        "corruption_perception_index": 74,
        "region": "north_america",
    },

    "AU": {
        "country_code": "AU",
        "country_name": "Australia",
        "ec_risk_classification": "low",
        "base_risk_score": 18,
        "forest_cover_pct": 16.2,
        "annual_deforestation_rate": -0.10,
        "primary_forest_pct": 14.7,
        "primary_deforestation_drivers": [
            "Bushfires",
            "Agricultural clearing",
            "Mining",
        ],
        "regions_of_concern": [],
        "commodity_production": ["cattle", "wood"],
        "governance_score": 84,
        "enforcement_score": 82,
        "corruption_perception_index": 75,
        "region": "oceania",
    },

    "NZ": {
        "country_code": "NZ",
        "country_name": "New Zealand",
        "ec_risk_classification": "low",
        "base_risk_score": 8,
        "forest_cover_pct": 38.6,
        "annual_deforestation_rate": 0.10,
        "primary_forest_pct": 15.0,
        "primary_deforestation_drivers": [
            "Plantation harvest cycles",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood", "cattle"],
        "governance_score": 94,
        "enforcement_score": 92,
        "corruption_perception_index": 87,
        "region": "oceania",
    },

    "JP": {
        "country_code": "JP",
        "country_name": "Japan",
        "ec_risk_classification": "low",
        "base_risk_score": 8,
        "forest_cover_pct": 68.4,
        "annual_deforestation_rate": 0.00,
        "primary_forest_pct": 5.8,
        "primary_deforestation_drivers": [
            "Infrastructure",
            "Natural disasters",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 88,
        "enforcement_score": 88,
        "corruption_perception_index": 73,
        "region": "east_asia",
    },

    "KR": {
        "country_code": "KR",
        "country_name": "South Korea",
        "ec_risk_classification": "low",
        "base_risk_score": 8,
        "forest_cover_pct": 63.7,
        "annual_deforestation_rate": 0.10,
        "primary_forest_pct": 0.5,
        "primary_deforestation_drivers": [
            "Infrastructure",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 82,
        "enforcement_score": 80,
        "corruption_perception_index": 63,
        "region": "east_asia",
    },

    "GB": {
        "country_code": "GB",
        "country_name": "United Kingdom",
        "ec_risk_classification": "low",
        "base_risk_score": 6,
        "forest_cover_pct": 13.2,
        "annual_deforestation_rate": 0.50,
        "primary_forest_pct": 0.0,
        "primary_deforestation_drivers": [],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 92,
        "enforcement_score": 90,
        "corruption_perception_index": 73,
        "region": "western_europe",
    },

    "NO": {
        "country_code": "NO",
        "country_name": "Norway",
        "ec_risk_classification": "low",
        "base_risk_score": 4,
        "forest_cover_pct": 33.2,
        "annual_deforestation_rate": 0.20,
        "primary_forest_pct": 1.0,
        "primary_deforestation_drivers": [
            "Managed forestry",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 96,
        "enforcement_score": 96,
        "corruption_perception_index": 84,
        "region": "northern_europe",
    },

    "CH": {
        "country_code": "CH",
        "country_name": "Switzerland",
        "ec_risk_classification": "low",
        "base_risk_score": 4,
        "forest_cover_pct": 31.9,
        "annual_deforestation_rate": 0.20,
        "primary_forest_pct": 0.0,
        "primary_deforestation_drivers": [],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 96,
        "enforcement_score": 95,
        "corruption_perception_index": 82,
        "region": "western_europe",
    },

    "UY": {
        "country_code": "UY",
        "country_name": "Uruguay",
        "ec_risk_classification": "low",
        "base_risk_score": 18,
        "forest_cover_pct": 11.8,
        "annual_deforestation_rate": 1.50,
        "primary_forest_pct": 1.5,
        "primary_deforestation_drivers": [],
        "regions_of_concern": [],
        "commodity_production": ["cattle", "wood", "soya"],
        "governance_score": 72,
        "enforcement_score": 68,
        "corruption_perception_index": 74,
        "region": "south_america",
    },

    "CL": {
        "country_code": "CL",
        "country_name": "Chile",
        "ec_risk_classification": "low",
        "base_risk_score": 18,
        "forest_cover_pct": 24.4,
        "annual_deforestation_rate": 0.00,
        "primary_forest_pct": 17.3,
        "primary_deforestation_drivers": [
            "Wildfires",
            "Plantation expansion",
        ],
        "regions_of_concern": [],
        "commodity_production": ["wood"],
        "governance_score": 72,
        "enforcement_score": 68,
        "corruption_perception_index": 67,
        "region": "south_america",
    },
}

# ---------------------------------------------------------------------------
# Totals
# ---------------------------------------------------------------------------

TOTAL_COUNTRIES: int = len(COUNTRY_RISK_DATABASE)

# ---------------------------------------------------------------------------
# Accessor Functions
# ---------------------------------------------------------------------------


def get_country_risk(country_code: str) -> Optional[CountryRiskRecord]:
    """Get the full risk record for a country by ISO alpha-2 code.

    Args:
        country_code: ISO 3166-1 alpha-2 country code (e.g. 'BR', 'ID').

    Returns:
        Dictionary with risk data, or None if not found.

    Example:
        >>> data = get_country_risk("BR")
        >>> assert data["base_risk_score"] == 78
    """
    if not country_code:
        return None
    return COUNTRY_RISK_DATABASE.get(country_code.upper())


def list_countries_by_risk(
    risk_level: str,
) -> List[CountryRiskRecord]:
    """List all countries with a given EC risk classification.

    Args:
        risk_level: One of 'low', 'standard', 'high'.

    Returns:
        List of country risk records sorted by base_risk_score descending.

    Example:
        >>> high = list_countries_by_risk("high")
        >>> assert all(c["ec_risk_classification"] == "high" for c in high)
    """
    level = risk_level.lower().strip()
    results = [
        r for r in COUNTRY_RISK_DATABASE.values()
        if r["ec_risk_classification"] == level
    ]
    results.sort(key=lambda x: x["base_risk_score"], reverse=True)
    return results


def get_countries_for_commodity(
    commodity: str,
) -> List[CountryRiskRecord]:
    """Get all countries that produce a given EUDR commodity.

    Args:
        commodity: EUDR commodity key (e.g. 'cocoa', 'oil_palm').

    Returns:
        List of country records sorted by base_risk_score descending.

    Example:
        >>> cocoa = get_countries_for_commodity("cocoa")
        >>> assert any(c["country_code"] == "CI" for c in cocoa)
    """
    commodity_lower = commodity.lower().strip()
    results = [
        r for r in COUNTRY_RISK_DATABASE.values()
        if commodity_lower in r.get("commodity_production", [])
    ]
    results.sort(key=lambda x: x["base_risk_score"], reverse=True)
    return results


def is_high_risk(country_code: str) -> bool:
    """Check whether a country is classified as high risk.

    Args:
        country_code: ISO 3166-1 alpha-2 code.

    Returns:
        True if the country is classified as high risk. False for unknown.

    Example:
        >>> assert is_high_risk("BR") is True
        >>> assert is_high_risk("DE") is False
    """
    record = get_country_risk(country_code)
    if record is None:
        return False
    return record["ec_risk_classification"] == "high"


def get_regional_context(
    country_code: str,
) -> List[CountryRiskRecord]:
    """Get all countries in the same region as the given country.

    Args:
        country_code: ISO 3166-1 alpha-2 code.

    Returns:
        List of country records in the same region, sorted by risk score.

    Example:
        >>> peers = get_regional_context("BR")
        >>> assert all(c["region"] == "south_america" for c in peers)
    """
    record = get_country_risk(country_code)
    if record is None:
        return []
    region = record["region"]
    results = [
        r for r in COUNTRY_RISK_DATABASE.values()
        if r["region"] == region
    ]
    results.sort(key=lambda x: x["base_risk_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "COUNTRY_RISK_DATABASE",
    "EUDR_COMMODITIES",
    "TOTAL_COUNTRIES",
    "DATA_VERSION",
    "DATA_SOURCES",
    "CountryRiskRecord",
    "get_country_risk",
    "list_countries_by_risk",
    "get_countries_for_commodity",
    "is_high_risk",
    "get_regional_context",
]
