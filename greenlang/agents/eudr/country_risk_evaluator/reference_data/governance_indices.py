# -*- coding: utf-8 -*-
"""
Governance Indices Reference Data - AGENT-EUDR-016

World Bank Worldwide Governance Indicators (WGI), Transparency International
Corruption Perceptions Index (CPI), and forest governance scores for 50+
countries relevant to EUDR compliance.

Data Sources:
    - World Bank WGI 2024 (6 dimensions: voice_accountability, political_stability,
      government_effectiveness, regulatory_quality, rule_of_law, control_of_corruption)
    - Transparency International CPI 2024 (0-100, higher = less corrupt)
    - FAO/ITTO Forest Governance Framework (forest_law_quality, enforcement_capacity,
      institutional_strength)
    - Custom enforcement effectiveness scores (prosecution_rate, penalty_adequacy,
      response_time_days)

All scores are normalized to 0-100 scale for consistency, with higher scores
indicating better governance/lower corruption/stronger enforcement.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

WGIRecord = Dict[str, float]
ForestGovernanceRecord = Dict[str, float]
EnforcementRecord = Dict[str, Any]

# ---------------------------------------------------------------------------
# WGI Dimensions (6)
# ---------------------------------------------------------------------------

WGI_DIMENSIONS: List[str] = [
    "voice_accountability",
    "political_stability",
    "government_effectiveness",
    "regulatory_quality",
    "rule_of_law",
    "control_of_corruption",
]

FOREST_GOVERNANCE_DIMENSIONS: List[str] = [
    "forest_law_quality",
    "enforcement_capacity",
    "institutional_strength",
]

ENFORCEMENT_DIMENSIONS: List[str] = [
    "prosecution_rate",
    "penalty_adequacy",
    "response_time_days",
]

# ===========================================================================
# WORLD BANK WGI (Worldwide Governance Indicators) 2024
# ===========================================================================
# Scores normalized to 0-100 scale (higher = better governance)
# Based on WGI 2024 percentile rank estimates

WORLD_BANK_WGI: Dict[str, WGIRecord] = {
    # -- High Risk Countries --
    "BRA": {  # Brazil
        "voice_accountability": 55.2,
        "political_stability": 38.1,
        "government_effectiveness": 47.6,
        "regulatory_quality": 52.4,
        "rule_of_law": 45.7,
        "control_of_corruption": 42.9,
    },
    "IDN": {  # Indonesia
        "voice_accountability": 42.9,
        "political_stability": 32.4,
        "government_effectiveness": 55.2,
        "regulatory_quality": 49.5,
        "rule_of_law": 41.9,
        "control_of_corruption": 38.1,
    },
    "COD": {  # Democratic Republic of Congo
        "voice_accountability": 12.4,
        "political_stability": 5.7,
        "government_effectiveness": 8.6,
        "regulatory_quality": 10.5,
        "rule_of_law": 6.7,
        "control_of_corruption": 9.5,
    },
    "MYS": {  # Malaysia
        "voice_accountability": 41.9,
        "political_stability": 48.6,
        "government_effectiveness": 72.4,
        "regulatory_quality": 67.6,
        "rule_of_law": 62.9,
        "control_of_corruption": 55.2,
    },
    "COL": {  # Colombia
        "voice_accountability": 48.6,
        "political_stability": 25.7,
        "government_effectiveness": 51.4,
        "regulatory_quality": 61.9,
        "rule_of_law": 38.1,
        "control_of_corruption": 40.0,
    },
    "MMR": {  # Myanmar
        "voice_accountability": 5.7,
        "political_stability": 8.6,
        "government_effectiveness": 12.4,
        "regulatory_quality": 10.5,
        "rule_of_law": 9.5,
        "control_of_corruption": 11.4,
    },
    "KHM": {  # Cambodia
        "voice_accountability": 8.6,
        "political_stability": 28.6,
        "government_effectiveness": 31.4,
        "regulatory_quality": 28.6,
        "rule_of_law": 18.1,
        "control_of_corruption": 15.2,
    },
    "PNG": {  # Papua New Guinea
        "voice_accountability": 35.2,
        "political_stability": 22.9,
        "government_effectiveness": 18.1,
        "regulatory_quality": 25.7,
        "rule_of_law": 20.0,
        "control_of_corruption": 17.1,
    },
    "LAO": {  # Laos
        "voice_accountability": 4.8,
        "political_stability": 38.1,
        "government_effectiveness": 28.6,
        "regulatory_quality": 21.9,
        "rule_of_law": 22.9,
        "control_of_corruption": 19.0,
    },
    "BOL": {  # Bolivia
        "voice_accountability": 32.4,
        "political_stability": 30.5,
        "government_effectiveness": 28.6,
        "regulatory_quality": 20.0,
        "rule_of_law": 25.7,
        "control_of_corruption": 24.8,
    },
    "PRY": {  # Paraguay
        "voice_accountability": 45.7,
        "political_stability": 45.7,
        "government_effectiveness": 31.4,
        "regulatory_quality": 32.4,
        "rule_of_law": 28.6,
        "control_of_corruption": 27.6,
    },
    "PER": {  # Peru
        "voice_accountability": 51.4,
        "political_stability": 18.1,
        "government_effectiveness": 45.7,
        "regulatory_quality": 58.1,
        "rule_of_law": 38.1,
        "control_of_corruption": 40.0,
    },
    "GHA": {  # Ghana
        "voice_accountability": 61.9,
        "political_stability": 48.6,
        "government_effectiveness": 51.4,
        "regulatory_quality": 52.4,
        "rule_of_law": 52.4,
        "control_of_corruption": 48.6,
    },
    "CMR": {  # Cameroon
        "voice_accountability": 15.2,
        "political_stability": 22.9,
        "government_effectiveness": 20.0,
        "regulatory_quality": 25.7,
        "rule_of_law": 18.1,
        "control_of_corruption": 20.0,
    },
    "CIV": {  # Cote d'Ivoire
        "voice_accountability": 28.6,
        "political_stability": 35.2,
        "government_effectiveness": 31.4,
        "regulatory_quality": 38.1,
        "rule_of_law": 30.5,
        "control_of_corruption": 32.4,
    },
    # -- Standard Risk Countries --
    "IND": {  # India
        "voice_accountability": 58.1,
        "political_stability": 25.7,
        "government_effectiveness": 58.1,
        "regulatory_quality": 48.6,
        "rule_of_law": 55.2,
        "control_of_corruption": 42.9,
    },
    "THA": {  # Thailand
        "voice_accountability": 28.6,
        "political_stability": 28.6,
        "government_effectiveness": 61.9,
        "regulatory_quality": 61.9,
        "rule_of_law": 51.4,
        "control_of_corruption": 45.7,
    },
    "VNM": {  # Vietnam
        "voice_accountability": 11.4,
        "political_stability": 58.1,
        "government_effectiveness": 58.1,
        "regulatory_quality": 42.9,
        "rule_of_law": 48.6,
        "control_of_corruption": 41.9,
    },
    "PHL": {  # Philippines
        "voice_accountability": 48.6,
        "political_stability": 22.9,
        "government_effectiveness": 55.2,
        "regulatory_quality": 51.4,
        "rule_of_law": 35.2,
        "control_of_corruption": 38.1,
    },
    "CHN": {  # China
        "voice_accountability": 5.7,
        "political_stability": 45.7,
        "government_effectiveness": 71.4,
        "regulatory_quality": 55.2,
        "rule_of_law": 48.6,
        "control_of_corruption": 52.4,
    },
    "MEX": {  # Mexico
        "voice_accountability": 52.4,
        "political_stability": 28.6,
        "government_effectiveness": 52.4,
        "regulatory_quality": 61.9,
        "rule_of_law": 32.4,
        "control_of_corruption": 31.4,
    },
    "ARG": {  # Argentina
        "voice_accountability": 61.9,
        "political_stability": 42.9,
        "government_effectiveness": 42.9,
        "regulatory_quality": 38.1,
        "rule_of_law": 42.9,
        "control_of_corruption": 42.9,
    },
    "ECU": {  # Ecuador
        "voice_accountability": 41.9,
        "political_stability": 25.7,
        "government_effectiveness": 32.4,
        "regulatory_quality": 28.6,
        "rule_of_law": 25.7,
        "control_of_corruption": 30.5,
    },
    "KEN": {  # Kenya
        "voice_accountability": 45.7,
        "political_stability": 22.9,
        "government_effectiveness": 45.7,
        "regulatory_quality": 51.4,
        "rule_of_law": 42.9,
        "control_of_corruption": 32.4,
    },
    "TZA": {  # Tanzania
        "voice_accountability": 28.6,
        "political_stability": 52.4,
        "government_effectiveness": 41.9,
        "regulatory_quality": 45.7,
        "rule_of_law": 45.7,
        "control_of_corruption": 38.1,
    },
    "UGA": {  # Uganda
        "voice_accountability": 25.7,
        "political_stability": 32.4,
        "government_effectiveness": 35.2,
        "regulatory_quality": 42.9,
        "rule_of_law": 38.1,
        "control_of_corruption": 28.6,
    },
    "ETH": {  # Ethiopia
        "voice_accountability": 15.2,
        "political_stability": 8.6,
        "government_effectiveness": 38.1,
        "regulatory_quality": 28.6,
        "rule_of_law": 32.4,
        "control_of_corruption": 35.2,
    },
    "NGA": {  # Nigeria
        "voice_accountability": 38.1,
        "political_stability": 10.5,
        "government_effectiveness": 22.9,
        "regulatory_quality": 32.4,
        "rule_of_law": 20.0,
        "control_of_corruption": 18.1,
    },
    "GTM": {  # Guatemala
        "voice_accountability": 38.1,
        "political_stability": 38.1,
        "government_effectiveness": 32.4,
        "regulatory_quality": 42.9,
        "rule_of_law": 25.7,
        "control_of_corruption": 28.6,
    },
    "HND": {  # Honduras
        "voice_accountability": 41.9,
        "political_stability": 42.9,
        "government_effectiveness": 28.6,
        "regulatory_quality": 38.1,
        "rule_of_law": 22.9,
        "control_of_corruption": 25.7,
    },
    "CRI": {  # Costa Rica
        "voice_accountability": 74.3,
        "political_stability": 65.7,
        "government_effectiveness": 61.9,
        "regulatory_quality": 65.7,
        "rule_of_law": 61.9,
        "control_of_corruption": 61.9,
    },
    "ZAF": {  # South Africa
        "voice_accountability": 65.7,
        "political_stability": 45.7,
        "government_effectiveness": 61.9,
        "regulatory_quality": 61.9,
        "rule_of_law": 58.1,
        "control_of_corruption": 52.4,
    },
    # -- Low Risk Countries --
    "DEU": {  # Germany
        "voice_accountability": 91.4,
        "political_stability": 78.1,
        "government_effectiveness": 91.4,
        "regulatory_quality": 91.4,
        "rule_of_law": 91.4,
        "control_of_corruption": 91.4,
    },
    "FRA": {  # France
        "voice_accountability": 88.6,
        "political_stability": 61.9,
        "government_effectiveness": 85.7,
        "regulatory_quality": 85.7,
        "rule_of_law": 88.6,
        "control_of_corruption": 85.7,
    },
    "GBR": {  # United Kingdom
        "voice_accountability": 91.4,
        "political_stability": 65.7,
        "government_effectiveness": 88.6,
        "regulatory_quality": 94.3,
        "rule_of_law": 91.4,
        "control_of_corruption": 88.6,
    },
    "USA": {  # United States
        "voice_accountability": 85.7,
        "political_stability": 58.1,
        "government_effectiveness": 85.7,
        "regulatory_quality": 88.6,
        "rule_of_law": 85.7,
        "control_of_corruption": 82.9,
    },
    "CAN": {  # Canada
        "voice_accountability": 94.3,
        "political_stability": 78.1,
        "government_effectiveness": 91.4,
        "regulatory_quality": 91.4,
        "rule_of_law": 91.4,
        "control_of_corruption": 91.4,
    },
    "AUS": {  # Australia
        "voice_accountability": 94.3,
        "political_stability": 78.1,
        "government_effectiveness": 91.4,
        "regulatory_quality": 94.3,
        "rule_of_law": 94.3,
        "control_of_corruption": 88.6,
    },
    "NZL": {  # New Zealand
        "voice_accountability": 97.1,
        "political_stability": 91.4,
        "government_effectiveness": 94.3,
        "regulatory_quality": 97.1,
        "rule_of_law": 97.1,
        "control_of_corruption": 97.1,
    },
    "JPN": {  # Japan
        "voice_accountability": 78.1,
        "political_stability": 85.7,
        "government_effectiveness": 85.7,
        "regulatory_quality": 85.7,
        "rule_of_law": 88.6,
        "control_of_corruption": 85.7,
    },
    "KOR": {  # South Korea
        "voice_accountability": 74.3,
        "political_stability": 61.9,
        "government_effectiveness": 82.9,
        "regulatory_quality": 82.9,
        "rule_of_law": 82.9,
        "control_of_corruption": 71.4,
    },
    "NOR": {  # Norway
        "voice_accountability": 97.1,
        "political_stability": 91.4,
        "government_effectiveness": 94.3,
        "regulatory_quality": 91.4,
        "rule_of_law": 97.1,
        "control_of_corruption": 97.1,
    },
    "SWE": {  # Sweden
        "voice_accountability": 97.1,
        "political_stability": 85.7,
        "government_effectiveness": 94.3,
        "regulatory_quality": 94.3,
        "rule_of_law": 94.3,
        "control_of_corruption": 97.1,
    },
    "CHE": {  # Switzerland
        "voice_accountability": 94.3,
        "political_stability": 91.4,
        "government_effectiveness": 94.3,
        "regulatory_quality": 91.4,
        "rule_of_law": 94.3,
        "control_of_corruption": 94.3,
    },
    "NLD": {  # Netherlands
        "voice_accountability": 94.3,
        "political_stability": 78.1,
        "government_effectiveness": 91.4,
        "regulatory_quality": 94.3,
        "rule_of_law": 91.4,
        "control_of_corruption": 91.4,
    },
    "DNK": {  # Denmark
        "voice_accountability": 97.1,
        "political_stability": 85.7,
        "government_effectiveness": 94.3,
        "regulatory_quality": 97.1,
        "rule_of_law": 97.1,
        "control_of_corruption": 97.1,
    },
    "FIN": {  # Finland
        "voice_accountability": 97.1,
        "political_stability": 88.6,
        "government_effectiveness": 97.1,
        "regulatory_quality": 94.3,
        "rule_of_law": 97.1,
        "control_of_corruption": 97.1,
    },
}

# ===========================================================================
# TRANSPARENCY INTERNATIONAL CPI (Corruption Perceptions Index) 2024
# ===========================================================================
# Scores 0-100 (higher = less corrupt)

TI_CPI_SCORES: Dict[str, float] = {
    # -- High Risk Countries --
    "BRA": 36.0,
    "IDN": 34.0,
    "COD": 20.0,
    "MYS": 50.0,
    "COL": 40.0,
    "MMR": 23.0,
    "KHM": 24.0,
    "PNG": 31.0,
    "LAO": 30.0,
    "BOL": 30.0,
    "PRY": 31.0,
    "PER": 38.0,
    "GHA": 43.0,
    "CMR": 26.0,
    "CIV": 36.0,
    # -- Standard Risk Countries --
    "IND": 40.0,
    "THA": 35.0,
    "VNM": 41.0,
    "PHL": 34.0,
    "CHN": 42.0,
    "MEX": 31.0,
    "ARG": 38.0,
    "ECU": 36.0,
    "KEN": 32.0,
    "TZA": 38.0,
    "UGA": 26.0,
    "ETH": 38.0,
    "NGA": 25.0,
    "GTM": 24.0,
    "HND": 23.0,
    "CRI": 58.0,
    "ZAF": 43.0,
    # -- Low Risk Countries --
    "DEU": 78.0,
    "FRA": 72.0,
    "GBR": 71.0,
    "USA": 69.0,
    "CAN": 76.0,
    "AUS": 75.0,
    "NZL": 87.0,
    "JPN": 73.0,
    "KOR": 63.0,
    "NOR": 84.0,
    "SWE": 82.0,
    "CHE": 82.0,
    "NLD": 79.0,
    "DNK": 90.0,
    "FIN": 87.0,
}

# ===========================================================================
# FOREST GOVERNANCE SCORES (FAO/ITTO-based)
# ===========================================================================
# Custom composite scores for forest governance framework strength
# Scores 0-100 (higher = better forest governance)

FOREST_GOVERNANCE_SCORES: Dict[str, ForestGovernanceRecord] = {
    # -- High Risk Countries --
    "BRA": {
        "forest_law_quality": 65.0,  # Strong laws but weak implementation
        "enforcement_capacity": 35.0,  # Limited enforcement resources
        "institutional_strength": 48.0,  # IBAMA exists but underfunded
    },
    "IDN": {
        "forest_law_quality": 58.0,
        "enforcement_capacity": 28.0,
        "institutional_strength": 42.0,
    },
    "COD": {
        "forest_law_quality": 25.0,
        "enforcement_capacity": 8.0,
        "institutional_strength": 12.0,
    },
    "MYS": {
        "forest_law_quality": 72.0,
        "enforcement_capacity": 55.0,
        "institutional_strength": 68.0,
    },
    "COL": {
        "forest_law_quality": 60.0,
        "enforcement_capacity": 32.0,
        "institutional_strength": 45.0,
    },
    "MMR": {
        "forest_law_quality": 20.0,
        "enforcement_capacity": 10.0,
        "institutional_strength": 15.0,
    },
    "KHM": {
        "forest_law_quality": 28.0,
        "enforcement_capacity": 12.0,
        "institutional_strength": 18.0,
    },
    "PNG": {
        "forest_law_quality": 35.0,
        "enforcement_capacity": 15.0,
        "institutional_strength": 22.0,
    },
    "LAO": {
        "forest_law_quality": 30.0,
        "enforcement_capacity": 18.0,
        "institutional_strength": 25.0,
    },
    "BOL": {
        "forest_law_quality": 45.0,
        "enforcement_capacity": 25.0,
        "institutional_strength": 35.0,
    },
    "PRY": {
        "forest_law_quality": 42.0,
        "enforcement_capacity": 28.0,
        "institutional_strength": 32.0,
    },
    "PER": {
        "forest_law_quality": 55.0,
        "enforcement_capacity": 35.0,
        "institutional_strength": 45.0,
    },
    "GHA": {
        "forest_law_quality": 58.0,
        "enforcement_capacity": 42.0,
        "institutional_strength": 52.0,
    },
    "CMR": {
        "forest_law_quality": 38.0,
        "enforcement_capacity": 18.0,
        "institutional_strength": 25.0,
    },
    "CIV": {
        "forest_law_quality": 48.0,
        "enforcement_capacity": 28.0,
        "institutional_strength": 35.0,
    },
    # -- Standard Risk Countries --
    "IND": {
        "forest_law_quality": 68.0,
        "enforcement_capacity": 52.0,
        "institutional_strength": 62.0,
    },
    "THA": {
        "forest_law_quality": 65.0,
        "enforcement_capacity": 48.0,
        "institutional_strength": 58.0,
    },
    "VNM": {
        "forest_law_quality": 62.0,
        "enforcement_capacity": 45.0,
        "institutional_strength": 55.0,
    },
    "PHL": {
        "forest_law_quality": 55.0,
        "enforcement_capacity": 38.0,
        "institutional_strength": 48.0,
    },
    "CHN": {
        "forest_law_quality": 70.0,
        "enforcement_capacity": 65.0,
        "institutional_strength": 72.0,
    },
    "MEX": {
        "forest_law_quality": 62.0,
        "enforcement_capacity": 42.0,
        "institutional_strength": 52.0,
    },
    "ARG": {
        "forest_law_quality": 58.0,
        "enforcement_capacity": 45.0,
        "institutional_strength": 52.0,
    },
    "ECU": {
        "forest_law_quality": 48.0,
        "enforcement_capacity": 32.0,
        "institutional_strength": 42.0,
    },
    "KEN": {
        "forest_law_quality": 55.0,
        "enforcement_capacity": 38.0,
        "institutional_strength": 48.0,
    },
    "TZA": {
        "forest_law_quality": 52.0,
        "enforcement_capacity": 42.0,
        "institutional_strength": 48.0,
    },
    "UGA": {
        "forest_law_quality": 48.0,
        "enforcement_capacity": 35.0,
        "institutional_strength": 42.0,
    },
    "ETH": {
        "forest_law_quality": 48.0,
        "enforcement_capacity": 32.0,
        "institutional_strength": 42.0,
    },
    "NGA": {
        "forest_law_quality": 42.0,
        "enforcement_capacity": 22.0,
        "institutional_strength": 32.0,
    },
    "GTM": {
        "forest_law_quality": 45.0,
        "enforcement_capacity": 28.0,
        "institutional_strength": 38.0,
    },
    "HND": {
        "forest_law_quality": 42.0,
        "enforcement_capacity": 25.0,
        "institutional_strength": 35.0,
    },
    "CRI": {
        "forest_law_quality": 82.0,
        "enforcement_capacity": 72.0,
        "institutional_strength": 78.0,
    },
    "ZAF": {
        "forest_law_quality": 75.0,
        "enforcement_capacity": 62.0,
        "institutional_strength": 68.0,
    },
    # -- Low Risk Countries --
    "DEU": {
        "forest_law_quality": 95.0,
        "enforcement_capacity": 92.0,
        "institutional_strength": 94.0,
    },
    "FRA": {
        "forest_law_quality": 92.0,
        "enforcement_capacity": 88.0,
        "institutional_strength": 90.0,
    },
    "GBR": {
        "forest_law_quality": 94.0,
        "enforcement_capacity": 90.0,
        "institutional_strength": 92.0,
    },
    "USA": {
        "forest_law_quality": 90.0,
        "enforcement_capacity": 85.0,
        "institutional_strength": 88.0,
    },
    "CAN": {
        "forest_law_quality": 92.0,
        "enforcement_capacity": 88.0,
        "institutional_strength": 90.0,
    },
    "AUS": {
        "forest_law_quality": 91.0,
        "enforcement_capacity": 87.0,
        "institutional_strength": 89.0,
    },
    "NZL": {
        "forest_law_quality": 94.0,
        "enforcement_capacity": 91.0,
        "institutional_strength": 93.0,
    },
    "JPN": {
        "forest_law_quality": 88.0,
        "enforcement_capacity": 85.0,
        "institutional_strength": 87.0,
    },
    "KOR": {
        "forest_law_quality": 85.0,
        "enforcement_capacity": 82.0,
        "institutional_strength": 84.0,
    },
    "NOR": {
        "forest_law_quality": 95.0,
        "enforcement_capacity": 93.0,
        "institutional_strength": 94.0,
    },
    "SWE": {
        "forest_law_quality": 96.0,
        "enforcement_capacity": 94.0,
        "institutional_strength": 95.0,
    },
    "CHE": {
        "forest_law_quality": 94.0,
        "enforcement_capacity": 92.0,
        "institutional_strength": 93.0,
    },
    "NLD": {
        "forest_law_quality": 93.0,
        "enforcement_capacity": 90.0,
        "institutional_strength": 92.0,
    },
    "DNK": {
        "forest_law_quality": 96.0,
        "enforcement_capacity": 94.0,
        "institutional_strength": 95.0,
    },
    "FIN": {
        "forest_law_quality": 97.0,
        "enforcement_capacity": 95.0,
        "institutional_strength": 96.0,
    },
}

# ===========================================================================
# ENFORCEMENT EFFECTIVENESS SCORES
# ===========================================================================
# Custom enforcement effectiveness metrics

ENFORCEMENT_EFFECTIVENESS: Dict[str, EnforcementRecord] = {
    # -- High Risk Countries --
    "BRA": {
        "prosecution_rate": 12.5,  # % of detected violations prosecuted
        "penalty_adequacy": 25.0,  # % of penalties that are adequate deterrents
        "response_time_days": 180,  # Average days from detection to enforcement action
    },
    "IDN": {
        "prosecution_rate": 8.0,
        "penalty_adequacy": 18.0,
        "response_time_days": 240,
    },
    "COD": {
        "prosecution_rate": 2.0,
        "penalty_adequacy": 5.0,
        "response_time_days": 365,
    },
    "MYS": {
        "prosecution_rate": 35.0,
        "penalty_adequacy": 48.0,
        "response_time_days": 90,
    },
    "COL": {
        "prosecution_rate": 15.0,
        "penalty_adequacy": 28.0,
        "response_time_days": 150,
    },
    "MMR": {
        "prosecution_rate": 3.0,
        "penalty_adequacy": 8.0,
        "response_time_days": 300,
    },
    "KHM": {
        "prosecution_rate": 5.0,
        "penalty_adequacy": 12.0,
        "response_time_days": 270,
    },
    "PNG": {
        "prosecution_rate": 6.0,
        "penalty_adequacy": 15.0,
        "response_time_days": 210,
    },
    "LAO": {
        "prosecution_rate": 7.0,
        "penalty_adequacy": 18.0,
        "response_time_days": 240,
    },
    "BOL": {
        "prosecution_rate": 10.0,
        "penalty_adequacy": 22.0,
        "response_time_days": 180,
    },
    "PRY": {
        "prosecution_rate": 11.0,
        "penalty_adequacy": 25.0,
        "response_time_days": 165,
    },
    "PER": {
        "prosecution_rate": 18.0,
        "penalty_adequacy": 32.0,
        "response_time_days": 135,
    },
    "GHA": {
        "prosecution_rate": 22.0,
        "penalty_adequacy": 38.0,
        "response_time_days": 120,
    },
    "CMR": {
        "prosecution_rate": 8.0,
        "penalty_adequacy": 18.0,
        "response_time_days": 210,
    },
    "CIV": {
        "prosecution_rate": 12.0,
        "penalty_adequacy": 25.0,
        "response_time_days": 180,
    },
    # -- Standard Risk Countries --
    "IND": {
        "prosecution_rate": 28.0,
        "penalty_adequacy": 42.0,
        "response_time_days": 105,
    },
    "THA": {
        "prosecution_rate": 25.0,
        "penalty_adequacy": 38.0,
        "response_time_days": 120,
    },
    "VNM": {
        "prosecution_rate": 30.0,
        "penalty_adequacy": 45.0,
        "response_time_days": 90,
    },
    "PHL": {
        "prosecution_rate": 20.0,
        "penalty_adequacy": 35.0,
        "response_time_days": 135,
    },
    "CHN": {
        "prosecution_rate": 45.0,
        "penalty_adequacy": 58.0,
        "response_time_days": 60,
    },
    "MEX": {
        "prosecution_rate": 22.0,
        "penalty_adequacy": 38.0,
        "response_time_days": 120,
    },
    "ARG": {
        "prosecution_rate": 25.0,
        "penalty_adequacy": 40.0,
        "response_time_days": 105,
    },
    "ECU": {
        "prosecution_rate": 15.0,
        "penalty_adequacy": 28.0,
        "response_time_days": 150,
    },
    "KEN": {
        "prosecution_rate": 20.0,
        "penalty_adequacy": 35.0,
        "response_time_days": 135,
    },
    "TZA": {
        "prosecution_rate": 22.0,
        "penalty_adequacy": 38.0,
        "response_time_days": 120,
    },
    "UGA": {
        "prosecution_rate": 18.0,
        "penalty_adequacy": 32.0,
        "response_time_days": 150,
    },
    "ETH": {
        "prosecution_rate": 20.0,
        "penalty_adequacy": 35.0,
        "response_time_days": 135,
    },
    "NGA": {
        "prosecution_rate": 12.0,
        "penalty_adequacy": 25.0,
        "response_time_days": 180,
    },
    "GTM": {
        "prosecution_rate": 15.0,
        "penalty_adequacy": 28.0,
        "response_time_days": 150,
    },
    "HND": {
        "prosecution_rate": 14.0,
        "penalty_adequacy": 26.0,
        "response_time_days": 165,
    },
    "CRI": {
        "prosecution_rate": 58.0,
        "penalty_adequacy": 68.0,
        "response_time_days": 45,
    },
    "ZAF": {
        "prosecution_rate": 42.0,
        "penalty_adequacy": 55.0,
        "response_time_days": 75,
    },
    # -- Low Risk Countries --
    "DEU": {
        "prosecution_rate": 85.0,
        "penalty_adequacy": 88.0,
        "response_time_days": 21,
    },
    "FRA": {
        "prosecution_rate": 82.0,
        "penalty_adequacy": 85.0,
        "response_time_days": 28,
    },
    "GBR": {
        "prosecution_rate": 84.0,
        "penalty_adequacy": 87.0,
        "response_time_days": 24,
    },
    "USA": {
        "prosecution_rate": 78.0,
        "penalty_adequacy": 82.0,
        "response_time_days": 35,
    },
    "CAN": {
        "prosecution_rate": 83.0,
        "penalty_adequacy": 86.0,
        "response_time_days": 26,
    },
    "AUS": {
        "prosecution_rate": 81.0,
        "penalty_adequacy": 84.0,
        "response_time_days": 30,
    },
    "NZL": {
        "prosecution_rate": 87.0,
        "penalty_adequacy": 90.0,
        "response_time_days": 18,
    },
    "JPN": {
        "prosecution_rate": 80.0,
        "penalty_adequacy": 83.0,
        "response_time_days": 32,
    },
    "KOR": {
        "prosecution_rate": 76.0,
        "penalty_adequacy": 80.0,
        "response_time_days": 38,
    },
    "NOR": {
        "prosecution_rate": 88.0,
        "penalty_adequacy": 91.0,
        "response_time_days": 16,
    },
    "SWE": {
        "prosecution_rate": 89.0,
        "penalty_adequacy": 92.0,
        "response_time_days": 14,
    },
    "CHE": {
        "prosecution_rate": 86.0,
        "penalty_adequacy": 89.0,
        "response_time_days": 20,
    },
    "NLD": {
        "prosecution_rate": 85.0,
        "penalty_adequacy": 88.0,
        "response_time_days": 22,
    },
    "DNK": {
        "prosecution_rate": 90.0,
        "penalty_adequacy": 93.0,
        "response_time_days": 12,
    },
    "FIN": {
        "prosecution_rate": 91.0,
        "penalty_adequacy": 94.0,
        "response_time_days": 10,
    },
}

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================


def get_wgi_score(country_code: str) -> Optional[WGIRecord]:
    """
    Get World Bank WGI scores for a country.

    Args:
        country_code: ISO 3166-1 alpha-3 country code (e.g., "BRA")

    Returns:
        Dictionary with 6 WGI dimensions or None if not found
    """
    return WORLD_BANK_WGI.get(country_code)


def get_wgi_dimension(country_code: str, dimension: str) -> Optional[float]:
    """
    Get a specific WGI dimension score for a country.

    Args:
        country_code: ISO 3166-1 alpha-3 country code
        dimension: One of WGI_DIMENSIONS

    Returns:
        Score (0-100) or None if not found
    """
    wgi = WORLD_BANK_WGI.get(country_code)
    if wgi and dimension in wgi:
        return wgi[dimension]
    return None


def get_cpi_score(country_code: str) -> Optional[float]:
    """
    Get Transparency International CPI score for a country.

    Args:
        country_code: ISO 3166-1 alpha-3 country code

    Returns:
        CPI score (0-100, higher = less corrupt) or None if not found
    """
    return TI_CPI_SCORES.get(country_code)


def get_forest_governance(country_code: str) -> Optional[ForestGovernanceRecord]:
    """
    Get forest governance scores for a country.

    Args:
        country_code: ISO 3166-1 alpha-3 country code

    Returns:
        Dictionary with 3 forest governance dimensions or None if not found
    """
    return FOREST_GOVERNANCE_SCORES.get(country_code)


def get_forest_governance_dimension(
    country_code: str, dimension: str
) -> Optional[float]:
    """
    Get a specific forest governance dimension score.

    Args:
        country_code: ISO 3166-1 alpha-3 country code
        dimension: One of FOREST_GOVERNANCE_DIMENSIONS

    Returns:
        Score (0-100) or None if not found
    """
    fg = FOREST_GOVERNANCE_SCORES.get(country_code)
    if fg and dimension in fg:
        return fg[dimension]
    return None


def get_enforcement_score(country_code: str) -> Optional[EnforcementRecord]:
    """
    Get enforcement effectiveness scores for a country.

    Args:
        country_code: ISO 3166-1 alpha-3 country code

    Returns:
        Dictionary with enforcement metrics or None if not found
    """
    return ENFORCEMENT_EFFECTIVENESS.get(country_code)


def get_enforcement_dimension(country_code: str, dimension: str) -> Optional[float]:
    """
    Get a specific enforcement dimension value.

    Args:
        country_code: ISO 3166-1 alpha-3 country code
        dimension: One of ENFORCEMENT_DIMENSIONS

    Returns:
        Value or None if not found
    """
    enf = ENFORCEMENT_EFFECTIVENESS.get(country_code)
    if enf and dimension in enf:
        return enf[dimension]
    return None


def calculate_governance_composite(
    country_code: str,
    *,
    wgi_weight: float = 0.40,
    cpi_weight: float = 0.30,
    forest_governance_weight: float = 0.20,
    enforcement_weight: float = 0.10,
) -> Optional[float]:
    """
    Calculate composite governance score from all sources.

    Args:
        country_code: ISO 3166-1 alpha-3 country code
        wgi_weight: Weight for WGI composite (default 40%)
        cpi_weight: Weight for CPI (default 30%)
        forest_governance_weight: Weight for forest governance (default 20%)
        enforcement_weight: Weight for enforcement (default 10%)

    Returns:
        Composite governance score (0-100) or None if data not available

    Raises:
        ValueError: If weights don't sum to 1.0
    """
    total_weight = wgi_weight + cpi_weight + forest_governance_weight + enforcement_weight
    if abs(total_weight - 1.0) > 0.001:
        raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    # Get WGI composite (average of 6 dimensions)
    wgi = get_wgi_score(country_code)
    if not wgi:
        return None
    wgi_composite = sum(wgi.values()) / len(wgi)

    # Get CPI
    cpi = get_cpi_score(country_code)
    if cpi is None:
        return None

    # Get forest governance composite
    fg = get_forest_governance(country_code)
    if not fg:
        return None
    fg_composite = sum(fg.values()) / len(fg)

    # Get enforcement composite
    enf = get_enforcement_score(country_code)
    if not enf:
        return None
    # Normalize enforcement metrics to 0-100
    enf_composite = (
        (enf["prosecution_rate"] + enf["penalty_adequacy"]) / 2.0
    )  # Average %

    # Calculate weighted composite
    composite = (
        wgi_composite * wgi_weight
        + cpi * cpi_weight
        + fg_composite * forest_governance_weight
        + enf_composite * enforcement_weight
    )

    return round(composite, 2)


__all__ = [
    "WORLD_BANK_WGI",
    "TI_CPI_SCORES",
    "FOREST_GOVERNANCE_SCORES",
    "ENFORCEMENT_EFFECTIVENESS",
    "WGI_DIMENSIONS",
    "FOREST_GOVERNANCE_DIMENSIONS",
    "ENFORCEMENT_DIMENSIONS",
    "WGIRecord",
    "ForestGovernanceRecord",
    "EnforcementRecord",
    "get_wgi_score",
    "get_wgi_dimension",
    "get_cpi_score",
    "get_forest_governance",
    "get_forest_governance_dimension",
    "get_enforcement_score",
    "get_enforcement_dimension",
    "calculate_governance_composite",
]
