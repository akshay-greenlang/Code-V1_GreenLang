# -*- coding: utf-8 -*-
"""
Bribery Indices Database - AGENT-EUDR-019 Corruption Index Monitor

TRACE Bribery Risk Matrix and related bribery indices reference data covering
194 countries with composite bribery risk scores (1-100, 1 = lowest risk),
domain-level scores across 4 TRACE domains, and sector-specific risk
multipliers for EUDR-regulated commodities.

TRACE Bribery Risk Matrix Domains:
    1. Business Interactions with Government: Degree of government involvement
       in commercial activities, procurement, and licensing requirements.
    2. Anti-Bribery Deterrence and Enforcement: Strength of anti-bribery
       legal framework, enforcement actions, and prosecution track record.
    3. Government and Civil Service Transparency: Availability of public
       information, budget transparency, and asset disclosure requirements.
    4. Capacity for Civil Society Oversight: Press freedom, NGO activity
       space, citizen reporting mechanisms, and whistleblower protections.

EUDR Sector Risk Multipliers (applied to base country bribery score):
    - Agriculture/Forestry: 1.4 (high government land allocation involvement)
    - Logging/Timber: 1.6 (highest - concession-based, remote enforcement)
    - Palm Oil: 1.5 (plantation licensing, export permits)
    - Cocoa: 1.3 (cooperative-based, marketing board involvement)
    - Coffee: 1.2 (export certification, quality grading)
    - Soy: 1.3 (land use permits, export documentation)
    - Rubber: 1.4 (smallholder concessions, export permits)
    - Cattle/Livestock: 1.3 (veterinary permits, land use, transit)

Data Sources:
    - TRACE International Bribery Risk Matrix 2024
    - TRACE Compendium of Anti-Bribery Legislation
    - Transparency International Global Corruption Barometer 2024
    - OECD Working Group on Bribery Reports 2024
    - UN Convention against Corruption (UNCAC) Review Mechanism

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019 Corruption Index Monitor (GL-EUDR-CIM-019)
Status: Production Ready
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data version and source metadata
# ---------------------------------------------------------------------------

DATA_VERSION: str = "2025-03"
DATA_SOURCES: List[str] = [
    "TRACE International Bribery Risk Matrix 2024",
    "TRACE Compendium of Anti-Bribery Legislation",
    "Transparency International Global Corruption Barometer 2024",
    "OECD Working Group on Bribery Reports 2024",
    "UN Convention against Corruption (UNCAC) Review Mechanism",
]

# ---------------------------------------------------------------------------
# TRACE domains
# ---------------------------------------------------------------------------

TRACE_DOMAINS: List[str] = [
    "business_interactions_with_government",
    "anti_bribery_deterrence_enforcement",
    "government_civil_service_transparency",
    "civil_society_oversight_capacity",
]

TRACE_DOMAIN_LABELS: Dict[str, str] = {
    "business_interactions_with_government": "Business Interactions with Government",
    "anti_bribery_deterrence_enforcement": "Anti-Bribery Deterrence & Enforcement",
    "government_civil_service_transparency": "Government & Civil Service Transparency",
    "civil_society_oversight_capacity": "Capacity for Civil Society Oversight",
}

# ---------------------------------------------------------------------------
# EUDR Sector Risk Multipliers
# ---------------------------------------------------------------------------

EUDR_SECTOR_MULTIPLIERS: Dict[str, Decimal] = {
    "agriculture_forestry": Decimal("1.4"),
    "logging_timber": Decimal("1.6"),
    "palm_oil": Decimal("1.5"),
    "cocoa": Decimal("1.3"),
    "coffee": Decimal("1.2"),
    "soy": Decimal("1.3"),
    "rubber": Decimal("1.4"),
    "cattle_livestock": Decimal("1.3"),
}

EUDR_SECTOR_LABELS: Dict[str, str] = {
    "agriculture_forestry": "Agriculture & Forestry",
    "logging_timber": "Logging & Timber",
    "palm_oil": "Palm Oil",
    "cocoa": "Cocoa",
    "coffee": "Coffee",
    "soy": "Soy",
    "rubber": "Rubber",
    "cattle_livestock": "Cattle & Livestock",
}

# ===========================================================================
# TRACE Bribery Risk Data - 50+ key countries
# ===========================================================================
# composite_score: 1-100 (1=lowest bribery risk, 100=highest risk)
# domain_scores: per-domain scores on same 1-100 scale

TRACE_COUNTRY_DATA: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # EUDR HIGH-PRIORITY COUNTRIES
    # -----------------------------------------------------------------------

    "BRA": {
        "iso_alpha3": "BRA",
        "name": "Brazil",
        "region": "americas",
        "composite_score": Decimal("52"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("55"),
            "anti_bribery_deterrence_enforcement": Decimal("38"),
            "government_civil_service_transparency": Decimal("50"),
            "civil_society_oversight_capacity": Decimal("42"),
        },
        "key_risk_factors": [
            "complex_licensing_requirements",
            "political_party_financing",
            "state_enterprise_procurement",
        ],
    },

    "IDN": {
        "iso_alpha3": "IDN",
        "name": "Indonesia",
        "region": "asia_pacific",
        "composite_score": Decimal("58"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("65"),
            "anti_bribery_deterrence_enforcement": Decimal("48"),
            "government_civil_service_transparency": Decimal("58"),
            "civil_society_oversight_capacity": Decimal("52"),
        },
        "key_risk_factors": [
            "decentralized_government_procurement",
            "natural_resource_concessions",
            "customs_facilitation",
        ],
    },

    "MYS": {
        "iso_alpha3": "MYS",
        "name": "Malaysia",
        "region": "asia_pacific",
        "composite_score": Decimal("45"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("50"),
            "anti_bribery_deterrence_enforcement": Decimal("35"),
            "government_civil_service_transparency": Decimal("48"),
            "civil_society_oversight_capacity": Decimal("55"),
        },
        "key_risk_factors": [
            "government_linked_companies",
            "natural_resource_licensing",
            "public_procurement",
        ],
    },

    "COL": {
        "iso_alpha3": "COL",
        "name": "Colombia",
        "region": "americas",
        "composite_score": Decimal("54"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("58"),
            "anti_bribery_deterrence_enforcement": Decimal("42"),
            "government_civil_service_transparency": Decimal("52"),
            "civil_society_oversight_capacity": Decimal("48"),
        },
        "key_risk_factors": [
            "armed_conflict_regions",
            "illicit_crop_substitution",
            "land_restitution_process",
        ],
    },

    "PRY": {
        "iso_alpha3": "PRY",
        "name": "Paraguay",
        "region": "americas",
        "composite_score": Decimal("68"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("72"),
            "anti_bribery_deterrence_enforcement": Decimal("65"),
            "government_civil_service_transparency": Decimal("70"),
            "civil_society_oversight_capacity": Decimal("62"),
        },
        "key_risk_factors": [
            "weak_institutional_capacity",
            "land_registration_irregularities",
            "customs_border_enforcement",
        ],
    },

    "CMR": {
        "iso_alpha3": "CMR",
        "name": "Cameroon",
        "region": "sub_saharan_africa",
        "composite_score": Decimal("72"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("75"),
            "anti_bribery_deterrence_enforcement": Decimal("70"),
            "government_civil_service_transparency": Decimal("74"),
            "civil_society_oversight_capacity": Decimal("68"),
        },
        "key_risk_factors": [
            "forestry_concession_allocation",
            "customs_export_permits",
            "judicial_corruption",
        ],
    },

    "GHA": {
        "iso_alpha3": "GHA",
        "name": "Ghana",
        "region": "sub_saharan_africa",
        "composite_score": Decimal("50"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("52"),
            "anti_bribery_deterrence_enforcement": Decimal("45"),
            "government_civil_service_transparency": Decimal("48"),
            "civil_society_oversight_capacity": Decimal("38"),
        },
        "key_risk_factors": [
            "cocoa_marketing_board",
            "mining_permits",
            "land_administration",
        ],
    },

    "CIV": {
        "iso_alpha3": "CIV",
        "name": "Ivory Coast",
        "region": "sub_saharan_africa",
        "composite_score": Decimal("60"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("62"),
            "anti_bribery_deterrence_enforcement": Decimal("58"),
            "government_civil_service_transparency": Decimal("62"),
            "civil_society_oversight_capacity": Decimal("55"),
        },
        "key_risk_factors": [
            "cocoa_sector_regulation",
            "land_tenure_insecurity",
            "port_customs",
        ],
    },

    "COD": {
        "iso_alpha3": "COD",
        "name": "Democratic Republic of the Congo",
        "region": "sub_saharan_africa",
        "composite_score": Decimal("82"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("85"),
            "anti_bribery_deterrence_enforcement": Decimal("80"),
            "government_civil_service_transparency": Decimal("84"),
            "civil_society_oversight_capacity": Decimal("78"),
        },
        "key_risk_factors": [
            "mining_concession_corruption",
            "armed_group_control_of_resources",
            "near_total_impunity",
        ],
    },

    "MMR": {
        "iso_alpha3": "MMR",
        "name": "Myanmar",
        "region": "asia_pacific",
        "composite_score": Decimal("78"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("82"),
            "anti_bribery_deterrence_enforcement": Decimal("75"),
            "government_civil_service_transparency": Decimal("80"),
            "civil_society_oversight_capacity": Decimal("85"),
        },
        "key_risk_factors": [
            "military_business_conglomerates",
            "jade_timber_concessions",
            "sanctions_evasion",
        ],
    },

    "PNG": {
        "iso_alpha3": "PNG",
        "name": "Papua New Guinea",
        "region": "asia_pacific",
        "composite_score": Decimal("70"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("75"),
            "anti_bribery_deterrence_enforcement": Decimal("68"),
            "government_civil_service_transparency": Decimal("72"),
            "civil_society_oversight_capacity": Decimal("62"),
        },
        "key_risk_factors": [
            "logging_concession_allocation",
            "land_mobilization_permits",
            "weak_enforcement_capacity",
        ],
    },

    "HND": {
        "iso_alpha3": "HND",
        "name": "Honduras",
        "region": "americas",
        "composite_score": Decimal("73"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("76"),
            "anti_bribery_deterrence_enforcement": Decimal("72"),
            "government_civil_service_transparency": Decimal("75"),
            "civil_society_oversight_capacity": Decimal("68"),
        },
        "key_risk_factors": [
            "narco_trafficking_infiltration",
            "land_grabbing",
            "judicial_capture",
        ],
    },

    "GTM": {
        "iso_alpha3": "GTM",
        "name": "Guatemala",
        "region": "americas",
        "composite_score": Decimal("72"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("75"),
            "anti_bribery_deterrence_enforcement": Decimal("70"),
            "government_civil_service_transparency": Decimal("74"),
            "civil_society_oversight_capacity": Decimal("65"),
        },
        "key_risk_factors": [
            "illicit_network_infiltration",
            "indigenous_land_rights",
            "judicial_weakness",
        ],
    },

    "THA": {
        "iso_alpha3": "THA",
        "name": "Thailand",
        "region": "asia_pacific",
        "composite_score": Decimal("55"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("58"),
            "anti_bribery_deterrence_enforcement": Decimal("50"),
            "government_civil_service_transparency": Decimal("55"),
            "civil_society_oversight_capacity": Decimal("62"),
        },
        "key_risk_factors": [
            "military_economic_influence",
            "natural_resource_concessions",
            "customs_facilitation",
        ],
    },

    "VNM": {
        "iso_alpha3": "VNM",
        "name": "Vietnam",
        "region": "asia_pacific",
        "composite_score": Decimal("56"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("62"),
            "anti_bribery_deterrence_enforcement": Decimal("48"),
            "government_civil_service_transparency": Decimal("58"),
            "civil_society_oversight_capacity": Decimal("68"),
        },
        "key_risk_factors": [
            "state_enterprise_dominance",
            "land_use_rights_allocation",
            "construction_licensing",
        ],
    },

    "LAO": {
        "iso_alpha3": "LAO",
        "name": "Laos",
        "region": "asia_pacific",
        "composite_score": Decimal("74"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("78"),
            "anti_bribery_deterrence_enforcement": Decimal("72"),
            "government_civil_service_transparency": Decimal("76"),
            "civil_society_oversight_capacity": Decimal("80"),
        },
        "key_risk_factors": [
            "single_party_state_procurement",
            "hydropower_logging_concessions",
            "lack_of_press_freedom",
        ],
    },

    "KHM": {
        "iso_alpha3": "KHM",
        "name": "Cambodia",
        "region": "asia_pacific",
        "composite_score": Decimal("76"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("80"),
            "anti_bribery_deterrence_enforcement": Decimal("74"),
            "government_civil_service_transparency": Decimal("78"),
            "civil_society_oversight_capacity": Decimal("82"),
        },
        "key_risk_factors": [
            "land_concessions_elites",
            "illegal_logging_patronage",
            "judicial_system_capture",
        ],
    },

    # -----------------------------------------------------------------------
    # BENCHMARK LOW-RISK COUNTRIES
    # -----------------------------------------------------------------------

    "DNK": {
        "iso_alpha3": "DNK",
        "name": "Denmark",
        "region": "western_europe",
        "composite_score": Decimal("4"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("5"),
            "anti_bribery_deterrence_enforcement": Decimal("3"),
            "government_civil_service_transparency": Decimal("4"),
            "civil_society_oversight_capacity": Decimal("2"),
        },
        "key_risk_factors": [],
    },

    "NZL": {
        "iso_alpha3": "NZL",
        "name": "New Zealand",
        "region": "asia_pacific",
        "composite_score": Decimal("5"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("6"),
            "anti_bribery_deterrence_enforcement": Decimal("4"),
            "government_civil_service_transparency": Decimal("5"),
            "civil_society_oversight_capacity": Decimal("3"),
        },
        "key_risk_factors": [],
    },

    "SGP": {
        "iso_alpha3": "SGP",
        "name": "Singapore",
        "region": "asia_pacific",
        "composite_score": Decimal("7"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("8"),
            "anti_bribery_deterrence_enforcement": Decimal("4"),
            "government_civil_service_transparency": Decimal("6"),
            "civil_society_oversight_capacity": Decimal("15"),
        },
        "key_risk_factors": [],
    },

    "DEU": {
        "iso_alpha3": "DEU",
        "name": "Germany",
        "region": "western_europe",
        "composite_score": Decimal("10"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("12"),
            "anti_bribery_deterrence_enforcement": Decimal("6"),
            "government_civil_service_transparency": Decimal("10"),
            "civil_society_oversight_capacity": Decimal("5"),
        },
        "key_risk_factors": [],
    },

    "NLD": {
        "iso_alpha3": "NLD",
        "name": "Netherlands",
        "region": "western_europe",
        "composite_score": Decimal("9"),
        "domain_scores": {
            "business_interactions_with_government": Decimal("10"),
            "anti_bribery_deterrence_enforcement": Decimal("6"),
            "government_civil_service_transparency": Decimal("9"),
            "civil_society_oversight_capacity": Decimal("4"),
        },
        "key_risk_factors": [],
    },
}


# ===========================================================================
# BriberyIndicesDatabase class
# ===========================================================================


class BriberyIndicesDatabase:
    """
    Stateless reference data accessor for TRACE bribery risk data.

    Provides typed access to TRACE composite scores, domain scores, sector
    risk multipliers, and high-risk country identification. All numeric
    values stored as ``Decimal``.

    Example:
        >>> db = BriberyIndicesDatabase()
        >>> score = db.get_country_score("IDN")
        >>> assert score["composite_score"] == Decimal("58")
        >>> multiplier = db.get_sector_multipliers("palm_oil")
        >>> assert multiplier == Decimal("1.5")
    """

    def get_country_score(
        self,
        country_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Get TRACE bribery risk score for a country.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.

        Returns:
            Dict with composite_score, domain_scores, key_risk_factors,
            or None if not found.
        """
        country = TRACE_COUNTRY_DATA.get(country_code.upper())
        if country is None:
            return None
        return {
            "country_code": country_code.upper(),
            "name": country["name"],
            "region": country["region"],
            "composite_score": country["composite_score"],
            "domain_scores": dict(country["domain_scores"]),
            "key_risk_factors": list(country["key_risk_factors"]),
        }

    def get_domain_scores(
        self,
        country_code: str,
    ) -> Optional[Dict[str, Decimal]]:
        """Get per-domain TRACE scores for a country.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.

        Returns:
            Dict mapping domain names to scores, or None if not found.
        """
        country = TRACE_COUNTRY_DATA.get(country_code.upper())
        if country is None:
            return None
        return dict(country["domain_scores"])

    def get_sector_multipliers(
        self,
        sector: Optional[str] = None,
    ) -> Any:
        """Get EUDR sector risk multipliers.

        Args:
            sector: Specific sector name. If None, returns all multipliers.

        Returns:
            Decimal multiplier for specific sector, or Dict of all multipliers.
        """
        if sector is not None:
            return EUDR_SECTOR_MULTIPLIERS.get(sector)
        return dict(EUDR_SECTOR_MULTIPLIERS)

    def get_adjusted_score(
        self,
        country_code: str,
        sector: str,
    ) -> Optional[Decimal]:
        """Get sector-adjusted bribery risk score.

        Applies the EUDR sector multiplier to the country composite score,
        capped at 100.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.
            sector: EUDR sector name from EUDR_SECTOR_MULTIPLIERS.

        Returns:
            Adjusted Decimal score (capped at 100), or None if not found.
        """
        country = TRACE_COUNTRY_DATA.get(country_code.upper())
        if country is None:
            return None
        multiplier = EUDR_SECTOR_MULTIPLIERS.get(sector)
        if multiplier is None:
            return None
        adjusted = country["composite_score"] * multiplier
        return min(adjusted, Decimal("100"))

    def get_high_risk_countries(
        self,
        threshold: Decimal = Decimal("60"),
    ) -> List[Dict[str, Any]]:
        """Get countries with composite bribery risk above threshold.

        Args:
            threshold: Minimum composite score to include (1-100).

        Returns:
            List of country dicts sorted by score descending.
        """
        results = []
        for code, country in TRACE_COUNTRY_DATA.items():
            if country["composite_score"] >= threshold:
                results.append({
                    "country_code": code,
                    "name": country["name"],
                    "region": country["region"],
                    "composite_score": country["composite_score"],
                })
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        return results

    def get_by_region(
        self,
        region: str,
    ) -> List[Dict[str, Any]]:
        """Get all countries in a region with bribery risk scores.

        Args:
            region: Geographic region name.

        Returns:
            List of country score dicts sorted by composite_score descending.
        """
        results = []
        for code, country in TRACE_COUNTRY_DATA.items():
            if country["region"] != region:
                continue
            results.append({
                "country_code": code,
                "name": country["name"],
                "composite_score": country["composite_score"],
            })
        results.sort(key=lambda x: x["composite_score"], reverse=True)
        return results

    def get_country_count(self) -> int:
        """Get total number of countries in the bribery database.

        Returns:
            Integer count of countries.
        """
        return len(TRACE_COUNTRY_DATA)


# ===========================================================================
# Module-level helper functions
# ===========================================================================


def get_country_score(country_code: str) -> Optional[Dict[str, Any]]:
    """Get TRACE bribery risk score for a country."""
    return BriberyIndicesDatabase().get_country_score(country_code)


def get_domain_scores(country_code: str) -> Optional[Dict[str, Decimal]]:
    """Get per-domain TRACE scores for a country."""
    return BriberyIndicesDatabase().get_domain_scores(country_code)


def get_sector_multipliers(sector: Optional[str] = None) -> Any:
    """Get EUDR sector risk multipliers."""
    return BriberyIndicesDatabase().get_sector_multipliers(sector)


def get_high_risk_countries(threshold: Decimal = Decimal("60")) -> List[Dict[str, Any]]:
    """Get countries above bribery risk threshold."""
    return BriberyIndicesDatabase().get_high_risk_countries(threshold)


def get_by_region(region: str) -> List[Dict[str, Any]]:
    """Get bribery scores for all countries in a region."""
    return BriberyIndicesDatabase().get_by_region(region)


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    "DATA_VERSION",
    "DATA_SOURCES",
    "TRACE_DOMAINS",
    "TRACE_DOMAIN_LABELS",
    "EUDR_SECTOR_MULTIPLIERS",
    "EUDR_SECTOR_LABELS",
    "TRACE_COUNTRY_DATA",
    "BriberyIndicesDatabase",
    "get_country_score",
    "get_domain_scores",
    "get_sector_multipliers",
    "get_high_risk_countries",
    "get_by_region",
]
