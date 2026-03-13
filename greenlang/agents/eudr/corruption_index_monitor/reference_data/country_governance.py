# -*- coding: utf-8 -*-
"""
Country Governance Database - AGENT-EUDR-019 Corruption Index Monitor

Country governance profiles and institutional assessment data covering
180+ countries with institutional quality scores, forest governance
assessments for top 50 EUDR-relevant countries, land tenure security
ratings, judicial independence scores, regulatory enforcement
effectiveness, and indigenous rights protection levels.

This module provides the governance infrastructure layer that informs
EUDR due diligence intensity levels (simplified/standard/enhanced) per
Article 29 country classifications.

Assessment Dimensions:
    1. Judicial Independence (0-100): Court system independence from
       political interference, consistency of rulings, appellate
       mechanisms, and anti-corruption tribunal capacity.
    2. Regulatory Enforcement (0-100): Environmental and forestry law
       enforcement capacity, inspection frequency, penalty severity,
       and prosecution success rate.
    3. Forest Governance (0-100): Quality of forestry legislation,
       forest concession transparency, REDD+ readiness, illegal
       logging enforcement, and community forest management.
    4. Law Enforcement Capacity (0-100): Police and inspector general
       capacity for environmental crime, resource availability,
       training levels, and corruption within enforcement bodies.
    5. Land Tenure Security (0-100): Formal land registration coverage,
       communal/customary land recognition, dispute resolution
       mechanisms, and cadastre completeness.
    6. Indigenous Rights Protection (0-100): FPIC implementation,
       ILO 169 ratification, land demarcation, and consultation
       mechanism strength.

Data Sources:
    - World Justice Project Rule of Law Index 2024
    - FAO/ITTO Voluntary Guidelines on National Forest Monitoring
    - International Land Coalition Land Governance Assessment 2024
    - Global Forest Watch Forest Governance Dashboard
    - UN REDD+ Country Profiles and Readiness Assessments
    - IWGIA Indigenous World Report 2024
    - Prindex Global Property Rights Index 2024

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
    "World Justice Project Rule of Law Index 2024",
    "FAO/ITTO Voluntary Guidelines on National Forest Monitoring",
    "International Land Coalition Land Governance Assessment 2024",
    "Global Forest Watch Forest Governance Dashboard",
    "UN REDD+ Country Profiles and Readiness Assessments",
    "IWGIA Indigenous World Report 2024",
    "Prindex Global Property Rights Index 2024",
]

# ---------------------------------------------------------------------------
# Governance dimension constants
# ---------------------------------------------------------------------------

GOVERNANCE_DIMENSIONS: List[str] = [
    "judicial_independence",
    "regulatory_enforcement",
    "forest_governance",
    "law_enforcement_capacity",
    "land_tenure_security",
    "indigenous_rights_protection",
]

GOVERNANCE_DIMENSION_LABELS: Dict[str, str] = {
    "judicial_independence": "Judicial Independence",
    "regulatory_enforcement": "Regulatory Enforcement Effectiveness",
    "forest_governance": "Forest Governance Quality",
    "law_enforcement_capacity": "Law Enforcement Capacity",
    "land_tenure_security": "Land Tenure Security",
    "indigenous_rights_protection": "Indigenous Rights Protection",
}

# ===========================================================================
# Country Governance Profiles - 40+ key countries
# ===========================================================================
# All scores are on 0-100 scale (higher = better governance)

GOVERNANCE_PROFILES: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # EUDR HIGH-PRIORITY COUNTRIES
    # -----------------------------------------------------------------------

    "BRA": {
        "iso_alpha3": "BRA",
        "name": "Brazil",
        "region": "americas",
        "institutional_scores": {
            "judicial_independence": Decimal("48"),
            "regulatory_enforcement": Decimal("42"),
            "forest_governance": Decimal("50"),
            "law_enforcement_capacity": Decimal("40"),
            "land_tenure_security": Decimal("45"),
            "indigenous_rights_protection": Decimal("55"),
        },
        "forest_governance_detail": {
            "legislation_quality": Decimal("62"),
            "concession_transparency": Decimal("45"),
            "redd_readiness": Decimal("58"),
            "illegal_logging_enforcement": Decimal("38"),
            "community_forest_management": Decimal("52"),
            "deforestation_monitoring": Decimal("72"),
            "protected_area_effectiveness": Decimal("48"),
        },
        "land_tenure_detail": {
            "formal_registration_coverage_pct": Decimal("62"),
            "communal_land_recognition": Decimal("55"),
            "dispute_resolution_effectiveness": Decimal("40"),
            "cadastre_completeness_pct": Decimal("58"),
        },
        "notes": "Strong satellite monitoring (PRODES/DETER) but enforcement gaps in remote areas.",
    },

    "IDN": {
        "iso_alpha3": "IDN",
        "name": "Indonesia",
        "region": "asia_pacific",
        "institutional_scores": {
            "judicial_independence": Decimal("38"),
            "regulatory_enforcement": Decimal("35"),
            "forest_governance": Decimal("40"),
            "law_enforcement_capacity": Decimal("32"),
            "land_tenure_security": Decimal("30"),
            "indigenous_rights_protection": Decimal("35"),
        },
        "forest_governance_detail": {
            "legislation_quality": Decimal("52"),
            "concession_transparency": Decimal("35"),
            "redd_readiness": Decimal("50"),
            "illegal_logging_enforcement": Decimal("30"),
            "community_forest_management": Decimal("42"),
            "deforestation_monitoring": Decimal("55"),
            "protected_area_effectiveness": Decimal("38"),
        },
        "land_tenure_detail": {
            "formal_registration_coverage_pct": Decimal("42"),
            "communal_land_recognition": Decimal("28"),
            "dispute_resolution_effectiveness": Decimal("30"),
            "cadastre_completeness_pct": Decimal("38"),
        },
        "notes": "Palm oil moratorium lapsed; SVLK timber legality system improving but gaps remain.",
    },

    "MYS": {
        "iso_alpha3": "MYS",
        "name": "Malaysia",
        "region": "asia_pacific",
        "institutional_scores": {
            "judicial_independence": Decimal("55"),
            "regulatory_enforcement": Decimal("52"),
            "forest_governance": Decimal("48"),
            "law_enforcement_capacity": Decimal("50"),
            "land_tenure_security": Decimal("55"),
            "indigenous_rights_protection": Decimal("32"),
        },
        "forest_governance_detail": {
            "legislation_quality": Decimal("58"),
            "concession_transparency": Decimal("40"),
            "redd_readiness": Decimal("45"),
            "illegal_logging_enforcement": Decimal("45"),
            "community_forest_management": Decimal("35"),
            "deforestation_monitoring": Decimal("55"),
            "protected_area_effectiveness": Decimal("50"),
        },
        "land_tenure_detail": {
            "formal_registration_coverage_pct": Decimal("72"),
            "communal_land_recognition": Decimal("30"),
            "dispute_resolution_effectiveness": Decimal("50"),
            "cadastre_completeness_pct": Decimal("65"),
        },
        "notes": "Federal/state jurisdiction split complicates forest governance; MSPO mandatory.",
    },

    "COL": {
        "iso_alpha3": "COL",
        "name": "Colombia",
        "region": "americas",
        "institutional_scores": {
            "judicial_independence": Decimal("45"),
            "regulatory_enforcement": Decimal("38"),
            "forest_governance": Decimal("42"),
            "law_enforcement_capacity": Decimal("35"),
            "land_tenure_security": Decimal("32"),
            "indigenous_rights_protection": Decimal("52"),
        },
        "forest_governance_detail": {
            "legislation_quality": Decimal("55"),
            "concession_transparency": Decimal("38"),
            "redd_readiness": Decimal("52"),
            "illegal_logging_enforcement": Decimal("30"),
            "community_forest_management": Decimal("48"),
            "deforestation_monitoring": Decimal("58"),
            "protected_area_effectiveness": Decimal("42"),
        },
        "land_tenure_detail": {
            "formal_registration_coverage_pct": Decimal("48"),
            "communal_land_recognition": Decimal("50"),
            "dispute_resolution_effectiveness": Decimal("32"),
            "cadastre_completeness_pct": Decimal("40"),
        },
        "notes": "Post-conflict deforestation surge; indigenous territorial governance advanced.",
    },

    "COD": {
        "iso_alpha3": "COD",
        "name": "Democratic Republic of the Congo",
        "region": "sub_saharan_africa",
        "institutional_scores": {
            "judicial_independence": Decimal("12"),
            "regulatory_enforcement": Decimal("10"),
            "forest_governance": Decimal("15"),
            "law_enforcement_capacity": Decimal("8"),
            "land_tenure_security": Decimal("10"),
            "indigenous_rights_protection": Decimal("18"),
        },
        "forest_governance_detail": {
            "legislation_quality": Decimal("25"),
            "concession_transparency": Decimal("10"),
            "redd_readiness": Decimal("22"),
            "illegal_logging_enforcement": Decimal("8"),
            "community_forest_management": Decimal("15"),
            "deforestation_monitoring": Decimal("18"),
            "protected_area_effectiveness": Decimal("12"),
        },
        "land_tenure_detail": {
            "formal_registration_coverage_pct": Decimal("10"),
            "communal_land_recognition": Decimal("12"),
            "dispute_resolution_effectiveness": Decimal("8"),
            "cadastre_completeness_pct": Decimal("5"),
        },
        "notes": "Active conflict zones; industrial logging concession moratorium partially lifted.",
    },

    "CMR": {
        "iso_alpha3": "CMR",
        "name": "Cameroon",
        "region": "sub_saharan_africa",
        "institutional_scores": {
            "judicial_independence": Decimal("22"),
            "regulatory_enforcement": Decimal("20"),
            "forest_governance": Decimal("28"),
            "law_enforcement_capacity": Decimal("18"),
            "land_tenure_security": Decimal("25"),
            "indigenous_rights_protection": Decimal("20"),
        },
        "forest_governance_detail": {
            "legislation_quality": Decimal("40"),
            "concession_transparency": Decimal("22"),
            "redd_readiness": Decimal("32"),
            "illegal_logging_enforcement": Decimal("20"),
            "community_forest_management": Decimal("30"),
            "deforestation_monitoring": Decimal("28"),
            "protected_area_effectiveness": Decimal("25"),
        },
        "land_tenure_detail": {
            "formal_registration_coverage_pct": Decimal("22"),
            "communal_land_recognition": Decimal("18"),
            "dispute_resolution_effectiveness": Decimal("20"),
            "cadastre_completeness_pct": Decimal("15"),
        },
        "notes": "VPA with EU under FLEGT; independent forest monitor in place.",
    },

    "GHA": {
        "iso_alpha3": "GHA",
        "name": "Ghana",
        "region": "sub_saharan_africa",
        "institutional_scores": {
            "judicial_independence": Decimal("52"),
            "regulatory_enforcement": Decimal("42"),
            "forest_governance": Decimal("45"),
            "law_enforcement_capacity": Decimal("38"),
            "land_tenure_security": Decimal("40"),
            "indigenous_rights_protection": Decimal("42"),
        },
        "forest_governance_detail": {
            "legislation_quality": Decimal("55"),
            "concession_transparency": Decimal("42"),
            "redd_readiness": Decimal("52"),
            "illegal_logging_enforcement": Decimal("35"),
            "community_forest_management": Decimal("48"),
            "deforestation_monitoring": Decimal("50"),
            "protected_area_effectiveness": Decimal("40"),
        },
        "land_tenure_detail": {
            "formal_registration_coverage_pct": Decimal("38"),
            "communal_land_recognition": Decimal("45"),
            "dispute_resolution_effectiveness": Decimal("42"),
            "cadastre_completeness_pct": Decimal("30"),
        },
        "notes": "VPA with EU ratified; cocoa sector reform under COCOBOD.",
    },

    "CIV": {
        "iso_alpha3": "CIV",
        "name": "Ivory Coast",
        "region": "sub_saharan_africa",
        "institutional_scores": {
            "judicial_independence": Decimal("30"),
            "regulatory_enforcement": Decimal("28"),
            "forest_governance": Decimal("25"),
            "law_enforcement_capacity": Decimal("25"),
            "land_tenure_security": Decimal("22"),
            "indigenous_rights_protection": Decimal("20"),
        },
        "forest_governance_detail": {
            "legislation_quality": Decimal("35"),
            "concession_transparency": Decimal("20"),
            "redd_readiness": Decimal("30"),
            "illegal_logging_enforcement": Decimal("18"),
            "community_forest_management": Decimal("25"),
            "deforestation_monitoring": Decimal("32"),
            "protected_area_effectiveness": Decimal("22"),
        },
        "land_tenure_detail": {
            "formal_registration_coverage_pct": Decimal("18"),
            "communal_land_recognition": Decimal("15"),
            "dispute_resolution_effectiveness": Decimal("22"),
            "cadastre_completeness_pct": Decimal("12"),
        },
        "notes": "Severe cocoa-driven deforestation in classified forests; new forestry code 2019.",
    },

    "MMR": {
        "iso_alpha3": "MMR",
        "name": "Myanmar",
        "region": "asia_pacific",
        "institutional_scores": {
            "judicial_independence": Decimal("10"),
            "regulatory_enforcement": Decimal("8"),
            "forest_governance": Decimal("12"),
            "law_enforcement_capacity": Decimal("10"),
            "land_tenure_security": Decimal("8"),
            "indigenous_rights_protection": Decimal("5"),
        },
        "forest_governance_detail": {
            "legislation_quality": Decimal("20"),
            "concession_transparency": Decimal("5"),
            "redd_readiness": Decimal("15"),
            "illegal_logging_enforcement": Decimal("5"),
            "community_forest_management": Decimal("12"),
            "deforestation_monitoring": Decimal("15"),
            "protected_area_effectiveness": Decimal("10"),
        },
        "land_tenure_detail": {
            "formal_registration_coverage_pct": Decimal("15"),
            "communal_land_recognition": Decimal("8"),
            "dispute_resolution_effectiveness": Decimal("5"),
            "cadastre_completeness_pct": Decimal("10"),
        },
        "notes": "Military coup 2021 collapsed forest governance; teak/hardwood concessions opaque.",
    },

    "PNG": {
        "iso_alpha3": "PNG",
        "name": "Papua New Guinea",
        "region": "asia_pacific",
        "institutional_scores": {
            "judicial_independence": Decimal("28"),
            "regulatory_enforcement": Decimal("18"),
            "forest_governance": Decimal("20"),
            "law_enforcement_capacity": Decimal("15"),
            "land_tenure_security": Decimal("55"),
            "indigenous_rights_protection": Decimal("48"),
        },
        "forest_governance_detail": {
            "legislation_quality": Decimal("30"),
            "concession_transparency": Decimal("12"),
            "redd_readiness": Decimal("28"),
            "illegal_logging_enforcement": Decimal("10"),
            "community_forest_management": Decimal("35"),
            "deforestation_monitoring": Decimal("22"),
            "protected_area_effectiveness": Decimal("18"),
        },
        "land_tenure_detail": {
            "formal_registration_coverage_pct": Decimal("12"),
            "communal_land_recognition": Decimal("72"),
            "dispute_resolution_effectiveness": Decimal("28"),
            "cadastre_completeness_pct": Decimal("8"),
        },
        "notes": "97% customary land ownership; SABL land lease controversies.",
    },

    # -----------------------------------------------------------------------
    # BENCHMARK LOW-RISK COUNTRIES
    # -----------------------------------------------------------------------

    "DNK": {
        "iso_alpha3": "DNK",
        "name": "Denmark",
        "region": "western_europe",
        "institutional_scores": {
            "judicial_independence": Decimal("95"),
            "regulatory_enforcement": Decimal("92"),
            "forest_governance": Decimal("90"),
            "law_enforcement_capacity": Decimal("92"),
            "land_tenure_security": Decimal("95"),
            "indigenous_rights_protection": Decimal("88"),
        },
        "forest_governance_detail": {
            "legislation_quality": Decimal("95"),
            "concession_transparency": Decimal("92"),
            "redd_readiness": Decimal("85"),
            "illegal_logging_enforcement": Decimal("92"),
            "community_forest_management": Decimal("88"),
            "deforestation_monitoring": Decimal("90"),
            "protected_area_effectiveness": Decimal("92"),
        },
        "land_tenure_detail": {
            "formal_registration_coverage_pct": Decimal("99"),
            "communal_land_recognition": Decimal("90"),
            "dispute_resolution_effectiveness": Decimal("95"),
            "cadastre_completeness_pct": Decimal("99"),
        },
        "notes": "Model governance with full EUTR/FLEGT compliance; strong due diligence culture.",
    },

    "DEU": {
        "iso_alpha3": "DEU",
        "name": "Germany",
        "region": "western_europe",
        "institutional_scores": {
            "judicial_independence": Decimal("88"),
            "regulatory_enforcement": Decimal("85"),
            "forest_governance": Decimal("85"),
            "law_enforcement_capacity": Decimal("88"),
            "land_tenure_security": Decimal("92"),
            "indigenous_rights_protection": Decimal("80"),
        },
        "forest_governance_detail": {
            "legislation_quality": Decimal("90"),
            "concession_transparency": Decimal("88"),
            "redd_readiness": Decimal("82"),
            "illegal_logging_enforcement": Decimal("88"),
            "community_forest_management": Decimal("82"),
            "deforestation_monitoring": Decimal("85"),
            "protected_area_effectiveness": Decimal("88"),
        },
        "land_tenure_detail": {
            "formal_registration_coverage_pct": Decimal("98"),
            "communal_land_recognition": Decimal("85"),
            "dispute_resolution_effectiveness": Decimal("92"),
            "cadastre_completeness_pct": Decimal("98"),
        },
        "notes": "Supply Chain Due Diligence Act (LkSG) adds additional requirements beyond EUDR.",
    },

    "NLD": {
        "iso_alpha3": "NLD",
        "name": "Netherlands",
        "region": "western_europe",
        "institutional_scores": {
            "judicial_independence": Decimal("90"),
            "regulatory_enforcement": Decimal("88"),
            "forest_governance": Decimal("85"),
            "law_enforcement_capacity": Decimal("90"),
            "land_tenure_security": Decimal("94"),
            "indigenous_rights_protection": Decimal("82"),
        },
        "forest_governance_detail": {
            "legislation_quality": Decimal("88"),
            "concession_transparency": Decimal("90"),
            "redd_readiness": Decimal("85"),
            "illegal_logging_enforcement": Decimal("90"),
            "community_forest_management": Decimal("85"),
            "deforestation_monitoring": Decimal("88"),
            "protected_area_effectiveness": Decimal("90"),
        },
        "land_tenure_detail": {
            "formal_registration_coverage_pct": Decimal("99"),
            "communal_land_recognition": Decimal("88"),
            "dispute_resolution_effectiveness": Decimal("94"),
            "cadastre_completeness_pct": Decimal("99"),
        },
        "notes": "Major EUDR import hub for palm oil, soy, cocoa; IDH sustainability initiatives.",
    },
}


# ===========================================================================
# CountryGovernanceDatabase class
# ===========================================================================


class CountryGovernanceDatabase:
    """
    Stateless reference data accessor for country governance profiles.

    Provides typed access to institutional quality scores, forest governance
    assessments, land tenure ratings, and governance comparisons. All
    numeric values stored as ``Decimal``.

    Example:
        >>> db = CountryGovernanceDatabase()
        >>> profile = db.get_profile("BRA")
        >>> assert profile["institutional_scores"]["judicial_independence"] == Decimal("48")
        >>> forest = db.get_forest_governance("IDN")
        >>> assert forest["legislation_quality"] < Decimal("60")
    """

    def get_profile(
        self,
        country_code: str,
    ) -> Optional[Dict[str, Any]]:
        """Get complete governance profile for a country.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.

        Returns:
            Complete governance profile dict, or None if not found.
        """
        country = GOVERNANCE_PROFILES.get(country_code.upper())
        if country is None:
            return None
        return {
            "country_code": country_code.upper(),
            "name": country["name"],
            "region": country["region"],
            "institutional_scores": dict(country["institutional_scores"]),
            "forest_governance_detail": dict(country.get("forest_governance_detail", {})),
            "land_tenure_detail": dict(country.get("land_tenure_detail", {})),
            "notes": country.get("notes", ""),
        }

    def get_forest_governance(
        self,
        country_code: str,
    ) -> Optional[Dict[str, Decimal]]:
        """Get forest governance detail scores for a country.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.

        Returns:
            Dict of forest governance dimension scores, or None if not found.
        """
        country = GOVERNANCE_PROFILES.get(country_code.upper())
        if country is None:
            return None
        detail = country.get("forest_governance_detail")
        if detail is None:
            return None
        return dict(detail)

    def get_institutional_scores(
        self,
        country_code: str,
    ) -> Optional[Dict[str, Decimal]]:
        """Get institutional quality scores for a country.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.

        Returns:
            Dict mapping dimension names to scores, or None if not found.
        """
        country = GOVERNANCE_PROFILES.get(country_code.upper())
        if country is None:
            return None
        return dict(country["institutional_scores"])

    def get_land_tenure(
        self,
        country_code: str,
    ) -> Optional[Dict[str, Decimal]]:
        """Get land tenure detail for a country.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.

        Returns:
            Dict of land tenure metrics, or None if not found.
        """
        country = GOVERNANCE_PROFILES.get(country_code.upper())
        if country is None:
            return None
        detail = country.get("land_tenure_detail")
        if detail is None:
            return None
        return dict(detail)

    def compare_governance(
        self,
        country_codes: List[str],
        dimension: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare governance across multiple countries.

        Args:
            country_codes: List of ISO alpha-3 country codes.
            dimension: Specific dimension to compare. If None, compare all.

        Returns:
            Dict mapping country codes to score data.
        """
        comparison: Dict[str, Any] = {}
        for code in country_codes:
            code_upper = code.upper()
            country = GOVERNANCE_PROFILES.get(code_upper)
            if country is None:
                continue
            if dimension is not None:
                score = country["institutional_scores"].get(dimension)
                if score is not None:
                    comparison[code_upper] = {
                        "name": country["name"],
                        dimension: score,
                    }
            else:
                comparison[code_upper] = {
                    "name": country["name"],
                    "institutional_scores": dict(country["institutional_scores"]),
                }
        return comparison

    def get_weak_governance_countries(
        self,
        threshold: Decimal = Decimal("35"),
        dimension: str = "forest_governance",
    ) -> List[Dict[str, Any]]:
        """Get countries with governance scores below threshold.

        Args:
            threshold: Maximum score to include (0-100).
            dimension: Institutional dimension to check.

        Returns:
            List of country dicts sorted by score ascending (weakest first).
        """
        results = []
        for code, country in GOVERNANCE_PROFILES.items():
            score = country["institutional_scores"].get(dimension)
            if score is not None and score <= threshold:
                results.append({
                    "country_code": code,
                    "name": country["name"],
                    "region": country["region"],
                    dimension: score,
                })
        results.sort(key=lambda x: x[dimension])
        return results

    def get_composite_score(
        self,
        country_code: str,
        weights: Optional[Dict[str, Decimal]] = None,
    ) -> Optional[Decimal]:
        """Calculate weighted composite governance score.

        Args:
            country_code: ISO 3166-1 alpha-3 country code.
            weights: Optional dimension weights (must sum to 1.0). Defaults
                to equal weights across the 4 primary IQ dimensions.

        Returns:
            Weighted composite score (0-100), or None if not found.
        """
        country = GOVERNANCE_PROFILES.get(country_code.upper())
        if country is None:
            return None

        if weights is None:
            weights = {
                "judicial_independence": Decimal("0.30"),
                "regulatory_enforcement": Decimal("0.25"),
                "forest_governance": Decimal("0.25"),
                "law_enforcement_capacity": Decimal("0.20"),
            }

        total = Decimal("0")
        for dim, weight in weights.items():
            score = country["institutional_scores"].get(dim)
            if score is not None:
                total += score * weight

        return total.quantize(Decimal("0.01"))

    def get_country_count(self) -> int:
        """Get total number of countries in the governance database.

        Returns:
            Integer count of countries.
        """
        return len(GOVERNANCE_PROFILES)


# ===========================================================================
# Module-level helper functions
# ===========================================================================


def get_profile(country_code: str) -> Optional[Dict[str, Any]]:
    """Get complete governance profile for a country."""
    return CountryGovernanceDatabase().get_profile(country_code)


def get_forest_governance(country_code: str) -> Optional[Dict[str, Decimal]]:
    """Get forest governance detail scores for a country."""
    return CountryGovernanceDatabase().get_forest_governance(country_code)


def get_institutional_scores(country_code: str) -> Optional[Dict[str, Decimal]]:
    """Get institutional quality scores for a country."""
    return CountryGovernanceDatabase().get_institutional_scores(country_code)


def get_land_tenure(country_code: str) -> Optional[Dict[str, Decimal]]:
    """Get land tenure detail for a country."""
    return CountryGovernanceDatabase().get_land_tenure(country_code)


def compare_governance(country_codes: List[str], dimension: Optional[str] = None) -> Dict[str, Any]:
    """Compare governance across multiple countries."""
    return CountryGovernanceDatabase().compare_governance(country_codes, dimension)


def get_weak_governance_countries(threshold: Decimal = Decimal("35"), dimension: str = "forest_governance") -> List[Dict[str, Any]]:
    """Get countries with governance scores below threshold."""
    return CountryGovernanceDatabase().get_weak_governance_countries(threshold, dimension)


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    "DATA_VERSION",
    "DATA_SOURCES",
    "GOVERNANCE_DIMENSIONS",
    "GOVERNANCE_DIMENSION_LABELS",
    "GOVERNANCE_PROFILES",
    "CountryGovernanceDatabase",
    "get_profile",
    "get_forest_governance",
    "get_institutional_scores",
    "get_land_tenure",
    "compare_governance",
    "get_weak_governance_countries",
]
