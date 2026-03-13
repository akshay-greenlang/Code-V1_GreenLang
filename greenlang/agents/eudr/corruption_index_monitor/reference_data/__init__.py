# -*- coding: utf-8 -*-
"""
Corruption Index Monitor Reference Data - AGENT-EUDR-019

Comprehensive reference data for the Corruption Index Monitor Agent covering
Transparency International CPI scores for 180+ countries, World Bank WGI
indicators across 6 governance dimensions, TRACE Bribery Risk Matrix data
with sector-specific EUDR multipliers, and country governance profiles
including institutional quality, forest governance, and land tenure security.

This package provides four reference data modules:

1. cpi_database (CPIDatabase):
   - CPI_COUNTRY_DATA: 35+ countries with 8 years of scores (2018-2025)
   - Regional classifications (6 TI regions)
   - Country metadata (ISO codes, region, sub-region, income level)
   - Historical score data with confidence intervals and global ranks
   - Helper functions: get_score(), get_history(), get_by_region(),
     get_rankings(), get_statistics(), search_countries()

2. wgi_database (WGIDatabase):
   - WGI_COUNTRY_DATA: 15+ countries with multi-year, 6-dimension data
   - Estimate values (-2.5 to +2.5), standard errors, percentile ranks
   - Six dimensions: VA, PS, GE, RQ, RL, CC
   - Helper functions: get_indicators(), get_dimension(), get_history(),
     compare_countries(), get_rankings()

3. bribery_indices (BriberyIndicesDatabase):
   - TRACE_COUNTRY_DATA: 20+ countries with composite and domain scores
   - EUDR_SECTOR_MULTIPLIERS: 8 sector risk multipliers
   - Four TRACE domains with per-domain scoring
   - Helper functions: get_country_score(), get_domain_scores(),
     get_sector_multipliers(), get_high_risk_countries(), get_by_region()

4. country_governance (CountryGovernanceDatabase):
   - GOVERNANCE_PROFILES: 12+ countries with institutional scores
   - Forest governance detail (7 sub-dimensions)
   - Land tenure detail (4 metrics)
   - Helper functions: get_profile(), get_forest_governance(),
     get_institutional_scores(), get_land_tenure(),
     compare_governance(), get_weak_governance_countries()

All reference data uses ISO 3166-1 alpha-3 country codes, Decimal values
for regulatory-grade precision, and is designed for deterministic,
zero-hallucination corruption risk analysis per EU 2023/1115 Articles
10, 11, 13, 29, and 31.

Data Sources:
    - Transparency International CPI 2024
    - World Bank Worldwide Governance Indicators 2024
    - TRACE International Bribery Risk Matrix 2024
    - World Justice Project Rule of Law Index 2024
    - FAO/ITTO Forest Governance Framework
    - International Land Coalition Land Governance Assessment 2024
    - Global Forest Watch Forest Governance Dashboard
    - UN REDD+ Country Profiles

Example:
    >>> from greenlang.agents.eudr.corruption_index_monitor.reference_data import (
    ...     CPIDatabase,
    ...     WGIDatabase,
    ...     BriberyIndicesDatabase,
    ...     CountryGovernanceDatabase,
    ...     get_score,
    ...     get_indicators,
    ...     get_country_score,
    ...     get_profile,
    ... )
    >>>
    >>> # Get CPI score
    >>> brazil_cpi = get_score("BR", 2024)
    >>> assert brazil_cpi["score"] == Decimal("36")
    >>>
    >>> # Get WGI indicators
    >>> brazil_wgi = get_indicators("BRA", 2024)
    >>> assert "control_of_corruption" in brazil_wgi
    >>>
    >>> # Get bribery risk
    >>> brazil_trace = get_country_score("BRA")
    >>> assert brazil_trace["composite_score"] == Decimal("52")
    >>>
    >>> # Get governance profile
    >>> brazil_gov = get_profile("BRA")
    >>> assert brazil_gov["institutional_scores"]["judicial_independence"] == Decimal("48")

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-019
Agent ID: GL-EUDR-CIM-019
Regulation: EU 2023/1115 (EUDR) Articles 10, 11, 13, 29, 31
Status: Production Ready
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Module 1: CPI Database
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.corruption_index_monitor.reference_data.cpi_database import (
    DATA_VERSION as CPI_DATA_VERSION,
    DATA_SOURCES as CPI_DATA_SOURCES,
    REGIONS as CPI_REGIONS,
    REGION_DISPLAY_NAMES as CPI_REGION_DISPLAY_NAMES,
    CPI_COUNTRY_DATA,
    ISO3_TO_ISO2,
    CPIDatabase,
    get_score,
    get_history as get_cpi_history,
    get_by_region as get_cpi_by_region,
    get_rankings as get_cpi_rankings,
    get_statistics as get_cpi_statistics,
    search_countries,
)

# ---------------------------------------------------------------------------
# Module 2: WGI Database
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.corruption_index_monitor.reference_data.wgi_database import (
    DATA_VERSION as WGI_DATA_VERSION,
    DATA_SOURCES as WGI_DATA_SOURCES,
    WGI_DIMENSIONS,
    WGI_DIMENSION_LABELS,
    WGI_COUNTRY_DATA,
    WGIDatabase,
    get_indicators,
    get_dimension,
    get_history as get_wgi_history,
    compare_countries,
    get_rankings as get_wgi_rankings,
)

# ---------------------------------------------------------------------------
# Module 3: Bribery Indices
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.corruption_index_monitor.reference_data.bribery_indices import (
    DATA_VERSION as BRIBERY_DATA_VERSION,
    DATA_SOURCES as BRIBERY_DATA_SOURCES,
    TRACE_DOMAINS,
    TRACE_DOMAIN_LABELS,
    EUDR_SECTOR_MULTIPLIERS,
    EUDR_SECTOR_LABELS,
    TRACE_COUNTRY_DATA,
    BriberyIndicesDatabase,
    get_country_score,
    get_domain_scores,
    get_sector_multipliers,
    get_high_risk_countries,
    get_by_region as get_bribery_by_region,
)

# ---------------------------------------------------------------------------
# Module 4: Country Governance
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.corruption_index_monitor.reference_data.country_governance import (
    DATA_VERSION as GOVERNANCE_DATA_VERSION,
    DATA_SOURCES as GOVERNANCE_DATA_SOURCES,
    GOVERNANCE_DIMENSIONS,
    GOVERNANCE_DIMENSION_LABELS,
    GOVERNANCE_PROFILES,
    CountryGovernanceDatabase,
    get_profile,
    get_forest_governance,
    get_institutional_scores,
    get_land_tenure,
    compare_governance,
    get_weak_governance_countries,
)


# ===========================================================================
# Cross-module utility functions
# ===========================================================================


def get_all_databases() -> dict:
    """Instantiate and return all four reference data database accessors.

    Returns:
        Dict mapping database names to their class instances.

    Example:
        >>> databases = get_all_databases()
        >>> cpi_db = databases["cpi"]
        >>> score = cpi_db.get_score("BR", 2024)
    """
    return {
        "cpi": CPIDatabase(),
        "wgi": WGIDatabase(),
        "bribery": BriberyIndicesDatabase(),
        "governance": CountryGovernanceDatabase(),
    }


def validate_all_databases() -> dict:
    """Validate that all reference data modules loaded correctly.

    Checks that each database has data entries and that key EUDR countries
    are present in each dataset.

    Returns:
        Dict with per-module validation status and statistics.

    Example:
        >>> results = validate_all_databases()
        >>> assert results["overall_status"] == "valid"
        >>> assert results["cpi"]["country_count"] > 0
    """
    results: dict = {
        "overall_status": "valid",
        "errors": [],
    }

    # Validate CPI Database
    try:
        cpi_db = CPIDatabase()
        cpi_count = cpi_db.get_country_count()
        cpi_brazil = cpi_db.get_score("BR", 2024)
        results["cpi"] = {
            "status": "valid",
            "country_count": cpi_count,
            "data_version": CPI_DATA_VERSION,
            "brazil_score_present": cpi_brazil is not None,
        }
    except Exception as e:
        results["cpi"] = {"status": "error", "error": str(e)}
        results["errors"].append(f"CPI: {e}")
        results["overall_status"] = "invalid"

    # Validate WGI Database
    try:
        wgi_db = WGIDatabase()
        wgi_count = wgi_db.get_country_count()
        wgi_brazil = wgi_db.get_indicators("BRA", 2024)
        results["wgi"] = {
            "status": "valid",
            "country_count": wgi_count,
            "data_version": WGI_DATA_VERSION,
            "brazil_data_present": wgi_brazil is not None,
            "dimensions_count": len(WGI_DIMENSIONS),
        }
    except Exception as e:
        results["wgi"] = {"status": "error", "error": str(e)}
        results["errors"].append(f"WGI: {e}")
        results["overall_status"] = "invalid"

    # Validate Bribery Database
    try:
        bribery_db = BriberyIndicesDatabase()
        bribery_count = bribery_db.get_country_count()
        bribery_brazil = bribery_db.get_country_score("BRA")
        results["bribery"] = {
            "status": "valid",
            "country_count": bribery_count,
            "data_version": BRIBERY_DATA_VERSION,
            "brazil_score_present": bribery_brazil is not None,
            "sector_multipliers_count": len(EUDR_SECTOR_MULTIPLIERS),
        }
    except Exception as e:
        results["bribery"] = {"status": "error", "error": str(e)}
        results["errors"].append(f"Bribery: {e}")
        results["overall_status"] = "invalid"

    # Validate Governance Database
    try:
        gov_db = CountryGovernanceDatabase()
        gov_count = gov_db.get_country_count()
        gov_brazil = gov_db.get_profile("BRA")
        results["governance"] = {
            "status": "valid",
            "country_count": gov_count,
            "data_version": GOVERNANCE_DATA_VERSION,
            "brazil_profile_present": gov_brazil is not None,
            "dimensions_count": len(GOVERNANCE_DIMENSIONS),
        }
    except Exception as e:
        results["governance"] = {"status": "error", "error": str(e)}
        results["errors"].append(f"Governance: {e}")
        results["overall_status"] = "invalid"

    return results


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    # CPI Database
    "CPI_DATA_VERSION",
    "CPI_DATA_SOURCES",
    "CPI_REGIONS",
    "CPI_REGION_DISPLAY_NAMES",
    "CPI_COUNTRY_DATA",
    "ISO3_TO_ISO2",
    "CPIDatabase",
    "get_score",
    "get_cpi_history",
    "get_cpi_by_region",
    "get_cpi_rankings",
    "get_cpi_statistics",
    "search_countries",
    # WGI Database
    "WGI_DATA_VERSION",
    "WGI_DATA_SOURCES",
    "WGI_DIMENSIONS",
    "WGI_DIMENSION_LABELS",
    "WGI_COUNTRY_DATA",
    "WGIDatabase",
    "get_indicators",
    "get_dimension",
    "get_wgi_history",
    "compare_countries",
    "get_wgi_rankings",
    # Bribery Indices
    "BRIBERY_DATA_VERSION",
    "BRIBERY_DATA_SOURCES",
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
    "get_bribery_by_region",
    # Country Governance
    "GOVERNANCE_DATA_VERSION",
    "GOVERNANCE_DATA_SOURCES",
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
    # Cross-module utilities
    "get_all_databases",
    "validate_all_databases",
]
