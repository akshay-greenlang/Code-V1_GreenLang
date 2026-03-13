# -*- coding: utf-8 -*-
"""
Commodity Risk Analyzer Reference Data - AGENT-EUDR-018

Comprehensive reference data for the Commodity Risk Analyzer Agent covering
all 7 EUDR-regulated commodities (cattle, cocoa, coffee, oil palm, rubber,
soya, wood) and their Annex I derived products, processing chains,
production statistics, and regulatory requirements.

This package provides four reference data modules:

1. commodity_database (CommodityDatabase):
   - EUDR_COMMODITIES: 7 EUDR commodities with HS codes, CN codes,
     intrinsic risk factors (deforestation_pressure, supply_chain_complexity,
     traceability_difficulty, processing_variability), key producing countries,
     supply chain depth, processing stages, and sustainability certifications
   - DERIVED_PRODUCTS: 70+ derived products mapped to source commodities
     with product codes, processing stages, and traceability requirements
   - COUNTRY_PRODUCTION_DATA: Top producers per commodity with volumes
   - HS_CODE_MAPPING: Complete HS code to commodity type mapping
   - Helper functions: get_commodity_info(), get_derived_products(),
     lookup_hs_code()

2. processing_chains (ProcessingChainDatabase):
   - PROCESSING_CHAINS: Transformation pathways for all 7 commodities
     with per-stage risk addition, traceability loss, waste percentage,
     and transformation ratios
   - PROCESSING_RISK_FACTORS: Risk multipliers per processing type
   - TRACEABILITY_LOSS_FACTORS: Traceability degradation coefficients
   - Helper functions: get_processing_chain(), calculate_chain_risk(),
     get_transformation_ratio()

3. production_statistics (ProductionStatistics):
   - PRODUCTION_STATISTICS: Annual global production volumes per commodity
     with top-10 producing countries, YoY growth rates, historical yields
   - SEASONAL_PATTERNS: Month-by-month production patterns per commodity
     per region (planting/harvest calendars)
   - CLIMATE_SENSITIVITY: Temperature and rainfall sensitivity coefficients
   - Helper functions: get_production_stats(), get_seasonal_pattern(),
     get_yield_data()

4. regulatory_requirements (RegulatoryRequirementDatabase):
   - EUDR_ARTICLES: Per-article requirements mapped to each commodity
     (Articles 3, 4, 9, 10, 11, 12, 13, 29)
   - DOCUMENTATION_REQUIREMENTS: Common and commodity-specific documents
   - PENALTY_MATRIX: Penalty categories per violation type per commodity
   - Helper functions: get_requirements(), get_documentation_list(),
     get_penalty_info()

All reference data uses ISO 3166-1 alpha-3 country codes for production
data, Harmonized System (HS) codes for trade classification, and EUDR
Annex I product codes for derived products. Data is designed for
deterministic, zero-hallucination commodity risk analysis per EU 2023/1115.

Data Sources:
    - FAO FAOSTAT Production Statistics 2024
    - USDA Foreign Agricultural Service (FAS) Commodity Reports 2024
    - International Cocoa Organization (ICCO) Quarterly Bulletin
    - International Coffee Organization (ICO) Market Reports
    - Malaysian Palm Oil Board (MPOB) Statistics 2024
    - International Rubber Study Group (IRSG) Outlook 2024
    - International Tropical Timber Organization (ITTO) Market Report
    - European Commission EUDR Implementing Regulation (EU) 2023/1115
    - WCO Harmonized System Nomenclature 2024

Example:
    >>> from greenlang.agents.eudr.commodity_risk_analyzer.reference_data import (
    ...     CommodityDatabase,
    ...     ProcessingChainDatabase,
    ...     ProductionStatistics,
    ...     RegulatoryRequirementDatabase,
    ...     EUDR_COMMODITIES,
    ...     DERIVED_PRODUCTS,
    ...     get_commodity_info,
    ...     get_derived_products,
    ...     lookup_hs_code,
    ... )
    >>>
    >>> # Get commodity info
    >>> cocoa = get_commodity_info("cocoa")
    >>> assert cocoa["commodity_type"] == "cocoa"
    >>> assert len(cocoa["hs_codes"]) > 0
    >>>
    >>> # Get derived products
    >>> palm_products = get_derived_products("oil_palm")
    >>> assert any(p["name"] == "crude_palm_oil" for p in palm_products)
    >>>
    >>> # Look up HS code
    >>> commodity = lookup_hs_code("1801.00")
    >>> assert commodity == "cocoa"

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-018
Agent ID: GL-EUDR-CRA-018
Regulation: EU 2023/1115 (EUDR) Articles 1, 2, 3, 4, 8, 9, 10, Annex I
Status: Production Ready
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Module 1: Commodity Database
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.commodity_risk_analyzer.reference_data.commodity_database import (
    DATA_VERSION as COMMODITY_DATA_VERSION,
    DATA_SOURCES as COMMODITY_DATA_SOURCES,
    EUDR_COMMODITIES,
    DERIVED_PRODUCTS,
    COUNTRY_PRODUCTION_DATA,
    HS_CODE_MAPPING,
    CommodityDatabase,
    get_commodity_info,
    get_derived_products,
    lookup_hs_code,
    get_hs_codes_for_commodity,
    get_producing_countries,
)

# ---------------------------------------------------------------------------
# Module 2: Processing Chains
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.commodity_risk_analyzer.reference_data.processing_chains import (
    DATA_VERSION as PROCESSING_DATA_VERSION,
    DATA_SOURCES as PROCESSING_DATA_SOURCES,
    PROCESSING_CHAINS,
    PROCESSING_RISK_FACTORS,
    TRACEABILITY_LOSS_FACTORS,
    ProcessingChainDatabase,
    get_processing_chain,
    calculate_chain_risk,
    get_transformation_ratio,
    get_processing_stages,
)

# ---------------------------------------------------------------------------
# Module 3: Production Statistics
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.commodity_risk_analyzer.reference_data.production_statistics import (
    DATA_VERSION as PRODUCTION_DATA_VERSION,
    DATA_SOURCES as PRODUCTION_DATA_SOURCES,
    PRODUCTION_STATISTICS,
    SEASONAL_PATTERNS,
    CLIMATE_SENSITIVITY,
    ProductionStatistics,
    get_production_stats,
    get_seasonal_pattern,
    get_yield_data,
    get_top_producers,
)

# ---------------------------------------------------------------------------
# Module 4: Regulatory Requirements
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.commodity_risk_analyzer.reference_data.regulatory_requirements import (
    DATA_VERSION as REGULATORY_DATA_VERSION,
    DATA_SOURCES as REGULATORY_DATA_SOURCES,
    EUDR_ARTICLES,
    DOCUMENTATION_REQUIREMENTS,
    PENALTY_MATRIX,
    RegulatoryRequirementDatabase,
    get_requirements,
    get_documentation_list,
    get_penalty_info,
    get_article_requirements,
)


# ===========================================================================
# Module exports
# ===========================================================================

__all__ = [
    # Commodity Database
    "COMMODITY_DATA_VERSION",
    "COMMODITY_DATA_SOURCES",
    "EUDR_COMMODITIES",
    "DERIVED_PRODUCTS",
    "COUNTRY_PRODUCTION_DATA",
    "HS_CODE_MAPPING",
    "CommodityDatabase",
    "get_commodity_info",
    "get_derived_products",
    "lookup_hs_code",
    "get_hs_codes_for_commodity",
    "get_producing_countries",
    # Processing Chains
    "PROCESSING_DATA_VERSION",
    "PROCESSING_DATA_SOURCES",
    "PROCESSING_CHAINS",
    "PROCESSING_RISK_FACTORS",
    "TRACEABILITY_LOSS_FACTORS",
    "ProcessingChainDatabase",
    "get_processing_chain",
    "calculate_chain_risk",
    "get_transformation_ratio",
    "get_processing_stages",
    # Production Statistics
    "PRODUCTION_DATA_VERSION",
    "PRODUCTION_DATA_SOURCES",
    "PRODUCTION_STATISTICS",
    "SEASONAL_PATTERNS",
    "CLIMATE_SENSITIVITY",
    "ProductionStatistics",
    "get_production_stats",
    "get_seasonal_pattern",
    "get_yield_data",
    "get_top_producers",
    # Regulatory Requirements
    "REGULATORY_DATA_VERSION",
    "REGULATORY_DATA_SOURCES",
    "EUDR_ARTICLES",
    "DOCUMENTATION_REQUIREMENTS",
    "PENALTY_MATRIX",
    "RegulatoryRequirementDatabase",
    "get_requirements",
    "get_documentation_list",
    "get_penalty_info",
    "get_article_requirements",
]
