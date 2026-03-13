# -*- coding: utf-8 -*-
"""
Reference Data Package - AGENT-EUDR-016 Country Risk Evaluator

Centralized reference data for country risk evaluation covering:
    - Country risk database: 60+ countries with EC benchmarking classifications
    - Governance indices: World Bank WGI, Transparency International CPI,
      FAO/ITTO forest governance scores
    - Trade flow data: Major bilateral trade flows, transshipment hubs,
      HS code mapping, production volumes, certification coverage

All reference data is version-controlled and deterministically serializable
for SHA-256 provenance hashing and zero-hallucination compliance.

Modules:
    - country_risk_database: 60+ countries with composite risk data
    - governance_indices: WGI, CPI, forest governance, enforcement scores
    - trade_flow_data: Bilateral trade flows, transshipment hubs, HS codes

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
Status: Production Ready
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Import from country_risk_database
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.country_risk_evaluator.reference_data.country_risk_database import (
    COUNTRY_RISK_DATABASE,
    CountryRiskRecord,
    DATA_VERSION,
    DATA_SOURCES,
    EUDR_COMMODITIES,
    get_country_risk_data,
    get_high_risk_countries,
    get_low_risk_countries,
    get_standard_risk_countries,
    get_countries_by_region,
    get_countries_producing_commodity,
    search_countries,
)

# ---------------------------------------------------------------------------
# Import from governance_indices
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.country_risk_evaluator.reference_data.governance_indices import (
    WORLD_BANK_WGI,
    TI_CPI_SCORES,
    FOREST_GOVERNANCE_SCORES,
    ENFORCEMENT_EFFECTIVENESS,
    WGI_DIMENSIONS,
    FOREST_GOVERNANCE_DIMENSIONS,
    ENFORCEMENT_DIMENSIONS,
    get_wgi_score,
    get_wgi_dimension,
    get_cpi_score,
    get_forest_governance,
    get_forest_governance_dimension,
    get_enforcement_score,
    get_enforcement_dimension,
    calculate_governance_composite,
)

# ---------------------------------------------------------------------------
# Import from trade_flow_data
# ---------------------------------------------------------------------------

from greenlang.agents.eudr.country_risk_evaluator.reference_data.trade_flow_data import (
    MAJOR_TRADE_FLOWS,
    TRANSSHIPMENT_HUBS,
    HS_CODE_MAPPING,
    COMMODITY_PRODUCTION_DATA,
    CERTIFICATION_COVERAGE,
    TradeFlowRecord,
    TransshipmentHub,
    HSCodeMapping,
    ProductionRecord,
    CertificationCoverageRecord,
    get_trade_flows,
    get_trade_flows_by_commodity,
    get_trade_flows_by_origin,
    get_trade_flows_by_destination,
    get_transshipment_risk,
    get_transshipment_hubs_for_commodity,
    map_hs_to_commodity,
    get_hs_codes_for_commodity,
    get_production_volume,
    get_production_by_commodity,
    get_certification_coverage,
    get_certification_by_commodity,
    get_major_exporters,
    get_major_importers,
)

__all__ = [
    # -- Country Risk Database --
    "COUNTRY_RISK_DATABASE",
    "CountryRiskRecord",
    "DATA_VERSION",
    "DATA_SOURCES",
    "EUDR_COMMODITIES",
    "get_country_risk_data",
    "get_high_risk_countries",
    "get_low_risk_countries",
    "get_standard_risk_countries",
    "get_countries_by_region",
    "get_countries_producing_commodity",
    "search_countries",
    # -- Governance Indices --
    "WORLD_BANK_WGI",
    "TI_CPI_SCORES",
    "FOREST_GOVERNANCE_SCORES",
    "ENFORCEMENT_EFFECTIVENESS",
    "WGI_DIMENSIONS",
    "FOREST_GOVERNANCE_DIMENSIONS",
    "ENFORCEMENT_DIMENSIONS",
    "get_wgi_score",
    "get_wgi_dimension",
    "get_cpi_score",
    "get_forest_governance",
    "get_forest_governance_dimension",
    "get_enforcement_score",
    "get_enforcement_dimension",
    "calculate_governance_composite",
    # -- Trade Flow Data --
    "MAJOR_TRADE_FLOWS",
    "TRANSSHIPMENT_HUBS",
    "HS_CODE_MAPPING",
    "COMMODITY_PRODUCTION_DATA",
    "CERTIFICATION_COVERAGE",
    "TradeFlowRecord",
    "TransshipmentHub",
    "HSCodeMapping",
    "ProductionRecord",
    "CertificationCoverageRecord",
    "get_trade_flows",
    "get_trade_flows_by_commodity",
    "get_trade_flows_by_origin",
    "get_trade_flows_by_destination",
    "get_transshipment_risk",
    "get_transshipment_hubs_for_commodity",
    "map_hs_to_commodity",
    "get_hs_codes_for_commodity",
    "get_production_volume",
    "get_production_by_commodity",
    "get_certification_coverage",
    "get_certification_by_commodity",
    "get_major_exporters",
    "get_major_importers",
]
