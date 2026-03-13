# -*- coding: utf-8 -*-
"""
Reference Data Package - AGENT-EUDR-008: Multi-Tier Supplier Tracker Agent

Provides built-in reference datasets for multi-tier supplier tracking:
    - country_risk_scores: 100+ countries with deforestation risk scores
    - certification_standards: 10+ certification types with EUDR rules
    - commodity_supply_chains: 7 EUDR commodities with tier structures

These datasets enable deterministic, zero-hallucination supplier risk
assessment, compliance checking, tier depth benchmarking, and gap analysis
without external API dependencies. All data is version-tracked and
provenance-auditable.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from greenlang.agents.eudr.multi_tier_supplier.reference_data.country_risk_scores import (
    COUNTRY_RISK_SCORES,
    REGIONAL_RISK_AGGREGATES,
    TOTAL_COUNTRIES,
    get_composite_score,
    get_countries_by_commodity,
    get_country_risk,
    get_high_risk_countries,
    get_risk_by_region,
    get_risk_factors,
    is_high_risk,
)
from greenlang.agents.eudr.multi_tier_supplier.reference_data.certification_standards import (
    CERTIFICATION_STANDARDS,
    EUDR_ACCEPTANCE_LEVELS,
    TOTAL_CERTIFICATIONS,
    check_validity,
    get_all_certifications,
    get_certification,
    get_certification_hierarchy,
    get_certifications_for_commodity,
    is_eudr_accepted,
)
from greenlang.agents.eudr.multi_tier_supplier.reference_data.commodity_supply_chains import (
    COMMODITY_SUPPLY_CHAINS,
    INDUSTRY_VISIBILITY_BENCHMARKS,
    TOTAL_COMMODITIES,
    get_actor_types_for_tier,
    get_all_commodities,
    get_avg_tier_depth,
    get_industry_benchmark,
    get_tier_info,
    get_traceability_gaps,
    get_typical_chain,
    get_visibility_benchmark,
)

__all__ = [
    # country_risk_scores
    "COUNTRY_RISK_SCORES",
    "REGIONAL_RISK_AGGREGATES",
    "TOTAL_COUNTRIES",
    "get_country_risk",
    "get_risk_factors",
    "get_high_risk_countries",
    "get_risk_by_region",
    "get_countries_by_commodity",
    "get_composite_score",
    "is_high_risk",
    # certification_standards
    "CERTIFICATION_STANDARDS",
    "EUDR_ACCEPTANCE_LEVELS",
    "TOTAL_CERTIFICATIONS",
    "get_certification",
    "is_eudr_accepted",
    "get_certifications_for_commodity",
    "check_validity",
    "get_certification_hierarchy",
    "get_all_certifications",
    # commodity_supply_chains
    "COMMODITY_SUPPLY_CHAINS",
    "INDUSTRY_VISIBILITY_BENCHMARKS",
    "TOTAL_COMMODITIES",
    "get_typical_chain",
    "get_industry_benchmark",
    "get_actor_types_for_tier",
    "get_avg_tier_depth",
    "get_visibility_benchmark",
    "get_tier_info",
    "get_traceability_gaps",
    "get_all_commodities",
]
