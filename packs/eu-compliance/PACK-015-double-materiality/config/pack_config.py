"""
PACK-015 Double Materiality Assessment Pack - Configuration Manager

This module implements the DMAConfig and PackConfig classes that load,
merge, and validate all configuration for the Double Materiality Assessment
Pack. It provides comprehensive Pydantic v2 models for every aspect of the
DMA process per ESRS 1, Chapter 3: impact materiality scoring, financial
materiality scoring, stakeholder engagement, IRO identification and
classification, materiality matrix generation, ESRS topic mapping,
threshold calibration, and DMA report generation.

ESRS 1 Double Materiality Perspectives:
    - Impact Materiality: The undertaking's actual or potential, positive
      or negative impacts on people and the environment (inside-out).
      Assessed on severity (scale, scope, irremediable character) and
      likelihood (for potential impacts). ESRS 1 sect. 43-48.
    - Financial Materiality: Sustainability matters that create risks
      and opportunities affecting (or reasonably expected to affect) the
      undertaking's financial position, performance, and cash flows
      (outside-in). Assessed on magnitude and likelihood. ESRS 1 sect. 49-51.

Sector Types (NACE macro-sectors):
    - AGRICULTURE: NACE A - Agriculture, forestry and fishing
    - MINING: NACE B - Mining and quarrying
    - MANUFACTURING: NACE C - Manufacturing
    - ENERGY: NACE D - Electricity, gas, steam and air conditioning
    - WATER: NACE E - Water supply, sewerage, waste management
    - CONSTRUCTION: NACE F - Construction
    - RETAIL: NACE G - Wholesale and retail trade
    - TRANSPORT: NACE H - Transport and storage
    - HOSPITALITY: NACE I - Accommodation and food service
    - ICT: NACE J - Information and communication
    - FINANCIAL_SERVICES: NACE K - Financial and insurance activities
    - REAL_ESTATE: NACE L - Real estate activities
    - PROFESSIONAL_SERVICES: NACE M - Professional, scientific, technical
    - ADMINISTRATIVE: NACE N - Administrative and support services
    - PUBLIC_ADMINISTRATION: NACE O - Public administration and defence
    - EDUCATION: NACE P - Education
    - HEALTHCARE: NACE Q - Human health and social work
    - OTHER_SERVICES: NACE R-U - Other services

Company Sizes:
    - LARGE_ENTERPRISE: >500 employees, listed, first wave CSRD
    - LARGE_NON_LISTED: >250 employees, non-listed, second wave CSRD
    - MID_MARKET: 250-500 employees, third wave or voluntary
    - SME_LISTED: Listed SME, third wave CSRD (LSME standard)
    - SME: <250 employees, voluntary reporter
    - MICRO: <10 employees, voluntary reporter

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (large_enterprise / mid_market / sme /
       financial_services / manufacturing / multi_sector)
    3. Environment overrides (DMA_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - ESRS 1: General Requirements (Chapter 3 Double Materiality)
    - ESRS 2: General Disclosures (IRO-1, IRO-2, SBM-3)
    - CSRD: Directive (EU) 2022/2464
    - EFRAG IG 1: Materiality Assessment Implementation Guidance
    - EFRAG IG 2: Value Chain Implementation Guidance
    - EFRAG IG 3: ESRS Datapoints Implementation Guidance

Example:
    >>> config = PackConfig.from_preset("large_enterprise")
    >>> print(config.pack.sectors)
    [SectorType.GENERAL]
    >>> print(config.pack.impact_materiality.scoring_scale)
    10
    >>> print(config.pack.stakeholder_engagement.enabled)
    True
"""

import hashlib
import json
import logging
import os
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Enums - DMA-specific enumeration types (14 enums)
# =============================================================================


class SectorType(str, Enum):
    """NACE macro-sector classification for sector-specific materiality profiles."""

    AGRICULTURE = "AGRICULTURE"
    MINING = "MINING"
    MANUFACTURING = "MANUFACTURING"
    ENERGY = "ENERGY"
    WATER = "WATER"
    CONSTRUCTION = "CONSTRUCTION"
    RETAIL = "RETAIL"
    TRANSPORT = "TRANSPORT"
    HOSPITALITY = "HOSPITALITY"
    ICT = "ICT"
    FINANCIAL_SERVICES = "FINANCIAL_SERVICES"
    REAL_ESTATE = "REAL_ESTATE"
    PROFESSIONAL_SERVICES = "PROFESSIONAL_SERVICES"
    ADMINISTRATIVE = "ADMINISTRATIVE"
    PUBLIC_ADMINISTRATION = "PUBLIC_ADMINISTRATION"
    EDUCATION = "EDUCATION"
    HEALTHCARE = "HEALTHCARE"
    OTHER_SERVICES = "OTHER_SERVICES"
    GENERAL = "GENERAL"


class CompanySize(str, Enum):
    """Company size classification driving DMA complexity and scope."""

    LARGE_ENTERPRISE = "LARGE_ENTERPRISE"
    LARGE_NON_LISTED = "LARGE_NON_LISTED"
    MID_MARKET = "MID_MARKET"
    SME_LISTED = "SME_LISTED"
    SME = "SME"
    MICRO = "MICRO"


class StakeholderCategory(str, Enum):
    """Stakeholder categories per ESRS 1 sect. 22."""

    # Affected stakeholders
    EMPLOYEES = "EMPLOYEES"
    VALUE_CHAIN_WORKERS = "VALUE_CHAIN_WORKERS"
    LOCAL_COMMUNITIES = "LOCAL_COMMUNITIES"
    CONSUMERS = "CONSUMERS"
    INDIGENOUS_PEOPLES = "INDIGENOUS_PEOPLES"
    ECOSYSTEMS = "ECOSYSTEMS"
    # Users of sustainability statements
    INVESTORS = "INVESTORS"
    LENDERS = "LENDERS"
    CREDITORS = "CREDITORS"
    ASSET_MANAGERS = "ASSET_MANAGERS"
    INSURERS = "INSURERS"
    RATING_AGENCIES = "RATING_AGENCIES"
    REGULATORS = "REGULATORS"
    NGOS = "NGOS"
    TRADE_UNIONS = "TRADE_UNIONS"
    CUSTOMERS_B2B = "CUSTOMERS_B2B"


class MaterialityLevel(str, Enum):
    """Materiality determination level for a sustainability matter."""

    MATERIAL = "MATERIAL"
    NOT_MATERIAL = "NOT_MATERIAL"
    BORDERLINE = "BORDERLINE"
    NOT_ASSESSED = "NOT_ASSESSED"


class TimeHorizon(str, Enum):
    """Time horizon classification per ESRS 1 sect. 77."""

    SHORT_TERM = "SHORT_TERM"      # 0-1 year
    MEDIUM_TERM = "MEDIUM_TERM"    # 1-5 years
    LONG_TERM = "LONG_TERM"        # 5+ years (up to 10 years typically)
    VERY_LONG_TERM = "VERY_LONG_TERM"  # 10+ years (climate scenarios)


class ESRSTopic(str, Enum):
    """ESRS topical standard identifiers."""

    E1 = "E1"  # Climate change
    E2 = "E2"  # Pollution
    E3 = "E3"  # Water and marine resources
    E4 = "E4"  # Biodiversity and ecosystems
    E5 = "E5"  # Resource use and circular economy
    S1 = "S1"  # Own workforce
    S2 = "S2"  # Workers in the value chain
    S3 = "S3"  # Affected communities
    S4 = "S4"  # Consumers and end-users
    G1 = "G1"  # Business conduct


class ESRSSubTopic(str, Enum):
    """ESRS sub-topics per Appendix A of ESRS 1."""

    # E1 sub-topics
    E1_CLIMATE_CHANGE_ADAPTATION = "E1_CLIMATE_CHANGE_ADAPTATION"
    E1_CLIMATE_CHANGE_MITIGATION = "E1_CLIMATE_CHANGE_MITIGATION"
    E1_ENERGY = "E1_ENERGY"
    # E2 sub-topics
    E2_POLLUTION_OF_AIR = "E2_POLLUTION_OF_AIR"
    E2_POLLUTION_OF_WATER = "E2_POLLUTION_OF_WATER"
    E2_POLLUTION_OF_SOIL = "E2_POLLUTION_OF_SOIL"
    E2_POLLUTION_OF_LIVING_ORGANISMS = "E2_POLLUTION_OF_LIVING_ORGANISMS"
    E2_SUBSTANCES_OF_CONCERN = "E2_SUBSTANCES_OF_CONCERN"
    E2_SUBSTANCES_OF_VERY_HIGH_CONCERN = "E2_SUBSTANCES_OF_VERY_HIGH_CONCERN"
    E2_MICROPLASTICS = "E2_MICROPLASTICS"
    # E3 sub-topics
    E3_WATER_CONSUMPTION = "E3_WATER_CONSUMPTION"
    E3_WATER_WITHDRAWALS = "E3_WATER_WITHDRAWALS"
    E3_WATER_DISCHARGES = "E3_WATER_DISCHARGES"
    E3_WATER_DISCHARGES_IN_OCEANS = "E3_WATER_DISCHARGES_IN_OCEANS"
    E3_EXTRACTION_AND_USE_OF_MARINE_RESOURCES = "E3_EXTRACTION_AND_USE_OF_MARINE_RESOURCES"
    # E4 sub-topics
    E4_DIRECT_IMPACT_DRIVERS = "E4_DIRECT_IMPACT_DRIVERS"
    E4_IMPACTS_ON_SPECIES = "E4_IMPACTS_ON_SPECIES"
    E4_IMPACTS_ON_ECOSYSTEMS = "E4_IMPACTS_ON_ECOSYSTEMS"
    E4_IMPACTS_ON_ECOSYSTEM_SERVICES = "E4_IMPACTS_ON_ECOSYSTEM_SERVICES"
    # E5 sub-topics
    E5_RESOURCE_INFLOWS = "E5_RESOURCE_INFLOWS"
    E5_RESOURCE_OUTFLOWS = "E5_RESOURCE_OUTFLOWS"
    E5_WASTE = "E5_WASTE"
    # S1 sub-topics
    S1_WORKING_CONDITIONS = "S1_WORKING_CONDITIONS"
    S1_EQUAL_TREATMENT = "S1_EQUAL_TREATMENT"
    S1_OTHER_WORK_RELATED_RIGHTS = "S1_OTHER_WORK_RELATED_RIGHTS"
    # S2 sub-topics
    S2_WORKING_CONDITIONS = "S2_WORKING_CONDITIONS"
    S2_EQUAL_TREATMENT = "S2_EQUAL_TREATMENT"
    S2_OTHER_WORK_RELATED_RIGHTS = "S2_OTHER_WORK_RELATED_RIGHTS"
    # S3 sub-topics
    S3_COMMUNITIES_ECONOMIC_SOCIAL_CULTURAL_RIGHTS = "S3_COMMUNITIES_ECONOMIC_SOCIAL_CULTURAL_RIGHTS"
    S3_COMMUNITIES_CIVIL_POLITICAL_RIGHTS = "S3_COMMUNITIES_CIVIL_POLITICAL_RIGHTS"
    S3_INDIGENOUS_PEOPLES_RIGHTS = "S3_INDIGENOUS_PEOPLES_RIGHTS"
    # S4 sub-topics
    S4_INFORMATION_RELATED_IMPACTS = "S4_INFORMATION_RELATED_IMPACTS"
    S4_PERSONAL_SAFETY = "S4_PERSONAL_SAFETY"
    S4_SOCIAL_INCLUSION = "S4_SOCIAL_INCLUSION"
    # G1 sub-topics
    G1_CORPORATE_CULTURE = "G1_CORPORATE_CULTURE"
    G1_PROTECTION_OF_WHISTLEBLOWERS = "G1_PROTECTION_OF_WHISTLEBLOWERS"
    G1_ANIMAL_WELFARE = "G1_ANIMAL_WELFARE"
    G1_POLITICAL_ENGAGEMENT = "G1_POLITICAL_ENGAGEMENT"
    G1_MANAGEMENT_OF_RELATIONSHIPS_WITH_SUPPLIERS = "G1_MANAGEMENT_OF_RELATIONSHIPS_WITH_SUPPLIERS"
    G1_CORRUPTION_AND_BRIBERY = "G1_CORRUPTION_AND_BRIBERY"


class ScoringMethodology(str, Enum):
    """Scoring methodology for materiality assessment."""

    ABSOLUTE_CUTOFF = "ABSOLUTE_CUTOFF"       # Fixed score threshold
    PERCENTILE = "PERCENTILE"                  # Top N% of scored matters
    SECTOR_CALIBRATED = "SECTOR_CALIBRATED"    # Sector-adjusted thresholds
    EXPERT_JUDGMENT = "EXPERT_JUDGMENT"         # Sustainability committee validation
    COMBINED = "COMBINED"                       # Multiple methods combined


class IROType(str, Enum):
    """Impact, Risk, or Opportunity classification."""

    IMPACT = "IMPACT"
    RISK = "RISK"
    OPPORTUNITY = "OPPORTUNITY"


class IRODirection(str, Enum):
    """Direction of an IRO (positive or negative)."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class IROTemporality(str, Enum):
    """Temporality of an IRO (actual or potential)."""

    ACTUAL = "ACTUAL"
    POTENTIAL = "POTENTIAL"


class ValueChainPosition(str, Enum):
    """Position in the value chain where an IRO occurs."""

    OWN_OPERATIONS = "OWN_OPERATIONS"
    UPSTREAM = "UPSTREAM"
    DOWNSTREAM = "DOWNSTREAM"
    FULL_VALUE_CHAIN = "FULL_VALUE_CHAIN"


class ReportingFrequency(str, Enum):
    """Reporting and disclosure frequency."""

    ANNUAL = "ANNUAL"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    QUARTERLY = "QUARTERLY"


class DisclosureFormat(str, Enum):
    """Output format for disclosure documents."""

    XBRL = "XBRL"
    PDF = "PDF"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"


# =============================================================================
# Reference Data Constants
# =============================================================================

# ESRS topic display names and descriptions
ESRS_TOPIC_INFO: Dict[str, Dict[str, Any]] = {
    "E1": {
        "name": "Climate Change",
        "standard": "ESRS E1",
        "pillar": "Environmental",
        "description": "GHG emissions, energy consumption, climate change adaptation and mitigation",
        "mandatory_for_all": False,
        "key_disclosures": ["E1-1", "E1-2", "E1-3", "E1-4", "E1-5", "E1-6", "E1-7", "E1-8", "E1-9"],
    },
    "E2": {
        "name": "Pollution",
        "standard": "ESRS E2",
        "pillar": "Environmental",
        "description": "Pollution of air, water, soil; substances of concern and very high concern",
        "mandatory_for_all": False,
        "key_disclosures": ["E2-1", "E2-2", "E2-3", "E2-4", "E2-5", "E2-6"],
    },
    "E3": {
        "name": "Water and Marine Resources",
        "standard": "ESRS E3",
        "pillar": "Environmental",
        "description": "Water consumption, withdrawals, discharges; marine resource use",
        "mandatory_for_all": False,
        "key_disclosures": ["E3-1", "E3-2", "E3-3", "E3-4", "E3-5"],
    },
    "E4": {
        "name": "Biodiversity and Ecosystems",
        "standard": "ESRS E4",
        "pillar": "Environmental",
        "description": "Direct impact drivers on biodiversity; impacts on species and ecosystems",
        "mandatory_for_all": False,
        "key_disclosures": ["E4-1", "E4-2", "E4-3", "E4-4", "E4-5", "E4-6"],
    },
    "E5": {
        "name": "Resource Use and Circular Economy",
        "standard": "ESRS E5",
        "pillar": "Environmental",
        "description": "Resource inflows, outflows, waste management, circularity",
        "mandatory_for_all": False,
        "key_disclosures": ["E5-1", "E5-2", "E5-3", "E5-4", "E5-5", "E5-6"],
    },
    "S1": {
        "name": "Own Workforce",
        "standard": "ESRS S1",
        "pillar": "Social",
        "description": "Working conditions, equal treatment, other work-related rights",
        "mandatory_for_all": False,
        "key_disclosures": ["S1-1", "S1-2", "S1-3", "S1-4", "S1-5", "S1-6",
                            "S1-7", "S1-8", "S1-9", "S1-10", "S1-11", "S1-12",
                            "S1-13", "S1-14", "S1-15", "S1-16", "S1-17"],
    },
    "S2": {
        "name": "Workers in the Value Chain",
        "standard": "ESRS S2",
        "pillar": "Social",
        "description": "Working conditions, equal treatment for value chain workers",
        "mandatory_for_all": False,
        "key_disclosures": ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5"],
    },
    "S3": {
        "name": "Affected Communities",
        "standard": "ESRS S3",
        "pillar": "Social",
        "description": "Community rights, indigenous peoples' rights, civil and political rights",
        "mandatory_for_all": False,
        "key_disclosures": ["S3-1", "S3-2", "S3-3", "S3-4", "S3-5"],
    },
    "S4": {
        "name": "Consumers and End-Users",
        "standard": "ESRS S4",
        "pillar": "Social",
        "description": "Information-related impacts, personal safety, social inclusion",
        "mandatory_for_all": False,
        "key_disclosures": ["S4-1", "S4-2", "S4-3", "S4-4", "S4-5"],
    },
    "G1": {
        "name": "Business Conduct",
        "standard": "ESRS G1",
        "pillar": "Governance",
        "description": "Corporate culture, whistleblower protection, corruption, political engagement",
        "mandatory_for_all": False,
        "key_disclosures": ["G1-1", "G1-2", "G1-3", "G1-4", "G1-5", "G1-6"],
    },
}

# ESRS sub-topic mapping (topic -> list of sub-topics)
ESRS_SUBTOPIC_MAP: Dict[str, List[str]] = {
    "E1": [
        "E1_CLIMATE_CHANGE_ADAPTATION",
        "E1_CLIMATE_CHANGE_MITIGATION",
        "E1_ENERGY",
    ],
    "E2": [
        "E2_POLLUTION_OF_AIR",
        "E2_POLLUTION_OF_WATER",
        "E2_POLLUTION_OF_SOIL",
        "E2_POLLUTION_OF_LIVING_ORGANISMS",
        "E2_SUBSTANCES_OF_CONCERN",
        "E2_SUBSTANCES_OF_VERY_HIGH_CONCERN",
        "E2_MICROPLASTICS",
    ],
    "E3": [
        "E3_WATER_CONSUMPTION",
        "E3_WATER_WITHDRAWALS",
        "E3_WATER_DISCHARGES",
        "E3_WATER_DISCHARGES_IN_OCEANS",
        "E3_EXTRACTION_AND_USE_OF_MARINE_RESOURCES",
    ],
    "E4": [
        "E4_DIRECT_IMPACT_DRIVERS",
        "E4_IMPACTS_ON_SPECIES",
        "E4_IMPACTS_ON_ECOSYSTEMS",
        "E4_IMPACTS_ON_ECOSYSTEM_SERVICES",
    ],
    "E5": [
        "E5_RESOURCE_INFLOWS",
        "E5_RESOURCE_OUTFLOWS",
        "E5_WASTE",
    ],
    "S1": [
        "S1_WORKING_CONDITIONS",
        "S1_EQUAL_TREATMENT",
        "S1_OTHER_WORK_RELATED_RIGHTS",
    ],
    "S2": [
        "S2_WORKING_CONDITIONS",
        "S2_EQUAL_TREATMENT",
        "S2_OTHER_WORK_RELATED_RIGHTS",
    ],
    "S3": [
        "S3_COMMUNITIES_ECONOMIC_SOCIAL_CULTURAL_RIGHTS",
        "S3_COMMUNITIES_CIVIL_POLITICAL_RIGHTS",
        "S3_INDIGENOUS_PEOPLES_RIGHTS",
    ],
    "S4": [
        "S4_INFORMATION_RELATED_IMPACTS",
        "S4_PERSONAL_SAFETY",
        "S4_SOCIAL_INCLUSION",
    ],
    "G1": [
        "G1_CORPORATE_CULTURE",
        "G1_PROTECTION_OF_WHISTLEBLOWERS",
        "G1_ANIMAL_WELFARE",
        "G1_POLITICAL_ENGAGEMENT",
        "G1_MANAGEMENT_OF_RELATIONSHIPS_WITH_SUPPLIERS",
        "G1_CORRUPTION_AND_BRIBERY",
    ],
}

# Sector-specific materiality profiles: topics typically material per sector
SECTOR_MATERIALITY_PROFILES: Dict[str, Dict[str, str]] = {
    "MANUFACTURING": {
        "E1": "CRITICAL", "E2": "HIGH", "E3": "HIGH", "E4": "MEDIUM",
        "E5": "CRITICAL", "S1": "HIGH", "S2": "HIGH", "S3": "MEDIUM",
        "S4": "MEDIUM", "G1": "MEDIUM",
    },
    "FINANCIAL_SERVICES": {
        "E1": "CRITICAL", "E2": "LOW", "E3": "LOW", "E4": "MEDIUM",
        "E5": "LOW", "S1": "HIGH", "S2": "MEDIUM", "S3": "MEDIUM",
        "S4": "HIGH", "G1": "CRITICAL",
    },
    "ENERGY": {
        "E1": "CRITICAL", "E2": "CRITICAL", "E3": "HIGH", "E4": "HIGH",
        "E5": "HIGH", "S1": "HIGH", "S2": "HIGH", "S3": "HIGH",
        "S4": "MEDIUM", "G1": "MEDIUM",
    },
    "RETAIL": {
        "E1": "HIGH", "E2": "MEDIUM", "E3": "LOW", "E4": "MEDIUM",
        "E5": "HIGH", "S1": "HIGH", "S2": "CRITICAL", "S3": "MEDIUM",
        "S4": "HIGH", "G1": "MEDIUM",
    },
    "MINING": {
        "E1": "CRITICAL", "E2": "CRITICAL", "E3": "CRITICAL", "E4": "CRITICAL",
        "E5": "HIGH", "S1": "CRITICAL", "S2": "HIGH", "S3": "CRITICAL",
        "S4": "LOW", "G1": "HIGH",
    },
    "CONSTRUCTION": {
        "E1": "HIGH", "E2": "HIGH", "E3": "MEDIUM", "E4": "HIGH",
        "E5": "CRITICAL", "S1": "CRITICAL", "S2": "HIGH", "S3": "MEDIUM",
        "S4": "LOW", "G1": "MEDIUM",
    },
    "TRANSPORT": {
        "E1": "CRITICAL", "E2": "HIGH", "E3": "MEDIUM", "E4": "MEDIUM",
        "E5": "MEDIUM", "S1": "HIGH", "S2": "MEDIUM", "S3": "HIGH",
        "S4": "MEDIUM", "G1": "MEDIUM",
    },
    "ICT": {
        "E1": "HIGH", "E2": "LOW", "E3": "MEDIUM", "E4": "LOW",
        "E5": "HIGH", "S1": "HIGH", "S2": "MEDIUM", "S3": "LOW",
        "S4": "CRITICAL", "G1": "HIGH",
    },
    "HEALTHCARE": {
        "E1": "MEDIUM", "E2": "HIGH", "E3": "HIGH", "E4": "LOW",
        "E5": "HIGH", "S1": "CRITICAL", "S2": "MEDIUM", "S3": "HIGH",
        "S4": "CRITICAL", "G1": "HIGH",
    },
    "AGRICULTURE": {
        "E1": "CRITICAL", "E2": "HIGH", "E3": "CRITICAL", "E4": "CRITICAL",
        "E5": "HIGH", "S1": "HIGH", "S2": "HIGH", "S3": "HIGH",
        "S4": "MEDIUM", "G1": "MEDIUM",
    },
    "REAL_ESTATE": {
        "E1": "CRITICAL", "E2": "MEDIUM", "E3": "MEDIUM", "E4": "MEDIUM",
        "E5": "HIGH", "S1": "MEDIUM", "S2": "MEDIUM", "S3": "HIGH",
        "S4": "MEDIUM", "G1": "MEDIUM",
    },
    "HOSPITALITY": {
        "E1": "HIGH", "E2": "MEDIUM", "E3": "HIGH", "E4": "MEDIUM",
        "E5": "HIGH", "S1": "HIGH", "S2": "MEDIUM", "S3": "MEDIUM",
        "S4": "HIGH", "G1": "MEDIUM",
    },
    "GENERAL": {
        "E1": "HIGH", "E2": "MEDIUM", "E3": "MEDIUM", "E4": "MEDIUM",
        "E5": "MEDIUM", "S1": "HIGH", "S2": "MEDIUM", "S3": "MEDIUM",
        "S4": "MEDIUM", "G1": "MEDIUM",
    },
}

# Severity dimension weights for impact materiality
SEVERITY_DIMENSION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "EQUAL": {
        "scale": 0.333,
        "scope": 0.333,
        "irremediable_character": 0.334,
    },
    "SCALE_WEIGHTED": {
        "scale": 0.50,
        "scope": 0.25,
        "irremediable_character": 0.25,
    },
    "IRREMEDIABLE_WEIGHTED": {
        "scale": 0.25,
        "scope": 0.25,
        "irremediable_character": 0.50,
    },
}

# Financial magnitude thresholds by company size (EUR)
FINANCIAL_MAGNITUDE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "LARGE_ENTERPRISE": {
        "very_high": 50_000_000,
        "high": 10_000_000,
        "medium": 1_000_000,
        "low": 100_000,
    },
    "LARGE_NON_LISTED": {
        "very_high": 25_000_000,
        "high": 5_000_000,
        "medium": 500_000,
        "low": 50_000,
    },
    "MID_MARKET": {
        "very_high": 10_000_000,
        "high": 2_000_000,
        "medium": 200_000,
        "low": 20_000,
    },
    "SME_LISTED": {
        "very_high": 5_000_000,
        "high": 1_000_000,
        "medium": 100_000,
        "low": 10_000,
    },
    "SME": {
        "very_high": 2_000_000,
        "high": 500_000,
        "medium": 50_000,
        "low": 5_000,
    },
    "MICRO": {
        "very_high": 500_000,
        "high": 100_000,
        "medium": 10_000,
        "low": 1_000,
    },
}

# Available presets
AVAILABLE_PRESETS: Dict[str, str] = {
    "large_enterprise": "Large listed enterprise with full DMA granularity and multi-scorer",
    "mid_market": "Mid-market company with streamlined DMA scoring",
    "sme": "Listed SME or voluntary reporter with simplified DMA",
    "financial_services": "Financial sector with financed emissions and portfolio risk focus",
    "manufacturing": "Manufacturing sector with environmental impact emphasis",
    "multi_sector": "Diversified group with division-level DMA consolidation",
}


# =============================================================================
# Pydantic Sub-Config Models (10 models)
# =============================================================================


class ImpactMaterialityConfig(BaseModel):
    """Configuration for impact materiality scoring engine.

    Implements ESRS 1 sect. 43-48 scoring criteria for actual and potential,
    positive and negative impacts on people and the environment.
    """

    enabled: bool = Field(
        True,
        description="Enable impact materiality assessment",
    )
    scoring_scale: int = Field(
        5,
        ge=3,
        le=10,
        description="Scoring scale maximum (e.g., 5 for 1-5 scale, 10 for 1-10)",
    )
    severity_dimensions: List[str] = Field(
        default_factory=lambda: ["scale", "scope", "irremediable_character"],
        description="Severity dimensions per ESRS 1 sect. 44 (scale, scope, irremediable character)",
    )
    severity_weighting: str = Field(
        "EQUAL",
        description="Weighting scheme for severity dimensions: EQUAL, SCALE_WEIGHTED, IRREMEDIABLE_WEIGHTED",
    )
    likelihood_enabled: bool = Field(
        True,
        description="Include likelihood dimension for potential impacts (ESRS 1 sect. 46)",
    )
    likelihood_scale: int = Field(
        5,
        ge=3,
        le=10,
        description="Likelihood scoring scale (should match scoring_scale)",
    )
    positive_impacts_enabled: bool = Field(
        True,
        description="Assess positive impacts (actual and potential) alongside negatives",
    )
    value_chain_boundary: bool = Field(
        True,
        description="Include value chain impacts (upstream/downstream) in assessment",
    )
    multi_scorer: bool = Field(
        False,
        description="Enable multi-scorer mode (multiple assessors, aggregated scores)",
    )
    scorer_aggregation: str = Field(
        "mean",
        description="Aggregation method for multi-scorer: mean, median, weighted_mean",
    )
    evidence_based_scoring: bool = Field(
        True,
        description="Require evidence/data linkage for each score (MRV data, reports)",
    )
    sub_topic_granularity: bool = Field(
        True,
        description="Score at sub-topic level (not just topic level)",
    )
    sub_sub_topic_granularity: bool = Field(
        False,
        description="Score at sub-sub-topic level (maximum ESRS granularity)",
    )

    @field_validator("severity_dimensions")
    @classmethod
    def validate_severity_dimensions(cls, v: List[str]) -> List[str]:
        """Validate severity dimensions are valid ESRS dimensions."""
        valid = {"scale", "scope", "irremediable_character"}
        invalid = [d for d in v if d not in valid]
        if invalid:
            raise ValueError(
                f"Invalid severity dimensions: {invalid}. "
                f"Valid: {sorted(valid)}"
            )
        return v


class FinancialMaterialityConfig(BaseModel):
    """Configuration for financial materiality scoring engine.

    Implements ESRS 1 sect. 49-51 scoring criteria for sustainability-related
    risks and opportunities that affect or may affect financial performance.
    """

    enabled: bool = Field(
        True,
        description="Enable financial materiality assessment",
    )
    scoring_scale: int = Field(
        5,
        ge=3,
        le=10,
        description="Scoring scale maximum for magnitude and likelihood",
    )
    magnitude_type: str = Field(
        "qualitative",
        description="Magnitude assessment type: qualitative, quantitative, mixed",
    )
    magnitude_currency: str = Field(
        "EUR",
        description="Currency for quantitative magnitude assessment",
    )
    likelihood_enabled: bool = Field(
        True,
        description="Include likelihood dimension in financial scoring",
    )
    time_horizon_analysis: bool = Field(
        True,
        description="Assess financial materiality across time horizons per ESRS 1 sect. 77",
    )
    time_horizons: List[TimeHorizon] = Field(
        default_factory=lambda: [
            TimeHorizon.SHORT_TERM,
            TimeHorizon.MEDIUM_TERM,
            TimeHorizon.LONG_TERM,
        ],
        description="Time horizons to assess",
    )
    short_term_years: int = Field(
        1,
        ge=1,
        le=2,
        description="Short-term horizon definition in years",
    )
    medium_term_years: int = Field(
        5,
        ge=2,
        le=10,
        description="Medium-term horizon definition in years",
    )
    long_term_years: int = Field(
        10,
        ge=5,
        le=30,
        description="Long-term horizon definition in years",
    )
    scenario_analysis: bool = Field(
        False,
        description="Enable scenario analysis for climate-related financial risks",
    )
    scenario_types: List[str] = Field(
        default_factory=lambda: ["orderly_transition", "disorderly_transition", "hot_house"],
        description="Climate scenario types for financial risk assessment",
    )
    risk_opportunity_asymmetry: bool = Field(
        True,
        description="Handle asymmetric treatment of risks vs. opportunities",
    )
    cost_of_capital_impact: bool = Field(
        False,
        description="Assess impact on cost of capital and access to finance",
    )
    multi_scorer: bool = Field(
        False,
        description="Enable multi-scorer mode for financial assessment",
    )
    erp_data_integration: bool = Field(
        True,
        description="Integrate ERP financial data for quantitative magnitude estimates",
    )


class StakeholderConfig(BaseModel):
    """Configuration for stakeholder engagement engine.

    Implements ESRS 1 sect. 22-23 stakeholder identification and engagement.
    """

    enabled: bool = Field(
        True,
        description="Enable stakeholder engagement management",
    )
    affected_stakeholders: List[StakeholderCategory] = Field(
        default_factory=lambda: [
            StakeholderCategory.EMPLOYEES,
            StakeholderCategory.VALUE_CHAIN_WORKERS,
            StakeholderCategory.LOCAL_COMMUNITIES,
            StakeholderCategory.CONSUMERS,
        ],
        description="Affected stakeholder categories to engage per ESRS 1 sect. 22",
    )
    statement_users: List[StakeholderCategory] = Field(
        default_factory=lambda: [
            StakeholderCategory.INVESTORS,
            StakeholderCategory.LENDERS,
            StakeholderCategory.RATING_AGENCIES,
        ],
        description="Users of sustainability statements per ESRS 1 sect. 22",
    )
    engagement_methods: List[str] = Field(
        default_factory=lambda: [
            "surveys",
            "interviews",
            "workshops",
        ],
        description="Engagement methods: surveys, interviews, workshops, advisory_panels, grievance_mechanisms",
    )
    stakeholder_weighting: bool = Field(
        True,
        description="Apply weighting to stakeholder inputs based on proximity and vulnerability",
    )
    weighting_scheme: str = Field(
        "proximity_based",
        description="Weighting scheme: equal, proximity_based, vulnerability_weighted, expert_assigned",
    )
    sentiment_analysis: bool = Field(
        False,
        description="Enable LLM-assisted sentiment analysis on stakeholder responses",
    )
    min_response_rate: float = Field(
        0.20,
        ge=0.0,
        le=1.0,
        description="Minimum response rate threshold for engagement validity",
    )
    annual_refresh: bool = Field(
        True,
        description="Refresh stakeholder engagement annually",
    )
    grievance_mechanism_tracking: bool = Field(
        False,
        description="Track grievance mechanism inputs as stakeholder feedback",
    )

    @field_validator("affected_stakeholders")
    @classmethod
    def validate_affected_stakeholders(cls, v: List[StakeholderCategory]) -> List[StakeholderCategory]:
        """Deduplicate affected stakeholder categories."""
        seen: set = set()
        result: List[StakeholderCategory] = []
        for s in v:
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result


class IROConfig(BaseModel):
    """Configuration for IRO identification and classification engine.

    Implements ESRS 1 sect. 28-39 for impact, risk, and opportunity
    identification across all ESRS topics.
    """

    enabled: bool = Field(
        True,
        description="Enable IRO identification engine",
    )
    iro_types: List[IROType] = Field(
        default_factory=lambda: [IROType.IMPACT, IROType.RISK, IROType.OPPORTUNITY],
        description="IRO types to identify",
    )
    directions: List[IRODirection] = Field(
        default_factory=lambda: [IRODirection.POSITIVE, IRODirection.NEGATIVE],
        description="IRO directions to assess",
    )
    temporalities: List[IROTemporality] = Field(
        default_factory=lambda: [IROTemporality.ACTUAL, IROTemporality.POTENTIAL],
        description="IRO temporalities to assess",
    )
    value_chain_positions: List[ValueChainPosition] = Field(
        default_factory=lambda: [
            ValueChainPosition.OWN_OPERATIONS,
            ValueChainPosition.UPSTREAM,
            ValueChainPosition.DOWNSTREAM,
        ],
        description="Value chain positions to scan for IROs",
    )
    esrs_topics_in_scope: List[ESRSTopic] = Field(
        default_factory=lambda: [
            ESRSTopic.E1, ESRSTopic.E2, ESRSTopic.E3, ESRSTopic.E4, ESRSTopic.E5,
            ESRSTopic.S1, ESRSTopic.S2, ESRSTopic.S3, ESRSTopic.S4,
            ESRSTopic.G1,
        ],
        description="ESRS topics in scope for IRO identification",
    )
    sector_profile_enabled: bool = Field(
        True,
        description="Use sector materiality profile for IRO pre-population",
    )
    llm_assisted_identification: bool = Field(
        True,
        description="Use LLM for initial IRO identification suggestions (human-validated)",
    )
    max_iros_per_topic: int = Field(
        20,
        ge=1,
        le=50,
        description="Maximum IROs per ESRS topic (prevents over-proliferation)",
    )
    iro_register_export: bool = Field(
        True,
        description="Export IRO register in structured format (JSON, CSV)",
    )
    value_chain_depth: int = Field(
        3,
        ge=1,
        le=5,
        description="Depth of value chain tiers to scan for IROs",
    )


class MatrixConfig(BaseModel):
    """Configuration for materiality matrix generation engine.

    Produces the dual-axis materiality matrix combining impact and financial
    materiality scores.
    """

    enabled: bool = Field(
        True,
        description="Enable materiality matrix generation",
    )
    x_axis: str = Field(
        "financial_materiality",
        description="Matrix X-axis: financial_materiality or impact_materiality",
    )
    y_axis: str = Field(
        "impact_materiality",
        description="Matrix Y-axis: impact_materiality or financial_materiality",
    )
    output_formats: List[str] = Field(
        default_factory=lambda: ["html", "pdf", "svg"],
        description="Matrix visualization formats: html, pdf, svg, png",
    )
    interactive_drill_down: bool = Field(
        True,
        description="Enable interactive drill-down to underlying IROs (HTML format)",
    )
    color_by_pillar: bool = Field(
        True,
        description="Color-code matters by ESRS pillar (Environmental, Social, Governance)",
    )
    year_over_year_comparison: bool = Field(
        True,
        description="Show year-over-year movement of topics on matrix",
    )
    quadrant_labels: bool = Field(
        True,
        description="Label matrix quadrants (material-both, impact-only, financial-only, not-material)",
    )
    sensitivity_visualization: bool = Field(
        True,
        description="Include sensitivity analysis chart showing threshold impact",
    )
    borderline_highlighting: bool = Field(
        True,
        description="Highlight borderline matters near threshold boundaries",
    )
    topic_label_display: str = Field(
        "short_name",
        description="Topic label display: short_name, full_name, code_only",
    )


class ThresholdConfig(BaseModel):
    """Configuration for threshold scoring engine.

    Applies materiality thresholds to determine which sustainability
    matters cross the materiality boundary.
    """

    enabled: bool = Field(
        True,
        description="Enable threshold scoring",
    )
    methodology: ScoringMethodology = Field(
        ScoringMethodology.ABSOLUTE_CUTOFF,
        description="Primary threshold methodology",
    )
    impact_threshold: float = Field(
        3.0,
        ge=1.0,
        le=10.0,
        description="Impact materiality threshold (on scoring scale)",
    )
    financial_threshold: float = Field(
        3.0,
        ge=1.0,
        le=10.0,
        description="Financial materiality threshold (on scoring scale)",
    )
    percentile_cutoff: float = Field(
        0.60,
        ge=0.0,
        le=1.0,
        description="Percentile cutoff for percentile-based methodology (top 60%)",
    )
    sector_calibration: bool = Field(
        False,
        description="Calibrate thresholds using sector materiality profiles",
    )
    sensitivity_analysis: bool = Field(
        True,
        description="Run sensitivity analysis on threshold changes",
    )
    sensitivity_range_pct: float = Field(
        20.0,
        ge=5.0,
        le=50.0,
        description="Sensitivity analysis range (e.g., +/- 20% of threshold)",
    )
    borderline_treatment: str = Field(
        "include",
        description="Treatment of borderline matters: include, exclude, flag_for_review",
    )
    borderline_margin_pct: float = Field(
        10.0,
        ge=0.0,
        le=25.0,
        description="Margin around threshold that defines borderline zone (%)",
    )
    committee_validation: bool = Field(
        False,
        description="Require sustainability committee validation of threshold outcomes",
    )
    rationale_documentation: bool = Field(
        True,
        description="Document threshold rationale for audit trail",
    )

    @model_validator(mode="after")
    def validate_thresholds_within_scale(self) -> "ThresholdConfig":
        """Warn if thresholds seem miscalibrated (informational only)."""
        if self.impact_threshold > 7.0:
            logger.warning(
                "Impact materiality threshold %.1f is very high. "
                "This may result in very few material topics.",
                self.impact_threshold,
            )
        if self.financial_threshold > 7.0:
            logger.warning(
                "Financial materiality threshold %.1f is very high. "
                "This may result in very few material topics.",
                self.financial_threshold,
            )
        return self


class ESRSMappingConfig(BaseModel):
    """Configuration for ESRS topic mapping engine.

    Maps material sustainability matters to ESRS disclosure requirements.
    """

    enabled: bool = Field(
        True,
        description="Enable ESRS topic mapping",
    )
    include_mandatory_disclosures: bool = Field(
        True,
        description="Always include ESRS 2 mandatory disclosures regardless of materiality",
    )
    sub_topic_mapping: bool = Field(
        True,
        description="Map at sub-topic level (not just topic level)",
    )
    datapoint_mapping: bool = Field(
        False,
        description="Map to individual EFRAG XBRL datapoint level (most granular)",
    )
    iro_2_table_generation: bool = Field(
        True,
        description="Generate ESRS 2 IRO-2 disclosure table",
    )
    omission_rationale: bool = Field(
        True,
        description="Require rationale documentation for excluded disclosure requirements",
    )
    xbrl_tagging: bool = Field(
        True,
        description="Tag mapped disclosures with EFRAG XBRL taxonomy identifiers",
    )
    sector_standard_readiness: bool = Field(
        False,
        description="Pre-map to anticipated sector-specific ESRS standards",
    )
    cross_reference_validation: bool = Field(
        True,
        description="Validate cross-references between disclosure requirements",
    )
    lsme_standard_mapping: bool = Field(
        False,
        description="Map to LSME (Listed SME) standard instead of full ESRS",
    )


class ReportingConfig(BaseModel):
    """Configuration for DMA report generation and disclosure output."""

    enabled: bool = Field(
        True,
        description="Enable DMA report generation",
    )
    iro_1_disclosure: bool = Field(
        True,
        description="Generate ESRS 2 IRO-1 process description",
    )
    iro_2_disclosure: bool = Field(
        True,
        description="Generate ESRS 2 IRO-2 disclosure requirement table",
    )
    sbm_3_disclosure: bool = Field(
        True,
        description="Generate ESRS 2 SBM-3 material IROs and strategy narrative",
    )
    executive_summary: bool = Field(
        True,
        description="Generate board-level executive summary",
    )
    full_dma_report: bool = Field(
        True,
        description="Generate comprehensive DMA report",
    )
    audit_trail_report: bool = Field(
        True,
        description="Generate audit trail report for assurance",
    )
    output_formats: List[DisclosureFormat] = Field(
        default_factory=lambda: [DisclosureFormat.PDF, DisclosureFormat.XBRL],
        description="Output formats for DMA reports",
    )
    multi_language: bool = Field(
        False,
        description="Enable multi-language report generation",
    )
    supported_languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Supported languages for DMA reports",
    )
    review_workflow: bool = Field(
        True,
        description="Enable review and approval workflow for DMA reports",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved DMA documents",
    )
    llm_narrative_generation: bool = Field(
        True,
        description="Use LLM for narrative sections (methodology, IRO descriptions, SBM-3)",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for DMA audit trail and provenance tracking."""

    enabled: bool = Field(
        True,
        description="Enable audit trail for all DMA scoring and decisions",
    )
    sha256_provenance: bool = Field(
        True,
        description="Generate SHA-256 provenance hashes for all scoring outputs",
    )
    scoring_log: bool = Field(
        True,
        description="Log all individual scoring inputs and intermediate calculations",
    )
    assumption_tracking: bool = Field(
        True,
        description="Track all assumptions in DMA (weighting, boundaries, thresholds)",
    )
    data_lineage: bool = Field(
        True,
        description="Track full data lineage from source evidence to final scores",
    )
    retention_years: int = Field(
        10,
        ge=1,
        le=15,
        description="Audit trail retention period in years",
    )
    external_audit_export: bool = Field(
        True,
        description="Enable export format for external assurance providers",
    )
    version_control: bool = Field(
        True,
        description="Version-control DMA configurations and scoring results",
    )
    change_log: bool = Field(
        True,
        description="Maintain change log for all DMA parameter modifications",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class DMAConfig(BaseModel):
    """Main configuration for PACK-015 Double Materiality Assessment Pack.

    This is the root configuration model that contains all sub-configurations
    for the complete DMA process per ESRS 1 Chapter 3. The sectors and
    company_size fields drive scoring complexity, threshold calibration,
    and sector-specific materiality profiles.
    """

    # Company identification
    company_name: str = Field(
        "",
        description="Legal entity name of the undertaking",
    )
    reporting_year: int = Field(
        2025,
        ge=2024,
        le=2035,
        description="Reporting year for CSRD DMA",
    )
    company_size: CompanySize = Field(
        CompanySize.LARGE_ENTERPRISE,
        description="Company size classification (drives DMA complexity)",
    )
    sectors: List[SectorType] = Field(
        default_factory=lambda: [SectorType.GENERAL],
        description="NACE macro-sectors of the undertaking (may be multi-sector)",
    )
    listed_entity: bool = Field(
        False,
        description="Whether the entity is listed on a regulated market",
    )
    employee_count: Optional[int] = Field(
        None,
        ge=0,
        description="Average number of employees (for Omnibus threshold assessment)",
    )
    net_turnover_eur: Optional[float] = Field(
        None,
        ge=0,
        description="Net turnover in EUR (for Omnibus threshold assessment)",
    )
    total_assets_eur: Optional[float] = Field(
        None,
        ge=0,
        description="Total assets in EUR (for Omnibus threshold assessment)",
    )

    # Engine sub-configurations
    impact_materiality: ImpactMaterialityConfig = Field(
        default_factory=ImpactMaterialityConfig,
        description="Impact materiality engine configuration (ESRS 1 sect. 43-48)",
    )
    financial_materiality: FinancialMaterialityConfig = Field(
        default_factory=FinancialMaterialityConfig,
        description="Financial materiality engine configuration (ESRS 1 sect. 49-51)",
    )
    stakeholder_engagement: StakeholderConfig = Field(
        default_factory=StakeholderConfig,
        description="Stakeholder engagement engine configuration (ESRS 1 sect. 22-23)",
    )
    iro_identification: IROConfig = Field(
        default_factory=IROConfig,
        description="IRO identification engine configuration (ESRS 1 sect. 28-39)",
    )
    materiality_matrix: MatrixConfig = Field(
        default_factory=MatrixConfig,
        description="Materiality matrix engine configuration",
    )
    threshold: ThresholdConfig = Field(
        default_factory=ThresholdConfig,
        description="Threshold scoring engine configuration",
    )
    esrs_mapping: ESRSMappingConfig = Field(
        default_factory=ESRSMappingConfig,
        description="ESRS topic mapping engine configuration",
    )

    # Supporting configurations
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="DMA report generation configuration",
    )
    audit_trail: AuditTrailConfig = Field(
        default_factory=AuditTrailConfig,
        description="Audit trail and provenance configuration",
    )

    @model_validator(mode="after")
    def validate_scoring_scales_consistent(self) -> "DMAConfig":
        """Ensure impact and financial scoring scales are consistent."""
        if self.impact_materiality.scoring_scale != self.financial_materiality.scoring_scale:
            logger.warning(
                "Impact scoring scale (%d) differs from financial scoring scale (%d). "
                "This may cause confusion in the materiality matrix. Consider "
                "using the same scale for both.",
                self.impact_materiality.scoring_scale,
                self.financial_materiality.scoring_scale,
            )
        return self

    @model_validator(mode="after")
    def validate_thresholds_match_scale(self) -> "DMAConfig":
        """Ensure thresholds are within the scoring scale range."""
        impact_scale = self.impact_materiality.scoring_scale
        financial_scale = self.financial_materiality.scoring_scale
        if self.threshold.impact_threshold > impact_scale:
            logger.warning(
                "Impact threshold (%.1f) exceeds impact scoring scale (%d). "
                "Adjusting threshold to scale maximum.",
                self.threshold.impact_threshold,
                impact_scale,
            )
            object.__setattr__(
                self.threshold, "impact_threshold", float(impact_scale)
            )
        if self.threshold.financial_threshold > financial_scale:
            logger.warning(
                "Financial threshold (%.1f) exceeds financial scoring scale (%d). "
                "Adjusting threshold to scale maximum.",
                self.threshold.financial_threshold,
                financial_scale,
            )
            object.__setattr__(
                self.threshold, "financial_threshold", float(financial_scale)
            )
        return self

    @model_validator(mode="after")
    def validate_sme_simplification(self) -> "DMAConfig":
        """Apply SME simplifications if company_size indicates SME."""
        sme_sizes = {CompanySize.SME, CompanySize.SME_LISTED, CompanySize.MICRO}
        if self.company_size in sme_sizes:
            if self.impact_materiality.sub_sub_topic_granularity:
                logger.warning(
                    "Sub-sub-topic granularity is unusual for SMEs. "
                    "Consider disabling for efficiency."
                )
            if self.financial_materiality.scenario_analysis:
                logger.warning(
                    "Scenario analysis is typically not required for SMEs. "
                    "Consider disabling for efficiency."
                )
        return self

    @model_validator(mode="after")
    def validate_financial_services_profile(self) -> "DMAConfig":
        """Apply financial services specific validations."""
        if SectorType.FINANCIAL_SERVICES in self.sectors:
            if not self.financial_materiality.cost_of_capital_impact:
                logger.warning(
                    "Financial services sector: cost-of-capital impact assessment "
                    "is recommended. Consider enabling cost_of_capital_impact."
                )
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper.

    Handles preset loading, environment variable overrides, and
    configuration merging.
    """

    pack: DMAConfig = Field(
        default_factory=DMAConfig,
        description="Main DMA configuration",
    )
    preset_name: Optional[str] = Field(
        None,
        description="Name of the loaded preset",
    )
    config_version: str = Field(
        "1.0.0",
        description="Configuration schema version",
    )
    pack_id: str = Field(
        "PACK-015-double-materiality",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (large_enterprise, mid_market, etc.)
            overrides: Optional dictionary of configuration overrides.

        Returns:
            PackConfig instance with preset values applied.

        Raises:
            FileNotFoundError: If preset YAML file does not exist.
            ValueError: If preset_name is not in AVAILABLE_PRESETS.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available presets: {sorted(AVAILABLE_PRESETS.keys())}"
            )

        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(
                f"Preset file not found: {preset_path}. "
                f"Run setup wizard to generate presets."
            )

        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            preset_data = cls._deep_merge(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = cls._deep_merge(preset_data, overrides)

        pack_config = DMAConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            PackConfig instance with YAML values applied.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        pack_config = DMAConfig(**config_data)
        return cls(pack=pack_config)

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """Load configuration overrides from environment variables.

        Environment variables prefixed with DMA_PACK_ are loaded and mapped
        to configuration keys. Nested keys use double underscore.

        Example: DMA_PACK_IMPACT_MATERIALITY__SCORING_SCALE=10
        """
        overrides: Dict[str, Any] = {}
        prefix = "DMA_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                current = overrides
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
                # Parse value
                if value.lower() in ("true", "yes", "1"):
                    current[parts[-1]] = True
                elif value.lower() in ("false", "no", "0"):
                    current[parts[-1]] = False
                else:
                    try:
                        current[parts[-1]] = int(value)
                    except ValueError:
                        try:
                            current[parts[-1]] = float(value)
                        except ValueError:
                            current[parts[-1]] = value
        return overrides

    @staticmethod
    def _deep_merge(
        base: Dict[str, Any],
        override: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = PackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_config_hash(self) -> str:
        """Generate SHA-256 hash of the current configuration for provenance."""
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


# =============================================================================
# Utility Functions
# =============================================================================


def load_preset(
    preset_name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> PackConfig:
    """Load a named preset configuration.

    Convenience wrapper around PackConfig.from_preset().

    Args:
        preset_name: Name of the preset to load.
        overrides: Optional configuration overrides.

    Returns:
        PackConfig instance with preset applied.
    """
    return PackConfig.from_preset(preset_name, overrides)


def validate_config(config: DMAConfig) -> List[str]:
    """Validate a DMA configuration and return any warnings.

    Args:
        config: DMAConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check that at least one sector is configured
    if not config.sectors:
        warnings.append(
            "No sectors configured. Add at least one sector for meaningful "
            "sector-specific materiality profiles."
        )

    # Check scoring scale consistency
    if (
        config.impact_materiality.scoring_scale
        != config.financial_materiality.scoring_scale
    ):
        warnings.append(
            f"Impact scale ({config.impact_materiality.scoring_scale}) differs "
            f"from financial scale ({config.financial_materiality.scoring_scale}). "
            f"Using different scales complicates matrix interpretation."
        )

    # Check threshold within scale
    impact_scale = config.impact_materiality.scoring_scale
    if config.threshold.impact_threshold > impact_scale:
        warnings.append(
            f"Impact threshold ({config.threshold.impact_threshold}) exceeds "
            f"impact scoring scale ({impact_scale})."
        )

    financial_scale = config.financial_materiality.scoring_scale
    if config.threshold.financial_threshold > financial_scale:
        warnings.append(
            f"Financial threshold ({config.threshold.financial_threshold}) exceeds "
            f"financial scoring scale ({financial_scale})."
        )

    # Check all 10 ESRS topics are in scope for comprehensive DMA
    all_topics = {
        ESRSTopic.E1, ESRSTopic.E2, ESRSTopic.E3, ESRSTopic.E4, ESRSTopic.E5,
        ESRSTopic.S1, ESRSTopic.S2, ESRSTopic.S3, ESRSTopic.S4, ESRSTopic.G1,
    }
    configured_topics = set(config.iro_identification.esrs_topics_in_scope)
    missing_topics = all_topics - configured_topics
    if missing_topics:
        topic_names = ", ".join(t.value for t in sorted(missing_topics, key=lambda t: t.value))
        warnings.append(
            f"ESRS topics not in scope: {topic_names}. A comprehensive DMA "
            f"should initially consider all 10 topical standards."
        )

    # Check stakeholder engagement completeness
    if config.stakeholder_engagement.enabled:
        if len(config.stakeholder_engagement.affected_stakeholders) < 2:
            warnings.append(
                "Fewer than 2 affected stakeholder categories configured. "
                "ESRS 1 sect. 22 expects engagement with multiple stakeholder groups."
            )
        if len(config.stakeholder_engagement.statement_users) < 1:
            warnings.append(
                "No users of sustainability statements configured. "
                "ESRS 1 sect. 22 requires consideration of investor/lender perspectives."
            )

    # Check financial services specific requirements
    if SectorType.FINANCIAL_SERVICES in config.sectors:
        if not config.financial_materiality.scenario_analysis:
            warnings.append(
                "Financial services sector: scenario analysis is strongly "
                "recommended for climate-related financial risk assessment."
            )

    # Check SME appropriateness
    sme_sizes = {CompanySize.SME, CompanySize.SME_LISTED, CompanySize.MICRO}
    if config.company_size in sme_sizes:
        if config.impact_materiality.scoring_scale > 5:
            warnings.append(
                "Scoring scale >5 is unusual for SMEs. Consider 1-5 for simplicity."
            )
        if config.impact_materiality.multi_scorer:
            warnings.append(
                "Multi-scorer mode is unusual for SMEs. Single-scorer is typically sufficient."
            )

    return warnings


def get_default_config(
    sector: SectorType = SectorType.GENERAL,
    company_size: CompanySize = CompanySize.LARGE_ENTERPRISE,
) -> DMAConfig:
    """Get default DMA configuration for a given sector and company size.

    Args:
        sector: NACE macro-sector.
        company_size: Company size classification.

    Returns:
        DMAConfig instance with sector/size-appropriate defaults.
    """
    return DMAConfig(
        sectors=[sector],
        company_size=company_size,
    )


def get_sector_info(sector: Union[str, SectorType]) -> Dict[str, Any]:
    """Get sector materiality profile information.

    Args:
        sector: Sector enum or string value.

    Returns:
        Dictionary with sector-specific materiality importance levels per ESRS topic.
    """
    key = sector.value if isinstance(sector, SectorType) else sector
    profile = SECTOR_MATERIALITY_PROFILES.get(key, SECTOR_MATERIALITY_PROFILES["GENERAL"])
    return {
        "sector": key,
        "materiality_profile": profile,
    }


def get_esrs_topic_info(topic: Union[str, ESRSTopic]) -> Dict[str, Any]:
    """Get detailed information about an ESRS topic.

    Args:
        topic: ESRS topic enum or string value.

    Returns:
        Dictionary with topic name, standard, pillar, description, and disclosures.
    """
    key = topic.value if isinstance(topic, ESRSTopic) else topic
    return ESRS_TOPIC_INFO.get(key, {
        "name": key,
        "standard": f"ESRS {key}",
        "pillar": "Unknown",
        "description": "Unknown topic",
        "mandatory_for_all": False,
        "key_disclosures": [],
    })


def get_subtopics_for_topic(topic: Union[str, ESRSTopic]) -> List[str]:
    """Get sub-topics for a given ESRS topic.

    Args:
        topic: ESRS topic enum or string value.

    Returns:
        List of sub-topic identifier strings.
    """
    key = topic.value if isinstance(topic, ESRSTopic) else topic
    return ESRS_SUBTOPIC_MAP.get(key, [])


def list_available_presets() -> Dict[str, str]:
    """List all available DMA configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return AVAILABLE_PRESETS.copy()
