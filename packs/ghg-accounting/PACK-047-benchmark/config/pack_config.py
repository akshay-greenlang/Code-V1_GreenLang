"""
PACK-047 GHG Emissions Benchmark Pack - Configuration Manager

Pydantic v2 configuration for GHG emissions benchmarking including peer group
construction, scope normalisation, external dataset integration, pathway
alignment scoring, implied temperature rise (ITR) calculation, trajectory
benchmarking, portfolio-level carbon benchmarking, data quality scoring,
transition risk scoring, and automated benchmark reporting.

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (sector-specific defaults)
    3. Environment overrides (BENCHMARK_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    EU CSRD / ESRS E1-6 - Climate change benchmark disclosures
    SBTi Corporate Manual v2.1 - Sectoral Decarbonisation Approach (SDA)
    CDP Climate Change Questionnaire C4.1/C4.2 (2026) - Targets and performance
    US SEC Climate Disclosure Rules (2024) - Peer comparison metrics
    ISO 14064-1:2018 Clause 5 - Quantification benchmarks
    TCFD Recommendations - Metrics and Targets (cross-industry benchmarks)
    PCAF Global GHG Accounting Standard v3 (2024) - Data quality scoring
    IFRS S2 (2023) - Climate-related disclosures benchmark metrics
    EU SFDR (2021) - PAI indicators and benchmark alignment
    IEA Net Zero by 2050 Roadmap (2023 update)
    IPCC AR6 WGIII - Mitigation pathways (C1-C3 scenarios)
    TPI Carbon Performance methodology v5.0

Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Helper Functions
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime (mockable for testing)."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Return new UUID4 string (mockable for testing)."""
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string for provenance tracking."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# Enums (18 total)
# =============================================================================


class SectorClassification(str, Enum):
    """Sector classification system for peer group construction."""
    GICS_2DIG = "GICS_2DIG"
    GICS_4DIG = "GICS_4DIG"
    GICS_6DIG = "GICS_6DIG"
    GICS_8DIG = "GICS_8DIG"
    NACE_2DIG = "NACE_2DIG"
    NACE_3DIG = "NACE_3DIG"
    NACE_4DIG = "NACE_4DIG"
    ISIC_2DIG = "ISIC_2DIG"
    ISIC_3DIG = "ISIC_3DIG"
    ISIC_4DIG = "ISIC_4DIG"
    SIC_2DIG = "SIC_2DIG"
    SIC_3DIG = "SIC_3DIG"
    SIC_4DIG = "SIC_4DIG"
    CUSTOM = "CUSTOM"


class PeerSizeBand(str, Enum):
    """Revenue-based size band for peer group matching."""
    MICRO = "MICRO"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    ENTERPRISE = "ENTERPRISE"
    MEGA = "MEGA"


class ScopeAlignment(str, Enum):
    """Scope alignment configuration for normalisation."""
    S1_ONLY = "S1_ONLY"
    S1_S2L = "S1_S2L"
    S1_S2M = "S1_S2M"
    S1_S2_S3 = "S1_S2_S3"
    CUSTOM = "CUSTOM"


class ConsolidationApproach(str, Enum):
    """Organisational boundary consolidation approach per GHG Protocol Ch 3."""
    EQUITY_SHARE = "EQUITY_SHARE"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"


class GWPVersion(str, Enum):
    """IPCC Global Warming Potential assessment report version."""
    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"


class PathwayType(str, Enum):
    """Science-based pathway type for alignment scoring."""
    IEA_NZE = "IEA_NZE"
    IPCC_AR6_C1 = "IPCC_AR6_C1"
    IPCC_AR6_C2 = "IPCC_AR6_C2"
    IPCC_AR6_C3 = "IPCC_AR6_C3"
    SBTI_SDA = "SBTI_SDA"
    OECM = "OECM"
    TPI_CP = "TPI_CP"
    CRREM = "CRREM"


class PathwayScenario(str, Enum):
    """Temperature scenario for pathway alignment."""
    ONE_POINT_FIVE_C = "ONE_POINT_FIVE_C"
    WELL_BELOW_2C = "WELL_BELOW_2C"
    BELOW_2C = "BELOW_2C"


class ITRMethod(str, Enum):
    """Implied temperature rise calculation method."""
    BUDGET_BASED = "BUDGET_BASED"
    SECTOR_RELATIVE = "SECTOR_RELATIVE"
    RATE_OF_REDUCTION = "RATE_OF_REDUCTION"


class PortfolioAssetClass(str, Enum):
    """Asset class for portfolio-level benchmarking (PCAF aligned)."""
    LISTED_EQUITY = "LISTED_EQUITY"
    CORPORATE_BONDS = "CORPORATE_BONDS"
    PROJECT_FINANCE = "PROJECT_FINANCE"
    COMMERCIAL_RE = "COMMERCIAL_RE"
    MORTGAGES = "MORTGAGES"
    SOVEREIGN_DEBT = "SOVEREIGN_DEBT"


class PCAFScore(int, Enum):
    """PCAF data quality score (1 = best, 5 = worst)."""
    SCORE_1 = 1
    SCORE_2 = 2
    SCORE_3 = 3
    SCORE_4 = 4
    SCORE_5 = 5


class DataSourceType(str, Enum):
    """External benchmark data source type."""
    CDP = "CDP"
    TPI = "TPI"
    GRESB = "GRESB"
    CRREM = "CRREM"
    ISS_ESG = "ISS_ESG"
    CUSTOM = "CUSTOM"


class TransitionRiskCategory(str, Enum):
    """Transition risk category for scoring."""
    CARBON_BUDGET = "CARBON_BUDGET"
    STRANDING = "STRANDING"
    REGULATORY = "REGULATORY"
    COMPETITIVE = "COMPETITIVE"
    FINANCIAL = "FINANCIAL"


class BenchmarkMetric(str, Enum):
    """Benchmark metric type for reporting."""
    ABSOLUTE = "ABSOLUTE"
    INTENSITY = "INTENSITY"
    WACI = "WACI"
    CARBON_FOOTPRINT = "CARBON_FOOTPRINT"
    ITR = "ITR"
    TRANSITION_RISK = "TRANSITION_RISK"


class ReportFormat(str, Enum):
    """Supported report output formats."""
    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    PDF = "PDF"
    JSON = "JSON"
    CSV = "CSV"
    XBRL = "XBRL"


class DisclosureFramework(str, Enum):
    """Supported regulatory and voluntary reporting frameworks."""
    ESRS = "ESRS"
    CDP = "CDP"
    SFDR = "SFDR"
    TCFD = "TCFD"
    SEC = "SEC"
    GRI = "GRI"


class AlertType(str, Enum):
    """Alert trigger type for benchmark monitoring."""
    THRESHOLD = "THRESHOLD"
    DEADLINE = "DEADLINE"
    PATHWAY_DEVIATION = "PATHWAY_DEVIATION"
    DATA_STALENESS = "DATA_STALENESS"
    QUALITY_DROP = "QUALITY_DROP"
    RANK_CHANGE = "RANK_CHANGE"


class QualityDimension(str, Enum):
    """Data quality dimension for PCAF scoring."""
    TEMPORAL = "TEMPORAL"
    GEOGRAPHIC = "GEOGRAPHIC"
    TECHNOLOGICAL = "TECHNOLOGICAL"
    COMPLETENESS = "COMPLETENESS"
    RELIABILITY = "RELIABILITY"


class NormalisationStep(str, Enum):
    """Normalisation step type in the pipeline."""
    SCOPE_ALIGN = "SCOPE_ALIGN"
    CONSOLIDATION = "CONSOLIDATION"
    GWP = "GWP"
    CURRENCY = "CURRENCY"
    PERIOD = "PERIOD"
    DATA_GAP = "DATA_GAP"
    BIOGENIC = "BIOGENIC"
    CLIMATE = "CLIMATE"


# =============================================================================
# Reference Data Constants
# =============================================================================


IPCC_CARBON_BUDGETS: Dict[str, Dict[str, Any]] = {
    "1.5C": {
        "temperature": Decimal("1.5"),
        "remaining_budget_gt_co2": Decimal("400"),
        "from_year": 2020,
        "probability_pct": 50,
        "source": "IPCC AR6 WGIII Table SPM.2 (2022)",
        "description": "Remaining carbon budget for 50% chance of limiting warming to 1.5C",
    },
    "1.7C": {
        "temperature": Decimal("1.7"),
        "remaining_budget_gt_co2": Decimal("700"),
        "from_year": 2020,
        "probability_pct": 50,
        "source": "IPCC AR6 WGIII Table SPM.2 (2022)",
        "description": "Remaining carbon budget for 50% chance of limiting warming to 1.7C",
    },
    "2.0C": {
        "temperature": Decimal("2.0"),
        "remaining_budget_gt_co2": Decimal("1150"),
        "from_year": 2020,
        "probability_pct": 67,
        "source": "IPCC AR6 WGIII Table SPM.2 (2022)",
        "description": "Remaining carbon budget for 67% chance of limiting warming to 2.0C",
    },
}


SBTI_SECTOR_PATHWAYS: Dict[str, Dict[str, Any]] = {
    "power": {
        "sector_name": "Electricity Generation",
        "metric": "tCO2e/MWh",
        "base_year": 2020,
        "base_intensity": Decimal("0.48"),
        "target_year": 2030,
        "target_intensity": Decimal("0.14"),
        "pathway_1_5c": {
            "2025": Decimal("0.28"),
            "2030": Decimal("0.07"),
            "2035": Decimal("0.01"),
            "2040": Decimal("0.00"),
            "2050": Decimal("0.00"),
        },
        "source": "SBTi SDA Power Sector v2.0",
    },
    "steel": {
        "sector_name": "Iron and Steel",
        "metric": "tCO2e/tonne_steel",
        "base_year": 2020,
        "base_intensity": Decimal("1.81"),
        "target_year": 2030,
        "target_intensity": Decimal("1.10"),
        "pathway_1_5c": {
            "2025": Decimal("1.50"),
            "2030": Decimal("1.10"),
            "2035": Decimal("0.80"),
            "2040": Decimal("0.55"),
            "2050": Decimal("0.18"),
        },
        "source": "SBTi SDA Iron & Steel Sector v2.0",
    },
    "cement": {
        "sector_name": "Cement",
        "metric": "tCO2e/tonne_cite",
        "base_year": 2020,
        "base_intensity": Decimal("0.63"),
        "target_year": 2030,
        "target_intensity": Decimal("0.40"),
        "pathway_1_5c": {
            "2025": Decimal("0.52"),
            "2030": Decimal("0.40"),
            "2035": Decimal("0.29"),
            "2040": Decimal("0.20"),
            "2050": Decimal("0.06"),
        },
        "source": "SBTi SDA Cement Sector v2.0",
    },
    "aluminium": {
        "sector_name": "Aluminium",
        "metric": "tCO2e/tonne_aluminium",
        "base_year": 2020,
        "base_intensity": Decimal("8.60"),
        "target_year": 2030,
        "target_intensity": Decimal("4.80"),
        "pathway_1_5c": {
            "2025": Decimal("7.00"),
            "2030": Decimal("4.80"),
            "2035": Decimal("3.10"),
            "2040": Decimal("1.80"),
            "2050": Decimal("0.50"),
        },
        "source": "SBTi SDA Aluminium Sector v2.0",
    },
    "buildings": {
        "sector_name": "Commercial Buildings",
        "metric": "kgCO2e/m2",
        "base_year": 2020,
        "base_intensity": Decimal("50.00"),
        "target_year": 2030,
        "target_intensity": Decimal("24.00"),
        "pathway_1_5c": {
            "2025": Decimal("38.00"),
            "2030": Decimal("24.00"),
            "2035": Decimal("14.00"),
            "2040": Decimal("7.00"),
            "2050": Decimal("0.50"),
        },
        "source": "SBTi SDA Buildings Sector v2.0 / CRREM 1.5C",
    },
    "transport": {
        "sector_name": "Road Freight Transport",
        "metric": "gCO2e/tkm",
        "base_year": 2020,
        "base_intensity": Decimal("85.00"),
        "target_year": 2030,
        "target_intensity": Decimal("50.00"),
        "pathway_1_5c": {
            "2025": Decimal("72.00"),
            "2030": Decimal("50.00"),
            "2035": Decimal("33.00"),
            "2040": Decimal("20.00"),
            "2050": Decimal("3.00"),
        },
        "source": "SBTi SDA Transport Sector v2.0",
    },
    "paper": {
        "sector_name": "Pulp and Paper",
        "metric": "tCO2e/tonne_paper",
        "base_year": 2020,
        "base_intensity": Decimal("0.45"),
        "target_year": 2030,
        "target_intensity": Decimal("0.28"),
        "pathway_1_5c": {
            "2025": Decimal("0.37"),
            "2030": Decimal("0.28"),
            "2035": Decimal("0.20"),
            "2040": Decimal("0.13"),
            "2050": Decimal("0.03"),
        },
        "source": "SBTi SDA Pulp & Paper Sector v2.0",
    },
    "food": {
        "sector_name": "Food and Agriculture (FLAG)",
        "metric": "tCO2e/tonne_product",
        "base_year": 2020,
        "base_intensity": Decimal("0.95"),
        "target_year": 2030,
        "target_intensity": Decimal("0.60"),
        "pathway_1_5c": {
            "2025": Decimal("0.82"),
            "2030": Decimal("0.60"),
            "2035": Decimal("0.42"),
            "2040": Decimal("0.30"),
            "2050": Decimal("0.15"),
        },
        "source": "SBTi FLAG Guidance v1.1",
    },
    "chemicals": {
        "sector_name": "Chemicals",
        "metric": "tCO2e/tonne_product",
        "base_year": 2020,
        "base_intensity": Decimal("1.20"),
        "target_year": 2030,
        "target_intensity": Decimal("0.82"),
        "pathway_1_5c": {
            "2025": Decimal("1.02"),
            "2030": Decimal("0.82"),
            "2035": Decimal("0.60"),
            "2040": Decimal("0.40"),
            "2050": Decimal("0.12"),
        },
        "source": "SBTi SDA Chemicals Sector v2.0",
    },
}


GWP_CONVERSION_FACTORS: Dict[str, Dict[str, Dict[str, Decimal]]] = {
    "AR4_to_AR6": {
        "CO2": {"factor": Decimal("1.000"), "note": "No change across ARs"},
        "CH4": {"factor": Decimal("1.107"), "note": "AR4=25, AR6=27.9 (fossil); ratio=27.9/25"},
        "N2O": {"factor": Decimal("0.930"), "note": "AR4=298, AR6=273; ratio=273/298"},
        "HFCs": {"factor": Decimal("1.050"), "note": "Approximate weighted average for HFC mix"},
    },
    "AR5_to_AR6": {
        "CO2": {"factor": Decimal("1.000"), "note": "No change across ARs"},
        "CH4": {"factor": Decimal("0.996"), "note": "AR5=28, AR6=27.9 (fossil); ratio=27.9/28"},
        "N2O": {"factor": Decimal("1.028"), "note": "AR5=265, AR6=273; ratio=273/265"},
        "HFCs": {"factor": Decimal("1.010"), "note": "Approximate weighted average for HFC mix"},
    },
}


PCAF_QUALITY_THRESHOLDS: Dict[int, Dict[str, Any]] = {
    1: {
        "label": "Audited / Verified",
        "description": "Reported emissions from the company, verified by a third party",
        "typical_uncertainty_pct": Decimal("2.0"),
        "min_uncertainty_pct": Decimal("1.0"),
        "max_uncertainty_pct": Decimal("5.0"),
        "requirements": [
            "Third-party verified emissions data",
            "Direct measurement or calculation from primary data",
            "Company-specific emission factors",
        ],
    },
    2: {
        "label": "Reported / Unverified",
        "description": "Reported emissions from the company, not verified",
        "typical_uncertainty_pct": Decimal("10.0"),
        "min_uncertainty_pct": Decimal("5.0"),
        "max_uncertainty_pct": Decimal("15.0"),
        "requirements": [
            "Company-reported emissions data",
            "Calculated from company activity data",
            "Standard emission factors applied",
        ],
    },
    3: {
        "label": "Estimated / Physical Activity",
        "description": "Estimated from physical activity data of borrower or investee",
        "typical_uncertainty_pct": Decimal("25.0"),
        "min_uncertainty_pct": Decimal("15.0"),
        "max_uncertainty_pct": Decimal("40.0"),
        "requirements": [
            "Physical activity data available (e.g., MWh, tonnes)",
            "Sector-average emission factors",
            "Partial company-specific data",
        ],
    },
    4: {
        "label": "Estimated / Economic Activity",
        "description": "Estimated from economic activity data of borrower or investee",
        "typical_uncertainty_pct": Decimal("45.0"),
        "min_uncertainty_pct": Decimal("30.0"),
        "max_uncertainty_pct": Decimal("60.0"),
        "requirements": [
            "Revenue or financial data available",
            "Economic emission factors (tCO2e/MEUR)",
            "Sector-level averages",
        ],
    },
    5: {
        "label": "Estimated / Asset Class",
        "description": "Estimated from asset class or sector average data",
        "typical_uncertainty_pct": Decimal("65.0"),
        "min_uncertainty_pct": Decimal("50.0"),
        "max_uncertainty_pct": Decimal("100.0"),
        "requirements": [
            "No company-specific data available",
            "Broad sector or asset class averages only",
            "National or global average emission factors",
        ],
    },
}


PEER_SIZE_BANDS: Dict[str, Dict[str, Any]] = {
    "MICRO": {
        "label": "Micro",
        "revenue_min_meur": Decimal("0"),
        "revenue_max_meur": Decimal("2"),
        "employees_min": 0,
        "employees_max": 10,
    },
    "SMALL": {
        "label": "Small",
        "revenue_min_meur": Decimal("2"),
        "revenue_max_meur": Decimal("50"),
        "employees_min": 10,
        "employees_max": 250,
    },
    "MEDIUM": {
        "label": "Medium",
        "revenue_min_meur": Decimal("50"),
        "revenue_max_meur": Decimal("500"),
        "employees_min": 250,
        "employees_max": 2500,
    },
    "LARGE": {
        "label": "Large",
        "revenue_min_meur": Decimal("500"),
        "revenue_max_meur": Decimal("5000"),
        "employees_min": 2500,
        "employees_max": 25000,
    },
    "ENTERPRISE": {
        "label": "Enterprise",
        "revenue_min_meur": Decimal("5000"),
        "revenue_max_meur": Decimal("50000"),
        "employees_min": 25000,
        "employees_max": 250000,
    },
    "MEGA": {
        "label": "Mega",
        "revenue_min_meur": Decimal("50000"),
        "revenue_max_meur": Decimal("999999"),
        "employees_min": 250000,
        "employees_max": 9999999,
    },
}


TRANSITION_RISK_WEIGHTS: Dict[str, Decimal] = {
    "CARBON_BUDGET": Decimal("0.25"),
    "STRANDING": Decimal("0.20"),
    "REGULATORY": Decimal("0.25"),
    "COMPETITIVE": Decimal("0.15"),
    "FINANCIAL": Decimal("0.15"),
}


AVAILABLE_PRESETS: Dict[str, str] = {
    "corporate_general": (
        "General corporate benchmark with multi-sector peer grouping, "
        "ITR calculation, and ESRS/CDP/TCFD disclosure alignment"
    ),
    "power_utilities": (
        "Power and utilities with tCO2e/MWh intensity benchmarking, "
        "IEA NZE pathway alignment, and TPI carbon performance"
    ),
    "heavy_industry": (
        "Heavy industry (steel, cement, aluminium, chemicals) with "
        "SBTi SDA convergence pathways and sector-specific thresholds"
    ),
    "real_estate": (
        "Real estate portfolio with kgCO2e/m2 benchmarking, CRREM "
        "pathway alignment, and GRESB score integration"
    ),
    "financial_services": (
        "Financial services with PCAF-aligned portfolio benchmarking, "
        "WACI, carbon footprint, financed emissions ITR"
    ),
    "transport_logistics": (
        "Transport and logistics with gCO2e/tkm intensity benchmarking, "
        "GLEC Framework, and SBTi transport pathway"
    ),
    "oil_gas": (
        "Oil and gas with upstream/downstream scope normalisation, "
        "IEA NZE alignment, and stranding risk assessment"
    ),
    "food_agriculture": (
        "Food and agriculture with SBTi FLAG pathway, tCO2e/tonne "
        "product benchmarking, and land use change scope"
    ),
}


# =============================================================================
# Sub-Config Models (15+ Pydantic v2 models)
# =============================================================================


class PeerGroupConfig(BaseModel):
    """Configuration for peer group construction and matching."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sector_classification: SectorClassification = Field(
        SectorClassification.GICS_4DIG,
        description="Sector classification system for peer matching",
    )
    size_band: PeerSizeBand = Field(
        PeerSizeBand.LARGE,
        description="Revenue-based size band for peer matching",
    )
    require_same_region: bool = Field(
        False,
        description="Require peers to be in the same geographic region",
    )
    region_scope: str = Field(
        "GLOBAL",
        description="Geographic scope for peer matching (GLOBAL, EU, NORTH_AMERICA, APAC, CUSTOM)",
    )
    min_peers: int = Field(
        5, ge=3, le=100,
        description="Minimum number of peers required in the group",
    )
    max_peers: int = Field(
        50, ge=5, le=500,
        description="Maximum number of peers to include",
    )
    revenue_tolerance_pct: Decimal = Field(
        Decimal("50.0"),
        description="Revenue range tolerance percentage for peer matching",
    )
    exclude_entities: List[str] = Field(
        default_factory=list,
        description="Entity IDs to exclude from peer group",
    )
    custom_peer_ids: List[str] = Field(
        default_factory=list,
        description="Manually specified peer entity IDs (bypasses automatic matching)",
    )
    require_disclosure_coverage: bool = Field(
        True,
        description="Require peers to have sufficient GHG disclosure coverage",
    )
    min_disclosure_pct: Decimal = Field(
        Decimal("60.0"), ge=Decimal("0"), le=Decimal("100"),
        description="Minimum disclosure coverage percentage required for peers",
    )

    @field_validator("region_scope")
    @classmethod
    def validate_region_scope(cls, v: str) -> str:
        """Validate region scope value."""
        allowed = {"GLOBAL", "EU", "NORTH_AMERICA", "APAC", "CUSTOM"}
        if v.upper() not in allowed:
            raise ValueError(f"region_scope must be one of {allowed}, got '{v}'")
        return v.upper()


class NormalisationConfig(BaseModel):
    """Configuration for scope normalisation and data alignment."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scope_alignment: ScopeAlignment = Field(
        ScopeAlignment.S1_S2M,
        description="Scope alignment rule for benchmark comparisons",
    )
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Organisational boundary consolidation approach",
    )
    target_gwp: GWPVersion = Field(
        GWPVersion.AR6,
        description="Target GWP version for harmonisation",
    )
    currency_normalisation: bool = Field(
        True,
        description="Normalise financial data to common currency using PPP",
    )
    target_currency: str = Field(
        "EUR",
        description="Target currency for financial normalisation (ISO 4217)",
    )
    period_alignment: str = Field(
        "CALENDAR_YEAR",
        description="Reporting period alignment (CALENDAR_YEAR, FISCAL_YEAR)",
    )
    data_gap_strategy: str = Field(
        "INTERPOLATE",
        description="Strategy for filling data gaps (INTERPOLATE, PREVIOUS_YEAR, EXCLUDE, ZERO_FILL)",
    )
    biogenic_treatment: str = Field(
        "EXCLUDE",
        description="Treatment of biogenic CO2 (EXCLUDE, INCLUDE, SEPARATE)",
    )
    climate_zone_adjustment: bool = Field(
        False,
        description="Adjust building benchmarks by climate zone (HDD/CDD)",
    )
    normalisation_steps: List[NormalisationStep] = Field(
        default_factory=lambda: [
            NormalisationStep.SCOPE_ALIGN,
            NormalisationStep.CONSOLIDATION,
            NormalisationStep.GWP,
            NormalisationStep.CURRENCY,
            NormalisationStep.PERIOD,
            NormalisationStep.DATA_GAP,
        ],
        description="Ordered list of normalisation steps to apply",
    )

    @field_validator("period_alignment")
    @classmethod
    def validate_period_alignment(cls, v: str) -> str:
        """Validate period alignment value."""
        allowed = {"CALENDAR_YEAR", "FISCAL_YEAR"}
        if v.upper() not in allowed:
            raise ValueError(f"period_alignment must be one of {allowed}, got '{v}'")
        return v.upper()

    @field_validator("data_gap_strategy")
    @classmethod
    def validate_data_gap_strategy(cls, v: str) -> str:
        """Validate data gap strategy value."""
        allowed = {"INTERPOLATE", "PREVIOUS_YEAR", "EXCLUDE", "ZERO_FILL"}
        if v.upper() not in allowed:
            raise ValueError(f"data_gap_strategy must be one of {allowed}, got '{v}'")
        return v.upper()

    @field_validator("biogenic_treatment")
    @classmethod
    def validate_biogenic_treatment(cls, v: str) -> str:
        """Validate biogenic treatment value."""
        allowed = {"EXCLUDE", "INCLUDE", "SEPARATE"}
        if v.upper() not in allowed:
            raise ValueError(f"biogenic_treatment must be one of {allowed}, got '{v}'")
        return v.upper()


class ExternalDataConfig(BaseModel):
    """Configuration for external benchmark data source integration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sources: List[DataSourceType] = Field(
        default_factory=lambda: [DataSourceType.CDP, DataSourceType.TPI],
        description="External data sources to integrate",
    )
    cache_ttl_hours: int = Field(
        168, ge=1, le=8760,
        description="Cache TTL in hours for external data (default 7 days)",
    )
    auto_refresh: bool = Field(
        True,
        description="Automatically refresh external data when stale",
    )
    cdp_year: int = Field(
        2025, ge=2020, le=2030,
        description="CDP questionnaire year for data retrieval",
    )
    tpi_version: str = Field(
        "5.0",
        description="TPI Carbon Performance methodology version",
    )
    gresb_year: int = Field(
        2025, ge=2020, le=2030,
        description="GRESB assessment year for data retrieval",
    )
    crrem_version: str = Field(
        "2.1",
        description="CRREM decarbonisation pathway version",
    )
    iss_esg_enabled: bool = Field(
        False,
        description="Enable ISS ESG Climate Solutions data integration",
    )
    custom_data_path: Optional[str] = Field(
        None,
        description="Path to custom benchmark dataset (CSV or JSON)",
    )


class PathwayConfig(BaseModel):
    """Configuration for science-based pathway alignment scoring."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pathways: List[PathwayType] = Field(
        default_factory=lambda: [PathwayType.IEA_NZE, PathwayType.SBTI_SDA],
        description="Pathway types to assess alignment against",
    )
    primary_scenario: PathwayScenario = Field(
        PathwayScenario.ONE_POINT_FIVE_C,
        description="Primary temperature scenario for alignment scoring",
    )
    assessment_years: List[int] = Field(
        default_factory=lambda: [2025, 2030, 2035, 2040, 2050],
        description="Target years for pathway alignment assessment",
    )
    sbti_sector: str = Field(
        "buildings",
        description="SBTi SDA sector key from SBTI_SECTOR_PATHWAYS",
    )
    convergence_year: int = Field(
        2050, ge=2030, le=2070,
        description="Year by which emissions should converge to the pathway",
    )
    include_overshoot_analysis: bool = Field(
        True,
        description="Include cumulative overshoot analysis for budget-based pathways",
    )
    alignment_threshold_pct: Decimal = Field(
        Decimal("10.0"),
        description="Percentage threshold for pathway alignment classification",
    )

    @field_validator("sbti_sector")
    @classmethod
    def validate_sbti_sector(cls, v: str) -> str:
        """Validate SBTi sector key exists."""
        if v not in SBTI_SECTOR_PATHWAYS and v != "custom":
            logger.warning(
                "SBTi sector '%s' not in SBTI_SECTOR_PATHWAYS. Available: %s",
                v, sorted(SBTI_SECTOR_PATHWAYS.keys()),
            )
        return v


class ITRConfig(BaseModel):
    """Configuration for implied temperature rise calculation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    methods: List[ITRMethod] = Field(
        default_factory=lambda: [ITRMethod.BUDGET_BASED, ITRMethod.SECTOR_RELATIVE],
        description="ITR calculation methods to apply",
    )
    primary_method: ITRMethod = Field(
        ITRMethod.BUDGET_BASED,
        description="Primary ITR method for headline reporting",
    )
    budget_start_year: int = Field(
        2020, ge=2015, le=2030,
        description="Carbon budget start year for budget-based ITR",
    )
    target_horizon_year: int = Field(
        2050, ge=2030, le=2100,
        description="Target horizon year for ITR calculation",
    )
    confidence_level: Decimal = Field(
        Decimal("0.50"),
        description="Probability level for carbon budget (0.50 = 50th percentile)",
    )
    include_scope_3: bool = Field(
        False,
        description="Include Scope 3 emissions in ITR calculation",
    )
    tcre_value: Decimal = Field(
        Decimal("0.00045"),
        description="Transient Climate Response to Cumulative Emissions (C per GtCO2)",
    )
    pre_industrial_baseline: Decimal = Field(
        Decimal("1.07"),
        description="Current warming above pre-industrial levels (degrees C)",
    )


class TrajectoryConfig(BaseModel):
    """Configuration for trajectory benchmarking and forward-looking assessment."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    historical_years: int = Field(
        5, ge=3, le=15,
        description="Number of historical years for trend extraction",
    )
    projection_years: int = Field(
        10, ge=5, le=30,
        description="Number of years to project forward",
    )
    regression_model: str = Field(
        "OLS",
        description="Regression model for trajectory projection (OLS, WEIGHTED_LS, ROBUST)",
    )
    min_data_points: int = Field(
        3, ge=2, le=10,
        description="Minimum data points required for trajectory analysis",
    )
    include_confidence_bands: bool = Field(
        True,
        description="Include confidence bands on trajectory projections",
    )
    confidence_levels: List[Decimal] = Field(
        default_factory=lambda: [Decimal("0.90"), Decimal("0.95")],
        description="Confidence levels for trajectory bands",
    )
    convergence_test: bool = Field(
        True,
        description="Test whether projected trajectory converges with pathway",
    )

    @field_validator("regression_model")
    @classmethod
    def validate_regression_model(cls, v: str) -> str:
        """Validate regression model value."""
        allowed = {"OLS", "WEIGHTED_LS", "ROBUST"}
        if v.upper() not in allowed:
            raise ValueError(f"regression_model must be one of {allowed}, got '{v}'")
        return v.upper()


class PortfolioConfig(BaseModel):
    """Configuration for portfolio-level carbon benchmarking (PCAF aligned)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    asset_classes: List[PortfolioAssetClass] = Field(
        default_factory=lambda: [
            PortfolioAssetClass.LISTED_EQUITY,
            PortfolioAssetClass.CORPORATE_BONDS,
        ],
        description="Asset classes to include in portfolio benchmarking",
    )
    waci_enabled: bool = Field(
        True,
        description="Calculate Weighted Average Carbon Intensity (WACI)",
    )
    carbon_footprint_enabled: bool = Field(
        True,
        description="Calculate financed emissions carbon footprint",
    )
    attribution_method: str = Field(
        "EVIC",
        description="Attribution method for financed emissions (EVIC, REVENUE, BALANCE_SHEET)",
    )
    include_sovereign: bool = Field(
        False,
        description="Include sovereign debt in portfolio benchmarking",
    )
    benchmark_index: str = Field(
        "MSCI_WORLD",
        description="Reference benchmark index for portfolio comparison",
    )
    pcaf_alignment: bool = Field(
        True,
        description="Follow PCAF Global Standard v3 methodology",
    )
    decimal_places: int = Field(
        4, ge=0, le=10,
        description="Decimal places for portfolio metric calculations",
    )

    @field_validator("attribution_method")
    @classmethod
    def validate_attribution_method(cls, v: str) -> str:
        """Validate attribution method value."""
        allowed = {"EVIC", "REVENUE", "BALANCE_SHEET"}
        if v.upper() not in allowed:
            raise ValueError(f"attribution_method must be one of {allowed}, got '{v}'")
        return v.upper()


class DataQualityConfig(BaseModel):
    """Configuration for PCAF data quality scoring."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scoring_method: str = Field(
        "PCAF_V3",
        description="Data quality scoring methodology (PCAF_V3, CUSTOM)",
    )
    dimensions: List[QualityDimension] = Field(
        default_factory=lambda: [
            QualityDimension.TEMPORAL,
            QualityDimension.GEOGRAPHIC,
            QualityDimension.TECHNOLOGICAL,
            QualityDimension.COMPLETENESS,
            QualityDimension.RELIABILITY,
        ],
        description="Quality dimensions to assess",
    )
    dimension_weights: Dict[str, Decimal] = Field(
        default_factory=lambda: {
            "TEMPORAL": Decimal("0.20"),
            "GEOGRAPHIC": Decimal("0.20"),
            "TECHNOLOGICAL": Decimal("0.20"),
            "COMPLETENESS": Decimal("0.20"),
            "RELIABILITY": Decimal("0.20"),
        },
        description="Weights for each quality dimension (must sum to 1.0)",
    )
    target_score: PCAFScore = Field(
        PCAFScore.SCORE_2,
        description="Target PCAF quality score to achieve",
    )
    improvement_roadmap: bool = Field(
        True,
        description="Generate data quality improvement roadmap",
    )
    staleness_threshold_days: int = Field(
        365, ge=30, le=730,
        description="Number of days after which data is considered stale",
    )

    @model_validator(mode="after")
    def validate_dimension_weights_sum(self) -> DataQualityConfig:
        """Ensure dimension weights sum to approximately 1.0."""
        total = sum(self.dimension_weights.values())
        if abs(total - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(
                f"Dimension weights must sum to 1.0, got {total}"
            )
        return self


class TransitionRiskConfig(BaseModel):
    """Configuration for transition risk scoring."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    categories: List[TransitionRiskCategory] = Field(
        default_factory=lambda: [
            TransitionRiskCategory.CARBON_BUDGET,
            TransitionRiskCategory.STRANDING,
            TransitionRiskCategory.REGULATORY,
            TransitionRiskCategory.COMPETITIVE,
            TransitionRiskCategory.FINANCIAL,
        ],
        description="Transition risk categories to assess",
    )
    category_weights: Dict[str, Decimal] = Field(
        default_factory=lambda: {
            "CARBON_BUDGET": Decimal("0.25"),
            "STRANDING": Decimal("0.20"),
            "REGULATORY": Decimal("0.25"),
            "COMPETITIVE": Decimal("0.15"),
            "FINANCIAL": Decimal("0.15"),
        },
        description="Weights for each risk category (must sum to 1.0)",
    )
    carbon_price_scenario: str = Field(
        "IEA_NZE",
        description="Carbon price scenario for financial impact (IEA_NZE, EU_ETS, NGFS)",
    )
    carbon_price_2030_eur: Decimal = Field(
        Decimal("130.00"),
        description="Assumed carbon price in EUR/tCO2e for 2030",
    )
    carbon_price_2050_eur: Decimal = Field(
        Decimal("250.00"),
        description="Assumed carbon price in EUR/tCO2e for 2050",
    )
    stranding_threshold_year: int = Field(
        2035, ge=2025, le=2050,
        description="Year by which stranding risk is assessed",
    )
    regulatory_jurisdictions: List[str] = Field(
        default_factory=lambda: ["EU", "UK", "US"],
        description="Jurisdictions for regulatory risk assessment",
    )
    include_physical_risk: bool = Field(
        False,
        description="Include physical climate risk in overall assessment (future extension)",
    )

    @model_validator(mode="after")
    def validate_category_weights_sum(self) -> TransitionRiskConfig:
        """Ensure category weights sum to approximately 1.0."""
        total = sum(self.category_weights.values())
        if abs(total - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(
                f"Category weights must sum to 1.0, got {total}"
            )
        return self

    @field_validator("carbon_price_scenario")
    @classmethod
    def validate_carbon_price_scenario(cls, v: str) -> str:
        """Validate carbon price scenario value."""
        allowed = {"IEA_NZE", "EU_ETS", "NGFS"}
        if v.upper() not in allowed:
            raise ValueError(f"carbon_price_scenario must be one of {allowed}, got '{v}'")
        return v.upper()


class ReportingConfig(BaseModel):
    """Configuration for benchmark report generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.HTML, ReportFormat.JSON],
        description="Output format(s) for generated reports",
    )
    metrics: List[BenchmarkMetric] = Field(
        default_factory=lambda: [
            BenchmarkMetric.INTENSITY,
            BenchmarkMetric.ITR,
            BenchmarkMetric.TRANSITION_RISK,
        ],
        description="Benchmark metrics to include in reports",
    )
    sections: List[str] = Field(
        default_factory=lambda: [
            "executive_summary",
            "peer_comparison",
            "pathway_alignment",
            "implied_temperature_rise",
            "trajectory_analysis",
            "data_quality",
            "transition_risk",
            "methodology",
        ],
        description="Report sections to include",
    )
    branding: Dict[str, str] = Field(
        default_factory=lambda: {
            "logo_url": "",
            "primary_colour": "#1B5E20",
            "company_name": "",
        },
        description="Report branding configuration",
    )
    language: str = Field("en", description="Report language (ISO 639-1)")
    include_charts: bool = Field(True, description="Include charts and visualisations")
    include_data_tables: bool = Field(True, description="Include detailed data tables")
    include_appendices: bool = Field(True, description="Include technical appendices")
    decimal_places_display: int = Field(
        2, ge=0, le=6,
        description="Decimal places for display in reports",
    )


class DisclosureConfig(BaseModel):
    """Configuration for multi-framework benchmark disclosure mapping."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    frameworks: List[DisclosureFramework] = Field(
        default_factory=lambda: [
            DisclosureFramework.ESRS,
            DisclosureFramework.CDP,
            DisclosureFramework.TCFD,
        ],
        description="Target disclosure frameworks for benchmark data",
    )
    mandatory_only: bool = Field(
        False,
        description="Only map mandatory (required) disclosure fields",
    )
    xbrl_taxonomy: str = Field(
        "ESRS_2024",
        description="XBRL taxonomy version for machine-readable tagging",
    )
    include_methodology_notes: bool = Field(
        True,
        description="Include methodology descriptions in disclosure output",
    )
    include_data_quality_notes: bool = Field(
        True,
        description="Include data quality information in disclosure output",
    )
    sfdr_pai_indicators: List[int] = Field(
        default_factory=lambda: [1, 2, 3],
        description="SFDR Principal Adverse Impact indicators to include",
    )
    field_mapping_overrides: Dict[str, str] = Field(
        default_factory=dict,
        description="Manual overrides for framework field mappings",
    )


class AlertConfig(BaseModel):
    """Configuration for benchmark monitoring and alerting."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    alert_types: List[AlertType] = Field(
        default_factory=lambda: [
            AlertType.THRESHOLD,
            AlertType.PATHWAY_DEVIATION,
            AlertType.DATA_STALENESS,
        ],
        description="Types of alerts to enable",
    )
    channels: List[str] = Field(
        default_factory=lambda: ["EMAIL"],
        description="Notification delivery channels (EMAIL, SLACK, TEAMS, WEBHOOK)",
    )
    pathway_deviation_threshold_pct: Decimal = Field(
        Decimal("15.0"),
        description="Percentage deviation from pathway before alerting",
    )
    rank_change_threshold: int = Field(
        5, ge=1, le=50,
        description="Minimum rank position change to trigger alert",
    )
    data_staleness_days: int = Field(
        180, ge=30, le=730,
        description="Days after which data staleness alert is triggered",
    )
    quality_drop_threshold: int = Field(
        1, ge=1, le=3,
        description="PCAF score drop amount to trigger quality alert",
    )


class PerformanceConfig(BaseModel):
    """Configuration for computational performance tuning."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_calculation_time_seconds: int = Field(
        300, ge=30, le=3600,
        description="Maximum allowed calculation time in seconds",
    )
    cache_benchmark_results: bool = Field(
        True, description="Cache benchmark results for repeated access",
    )
    parallel_peer_processing: bool = Field(
        True, description="Process peer group entities in parallel",
    )
    batch_size: int = Field(
        500, ge=50, le=5000,
        description="Batch size for bulk entity processing",
    )
    cache_ttl_seconds: int = Field(
        3600, ge=60, le=86400,
        description="Cache TTL in seconds",
    )
    lazy_load_external_data: bool = Field(
        True, description="Lazy-load external benchmark data only when needed",
    )


class SecurityConfig(BaseModel):
    """Configuration for access control and data protection."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rbac_enabled: bool = Field(True, description="Enable role-based access control")
    audit_trail_enabled: bool = Field(True, description="Enable audit trail for all operations")
    encryption_at_rest: bool = Field(True, description="Encrypt benchmark data at rest (AES-256)")
    roles: List[str] = Field(
        default_factory=lambda: [
            "benchmark_analyst", "benchmark_manager", "reviewer",
            "approver", "data_admin", "viewer", "admin",
        ],
        description="Available RBAC roles for benchmark management",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class BenchmarkPackConfig(BaseModel):
    """
    Top-level configuration for PACK-047 GHG Emissions Benchmark.

    Combines all sub-configurations required for peer group construction,
    scope normalisation, external dataset integration, pathway alignment,
    implied temperature rise, trajectory benchmarking, portfolio benchmarking,
    data quality scoring, transition risk scoring, and reporting.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    company_name: str = Field("", description="Reporting company legal name")
    sector_classification: SectorClassification = Field(
        SectorClassification.GICS_4DIG, description="Sector classification system",
    )
    sector_code: str = Field(
        "", description="Sector code within the classification system (e.g., '2010' for GICS)",
    )
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Organisational boundary approach",
    )
    country: str = Field("DE", description="Primary country (ISO 3166-1 alpha-2)")
    reporting_year: int = Field(2026, ge=2020, le=2035, description="Current reporting year")
    base_year: int = Field(2020, ge=2015, le=2030, description="Base year for benchmark trends")
    revenue_meur: Optional[Decimal] = Field(None, ge=Decimal("0"), description="Annual revenue in MEUR")
    employees_fte: Optional[int] = Field(None, ge=0, description="Full-time equivalent employees")

    peer_group: PeerGroupConfig = Field(default_factory=PeerGroupConfig)
    normalisation: NormalisationConfig = Field(default_factory=NormalisationConfig)
    external_data: ExternalDataConfig = Field(default_factory=ExternalDataConfig)
    pathway: PathwayConfig = Field(default_factory=PathwayConfig)
    itr: ITRConfig = Field(default_factory=ITRConfig)
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    transition_risk: TransitionRiskConfig = Field(default_factory=TransitionRiskConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    disclosure: DisclosureConfig = Field(default_factory=DisclosureConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @model_validator(mode="after")
    def validate_base_year_consistency(self) -> BenchmarkPackConfig:
        """Ensure base year is before reporting year."""
        if self.base_year > self.reporting_year:
            raise ValueError(
                f"base_year ({self.base_year}) cannot be after "
                f"reporting_year ({self.reporting_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_itr_scope_3_integration(self) -> BenchmarkPackConfig:
        """Warn if ITR includes Scope 3 but scope alignment does not."""
        if self.itr.include_scope_3:
            if self.normalisation.scope_alignment != ScopeAlignment.S1_S2_S3:
                logger.warning(
                    "ITR includes Scope 3 but normalisation scope_alignment is %s. "
                    "Consider aligning to S1_S2_S3 for consistency.",
                    self.normalisation.scope_alignment.value,
                )
        return self

    @model_validator(mode="after")
    def validate_pathway_sector_exists(self) -> BenchmarkPackConfig:
        """Warn if pathway SBTi sector is not in reference data."""
        if self.pathway.sbti_sector not in SBTI_SECTOR_PATHWAYS:
            if PathwayType.SBTI_SDA in self.pathway.pathways:
                logger.warning(
                    "SBTi SDA pathway selected but sector '%s' not in "
                    "SBTI_SECTOR_PATHWAYS. Available: %s",
                    self.pathway.sbti_sector,
                    sorted(SBTI_SECTOR_PATHWAYS.keys()),
                )
        return self

    @model_validator(mode="after")
    def validate_portfolio_pcaf_consistency(self) -> BenchmarkPackConfig:
        """Ensure PCAF alignment is on when portfolio benchmarking is enabled."""
        if self.portfolio.waci_enabled or self.portfolio.carbon_footprint_enabled:
            if not self.portfolio.pcaf_alignment:
                logger.warning(
                    "Portfolio metrics enabled but pcaf_alignment is False. "
                    "PCAF alignment is recommended for financed emissions."
                )
        return self

    @model_validator(mode="after")
    def validate_gresb_real_estate(self) -> BenchmarkPackConfig:
        """Warn if GRESB is enabled for non-real-estate sectors."""
        if DataSourceType.GRESB in self.external_data.sources:
            if self.sector_code and not self.sector_code.startswith("60"):
                logger.warning(
                    "GRESB data source is enabled but sector code '%s' "
                    "does not appear to be real estate (GICS 60xx).",
                    self.sector_code,
                )
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """
    Top-level wrapper for PACK-047 configuration.

    Provides factory methods for loading from presets, YAML files,
    environment overrides, and runtime merges. Includes SHA-256
    config hashing for provenance tracking.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pack: BenchmarkPackConfig = Field(default_factory=BenchmarkPackConfig)
    preset_name: Optional[str] = Field(None, description="Name of the loaded preset")
    config_version: str = Field("1.0.0", description="Configuration schema version")
    pack_id: str = Field("PACK-047-benchmark", description="Unique pack identifier")

    @classmethod
    def from_preset(cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> PackConfig:
        """
        Load configuration from a named sector preset.

        Args:
            preset_name: Key from AVAILABLE_PRESETS (e.g., 'corporate_general').
            overrides: Optional dict of overrides applied after preset load.

        Returns:
            Fully initialised PackConfig.

        Raises:
            ValueError: If preset_name is not recognised.
            FileNotFoundError: If preset YAML file is missing.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {sorted(AVAILABLE_PRESETS.keys())}"
            )
        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset file not found: {preset_path}")
        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            preset_data = cls._deep_merge(preset_data, env_overrides)
        if overrides:
            preset_data = cls._deep_merge(preset_data, overrides)
        pack_config = BenchmarkPackConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> PackConfig:
        """
        Load configuration from an arbitrary YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Fully initialised PackConfig.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        pack_config = BenchmarkPackConfig(**config_data)
        return cls(pack=pack_config)

    @classmethod
    def merge(cls, base: PackConfig, overrides: Dict[str, Any]) -> PackConfig:
        """
        Create a new PackConfig by merging overrides into a base config.

        Args:
            base: Existing PackConfig to use as the base.
            overrides: Dict of overrides (supports nested keys).

        Returns:
            New PackConfig with merged values.
        """
        base_dict = base.pack.model_dump()
        merged = cls._deep_merge(base_dict, overrides)
        pack_config = BenchmarkPackConfig(**merged)
        return cls(pack=pack_config, preset_name=base.preset_name, config_version=base.config_version)

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.

        Environment variables prefixed with BENCHMARK_PACK_ are parsed.
        Double underscores denote nested keys.
        Example: BENCHMARK_PACK_PEER_GROUP__MIN_PEERS=10
        """
        overrides: Dict[str, Any] = {}
        prefix = "BENCHMARK_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                current = overrides
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
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
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override dict into base dict."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_config_hash(self) -> str:
        """
        Compute SHA-256 hash of the full configuration.

        Returns:
            Hex-encoded SHA-256 hash string for provenance tracking.
        """
        config_json = self.model_dump_json(indent=None)
        return _compute_hash(config_json)

    def validate_completeness(self) -> List[str]:
        """
        Run domain-specific validation checks on the configuration.

        Returns:
            List of warning messages (empty list means no issues).
        """
        return validate_config(self.pack)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the full configuration to a plain dictionary.

        Returns:
            Dict representation of the entire PackConfig.
        """
        return self.model_dump()


# =============================================================================
# Utility Functions
# =============================================================================


def load_preset(preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> PackConfig:
    """
    Convenience function to load a preset configuration.

    Args:
        preset_name: Key from AVAILABLE_PRESETS.
        overrides: Optional dict of overrides.

    Returns:
        Initialised PackConfig from the named preset.
    """
    return PackConfig.from_preset(preset_name, overrides)


def validate_config(config: BenchmarkPackConfig) -> List[str]:
    """
    Validate configuration for domain-specific consistency.

    Args:
        config: The benchmark pack configuration to validate.

    Returns:
        List of warning strings. Empty list indicates no issues found.
    """
    warnings: List[str] = []

    # Company name check
    if not config.company_name:
        warnings.append("No company_name configured.")

    # Sector code check
    if not config.sector_code:
        warnings.append("No sector_code configured. Peer group matching may be imprecise.")

    # Peer group minimum size
    if config.peer_group.min_peers < 5:
        warnings.append(
            f"Peer group min_peers is {config.peer_group.min_peers}. "
            "A minimum of 5 peers is recommended for statistical significance."
        )

    # Normalisation scope alignment consistency
    if config.normalisation.scope_alignment == ScopeAlignment.S1_S2_S3:
        if not config.itr.include_scope_3:
            warnings.append(
                "Normalisation includes Scope 3 but ITR excludes it. "
                "Consider enabling itr.include_scope_3 for consistency."
            )

    # SBTi pathway sector check
    if PathwayType.SBTI_SDA in config.pathway.pathways:
        if config.pathway.sbti_sector not in SBTI_SECTOR_PATHWAYS:
            warnings.append(
                f"SBTi SDA pathway selected but sector '{config.pathway.sbti_sector}' "
                f"is not in SBTI_SECTOR_PATHWAYS. Available: {sorted(SBTI_SECTOR_PATHWAYS.keys())}."
            )

    # GRESB non-real-estate check
    if DataSourceType.GRESB in config.external_data.sources:
        if config.sector_code and not config.sector_code.startswith("60"):
            warnings.append(
                "GRESB data source is enabled for a non-real-estate sector. "
                "GRESB is primarily relevant for sector code 60xx (Real Estate)."
            )

    # CRREM non-real-estate check
    if DataSourceType.CRREM in config.external_data.sources:
        if config.sector_code and not config.sector_code.startswith("60"):
            warnings.append(
                "CRREM data source is enabled for a non-real-estate sector."
            )

    # Portfolio asset class consistency
    if config.portfolio.include_sovereign:
        if PortfolioAssetClass.SOVEREIGN_DEBT not in config.portfolio.asset_classes:
            warnings.append(
                "include_sovereign is True but SOVEREIGN_DEBT not in asset_classes."
            )

    # Data quality dimension weights
    total_weights = sum(config.data_quality.dimension_weights.values())
    if abs(total_weights - Decimal("1.0")) > Decimal("0.01"):
        warnings.append(
            f"Data quality dimension weights sum to {total_weights}, expected 1.0."
        )

    # Transition risk category weights
    total_risk_weights = sum(config.transition_risk.category_weights.values())
    if abs(total_risk_weights - Decimal("1.0")) > Decimal("0.01"):
        warnings.append(
            f"Transition risk category weights sum to {total_risk_weights}, expected 1.0."
        )

    # Base year vs ITR budget start year
    if config.itr.budget_start_year != config.base_year:
        warnings.append(
            f"Pack base_year ({config.base_year}) differs from ITR budget_start_year "
            f"({config.itr.budget_start_year}). Ensure alignment for consistent tracking."
        )

    # Security configuration
    if config.security.audit_trail_enabled and not config.security.rbac_enabled:
        warnings.append(
            "Audit trail is enabled but RBAC is disabled. "
            "Consider enabling RBAC for proper identity tracking."
        )

    return warnings


def get_default_config(
    sector: SectorClassification = SectorClassification.GICS_4DIG,
) -> BenchmarkPackConfig:
    """
    Create a default configuration for the given sector classification.

    Args:
        sector: Sector classification system.

    Returns:
        Default BenchmarkPackConfig for the sector.
    """
    return BenchmarkPackConfig(sector_classification=sector)


def list_available_presets() -> Dict[str, str]:
    """
    Return a copy of all available preset names and descriptions.

    Returns:
        Dict mapping preset name to human-readable description.
    """
    return AVAILABLE_PRESETS.copy()


def get_pcaf_quality_info(score: int) -> Optional[Dict[str, Any]]:
    """
    Return PCAF quality threshold info for a given score.

    Args:
        score: PCAF quality score (1-5).

    Returns:
        Dict of quality threshold metadata, or None if score invalid.
    """
    return PCAF_QUALITY_THRESHOLDS.get(score)


def get_sbti_pathway(
    sector_key: str,
) -> Optional[Dict[str, Any]]:
    """
    Return SBTi SDA pathway data for a given sector.

    Args:
        sector_key: Key from SBTI_SECTOR_PATHWAYS.

    Returns:
        Dict of pathway data, or None if sector not found.
    """
    return SBTI_SECTOR_PATHWAYS.get(sector_key)


def get_carbon_budget(
    scenario: str,
) -> Optional[Dict[str, Any]]:
    """
    Return IPCC carbon budget for a given temperature scenario.

    Args:
        scenario: Temperature scenario key (e.g., '1.5C', '1.7C', '2.0C').

    Returns:
        Dict of carbon budget data, or None if scenario not found.
    """
    return IPCC_CARBON_BUDGETS.get(scenario)


def get_peer_size_band(
    revenue_meur: Decimal,
) -> str:
    """
    Determine peer size band from revenue in MEUR.

    Args:
        revenue_meur: Annual revenue in millions of euros.

    Returns:
        Size band string (MICRO, SMALL, MEDIUM, LARGE, ENTERPRISE, MEGA).
    """
    for band_key, band_data in PEER_SIZE_BANDS.items():
        if band_data["revenue_min_meur"] <= revenue_meur < band_data["revenue_max_meur"]:
            return band_key
    return "MEGA"
