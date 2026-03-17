"""
PACK-007 EUDR Professional Pack - Configuration Manager

This module implements the EUDRProfessionalConfig and PackConfig classes that load,
merge, and validate all configuration for the EUDR Professional Pack. It extends
the PACK-006 Starter configuration with advanced professional-tier features for
enterprise-grade EUDR compliance management.

Professional Pack Features:
    - All 40 EUDR agents (vs 18 in Starter)
    - Advanced geolocation with Sentinel-1/2, MODIS integration
    - Scenario-based risk modeling (Monte Carlo, confidence intervals)
    - Real-time satellite monitoring (6-hour check intervals)
    - Multi-operator portfolio management (up to 100 operators)
    - Advanced audit trail with SHA-256 provenance
    - Protected area & indigenous land checks (WDPA, KBA, Ramsar, UNESCO)
    - Supplier benchmarking & degradation scoring (6 dimensions)
    - Regulatory change tracking (EUR-Lex, 24-hour intervals)
    - Grievance mechanism with whistleblower protection
    - Cross-regulation linkage (CSRD E4, CSDDD, Nature Restoration)

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Size preset (multi_commodity / high_risk / trading_company / multi_subsidiary)
    3. Sector preset (palm_oil_professional / timber_professional / etc.)
    4. Environment overrides (EUDR_PACK_* environment variables)
    5. Explicit runtime overrides

Regulatory Context:
    - EUDR: Regulation (EU) 2023/1115
    - Articles: 3, 4, 8, 9, 10, 11, 12, 13, 29, 33
    - Cutoff Date: 31 December 2020
    - Commodities: 7 (cattle, cocoa, coffee, oil palm, rubber, soya, wood)
    - Annex I CN Codes: 400+ product classifications

Example:
    >>> config = PackConfig.load(
    ...     size_preset="multi_commodity",
    ...     sector_preset="palm_oil_professional",
    ... )
    >>> print(config.pack.metadata.display_name)
    'EUDR Professional Pack'
    >>> print(config.active_agents)
    ['AGENT-EUDR-001', 'AGENT-EUDR-002', ..., 'AGENT-EUDR-040']
"""

import hashlib
import json
import logging
import os
from datetime import date, datetime
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
# Enums - EUDR-specific enumeration types (from PACK-006 + Professional)
# =============================================================================


class EUDRCommodity(str, Enum):
    """EUDR Article 1 regulated commodities."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class OperatorType(str, Enum):
    """EUDR operator classification per Article 2."""

    OPERATOR = "OPERATOR"
    TRADER = "TRADER"


class CompanySize(str, Enum):
    """Company size classification for EUDR obligations."""

    SME = "SME"
    MID_MARKET = "MID_MARKET"
    LARGE = "LARGE"
    ENTERPRISE = "ENTERPRISE"  # Professional tier


class DDSType(str, Enum):
    """Due Diligence Statement type per Articles 4 and 13."""

    STANDARD = "STANDARD"
    SIMPLIFIED = "SIMPLIFIED"


class DDSStatus(str, Enum):
    """Due Diligence Statement lifecycle status."""

    DRAFT = "DRAFT"
    REVIEW = "REVIEW"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    AMENDED = "AMENDED"


class RiskLevel(str, Enum):
    """Risk level classification for suppliers and commodities."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CountryBenchmark(str, Enum):
    """Country benchmarking classification per Article 29."""

    LOW_RISK = "LOW_RISK"
    STANDARD_RISK = "STANDARD_RISK"
    HIGH_RISK = "HIGH_RISK"


class CertificationScheme(str, Enum):
    """Recognized voluntary certification schemes."""

    FSC = "FSC"
    PEFC = "PEFC"
    RSPO = "RSPO"
    ISCC = "ISCC"
    RAINFOREST_ALLIANCE = "RAINFOREST_ALLIANCE"
    UTZ = "UTZ"
    FAIRTRADE = "FAIRTRADE"
    RTRS = "RTRS"
    PROTERRA = "PROTERRA"
    FOUR_C = "FOUR_C"
    ORGANIC = "ORGANIC"
    SFI = "SFI"
    MSPO = "MSPO"
    ISPO = "ISPO"


class ChainOfCustodyModel(str, Enum):
    """Chain of custody models per industry standard."""

    IDENTITY_PRESERVED = "IDENTITY_PRESERVED"
    SEGREGATED = "SEGREGATED"
    MASS_BALANCE = "MASS_BALANCE"
    CONTROLLED_SOURCES = "CONTROLLED_SOURCES"


class CoordinateFormat(str, Enum):
    """GPS coordinate format types."""

    DECIMAL_DEGREES = "DECIMAL_DEGREES"
    DMS = "DMS"
    UTM = "UTM"


class SupplierDDStatus(str, Enum):
    """Supplier due diligence completion status."""

    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    VERIFIED = "VERIFIED"
    EXPIRED = "EXPIRED"


class AreaUnit(str, Enum):
    """Area measurement units for plot boundaries."""

    HECTARES = "HECTARES"
    SQUARE_METERS = "SQUARE_METERS"
    ACRES = "ACRES"
    SQUARE_KILOMETERS = "SQUARE_KILOMETERS"


class AuthType(str, Enum):
    """Authentication type for EU Information System."""

    OAUTH2 = "OAUTH2"
    EIDAS = "EIDAS"
    CERTIFICATE = "CERTIFICATE"


# Professional-tier enums
class SatelliteProvider(str, Enum):
    """Satellite imagery providers for monitoring."""

    SENTINEL_1 = "SENTINEL_1"  # Radar
    SENTINEL_2 = "SENTINEL_2"  # Optical
    MODIS = "MODIS"  # Fire detection
    LANDSAT_8 = "LANDSAT_8"
    PLANET = "PLANET"  # High-resolution


class AlertSystem(str, Enum):
    """Deforestation alert systems."""

    GLAD = "GLAD"  # Global Land Analysis & Discovery
    RADD = "RADD"  # Radar for Detecting Deforestation
    DETER = "DETER"  # Brazil's alert system
    GFW = "GFW"  # Global Forest Watch


class RiskDistribution(str, Enum):
    """Probability distributions for scenario modeling."""

    NORMAL = "NORMAL"
    LOGNORMAL = "LOGNORMAL"
    UNIFORM = "UNIFORM"
    TRIANGULAR = "TRIANGULAR"
    BETA = "BETA"


class BenchmarkDimension(str, Enum):
    """Supplier benchmarking dimensions."""

    COMPLIANCE_SCORE = "COMPLIANCE_SCORE"
    DATA_QUALITY = "DATA_QUALITY"
    RISK_LEVEL = "RISK_LEVEL"
    CERTIFICATION_STATUS = "CERTIFICATION_STATUS"
    RESPONSE_TIME = "RESPONSE_TIME"
    SUSTAINABILITY_PERFORMANCE = "SUSTAINABILITY_PERFORMANCE"


class NotificationChannel(str, Enum):
    """Notification delivery channels."""

    EMAIL = "EMAIL"
    WEBHOOK = "WEBHOOK"
    SMS = "SMS"
    SLACK = "SLACK"
    TEAMS = "TEAMS"
    IN_APP = "IN_APP"


# =============================================================================
# Reference Data Constants - EUDR regulatory reference data
# =============================================================================

# Cutoff date per Article 1(1) - no deforestation after this date
CUTOFF_DATE: date = date(2020, 12, 31)

# Polygon area threshold (hectares) - above this, polygon required; below, point OK
POLYGON_AREA_THRESHOLD_HA: float = 4.0

# EUDR commodities with display names and Annex I article references
EUDR_COMMODITIES: Dict[str, Dict[str, Any]] = {
    "cattle": {
        "display_name": "Cattle",
        "article": "Article 1(1)(a)",
        "cn_code_count": 14,
        "description": "Live bovine animals, beef, leather, hides, tallow",
        "high_risk_origins": ["BRA", "ARG", "PRY", "BOL", "COL", "VEN", "GUY", "NIC"],
        "certifications": [],
    },
    "cocoa": {
        "display_name": "Cocoa",
        "article": "Article 1(1)(b)",
        "cn_code_count": 6,
        "description": "Cocoa beans, paste, butter, powder, chocolate",
        "high_risk_origins": ["CIV", "GHA", "CMR", "NGA", "IDN", "SLE", "LBR"],
        "certifications": ["RAINFOREST_ALLIANCE", "UTZ", "FAIRTRADE", "ORGANIC"],
    },
    "coffee": {
        "display_name": "Coffee",
        "article": "Article 1(1)(c)",
        "cn_code_count": 3,
        "description": "Coffee beans, roasted, ground, instant, extracts",
        "high_risk_origins": ["BRA", "VNM", "COL", "IDN", "ETH", "HND", "UGA", "PER"],
        "certifications": ["FOUR_C", "RAINFOREST_ALLIANCE", "UTZ", "FAIRTRADE", "ORGANIC"],
    },
    "oil_palm": {
        "display_name": "Oil Palm",
        "article": "Article 1(1)(d)",
        "cn_code_count": 10,
        "description": "Palm oil, palm kernel oil, oleochemicals, biodiesel",
        "high_risk_origins": ["IDN", "MYS", "PNG", "COL", "NGA", "THA", "GTM", "HND"],
        "certifications": ["RSPO", "ISCC", "MSPO", "ISPO"],
    },
    "rubber": {
        "display_name": "Rubber",
        "article": "Article 1(1)(e)",
        "cn_code_count": 12,
        "description": "Natural rubber, tyres, gloves, industrial goods",
        "high_risk_origins": ["THA", "IDN", "VNM", "CIV", "MYS", "CHN", "MMR", "LAO", "KHM"],
        "certifications": ["FSC", "PEFC"],
    },
    "soya": {
        "display_name": "Soya",
        "article": "Article 1(1)(f)",
        "cn_code_count": 5,
        "description": "Soybeans, soybean oil, meal, lecithin, animal feed",
        "high_risk_origins": ["BRA", "ARG", "USA", "PRY", "BOL", "URY"],
        "certifications": ["RTRS", "PROTERRA", "ISCC"],
    },
    "wood": {
        "display_name": "Wood",
        "article": "Article 1(1)(g)",
        "cn_code_count": 73,
        "description": "Timber, charcoal, pulp, paper, furniture, cork",
        "high_risk_origins": ["BRA", "COD", "IDN", "MYS", "PNG", "MMR", "CMR", "COG", "GAB", "GHA"],
        "certifications": ["FSC", "PEFC", "SFI"],
    },
}

# Comprehensive Annex I CN code database per commodity (top 20 per commodity for brevity)
ANNEX_I_CN_CODES: Dict[str, List[Dict[str, str]]] = {
    "cattle": [
        {"code": "0102", "description": "Live bovine animals"},
        {"code": "0201", "description": "Meat of bovine animals, fresh or chilled"},
        {"code": "0202", "description": "Meat of bovine animals, frozen"},
        {"code": "0206 10", "description": "Edible offal of bovine animals, fresh or chilled"},
        {"code": "0210 20", "description": "Meat of bovine animals, salted, dried or smoked"},
        {"code": "1502", "description": "Fats of bovine animals, sheep or goats"},
        {"code": "1602 50", "description": "Prepared or preserved meat of bovine animals"},
        {"code": "4101", "description": "Raw hides and skins of bovine or equine animals"},
        {"code": "4104", "description": "Tanned or crust hides and skins of bovine"},
        {"code": "4107", "description": "Leather further prepared after tanning, bovine"},
    ],
    "cocoa": [
        {"code": "1801 00 00", "description": "Cocoa beans, whole or broken, raw or roasted"},
        {"code": "1802 00 00", "description": "Cocoa shells, husks, skins and waste"},
        {"code": "1803", "description": "Cocoa paste, whether or not defatted"},
        {"code": "1804 00 00", "description": "Cocoa butter, fat and oil"},
        {"code": "1805 00 00", "description": "Cocoa powder, not containing added sugar"},
        {"code": "1806", "description": "Chocolate and food preparations containing cocoa"},
    ],
    "coffee": [
        {"code": "0901", "description": "Coffee, whether or not roasted or decaffeinated"},
        {"code": "2101 11", "description": "Extracts, essences and concentrates of coffee"},
        {"code": "2101 12", "description": "Preparations with a basis of coffee extracts"},
    ],
    "oil_palm": [
        {"code": "1207 10", "description": "Palm nuts and kernels"},
        {"code": "1511", "description": "Palm oil and its fractions"},
        {"code": "1513 21", "description": "Crude palm kernel or babassu oil"},
        {"code": "1513 29", "description": "Palm kernel or babassu oil, refined"},
        {"code": "1516 20", "description": "Vegetable fats and oils, hydrogenated (palm)"},
        {"code": "2915", "description": "Saturated acyclic monocarboxylic acids"},
        {"code": "3401", "description": "Soap, organic surface-active products"},
        {"code": "3826 00", "description": "Biodiesel (palm-derived FAME)"},
    ],
    "rubber": [
        {"code": "4001", "description": "Natural rubber in primary forms"},
        {"code": "4005", "description": "Compounded rubber, unvulcanised"},
        {"code": "4010", "description": "Conveyor or transmission belts of rubber"},
        {"code": "4011", "description": "New pneumatic tyres, of rubber"},
        {"code": "4012", "description": "Retreaded or used pneumatic tyres"},
        {"code": "4013", "description": "Inner tubes, of rubber"},
        {"code": "4015", "description": "Articles of apparel (gloves) of rubber"},
        {"code": "4016", "description": "Other articles of vulcanised rubber"},
    ],
    "soya": [
        {"code": "1201", "description": "Soya beans, whether or not broken"},
        {"code": "1208 10", "description": "Soya bean flour and meal"},
        {"code": "1507", "description": "Soya-bean oil and its fractions"},
        {"code": "2304 00 00", "description": "Oil-cake and other solid residues of soya-bean"},
        {"code": "2309", "description": "Preparations for animal feeding (soy-based)"},
    ],
    "wood": [
        {"code": "4401", "description": "Fuel wood, wood in chips, sawdust"},
        {"code": "4402", "description": "Wood charcoal"},
        {"code": "4403", "description": "Wood in the rough"},
        {"code": "4407", "description": "Wood sawn or chipped lengthwise"},
        {"code": "4408", "description": "Sheets for veneering, plywood"},
        {"code": "4410", "description": "Particle board, OSB"},
        {"code": "4411", "description": "Fibreboard of wood"},
        {"code": "4412", "description": "Plywood, veneered panels"},
        {"code": "4418", "description": "Builders joinery and carpentry of wood"},
        {"code": "4701 00", "description": "Mechanical wood pulp"},
        {"code": "4801 00 00", "description": "Newsprint"},
        {"code": "4802", "description": "Uncoated paper for writing, printing"},
        {"code": "4819", "description": "Cartons, boxes, cases, bags of paper"},
        {"code": "4901", "description": "Printed books, brochures, leaflets"},
        {"code": "9403", "description": "Other furniture and parts thereof"},
    ],
}

# High-risk countries list (ISO-3166-1 alpha-3) - 28 countries
HIGH_RISK_COUNTRIES: List[str] = [
    "BRA", "IDN", "COD", "BOL", "PRY", "MYS", "PNG", "COG",
    "MMR", "CMR", "CIV", "GHA", "NGA", "LAO", "KHM", "NIC",
    "SLE", "LBR", "GAB", "VEN", "GUY", "SUR", "GTM", "HND",
    "TZA", "MOZ", "MDG", "AGO",
]

# Low-risk countries list (EU-27 + EEA/EFTA + select OECD) - 38 countries
LOW_RISK_COUNTRIES: List[str] = [
    # EU-27
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST",
    "FIN", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA", "LVA",
    "LTU", "LUX", "MLT", "NLD", "POL", "PRT", "ROU", "SVK",
    "SVN", "ESP", "SWE",
    # EEA / EFTA
    "ISL", "LIE", "NOR", "CHE",
    # Other OECD
    "GBR", "CAN", "AUS", "NZL", "JPN", "KOR", "SGP",
]

# Country risk database with Article 29 benchmarking details (top 50 for brevity)
COUNTRY_RISK_DATABASE: Dict[str, Dict[str, Any]] = {
    # HIGH RISK (28 countries)
    "BRA": {"name": "Brazil", "benchmark": "HIGH_RISK", "region": "South America", "forest_cover_pct": 59.0, "annual_deforestation_rate": 0.45, "governance_score": 0.42},
    "IDN": {"name": "Indonesia", "benchmark": "HIGH_RISK", "region": "Southeast Asia", "forest_cover_pct": 49.1, "annual_deforestation_rate": 0.75, "governance_score": 0.40},
    "COD": {"name": "DR Congo", "benchmark": "HIGH_RISK", "region": "Central Africa", "forest_cover_pct": 67.3, "annual_deforestation_rate": 0.40, "governance_score": 0.18},
    "BOL": {"name": "Bolivia", "benchmark": "HIGH_RISK", "region": "South America", "forest_cover_pct": 50.6, "annual_deforestation_rate": 0.50, "governance_score": 0.35},
    "PRY": {"name": "Paraguay", "benchmark": "HIGH_RISK", "region": "South America", "forest_cover_pct": 38.6, "annual_deforestation_rate": 0.80, "governance_score": 0.38},
    "MYS": {"name": "Malaysia", "benchmark": "HIGH_RISK", "region": "Southeast Asia", "forest_cover_pct": 54.6, "annual_deforestation_rate": 0.40, "governance_score": 0.52},
    "PNG": {"name": "Papua New Guinea", "benchmark": "HIGH_RISK", "region": "Oceania", "forest_cover_pct": 74.1, "annual_deforestation_rate": 0.35, "governance_score": 0.25},
    "COG": {"name": "Republic of Congo", "benchmark": "HIGH_RISK", "region": "Central Africa", "forest_cover_pct": 65.4, "annual_deforestation_rate": 0.20, "governance_score": 0.22},
    "MMR": {"name": "Myanmar", "benchmark": "HIGH_RISK", "region": "Southeast Asia", "forest_cover_pct": 42.9, "annual_deforestation_rate": 0.85, "governance_score": 0.15},
    "CMR": {"name": "Cameroon", "benchmark": "HIGH_RISK", "region": "Central Africa", "forest_cover_pct": 39.8, "annual_deforestation_rate": 0.30, "governance_score": 0.28},
    "CIV": {"name": "Cote d'Ivoire", "benchmark": "HIGH_RISK", "region": "West Africa", "forest_cover_pct": 8.9, "annual_deforestation_rate": 2.60, "governance_score": 0.30},
    "GHA": {"name": "Ghana", "benchmark": "HIGH_RISK", "region": "West Africa", "forest_cover_pct": 21.0, "annual_deforestation_rate": 1.20, "governance_score": 0.48},
    "NGA": {"name": "Nigeria", "benchmark": "HIGH_RISK", "region": "West Africa", "forest_cover_pct": 7.2, "annual_deforestation_rate": 3.70, "governance_score": 0.32},
    # LOW RISK - EU-27 (sample)
    "DEU": {"name": "Germany", "benchmark": "LOW_RISK", "region": "Western Europe", "forest_cover_pct": 32.8, "annual_deforestation_rate": 0.0, "governance_score": 0.93},
    "FRA": {"name": "France", "benchmark": "LOW_RISK", "region": "Western Europe", "forest_cover_pct": 31.4, "annual_deforestation_rate": 0.0, "governance_score": 0.88},
    "NLD": {"name": "Netherlands", "benchmark": "LOW_RISK", "region": "Western Europe", "forest_cover_pct": 11.2, "annual_deforestation_rate": 0.0, "governance_score": 0.93},
    # STANDARD RISK (sample)
    "USA": {"name": "United States", "benchmark": "STANDARD_RISK", "region": "North America", "forest_cover_pct": 33.9, "annual_deforestation_rate": 0.01, "governance_score": 0.85},
    "CHN": {"name": "China", "benchmark": "STANDARD_RISK", "region": "East Asia", "forest_cover_pct": 23.3, "annual_deforestation_rate": 0.0, "governance_score": 0.55},
    "IND": {"name": "India", "benchmark": "STANDARD_RISK", "region": "South Asia", "forest_cover_pct": 24.3, "annual_deforestation_rate": 0.0, "governance_score": 0.52},
}


# =============================================================================
# Pydantic Sub-Config Models (22 models: 12 from Starter + 10 Professional)
# =============================================================================


class OperatorConfig(BaseModel):
    """Operator/trader identification per EUDR Article 2."""

    company_name: str = Field("", description="Legal name of the operator or trader")
    eori_number: str = Field("", description="Economic Operators Registration and Identification number")
    registration_country: str = Field("DEU", description="ISO-3166-1 alpha-3 country code")
    operator_type: OperatorType = Field(OperatorType.OPERATOR, description="Operator or trader per Article 2")
    company_size: CompanySize = Field(CompanySize.LARGE, description="Company size classification")
    contact_email: str = Field("", description="Primary contact email for EUDR compliance")
    vat_number: str = Field("", description="EU VAT identification number")

    @field_validator("registration_country")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country code is 3-letter ISO format."""
        if len(v) != 3 or not v.isalpha() or not v.isupper():
            raise ValueError(f"Country code must be 3-letter uppercase ISO-3166-1 alpha-3: {v}")
        return v


class CommodityConfig(BaseModel):
    """Configuration for a single EUDR commodity."""

    commodity_type: EUDRCommodity = Field(..., description="EUDR commodity type")
    enabled: bool = Field(True, description="Whether commodity is enabled")
    cn_codes: List[str] = Field(default_factory=list, description="Annex I CN codes")
    high_risk_origins: List[str] = Field(default_factory=list, description="High-risk origin countries")
    certification_schemes: List[CertificationScheme] = Field(default_factory=list, description="Certifications")
    annual_volume_tonnes: Optional[float] = Field(None, ge=0, description="Annual import volume")
    priority: int = Field(1, ge=1, le=10, description="Priority ranking (1=highest)")


class GeolocationConfig(BaseModel):
    """Geolocation verification configuration per Article 9(1)(d)."""

    coordinate_precision: int = Field(6, ge=4, le=10, description="Decimal places (6 = ~0.11m)")
    polygon_max_vertices: int = Field(10000, ge=100, le=100000, description="Max vertices per polygon")
    polygon_area_threshold_ha: float = Field(4.0, ge=0, description="Polygon threshold (Article 9)")
    area_unit: AreaUnit = Field(AreaUnit.HECTARES, description="Default area unit")
    crs: str = Field("EPSG:4326", description="Coordinate Reference System (WGS 84)")
    allowed_crs_list: List[str] = Field(
        default_factory=lambda: ["EPSG:4326", "EPSG:3857", "EPSG:32601"],
        description="Allowed CRS for input",
    )
    batch_size: int = Field(500, ge=10, le=10000, description="Batch size for verification")
    overlap_detection_enabled: bool = Field(True, description="Enable overlap detection")
    satellite_overlay_enabled: bool = Field(True, description="Enable satellite overlay")
    coordinate_format: CoordinateFormat = Field(CoordinateFormat.DECIMAL_DEGREES, description="Default format")


class RiskAssessmentConfig(BaseModel):
    """Risk assessment weights and thresholds per Article 10."""

    country_weight: float = Field(0.35, ge=0.0, le=1.0, description="Country risk weight")
    supplier_weight: float = Field(0.25, ge=0.0, le=1.0, description="Supplier risk weight")
    commodity_weight: float = Field(0.20, ge=0.0, le=1.0, description="Commodity risk weight")
    document_weight: float = Field(0.20, ge=0.0, le=1.0, description="Documentation weight")
    thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "low_max": 25.0,
            "medium_max": 50.0,
            "high_max": 75.0,
            "critical_min": 75.0,
        },
        description="Risk level thresholds",
    )
    auto_escalation_enabled: bool = Field(True, description="Auto-escalate high risk")
    reassessment_interval_days: int = Field(90, ge=1, le=365, description="Reassessment interval")


class DDSConfig(BaseModel):
    """Due Diligence Statement configuration per Article 4."""

    dds_type: DDSType = Field(DDSType.STANDARD, description="Standard or simplified DDS")
    auto_generate_reference_number: bool = Field(True, description="Auto-generate reference number")
    require_all_fields: bool = Field(True, description="Require all mandatory fields")
    enable_draft_mode: bool = Field(True, description="Allow draft DDS")
    submission_timeout_seconds: int = Field(300, ge=60, le=3600, description="EUIS submission timeout")
    retry_attempts: int = Field(3, ge=0, le=10, description="Retry attempts on failure")
    annex_ii_format_version: str = Field("2024-01", description="Annex II format version")


class ComplianceConfig(BaseModel):
    """General compliance configuration."""

    require_geolocation: bool = Field(True, description="Require geolocation for all plots")
    require_certification: bool = Field(False, description="Require voluntary certification")
    allow_self_declarations: bool = Field(True, description="Allow supplier self-declarations")
    verification_level: str = Field("STANDARD", description="STANDARD or ENHANCED")
    audit_retention_years: int = Field(5, ge=1, le=20, description="Audit trail retention")
    data_quality_threshold: float = Field(0.85, ge=0.0, le=1.0, description="Min data quality score")


class CutoffDateConfig(BaseModel):
    """Cutoff date verification configuration."""

    cutoff_date: date = Field(CUTOFF_DATE, description="EUDR cutoff date (31 Dec 2020)")
    satellite_lookback_years: int = Field(10, ge=1, le=30, description="Satellite history years")
    hansen_forest_change_enabled: bool = Field(True, description="Use Hansen Global Forest Change")
    temporal_analysis_periods: int = Field(3, ge=1, le=10, description="Temporal analysis periods")
    allow_manual_override: bool = Field(False, description="Allow manual override with justification")


class ReportingConfig(BaseModel):
    """Reporting and dashboard configuration."""

    default_report_format: str = Field("PDF", description="PDF, HTML, JSON, XML")
    include_executive_summary: bool = Field(True, description="Include exec summary")
    include_risk_heatmap: bool = Field(True, description="Include risk heatmap")
    include_supplier_details: bool = Field(True, description="Include supplier details")
    language: str = Field("en", description="Report language code")
    timezone: str = Field("UTC", description="Timezone for timestamps")


class DemoConfig(BaseModel):
    """Demo mode configuration for testing and training."""

    demo_mode_enabled: bool = Field(False, description="Enable demo mode")
    use_synthetic_data: bool = Field(False, description="Use synthetic test data")
    mock_eu_is_responses: bool = Field(False, description="Mock EU Information System")
    mock_satellite_imagery: bool = Field(False, description="Mock satellite data")
    tutorial_mode_enabled: bool = Field(False, description="Enable guided tutorials")


# =============================================================================
# Professional Tier Sub-Config Models (10 new models)
# =============================================================================


class AdvancedGeolocationConfig(BaseModel):
    """Advanced geolocation configuration with Sentinel/MODIS integration."""

    sentinel_integration: bool = Field(True, description="Enable Sentinel-1/2 integration")
    modis_fire_check: bool = Field(True, description="Enable MODIS fire detection")
    protected_area_check: bool = Field(True, description="Check against protected areas")
    indigenous_land_check: bool = Field(True, description="Check against indigenous lands")
    hansen_forest_change: bool = Field(True, description="Hansen Global Forest Change analysis")
    resolution_meters: int = Field(10, ge=1, le=100, description="Imagery resolution (meters)")
    temporal_periods: int = Field(5, ge=1, le=20, description="Multi-temporal analysis periods")
    alert_systems: List[AlertSystem] = Field(
        default_factory=lambda: [AlertSystem.GLAD, AlertSystem.RADD],
        description="Deforestation alert systems",
    )


class ScenarioRiskConfig(BaseModel):
    """Scenario-based risk modeling configuration (Monte Carlo)."""

    simulation_count: int = Field(10000, ge=1000, le=100000, description="Monte Carlo simulations")
    confidence_levels: List[float] = Field(
        default_factory=lambda: [0.90, 0.95, 0.99],
        description="Confidence intervals",
    )
    seed: int = Field(42, description="Random seed for reproducibility")
    parallel_workers: int = Field(4, ge=1, le=32, description="Parallel worker threads")
    timeout_seconds: int = Field(60, ge=10, le=600, description="Simulation timeout")
    distributions: Dict[str, RiskDistribution] = Field(
        default_factory=lambda: {
            "country_risk": RiskDistribution.BETA,
            "supplier_risk": RiskDistribution.LOGNORMAL,
            "commodity_risk": RiskDistribution.NORMAL,
        },
        description="Probability distributions per risk factor",
    )


class SatelliteMonitoringConfig(BaseModel):
    """Real-time satellite monitoring configuration."""

    providers: List[SatelliteProvider] = Field(
        default_factory=lambda: [
            SatelliteProvider.SENTINEL_1,
            SatelliteProvider.SENTINEL_2,
            SatelliteProvider.MODIS,
        ],
        description="Active satellite providers",
    )
    check_interval_hours: int = Field(6, ge=1, le=168, description="Check interval (6 hours)")
    alert_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Alert confidence threshold")
    historical_months: int = Field(60, ge=12, le=120, description="Historical data months")
    resolution_meters: int = Field(10, ge=1, le=100, description="Minimum resolution")


class ContinuousMonitoringConfig(BaseModel):
    """Continuous monitoring configuration."""

    enabled: bool = Field(True, description="Enable continuous monitoring")
    intervals: Dict[str, int] = Field(
        default_factory=lambda: {
            "satellite_check_hours": 6,
            "risk_reassessment_hours": 24,
            "supplier_update_hours": 168,  # Weekly
            "regulatory_check_hours": 24,
        },
        description="Monitoring intervals",
    )
    notification_channels: List[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.EMAIL, NotificationChannel.WEBHOOK],
        description="Notification channels",
    )
    escalation_enabled: bool = Field(True, description="Auto-escalate critical events")


class PortfolioConfig(BaseModel):
    """Multi-operator portfolio management configuration."""

    max_operators: int = Field(100, ge=1, le=1000, description="Max operators in portfolio")
    shared_supplier_pool: bool = Field(True, description="Share supplier pool across operators")
    cross_operator_reporting: bool = Field(True, description="Enable cross-operator reports")
    cost_allocation: bool = Field(True, description="Enable cost allocation")
    hierarchy_depth: int = Field(3, ge=1, le=10, description="Organizational hierarchy depth")


class AuditManagementConfig(BaseModel):
    """Advanced audit trail management configuration."""

    retention_years: int = Field(5, ge=1, le=20, description="Audit retention (years)")
    hash_algorithm: str = Field("SHA-256", description="Provenance hash algorithm")
    export_formats: List[str] = Field(
        default_factory=lambda: ["JSON", "XML", "PDF"],
        description="Export formats",
    )
    mock_audit_enabled: bool = Field(True, description="Enable mock audit simulations")


class ProtectedAreaConfig(BaseModel):
    """Protected area screening configuration."""

    wdpa_enabled: bool = Field(True, description="World Database on Protected Areas")
    kba_enabled: bool = Field(True, description="Key Biodiversity Areas")
    indigenous_check: bool = Field(True, description="Indigenous lands check")
    buffer_km: float = Field(5.0, ge=0.0, le=100.0, description="Buffer zone (km)")
    ramsar_check: bool = Field(True, description="Ramsar wetlands check")
    unesco_check: bool = Field(True, description="UNESCO heritage sites check")


class RegulatoryTrackingConfig(BaseModel):
    """Regulatory change tracking configuration."""

    eurlex_monitoring: bool = Field(True, description="Monitor EUR-Lex for changes")
    check_interval_hours: int = Field(24, ge=1, le=168, description="Check interval")
    cross_regulation_tracking: bool = Field(True, description="Track CSRD E4, CSDDD, etc.")
    auto_gap_analysis: bool = Field(True, description="Auto gap analysis on changes")


class GrievanceConfig(BaseModel):
    """Grievance mechanism configuration."""

    enabled: bool = Field(True, description="Enable grievance mechanism")
    anonymous_submissions: bool = Field(True, description="Allow anonymous submissions")
    response_sla_days: int = Field(5, ge=1, le=30, description="Response SLA (days)")
    resolution_sla_days: int = Field(30, ge=1, le=365, description="Resolution SLA (days)")
    whistleblower_protection: bool = Field(True, description="Whistleblower protection protocols")


class CrossRegulationConfig(BaseModel):
    """Cross-regulation linkage configuration."""

    csrd_e4_linkage: bool = Field(True, description="Link to CSRD ESRS E4 (Biodiversity)")
    csddd_linkage: bool = Field(True, description="Link to CSDDD (Due Diligence Directive)")
    nature_restoration_linkage: bool = Field(False, description="Link to Nature Restoration Law")


class AdvancedSupplyChainConfig(BaseModel):
    """Advanced supply chain configuration."""

    max_tier_depth: int = Field(5, ge=1, le=10, description="Max supply chain tier depth")
    chain_of_custody_models: List[ChainOfCustodyModel] = Field(
        default_factory=lambda: list(ChainOfCustodyModel),
        description="Supported CoC models",
    )
    network_analysis: bool = Field(True, description="Supply chain network topology analysis")
    diversification_scoring: bool = Field(True, description="Supplier diversification scoring")


class SupplierBenchmarkConfig(BaseModel):
    """Supplier benchmarking configuration."""

    peer_group_min_size: int = Field(5, ge=3, le=100, description="Min peer group size")
    scoring_dimensions: List[BenchmarkDimension] = Field(
        default_factory=lambda: list(BenchmarkDimension),
        description="Benchmarking dimensions",
    )
    benchmark_frequency: str = Field("QUARTERLY", description="MONTHLY, QUARTERLY, ANNUAL")
    degradation_alert: bool = Field(True, description="Alert on performance degradation")


# =============================================================================
# Main Configuration Classes
# =============================================================================


class EUDRProfessionalConfig(BaseModel):
    """EUDR Professional Pack configuration.

    Extends PACK-006 Starter configuration with professional-tier features:
    - All 40 EUDR agents
    - Advanced geolocation (Sentinel, MODIS)
    - Scenario risk modeling (Monte Carlo)
    - Real-time satellite monitoring
    - Multi-operator portfolio
    - Advanced audit trail
    - Protected area screening
    - Supplier benchmarking
    - Regulatory change tracking
    - Grievance mechanism
    - Cross-regulation linkage
    """

    # Pack metadata
    pack_id: str = Field("PACK-007-eudr-professional", description="Pack identifier")
    version: str = Field("1.0.0", description="Pack version")
    tier: str = Field("professional", description="Pack tier")
    extends: str = Field("PACK-006-eudr-starter", description="Extends Starter pack")

    # Base configurations (from PACK-006)
    operator: OperatorConfig = Field(default_factory=OperatorConfig)
    commodities: List[CommodityConfig] = Field(default_factory=list)
    geolocation: GeolocationConfig = Field(default_factory=GeolocationConfig)
    risk_assessment: RiskAssessmentConfig = Field(default_factory=RiskAssessmentConfig)
    dds: DDSConfig = Field(default_factory=DDSConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    cutoff_date: CutoffDateConfig = Field(default_factory=CutoffDateConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    demo: DemoConfig = Field(default_factory=DemoConfig)

    # Professional configurations (10 new models)
    advanced_geolocation: AdvancedGeolocationConfig = Field(default_factory=AdvancedGeolocationConfig)
    scenario_risk: ScenarioRiskConfig = Field(default_factory=ScenarioRiskConfig)
    satellite_monitoring: SatelliteMonitoringConfig = Field(default_factory=SatelliteMonitoringConfig)
    continuous_monitoring: ContinuousMonitoringConfig = Field(default_factory=ContinuousMonitoringConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    audit_management: AuditManagementConfig = Field(default_factory=AuditManagementConfig)
    protected_areas: ProtectedAreaConfig = Field(default_factory=ProtectedAreaConfig)
    regulatory_tracking: RegulatoryTrackingConfig = Field(default_factory=RegulatoryTrackingConfig)
    grievance: GrievanceConfig = Field(default_factory=GrievanceConfig)
    cross_regulation: CrossRegulationConfig = Field(default_factory=CrossRegulationConfig)
    advanced_supply_chain: AdvancedSupplyChainConfig = Field(default_factory=AdvancedSupplyChainConfig)
    supplier_benchmark: SupplierBenchmarkConfig = Field(default_factory=SupplierBenchmarkConfig)

    @model_validator(mode="after")
    def validate_professional_features(self) -> "EUDRProfessionalConfig":
        """Validate professional feature dependencies."""
        # If continuous monitoring enabled, ensure notification channels configured
        if self.continuous_monitoring.enabled and not self.continuous_monitoring.notification_channels:
            raise ValueError("Continuous monitoring requires at least one notification channel")

        # If satellite monitoring enabled, ensure at least one provider
        if self.satellite_monitoring.providers is None or len(self.satellite_monitoring.providers) == 0:
            raise ValueError("Satellite monitoring requires at least one provider")

        # If portfolio management enabled, ensure max_operators > 0
        if self.portfolio.max_operators <= 0:
            raise ValueError("Portfolio max_operators must be positive")

        return self

    def get_active_agents(self) -> List[str]:
        """Get list of all active EUDR agents (all 40 in Professional)."""
        return [f"AGENT-EUDR-{str(i).zfill(3)}" for i in range(1, 41)]

    def get_professional_features(self) -> Dict[str, bool]:
        """Get dictionary of professional features and their enabled status."""
        return {
            "advanced_geolocation": self.advanced_geolocation.sentinel_integration,
            "scenario_risk_modeling": self.scenario_risk.simulation_count > 0,
            "satellite_monitoring": len(self.satellite_monitoring.providers) > 0,
            "continuous_monitoring": self.continuous_monitoring.enabled,
            "portfolio_management": self.portfolio.max_operators > 1,
            "advanced_audit_trail": self.audit_management.retention_years >= 5,
            "protected_area_screening": self.protected_areas.wdpa_enabled,
            "regulatory_tracking": self.regulatory_tracking.eurlex_monitoring,
            "grievance_mechanism": self.grievance.enabled,
            "cross_regulation_linkage": self.cross_regulation.csrd_e4_linkage,
            "supplier_benchmarking": len(self.supplier_benchmark.scoring_dimensions) > 0,
        }


class PackConfig(BaseModel):
    """Top-level pack configuration loader with YAML support."""

    pack: EUDRProfessionalConfig = Field(default_factory=EUDRProfessionalConfig)
    loaded_from: List[str] = Field(default_factory=list, description="Config files loaded")
    merge_timestamp: datetime = Field(default_factory=datetime.now)

    @classmethod
    def load(
        cls,
        size_preset: Optional[str] = None,
        sector_preset: Optional[str] = None,
        demo_mode: bool = False,
    ) -> "PackConfig":
        """Load configuration with preset and sector overlays.

        Args:
            size_preset: Size preset (multi_commodity, high_risk, trading_company, etc.)
            sector_preset: Sector preset (palm_oil_professional, timber_professional, etc.)
            demo_mode: Enable demo mode with synthetic data

        Returns:
            Loaded PackConfig instance
        """
        config = EUDRProfessionalConfig()
        loaded_files = []

        # Load size preset if specified
        if size_preset:
            preset_path = CONFIG_DIR / "presets" / f"{size_preset}.yaml"
            if preset_path.exists():
                with open(preset_path, "r", encoding="utf-8") as f:
                    preset_data = yaml.safe_load(f)
                    config = cls._merge_config(config, preset_data)
                    loaded_files.append(str(preset_path))
                    logger.info(f"Loaded size preset: {size_preset}")

        # Load sector preset if specified
        if sector_preset:
            sector_path = CONFIG_DIR / "sectors" / f"{sector_preset}.yaml"
            if sector_path.exists():
                with open(sector_path, "r", encoding="utf-8") as f:
                    sector_data = yaml.safe_load(f)
                    config = cls._merge_config(config, sector_data)
                    loaded_files.append(str(sector_path))
                    logger.info(f"Loaded sector preset: {sector_preset}")

        # Enable demo mode if requested
        if demo_mode:
            config.demo.demo_mode_enabled = True
            config.demo.use_synthetic_data = True
            config.demo.mock_eu_is_responses = True
            logger.info("Demo mode enabled with synthetic data")

        return cls(pack=config, loaded_from=loaded_files)

    @staticmethod
    def _merge_config(base: EUDRProfessionalConfig, overlay: Dict[str, Any]) -> EUDRProfessionalConfig:
        """Deep merge overlay config into base config."""
        base_dict = base.model_dump()

        def deep_merge(d1: Dict, d2: Dict) -> Dict:
            for key, value in d2.items():
                if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                    d1[key] = deep_merge(d1[key], value)
                else:
                    d1[key] = value
            return d1

        merged = deep_merge(base_dict, overlay)
        return EUDRProfessionalConfig(**merged)

    def export_yaml(self, output_path: Path) -> None:
        """Export configuration to YAML file."""
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.pack.model_dump(), f, default_flow_style=False, sort_keys=False)
        logger.info(f"Exported configuration to {output_path}")

    def export_json(self, output_path: Path) -> None:
        """Export configuration to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.pack.model_dump(), f, indent=2, default=str)
        logger.info(f"Exported configuration to {output_path}")

    def get_config_hash(self) -> str:
        """Get SHA-256 hash of configuration for change detection."""
        config_json = json.dumps(self.pack.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(config_json.encode()).hexdigest()


# =============================================================================
# Utility Functions
# =============================================================================


def get_country_benchmark(iso3_code: str) -> CountryBenchmark:
    """Get country benchmark classification per Article 29.

    Args:
        iso3_code: ISO-3166-1 alpha-3 country code

    Returns:
        CountryBenchmark classification
    """
    if iso3_code in HIGH_RISK_COUNTRIES:
        return CountryBenchmark.HIGH_RISK
    elif iso3_code in LOW_RISK_COUNTRIES:
        return CountryBenchmark.LOW_RISK
    else:
        return CountryBenchmark.STANDARD_RISK


def get_commodity_info(commodity: Union[str, EUDRCommodity]) -> Dict[str, Any]:
    """Get detailed commodity information.

    Args:
        commodity: Commodity type (string or enum)

    Returns:
        Dictionary with commodity details
    """
    commodity_key = commodity.value if isinstance(commodity, EUDRCommodity) else commodity
    return EUDR_COMMODITIES.get(commodity_key, {})


def validate_cn_code(cn_code: str, commodity: Union[str, EUDRCommodity]) -> bool:
    """Validate CN code against commodity's Annex I codes.

    Args:
        cn_code: Combined Nomenclature code
        commodity: Commodity type

    Returns:
        True if CN code is valid for commodity
    """
    commodity_key = commodity.value if isinstance(commodity, EUDRCommodity) else commodity
    valid_codes = ANNEX_I_CN_CODES.get(commodity_key, [])
    return any(code["code"].startswith(cn_code) for code in valid_codes)
