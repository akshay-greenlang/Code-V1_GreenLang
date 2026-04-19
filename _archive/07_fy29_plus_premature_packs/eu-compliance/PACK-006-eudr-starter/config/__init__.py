"""
PACK-006 EUDR Starter Pack - Configuration Module

This module provides configuration management for the EUDR Starter Pack,
including pack manifest loading, preset resolution, sector-specific
configuration, and environment-based overrides.

Usage:
    >>> from packs.eu_compliance.pack_006_eudr_starter.config import PackConfig
    >>> config = PackConfig.load()
    >>> config = PackConfig.load(size_preset="mid_market", sector_preset="palm_oil")

Presets:
    Size presets: large_operator, mid_market, sme, first_time
    Sector presets: palm_oil, timber_wood, cocoa_coffee, soy_cattle, rubber

Regulatory Context:
    - EUDR: Regulation (EU) 2023/1115
    - Cutoff Date: 31 December 2020
    - Commodities: cattle, cocoa, coffee, oil_palm, rubber, soya, wood
"""

from config.pack_config import (
    # Main config classes
    PackConfig,
    EUDRPackConfig,
    PresetConfig,
    # Sub-config models
    OperatorConfig,
    CommodityConfig,
    GeolocationConfig,
    RiskAssessmentConfig,
    DDSConfig,
    EUISConfig,
    SupplyChainConfig,
    SupplierConfig,
    ComplianceConfig,
    CutoffDateConfig,
    ReportingConfig,
    DemoConfig,
    # Component configs
    AgentComponentConfig,
    ComponentsConfig,
    WorkflowConfig,
    WorkflowPhaseConfig,
    TemplateConfig,
    PerformanceTargets,
    RequirementsConfig,
    # Enums
    EUDRCommodity,
    OperatorType,
    CompanySize,
    DDSType,
    DDSStatus,
    RiskLevel,
    CountryBenchmark,
    CertificationScheme,
    ChainOfCustodyModel,
    CoordinateFormat,
    SupplierDDStatus,
    AreaUnit,
    AuthType,
    # Reference data constants
    CUTOFF_DATE,
    POLYGON_AREA_THRESHOLD_HA,
    EUDR_COMMODITIES,
    ANNEX_I_CN_CODES,
    CERTIFICATION_REGISTRIES,
    HIGH_RISK_COUNTRIES,
    LOW_RISK_COUNTRIES,
    COUNTRY_RISK_DATABASE,
    # Utility functions
    get_country_risk,
    is_eudr_commodity,
    get_all_cn_codes,
    calculate_config_hash,
)

__all__ = [
    # Main config classes
    "PackConfig",
    "EUDRPackConfig",
    "PresetConfig",
    # Sub-config models
    "OperatorConfig",
    "CommodityConfig",
    "GeolocationConfig",
    "RiskAssessmentConfig",
    "DDSConfig",
    "EUISConfig",
    "SupplyChainConfig",
    "SupplierConfig",
    "ComplianceConfig",
    "CutoffDateConfig",
    "ReportingConfig",
    "DemoConfig",
    # Component configs
    "AgentComponentConfig",
    "ComponentsConfig",
    "WorkflowConfig",
    "WorkflowPhaseConfig",
    "TemplateConfig",
    "PerformanceTargets",
    "RequirementsConfig",
    # Enums
    "EUDRCommodity",
    "OperatorType",
    "CompanySize",
    "DDSType",
    "DDSStatus",
    "RiskLevel",
    "CountryBenchmark",
    "CertificationScheme",
    "ChainOfCustodyModel",
    "CoordinateFormat",
    "SupplierDDStatus",
    "AreaUnit",
    "AuthType",
    # Reference data constants
    "CUTOFF_DATE",
    "POLYGON_AREA_THRESHOLD_HA",
    "EUDR_COMMODITIES",
    "ANNEX_I_CN_CODES",
    "CERTIFICATION_REGISTRIES",
    "HIGH_RISK_COUNTRIES",
    "LOW_RISK_COUNTRIES",
    "COUNTRY_RISK_DATABASE",
    # Utility functions
    "get_country_risk",
    "is_eudr_commodity",
    "get_all_cn_codes",
    "calculate_config_hash",
]
