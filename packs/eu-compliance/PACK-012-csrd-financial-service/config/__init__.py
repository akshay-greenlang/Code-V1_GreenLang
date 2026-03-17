"""
PACK-012 CSRD Financial Service Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the CSRD Financial Service Pack. Import from this module to
access the full configuration API.

Usage:
    >>> from packs.eu_compliance.PACK_012_csrd_financial_service.config import (
    ...     PackConfig,
    ...     CSRDFinancialServiceConfig,
    ...     FinancialInstitutionType,
    ...     PCAFAssetClass,
    ...     PCAFDataQuality,
    ...     get_institution_display_name,
    ...     get_required_disclosures,
    ...     get_pcaf_asset_class_info,
    ... )
    >>> config = PackConfig.from_preset("bank")
    >>> print(config.pack.institution_type)
    FinancialInstitutionType.BANK
"""

from .pack_config import (
    # Enums
    ClimateRiskType,
    ComplianceStatus,
    DisclosureFormat,
    ESRSTopic,
    FinancialInstitutionType,
    GARExposureCategory,
    GARScope,
    InsuranceLineType,
    MaterialityLevel,
    NGFSScenario,
    PCAFAssetClass,
    PCAFDataQuality,
    PhysicalHazardType,
    Pillar3Template,
    ReportingFrequency,
    SBTiFIMethod,
    TransitionRiskChannel,
    # Sub-config models
    AuditTrailConfig,
    ClimateRiskConfig,
    DataQualityConfig,
    DisclosureConfig,
    FSMaterialityConfig,
    FSTransitionPlanConfig,
    GARBTARConfig,
    InsuranceConfig,
    PCAFConfig,
    Pillar3Config,
    SBTiFIConfig,
    # Main config models
    CSRDFinancialServiceConfig,
    PackConfig,
    # Constants
    AVAILABLE_PRESETS,
    INSTITUTION_DISPLAY_NAMES,
    NGFS_SCENARIO_DESCRIPTIONS,
    PCAF_ASSET_CLASS_INFO,
    REQUIRED_DISCLOSURES,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_institution_display_name,
    get_ngfs_scenario_info,
    get_pcaf_asset_class_info,
    get_required_disclosures,
    list_available_presets,
)

__all__ = [
    # Enums
    "ClimateRiskType",
    "ComplianceStatus",
    "DisclosureFormat",
    "ESRSTopic",
    "FinancialInstitutionType",
    "GARExposureCategory",
    "GARScope",
    "InsuranceLineType",
    "MaterialityLevel",
    "NGFSScenario",
    "PCAFAssetClass",
    "PCAFDataQuality",
    "PhysicalHazardType",
    "Pillar3Template",
    "ReportingFrequency",
    "SBTiFIMethod",
    "TransitionRiskChannel",
    # Sub-config models
    "AuditTrailConfig",
    "ClimateRiskConfig",
    "DataQualityConfig",
    "DisclosureConfig",
    "FSMaterialityConfig",
    "FSTransitionPlanConfig",
    "GARBTARConfig",
    "InsuranceConfig",
    "PCAFConfig",
    "Pillar3Config",
    "SBTiFIConfig",
    # Main config models
    "CSRDFinancialServiceConfig",
    "PackConfig",
    # Constants
    "AVAILABLE_PRESETS",
    "INSTITUTION_DISPLAY_NAMES",
    "NGFS_SCENARIO_DESCRIPTIONS",
    "PCAF_ASSET_CLASS_INFO",
    "REQUIRED_DISCLOSURES",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_institution_display_name",
    "get_ngfs_scenario_info",
    "get_pcaf_asset_class_info",
    "get_required_disclosures",
    "list_available_presets",
]