"""
PACK-005 CBAM Complete Pack - Configuration Module

This module provides configuration management for the CBAM Complete Pack,
including pack manifest loading, preset resolution, sector-specific overlays,
environment-based overrides, and a factory function for flexible instantiation.

Usage:
    >>> from packs.eu_compliance.pack_005_cbam_complete.config import CBAMCompleteConfig
    >>> config = CBAMCompleteConfig.from_preset("enterprise_importer")

    >>> from packs.eu_compliance.pack_005_cbam_complete.config import load_preset
    >>> config = load_preset("steel_group", sector_name="automotive_oem")

    >>> from packs.eu_compliance.pack_005_cbam_complete.config import config_factory
    >>> config = config_factory(source="demo")
"""

from config.pack_config import (
    # Enums - inherited from PACK-004
    CBAMGoodsCategory,
    CalculationMethod,
    CostScenario,
    DataSubmissionFormat,
    EUMemberState,
    ETSPriceSource,
    EmissionFactorSource,
    ReportLanguage,
    ReportingPeriod,
    VerificationFrequency,
    # Enums - new in PACK-005
    AllocationMethod,
    AntiCircumventionRule,
    BuyingStrategy,
    CostAllocationMethod,
    DeclarantStatus,
    EntityRole,
    RegulationTarget,
    ValuationMethod,
    # Config models - inherited from PACK-004
    CertificateConfig,
    DeMinimisConfig,
    EmissionConfig,
    GoodsCategoryConfig,
    ImporterConfig,
    QuarterlyConfig,
    SupplierConfig,
    VerificationConfig,
    # Config models - new in PACK-005
    AdvancedAnalyticsConfig,
    AuditManagementConfig,
    CertificateTradingConfig,
    CrossRegulationConfig,
    CustomsAutomationConfig,
    EntityConfig,
    EntityGroupConfig,
    PrecursorChainConfig,
    RegistryAPIConfig,
    # Main config class
    CBAMCompleteConfig,
    # Utility functions
    config_factory,
    load_preset,
    # Reference data
    ANTI_CIRCUMVENTION_THRESHOLDS,
    CBAM_PENALTIES,
    COUNTRY_DEFAULT_FACTORS,
    EU_DEFAULT_EMISSION_FACTORS,
    EXPANDED_CN_CODES,
    FREE_ALLOCATION_PHASEOUT,
    THIRD_COUNTRY_CARBON_PRICING,
)

__all__ = [
    # Main config
    "CBAMCompleteConfig",
    # Factory / loader
    "config_factory",
    "load_preset",
    # Enums - inherited
    "CBAMGoodsCategory",
    "CalculationMethod",
    "CostScenario",
    "DataSubmissionFormat",
    "EUMemberState",
    "ETSPriceSource",
    "EmissionFactorSource",
    "ReportLanguage",
    "ReportingPeriod",
    "VerificationFrequency",
    # Enums - new
    "AllocationMethod",
    "AntiCircumventionRule",
    "BuyingStrategy",
    "CostAllocationMethod",
    "DeclarantStatus",
    "EntityRole",
    "RegulationTarget",
    "ValuationMethod",
    # Sub-configs - inherited
    "CertificateConfig",
    "DeMinimisConfig",
    "EmissionConfig",
    "GoodsCategoryConfig",
    "ImporterConfig",
    "QuarterlyConfig",
    "SupplierConfig",
    "VerificationConfig",
    # Sub-configs - new
    "AdvancedAnalyticsConfig",
    "AuditManagementConfig",
    "CertificateTradingConfig",
    "CrossRegulationConfig",
    "CustomsAutomationConfig",
    "EntityConfig",
    "EntityGroupConfig",
    "PrecursorChainConfig",
    "RegistryAPIConfig",
    # Reference data
    "ANTI_CIRCUMVENTION_THRESHOLDS",
    "CBAM_PENALTIES",
    "COUNTRY_DEFAULT_FACTORS",
    "EU_DEFAULT_EMISSION_FACTORS",
    "EXPANDED_CN_CODES",
    "FREE_ALLOCATION_PHASEOUT",
    "THIRD_COUNTRY_CARBON_PRICING",
]
