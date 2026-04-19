"""
PACK-024 Carbon Neutral Pack - Configuration Module

Provides comprehensive configuration management for the Carbon Neutral Pack,
including runtime configuration models, preset loading, and validation.

Classes:
    CarbonNeutralConfig: Main configuration container for the pack
    PackConfig: Top-level wrapper with preset/YAML loading support

Enums:
    NeutralityType: Carbon neutrality boundary type (CORPORATE, SME, EVENT, etc.)
    ConsolidationApproach: GHG Protocol consolidation approach
    CreditCategory: Carbon credit category (AVOIDANCE, REMOVAL)
    CreditStandard: Carbon credit standard/registry
    NeutralizationStandard: Neutrality standard (ISO 14068-1, PAS 2060)
    ClaimType: Carbon neutrality claim type
    AssuranceLevel: Verification assurance level (LIMITED, REASONABLE)
    BalanceMethod: Neutralization balance method
    VCMIClaimTier: VCMI offset portfolio tier
    PermanenceCategory: Credit permanence category
    VerificationStatus: Neutrality verification status

Functions:
    load_preset(preset_name, overrides): Load a named preset configuration
    list_available_presets(): List all available presets
    validate_config(config): Validate configuration and return warnings

Example:
    >>> from packs.net_zero.PACK_024_carbon_neutral.config import (
    ...     CarbonNeutralConfig, PackConfig, load_preset
    ... )
    >>> config = load_preset("corporate_neutrality")
    >>> warnings = config.validate_config()
    >>> engines = config.pack.get_enabled_engines()
    >>> config_hash = config.get_config_hash()
"""

from .runtime_config import (
    CarbonNeutralConfig,
    PackConfig,
    # Enums
    NeutralityType,
    OrganizationSize,
    ConsolidationApproach,
    CreditCategory,
    CreditStandard,
    NeutralizationStandard,
    ClaimType,
    AssuranceLevel,
    BalanceMethod,
    DataSourceType,
    ReportFormat,
    Scope3Method,
    ERPType,
    VCMIClaimTier,
    PermanenceCategory,
    VerificationStatus,
    MilestoneFrequency,
    OrganizationSector,
    # Sub-Config Models
    OrganizationConfig,
    BoundaryConfig,
    FootprintConfig,
    CarbonManagementPlanConfig,
    CreditQualityConfig,
    PortfolioOptimizationConfig,
    RegistryRetirementConfig,
    NeutralizationBalanceConfig,
    ClaimsSubstantiationConfig,
    VerificationPackageConfig,
    AnnualCycleConfig,
    PermanenceRiskConfig,
    ReportingConfig,
    PerformanceConfig,
    AuditTrailConfig,
    # Utility Functions
    load_preset,
    list_available_presets,
    validate_config,
    merge_config,
    get_env_overrides,
    # Constants
    SUPPORTED_PRESETS,
    DEFAULT_BASE_YEAR,
    DEFAULT_REPORTING_YEAR,
    DEFAULT_NEUTRALITY_TARGET_YEAR,
    ICVCM_CCP_DIMENSIONS,
    CREDIT_QUALITY_RATINGS,
    CREDIT_PROJECT_TYPES,
    SUPPORTED_REGISTRIES,
    SCOPE3_CATEGORIES_INFO,
)

__all__ = [
    # Main Config Classes
    "CarbonNeutralConfig",
    "PackConfig",
    # Enums
    "NeutralityType",
    "OrganizationSize",
    "ConsolidationApproach",
    "CreditCategory",
    "CreditStandard",
    "NeutralizationStandard",
    "ClaimType",
    "AssuranceLevel",
    "BalanceMethod",
    "DataSourceType",
    "ReportFormat",
    "Scope3Method",
    "ERPType",
    "VCMIClaimTier",
    "PermanenceCategory",
    "VerificationStatus",
    "MilestoneFrequency",
    "OrganizationSector",
    # Sub-Config Models
    "OrganizationConfig",
    "BoundaryConfig",
    "FootprintConfig",
    "CarbonManagementPlanConfig",
    "CreditQualityConfig",
    "PortfolioOptimizationConfig",
    "RegistryRetirementConfig",
    "NeutralizationBalanceConfig",
    "ClaimsSubstantiationConfig",
    "VerificationPackageConfig",
    "AnnualCycleConfig",
    "PermanenceRiskConfig",
    "ReportingConfig",
    "PerformanceConfig",
    "AuditTrailConfig",
    # Functions
    "load_preset",
    "list_available_presets",
    "validate_config",
    "merge_config",
    "get_env_overrides",
    # Constants
    "SUPPORTED_PRESETS",
    "DEFAULT_BASE_YEAR",
    "DEFAULT_REPORTING_YEAR",
    "DEFAULT_NEUTRALITY_TARGET_YEAR",
    "ICVCM_CCP_DIMENSIONS",
    "CREDIT_QUALITY_RATINGS",
    "CREDIT_PROJECT_TYPES",
    "SUPPORTED_REGISTRIES",
    "SCOPE3_CATEGORIES_INFO",
]

__version__ = "1.0.0"
__author__ = "GreenLang Platform Team"
