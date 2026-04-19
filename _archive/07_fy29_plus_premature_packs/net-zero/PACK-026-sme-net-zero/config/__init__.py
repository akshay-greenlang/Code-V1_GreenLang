"""
PACK-026 SME Net Zero Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the SME Net Zero Pack. Import from this module to access the
full configuration API.

Usage:
    >>> from packs.net_zero.PACK_026_sme_net_zero.config import (
    ...     PackConfig,
    ...     SMENetZeroConfig,
    ...     SMESize,
    ...     SMEDataQualityTier,
    ...     get_sector_info,
    ...     get_default_config,
    ... )
    >>> config = PackConfig.from_preset("small_business")
    >>> print(config.pack.organization.sme_size)
    SMESize.SMALL
"""

from .pack_config import (
    # Enums
    AccountingSoftware,
    BoundaryMethod,
    CertificationPathway,
    EmissionsProfileType,
    GrantCategory,
    MaturityAssessment,
    OffsetStrategy,
    PathwayType,
    QuickWinCategory,
    ReportFormat,
    Scope3Method,
    SMEDataQualityTier,
    SMESector,
    SMESize,
    VerificationLevel,
    # Sub-config models
    SMEAuditTrailConfig,
    SMEBoundaryConfig,
    SMECertificationConfig,
    SMEDataQualityConfig,
    SMEGrantConfig,
    SMEOrganizationConfig,
    SMEPerformanceConfig,
    SMEReductionConfig,
    SMEReportingConfig,
    SMEScopeConfig,
    SMEScorecardConfig,
    SMETargetConfig,
    SMEVerificationConfig,
    # Main config models
    SMENetZeroConfig,
    PackConfig,
    # Constants
    CERTIFICATION_INFO,
    DEFAULT_BASE_YEAR,
    DEFAULT_SCOPE3_MEDIUM_CATEGORIES,
    DEFAULT_SCOPE3_SME_CATEGORIES,
    DEFAULT_TARGET_YEAR_LONG,
    DEFAULT_TARGET_YEAR_NEAR,
    IPCC_AR6_GWP100,
    SBTI_SME_PARAMETERS,
    SECTOR_EMISSIONS_PROFILE,
    SME_BUDGET_RANGES,
    SME_QUICK_WINS,
    SME_REPORT_TEMPLATES,
    SME_SCOPE3_PRIORITIES,
    SME_SECTOR_INFO,
    SME_SIZE_THRESHOLDS,
    SUPPORTED_PRESETS,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_certification_info,
    get_default_config,
    get_emissions_profile,
    get_env_overrides,
    get_gwp100,
    get_sbti_sme_parameters,
    get_sector_info,
    get_sme_budget_range,
    list_available_presets,
    load_preset,
    merge_config,
    validate_config,
)

__all__ = [
    # Enums
    "AccountingSoftware",
    "BoundaryMethod",
    "CertificationPathway",
    "EmissionsProfileType",
    "GrantCategory",
    "MaturityAssessment",
    "OffsetStrategy",
    "PathwayType",
    "QuickWinCategory",
    "ReportFormat",
    "Scope3Method",
    "SMEDataQualityTier",
    "SMESector",
    "SMESize",
    "VerificationLevel",
    # Sub-config models
    "SMEAuditTrailConfig",
    "SMEBoundaryConfig",
    "SMECertificationConfig",
    "SMEDataQualityConfig",
    "SMEGrantConfig",
    "SMEOrganizationConfig",
    "SMEPerformanceConfig",
    "SMEReductionConfig",
    "SMEReportingConfig",
    "SMEScopeConfig",
    "SMEScorecardConfig",
    "SMETargetConfig",
    "SMEVerificationConfig",
    # Main config models
    "SMENetZeroConfig",
    "PackConfig",
    # Constants
    "CERTIFICATION_INFO",
    "DEFAULT_BASE_YEAR",
    "DEFAULT_SCOPE3_MEDIUM_CATEGORIES",
    "DEFAULT_SCOPE3_SME_CATEGORIES",
    "DEFAULT_TARGET_YEAR_LONG",
    "DEFAULT_TARGET_YEAR_NEAR",
    "IPCC_AR6_GWP100",
    "SBTI_SME_PARAMETERS",
    "SECTOR_EMISSIONS_PROFILE",
    "SME_BUDGET_RANGES",
    "SME_QUICK_WINS",
    "SME_REPORT_TEMPLATES",
    "SME_SCOPE3_PRIORITIES",
    "SME_SECTOR_INFO",
    "SME_SIZE_THRESHOLDS",
    "SUPPORTED_PRESETS",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_certification_info",
    "get_default_config",
    "get_emissions_profile",
    "get_env_overrides",
    "get_gwp100",
    "get_sbti_sme_parameters",
    "get_sector_info",
    "get_sme_budget_range",
    "list_available_presets",
    "load_preset",
    "merge_config",
    "validate_config",
]
