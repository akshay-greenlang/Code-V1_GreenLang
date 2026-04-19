"""
PACK-030 Net Zero Reporting Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the Net Zero Reporting Pack. Import from this module to access
the full configuration API for multi-framework report generation (SBTi, CDP,
TCFD, GRI, ISSB, SEC, CSRD), data aggregation, narrative generation,
XBRL/iXBRL tagging, assurance evidence packaging, dashboard creation,
format rendering, and cross-framework consistency validation.

Usage:
    >>> from packs.net_zero.PACK_030_net_zero_reporting.config import (
    ...     PackConfig,
    ...     NetZeroReportingConfig,
    ...     ReportingFramework,
    ...     OutputFormat,
    ...     load_preset,
    ... )
    >>> config = PackConfig.from_preset("multi_framework")
    >>> print(config.pack.frameworks.frameworks_enabled)
    ['SBTi', 'CDP', 'TCFD', 'GRI', 'ISSB', 'SEC', 'CSRD']
    >>> print(config.pack.output_formats)
    ['PDF', 'HTML', 'Excel', 'JSON', 'XBRL', 'iXBRL']

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (8 presets)
    3. Environment overrides (NZ_REPORTING_* prefix)
    4. Explicit runtime overrides

Available Presets:
    - csrd_focus: CSRD ESRS E1 Focus with digital taxonomy
    - cdp_alist: CDP A-List Target with completeness scoring
    - tcfd_investor: TCFD Investor-Grade with scenario analysis
    - sbti_validation: SBTi Validation Ready with evidence bundles
    - sec_10k: SEC 10-K Compliance with XBRL/iXBRL tagging
    - multi_framework: Multi-Framework Comprehensive (all 7 frameworks)
    - investor_relations: Investor Relations Package (TCFD+ISSB focus)
    - assurance_ready: Assurance-Ready Package (ISAE 3410 evidence)
"""

from .pack_config import (
    # Enums
    AssuranceLevel,
    BrandingStyle,
    ConsistencyStrictness,
    DataSourceRequirement,
    NarrativeQuality,
    OutputFormat,
    ReportingFramework,
    ReportStatus,
    StakeholderViewType,
    TranslationService,
    # Sub-config models
    AssuranceConfig,
    BrandingConfig,
    DashboardConfig,
    DataAggregationConfig,
    FrameworkConfig,
    FrameworkOutputConfig,
    NarrativeConfig,
    NotificationConfig,
    PerformanceConfig,
    TranslationConfig,
    ValidationConfig,
    XBRLConfig,
    # Main config models
    NetZeroReportingConfig,
    PackConfig,
    # Constants
    ASSURANCE_STANDARDS,
    BRANDING_STYLES,
    CONSISTENCY_RULE_CATEGORIES,
    DATA_SOURCE_APPS,
    DATA_SOURCE_PACKS,
    DEFAULT_API_RESPONSE_TIMEOUT_MS,
    DEFAULT_BASELINE_YEAR,
    DEFAULT_CACHE_HIT_RATIO_TARGET_PCT,
    DEFAULT_MAX_CONCURRENT_REPORTS,
    DEFAULT_NARRATIVE_CONSISTENCY_TARGET_PCT,
    DEFAULT_NET_ZERO_YEAR,
    DEFAULT_PDF_RENDER_TIMEOUT_SECONDS,
    DEFAULT_REPORT_GENERATION_TIMEOUT_SECONDS,
    DEFAULT_REPORTING_YEAR,
    DEFAULT_RETENTION_YEARS,
    DEFAULT_XBRL_RENDER_TIMEOUT_SECONDS,
    EVIDENCE_BUNDLE_COMPONENTS,
    FRAMEWORK_DEADLINE_TEMPLATES,
    OUTPUT_FORMAT_SPECS,
    STAKEHOLDER_VIEW_TYPES,
    SUPPORTED_FRAMEWORKS,
    SUPPORTED_LANGUAGES,
    SUPPORTED_PRESETS,
    XBRL_TAXONOMY_SPECS,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_assurance_standard_info,
    get_env_overrides,
    get_evidence_bundle_info,
    get_framework_info,
    get_output_format_info,
    get_stakeholder_view_info,
    get_xbrl_taxonomy_info,
    list_available_presets,
    list_branding_styles,
    list_consistency_rules,
    list_evidence_bundle_components,
    list_output_formats,
    list_stakeholder_views,
    list_supported_frameworks,
    list_supported_languages,
    load_config,
    load_preset,
    merge_config,
    validate_config,
)

__all__ = [
    # Enums
    "AssuranceLevel",
    "BrandingStyle",
    "ConsistencyStrictness",
    "DataSourceRequirement",
    "NarrativeQuality",
    "OutputFormat",
    "ReportingFramework",
    "ReportStatus",
    "StakeholderViewType",
    "TranslationService",
    # Sub-config models
    "AssuranceConfig",
    "BrandingConfig",
    "DashboardConfig",
    "DataAggregationConfig",
    "FrameworkConfig",
    "FrameworkOutputConfig",
    "NarrativeConfig",
    "NotificationConfig",
    "PerformanceConfig",
    "TranslationConfig",
    "ValidationConfig",
    "XBRLConfig",
    # Main config models
    "NetZeroReportingConfig",
    "PackConfig",
    # Constants
    "ASSURANCE_STANDARDS",
    "BRANDING_STYLES",
    "CONSISTENCY_RULE_CATEGORIES",
    "DATA_SOURCE_APPS",
    "DATA_SOURCE_PACKS",
    "DEFAULT_API_RESPONSE_TIMEOUT_MS",
    "DEFAULT_BASELINE_YEAR",
    "DEFAULT_CACHE_HIT_RATIO_TARGET_PCT",
    "DEFAULT_MAX_CONCURRENT_REPORTS",
    "DEFAULT_NARRATIVE_CONSISTENCY_TARGET_PCT",
    "DEFAULT_NET_ZERO_YEAR",
    "DEFAULT_PDF_RENDER_TIMEOUT_SECONDS",
    "DEFAULT_REPORT_GENERATION_TIMEOUT_SECONDS",
    "DEFAULT_REPORTING_YEAR",
    "DEFAULT_RETENTION_YEARS",
    "DEFAULT_XBRL_RENDER_TIMEOUT_SECONDS",
    "EVIDENCE_BUNDLE_COMPONENTS",
    "FRAMEWORK_DEADLINE_TEMPLATES",
    "OUTPUT_FORMAT_SPECS",
    "STAKEHOLDER_VIEW_TYPES",
    "SUPPORTED_FRAMEWORKS",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_PRESETS",
    "XBRL_TAXONOMY_SPECS",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_assurance_standard_info",
    "get_env_overrides",
    "get_evidence_bundle_info",
    "get_framework_info",
    "get_output_format_info",
    "get_stakeholder_view_info",
    "get_xbrl_taxonomy_info",
    "list_available_presets",
    "list_branding_styles",
    "list_consistency_rules",
    "list_evidence_bundle_components",
    "list_output_formats",
    "list_stakeholder_views",
    "list_supported_frameworks",
    "list_supported_languages",
    "load_config",
    "load_preset",
    "merge_config",
    "validate_config",
]
