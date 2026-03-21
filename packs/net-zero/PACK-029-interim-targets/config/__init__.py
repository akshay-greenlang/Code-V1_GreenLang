"""
PACK-029 Interim Targets Pack - Configuration Module

This module exports all configuration classes, enums, constants, and utility
functions for the Interim Targets Pack. Import from this module to access
the full configuration API for interim target setting, progress monitoring,
variance analysis, corrective action planning, carbon budget allocation,
trend extrapolation, SBTi validation, and multi-framework reporting.

Usage:
    >>> from packs.net_zero.PACK_029_interim_targets.config import (
    ...     PackConfig,
    ...     InterimTargetsConfig,
    ...     SBTiPathwayLevel,
    ...     PathwayType,
    ...     get_pathway_defaults,
    ...     load_preset,
    ... )
    >>> config = PackConfig.from_preset("sbti_1_5c_pathway")
    >>> print(config.pack.sbti_pathway)
    SBTiPathwayLevel.CELSIUS_1_5
    >>> print(config.pack.interim_target_5yr.min_reduction_pct)
    42.0

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (7 presets)
    3. Environment overrides (INTERIM_TARGETS_* prefix)
    4. Explicit runtime overrides

Available Presets:
    - sbti_1_5c_pathway: SBTi 1.5C-aligned with 42% near-term reduction
    - sbti_wb2c_pathway: SBTi Well-Below 2C with 30% near-term reduction
    - quarterly_monitoring: Frequent monitoring with real-time alerting
    - annual_review: Comprehensive annual review with dual variance analysis
    - corrective_action: Proactive corrective action with MACC integration
    - sector_specific: Sector pathway integration with PACK-028
    - scope_3_extended: Extended Scope 3 timeline per SBTi guidance
"""

from .pack_config import (
    # Enums
    AssuranceLevel,
    BudgetAllocationMethod,
    CorrectiveActionTrigger,
    ExtrapolationMethod,
    MonitoringFrequency,
    PathwayType,
    PerformanceScore,
    ReportingFrequency,
    SBTiPathwayLevel,
    VarianceMethod,
    # Sub-config models
    AlertingConfig,
    CarbonBudgetConfig,
    CorrectiveActionConfig,
    ExtrapolationConfig,
    InterimTargetConfig,
    MonitoringConfig,
    PerformanceConfig,
    ReportingConfig,
    SBTiValidationConfig,
    ScopeConfig,
    VarianceAnalysisConfig,
    # Main config models
    InterimTargetsConfig,
    PackConfig,
    # Constants
    BUDGET_ALLOCATION_PROFILES,
    CORRECTIVE_ACTION_PRIORITIES,
    DEFAULT_5YR_TARGET_YEAR,
    DEFAULT_10YR_TARGET_YEAR,
    DEFAULT_BASELINE_YEAR,
    DEFAULT_CONFIDENCE_INTERVAL,
    DEFAULT_CURRENT_YEAR,
    DEFAULT_FORECAST_HORIZON_YEARS,
    DEFAULT_LONG_TERM_REDUCTION_PCT,
    DEFAULT_LONG_TERM_TARGET_YEAR,
    DEFAULT_NET_ZERO_YEAR,
    DEFAULT_REPORTING_YEAR,
    DEFAULT_RETENTION_YEARS,
    DEFAULT_SBTI_CRITERIA_COUNT,
    DEFAULT_SCOPE_3_LAG_YEARS,
    DEFAULT_VARIANCE_TOLERANCE_PCT,
    EXTRAPOLATION_MODEL_PARAMS,
    PERFORMANCE_SCORES,
    REPORTING_FRAMEWORK_MAPPING,
    ROOT_CAUSE_CATEGORIES,
    SBTI_COVERAGE_THRESHOLDS,
    SBTI_NEAR_TERM_MINIMUMS,
    SBTI_VALIDATION_CRITERIA,
    SUPPORTED_PRESETS,
    VARIANCE_DECOMPOSITION_LEVELS,
    # Directories
    CONFIG_DIR,
    PACK_BASE_DIR,
    # Utility functions
    get_budget_allocation_info,
    get_corrective_action_priority,
    get_env_overrides,
    get_extrapolation_model_info,
    get_pathway_defaults,
    get_performance_score_info,
    get_reporting_framework_info,
    get_root_cause_categories,
    get_sbti_criteria,
    get_sbti_minimums,
    list_available_presets,
    list_extrapolation_methods,
    list_pathway_types,
    list_reporting_frameworks,
    list_sbti_pathways,
    list_variance_methods,
    load_config,
    load_preset,
    merge_config,
    validate_config,
)

__all__ = [
    # Enums
    "AssuranceLevel",
    "BudgetAllocationMethod",
    "CorrectiveActionTrigger",
    "ExtrapolationMethod",
    "MonitoringFrequency",
    "PathwayType",
    "PerformanceScore",
    "ReportingFrequency",
    "SBTiPathwayLevel",
    "VarianceMethod",
    # Sub-config models
    "AlertingConfig",
    "CarbonBudgetConfig",
    "CorrectiveActionConfig",
    "ExtrapolationConfig",
    "InterimTargetConfig",
    "MonitoringConfig",
    "PerformanceConfig",
    "ReportingConfig",
    "SBTiValidationConfig",
    "ScopeConfig",
    "VarianceAnalysisConfig",
    # Main config models
    "InterimTargetsConfig",
    "PackConfig",
    # Constants
    "BUDGET_ALLOCATION_PROFILES",
    "CORRECTIVE_ACTION_PRIORITIES",
    "DEFAULT_5YR_TARGET_YEAR",
    "DEFAULT_10YR_TARGET_YEAR",
    "DEFAULT_BASELINE_YEAR",
    "DEFAULT_CONFIDENCE_INTERVAL",
    "DEFAULT_CURRENT_YEAR",
    "DEFAULT_FORECAST_HORIZON_YEARS",
    "DEFAULT_LONG_TERM_REDUCTION_PCT",
    "DEFAULT_LONG_TERM_TARGET_YEAR",
    "DEFAULT_NET_ZERO_YEAR",
    "DEFAULT_REPORTING_YEAR",
    "DEFAULT_RETENTION_YEARS",
    "DEFAULT_SBTI_CRITERIA_COUNT",
    "DEFAULT_SCOPE_3_LAG_YEARS",
    "DEFAULT_VARIANCE_TOLERANCE_PCT",
    "EXTRAPOLATION_MODEL_PARAMS",
    "PERFORMANCE_SCORES",
    "REPORTING_FRAMEWORK_MAPPING",
    "ROOT_CAUSE_CATEGORIES",
    "SBTI_COVERAGE_THRESHOLDS",
    "SBTI_NEAR_TERM_MINIMUMS",
    "SBTI_VALIDATION_CRITERIA",
    "SUPPORTED_PRESETS",
    "VARIANCE_DECOMPOSITION_LEVELS",
    # Directories
    "CONFIG_DIR",
    "PACK_BASE_DIR",
    # Utility functions
    "get_budget_allocation_info",
    "get_corrective_action_priority",
    "get_env_overrides",
    "get_extrapolation_model_info",
    "get_pathway_defaults",
    "get_performance_score_info",
    "get_reporting_framework_info",
    "get_root_cause_categories",
    "get_sbti_criteria",
    "get_sbti_minimums",
    "list_available_presets",
    "list_extrapolation_methods",
    "list_pathway_types",
    "list_reporting_frameworks",
    "list_sbti_pathways",
    "list_variance_methods",
    "load_config",
    "load_preset",
    "merge_config",
    "validate_config",
]
