"""
Policy Engine module for GL-FOUND-X-003 Unit & Reference Normalizer.

This module provides the complete policy engine implementation for
controlling normalization behavior based on organizational policies,
compliance profiles, and request-level overrides.

Key Components:
    - PolicyEngine: Core evaluation engine
    - Policy: Policy configuration model
    - PolicyDecision: Evaluation result model
    - ComplianceProfile: Supported regulatory profiles
    - PolicyMode: STRICT and LENIENT operating modes

Operating Modes:
    - STRICT: Fail on any missing required context; no defaults applied
      silently. Use for production compliance scenarios.
    - LENIENT: Apply defaults with warnings when context is missing.
      Use for exploratory or legacy data processing.

Compliance Profiles:
    - GHG_PROTOCOL: GHG Protocol Corporate Standard (Scope 1, 2, 3)
    - EU_CSRD: EU Corporate Sustainability Reporting Directive (ESRS E1)
    - IFRS_S2: IFRS Sustainability Disclosure Standard S2 (Climate)
    - EU_TAXONOMY: EU Taxonomy Regulation for sustainable activities
    - INDIA_BRSR: Business Responsibility and Sustainability Reporting
    - CALIFORNIA_SB253: California Climate Corporate Data Accountability Act
    - US_SEC: US SEC Climate Disclosure Rules

Example:
    >>> from gl_normalizer_core.policy import PolicyEngine, PolicyMode
    >>> engine = PolicyEngine(default_mode=PolicyMode.STRICT)
    >>> request = {"source_unit": "kg", "target_unit": "t"}
    >>> context = {"org_id": "org-acme", "gwp_version": "AR5"}
    >>> decision = engine.evaluate(request, context)
    >>> if decision.allowed:
    ...     print(f"GWP Version: {decision.effective_config.gwp_version}")

Quick Usage:
    >>> from gl_normalizer_core.policy import evaluate
    >>> decision = evaluate(request, context)

Loading Policies:
    >>> from gl_normalizer_core.policy import load_org_policy, merge_policies
    >>> org_policy = load_org_policy("org-acme")
    >>> merged = merge_policies(base_policy, org_policy)

Compliance Validation:
    >>> from gl_normalizer_core.policy import (
    ...     validate_against_profile,
    ...     ComplianceProfile,
    ... )
    >>> result = validate_against_profile(context, ComplianceProfile.GHG_PROTOCOL)
    >>> if not result.passed:
    ...     print(result.errors)
"""

# Core engine
from gl_normalizer_core.policy.engine import (
    PolicyEngine,
    get_default_engine,
    set_default_engine,
    evaluate,
)

# Models
from gl_normalizer_core.policy.models import (
    Policy,
    PolicyMode,
    PolicyDecision,
    PolicyDefaults,
    PolicyOverrides,
    PolicyWarning,
    AppliedDefault,
    EffectiveConfig,
    ReferenceConditions,
    ComplianceProfile,
    ConversionPolicy,
)

# Defaults
from gl_normalizer_core.policy.defaults import (
    DEFAULT_GWP_VERSION,
    DEFAULT_BASIS,
    DEFAULT_TEMPERATURE_REF,
    DEFAULT_PRESSURE_REF,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_PRECISION_DIGITS,
    REFERENCE_CONDITIONS_ISO,
    REFERENCE_CONDITIONS_US,
    REFERENCE_CONDITIONS_NTP,
    REFERENCE_CONDITIONS_STP,
    REFERENCE_CONDITIONS_PRESETS,
    PROFILE_DEFAULTS,
    get_system_defaults,
    get_profile_defaults,
    get_org_defaults,
    clear_org_defaults_cache,
    merge_defaults,
    get_gwp_value,
    GWP_VALUES,
)

# Compliance
from gl_normalizer_core.policy.compliance import (
    ComplianceRule,
    RuleSeverity,
    RuleCategory,
    ValidationResult,
    ProfileValidationResult,
    GHG_PROTOCOL_RULES,
    EU_CSRD_RULES,
    IFRS_S2_RULES,
    EU_TAXONOMY_RULES,
    INDIA_BRSR_RULES,
    CALIFORNIA_SB253_RULES,
    US_SEC_RULES,
    PROFILE_RULES,
    get_profile_rules,
    get_all_rules,
    get_rules_by_category,
    validate_rule,
    validate_against_profile,
    validate_against_profiles,
    get_required_fields_for_profile,
    create_warnings_from_validation,
)

# Loader
from gl_normalizer_core.policy.loader import (
    set_cache_ttl,
    clear_policy_cache,
    load_policy_from_file,
    load_policy_from_dict,
    load_org_policy,
    load_compliance_profile,
    load_default_policy,
    merge_policies,
    merge_policy_chain,
    validate_policy,
)


__all__ = [
    # Engine
    "PolicyEngine",
    "get_default_engine",
    "set_default_engine",
    "evaluate",
    # Models
    "Policy",
    "PolicyMode",
    "PolicyDecision",
    "PolicyDefaults",
    "PolicyOverrides",
    "PolicyWarning",
    "AppliedDefault",
    "EffectiveConfig",
    "ReferenceConditions",
    "ComplianceProfile",
    "ConversionPolicy",
    # Defaults
    "DEFAULT_GWP_VERSION",
    "DEFAULT_BASIS",
    "DEFAULT_TEMPERATURE_REF",
    "DEFAULT_PRESSURE_REF",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_PRECISION_DIGITS",
    "REFERENCE_CONDITIONS_ISO",
    "REFERENCE_CONDITIONS_US",
    "REFERENCE_CONDITIONS_NTP",
    "REFERENCE_CONDITIONS_STP",
    "REFERENCE_CONDITIONS_PRESETS",
    "PROFILE_DEFAULTS",
    "get_system_defaults",
    "get_profile_defaults",
    "get_org_defaults",
    "clear_org_defaults_cache",
    "merge_defaults",
    "get_gwp_value",
    "GWP_VALUES",
    # Compliance
    "ComplianceRule",
    "RuleSeverity",
    "RuleCategory",
    "ValidationResult",
    "ProfileValidationResult",
    "GHG_PROTOCOL_RULES",
    "EU_CSRD_RULES",
    "IFRS_S2_RULES",
    "EU_TAXONOMY_RULES",
    "INDIA_BRSR_RULES",
    "CALIFORNIA_SB253_RULES",
    "US_SEC_RULES",
    "PROFILE_RULES",
    "get_profile_rules",
    "get_all_rules",
    "get_rules_by_category",
    "validate_rule",
    "validate_against_profile",
    "validate_against_profiles",
    "get_required_fields_for_profile",
    "create_warnings_from_validation",
    # Loader
    "set_cache_ttl",
    "clear_policy_cache",
    "load_policy_from_file",
    "load_policy_from_dict",
    "load_org_policy",
    "load_compliance_profile",
    "load_default_policy",
    "merge_policies",
    "merge_policy_chain",
    "validate_policy",
]
