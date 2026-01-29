"""
Compliance profile rules for GL-FOUND-X-003 Unit & Reference Normalizer.

This module defines the validation rules and requirements for each
compliance profile. Rules are applied during policy evaluation to
ensure conversions meet regulatory requirements.

Key Design Principles:
    - Deterministic rule evaluation for reproducibility
    - Clear separation between profile rules and generic validation
    - Comprehensive documentation of regulatory requirements
    - Extensible rule structure for new regulations

Example:
    >>> from gl_normalizer_core.policy.compliance import (
    ...     get_profile_rules,
    ...     validate_against_profile,
    ...     ComplianceProfile,
    ... )
    >>> rules = get_profile_rules(ComplianceProfile.GHG_PROTOCOL)
    >>> result = validate_against_profile(context, ComplianceProfile.GHG_PROTOCOL)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum

import structlog

from gl_normalizer_core.policy.models import (
    ComplianceProfile,
    PolicyWarning,
    EffectiveConfig,
)
from gl_normalizer_core.errors.codes import GLNORMErrorCode

logger = structlog.get_logger(__name__)


# =============================================================================
# Rule Data Structures
# =============================================================================


class RuleSeverity(str, Enum):
    """Severity level for compliance rule violations."""

    ERROR = "error"  # Blocks processing
    WARNING = "warning"  # Allows processing with warning
    INFO = "info"  # Informational only


class RuleCategory(str, Enum):
    """Category of compliance rule."""

    GWP = "gwp"  # GWP version requirements
    BASIS = "basis"  # Energy basis requirements
    REFERENCE = "reference"  # Reference condition requirements
    PRECISION = "precision"  # Precision requirements
    UNIT = "unit"  # Unit validation requirements
    ENTITY = "entity"  # Entity resolution requirements
    AUDIT = "audit"  # Audit trail requirements


@dataclass
class ComplianceRule:
    """
    Definition of a compliance rule.

    Attributes:
        rule_id: Unique identifier for the rule.
        profile: Compliance profile this rule belongs to.
        category: Rule category.
        severity: Severity of violations.
        description: Human-readable description.
        error_code: GLNORM error code for violations.
        validator: Function that validates the rule.
        hint: Suggestion for fixing violations.
    """

    rule_id: str
    profile: ComplianceProfile
    category: RuleCategory
    severity: RuleSeverity
    description: str
    error_code: GLNORMErrorCode
    validator: Callable[[Dict[str, Any]], bool]
    hint: Optional[str] = None


@dataclass
class ValidationResult:
    """
    Result of validating against a compliance rule.

    Attributes:
        rule_id: ID of the rule that was validated.
        passed: Whether validation passed.
        severity: Severity of any violation.
        message: Validation message.
        error_code: Error code if failed.
        hint: Hint for resolution if failed.
    """

    rule_id: str
    passed: bool
    severity: RuleSeverity
    message: str
    error_code: Optional[GLNORMErrorCode] = None
    hint: Optional[str] = None


@dataclass
class ProfileValidationResult:
    """
    Complete result of validating against a compliance profile.

    Attributes:
        profile: The compliance profile validated against.
        passed: Whether all required rules passed.
        results: Individual rule validation results.
        errors: List of error-level violations.
        warnings: List of warning-level violations.
    """

    profile: ComplianceProfile
    passed: bool
    results: List[ValidationResult]
    errors: List[ValidationResult]
    warnings: List[ValidationResult]


# =============================================================================
# GHG Protocol Rules
# =============================================================================

def _validate_gwp_version_ghg(context: Dict[str, Any]) -> bool:
    """Validate GWP version for GHG Protocol."""
    gwp_version = context.get("gwp_version", "")
    # GHG Protocol accepts AR4, AR5, AR6 (with preference for AR5)
    return gwp_version in {"AR4", "AR5", "AR6"}


def _validate_basis_ghg(context: Dict[str, Any]) -> bool:
    """Validate energy basis for GHG Protocol."""
    basis = context.get("basis", "")
    # GHG Protocol requires LHV for fuel heating values
    return basis.upper() == "LHV"


def _validate_scope_classification(context: Dict[str, Any]) -> bool:
    """Validate scope classification is provided."""
    return "scope" in context and context["scope"] in {1, 2, 3, "1", "2", "3"}


GHG_PROTOCOL_RULES: List[ComplianceRule] = [
    ComplianceRule(
        rule_id="GHG-001",
        profile=ComplianceProfile.GHG_PROTOCOL,
        category=RuleCategory.GWP,
        severity=RuleSeverity.ERROR,
        description="GWP version must be AR4, AR5, or AR6 for GHG Protocol compliance",
        error_code=GLNORMErrorCode.E305_GWP_VERSION_MISSING,
        validator=_validate_gwp_version_ghg,
        hint="Specify gwp_version as 'AR5' (recommended) or 'AR4'/'AR6'",
    ),
    ComplianceRule(
        rule_id="GHG-002",
        profile=ComplianceProfile.GHG_PROTOCOL,
        category=RuleCategory.BASIS,
        severity=RuleSeverity.WARNING,
        description="GHG Protocol recommends LHV basis for fuel heating values",
        error_code=GLNORMErrorCode.E306_BASIS_MISSING,
        validator=_validate_basis_ghg,
        hint="Use 'LHV' (Lower Heating Value) for GHG Protocol compliance",
    ),
    ComplianceRule(
        rule_id="GHG-003",
        profile=ComplianceProfile.GHG_PROTOCOL,
        category=RuleCategory.AUDIT,
        severity=RuleSeverity.INFO,
        description="Scope classification should be provided for emissions",
        error_code=GLNORMErrorCode.E301_MISSING_REFERENCE_CONDITIONS,
        validator=_validate_scope_classification,
        hint="Specify 'scope' as 1, 2, or 3",
    ),
]


# =============================================================================
# EU CSRD / ESRS E1 Rules
# =============================================================================

def _validate_gwp_version_csrd(context: Dict[str, Any]) -> bool:
    """Validate GWP version for EU CSRD."""
    gwp_version = context.get("gwp_version", "")
    # CSRD requires AR5 or later per ESRS E1
    return gwp_version in {"AR5", "AR6"}


def _validate_activity_data_source(context: Dict[str, Any]) -> bool:
    """Validate activity data source is documented."""
    return "data_source" in context and context["data_source"] is not None


def _validate_emission_factor_source(context: Dict[str, Any]) -> bool:
    """Validate emission factor source is documented."""
    return "ef_source" in context and context["ef_source"] is not None


EU_CSRD_RULES: List[ComplianceRule] = [
    ComplianceRule(
        rule_id="CSRD-001",
        profile=ComplianceProfile.EU_CSRD,
        category=RuleCategory.GWP,
        severity=RuleSeverity.ERROR,
        description="EU CSRD (ESRS E1) requires AR5 or AR6 GWP values",
        error_code=GLNORMErrorCode.E305_GWP_VERSION_MISSING,
        validator=_validate_gwp_version_csrd,
        hint="Use 'AR5' or 'AR6' GWP version for EU CSRD compliance",
    ),
    ComplianceRule(
        rule_id="CSRD-002",
        profile=ComplianceProfile.EU_CSRD,
        category=RuleCategory.AUDIT,
        severity=RuleSeverity.WARNING,
        description="Activity data source should be documented per ESRS E1",
        error_code=GLNORMErrorCode.E600_AUDIT_WRITE_FAILED,
        validator=_validate_activity_data_source,
        hint="Specify 'data_source' to document the source of activity data",
    ),
    ComplianceRule(
        rule_id="CSRD-003",
        profile=ComplianceProfile.EU_CSRD,
        category=RuleCategory.AUDIT,
        severity=RuleSeverity.WARNING,
        description="Emission factor source should be documented per ESRS E1",
        error_code=GLNORMErrorCode.E600_AUDIT_WRITE_FAILED,
        validator=_validate_emission_factor_source,
        hint="Specify 'ef_source' to document the emission factor source",
    ),
]


# =============================================================================
# IFRS S2 Rules
# =============================================================================

def _validate_gwp_version_ifrs(context: Dict[str, Any]) -> bool:
    """Validate GWP version for IFRS S2."""
    gwp_version = context.get("gwp_version", "")
    # IFRS S2 requires AR5 or AR6
    return gwp_version in {"AR5", "AR6"}


def _validate_materiality_assessment(context: Dict[str, Any]) -> bool:
    """Validate materiality assessment is documented."""
    return "materiality_assessed" in context


IFRS_S2_RULES: List[ComplianceRule] = [
    ComplianceRule(
        rule_id="IFRS-001",
        profile=ComplianceProfile.IFRS_S2,
        category=RuleCategory.GWP,
        severity=RuleSeverity.ERROR,
        description="IFRS S2 requires AR5 or AR6 GWP values per GHG Protocol",
        error_code=GLNORMErrorCode.E305_GWP_VERSION_MISSING,
        validator=_validate_gwp_version_ifrs,
        hint="Use 'AR5' or 'AR6' GWP version for IFRS S2 compliance",
    ),
    ComplianceRule(
        rule_id="IFRS-002",
        profile=ComplianceProfile.IFRS_S2,
        category=RuleCategory.AUDIT,
        severity=RuleSeverity.INFO,
        description="IFRS S2 recommends documenting materiality assessment",
        error_code=GLNORMErrorCode.E600_AUDIT_WRITE_FAILED,
        validator=_validate_materiality_assessment,
        hint="Include 'materiality_assessed' flag in context",
    ),
]


# =============================================================================
# EU Taxonomy Rules
# =============================================================================

def _validate_taxonomy_alignment(context: Dict[str, Any]) -> bool:
    """Validate taxonomy alignment is assessed."""
    return "taxonomy_eligible" in context or "taxonomy_aligned" in context


def _validate_dnsh_assessment(context: Dict[str, Any]) -> bool:
    """Validate Do No Significant Harm assessment is documented."""
    return "dnsh_assessed" in context


EU_TAXONOMY_RULES: List[ComplianceRule] = [
    ComplianceRule(
        rule_id="TAX-001",
        profile=ComplianceProfile.EU_TAXONOMY,
        category=RuleCategory.GWP,
        severity=RuleSeverity.ERROR,
        description="EU Taxonomy requires AR5 or AR6 GWP values",
        error_code=GLNORMErrorCode.E305_GWP_VERSION_MISSING,
        validator=_validate_gwp_version_csrd,  # Reuse CSRD validator
        hint="Use 'AR5' or 'AR6' GWP version for EU Taxonomy compliance",
    ),
    ComplianceRule(
        rule_id="TAX-002",
        profile=ComplianceProfile.EU_TAXONOMY,
        category=RuleCategory.AUDIT,
        severity=RuleSeverity.WARNING,
        description="Taxonomy eligibility/alignment should be assessed",
        error_code=GLNORMErrorCode.E600_AUDIT_WRITE_FAILED,
        validator=_validate_taxonomy_alignment,
        hint="Include 'taxonomy_eligible' or 'taxonomy_aligned' in context",
    ),
    ComplianceRule(
        rule_id="TAX-003",
        profile=ComplianceProfile.EU_TAXONOMY,
        category=RuleCategory.AUDIT,
        severity=RuleSeverity.WARNING,
        description="DNSH assessment should be documented for aligned activities",
        error_code=GLNORMErrorCode.E600_AUDIT_WRITE_FAILED,
        validator=_validate_dnsh_assessment,
        hint="Include 'dnsh_assessed' flag in context",
    ),
]


# =============================================================================
# India BRSR Rules
# =============================================================================

def _validate_gwp_version_brsr(context: Dict[str, Any]) -> bool:
    """Validate GWP version for India BRSR."""
    gwp_version = context.get("gwp_version", "")
    # BRSR accepts AR4, AR5, AR6
    return gwp_version in {"AR4", "AR5", "AR6"}


INDIA_BRSR_RULES: List[ComplianceRule] = [
    ComplianceRule(
        rule_id="BRSR-001",
        profile=ComplianceProfile.INDIA_BRSR,
        category=RuleCategory.GWP,
        severity=RuleSeverity.ERROR,
        description="India BRSR requires GWP values from AR4, AR5, or AR6",
        error_code=GLNORMErrorCode.E305_GWP_VERSION_MISSING,
        validator=_validate_gwp_version_brsr,
        hint="Use 'AR4', 'AR5', or 'AR6' GWP version for BRSR compliance",
    ),
]


# =============================================================================
# California SB 253 Rules
# =============================================================================

def _validate_gwp_version_sb253(context: Dict[str, Any]) -> bool:
    """Validate GWP version for California SB 253."""
    gwp_version = context.get("gwp_version", "")
    # SB 253 aligns with GHG Protocol, accepts AR5 or AR6
    return gwp_version in {"AR5", "AR6"}


def _validate_scope3_categories(context: Dict[str, Any]) -> bool:
    """Validate Scope 3 categories are specified."""
    scope = context.get("scope")
    if scope in {3, "3"}:
        return "scope3_categories" in context and len(context.get("scope3_categories", [])) > 0
    return True


CALIFORNIA_SB253_RULES: List[ComplianceRule] = [
    ComplianceRule(
        rule_id="SB253-001",
        profile=ComplianceProfile.CALIFORNIA_SB253,
        category=RuleCategory.GWP,
        severity=RuleSeverity.ERROR,
        description="California SB 253 requires AR5 or AR6 GWP values",
        error_code=GLNORMErrorCode.E305_GWP_VERSION_MISSING,
        validator=_validate_gwp_version_sb253,
        hint="Use 'AR5' or 'AR6' GWP version for SB 253 compliance",
    ),
    ComplianceRule(
        rule_id="SB253-002",
        profile=ComplianceProfile.CALIFORNIA_SB253,
        category=RuleCategory.AUDIT,
        severity=RuleSeverity.WARNING,
        description="Scope 3 emissions should specify relevant categories",
        error_code=GLNORMErrorCode.E600_AUDIT_WRITE_FAILED,
        validator=_validate_scope3_categories,
        hint="Specify 'scope3_categories' list for Scope 3 emissions",
    ),
]


# =============================================================================
# US SEC Climate Disclosure Rules
# =============================================================================

def _validate_gwp_version_sec(context: Dict[str, Any]) -> bool:
    """Validate GWP version for US SEC rules."""
    gwp_version = context.get("gwp_version", "")
    # SEC aligns with GHG Protocol
    return gwp_version in {"AR5", "AR6"}


def _validate_assurance_level(context: Dict[str, Any]) -> bool:
    """Validate assurance level is documented."""
    return "assurance_level" in context


US_SEC_RULES: List[ComplianceRule] = [
    ComplianceRule(
        rule_id="SEC-001",
        profile=ComplianceProfile.US_SEC,
        category=RuleCategory.GWP,
        severity=RuleSeverity.ERROR,
        description="US SEC Climate Disclosure requires AR5 or AR6 GWP values",
        error_code=GLNORMErrorCode.E305_GWP_VERSION_MISSING,
        validator=_validate_gwp_version_sec,
        hint="Use 'AR5' or 'AR6' GWP version for SEC compliance",
    ),
    ComplianceRule(
        rule_id="SEC-002",
        profile=ComplianceProfile.US_SEC,
        category=RuleCategory.AUDIT,
        severity=RuleSeverity.INFO,
        description="Assurance level should be documented for SEC filings",
        error_code=GLNORMErrorCode.E600_AUDIT_WRITE_FAILED,
        validator=_validate_assurance_level,
        hint="Specify 'assurance_level' (e.g., 'limited', 'reasonable')",
    ),
]


# =============================================================================
# Rule Registry
# =============================================================================

# All rules indexed by profile
PROFILE_RULES: Dict[ComplianceProfile, List[ComplianceRule]] = {
    ComplianceProfile.GHG_PROTOCOL: GHG_PROTOCOL_RULES,
    ComplianceProfile.EU_CSRD: EU_CSRD_RULES,
    ComplianceProfile.IFRS_S2: IFRS_S2_RULES,
    ComplianceProfile.EU_TAXONOMY: EU_TAXONOMY_RULES,
    ComplianceProfile.INDIA_BRSR: INDIA_BRSR_RULES,
    ComplianceProfile.CALIFORNIA_SB253: CALIFORNIA_SB253_RULES,
    ComplianceProfile.US_SEC: US_SEC_RULES,
}


def get_profile_rules(profile: ComplianceProfile) -> List[ComplianceRule]:
    """
    Get all compliance rules for a profile.

    Args:
        profile: The compliance profile.

    Returns:
        List of ComplianceRule for the profile.

    Example:
        >>> rules = get_profile_rules(ComplianceProfile.GHG_PROTOCOL)
        >>> print(len(rules))
        3
    """
    return PROFILE_RULES.get(profile, [])


def get_all_rules() -> List[ComplianceRule]:
    """
    Get all compliance rules across all profiles.

    Returns:
        List of all ComplianceRule.
    """
    all_rules = []
    for rules in PROFILE_RULES.values():
        all_rules.extend(rules)
    return all_rules


def get_rules_by_category(category: RuleCategory) -> List[ComplianceRule]:
    """
    Get all rules in a specific category.

    Args:
        category: The rule category.

    Returns:
        List of ComplianceRule in the category.
    """
    return [rule for rule in get_all_rules() if rule.category == category]


# =============================================================================
# Validation Functions
# =============================================================================

def validate_rule(
    rule: ComplianceRule,
    context: Dict[str, Any],
) -> ValidationResult:
    """
    Validate a single compliance rule against context.

    Args:
        rule: The rule to validate.
        context: The context to validate against.

    Returns:
        ValidationResult indicating pass/fail.

    Example:
        >>> rule = GHG_PROTOCOL_RULES[0]
        >>> result = validate_rule(rule, {"gwp_version": "AR5"})
        >>> print(result.passed)
        True
    """
    try:
        passed = rule.validator(context)
        return ValidationResult(
            rule_id=rule.rule_id,
            passed=passed,
            severity=rule.severity,
            message=rule.description if not passed else f"Rule {rule.rule_id} passed",
            error_code=rule.error_code if not passed else None,
            hint=rule.hint if not passed else None,
        )
    except Exception as e:
        logger.error(
            "Rule validation failed",
            rule_id=rule.rule_id,
            error=str(e),
            exc_info=True,
        )
        return ValidationResult(
            rule_id=rule.rule_id,
            passed=False,
            severity=RuleSeverity.ERROR,
            message=f"Rule validation error: {str(e)}",
            error_code=GLNORMErrorCode.E904_INTERNAL_ERROR,
            hint="Check rule configuration and context",
        )


def validate_against_profile(
    context: Dict[str, Any],
    profile: ComplianceProfile,
) -> ProfileValidationResult:
    """
    Validate context against all rules in a compliance profile.

    Args:
        context: The context to validate.
        profile: The compliance profile to validate against.

    Returns:
        ProfileValidationResult with all rule results.

    Example:
        >>> context = {"gwp_version": "AR5", "basis": "LHV"}
        >>> result = validate_against_profile(context, ComplianceProfile.GHG_PROTOCOL)
        >>> print(result.passed)
        True
    """
    rules = get_profile_rules(profile)
    results: List[ValidationResult] = []
    errors: List[ValidationResult] = []
    warnings: List[ValidationResult] = []

    for rule in rules:
        result = validate_rule(rule, context)
        results.append(result)

        if not result.passed:
            if result.severity == RuleSeverity.ERROR:
                errors.append(result)
            elif result.severity == RuleSeverity.WARNING:
                warnings.append(result)

    # Profile passes if no error-level violations
    passed = len(errors) == 0

    logger.debug(
        "Profile validation complete",
        profile=profile.value,
        passed=passed,
        error_count=len(errors),
        warning_count=len(warnings),
    )

    return ProfileValidationResult(
        profile=profile,
        passed=passed,
        results=results,
        errors=errors,
        warnings=warnings,
    )


def validate_against_profiles(
    context: Dict[str, Any],
    profiles: List[ComplianceProfile],
) -> Dict[ComplianceProfile, ProfileValidationResult]:
    """
    Validate context against multiple compliance profiles.

    Args:
        context: The context to validate.
        profiles: List of compliance profiles to validate against.

    Returns:
        Dict mapping profiles to their validation results.

    Example:
        >>> profiles = [ComplianceProfile.GHG_PROTOCOL, ComplianceProfile.EU_CSRD]
        >>> results = validate_against_profiles(context, profiles)
    """
    return {
        profile: validate_against_profile(context, profile)
        for profile in profiles
    }


def get_required_fields_for_profile(profile: ComplianceProfile) -> Set[str]:
    """
    Get the set of fields required by error-level rules in a profile.

    Args:
        profile: The compliance profile.

    Returns:
        Set of field names that are required.

    Example:
        >>> fields = get_required_fields_for_profile(ComplianceProfile.GHG_PROTOCOL)
        >>> print("gwp_version" in fields)
        True
    """
    rules = get_profile_rules(profile)
    required_fields: Set[str] = set()

    for rule in rules:
        if rule.severity == RuleSeverity.ERROR:
            # Infer fields from category
            if rule.category == RuleCategory.GWP:
                required_fields.add("gwp_version")
            elif rule.category == RuleCategory.BASIS:
                required_fields.add("basis")
            elif rule.category == RuleCategory.REFERENCE:
                required_fields.add("reference_conditions")

    return required_fields


def create_warnings_from_validation(
    validation_result: ProfileValidationResult,
) -> List[PolicyWarning]:
    """
    Create PolicyWarning objects from validation results.

    Args:
        validation_result: The profile validation result.

    Returns:
        List of PolicyWarning for warnings and info-level results.

    Example:
        >>> result = validate_against_profile(context, profile)
        >>> warnings = create_warnings_from_validation(result)
    """
    warnings: List[PolicyWarning] = []

    for result in validation_result.results:
        if not result.passed and result.severity in {RuleSeverity.WARNING, RuleSeverity.INFO}:
            warnings.append(PolicyWarning(
                code=result.error_code.value if result.error_code else "POL_WARN_COMPLIANCE",
                message=result.message,
                field=None,
                severity="medium" if result.severity == RuleSeverity.WARNING else "low",
                hint=result.hint,
            ))

    return warnings


__all__ = [
    # Data structures
    "ComplianceRule",
    "RuleSeverity",
    "RuleCategory",
    "ValidationResult",
    "ProfileValidationResult",
    # Rule sets
    "GHG_PROTOCOL_RULES",
    "EU_CSRD_RULES",
    "IFRS_S2_RULES",
    "EU_TAXONOMY_RULES",
    "INDIA_BRSR_RULES",
    "CALIFORNIA_SB253_RULES",
    "US_SEC_RULES",
    "PROFILE_RULES",
    # Functions
    "get_profile_rules",
    "get_all_rules",
    "get_rules_by_category",
    "validate_rule",
    "validate_against_profile",
    "validate_against_profiles",
    "get_required_fields_for_profile",
    "create_warnings_from_validation",
]
