"""
Policy data models for GL-FOUND-X-003 Unit & Reference Normalizer.

This module defines the Pydantic models for the Policy Engine, including
policy configurations, decisions, and compliance profiles. These models
ensure type safety and validation for all policy-related operations.

Key Design Principles:
    - Immutable policy records for audit compliance
    - Complete typing with Pydantic validation
    - Deterministic serialization for reproducibility
    - Clear separation between input policies and output decisions

Example:
    >>> from gl_normalizer_core.policy.models import Policy, PolicyMode, ComplianceProfile
    >>> policy = Policy(
    ...     mode=PolicyMode.STRICT,
    ...     compliance_profiles=[ComplianceProfile.GHG_PROTOCOL],
    ...     defaults={"gwp_version": "AR5", "basis": "LHV"},
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, field_validator, model_validator
import hashlib


class PolicyMode(str, Enum):
    """
    Operating mode for the Policy Engine.

    The mode determines how the engine handles missing or ambiguous context:
    - STRICT: Fail on any missing required context; no defaults applied silently
    - LENIENT: Apply defaults with warnings when context is missing

    Attributes:
        STRICT: Fail-fast mode for production compliance scenarios.
        LENIENT: Permissive mode for exploratory or legacy data processing.
    """

    STRICT = "STRICT"
    LENIENT = "LENIENT"


class ComplianceProfile(str, Enum):
    """
    Supported regulatory compliance profiles.

    Each profile defines specific requirements for GWP versions,
    reporting formats, calculation methodologies, and validation rules.

    Attributes:
        GHG_PROTOCOL: GHG Protocol Corporate Standard (Scope 1, 2, 3).
        EU_CSRD: EU Corporate Sustainability Reporting Directive (ESRS E1).
        IFRS_S2: IFRS Sustainability Disclosure Standard S2 (Climate).
        EU_TAXONOMY: EU Taxonomy Regulation for sustainable activities.
        INDIA_BRSR: Business Responsibility and Sustainability Reporting (India).
        CALIFORNIA_SB253: California Climate Corporate Data Accountability Act.
        US_SEC: US SEC Climate Disclosure Rules.

    Example:
        >>> profile = ComplianceProfile.GHG_PROTOCOL
        >>> print(profile.value)
        'GHG_PROTOCOL'
    """

    GHG_PROTOCOL = "GHG_PROTOCOL"
    EU_CSRD = "EU_CSRD"
    IFRS_S2 = "IFRS_S2"
    EU_TAXONOMY = "EU_TAXONOMY"
    INDIA_BRSR = "INDIA_BRSR"
    CALIFORNIA_SB253 = "CALIFORNIA_SB253"
    US_SEC = "US_SEC"

    @property
    def display_name(self) -> str:
        """Get human-readable display name for the profile."""
        display_names = {
            "GHG_PROTOCOL": "GHG Protocol Corporate Standard",
            "EU_CSRD": "EU Corporate Sustainability Reporting Directive",
            "IFRS_S2": "IFRS S2 Climate-related Disclosures",
            "EU_TAXONOMY": "EU Taxonomy Regulation",
            "INDIA_BRSR": "India BRSR",
            "CALIFORNIA_SB253": "California SB 253",
            "US_SEC": "US SEC Climate Disclosure",
        }
        return display_names.get(self.value, self.value)


class ReferenceConditions(BaseModel):
    """
    Reference conditions for gas conversions and volume normalization.

    These conditions are critical for converting between standard and
    normal volume measurements (e.g., Nm3, scf) and for fuel density
    calculations.

    Attributes:
        temperature_c: Reference temperature in degrees Celsius.
        pressure_kpa: Reference pressure in kilopascals (absolute).

    Example:
        >>> conditions = ReferenceConditions(temperature_c=15.0, pressure_kpa=101.325)
    """

    temperature_c: float = Field(
        ...,
        description="Reference temperature in degrees Celsius",
        alias="temperature_C",
    )
    pressure_kpa: float = Field(
        default=101.325,
        ge=0,
        description="Reference pressure in kilopascals (absolute)",
        alias="pressure_kPa",
    )

    model_config = {"populate_by_name": True}

    @field_validator("temperature_c")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within reasonable bounds."""
        if v < -273.15:
            raise ValueError(
                f"Temperature {v}C is below absolute zero (-273.15C)"
            )
        if v > 1000:
            raise ValueError(
                f"Temperature {v}C exceeds reasonable maximum (1000C)"
            )
        return v


class PolicyDefaults(BaseModel):
    """
    Default values applied by the Policy Engine when context is missing.

    In LENIENT mode, these defaults are applied automatically with warnings.
    In STRICT mode, missing values that would require defaults cause errors.

    Attributes:
        gwp_version: Default GWP Assessment Report version (e.g., "AR5", "AR6").
        basis: Default energy basis (LHV or HHV).
        reference_conditions: Default reference conditions for gas conversions.
        confidence_threshold: Minimum confidence for entity resolution.
        precision_digits: Default significant digits for output values.

    Example:
        >>> defaults = PolicyDefaults(
        ...     gwp_version="AR5",
        ...     basis="LHV",
        ...     confidence_threshold=0.8,
        ... )
    """

    gwp_version: str = Field(
        default="AR5",
        description="Default GWP Assessment Report version",
        pattern=r"^(AR[1-6]|SAR|TAR|FAR)$",
    )
    basis: str = Field(
        default="LHV",
        description="Default energy basis (LHV or HHV)",
        pattern=r"^(LHV|HHV)$",
    )
    reference_conditions: ReferenceConditions = Field(
        default_factory=lambda: ReferenceConditions(
            temperature_c=15.0,
            pressure_kpa=101.325,
        ),
        description="Default reference conditions for gas conversions",
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for entity resolution",
    )
    precision_digits: int = Field(
        default=6,
        ge=1,
        le=15,
        description="Default significant digits for output values",
    )
    allow_deprecated_factors: bool = Field(
        default=False,
        description="Whether to allow deprecated conversion factors",
    )
    require_unit_validation: bool = Field(
        default=True,
        description="Whether to validate all unit strings",
    )


class PolicyOverrides(BaseModel):
    """
    Override values that take precedence over defaults and compliance rules.

    Overrides allow organizations to customize behavior for specific
    use cases while maintaining audit trails of the customization.

    Attributes:
        gwp_version: Override GWP version (takes precedence over profile).
        basis: Override energy basis.
        reference_conditions: Override reference conditions.
        confidence_threshold: Override confidence threshold.
        allowed_units: Restrict to specific unit strings.
        blocked_units: Block specific unit strings.
        custom_conversions: Custom conversion factors.

    Example:
        >>> overrides = PolicyOverrides(
        ...     gwp_version="AR6",
        ...     custom_conversions={"my_unit": {"factor": 1.5, "target": "kg"}},
        ... )
    """

    gwp_version: Optional[str] = Field(
        default=None,
        description="Override GWP version",
    )
    basis: Optional[str] = Field(
        default=None,
        description="Override energy basis",
    )
    reference_conditions: Optional[ReferenceConditions] = Field(
        default=None,
        description="Override reference conditions",
    )
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override confidence threshold",
    )
    allowed_units: Optional[Set[str]] = Field(
        default=None,
        description="Restrict to specific unit strings",
    )
    blocked_units: Optional[Set[str]] = Field(
        default=None,
        description="Block specific unit strings",
    )
    custom_conversions: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Custom conversion factors",
    )

    @field_validator("gwp_version")
    @classmethod
    def validate_gwp_version(cls, v: Optional[str]) -> Optional[str]:
        """Validate GWP version format if provided."""
        if v is not None:
            import re
            if not re.match(r"^(AR[1-6]|SAR|TAR|FAR)$", v):
                raise ValueError(
                    f"Invalid GWP version '{v}'. "
                    "Expected AR1-AR6, SAR, TAR, or FAR."
                )
        return v

    @field_validator("basis")
    @classmethod
    def validate_basis(cls, v: Optional[str]) -> Optional[str]:
        """Validate energy basis if provided."""
        if v is not None:
            if v.upper() not in {"LHV", "HHV"}:
                raise ValueError(
                    f"Invalid basis '{v}'. Expected 'LHV' or 'HHV'."
                )
            return v.upper()
        return v


class Policy(BaseModel):
    """
    Complete policy configuration for the normalizer.

    A Policy combines operating mode, compliance profiles, defaults,
    and overrides into a single auditable configuration unit.

    Attributes:
        policy_id: Unique identifier for this policy.
        version: Version string for change tracking.
        mode: Operating mode (STRICT or LENIENT).
        compliance_profiles: Active compliance profiles.
        defaults: Default values for missing context.
        overrides: Override values that take precedence.
        org_id: Organization ID for multi-tenant deployments.
        effective_from: When this policy becomes effective.
        effective_until: When this policy expires (optional).
        created_at: Timestamp when policy was created.
        description: Human-readable description.

    Example:
        >>> policy = Policy(
        ...     policy_id="pol-001",
        ...     version="1.0.0",
        ...     mode=PolicyMode.STRICT,
        ...     compliance_profiles=[ComplianceProfile.GHG_PROTOCOL],
        ...     defaults=PolicyDefaults(),
        ... )
    """

    policy_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for this policy",
    )
    version: str = Field(
        default="1.0.0",
        min_length=1,
        max_length=50,
        description="Version string for change tracking",
    )
    mode: PolicyMode = Field(
        default=PolicyMode.STRICT,
        description="Operating mode (STRICT or LENIENT)",
    )
    compliance_profiles: List[ComplianceProfile] = Field(
        default_factory=list,
        description="Active compliance profiles",
    )
    defaults: PolicyDefaults = Field(
        default_factory=PolicyDefaults,
        description="Default values for missing context",
    )
    overrides: PolicyOverrides = Field(
        default_factory=PolicyOverrides,
        description="Override values that take precedence",
    )
    org_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Organization ID for multi-tenant deployments",
    )
    effective_from: Optional[datetime] = Field(
        default=None,
        description="When this policy becomes effective",
    )
    effective_until: Optional[datetime] = Field(
        default=None,
        description="When this policy expires",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when policy was created",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Human-readable description",
    )

    @model_validator(mode="after")
    def validate_effective_dates(self) -> "Policy":
        """Validate that effective_from is before effective_until."""
        if (
            self.effective_from is not None
            and self.effective_until is not None
            and self.effective_from >= self.effective_until
        ):
            raise ValueError(
                "effective_from must be before effective_until"
            )
        return self

    def is_active(self, at_time: Optional[datetime] = None) -> bool:
        """
        Check if the policy is active at a given time.

        Args:
            at_time: Time to check (defaults to now)

        Returns:
            True if the policy is active at the given time
        """
        check_time = at_time or datetime.utcnow()
        if self.effective_from and check_time < self.effective_from:
            return False
        if self.effective_until and check_time >= self.effective_until:
            return False
        return True

    def get_policy_hash(self) -> str:
        """
        Calculate SHA-256 hash of the policy for audit trails.

        Returns:
            SHA-256 hash string
        """
        # Create deterministic JSON representation
        policy_str = self.model_dump_json(exclude={"created_at"})
        return hashlib.sha256(policy_str.encode()).hexdigest()


class AppliedDefault(BaseModel):
    """
    Record of a default value that was applied during evaluation.

    This model tracks when defaults are used, enabling audit trails
    and warnings in LENIENT mode.

    Attributes:
        field: Name of the field that received a default.
        default_value: The default value that was applied.
        reason: Reason the default was needed.
        source: Source of the default (policy, profile, or system).

    Example:
        >>> applied = AppliedDefault(
        ...     field="gwp_version",
        ...     default_value="AR5",
        ...     reason="Not specified in request context",
        ...     source="policy_defaults",
        ... )
    """

    field: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name of the field that received a default",
    )
    default_value: Any = Field(
        ...,
        description="The default value that was applied",
    )
    reason: str = Field(
        ...,
        max_length=500,
        description="Reason the default was needed",
    )
    source: str = Field(
        ...,
        description="Source of the default (policy, profile, or system)",
    )


class PolicyWarning(BaseModel):
    """
    Warning generated during policy evaluation.

    Warnings indicate potential issues that did not prevent processing
    but should be reviewed for compliance or data quality.

    Attributes:
        code: Warning code for programmatic handling.
        message: Human-readable warning message.
        field: Optional field that triggered the warning.
        severity: Warning severity (low, medium, high).
        hint: Optional suggestion for resolution.

    Example:
        >>> warning = PolicyWarning(
        ...     code="POL_WARN_001",
        ...     message="Using default GWP version AR5",
        ...     field="gwp_version",
        ...     severity="medium",
        ... )
    """

    code: str = Field(
        ...,
        description="Warning code for programmatic handling",
    )
    message: str = Field(
        ...,
        max_length=1000,
        description="Human-readable warning message",
    )
    field: Optional[str] = Field(
        default=None,
        description="Field that triggered the warning",
    )
    severity: str = Field(
        default="medium",
        description="Warning severity (low, medium, high)",
    )
    hint: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Suggestion for resolution",
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Validate severity is one of the allowed values."""
        allowed = {"low", "medium", "high"}
        if v.lower() not in allowed:
            raise ValueError(
                f"Invalid severity '{v}'. Allowed: {', '.join(sorted(allowed))}"
            )
        return v.lower()


class EffectiveConfig(BaseModel):
    """
    Effective configuration after policy evaluation.

    This model represents the final resolved configuration after
    applying all policies, profiles, defaults, and overrides.

    Attributes:
        gwp_version: Effective GWP version.
        basis: Effective energy basis.
        reference_conditions: Effective reference conditions.
        confidence_threshold: Effective confidence threshold.
        precision_digits: Effective precision digits.
        compliance_profiles: Active compliance profiles.
        allowed_units: Allowed unit strings (None means all).
        blocked_units: Blocked unit strings (None means none).
        custom_conversions: Custom conversion factors.

    Example:
        >>> config = EffectiveConfig(
        ...     gwp_version="AR5",
        ...     basis="LHV",
        ...     confidence_threshold=0.8,
        ... )
    """

    gwp_version: str = Field(
        ...,
        description="Effective GWP version",
    )
    basis: str = Field(
        ...,
        description="Effective energy basis",
    )
    reference_conditions: ReferenceConditions = Field(
        ...,
        description="Effective reference conditions",
    )
    confidence_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Effective confidence threshold",
    )
    precision_digits: int = Field(
        ...,
        ge=1,
        le=15,
        description="Effective precision digits",
    )
    compliance_profiles: List[ComplianceProfile] = Field(
        default_factory=list,
        description="Active compliance profiles",
    )
    allowed_units: Optional[Set[str]] = Field(
        default=None,
        description="Allowed unit strings (None means all)",
    )
    blocked_units: Optional[Set[str]] = Field(
        default=None,
        description="Blocked unit strings (None means none)",
    )
    custom_conversions: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Custom conversion factors",
    )
    allow_deprecated_factors: bool = Field(
        default=False,
        description="Whether deprecated factors are allowed",
    )


class PolicyDecision(BaseModel):
    """
    Result of a policy evaluation.

    This model captures the complete outcome of evaluating a request
    against a policy, including the decision, warnings, applied defaults,
    and effective configuration.

    Attributes:
        allowed: Whether the request is allowed to proceed.
        decision_id: Unique identifier for this decision.
        policy_id: ID of the policy that was evaluated.
        policy_version: Version of the policy.
        policy_hash: SHA-256 hash of the policy for audit.
        mode: Policy mode that was used.
        warnings: List of warnings generated.
        applied_defaults: List of defaults that were applied.
        effective_config: Final resolved configuration.
        evaluation_time_ms: Time taken to evaluate in milliseconds.
        evaluated_at: Timestamp of evaluation.
        denial_reason: Reason for denial (if not allowed).

    Example:
        >>> decision = PolicyDecision(
        ...     allowed=True,
        ...     decision_id="dec-001",
        ...     policy_id="pol-001",
        ...     policy_version="1.0.0",
        ...     policy_hash="sha256:...",
        ...     mode=PolicyMode.LENIENT,
        ...     effective_config=EffectiveConfig(...),
        ...     evaluation_time_ms=1.5,
        ... )
    """

    allowed: bool = Field(
        ...,
        description="Whether the request is allowed to proceed",
    )
    decision_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for this decision",
    )
    policy_id: str = Field(
        ...,
        description="ID of the policy that was evaluated",
    )
    policy_version: str = Field(
        ...,
        description="Version of the policy",
    )
    policy_hash: str = Field(
        ...,
        description="SHA-256 hash of the policy for audit",
    )
    mode: PolicyMode = Field(
        ...,
        description="Policy mode that was used",
    )
    warnings: List[PolicyWarning] = Field(
        default_factory=list,
        description="List of warnings generated",
    )
    applied_defaults: List[AppliedDefault] = Field(
        default_factory=list,
        description="List of defaults that were applied",
    )
    effective_config: EffectiveConfig = Field(
        ...,
        description="Final resolved configuration",
    )
    evaluation_time_ms: float = Field(
        ...,
        ge=0,
        description="Time taken to evaluate in milliseconds",
    )
    evaluated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of evaluation",
    )
    denial_reason: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Reason for denial (if not allowed)",
    )

    @model_validator(mode="after")
    def validate_denial_reason(self) -> "PolicyDecision":
        """Validate that denial_reason is set when not allowed."""
        if not self.allowed and not self.denial_reason:
            raise ValueError(
                "denial_reason must be set when allowed is False"
            )
        return self

    def get_decision_hash(self) -> str:
        """
        Calculate SHA-256 hash of the decision for audit trails.

        Returns:
            SHA-256 hash string
        """
        # Create deterministic representation
        decision_str = self.model_dump_json(exclude={"evaluated_at"})
        return hashlib.sha256(decision_str.encode()).hexdigest()


class ConversionPolicy(BaseModel):
    """
    Policy for a specific type of unit conversion.

    This model defines rules for how specific conversions should be
    handled, including required context and compliance requirements.

    Attributes:
        conversion_type: Type of conversion (unit, gwp, basis).
        requires_gwp_version: Whether GWP version is required.
        requires_basis: Whether energy basis is required.
        requires_reference_conditions: Whether reference conditions are required.
        allowed_source_units: Allowed source units.
        allowed_target_units: Allowed target units.
        compliance_profiles: Profiles that mandate this conversion policy.

    Example:
        >>> conversion_policy = ConversionPolicy(
        ...     conversion_type="gwp",
        ...     requires_gwp_version=True,
        ...     compliance_profiles=[ComplianceProfile.GHG_PROTOCOL],
        ... )
    """

    conversion_type: str = Field(
        ...,
        description="Type of conversion (unit, gwp, basis)",
    )
    requires_gwp_version: bool = Field(
        default=False,
        description="Whether GWP version is required",
    )
    requires_basis: bool = Field(
        default=False,
        description="Whether energy basis is required",
    )
    requires_reference_conditions: bool = Field(
        default=False,
        description="Whether reference conditions are required",
    )
    allowed_source_units: Optional[Set[str]] = Field(
        default=None,
        description="Allowed source units",
    )
    allowed_target_units: Optional[Set[str]] = Field(
        default=None,
        description="Allowed target units",
    )
    compliance_profiles: List[ComplianceProfile] = Field(
        default_factory=list,
        description="Profiles that mandate this conversion policy",
    )


# Type aliases for convenience
PolicyType = Policy
PolicyDecisionType = PolicyDecision
ComplianceProfileType = ComplianceProfile
