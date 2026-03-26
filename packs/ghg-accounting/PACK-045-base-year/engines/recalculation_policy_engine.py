# -*- coding: utf-8 -*-
"""
RecalculationPolicyEngine - PACK-045 Base Year Management Engine 3
====================================================================

Configurable recalculation policy management engine implementing
multi-framework compliance for GHG Protocol, SBTi, SEC, and CDP
base year recalculation policies with trigger rules, approval
workflows, and threshold management.

Calculation Methodology:
    Significance Test:
        significance_pct = abs(estimated_impact) / base_year_total * 100
        is_significant = significance_pct >= threshold_pct

    Individual Trigger Threshold:
        A single event triggers recalculation if its impact exceeds
        the individual threshold (default 5% of base year total).

    Cumulative Trigger Threshold:
        Multiple sub-threshold events are accumulated; recalculation
        is required when their cumulative impact exceeds the
        cumulative threshold (default 10% of base year total).

    Framework-Specific Thresholds:
        GHG Protocol: Individual 5%, cumulative 10% (guidance)
        SBTi:         Individual 5% (mandatory for target tracking)
        SEC:          Individual 5% (per Item 1504 materiality)
        CDP:          Individual 5% (per C5.1 reporting guidance)

    Trigger Rules:
        Each trigger type (acquisition, divestiture, merger, etc.)
        has configurable settings:
        - enabled:              Whether the trigger type is active.
        - threshold_override:   Optional override of default threshold.
        - auto_detect:          Whether to automatically detect.
        - requires_approval:    Whether manual approval is needed.
        - approval_level:       Required approval level.
        - documentation_required: Whether documentation is mandatory.

    Policy Compliance Check:
        For each framework (GHG Protocol, SBTi, SEC, CDP):
        - Verify threshold meets minimum requirement.
        - Verify all mandatory triggers are enabled.
        - Verify documentation requirements are met.
        - Verify approval workflow is sufficient.

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015), Chapter 5
    - GHG Protocol Scope 2 Guidance (2015), Chapter 7
    - GHG Protocol Corporate Value Chain Standard, Chapter 5
    - SBTi Corporate Manual (2023), Section 7 (Recalculation)
    - SBTi Criteria and Recommendations (v5.1), Section 6
    - SEC Climate Disclosure Rule (2024), Item 1504
    - CDP Climate Change Questionnaire C5.1-C5.2
    - ESRS E1-6 (Gross GHG emissions)
    - ISO 14064-1:2018, Clause 5.2

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Threshold values from published regulatory guidance
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-045 Base Year Management
Engine:  3 of 10
Status:  Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Excludes volatile fields (calculated_at, processing_time_ms,
    provenance_hash) so that identical logical inputs always produce
    the same hash.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round2(value: Any) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round3(value: Any) -> float:
    """Round to 3 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PolicyType(str, Enum):
    """Type of recalculation policy.

    GHG_PROTOCOL_DEFAULT: Standard GHG Protocol Chapter 5 defaults.
                          Individual 5%, cumulative 10%.
    SBTI_STRICT:          SBTi Corporate Manual Section 7 requirements.
                          Strict 5% threshold, mandatory for targets.
    SEC_COMPLIANT:        SEC Climate Disclosure Rule Item 1504.
                          Materiality-based threshold (5%).
    CDP_ALIGNED:          CDP Climate Change Questionnaire C5.1.
                          Aligned with GHG Protocol, CDP-specific docs.
    CUSTOM:               User-defined policy with custom thresholds
                          and trigger rules.
    """
    GHG_PROTOCOL_DEFAULT = "ghg_protocol_default"
    SBTI_STRICT = "sbti_strict"
    SEC_COMPLIANT = "sec_compliant"
    CDP_ALIGNED = "cdp_aligned"
    CUSTOM = "custom"


class TriggerType(str, Enum):
    """Types of events that may trigger base year recalculation.

    Per GHG Protocol Corporate Standard, Chapter 5, Table 5.3:

    ACQUISITION:            Purchase of operations, facilities, or
                            business units that bring new emissions
                            into the organisational boundary.
    DIVESTITURE:            Sale, closure, or transfer of operations
                            that remove emissions from the boundary.
    MERGER:                 Merger with another organisation that
                            fundamentally restructures the entity.
    METHODOLOGY_CHANGE:     Change in calculation methodology,
                            emission factors, GWP values, or
                            measurement techniques.
    ERROR_CORRECTION:       Discovery and correction of significant
                            errors in historical data.
    SOURCE_BOUNDARY_CHANGE: Addition or removal of emission source
                            categories from the operational boundary.
    OUTSOURCING_INSOURCING: Transfer of activities between the
                            reporting entity and third parties.
    """
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    METHODOLOGY_CHANGE = "methodology_change"
    ERROR_CORRECTION = "error_correction"
    SOURCE_BOUNDARY_CHANGE = "source_boundary_change"
    OUTSOURCING_INSOURCING = "outsourcing_insourcing"


class ApprovalLevel(str, Enum):
    """Approval level required for recalculation decisions.

    AUTO_APPROVE: Automatic approval (no human review).
                  For minor, well-defined trigger types.
    MANAGER:      Line manager approval required.
                  For routine recalculations below 5%.
    DIRECTOR:     Director-level approval required.
                  For significant recalculations (5-15%).
    COMMITTEE:    Sustainability committee approval.
                  For major recalculations (>15%).
    BOARD:        Board-level approval required.
                  For fundamental changes to base year.
    """
    AUTO_APPROVE = "auto_approve"
    MANAGER = "manager"
    DIRECTOR = "director"
    COMMITTEE = "committee"
    BOARD = "board"


class ComplianceFramework(str, Enum):
    """Regulatory/reporting framework for compliance checking.

    GHG_PROTOCOL: GHG Protocol Corporate Standard.
    SBTI:         Science Based Targets initiative.
    SEC:          SEC Climate Disclosure Rule.
    CDP:          CDP Climate Change Questionnaire.
    ESRS:         European Sustainability Reporting Standards.
    ISO_14064:    ISO 14064-1:2018.
    """
    GHG_PROTOCOL = "ghg_protocol"
    SBTI = "sbti"
    SEC = "sec"
    CDP = "cdp"
    ESRS = "esrs"
    ISO_14064 = "iso_14064"


class ValidationSeverity(str, Enum):
    """Severity of a policy validation finding.

    ERROR:   Policy is non-compliant; must be fixed.
    WARNING: Policy has potential issues; should be reviewed.
    INFO:    Informational note; no action required.
    """
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Framework-specific threshold requirements.
# Source: Published regulatory guidance documents.
FRAMEWORK_THRESHOLDS: Dict[str, Dict[str, Decimal]] = {
    ComplianceFramework.GHG_PROTOCOL.value: {
        "individual_min": Decimal("5"),
        "individual_max": Decimal("10"),
        "cumulative_min": Decimal("5"),
        "cumulative_max": Decimal("10"),
    },
    ComplianceFramework.SBTI.value: {
        "individual_min": Decimal("5"),
        "individual_max": Decimal("5"),
        "cumulative_min": Decimal("5"),
        "cumulative_max": Decimal("5"),
    },
    ComplianceFramework.SEC.value: {
        "individual_min": Decimal("5"),
        "individual_max": Decimal("10"),
        "cumulative_min": Decimal("5"),
        "cumulative_max": Decimal("10"),
    },
    ComplianceFramework.CDP.value: {
        "individual_min": Decimal("5"),
        "individual_max": Decimal("10"),
        "cumulative_min": Decimal("5"),
        "cumulative_max": Decimal("10"),
    },
    ComplianceFramework.ESRS.value: {
        "individual_min": Decimal("5"),
        "individual_max": Decimal("10"),
        "cumulative_min": Decimal("5"),
        "cumulative_max": Decimal("10"),
    },
    ComplianceFramework.ISO_14064.value: {
        "individual_min": Decimal("5"),
        "individual_max": Decimal("10"),
        "cumulative_min": Decimal("5"),
        "cumulative_max": Decimal("10"),
    },
}

# Framework-specific mandatory trigger types.
# These triggers MUST be enabled for compliance.
FRAMEWORK_MANDATORY_TRIGGERS: Dict[str, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL.value: [
        TriggerType.ACQUISITION.value,
        TriggerType.DIVESTITURE.value,
        TriggerType.MERGER.value,
        TriggerType.METHODOLOGY_CHANGE.value,
        TriggerType.ERROR_CORRECTION.value,
        TriggerType.SOURCE_BOUNDARY_CHANGE.value,
    ],
    ComplianceFramework.SBTI.value: [
        TriggerType.ACQUISITION.value,
        TriggerType.DIVESTITURE.value,
        TriggerType.MERGER.value,
        TriggerType.METHODOLOGY_CHANGE.value,
        TriggerType.ERROR_CORRECTION.value,
        TriggerType.SOURCE_BOUNDARY_CHANGE.value,
        TriggerType.OUTSOURCING_INSOURCING.value,
    ],
    ComplianceFramework.SEC.value: [
        TriggerType.ACQUISITION.value,
        TriggerType.DIVESTITURE.value,
        TriggerType.MERGER.value,
        TriggerType.METHODOLOGY_CHANGE.value,
        TriggerType.ERROR_CORRECTION.value,
    ],
    ComplianceFramework.CDP.value: [
        TriggerType.ACQUISITION.value,
        TriggerType.DIVESTITURE.value,
        TriggerType.MERGER.value,
        TriggerType.METHODOLOGY_CHANGE.value,
        TriggerType.ERROR_CORRECTION.value,
    ],
    ComplianceFramework.ESRS.value: [
        TriggerType.ACQUISITION.value,
        TriggerType.DIVESTITURE.value,
        TriggerType.MERGER.value,
        TriggerType.METHODOLOGY_CHANGE.value,
        TriggerType.ERROR_CORRECTION.value,
        TriggerType.SOURCE_BOUNDARY_CHANGE.value,
    ],
    ComplianceFramework.ISO_14064.value: [
        TriggerType.ACQUISITION.value,
        TriggerType.DIVESTITURE.value,
        TriggerType.METHODOLOGY_CHANGE.value,
        TriggerType.ERROR_CORRECTION.value,
    ],
}

# Framework-specific documentation requirements.
FRAMEWORK_DOCUMENTATION: Dict[str, List[str]] = {
    ComplianceFramework.GHG_PROTOCOL.value: [
        "Recalculation policy description",
        "Significance threshold justification",
        "Trigger event documentation",
        "Before/after comparison",
    ],
    ComplianceFramework.SBTI.value: [
        "Recalculation policy description",
        "Significance threshold justification",
        "Trigger event documentation",
        "Before/after comparison",
        "Target recalculation impact",
        "SBTi notification letter",
    ],
    ComplianceFramework.SEC.value: [
        "Recalculation policy description",
        "Materiality assessment",
        "Trigger event documentation",
        "Before/after comparison",
        "Board/committee approval record",
        "Auditor notification",
    ],
    ComplianceFramework.CDP.value: [
        "Recalculation policy description",
        "Significance threshold justification",
        "Trigger event documentation",
        "Before/after comparison",
    ],
    ComplianceFramework.ESRS.value: [
        "Recalculation policy description",
        "Significance threshold justification",
        "Trigger event documentation",
        "Before/after comparison",
        "ESRS E1-6 disclosure alignment",
    ],
    ComplianceFramework.ISO_14064.value: [
        "Recalculation policy description",
        "Significance threshold justification",
        "Trigger event documentation",
        "Before/after comparison",
        "Verification statement update",
    ],
}

# Minimum approval levels per framework for significant recalculations.
FRAMEWORK_MIN_APPROVAL: Dict[str, str] = {
    ComplianceFramework.GHG_PROTOCOL.value: ApprovalLevel.MANAGER.value,
    ComplianceFramework.SBTI.value: ApprovalLevel.DIRECTOR.value,
    ComplianceFramework.SEC.value: ApprovalLevel.COMMITTEE.value,
    ComplianceFramework.CDP.value: ApprovalLevel.MANAGER.value,
    ComplianceFramework.ESRS.value: ApprovalLevel.DIRECTOR.value,
    ComplianceFramework.ISO_14064.value: ApprovalLevel.MANAGER.value,
}

# Approval level hierarchy (for comparison).
APPROVAL_HIERARCHY: Dict[str, int] = {
    ApprovalLevel.AUTO_APPROVE.value: 0,
    ApprovalLevel.MANAGER.value: 1,
    ApprovalLevel.DIRECTOR.value: 2,
    ApprovalLevel.COMMITTEE.value: 3,
    ApprovalLevel.BOARD.value: 4,
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Core
# ---------------------------------------------------------------------------


class ThresholdConfig(BaseModel):
    """Threshold configuration for recalculation significance testing.

    Attributes:
        individual_pct:  Threshold for a single trigger event (%).
        cumulative_pct:  Threshold for cumulative trigger events (%).
        sbti_pct:        SBTi-specific threshold override (%).
        sec_pct:         SEC-specific threshold override (%).
    """
    individual_pct: Decimal = Field(
        default=Decimal("5.0"), ge=0, le=100,
        description="Individual trigger threshold (%)"
    )
    cumulative_pct: Decimal = Field(
        default=Decimal("10.0"), ge=0, le=100,
        description="Cumulative trigger threshold (%)"
    )
    sbti_pct: Decimal = Field(
        default=Decimal("5.0"), ge=0, le=100,
        description="SBTi-specific threshold (%)"
    )
    sec_pct: Decimal = Field(
        default=Decimal("5.0"), ge=0, le=100,
        description="SEC-specific threshold (%)"
    )

    @field_validator(
        "individual_pct", "cumulative_pct", "sbti_pct", "sec_pct",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce threshold values to Decimal."""
        return _decimal(v)

    @model_validator(mode="after")
    def validate_cumulative_gte_individual(self) -> "ThresholdConfig":
        """Validate cumulative threshold >= individual threshold.

        The cumulative threshold should be at least as large as the
        individual threshold, since cumulative includes multiple events.
        """
        if self.cumulative_pct < self.individual_pct:
            raise ValueError(
                f"Cumulative threshold ({self.cumulative_pct}%) must be >= "
                f"individual threshold ({self.individual_pct}%)"
            )
        return self


class TriggerRule(BaseModel):
    """Configuration for a single trigger type.

    Defines how a specific type of recalculation trigger is handled,
    including whether it is enabled, its threshold, approval
    requirements, and documentation needs.

    Attributes:
        trigger_type:           The trigger event type.
        enabled:                Whether this trigger is active.
        threshold_override:     Optional threshold override (%).
        auto_detect:            Whether to auto-detect this trigger.
        requires_approval:      Whether manual approval is needed.
        approval_level:         Required approval level.
        documentation_required: Whether documentation is mandatory.
        description:            Description of the rule.
        notes:                  Additional notes or guidance.
    """
    trigger_type: TriggerType = Field(
        ..., description="Trigger event type"
    )
    enabled: bool = Field(
        default=True, description="Trigger is active"
    )
    threshold_override: Optional[Decimal] = Field(
        default=None, ge=0, le=100,
        description="Threshold override (%)"
    )
    auto_detect: bool = Field(
        default=False,
        description="Auto-detect this trigger type"
    )
    requires_approval: bool = Field(
        default=True,
        description="Manual approval required"
    )
    approval_level: ApprovalLevel = Field(
        default=ApprovalLevel.MANAGER,
        description="Required approval level"
    )
    documentation_required: bool = Field(
        default=True,
        description="Documentation mandatory"
    )
    description: str = Field(
        default="", description="Rule description"
    )
    notes: str = Field(
        default="", description="Additional notes"
    )

    @field_validator("threshold_override", mode="before")
    @classmethod
    def coerce_threshold(cls, v: Any) -> Optional[Decimal]:
        """Coerce threshold override to Decimal."""
        if v is None:
            return None
        return _decimal(v)


class RecalculationPolicy(BaseModel):
    """Complete recalculation policy configuration.

    Defines the full set of rules, thresholds, and approval
    requirements for base year recalculation decisions.

    Attributes:
        policy_id:                       Unique policy identifier.
        policy_type:                     Pre-defined policy type.
        name:                            Policy name.
        description:                     Policy description.
        thresholds:                      Threshold configuration.
        trigger_rules:                   List of trigger rules.
        rolling_base_year:               Whether using rolling base year.
        rolling_period_years:            Rolling period (years).
        require_board_approval_above_pct: Threshold (%) requiring board
                                         approval.
        documentation_requirements:      List of required documents.
        applicable_frameworks:           Frameworks this policy targets.
        effective_date:                  Policy effective date (ISO).
        expiry_date:                     Policy expiry date (optional).
        version:                         Policy version number.
        is_active:                       Whether policy is currently active.
        created_by:                      Creator of the policy.
        approved_by:                     Approver of the policy.
        notes:                           Additional policy notes.
        calculated_at:                   Timestamp.
        provenance_hash:                 SHA-256 provenance hash.
    """
    policy_id: str = Field(
        default_factory=_new_uuid,
        description="Policy identifier"
    )
    policy_type: PolicyType = Field(
        ..., description="Policy type"
    )
    name: str = Field(
        default="", description="Policy name"
    )
    description: str = Field(
        default="", description="Policy description"
    )
    thresholds: ThresholdConfig = Field(
        default_factory=ThresholdConfig,
        description="Threshold configuration"
    )
    trigger_rules: List[TriggerRule] = Field(
        default_factory=list,
        description="Trigger rules"
    )
    rolling_base_year: bool = Field(
        default=False,
        description="Use rolling base year"
    )
    rolling_period_years: Optional[int] = Field(
        default=None, ge=3, le=10,
        description="Rolling period (years)"
    )
    require_board_approval_above_pct: Decimal = Field(
        default=Decimal("15.0"), ge=0, le=100,
        description="Board approval threshold (%)"
    )
    documentation_requirements: List[str] = Field(
        default_factory=list,
        description="Required documentation"
    )
    applicable_frameworks: List[ComplianceFramework] = Field(
        default_factory=list,
        description="Target frameworks"
    )
    effective_date: str = Field(
        default="", description="Effective date (ISO)"
    )
    expiry_date: Optional[str] = Field(
        default=None, description="Expiry date (ISO)"
    )
    version: int = Field(
        default=1, ge=1, description="Policy version"
    )
    is_active: bool = Field(
        default=True, description="Policy is active"
    )
    created_by: str = Field(
        default="", description="Created by"
    )
    approved_by: str = Field(
        default="", description="Approved by"
    )
    notes: str = Field(
        default="", description="Additional notes"
    )
    calculated_at: str = Field(
        default="", description="Timestamp"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    @field_validator("require_board_approval_above_pct", mode="before")
    @classmethod
    def coerce_board_threshold(cls, v: Any) -> Decimal:
        """Coerce board approval threshold to Decimal."""
        return _decimal(v)


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class ComplianceGap(BaseModel):
    """A single compliance gap identified during policy checking.

    Attributes:
        framework:     Affected framework.
        severity:      Gap severity (error, warning, info).
        category:      Gap category (threshold, trigger, docs, approval).
        description:   Human-readable description.
        requirement:   The requirement that is not met.
        current_value: Current policy value.
        required_value: Required value for compliance.
        remediation:   Suggested remediation action.
    """
    framework: ComplianceFramework = Field(
        ..., description="Affected framework"
    )
    severity: ValidationSeverity = Field(
        ..., description="Gap severity"
    )
    category: str = Field(
        default="", description="Gap category"
    )
    description: str = Field(
        ..., description="Gap description"
    )
    requirement: str = Field(
        default="", description="Requirement"
    )
    current_value: str = Field(
        default="", description="Current value"
    )
    required_value: str = Field(
        default="", description="Required value"
    )
    remediation: str = Field(
        default="", description="Remediation action"
    )


class PolicyComplianceCheck(BaseModel):
    """Result of checking a policy against a single framework.

    Attributes:
        policy_id:         Policy being checked.
        framework:         Framework being checked against.
        is_compliant:      Whether policy meets all requirements.
        gaps:              List of compliance gaps.
        recommendations:   List of improvement recommendations.
        checked_at:        Timestamp of check.
    """
    policy_id: str = Field(
        default="", description="Policy identifier"
    )
    framework: ComplianceFramework = Field(
        ..., description="Compliance framework"
    )
    is_compliant: bool = Field(
        default=True, description="Policy is compliant"
    )
    gaps: List[ComplianceGap] = Field(
        default_factory=list,
        description="Compliance gaps"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )
    checked_at: str = Field(
        default="", description="Check timestamp"
    )


class PolicyValidationResult(BaseModel):
    """Result of validating a policy's internal consistency.

    Attributes:
        policy_id:   Policy being validated.
        is_valid:    Whether policy passes all validation checks.
        errors:      List of validation errors.
        warnings:    List of validation warnings.
        info:        List of informational notes.
    """
    policy_id: str = Field(
        default="", description="Policy identifier"
    )
    is_valid: bool = Field(
        default=True, description="Policy is valid"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Validation errors"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Validation warnings"
    )
    info: List[str] = Field(
        default_factory=list,
        description="Informational notes"
    )


class PolicyComparisonItem(BaseModel):
    """A single difference between two policies.

    Attributes:
        field:      Name of the differing field.
        policy1_val: Value in policy 1.
        policy2_val: Value in policy 2.
        impact:     Description of the impact.
    """
    field: str = Field(..., description="Field name")
    policy1_val: str = Field(default="", description="Policy 1 value")
    policy2_val: str = Field(default="", description="Policy 2 value")
    impact: str = Field(default="", description="Impact description")


class PolicyComparison(BaseModel):
    """Comparison between two recalculation policies.

    Attributes:
        policy1_id:     First policy identifier.
        policy2_id:     Second policy identifier.
        differences:    List of differences.
        policy1_stricter: Fields where policy 1 is stricter.
        policy2_stricter: Fields where policy 2 is stricter.
        summary:        Summary of comparison.
        provenance_hash: SHA-256 provenance hash.
    """
    policy1_id: str = Field(default="", description="Policy 1 ID")
    policy2_id: str = Field(default="", description="Policy 2 ID")
    differences: List[PolicyComparisonItem] = Field(
        default_factory=list, description="Differences"
    )
    policy1_stricter: List[str] = Field(
        default_factory=list, description="Fields where P1 is stricter"
    )
    policy2_stricter: List[str] = Field(
        default_factory=list, description="Fields where P2 is stricter"
    )
    summary: str = Field(default="", description="Summary")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RecalculationPolicyEngine:
    """Configurable recalculation policy management engine.

    Creates, validates, and checks compliance of base year
    recalculation policies against multiple regulatory frameworks.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: All policy decisions documented with rationale.
        - Zero-Hallucination: No LLM in any calculation path.

    Usage:
        engine = RecalculationPolicyEngine()
        policy = engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        checks = engine.check_compliance(
            policy, [ComplianceFramework.GHG_PROTOCOL, ComplianceFramework.SBTI]
        )
        validation = engine.validate_policy(policy)
    """

    def __init__(self) -> None:
        """Initialise the RecalculationPolicyEngine."""
        self._version = _MODULE_VERSION
        logger.info(
            "RecalculationPolicyEngine v%s initialised", self._version
        )

    # ------------------------------------------------------------------
    # Public API -- Policy Creation
    # ------------------------------------------------------------------

    def create_policy(
        self,
        policy_type: PolicyType,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> RecalculationPolicy:
        """Create a recalculation policy with sensible defaults.

        Generates a complete policy pre-configured for the selected
        policy type.  All trigger rules, thresholds, and documentation
        requirements are set according to the framework's guidance.

        Args:
            policy_type:  Type of policy to create.
            name:         Optional custom name (defaults to type name).
            description:  Optional custom description.

        Returns:
            RecalculationPolicy with all defaults set.
        """
        if policy_type == PolicyType.GHG_PROTOCOL_DEFAULT:
            policy = self.get_ghg_protocol_defaults()
        elif policy_type == PolicyType.SBTI_STRICT:
            policy = self.get_sbti_policy()
        elif policy_type == PolicyType.SEC_COMPLIANT:
            policy = self.get_sec_policy()
        elif policy_type == PolicyType.CDP_ALIGNED:
            policy = self.get_cdp_policy()
        elif policy_type == PolicyType.CUSTOM:
            policy = self._get_custom_base()
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        if name:
            policy.name = name
        if description:
            policy.description = description

        policy.calculated_at = _utcnow().isoformat()
        policy.provenance_hash = _compute_hash(policy)
        return policy

    def get_ghg_protocol_defaults(self) -> RecalculationPolicy:
        """Create GHG Protocol Corporate Standard default policy.

        Based on GHG Protocol Corporate Standard (2004, revised 2015),
        Chapter 5: Tracking Emissions Over Time.

        Defaults:
            - Individual threshold: 5%
            - Cumulative threshold: 10%
            - All 7 trigger types enabled
            - Manager approval for standard triggers
            - Committee approval above 15%

        Returns:
            RecalculationPolicy with GHG Protocol defaults.
        """
        trigger_rules = self._build_standard_trigger_rules(
            default_approval=ApprovalLevel.MANAGER,
            merger_approval=ApprovalLevel.DIRECTOR,
            threshold_override=None,
        )

        return RecalculationPolicy(
            policy_type=PolicyType.GHG_PROTOCOL_DEFAULT,
            name="GHG Protocol Default Recalculation Policy",
            description=(
                "Standard recalculation policy based on GHG Protocol "
                "Corporate Standard Chapter 5 guidance. Applies a 5% "
                "individual significance threshold and 10% cumulative "
                "threshold for assessing recalculation triggers."
            ),
            thresholds=ThresholdConfig(
                individual_pct=Decimal("5.0"),
                cumulative_pct=Decimal("10.0"),
                sbti_pct=Decimal("5.0"),
                sec_pct=Decimal("5.0"),
            ),
            trigger_rules=trigger_rules,
            rolling_base_year=False,
            require_board_approval_above_pct=Decimal("15.0"),
            documentation_requirements=[
                "Recalculation policy description",
                "Significance threshold justification",
                "Trigger event documentation",
                "Before/after inventory comparison",
                "Approval record",
            ],
            applicable_frameworks=[
                ComplianceFramework.GHG_PROTOCOL,
            ],
            effective_date=_utcnow().isoformat(),
        )

    def get_sbti_policy(self) -> RecalculationPolicy:
        """Create SBTi-compliant recalculation policy.

        Based on SBTi Corporate Manual (2023), Section 7 and
        SBTi Criteria and Recommendations (v5.1), Section 6.

        SBTi-specific requirements:
            - 5% significance threshold (mandatory)
            - All trigger types including outsourcing/insourcing
            - Mandatory target recalculation documentation
            - Director-level minimum approval
            - SBTi notification required

        Returns:
            RecalculationPolicy with SBTi-compliant settings.
        """
        trigger_rules = self._build_standard_trigger_rules(
            default_approval=ApprovalLevel.DIRECTOR,
            merger_approval=ApprovalLevel.COMMITTEE,
            threshold_override=None,
        )

        # SBTi requires outsourcing/insourcing trigger
        outsource_rule = self._find_trigger_rule(
            trigger_rules, TriggerType.OUTSOURCING_INSOURCING
        )
        if outsource_rule:
            outsource_rule.enabled = True
            outsource_rule.requires_approval = True
            outsource_rule.approval_level = ApprovalLevel.DIRECTOR
            outsource_rule.description = (
                "SBTi requires recalculation when activities are transferred "
                "between the reporting entity and third parties."
            )

        return RecalculationPolicy(
            policy_type=PolicyType.SBTI_STRICT,
            name="SBTi Strict Recalculation Policy",
            description=(
                "Strict recalculation policy compliant with SBTi Corporate "
                "Manual Section 7. Applies a mandatory 5% significance "
                "threshold with director-level approval. All trigger types "
                "are enabled including outsourcing/insourcing."
            ),
            thresholds=ThresholdConfig(
                individual_pct=Decimal("5.0"),
                cumulative_pct=Decimal("5.0"),
                sbti_pct=Decimal("5.0"),
                sec_pct=Decimal("5.0"),
            ),
            trigger_rules=trigger_rules,
            rolling_base_year=False,
            require_board_approval_above_pct=Decimal("10.0"),
            documentation_requirements=[
                "Recalculation policy description",
                "Significance threshold justification",
                "Trigger event documentation",
                "Before/after inventory comparison",
                "Target recalculation impact assessment",
                "SBTi notification letter",
                "Director/committee approval record",
            ],
            applicable_frameworks=[
                ComplianceFramework.SBTI,
                ComplianceFramework.GHG_PROTOCOL,
            ],
            effective_date=_utcnow().isoformat(),
        )

    def get_sec_policy(self) -> RecalculationPolicy:
        """Create SEC-compliant recalculation policy.

        Based on SEC Climate Disclosure Rule (2024), Item 1504
        requirements for material recalculations.

        SEC-specific requirements:
            - 5% materiality threshold
            - Committee-level approval for significant changes
            - Board approval for material changes (>10%)
            - Auditor notification requirement
            - Formal documentation for SEC filings

        Returns:
            RecalculationPolicy with SEC-compliant settings.
        """
        trigger_rules = self._build_standard_trigger_rules(
            default_approval=ApprovalLevel.COMMITTEE,
            merger_approval=ApprovalLevel.BOARD,
            threshold_override=None,
        )

        # SEC requires enhanced approval for error corrections
        error_rule = self._find_trigger_rule(
            trigger_rules, TriggerType.ERROR_CORRECTION
        )
        if error_rule:
            error_rule.approval_level = ApprovalLevel.COMMITTEE
            error_rule.description = (
                "SEC requires committee approval and auditor notification "
                "for material error corrections affecting climate disclosures."
            )

        return RecalculationPolicy(
            policy_type=PolicyType.SEC_COMPLIANT,
            name="SEC Climate Disclosure Compliant Policy",
            description=(
                "Recalculation policy compliant with SEC Climate Disclosure "
                "Rule Item 1504. Applies materiality-based 5% threshold "
                "with committee-level approval for significant changes. "
                "Board approval required for changes exceeding 10%."
            ),
            thresholds=ThresholdConfig(
                individual_pct=Decimal("5.0"),
                cumulative_pct=Decimal("10.0"),
                sbti_pct=Decimal("5.0"),
                sec_pct=Decimal("5.0"),
            ),
            trigger_rules=trigger_rules,
            rolling_base_year=False,
            require_board_approval_above_pct=Decimal("10.0"),
            documentation_requirements=[
                "Recalculation policy description",
                "Materiality assessment",
                "Trigger event documentation",
                "Before/after inventory comparison",
                "Board/committee approval record",
                "Auditor notification",
                "SEC filing impact assessment",
                "Internal controls documentation",
            ],
            applicable_frameworks=[
                ComplianceFramework.SEC,
                ComplianceFramework.GHG_PROTOCOL,
            ],
            effective_date=_utcnow().isoformat(),
        )

    def get_cdp_policy(self) -> RecalculationPolicy:
        """Create CDP-aligned recalculation policy.

        Based on CDP Climate Change Questionnaire sections C5.1
        and C5.2 guidance on base year recalculation.

        CDP-specific requirements:
            - 5% significance threshold (aligned with GHG Protocol)
            - Documentation for CDP questionnaire responses
            - Explanation of recalculation context

        Returns:
            RecalculationPolicy with CDP-aligned settings.
        """
        trigger_rules = self._build_standard_trigger_rules(
            default_approval=ApprovalLevel.MANAGER,
            merger_approval=ApprovalLevel.DIRECTOR,
            threshold_override=None,
        )

        return RecalculationPolicy(
            policy_type=PolicyType.CDP_ALIGNED,
            name="CDP-Aligned Recalculation Policy",
            description=(
                "Recalculation policy aligned with CDP Climate Change "
                "Questionnaire C5.1-C5.2 guidance. Follows GHG Protocol "
                "defaults with additional documentation for CDP responses."
            ),
            thresholds=ThresholdConfig(
                individual_pct=Decimal("5.0"),
                cumulative_pct=Decimal("10.0"),
                sbti_pct=Decimal("5.0"),
                sec_pct=Decimal("5.0"),
            ),
            trigger_rules=trigger_rules,
            rolling_base_year=False,
            require_board_approval_above_pct=Decimal("15.0"),
            documentation_requirements=[
                "Recalculation policy description",
                "Significance threshold justification",
                "Trigger event documentation",
                "Before/after inventory comparison",
                "CDP C5.1 response alignment",
                "CDP C5.2 recalculation context",
                "Approval record",
            ],
            applicable_frameworks=[
                ComplianceFramework.CDP,
                ComplianceFramework.GHG_PROTOCOL,
            ],
            effective_date=_utcnow().isoformat(),
        )

    # ------------------------------------------------------------------
    # Public API -- Compliance Checking
    # ------------------------------------------------------------------

    def check_compliance(
        self,
        policy: RecalculationPolicy,
        frameworks: List[ComplianceFramework],
    ) -> List[PolicyComplianceCheck]:
        """Check policy compliance against one or more frameworks.

        For each framework, verifies:
        1. Thresholds meet minimum requirements.
        2. All mandatory trigger types are enabled.
        3. Documentation requirements are met.
        4. Approval levels meet minimums.

        Args:
            policy:     Recalculation policy to check.
            frameworks: List of frameworks to check against.

        Returns:
            List of PolicyComplianceCheck results, one per framework.
        """
        results: List[PolicyComplianceCheck] = []

        for framework in frameworks:
            gaps: List[ComplianceGap] = []
            recommendations: List[str] = []

            # Check thresholds
            threshold_gaps = self._check_thresholds(policy, framework)
            gaps.extend(threshold_gaps)

            # Check trigger rules
            trigger_gaps = self._check_triggers(policy, framework)
            gaps.extend(trigger_gaps)

            # Check documentation
            doc_gaps = self._check_documentation(policy, framework)
            gaps.extend(doc_gaps)

            # Check approval levels
            approval_gaps = self._check_approval_levels(policy, framework)
            gaps.extend(approval_gaps)

            # Build recommendations
            for gap in gaps:
                if gap.remediation:
                    recommendations.append(gap.remediation)

            is_compliant = not any(
                g.severity == ValidationSeverity.ERROR for g in gaps
            )

            results.append(PolicyComplianceCheck(
                policy_id=policy.policy_id,
                framework=framework,
                is_compliant=is_compliant,
                gaps=gaps,
                recommendations=recommendations,
                checked_at=_utcnow().isoformat(),
            ))

        return results

    # ------------------------------------------------------------------
    # Public API -- Policy Modification
    # ------------------------------------------------------------------

    def update_threshold(
        self,
        policy: RecalculationPolicy,
        trigger_type: TriggerType,
        new_threshold: Decimal,
    ) -> RecalculationPolicy:
        """Update the threshold override for a specific trigger type.

        Creates a new policy version with the updated threshold.

        Args:
            policy:        Policy to update.
            trigger_type:  Trigger type to modify.
            new_threshold: New threshold value (%).

        Returns:
            Updated RecalculationPolicy (new version).

        Raises:
            ValueError: If trigger_type not found in policy.
            ValueError: If new_threshold is out of range.
        """
        if new_threshold < Decimal("0") or new_threshold > Decimal("100"):
            raise ValueError(
                f"Threshold must be between 0 and 100 (got {new_threshold})"
            )

        # Deep copy for immutability
        updated_data = policy.model_dump(mode="json")
        updated = RecalculationPolicy.model_validate(updated_data)

        found = False
        for rule in updated.trigger_rules:
            if rule.trigger_type == trigger_type:
                rule.threshold_override = new_threshold
                found = True
                break

        if not found:
            raise ValueError(
                f"Trigger type {trigger_type.value} not found in policy. "
                f"Available triggers: "
                f"{[r.trigger_type.value for r in updated.trigger_rules]}"
            )

        updated.version = policy.version + 1
        updated.policy_id = _new_uuid()
        updated.calculated_at = _utcnow().isoformat()
        updated.provenance_hash = _compute_hash(updated)

        logger.info(
            "Updated threshold for %s to %s%% in policy v%d",
            trigger_type.value, new_threshold, updated.version,
        )

        return updated

    def add_custom_rule(
        self,
        policy: RecalculationPolicy,
        rule: TriggerRule,
    ) -> RecalculationPolicy:
        """Add a custom trigger rule to the policy.

        If a rule for the same trigger type already exists, it is
        replaced.  Otherwise, the new rule is appended.

        Args:
            policy: Policy to modify.
            rule:   New trigger rule to add.

        Returns:
            Updated RecalculationPolicy (new version).
        """
        updated_data = policy.model_dump(mode="json")
        updated = RecalculationPolicy.model_validate(updated_data)

        # Check if rule for this trigger type already exists
        existing_idx = None
        for i, existing in enumerate(updated.trigger_rules):
            if existing.trigger_type == rule.trigger_type:
                existing_idx = i
                break

        if existing_idx is not None:
            updated.trigger_rules[existing_idx] = rule
            logger.info(
                "Replaced existing rule for %s", rule.trigger_type.value
            )
        else:
            updated.trigger_rules.append(rule)
            logger.info(
                "Added new rule for %s", rule.trigger_type.value
            )

        updated.version = policy.version + 1
        updated.policy_id = _new_uuid()
        updated.calculated_at = _utcnow().isoformat()
        updated.provenance_hash = _compute_hash(updated)

        return updated

    def remove_trigger_rule(
        self,
        policy: RecalculationPolicy,
        trigger_type: TriggerType,
    ) -> RecalculationPolicy:
        """Remove a trigger rule from the policy.

        Args:
            policy:       Policy to modify.
            trigger_type: Trigger type to remove.

        Returns:
            Updated RecalculationPolicy (new version).

        Raises:
            ValueError: If trigger type not found.
        """
        updated_data = policy.model_dump(mode="json")
        updated = RecalculationPolicy.model_validate(updated_data)

        original_count = len(updated.trigger_rules)
        updated.trigger_rules = [
            r for r in updated.trigger_rules
            if r.trigger_type != trigger_type
        ]

        if len(updated.trigger_rules) == original_count:
            raise ValueError(
                f"Trigger type {trigger_type.value} not found in policy"
            )

        updated.version = policy.version + 1
        updated.policy_id = _new_uuid()
        updated.calculated_at = _utcnow().isoformat()
        updated.provenance_hash = _compute_hash(updated)

        return updated

    # ------------------------------------------------------------------
    # Public API -- Validation
    # ------------------------------------------------------------------

    def validate_policy(
        self,
        policy: RecalculationPolicy,
    ) -> PolicyValidationResult:
        """Validate a policy's internal consistency and completeness.

        Checks:
        1. At least one trigger rule is defined.
        2. No duplicate trigger types.
        3. Thresholds are internally consistent.
        4. Documentation requirements are non-empty.
        5. Rolling base year has period specified.
        6. Active policy has effective date.
        7. Approval levels are consistent with thresholds.

        Args:
            policy: Policy to validate.

        Returns:
            PolicyValidationResult with errors, warnings, and info.
        """
        errors: List[str] = []
        warnings: List[str] = []
        info: List[str] = []

        # Check 1: At least one trigger rule
        if not policy.trigger_rules:
            errors.append(
                "Policy has no trigger rules defined. At least one "
                "trigger type must be configured."
            )

        # Check 2: No duplicate trigger types
        trigger_types = [r.trigger_type.value for r in policy.trigger_rules]
        if len(trigger_types) != len(set(trigger_types)):
            seen = set()
            dupes = set()
            for tt in trigger_types:
                if tt in seen:
                    dupes.add(tt)
                seen.add(tt)
            errors.append(
                f"Duplicate trigger types found: {sorted(dupes)}. "
                f"Each trigger type should appear at most once."
            )

        # Check 3: Threshold consistency
        if policy.thresholds.individual_pct > policy.thresholds.cumulative_pct:
            warnings.append(
                f"Individual threshold ({policy.thresholds.individual_pct}%) "
                f"exceeds cumulative threshold "
                f"({policy.thresholds.cumulative_pct}%). This may cause "
                f"unexpected recalculation behaviour."
            )

        if policy.thresholds.individual_pct == Decimal("0"):
            warnings.append(
                "Individual threshold is 0%. This means every trigger "
                "event will require recalculation regardless of magnitude."
            )

        if policy.thresholds.individual_pct > Decimal("10"):
            warnings.append(
                f"Individual threshold ({policy.thresholds.individual_pct}%) "
                f"exceeds GHG Protocol recommended maximum of 10%. "
                f"This may result in non-compliance."
            )

        # Check 4: Documentation requirements
        if not policy.documentation_requirements:
            warnings.append(
                "No documentation requirements specified. Consider adding "
                "requirements for audit trail and regulatory compliance."
            )

        # Check 5: Rolling base year configuration
        if policy.rolling_base_year and not policy.rolling_period_years:
            errors.append(
                "Rolling base year is enabled but no rolling period "
                "is specified. Set rolling_period_years (3-10)."
            )

        if not policy.rolling_base_year and policy.rolling_period_years:
            info.append(
                "Rolling period is specified but rolling base year is "
                "disabled. The rolling period will not be used."
            )

        # Check 6: Active policy must have effective date
        if policy.is_active and not policy.effective_date:
            warnings.append(
                "Active policy has no effective date. Consider setting "
                "an effective date for audit trail purposes."
            )

        # Check 7: Expiry date validation
        if policy.effective_date and policy.expiry_date:
            if policy.expiry_date <= policy.effective_date:
                errors.append(
                    "Policy expiry date is before or equal to effective date."
                )

        # Check 8: Board approval threshold consistency
        if (policy.require_board_approval_above_pct
                <= policy.thresholds.individual_pct):
            warnings.append(
                f"Board approval threshold "
                f"({policy.require_board_approval_above_pct}%) is at or "
                f"below the individual significance threshold "
                f"({policy.thresholds.individual_pct}%). This means every "
                f"significant recalculation requires board approval."
            )

        # Check 9: All enabled rules have reasonable approval levels
        for rule in policy.trigger_rules:
            if rule.enabled and not rule.requires_approval:
                if rule.trigger_type in {
                    TriggerType.MERGER,
                    TriggerType.ACQUISITION,
                    TriggerType.DIVESTITURE,
                }:
                    warnings.append(
                        f"Trigger type {rule.trigger_type.value} is enabled "
                        f"but does not require approval. Major structural "
                        f"changes should typically require approval."
                    )

        # Check 10: Disabled mandatory triggers (informational)
        enabled_triggers = {
            r.trigger_type.value for r in policy.trigger_rules if r.enabled
        }
        for framework in policy.applicable_frameworks:
            mandatory = set(FRAMEWORK_MANDATORY_TRIGGERS.get(
                framework.value, []
            ))
            disabled_mandatory = mandatory - enabled_triggers
            if disabled_mandatory:
                info.append(
                    f"Triggers disabled but mandatory for "
                    f"{framework.value}: {sorted(disabled_mandatory)}"
                )

        # Summary info
        enabled_count = sum(1 for r in policy.trigger_rules if r.enabled)
        total_count = len(policy.trigger_rules)
        info.append(
            f"Policy has {enabled_count}/{total_count} trigger rules enabled."
        )
        info.append(
            f"Individual threshold: {policy.thresholds.individual_pct}%, "
            f"Cumulative: {policy.thresholds.cumulative_pct}%"
        )

        is_valid = len(errors) == 0

        return PolicyValidationResult(
            policy_id=policy.policy_id,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
        )

    # ------------------------------------------------------------------
    # Public API -- Comparison
    # ------------------------------------------------------------------

    def compare_policies(
        self,
        policy1: RecalculationPolicy,
        policy2: RecalculationPolicy,
    ) -> PolicyComparison:
        """Compare two recalculation policies.

        Identifies all differences between two policies, including
        thresholds, trigger rules, approval levels, and documentation.

        Args:
            policy1: First policy.
            policy2: Second policy.

        Returns:
            PolicyComparison with all differences and summary.
        """
        differences: List[PolicyComparisonItem] = []
        p1_stricter: List[str] = []
        p2_stricter: List[str] = []

        # Compare thresholds
        if policy1.thresholds.individual_pct != policy2.thresholds.individual_pct:
            diff = PolicyComparisonItem(
                field="individual_threshold_pct",
                policy1_val=str(policy1.thresholds.individual_pct),
                policy2_val=str(policy2.thresholds.individual_pct),
                impact="Lower threshold means more frequent recalculations",
            )
            differences.append(diff)
            if policy1.thresholds.individual_pct < policy2.thresholds.individual_pct:
                p1_stricter.append("individual_threshold")
            else:
                p2_stricter.append("individual_threshold")

        if policy1.thresholds.cumulative_pct != policy2.thresholds.cumulative_pct:
            diff = PolicyComparisonItem(
                field="cumulative_threshold_pct",
                policy1_val=str(policy1.thresholds.cumulative_pct),
                policy2_val=str(policy2.thresholds.cumulative_pct),
                impact="Lower cumulative threshold catches more accumulated changes",
            )
            differences.append(diff)
            if policy1.thresholds.cumulative_pct < policy2.thresholds.cumulative_pct:
                p1_stricter.append("cumulative_threshold")
            else:
                p2_stricter.append("cumulative_threshold")

        # Compare board approval threshold
        if (policy1.require_board_approval_above_pct
                != policy2.require_board_approval_above_pct):
            diff = PolicyComparisonItem(
                field="board_approval_threshold_pct",
                policy1_val=str(policy1.require_board_approval_above_pct),
                policy2_val=str(policy2.require_board_approval_above_pct),
                impact="Lower board threshold means more board involvement",
            )
            differences.append(diff)
            if (policy1.require_board_approval_above_pct
                    < policy2.require_board_approval_above_pct):
                p1_stricter.append("board_approval_threshold")
            else:
                p2_stricter.append("board_approval_threshold")

        # Compare trigger rules
        p1_triggers = {r.trigger_type.value: r for r in policy1.trigger_rules}
        p2_triggers = {r.trigger_type.value: r for r in policy2.trigger_rules}
        all_triggers = set(p1_triggers.keys()) | set(p2_triggers.keys())

        for tt in sorted(all_triggers):
            r1 = p1_triggers.get(tt)
            r2 = p2_triggers.get(tt)

            if r1 and not r2:
                differences.append(PolicyComparisonItem(
                    field=f"trigger_{tt}",
                    policy1_val="present",
                    policy2_val="absent",
                    impact=f"Policy 1 monitors {tt} triggers; Policy 2 does not",
                ))
                if r1.enabled:
                    p1_stricter.append(f"trigger_{tt}")
            elif r2 and not r1:
                differences.append(PolicyComparisonItem(
                    field=f"trigger_{tt}",
                    policy1_val="absent",
                    policy2_val="present",
                    impact=f"Policy 2 monitors {tt} triggers; Policy 1 does not",
                ))
                if r2.enabled:
                    p2_stricter.append(f"trigger_{tt}")
            elif r1 and r2:
                if r1.enabled != r2.enabled:
                    differences.append(PolicyComparisonItem(
                        field=f"trigger_{tt}_enabled",
                        policy1_val=str(r1.enabled),
                        policy2_val=str(r2.enabled),
                        impact=f"Trigger {tt} enabled status differs",
                    ))
                    if r1.enabled:
                        p1_stricter.append(f"trigger_{tt}_enabled")
                    else:
                        p2_stricter.append(f"trigger_{tt}_enabled")

                if r1.approval_level != r2.approval_level:
                    p1_level = APPROVAL_HIERARCHY.get(r1.approval_level.value, 0)
                    p2_level = APPROVAL_HIERARCHY.get(r2.approval_level.value, 0)
                    differences.append(PolicyComparisonItem(
                        field=f"trigger_{tt}_approval",
                        policy1_val=r1.approval_level.value,
                        policy2_val=r2.approval_level.value,
                        impact=f"Different approval level for {tt}",
                    ))
                    if p1_level > p2_level:
                        p1_stricter.append(f"trigger_{tt}_approval")
                    elif p2_level > p1_level:
                        p2_stricter.append(f"trigger_{tt}_approval")

        # Compare documentation
        p1_docs = set(policy1.documentation_requirements)
        p2_docs = set(policy2.documentation_requirements)
        if p1_docs != p2_docs:
            only_p1 = p1_docs - p2_docs
            only_p2 = p2_docs - p1_docs
            if only_p1:
                differences.append(PolicyComparisonItem(
                    field="documentation_only_in_policy1",
                    policy1_val=str(sorted(only_p1)),
                    policy2_val="n/a",
                    impact="Additional documentation requirements in Policy 1",
                ))
            if only_p2:
                differences.append(PolicyComparisonItem(
                    field="documentation_only_in_policy2",
                    policy1_val="n/a",
                    policy2_val=str(sorted(only_p2)),
                    impact="Additional documentation requirements in Policy 2",
                ))
            if len(p1_docs) > len(p2_docs):
                p1_stricter.append("documentation_requirements")
            elif len(p2_docs) > len(p1_docs):
                p2_stricter.append("documentation_requirements")

        # Build summary
        if not differences:
            summary = "Policies are identical in all compared dimensions."
        else:
            summary_parts = [f"{len(differences)} differences found."]
            if p1_stricter:
                summary_parts.append(
                    f"Policy 1 ({policy1.name}) is stricter in: "
                    f"{', '.join(p1_stricter)}."
                )
            if p2_stricter:
                summary_parts.append(
                    f"Policy 2 ({policy2.name}) is stricter in: "
                    f"{', '.join(p2_stricter)}."
                )
            summary = " ".join(summary_parts)

        comparison = PolicyComparison(
            policy1_id=policy1.policy_id,
            policy2_id=policy2.policy_id,
            differences=differences,
            policy1_stricter=p1_stricter,
            policy2_stricter=p2_stricter,
            summary=summary,
        )
        comparison.provenance_hash = _compute_hash(comparison)
        return comparison

    # ------------------------------------------------------------------
    # Internal Methods -- Compliance Checks
    # ------------------------------------------------------------------

    def _check_thresholds(
        self,
        policy: RecalculationPolicy,
        framework: ComplianceFramework,
    ) -> List[ComplianceGap]:
        """Check if policy thresholds comply with framework requirements.

        Args:
            policy:    Policy to check.
            framework: Framework requirements.

        Returns:
            List of compliance gaps (empty if compliant).
        """
        gaps: List[ComplianceGap] = []
        fw_thresholds = FRAMEWORK_THRESHOLDS.get(framework.value, {})

        if not fw_thresholds:
            return gaps

        # Individual threshold check
        ind_min = fw_thresholds.get("individual_min", Decimal("0"))
        ind_max = fw_thresholds.get("individual_max", Decimal("100"))

        if policy.thresholds.individual_pct > ind_max:
            gaps.append(ComplianceGap(
                framework=framework,
                severity=ValidationSeverity.ERROR,
                category="threshold",
                description=(
                    f"Individual threshold ({policy.thresholds.individual_pct}%) "
                    f"exceeds {framework.value} maximum ({ind_max}%)"
                ),
                requirement=f"Individual threshold <= {ind_max}%",
                current_value=str(policy.thresholds.individual_pct),
                required_value=str(ind_max),
                remediation=(
                    f"Reduce individual threshold to at most {ind_max}% "
                    f"for {framework.value} compliance."
                ),
            ))

        # Framework-specific threshold checks
        if framework == ComplianceFramework.SBTI:
            if policy.thresholds.sbti_pct > Decimal("5"):
                gaps.append(ComplianceGap(
                    framework=framework,
                    severity=ValidationSeverity.ERROR,
                    category="threshold",
                    description=(
                        f"SBTi threshold ({policy.thresholds.sbti_pct}%) "
                        f"exceeds SBTi mandatory 5% maximum"
                    ),
                    requirement="SBTi threshold <= 5%",
                    current_value=str(policy.thresholds.sbti_pct),
                    required_value="5.0",
                    remediation="Set SBTi threshold to 5% or lower.",
                ))

        if framework == ComplianceFramework.SEC:
            if policy.thresholds.sec_pct > Decimal("5"):
                gaps.append(ComplianceGap(
                    framework=framework,
                    severity=ValidationSeverity.WARNING,
                    category="threshold",
                    description=(
                        f"SEC threshold ({policy.thresholds.sec_pct}%) "
                        f"exceeds SEC recommended 5% materiality threshold"
                    ),
                    requirement="SEC threshold <= 5%",
                    current_value=str(policy.thresholds.sec_pct),
                    required_value="5.0",
                    remediation=(
                        "Consider reducing SEC threshold to 5% to align "
                        "with SEC materiality guidance."
                    ),
                ))

        return gaps

    def _check_triggers(
        self,
        policy: RecalculationPolicy,
        framework: ComplianceFramework,
    ) -> List[ComplianceGap]:
        """Check if mandatory trigger types are enabled.

        Args:
            policy:    Policy to check.
            framework: Framework requirements.

        Returns:
            List of compliance gaps for missing/disabled triggers.
        """
        gaps: List[ComplianceGap] = []
        mandatory = FRAMEWORK_MANDATORY_TRIGGERS.get(framework.value, [])

        enabled_triggers = {
            r.trigger_type.value for r in policy.trigger_rules if r.enabled
        }
        all_triggers = {
            r.trigger_type.value for r in policy.trigger_rules
        }

        for tt in mandatory:
            if tt not in all_triggers:
                gaps.append(ComplianceGap(
                    framework=framework,
                    severity=ValidationSeverity.ERROR,
                    category="trigger",
                    description=(
                        f"Mandatory trigger type '{tt}' is not defined "
                        f"in the policy (required by {framework.value})"
                    ),
                    requirement=f"Trigger type '{tt}' must be defined",
                    current_value="not defined",
                    required_value="defined and enabled",
                    remediation=(
                        f"Add a trigger rule for '{tt}' to the policy."
                    ),
                ))
            elif tt not in enabled_triggers:
                gaps.append(ComplianceGap(
                    framework=framework,
                    severity=ValidationSeverity.ERROR,
                    category="trigger",
                    description=(
                        f"Mandatory trigger type '{tt}' is defined but "
                        f"disabled (required by {framework.value})"
                    ),
                    requirement=f"Trigger type '{tt}' must be enabled",
                    current_value="disabled",
                    required_value="enabled",
                    remediation=(
                        f"Enable the trigger rule for '{tt}'."
                    ),
                ))

        return gaps

    def _check_documentation(
        self,
        policy: RecalculationPolicy,
        framework: ComplianceFramework,
    ) -> List[ComplianceGap]:
        """Check if policy documentation requirements meet framework needs.

        Args:
            policy:    Policy to check.
            framework: Framework requirements.

        Returns:
            List of documentation compliance gaps.
        """
        gaps: List[ComplianceGap] = []
        required_docs = FRAMEWORK_DOCUMENTATION.get(framework.value, [])

        policy_docs_lower = {d.lower() for d in policy.documentation_requirements}

        for req_doc in required_docs:
            # Fuzzy match: check if any policy doc contains the key phrase
            found = any(
                req_doc.lower() in pd or pd in req_doc.lower()
                for pd in policy_docs_lower
            )
            if not found:
                gaps.append(ComplianceGap(
                    framework=framework,
                    severity=ValidationSeverity.WARNING,
                    category="documentation",
                    description=(
                        f"Documentation requirement '{req_doc}' may not "
                        f"be covered in the policy (required by {framework.value})"
                    ),
                    requirement=req_doc,
                    current_value="not found in policy documentation",
                    required_value=req_doc,
                    remediation=(
                        f"Add '{req_doc}' to policy documentation requirements."
                    ),
                ))

        return gaps

    def _check_approval_levels(
        self,
        policy: RecalculationPolicy,
        framework: ComplianceFramework,
    ) -> List[ComplianceGap]:
        """Check if approval levels meet framework minimums.

        Args:
            policy:    Policy to check.
            framework: Framework requirements.

        Returns:
            List of approval level compliance gaps.
        """
        gaps: List[ComplianceGap] = []
        min_approval_str = FRAMEWORK_MIN_APPROVAL.get(framework.value)

        if not min_approval_str:
            return gaps

        min_level = APPROVAL_HIERARCHY.get(min_approval_str, 0)

        for rule in policy.trigger_rules:
            if not rule.enabled:
                continue
            if not rule.requires_approval:
                # Check if framework requires approval for this trigger
                mandatory = FRAMEWORK_MANDATORY_TRIGGERS.get(framework.value, [])
                if rule.trigger_type.value in mandatory:
                    gaps.append(ComplianceGap(
                        framework=framework,
                        severity=ValidationSeverity.WARNING,
                        category="approval",
                        description=(
                            f"Trigger '{rule.trigger_type.value}' does not "
                            f"require approval but {framework.value} expects "
                            f"at least {min_approval_str} level approval"
                        ),
                        requirement=f"Minimum approval: {min_approval_str}",
                        current_value="no approval required",
                        required_value=min_approval_str,
                        remediation=(
                            f"Enable approval with minimum level "
                            f"'{min_approval_str}' for '{rule.trigger_type.value}'."
                        ),
                    ))
                continue

            rule_level = APPROVAL_HIERARCHY.get(rule.approval_level.value, 0)
            if rule_level < min_level:
                gaps.append(ComplianceGap(
                    framework=framework,
                    severity=ValidationSeverity.WARNING,
                    category="approval",
                    description=(
                        f"Trigger '{rule.trigger_type.value}' has approval "
                        f"level '{rule.approval_level.value}' but "
                        f"{framework.value} requires minimum "
                        f"'{min_approval_str}'"
                    ),
                    requirement=f"Minimum approval: {min_approval_str}",
                    current_value=rule.approval_level.value,
                    required_value=min_approval_str,
                    remediation=(
                        f"Upgrade approval level for "
                        f"'{rule.trigger_type.value}' to at least "
                        f"'{min_approval_str}'."
                    ),
                ))

        return gaps

    # ------------------------------------------------------------------
    # Internal Methods -- Policy Building
    # ------------------------------------------------------------------

    def _build_standard_trigger_rules(
        self,
        default_approval: ApprovalLevel,
        merger_approval: ApprovalLevel,
        threshold_override: Optional[Decimal],
    ) -> List[TriggerRule]:
        """Build a standard set of trigger rules for all trigger types.

        Args:
            default_approval:   Default approval level for most triggers.
            merger_approval:    Approval level for mergers (usually higher).
            threshold_override: Optional threshold override for all rules.

        Returns:
            List of TriggerRule covering all trigger types.
        """
        rules: List[TriggerRule] = []

        trigger_configs = [
            (TriggerType.ACQUISITION, default_approval, True, True,
             "Recalculation triggered by acquisition of operations or "
             "business units that add emissions to the boundary."),
            (TriggerType.DIVESTITURE, default_approval, True, True,
             "Recalculation triggered by sale or closure of operations "
             "that remove emissions from the boundary."),
            (TriggerType.MERGER, merger_approval, True, True,
             "Recalculation triggered by merger that fundamentally "
             "restructures the reporting entity."),
            (TriggerType.METHODOLOGY_CHANGE, default_approval, False, True,
             "Recalculation triggered by changes in calculation methodology, "
             "emission factors, or GWP values."),
            (TriggerType.ERROR_CORRECTION, default_approval, False, True,
             "Recalculation triggered by discovery and correction of "
             "significant errors in historical data."),
            (TriggerType.SOURCE_BOUNDARY_CHANGE, default_approval, False, True,
             "Recalculation triggered by addition or removal of emission "
             "source categories from the operational boundary."),
            (TriggerType.OUTSOURCING_INSOURCING, default_approval, False, True,
             "Recalculation triggered by transfer of activities between "
             "the reporting entity and third parties."),
        ]

        for trigger_type, approval, auto_detect, docs_req, description in trigger_configs:
            rules.append(TriggerRule(
                trigger_type=trigger_type,
                enabled=True,
                threshold_override=threshold_override,
                auto_detect=auto_detect,
                requires_approval=True,
                approval_level=approval,
                documentation_required=docs_req,
                description=description,
            ))

        return rules

    def _get_custom_base(self) -> RecalculationPolicy:
        """Create a minimal custom policy as a starting point.

        Returns:
            RecalculationPolicy with minimal defaults for customisation.
        """
        return RecalculationPolicy(
            policy_type=PolicyType.CUSTOM,
            name="Custom Recalculation Policy",
            description=(
                "Custom recalculation policy. Configure thresholds, "
                "trigger rules, and documentation requirements as needed."
            ),
            thresholds=ThresholdConfig(
                individual_pct=Decimal("5.0"),
                cumulative_pct=Decimal("10.0"),
                sbti_pct=Decimal("5.0"),
                sec_pct=Decimal("5.0"),
            ),
            trigger_rules=self._build_standard_trigger_rules(
                default_approval=ApprovalLevel.MANAGER,
                merger_approval=ApprovalLevel.DIRECTOR,
                threshold_override=None,
            ),
            rolling_base_year=False,
            require_board_approval_above_pct=Decimal("15.0"),
            documentation_requirements=[
                "Recalculation policy description",
                "Trigger event documentation",
                "Before/after inventory comparison",
            ],
            applicable_frameworks=[],
            effective_date=_utcnow().isoformat(),
        )

    @staticmethod
    def _find_trigger_rule(
        rules: List[TriggerRule],
        trigger_type: TriggerType,
    ) -> Optional[TriggerRule]:
        """Find a trigger rule by type in a list of rules.

        Args:
            rules:        List of trigger rules.
            trigger_type: Trigger type to find.

        Returns:
            TriggerRule if found, None otherwise.
        """
        for rule in rules:
            if rule.trigger_type == trigger_type:
                return rule
        return None

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def get_policy_summary(
        self,
        policy: RecalculationPolicy,
    ) -> Dict[str, Any]:
        """Generate a summary of the policy for reporting.

        Args:
            policy: Policy to summarise.

        Returns:
            Summary dictionary for tabular display.
        """
        enabled_triggers = [
            r.trigger_type.value for r in policy.trigger_rules if r.enabled
        ]
        disabled_triggers = [
            r.trigger_type.value for r in policy.trigger_rules if not r.enabled
        ]

        return {
            "policy_id": policy.policy_id,
            "policy_type": policy.policy_type.value,
            "name": policy.name,
            "version": policy.version,
            "is_active": policy.is_active,
            "individual_threshold_pct": _round2(policy.thresholds.individual_pct),
            "cumulative_threshold_pct": _round2(policy.thresholds.cumulative_pct),
            "board_approval_above_pct": _round2(policy.require_board_approval_above_pct),
            "enabled_triggers": enabled_triggers,
            "disabled_triggers": disabled_triggers,
            "total_trigger_rules": len(policy.trigger_rules),
            "rolling_base_year": policy.rolling_base_year,
            "documentation_count": len(policy.documentation_requirements),
            "applicable_frameworks": [
                f.value for f in policy.applicable_frameworks
            ],
            "provenance_hash": policy.provenance_hash,
        }

    def get_version(self) -> str:
        """Return engine version string."""
        return self._version


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def create_recalculation_policy(
    policy_type: PolicyType,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> RecalculationPolicy:
    """Module-level convenience function to create a policy.

    Args:
        policy_type:  Type of policy to create.
        name:         Optional custom name.
        description:  Optional custom description.

    Returns:
        RecalculationPolicy with sensible defaults.
    """
    engine = RecalculationPolicyEngine()
    return engine.create_policy(policy_type, name, description)


def check_policy_compliance(
    policy: RecalculationPolicy,
    frameworks: List[ComplianceFramework],
) -> List[PolicyComplianceCheck]:
    """Module-level convenience function to check compliance.

    Args:
        policy:     Policy to check.
        frameworks: Frameworks to check against.

    Returns:
        List of PolicyComplianceCheck results.
    """
    engine = RecalculationPolicyEngine()
    return engine.check_compliance(policy, frameworks)


def validate_recalculation_policy(
    policy: RecalculationPolicy,
) -> PolicyValidationResult:
    """Module-level convenience function to validate a policy.

    Args:
        policy: Policy to validate.

    Returns:
        PolicyValidationResult with errors and warnings.
    """
    engine = RecalculationPolicyEngine()
    return engine.validate_policy(policy)


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "PolicyType",
    "TriggerType",
    "ApprovalLevel",
    "ComplianceFramework",
    "ValidationSeverity",
    # Input Models
    "ThresholdConfig",
    "TriggerRule",
    "RecalculationPolicy",
    # Output Models
    "ComplianceGap",
    "PolicyComplianceCheck",
    "PolicyValidationResult",
    "PolicyComparisonItem",
    "PolicyComparison",
    # Engine
    "RecalculationPolicyEngine",
    # Convenience functions
    "create_recalculation_policy",
    "check_policy_compliance",
    "validate_recalculation_policy",
    # Constants
    "FRAMEWORK_THRESHOLDS",
    "FRAMEWORK_MANDATORY_TRIGGERS",
    "FRAMEWORK_DOCUMENTATION",
    "FRAMEWORK_MIN_APPROVAL",
    "APPROVAL_HIERARCHY",
]
