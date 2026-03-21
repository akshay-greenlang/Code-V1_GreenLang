# -*- coding: utf-8 -*-
"""
RecalculationEngine - PACK-023 SBTi Alignment Engine 8
=========================================================

Implements base year recalculation logic for SBTi targets when
structural changes occur.  The SBTi Corporate Manual V5.3 requires
organizations to recalculate their base year emissions when changes
exceed the 5% significance threshold, ensuring that targets remain
meaningful and comparable over time.

Recalculation Triggers (SBTi Corporate Manual V5.3):
    - ACQUISITION: Acquiring another entity (>5% emissions impact)
    - DIVESTITURE: Selling/spinning-off business units (>5%)
    - MERGER: Corporate merger changing organizational boundary
    - METHODOLOGY_CHANGE: Updated emission factors or calculation methods
    - STRUCTURAL_CHANGE: Outsourcing, insourcing, organizational restructuring
    - ORGANIC_GROWTH: Growth in emissions unrelated to structural changes (>5%)

Significance Assessment:
    The engine calculates the significance of each change as a percentage
    of base year emissions.  If the absolute percentage change exceeds
    the 5% significance threshold, a recalculation is required.

Base Year Recalculation Approach:
    When a recalculation is triggered, the engine adjusts base year
    emissions to reflect the structural change as if it had occurred
    in the base year.  This ensures like-for-like comparison between
    base year and current year emissions.

Target Adjustment:
    After recalculating the base year, all targets (near-term, long-term,
    net-zero) are proportionally adjusted to maintain the same reduction
    percentage relative to the new base year emissions.

Regulatory References:
    - SBTi Corporate Manual V5.3 (2024), Section 8: Recalculation
    - SBTi Corporate Net-Zero Standard V1.3 (2024)
    - GHG Protocol Corporate Standard Chapter 5: Tracking Emissions
    - ISO 14064-1:2018 Section 8: Base Year Recalculation

Zero-Hallucination:
    - Significance calculated as deterministic percentage of base emissions
    - Adjustments use proportional arithmetic only
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-023 SBTi Alignment
Engine:  8 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
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
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning default on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(
    part: Decimal, whole: Decimal, places: int = 2
) -> Decimal:
    """Calculate percentage safely, returning 0 on zero denominator."""
    if whole == Decimal("0"):
        return Decimal("0")
    return (part / whole * Decimal("100")).quantize(
        Decimal("0." + "0" * places), rounding=ROUND_HALF_UP
    )


def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal value to the specified number of decimal places."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SBTi significance threshold for base year recalculation
SIGNIFICANCE_THRESHOLD: Decimal = Decimal("0.05")  # 5%

# Borderline band: within 1% of the threshold on either side
BORDERLINE_LOWER: Decimal = Decimal("0.04")
BORDERLINE_UPPER: Decimal = Decimal("0.06")

# Triggers that always require methodology review
METHODOLOGY_TRIGGERS: List[str] = [
    "emission_factor_update",
    "gwp_value_change",
    "calculation_methodology_change",
    "scope_boundary_change",
    "data_quality_improvement",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RecalculationTrigger(str, Enum):
    """Type of change that may trigger a base year recalculation."""
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    METHODOLOGY_CHANGE = "methodology_change"
    STRUCTURAL_CHANGE = "structural_change"
    ORGANIC_GROWTH = "organic_growth"


class RecalculationStatus(str, Enum):
    """Whether recalculation is required."""
    REQUIRED = "required"
    NOT_REQUIRED = "not_required"
    PENDING_REVIEW = "pending_review"


class SignificanceLevel(str, Enum):
    """Significance of the change relative to threshold."""
    ABOVE_THRESHOLD = "above_threshold"
    BELOW_THRESHOLD = "below_threshold"
    BORDERLINE = "borderline"


class TargetScope(str, Enum):
    """Target scope classification."""
    S1S2 = "s1s2"
    S3 = "s3"
    S1S2S3 = "s1s2s3"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class TargetDefinition(BaseModel):
    """An existing SBTi target that may need adjustment."""
    target_id: str = Field(default_factory=_new_uuid, description="Target ID")
    target_name: str = Field(default="", description="Target name")
    scope: TargetScope = Field(default=TargetScope.S1S2, description="Scope")
    base_year: int = Field(description="Original base year")
    target_year: int = Field(description="Target year")
    base_year_emissions: Decimal = Field(description="Original base year emissions (tCO2e)")
    target_year_emissions: Decimal = Field(description="Target year emissions (tCO2e)")
    reduction_pct: Decimal = Field(description="Original reduction percentage")
    is_validated: bool = Field(default=False, description="SBTi validated flag")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("base_year_emissions", "target_year_emissions",
                     "reduction_pct", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class TargetAdjustment(BaseModel):
    """Adjustment made to a target after base year recalculation."""
    target_id: str = Field(description="Original target ID")
    target_name: str = Field(default="", description="Target name")
    scope: TargetScope = Field(default=TargetScope.S1S2, description="Scope")
    original_base_emissions: Decimal = Field(
        description="Original base year emissions (tCO2e)"
    )
    adjusted_base_emissions: Decimal = Field(
        description="Adjusted base year emissions (tCO2e)"
    )
    original_target_emissions: Decimal = Field(
        description="Original target year emissions (tCO2e)"
    )
    adjusted_target_emissions: Decimal = Field(
        description="Adjusted target year emissions (tCO2e)"
    )
    original_reduction_pct: Decimal = Field(
        description="Original reduction percentage"
    )
    adjusted_reduction_pct: Decimal = Field(
        description="Adjusted reduction percentage (should match original)"
    )
    adjustment_delta_tco2e: Decimal = Field(
        description="Change in base year emissions (tCO2e)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("original_base_emissions", "adjusted_base_emissions",
                     "original_target_emissions", "adjusted_target_emissions",
                     "original_reduction_pct", "adjusted_reduction_pct",
                     "adjustment_delta_tco2e", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class AuditEntry(BaseModel):
    """An entry in the recalculation audit trail."""
    entry_id: str = Field(default_factory=_new_uuid, description="Entry ID")
    timestamp: datetime = Field(default_factory=_utcnow, description="Timestamp")
    action: str = Field(description="Action performed")
    trigger: RecalculationTrigger = Field(description="Trigger type")
    details: str = Field(default="", description="Detailed description")
    old_value: str = Field(default="", description="Value before change")
    new_value: str = Field(default="", description="Value after change")
    performed_by: str = Field(default="system", description="Actor")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class RecalculationInput(BaseModel):
    """Input data for base year recalculation assessment."""
    assessment_id: str = Field(default_factory=_new_uuid, description="Assessment ID")
    entity_name: str = Field(description="Organization name")
    entity_id: str = Field(default_factory=_new_uuid, description="Entity ID")
    trigger_type: RecalculationTrigger = Field(
        description="Type of change triggering assessment"
    )
    change_description: str = Field(
        default="", description="Description of the structural change"
    )
    change_date: datetime = Field(
        default_factory=_utcnow, description="Date of the structural change"
    )
    pre_change_emissions: Decimal = Field(
        description="Emissions before the change (tCO2e)"
    )
    post_change_emissions: Decimal = Field(
        description="Emissions after the change (tCO2e)"
    )
    base_year_emissions: Decimal = Field(
        description="Current base year emissions (tCO2e)"
    )
    scope: TargetScope = Field(
        default=TargetScope.S1S2, description="Scope affected"
    )
    existing_targets: List[TargetDefinition] = Field(
        default_factory=list, description="Existing targets to adjust"
    )
    additional_context: str = Field(
        default="", description="Additional context or justification"
    )
    requested_at: datetime = Field(default_factory=_utcnow, description="Request timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("pre_change_emissions", "post_change_emissions",
                     "base_year_emissions", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class RecalculationResult(BaseModel):
    """Complete result of a recalculation assessment."""
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    entity_name: str = Field(default="", description="Organization name")
    entity_id: str = Field(default="", description="Entity ID")
    trigger: RecalculationTrigger = Field(description="Trigger type")
    change_description: str = Field(default="", description="Change description")
    emissions_change_tco2e: Decimal = Field(
        description="Absolute emissions change (tCO2e)"
    )
    significance_pct: Decimal = Field(
        description="Change as percentage of base year emissions"
    )
    significance_level: SignificanceLevel = Field(
        description="Significance classification"
    )
    recalculation_required: bool = Field(
        description="Whether recalculation is required"
    )
    recalculation_status: RecalculationStatus = Field(
        description="Recalculation status"
    )
    original_base_year_emissions: Decimal = Field(
        description="Original base year emissions (tCO2e)"
    )
    adjusted_base_year_emissions: Decimal = Field(
        description="Adjusted base year emissions (tCO2e)"
    )
    adjusted_targets: List[TargetAdjustment] = Field(
        default_factory=list, description="Adjusted targets"
    )
    audit_trail: List[AuditEntry] = Field(
        default_factory=list, description="Audit trail entries"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator("emissions_change_tco2e", "significance_pct",
                     "original_base_year_emissions",
                     "adjusted_base_year_emissions", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------


class RecalculationConfig(BaseModel):
    """Configuration for the RecalculationEngine."""
    significance_threshold: Decimal = Field(
        default=SIGNIFICANCE_THRESHOLD,
        description="Significance threshold (fraction, e.g. 0.05 = 5%)",
    )
    borderline_lower: Decimal = Field(
        default=BORDERLINE_LOWER,
        description="Lower borderline threshold (fraction)",
    )
    borderline_upper: Decimal = Field(
        default=BORDERLINE_UPPER,
        description="Upper borderline threshold (fraction)",
    )
    score_precision: int = Field(
        default=4, description="Decimal places for calculated values"
    )
    auto_recalculate_methodology: bool = Field(
        default=True,
        description="Automatically flag methodology changes for recalculation",
    )
    auto_recalculate_merger: bool = Field(
        default=True,
        description="Automatically flag mergers for recalculation",
    )


# ---------------------------------------------------------------------------
# Pydantic model_rebuild
# ---------------------------------------------------------------------------

TargetDefinition.model_rebuild()
TargetAdjustment.model_rebuild()
AuditEntry.model_rebuild()
RecalculationInput.model_rebuild()
RecalculationResult.model_rebuild()
RecalculationConfig.model_rebuild()


# ---------------------------------------------------------------------------
# RecalculationEngine
# ---------------------------------------------------------------------------


class RecalculationEngine:
    """
    Base year recalculation engine for SBTi targets.

    Assesses whether structural changes (acquisitions, divestitures,
    mergers, methodology changes) exceed the 5% significance threshold
    and, if so, recalculates the base year emissions and adjusts all
    targets proportionally.

    Attributes:
        config: Engine configuration.
        _history: List of past recalculation results for audit.

    Example:
        >>> engine = RecalculationEngine()
        >>> inp = RecalculationInput(
        ...     entity_name="Acme Corp",
        ...     trigger_type=RecalculationTrigger.ACQUISITION,
        ...     pre_change_emissions=100000,
        ...     post_change_emissions=115000,
        ...     base_year_emissions=100000,
        ... )
        >>> result = engine.assess_trigger(inp)
        >>> assert result.recalculation_required is True
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RecalculationEngine.

        Args:
            config: Optional configuration dictionary or RecalculationConfig.
        """
        if config and isinstance(config, dict):
            self.config = RecalculationConfig(**config)
        elif config and isinstance(config, RecalculationConfig):
            self.config = config
        else:
            self.config = RecalculationConfig()

        self._history: List[RecalculationResult] = []
        logger.info("RecalculationEngine initialized (v%s)", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Core Assessment
    # -------------------------------------------------------------------

    def assess_trigger(self, inp: RecalculationInput) -> RecalculationResult:
        """Assess whether a structural change triggers base year recalculation.

        This is the primary entry point.  It evaluates the significance
        of the change, determines whether recalculation is required, and
        if so, performs the recalculation and adjusts all targets.

        Args:
            inp: RecalculationInput with change details and existing targets.

        Returns:
            RecalculationResult with full assessment and adjusted targets.

        Raises:
            ValueError: If input data is invalid.
        """
        self._validate_input(inp)
        audit_trail: List[AuditEntry] = []

        # Step 1: Calculate significance
        significance_pct, significance_level = self.calculate_significance(inp)
        audit_trail.append(AuditEntry(
            action="significance_assessment",
            trigger=inp.trigger_type,
            details=f"Change significance: {significance_pct}% of base year emissions",
            old_value=str(inp.pre_change_emissions),
            new_value=str(inp.post_change_emissions),
        ))

        # Step 2: Determine if recalculation is required
        recalc_required, recalc_status = self._determine_recalculation_need(
            inp.trigger_type, significance_level
        )
        audit_trail.append(AuditEntry(
            action="recalculation_decision",
            trigger=inp.trigger_type,
            details=f"Recalculation {recalc_status.value}: significance={significance_level.value}",
            old_value="",
            new_value=recalc_status.value,
        ))

        # Step 3: Calculate adjusted base year emissions
        emissions_change = inp.post_change_emissions - inp.pre_change_emissions
        adjusted_base = inp.base_year_emissions
        adjusted_targets: List[TargetAdjustment] = []

        if recalc_required:
            adjusted_base = self.recalculate_base_year(inp)
            audit_trail.append(AuditEntry(
                action="base_year_recalculation",
                trigger=inp.trigger_type,
                details="Base year emissions recalculated",
                old_value=str(inp.base_year_emissions),
                new_value=str(adjusted_base),
            ))

            # Step 4: Adjust targets
            adjusted_targets = self.adjust_targets(
                inp.existing_targets, inp.base_year_emissions, adjusted_base
            )
            for adj in adjusted_targets:
                audit_trail.append(AuditEntry(
                    action="target_adjustment",
                    trigger=inp.trigger_type,
                    details=f"Target {adj.target_name} adjusted",
                    old_value=str(adj.original_base_emissions),
                    new_value=str(adj.adjusted_base_emissions),
                ))

        # Step 5: Generate recommendations
        recommendations = self._generate_recommendations(
            inp.trigger_type, significance_level, recalc_required
        )

        # Step 6: Finalize audit trail
        audit_trail = self.generate_audit_trail(audit_trail)

        result = RecalculationResult(
            entity_name=inp.entity_name,
            entity_id=inp.entity_id,
            trigger=inp.trigger_type,
            change_description=inp.change_description,
            emissions_change_tco2e=_round_val(emissions_change, self.config.score_precision),
            significance_pct=significance_pct,
            significance_level=significance_level,
            recalculation_required=recalc_required,
            recalculation_status=recalc_status,
            original_base_year_emissions=inp.base_year_emissions,
            adjusted_base_year_emissions=adjusted_base,
            adjusted_targets=adjusted_targets,
            audit_trail=audit_trail,
            recommendations=recommendations,
        )
        result.provenance_hash = _compute_hash(result)
        self._history.append(result)

        logger.info(
            "Recalculation assessment for %s: trigger=%s, significance=%.2f%%, required=%s",
            inp.entity_name, inp.trigger_type.value,
            float(significance_pct), str(recalc_required),
        )
        return result

    # -------------------------------------------------------------------
    # Input Validation
    # -------------------------------------------------------------------

    def _validate_input(self, inp: RecalculationInput) -> None:
        """Validate input data.

        Args:
            inp: RecalculationInput to validate.

        Raises:
            ValueError: If input is invalid.
        """
        if inp.base_year_emissions <= Decimal("0"):
            raise ValueError("Base year emissions must be positive")
        if inp.pre_change_emissions < Decimal("0"):
            raise ValueError("Pre-change emissions cannot be negative")
        if inp.post_change_emissions < Decimal("0"):
            raise ValueError("Post-change emissions cannot be negative")

    # -------------------------------------------------------------------
    # Significance Calculation
    # -------------------------------------------------------------------

    def calculate_significance(
        self, inp: RecalculationInput
    ) -> Tuple[Decimal, SignificanceLevel]:
        """Calculate the significance of the structural change.

        Significance = abs(post_change - pre_change) / base_year * 100

        Args:
            inp: RecalculationInput.

        Returns:
            Tuple of (significance_pct, SignificanceLevel).
        """
        abs_change = abs(inp.post_change_emissions - inp.pre_change_emissions)
        significance_fraction = _safe_divide(abs_change, inp.base_year_emissions)
        significance_pct = _round_val(significance_fraction * Decimal("100"), 2)

        level = self._classify_significance(significance_fraction)

        logger.info(
            "Significance calculation: change=%.2f tCO2e, base=%.2f tCO2e, pct=%.2f%%",
            float(abs_change), float(inp.base_year_emissions),
            float(significance_pct),
        )
        return significance_pct, level

    def _classify_significance(
        self, significance_fraction: Decimal
    ) -> SignificanceLevel:
        """Classify a significance fraction into a level.

        Args:
            significance_fraction: Significance as a fraction (e.g. 0.05).

        Returns:
            SignificanceLevel.
        """
        if significance_fraction >= self.config.significance_threshold:
            return SignificanceLevel.ABOVE_THRESHOLD
        elif significance_fraction >= self.config.borderline_lower:
            return SignificanceLevel.BORDERLINE
        else:
            return SignificanceLevel.BELOW_THRESHOLD

    # -------------------------------------------------------------------
    # Recalculation Decision
    # -------------------------------------------------------------------

    def _determine_recalculation_need(
        self,
        trigger: RecalculationTrigger,
        significance: SignificanceLevel,
    ) -> Tuple[bool, RecalculationStatus]:
        """Determine whether recalculation is required.

        Args:
            trigger: Type of change.
            significance: Significance level.

        Returns:
            Tuple of (required, RecalculationStatus).
        """
        # Mergers always require recalculation
        if trigger == RecalculationTrigger.MERGER and self.config.auto_recalculate_merger:
            return True, RecalculationStatus.REQUIRED

        # Methodology changes: configurable auto-recalculation
        if trigger == RecalculationTrigger.METHODOLOGY_CHANGE:
            if self.config.auto_recalculate_methodology:
                return True, RecalculationStatus.REQUIRED
            return False, RecalculationStatus.PENDING_REVIEW

        # Standard significance-based decision
        if significance == SignificanceLevel.ABOVE_THRESHOLD:
            return True, RecalculationStatus.REQUIRED
        elif significance == SignificanceLevel.BORDERLINE:
            return False, RecalculationStatus.PENDING_REVIEW
        else:
            return False, RecalculationStatus.NOT_REQUIRED

    # -------------------------------------------------------------------
    # Base Year Recalculation
    # -------------------------------------------------------------------

    def recalculate_base_year(self, inp: RecalculationInput) -> Decimal:
        """Recalculate base year emissions to reflect the structural change.

        The adjustment accounts for the change as if it had occurred in
        the base year.  For acquisitions, the acquired entity's base-year
        emissions are added.  For divestitures, the divested entity's
        emissions are removed.

        Args:
            inp: RecalculationInput with change details.

        Returns:
            Adjusted base year emissions (tCO2e).
        """
        change = inp.post_change_emissions - inp.pre_change_emissions

        if inp.trigger_type == RecalculationTrigger.ACQUISITION:
            # Add acquired emissions to base year
            adjusted = inp.base_year_emissions + change
        elif inp.trigger_type == RecalculationTrigger.DIVESTITURE:
            # Remove divested emissions from base year
            adjusted = inp.base_year_emissions + change  # change is negative
        elif inp.trigger_type == RecalculationTrigger.MERGER:
            # Merger: combine both entities' base year emissions
            adjusted = inp.base_year_emissions + change
        elif inp.trigger_type == RecalculationTrigger.METHODOLOGY_CHANGE:
            # Apply the methodology correction factor
            if inp.pre_change_emissions > Decimal("0"):
                correction_factor = _safe_divide(
                    inp.post_change_emissions,
                    inp.pre_change_emissions,
                    Decimal("1"),
                )
                adjusted = inp.base_year_emissions * correction_factor
            else:
                adjusted = inp.base_year_emissions
        elif inp.trigger_type == RecalculationTrigger.STRUCTURAL_CHANGE:
            adjusted = inp.base_year_emissions + change
        elif inp.trigger_type == RecalculationTrigger.ORGANIC_GROWTH:
            # Organic growth does not adjust base year (targets remain same)
            # Only structural changes adjust the base year
            adjusted = inp.base_year_emissions
        else:
            adjusted = inp.base_year_emissions + change

        adjusted = max(adjusted, Decimal("0"))
        adjusted = _round_val(adjusted, self.config.score_precision)

        logger.info(
            "Base year recalculated: %.2f -> %.2f tCO2e (trigger=%s)",
            float(inp.base_year_emissions), float(adjusted),
            inp.trigger_type.value,
        )
        return adjusted

    # -------------------------------------------------------------------
    # Target Adjustment
    # -------------------------------------------------------------------

    def adjust_targets(
        self,
        targets: List[TargetDefinition],
        original_base: Decimal,
        adjusted_base: Decimal,
    ) -> List[TargetAdjustment]:
        """Adjust all targets proportionally after base year recalculation.

        Each target maintains its original reduction percentage, but the
        absolute emissions values are recalculated relative to the
        adjusted base year emissions.

        Args:
            targets: List of existing target definitions.
            original_base: Original base year emissions.
            adjusted_base: Adjusted base year emissions.

        Returns:
            List of TargetAdjustment.
        """
        adjustments: List[TargetAdjustment] = []

        for target in targets:
            # Maintain original reduction percentage
            reduction_pct = target.reduction_pct
            if reduction_pct <= Decimal("0"):
                # Calculate from emissions if not explicitly set
                reduction_pct = _safe_pct(
                    target.base_year_emissions - target.target_year_emissions,
                    target.base_year_emissions,
                )

            # Calculate adjusted target year emissions
            adjusted_target_emissions = adjusted_base * (
                Decimal("1") - reduction_pct / Decimal("100")
            )
            adjusted_target_emissions = _round_val(
                adjusted_target_emissions, self.config.score_precision
            )

            # Verify reduction percentage is maintained
            adjusted_reduction_pct = _safe_pct(
                adjusted_base - adjusted_target_emissions,
                adjusted_base,
            )

            delta = adjusted_base - target.base_year_emissions
            delta = _round_val(delta, self.config.score_precision)

            adj = TargetAdjustment(
                target_id=target.target_id,
                target_name=target.target_name,
                scope=target.scope,
                original_base_emissions=target.base_year_emissions,
                adjusted_base_emissions=adjusted_base,
                original_target_emissions=target.target_year_emissions,
                adjusted_target_emissions=adjusted_target_emissions,
                original_reduction_pct=reduction_pct,
                adjusted_reduction_pct=adjusted_reduction_pct,
                adjustment_delta_tco2e=delta,
            )
            adj.provenance_hash = _compute_hash(adj)
            adjustments.append(adj)

            logger.info(
                "Target %s adjusted: base %.2f->%.2f, target %.2f->%.2f, reduction %.2f%%",
                target.target_name,
                float(target.base_year_emissions), float(adjusted_base),
                float(target.target_year_emissions), float(adjusted_target_emissions),
                float(adjusted_reduction_pct),
            )

        return adjustments

    # -------------------------------------------------------------------
    # Audit Trail
    # -------------------------------------------------------------------

    def generate_audit_trail(
        self, entries: List[AuditEntry]
    ) -> List[AuditEntry]:
        """Add provenance hashes to audit entries.

        Args:
            entries: List of AuditEntry objects.

        Returns:
            List of AuditEntry with provenance hashes.
        """
        for entry in entries:
            entry.provenance_hash = _compute_hash(entry)
        return entries

    # -------------------------------------------------------------------
    # Recommendations
    # -------------------------------------------------------------------

    def _generate_recommendations(
        self,
        trigger: RecalculationTrigger,
        significance: SignificanceLevel,
        recalc_required: bool,
    ) -> List[str]:
        """Generate recommendations based on assessment outcome.

        Args:
            trigger: Trigger type.
            significance: Significance level.
            recalc_required: Whether recalculation is required.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if recalc_required:
            recs.append(
                "Base year recalculation is REQUIRED. Update all target documentation "
                "to reflect the adjusted base year emissions."
            )
            recs.append(
                "Notify the SBTi of the structural change and submit updated target "
                "information for re-validation within the next reporting cycle."
            )
            recs.append(
                "Update all internal and external disclosures (CDP, TCFD, CSRD) to "
                "reflect the recalculated base year."
            )

        if significance == SignificanceLevel.BORDERLINE:
            recs.append(
                "The change is near the 5% significance threshold. Consider voluntary "
                "recalculation for transparency and comparability."
            )
            recs.append(
                "Document the rationale for the recalculation decision in the "
                "organization's GHG management plan."
            )

        if trigger == RecalculationTrigger.ACQUISITION:
            recs.append(
                "Integrate the acquired entity's historical emissions data into the "
                "base year inventory. Ensure consistent boundary treatment."
            )
        elif trigger == RecalculationTrigger.DIVESTITURE:
            recs.append(
                "Remove the divested entity's emissions from the base year inventory. "
                "Retain documentation of the original boundary."
            )
        elif trigger == RecalculationTrigger.MERGER:
            recs.append(
                "Establish a combined base year inventory reflecting both entities. "
                "Align on consistent methodology and emission factors."
            )
        elif trigger == RecalculationTrigger.METHODOLOGY_CHANGE:
            recs.append(
                "Apply the updated methodology consistently to both the base year and "
                "all subsequent reporting years."
            )
        elif trigger == RecalculationTrigger.ORGANIC_GROWTH:
            recs.append(
                "Organic growth does not typically trigger base year recalculation. "
                "Monitor whether intensity targets are more appropriate."
            )

        if not recalc_required and significance == SignificanceLevel.BELOW_THRESHOLD:
            recs.append(
                "No recalculation required. Document the change and monitor cumulative "
                "impacts for future reporting periods."
            )

        return recs

    # -------------------------------------------------------------------
    # Batch Assessment
    # -------------------------------------------------------------------

    def assess_multiple_triggers(
        self, inputs: List[RecalculationInput]
    ) -> List[RecalculationResult]:
        """Assess multiple recalculation triggers.

        Args:
            inputs: List of RecalculationInput.

        Returns:
            List of RecalculationResult.
        """
        results: List[RecalculationResult] = []
        for inp in inputs:
            try:
                result = self.assess_trigger(inp)
                results.append(result)
            except ValueError as e:
                logger.error("Failed to assess trigger for %s: %s", inp.entity_name, str(e))
        logger.info("Assessed %d recalculation triggers", len(results))
        return results

    # -------------------------------------------------------------------
    # Cumulative Significance
    # -------------------------------------------------------------------

    def assess_cumulative_significance(
        self, changes: List[RecalculationInput]
    ) -> Tuple[Decimal, SignificanceLevel]:
        """Assess the cumulative significance of multiple changes.

        Multiple changes below the threshold individually may exceed it
        when considered cumulatively.

        Args:
            changes: List of RecalculationInput for individual changes.

        Returns:
            Tuple of (cumulative_significance_pct, SignificanceLevel).
        """
        if not changes:
            return Decimal("0"), SignificanceLevel.BELOW_THRESHOLD

        base = changes[0].base_year_emissions
        if base <= Decimal("0"):
            return Decimal("0"), SignificanceLevel.BELOW_THRESHOLD

        total_change = Decimal("0")
        for change in changes:
            total_change += abs(change.post_change_emissions - change.pre_change_emissions)

        cumulative_fraction = _safe_divide(total_change, base)
        cumulative_pct = _round_val(cumulative_fraction * Decimal("100"), 2)
        level = self._classify_significance(cumulative_fraction)

        logger.info(
            "Cumulative significance: %.2f%% across %d changes",
            float(cumulative_pct), len(changes),
        )
        return cumulative_pct, level

    # -------------------------------------------------------------------
    # History / Utility
    # -------------------------------------------------------------------

    def get_history(self) -> List[RecalculationResult]:
        """Get all past recalculation results.

        Returns:
            List of RecalculationResult.
        """
        return list(self._history)

    def clear_history(self) -> None:
        """Clear recalculation history."""
        self._history.clear()
        logger.info("Recalculation history cleared")
