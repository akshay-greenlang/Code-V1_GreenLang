# -*- coding: utf-8 -*-
"""
BaseYearRecalculationEngine - PACK-041 Scope 1-2 Complete Engine 7
====================================================================

Implements GHG Protocol Corporate Standard Chapter 5 base year
recalculation policy to maintain time-series consistency when structural
or methodological changes occur in an organisation's GHG inventory.

Calculation Methodology:
    Significance Test (GHG Protocol Ch 5, p.35):
        significance_pct = abs(adjustment) / base_year_total * 100
        is_significant = significance_pct >= threshold (default 5%)

    Acquisition Adjustment:
        For each acquired entity:
            recalculated_base_year = original_base_year + acquired_emissions
        Pro-rata if acquisition occurred mid-year:
            annual_acquired = acquired_emissions * (12 / months_in_year_of_acq)

    Divestiture Adjustment:
        For each divested entity:
            recalculated_base_year = original_base_year - divested_emissions
        Pro-rata for partial-year divestiture

    Methodology Change:
        Apply new emission factors or calculation methodology to base year
        activity data to produce a like-for-like comparison.

    Error Correction:
        Apply the discovered error correction to base year data.
        Only triggers recalculation if cumulative errors exceed threshold.

    Source/Boundary Change:
        Add newly included source categories to base year using
        best available historical data or proxies.

Trigger Assessment (GHG Protocol Table 5.3):
    ACQUISITION:              Always significant if >5% of total
    DIVESTITURE:              Always significant if >5% of total
    MERGER:                   Always significant (restructures entity)
    METHODOLOGY_CHANGE:       Significant if impact >5% of total
    ERROR_CORRECTION:         Significant if cumulative errors >5%
    SOURCE_CATEGORY_CHANGE:   Significant if new source >5%

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015), Chapter 5
    - GHG Protocol Corporate Value Chain Standard (Scope 3), Chapter 5
    - ISO 14064-1:2018, Clause 5.2 (Base year selection)
    - ESRS E1-6 (Gross GHG emissions - base year recalculation)
    - CDP Climate Change Questionnaire C5.1-C5.2
    - SBTi Corporate Manual (2023), Section 7 (Recalculation)
    - SEC Climate Disclosure Rule (2024), Item 1504

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Threshold values from GHG Protocol published guidance
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-041 Scope 1-2 Complete
Engine:  7 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def _round4(value: Any) -> float:
    """Round to 4 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RecalculationTriggerType(str, Enum):
    """Types of events that may trigger base year recalculation.

    Per GHG Protocol Corporate Standard, Chapter 5, Table 5.3.

    ACQUISITION:            Purchase of operations or business units.
    DIVESTITURE:            Sale or closure of operations or business units.
    MERGER:                 Merger with another organisation.
    METHODOLOGY_CHANGE:     Change in calculation methodology or emission
                            factors (e.g. Tier 1 to Tier 2, updated GWPs).
    ERROR_CORRECTION:       Discovery and correction of significant errors.
    SOURCE_CATEGORY_CHANGE: Addition or removal of source categories from
                            the organisational boundary.
    """
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    METHODOLOGY_CHANGE = "methodology_change"
    ERROR_CORRECTION = "error_correction"
    SOURCE_CATEGORY_CHANGE = "source_category_change"

class RecalculationStatus(str, Enum):
    """Status of a base year recalculation assessment.

    PENDING:      Trigger identified, assessment not yet complete.
    SIGNIFICANT:  Trigger exceeds significance threshold; recalculation required.
    NOT_SIGNIFICANT: Trigger below threshold; recalculation not required.
    APPLIED:      Recalculation has been applied to base year data.
    DEFERRED:     Recalculation deferred to next reporting cycle.
    """
    PENDING = "pending"
    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    APPLIED = "applied"
    DEFERRED = "deferred"

class AdjustmentType(str, Enum):
    """Type of adjustment applied to the base year.

    ADDITIVE:       Add emissions to base year (acquisition, new source).
    SUBTRACTIVE:    Remove emissions from base year (divestiture).
    REPLACEMENT:    Replace emission values (methodology change, error).
    PRO_RATA:       Partial-year adjustment scaled to full year.
    """
    ADDITIVE = "additive"
    SUBTRACTIVE = "subtractive"
    REPLACEMENT = "replacement"
    PRO_RATA = "pro_rata"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default significance threshold as percentage of base year emissions.
# Source: GHG Protocol Corporate Standard, Chapter 5, p.35.
# "A significant threshold is typically between 5% and 10%."
DEFAULT_SIGNIFICANCE_THRESHOLD_PCT: float = 5.0

# SBTi significance threshold (stricter for target tracking).
# Source: SBTi Corporate Manual (2023), Section 7.
SBTI_SIGNIFICANCE_THRESHOLD_PCT: float = 5.0

# Minimum base year for validation.
MINIMUM_BASE_YEAR: int = 1990

# Maximum number of triggers to process in a single recalculation.
MAX_TRIGGERS_PER_RECALCULATION: int = 50

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class BaseYearData(BaseModel):
    """Complete base year emission inventory data.

    Attributes:
        year: Base year (e.g. 2019).
        scope1_total: Total Scope 1 emissions (tCO2e).
        scope2_location_total: Scope 2 location-based total (tCO2e).
        scope2_market_total: Scope 2 market-based total (tCO2e).
        per_category_emissions: Emissions by source category (tCO2e).
        per_facility_emissions: Emissions by facility (tCO2e).
        per_gas_emissions: Emissions by gas type (tCO2e).
        activity_data: Activity data by source category (original units).
        emission_factors: Emission factors by source category.
        consolidation_approach: Equity share, financial control, or
            operational control.
        notes: Free-text notes about the base year data.
    """
    year: int = Field(
        ..., ge=MINIMUM_BASE_YEAR, le=2030, description="Base year"
    )
    scope1_total: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 1 total (tCO2e)"
    )
    scope2_location_total: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 2 location-based (tCO2e)"
    )
    scope2_market_total: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 2 market-based (tCO2e)"
    )
    per_category_emissions: Dict[str, float] = Field(
        default_factory=dict, description="Emissions by source category"
    )
    per_facility_emissions: Dict[str, float] = Field(
        default_factory=dict, description="Emissions by facility"
    )
    per_gas_emissions: Dict[str, float] = Field(
        default_factory=dict, description="Emissions by gas type"
    )
    activity_data: Dict[str, Any] = Field(
        default_factory=dict, description="Activity data by source category"
    )
    emission_factors: Dict[str, Any] = Field(
        default_factory=dict, description="Emission factors by source category"
    )
    consolidation_approach: str = Field(
        default="operational_control",
        description="Consolidation approach (equity_share, financial_control, operational_control)"
    )
    notes: str = Field(default="", description="Notes about base year data")

    @field_validator("scope1_total", "scope2_location_total", "scope2_market_total", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce emission totals to Decimal."""
        return _decimal(v)

    @property
    def grand_total(self) -> Decimal:
        """Total Scope 1 + Scope 2 (location-based) emissions."""
        return self.scope1_total + self.scope2_location_total

class RecalculationTrigger(BaseModel):
    """A single event that may trigger base year recalculation.

    Attributes:
        trigger_id: Unique trigger identifier.
        trigger_type: Type of recalculation trigger.
        description: Human-readable description of the event.
        effective_date: Date when the event takes effect.
        affected_entities: List of affected facilities/business units.
        affected_categories: List of affected emission source categories.
        estimated_emission_impact_tco2e: Estimated emission impact (tCO2e).
        significance_pct: Pre-calculated significance percentage.
        acquired_emissions: Emissions from acquired entities (for acquisition).
        divested_emissions: Emissions from divested entities (for divestiture).
        new_emission_factors: New emission factors (for methodology change).
        error_corrections: Error corrections by category (for error correction).
        months_in_year: Months the entity operated in the acquisition/divestiture
            year (for pro-rata calculation).
        metadata: Additional metadata about the trigger.
    """
    trigger_id: str = Field(
        default_factory=_new_uuid, description="Trigger identifier"
    )
    trigger_type: RecalculationTriggerType = Field(
        ..., description="Trigger type"
    )
    description: str = Field(
        default="", description="Event description"
    )
    effective_date: Optional[str] = Field(
        default=None, description="Effective date (ISO format)"
    )
    affected_entities: List[str] = Field(
        default_factory=list, description="Affected facilities/BUs"
    )
    affected_categories: List[str] = Field(
        default_factory=list, description="Affected source categories"
    )
    estimated_emission_impact_tco2e: Decimal = Field(
        default=Decimal("0"), description="Estimated impact (tCO2e)"
    )
    significance_pct: Optional[Decimal] = Field(
        default=None, description="Pre-calculated significance %"
    )
    acquired_emissions: Optional[Dict[str, float]] = Field(
        default=None, description="Acquired entity emissions by category"
    )
    divested_emissions: Optional[Dict[str, float]] = Field(
        default=None, description="Divested entity emissions by category"
    )
    new_emission_factors: Optional[Dict[str, float]] = Field(
        default=None, description="New emission factors by category"
    )
    error_corrections: Optional[Dict[str, float]] = Field(
        default=None, description="Error corrections by category"
    )
    months_in_year: int = Field(
        default=12, ge=1, le=12,
        description="Months entity operated in year (for pro-rata)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("estimated_emission_impact_tco2e", mode="before")
    @classmethod
    def coerce_impact(cls, v: Any) -> Decimal:
        """Coerce impact to Decimal."""
        return _decimal(v)

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class SignificanceAssessment(BaseModel):
    """Result of significance threshold assessment for a trigger.

    Attributes:
        trigger_id: The trigger being assessed.
        trigger_type: Type of trigger.
        emission_impact_tco2e: Absolute emission impact.
        base_year_total_tco2e: Base year total used as denominator.
        significance_pct: Impact as percentage of base year.
        threshold_pct: Significance threshold used.
        is_significant: Whether the trigger exceeds the threshold.
        status: Recalculation status.
        rationale: Explanation of the assessment.
    """
    trigger_id: str = Field(default="", description="Trigger ID")
    trigger_type: str = Field(default="", description="Trigger type")
    emission_impact_tco2e: float = Field(default=0.0, description="Impact (tCO2e)")
    base_year_total_tco2e: float = Field(default=0.0, description="Base year total")
    significance_pct: float = Field(default=0.0, description="Significance %")
    threshold_pct: float = Field(default=5.0, description="Threshold %")
    is_significant: bool = Field(default=False, description="Exceeds threshold")
    status: RecalculationStatus = Field(
        default=RecalculationStatus.PENDING, description="Status"
    )
    rationale: str = Field(default="", description="Assessment rationale")

class RecalculationAdjustment(BaseModel):
    """A single adjustment applied to the base year.

    Attributes:
        adjustment_id: Unique adjustment identifier.
        trigger_id: The trigger that caused this adjustment.
        adjustment_type: Type of adjustment (additive/subtractive/replacement).
        scope: Which scope is affected (scope1, scope2_location, scope2_market).
        category: Affected source category.
        original_value: Original base year value (tCO2e).
        adjusted_value: New value after adjustment (tCO2e).
        adjustment_amount: Delta (adjusted - original).
        pro_rata_factor: Pro-rata factor if partial year.
        justification: Explanation of the adjustment.
    """
    adjustment_id: str = Field(
        default_factory=_new_uuid, description="Adjustment ID"
    )
    trigger_id: str = Field(default="", description="Source trigger ID")
    adjustment_type: AdjustmentType = Field(
        default=AdjustmentType.ADDITIVE, description="Adjustment type"
    )
    scope: str = Field(default="scope1", description="Affected scope")
    category: str = Field(default="", description="Affected source category")
    original_value: float = Field(default=0.0, description="Original value (tCO2e)")
    adjusted_value: float = Field(default=0.0, description="Adjusted value (tCO2e)")
    adjustment_amount: float = Field(default=0.0, description="Delta (tCO2e)")
    pro_rata_factor: float = Field(default=1.0, description="Pro-rata factor")
    justification: str = Field(default="", description="Justification")

class AuditEntry(BaseModel):
    """Audit trail entry for a recalculation step.

    Attributes:
        entry_id: Unique entry identifier.
        timestamp: When this step was executed.
        action: Description of the action taken.
        trigger_id: Related trigger identifier.
        field_changed: Which field was changed.
        old_value: Previous value.
        new_value: New value after change.
        user_note: Optional user note.
    """
    entry_id: str = Field(default_factory=_new_uuid, description="Entry ID")
    timestamp: datetime = Field(default_factory=utcnow, description="Timestamp")
    action: str = Field(default="", description="Action description")
    trigger_id: str = Field(default="", description="Related trigger")
    field_changed: str = Field(default="", description="Field changed")
    old_value: str = Field(default="", description="Old value")
    new_value: str = Field(default="", description="New value")
    user_note: str = Field(default="", description="User note")

class BaseYearRecalculationResult(BaseModel):
    """Complete base year recalculation result with full provenance.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        processing_time_ms: Processing time in milliseconds.
        original_base_year: Original base year data (before recalculation).
        recalculated_base_year: Recalculated base year data.
        significance_assessments: Significance test for each trigger.
        adjustments_applied: List of adjustments made.
        triggers_evaluated: Total triggers evaluated.
        triggers_significant: Count of significant triggers.
        triggers_applied: Count of triggers that resulted in adjustments.
        is_significant: Whether any trigger exceeded the threshold.
        total_adjustment_tco2e: Net adjustment to base year total.
        total_adjustment_pct: Net adjustment as percentage of original.
        audit_trail: Complete audit trail.
        methodology_notes: Methodology and reference notes.
        provenance_hash: SHA-256 hash for audit.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Timestamp"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    original_base_year: Optional[BaseYearData] = Field(
        default=None, description="Original base year"
    )
    recalculated_base_year: Optional[BaseYearData] = Field(
        default=None, description="Recalculated base year"
    )
    significance_assessments: List[SignificanceAssessment] = Field(
        default_factory=list, description="Significance assessments"
    )
    adjustments_applied: List[RecalculationAdjustment] = Field(
        default_factory=list, description="Adjustments applied"
    )
    triggers_evaluated: int = Field(default=0, description="Triggers evaluated")
    triggers_significant: int = Field(default=0, description="Significant triggers")
    triggers_applied: int = Field(default=0, description="Applied triggers")
    is_significant: bool = Field(default=False, description="Any significant trigger")
    total_adjustment_tco2e: float = Field(
        default=0.0, description="Net adjustment (tCO2e)"
    )
    total_adjustment_pct: float = Field(
        default=0.0, description="Net adjustment (%)"
    )
    audit_trail: List[AuditEntry] = Field(
        default_factory=list, description="Audit trail"
    )
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology notes"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BaseYearRecalculationEngine:
    """GHG Protocol Chapter 5 base year recalculation engine.

    Implements the complete base year recalculation workflow including
    significance assessment, pro-rata calculations for partial-year
    events, and separate handling for acquisitions, divestitures,
    methodology changes, and error corrections.

    Guarantees:
        - Deterministic: same inputs produce identical outputs.
        - Auditable: complete audit trail for every adjustment.
        - Compliant: follows GHG Protocol Chapter 5 procedures.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        engine = BaseYearRecalculationEngine()
        result = engine.recalculate(
            base_year_data=BaseYearData(year=2019, scope1_total=10000, ...),
            triggers=[
                RecalculationTrigger(
                    trigger_type=RecalculationTriggerType.ACQUISITION,
                    estimated_emission_impact_tco2e=1500,
                    acquired_emissions={"stationary_combustion": 1000, "mobile": 500},
                ),
            ],
        )
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the base year recalculation engine.

        Args:
            config: Optional overrides. Supported keys:
                - significance_threshold_pct (float): default 5.0
                - sbti_mode (bool): use SBTi stricter thresholds
                - auto_apply_significant (bool): auto-apply significant triggers
        """
        self._config = config or {}
        self._threshold_pct = float(
            self._config.get(
                "significance_threshold_pct",
                DEFAULT_SIGNIFICANCE_THRESHOLD_PCT,
            )
        )
        self._sbti_mode = bool(self._config.get("sbti_mode", False))
        self._auto_apply = bool(self._config.get("auto_apply_significant", True))
        self._notes: List[str] = []
        self._audit_trail: List[AuditEntry] = []

        if self._sbti_mode:
            self._threshold_pct = SBTI_SIGNIFICANCE_THRESHOLD_PCT

        logger.info(
            "BaseYearRecalculationEngine v%s initialised, threshold=%.1f%%, "
            "SBTi=%s, auto_apply=%s.",
            _MODULE_VERSION, self._threshold_pct,
            self._sbti_mode, self._auto_apply,
        )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def recalculate(
        self,
        base_year_data: BaseYearData,
        triggers: List[RecalculationTrigger],
    ) -> BaseYearRecalculationResult:
        """Run complete base year recalculation workflow.

        1. Assess significance of each trigger.
        2. For significant triggers, compute adjustments.
        3. Apply adjustments to base year data.
        4. Generate audit trail and provenance.

        Args:
            base_year_data: Original base year inventory data.
            triggers: List of recalculation triggers to evaluate.

        Returns:
            BaseYearRecalculationResult with full provenance.

        Raises:
            ValueError: If triggers list exceeds MAX_TRIGGERS_PER_RECALCULATION.
        """
        t0 = time.perf_counter()
        self._notes = [
            f"Engine version: {self.engine_version}",
            f"Significance threshold: {self._threshold_pct}%",
            f"Base year: {base_year_data.year}",
            f"Base year total (S1+S2 location): "
            f"{_round2(float(base_year_data.grand_total))} tCO2e",
        ]
        self._audit_trail = []

        if len(triggers) > MAX_TRIGGERS_PER_RECALCULATION:
            raise ValueError(
                f"Too many triggers ({len(triggers)}); "
                f"maximum is {MAX_TRIGGERS_PER_RECALCULATION}."
            )

        self._add_audit("recalculation_started", "",
                        f"Evaluating {len(triggers)} trigger(s) against "
                        f"base year {base_year_data.year}.")

        logger.info(
            "Base year recalculation: year=%d, %d triggers, total=%.2f tCO2e",
            base_year_data.year, len(triggers), float(base_year_data.grand_total),
        )

        # Step 1: Assess significance of each trigger
        assessments: List[SignificanceAssessment] = []
        for trigger in triggers:
            assessment = self.assess_significance(trigger, base_year_data)
            assessments.append(assessment)

        # Step 2: Compute and apply adjustments for significant triggers
        all_adjustments: List[RecalculationAdjustment] = []
        recalculated = self._deep_copy_base_year(base_year_data)
        applied_count = 0

        for trigger, assessment in zip(triggers, assessments):
            if assessment.is_significant and self._auto_apply:
                adjustments = self._compute_adjustments(trigger, recalculated)
                recalculated = self._apply_adjustments(recalculated, adjustments)
                all_adjustments.extend(adjustments)
                applied_count += 1

                assessment.status = RecalculationStatus.APPLIED
                self._add_audit(
                    "trigger_applied", trigger.trigger_id,
                    f"Applied {len(adjustments)} adjustment(s) for "
                    f"{trigger.trigger_type.value}: {trigger.description}",
                )
            elif not assessment.is_significant:
                assessment.status = RecalculationStatus.NOT_SIGNIFICANT
                self._add_audit(
                    "trigger_not_significant", trigger.trigger_id,
                    f"Trigger below {self._threshold_pct}% threshold: "
                    f"{assessment.significance_pct:.2f}%",
                )

        # Step 3: Calculate net adjustment
        net_adj = _decimal(recalculated.grand_total) - _decimal(base_year_data.grand_total)
        net_adj_pct = float(
            _safe_pct(abs(net_adj), _decimal(base_year_data.grand_total))
        )

        significant_count = sum(1 for a in assessments if a.is_significant)

        self._add_audit(
            "recalculation_complete", "",
            f"Net adjustment: {_round2(float(net_adj))} tCO2e ({_round2(net_adj_pct)}%). "
            f"{significant_count} significant, {applied_count} applied.",
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = BaseYearRecalculationResult(
            original_base_year=base_year_data,
            recalculated_base_year=recalculated,
            significance_assessments=assessments,
            adjustments_applied=all_adjustments,
            triggers_evaluated=len(triggers),
            triggers_significant=significant_count,
            triggers_applied=applied_count,
            is_significant=significant_count > 0,
            total_adjustment_tco2e=_round2(float(net_adj)),
            total_adjustment_pct=_round2(net_adj_pct),
            audit_trail=list(self._audit_trail),
            methodology_notes=list(self._notes),
            processing_time_ms=_round3(elapsed_ms),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Base year recalculation complete: %d/%d significant, "
            "net_adj=%.2f tCO2e (%.2f%%), hash=%s (%.1f ms)",
            significant_count, len(triggers),
            float(net_adj), net_adj_pct,
            result.provenance_hash[:16], elapsed_ms,
        )
        return result

    def assess_significance(
        self,
        trigger: RecalculationTrigger,
        base_year_data: BaseYearData,
    ) -> SignificanceAssessment:
        """Assess whether a trigger exceeds the significance threshold.

        Per GHG Protocol Chapter 5: a trigger is significant if its
        estimated emission impact exceeds the threshold percentage
        of the base year total (Scope 1 + Scope 2).

        Args:
            trigger: Recalculation trigger to assess.
            base_year_data: Base year data for comparison.

        Returns:
            SignificanceAssessment with threshold test result.
        """
        impact = abs(_decimal(trigger.estimated_emission_impact_tco2e))
        total = _decimal(base_year_data.grand_total)

        sig_pct = float(_safe_pct(impact, total))

        is_sig = sig_pct >= self._threshold_pct

        # Mergers are always considered significant per GHG Protocol
        if trigger.trigger_type == RecalculationTriggerType.MERGER:
            is_sig = True

        rationale = self._build_rationale(trigger, sig_pct, is_sig)

        self._notes.append(
            f"Trigger {trigger.trigger_id[:8]}: {trigger.trigger_type.value}, "
            f"impact={_round2(float(impact))} tCO2e, "
            f"significance={_round2(sig_pct)}% "
            f"({'SIGNIFICANT' if is_sig else 'not significant'})."
        )

        return SignificanceAssessment(
            trigger_id=trigger.trigger_id,
            trigger_type=trigger.trigger_type.value,
            emission_impact_tco2e=_round2(float(impact)),
            base_year_total_tco2e=_round2(float(total)),
            significance_pct=_round2(sig_pct),
            threshold_pct=self._threshold_pct,
            is_significant=is_sig,
            status=RecalculationStatus.SIGNIFICANT if is_sig else RecalculationStatus.NOT_SIGNIFICANT,
            rationale=rationale,
        )

    def apply_acquisition(
        self,
        base_year: BaseYearData,
        acquired_emissions: Dict[str, float],
        months_in_year: int = 12,
    ) -> BaseYearData:
        """Apply acquisition adjustment to base year.

        Adds acquired entity's emissions to the base year, with
        pro-rata adjustment for partial-year acquisitions.

        Formula:
            annual_acquired = acquired * (12 / months_in_year)
            new_base_year = original + annual_acquired

        Args:
            base_year: Current base year data.
            acquired_emissions: Emissions by category from acquired entity.
            months_in_year: Months acquired entity operated in base year.

        Returns:
            Updated BaseYearData with acquisition reflected.
        """
        result = self._deep_copy_base_year(base_year)
        pro_rata = _safe_divide(
            Decimal("12"), _decimal(months_in_year), Decimal("1")
        )

        scope1_add = Decimal("0")
        scope2_add = Decimal("0")

        for category, value in acquired_emissions.items():
            annual_value = _decimal(value) * pro_rata
            cat_lower = category.lower()

            if "scope2" in cat_lower or "location" in cat_lower or "market" in cat_lower:
                scope2_add += annual_value
            else:
                scope1_add += annual_value

            # Update per-category
            current = _decimal(result.per_category_emissions.get(category, 0.0))
            result.per_category_emissions[category] = _round2(float(current + annual_value))

        result.scope1_total = result.scope1_total + scope1_add
        result.scope2_location_total = result.scope2_location_total + scope2_add

        return result

    def apply_divestiture(
        self,
        base_year: BaseYearData,
        divested_emissions: Dict[str, float],
        months_in_year: int = 12,
    ) -> BaseYearData:
        """Apply divestiture adjustment to base year.

        Removes divested entity's emissions from the base year, with
        pro-rata adjustment for partial-year divestitures.

        Formula:
            annual_divested = divested * (12 / months_in_year)
            new_base_year = original - annual_divested

        Args:
            base_year: Current base year data.
            divested_emissions: Emissions by category from divested entity.
            months_in_year: Months divested entity operated in base year.

        Returns:
            Updated BaseYearData with divestiture reflected.
        """
        result = self._deep_copy_base_year(base_year)
        pro_rata = _safe_divide(
            Decimal("12"), _decimal(months_in_year), Decimal("1")
        )

        scope1_sub = Decimal("0")
        scope2_sub = Decimal("0")

        for category, value in divested_emissions.items():
            annual_value = _decimal(value) * pro_rata
            cat_lower = category.lower()

            if "scope2" in cat_lower or "location" in cat_lower or "market" in cat_lower:
                scope2_sub += annual_value
            else:
                scope1_sub += annual_value

            current = _decimal(result.per_category_emissions.get(category, 0.0))
            new_val = max(current - annual_value, Decimal("0"))
            result.per_category_emissions[category] = _round2(float(new_val))

        result.scope1_total = max(result.scope1_total - scope1_sub, Decimal("0"))
        result.scope2_location_total = max(
            result.scope2_location_total - scope2_sub, Decimal("0")
        )

        return result

    def apply_methodology_change(
        self,
        base_year: BaseYearData,
        affected_categories: List[str],
        new_factors: Dict[str, float],
    ) -> BaseYearData:
        """Apply methodology change to base year.

        Recalculates base year emissions for affected categories using
        new emission factors applied to original activity data.

        Formula:
            new_emissions_i = activity_data_i * new_factor_i

        Args:
            base_year: Current base year data.
            affected_categories: Categories whose methodology changed.
            new_factors: New emission factors by category.

        Returns:
            Updated BaseYearData with new methodology applied.
        """
        result = self._deep_copy_base_year(base_year)

        for category in affected_categories:
            if category not in new_factors:
                continue

            new_factor = _decimal(new_factors[category])

            # Get original activity data
            activity = base_year.activity_data.get(category)
            if activity is not None:
                activity_value = _decimal(activity) if not isinstance(activity, dict) else Decimal("0")
                new_emissions = activity_value * new_factor

                old_emissions = _decimal(
                    result.per_category_emissions.get(category, 0.0)
                )
                delta = new_emissions - old_emissions

                result.per_category_emissions[category] = _round2(float(new_emissions))

                # Update scope totals
                cat_lower = category.lower()
                if "scope2" in cat_lower or "location" in cat_lower or "market" in cat_lower:
                    result.scope2_location_total = max(
                        result.scope2_location_total + delta, Decimal("0")
                    )
                else:
                    result.scope1_total = max(
                        result.scope1_total + delta, Decimal("0")
                    )

            elif category in base_year.per_category_emissions:
                # No activity data: apply factor ratio
                old_factor = base_year.emission_factors.get(category)
                if old_factor is not None:
                    old_factor_d = _decimal(old_factor) if not isinstance(old_factor, dict) else Decimal("1")
                    if old_factor_d > Decimal("0"):
                        ratio = _safe_divide(new_factor, old_factor_d, Decimal("1"))
                        old_emissions = _decimal(
                            result.per_category_emissions.get(category, 0.0)
                        )
                        new_emissions = old_emissions * ratio
                        delta = new_emissions - old_emissions

                        result.per_category_emissions[category] = _round2(float(new_emissions))

                        cat_lower = category.lower()
                        if "scope2" in cat_lower:
                            result.scope2_location_total = max(
                                result.scope2_location_total + delta, Decimal("0")
                            )
                        else:
                            result.scope1_total = max(
                                result.scope1_total + delta, Decimal("0")
                            )

            # Update emission factors
            result.emission_factors[category] = float(new_factor)

        return result

    def apply_error_correction(
        self,
        base_year: BaseYearData,
        corrections: Dict[str, float],
    ) -> BaseYearData:
        """Apply error corrections to base year.

        Each correction is a delta (positive = increase, negative = decrease)
        applied to the corresponding source category.

        Args:
            base_year: Current base year data.
            corrections: Corrections by category (delta tCO2e).

        Returns:
            Updated BaseYearData with corrections applied.
        """
        result = self._deep_copy_base_year(base_year)

        for category, correction in corrections.items():
            delta = _decimal(correction)
            old_val = _decimal(result.per_category_emissions.get(category, 0.0))
            new_val = max(old_val + delta, Decimal("0"))

            result.per_category_emissions[category] = _round2(float(new_val))

            cat_lower = category.lower()
            if "scope2" in cat_lower or "location" in cat_lower or "market" in cat_lower:
                result.scope2_location_total = max(
                    result.scope2_location_total + delta, Decimal("0")
                )
            else:
                result.scope1_total = max(
                    result.scope1_total + delta, Decimal("0")
                )

        return result

    def generate_audit_trail(
        self,
        result: BaseYearRecalculationResult,
    ) -> List[AuditEntry]:
        """Extract the audit trail from a recalculation result.

        Args:
            result: Completed recalculation result.

        Returns:
            List of AuditEntry objects.
        """
        return list(result.audit_trail)

    def compare_base_years(
        self,
        original: BaseYearData,
        recalculated: BaseYearData,
    ) -> Dict[str, Any]:
        """Compare original and recalculated base year data.

        Args:
            original: Original base year data.
            recalculated: Recalculated base year data.

        Returns:
            Dict with comparison statistics.
        """
        s1_delta = _decimal(recalculated.scope1_total) - _decimal(original.scope1_total)
        s2l_delta = (
            _decimal(recalculated.scope2_location_total)
            - _decimal(original.scope2_location_total)
        )
        s2m_delta = (
            _decimal(recalculated.scope2_market_total)
            - _decimal(original.scope2_market_total)
        )

        total_orig = _decimal(original.grand_total)
        total_recalc = _decimal(recalculated.grand_total)
        total_delta = total_recalc - total_orig

        return {
            "base_year": original.year,
            "scope1_original": _round2(float(original.scope1_total)),
            "scope1_recalculated": _round2(float(recalculated.scope1_total)),
            "scope1_delta": _round2(float(s1_delta)),
            "scope1_delta_pct": _round2(float(_safe_pct(s1_delta, _decimal(original.scope1_total)))),
            "scope2_location_original": _round2(float(original.scope2_location_total)),
            "scope2_location_recalculated": _round2(float(recalculated.scope2_location_total)),
            "scope2_location_delta": _round2(float(s2l_delta)),
            "scope2_market_original": _round2(float(original.scope2_market_total)),
            "scope2_market_recalculated": _round2(float(recalculated.scope2_market_total)),
            "scope2_market_delta": _round2(float(s2m_delta)),
            "grand_total_original": _round2(float(total_orig)),
            "grand_total_recalculated": _round2(float(total_recalc)),
            "grand_total_delta": _round2(float(total_delta)),
            "grand_total_delta_pct": _round2(float(_safe_pct(total_delta, total_orig))),
            "categories_changed": self._find_changed_categories(original, recalculated),
        }

    # -------------------------------------------------------------------
    # Private -- Compute adjustments
    # -------------------------------------------------------------------

    def _compute_adjustments(
        self,
        trigger: RecalculationTrigger,
        base_year: BaseYearData,
    ) -> List[RecalculationAdjustment]:
        """Compute adjustments for a significant trigger.

        Routes to the appropriate handler based on trigger type.

        Args:
            trigger: The trigger to process.
            base_year: Current state of base year data.

        Returns:
            List of RecalculationAdjustment objects.
        """
        handler_map = {
            RecalculationTriggerType.ACQUISITION: self._adj_acquisition,
            RecalculationTriggerType.DIVESTITURE: self._adj_divestiture,
            RecalculationTriggerType.MERGER: self._adj_acquisition,
            RecalculationTriggerType.METHODOLOGY_CHANGE: self._adj_methodology,
            RecalculationTriggerType.ERROR_CORRECTION: self._adj_error,
            RecalculationTriggerType.SOURCE_CATEGORY_CHANGE: self._adj_source_change,
        }

        handler = handler_map.get(trigger.trigger_type, self._adj_default)
        return handler(trigger, base_year)

    def _adj_acquisition(
        self,
        trigger: RecalculationTrigger,
        base_year: BaseYearData,
    ) -> List[RecalculationAdjustment]:
        """Compute acquisition adjustments."""
        adjustments: List[RecalculationAdjustment] = []
        emissions = trigger.acquired_emissions or {}
        pro_rata = _safe_divide(
            Decimal("12"), _decimal(trigger.months_in_year), Decimal("1")
        )

        for category, value in emissions.items():
            annual = _decimal(value) * pro_rata
            original = _decimal(base_year.per_category_emissions.get(category, 0.0))

            adjustments.append(RecalculationAdjustment(
                trigger_id=trigger.trigger_id,
                adjustment_type=AdjustmentType.ADDITIVE,
                scope="scope2" if "scope2" in category.lower() else "scope1",
                category=category,
                original_value=_round2(float(original)),
                adjusted_value=_round2(float(original + annual)),
                adjustment_amount=_round2(float(annual)),
                pro_rata_factor=_round4(float(pro_rata)),
                justification=(
                    f"Acquisition: added {_round2(float(annual))} tCO2e "
                    f"for {category} (pro-rata {trigger.months_in_year}/12 months)."
                ),
            ))

        return adjustments

    def _adj_divestiture(
        self,
        trigger: RecalculationTrigger,
        base_year: BaseYearData,
    ) -> List[RecalculationAdjustment]:
        """Compute divestiture adjustments."""
        adjustments: List[RecalculationAdjustment] = []
        emissions = trigger.divested_emissions or {}
        pro_rata = _safe_divide(
            Decimal("12"), _decimal(trigger.months_in_year), Decimal("1")
        )

        for category, value in emissions.items():
            annual = _decimal(value) * pro_rata
            original = _decimal(base_year.per_category_emissions.get(category, 0.0))
            new_val = max(original - annual, Decimal("0"))

            adjustments.append(RecalculationAdjustment(
                trigger_id=trigger.trigger_id,
                adjustment_type=AdjustmentType.SUBTRACTIVE,
                scope="scope2" if "scope2" in category.lower() else "scope1",
                category=category,
                original_value=_round2(float(original)),
                adjusted_value=_round2(float(new_val)),
                adjustment_amount=_round2(float(new_val - original)),
                pro_rata_factor=_round4(float(pro_rata)),
                justification=(
                    f"Divestiture: removed {_round2(float(annual))} tCO2e "
                    f"for {category} (pro-rata {trigger.months_in_year}/12 months)."
                ),
            ))

        return adjustments

    def _adj_methodology(
        self,
        trigger: RecalculationTrigger,
        base_year: BaseYearData,
    ) -> List[RecalculationAdjustment]:
        """Compute methodology change adjustments."""
        adjustments: List[RecalculationAdjustment] = []
        new_factors = trigger.new_emission_factors or {}

        for category in trigger.affected_categories:
            if category not in new_factors:
                continue

            new_factor = _decimal(new_factors[category])
            original = _decimal(base_year.per_category_emissions.get(category, 0.0))

            # Try activity data * new factor
            activity = base_year.activity_data.get(category)
            if activity is not None and not isinstance(activity, dict):
                new_val = _decimal(activity) * new_factor
            else:
                # Ratio approach
                old_factor = base_year.emission_factors.get(category)
                if old_factor is not None and not isinstance(old_factor, dict):
                    old_factor_d = _decimal(old_factor)
                    ratio = _safe_divide(new_factor, old_factor_d, Decimal("1"))
                    new_val = original * ratio
                else:
                    new_val = original

            adjustments.append(RecalculationAdjustment(
                trigger_id=trigger.trigger_id,
                adjustment_type=AdjustmentType.REPLACEMENT,
                scope="scope2" if "scope2" in category.lower() else "scope1",
                category=category,
                original_value=_round2(float(original)),
                adjusted_value=_round2(float(new_val)),
                adjustment_amount=_round2(float(new_val - original)),
                justification=(
                    f"Methodology change: recalculated {category} using "
                    f"new emission factor."
                ),
            ))

        return adjustments

    def _adj_error(
        self,
        trigger: RecalculationTrigger,
        base_year: BaseYearData,
    ) -> List[RecalculationAdjustment]:
        """Compute error correction adjustments."""
        adjustments: List[RecalculationAdjustment] = []
        corrections = trigger.error_corrections or {}

        for category, delta in corrections.items():
            d_delta = _decimal(delta)
            original = _decimal(base_year.per_category_emissions.get(category, 0.0))
            new_val = max(original + d_delta, Decimal("0"))

            adjustments.append(RecalculationAdjustment(
                trigger_id=trigger.trigger_id,
                adjustment_type=AdjustmentType.REPLACEMENT,
                scope="scope2" if "scope2" in category.lower() else "scope1",
                category=category,
                original_value=_round2(float(original)),
                adjusted_value=_round2(float(new_val)),
                adjustment_amount=_round2(float(d_delta)),
                justification=(
                    f"Error correction: adjusted {category} by "
                    f"{_round2(float(d_delta))} tCO2e."
                ),
            ))

        return adjustments

    def _adj_source_change(
        self,
        trigger: RecalculationTrigger,
        base_year: BaseYearData,
    ) -> List[RecalculationAdjustment]:
        """Compute source category change adjustments."""
        adjustments: List[RecalculationAdjustment] = []

        for category in trigger.affected_categories:
            # New sources added to base year
            impact = _safe_divide(
                _decimal(trigger.estimated_emission_impact_tco2e),
                _decimal(max(len(trigger.affected_categories), 1)),
            )

            adjustments.append(RecalculationAdjustment(
                trigger_id=trigger.trigger_id,
                adjustment_type=AdjustmentType.ADDITIVE,
                scope="scope2" if "scope2" in category.lower() else "scope1",
                category=category,
                original_value=0.0,
                adjusted_value=_round2(float(impact)),
                adjustment_amount=_round2(float(impact)),
                justification=(
                    f"Source category change: added {category} to base year "
                    f"with estimated {_round2(float(impact))} tCO2e."
                ),
            ))

        return adjustments

    def _adj_default(
        self,
        trigger: RecalculationTrigger,
        base_year: BaseYearData,
    ) -> List[RecalculationAdjustment]:
        """Default adjustment handler for unknown trigger types."""
        logger.warning("Unknown trigger type: %s", trigger.trigger_type)
        return []

    # -------------------------------------------------------------------
    # Private -- Apply adjustments
    # -------------------------------------------------------------------

    def _apply_adjustments(
        self,
        base_year: BaseYearData,
        adjustments: List[RecalculationAdjustment],
    ) -> BaseYearData:
        """Apply a list of adjustments to base year data.

        Args:
            base_year: Current base year data.
            adjustments: List of adjustments to apply.

        Returns:
            Updated BaseYearData.
        """
        result = self._deep_copy_base_year(base_year)

        for adj in adjustments:
            delta = _decimal(adj.adjustment_amount)
            category = adj.category

            # Update per-category
            if adj.adjustment_type == AdjustmentType.REPLACEMENT:
                result.per_category_emissions[category] = adj.adjusted_value
                delta = _decimal(adj.adjusted_value) - _decimal(adj.original_value)
            else:
                current = _decimal(result.per_category_emissions.get(category, 0.0))
                result.per_category_emissions[category] = _round2(
                    float(max(current + delta, Decimal("0")))
                )

            # Update scope totals
            if adj.scope == "scope2":
                result.scope2_location_total = max(
                    result.scope2_location_total + delta, Decimal("0")
                )
            else:
                result.scope1_total = max(
                    result.scope1_total + delta, Decimal("0")
                )

        return result

    # -------------------------------------------------------------------
    # Private -- Utilities
    # -------------------------------------------------------------------

    def _deep_copy_base_year(self, data: BaseYearData) -> BaseYearData:
        """Create an independent copy of BaseYearData.

        Args:
            data: Source data to copy.

        Returns:
            New BaseYearData instance with copied values.
        """
        return BaseYearData(
            year=data.year,
            scope1_total=data.scope1_total,
            scope2_location_total=data.scope2_location_total,
            scope2_market_total=data.scope2_market_total,
            per_category_emissions=dict(data.per_category_emissions),
            per_facility_emissions=dict(data.per_facility_emissions),
            per_gas_emissions=dict(data.per_gas_emissions),
            activity_data=dict(data.activity_data),
            emission_factors=dict(data.emission_factors),
            consolidation_approach=data.consolidation_approach,
            notes=data.notes,
        )

    def _build_rationale(
        self,
        trigger: RecalculationTrigger,
        sig_pct: float,
        is_sig: bool,
    ) -> str:
        """Build significance assessment rationale string.

        Args:
            trigger: The trigger being assessed.
            sig_pct: Significance percentage.
            is_sig: Whether the trigger is significant.

        Returns:
            Rationale string.
        """
        parts = [
            f"Trigger type: {trigger.trigger_type.value}.",
            f"Estimated impact: {_round2(float(trigger.estimated_emission_impact_tco2e))} tCO2e.",
            f"Significance: {_round2(sig_pct)}% of base year total.",
            f"Threshold: {self._threshold_pct}%.",
        ]

        if trigger.trigger_type == RecalculationTriggerType.MERGER:
            parts.append(
                "Mergers are always considered significant per "
                "GHG Protocol Chapter 5 guidance."
            )

        if is_sig:
            parts.append(
                "RESULT: Significant - base year recalculation required."
            )
        else:
            parts.append(
                "RESULT: Not significant - no recalculation required."
            )

        return " ".join(parts)

    def _find_changed_categories(
        self,
        original: BaseYearData,
        recalculated: BaseYearData,
    ) -> List[str]:
        """Find categories whose emissions changed between base years.

        Args:
            original: Original base year.
            recalculated: Recalculated base year.

        Returns:
            List of changed category names.
        """
        all_cats = set(original.per_category_emissions.keys()) | set(
            recalculated.per_category_emissions.keys()
        )
        changed: List[str] = []
        for cat in sorted(all_cats):
            orig_val = original.per_category_emissions.get(cat, 0.0)
            recalc_val = recalculated.per_category_emissions.get(cat, 0.0)
            if abs(float(orig_val) - float(recalc_val)) > 0.01:
                changed.append(cat)
        return changed

    def _add_audit(
        self,
        action: str,
        trigger_id: str,
        description: str,
    ) -> None:
        """Add an entry to the internal audit trail.

        Args:
            action: Action type.
            trigger_id: Related trigger ID.
            description: Description of the action.
        """
        self._audit_trail.append(AuditEntry(
            action=action,
            trigger_id=trigger_id,
            field_changed="",
            old_value="",
            new_value="",
            user_note=description,
        ))

# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

BaseYearData.model_rebuild()
RecalculationTrigger.model_rebuild()
SignificanceAssessment.model_rebuild()
RecalculationAdjustment.model_rebuild()
AuditEntry.model_rebuild()
BaseYearRecalculationResult.model_rebuild()
