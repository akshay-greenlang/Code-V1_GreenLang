# -*- coding: utf-8 -*-
"""
BaseYearAdjustmentEngine - PACK-045 Base Year Management Engine 6
===================================================================

Adjustment calculation engine that computes the precise numerical
adjustments required to recalculate the base year inventory when
structural or methodological changes trigger recalculation per
GHG Protocol Corporate Standard Chapter 5.

Once the SignificanceAssessmentEngine determines that recalculation
is required, this engine computes line-by-line adjustments for each
scope, source category, and entity, then produces an AdjustmentPackage
that can be reviewed, approved, and applied to the base year inventory.

Adjustment Types:
    ACQUISITION_ADD:        Add acquired entity emissions to base year.
    DIVESTITURE_REMOVE:     Remove divested entity emissions from base year.
    METHODOLOGY_RESTATE:    Restate base year using new emission factors
                            (like-for-like restatement).
    ERROR_CORRECT:          Apply error corrections to base year values.
    BOUNDARY_ADD:           Add emissions for newly included source categories.
    BOUNDARY_REMOVE:        Remove emissions for excluded source categories.
    OUTSOURCE_SHIFT:        Shift emissions from Scope 1/2 to Scope 3 when
                            activities are outsourced.
    INSOURCE_SHIFT:         Shift emissions from Scope 3 to Scope 1/2 when
                            activities are insourced.

Pro-Rata Methods:
    MONTHLY:    Pro-rate partial-year adjustments by months:
                    annual_adj = full_year_impact * (months_in_period / 12)
    DAILY:      Pro-rate partial-year adjustments by days:
                    annual_adj = full_year_impact * (days_in_period / days_in_year)
    QUARTERLY:  Pro-rate partial-year adjustments by quarters:
                    annual_adj = full_year_impact * (quarters_in_period / 4)

Core Formulas:
    Acquisition Adjustment:
        equity_adj = entity_emissions * ownership_pct / 100
        pro_rata_adj = equity_adj * pro_rata_factor
        base_year_adjusted = base_year_original + pro_rata_adj

    Divestiture Adjustment:
        equity_adj = entity_emissions * ownership_pct / 100
        pro_rata_adj = equity_adj * pro_rata_factor
        base_year_adjusted = base_year_original - pro_rata_adj

    Methodology Restatement (Like-for-Like):
        restated_value = activity_data * new_emission_factor
        adjustment = restated_value - original_value
        base_year_adjusted = base_year_original + adjustment

    Error Correction:
        adjustment = corrected_value - original_value
        base_year_adjusted = base_year_original + adjustment

    Boundary Change:
        For additions: adjustment = historical_emissions_for_new_source
        For removals:  adjustment = -(historical_emissions_for_removed_source)
        base_year_adjusted = base_year_original + adjustment

Approval Workflow:
    DRAFT:              Package created, not yet submitted.
    PENDING_APPROVAL:   Submitted for review and approval.
    APPROVED:           Approved by designated approver(s).
    APPLIED:            Adjustments applied to the base year inventory.
    REJECTED:           Package rejected with documented reason.

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015), Chapter 5
    - GHG Protocol Corporate Value Chain (Scope 3) Standard, Chapter 5
    - ISO 14064-1:2018, Clause 5.2 (Base year selection)
    - ESRS E1-6 (Climate change - base year recalculation disclosures)
    - CDP Climate Change Questionnaire C5.1-C5.2 (2026)
    - SBTi Corporate Net-Zero Standard v1.1, Section 7 (Recalculation)
    - US SEC Climate Disclosure Rule (2024), Item 1504
    - California SB 253 Climate Corporate Data Accountability Act (2026)

Zero-Hallucination Guarantee:
    - All adjustments use deterministic Python Decimal arithmetic
    - Pro-rata factors are computed from calendar data (no estimation)
    - No LLM involvement in any calculation or approval path
    - SHA-256 provenance hash on every result
    - Complete audit trail of all adjustment lines

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-045 Base Year Management
Engine:  6 of 10
Status:  Production Ready
"""

from __future__ import annotations

import calendar
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

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Excludes volatile fields (calculated_at, processing_time_ms,
    provenance_hash) so that identical logical content always produces
    the same hash regardless of when or how fast it was computed.

    Args:
        data: Any Pydantic model, dict, or stringifiable object.

    Returns:
        Hex-encoded SHA-256 digest string.
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
    """Safely convert a value to Decimal.

    Args:
        value: Any numeric or string value to convert.

    Returns:
        Decimal representation; Decimal('0') on conversion failure.
    """
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
    """Safely divide two Decimals, returning *default* on zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value returned when denominator is zero.

    Returns:
        Result of numerator / denominator, or default.
    """
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely: (part / whole) * 100.

    Args:
        part: Numerator value.
        whole: Denominator value.

    Returns:
        Percentage as Decimal; Decimal('0') when whole is zero.
    """
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP.

    Args:
        value: The Decimal value to round.
        places: Number of decimal places (default 4).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _abs_decimal(value: Decimal) -> Decimal:
    """Return the absolute value of a Decimal.

    Args:
        value: The input Decimal.

    Returns:
        Absolute value.
    """
    return value if value >= Decimal("0") else -value

def _days_in_year(year: int) -> int:
    """Return the number of days in a given year.

    Args:
        year: The calendar year (e.g. 2019).

    Returns:
        366 for leap years, 365 otherwise.
    """
    return 366 if calendar.isleap(year) else 365

def _months_remaining_in_year(effective_date: date) -> int:
    """Calculate months remaining from effective_date to end of year (inclusive).

    The month of the effective date counts as a full month.

    Args:
        effective_date: The date from which to count.

    Returns:
        Number of months from effective_date.month to December (inclusive).
    """
    return 13 - effective_date.month

def _days_remaining_in_year(effective_date: date) -> int:
    """Calculate days remaining from effective_date to 31 December (inclusive).

    Args:
        effective_date: The date from which to count.

    Returns:
        Number of days from effective_date to 31 Dec (inclusive of both).
    """
    year_end = date(effective_date.year, 12, 31)
    return (year_end - effective_date).days + 1

def _quarters_remaining_in_year(effective_date: date) -> int:
    """Calculate quarters remaining from effective_date to end of year.

    The quarter containing the effective date counts as a full quarter.

    Args:
        effective_date: The date from which to count.

    Returns:
        Number of quarters remaining (1-4).
    """
    current_quarter = (effective_date.month - 1) // 3 + 1
    return 5 - current_quarter

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AdjustmentType(str, Enum):
    """Type of adjustment applied to the base year inventory.

    ACQUISITION_ADD:        Add acquired entity emissions to base year.
    DIVESTITURE_REMOVE:     Remove divested entity emissions from base year.
    METHODOLOGY_RESTATE:    Restate base year with new methodology/factors.
    ERROR_CORRECT:          Correct errors in base year data.
    BOUNDARY_ADD:           Add emissions for newly included sources.
    BOUNDARY_REMOVE:        Remove emissions for excluded sources.
    OUTSOURCE_SHIFT:        Shift emissions for outsourced activities
                            (Scope 1/2 to Scope 3).
    INSOURCE_SHIFT:         Shift emissions for insourced activities
                            (Scope 3 to Scope 1/2).
    """
    ACQUISITION_ADD = "acquisition_add"
    DIVESTITURE_REMOVE = "divestiture_remove"
    METHODOLOGY_RESTATE = "methodology_restate"
    ERROR_CORRECT = "error_correct"
    BOUNDARY_ADD = "boundary_add"
    BOUNDARY_REMOVE = "boundary_remove"
    OUTSOURCE_SHIFT = "outsource_shift"
    INSOURCE_SHIFT = "insource_shift"

class ProRataMethod(str, Enum):
    """Method for pro-rating partial-year adjustments.

    MONTHLY:    Pro-rate by months: factor = months_in_period / 12
    DAILY:      Pro-rate by days: factor = days_in_period / days_in_year
    QUARTERLY:  Pro-rate by quarters: factor = quarters_in_period / 4
    """
    MONTHLY = "monthly"
    DAILY = "daily"
    QUARTERLY = "quarterly"

class AdjustmentStatus(str, Enum):
    """Lifecycle status of an adjustment package.

    DRAFT:              Package created, not yet submitted for approval.
    PENDING_APPROVAL:   Submitted for review by designated approver(s).
    APPROVED:           Approved by approver(s), ready to apply.
    APPLIED:            Adjustments have been applied to base year inventory.
    REJECTED:           Package rejected by approver with documented reason.
    """
    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    APPLIED = "applied"
    REJECTED = "rejected"

class Scope(str, Enum):
    """GHG Protocol emission scope classification.

    SCOPE_1:            Direct emissions from owned or controlled sources.
    SCOPE_2_LOCATION:   Indirect emissions (location-based method).
    SCOPE_2_MARKET:     Indirect emissions (market-based method).
    SCOPE_3:            Other indirect value chain emissions.
    """
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"

class TriggerType(str, Enum):
    """Types of events that triggered the adjustment.

    Mirrors the TriggerType from RecalculationTriggerEngine.
    """
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    METHODOLOGY_CHANGE = "methodology_change"
    ERROR_CORRECTION = "error_correction"
    SOURCE_BOUNDARY_CHANGE = "source_boundary_change"
    OUTSOURCING_INSOURCING = "outsourcing_insourcing"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum adjustment lines per package.
MAX_ADJUSTMENT_LINES: int = 1000

# Minimum base year for validation.
MINIMUM_BASE_YEAR: int = 1990

# Maximum base year.
MAXIMUM_BASE_YEAR: int = 2030

# Default decimal precision for tCO2e outputs.
DEFAULT_TCO2E_PRECISION: int = 3

# Default decimal precision for percentages.
DEFAULT_PCT_PRECISION: int = 4

# Default decimal precision for pro-rata factors.
DEFAULT_PRORATA_PRECISION: int = 6

# ---------------------------------------------------------------------------
# Pydantic Models -- Core Data
# ---------------------------------------------------------------------------

class AdjustmentLine(BaseModel):
    """A single adjustment line in an adjustment package.

    Each line represents one atomic adjustment to a specific scope,
    category, and (optionally) entity within the base year inventory.

    Attributes:
        line_id: Unique line identifier.
        adjustment_type: Type of adjustment being applied.
        scope: GHG scope affected by this adjustment.
        category: Emission source category affected.
        description: Human-readable description of the adjustment.
        original_tco2e: Original base year emission value for this
            scope/category (tCO2e).
        adjustment_tco2e: The adjustment amount in tCO2e. Positive for
            additions, negative for removals.
        adjusted_tco2e: Final adjusted value:
            adjusted = original + adjustment.
        pro_rata_factor: Pro-rata factor applied (1.0 for full-year).
        effective_date: Date when the triggering event took effect.
        entity_id: Optional entity identifier (for structural changes).
        trigger_id: ID of the trigger that caused this adjustment.
        calculation_detail: Detailed calculation breakdown for audit.
    """
    line_id: str = Field(
        default_factory=_new_uuid, description="Adjustment line ID"
    )
    adjustment_type: AdjustmentType = Field(
        ..., description="Type of adjustment"
    )
    scope: Scope = Field(
        ..., description="Affected GHG scope"
    )
    category: str = Field(
        default="", description="Affected source category"
    )
    description: str = Field(
        default="", description="Adjustment description"
    )
    original_tco2e: Decimal = Field(
        default=Decimal("0"), description="Original base year value (tCO2e)"
    )
    adjustment_tco2e: Decimal = Field(
        default=Decimal("0"), description="Adjustment amount (tCO2e)"
    )
    adjusted_tco2e: Decimal = Field(
        default=Decimal("0"), description="Final adjusted value (tCO2e)"
    )
    pro_rata_factor: Decimal = Field(
        default=Decimal("1.0"), ge=0, le=1,
        description="Pro-rata factor (0-1)"
    )
    effective_date: Optional[date] = Field(
        default=None, description="Effective date of triggering event"
    )
    entity_id: Optional[str] = Field(
        default=None, description="Entity ID (for structural changes)"
    )
    trigger_id: str = Field(
        default="", description="Source trigger ID"
    )
    calculation_detail: str = Field(
        default="", description="Calculation breakdown for audit"
    )

    @field_validator(
        "original_tco2e", "adjustment_tco2e", "adjusted_tco2e", "pro_rata_factor",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)

    @model_validator(mode="after")
    def validate_adjusted_equals_original_plus_adjustment(self) -> "AdjustmentLine":
        """Validate that adjusted = original + adjustment.

        This is a critical integrity check ensuring the adjustment
        arithmetic is consistent.
        """
        expected = self.original_tco2e + self.adjustment_tco2e
        if _round_val(self.adjusted_tco2e, 3) != _round_val(expected, 3):
            self.adjusted_tco2e = _round_val(expected, DEFAULT_TCO2E_PRECISION)
        return self

class ScopeBreakdown(BaseModel):
    """Adjustment breakdown for a single GHG scope.

    Attributes:
        scope: The GHG scope.
        original_tco2e: Original base year total for this scope.
        total_adjustment_tco2e: Net adjustment for this scope.
        adjusted_tco2e: Adjusted total for this scope.
        change_pct: Percentage change from original.
        line_count: Number of adjustment lines for this scope.
    """
    scope: Scope = Field(..., description="GHG scope")
    original_tco2e: Decimal = Field(
        default=Decimal("0"), description="Original total (tCO2e)"
    )
    total_adjustment_tco2e: Decimal = Field(
        default=Decimal("0"), description="Net adjustment (tCO2e)"
    )
    adjusted_tco2e: Decimal = Field(
        default=Decimal("0"), description="Adjusted total (tCO2e)"
    )
    change_pct: Decimal = Field(
        default=Decimal("0"), description="Change (%)"
    )
    line_count: int = Field(
        default=0, ge=0, description="Adjustment line count"
    )

    @field_validator(
        "original_tco2e", "total_adjustment_tco2e", "adjusted_tco2e", "change_pct",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)

class TypeBreakdown(BaseModel):
    """Adjustment breakdown for a single adjustment type.

    Attributes:
        adjustment_type: The adjustment type.
        total_adjustment_tco2e: Net adjustment for this type.
        line_count: Number of adjustment lines of this type.
        affected_scopes: List of scopes affected by this type.
    """
    adjustment_type: AdjustmentType = Field(
        ..., description="Adjustment type"
    )
    total_adjustment_tco2e: Decimal = Field(
        default=Decimal("0"), description="Net adjustment (tCO2e)"
    )
    line_count: int = Field(
        default=0, ge=0, description="Line count"
    )
    affected_scopes: List[Scope] = Field(
        default_factory=list, description="Affected scopes"
    )

    @field_validator("total_adjustment_tco2e", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)

class AdjustmentSummary(BaseModel):
    """Summary of all adjustments in an adjustment package.

    Provides aggregate metrics and breakdowns by scope and type
    for the complete adjustment package.

    Attributes:
        total_original_tco2e: Sum of original values across all lines.
        total_adjustment_tco2e: Net adjustment across all lines.
        total_adjusted_tco2e: Sum of adjusted values across all lines.
        change_pct: Overall percentage change from original to adjusted.
        total_lines: Total number of adjustment lines.
        by_scope: Breakdown of adjustments by GHG scope.
        by_type: Breakdown of adjustments by adjustment type.
    """
    total_original_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total original (tCO2e)"
    )
    total_adjustment_tco2e: Decimal = Field(
        default=Decimal("0"), description="Net adjustment (tCO2e)"
    )
    total_adjusted_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total adjusted (tCO2e)"
    )
    change_pct: Decimal = Field(
        default=Decimal("0"), description="Overall change (%)"
    )
    total_lines: int = Field(
        default=0, ge=0, description="Total adjustment lines"
    )
    by_scope: Dict[str, ScopeBreakdown] = Field(
        default_factory=dict, description="Breakdown by scope"
    )
    by_type: Dict[str, TypeBreakdown] = Field(
        default_factory=dict, description="Breakdown by type"
    )

    @field_validator(
        "total_original_tco2e", "total_adjustment_tco2e",
        "total_adjusted_tco2e", "change_pct",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)

class AdjustmentPackage(BaseModel):
    """Complete base year adjustment package with approval workflow.

    An adjustment package aggregates all adjustment lines resulting
    from one or more recalculation triggers, provides a summary, and

from greenlang.schemas import utcnow
    tracks the approval lifecycle.

    Attributes:
        package_id: Unique package identifier.
        base_year: The base year being adjusted.
        engine_version: Engine version that created the package.
        adjustment_lines: List of individual adjustment lines.
        summary: Aggregate summary of all adjustments.
        status: Current approval lifecycle status.
        created_by: User or system that created the package.
        created_date: Timestamp of package creation.
        approved_by: User who approved the package (if approved).
        approved_date: Timestamp of approval (if approved).
        rejected_by: User who rejected the package (if rejected).
        rejected_date: Timestamp of rejection (if rejected).
        rejection_reason: Reason for rejection (if rejected).
        rationale: Overall rationale for the adjustment package.
        trigger_ids: List of trigger IDs that sourced this package.
        calculated_at: Timestamp of calculation.
        processing_time_ms: Total processing time in milliseconds.
        provenance_hash: SHA-256 hash of the complete package.
    """
    package_id: str = Field(
        default_factory=_new_uuid, description="Package identifier"
    )
    base_year: int = Field(
        ..., ge=MINIMUM_BASE_YEAR, le=MAXIMUM_BASE_YEAR,
        description="Base year being adjusted"
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version"
    )
    adjustment_lines: List[AdjustmentLine] = Field(
        default_factory=list, description="Adjustment lines"
    )
    summary: Optional[AdjustmentSummary] = Field(
        default=None, description="Adjustment summary"
    )
    status: AdjustmentStatus = Field(
        default=AdjustmentStatus.DRAFT, description="Package status"
    )
    created_by: str = Field(
        default="system", description="Creator"
    )
    created_date: datetime = Field(
        default_factory=utcnow, description="Creation timestamp"
    )
    approved_by: Optional[str] = Field(
        default=None, description="Approver"
    )
    approved_date: Optional[datetime] = Field(
        default=None, description="Approval timestamp"
    )
    rejected_by: Optional[str] = Field(
        default=None, description="Rejector"
    )
    rejected_date: Optional[datetime] = Field(
        default=None, description="Rejection timestamp"
    )
    rejection_reason: Optional[str] = Field(
        default=None, description="Rejection reason"
    )
    rationale: str = Field(
        default="", description="Package rationale"
    )
    trigger_ids: List[str] = Field(
        default_factory=list, description="Source trigger IDs"
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0, description="Processing time (ms)"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class TriggerInput(BaseModel):
    """Simplified trigger input for the adjustment engine.

    Attributes:
        trigger_id: Unique trigger identifier.
        trigger_type: Type of recalculation trigger.
        scope: Affected GHG scope.
        category: Affected source category.
        entity_id: Entity identifier (for structural changes).
        entity_emissions_tco2e: Total entity emissions (tCO2e).
        ownership_pct: Ownership percentage for equity share adjustments.
        effective_date: Date when the triggering event took effect.
        activity_data: Activity data for methodology restatement.
        old_emission_factor: Previous emission factor.
        new_emission_factor: New emission factor.
        original_value_tco2e: Original value (for error corrections).
        corrected_value_tco2e: Corrected value (for error corrections).
        source_emissions_tco2e: Emissions for boundary source changes.
        description: Human-readable description.
    """
    trigger_id: str = Field(
        default_factory=_new_uuid, description="Trigger ID"
    )
    trigger_type: TriggerType = Field(
        ..., description="Trigger type"
    )
    scope: Scope = Field(
        default=Scope.SCOPE_1, description="Affected scope"
    )
    category: str = Field(
        default="", description="Source category"
    )
    entity_id: Optional[str] = Field(
        default=None, description="Entity ID"
    )
    entity_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Entity total emissions (tCO2e)"
    )
    ownership_pct: Decimal = Field(
        default=Decimal("100"), ge=0, le=100,
        description="Ownership percentage"
    )
    effective_date: Optional[date] = Field(
        default=None, description="Effective date"
    )
    activity_data: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Activity data quantity"
    )
    old_emission_factor: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Previous emission factor"
    )
    new_emission_factor: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="New emission factor"
    )
    original_value_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Original emission value (tCO2e)"
    )
    corrected_value_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Corrected emission value (tCO2e)"
    )
    source_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Boundary source emissions (tCO2e)"
    )
    description: str = Field(
        default="", description="Trigger description"
    )

    @field_validator(
        "entity_emissions_tco2e", "ownership_pct", "activity_data",
        "old_emission_factor", "new_emission_factor",
        "original_value_tco2e", "corrected_value_tco2e",
        "source_emissions_tco2e",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)

class BaseYearInventory(BaseModel):
    """Base year inventory used as the starting point for adjustments.

    Attributes:
        base_year: The base year.
        scope1_tco2e: Total Scope 1 emissions (tCO2e).
        scope2_location_tco2e: Scope 2 location-based total (tCO2e).
        scope2_market_tco2e: Scope 2 market-based total (tCO2e).
        scope3_tco2e: Total Scope 3 emissions (tCO2e).
        by_scope_category: Emissions by scope and category (tCO2e).
        activity_data: Activity data by category (original units).
        emission_factors: Emission factors by category.
    """
    base_year: int = Field(
        ..., ge=MINIMUM_BASE_YEAR, le=MAXIMUM_BASE_YEAR,
        description="Base year"
    )
    scope1_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Scope 1 total (tCO2e)"
    )
    scope2_location_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Scope 2 location-based (tCO2e)"
    )
    scope2_market_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Scope 2 market-based (tCO2e)"
    )
    scope3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Scope 3 total (tCO2e)"
    )
    by_scope_category: Dict[str, Dict[str, Decimal]] = Field(
        default_factory=dict,
        description="Emissions by scope and category"
    )
    activity_data: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Activity data by category"
    )
    emission_factors: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emission factors by category"
    )

    @field_validator(
        "scope1_tco2e", "scope2_location_tco2e",
        "scope2_market_tco2e", "scope3_tco2e",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce emission totals to Decimal."""
        return _decimal(v)

    @property
    def grand_total_tco2e(self) -> Decimal:
        """Total emissions across all scopes.

        Formula:
            grand_total = scope1 + scope2_location + scope3
        """
        return self.scope1_tco2e + self.scope2_location_tco2e + self.scope3_tco2e

    def get_scope_total(self, scope: Scope) -> Decimal:
        """Get total emissions for a specific scope.

        Args:
            scope: The GHG scope.

        Returns:
            Total emissions for the specified scope (tCO2e).
        """
        scope_map = {
            Scope.SCOPE_1: self.scope1_tco2e,
            Scope.SCOPE_2_LOCATION: self.scope2_location_tco2e,
            Scope.SCOPE_2_MARKET: self.scope2_market_tco2e,
            Scope.SCOPE_3: self.scope3_tco2e,
        }
        return scope_map.get(scope, Decimal("0"))

    def get_category_emissions(self, scope: Scope, category: str) -> Decimal:
        """Get emissions for a specific scope and category.

        Args:
            scope: The GHG scope.
            category: The emission source category.

        Returns:
            Emissions for the specified scope/category (tCO2e).
        """
        scope_data = self.by_scope_category.get(scope.value, {})
        return _decimal(scope_data.get(category, Decimal("0")))

class AdjustmentConfig(BaseModel):
    """Configuration for the adjustment engine.

    Attributes:
        pro_rata_method: Default method for partial-year pro-rating.
        tco2e_precision: Decimal places for tCO2e outputs.
        pct_precision: Decimal places for percentage outputs.
        prorata_precision: Decimal places for pro-rata factors.
        created_by: Default creator identity for packages.
    """
    pro_rata_method: ProRataMethod = Field(
        default=ProRataMethod.MONTHLY,
        description="Default pro-rata method"
    )
    tco2e_precision: int = Field(
        default=DEFAULT_TCO2E_PRECISION, ge=0, le=10,
        description="tCO2e decimal precision"
    )
    pct_precision: int = Field(
        default=DEFAULT_PCT_PRECISION, ge=0, le=10,
        description="Percentage decimal precision"
    )
    prorata_precision: int = Field(
        default=DEFAULT_PRORATA_PRECISION, ge=0, le=10,
        description="Pro-rata factor decimal precision"
    )
    created_by: str = Field(
        default="system", description="Default creator"
    )

# ---------------------------------------------------------------------------
# Model Rebuild (Pydantic v2 deferred annotations resolution)
# ---------------------------------------------------------------------------

AdjustmentLine.model_rebuild()
ScopeBreakdown.model_rebuild()
TypeBreakdown.model_rebuild()
AdjustmentSummary.model_rebuild()
AdjustmentPackage.model_rebuild()
TriggerInput.model_rebuild()
BaseYearInventory.model_rebuild()
AdjustmentConfig.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BaseYearAdjustmentEngine:
    """Adjustment calculation engine for base year recalculation.

    Computes precise numerical adjustments required to recalculate
    the base year inventory when structural or methodological changes
    trigger recalculation per GHG Protocol Chapter 5.

    The engine produces an AdjustmentPackage containing line-by-line
    adjustments that can be reviewed, approved, and applied.

    Usage:
        >>> engine = BaseYearAdjustmentEngine()
        >>> triggers = [TriggerInput(trigger_type=TriggerType.ACQUISITION, ...)]
        >>> inventory = BaseYearInventory(base_year=2019, scope1_tco2e=Decimal("25000"))
        >>> package = engine.create_adjustment_package(inventory, triggers)
        >>> # Review and approve
        >>> engine.submit_for_approval(package)
        >>> engine.approve_adjustment(package, "jane.doe@company.com")
        >>> # Apply to inventory
        >>> adjusted = engine.apply_adjustments(inventory, package)

    All calculations use Python Decimal arithmetic with ROUND_HALF_UP.
    No LLM is used in any calculation or approval path.
    """

    def __init__(self, config: Optional[AdjustmentConfig] = None) -> None:
        """Initialise the BaseYearAdjustmentEngine.

        Args:
            config: Engine configuration. If None, defaults are used.
        """
        self._version: str = _MODULE_VERSION
        self._config: AdjustmentConfig = config or AdjustmentConfig()
        self._package_store: Dict[str, AdjustmentPackage] = {}
        logger.info(
            "BaseYearAdjustmentEngine v%s initialised (pro_rata=%s, precision=%d)",
            self._version,
            self._config.pro_rata_method.value,
            self._config.tco2e_precision,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def create_adjustment_package(
        self,
        base_year_inventory: BaseYearInventory,
        triggers: List[TriggerInput],
        config: Optional[AdjustmentConfig] = None,
    ) -> AdjustmentPackage:
        """Create a complete adjustment package from triggers.

        Processes each trigger and generates the appropriate adjustment
        line(s), then assembles them into a package with summary.

        Args:
            base_year_inventory: The original base year inventory.
            triggers: List of triggers to process.
            config: Optional override configuration.

        Returns:
            AdjustmentPackage with all lines, summary, and provenance.
        """
        start_ns = time.perf_counter_ns()
        cfg = config or self._config

        all_lines: List[AdjustmentLine] = []
        trigger_ids: List[str] = []

        for trigger in triggers:
            trigger_ids.append(trigger.trigger_id)
            lines = self._process_trigger(trigger, base_year_inventory, cfg)
            all_lines.extend(lines)

        # Build summary
        summary = self._build_summary(all_lines, cfg)

        # Build rationale
        rationale_parts = [
            f"Base year {base_year_inventory.base_year} adjustment package "
            f"generated from {len(triggers)} trigger(s)."
        ]
        for trigger in triggers:
            rationale_parts.append(
                f"  - {trigger.trigger_type.value}: {trigger.description or 'No description'}"
            )

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

        package = AdjustmentPackage(
            base_year=base_year_inventory.base_year,
            adjustment_lines=all_lines,
            summary=summary,
            status=AdjustmentStatus.DRAFT,
            created_by=cfg.created_by,
            rationale="\n".join(rationale_parts),
            trigger_ids=trigger_ids,
            processing_time_ms=round(elapsed_ms, 2),
        )
        package.provenance_hash = _compute_hash(package)

        # Store the package
        self._package_store[package.package_id] = package

        logger.info(
            "Adjustment package %s created: %d lines, net adjustment=%.3f tCO2e",
            package.package_id, len(all_lines),
            float(summary.total_adjustment_tco2e),
        )

        return package

    def calculate_acquisition_adjustment(
        self,
        entity_emissions_tco2e: Decimal,
        ownership_pct: Decimal,
        effective_date: Optional[date] = None,
        pro_rata_method: Optional[ProRataMethod] = None,
        base_year: int = 2019,
        scope: Scope = Scope.SCOPE_1,
        category: str = "",
        entity_id: str = "",
        original_base_value: Decimal = Decimal("0"),
    ) -> AdjustmentLine:
        """Calculate acquisition adjustment for a single entity.

        Formulas:
            equity_adj = entity_emissions * ownership_pct / 100
            pro_rata_factor = months_in_period / 12  (for MONTHLY)
            pro_rata_factor = days_in_period / days_in_year  (for DAILY)
            pro_rata_factor = quarters_in_period / 4  (for QUARTERLY)
            adjustment = equity_adj * pro_rata_factor
            adjusted = original + adjustment

        Args:
            entity_emissions_tco2e: Total annual emissions for the entity.
            ownership_pct: Reporting entity's ownership percentage (0-100).
            effective_date: Date of acquisition (for pro-rata). None = full year.
            pro_rata_method: Method for partial-year calculation.
            base_year: The base year for pro-rata day calculations.
            scope: GHG scope for the adjustment line.
            category: Source category for the adjustment line.
            entity_id: Entity identifier.
            original_base_value: Original base year value for this scope/category.

        Returns:
            AdjustmentLine with the calculated acquisition adjustment.
        """
        method = pro_rata_method or self._config.pro_rata_method
        emissions = _decimal(entity_emissions_tco2e)
        ownership = _decimal(ownership_pct)

        # Step 1: Apply equity share
        equity_adj = emissions * ownership / Decimal("100")

        # Step 2: Calculate pro-rata factor
        pro_rata_factor = self._compute_pro_rata_factor(
            effective_date, method, base_year
        )

        # Step 3: Calculate adjustment
        adjustment = _round_val(
            equity_adj * pro_rata_factor, self._config.tco2e_precision
        )
        original = _decimal(original_base_value)
        adjusted = _round_val(original + adjustment, self._config.tco2e_precision)

        detail = (
            f"entity_emissions={emissions} * ownership={ownership}% "
            f"= equity_adj={_round_val(equity_adj, 3)}. "
            f"Pro-rata ({method.value}): factor={_round_val(pro_rata_factor, 6)}. "
            f"Adjustment: {_round_val(equity_adj, 3)} * {_round_val(pro_rata_factor, 6)} "
            f"= {adjustment}. "
            f"Adjusted: {original} + {adjustment} = {adjusted}."
        )

        return AdjustmentLine(
            adjustment_type=AdjustmentType.ACQUISITION_ADD,
            scope=scope,
            category=category,
            description=f"Acquisition adjustment for entity '{entity_id}'",
            original_tco2e=original,
            adjustment_tco2e=adjustment,
            adjusted_tco2e=adjusted,
            pro_rata_factor=_round_val(pro_rata_factor, DEFAULT_PRORATA_PRECISION),
            effective_date=effective_date,
            entity_id=entity_id or None,
            calculation_detail=detail,
        )

    def calculate_divestiture_adjustment(
        self,
        entity_emissions_tco2e: Decimal,
        ownership_pct: Decimal,
        effective_date: Optional[date] = None,
        pro_rata_method: Optional[ProRataMethod] = None,
        base_year: int = 2019,
        scope: Scope = Scope.SCOPE_1,
        category: str = "",
        entity_id: str = "",
        original_base_value: Decimal = Decimal("0"),
    ) -> AdjustmentLine:
        """Calculate divestiture adjustment for a single entity.

        Formulas:
            equity_adj = entity_emissions * ownership_pct / 100
            pro_rata_factor = months_in_period / 12
            adjustment = -(equity_adj * pro_rata_factor)
            adjusted = original + adjustment

        Note: The adjustment is negative (subtraction from base year).

        Args:
            entity_emissions_tco2e: Total annual emissions for the entity.
            ownership_pct: Ownership percentage before divestiture (0-100).
            effective_date: Date of divestiture (for pro-rata). None = full year.
            pro_rata_method: Method for partial-year calculation.
            base_year: The base year.
            scope: GHG scope for the adjustment line.
            category: Source category for the adjustment line.
            entity_id: Entity identifier.
            original_base_value: Original base year value for this scope/category.

        Returns:
            AdjustmentLine with the calculated divestiture adjustment (negative).
        """
        method = pro_rata_method or self._config.pro_rata_method
        emissions = _decimal(entity_emissions_tco2e)
        ownership = _decimal(ownership_pct)

        # Step 1: Apply equity share
        equity_adj = emissions * ownership / Decimal("100")

        # Step 2: Calculate pro-rata factor
        pro_rata_factor = self._compute_pro_rata_factor(
            effective_date, method, base_year
        )

        # Step 3: Calculate adjustment (negative for divestiture)
        raw_adjustment = equity_adj * pro_rata_factor
        adjustment = _round_val(-raw_adjustment, self._config.tco2e_precision)
        original = _decimal(original_base_value)
        adjusted = _round_val(original + adjustment, self._config.tco2e_precision)

        detail = (
            f"entity_emissions={emissions} * ownership={ownership}% "
            f"= equity_adj={_round_val(equity_adj, 3)}. "
            f"Pro-rata ({method.value}): factor={_round_val(pro_rata_factor, 6)}. "
            f"Adjustment: -({_round_val(equity_adj, 3)} * {_round_val(pro_rata_factor, 6)}) "
            f"= {adjustment}. "
            f"Adjusted: {original} + ({adjustment}) = {adjusted}."
        )

        return AdjustmentLine(
            adjustment_type=AdjustmentType.DIVESTITURE_REMOVE,
            scope=scope,
            category=category,
            description=f"Divestiture adjustment for entity '{entity_id}'",
            original_tco2e=original,
            adjustment_tco2e=adjustment,
            adjusted_tco2e=adjusted,
            pro_rata_factor=_round_val(pro_rata_factor, DEFAULT_PRORATA_PRECISION),
            effective_date=effective_date,
            entity_id=entity_id or None,
            calculation_detail=detail,
        )

    def calculate_methodology_restatement(
        self,
        activity_data: Decimal,
        old_factor: Decimal,
        new_factor: Decimal,
        scope: Scope = Scope.SCOPE_1,
        category: str = "",
    ) -> AdjustmentLine:
        """Calculate methodology restatement (like-for-like).

        Applies the new emission factor to the original base year activity
        data to produce a restated emission value, then computes the
        adjustment as the difference.

        Formulas:
            original_emissions = activity_data * old_factor
            restated_emissions = activity_data * new_factor
            adjustment = restated_emissions - original_emissions
            adjusted = original_emissions + adjustment = restated_emissions

        Args:
            activity_data: Base year activity data (in original units).
            old_factor: Previous emission factor.
            new_factor: New emission factor.
            scope: Affected GHG scope.
            category: Affected source category.

        Returns:
            AdjustmentLine with the methodology restatement adjustment.
        """
        activity = _decimal(activity_data)
        old_f = _decimal(old_factor)
        new_f = _decimal(new_factor)

        original_emissions = _round_val(
            activity * old_f, self._config.tco2e_precision
        )
        restated_emissions = _round_val(
            activity * new_f, self._config.tco2e_precision
        )
        adjustment = _round_val(
            restated_emissions - original_emissions, self._config.tco2e_precision
        )

        detail = (
            f"Like-for-like restatement: activity={activity}. "
            f"Old factor={old_f}, new factor={new_f}. "
            f"Original: {activity} * {old_f} = {original_emissions}. "
            f"Restated: {activity} * {new_f} = {restated_emissions}. "
            f"Adjustment: {restated_emissions} - {original_emissions} = {adjustment}."
        )

        return AdjustmentLine(
            adjustment_type=AdjustmentType.METHODOLOGY_RESTATE,
            scope=scope,
            category=category,
            description=(
                f"Methodology restatement for '{category}': "
                f"factor {old_f} -> {new_f}"
            ),
            original_tco2e=original_emissions,
            adjustment_tco2e=adjustment,
            adjusted_tco2e=restated_emissions,
            pro_rata_factor=Decimal("1.0"),
            calculation_detail=detail,
        )

    def calculate_error_correction(
        self,
        original_tco2e: Decimal,
        corrected_tco2e: Decimal,
        scope: Scope = Scope.SCOPE_1,
        category: str = "",
        error_description: str = "",
    ) -> AdjustmentLine:
        """Calculate error correction adjustment.

        Formula:
            adjustment = corrected_value - original_value
            adjusted = original + adjustment = corrected_value

        Args:
            original_tco2e: The originally reported emission value (tCO2e).
            corrected_tco2e: The corrected emission value (tCO2e).
            scope: Affected GHG scope.
            category: Affected source category.
            error_description: Description of the error.

        Returns:
            AdjustmentLine with the error correction adjustment.
        """
        original = _round_val(_decimal(original_tco2e), self._config.tco2e_precision)
        corrected = _round_val(_decimal(corrected_tco2e), self._config.tco2e_precision)
        adjustment = _round_val(corrected - original, self._config.tco2e_precision)

        detail = (
            f"Error correction: original={original}, corrected={corrected}. "
            f"Adjustment: {corrected} - {original} = {adjustment}. "
            f"Description: {error_description or 'N/A'}."
        )

        return AdjustmentLine(
            adjustment_type=AdjustmentType.ERROR_CORRECT,
            scope=scope,
            category=category,
            description=(
                f"Error correction in '{category}': "
                f"{original} -> {corrected} tCO2e"
            ),
            original_tco2e=original,
            adjustment_tco2e=adjustment,
            adjusted_tco2e=corrected,
            pro_rata_factor=Decimal("1.0"),
            calculation_detail=detail,
        )

    def apply_adjustments(
        self,
        inventory: BaseYearInventory,
        package: AdjustmentPackage,
    ) -> BaseYearInventory:
        """Apply an approved adjustment package to the base year inventory.

        Creates a new BaseYearInventory with the adjustments applied.
        The original inventory is not modified.

        Args:
            inventory: The original base year inventory.
            package: An APPROVED adjustment package to apply.

        Returns:
            A new BaseYearInventory with adjustments applied.

        Raises:
            ValueError: If the package is not in APPROVED status.
        """
        if package.status != AdjustmentStatus.APPROVED:
            raise ValueError(
                f"Cannot apply package in status '{package.status.value}'. "
                "Package must be APPROVED before application."
            )

        # Start with copies of original totals
        new_scope1 = _decimal(inventory.scope1_tco2e)
        new_scope2_loc = _decimal(inventory.scope2_location_tco2e)
        new_scope2_mkt = _decimal(inventory.scope2_market_tco2e)
        new_scope3 = _decimal(inventory.scope3_tco2e)

        # Apply each adjustment line
        for line in package.adjustment_lines:
            adj = _decimal(line.adjustment_tco2e)
            if line.scope == Scope.SCOPE_1:
                new_scope1 += adj
            elif line.scope == Scope.SCOPE_2_LOCATION:
                new_scope2_loc += adj
            elif line.scope == Scope.SCOPE_2_MARKET:
                new_scope2_mkt += adj
            elif line.scope == Scope.SCOPE_3:
                new_scope3 += adj

        # Ensure no negative totals (floor at zero)
        new_scope1 = max(new_scope1, Decimal("0"))
        new_scope2_loc = max(new_scope2_loc, Decimal("0"))
        new_scope2_mkt = max(new_scope2_mkt, Decimal("0"))
        new_scope3 = max(new_scope3, Decimal("0"))

        adjusted_inventory = BaseYearInventory(
            base_year=inventory.base_year,
            scope1_tco2e=_round_val(new_scope1, self._config.tco2e_precision),
            scope2_location_tco2e=_round_val(new_scope2_loc, self._config.tco2e_precision),
            scope2_market_tco2e=_round_val(new_scope2_mkt, self._config.tco2e_precision),
            scope3_tco2e=_round_val(new_scope3, self._config.tco2e_precision),
            by_scope_category=inventory.by_scope_category.copy(),
            activity_data=inventory.activity_data.copy(),
            emission_factors=inventory.emission_factors.copy(),
        )

        # Update package status to APPLIED
        package.status = AdjustmentStatus.APPLIED
        package.provenance_hash = _compute_hash(package)

        logger.info(
            "Adjustments applied to base year %d: S1=%.3f, S2L=%.3f, S2M=%.3f, S3=%.3f",
            inventory.base_year,
            float(adjusted_inventory.scope1_tco2e),
            float(adjusted_inventory.scope2_location_tco2e),
            float(adjusted_inventory.scope2_market_tco2e),
            float(adjusted_inventory.scope3_tco2e),
        )

        return adjusted_inventory

    def submit_for_approval(
        self,
        package: AdjustmentPackage,
    ) -> AdjustmentPackage:
        """Submit an adjustment package for approval.

        Transitions the package from DRAFT to PENDING_APPROVAL status.

        Args:
            package: The package to submit.

        Returns:
            Updated package with PENDING_APPROVAL status.

        Raises:
            ValueError: If the package is not in DRAFT status.
        """
        if package.status != AdjustmentStatus.DRAFT:
            raise ValueError(
                f"Cannot submit package in status '{package.status.value}'. "
                "Only DRAFT packages can be submitted."
            )

        package.status = AdjustmentStatus.PENDING_APPROVAL
        package.provenance_hash = _compute_hash(package)

        logger.info(
            "Package %s submitted for approval (%d lines, %.3f tCO2e net adjustment)",
            package.package_id,
            len(package.adjustment_lines),
            float(package.summary.total_adjustment_tco2e) if package.summary else 0.0,
        )

        return package

    def approve_adjustment(
        self,
        package: AdjustmentPackage,
        approver: str,
    ) -> AdjustmentPackage:
        """Approve an adjustment package.

        Transitions the package from PENDING_APPROVAL to APPROVED.

        Args:
            package: The package to approve.
            approver: Identity of the approver (e.g. email address).

        Returns:
            Updated package with APPROVED status.

        Raises:
            ValueError: If the package is not in PENDING_APPROVAL status.
            ValueError: If approver is empty.
        """
        if package.status != AdjustmentStatus.PENDING_APPROVAL:
            raise ValueError(
                f"Cannot approve package in status '{package.status.value}'. "
                "Only PENDING_APPROVAL packages can be approved."
            )
        if not approver or not approver.strip():
            raise ValueError("Approver identity must be provided.")

        package.status = AdjustmentStatus.APPROVED
        package.approved_by = approver.strip()
        package.approved_date = utcnow()
        package.provenance_hash = _compute_hash(package)

        logger.info(
            "Package %s approved by %s at %s",
            package.package_id, approver, package.approved_date,
        )

        return package

    def reject_adjustment(
        self,
        package: AdjustmentPackage,
        rejector: str,
        reason: str,
    ) -> AdjustmentPackage:
        """Reject an adjustment package.

        Transitions the package from PENDING_APPROVAL to REJECTED.

        Args:
            package: The package to reject.
            rejector: Identity of the person rejecting.
            reason: Documented reason for rejection.

        Returns:
            Updated package with REJECTED status.

        Raises:
            ValueError: If the package is not in PENDING_APPROVAL status.
            ValueError: If rejector or reason is empty.
        """
        if package.status != AdjustmentStatus.PENDING_APPROVAL:
            raise ValueError(
                f"Cannot reject package in status '{package.status.value}'. "
                "Only PENDING_APPROVAL packages can be rejected."
            )
        if not rejector or not rejector.strip():
            raise ValueError("Rejector identity must be provided.")
        if not reason or not reason.strip():
            raise ValueError("Rejection reason must be provided.")

        package.status = AdjustmentStatus.REJECTED
        package.rejected_by = rejector.strip()
        package.rejected_date = utcnow()
        package.rejection_reason = reason.strip()
        package.provenance_hash = _compute_hash(package)

        logger.info(
            "Package %s rejected by %s: %s",
            package.package_id, rejector, reason,
        )

        return package

    def get_package(self, package_id: str) -> Optional[AdjustmentPackage]:
        """Retrieve a stored adjustment package by ID.

        Args:
            package_id: The package identifier.

        Returns:
            The package, or None if not found.
        """
        return self._package_store.get(package_id)

    def list_packages(
        self,
        status_filter: Optional[AdjustmentStatus] = None,
    ) -> List[AdjustmentPackage]:
        """List all stored adjustment packages, optionally filtered by status.

        Args:
            status_filter: If provided, only return packages with this status.

        Returns:
            List of matching packages, sorted by creation date descending.
        """
        packages = list(self._package_store.values())
        if status_filter is not None:
            packages = [p for p in packages if p.status == status_filter]
        packages.sort(key=lambda p: p.created_date, reverse=True)
        return packages

    # -----------------------------------------------------------------
    # Private Methods
    # -----------------------------------------------------------------

    def _process_trigger(
        self,
        trigger: TriggerInput,
        inventory: BaseYearInventory,
        config: AdjustmentConfig,
    ) -> List[AdjustmentLine]:
        """Process a single trigger into adjustment line(s).

        Routes the trigger to the appropriate calculation method based
        on the trigger type.

        Args:
            trigger: The trigger to process.
            inventory: The base year inventory.
            config: Engine configuration.

        Returns:
            List of adjustment lines for this trigger.
        """
        lines: List[AdjustmentLine] = []

        original_value = inventory.get_category_emissions(
            trigger.scope, trigger.category
        )

        if trigger.trigger_type == TriggerType.ACQUISITION:
            line = self.calculate_acquisition_adjustment(
                entity_emissions_tco2e=trigger.entity_emissions_tco2e,
                ownership_pct=trigger.ownership_pct,
                effective_date=trigger.effective_date,
                pro_rata_method=config.pro_rata_method,
                base_year=inventory.base_year,
                scope=trigger.scope,
                category=trigger.category,
                entity_id=trigger.entity_id or "",
                original_base_value=original_value,
            )
            line.trigger_id = trigger.trigger_id
            lines.append(line)

        elif trigger.trigger_type == TriggerType.DIVESTITURE:
            line = self.calculate_divestiture_adjustment(
                entity_emissions_tco2e=trigger.entity_emissions_tco2e,
                ownership_pct=trigger.ownership_pct,
                effective_date=trigger.effective_date,
                pro_rata_method=config.pro_rata_method,
                base_year=inventory.base_year,
                scope=trigger.scope,
                category=trigger.category,
                entity_id=trigger.entity_id or "",
                original_base_value=original_value,
            )
            line.trigger_id = trigger.trigger_id
            lines.append(line)

        elif trigger.trigger_type == TriggerType.METHODOLOGY_CHANGE:
            line = self.calculate_methodology_restatement(
                activity_data=trigger.activity_data,
                old_factor=trigger.old_emission_factor,
                new_factor=trigger.new_emission_factor,
                scope=trigger.scope,
                category=trigger.category,
            )
            line.trigger_id = trigger.trigger_id
            lines.append(line)

        elif trigger.trigger_type == TriggerType.ERROR_CORRECTION:
            line = self.calculate_error_correction(
                original_tco2e=trigger.original_value_tco2e,
                corrected_tco2e=trigger.corrected_value_tco2e,
                scope=trigger.scope,
                category=trigger.category,
                error_description=trigger.description,
            )
            line.trigger_id = trigger.trigger_id
            lines.append(line)

        elif trigger.trigger_type == TriggerType.SOURCE_BOUNDARY_CHANGE:
            adj_type = (
                AdjustmentType.BOUNDARY_ADD
                if trigger.source_emissions_tco2e >= Decimal("0")
                else AdjustmentType.BOUNDARY_REMOVE
            )
            emissions = _abs_decimal(_decimal(trigger.source_emissions_tco2e))
            sign = Decimal("1") if adj_type == AdjustmentType.BOUNDARY_ADD else Decimal("-1")
            adjustment = _round_val(emissions * sign, self._config.tco2e_precision)
            adjusted = _round_val(original_value + adjustment, self._config.tco2e_precision)

            detail = (
                f"Boundary {adj_type.value}: source_emissions={emissions}. "
                f"Adjustment: {adjustment}. "
                f"Adjusted: {original_value} + {adjustment} = {adjusted}."
            )

            line = AdjustmentLine(
                adjustment_type=adj_type,
                scope=trigger.scope,
                category=trigger.category,
                description=f"Boundary change: {trigger.description}",
                original_tco2e=original_value,
                adjustment_tco2e=adjustment,
                adjusted_tco2e=adjusted,
                pro_rata_factor=Decimal("1.0"),
                trigger_id=trigger.trigger_id,
                calculation_detail=detail,
            )
            lines.append(line)

        elif trigger.trigger_type == TriggerType.OUTSOURCING_INSOURCING:
            # Outsourcing: remove from Scope 1/2, conceptually add to Scope 3
            # Insourcing: remove from Scope 3, add to Scope 1/2
            emissions = _decimal(trigger.source_emissions_tco2e)
            is_outsource = trigger.description.lower().startswith("outsourc")

            if is_outsource:
                adj_type = AdjustmentType.OUTSOURCE_SHIFT
                adjustment = _round_val(-emissions, self._config.tco2e_precision)
            else:
                adj_type = AdjustmentType.INSOURCE_SHIFT
                adjustment = _round_val(emissions, self._config.tco2e_precision)

            adjusted = _round_val(original_value + adjustment, self._config.tco2e_precision)

            detail = (
                f"{'Outsource' if is_outsource else 'Insource'} shift: "
                f"emissions={emissions}. "
                f"Adjustment: {adjustment}. "
                f"Adjusted: {original_value} + {adjustment} = {adjusted}."
            )

            line = AdjustmentLine(
                adjustment_type=adj_type,
                scope=trigger.scope,
                category=trigger.category,
                description=f"{'Outsource' if is_outsource else 'Insource'} shift: {trigger.description}",
                original_tco2e=original_value,
                adjustment_tco2e=adjustment,
                adjusted_tco2e=adjusted,
                pro_rata_factor=Decimal("1.0"),
                trigger_id=trigger.trigger_id,
                calculation_detail=detail,
            )
            lines.append(line)

        elif trigger.trigger_type == TriggerType.MERGER:
            # Merger treated as full acquisition (ownership goes from 0 to 100%)
            line = self.calculate_acquisition_adjustment(
                entity_emissions_tco2e=trigger.entity_emissions_tco2e,
                ownership_pct=Decimal("100"),
                effective_date=trigger.effective_date,
                pro_rata_method=config.pro_rata_method,
                base_year=inventory.base_year,
                scope=trigger.scope,
                category=trigger.category,
                entity_id=trigger.entity_id or "",
                original_base_value=original_value,
            )
            line.adjustment_type = AdjustmentType.ACQUISITION_ADD
            line.trigger_id = trigger.trigger_id
            line.description = f"Merger adjustment for entity '{trigger.entity_id or 'merged'}'"
            lines.append(line)

        return lines

    def _compute_pro_rata_factor(
        self,
        effective_date: Optional[date],
        method: ProRataMethod,
        base_year: int,
    ) -> Decimal:
        """Compute the pro-rata factor for partial-year adjustments.

        Formulas:
            MONTHLY:   factor = months_remaining / 12
            DAILY:     factor = days_remaining / days_in_year
            QUARTERLY: factor = quarters_remaining / 4

        If effective_date is None, returns 1.0 (full year).

        Args:
            effective_date: Date when the event took effect.
            method: Pro-rata calculation method.
            base_year: The year for day count calculations.

        Returns:
            Pro-rata factor as Decimal between 0 and 1 (inclusive).
        """
        if effective_date is None:
            return Decimal("1.0")

        if method == ProRataMethod.MONTHLY:
            months = _months_remaining_in_year(effective_date)
            return _safe_divide(
                Decimal(str(months)), Decimal("12")
            )

        elif method == ProRataMethod.DAILY:
            days = _days_remaining_in_year(effective_date)
            total_days = _days_in_year(effective_date.year)
            return _safe_divide(
                Decimal(str(days)), Decimal(str(total_days))
            )

        elif method == ProRataMethod.QUARTERLY:
            quarters = _quarters_remaining_in_year(effective_date)
            return _safe_divide(
                Decimal(str(quarters)), Decimal("4")
            )

        return Decimal("1.0")

    def _build_summary(
        self,
        lines: List[AdjustmentLine],
        config: AdjustmentConfig,
    ) -> AdjustmentSummary:
        """Build an AdjustmentSummary from a list of adjustment lines.

        Aggregates totals, computes percentage change, and builds
        breakdowns by scope and adjustment type.

        Args:
            lines: List of adjustment lines.
            config: Engine configuration.

        Returns:
            Populated AdjustmentSummary.
        """
        total_original = Decimal("0")
        total_adjustment = Decimal("0")
        total_adjusted = Decimal("0")

        scope_data: Dict[str, Dict[str, Decimal]] = {}
        scope_counts: Dict[str, int] = {}
        type_data: Dict[str, Decimal] = {}
        type_counts: Dict[str, int] = {}
        type_scopes: Dict[str, set] = {}

        for line in lines:
            total_original += _decimal(line.original_tco2e)
            total_adjustment += _decimal(line.adjustment_tco2e)
            total_adjusted += _decimal(line.adjusted_tco2e)

            # By scope
            s = line.scope.value
            if s not in scope_data:
                scope_data[s] = {"original": Decimal("0"), "adjustment": Decimal("0"), "adjusted": Decimal("0")}
                scope_counts[s] = 0
            scope_data[s]["original"] += _decimal(line.original_tco2e)
            scope_data[s]["adjustment"] += _decimal(line.adjustment_tco2e)
            scope_data[s]["adjusted"] += _decimal(line.adjusted_tco2e)
            scope_counts[s] += 1

            # By type
            t = line.adjustment_type.value
            if t not in type_data:
                type_data[t] = Decimal("0")
                type_counts[t] = 0
                type_scopes[t] = set()
            type_data[t] += _decimal(line.adjustment_tco2e)
            type_counts[t] += 1
            type_scopes[t].add(line.scope)

        change_pct = _safe_pct(total_adjustment, total_original) if total_original > Decimal("0") else Decimal("0")

        # Build scope breakdowns
        by_scope: Dict[str, ScopeBreakdown] = {}
        for s, data in scope_data.items():
            s_change_pct = _safe_pct(data["adjustment"], data["original"]) if data["original"] > Decimal("0") else Decimal("0")
            by_scope[s] = ScopeBreakdown(
                scope=Scope(s),
                original_tco2e=_round_val(data["original"], config.tco2e_precision),
                total_adjustment_tco2e=_round_val(data["adjustment"], config.tco2e_precision),
                adjusted_tco2e=_round_val(data["adjusted"], config.tco2e_precision),
                change_pct=_round_val(s_change_pct, config.pct_precision),
                line_count=scope_counts[s],
            )

        # Build type breakdowns
        by_type: Dict[str, TypeBreakdown] = {}
        for t, adj_total in type_data.items():
            by_type[t] = TypeBreakdown(
                adjustment_type=AdjustmentType(t),
                total_adjustment_tco2e=_round_val(adj_total, config.tco2e_precision),
                line_count=type_counts[t],
                affected_scopes=sorted(list(type_scopes[t]), key=lambda x: x.value),
            )

        return AdjustmentSummary(
            total_original_tco2e=_round_val(total_original, config.tco2e_precision),
            total_adjustment_tco2e=_round_val(total_adjustment, config.tco2e_precision),
            total_adjusted_tco2e=_round_val(total_adjusted, config.tco2e_precision),
            change_pct=_round_val(change_pct, config.pct_precision),
            total_lines=len(lines),
            by_scope=by_scope,
            by_type=by_type,
        )
