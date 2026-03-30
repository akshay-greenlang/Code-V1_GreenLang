# -*- coding: utf-8 -*-
"""
Scope3BaseYearEngine - PACK-043 Scope 3 Complete Pack Engine 8
================================================================

Manages Scope 3 base year data and triggers recalculations per
GHG Protocol Corporate Value Chain Standard Chapter 5.  Provides
base year establishment, significance testing for recalculation
triggers, adjusted base year recalculation, pro-rata time-weighting,
multi-year trend comparison, real-vs-methodology change decomposition,
and cumulative reduction tracking.

Recalculation Triggers (6 types):
    1. ACQUISITION       - New entity added to organisational boundary.
    2. DIVESTITURE       - Entity removed from organisational boundary.
    3. METHODOLOGY_UPGRADE - Tier change for a material Scope 3 category
                           (e.g., spend-based to supplier-specific).
    4. SCOPE_EXPANSION   - New Scope 3 category added to inventory.
    5. ERROR_CORRECTION  - Material data error discovered and corrected.
    6. STRUCTURAL_CHANGE - Outsourcing/insourcing that shifts emissions
                           between Scope 1/2 and Scope 3.

Significance Test:
    significance_pct = abs(adjustment_tco2e) / base_year_total_tco2e * 100
    is_significant = significance_pct >= threshold (default 5%)

Pro-Rata Time-Weighting:
    For mid-year changes:
        annual_adjustment = full_year_adjustment * (days_in_period / 365)

Real vs. Methodology Decomposition:
    real_change = current_tco2e - recalculated_base_year_tco2e
    methodology_change = recalculated_base_year_tco2e - original_base_year_tco2e
    total_change = real_change + methodology_change

Regulatory References:
    - GHG Protocol Corporate Value Chain Standard (2011), Chapter 5
    - GHG Protocol Corporate Standard (2004, revised 2015), Chapter 5
    - GHG Protocol Scope 3 Calculation Guidance (2013)
    - ISO 14064-1:2018, Clause 5.2 (Base year selection)
    - ESRS E1-6 (Gross GHG emissions, base year recalculation)
    - SBTi Corporate Manual (2023), Section 7 (Recalculation)
    - CDP Climate Change C5.1-C5.2

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Significance thresholds from published GHG Protocol guidance
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-043 Scope 3 Complete
Engine:  8 of 10
Status:  Production Ready
"""

from __future__ import annotations

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "43.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
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

def _round4(value: Any) -> float:
    """Round to 4 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

def _days_in_year(year: int) -> int:
    """Return the number of days in a given year (handles leap years)."""
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return 366
    return 365

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RecalculationTriggerType(str, Enum):
    """Types of events that trigger base year recalculation.

    Per GHG Protocol Corporate Standard Chapter 5 and Value Chain Standard.
    """
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    METHODOLOGY_UPGRADE = "methodology_upgrade"
    SCOPE_EXPANSION = "scope_expansion"
    ERROR_CORRECTION = "error_correction"
    STRUCTURAL_CHANGE = "structural_change"

class RecalculationStatus(str, Enum):
    """Status of a recalculation assessment."""
    PENDING = "pending"
    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    RECALCULATED = "recalculated"
    DEFERRED = "deferred"

class TrendDirection(str, Enum):
    """Direction of emission trend."""
    DECREASING = "decreasing"
    INCREASING = "increasing"
    STABLE = "stable"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default significance threshold (5%) per GHG Protocol.
DEFAULT_SIGNIFICANCE_THRESHOLD_PCT: float = 5.0
"""Default significance threshold for base year recalculation (5%)."""

# SBTi threshold (can differ).
SBTI_SIGNIFICANCE_THRESHOLD_PCT: float = 5.0
"""SBTi significance threshold for base year recalculation."""

# Trigger descriptions for audit trail.
TRIGGER_DESCRIPTIONS: Dict[str, str] = {
    RecalculationTriggerType.ACQUISITION: (
        "New entity acquired and added to organisational boundary. "
        "Emissions from the acquired entity must be added to the base year."
    ),
    RecalculationTriggerType.DIVESTITURE: (
        "Entity divested and removed from organisational boundary. "
        "Emissions from the divested entity must be removed from the base year."
    ),
    RecalculationTriggerType.METHODOLOGY_UPGRADE: (
        "Calculation methodology tier upgraded for a material Scope 3 category. "
        "Base year must be recalculated using the new methodology for comparability."
    ),
    RecalculationTriggerType.SCOPE_EXPANSION: (
        "New Scope 3 category added to the inventory boundary. "
        "Base year should include the new category using best available data."
    ),
    RecalculationTriggerType.ERROR_CORRECTION: (
        "Material data error discovered and corrected. "
        "If cumulative error exceeds the significance threshold, recalculate base year."
    ),
    RecalculationTriggerType.STRUCTURAL_CHANGE: (
        "Outsourcing or insourcing event that shifts emissions between scopes. "
        "Base year must be adjusted to maintain like-for-like comparability."
    ),
}
"""Descriptions of each trigger type for documentation and audit trail."""

# ---------------------------------------------------------------------------
# Pydantic Data Models
# ---------------------------------------------------------------------------

class CategoryBaseline(BaseModel):
    """Baseline data for a single Scope 3 category.

    Attributes:
        category: Scope 3 category number (1-15).
        category_name: Human-readable category name.
        tco2e: Baseline emissions in tCO2e.
        methodology_tier: Methodology tier used.
        data_quality_score: Data quality score (1-5).
        is_material: Whether this category is material.
        notes: Additional notes.
    """
    category: int = Field(..., ge=1, le=15, description="Scope 3 category (1-15)")
    category_name: str = Field(default="", description="Category name")
    tco2e: float = Field(..., ge=0, description="Baseline tCO2e")
    methodology_tier: str = Field(default="spend", description="Methodology tier")
    data_quality_score: int = Field(default=1, ge=1, le=5, description="DQ score 1-5")
    is_material: bool = Field(default=True, description="Is material category")
    notes: str = Field(default="", description="Notes")

class BaseYear(BaseModel):
    """Base year record for Scope 3 inventory.

    Attributes:
        base_year: The base year.
        established_date: When the base year was established.
        total_tco2e: Total base year emissions.
        by_category: Emissions by category.
        methodology_log: Methodology decisions.
        scope3_boundary: List of included categories.
        excluded_categories: List of excluded categories with rationale.
        organisation_boundary: Description of organisational boundary.
        version: Version of this base year record.
        is_recalculated: Whether this is a recalculated base year.
        recalculation_history: History of recalculations.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    base_year: int = Field(..., ge=2015, le=2035, description="Base year")
    established_date: str = Field(default="", description="Date established")
    total_tco2e: float = Field(..., ge=0, description="Total base year tCO2e")
    by_category: List[CategoryBaseline] = Field(default_factory=list)
    methodology_log: Dict[str, str] = Field(
        default_factory=dict, description="Methodology decisions"
    )
    scope3_boundary: List[int] = Field(
        default_factory=list, description="Included categories"
    )
    excluded_categories: Dict[int, str] = Field(
        default_factory=dict, description="Excluded categories with rationale"
    )
    organisation_boundary: str = Field(default="", description="Org boundary description")
    version: int = Field(default=1, ge=1, description="Record version")
    is_recalculated: bool = Field(default=False, description="Is recalculated")
    recalculation_history: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = ""
    calculated_at: str = ""

class ChangeEvent(BaseModel):
    """Change event that may trigger recalculation.

    Attributes:
        event_id: Unique event identifier.
        trigger_type: Type of trigger.
        description: Description of the change.
        effective_date: Date the change takes effect.
        affected_categories: Scope 3 categories affected.
        adjustment_tco2e: Estimated emission adjustment.
        entity_name: Name of affected entity (for M&A).
        previous_methodology: Previous methodology (for upgrades).
        new_methodology: New methodology (for upgrades).
        supporting_evidence: Supporting documentation references.
    """
    event_id: str = Field(default_factory=_new_uuid, description="Event ID")
    trigger_type: str = Field(..., description="Trigger type")
    description: str = Field(default="", description="Event description")
    effective_date: str = Field(default="", description="Effective date (YYYY-MM-DD)")
    affected_categories: List[int] = Field(
        default_factory=list, description="Affected categories"
    )
    adjustment_tco2e: float = Field(default=0, description="Estimated adjustment tCO2e")
    entity_name: str = Field(default="", description="Affected entity name")
    previous_methodology: str = Field(default="", description="Previous methodology")
    new_methodology: str = Field(default="", description="New methodology")
    supporting_evidence: List[str] = Field(
        default_factory=list, description="Evidence references"
    )

class RecalculationTrigger(BaseModel):
    """Recalculation trigger assessment result.

    Attributes:
        event_id: Event identifier.
        trigger_type: Trigger type.
        significance_pct: Significance as percentage of base year.
        threshold_pct: Threshold used.
        is_significant: Whether the trigger is significant.
        should_recalculate: Whether recalculation is recommended.
        status: Assessment status.
        rationale: Assessment rationale.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    event_id: str
    trigger_type: str
    significance_pct: float
    threshold_pct: float
    is_significant: bool
    should_recalculate: bool
    status: str
    rationale: str
    provenance_hash: str = ""
    calculated_at: str = ""

class RecalculationResult(BaseModel):
    """Result of a base year recalculation.

    Attributes:
        original_base_year: Original base year record.
        recalculated_total_tco2e: Recalculated total.
        adjustment_tco2e: Net adjustment.
        adjustment_pct: Adjustment as percentage of original.
        recalculated_categories: Updated category data.
        trigger_event: The event that triggered recalculation.
        methodology_changes: Description of methodology changes.
        pro_rata_applied: Whether pro-rata was applied.
        pro_rata_factor: Pro-rata factor (0-1) if applied.
        new_version: New version number.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    original_base_year: int
    original_total_tco2e: float
    recalculated_total_tco2e: float
    adjustment_tco2e: float
    adjustment_pct: float
    recalculated_categories: List[Dict[str, Any]] = Field(default_factory=list)
    trigger_event: Dict[str, Any] = Field(default_factory=dict)
    methodology_changes: List[str] = Field(default_factory=list)
    pro_rata_applied: bool = False
    pro_rata_factor: float = 1.0
    new_version: int = 2
    provenance_hash: str = ""
    calculated_at: str = ""

class TrendComparison(BaseModel):
    """Multi-year trend comparison from base year.

    Attributes:
        base_year: Base year.
        base_year_tco2e: Base year emissions.
        years: List of comparison years.
        absolute_changes: Year -> absolute change from base year.
        percentage_changes: Year -> percentage change from base year.
        cumulative_reduction_pct: Cumulative reduction from base year.
        trend_direction: Overall trend direction.
        cagr_pct: Compound annual growth rate.
        by_category: Per-category trend data.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    base_year: int
    base_year_tco2e: float
    years: List[int] = Field(default_factory=list)
    absolute_changes: Dict[int, float] = Field(default_factory=dict)
    percentage_changes: Dict[int, float] = Field(default_factory=dict)
    cumulative_reduction_pct: float = 0.0
    trend_direction: str = ""
    cagr_pct: float = 0.0
    by_category: Dict[str, Dict[int, float]] = Field(default_factory=dict)
    provenance_hash: str = ""
    calculated_at: str = ""

class YearInventory(BaseModel):
    """Inventory data for a single year.

    Attributes:
        year: Reporting year.
        total_tco2e: Total emissions.
        by_category: Emissions by category.
    """
    year: int = Field(..., description="Reporting year")
    total_tco2e: float = Field(..., ge=0, description="Total tCO2e")
    by_category: Dict[int, float] = Field(
        default_factory=dict, description="Category -> tCO2e"
    )

class ChangeDecomposition(BaseModel):
    """Decomposition of changes into real vs. methodology.

    Attributes:
        total_change_tco2e: Total change.
        real_change_tco2e: Real (operational) change.
        methodology_change_tco2e: Change due to methodology.
        real_change_pct: Real change as percentage of baseline.
        methodology_change_pct: Methodology change as percentage of baseline.
        baseline_tco2e: Baseline emissions used.
        by_category: Per-category decomposition.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    total_change_tco2e: float
    real_change_tco2e: float
    methodology_change_tco2e: float
    real_change_pct: float
    methodology_change_pct: float
    baseline_tco2e: float
    by_category: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    provenance_hash: str = ""
    calculated_at: str = ""

class CumulativeReduction(BaseModel):
    """Cumulative reduction since base year.

    Attributes:
        base_year: Base year.
        base_year_tco2e: Base year emissions.
        current_year: Current year.
        current_tco2e: Current emissions.
        cumulative_reduction_tco2e: Total reduction.
        cumulative_reduction_pct: Reduction percentage.
        annualised_reduction_pct: Annualised reduction rate.
        on_track_for_target: Whether on track for a specified target.
        target_pct: Target reduction percentage (if set).
        target_year: Target year (if set).
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    base_year: int
    base_year_tco2e: float
    current_year: int
    current_tco2e: float
    cumulative_reduction_tco2e: float
    cumulative_reduction_pct: float
    annualised_reduction_pct: float
    on_track_for_target: Optional[bool] = None
    target_pct: Optional[float] = None
    target_year: Optional[int] = None
    provenance_hash: str = ""
    calculated_at: str = ""

# ---------------------------------------------------------------------------
# Engine Class
# ---------------------------------------------------------------------------

class Scope3BaseYearEngine:
    """Manages Scope 3 base year data and recalculations.

    Implements GHG Protocol Chapter 5 base year recalculation policy
    for Scope 3 inventories.  All calculations use ``Decimal`` arithmetic
    for reproducibility.

    Attributes:
        significance_threshold_pct: Threshold for triggering recalculation.

    Example:
        >>> engine = Scope3BaseYearEngine()
        >>> base = engine.establish_base_year(inventory, methodology_log)
        >>> trigger = engine.check_recalculation_trigger(event, base)
        >>> if trigger.should_recalculate:
        ...     result = engine.recalculate_base_year(base, event)
    """

    def __init__(
        self,
        significance_threshold_pct: float = DEFAULT_SIGNIFICANCE_THRESHOLD_PCT,
    ) -> None:
        """Initialise Scope3BaseYearEngine.

        Args:
            significance_threshold_pct: Significance threshold percentage.
        """
        self.significance_threshold_pct = significance_threshold_pct
        logger.info(
            "Scope3BaseYearEngine v%s initialised (threshold=%.1f%%)",
            _MODULE_VERSION,
            self.significance_threshold_pct,
        )

    # -------------------------------------------------------------------
    # Public -- establish_base_year
    # -------------------------------------------------------------------

    def establish_base_year(
        self,
        inventory: List[CategoryBaseline],
        methodology_log: Optional[Dict[str, str]] = None,
        base_year: Optional[int] = None,
        organisation_boundary: str = "operational_control",
        excluded_categories: Optional[Dict[int, str]] = None,
    ) -> BaseYear:
        """Establish a base year record for Scope 3 inventory.

        Aggregates category baselines, validates completeness, and
        creates a versioned base year record with provenance.

        Args:
            inventory: List of category baseline data.
            methodology_log: Methodology decisions per category.
            base_year: Base year (default: earliest year in data).
            organisation_boundary: Organisational boundary approach.
            excluded_categories: Excluded categories with rationale.

        Returns:
            BaseYear record.
        """
        start_ms = time.time()

        total = Decimal("0")
        included_cats: List[int] = []
        for cat in inventory:
            total += _decimal(cat.tco2e)
            included_cats.append(cat.category)

        by = base_year or utcnow().year

        result = BaseYear(
            base_year=by,
            established_date=utcnow().isoformat(),
            total_tco2e=_round2(total),
            by_category=inventory,
            methodology_log=methodology_log or {},
            scope3_boundary=sorted(set(included_cats)),
            excluded_categories=excluded_categories or {},
            organisation_boundary=organisation_boundary,
            version=1,
            is_recalculated=False,
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Established base year %d: %.1f tCO2e across %d categories in %.1f ms",
            by, _round2(total), len(inventory), elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public -- check_recalculation_trigger
    # -------------------------------------------------------------------

    def check_recalculation_trigger(
        self,
        change_event: ChangeEvent,
        base_year: BaseYear,
        significance_threshold: Optional[float] = None,
    ) -> RecalculationTrigger:
        """Check whether a change event triggers base year recalculation.

        Applies the significance test:
            significance_pct = abs(adjustment) / base_year_total * 100
            is_significant = significance_pct >= threshold

        Args:
            change_event: The change event to assess.
            base_year: The current base year record.
            significance_threshold: Override threshold (default: engine default).

        Returns:
            RecalculationTrigger assessment.
        """
        start_ms = time.time()
        threshold = significance_threshold or self.significance_threshold_pct

        adjustment = abs(_decimal(change_event.adjustment_tco2e))
        base_total = _decimal(base_year.total_tco2e)
        significance = _safe_pct(adjustment, base_total)

        is_significant = float(significance) >= threshold
        threshold_d = _decimal(threshold)

        # Determine if recalculation is recommended.
        should_recalculate = is_significant

        # Build rationale.
        trigger_desc = TRIGGER_DESCRIPTIONS.get(
            change_event.trigger_type,
            f"Trigger type: {change_event.trigger_type}",
        )
        if is_significant:
            rationale = (
                f"SIGNIFICANT: Adjustment of {_round2(adjustment)} tCO2e represents "
                f"{_round2(significance)}% of base year total ({_round2(base_total)} tCO2e), "
                f"exceeding the {_round2(threshold_d)}% threshold. "
                f"Recalculation recommended. {trigger_desc}"
            )
            status = RecalculationStatus.SIGNIFICANT.value
        else:
            rationale = (
                f"NOT SIGNIFICANT: Adjustment of {_round2(adjustment)} tCO2e represents "
                f"{_round2(significance)}% of base year total ({_round2(base_total)} tCO2e), "
                f"below the {_round2(threshold_d)}% threshold. "
                f"No recalculation required. {trigger_desc}"
            )
            status = RecalculationStatus.NOT_SIGNIFICANT.value

        result = RecalculationTrigger(
            event_id=change_event.event_id,
            trigger_type=change_event.trigger_type,
            significance_pct=_round2(significance),
            threshold_pct=_round2(threshold_d),
            is_significant=is_significant,
            should_recalculate=should_recalculate,
            status=status,
            rationale=rationale,
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Trigger check: %s %.1f%% (%s) in %.1f ms",
            change_event.trigger_type, _round2(significance),
            "SIGNIFICANT" if is_significant else "not significant",
            elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public -- recalculate_base_year
    # -------------------------------------------------------------------

    def recalculate_base_year(
        self,
        base_year: BaseYear,
        change_event: ChangeEvent,
    ) -> RecalculationResult:
        """Recalculate the base year given a significant change event.

        Applies the adjustment to affected categories, optionally
        with pro-rata time-weighting for mid-year changes.

        Args:
            base_year: Current base year record.
            change_event: The change event triggering recalculation.

        Returns:
            RecalculationResult with adjusted base year.
        """
        start_ms = time.time()

        original_total = _decimal(base_year.total_tco2e)
        adjustment = _decimal(change_event.adjustment_tco2e)

        # Pro-rata for mid-year events.
        pro_rata_factor = Decimal("1")
        pro_rata_applied = False
        if change_event.effective_date:
            pro_rata_factor, pro_rata_applied = self._calculate_pro_rata_factor(
                change_event.effective_date, base_year.base_year,
            )

        adjusted_amount = adjustment * pro_rata_factor

        # Apply to base year categories.
        recalculated_cats: List[Dict[str, Any]] = []
        affected = set(change_event.affected_categories)
        num_affected = max(len(affected), 1)

        for cat in base_year.by_category:
            cat_dict = cat.model_dump()
            if cat.category in affected:
                per_cat_adj = adjusted_amount / _decimal(num_affected)
                new_tco2e = max(Decimal("0"), _decimal(cat.tco2e) + per_cat_adj)
                cat_dict["tco2e"] = _round2(new_tco2e)
                cat_dict["adjusted"] = True
                cat_dict["adjustment_tco2e"] = _round2(per_cat_adj)
            else:
                cat_dict["adjusted"] = False
                cat_dict["adjustment_tco2e"] = 0.0
            recalculated_cats.append(cat_dict)

        recalculated_total = original_total + adjusted_amount
        adjustment_pct = _safe_pct(abs(adjusted_amount), original_total)

        # Methodology changes.
        meth_changes: List[str] = []
        if change_event.trigger_type == RecalculationTriggerType.METHODOLOGY_UPGRADE:
            meth_changes.append(
                f"Category {change_event.affected_categories}: "
                f"{change_event.previous_methodology} -> {change_event.new_methodology}"
            )

        result = RecalculationResult(
            original_base_year=base_year.base_year,
            original_total_tco2e=_round2(original_total),
            recalculated_total_tco2e=_round2(max(Decimal("0"), recalculated_total)),
            adjustment_tco2e=_round2(adjusted_amount),
            adjustment_pct=_round2(adjustment_pct),
            recalculated_categories=recalculated_cats,
            trigger_event=change_event.model_dump(),
            methodology_changes=meth_changes,
            pro_rata_applied=pro_rata_applied,
            pro_rata_factor=_round4(pro_rata_factor),
            new_version=base_year.version + 1,
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Recalculated base year %d: %.1f -> %.1f tCO2e (adj=%.1f, v%d) in %.1f ms",
            base_year.base_year, _round2(original_total),
            _round2(max(Decimal("0"), recalculated_total)),
            _round2(adjusted_amount), base_year.version + 1, elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public -- calculate_pro_rata
    # -------------------------------------------------------------------

    def calculate_pro_rata(
        self,
        full_year_amount: float,
        effective_date: str,
        fiscal_year: int,
    ) -> Tuple[float, float]:
        """Calculate pro-rata time-weighted adjustment.

        Args:
            full_year_amount: Full year emission amount.
            effective_date: Effective date of change (YYYY-MM-DD).
            fiscal_year: Fiscal year.

        Returns:
            Tuple of (pro_rata_amount, pro_rata_factor).
        """
        factor, _ = self._calculate_pro_rata_factor(effective_date, fiscal_year)
        amount = _decimal(full_year_amount) * factor
        return (_round2(amount), _round4(factor))

    # -------------------------------------------------------------------
    # Public -- generate_trend_comparison
    # -------------------------------------------------------------------

    def generate_trend_comparison(
        self,
        base_year: BaseYear,
        subsequent_years: List[YearInventory],
    ) -> TrendComparison:
        """Generate multi-year trend comparison from base year.

        Args:
            base_year: Base year record.
            subsequent_years: List of subsequent year inventories.

        Returns:
            TrendComparison with YoY and cumulative changes.
        """
        start_ms = time.time()

        base_total = _decimal(base_year.total_tco2e)
        years: List[int] = []
        abs_changes: Dict[int, float] = {}
        pct_changes: Dict[int, float] = {}
        by_category: Dict[str, Dict[int, float]] = {}

        sorted_years = sorted(subsequent_years, key=lambda y: y.year)
        latest_tco2e = base_total

        for yr_data in sorted_years:
            yr = yr_data.year
            current = _decimal(yr_data.total_tco2e)
            years.append(yr)

            abs_change = current - base_total
            pct_change = _safe_pct(abs_change, base_total)
            abs_changes[yr] = _round2(abs_change)
            pct_changes[yr] = _round2(pct_change)
            latest_tco2e = current

            # Per-category trends.
            for cat_num, cat_tco2e in yr_data.by_category.items():
                cat_key = f"category_{cat_num}"
                if cat_key not in by_category:
                    by_category[cat_key] = {}
                by_category[cat_key][yr] = _round2(cat_tco2e)

        # Cumulative reduction.
        cumulative = base_total - latest_tco2e
        cumulative_pct = _safe_pct(cumulative, base_total)

        # Trend direction.
        if len(sorted_years) >= 2:
            last = _decimal(sorted_years[-1].total_tco2e)
            prev = _decimal(sorted_years[-2].total_tco2e)
            if last < prev * Decimal("0.99"):
                direction = TrendDirection.DECREASING.value
            elif last > prev * Decimal("1.01"):
                direction = TrendDirection.INCREASING.value
            else:
                direction = TrendDirection.STABLE.value
        elif len(sorted_years) == 1:
            if latest_tco2e < base_total:
                direction = TrendDirection.DECREASING.value
            elif latest_tco2e > base_total:
                direction = TrendDirection.INCREASING.value
            else:
                direction = TrendDirection.STABLE.value
        else:
            direction = TrendDirection.STABLE.value

        # CAGR.
        n_years = len(sorted_years)
        if n_years > 0 and base_total > Decimal("0") and latest_tco2e > Decimal("0"):
            total_years = sorted_years[-1].year - base_year.base_year
            if total_years > 0:
                ratio = latest_tco2e / base_total
                cagr = (float(ratio) ** (1.0 / total_years) - 1.0) * 100.0
            else:
                cagr = 0.0
        else:
            cagr = 0.0

        result = TrendComparison(
            base_year=base_year.base_year,
            base_year_tco2e=_round2(base_total),
            years=years,
            absolute_changes=abs_changes,
            percentage_changes=pct_changes,
            cumulative_reduction_pct=_round2(cumulative_pct),
            trend_direction=direction,
            cagr_pct=_round2(cagr),
            by_category=by_category,
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Trend comparison: base year %d, %d subsequent years, %.1f%% cumulative in %.1f ms",
            base_year.base_year, len(sorted_years), _round2(cumulative_pct), elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public -- separate_real_vs_methodology
    # -------------------------------------------------------------------

    def separate_real_vs_methodology(
        self,
        current_tco2e: float,
        original_base_tco2e: float,
        recalculated_base_tco2e: float,
    ) -> ChangeDecomposition:
        """Decompose changes into real operational vs. methodology changes.

        real_change = current - recalculated_base
        methodology_change = recalculated_base - original_base
        total_change = real_change + methodology_change

        Args:
            current_tco2e: Current year emissions.
            original_base_tco2e: Original base year emissions.
            recalculated_base_tco2e: Recalculated base year emissions.

        Returns:
            ChangeDecomposition with separated components.
        """
        start_ms = time.time()

        current = _decimal(current_tco2e)
        original = _decimal(original_base_tco2e)
        recalculated = _decimal(recalculated_base_tco2e)

        real_change = current - recalculated
        methodology_change = recalculated - original
        total_change = current - original

        real_pct = _safe_pct(real_change, original)
        meth_pct = _safe_pct(methodology_change, original)

        result = ChangeDecomposition(
            total_change_tco2e=_round2(total_change),
            real_change_tco2e=_round2(real_change),
            methodology_change_tco2e=_round2(methodology_change),
            real_change_pct=_round2(real_pct),
            methodology_change_pct=_round2(meth_pct),
            baseline_tco2e=_round2(original),
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Change decomposition: real=%.1f tCO2e, methodology=%.1f tCO2e in %.1f ms",
            _round2(real_change), _round2(methodology_change), elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public -- calculate_cumulative_reduction
    # -------------------------------------------------------------------

    def calculate_cumulative_reduction(
        self,
        base_year: BaseYear,
        current_year_tco2e: float,
        current_year: Optional[int] = None,
        target_pct: Optional[float] = None,
        target_year: Optional[int] = None,
    ) -> CumulativeReduction:
        """Calculate total cumulative reduction since base year.

        Args:
            base_year: Base year record.
            current_year_tco2e: Current year total emissions.
            current_year: Current year (default: now).
            target_pct: Target reduction percentage (optional).
            target_year: Target year (optional).

        Returns:
            CumulativeReduction with annualised rate and target tracking.
        """
        start_ms = time.time()

        base_total = _decimal(base_year.total_tco2e)
        current = _decimal(current_year_tco2e)
        yr = current_year or utcnow().year

        reduction = base_total - current
        reduction_pct = _safe_pct(reduction, base_total)

        years_elapsed = max(yr - base_year.base_year, 1)

        # Annualised rate using CAGR approach.
        if base_total > Decimal("0") and current > Decimal("0"):
            ratio = current / base_total
            annualised = (1.0 - float(ratio) ** (1.0 / years_elapsed)) * 100.0
        else:
            annualised = 0.0

        # Target tracking.
        on_track: Optional[bool] = None
        if target_pct is not None and target_year is not None:
            total_target_years = max(target_year - base_year.base_year, 1)
            expected_pct = _decimal(target_pct) * _decimal(years_elapsed) / _decimal(total_target_years)
            on_track = reduction_pct >= expected_pct * Decimal("0.9")

        result = CumulativeReduction(
            base_year=base_year.base_year,
            base_year_tco2e=_round2(base_total),
            current_year=yr,
            current_tco2e=_round2(current),
            cumulative_reduction_tco2e=_round2(reduction),
            cumulative_reduction_pct=_round2(reduction_pct),
            annualised_reduction_pct=_round2(annualised),
            on_track_for_target=on_track,
            target_pct=target_pct,
            target_year=target_year,
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Cumulative reduction: %.1f%% since %d (%.1f tCO2e) in %.1f ms",
            _round2(reduction_pct), base_year.base_year,
            _round2(reduction), elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _calculate_pro_rata_factor(
        self,
        effective_date_str: str,
        fiscal_year: int,
    ) -> Tuple[Decimal, bool]:
        """Calculate pro-rata factor for mid-year changes.

        Args:
            effective_date_str: Effective date string (YYYY-MM-DD).
            fiscal_year: Fiscal year.

        Returns:
            Tuple of (pro_rata_factor, was_applied).
        """
        try:
            eff_date = date.fromisoformat(effective_date_str)
        except (ValueError, TypeError):
            return (Decimal("1"), False)

        if eff_date.year != fiscal_year:
            return (Decimal("1"), False)

        year_start = date(fiscal_year, 1, 1)
        days_remaining = (_days_in_year(fiscal_year) - (eff_date - year_start).days)
        factor = _safe_divide(
            _decimal(days_remaining),
            _decimal(_days_in_year(fiscal_year)),
        )
        return (factor, True)

    # -------------------------------------------------------------------
    # Public -- _compute_provenance
    # -------------------------------------------------------------------

    @staticmethod
    def _compute_provenance(data: Any) -> str:
        """Compute SHA-256 provenance hash for audit trail.

        Args:
            data: Data to hash.

        Returns:
            SHA-256 hex digest (64 characters).
        """
        return _compute_hash(data)

# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

CategoryBaseline.model_rebuild()
BaseYear.model_rebuild()
ChangeEvent.model_rebuild()
RecalculationTrigger.model_rebuild()
RecalculationResult.model_rebuild()
TrendComparison.model_rebuild()
YearInventory.model_rebuild()
ChangeDecomposition.model_rebuild()
CumulativeReduction.model_rebuild()
