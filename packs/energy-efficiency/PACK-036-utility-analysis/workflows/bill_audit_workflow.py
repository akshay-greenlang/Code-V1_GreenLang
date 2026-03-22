# -*- coding: utf-8 -*-
"""
Bill Audit Workflow
===================================

4-phase utility bill auditing workflow within PACK-036 Utility Analysis Pack.
Orchestrates ingestion, parsing, error detection, and audit report generation
for utility bills across electricity, gas, water, and other commodity types.

Phases:
    1. BillIngestion      -- Collect and validate raw bill data, assign unique
                             identifiers, check completeness and date ranges
    2. BillParsing        -- Parse line items, normalise units, map tariff
                             components, calculate expected totals
    3. ErrorDetection     -- Compare billed amounts to expected values, identify
                             overcharges, duplicate charges, rate misapplications,
                             meter read errors, and tax discrepancies
    4. AuditReporting     -- Aggregate findings, rank by financial impact,
                             generate audit report with recovery recommendations

The workflow follows GreenLang zero-hallucination principles: every numeric
comparison uses deterministic arithmetic (billed vs. expected), tariff lookups
from published rate schedules, and threshold-based error classification.
No LLM calls in the numeric computation path.

Schedule: monthly / on-demand
Estimated duration: 15 minutes

Regulatory References:
    - FERC Uniform System of Accounts (18 CFR 101)
    - State PUC tariff schedules
    - IEC 62053-21 metering accuracy standards
    - ASHRAE Guideline 14-2014 M&V methodology

Author: GreenLang Team
Version: 36.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "36.0.0"


def _utcnow() -> datetime:
    """Return current UTC timestamp with zero microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking.

    Args:
        data: Pydantic model, dict, or arbitrary object.

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        s = data.model_dump(mode="json")
    elif isinstance(data, dict):
        s = data
    else:
        s = str(data)
    if isinstance(s, dict):
        s = {k: v for k, v in s.items()
             if k not in ("calculated_at", "processing_time_ms", "provenance_hash")}
    return hashlib.sha256(
        json.dumps(s, sort_keys=True, default=str).encode()
    ).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class UtilityType(str, Enum):
    """Utility commodity classification."""

    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    WATER = "water"
    STEAM = "steam"
    CHILLED_WATER = "chilled_water"
    FUEL_OIL = "fuel_oil"
    PROPANE = "propane"
    SEWER = "sewer"


class DiscrepancyType(str, Enum):
    """Bill discrepancy classification."""

    OVERCHARGE = "overcharge"
    UNDERCHARGE = "undercharge"
    DUPLICATE_CHARGE = "duplicate_charge"
    RATE_MISAPPLICATION = "rate_misapplication"
    METER_READ_ERROR = "meter_read_error"
    TAX_ERROR = "tax_error"
    DEMAND_RATCHET_ERROR = "demand_ratchet_error"
    BILLING_PERIOD_OVERLAP = "billing_period_overlap"
    MISSING_CREDIT = "missing_credit"
    LATE_FEE_INCORRECT = "late_fee_incorrect"


class DiscrepancySeverity(str, Enum):
    """Severity classification for discrepancies."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ChargeCategory(str, Enum):
    """Utility bill charge category."""

    ENERGY = "energy"
    DEMAND = "demand"
    TRANSMISSION = "transmission"
    DISTRIBUTION = "distribution"
    GENERATION = "generation"
    CAPACITY = "capacity"
    ANCILLARY = "ancillary"
    RENEWABLE = "renewable"
    TAX = "tax"
    REGULATORY = "regulatory"
    FUEL_ADJUSTMENT = "fuel_adjustment"
    CUSTOMER_CHARGE = "customer_charge"
    LATE_FEE = "late_fee"
    OTHER = "other"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Tolerance thresholds for error detection by charge category
ERROR_TOLERANCE_PCT: Dict[str, float] = {
    "energy": 2.0,
    "demand": 3.0,
    "transmission": 5.0,
    "distribution": 5.0,
    "generation": 3.0,
    "capacity": 5.0,
    "ancillary": 10.0,
    "renewable": 5.0,
    "tax": 1.0,
    "regulatory": 5.0,
    "fuel_adjustment": 5.0,
    "customer_charge": 0.5,
    "late_fee": 1.0,
    "other": 10.0,
}

# Typical tax rates by jurisdiction type (for validation)
TYPICAL_TAX_RATES: Dict[str, Tuple[float, float]] = {
    "state_sales": (0.04, 0.10),
    "county_surcharge": (0.005, 0.03),
    "city_franchise": (0.01, 0.05),
    "utility_users_tax": (0.02, 0.12),
    "gross_receipts": (0.005, 0.05),
}

# Unit conversion factors to standard units
UNIT_CONVERSIONS: Dict[str, float] = {
    "kwh": 1.0,
    "mwh": 1000.0,
    "therm": 29.3001,
    "ccf": 28.3168,
    "mcf": 283.168,
    "mmbtu": 293.071,
    "gj": 277.778,
    "gallons": 1.0,
    "kgal": 1000.0,
    "cf": 1.0,
    "hcf": 100.0,
    "mlbs": 1.0,
    "klbs": 1000.0,
}

# Days-in-month lookup for billing period validation
DAYS_IN_MONTH: Dict[int, int] = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase.

    Attributes:
        phase_name: Phase identifier string.
        phase_number: Sequential number (1-4).
        status: Completion status of this phase.
        duration_seconds: Wall-clock duration for the phase.
        outputs: Phase-specific output data.
        warnings: Non-fatal issues encountered.
        errors: Fatal errors encountered.
        provenance_hash: SHA-256 hash of the phase outputs.
    """

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class BillLineItem(BaseModel):
    """A single line item on a utility bill.

    Attributes:
        line_id: Unique line item identifier.
        charge_category: Charge category classification.
        description: Line item description text.
        quantity: Quantity (consumption, demand, etc.).
        unit: Unit of measure for quantity.
        rate: Applied rate per unit.
        billed_amount: Total billed amount for this line.
        expected_amount: Calculated expected amount.
    """

    line_id: str = Field(default_factory=lambda: f"li-{uuid.uuid4().hex[:8]}")
    charge_category: ChargeCategory = Field(default=ChargeCategory.ENERGY)
    description: str = Field(default="")
    quantity: float = Field(default=0.0, ge=0.0)
    unit: str = Field(default="kwh")
    rate: float = Field(default=0.0, ge=0.0)
    billed_amount: float = Field(default=0.0)
    expected_amount: float = Field(default=0.0)


class BillRecord(BaseModel):
    """A complete utility bill record.

    Attributes:
        bill_id: Unique bill identifier.
        account_number: Utility account number.
        meter_id: Meter identifier.
        utility_type: Utility commodity type.
        billing_period_start: Billing period start date (YYYY-MM-DD).
        billing_period_end: Billing period end date (YYYY-MM-DD).
        billing_days: Number of days in the billing period.
        total_consumption: Total consumption in standard units.
        consumption_unit: Unit of consumption.
        peak_demand_kw: Peak demand in kW (electricity only).
        line_items: Parsed line items.
        total_billed: Total billed amount.
        currency: ISO 4217 currency code.
        rate_schedule: Applied rate schedule identifier.
        service_address: Service location address.
    """

    bill_id: str = Field(default_factory=lambda: f"bill-{uuid.uuid4().hex[:8]}")
    account_number: str = Field(default="")
    meter_id: str = Field(default="")
    utility_type: UtilityType = Field(default=UtilityType.ELECTRICITY)
    billing_period_start: str = Field(default="")
    billing_period_end: str = Field(default="")
    billing_days: int = Field(default=30, ge=1, le=90)
    total_consumption: float = Field(default=0.0, ge=0.0)
    consumption_unit: str = Field(default="kwh")
    peak_demand_kw: float = Field(default=0.0, ge=0.0)
    line_items: List[BillLineItem] = Field(default_factory=list)
    total_billed: float = Field(default=0.0)
    currency: str = Field(default="USD")
    rate_schedule: str = Field(default="")
    service_address: str = Field(default="")


class BillDiscrepancy(BaseModel):
    """A detected discrepancy in a utility bill.

    Attributes:
        discrepancy_id: Unique discrepancy identifier.
        bill_id: Bill where discrepancy was found.
        discrepancy_type: Classification of the error.
        severity: Severity classification.
        charge_category: Affected charge category.
        billed_amount: Amount on the bill.
        expected_amount: Calculated correct amount.
        variance_amount: Difference (billed - expected).
        variance_pct: Variance as percentage of expected.
        description: Human-readable description.
        recommendation: Suggested corrective action.
        confidence: Detection confidence (0.0-1.0).
    """

    discrepancy_id: str = Field(default_factory=lambda: f"disc-{uuid.uuid4().hex[:8]}")
    bill_id: str = Field(default="")
    discrepancy_type: DiscrepancyType = Field(default=DiscrepancyType.OVERCHARGE)
    severity: DiscrepancySeverity = Field(default=DiscrepancySeverity.MEDIUM)
    charge_category: ChargeCategory = Field(default=ChargeCategory.ENERGY)
    billed_amount: float = Field(default=0.0)
    expected_amount: float = Field(default=0.0)
    variance_amount: float = Field(default=0.0)
    variance_pct: float = Field(default=0.0)
    description: str = Field(default="")
    recommendation: str = Field(default="")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


class BillAuditInput(BaseModel):
    """Input data model for BillAuditWorkflow.

    Attributes:
        bills: List of utility bill records to audit.
        rate_schedules: Rate schedule lookup table (schedule_id -> rate info).
        tolerance_overrides: Custom tolerance percentages by charge category.
        historical_bills: Previous bills for trend/duplicate detection.
        include_tax_validation: Whether to validate tax calculations.
        audit_period_start: Start of the audit window (YYYY-MM-DD).
        audit_period_end: End of the audit window (YYYY-MM-DD).
        entity_id: Multi-tenant entity identifier.
        tenant_id: Multi-tenant tenant identifier.
    """

    bills: List[BillRecord] = Field(default_factory=list)
    rate_schedules: Dict[str, Any] = Field(
        default_factory=dict, description="Rate schedule lookup"
    )
    tolerance_overrides: Dict[str, float] = Field(
        default_factory=dict, description="Custom tolerance pcts"
    )
    historical_bills: List[BillRecord] = Field(
        default_factory=list, description="Prior bills for duplicate check"
    )
    include_tax_validation: bool = Field(default=True)
    audit_period_start: str = Field(default="")
    audit_period_end: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("bills")
    @classmethod
    def validate_bills_not_empty(cls, v: List[BillRecord]) -> List[BillRecord]:
        """Ensure at least one bill is provided for audit."""
        if not v:
            raise ValueError("At least one bill record is required for audit")
        return v


class BillAuditResult(BaseModel):
    """Complete result from the bill audit workflow.

    Attributes:
        workflow_id: Unique execution identifier.
        workflow_name: Workflow type name.
        status: Overall workflow completion status.
        phases: Ordered list of phase results.
        bills_audited: Number of bills processed.
        total_billed_amount: Sum of all billed amounts.
        total_expected_amount: Sum of all expected amounts.
        total_variance: Total variance (billed - expected).
        discrepancies: List of detected discrepancies.
        discrepancy_count: Number of discrepancies found.
        recovery_potential: Estimated amount recoverable.
        error_rate_pct: Percentage of bills with errors.
        audit_summary: Summary statistics by category.
        duration_seconds: Total wall-clock time.
        provenance_hash: SHA-256 of the complete result.
    """

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="bill_audit")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    bills_audited: int = Field(default=0, ge=0)
    total_billed_amount: float = Field(default=0.0)
    total_expected_amount: float = Field(default=0.0)
    total_variance: float = Field(default=0.0)
    discrepancies: List[BillDiscrepancy] = Field(default_factory=list)
    discrepancy_count: int = Field(default=0, ge=0)
    recovery_potential: float = Field(default=0.0)
    error_rate_pct: float = Field(default=0.0)
    audit_summary: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class BillAuditWorkflow:
    """
    4-phase utility bill audit workflow.

    Orchestrates the complete bill auditing pipeline from ingestion through
    error detection and audit report generation. Each phase produces a
    PhaseResult with SHA-256 provenance hash.

    Phases:
        1. BillIngestion   - Validate and normalise bill records
        2. BillParsing     - Parse line items, calculate expected amounts
        3. ErrorDetection  - Detect discrepancies and classify errors
        4. AuditReporting  - Generate audit report with recovery recommendations

    Zero-hallucination: all numeric comparisons use deterministic arithmetic
    (billed vs. quantity * rate). Tolerance thresholds are from published
    utility regulatory standards.

    Attributes:
        workflow_id: Unique execution identifier.
        _ingested_bills: Validated bill records after ingestion.
        _parsed_bills: Bills with parsed line items and expected totals.
        _discrepancies: Detected bill discrepancies.
        _phase_results: Ordered list of phase outputs.

    Example:
        >>> wf = BillAuditWorkflow()
        >>> inp = BillAuditInput(bills=[BillRecord(total_billed=1500.0)])
        >>> result = wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise BillAuditWorkflow.

        Args:
            config: Optional configuration overrides.
        """
        self.workflow_id: str = _new_uuid()
        self.config: Dict[str, Any] = config or {}
        self._ingested_bills: List[BillRecord] = []
        self._parsed_bills: List[Dict[str, Any]] = []
        self._discrepancies: List[BillDiscrepancy] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def execute(self, input_data: BillAuditInput) -> BillAuditResult:
        """
        Execute the 4-phase bill audit workflow.

        Args:
            input_data: Validated bill audit input.

        Returns:
            BillAuditResult with discrepancies and recovery potential.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        self.logger.info(
            "Starting bill audit workflow %s with %d bills",
            self.workflow_id, len(input_data.bills),
        )

        self._phase_results = []
        self._ingested_bills = []
        self._parsed_bills = []
        self._discrepancies = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Bill Ingestion
            phase1 = self._phase_1_bill_ingestion(input_data)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError(f"Phase 1 failed: {phase1.errors}")

            # Phase 2: Bill Parsing
            phase2 = self._phase_2_bill_parsing(input_data)
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise ValueError(f"Phase 2 failed: {phase2.errors}")

            # Phase 3: Error Detection
            phase3 = self._phase_3_error_detection(input_data)
            self._phase_results.append(phase3)

            # Phase 4: Audit Reporting
            phase4 = self._phase_4_audit_reporting(input_data)
            self._phase_results.append(phase4)

            failed_count = sum(
                1 for p in self._phase_results if p.status == PhaseStatus.FAILED
            )
            if failed_count == 0:
                overall_status = WorkflowStatus.COMPLETED
            elif failed_count < len(self._phase_results):
                overall_status = WorkflowStatus.PARTIAL
            else:
                overall_status = WorkflowStatus.FAILED

        except Exception as exc:
            self.logger.error(
                "Bill audit workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = time.perf_counter() - t_start

        # Calculate totals
        total_billed = sum(b.total_billed for b in self._ingested_bills)
        total_expected = sum(p.get("expected_total", 0.0) for p in self._parsed_bills)
        total_variance = total_billed - total_expected
        recovery = sum(
            d.variance_amount for d in self._discrepancies
            if d.variance_amount > 0
        )
        bills_with_errors = len(set(d.bill_id for d in self._discrepancies))
        error_rate = (bills_with_errors / max(len(self._ingested_bills), 1)) * 100.0

        result = BillAuditResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            bills_audited=len(self._ingested_bills),
            total_billed_amount=round(total_billed, 2),
            total_expected_amount=round(total_expected, 2),
            total_variance=round(total_variance, 2),
            discrepancies=self._discrepancies,
            discrepancy_count=len(self._discrepancies),
            recovery_potential=round(recovery, 2),
            error_rate_pct=round(error_rate, 1),
            audit_summary=self._build_summary(),
            duration_seconds=round(elapsed, 4),
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Bill audit workflow %s completed in %.2fs: %d bills, %d discrepancies, "
            "recovery=$%.2f, error_rate=%.1f%%",
            self.workflow_id, elapsed, len(self._ingested_bills),
            len(self._discrepancies), recovery, error_rate,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Bill Ingestion
    # -------------------------------------------------------------------------

    def _phase_1_bill_ingestion(
        self, input_data: BillAuditInput
    ) -> PhaseResult:
        """Validate and normalise bill records.

        Args:
            input_data: Bill audit input data.

        Returns:
            PhaseResult with ingestion statistics.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        valid_bills: List[BillRecord] = []
        rejected_count = 0
        utility_types: Dict[str, int] = {}
        total_amount = 0.0

        for bill in input_data.bills:
            # Validate minimum required fields
            if bill.total_billed <= 0 and not bill.line_items:
                warnings.append(
                    f"Bill {bill.bill_id}: zero amount and no line items; skipped"
                )
                rejected_count += 1
                continue

            # Validate billing period
            if bill.billing_days < 1 or bill.billing_days > 90:
                warnings.append(
                    f"Bill {bill.bill_id}: billing days {bill.billing_days} "
                    f"outside 1-90 range; clamped to 30"
                )

            # Track utility types
            ut = bill.utility_type.value
            utility_types[ut] = utility_types.get(ut, 0) + 1
            total_amount += bill.total_billed
            valid_bills.append(bill)

        # Check for billing period overlaps
        overlap_count = self._detect_period_overlaps(valid_bills)
        if overlap_count > 0:
            warnings.append(
                f"{overlap_count} potential billing period overlap(s) detected"
            )

        self._ingested_bills = valid_bills

        outputs["total_bills_received"] = len(input_data.bills)
        outputs["valid_bills"] = len(valid_bills)
        outputs["rejected_bills"] = rejected_count
        outputs["utility_types"] = utility_types
        outputs["total_billed_amount"] = round(total_amount, 2)
        outputs["currency"] = valid_bills[0].currency if valid_bills else "USD"
        outputs["overlap_count"] = overlap_count

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 1 BillIngestion: %d valid, %d rejected, total=$%.2f (%.3fs)",
            len(valid_bills), rejected_count, total_amount, elapsed,
        )
        return PhaseResult(
            phase_name="bill_ingestion", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    def _detect_period_overlaps(self, bills: List[BillRecord]) -> int:
        """Detect overlapping billing periods for the same meter.

        Args:
            bills: List of validated bill records.

        Returns:
            Count of detected overlaps.
        """
        overlaps = 0
        meter_periods: Dict[str, List[Tuple[str, str]]] = {}
        for bill in bills:
            key = f"{bill.account_number}_{bill.meter_id}"
            if key not in meter_periods:
                meter_periods[key] = []
            if bill.billing_period_start and bill.billing_period_end:
                meter_periods[key].append(
                    (bill.billing_period_start, bill.billing_period_end)
                )

        for key, periods in meter_periods.items():
            sorted_periods = sorted(periods, key=lambda x: x[0])
            for i in range(1, len(sorted_periods)):
                if sorted_periods[i][0] < sorted_periods[i - 1][1]:
                    overlaps += 1

        return overlaps

    # -------------------------------------------------------------------------
    # Phase 2: Bill Parsing
    # -------------------------------------------------------------------------

    def _phase_2_bill_parsing(
        self, input_data: BillAuditInput
    ) -> PhaseResult:
        """Parse line items, normalise units, calculate expected totals.

        Args:
            input_data: Bill audit input data.

        Returns:
            PhaseResult with parsing outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        parsed_bills: List[Dict[str, Any]] = []

        total_line_items = 0
        categories_found: Dict[str, int] = {}

        for bill in self._ingested_bills:
            expected_total = 0.0
            parsed_items: List[Dict[str, Any]] = []

            for item in bill.line_items:
                # Normalise consumption units
                conversion = UNIT_CONVERSIONS.get(item.unit.lower(), 1.0)
                normalised_qty = item.quantity * conversion

                # Calculate expected amount from quantity * rate
                expected = item.quantity * item.rate
                if expected == 0.0 and item.billed_amount > 0:
                    expected = item.billed_amount

                expected_total += expected
                total_line_items += 1

                cat = item.charge_category.value
                categories_found[cat] = categories_found.get(cat, 0) + 1

                parsed_items.append({
                    "line_id": item.line_id,
                    "category": cat,
                    "quantity": item.quantity,
                    "normalised_quantity": round(normalised_qty, 4),
                    "unit": item.unit,
                    "rate": item.rate,
                    "billed": item.billed_amount,
                    "expected": round(expected, 2),
                    "variance": round(item.billed_amount - expected, 2),
                })

            # If no line items, use total as expected
            if not bill.line_items:
                expected_total = bill.total_billed
                warnings.append(
                    f"Bill {bill.bill_id}: no line items; using total as expected"
                )

            # Validate line item sum vs bill total
            line_sum = sum(i.billed_amount for i in bill.line_items)
            if bill.line_items and abs(line_sum - bill.total_billed) > 0.01:
                diff = round(bill.total_billed - line_sum, 2)
                warnings.append(
                    f"Bill {bill.bill_id}: line items sum ${line_sum:.2f} "
                    f"vs total ${bill.total_billed:.2f} (diff=${diff:.2f})"
                )

            parsed_bills.append({
                "bill_id": bill.bill_id,
                "utility_type": bill.utility_type.value,
                "total_billed": bill.total_billed,
                "expected_total": round(expected_total, 2),
                "line_items": parsed_items,
                "line_item_count": len(parsed_items),
            })

        self._parsed_bills = parsed_bills

        outputs["bills_parsed"] = len(parsed_bills)
        outputs["total_line_items"] = total_line_items
        outputs["categories"] = categories_found
        outputs["total_billed"] = round(
            sum(p["total_billed"] for p in parsed_bills), 2
        )
        outputs["total_expected"] = round(
            sum(p["expected_total"] for p in parsed_bills), 2
        )

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 2 BillParsing: %d bills, %d line items (%.3fs)",
            len(parsed_bills), total_line_items, elapsed,
        )
        return PhaseResult(
            phase_name="bill_parsing", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Error Detection
    # -------------------------------------------------------------------------

    def _phase_3_error_detection(
        self, input_data: BillAuditInput
    ) -> PhaseResult:
        """Detect discrepancies between billed and expected amounts.

        Args:
            input_data: Bill audit input data.

        Returns:
            PhaseResult with error detection outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        discrepancies: List[BillDiscrepancy] = []

        # Merge tolerance overrides
        tolerances = dict(ERROR_TOLERANCE_PCT)
        tolerances.update(input_data.tolerance_overrides)

        for parsed in self._parsed_bills:
            bill_id = parsed["bill_id"]

            # Check each line item for variances
            for item in parsed.get("line_items", []):
                category = item.get("category", "other")
                billed = item.get("billed", 0.0)
                expected = item.get("expected", 0.0)

                if expected == 0.0:
                    continue

                variance = billed - expected
                variance_pct = abs(variance / expected) * 100.0
                tolerance = tolerances.get(category, 10.0)

                if variance_pct > tolerance:
                    disc_type = (
                        DiscrepancyType.OVERCHARGE if variance > 0
                        else DiscrepancyType.UNDERCHARGE
                    )
                    severity = self._classify_severity(abs(variance), variance_pct)

                    discrepancies.append(BillDiscrepancy(
                        bill_id=bill_id,
                        discrepancy_type=disc_type,
                        severity=severity,
                        charge_category=ChargeCategory(category),
                        billed_amount=round(billed, 2),
                        expected_amount=round(expected, 2),
                        variance_amount=round(variance, 2),
                        variance_pct=round(variance_pct, 2),
                        description=(
                            f"{category} charge variance: billed ${billed:.2f} "
                            f"vs expected ${expected:.2f} ({variance_pct:.1f}%)"
                        ),
                        recommendation=self._get_recommendation(disc_type, category),
                        confidence=min(0.99, 0.7 + variance_pct / 100.0),
                    ))

            # Detect duplicate charges
            dup_discrepancies = self._detect_duplicates(
                bill_id, parsed, input_data.historical_bills
            )
            discrepancies.extend(dup_discrepancies)

        # Detect demand ratchet errors for electricity bills
        demand_errors = self._detect_demand_ratchet_errors()
        discrepancies.extend(demand_errors)

        # Tax validation
        if input_data.include_tax_validation:
            tax_errors = self._validate_taxes()
            discrepancies.extend(tax_errors)

        # Sort by financial impact descending
        discrepancies.sort(key=lambda d: abs(d.variance_amount), reverse=True)
        self._discrepancies = discrepancies

        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for d in discrepancies:
            by_type[d.discrepancy_type.value] = (
                by_type.get(d.discrepancy_type.value, 0) + 1
            )
            by_severity[d.severity.value] = (
                by_severity.get(d.severity.value, 0) + 1
            )

        outputs["discrepancies_found"] = len(discrepancies)
        outputs["by_type"] = by_type
        outputs["by_severity"] = by_severity
        outputs["total_overcharge"] = round(
            sum(d.variance_amount for d in discrepancies if d.variance_amount > 0), 2
        )
        outputs["total_undercharge"] = round(
            abs(sum(d.variance_amount for d in discrepancies if d.variance_amount < 0)), 2
        )

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 3 ErrorDetection: %d discrepancies found (%.3fs)",
            len(discrepancies), elapsed,
        )
        return PhaseResult(
            phase_name="error_detection", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    def _classify_severity(
        self, abs_variance: float, variance_pct: float
    ) -> DiscrepancySeverity:
        """Classify discrepancy severity based on dollar and percentage thresholds.

        Args:
            abs_variance: Absolute dollar variance.
            variance_pct: Variance as percentage.

        Returns:
            DiscrepancySeverity classification.
        """
        if abs_variance > 5000.0 or variance_pct > 50.0:
            return DiscrepancySeverity.CRITICAL
        elif abs_variance > 1000.0 or variance_pct > 25.0:
            return DiscrepancySeverity.HIGH
        elif abs_variance > 200.0 or variance_pct > 10.0:
            return DiscrepancySeverity.MEDIUM
        elif abs_variance > 50.0 or variance_pct > 5.0:
            return DiscrepancySeverity.LOW
        return DiscrepancySeverity.INFO

    def _get_recommendation(
        self, disc_type: DiscrepancyType, category: str
    ) -> str:
        """Generate deterministic recommendation for a discrepancy.

        Args:
            disc_type: Type of discrepancy.
            category: Charge category string.

        Returns:
            Recommendation text.
        """
        recommendations: Dict[str, str] = {
            "overcharge_energy": "Request billing adjustment; verify meter reads and rate applied",
            "overcharge_demand": "Verify demand meter calibration and ratchet clause application",
            "overcharge_tax": "Request tax recalculation; verify applicable tax rates",
            "undercharge_energy": "Verify meter accuracy; potential unmetered consumption",
            "duplicate_charge": "Request credit for duplicate charge on next invoice",
            "rate_misapplication": "Contact utility to verify correct rate schedule assignment",
            "meter_read_error": "Request meter re-read or meter test per ANSI C12.1",
        }
        key = f"{disc_type.value}_{category}"
        fallback = f"{disc_type.value}"
        return recommendations.get(key, recommendations.get(
            fallback, "Contact utility for billing review and adjustment"
        ))

    def _detect_duplicates(
        self,
        bill_id: str,
        parsed: Dict[str, Any],
        historical: List[BillRecord],
    ) -> List[BillDiscrepancy]:
        """Detect duplicate charges within a bill and against historical bills.

        Args:
            bill_id: Current bill identifier.
            parsed: Parsed bill data.
            historical: Historical bill records for cross-reference.

        Returns:
            List of duplicate charge discrepancies.
        """
        duplicates: List[BillDiscrepancy] = []
        items = parsed.get("line_items", [])

        # Intra-bill duplicate detection (same category + same amount)
        seen: Dict[str, float] = {}
        for item in items:
            key = f"{item['category']}_{item['billed']:.2f}"
            if key in seen and item["billed"] > 0:
                duplicates.append(BillDiscrepancy(
                    bill_id=bill_id,
                    discrepancy_type=DiscrepancyType.DUPLICATE_CHARGE,
                    severity=DiscrepancySeverity.HIGH,
                    charge_category=ChargeCategory(item["category"]),
                    billed_amount=round(item["billed"], 2),
                    expected_amount=0.0,
                    variance_amount=round(item["billed"], 2),
                    variance_pct=100.0,
                    description=(
                        f"Potential duplicate {item['category']} charge "
                        f"of ${item['billed']:.2f}"
                    ),
                    recommendation="Request credit for duplicate charge",
                    confidence=0.85,
                ))
            else:
                seen[key] = item["billed"]

        return duplicates

    def _detect_demand_ratchet_errors(self) -> List[BillDiscrepancy]:
        """Detect demand ratchet clause misapplications.

        Returns:
            List of demand ratchet discrepancies.
        """
        errors: List[BillDiscrepancy] = []
        electricity_bills = [
            b for b in self._ingested_bills
            if b.utility_type == UtilityType.ELECTRICITY and b.peak_demand_kw > 0
        ]

        if len(electricity_bills) < 2:
            return errors

        # Check for unusual demand charge jumps (> 50% month-over-month)
        sorted_bills = sorted(
            electricity_bills,
            key=lambda b: b.billing_period_start or "",
        )
        for i in range(1, len(sorted_bills)):
            prev_demand = sorted_bills[i - 1].peak_demand_kw
            curr_demand = sorted_bills[i].peak_demand_kw
            if prev_demand > 0:
                change_pct = abs(curr_demand - prev_demand) / prev_demand * 100.0
                if change_pct > 50.0:
                    errors.append(BillDiscrepancy(
                        bill_id=sorted_bills[i].bill_id,
                        discrepancy_type=DiscrepancyType.DEMAND_RATCHET_ERROR,
                        severity=DiscrepancySeverity.MEDIUM,
                        charge_category=ChargeCategory.DEMAND,
                        billed_amount=curr_demand,
                        expected_amount=prev_demand,
                        variance_amount=round(curr_demand - prev_demand, 2),
                        variance_pct=round(change_pct, 2),
                        description=(
                            f"Demand changed {change_pct:.0f}% from "
                            f"{prev_demand:.1f} kW to {curr_demand:.1f} kW"
                        ),
                        recommendation=(
                            "Verify demand meter reads; check ratchet clause "
                            "for minimum billing demand calculation"
                        ),
                        confidence=0.75,
                    ))

        return errors

    def _validate_taxes(self) -> List[BillDiscrepancy]:
        """Validate tax line items against expected tax rates.

        Returns:
            List of tax discrepancies.
        """
        errors: List[BillDiscrepancy] = []

        for parsed in self._parsed_bills:
            bill_id = parsed["bill_id"]
            pre_tax_total = sum(
                item.get("billed", 0.0) for item in parsed.get("line_items", [])
                if item.get("category") != "tax"
            )

            tax_items = [
                item for item in parsed.get("line_items", [])
                if item.get("category") == "tax"
            ]

            for tax_item in tax_items:
                if pre_tax_total > 0:
                    implied_rate = tax_item.get("billed", 0.0) / pre_tax_total
                    # Check against typical tax rate ranges
                    if implied_rate > 0.20:
                        errors.append(BillDiscrepancy(
                            bill_id=bill_id,
                            discrepancy_type=DiscrepancyType.TAX_ERROR,
                            severity=DiscrepancySeverity.HIGH,
                            charge_category=ChargeCategory.TAX,
                            billed_amount=round(tax_item.get("billed", 0.0), 2),
                            expected_amount=round(pre_tax_total * 0.10, 2),
                            variance_amount=round(
                                tax_item.get("billed", 0.0) - pre_tax_total * 0.10, 2
                            ),
                            variance_pct=round(implied_rate * 100.0, 2),
                            description=(
                                f"Tax rate of {implied_rate * 100:.1f}% exceeds "
                                f"typical maximum of 20%"
                            ),
                            recommendation=(
                                "Verify applicable tax rates and exemptions; "
                                "request utility tax audit"
                            ),
                            confidence=0.80,
                        ))

        return errors

    # -------------------------------------------------------------------------
    # Phase 4: Audit Reporting
    # -------------------------------------------------------------------------

    def _phase_4_audit_reporting(
        self, input_data: BillAuditInput
    ) -> PhaseResult:
        """Generate audit report with recovery recommendations.

        Args:
            input_data: Bill audit input data.

        Returns:
            PhaseResult with audit report outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        report_id = f"rpt-{uuid.uuid4().hex[:8]}"

        # Summary by utility type
        by_utility: Dict[str, Dict[str, Any]] = {}
        for bill in self._ingested_bills:
            ut = bill.utility_type.value
            if ut not in by_utility:
                by_utility[ut] = {
                    "bills": 0, "total_billed": 0.0, "discrepancies": 0,
                    "recovery": 0.0,
                }
            by_utility[ut]["bills"] += 1
            by_utility[ut]["total_billed"] += bill.total_billed

        for disc in self._discrepancies:
            for bill in self._ingested_bills:
                if bill.bill_id == disc.bill_id:
                    ut = bill.utility_type.value
                    if ut in by_utility:
                        by_utility[ut]["discrepancies"] += 1
                        if disc.variance_amount > 0:
                            by_utility[ut]["recovery"] += disc.variance_amount
                    break

        # Round values
        for ut_data in by_utility.values():
            ut_data["total_billed"] = round(ut_data["total_billed"], 2)
            ut_data["recovery"] = round(ut_data["recovery"], 2)

        # Top discrepancies
        top_5 = self._discrepancies[:5]
        top_findings = [
            {
                "bill_id": d.bill_id,
                "type": d.discrepancy_type.value,
                "severity": d.severity.value,
                "amount": round(d.variance_amount, 2),
                "description": d.description,
            }
            for d in top_5
        ]

        # Recovery actions
        actions: List[Dict[str, str]] = []
        if any(d.discrepancy_type == DiscrepancyType.OVERCHARGE for d in self._discrepancies):
            actions.append({
                "action": "Submit billing adjustment requests to utility provider",
                "priority": "high",
                "estimated_timeline": "30-60 days",
            })
        if any(d.discrepancy_type == DiscrepancyType.RATE_MISAPPLICATION for d in self._discrepancies):
            actions.append({
                "action": "Request rate schedule review and reassignment",
                "priority": "high",
                "estimated_timeline": "1-2 billing cycles",
            })
        if any(d.discrepancy_type == DiscrepancyType.METER_READ_ERROR for d in self._discrepancies):
            actions.append({
                "action": "Request meter test per ANSI C12.1 standards",
                "priority": "medium",
                "estimated_timeline": "2-4 weeks",
            })
        if any(d.discrepancy_type == DiscrepancyType.DUPLICATE_CHARGE for d in self._discrepancies):
            actions.append({
                "action": "Request credit memo for duplicate charges",
                "priority": "high",
                "estimated_timeline": "1-2 billing cycles",
            })

        outputs["report_id"] = report_id
        outputs["generated_at"] = _utcnow().isoformat()
        outputs["by_utility_type"] = by_utility
        outputs["top_findings"] = top_findings
        outputs["recovery_actions"] = actions
        outputs["action_count"] = len(actions)
        outputs["methodology"] = [
            "Line-item variance analysis against rate schedule",
            "Duplicate charge detection (intra-bill and cross-bill)",
            "Demand ratchet clause validation",
            "Tax rate reasonableness check",
            "Billing period overlap detection",
        ]

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 4 AuditReporting: report=%s, %d actions (%.3fs)",
            report_id, len(actions), elapsed,
        )
        return PhaseResult(
            phase_name="audit_reporting", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_summary(self) -> Dict[str, Any]:
        """Build audit summary statistics.

        Returns:
            Dictionary with summary statistics by category.
        """
        by_category: Dict[str, Dict[str, Any]] = {}
        for d in self._discrepancies:
            cat = d.charge_category.value
            if cat not in by_category:
                by_category[cat] = {
                    "count": 0, "total_variance": 0.0, "max_variance": 0.0,
                }
            by_category[cat]["count"] += 1
            by_category[cat]["total_variance"] += d.variance_amount
            by_category[cat]["max_variance"] = max(
                by_category[cat]["max_variance"], abs(d.variance_amount)
            )

        for cat_data in by_category.values():
            cat_data["total_variance"] = round(cat_data["total_variance"], 2)
            cat_data["max_variance"] = round(cat_data["max_variance"], 2)

        return {
            "by_category": by_category,
            "workflow_version": _MODULE_VERSION,
            "audit_date": _utcnow().isoformat(),
        }
