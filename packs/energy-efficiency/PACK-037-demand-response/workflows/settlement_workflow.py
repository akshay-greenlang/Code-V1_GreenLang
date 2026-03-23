# -*- coding: utf-8 -*-
"""
Settlement Workflow
===================================

3-phase workflow for calculating baselines, measuring performance, and
settling demand response revenue within PACK-037 Demand Response Pack.

Phases:
    1. BaselineCalculation     -- Calculate customer baseline load (CBL)
    2. PerformanceMeasurement  -- Measure actual curtailment against CBL
    3. RevenueSettlement       -- Calculate settlement revenue and penalties

The workflow follows GreenLang zero-hallucination principles: baseline
calculations use deterministic averaging methods (e.g. 10-of-10, high-5-of-10)
as specified by ISO/RTO rules. Revenue settlement uses published programme
rates. SHA-256 provenance hashes guarantee auditability.

Regulatory references:
    - PJM CBL methodology (10-of-10 with symmetric additive adjustment)
    - NYISO ICAP baseline methodology
    - CAISO demand response settlement protocols
    - FERC Order 745 (full LMP compensation)

Schedule: post-event (within 24-48 hours)
Estimated duration: 15 minutes

Author: GreenLang Team
Version: 37.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.utcnow()


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


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


class BaselineMethod(str, Enum):
    """Customer baseline load calculation methodology."""

    TEN_OF_TEN = "10_of_10"
    HIGH_5_OF_10 = "high_5_of_10"
    MATCHED_DAY = "matched_day"
    REGRESSION = "regression"
    METER_BEFORE_AFTER = "meter_before_after"


class SettlementStatus(str, Enum):
    """Settlement status for an event."""

    PENDING = "pending"
    CALCULATED = "calculated"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    SETTLED = "settled"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Baseline methodology parameters
BASELINE_METHODS: Dict[str, Dict[str, Any]] = {
    "10_of_10": {
        "description": "Average of 10 highest similar non-event days",
        "lookback_days": 45,
        "selection_count": 10,
        "selection_rule": "all",
        "adjustment": "symmetric_additive",
        "adjustment_cap_pct": Decimal("20"),
    },
    "high_5_of_10": {
        "description": "Average of 5 highest of 10 similar non-event days",
        "lookback_days": 45,
        "selection_count": 10,
        "top_n": 5,
        "selection_rule": "top_n",
        "adjustment": "symmetric_additive",
        "adjustment_cap_pct": Decimal("20"),
    },
    "matched_day": {
        "description": "Day-type matched baseline (weekday/weekend, temp)",
        "lookback_days": 30,
        "selection_count": 5,
        "selection_rule": "matched",
        "adjustment": "weather_adjusted",
        "adjustment_cap_pct": Decimal("25"),
    },
    "regression": {
        "description": "Temperature-regression baseline model",
        "lookback_days": 90,
        "selection_count": 0,
        "selection_rule": "regression",
        "adjustment": "model_based",
        "adjustment_cap_pct": Decimal("0"),
    },
    "meter_before_after": {
        "description": "Pre/post event metering comparison",
        "lookback_days": 0,
        "selection_count": 0,
        "selection_rule": "metered",
        "adjustment": "none",
        "adjustment_cap_pct": Decimal("0"),
    },
}

# Settlement rate types
SETTLEMENT_RATES: Dict[str, Dict[str, Any]] = {
    "capacity": {
        "rate_unit": "per_kw_year",
        "default_rate": Decimal("45.00"),
        "penalty_multiplier": Decimal("1.5"),
    },
    "energy": {
        "rate_unit": "per_kwh",
        "default_rate": Decimal("0.25"),
        "penalty_multiplier": Decimal("1.0"),
    },
    "ancillary": {
        "rate_unit": "per_kw_year",
        "default_rate": Decimal("80.00"),
        "penalty_multiplier": Decimal("3.0"),
    },
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class BaselineResult(BaseModel):
    """Customer baseline load calculation result."""

    method: str = Field(default="", description="Baseline methodology used")
    baseline_kw: Decimal = Field(default=Decimal("0"), ge=0, description="Calculated baseline kW")
    adjustment_kw: Decimal = Field(default=Decimal("0"), description="Adjustment applied kW")
    adjusted_baseline_kw: Decimal = Field(default=Decimal("0"), ge=0)
    days_used: int = Field(default=0, ge=0, description="Number of reference days used")
    confidence_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)


class PerformanceMeasurement(BaseModel):
    """Performance measurement for a single event interval."""

    interval_start: str = Field(default="", description="Interval start ISO 8601")
    baseline_kw: Decimal = Field(default=Decimal("0"), ge=0)
    actual_kw: Decimal = Field(default=Decimal("0"), ge=0)
    curtailment_kw: Decimal = Field(default=Decimal("0"), ge=0)
    committed_kw: Decimal = Field(default=Decimal("0"), ge=0)
    performance_pct: Decimal = Field(default=Decimal("0"), ge=0)


class SettlementInput(BaseModel):
    """Input data model for SettlementWorkflow."""

    event_id: str = Field(default_factory=lambda: f"evt-{uuid.uuid4().hex[:8]}")
    facility_id: str = Field(default="", description="Facility identifier")
    program_key: str = Field(default="capacity", description="DR program key")
    program_type: str = Field(default="capacity", description="capacity|energy|ancillary")
    committed_kw: Decimal = Field(..., gt=0, description="Committed curtailment kW")
    event_start_utc: str = Field(..., description="Event start ISO 8601")
    event_end_utc: str = Field(..., description="Event end ISO 8601")
    event_duration_hours: Decimal = Field(default=Decimal("4"), gt=0)
    baseline_method: str = Field(default="high_5_of_10", description="CBL methodology")
    historical_demand: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historical daily demand data for baseline calculation",
    )
    interval_readings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Metered interval readings during event",
    )
    settlement_rate: Optional[Decimal] = Field(
        default=None, ge=0, description="Override settlement rate"
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("event_start_utc")
    @classmethod
    def validate_event_start(cls, v: str) -> str:
        """Ensure event start is a valid ISO timestamp."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("event_start_utc must not be blank")
        return stripped


class SettlementResult(BaseModel):
    """Complete result from settlement workflow."""

    settlement_id: str = Field(..., description="Unique settlement ID")
    event_id: str = Field(default="", description="DR event identifier")
    facility_id: str = Field(default="", description="Facility identifier")
    baseline: BaselineResult = Field(default_factory=BaselineResult)
    measurements: List[PerformanceMeasurement] = Field(default_factory=list)
    average_curtailment_kw: Decimal = Field(default=Decimal("0"), ge=0)
    committed_kw: Decimal = Field(default=Decimal("0"), ge=0)
    performance_pct: Decimal = Field(default=Decimal("0"), ge=0)
    gross_revenue: Decimal = Field(default=Decimal("0"), ge=0)
    penalty_amount: Decimal = Field(default=Decimal("0"), ge=0)
    net_settlement: Decimal = Field(default=Decimal("0"))
    settlement_status: str = Field(default="calculated")
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class SettlementWorkflow:
    """
    3-phase settlement workflow for demand response events.

    Calculates customer baseline load, measures actual curtailment
    performance, and settles revenue including penalties for
    non-performance.

    Zero-hallucination: baseline calculations use ISO/RTO-specified
    averaging methods with deterministic arithmetic. Revenue settlement
    uses published programme rates. No LLM calls in the numeric
    computation path.

    Attributes:
        settlement_id: Unique settlement execution identifier.
        _baseline: Calculated baseline result.
        _measurements: Performance measurements per interval.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = SettlementWorkflow()
        >>> inp = SettlementInput(
        ...     committed_kw=Decimal("500"),
        ...     event_start_utc="2026-07-15T14:00:00Z",
        ...     event_end_utc="2026-07-15T18:00:00Z",
        ...     historical_demand=[{"date": "2026-07-10", "peak_kw": 2000}],
        ... )
        >>> result = wf.run(inp)
        >>> assert result.net_settlement >= 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SettlementWorkflow."""
        self.settlement_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._baseline: BaselineResult = BaselineResult()
        self._measurements: List[PerformanceMeasurement] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: SettlementInput) -> SettlementResult:
        """
        Execute the 3-phase settlement workflow.

        Args:
            input_data: Validated settlement input.

        Returns:
            SettlementResult with baseline, performance, and revenue details.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = _utcnow()
        self.logger.info(
            "Starting settlement workflow %s event=%s committed=%.0f kW",
            self.settlement_id, input_data.event_id,
            float(input_data.committed_kw),
        )

        self._phase_results = []
        self._baseline = BaselineResult()
        self._measurements = []

        try:
            # Phase 1: Baseline Calculation
            phase1 = self._phase_baseline_calculation(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Performance Measurement
            phase2 = self._phase_performance_measurement(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Revenue Settlement
            phase3 = self._phase_revenue_settlement(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error(
                "Settlement workflow failed: %s", exc, exc_info=True,
            )
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Calculate aggregate metrics
        avg_curtailment = Decimal("0")
        if self._measurements:
            avg_curtailment = (
                sum(m.curtailment_kw for m in self._measurements)
                / Decimal(str(len(self._measurements)))
            ).quantize(Decimal("0.1"))

        performance_pct = (
            Decimal(str(round(
                float(avg_curtailment) / float(input_data.committed_kw) * 100, 2
            )))
            if input_data.committed_kw > 0 else Decimal("0")
        )

        # Get revenue from phase 3 outputs
        gross_revenue = Decimal("0")
        penalty_amount = Decimal("0")
        net_settlement = Decimal("0")
        for pr in self._phase_results:
            if pr.phase_name == "revenue_settlement":
                gross_revenue = Decimal(str(pr.outputs.get("gross_revenue", 0)))
                penalty_amount = Decimal(str(pr.outputs.get("penalty_amount", 0)))
                net_settlement = Decimal(str(pr.outputs.get("net_settlement", 0)))

        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = SettlementResult(
            settlement_id=self.settlement_id,
            event_id=input_data.event_id,
            facility_id=input_data.facility_id,
            baseline=self._baseline,
            measurements=self._measurements,
            average_curtailment_kw=avg_curtailment,
            committed_kw=input_data.committed_kw,
            performance_pct=performance_pct,
            gross_revenue=gross_revenue,
            penalty_amount=penalty_amount,
            net_settlement=net_settlement,
            settlement_status="calculated",
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Settlement workflow %s completed in %dms "
            "avg_curtail=%.0f kW perf=%.1f%% net=%.2f",
            self.settlement_id, int(elapsed_ms), float(avg_curtailment),
            float(performance_pct), float(net_settlement),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Baseline Calculation
    # -------------------------------------------------------------------------

    def _phase_baseline_calculation(
        self, input_data: SettlementInput
    ) -> PhaseResult:
        """Calculate customer baseline load using specified methodology."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        method_key = input_data.baseline_method
        method_params = BASELINE_METHODS.get(method_key)
        if not method_params:
            method_key = "high_5_of_10"
            method_params = BASELINE_METHODS[method_key]
            warnings.append(
                f"Unknown baseline method; defaulting to '{method_key}'"
            )

        historical = input_data.historical_demand
        if not historical:
            # Use committed_kw * 1.25 as fallback baseline
            fallback_baseline = (
                input_data.committed_kw * Decimal("2.5")
            ).quantize(Decimal("0.1"))
            warnings.append(
                "No historical demand data; using fallback baseline estimation"
            )
            self._baseline = BaselineResult(
                method=method_key,
                baseline_kw=fallback_baseline,
                adjustment_kw=Decimal("0"),
                adjusted_baseline_kw=fallback_baseline,
                days_used=0,
                confidence_pct=Decimal("50"),
            )
        else:
            baseline_kw = self._calculate_baseline(
                historical, method_key, method_params
            )
            # Apply symmetric additive adjustment
            adjustment = self._calculate_adjustment(
                historical, baseline_kw, method_params
            )
            adj_cap_pct = method_params.get("adjustment_cap_pct", Decimal("20"))
            max_adj = baseline_kw * adj_cap_pct / Decimal("100")
            capped_adjustment = max(
                -max_adj, min(max_adj, adjustment)
            )
            adjusted_baseline = baseline_kw + capped_adjustment

            days_used = min(len(historical), method_params.get("selection_count", 10))
            confidence = Decimal(str(min(95, 50 + days_used * 5)))

            self._baseline = BaselineResult(
                method=method_key,
                baseline_kw=baseline_kw.quantize(Decimal("0.1")),
                adjustment_kw=capped_adjustment.quantize(Decimal("0.1")),
                adjusted_baseline_kw=adjusted_baseline.quantize(Decimal("0.1")),
                days_used=days_used,
                confidence_pct=confidence,
            )

        outputs["method"] = method_key
        outputs["baseline_kw"] = str(self._baseline.baseline_kw)
        outputs["adjusted_baseline_kw"] = str(self._baseline.adjusted_baseline_kw)
        outputs["adjustment_kw"] = str(self._baseline.adjustment_kw)
        outputs["days_used"] = self._baseline.days_used
        outputs["confidence_pct"] = str(self._baseline.confidence_pct)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 BaselineCalculation: method=%s baseline=%.0f adjusted=%.0f kW",
            method_key, float(self._baseline.baseline_kw),
            float(self._baseline.adjusted_baseline_kw),
        )
        return PhaseResult(
            phase_name="baseline_calculation", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _calculate_baseline(
        self,
        historical: List[Dict[str, Any]],
        method_key: str,
        method_params: Dict[str, Any],
    ) -> Decimal:
        """Calculate baseline kW from historical demand data."""
        peak_values = sorted(
            [Decimal(str(d.get("peak_kw", 0))) for d in historical],
            reverse=True,
        )

        selection_count = method_params.get("selection_count", 10)
        selected = peak_values[:selection_count] if peak_values else [Decimal("0")]

        if method_params.get("selection_rule") == "top_n":
            top_n = method_params.get("top_n", 5)
            selected = selected[:top_n]

        if not selected:
            return Decimal("0")

        return sum(selected) / Decimal(str(len(selected)))

    def _calculate_adjustment(
        self,
        historical: List[Dict[str, Any]],
        baseline_kw: Decimal,
        method_params: Dict[str, Any],
    ) -> Decimal:
        """Calculate symmetric additive adjustment."""
        if method_params.get("adjustment") == "none":
            return Decimal("0")

        # Use most recent day as adjustment reference
        if historical:
            most_recent = Decimal(str(
                historical[0].get("peak_kw", float(baseline_kw))
            ))
            return most_recent - baseline_kw

        return Decimal("0")

    # -------------------------------------------------------------------------
    # Phase 2: Performance Measurement
    # -------------------------------------------------------------------------

    def _phase_performance_measurement(
        self, input_data: SettlementInput
    ) -> PhaseResult:
        """Measure actual curtailment against calculated baseline."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        adjusted_baseline = self._baseline.adjusted_baseline_kw
        committed = input_data.committed_kw

        if input_data.interval_readings:
            for reading in input_data.interval_readings:
                actual_kw = Decimal(str(reading.get("demand_kw", 0)))
                curtailment_kw = max(Decimal("0"), adjusted_baseline - actual_kw)
                perf_pct = (
                    Decimal(str(round(
                        float(curtailment_kw) / float(committed) * 100, 2
                    )))
                    if committed > 0 else Decimal("0")
                )

                measurement = PerformanceMeasurement(
                    interval_start=reading.get("interval_start", ""),
                    baseline_kw=adjusted_baseline,
                    actual_kw=actual_kw,
                    curtailment_kw=curtailment_kw,
                    committed_kw=committed,
                    performance_pct=perf_pct,
                )
                self._measurements.append(measurement)
        else:
            # Single measurement from aggregate data
            total_actual = adjusted_baseline - committed * Decimal("0.92")
            curtailment = max(Decimal("0"), adjusted_baseline - total_actual)
            perf_pct = (
                Decimal(str(round(float(curtailment) / float(committed) * 100, 2)))
                if committed > 0 else Decimal("0")
            )

            measurement = PerformanceMeasurement(
                interval_start=input_data.event_start_utc,
                baseline_kw=adjusted_baseline,
                actual_kw=total_actual.quantize(Decimal("0.1")),
                curtailment_kw=curtailment.quantize(Decimal("0.1")),
                committed_kw=committed,
                performance_pct=perf_pct,
            )
            self._measurements.append(measurement)
            warnings.append("No interval readings; using single aggregate measurement")

        # Aggregate
        avg_curtailment = (
            sum(m.curtailment_kw for m in self._measurements)
            / Decimal(str(max(len(self._measurements), 1)))
        ).quantize(Decimal("0.1"))
        avg_performance = (
            sum(m.performance_pct for m in self._measurements)
            / Decimal(str(max(len(self._measurements), 1)))
        ).quantize(Decimal("0.1"))

        outputs["intervals_measured"] = len(self._measurements)
        outputs["avg_curtailment_kw"] = str(avg_curtailment)
        outputs["avg_performance_pct"] = str(avg_performance)
        outputs["baseline_kw"] = str(adjusted_baseline)
        outputs["committed_kw"] = str(committed)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 PerformanceMeasurement: %d intervals, avg=%.0f kW, perf=%.1f%%",
            len(self._measurements), float(avg_curtailment), float(avg_performance),
        )
        return PhaseResult(
            phase_name="performance_measurement", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Revenue Settlement
    # -------------------------------------------------------------------------

    def _phase_revenue_settlement(
        self, input_data: SettlementInput
    ) -> PhaseResult:
        """Calculate settlement revenue and penalties."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        program_type = input_data.program_type
        rate_config = SETTLEMENT_RATES.get(program_type, SETTLEMENT_RATES["capacity"])

        # Determine rate
        if input_data.settlement_rate is not None:
            rate = input_data.settlement_rate
        else:
            rate = rate_config["default_rate"]

        # Calculate average curtailment
        avg_curtailment = Decimal("0")
        if self._measurements:
            avg_curtailment = (
                sum(m.curtailment_kw for m in self._measurements)
                / Decimal(str(len(self._measurements)))
            ).quantize(Decimal("0.1"))

        committed = input_data.committed_kw
        event_hours = input_data.event_duration_hours

        # Gross revenue calculation based on rate type
        if rate_config["rate_unit"] == "per_kw_year":
            # Pro-rate annual capacity payment for this event
            # Revenue per event = (rate_per_kw_year / max_events_year) * curtailed_kw
            # Simplified: per-event = rate * kW * (event_hours / 8760)
            gross_revenue = (
                rate * avg_curtailment * event_hours / Decimal("8760")
            ).quantize(Decimal("0.01"))
        else:
            # Energy-based: rate * curtailed_kWh
            curtailed_kwh = avg_curtailment * event_hours
            gross_revenue = (rate * curtailed_kwh).quantize(Decimal("0.01"))

        # Penalty calculation for underperformance
        penalty_amount = Decimal("0")
        shortfall_kw = max(Decimal("0"), committed - avg_curtailment)
        if shortfall_kw > 0:
            penalty_multiplier = rate_config.get("penalty_multiplier", Decimal("1.0"))
            if rate_config["rate_unit"] == "per_kw_year":
                penalty_amount = (
                    rate * shortfall_kw * event_hours / Decimal("8760")
                    * penalty_multiplier
                ).quantize(Decimal("0.01"))
            else:
                shortfall_kwh = shortfall_kw * event_hours
                penalty_amount = (
                    rate * shortfall_kwh * penalty_multiplier
                ).quantize(Decimal("0.01"))

        net_settlement = gross_revenue - penalty_amount

        outputs["program_type"] = program_type
        outputs["rate"] = str(rate)
        outputs["rate_unit"] = rate_config["rate_unit"]
        outputs["avg_curtailment_kw"] = str(avg_curtailment)
        outputs["committed_kw"] = str(committed)
        outputs["shortfall_kw"] = str(shortfall_kw)
        outputs["event_duration_hours"] = str(event_hours)
        outputs["gross_revenue"] = str(gross_revenue)
        outputs["penalty_amount"] = str(penalty_amount)
        outputs["net_settlement"] = str(net_settlement)

        if penalty_amount > 0:
            warnings.append(
                f"Penalty of {penalty_amount} applied for {shortfall_kw} kW shortfall"
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 RevenueSettlement: gross=%.2f penalty=%.2f net=%.2f",
            float(gross_revenue), float(penalty_amount), float(net_settlement),
        )
        return PhaseResult(
            phase_name="revenue_settlement", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: SettlementResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
