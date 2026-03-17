# -*- coding: utf-8 -*-
"""
Performance Testing Workflow
==================================

4-phase workflow for battery performance and durability testing per EU
Battery Regulation 2023/1542, Articles 10-11 and Annex IV. Implements
test data collection, metric calculation, threshold validation against
regulatory minimums, and report generation.

Phases:
    1. TestDataCollection     -- Gather performance test results
    2. MetricCalculation      -- Compute performance and durability metrics
    3. ThresholdValidation    -- Validate against regulatory thresholds
    4. ReportGeneration       -- Generate performance compliance report

Regulatory references:
    - EU Regulation 2023/1542 Art. 10 (performance and durability requirements)
    - EU Regulation 2023/1542 Art. 11 (removability and replaceability)
    - EU Regulation 2023/1542 Annex IV (performance and durability parameters)
    - IEC 62660-1 (secondary lithium-ion cells performance testing)
    - IEC 62620 (secondary lithium cells for industrial applications)

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class WorkflowPhase(str, Enum):
    """Phases of the performance testing workflow."""
    TEST_DATA_COLLECTION = "test_data_collection"
    METRIC_CALCULATION = "metric_calculation"
    THRESHOLD_VALIDATION = "threshold_validation"
    REPORT_GENERATION = "report_generation"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TestType(str, Enum):
    """Battery performance test types per Annex IV."""
    RATED_CAPACITY = "rated_capacity"
    CAPACITY_RETENTION = "capacity_retention"
    ROUND_TRIP_EFFICIENCY = "round_trip_efficiency"
    CYCLE_LIFE = "cycle_life"
    CALENDAR_LIFE = "calendar_life"
    INTERNAL_RESISTANCE = "internal_resistance"
    POWER_CAPABILITY = "power_capability"
    SELF_DISCHARGE = "self_discharge"
    TEMPERATURE_RANGE = "temperature_range"
    STATE_OF_HEALTH = "state_of_health"


class TestStandard(str, Enum):
    """Testing standard applied."""
    IEC_62660_1 = "IEC_62660_1"
    IEC_62620 = "IEC_62620"
    IEC_61960 = "IEC_61960"
    UN_38_3 = "UN_38_3"
    ISO_12405 = "ISO_12405"
    MANUFACTURER = "manufacturer_standard"


class ComplianceLevel(str, Enum):
    """Compliance assessment result."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    MARGINAL = "marginal"
    NOT_TESTED = "not_tested"


# =============================================================================
# REGULATORY THRESHOLDS (Annex IV minimum values)
# =============================================================================


EV_BATTERY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "rated_capacity_ah": {"min": 0.0, "tolerance_pct": 5.0},
    "capacity_retention_500_cycles_pct": {"min": 80.0, "tolerance_pct": 0.0},
    "capacity_retention_1000_cycles_pct": {"min": 70.0, "tolerance_pct": 0.0},
    "round_trip_efficiency_pct": {"min": 90.0, "tolerance_pct": 0.0},
    "cycle_life_80pct_soh": {"min": 1000, "tolerance_pct": 0.0},
    "internal_resistance_increase_pct": {"max": 50.0, "tolerance_pct": 0.0},
    "power_capability_w_per_kg": {"min": 0.0, "tolerance_pct": 0.0},
    "self_discharge_pct_per_month": {"max": 5.0, "tolerance_pct": 0.0},
}

INDUSTRIAL_BATTERY_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "rated_capacity_ah": {"min": 0.0, "tolerance_pct": 5.0},
    "capacity_retention_500_cycles_pct": {"min": 75.0, "tolerance_pct": 0.0},
    "capacity_retention_1000_cycles_pct": {"min": 65.0, "tolerance_pct": 0.0},
    "round_trip_efficiency_pct": {"min": 85.0, "tolerance_pct": 0.0},
    "cycle_life_80pct_soh": {"min": 500, "tolerance_pct": 0.0},
    "internal_resistance_increase_pct": {"max": 60.0, "tolerance_pct": 0.0},
    "self_discharge_pct_per_month": {"max": 8.0, "tolerance_pct": 0.0},
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class TestRecord(BaseModel):
    """Individual test result record."""
    test_id: str = Field(default_factory=lambda: f"tst-{_new_uuid()[:8]}")
    test_type: TestType = Field(..., description="Test type category")
    test_standard: TestStandard = Field(default=TestStandard.IEC_62660_1)
    parameter_name: str = Field(default="", description="Measured parameter")
    measured_value: float = Field(default=0.0, description="Measured value")
    unit: str = Field(default="", description="Measurement unit")
    test_temperature_c: float = Field(default=25.0, description="Test temperature")
    test_date: str = Field(default="", description="ISO date of test")
    lab_name: str = Field(default="", description="Testing laboratory")
    accreditation: str = Field(default="", description="Lab accreditation")
    sample_id: str = Field(default="", description="Test sample identifier")
    cycle_count: int = Field(default=0, ge=0, description="Cycle count at test")
    notes: str = Field(default="")


class MetricResult(BaseModel):
    """Computed performance metric."""
    metric_name: str = Field(..., description="Metric identifier")
    value: float = Field(default=0.0, description="Calculated value")
    unit: str = Field(default="", description="Unit of measurement")
    threshold_min: Optional[float] = Field(default=None)
    threshold_max: Optional[float] = Field(default=None)
    compliance: ComplianceLevel = Field(default=ComplianceLevel.NOT_TESTED)
    margin_pct: float = Field(default=0.0, description="Margin above/below threshold")
    source_test_ids: List[str] = Field(default_factory=list)


class PerformanceTestingInput(BaseModel):
    """Input data model for PerformanceTestingWorkflow."""
    battery_id: str = Field(default_factory=lambda: f"bat-{_new_uuid()[:8]}")
    battery_model: str = Field(default="", description="Battery model identifier")
    battery_category: str = Field(default="ev_battery")
    rated_capacity_ah: float = Field(default=0.0, ge=0.0, description="Rated capacity")
    nominal_voltage_v: float = Field(default=0.0, ge=0.0)
    energy_capacity_kwh: float = Field(default=0.0, ge=0.0)
    test_records: List[TestRecord] = Field(default_factory=list)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class PerformanceTestingResult(BaseModel):
    """Complete result from performance testing workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="performance_testing")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    battery_id: str = Field(default="")
    metrics: List[MetricResult] = Field(default_factory=list)
    metrics_compliant: int = Field(default=0, ge=0)
    metrics_non_compliant: int = Field(default=0, ge=0)
    metrics_marginal: int = Field(default=0, ge=0)
    overall_compliance: str = Field(default="not_tested")
    test_record_count: int = Field(default=0, ge=0)
    test_types_covered: List[str] = Field(default_factory=list)
    reporting_year: int = Field(default=2025)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class PerformanceTestingWorkflow:
    """
    4-phase battery performance testing workflow per EU Battery Regulation.

    Implements end-to-end performance testing evaluation following
    EU Regulation 2023/1542 Art. 10-11 and Annex IV. Collects test data,
    computes performance metrics, validates against regulatory thresholds,
    and generates compliance reports.

    Zero-hallucination: all metric calculations use deterministic arithmetic
    with documented threshold values. No LLM in validation paths.

    Example:
        >>> wf = PerformanceTestingWorkflow()
        >>> inp = PerformanceTestingInput(test_records=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.overall_compliance in ("compliant", "non_compliant", "marginal")
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize PerformanceTestingWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._tests: List[TestRecord] = []
        self._metrics: List[MetricResult] = []
        self._overall_compliance: str = ComplianceLevel.NOT_TESTED.value
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.TEST_DATA_COLLECTION.value, "description": "Gather performance test results"},
            {"name": WorkflowPhase.METRIC_CALCULATION.value, "description": "Compute performance and durability metrics"},
            {"name": WorkflowPhase.THRESHOLD_VALIDATION.value, "description": "Validate against regulatory thresholds"},
            {"name": WorkflowPhase.REPORT_GENERATION.value, "description": "Generate performance compliance report"},
        ]

    def validate_inputs(self, input_data: PerformanceTestingInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.test_records:
            issues.append("No test records provided")
        if input_data.rated_capacity_ah <= 0:
            issues.append("Rated capacity must be greater than zero")
        return issues

    async def execute(
        self,
        input_data: Optional[PerformanceTestingInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> PerformanceTestingResult:
        """
        Execute the 4-phase performance testing workflow.

        Args:
            input_data: Full input model (preferred).
            config: Configuration overrides.

        Returns:
            PerformanceTestingResult with metrics, compliance status, and report.
        """
        if input_data is None:
            input_data = PerformanceTestingInput(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting performance testing workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS
        phases_done = 0

        try:
            phase_results.append(await self._phase_test_data_collection(input_data))
            phases_done += 1
            phase_results.append(await self._phase_metric_calculation(input_data))
            phases_done += 1
            phase_results.append(await self._phase_threshold_validation(input_data))
            phases_done += 1
            phase_results.append(await self._phase_report_generation(input_data))
            phases_done += 1
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Performance testing workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        compliant = sum(1 for m in self._metrics if m.compliance == ComplianceLevel.COMPLIANT)
        non_compliant = sum(1 for m in self._metrics if m.compliance == ComplianceLevel.NON_COMPLIANT)
        marginal = sum(1 for m in self._metrics if m.compliance == ComplianceLevel.MARGINAL)
        types_covered = sorted(set(t.test_type.value for t in self._tests))

        result = PerformanceTestingResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=phases_done,
            total_duration_seconds=elapsed,
            duration_ms=round(elapsed * 1000, 2),
            battery_id=input_data.battery_id,
            metrics=self._metrics,
            metrics_compliant=compliant,
            metrics_non_compliant=non_compliant,
            metrics_marginal=marginal,
            overall_compliance=self._overall_compliance,
            test_record_count=len(self._tests),
            test_types_covered=types_covered,
            reporting_year=input_data.reporting_year,
            executed_at=_utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Performance testing %s completed in %.2fs: %d compliant, %d non-compliant",
            self.workflow_id, elapsed, compliant, non_compliant,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Test Data Collection
    # -------------------------------------------------------------------------

    async def _phase_test_data_collection(
        self, input_data: PerformanceTestingInput,
    ) -> PhaseResult:
        """Gather and organize performance test results."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._tests = list(input_data.test_records)

        type_counts: Dict[str, int] = {}
        standard_counts: Dict[str, int] = {}
        lab_counts: Dict[str, int] = {}
        for test in self._tests:
            tt = test.test_type.value
            type_counts[tt] = type_counts.get(tt, 0) + 1
            ts = test.test_standard.value
            standard_counts[ts] = standard_counts.get(ts, 0) + 1
            if test.lab_name:
                lab_counts[test.lab_name] = lab_counts.get(test.lab_name, 0) + 1

        outputs["tests_collected"] = len(self._tests)
        outputs["test_type_distribution"] = type_counts
        outputs["standard_distribution"] = standard_counts
        outputs["lab_distribution"] = lab_counts
        outputs["unique_test_types"] = len(type_counts)

        # Check for missing critical test types
        required_types = {
            TestType.RATED_CAPACITY.value,
            TestType.CAPACITY_RETENTION.value,
            TestType.ROUND_TRIP_EFFICIENCY.value,
            TestType.CYCLE_LIFE.value,
        }
        present_types = set(type_counts.keys())
        missing = required_types - present_types
        if missing:
            warnings.append(
                f"Missing required test types: {', '.join(sorted(missing))}"
            )

        if not self._tests:
            warnings.append("No test records provided; all metrics will be not_tested")

        # Check for unaccredited labs
        unaccredited = [
            t for t in self._tests if not t.accreditation
        ]
        if unaccredited:
            warnings.append(
                f"{len(unaccredited)} tests from labs without stated accreditation"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 TestDataCollection: %d tests, %d types",
            len(self._tests), len(type_counts),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.TEST_DATA_COLLECTION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Metric Calculation
    # -------------------------------------------------------------------------

    async def _phase_metric_calculation(
        self, input_data: PerformanceTestingInput,
    ) -> PhaseResult:
        """Compute performance and durability metrics from test data."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._metrics = []

        # Group tests by type
        test_groups: Dict[str, List[TestRecord]] = {}
        for test in self._tests:
            test_groups.setdefault(test.test_type.value, []).append(test)

        # Rated capacity metric
        if TestType.RATED_CAPACITY.value in test_groups:
            records = test_groups[TestType.RATED_CAPACITY.value]
            avg_val = sum(t.measured_value for t in records) / len(records)
            self._metrics.append(MetricResult(
                metric_name="rated_capacity_ah",
                value=round(avg_val, 2),
                unit="Ah",
                source_test_ids=[t.test_id for t in records],
            ))

        # Capacity retention at 500 cycles
        cap_ret_records = test_groups.get(TestType.CAPACITY_RETENTION.value, [])
        records_500 = [t for t in cap_ret_records if t.cycle_count == 500]
        if records_500:
            avg_ret = sum(t.measured_value for t in records_500) / len(records_500)
            self._metrics.append(MetricResult(
                metric_name="capacity_retention_500_cycles_pct",
                value=round(avg_ret, 2),
                unit="%",
                source_test_ids=[t.test_id for t in records_500],
            ))

        # Capacity retention at 1000 cycles
        records_1000 = [t for t in cap_ret_records if t.cycle_count == 1000]
        if records_1000:
            avg_ret = sum(t.measured_value for t in records_1000) / len(records_1000)
            self._metrics.append(MetricResult(
                metric_name="capacity_retention_1000_cycles_pct",
                value=round(avg_ret, 2),
                unit="%",
                source_test_ids=[t.test_id for t in records_1000],
            ))

        # Round-trip efficiency
        if TestType.ROUND_TRIP_EFFICIENCY.value in test_groups:
            records = test_groups[TestType.ROUND_TRIP_EFFICIENCY.value]
            avg_eff = sum(t.measured_value for t in records) / len(records)
            self._metrics.append(MetricResult(
                metric_name="round_trip_efficiency_pct",
                value=round(avg_eff, 2),
                unit="%",
                source_test_ids=[t.test_id for t in records],
            ))

        # Cycle life (cycles to 80% SoH)
        if TestType.CYCLE_LIFE.value in test_groups:
            records = test_groups[TestType.CYCLE_LIFE.value]
            max_cycles = max(t.measured_value for t in records)
            self._metrics.append(MetricResult(
                metric_name="cycle_life_80pct_soh",
                value=round(max_cycles, 0),
                unit="cycles",
                source_test_ids=[t.test_id for t in records],
            ))

        # Internal resistance increase
        if TestType.INTERNAL_RESISTANCE.value in test_groups:
            records = test_groups[TestType.INTERNAL_RESISTANCE.value]
            avg_val = sum(t.measured_value for t in records) / len(records)
            self._metrics.append(MetricResult(
                metric_name="internal_resistance_increase_pct",
                value=round(avg_val, 2),
                unit="%",
                source_test_ids=[t.test_id for t in records],
            ))

        # Power capability
        if TestType.POWER_CAPABILITY.value in test_groups:
            records = test_groups[TestType.POWER_CAPABILITY.value]
            avg_val = sum(t.measured_value for t in records) / len(records)
            self._metrics.append(MetricResult(
                metric_name="power_capability_w_per_kg",
                value=round(avg_val, 2),
                unit="W/kg",
                source_test_ids=[t.test_id for t in records],
            ))

        # Self-discharge rate
        if TestType.SELF_DISCHARGE.value in test_groups:
            records = test_groups[TestType.SELF_DISCHARGE.value]
            avg_val = sum(t.measured_value for t in records) / len(records)
            self._metrics.append(MetricResult(
                metric_name="self_discharge_pct_per_month",
                value=round(avg_val, 2),
                unit="%/month",
                source_test_ids=[t.test_id for t in records],
            ))

        outputs["metrics_calculated"] = len(self._metrics)
        outputs["metric_summary"] = {
            m.metric_name: m.value for m in self._metrics
        }

        if not self._metrics:
            warnings.append("No metrics could be calculated from the test data")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 MetricCalculation: %d metrics calculated",
            len(self._metrics),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.METRIC_CALCULATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Threshold Validation
    # -------------------------------------------------------------------------

    async def _phase_threshold_validation(
        self, input_data: PerformanceTestingInput,
    ) -> PhaseResult:
        """Validate computed metrics against regulatory thresholds."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        is_ev = input_data.battery_category in ("ev_battery", "lmt_battery")
        thresholds = EV_BATTERY_THRESHOLDS if is_ev else INDUSTRIAL_BATTERY_THRESHOLDS

        for metric in self._metrics:
            threshold_spec = thresholds.get(metric.metric_name)
            if threshold_spec is None:
                metric.compliance = ComplianceLevel.NOT_TESTED
                continue

            if "min" in threshold_spec:
                min_val = threshold_spec["min"]
                metric.threshold_min = min_val
                if min_val > 0:
                    margin = ((metric.value - min_val) / min_val) * 100
                    metric.margin_pct = round(margin, 2)
                    if metric.value >= min_val:
                        metric.compliance = ComplianceLevel.COMPLIANT
                    elif metric.value >= min_val * 0.95:
                        metric.compliance = ComplianceLevel.MARGINAL
                        warnings.append(
                            f"{metric.metric_name}: {metric.value} is marginally "
                            f"below threshold {min_val} ({metric.unit})"
                        )
                    else:
                        metric.compliance = ComplianceLevel.NON_COMPLIANT
                        warnings.append(
                            f"{metric.metric_name}: {metric.value} fails threshold "
                            f"{min_val} ({metric.unit})"
                        )
                else:
                    metric.compliance = ComplianceLevel.COMPLIANT

            if "max" in threshold_spec:
                max_val = threshold_spec["max"]
                metric.threshold_max = max_val
                if max_val > 0:
                    margin = ((max_val - metric.value) / max_val) * 100
                    metric.margin_pct = round(margin, 2)
                    if metric.value <= max_val:
                        metric.compliance = ComplianceLevel.COMPLIANT
                    elif metric.value <= max_val * 1.05:
                        metric.compliance = ComplianceLevel.MARGINAL
                        warnings.append(
                            f"{metric.metric_name}: {metric.value} is marginally "
                            f"above max threshold {max_val} ({metric.unit})"
                        )
                    else:
                        metric.compliance = ComplianceLevel.NON_COMPLIANT
                        warnings.append(
                            f"{metric.metric_name}: {metric.value} exceeds max "
                            f"threshold {max_val} ({metric.unit})"
                        )

        # Determine overall compliance
        compliant_count = sum(
            1 for m in self._metrics if m.compliance == ComplianceLevel.COMPLIANT
        )
        non_compliant_count = sum(
            1 for m in self._metrics if m.compliance == ComplianceLevel.NON_COMPLIANT
        )
        marginal_count = sum(
            1 for m in self._metrics if m.compliance == ComplianceLevel.MARGINAL
        )

        if non_compliant_count > 0:
            self._overall_compliance = ComplianceLevel.NON_COMPLIANT.value
        elif marginal_count > 0:
            self._overall_compliance = ComplianceLevel.MARGINAL.value
        elif compliant_count > 0:
            self._overall_compliance = ComplianceLevel.COMPLIANT.value
        else:
            self._overall_compliance = ComplianceLevel.NOT_TESTED.value

        outputs["overall_compliance"] = self._overall_compliance
        outputs["compliant_count"] = compliant_count
        outputs["non_compliant_count"] = non_compliant_count
        outputs["marginal_count"] = marginal_count
        outputs["battery_category"] = input_data.battery_category
        outputs["thresholds_applied"] = "ev_battery" if is_ev else "industrial_battery"

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ThresholdValidation: %s (%d compliant, %d non-compliant, %d marginal)",
            self._overall_compliance, compliant_count, non_compliant_count, marginal_count,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.THRESHOLD_VALIDATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(
        self, input_data: PerformanceTestingInput,
    ) -> PhaseResult:
        """Generate performance compliance report."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        report = {
            "report_id": f"ptr-{_new_uuid()[:8]}",
            "battery_id": input_data.battery_id,
            "battery_model": input_data.battery_model,
            "battery_category": input_data.battery_category,
            "regulation_reference": "EU Regulation 2023/1542 Art. 10, Annex IV",
            "reporting_year": input_data.reporting_year,
            "overall_compliance": self._overall_compliance,
            "rated_capacity_ah": input_data.rated_capacity_ah,
            "nominal_voltage_v": input_data.nominal_voltage_v,
            "energy_capacity_kwh": input_data.energy_capacity_kwh,
            "metrics": [
                {
                    "name": m.metric_name,
                    "value": m.value,
                    "unit": m.unit,
                    "compliance": m.compliance.value,
                    "threshold_min": m.threshold_min,
                    "threshold_max": m.threshold_max,
                    "margin_pct": m.margin_pct,
                }
                for m in self._metrics
            ],
            "test_summary": {
                "total_tests": len(self._tests),
                "test_types": sorted(set(t.test_type.value for t in self._tests)),
                "labs_used": sorted(set(t.lab_name for t in self._tests if t.lab_name)),
                "standards_applied": sorted(set(t.test_standard.value for t in self._tests)),
            },
            "issued_at": _utcnow().isoformat(),
        }

        # Coverage assessment
        all_test_types = set(t.value for t in TestType)
        covered = set(t.test_type.value for t in self._tests)
        coverage_pct = round(len(covered) / len(all_test_types) * 100, 1)

        outputs["report_id"] = report["report_id"]
        outputs["overall_compliance"] = self._overall_compliance
        outputs["test_coverage_pct"] = coverage_pct
        outputs["test_types_covered"] = sorted(covered)
        outputs["test_types_missing"] = sorted(all_test_types - covered)
        outputs["report_ready"] = True

        if coverage_pct < 50:
            warnings.append(
                f"Test coverage is low ({coverage_pct}%); "
                f"missing types: {', '.join(sorted(all_test_types - covered))}"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ReportGeneration: %s, coverage %.1f%%",
            report["report_id"], coverage_pct,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_GENERATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: PerformanceTestingResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
