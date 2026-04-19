# -*- coding: utf-8 -*-
"""
Full Energy Monitoring Lifecycle Workflow
===================================

8-phase end-to-end master workflow that orchestrates the complete energy
monitoring lifecycle within PACK-039 Energy Monitoring Pack.

Phases:
    1. MeterSetup           -- Meter registration, channels, hierarchy
    2. DataCollection       -- Protocol connect, poll, validate, store
    3. AnomalyResponse      -- Detect, investigate, resolve anomalies
    4. EnPITracking         -- Normalize, calculate EnPIs, review performance
    5. CostAllocation       -- Meter readings, cost calculation, billing
    6. BudgetReview         -- Budget setup, variance analysis, forecast
    7. Reporting            -- Data gathering, report generation, distribution
    8. Consolidation        -- Aggregate all results into unified assessment

The workflow follows GreenLang zero-hallucination principles: all numeric
results flow through deterministic engine calculations. Delegation to
sub-workflows ensures composability and auditability. SHA-256 provenance
hashes guarantee end-to-end traceability.

Regulatory references:
    - ISO 50001:2018 (energy management systems)
    - ISO 50006:2014 (energy performance indicators)
    - ISO 50015:2014 (measurement and verification)
    - IEC 62053 (metering equipment)
    - EN 15232 (building automation impact)
    - ASHRAE Guideline 14 (measurement uncertainty)

Schedule: on-demand
Estimated duration: 60 minutes

Author: GreenLang Team
Version: 39.0.0
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
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

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

# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

DEFAULT_GRID_EMISSION_FACTORS: Dict[str, float] = {
    "US": 0.390,
    "EU": 0.275,
    "UK": 0.207,
    "AU": 0.680,
    "IN": 0.820,
    "CN": 0.580,
    "DEFAULT": 0.400,
}

MONITORING_MATURITY_LEVELS: Dict[str, Dict[str, Any]] = {
    "level_1_basic": {
        "description": "Manual meter reads, monthly utility bills only",
        "typical_savings_pct": 0,
        "automation_pct": 0,
        "data_frequency": "monthly",
    },
    "level_2_interval": {
        "description": "Automated interval metering, basic dashboards",
        "typical_savings_pct": 5,
        "automation_pct": 40,
        "data_frequency": "15min",
    },
    "level_3_analytics": {
        "description": "Advanced analytics, anomaly detection, EnPI tracking",
        "typical_savings_pct": 10,
        "automation_pct": 70,
        "data_frequency": "5min",
    },
    "level_4_predictive": {
        "description": "Predictive analytics, automated optimization, M&V",
        "typical_savings_pct": 18,
        "automation_pct": 90,
        "data_frequency": "1min",
    },
    "level_5_autonomous": {
        "description": "AI-driven autonomous optimization, continuous commissioning",
        "typical_savings_pct": 25,
        "automation_pct": 98,
        "data_frequency": "real_time",
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

class FullMonitoringInput(BaseModel):
    """Input data model for FullMonitoringWorkflow."""

    facility_profile: Dict[str, Any] = Field(
        ...,
        description="Facility data: facility_name, facility_type, floor_area_m2, "
                    "peak_demand_kw, annual_energy_kwh",
    )
    meter_definitions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Meter definitions: meter_name, protocol, address, channels",
    )
    interval_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Interval readings: timestamp, value, unit, meter_id",
    )
    period_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Period data for EnPI: period_label, energy_kwh, floor_area_m2",
    )
    cost_centres: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Cost centres: cost_centre_name, meter_ids, floor_area_m2",
    )
    tariff: Dict[str, Any] = Field(
        default_factory=lambda: {
            "energy_rate_per_kwh": 0.12,
            "demand_rate_per_kw": 15.00,
            "fixed_charge_per_month": 250.00,
            "tax_rate_pct": 5.0,
        },
        description="Tariff structure",
    )
    budget_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Budget items: category, period_label, budget_amount, actual_amount",
    )
    report_types: List[str] = Field(
        default_factory=lambda: ["monthly_management"],
        description="Report types to generate",
    )
    region: str = Field(default="DEFAULT", description="Region for emission factors")
    billing_period: str = Field(default="2026-03", description="Current billing period")
    fiscal_year: str = Field(default="2026", description="Fiscal year")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("facility_profile")
    @classmethod
    def validate_facility_profile(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure facility profile has minimum required fields."""
        required = ["facility_name"]
        missing = [f for f in required if f not in v or v[f] is None]
        if missing:
            raise ValueError(f"facility_profile missing required fields: {missing}")
        return v

class FullMonitoringResult(BaseModel):
    """Complete result from full energy monitoring lifecycle workflow."""

    lifecycle_id: str = Field(..., description="Unique lifecycle assessment ID")
    facility_id: str = Field(default="", description="Facility identifier")
    meter_setup_data: Dict[str, Any] = Field(default_factory=dict)
    data_collection_data: Dict[str, Any] = Field(default_factory=dict)
    anomaly_response_data: Dict[str, Any] = Field(default_factory=dict)
    enpi_tracking_data: Dict[str, Any] = Field(default_factory=dict)
    cost_allocation_data: Dict[str, Any] = Field(default_factory=dict)
    budget_review_data: Dict[str, Any] = Field(default_factory=dict)
    reporting_data: Dict[str, Any] = Field(default_factory=dict)
    consolidation_data: Dict[str, Any] = Field(default_factory=dict)
    meters_registered: int = Field(default=0, ge=0)
    channels_configured: int = Field(default=0, ge=0)
    readings_collected: int = Field(default=0, ge=0)
    data_quality_pct: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    anomalies_detected: int = Field(default=0, ge=0)
    current_eui_kwh_m2: Decimal = Field(default=Decimal("0"), ge=0)
    total_energy_cost: Decimal = Field(default=Decimal("0"), ge=0)
    budget_variance_pct: Decimal = Field(default=Decimal("0"))
    carbon_emissions_tonnes: Decimal = Field(default=Decimal("0"), ge=0)
    monitoring_maturity_level: str = Field(default="level_2_interval")
    reports_generated: int = Field(default=0, ge=0)
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    phases_completed: List[str] = Field(default_factory=list)
    total_duration_ms: int = Field(default=0, ge=0)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class FullMonitoringWorkflow:
    """
    8-phase end-to-end energy monitoring lifecycle workflow.

    Orchestrates meter setup, data collection, anomaly response, EnPI
    tracking, cost allocation, budget review, reporting, and consolidation
    into a single comprehensive pipeline.

    Zero-hallucination: delegates numeric work to deterministic sub-workflow
    calculations. No LLM calls in the computation path. All inter-phase
    data flows through typed Pydantic models.

    Attributes:
        lifecycle_id: Unique lifecycle execution identifier.
        _meter_setup: Results from meter setup phase.
        _data_collection: Results from data collection phase.
        _anomaly_response: Results from anomaly response phase.
        _enpi_tracking: Results from EnPI tracking phase.
        _cost_allocation: Results from cost allocation phase.
        _budget_review: Results from budget review phase.
        _reporting: Results from reporting phase.
        _consolidation: Results from consolidation phase.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = FullMonitoringWorkflow()
        >>> inp = FullMonitoringInput(
        ...     facility_profile={"facility_name": "HQ", "peak_demand_kw": 500},
        ... )
        >>> result = wf.run(inp)
        >>> assert len(result.phases_completed) > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FullMonitoringWorkflow."""
        self.lifecycle_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._meter_setup: Dict[str, Any] = {}
        self._data_collection: Dict[str, Any] = {}
        self._anomaly_response: Dict[str, Any] = {}
        self._enpi_tracking: Dict[str, Any] = {}
        self._cost_allocation: Dict[str, Any] = {}
        self._budget_review: Dict[str, Any] = {}
        self._reporting: Dict[str, Any] = {}
        self._consolidation: Dict[str, Any] = {}
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: FullMonitoringInput) -> FullMonitoringResult:
        """
        Execute the 8-phase full energy monitoring lifecycle workflow.

        Args:
            input_data: Validated full lifecycle input.

        Returns:
            FullMonitoringResult with all sub-results and aggregate metrics.

        Raises:
            ValueError: If facility profile validation fails.
        """
        t_start = time.perf_counter()
        started_at = utcnow()
        facility_name = input_data.facility_profile.get("facility_name", "Unknown")
        self.logger.info(
            "Starting full monitoring lifecycle workflow %s for facility=%s",
            self.lifecycle_id, facility_name,
        )

        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Meter Setup
            phase1 = self._phase_meter_setup(input_data)
            self._phase_results.append(phase1)

            # Phase 2: Data Collection
            phase2 = self._phase_data_collection(input_data)
            self._phase_results.append(phase2)

            # Phase 3: Anomaly Response
            phase3 = self._phase_anomaly_response(input_data)
            self._phase_results.append(phase3)

            # Phase 4: EnPI Tracking
            phase4 = self._phase_enpi_tracking(input_data)
            self._phase_results.append(phase4)

            # Phase 5: Cost Allocation
            phase5 = self._phase_cost_allocation(input_data)
            self._phase_results.append(phase5)

            # Phase 6: Budget Review
            phase6 = self._phase_budget_review(input_data)
            self._phase_results.append(phase6)

            # Phase 7: Reporting
            phase7 = self._phase_reporting(input_data)
            self._phase_results.append(phase7)

            # Phase 8: Consolidation
            phase8 = self._phase_consolidation(input_data)
            self._phase_results.append(phase8)

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "Full monitoring lifecycle workflow failed: %s", exc, exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Aggregate final metrics
        fp = input_data.facility_profile
        facility_id = fp.get("facility_id", f"fac-{_new_uuid()[:8]}")

        meters_registered = int(self._meter_setup.get("meters_registered", 0))
        channels_configured = int(self._meter_setup.get("channels_configured", 0))
        readings_collected = int(self._data_collection.get("readings_collected", 0))
        data_quality = Decimal(str(self._data_collection.get("data_quality_pct", 0)))
        anomalies = int(self._anomaly_response.get("anomalies_detected", 0))
        current_eui = Decimal(str(self._enpi_tracking.get("current_eui_kwh_m2", 0)))
        total_cost = Decimal(str(self._cost_allocation.get("total_facility_cost", 0)))
        budget_var = Decimal(str(self._budget_review.get("ytd_variance_pct", 0)))
        reports = int(self._reporting.get("reports_generated", 0))

        # Carbon emissions
        annual_kwh = float(fp.get("annual_energy_kwh", 0))
        ef = DEFAULT_GRID_EMISSION_FACTORS.get(
            input_data.region, DEFAULT_GRID_EMISSION_FACTORS["DEFAULT"]
        )
        carbon = Decimal(str(round(annual_kwh * ef / 1000, 2)))

        # Monitoring maturity assessment
        maturity = self._assess_maturity(
            meters_registered, channels_configured, readings_collected,
            anomalies, float(current_eui), reports,
        )

        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = FullMonitoringResult(
            lifecycle_id=self.lifecycle_id,
            facility_id=facility_id,
            meter_setup_data=self._meter_setup,
            data_collection_data=self._data_collection,
            anomaly_response_data=self._anomaly_response,
            enpi_tracking_data=self._enpi_tracking,
            cost_allocation_data=self._cost_allocation,
            budget_review_data=self._budget_review,
            reporting_data=self._reporting,
            consolidation_data=self._consolidation,
            meters_registered=meters_registered,
            channels_configured=channels_configured,
            readings_collected=readings_collected,
            data_quality_pct=data_quality,
            anomalies_detected=anomalies,
            current_eui_kwh_m2=current_eui,
            total_energy_cost=total_cost,
            budget_variance_pct=budget_var,
            carbon_emissions_tonnes=carbon,
            monitoring_maturity_level=maturity,
            reports_generated=reports,
            status=overall_status,
            phases_completed=completed_phases,
            total_duration_ms=int(elapsed_ms),
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Full monitoring lifecycle %s completed in %dms status=%s "
            "meters=%d readings=%d anomalies=%d EUI=%.1f cost=$%.0f "
            "carbon=%.2f t maturity=%s",
            self.lifecycle_id, int(elapsed_ms), overall_status.value,
            meters_registered, readings_collected, anomalies,
            float(current_eui), float(total_cost), float(carbon), maturity,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Meter Setup
    # -------------------------------------------------------------------------

    def _phase_meter_setup(
        self, input_data: FullMonitoringInput
    ) -> PhaseResult:
        """Register meters, configure channels, build hierarchy."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        fp = input_data.facility_profile
        facility_name = fp.get("facility_name", "Unknown")

        from .meter_setup_workflow import METER_PROTOCOL_SPECS

        meters: List[Dict[str, Any]] = []
        channels_count = 0

        if input_data.meter_definitions:
            for idx, mdef in enumerate(input_data.meter_definitions):
                protocol = mdef.get("protocol", "modbus_tcp")
                spec = METER_PROTOCOL_SPECS.get(protocol, METER_PROTOCOL_SPECS.get("modbus_tcp", {}))

                meter_id = f"mtr-{_new_uuid()[:8]}"
                meters.append({
                    "meter_id": meter_id,
                    "meter_name": mdef.get("meter_name", f"Meter-{idx + 1}"),
                    "protocol": protocol,
                    "address": mdef.get("address", f"unit-{idx + 1}"),
                    "poll_interval_s": spec.get("typical_poll_interval_s", 15),
                    "status": "registered",
                })

                user_channels = mdef.get("channels", [])
                if user_channels:
                    channels_count += len(user_channels)
                else:
                    channels_count += 6  # Default electrical channels
        else:
            warnings.append("No meter definitions; creating default main incomer")
            meter_id = f"mtr-{_new_uuid()[:8]}"
            meters.append({
                "meter_id": meter_id,
                "meter_name": f"{facility_name} - Main Incomer",
                "protocol": "modbus_tcp",
                "address": "unit-1",
                "poll_interval_s": 5,
                "status": "registered",
            })
            channels_count = 6

        self._meter_setup = {
            "meters_registered": len(meters),
            "channels_configured": channels_count,
            "hierarchy_depth": 2,
            "meters": meters,
            "commission_pass_rate_pct": 100.0,
        }

        outputs.update({
            "meters_registered": len(meters),
            "channels_configured": channels_count,
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 MeterSetup: %d meters, %d channels",
            len(meters), channels_count,
        )
        return PhaseResult(
            phase_name="meter_setup", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Data Collection
    # -------------------------------------------------------------------------

    def _phase_data_collection(
        self, input_data: FullMonitoringInput
    ) -> PhaseResult:
        """Poll meters, validate readings, store data."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        meters = self._meter_setup.get("meters", [])
        readings_count = 0
        good_count = 0
        suspect_count = 0

        if input_data.interval_data:
            readings_count = len(input_data.interval_data)
            for reading in input_data.interval_data:
                quality = reading.get("quality", "good")
                if quality == "good":
                    good_count += 1
                else:
                    suspect_count += 1
        else:
            # Simulate one reading per channel
            channels = self._meter_setup.get("channels_configured", 6)
            readings_count = channels * len(meters)
            good_count = readings_count
            warnings.append("No interval data; simulated readings from meter config")

        data_quality = round(good_count / max(readings_count, 1) * 100, 1)

        self._data_collection = {
            "readings_collected": readings_count,
            "readings_valid": good_count,
            "readings_suspect": suspect_count,
            "data_quality_pct": data_quality,
            "records_stored": readings_count,
        }

        outputs.update({
            "readings_collected": readings_count,
            "data_quality_pct": data_quality,
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 DataCollection: %d readings, quality=%.1f%%",
            readings_count, data_quality,
        )
        return PhaseResult(
            phase_name="data_collection", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Anomaly Response
    # -------------------------------------------------------------------------

    def _phase_anomaly_response(
        self, input_data: FullMonitoringInput
    ) -> PhaseResult:
        """Detect and respond to energy anomalies."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        from .anomaly_response_workflow import ANOMALY_THRESHOLDS

        fp = input_data.facility_profile
        facility_type = fp.get("facility_type", "office_building")
        energy_type = f"electricity_{facility_type}" if facility_type in (
            "commercial", "industrial"
        ) else "electricity_commercial"

        thresholds = ANOMALY_THRESHOLDS.get(energy_type, ANOMALY_THRESHOLDS["electricity_commercial"])

        # Detect anomalies from interval data
        peak_kw = float(fp.get("peak_demand_kw", 500))
        baseline = peak_kw * 0.55  # Approximate load factor
        anomalies_detected = 0
        total_excess_kwh = 0.0

        if input_data.interval_data:
            for reading in input_data.interval_data:
                value = float(reading.get("value", 0))
                if baseline > 0:
                    deviation_pct = abs((value - baseline) / baseline) * 100
                    if deviation_pct > thresholds["spike_threshold_pct"]:
                        anomalies_detected += 1
                        excess = max(0, value - baseline)
                        total_excess_kwh += excess * 0.25  # 15-min interval
        else:
            # Estimate anomalies from typical rate
            readings_count = self._data_collection.get("readings_collected", 0)
            anomalies_detected = max(0, int(readings_count * 0.02))  # ~2% anomaly rate
            total_excess_kwh = anomalies_detected * baseline * 0.3 * 0.25

        self._anomaly_response = {
            "anomalies_detected": anomalies_detected,
            "total_excess_kwh": round(total_excess_kwh, 1),
            "estimated_cost_impact": round(total_excess_kwh * 0.12, 2),
            "investigations_completed": anomalies_detected,
            "actions_recommended": anomalies_detected,
        }

        outputs.update({
            "anomalies_detected": anomalies_detected,
            "excess_energy_kwh": round(total_excess_kwh, 1),
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 AnomalyResponse: %d anomalies, excess=%.1f kWh",
            anomalies_detected, total_excess_kwh,
        )
        return PhaseResult(
            phase_name="anomaly_response", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: EnPI Tracking
    # -------------------------------------------------------------------------

    def _phase_enpi_tracking(
        self, input_data: FullMonitoringInput
    ) -> PhaseResult:
        """Calculate energy performance indicators."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        from .enpi_tracking_workflow import ENPI_BENCHMARKS

        fp = input_data.facility_profile
        facility_type = fp.get("facility_type", "office_building")
        benchmark = ENPI_BENCHMARKS.get(facility_type, ENPI_BENCHMARKS["office_building"])

        annual_kwh = float(fp.get("annual_energy_kwh", 0))
        floor_area = float(fp.get("floor_area_m2", 0))

        if input_data.period_data:
            # Calculate from period data
            total_kwh = sum(float(p.get("energy_kwh", 0)) for p in input_data.period_data)
            total_area = max(
                max((float(p.get("floor_area_m2", 0)) for p in input_data.period_data), default=0),
                floor_area,
            )
            eui = round(total_kwh / max(total_area, 0.01), 2) if total_area > 0 else 0
        elif annual_kwh > 0 and floor_area > 0:
            eui = round(annual_kwh / floor_area, 2)
            total_kwh = annual_kwh
        else:
            eui = benchmark["typical_eui_kwh_m2"]
            total_kwh = eui * max(floor_area, 1000)
            warnings.append("No energy/area data; using benchmark EUI")

        # Benchmark rating
        typical = benchmark["typical_eui_kwh_m2"]
        best = benchmark["best_practice_eui_kwh_m2"]
        poor = benchmark["poor_eui_kwh_m2"]

        if eui <= best:
            rating = "best_in_class"
        elif eui <= (best + typical) / 2:
            rating = "above_average"
        elif eui <= typical:
            rating = "average"
        elif eui <= (typical + poor) / 2:
            rating = "below_average"
        else:
            rating = "poor"

        self._enpi_tracking = {
            "current_eui_kwh_m2": eui,
            "benchmark_rating": rating,
            "typical_eui": typical,
            "best_practice_eui": best,
            "total_energy_kwh": total_kwh,
            "floor_area_m2": floor_area,
            "iso_50006_category": benchmark.get("iso_50006_category", ""),
        }

        outputs.update({
            "current_eui_kwh_m2": eui,
            "benchmark_rating": rating,
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 EnPITracking: EUI=%.1f kWh/m2 rating=%s",
            eui, rating,
        )
        return PhaseResult(
            phase_name="enpi_tracking", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Cost Allocation
    # -------------------------------------------------------------------------

    def _phase_cost_allocation(
        self, input_data: FullMonitoringInput
    ) -> PhaseResult:
        """Calculate and allocate energy costs."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        from .cost_allocation_workflow import ALLOCATION_METHODS

        fp = input_data.facility_profile
        tariff = input_data.tariff
        energy_rate = float(tariff.get("energy_rate_per_kwh", 0.12))
        demand_rate = float(tariff.get("demand_rate_per_kw", 15.00))
        fixed_charge = float(tariff.get("fixed_charge_per_month", 250.00))
        tax_rate = float(tariff.get("tax_rate_pct", 5.0)) / 100.0

        total_kwh = float(self._enpi_tracking.get("total_energy_kwh", 0))
        peak_kw = float(fp.get("peak_demand_kw", 0))

        # Monthly energy calculation (1/12 of annual)
        monthly_kwh = total_kwh / 12.0
        energy_cost = round(monthly_kwh * energy_rate, 2)
        demand_cost = round(peak_kw * demand_rate, 2)
        subtotal = energy_cost + demand_cost + fixed_charge
        tax = round(subtotal * tax_rate, 2)
        total_cost = round(subtotal + tax, 2)

        # Allocate to cost centres
        bills: List[Dict[str, Any]] = []
        if input_data.cost_centres:
            total_area = sum(float(cc.get("floor_area_m2", 0)) for cc in input_data.cost_centres)
            for cc in input_data.cost_centres:
                cc_area = float(cc.get("floor_area_m2", 0))
                proportion = cc_area / max(total_area, 0.01)
                cc_cost = round(total_cost * proportion, 2)
                bills.append({
                    "cost_centre": cc.get("cost_centre_name", ""),
                    "proportion": round(proportion * 100, 1),
                    "allocated_cost": cc_cost,
                })
        else:
            bills.append({
                "cost_centre": "Facility Total",
                "proportion": 100.0,
                "allocated_cost": total_cost,
            })

        self._cost_allocation = {
            "total_facility_cost": total_cost,
            "energy_cost": energy_cost,
            "demand_cost": demand_cost,
            "fixed_cost": fixed_charge,
            "tax": tax,
            "cost_centres_billed": len(bills),
            "bills": bills,
            "allocation_method": "floor_area_pro_rata" if input_data.cost_centres else "single_facility",
        }

        outputs.update({
            "total_facility_cost": total_cost,
            "cost_centres_billed": len(bills),
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 5 CostAllocation: total=$%.2f centres=%d",
            total_cost, len(bills),
        )
        return PhaseResult(
            phase_name="cost_allocation", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 6: Budget Review
    # -------------------------------------------------------------------------

    def _phase_budget_review(
        self, input_data: FullMonitoringInput
    ) -> PhaseResult:
        """Review budget variances and update forecast."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        from .budget_review_workflow import VARIANCE_THRESHOLDS

        total_cost = float(self._cost_allocation.get("total_facility_cost", 0))

        if input_data.budget_items:
            ytd_budget = sum(
                float(b.get("budget_amount", 0)) for b in input_data.budget_items
                if b.get("period_label", "") <= input_data.billing_period
            )
            ytd_actual = sum(
                float(b.get("actual_amount", 0)) for b in input_data.budget_items
                if b.get("actual_amount") is not None
                and b.get("period_label", "") <= input_data.billing_period
            )
        else:
            # Estimate from current cost
            ytd_budget = total_cost * 1.05  # 5% buffer
            ytd_actual = total_cost
            warnings.append("No budget items; estimating from current cost")

        variance_abs = ytd_actual - ytd_budget
        variance_pct = round(
            variance_abs / max(ytd_budget, 0.01) * 100, 1
        )

        thresholds = VARIANCE_THRESHOLDS.get("energy_cost", VARIANCE_THRESHOLDS["energy_cost"])
        if abs(variance_pct) >= thresholds["critical_pct"]:
            budget_status = "critical"
        elif abs(variance_pct) >= thresholds["warning_pct"]:
            budget_status = "warning"
        else:
            budget_status = "on_target"

        # Forecast
        try:
            review_month = int(input_data.billing_period.split("-")[1])
        except (IndexError, ValueError):
            review_month = 3

        months_elapsed = review_month
        if months_elapsed > 0 and ytd_actual > 0:
            monthly_run = ytd_actual / months_elapsed
            forecast_year = round(monthly_run * 12, 2)
        else:
            forecast_year = ytd_budget * 12 / max(months_elapsed, 1)

        self._budget_review = {
            "ytd_budget": round(ytd_budget, 2),
            "ytd_actual": round(ytd_actual, 2),
            "ytd_variance_abs": round(variance_abs, 2),
            "ytd_variance_pct": variance_pct,
            "budget_status": budget_status,
            "forecast_year_end": round(forecast_year, 2),
        }

        outputs.update({
            "ytd_variance_pct": variance_pct,
            "budget_status": budget_status,
            "forecast_year_end": round(forecast_year, 2),
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 6 BudgetReview: variance=%.1f%% status=%s forecast=$%.0f",
            variance_pct, budget_status, forecast_year,
        )
        return PhaseResult(
            phase_name="budget_review", phase_number=6,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 7: Reporting
    # -------------------------------------------------------------------------

    def _phase_reporting(
        self, input_data: FullMonitoringInput
    ) -> PhaseResult:
        """Generate energy monitoring reports."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        from .reporting_workflow import REPORT_SCHEDULES


        report_types = input_data.report_types or ["monthly_management"]
        reports: List[Dict[str, Any]] = []

        now_iso = utcnow().isoformat() + "Z"

        for rt in report_types:
            schedule = REPORT_SCHEDULES.get(rt, REPORT_SCHEDULES.get("monthly_management", {}))
            report = {
                "report_id": f"rpt-{_new_uuid()[:8]}",
                "report_type": rt,
                "description": schedule.get("description", ""),
                "sections": schedule.get("sections", []),
                "estimated_pages": schedule.get("estimated_pages", 5),
                "generated_at": now_iso,
                "content_summary": {
                    "meters": self._meter_setup.get("meters_registered", 0),
                    "readings": self._data_collection.get("readings_collected", 0),
                    "anomalies": self._anomaly_response.get("anomalies_detected", 0),
                    "eui": self._enpi_tracking.get("current_eui_kwh_m2", 0),
                    "cost": self._cost_allocation.get("total_facility_cost", 0),
                    "budget_variance_pct": self._budget_review.get("ytd_variance_pct", 0),
                },
            }
            reports.append(report)

        self._reporting = {
            "reports_generated": len(reports),
            "report_types": report_types,
            "reports": reports,
        }

        outputs.update({
            "reports_generated": len(reports),
            "report_types": report_types,
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 7 Reporting: %d reports generated",
            len(reports),
        )
        return PhaseResult(
            phase_name="reporting", phase_number=7,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 8: Consolidation
    # -------------------------------------------------------------------------

    def _phase_consolidation(
        self, input_data: FullMonitoringInput
    ) -> PhaseResult:
        """Aggregate all results into unified monitoring assessment."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        fp = input_data.facility_profile
        facility_name = fp.get("facility_name", "Unknown")

        # Monitoring health score (0-100)
        scores: List[float] = []

        # Data quality component (25%)
        dq = float(self._data_collection.get("data_quality_pct", 0))
        scores.append(min(dq, 100))

        # Meter coverage component (25%)
        meters = self._meter_setup.get("meters_registered", 0)
        meter_score = min(meters * 20, 100)  # 5 meters = 100%
        scores.append(meter_score)

        # Anomaly resolution component (25%)
        anomalies = self._anomaly_response.get("anomalies_detected", 0)
        actions = self._anomaly_response.get("actions_recommended", 0)
        if anomalies > 0 and actions > 0:
            resolution_score = min((actions / max(anomalies, 1)) * 100, 100)
        else:
            resolution_score = 100  # No anomalies = perfect
        scores.append(resolution_score)

        # Budget adherence component (25%)
        var_pct = abs(float(self._budget_review.get("ytd_variance_pct", 0)))
        budget_score = max(0, 100 - var_pct * 5)  # Each 1% variance = -5 points
        scores.append(budget_score)

        health_score = round(sum(scores) / len(scores), 1)

        # Key recommendations
        recommendations: List[str] = []
        if dq < 95:
            recommendations.append(
                "Improve data quality to >95% by addressing meter communication issues"
            )
        if meters < 3:
            recommendations.append(
                "Add sub-meters to improve granularity and cost allocation accuracy"
            )
        eui = float(self._enpi_tracking.get("current_eui_kwh_m2", 0))
        typical = float(self._enpi_tracking.get("typical_eui", 180))
        if eui > typical:
            recommendations.append(
                f"EUI ({eui:.0f} kWh/m2) exceeds benchmark ({typical:.0f}); "
                f"investigate HVAC and lighting efficiency"
            )
        if var_pct > 10:
            recommendations.append(
                f"Budget variance ({var_pct:.1f}%) is significant; review forecasting model"
            )

        # Carbon intensity
        annual_kwh = float(fp.get("annual_energy_kwh", 0))
        floor_area = float(fp.get("floor_area_m2", 0))
        ef = DEFAULT_GRID_EMISSION_FACTORS.get(
            input_data.region, DEFAULT_GRID_EMISSION_FACTORS["DEFAULT"]
        )
        carbon_total = round(annual_kwh * ef / 1000, 2)
        carbon_intensity = round(
            carbon_total / max(floor_area, 0.01) * 1000, 2
        ) if floor_area > 0 else 0

        self._consolidation = {
            "facility_name": facility_name,
            "health_score": health_score,
            "component_scores": {
                "data_quality": round(scores[0], 1),
                "meter_coverage": round(scores[1], 1),
                "anomaly_resolution": round(scores[2], 1),
                "budget_adherence": round(scores[3], 1),
            },
            "carbon_emissions_tonnes": carbon_total,
            "carbon_intensity_kgco2_m2": carbon_intensity,
            "recommendations": recommendations,
            "monitoring_maturity": self._assess_maturity(
                self._meter_setup.get("meters_registered", 0),
                self._meter_setup.get("channels_configured", 0),
                self._data_collection.get("readings_collected", 0),
                self._anomaly_response.get("anomalies_detected", 0),
                eui, self._reporting.get("reports_generated", 0),
            ),
        }

        outputs.update({
            "health_score": health_score,
            "carbon_tonnes": carbon_total,
            "recommendations_count": len(recommendations),
            "monitoring_maturity": self._consolidation["monitoring_maturity"],
        })

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 8 Consolidation: health=%.1f carbon=%.2f t recommendations=%d",
            health_score, carbon_total, len(recommendations),
        )
        return PhaseResult(
            phase_name="consolidation", phase_number=8,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _assess_maturity(
        self,
        meters: int,
        channels: int,
        readings: int,
        anomalies_detected: int,
        eui: float,
        reports: int,
    ) -> str:
        """Assess monitoring maturity level based on system capabilities."""
        score = 0

        # Metering coverage
        if meters >= 10:
            score += 3
        elif meters >= 5:
            score += 2
        elif meters >= 1:
            score += 1

        # Channel depth
        if channels >= 50:
            score += 2
        elif channels >= 10:
            score += 1

        # Data collection
        if readings >= 10000:
            score += 3
        elif readings >= 1000:
            score += 2
        elif readings > 0:
            score += 1

        # Analytics capability
        if anomalies_detected > 0:
            score += 2
        if eui > 0:
            score += 1

        # Reporting
        if reports >= 3:
            score += 2
        elif reports >= 1:
            score += 1

        # Map score to maturity level
        if score >= 12:
            return "level_5_autonomous"
        elif score >= 9:
            return "level_4_predictive"
        elif score >= 6:
            return "level_3_analytics"
        elif score >= 3:
            return "level_2_interval"
        return "level_1_basic"

    def _compute_provenance(self, result: FullMonitoringResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
