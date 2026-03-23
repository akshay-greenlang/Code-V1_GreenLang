# -*- coding: utf-8 -*-
"""
Anomaly Response Workflow
===================================

3-phase workflow for detecting energy consumption anomalies, investigating
root causes, and resolving through corrective actions within PACK-039
Energy Monitoring Pack.

Phases:
    1. Detect       -- Apply statistical and rule-based anomaly detection
    2. Investigate   -- Root cause analysis and impact assessment
    3. Resolve       -- Corrective action planning and verification

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory references:
    - ISO 50001:2018 Clause 9.1 (monitoring, measurement, analysis)
    - ISO 50006:2014 (energy baselines and performance indicators)
    - ASHRAE Guideline 14 (measurement uncertainty)
    - EN 15232 (building automation anomaly management)

Schedule: continuous / event-triggered
Estimated duration: 10 minutes

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


class AnomalySeverity(str, Enum):
    """Anomaly severity classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(str, Enum):
    """Type of energy anomaly detected."""

    SPIKE = "spike"
    BASELINE_DRIFT = "baseline_drift"
    OFF_HOURS = "off_hours"
    EQUIPMENT_FAILURE = "equipment_failure"
    METER_ERROR = "meter_error"
    SEASONAL_DEVIATION = "seasonal_deviation"
    LOAD_IMBALANCE = "load_imbalance"


class ResolutionStatus(str, Enum):
    """Resolution action status."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DEFERRED = "deferred"
    FALSE_POSITIVE = "false_positive"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

ANOMALY_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "electricity_commercial": {
        "description": "Commercial building electricity consumption",
        "spike_threshold_pct": 30.0,
        "spike_absolute_kw": 100.0,
        "baseline_drift_pct": 15.0,
        "off_hours_threshold_pct": 25.0,
        "seasonal_deviation_pct": 20.0,
        "load_imbalance_pct": 10.0,
        "min_duration_minutes": 15,
        "cooldown_minutes": 60,
        "z_score_threshold": 3.0,
    },
    "electricity_industrial": {
        "description": "Industrial facility electricity consumption",
        "spike_threshold_pct": 20.0,
        "spike_absolute_kw": 500.0,
        "baseline_drift_pct": 10.0,
        "off_hours_threshold_pct": 40.0,
        "seasonal_deviation_pct": 15.0,
        "load_imbalance_pct": 8.0,
        "min_duration_minutes": 15,
        "cooldown_minutes": 30,
        "z_score_threshold": 2.5,
    },
    "gas_commercial": {
        "description": "Commercial building gas consumption",
        "spike_threshold_pct": 40.0,
        "spike_absolute_kw": 50.0,
        "baseline_drift_pct": 20.0,
        "off_hours_threshold_pct": 30.0,
        "seasonal_deviation_pct": 25.0,
        "load_imbalance_pct": 15.0,
        "min_duration_minutes": 60,
        "cooldown_minutes": 120,
        "z_score_threshold": 3.0,
    },
    "gas_industrial": {
        "description": "Industrial facility gas consumption",
        "spike_threshold_pct": 25.0,
        "spike_absolute_kw": 200.0,
        "baseline_drift_pct": 12.0,
        "off_hours_threshold_pct": 50.0,
        "seasonal_deviation_pct": 18.0,
        "load_imbalance_pct": 10.0,
        "min_duration_minutes": 30,
        "cooldown_minutes": 60,
        "z_score_threshold": 2.5,
    },
    "water_commercial": {
        "description": "Commercial building water consumption",
        "spike_threshold_pct": 50.0,
        "spike_absolute_kw": 10.0,
        "baseline_drift_pct": 25.0,
        "off_hours_threshold_pct": 20.0,
        "seasonal_deviation_pct": 30.0,
        "load_imbalance_pct": 20.0,
        "min_duration_minutes": 60,
        "cooldown_minutes": 240,
        "z_score_threshold": 3.5,
    },
    "steam_industrial": {
        "description": "Industrial steam consumption",
        "spike_threshold_pct": 20.0,
        "spike_absolute_kw": 300.0,
        "baseline_drift_pct": 10.0,
        "off_hours_threshold_pct": 60.0,
        "seasonal_deviation_pct": 15.0,
        "load_imbalance_pct": 8.0,
        "min_duration_minutes": 15,
        "cooldown_minutes": 30,
        "z_score_threshold": 2.0,
    },
    "thermal_commercial": {
        "description": "Commercial thermal (chilled/hot water) consumption",
        "spike_threshold_pct": 35.0,
        "spike_absolute_kw": 75.0,
        "baseline_drift_pct": 18.0,
        "off_hours_threshold_pct": 30.0,
        "seasonal_deviation_pct": 25.0,
        "load_imbalance_pct": 12.0,
        "min_duration_minutes": 30,
        "cooldown_minutes": 60,
        "z_score_threshold": 3.0,
    },
    "compressed_air_industrial": {
        "description": "Industrial compressed air consumption",
        "spike_threshold_pct": 25.0,
        "spike_absolute_kw": 150.0,
        "baseline_drift_pct": 15.0,
        "off_hours_threshold_pct": 35.0,
        "seasonal_deviation_pct": 10.0,
        "load_imbalance_pct": 10.0,
        "min_duration_minutes": 5,
        "cooldown_minutes": 15,
        "z_score_threshold": 2.5,
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


class EnergyReading(BaseModel):
    """A single energy consumption reading for anomaly analysis."""

    timestamp: str = Field(..., description="ISO 8601 timestamp")
    value: Decimal = Field(default=Decimal("0"), description="Measured value")
    unit: str = Field(default="kW", description="Engineering unit")
    meter_id: str = Field(default="", description="Source meter identifier")
    channel_name: str = Field(default="active_power", description="Channel name")
    quality: str = Field(default="good", description="Data quality flag")


class AnomalyResponseInput(BaseModel):
    """Input data model for AnomalyResponseWorkflow."""

    facility_id: str = Field(default_factory=lambda: f"fac-{uuid.uuid4().hex[:8]}")
    facility_name: str = Field(..., min_length=1, description="Facility name")
    energy_type: str = Field(
        default="electricity_commercial",
        description="Energy type and facility class key from thresholds",
    )
    readings: List[EnergyReading] = Field(
        default_factory=list,
        description="Recent energy readings for anomaly analysis",
    )
    baseline_value: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Expected baseline value for comparison",
    )
    baseline_std_dev: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Standard deviation of baseline values",
    )
    operating_schedule: Dict[str, Any] = Field(
        default_factory=lambda: {
            "weekday_start": "08:00",
            "weekday_end": "18:00",
            "weekend_occupied": False,
        },
        description="Operating schedule for off-hours detection",
    )
    active_anomalies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Currently open/active anomaly records",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("facility_name")
    @classmethod
    def validate_facility_name(cls, v: str) -> str:
        """Ensure facility name is non-empty after stripping."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("facility_name must not be blank")
        return stripped


class AnomalyResponseResult(BaseModel):
    """Complete result from anomaly response workflow."""

    response_id: str = Field(..., description="Unique anomaly response execution ID")
    facility_id: str = Field(default="", description="Facility identifier")
    anomalies_detected: int = Field(default=0, ge=0)
    anomalies_by_severity: Dict[str, int] = Field(default_factory=dict)
    anomalies_by_type: Dict[str, int] = Field(default_factory=dict)
    total_excess_energy_kwh: Decimal = Field(default=Decimal("0"), ge=0)
    estimated_cost_impact: Decimal = Field(default=Decimal("0"), ge=0)
    investigations_completed: int = Field(default=0, ge=0)
    root_causes_identified: int = Field(default=0, ge=0)
    actions_recommended: int = Field(default=0, ge=0)
    actions_resolved: int = Field(default=0, ge=0)
    anomaly_records: List[Dict[str, Any]] = Field(default_factory=list)
    response_duration_ms: int = Field(default=0, ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class AnomalyResponseWorkflow:
    """
    3-phase anomaly response workflow for energy monitoring systems.

    Detects energy consumption anomalies using statistical and rule-based
    methods, investigates root causes, and plans corrective actions.

    Zero-hallucination: all detection thresholds and severity classifications
    are sourced from validated reference data. No LLM calls in the anomaly
    detection or impact calculation path.

    Attributes:
        response_id: Unique anomaly response execution identifier.
        _anomalies: Detected anomaly records.
        _investigations: Root cause investigation results.
        _resolutions: Corrective action records.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = AnomalyResponseWorkflow()
        >>> reading = EnergyReading(timestamp="2026-03-01T14:00:00Z", value=Decimal("500"))
        >>> inp = AnomalyResponseInput(
        ...     facility_name="Office HQ",
        ...     baseline_value=Decimal("300"),
        ...     baseline_std_dev=Decimal("30"),
        ...     readings=[reading],
        ... )
        >>> result = wf.run(inp)
        >>> assert result.anomalies_detected >= 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AnomalyResponseWorkflow."""
        self.response_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._anomalies: List[Dict[str, Any]] = []
        self._investigations: List[Dict[str, Any]] = []
        self._resolutions: List[Dict[str, Any]] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, input_data: AnomalyResponseInput) -> AnomalyResponseResult:
        """
        Execute the 3-phase anomaly response workflow.

        Args:
            input_data: Validated anomaly response input.

        Returns:
            AnomalyResponseResult with detected anomalies, investigations,
            and corrective actions.

        Raises:
            ValueError: If input validation fails.
        """
        t_start = time.perf_counter()
        started_at = _utcnow()
        self.logger.info(
            "Starting anomaly response workflow %s for facility=%s type=%s",
            self.response_id, input_data.facility_name, input_data.energy_type,
        )

        self._phase_results = []
        self._anomalies = []
        self._investigations = []
        self._resolutions = []

        try:
            phase1 = self._phase_detect(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_investigate(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_resolve(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error("Anomaly response workflow failed: %s", exc, exc_info=True)
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        # Aggregate severity and type counts
        severity_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for anom in self._anomalies:
            sev = anom.get("severity", "low")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            atype = anom.get("anomaly_type", "unknown")
            type_counts[atype] = type_counts.get(atype, 0) + 1

        total_excess = sum(
            Decimal(str(a.get("excess_energy_kwh", 0))) for a in self._anomalies
        )
        total_cost = sum(
            Decimal(str(a.get("estimated_cost", 0))) for a in self._anomalies
        )

        result = AnomalyResponseResult(
            response_id=self.response_id,
            facility_id=input_data.facility_id,
            anomalies_detected=len(self._anomalies),
            anomalies_by_severity=severity_counts,
            anomalies_by_type=type_counts,
            total_excess_energy_kwh=total_excess,
            estimated_cost_impact=total_cost,
            investigations_completed=len(self._investigations),
            root_causes_identified=sum(
                1 for inv in self._investigations if inv.get("root_cause")
            ),
            actions_recommended=len(self._resolutions),
            actions_resolved=sum(
                1 for r in self._resolutions
                if r.get("status") == ResolutionStatus.RESOLVED.value
            ),
            anomaly_records=self._anomalies,
            response_duration_ms=int(elapsed_ms),
            phases_completed=completed_phases,
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Anomaly response workflow %s completed in %dms anomalies=%d "
            "excess=%.1f kWh cost=$%.2f",
            self.response_id, int(elapsed_ms), len(self._anomalies),
            float(total_excess), float(total_cost),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Detect
    # -------------------------------------------------------------------------

    def _phase_detect(
        self, input_data: AnomalyResponseInput
    ) -> PhaseResult:
        """Apply statistical and rule-based anomaly detection."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        thresholds = ANOMALY_THRESHOLDS.get(
            input_data.energy_type,
            ANOMALY_THRESHOLDS["electricity_commercial"],
        )
        baseline = float(input_data.baseline_value)
        std_dev = float(input_data.baseline_std_dev) if input_data.baseline_std_dev > 0 else baseline * 0.15
        z_threshold = thresholds["z_score_threshold"]
        spike_pct = thresholds["spike_threshold_pct"]

        if not input_data.readings:
            warnings.append("No readings provided for anomaly detection")
        else:
            for reading in input_data.readings:
                value = float(reading.value)
                anomalies_for_reading: List[str] = []

                # Z-score anomaly detection
                if std_dev > 0 and baseline > 0:
                    z_score = abs(value - baseline) / std_dev
                    if z_score >= z_threshold:
                        anomalies_for_reading.append("spike")

                # Percentage deviation from baseline
                if baseline > 0:
                    deviation_pct = ((value - baseline) / baseline) * 100
                    if deviation_pct > spike_pct:
                        if "spike" not in anomalies_for_reading:
                            anomalies_for_reading.append("spike")
                    elif deviation_pct < -spike_pct:
                        anomalies_for_reading.append("equipment_failure")

                    # Baseline drift check
                    if abs(deviation_pct) > thresholds["baseline_drift_pct"]:
                        anomalies_for_reading.append("baseline_drift")

                # Off-hours check
                if self._is_off_hours(reading.timestamp, input_data.operating_schedule):
                    off_hours_threshold = baseline * thresholds["off_hours_threshold_pct"] / 100
                    if value > off_hours_threshold:
                        anomalies_for_reading.append("off_hours")

                # Create anomaly records
                for atype in anomalies_for_reading:
                    severity = self._classify_severity(value, baseline, thresholds)
                    excess = max(0, value - baseline)
                    # Estimate cost at $0.10/kWh for 15-min interval
                    est_cost = round(excess * 0.25 * 0.10, 2)

                    self._anomalies.append({
                        "anomaly_id": f"anom-{_new_uuid()[:8]}",
                        "anomaly_type": atype,
                        "severity": severity,
                        "timestamp": reading.timestamp,
                        "measured_value": value,
                        "baseline_value": baseline,
                        "deviation_pct": round(
                            ((value - baseline) / max(baseline, 0.01)) * 100, 1
                        ),
                        "z_score": round(
                            abs(value - baseline) / max(std_dev, 0.01), 2
                        ),
                        "excess_energy_kwh": round(excess * 0.25, 2),
                        "estimated_cost": est_cost,
                        "meter_id": reading.meter_id,
                        "channel_name": reading.channel_name,
                        "status": "open",
                    })

        outputs["anomalies_detected"] = len(self._anomalies)
        outputs["readings_analysed"] = len(input_data.readings)
        outputs["thresholds_applied"] = input_data.energy_type

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 Detect: %d anomalies from %d readings",
            len(self._anomalies), len(input_data.readings),
        )
        return PhaseResult(
            phase_name="detect", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Investigate
    # -------------------------------------------------------------------------

    def _phase_investigate(
        self, input_data: AnomalyResponseInput
    ) -> PhaseResult:
        """Root cause analysis and impact assessment."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        root_cause_map = {
            "spike": [
                "equipment_startup_surge",
                "simultaneous_load_activation",
                "control_system_failure",
                "meter_communication_glitch",
            ],
            "baseline_drift": [
                "equipment_degradation",
                "occupancy_change",
                "weather_deviation",
                "operational_change",
            ],
            "off_hours": [
                "forgotten_equipment",
                "cleaning_crew_usage",
                "scheduled_maintenance",
                "security_systems",
            ],
            "equipment_failure": [
                "motor_failure",
                "compressor_shutdown",
                "hvac_fault",
                "circuit_breaker_trip",
            ],
            "meter_error": [
                "ct_ratio_mismatch",
                "communication_loss",
                "firmware_bug",
                "register_overflow",
            ],
            "seasonal_deviation": [
                "unexpected_weather",
                "heating_cooling_changeover",
                "schedule_not_updated",
            ],
            "load_imbalance": [
                "phase_imbalance",
                "single_phase_overload",
                "neutral_conductor_issue",
            ],
        }

        for anomaly in self._anomalies:
            atype = anomaly.get("anomaly_type", "spike")
            possible_causes = root_cause_map.get(atype, ["unknown"])

            # Select most likely cause based on severity and deviation
            deviation = abs(anomaly.get("deviation_pct", 0))
            if deviation > 100:
                root_cause = possible_causes[0] if possible_causes else "unknown"
                confidence = "high"
            elif deviation > 50:
                root_cause = possible_causes[0] if possible_causes else "unknown"
                confidence = "medium"
            else:
                root_cause = possible_causes[-1] if possible_causes else "unknown"
                confidence = "low"

            # Impact assessment
            excess_kwh = float(anomaly.get("excess_energy_kwh", 0))
            # Annual impact if recurring (assume daily)
            annual_impact_kwh = round(excess_kwh * 365, 1)
            annual_cost_impact = round(annual_impact_kwh * 0.10, 2)

            investigation = {
                "investigation_id": f"inv-{_new_uuid()[:8]}",
                "anomaly_id": anomaly["anomaly_id"],
                "root_cause": root_cause,
                "possible_causes": possible_causes,
                "confidence": confidence,
                "annual_impact_kwh": annual_impact_kwh,
                "annual_cost_impact": annual_cost_impact,
                "priority": anomaly.get("severity", "low"),
                "investigated_at": _utcnow().isoformat() + "Z",
            }
            self._investigations.append(investigation)

        outputs["investigations_completed"] = len(self._investigations)
        outputs["root_causes_identified"] = sum(
            1 for inv in self._investigations if inv.get("root_cause")
        )
        outputs["high_confidence_count"] = sum(
            1 for inv in self._investigations if inv.get("confidence") == "high"
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 Investigate: %d investigations, %d root causes identified",
            len(self._investigations), outputs["root_causes_identified"],
        )
        return PhaseResult(
            phase_name="investigate", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Resolve
    # -------------------------------------------------------------------------

    def _phase_resolve(
        self, input_data: AnomalyResponseInput
    ) -> PhaseResult:
        """Corrective action planning and verification."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        action_templates = {
            "equipment_startup_surge": {
                "action": "Implement soft-start or staggered startup sequence",
                "category": "controls",
                "effort": "medium",
                "timeline_days": 14,
            },
            "simultaneous_load_activation": {
                "action": "Configure load sequencing in BMS",
                "category": "controls",
                "effort": "low",
                "timeline_days": 7,
            },
            "control_system_failure": {
                "action": "Inspect and repair control system; verify setpoints",
                "category": "maintenance",
                "effort": "high",
                "timeline_days": 3,
            },
            "equipment_degradation": {
                "action": "Schedule preventive maintenance; assess replacement",
                "category": "maintenance",
                "effort": "medium",
                "timeline_days": 30,
            },
            "forgotten_equipment": {
                "action": "Implement automated off-hours shutdown schedules",
                "category": "scheduling",
                "effort": "low",
                "timeline_days": 3,
            },
            "occupancy_change": {
                "action": "Update baseline model with new occupancy patterns",
                "category": "analytics",
                "effort": "low",
                "timeline_days": 7,
            },
            "weather_deviation": {
                "action": "Adjust weather-normalized baseline parameters",
                "category": "analytics",
                "effort": "low",
                "timeline_days": 1,
            },
            "motor_failure": {
                "action": "Repair or replace failed motor; verify protection",
                "category": "maintenance",
                "effort": "high",
                "timeline_days": 7,
            },
            "ct_ratio_mismatch": {
                "action": "Verify and correct CT/PT ratios in meter configuration",
                "category": "metering",
                "effort": "low",
                "timeline_days": 1,
            },
            "meter_communication_glitch": {
                "action": "Check communication cables and protocol settings",
                "category": "metering",
                "effort": "low",
                "timeline_days": 1,
            },
            "cleaning_crew_usage": {
                "action": "Review cleaning schedule; install sub-metering",
                "category": "scheduling",
                "effort": "low",
                "timeline_days": 7,
            },
        }

        for investigation in self._investigations:
            root_cause = investigation.get("root_cause", "unknown")
            template = action_templates.get(root_cause, {
                "action": f"Investigate and address: {root_cause}",
                "category": "general",
                "effort": "medium",
                "timeline_days": 14,
            })

            # Determine resolution status based on severity
            severity = investigation.get("priority", "low")
            if severity == "critical":
                status = ResolutionStatus.IN_PROGRESS.value
            elif severity == "high":
                status = ResolutionStatus.OPEN.value
            else:
                status = ResolutionStatus.OPEN.value

            resolution = {
                "resolution_id": f"res-{_new_uuid()[:8]}",
                "investigation_id": investigation["investigation_id"],
                "anomaly_id": investigation["anomaly_id"],
                "root_cause": root_cause,
                "recommended_action": template["action"],
                "category": template["category"],
                "effort": template["effort"],
                "timeline_days": template["timeline_days"],
                "estimated_savings_kwh": investigation.get("annual_impact_kwh", 0),
                "estimated_savings_cost": investigation.get("annual_cost_impact", 0),
                "status": status,
                "assigned_to": "",
                "created_at": _utcnow().isoformat() + "Z",
            }
            self._resolutions.append(resolution)

        # Process active anomalies that may have been resolved
        for active in input_data.active_anomalies:
            if active.get("status") == "resolved":
                self._resolutions.append({
                    "resolution_id": f"res-{_new_uuid()[:8]}",
                    "anomaly_id": active.get("anomaly_id", ""),
                    "status": ResolutionStatus.RESOLVED.value,
                    "resolved_at": _utcnow().isoformat() + "Z",
                })

        outputs["actions_recommended"] = len(self._resolutions)
        outputs["actions_by_category"] = {}
        for res in self._resolutions:
            cat = res.get("category", "general")
            outputs["actions_by_category"][cat] = outputs["actions_by_category"].get(cat, 0) + 1
        outputs["total_potential_savings"] = round(
            sum(float(r.get("estimated_savings_cost", 0)) for r in self._resolutions), 2
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 Resolve: %d actions recommended, potential savings=$%.0f",
            len(self._resolutions), outputs["total_potential_savings"],
        )
        return PhaseResult(
            phase_name="resolve", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _is_off_hours(self, timestamp: str, schedule: Dict[str, Any]) -> bool:
        """Check if timestamp falls outside operating hours."""
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            hour = dt.hour
            weekday = dt.weekday()  # 0=Monday, 6=Sunday

            if weekday >= 5:  # Weekend
                return not schedule.get("weekend_occupied", False)

            start_str = schedule.get("weekday_start", "08:00")
            end_str = schedule.get("weekday_end", "18:00")
            start_hour = int(start_str.split(":")[0])
            end_hour = int(end_str.split(":")[0])

            return hour < start_hour or hour >= end_hour
        except (ValueError, TypeError):
            return False

    def _classify_severity(
        self, value: float, baseline: float, thresholds: Dict[str, Any]
    ) -> str:
        """Classify anomaly severity based on deviation magnitude."""
        if baseline <= 0:
            return AnomalySeverity.LOW.value

        deviation_pct = abs((value - baseline) / baseline) * 100
        spike_pct = thresholds.get("spike_threshold_pct", 30.0)

        if deviation_pct >= spike_pct * 3:
            return AnomalySeverity.CRITICAL.value
        elif deviation_pct >= spike_pct * 2:
            return AnomalySeverity.HIGH.value
        elif deviation_pct >= spike_pct:
            return AnomalySeverity.MEDIUM.value
        return AnomalySeverity.LOW.value

    def _compute_provenance(self, result: AnomalyResponseResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
