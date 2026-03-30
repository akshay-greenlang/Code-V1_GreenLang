# -*- coding: utf-8 -*-
"""
Operational Control Workflow - ISO 50001 Clause 8 Operations
===================================

3-phase workflow for establishing operational controls for significant
energy uses within PACK-034 ISO 50001 Energy Management System Pack.

Phases:
    1. CriteriaDefinition   -- Define operating criteria for each SEU
    2. MonitoringSetup       -- Configure monitoring parameters and alert thresholds
    3. DeviationResponse     -- Define deviation response procedures and escalation

The workflow follows GreenLang zero-hallucination principles: operating
criteria are derived from equipment nameplate data and engineering
standards, alert thresholds use statistical process control limits,
and response procedures follow ISO 50001 Clause 8 requirements.
SHA-256 provenance hashes guarantee auditability.

Schedule: on SEU change / annual review
Estimated duration: 25 minutes

Author: GreenLang Team
Version: 34.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from greenlang.schemas.enums import AlertSeverity

logger = logging.getLogger(__name__)


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


class ControlPhase(str, Enum):
    """Phases of the operational control workflow."""

    CRITERIA_DEFINITION = "criteria_definition"
    MONITORING_SETUP = "monitoring_setup"
    DEVIATION_RESPONSE = "deviation_response"


class EscalationLevel(str, Enum):
    """Escalation levels for deviation response."""

    LEVEL_1 = "level_1_operator"
    LEVEL_2 = "level_2_supervisor"
    LEVEL_3 = "level_3_manager"
    LEVEL_4 = "level_4_energy_team"


# =============================================================================
# OPERATING CRITERIA REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Default operating criteria by SEU category
DEFAULT_CRITERIA: Dict[str, Dict[str, Any]] = {
    "hvac": {
        "setpoints": {"cooling_setpoint_c": 24.0, "heating_setpoint_c": 21.0},
        "schedules": {"occupied_start": "07:00", "occupied_end": "19:00"},
        "limits": {"max_simultaneous_heat_cool": False, "min_oa_pct": 15.0},
        "alert_deadband_pct": 5.0,
    },
    "lighting": {
        "setpoints": {"illuminance_lux": 500.0, "daylight_threshold_lux": 300.0},
        "schedules": {"on_time": "06:30", "off_time": "20:00"},
        "limits": {"max_power_density_w_m2": 10.0},
        "alert_deadband_pct": 10.0,
    },
    "motors_drives": {
        "setpoints": {"vfd_min_speed_pct": 20.0, "vfd_max_speed_pct": 100.0},
        "schedules": {"operating_window": "production_hours"},
        "limits": {"max_amps_pct_nameplate": 90.0, "min_power_factor": 0.85},
        "alert_deadband_pct": 5.0,
    },
    "compressed_air": {
        "setpoints": {"discharge_pressure_bar": 7.0, "dewpoint_c": 3.0},
        "schedules": {"operating_window": "production_hours"},
        "limits": {"max_leak_rate_pct": 10.0, "max_pressure_drop_bar": 0.5},
        "alert_deadband_pct": 3.0,
    },
    "process_heat": {
        "setpoints": {"process_temp_c": 180.0, "flue_gas_temp_c": 200.0},
        "schedules": {"operating_window": "production_hours"},
        "limits": {"min_combustion_efficiency_pct": 80.0, "max_excess_air_pct": 20.0},
        "alert_deadband_pct": 5.0,
    },
    "refrigeration": {
        "setpoints": {"evaporator_temp_c": -25.0, "condenser_temp_c": 35.0},
        "schedules": {"defrost_interval_hrs": 8},
        "limits": {"max_subcooling_c": 5.0, "max_superheat_c": 8.0},
        "alert_deadband_pct": 3.0,
    },
}

# Statistical control chart parameters
CONTROL_CHART_SIGMA: float = 3.0  # 3-sigma control limits


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


class OperatingCriterion(BaseModel):
    """Operating criterion for an SEU."""

    criterion_id: str = Field(default_factory=lambda: f"crit-{uuid.uuid4().hex[:8]}")
    seu_id: str = Field(default="", description="Related SEU ID")
    category: str = Field(default="", description="SEU category")
    parameter_name: str = Field(default="", description="Controlled parameter name")
    setpoint_value: Decimal = Field(default=Decimal("0"), description="Target setpoint")
    setpoint_unit: str = Field(default="", description="Unit of measurement")
    upper_limit: Decimal = Field(default=Decimal("0"), description="Upper control limit")
    lower_limit: Decimal = Field(default=Decimal("0"), description="Lower control limit")
    schedule: str = Field(default="", description="Operating schedule")
    source: str = Field(default="iso50001_default", description="Criteria source")


class MonitoringParameter(BaseModel):
    """Monitoring configuration for a parameter."""

    parameter_id: str = Field(default_factory=lambda: f"mon-{uuid.uuid4().hex[:8]}")
    criterion_id: str = Field(default="", description="Related criterion ID")
    parameter_name: str = Field(default="", description="Parameter to monitor")
    measurement_frequency: str = Field(default="15min", description="Sampling interval")
    meter_id: str = Field(default="", description="Associated meter ID")
    alert_threshold_warning: Decimal = Field(default=Decimal("0"))
    alert_threshold_critical: Decimal = Field(default=Decimal("0"))
    ucl: Decimal = Field(default=Decimal("0"), description="Upper control limit (3-sigma)")
    lcl: Decimal = Field(default=Decimal("0"), description="Lower control limit (3-sigma)")
    data_quality_check: bool = Field(default=True)


class ResponseProcedure(BaseModel):
    """Deviation response procedure."""

    procedure_id: str = Field(default_factory=lambda: f"rsp-{uuid.uuid4().hex[:8]}")
    criterion_id: str = Field(default="", description="Related criterion ID")
    deviation_type: str = Field(default="", description="Type of deviation")
    severity: AlertSeverity = Field(default=AlertSeverity.WARNING)
    escalation_level: EscalationLevel = Field(default=EscalationLevel.LEVEL_1)
    response_time_minutes: int = Field(default=30, ge=1)
    corrective_steps: List[str] = Field(default_factory=list)
    requires_investigation: bool = Field(default=False)
    notification_channels: List[str] = Field(default_factory=list)


class OperationalControlInput(BaseModel):
    """Input data model for OperationalControlWorkflow."""

    enms_id: str = Field(default="", description="EnMS program identifier")
    seus: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="SEU data: [{seu_id, category, consumption_kwh, ...}]",
    )
    action_plans: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Action plans from planning phase",
    )
    operating_criteria: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom operating criteria overrides",
    )
    historical_data: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Historical parameter data for control chart calculation",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class OperationalControlResult(BaseModel):
    """Complete result from operational control workflow."""

    control_id: str = Field(..., description="Unique control setup ID")
    enms_id: str = Field(default="", description="EnMS program identifier")
    operating_criteria: List[OperatingCriterion] = Field(default_factory=list)
    monitoring_config: List[MonitoringParameter] = Field(default_factory=list)
    response_procedures: List[ResponseProcedure] = Field(default_factory=list)
    control_effectiveness_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Overall effectiveness score 0-100",
    )
    seus_covered: int = Field(default=0, ge=0)
    total_parameters: int = Field(default=0, ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class OperationalControlWorkflow:
    """
    3-phase operational control workflow per ISO 50001 Clause 8.

    Defines operating criteria for each SEU, configures monitoring
    parameters with statistical control limits, and establishes
    deviation response procedures with escalation paths.

    Zero-hallucination: operating criteria use engineering defaults,
    control limits use 3-sigma statistical process control, and
    response procedures follow ISO 50001 requirements. No LLM calls
    in the numeric computation path.

    Attributes:
        control_id: Unique control setup identifier.
        _criteria: Operating criteria for each SEU.
        _monitoring: Monitoring parameter configurations.
        _responses: Deviation response procedures.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = OperationalControlWorkflow()
        >>> inp = OperationalControlInput(
        ...     enms_id="enms-001",
        ...     seus=[{"seu_id": "seu-1", "category": "hvac", "consumption_kwh": 500000}],
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.control_effectiveness_score > 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize OperationalControlWorkflow."""
        self.control_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._criteria: List[OperatingCriterion] = []
        self._monitoring: List[MonitoringParameter] = []
        self._responses: List[ResponseProcedure] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def execute(self, input_data: OperationalControlInput) -> OperationalControlResult:
        """
        Execute the 3-phase operational control workflow.

        Args:
            input_data: Validated operational control input.

        Returns:
            OperationalControlResult with criteria, monitoring, and response procedures.
        """
        t_start = time.perf_counter()
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting operational control workflow %s enms=%s seus=%d",
            self.control_id, input_data.enms_id, len(input_data.seus),
        )

        self._phase_results = []
        self._criteria = []
        self._monitoring = []
        self._responses = []

        try:
            phase1 = self._phase_criteria_definition(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_monitoring_setup(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_deviation_response(input_data)
            self._phase_results.append(phase3)

        except Exception as exc:
            self.logger.error(
                "Operational control workflow failed: %s", exc, exc_info=True,
            )
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Calculate effectiveness score
        effectiveness = self._calculate_effectiveness_score(input_data)

        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = OperationalControlResult(
            control_id=self.control_id,
            enms_id=input_data.enms_id,
            operating_criteria=self._criteria,
            monitoring_config=self._monitoring,
            response_procedures=self._responses,
            control_effectiveness_score=effectiveness,
            seus_covered=len(set(c.seu_id for c in self._criteria)),
            total_parameters=len(self._monitoring),
            phases_completed=completed_phases,
            execution_time_ms=round(elapsed_ms, 2),
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Operational control workflow %s completed in %.0fms "
            "criteria=%d monitoring=%d responses=%d effectiveness=%.1f",
            self.control_id, elapsed_ms, len(self._criteria),
            len(self._monitoring), len(self._responses), float(effectiveness),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Criteria Definition
    # -------------------------------------------------------------------------

    def _phase_criteria_definition(
        self, input_data: OperationalControlInput
    ) -> PhaseResult:
        """Define operating criteria for each SEU."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for seu_dict in input_data.seus:
            seu_id = seu_dict.get("seu_id", f"seu-{uuid.uuid4().hex[:8]}")
            category = seu_dict.get("category", "other")

            # Get default criteria for category
            defaults = DEFAULT_CRITERIA.get(category, {})
            custom = input_data.operating_criteria.get(category, {})

            # Merge custom overrides with defaults
            setpoints = {**defaults.get("setpoints", {}), **custom.get("setpoints", {})}
            limits = {**defaults.get("limits", {}), **custom.get("limits", {})}
            schedules = defaults.get("schedules", {})
            deadband_pct = defaults.get("alert_deadband_pct", 5.0)

            # Create criteria for each setpoint
            for param_name, setpoint_val in setpoints.items():
                deadband = abs(float(setpoint_val) * deadband_pct / 100.0)
                unit = self._infer_unit(param_name)

                criterion = OperatingCriterion(
                    seu_id=seu_id,
                    category=category,
                    parameter_name=param_name,
                    setpoint_value=Decimal(str(round(float(setpoint_val), 2))),
                    setpoint_unit=unit,
                    upper_limit=Decimal(str(round(float(setpoint_val) + deadband, 2))),
                    lower_limit=Decimal(str(round(float(setpoint_val) - deadband, 2))),
                    schedule=json.dumps(schedules),
                    source="iso50001_default" if param_name not in custom.get("setpoints", {}) else "custom",
                )
                self._criteria.append(criterion)

            # Create criteria for each limit
            for param_name, limit_val in limits.items():
                if isinstance(limit_val, bool):
                    continue  # Skip boolean limits for now
                criterion = OperatingCriterion(
                    seu_id=seu_id,
                    category=category,
                    parameter_name=param_name,
                    setpoint_value=Decimal(str(round(float(limit_val), 2))),
                    setpoint_unit=self._infer_unit(param_name),
                    upper_limit=Decimal(str(round(float(limit_val), 2))),
                    lower_limit=Decimal("0"),
                    schedule="",
                    source="engineering_standard",
                )
                self._criteria.append(criterion)

        outputs["criteria_defined"] = len(self._criteria)
        outputs["seus_processed"] = len(input_data.seus)
        outputs["categories_covered"] = list(set(c.category for c in self._criteria))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 CriteriaDefinition: %d criteria for %d SEUs",
            len(self._criteria), len(input_data.seus),
        )
        return PhaseResult(
            phase_name=ControlPhase.CRITERIA_DEFINITION.value, phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _infer_unit(self, param_name: str) -> str:
        """Infer measurement unit from parameter name."""
        unit_map = {
            "_c": "degC", "_f": "degF", "_bar": "bar", "_psi": "psi",
            "_pct": "%", "_lux": "lux", "_w_m2": "W/m2",
            "_amps": "A", "_hrs": "hours",
        }
        for suffix, unit in unit_map.items():
            if param_name.endswith(suffix):
                return unit
        return "unit"

    # -------------------------------------------------------------------------
    # Phase 2: Monitoring Setup
    # -------------------------------------------------------------------------

    def _phase_monitoring_setup(
        self, input_data: OperationalControlInput
    ) -> PhaseResult:
        """Configure monitoring parameters and alert thresholds."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for criterion in self._criteria:
            # Calculate statistical control limits if historical data available
            hist_key = f"{criterion.category}_{criterion.parameter_name}"
            hist_data = input_data.historical_data.get(hist_key, [])

            ucl, lcl = self._calculate_control_limits(
                float(criterion.setpoint_value), hist_data
            )

            # Set alert thresholds
            deadband = float(criterion.upper_limit - criterion.lower_limit) / 2.0
            warning_threshold = Decimal(str(round(float(criterion.setpoint_value) + deadband * 0.7, 2)))
            critical_threshold = Decimal(str(round(float(criterion.setpoint_value) + deadband * 1.0, 2)))

            # Determine measurement frequency based on category
            freq_map = {
                "hvac": "15min", "lighting": "1hr", "motors_drives": "5min",
                "compressed_air": "5min", "process_heat": "1min", "refrigeration": "10min",
            }
            frequency = freq_map.get(criterion.category, "15min")

            monitoring = MonitoringParameter(
                criterion_id=criterion.criterion_id,
                parameter_name=criterion.parameter_name,
                measurement_frequency=frequency,
                meter_id=f"meter-{criterion.category}-{criterion.parameter_name}",
                alert_threshold_warning=warning_threshold,
                alert_threshold_critical=critical_threshold,
                ucl=Decimal(str(round(ucl, 2))),
                lcl=Decimal(str(round(lcl, 2))),
                data_quality_check=True,
            )
            self._monitoring.append(monitoring)

        outputs["parameters_configured"] = len(self._monitoring)
        outputs["with_historical_limits"] = sum(
            1 for m in self._monitoring if float(m.ucl) != float(m.lcl)
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 MonitoringSetup: %d parameters configured",
            len(self._monitoring),
        )
        return PhaseResult(
            phase_name=ControlPhase.MONITORING_SETUP.value, phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _calculate_control_limits(
        self, setpoint: float, historical_data: List[float]
    ) -> tuple:
        """Calculate 3-sigma control limits from historical data."""
        if len(historical_data) < 10:
            # Insufficient data; use setpoint +/- 10% as fallback
            margin = abs(setpoint) * 0.10
            return setpoint + margin, setpoint - margin

        mean = sum(historical_data) / len(historical_data)
        variance = sum((x - mean) ** 2 for x in historical_data) / (len(historical_data) - 1)
        std_dev = math.sqrt(variance)

        ucl = mean + CONTROL_CHART_SIGMA * std_dev
        lcl = mean - CONTROL_CHART_SIGMA * std_dev

        return ucl, lcl

    # -------------------------------------------------------------------------
    # Phase 3: Deviation Response
    # -------------------------------------------------------------------------

    def _phase_deviation_response(
        self, input_data: OperationalControlInput
    ) -> PhaseResult:
        """Define deviation response procedures and escalation paths."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for criterion in self._criteria:
            # Warning-level response
            warning_response = ResponseProcedure(
                criterion_id=criterion.criterion_id,
                deviation_type=f"{criterion.parameter_name}_warning",
                severity=AlertSeverity.WARNING,
                escalation_level=EscalationLevel.LEVEL_1,
                response_time_minutes=30,
                corrective_steps=[
                    f"Verify {criterion.parameter_name} reading accuracy",
                    f"Check {criterion.category} equipment status",
                    f"Adjust {criterion.parameter_name} back to setpoint {criterion.setpoint_value}",
                    "Log deviation in EnMS register",
                ],
                requires_investigation=False,
                notification_channels=["email", "dashboard"],
            )
            self._responses.append(warning_response)

            # Critical-level response
            critical_response = ResponseProcedure(
                criterion_id=criterion.criterion_id,
                deviation_type=f"{criterion.parameter_name}_critical",
                severity=AlertSeverity.CRITICAL,
                escalation_level=EscalationLevel.LEVEL_2,
                response_time_minutes=15,
                corrective_steps=[
                    f"Immediately investigate {criterion.parameter_name} deviation",
                    f"Check {criterion.category} system for failures",
                    "Activate backup systems if available",
                    f"Restore {criterion.parameter_name} to operating range",
                    "Document incident and root cause",
                    "Review and update operating procedures if needed",
                ],
                requires_investigation=True,
                notification_channels=["email", "sms", "dashboard"],
            )
            self._responses.append(critical_response)

            # Emergency-level response (for safety-critical parameters)
            if criterion.category in ("process_heat", "compressed_air", "refrigeration"):
                emergency_response = ResponseProcedure(
                    criterion_id=criterion.criterion_id,
                    deviation_type=f"{criterion.parameter_name}_emergency",
                    severity=AlertSeverity.EMERGENCY,
                    escalation_level=EscalationLevel.LEVEL_3,
                    response_time_minutes=5,
                    corrective_steps=[
                        "Activate emergency shutdown if safety risk",
                        "Notify facility manager immediately",
                        f"Isolate {criterion.category} system",
                        "Document emergency per ISO 50001 Clause 8.2",
                        "Conduct post-incident review within 24 hours",
                    ],
                    requires_investigation=True,
                    notification_channels=["email", "sms", "phone", "dashboard"],
                )
                self._responses.append(emergency_response)

        outputs["procedures_created"] = len(self._responses)
        outputs["warning_procedures"] = sum(
            1 for r in self._responses if r.severity == AlertSeverity.WARNING
        )
        outputs["critical_procedures"] = sum(
            1 for r in self._responses if r.severity == AlertSeverity.CRITICAL
        )
        outputs["emergency_procedures"] = sum(
            1 for r in self._responses if r.severity == AlertSeverity.EMERGENCY
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 DeviationResponse: %d procedures (W=%d, C=%d, E=%d)",
            len(self._responses),
            outputs["warning_procedures"],
            outputs["critical_procedures"],
            outputs["emergency_procedures"],
        )
        return PhaseResult(
            phase_name=ControlPhase.DEVIATION_RESPONSE.value, phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Supporting Methods
    # -------------------------------------------------------------------------

    def _calculate_effectiveness_score(
        self, input_data: OperationalControlInput
    ) -> Decimal:
        """Calculate overall control effectiveness score (0-100)."""
        total_seus = len(input_data.seus)
        if total_seus == 0:
            return Decimal("0")

        # Scoring components (weighted)
        covered_seus = len(set(c.seu_id for c in self._criteria))
        coverage_score = (covered_seus / total_seus) * 40.0

        # Monitoring completeness
        criteria_with_monitoring = len(set(m.criterion_id for m in self._monitoring))
        monitoring_score = (
            (criteria_with_monitoring / max(len(self._criteria), 1)) * 30.0
        )

        # Response procedure completeness
        criteria_with_response = len(set(r.criterion_id for r in self._responses))
        response_score = (
            (criteria_with_response / max(len(self._criteria), 1)) * 30.0
        )

        total_score = coverage_score + monitoring_score + response_score
        return Decimal(str(round(min(total_score, 100.0), 1)))

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: OperationalControlResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
