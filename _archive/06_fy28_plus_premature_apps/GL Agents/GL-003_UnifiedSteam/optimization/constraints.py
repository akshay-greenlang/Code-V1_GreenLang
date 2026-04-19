"""
GL-003 UNIFIEDSTEAM - Constraint Definitions

Provides constraint classes for steam system optimization:
- SafetyConstraints: Pressure, temperature, quality, and rate limits
- EquipmentConstraints: Valve travel, turndown ratios, nozzle limits
- OperationalConstraints: Production requirements, maintenance windows
- UncertaintyConstraints: Operator confirmation requirements

All optimizers must respect these constraints - safety is paramount.
"""

from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator


class ConstraintStatus(str, Enum):
    """Status of a constraint check."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    WARNING = "warning"  # Approaching limit
    UNKNOWN = "unknown"


class ConstraintSeverity(str, Enum):
    """Severity level of constraint."""

    CRITICAL = "critical"  # Safety-critical, never violate
    HIGH = "high"  # Equipment protection
    MEDIUM = "medium"  # Operational preference
    LOW = "low"  # Optimization preference


@dataclass
class ConstraintViolation:
    """Record of a constraint violation."""

    constraint_name: str
    constraint_type: str
    current_value: float
    limit_value: float
    severity: ConstraintSeverity
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""
    recommended_action: str = ""


class ConstraintCheckResult(BaseModel):
    """Result of checking a constraint."""

    constraint_name: str = Field(..., description="Name of the constraint")
    status: ConstraintStatus = Field(..., description="Constraint status")
    severity: ConstraintSeverity = Field(..., description="Constraint severity")
    current_value: float = Field(..., description="Current measured value")
    limit_value: float = Field(..., description="Constraint limit value")
    margin_percent: float = Field(
        default=0.0,
        description="Margin to limit as percentage"
    )
    message: str = Field(default="", description="Descriptive message")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


# =============================================================================
# Safety Constraints
# =============================================================================


class PressureConstraints(BaseModel):
    """Pressure safety constraints for steam systems."""

    # Header pressure limits (psig)
    hp_header_min_psig: float = Field(default=580.0, description="HP header minimum")
    hp_header_max_psig: float = Field(default=620.0, description="HP header maximum")
    hp_header_trip_psig: float = Field(default=650.0, description="HP header trip point")

    mp_header_min_psig: float = Field(default=145.0, description="MP header minimum")
    mp_header_max_psig: float = Field(default=155.0, description="MP header maximum")
    mp_header_trip_psig: float = Field(default=175.0, description="MP header trip point")

    lp_header_min_psig: float = Field(default=13.0, description="LP header minimum")
    lp_header_max_psig: float = Field(default=17.0, description="LP header maximum")
    lp_header_trip_psig: float = Field(default=25.0, description="LP header trip point")

    # Differential pressure limits
    min_prv_delta_psi: float = Field(
        default=15.0,
        description="Minimum pressure drop across PRV for proper operation"
    )
    max_control_valve_delta_psi: float = Field(
        default=100.0,
        description="Maximum pressure drop across control valve"
    )

    # Rate of change limits
    max_pressure_rate_psi_min: float = Field(
        default=10.0,
        description="Maximum pressure rate of change (psi/min)"
    )

    def check_header_pressure(
        self,
        header: str,
        pressure_psig: float,
    ) -> ConstraintCheckResult:
        """
        Check if header pressure is within limits.

        Args:
            header: Header identifier (hp, mp, lp)
            pressure_psig: Current pressure in psig

        Returns:
            Constraint check result
        """
        header = header.lower()

        if header == "hp":
            min_p, max_p, trip_p = (
                self.hp_header_min_psig,
                self.hp_header_max_psig,
                self.hp_header_trip_psig,
            )
        elif header == "mp":
            min_p, max_p, trip_p = (
                self.mp_header_min_psig,
                self.mp_header_max_psig,
                self.mp_header_trip_psig,
            )
        elif header == "lp":
            min_p, max_p, trip_p = (
                self.lp_header_min_psig,
                self.lp_header_max_psig,
                self.lp_header_trip_psig,
            )
        else:
            raise ValueError(f"Unknown header type: {header}")

        # Check against limits
        if pressure_psig >= trip_p:
            status = ConstraintStatus.VIOLATED
            message = f"{header.upper()} header at trip pressure"
            severity = ConstraintSeverity.CRITICAL
        elif pressure_psig > max_p:
            status = ConstraintStatus.WARNING
            message = f"{header.upper()} header above normal operating range"
            severity = ConstraintSeverity.HIGH
        elif pressure_psig < min_p:
            status = ConstraintStatus.WARNING
            message = f"{header.upper()} header below normal operating range"
            severity = ConstraintSeverity.HIGH
        else:
            status = ConstraintStatus.SATISFIED
            message = f"{header.upper()} header pressure within limits"
            severity = ConstraintSeverity.MEDIUM

        # Calculate margin
        if pressure_psig <= (max_p + min_p) / 2:
            margin_to_limit = pressure_psig - min_p
            limit_value = min_p
        else:
            margin_to_limit = max_p - pressure_psig
            limit_value = max_p

        margin_percent = (margin_to_limit / limit_value) * 100 if limit_value > 0 else 0

        return ConstraintCheckResult(
            constraint_name=f"{header}_header_pressure",
            status=status,
            severity=severity,
            current_value=pressure_psig,
            limit_value=limit_value,
            margin_percent=margin_percent,
            message=message,
        )


class TemperatureConstraints(BaseModel):
    """Temperature safety constraints for steam systems."""

    # Superheat limits
    min_superheat_f: float = Field(
        default=50.0,
        description="Minimum superheat above saturation (F)"
    )
    max_superheat_f: float = Field(
        default=200.0,
        description="Maximum superheat above saturation (F)"
    )

    # Desuperheater limits
    min_approach_to_saturation_f: float = Field(
        default=20.0,
        description="Minimum temperature above saturation after desuperheating"
    )
    max_spray_water_delta_t_f: float = Field(
        default=300.0,
        description="Maximum temperature difference for spray water"
    )

    # Thermal shock prevention
    max_temp_rate_f_min: float = Field(
        default=50.0,
        description="Maximum temperature rate of change (F/min)"
    )
    thermal_shock_limit_f: float = Field(
        default=100.0,
        description="Maximum sudden temperature change (F)"
    )

    # Metal temperature limits
    max_pipe_metal_temp_f: float = Field(
        default=1050.0,
        description="Maximum pipe metal temperature (F)"
    )
    min_pipe_metal_temp_f: float = Field(
        default=200.0,
        description="Minimum pipe metal temperature to prevent condensation"
    )

    def check_superheat(
        self,
        steam_temp_f: float,
        saturation_temp_f: float,
    ) -> ConstraintCheckResult:
        """
        Check if superheat is within limits.

        Args:
            steam_temp_f: Current steam temperature (F)
            saturation_temp_f: Saturation temperature at current pressure (F)

        Returns:
            Constraint check result
        """
        superheat = steam_temp_f - saturation_temp_f

        if superheat < self.min_superheat_f:
            status = ConstraintStatus.VIOLATED
            message = "Superheat below minimum - risk of wet steam"
            severity = ConstraintSeverity.CRITICAL
            limit_value = self.min_superheat_f
        elif superheat > self.max_superheat_f:
            status = ConstraintStatus.WARNING
            message = "Superheat above maximum - efficiency loss"
            severity = ConstraintSeverity.MEDIUM
            limit_value = self.max_superheat_f
        else:
            status = ConstraintStatus.SATISFIED
            message = "Superheat within normal range"
            severity = ConstraintSeverity.LOW
            limit_value = self.max_superheat_f

        margin = min(
            superheat - self.min_superheat_f,
            self.max_superheat_f - superheat,
        )
        margin_percent = (margin / (self.max_superheat_f - self.min_superheat_f)) * 100

        return ConstraintCheckResult(
            constraint_name="superheat",
            status=status,
            severity=severity,
            current_value=superheat,
            limit_value=limit_value,
            margin_percent=margin_percent,
            message=message,
        )

    def check_approach_to_saturation(
        self,
        outlet_temp_f: float,
        saturation_temp_f: float,
    ) -> ConstraintCheckResult:
        """
        Check approach to saturation after desuperheating.

        Args:
            outlet_temp_f: Desuperheater outlet temperature (F)
            saturation_temp_f: Saturation temperature (F)

        Returns:
            Constraint check result
        """
        approach = outlet_temp_f - saturation_temp_f

        if approach < self.min_approach_to_saturation_f:
            status = ConstraintStatus.VIOLATED
            message = "Too close to saturation - risk of wet steam"
            severity = ConstraintSeverity.CRITICAL
        elif approach < self.min_approach_to_saturation_f * 1.5:
            status = ConstraintStatus.WARNING
            message = "Approaching saturation limit"
            severity = ConstraintSeverity.HIGH
        else:
            status = ConstraintStatus.SATISFIED
            message = "Safe margin to saturation"
            severity = ConstraintSeverity.LOW

        margin_percent = (
            (approach - self.min_approach_to_saturation_f) /
            self.min_approach_to_saturation_f * 100
        )

        return ConstraintCheckResult(
            constraint_name="approach_to_saturation",
            status=status,
            severity=severity,
            current_value=approach,
            limit_value=self.min_approach_to_saturation_f,
            margin_percent=max(0, margin_percent),
            message=message,
        )


class QualityConstraints(BaseModel):
    """Steam quality constraints."""

    # Steam purity
    min_steam_quality_percent: float = Field(
        default=99.5,
        description="Minimum steam quality (dryness fraction %)"
    )
    max_tds_ppm: float = Field(
        default=0.5,
        description="Maximum total dissolved solids in steam (ppm)"
    )

    # Water quality for spray
    max_spray_water_conductivity_umho: float = Field(
        default=50.0,
        description="Maximum spray water conductivity (umho/cm)"
    )
    min_spray_water_temp_f: float = Field(
        default=60.0,
        description="Minimum spray water temperature (F)"
    )
    max_spray_water_temp_f: float = Field(
        default=250.0,
        description="Maximum spray water temperature (F)"
    )

    # Condensate quality
    min_condensate_purity_percent: float = Field(
        default=95.0,
        description="Minimum condensate purity for return (%)"
    )
    max_condensate_iron_ppb: float = Field(
        default=100.0,
        description="Maximum iron in condensate (ppb)"
    )

    def check_steam_quality(
        self,
        quality_percent: float,
    ) -> ConstraintCheckResult:
        """Check steam quality constraint."""
        if quality_percent < self.min_steam_quality_percent:
            status = ConstraintStatus.VIOLATED
            message = "Steam quality below minimum"
            severity = ConstraintSeverity.CRITICAL
        elif quality_percent < self.min_steam_quality_percent + 0.3:
            status = ConstraintStatus.WARNING
            message = "Steam quality approaching minimum"
            severity = ConstraintSeverity.HIGH
        else:
            status = ConstraintStatus.SATISFIED
            message = "Steam quality acceptable"
            severity = ConstraintSeverity.LOW

        margin_percent = (
            (quality_percent - self.min_steam_quality_percent) /
            (100 - self.min_steam_quality_percent) * 100
        )

        return ConstraintCheckResult(
            constraint_name="steam_quality",
            status=status,
            severity=severity,
            current_value=quality_percent,
            limit_value=self.min_steam_quality_percent,
            margin_percent=max(0, margin_percent),
            message=message,
        )


class RateLimitConstraints(BaseModel):
    """Rate of change constraints for thermal stress prevention."""

    # Pressure rates
    max_pressure_increase_psi_min: float = Field(
        default=10.0,
        description="Maximum pressure increase rate (psi/min)"
    )
    max_pressure_decrease_psi_min: float = Field(
        default=15.0,
        description="Maximum pressure decrease rate (psi/min)"
    )

    # Temperature rates
    max_temp_increase_f_min: float = Field(
        default=50.0,
        description="Maximum temperature increase rate (F/min)"
    )
    max_temp_decrease_f_min: float = Field(
        default=100.0,
        description="Maximum temperature decrease rate (F/min)"
    )

    # Flow rates
    max_flow_increase_percent_min: float = Field(
        default=20.0,
        description="Maximum flow increase rate (%/min)"
    )
    max_flow_decrease_percent_min: float = Field(
        default=30.0,
        description="Maximum flow decrease rate (%/min)"
    )

    # Valve position rates
    max_valve_rate_percent_sec: float = Field(
        default=5.0,
        description="Maximum valve travel rate (%/sec)"
    )

    def check_rate(
        self,
        parameter: str,
        rate: float,
        direction: str = "increase",
    ) -> ConstraintCheckResult:
        """
        Check if rate of change is within limits.

        Args:
            parameter: Parameter name (pressure, temp, flow, valve)
            rate: Current rate of change (absolute value)
            direction: Direction of change (increase/decrease)

        Returns:
            Constraint check result
        """
        rate = abs(rate)
        direction = direction.lower()

        # Get appropriate limit
        if parameter == "pressure":
            if direction == "increase":
                limit = self.max_pressure_increase_psi_min
            else:
                limit = self.max_pressure_decrease_psi_min
        elif parameter in ("temp", "temperature"):
            if direction == "increase":
                limit = self.max_temp_increase_f_min
            else:
                limit = self.max_temp_decrease_f_min
        elif parameter == "flow":
            if direction == "increase":
                limit = self.max_flow_increase_percent_min
            else:
                limit = self.max_flow_decrease_percent_min
        elif parameter == "valve":
            limit = self.max_valve_rate_percent_sec
        else:
            raise ValueError(f"Unknown parameter: {parameter}")

        if rate > limit * 1.2:
            status = ConstraintStatus.VIOLATED
            message = f"{parameter} rate of change exceeds limit"
            severity = ConstraintSeverity.HIGH
        elif rate > limit:
            status = ConstraintStatus.WARNING
            message = f"{parameter} rate approaching limit"
            severity = ConstraintSeverity.MEDIUM
        else:
            status = ConstraintStatus.SATISFIED
            message = f"{parameter} rate within limits"
            severity = ConstraintSeverity.LOW

        margin_percent = (limit - rate) / limit * 100 if limit > 0 else 0

        return ConstraintCheckResult(
            constraint_name=f"{parameter}_rate_{direction}",
            status=status,
            severity=severity,
            current_value=rate,
            limit_value=limit,
            margin_percent=max(0, margin_percent),
            message=message,
        )


class SafetyConstraints(BaseModel):
    """Combined safety constraints for steam systems."""

    pressure: PressureConstraints = Field(default_factory=PressureConstraints)
    temperature: TemperatureConstraints = Field(default_factory=TemperatureConstraints)
    quality: QualityConstraints = Field(default_factory=QualityConstraints)
    rate_limits: RateLimitConstraints = Field(default_factory=RateLimitConstraints)

    def check_all(
        self,
        state: Dict[str, Any],
    ) -> List[ConstraintCheckResult]:
        """
        Check all safety constraints against current state.

        Args:
            state: Current system state dictionary

        Returns:
            List of constraint check results
        """
        results = []

        # Check header pressures
        for header in ["hp", "mp", "lp"]:
            pressure_key = f"{header}_header_pressure_psig"
            if pressure_key in state:
                results.append(
                    self.pressure.check_header_pressure(
                        header, state[pressure_key]
                    )
                )

        # Check superheat
        if "steam_temp_f" in state and "saturation_temp_f" in state:
            results.append(
                self.temperature.check_superheat(
                    state["steam_temp_f"],
                    state["saturation_temp_f"],
                )
            )

        # Check steam quality
        if "steam_quality_percent" in state:
            results.append(
                self.quality.check_steam_quality(state["steam_quality_percent"])
            )

        return results

    def is_safe(self, state: Dict[str, Any]) -> bool:
        """Check if current state satisfies all critical safety constraints."""
        results = self.check_all(state)
        for result in results:
            if (
                result.status == ConstraintStatus.VIOLATED
                and result.severity == ConstraintSeverity.CRITICAL
            ):
                return False
        return True


# =============================================================================
# Equipment Constraints
# =============================================================================


class ValveConstraints(BaseModel):
    """Valve operating constraints."""

    # Position limits
    min_position_percent: float = Field(
        default=5.0,
        description="Minimum valve position (%)"
    )
    max_position_percent: float = Field(
        default=95.0,
        description="Maximum valve position (%)"
    )

    # Characteristic limits
    min_cv_percent: float = Field(
        default=10.0,
        description="Minimum usable Cv range (%)"
    )
    max_cv_percent: float = Field(
        default=100.0,
        description="Maximum Cv (%)"
    )

    # Travel rate
    max_travel_rate_percent_sec: float = Field(
        default=5.0,
        description="Maximum travel rate (%/sec)"
    )

    # Deadband
    deadband_percent: float = Field(
        default=0.5,
        description="Valve deadband (%)"
    )

    # Hysteresis
    hysteresis_percent: float = Field(
        default=1.0,
        description="Valve hysteresis (%)"
    )

    def check_position(
        self,
        position_percent: float,
        valve_id: str = "valve",
    ) -> ConstraintCheckResult:
        """Check valve position constraint."""
        if position_percent < self.min_position_percent:
            status = ConstraintStatus.WARNING
            message = f"{valve_id} near minimum - poor control"
            severity = ConstraintSeverity.MEDIUM
        elif position_percent > self.max_position_percent:
            status = ConstraintStatus.WARNING
            message = f"{valve_id} near maximum - limited capacity"
            severity = ConstraintSeverity.MEDIUM
        else:
            status = ConstraintStatus.SATISFIED
            message = f"{valve_id} position OK"
            severity = ConstraintSeverity.LOW

        # Calculate margin to nearest limit
        margin_to_min = position_percent - self.min_position_percent
        margin_to_max = self.max_position_percent - position_percent
        margin = min(margin_to_min, margin_to_max)
        margin_percent = margin / (
            (self.max_position_percent - self.min_position_percent) / 2
        ) * 100

        return ConstraintCheckResult(
            constraint_name=f"{valve_id}_position",
            status=status,
            severity=severity,
            current_value=position_percent,
            limit_value=(
                self.min_position_percent if margin_to_min < margin_to_max
                else self.max_position_percent
            ),
            margin_percent=max(0, margin_percent),
            message=message,
        )


class NozzleConstraints(BaseModel):
    """Spray nozzle operating constraints."""

    # Pressure differential
    min_nozzle_delta_p_psi: float = Field(
        default=30.0,
        description="Minimum pressure differential across nozzle (psi)"
    )
    max_nozzle_delta_p_psi: float = Field(
        default=150.0,
        description="Maximum pressure differential across nozzle (psi)"
    )

    # Flow range
    min_flow_gpm: float = Field(
        default=1.0,
        description="Minimum spray flow for atomization (gpm)"
    )
    max_flow_gpm: float = Field(
        default=50.0,
        description="Maximum spray flow capacity (gpm)"
    )

    # Turndown
    turndown_ratio: float = Field(
        default=10.0,
        description="Nozzle turndown ratio"
    )

    def check_delta_p(
        self,
        delta_p_psi: float,
        nozzle_id: str = "nozzle",
    ) -> ConstraintCheckResult:
        """Check nozzle pressure differential."""
        if delta_p_psi < self.min_nozzle_delta_p_psi:
            status = ConstraintStatus.VIOLATED
            message = f"{nozzle_id} dP too low - poor atomization"
            severity = ConstraintSeverity.HIGH
        elif delta_p_psi > self.max_nozzle_delta_p_psi:
            status = ConstraintStatus.WARNING
            message = f"{nozzle_id} dP high - excessive wear"
            severity = ConstraintSeverity.MEDIUM
        else:
            status = ConstraintStatus.SATISFIED
            message = f"{nozzle_id} dP OK"
            severity = ConstraintSeverity.LOW

        margin = min(
            delta_p_psi - self.min_nozzle_delta_p_psi,
            self.max_nozzle_delta_p_psi - delta_p_psi,
        )
        margin_percent = margin / (
            (self.max_nozzle_delta_p_psi - self.min_nozzle_delta_p_psi) / 2
        ) * 100

        return ConstraintCheckResult(
            constraint_name=f"{nozzle_id}_delta_p",
            status=status,
            severity=severity,
            current_value=delta_p_psi,
            limit_value=(
                self.min_nozzle_delta_p_psi if delta_p_psi < self.min_nozzle_delta_p_psi
                else self.max_nozzle_delta_p_psi
            ),
            margin_percent=max(0, margin_percent),
            message=message,
        )


class TrapConstraints(BaseModel):
    """Steam trap operating constraints."""

    # Temperature differential
    min_subcooling_f: float = Field(
        default=10.0,
        description="Minimum subcooling for proper trap operation (F)"
    )
    max_subcooling_f: float = Field(
        default=50.0,
        description="Maximum subcooling (indicates trap undersized)"
    )

    # Pressure differential
    min_trap_delta_p_psi: float = Field(
        default=5.0,
        description="Minimum pressure differential across trap (psi)"
    )

    # Capacity
    max_condensate_rate_lb_hr: float = Field(
        default=5000.0,
        description="Maximum condensate handling rate (lb/hr)"
    )
    safety_factor: float = Field(
        default=2.0,
        description="Trap sizing safety factor"
    )


class PumpConstraints(BaseModel):
    """Pump operating constraints."""

    # NPSH
    min_npsh_margin_ft: float = Field(
        default=3.0,
        description="Minimum NPSH margin above required (ft)"
    )

    # Flow range
    min_flow_percent_bep: float = Field(
        default=30.0,
        description="Minimum flow as % of BEP"
    )
    max_flow_percent_bep: float = Field(
        default=120.0,
        description="Maximum flow as % of BEP"
    )

    # Cavitation
    max_suction_specific_speed: float = Field(
        default=11000.0,
        description="Maximum suction specific speed"
    )

    def check_npsh_margin(
        self,
        npsh_available_ft: float,
        npsh_required_ft: float,
        pump_id: str = "pump",
    ) -> ConstraintCheckResult:
        """Check NPSH margin."""
        margin = npsh_available_ft - npsh_required_ft

        if margin < self.min_npsh_margin_ft:
            status = ConstraintStatus.VIOLATED
            message = f"{pump_id} NPSH margin insufficient - cavitation risk"
            severity = ConstraintSeverity.CRITICAL
        elif margin < self.min_npsh_margin_ft * 1.5:
            status = ConstraintStatus.WARNING
            message = f"{pump_id} NPSH margin low"
            severity = ConstraintSeverity.HIGH
        else:
            status = ConstraintStatus.SATISFIED
            message = f"{pump_id} NPSH margin adequate"
            severity = ConstraintSeverity.LOW

        margin_percent = margin / npsh_required_ft * 100 if npsh_required_ft > 0 else 0

        return ConstraintCheckResult(
            constraint_name=f"{pump_id}_npsh_margin",
            status=status,
            severity=severity,
            current_value=margin,
            limit_value=self.min_npsh_margin_ft,
            margin_percent=max(0, margin_percent),
            message=message,
        )


class EquipmentConstraints(BaseModel):
    """Combined equipment constraints."""

    valve: ValveConstraints = Field(default_factory=ValveConstraints)
    nozzle: NozzleConstraints = Field(default_factory=NozzleConstraints)
    trap: TrapConstraints = Field(default_factory=TrapConstraints)
    pump: PumpConstraints = Field(default_factory=PumpConstraints)

    # Turndown ratios
    boiler_turndown: float = Field(
        default=4.0,
        description="Boiler turndown ratio"
    )
    prv_turndown: float = Field(
        default=10.0,
        description="PRV turndown ratio"
    )
    desuperheater_turndown: float = Field(
        default=20.0,
        description="Desuperheater turndown ratio"
    )


# =============================================================================
# Operational Constraints
# =============================================================================


class ProductionConstraints(BaseModel):
    """Production-related constraints."""

    # Steam demand
    min_header_pressure_for_production_psig: float = Field(
        default=140.0,
        description="Minimum header pressure to maintain production"
    )
    max_steam_demand_lb_hr: float = Field(
        default=500000.0,
        description="Maximum steam demand (lb/hr)"
    )

    # Critical users
    critical_users: List[str] = Field(
        default_factory=list,
        description="List of critical steam users that cannot be shed"
    )

    # Load shedding priority
    load_shed_priority: Dict[str, int] = Field(
        default_factory=dict,
        description="Load shedding priority (lower = shed first)"
    )

    # Demand response
    demand_response_available: bool = Field(
        default=False,
        description="Whether demand response is available"
    )
    max_demand_reduction_percent: float = Field(
        default=20.0,
        description="Maximum demand reduction for DR events (%)"
    )


class MaintenanceWindow(BaseModel):
    """Maintenance window definition."""

    window_id: str = Field(..., description="Window identifier")
    equipment_id: str = Field(..., description="Equipment to maintain")
    start_time: datetime = Field(..., description="Window start time")
    end_time: datetime = Field(..., description="Window end time")
    required_redundancy: bool = Field(
        default=True,
        description="Whether redundancy is required during window"
    )


class MaintenanceConstraints(BaseModel):
    """Maintenance-related constraints."""

    # Scheduled windows
    maintenance_windows: List[MaintenanceWindow] = Field(
        default_factory=list,
        description="Scheduled maintenance windows"
    )

    # Minimum equipment availability
    min_boiler_availability: int = Field(
        default=2,
        description="Minimum number of available boilers"
    )
    min_prv_availability: int = Field(
        default=1,
        description="Minimum number of available PRVs per header"
    )

    # Maintenance intervals
    trap_inspection_interval_days: int = Field(
        default=90,
        description="Steam trap inspection interval (days)"
    )
    prv_test_interval_days: int = Field(
        default=365,
        description="PRV testing interval (days)"
    )

    def is_in_maintenance_window(
        self,
        equipment_id: str,
        check_time: Optional[datetime] = None,
    ) -> bool:
        """Check if equipment is currently in maintenance window."""
        if check_time is None:
            check_time = datetime.now(timezone.utc)

        for window in self.maintenance_windows:
            if (
                window.equipment_id == equipment_id
                and window.start_time <= check_time <= window.end_time
            ):
                return True
        return False


class OperationalConstraints(BaseModel):
    """Combined operational constraints."""

    production: ProductionConstraints = Field(default_factory=ProductionConstraints)
    maintenance: MaintenanceConstraints = Field(default_factory=MaintenanceConstraints)

    # Operating modes
    allowed_modes: List[str] = Field(
        default_factory=lambda: ["normal", "reduced", "startup", "shutdown"],
        description="Allowed operating modes"
    )
    current_mode: str = Field(default="normal", description="Current operating mode")

    # Time-based constraints
    peak_hours_start: dt_time = Field(
        default=dt_time(6, 0),
        description="Peak hours start time"
    )
    peak_hours_end: dt_time = Field(
        default=dt_time(18, 0),
        description="Peak hours end time"
    )

    # Cost constraints
    max_fuel_cost_per_hr: float = Field(
        default=10000.0,
        description="Maximum fuel cost ($/hr)"
    )
    max_water_cost_per_hr: float = Field(
        default=500.0,
        description="Maximum water cost ($/hr)"
    )


# =============================================================================
# Uncertainty Constraints
# =============================================================================


class UncertaintyConstraints(BaseModel):
    """Constraints related to uncertainty and operator confirmation."""

    # Confidence thresholds
    min_confidence_for_auto_action: float = Field(
        default=0.90,
        description="Minimum confidence for automatic action"
    )
    min_confidence_for_recommendation: float = Field(
        default=0.70,
        description="Minimum confidence for making recommendation"
    )

    # Uncertainty thresholds
    max_uncertainty_for_auto_action: float = Field(
        default=0.10,
        description="Maximum uncertainty (std) for automatic action"
    )
    max_uncertainty_for_recommendation: float = Field(
        default=0.25,
        description="Maximum uncertainty for making recommendation"
    )

    # Operator confirmation requirements
    require_confirmation_for_critical: bool = Field(
        default=True,
        description="Require operator confirmation for critical changes"
    )
    require_confirmation_for_large_changes: bool = Field(
        default=True,
        description="Require confirmation for changes above threshold"
    )
    large_change_threshold_percent: float = Field(
        default=20.0,
        description="Threshold for 'large' change requiring confirmation (%)"
    )

    # Escalation
    escalation_timeout_minutes: int = Field(
        default=15,
        description="Time before escalating unconfirmed recommendation"
    )
    max_escalation_level: int = Field(
        default=3,
        description="Maximum escalation level"
    )

    def requires_confirmation(
        self,
        confidence: float,
        change_magnitude_percent: float,
        is_critical: bool = False,
    ) -> Tuple[bool, str]:
        """
        Determine if action requires operator confirmation.

        Args:
            confidence: Confidence level (0-1)
            change_magnitude_percent: Size of proposed change (%)
            is_critical: Whether this affects critical equipment

        Returns:
            Tuple of (requires_confirmation, reason)
        """
        reasons = []

        if is_critical and self.require_confirmation_for_critical:
            reasons.append("critical equipment affected")

        if (
            self.require_confirmation_for_large_changes
            and change_magnitude_percent > self.large_change_threshold_percent
        ):
            reasons.append(
                f"change magnitude {change_magnitude_percent:.1f}% "
                f"exceeds threshold {self.large_change_threshold_percent:.1f}%"
            )

        if confidence < self.min_confidence_for_auto_action:
            reasons.append(
                f"confidence {confidence:.2f} below "
                f"auto-action threshold {self.min_confidence_for_auto_action:.2f}"
            )

        if reasons:
            return True, "; ".join(reasons)
        return False, "meets all auto-action criteria"

    def can_make_recommendation(
        self,
        confidence: float,
        uncertainty: float,
    ) -> Tuple[bool, str]:
        """
        Determine if recommendation can be made given uncertainty.

        Args:
            confidence: Confidence level (0-1)
            uncertainty: Uncertainty (standard deviation)

        Returns:
            Tuple of (can_recommend, reason)
        """
        if confidence < self.min_confidence_for_recommendation:
            return False, (
                f"confidence {confidence:.2f} below minimum "
                f"{self.min_confidence_for_recommendation:.2f}"
            )

        if uncertainty > self.max_uncertainty_for_recommendation:
            return False, (
                f"uncertainty {uncertainty:.2f} exceeds maximum "
                f"{self.max_uncertainty_for_recommendation:.2f}"
            )

        return True, "meets recommendation criteria"


# =============================================================================
# Master Constraint Container
# =============================================================================


class SteamSystemConstraints(BaseModel):
    """Master container for all steam system constraints."""

    safety: SafetyConstraints = Field(default_factory=SafetyConstraints)
    equipment: EquipmentConstraints = Field(default_factory=EquipmentConstraints)
    operational: OperationalConstraints = Field(default_factory=OperationalConstraints)
    uncertainty: UncertaintyConstraints = Field(default_factory=UncertaintyConstraints)

    def validate_action(
        self,
        action: Dict[str, Any],
        current_state: Dict[str, Any],
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """
        Validate proposed action against all constraints.

        Args:
            action: Proposed action dictionary
            current_state: Current system state

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []

        # Check safety constraints
        safety_results = self.safety.check_all(current_state)
        for result in safety_results:
            if (
                result.status == ConstraintStatus.VIOLATED
                and result.severity == ConstraintSeverity.CRITICAL
            ):
                violations.append(
                    ConstraintViolation(
                        constraint_name=result.constraint_name,
                        constraint_type="safety",
                        current_value=result.current_value,
                        limit_value=result.limit_value,
                        severity=ConstraintSeverity.CRITICAL,
                        message=result.message,
                        recommended_action="Do not proceed with action",
                    )
                )

        # Check uncertainty constraints
        if "confidence" in action:
            requires_confirm, reason = self.uncertainty.requires_confirmation(
                confidence=action.get("confidence", 0),
                change_magnitude_percent=action.get("change_magnitude_percent", 0),
                is_critical=action.get("is_critical", False),
            )
            if requires_confirm:
                violations.append(
                    ConstraintViolation(
                        constraint_name="operator_confirmation",
                        constraint_type="uncertainty",
                        current_value=action.get("confidence", 0),
                        limit_value=self.uncertainty.min_confidence_for_auto_action,
                        severity=ConstraintSeverity.MEDIUM,
                        message=reason,
                        recommended_action="Request operator confirmation",
                    )
                )

        is_valid = not any(
            v.severity == ConstraintSeverity.CRITICAL for v in violations
        )

        return is_valid, violations
