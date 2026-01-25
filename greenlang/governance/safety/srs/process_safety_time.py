"""
ProcessSafetyTimeCalculator - Process Safety Time (PST) Calculator

This module implements Process Safety Time calculation per IEC 61511.
PST is the time between a hazardous event initiation and the point
where the consequence becomes inevitable without intervention.

PST is critical for:
- Determining required SIF response time
- Setting appropriate proof test intervals
- Validating SIF design adequacy

Reference: IEC 61511-1 Clause 3.2.54, Clause 10.3.4

Example:
    >>> from greenlang.safety.srs.process_safety_time import ProcessSafetyTimeCalculator
    >>> calc = ProcessSafetyTimeCalculator()
    >>> result = calc.calculate_pst(process_inputs)
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
from enum import Enum
import hashlib
import logging
from datetime import datetime
import math

logger = logging.getLogger(__name__)


class ProcessType(str, Enum):
    """Process types for PST estimation."""

    PRESSURE_BUILDUP = "pressure_buildup"
    TEMPERATURE_RISE = "temperature_rise"
    LEVEL_RISE = "level_rise"
    LEVEL_FALL = "level_fall"
    FLOW_EXCURSION = "flow_excursion"
    COMPOSITION_CHANGE = "composition_change"
    RUNAWAY_REACTION = "runaway_reaction"
    FIRE_ESCALATION = "fire_escalation"
    TOXIC_RELEASE = "toxic_release"


class PSTMethod(str, Enum):
    """PST calculation methods."""

    ANALYTICAL = "analytical"  # Engineering calculation
    SIMULATION = "simulation"  # Process simulation
    EMPIRICAL = "empirical"  # Based on test data
    CONSERVATIVE = "conservative"  # Conservative estimate


class PSTInput(BaseModel):
    """Input parameters for PST calculation."""

    scenario_id: str = Field(
        ...,
        description="Scenario identifier"
    )
    process_type: ProcessType = Field(
        ...,
        description="Type of process deviation"
    )
    method: PSTMethod = Field(
        default=PSTMethod.ANALYTICAL,
        description="Calculation method"
    )
    # Process parameters
    initial_value: float = Field(
        ...,
        description="Initial process variable value"
    )
    trip_setpoint: float = Field(
        ...,
        description="SIF trip setpoint value"
    )
    hazard_threshold: float = Field(
        ...,
        description="Hazardous condition threshold"
    )
    rate_of_change: Optional[float] = Field(
        None,
        description="Rate of change of process variable (per second)"
    )
    # Equipment parameters
    vessel_volume_m3: Optional[float] = Field(
        None,
        gt=0,
        description="Vessel volume (m3)"
    )
    inlet_flow_rate: Optional[float] = Field(
        None,
        ge=0,
        description="Inlet flow rate (units/s)"
    )
    outlet_flow_rate: Optional[float] = Field(
        None,
        ge=0,
        description="Outlet flow rate (units/s)"
    )
    relief_capacity: Optional[float] = Field(
        None,
        ge=0,
        description="Relief device capacity (units/s)"
    )
    # Thermal parameters
    heat_input_kw: Optional[float] = Field(
        None,
        ge=0,
        description="Heat input rate (kW)"
    )
    heat_capacity_kj_k: Optional[float] = Field(
        None,
        gt=0,
        description="Heat capacity (kJ/K)"
    )
    # Safety factors
    safety_factor: float = Field(
        default=0.5,
        gt=0,
        le=1,
        description="Safety factor to apply to calculated PST"
    )
    description: str = Field(
        default="",
        description="Scenario description"
    )


class PSTResult(BaseModel):
    """Result of PST calculation."""

    scenario_id: str = Field(
        ...,
        description="Scenario identifier"
    )
    process_type: ProcessType = Field(
        ...,
        description="Type of process deviation"
    )
    method: PSTMethod = Field(
        ...,
        description="Calculation method used"
    )
    pst_calculated_ms: float = Field(
        ...,
        description="Calculated PST (milliseconds)"
    )
    pst_with_safety_factor_ms: float = Field(
        ...,
        description="PST with safety factor applied (milliseconds)"
    )
    safety_factor: float = Field(
        ...,
        description="Safety factor applied"
    )
    time_to_trip_ms: float = Field(
        ...,
        description="Time from initial condition to trip setpoint (ms)"
    )
    time_trip_to_hazard_ms: float = Field(
        ...,
        description="Time from trip to hazardous condition (ms)"
    )
    recommended_response_time_ms: float = Field(
        ...,
        description="Recommended SIF response time (ms)"
    )
    response_time_margin: float = Field(
        ...,
        description="Margin between recommended response and PST"
    )
    rate_of_change_used: float = Field(
        ...,
        description="Rate of change used in calculation"
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Calculation assumptions"
    )
    calculation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of calculation"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )
    formula_used: str = Field(
        default="",
        description="Formula used for calculation"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessSafetyTimeCalculator:
    """
    Process Safety Time Calculator.

    Calculates PST for various process scenarios per IEC 61511.
    Supports multiple calculation methods:
    - Analytical calculations for common scenarios
    - Rate-based calculations
    - Conservative estimates

    The calculator follows zero-hallucination principles:
    - All calculations are deterministic
    - No LLM involvement in numeric computations
    - Complete audit trail with provenance hashing

    Example:
        >>> calc = ProcessSafetyTimeCalculator()
        >>> input_data = PSTInput(
        ...     scenario_id="PST-001",
        ...     process_type=ProcessType.PRESSURE_BUILDUP,
        ...     initial_value=10.0,  # bar
        ...     trip_setpoint=15.0,  # bar
        ...     hazard_threshold=20.0,  # bar
        ...     rate_of_change=0.1  # bar/s
        ... )
        >>> result = calc.calculate_pst(input_data)
    """

    # Typical response time allocations
    TYPICAL_RESPONSE_ALLOCATIONS = {
        "sensor": 0.1,  # 10% for sensor
        "logic": 0.1,  # 10% for logic solver
        "actuator": 0.5,  # 50% for final element
        "margin": 0.3,  # 30% margin
    }

    def __init__(self):
        """Initialize ProcessSafetyTimeCalculator."""
        logger.info("ProcessSafetyTimeCalculator initialized")

    def calculate_pst(self, input_data: PSTInput) -> PSTResult:
        """
        Calculate Process Safety Time.

        Args:
            input_data: PSTInput with process parameters

        Returns:
            PSTResult with calculated PST

        Raises:
            ValueError: If input parameters are invalid
        """
        logger.info(f"Calculating PST for scenario: {input_data.scenario_id}")

        try:
            # Get appropriate calculation method
            if input_data.method == PSTMethod.ANALYTICAL:
                result = self._calculate_analytical(input_data)
            elif input_data.method == PSTMethod.CONSERVATIVE:
                result = self._calculate_conservative(input_data)
            else:
                # Default to rate-based calculation
                result = self._calculate_rate_based(input_data)

            # Apply safety factor
            pst_with_sf = result["pst_ms"] * input_data.safety_factor

            # Calculate recommended response time
            # Response time should be less than PST with margin
            recommended_response = pst_with_sf * 0.7  # 70% of PST

            # Build result
            pst_result = PSTResult(
                scenario_id=input_data.scenario_id,
                process_type=input_data.process_type,
                method=input_data.method,
                pst_calculated_ms=result["pst_ms"],
                pst_with_safety_factor_ms=pst_with_sf,
                safety_factor=input_data.safety_factor,
                time_to_trip_ms=result["time_to_trip_ms"],
                time_trip_to_hazard_ms=result["time_trip_to_hazard_ms"],
                recommended_response_time_ms=recommended_response,
                response_time_margin=pst_with_sf - recommended_response,
                rate_of_change_used=result["rate_of_change"],
                assumptions=result["assumptions"],
                formula_used=result["formula"],
            )

            # Calculate provenance hash
            pst_result.provenance_hash = self._calculate_provenance(
                input_data, pst_result.pst_calculated_ms
            )

            logger.info(
                f"PST calculated for {input_data.scenario_id}: "
                f"{pst_result.pst_with_safety_factor_ms:.0f}ms"
            )

            return pst_result

        except Exception as e:
            logger.error(f"PST calculation failed: {str(e)}", exc_info=True)
            raise

    def _calculate_rate_based(
        self,
        input_data: PSTInput
    ) -> Dict[str, Any]:
        """
        Calculate PST using rate of change.

        PST = (Hazard_Threshold - Trip_Setpoint) / Rate_of_Change

        Args:
            input_data: PSTInput parameters

        Returns:
            Dict with calculation results
        """
        if input_data.rate_of_change is None or input_data.rate_of_change == 0:
            raise ValueError("Rate of change required for rate-based calculation")

        rate = abs(input_data.rate_of_change)

        # Time from initial to trip
        time_to_trip_s = abs(
            input_data.trip_setpoint - input_data.initial_value
        ) / rate

        # Time from trip to hazard (this is the PST)
        time_trip_to_hazard_s = abs(
            input_data.hazard_threshold - input_data.trip_setpoint
        ) / rate

        # Convert to milliseconds
        time_to_trip_ms = time_to_trip_s * 1000
        pst_ms = time_trip_to_hazard_s * 1000

        return {
            "pst_ms": pst_ms,
            "time_to_trip_ms": time_to_trip_ms,
            "time_trip_to_hazard_ms": pst_ms,
            "rate_of_change": rate,
            "formula": "PST = (Hazard - Trip) / Rate",
            "assumptions": [
                "Constant rate of change assumed",
                "No process dynamics considered",
            ]
        }

    def _calculate_analytical(
        self,
        input_data: PSTInput
    ) -> Dict[str, Any]:
        """
        Calculate PST using analytical method based on process type.

        Args:
            input_data: PSTInput parameters

        Returns:
            Dict with calculation results
        """
        process_type = input_data.process_type

        if process_type == ProcessType.PRESSURE_BUILDUP:
            return self._calc_pressure_pst(input_data)
        elif process_type == ProcessType.TEMPERATURE_RISE:
            return self._calc_temperature_pst(input_data)
        elif process_type in [ProcessType.LEVEL_RISE, ProcessType.LEVEL_FALL]:
            return self._calc_level_pst(input_data)
        else:
            # Fall back to rate-based
            return self._calculate_rate_based(input_data)

    def _calc_pressure_pst(self, input_data: PSTInput) -> Dict[str, Any]:
        """Calculate PST for pressure buildup scenario."""
        assumptions = []

        if input_data.rate_of_change:
            rate = input_data.rate_of_change
            assumptions.append("Using provided rate of change")
        elif input_data.vessel_volume_m3 and input_data.inlet_flow_rate:
            # Estimate rate from flow balance
            net_flow = (input_data.inlet_flow_rate -
                       (input_data.outlet_flow_rate or 0) -
                       (input_data.relief_capacity or 0))
            # Simplified: dP/dt proportional to flow/volume
            rate = net_flow / input_data.vessel_volume_m3
            assumptions.append("Rate estimated from flow balance")
        else:
            raise ValueError(
                "Need rate_of_change or vessel/flow parameters for pressure PST"
            )

        time_to_trip_s = abs(
            input_data.trip_setpoint - input_data.initial_value
        ) / abs(rate)

        pst_s = abs(
            input_data.hazard_threshold - input_data.trip_setpoint
        ) / abs(rate)

        return {
            "pst_ms": pst_s * 1000,
            "time_to_trip_ms": time_to_trip_s * 1000,
            "time_trip_to_hazard_ms": pst_s * 1000,
            "rate_of_change": rate,
            "formula": "PST = (P_hazard - P_trip) / (dP/dt)",
            "assumptions": assumptions + [
                "Assuming gas behavior",
                "No phase change",
            ]
        }

    def _calc_temperature_pst(self, input_data: PSTInput) -> Dict[str, Any]:
        """Calculate PST for temperature rise scenario."""
        assumptions = []

        if input_data.rate_of_change:
            rate = input_data.rate_of_change
            assumptions.append("Using provided rate of change")
        elif input_data.heat_input_kw and input_data.heat_capacity_kj_k:
            # dT/dt = Q / Cp
            rate = input_data.heat_input_kw / input_data.heat_capacity_kj_k
            assumptions.append("Rate calculated from heat balance")
        else:
            raise ValueError(
                "Need rate_of_change or heat parameters for temperature PST"
            )

        time_to_trip_s = abs(
            input_data.trip_setpoint - input_data.initial_value
        ) / abs(rate)

        pst_s = abs(
            input_data.hazard_threshold - input_data.trip_setpoint
        ) / abs(rate)

        return {
            "pst_ms": pst_s * 1000,
            "time_to_trip_ms": time_to_trip_s * 1000,
            "time_trip_to_hazard_ms": pst_s * 1000,
            "rate_of_change": rate,
            "formula": "PST = (T_hazard - T_trip) / (dT/dt) where dT/dt = Q/Cp",
            "assumptions": assumptions + [
                "Constant heat input assumed",
                "No heat loss considered",
            ]
        }

    def _calc_level_pst(self, input_data: PSTInput) -> Dict[str, Any]:
        """Calculate PST for level rise/fall scenario."""
        assumptions = []

        if input_data.rate_of_change:
            rate = input_data.rate_of_change
            assumptions.append("Using provided rate of change")
        elif input_data.vessel_volume_m3 and input_data.inlet_flow_rate:
            # dL/dt proportional to net flow / area
            # Simplified for cylindrical vessel
            net_flow = (input_data.inlet_flow_rate -
                       (input_data.outlet_flow_rate or 0))
            rate = net_flow / (input_data.vessel_volume_m3 ** 0.5)  # Simplified
            assumptions.append("Rate estimated from flow balance")
        else:
            raise ValueError(
                "Need rate_of_change or vessel/flow parameters for level PST"
            )

        time_to_trip_s = abs(
            input_data.trip_setpoint - input_data.initial_value
        ) / abs(rate)

        pst_s = abs(
            input_data.hazard_threshold - input_data.trip_setpoint
        ) / abs(rate)

        return {
            "pst_ms": pst_s * 1000,
            "time_to_trip_ms": time_to_trip_s * 1000,
            "time_trip_to_hazard_ms": pst_s * 1000,
            "rate_of_change": rate,
            "formula": "PST = (L_hazard - L_trip) / (dL/dt)",
            "assumptions": assumptions + [
                "Constant cross-section assumed",
                "No wave effects",
            ]
        }

    def _calculate_conservative(
        self,
        input_data: PSTInput
    ) -> Dict[str, Any]:
        """
        Calculate PST using conservative method.

        Uses worst-case assumptions for safety.

        Args:
            input_data: PSTInput parameters

        Returns:
            Dict with calculation results
        """
        # First calculate rate-based or analytical
        try:
            if input_data.rate_of_change:
                base_result = self._calculate_rate_based(input_data)
            else:
                base_result = self._calculate_analytical(input_data)
        except ValueError:
            # If calculation fails, use very conservative estimate
            # Default PST of 10 seconds for unknown scenarios
            return {
                "pst_ms": 10000,
                "time_to_trip_ms": 5000,
                "time_trip_to_hazard_ms": 10000,
                "rate_of_change": 0,
                "formula": "Conservative default",
                "assumptions": [
                    "Unable to calculate - using conservative default",
                    "10 second PST assumed",
                ]
            }

        # Apply additional conservatism
        conservative_pst = base_result["pst_ms"] * 0.5  # 50% of calculated

        return {
            "pst_ms": conservative_pst,
            "time_to_trip_ms": base_result["time_to_trip_ms"],
            "time_trip_to_hazard_ms": conservative_pst,
            "rate_of_change": base_result["rate_of_change"],
            "formula": base_result["formula"] + " (50% conservative factor)",
            "assumptions": base_result["assumptions"] + [
                "50% conservative factor applied",
                "Worst-case rate assumed",
            ]
        }

    def allocate_response_time(
        self,
        pst_ms: float
    ) -> Dict[str, float]:
        """
        Allocate response time budget based on PST.

        Typical allocation:
        - Sensor: 10% of available time
        - Logic: 10% of available time
        - Actuator: 50% of available time
        - Margin: 30% of available time

        Args:
            pst_ms: Process Safety Time in milliseconds

        Returns:
            Dict with time allocations in milliseconds
        """
        available_time = pst_ms

        allocations = {}
        for component, fraction in self.TYPICAL_RESPONSE_ALLOCATIONS.items():
            allocations[f"{component}_time_ms"] = available_time * fraction

        allocations["total_available_ms"] = pst_ms
        allocations["total_allocated_ms"] = (
            allocations["sensor_time_ms"] +
            allocations["logic_time_ms"] +
            allocations["actuator_time_ms"]
        )

        return allocations

    def validate_response_time(
        self,
        pst_ms: float,
        sensor_time_ms: float,
        logic_time_ms: float,
        actuator_time_ms: float,
        margin_factor: float = 1.5
    ) -> Dict[str, Any]:
        """
        Validate that response time meets PST requirement.

        Per IEC 61511, response time must be less than PST
        with adequate margin.

        Args:
            pst_ms: Process Safety Time
            sensor_time_ms: Sensor response time
            logic_time_ms: Logic solver time
            actuator_time_ms: Actuator stroke time
            margin_factor: Required margin factor (default 1.5x)

        Returns:
            Validation result dictionary
        """
        total_response = sensor_time_ms + logic_time_ms + actuator_time_ms
        required_pst = total_response * margin_factor

        is_valid = required_pst <= pst_ms
        margin_ms = pst_ms - total_response
        margin_percent = (margin_ms / pst_ms) * 100 if pst_ms > 0 else 0

        result = {
            "is_valid": is_valid,
            "total_response_time_ms": total_response,
            "process_safety_time_ms": pst_ms,
            "margin_ms": margin_ms,
            "margin_percent": margin_percent,
            "margin_factor_achieved": pst_ms / total_response if total_response > 0 else float('inf'),
            "margin_factor_required": margin_factor,
            "sensor_time_ms": sensor_time_ms,
            "logic_time_ms": logic_time_ms,
            "actuator_time_ms": actuator_time_ms,
        }

        if not is_valid:
            result["recommendation"] = (
                f"Reduce response time by {required_pst - pst_ms:.0f}ms "
                f"or increase PST margin"
            )
            logger.warning(
                f"Response time validation failed: "
                f"{total_response}ms > {pst_ms}ms / {margin_factor}"
            )

        return result

    def _calculate_provenance(
        self,
        input_data: PSTInput,
        pst_ms: float
    ) -> str:
        """Calculate SHA-256 provenance hash for PST calculation."""
        provenance_str = (
            f"{input_data.scenario_id}|"
            f"{input_data.process_type.value}|"
            f"{input_data.trip_setpoint}|"
            f"{input_data.hazard_threshold}|"
            f"{pst_ms}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
