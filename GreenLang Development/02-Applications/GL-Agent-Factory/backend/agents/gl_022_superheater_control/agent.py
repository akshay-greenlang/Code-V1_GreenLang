"""GL-022 SUPERHEAT-CTRL: Superheater Control Agent.

Controls superheated steam temperature for process requirements through
intelligent desuperheater spray control with zero-hallucination calculations.

Standards: ASME PTC 4, IAPWS-IF97
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import (
    SuperheaterInput,
    SuperheaterOutput,
    SprayControlAction,
    ControlParameters,
)
from .formulas import (
    calculate_saturation_temperature,
    calculate_superheat,
    calculate_steam_enthalpy,
    calculate_spray_water_flow,
    calculate_valve_position,
    calculate_pid_parameters,
    calculate_spray_energy_loss,
    calculate_thermal_efficiency_impact,
    generate_calculation_hash,
)

logger = logging.getLogger(__name__)


class SuperheaterControlAgent:
    """
    Superheater temperature control agent with intelligent spray management.

    Features:
    - IAPWS-IF97 compliant steam property calculations
    - PID tuning with Lambda method
    - Energy loss tracking
    - Safety monitoring (tube metal temps)
    - Zero-hallucination deterministic calculations
    """

    AGENT_ID = "GL-022"
    AGENT_NAME = "SUPERHEAT-CTRL"
    VERSION = "1.0.0"

    # Control thresholds
    WARNING_TUBE_MARGIN_C = 50.0
    CRITICAL_TUBE_MARGIN_C = 25.0
    MIN_SUPERHEAT_MARGIN_C = 10.0

    # Efficiency thresholds
    HIGH_SPRAY_EFFICIENCY_IMPACT = 1.0  # % efficiency loss threshold

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the superheater control agent."""
        self.config = config or {}
        self.pid_params = calculate_pid_parameters(
            process_time_constant_s=self.config.get("time_constant", 60),
            process_dead_time_s=self.config.get("dead_time", 10),
            desired_response_time_s=self.config.get("response_time", 120)
        )
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous entry point for the agent.

        Args:
            input_data: Dictionary matching SuperheaterInput schema

        Returns:
            Dictionary matching SuperheaterOutput schema
        """
        validated_input = SuperheaterInput(**input_data)
        output = self._process(validated_input)
        return output.model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Async entry point for the agent."""
        return self.run(input_data)

    def _process(self, input_data: SuperheaterInput) -> SuperheaterOutput:
        """Process input and generate control recommendations."""
        recommendations = []
        warnings = []

        # Calculate saturation temperature and superheat
        t_sat = calculate_saturation_temperature(input_data.steam_pressure_bar)
        current_superheat = calculate_superheat(
            input_data.outlet_steam_temp_c,
            input_data.steam_pressure_bar
        )

        # Check minimum superheat requirement
        if current_superheat < input_data.min_superheat_c:
            warnings.append(
                f"Superheat {current_superheat}°C below minimum {input_data.min_superheat_c}°C"
            )

        # Calculate temperature deviation from target
        temp_deviation = input_data.outlet_steam_temp_c - input_data.target_steam_temp_c
        within_tolerance = abs(temp_deviation) <= input_data.process_temp_tolerance_c

        # Calculate required spray flow
        spray_flow, energy_absorbed = calculate_spray_water_flow(
            steam_flow_kg_s=input_data.steam_flow_kg_s,
            steam_temp_in_c=input_data.outlet_steam_temp_c,
            steam_temp_target_c=input_data.target_steam_temp_c,
            spray_water_temp_c=input_data.spray_water_temp_c,
            steam_pressure_bar=input_data.steam_pressure_bar
        )

        # Determine control action
        if temp_deviation > input_data.process_temp_tolerance_c:
            action_type = "INCREASE"
            target_spray = min(spray_flow, input_data.max_spray_flow_kg_s)
        elif temp_deviation < -input_data.process_temp_tolerance_c:
            action_type = "DECREASE"
            target_spray = max(0, input_data.current_spray_flow_kg_s * 0.8)
        else:
            action_type = "MAINTAIN"
            target_spray = input_data.current_spray_flow_kg_s

        # Check spray flow capacity
        if spray_flow > input_data.max_spray_flow_kg_s:
            warnings.append(
                f"Required spray {spray_flow:.3f} kg/s exceeds capacity "
                f"{input_data.max_spray_flow_kg_s:.3f} kg/s"
            )
            recommendations.append(
                "Consider reducing burner load or checking superheater fouling"
            )

        # Calculate valve position
        valve_position = calculate_valve_position(
            target_spray,
            input_data.max_spray_flow_kg_s
        )

        # Calculate rate of change (prevent thermal shock)
        current_position = input_data.spray_valve_position_pct
        max_rate = 10.0  # % per minute
        if abs(valve_position - current_position) > max_rate:
            rate_of_change = max_rate if valve_position > current_position else -max_rate
        else:
            rate_of_change = valve_position - current_position

        spray_control = SprayControlAction(
            target_spray_flow_kg_s=target_spray,
            valve_position_pct=valve_position,
            rate_of_change_pct_per_min=rate_of_change,
            action_type=action_type
        )

        # Calculate energy metrics
        enthalpy_reduction = calculate_steam_enthalpy(
            input_data.outlet_steam_temp_c,
            input_data.steam_pressure_bar
        ) - calculate_steam_enthalpy(
            input_data.target_steam_temp_c,
            input_data.steam_pressure_bar
        )

        spray_energy_loss = calculate_spray_energy_loss(
            target_spray,
            max(0, enthalpy_reduction)
        )

        # Estimate fuel input for efficiency impact
        # Typical boiler: 90% efficiency, steam enthalpy gain ~2500 kJ/kg
        fuel_input_kw = input_data.steam_flow_kg_s * 2500 / 0.9
        efficiency_impact = calculate_thermal_efficiency_impact(
            spray_energy_loss,
            fuel_input_kw
        )

        if efficiency_impact > self.HIGH_SPRAY_EFFICIENCY_IMPACT:
            warnings.append(
                f"High spray water usage impacting efficiency by {efficiency_impact:.2f}%"
            )
            recommendations.append(
                "Investigate superheater surface fouling or excess firing"
            )

        # Safety checks - tube metal temperature
        tube_metal_margin = input_data.max_tube_metal_temp_c
        if input_data.current_tube_metal_temp_c:
            tube_metal_margin = (
                input_data.max_tube_metal_temp_c -
                input_data.current_tube_metal_temp_c
            )

        if tube_metal_margin < self.CRITICAL_TUBE_MARGIN_C:
            safety_status = "CRITICAL"
            warnings.append(
                f"CRITICAL: Tube metal temp within {tube_metal_margin:.1f}°C of limit!"
            )
            recommendations.append("IMMEDIATELY reduce firing rate")
        elif tube_metal_margin < self.WARNING_TUBE_MARGIN_C:
            safety_status = "WARNING"
            warnings.append(
                f"Tube metal temp within {tube_metal_margin:.1f}°C of limit"
            )
        else:
            safety_status = "SAFE"

        # PID control parameters
        control_params = ControlParameters(
            kp=self.pid_params["kp"],
            ki=self.pid_params["ki"],
            kd=self.pid_params["kd"],
            deadband_c=self.pid_params["deadband_c"],
            max_rate_c_per_min=self.pid_params["max_rate_c_per_min"]
        )

        # Generate recommendations
        if not within_tolerance and action_type == "INCREASE":
            recommendations.append(
                f"Increase spray flow to {target_spray:.3f} kg/s to reduce temperature"
            )
        elif not within_tolerance and action_type == "DECREASE":
            recommendations.append(
                f"Reduce spray flow - temperature {temp_deviation:.1f}°C below target"
            )

        if current_superheat < input_data.min_superheat_c + self.MIN_SUPERHEAT_MARGIN_C:
            recommendations.append(
                f"Superheat margin low - consider reducing spray or increasing firing"
            )

        # Generate provenance hash
        calc_inputs = {
            "steam_temp": input_data.outlet_steam_temp_c,
            "target_temp": input_data.target_steam_temp_c,
            "pressure": input_data.steam_pressure_bar,
            "steam_flow": input_data.steam_flow_kg_s,
            "spray_water_temp": input_data.spray_water_temp_c
        }
        calc_outputs = {
            "spray_flow": target_spray,
            "valve_position": valve_position,
            "t_sat": t_sat,
            "superheat": current_superheat
        }
        calc_hash = generate_calculation_hash(calc_inputs, calc_outputs)

        return SuperheaterOutput(
            spray_control=spray_control,
            control_parameters=control_params,
            current_superheat_c=current_superheat,
            saturation_temp_c=t_sat,
            temperature_deviation_c=round(temp_deviation, 2),
            within_tolerance=within_tolerance,
            spray_energy_loss_kw=spray_energy_loss,
            spray_water_cost_per_hour=spray_energy_loss * 0.05,  # $0.05/kWh equiv
            tube_metal_margin_c=tube_metal_margin,
            safety_status=safety_status,
            thermal_efficiency_impact_pct=efficiency_impact,
            calculation_hash=calc_hash,
            calculation_timestamp=datetime.utcnow(),
            agent_version=self.VERSION,
            recommendations=recommendations,
            warnings=warnings
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Return agent metadata for registry."""
        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "category": "Steam Systems",
            "type": "Controller",
            "complexity": "Medium",
            "standards": ["ASME PTC 4", "IAPWS-IF97"],
            "description": "Controls superheated steam temperature for process requirements",
            "input_schema": SuperheaterInput.model_json_schema(),
            "output_schema": SuperheaterOutput.model_json_schema()
        }
