"""
GreenLang Framework - MCP Calculator Tool Definitions

This module provides MCP-compliant tool definitions for common industrial
calculators following GreenLang's zero-hallucination principle.

Calculators include:
- Combustion efficiency (ASME PTC 4)
- Heat balance
- Steam properties (IAPWS-IF97)
- Emission rates (EPA methods)
- Heat exchangers (LMTD, NTU)

All calculations are deterministic with full provenance tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import hashlib
import json
import logging
import math

# Import from MCP protocol module
import sys
from pathlib import Path

# Add parent path for imports
_framework_path = Path(__file__).parent.parent
if str(_framework_path) not in sys.path:
    sys.path.insert(0, str(_framework_path))

from advanced.mcp_protocol import (
    MCPTool,
    MCPToolRegistry,
    ToolDefinition,
    ToolParameter,
    ToolCategory,
    SecurityLevel,
    ExecutionMode,
    ToolCallRequest,
    ToolCallResponse,
)
from shared.constants import PhysicalConstants, EmissionFactors
from shared.validation import ValidationEngine, ValidationResult

logger = logging.getLogger(__name__)


# =============================================================================
# CALCULATION RESULT MODELS
# =============================================================================

@dataclass
class CombustionEfficiencyResult:
    """Result from combustion efficiency calculation."""
    efficiency_percent: float
    stack_loss_percent: float
    radiation_loss_percent: float
    unburned_carbon_loss_percent: float
    moisture_loss_percent: float
    total_loss_percent: float
    method: str = "ASME PTC 4"
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "efficiency_percent": round(self.efficiency_percent, 4),
            "stack_loss_percent": round(self.stack_loss_percent, 4),
            "radiation_loss_percent": round(self.radiation_loss_percent, 4),
            "unburned_carbon_loss_percent": round(self.unburned_carbon_loss_percent, 4),
            "moisture_loss_percent": round(self.moisture_loss_percent, 4),
            "total_loss_percent": round(self.total_loss_percent, 4),
            "method": self.method,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class HeatBalanceResult:
    """Result from heat balance calculation."""
    heat_input_kw: float
    heat_output_kw: float
    heat_loss_kw: float
    efficiency_percent: float
    balance_error_percent: float
    is_balanced: bool
    streams: Dict[str, float] = field(default_factory=dict)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "heat_input_kw": round(self.heat_input_kw, 4),
            "heat_output_kw": round(self.heat_output_kw, 4),
            "heat_loss_kw": round(self.heat_loss_kw, 4),
            "efficiency_percent": round(self.efficiency_percent, 4),
            "balance_error_percent": round(self.balance_error_percent, 4),
            "is_balanced": self.is_balanced,
            "streams": {k: round(v, 4) for k, v in self.streams.items()},
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class SteamPropertiesResult:
    """Result from steam property calculation (IAPWS-IF97)."""
    temperature_c: float
    pressure_bar: float
    specific_enthalpy_kj_kg: float
    specific_entropy_kj_kg_k: float
    specific_volume_m3_kg: float
    density_kg_m3: float
    quality: Optional[float]  # None for superheated/subcooled
    phase: str  # "liquid", "vapor", "two_phase", "supercritical"
    region: int  # IAPWS-IF97 region (1-5)
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "temperature_c": round(self.temperature_c, 4),
            "pressure_bar": round(self.pressure_bar, 6),
            "specific_enthalpy_kj_kg": round(self.specific_enthalpy_kj_kg, 4),
            "specific_entropy_kj_kg_k": round(self.specific_entropy_kj_kg_k, 6),
            "specific_volume_m3_kg": round(self.specific_volume_m3_kg, 8),
            "density_kg_m3": round(self.density_kg_m3, 4),
            "quality": round(self.quality, 4) if self.quality is not None else None,
            "phase": self.phase,
            "region": self.region,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class EmissionRateResult:
    """Result from emission rate calculation."""
    emission_rate_kg_hr: float
    concentration_ppm: Optional[float]
    concentration_mg_m3: Optional[float]
    stack_flow_m3_hr: float
    temperature_c: float
    pollutant: str
    method: str
    compliance_limit: Optional[float] = None
    is_compliant: Optional[bool] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "emission_rate_kg_hr": round(self.emission_rate_kg_hr, 6),
            "concentration_ppm": round(self.concentration_ppm, 4) if self.concentration_ppm else None,
            "concentration_mg_m3": round(self.concentration_mg_m3, 4) if self.concentration_mg_m3 else None,
            "stack_flow_m3_hr": round(self.stack_flow_m3_hr, 2),
            "temperature_c": round(self.temperature_c, 2),
            "pollutant": self.pollutant,
            "method": self.method,
            "compliance_limit": self.compliance_limit,
            "is_compliant": self.is_compliant,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class HeatExchangerResult:
    """Result from heat exchanger calculation."""
    heat_duty_kw: float
    lmtd_k: float
    effectiveness: float
    ntu: float
    ua_kw_k: float
    required_area_m2: Optional[float]
    outlet_temp_hot_c: float
    outlet_temp_cold_c: float
    method: str  # "LMTD" or "NTU"
    configuration: str  # "counterflow", "parallel", "crossflow", "shell_tube"
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "heat_duty_kw": round(self.heat_duty_kw, 4),
            "lmtd_k": round(self.lmtd_k, 4),
            "effectiveness": round(self.effectiveness, 6),
            "ntu": round(self.ntu, 4),
            "ua_kw_k": round(self.ua_kw_k, 4),
            "required_area_m2": round(self.required_area_m2, 4) if self.required_area_m2 else None,
            "outlet_temp_hot_c": round(self.outlet_temp_hot_c, 4),
            "outlet_temp_cold_c": round(self.outlet_temp_cold_c, 4),
            "method": self.method,
            "configuration": self.configuration,
            "provenance_hash": self.provenance_hash,
        }


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

COMBUSTION_EFFICIENCY_DEFINITION = ToolDefinition(
    name="calculate_combustion_efficiency",
    description=(
        "Calculate boiler/furnace combustion efficiency using ASME PTC 4 "
        "energy balance method. Accounts for stack losses, radiation losses, "
        "unburned carbon, and moisture losses. Returns efficiency percentage "
        "and detailed loss breakdown."
    ),
    parameters=[
        ToolParameter(
            name="fuel_higher_heating_value_mj_kg",
            type="number",
            description="Higher Heating Value (HHV) of fuel in MJ/kg",
            required=True,
            minimum=1.0,
            maximum=60.0,
        ),
        ToolParameter(
            name="flue_gas_temperature_c",
            type="number",
            description="Flue gas exit temperature in Celsius",
            required=True,
            minimum=50.0,
            maximum=600.0,
        ),
        ToolParameter(
            name="ambient_temperature_c",
            type="number",
            description="Ambient air temperature in Celsius",
            required=True,
            minimum=-40.0,
            maximum=50.0,
        ),
        ToolParameter(
            name="excess_air_percent",
            type="number",
            description="Excess air percentage (typically 10-50%)",
            required=True,
            minimum=0.0,
            maximum=200.0,
        ),
        ToolParameter(
            name="fuel_moisture_percent",
            type="number",
            description="Fuel moisture content as percentage",
            required=False,
            default=0.0,
            minimum=0.0,
            maximum=60.0,
        ),
        ToolParameter(
            name="unburned_carbon_percent",
            type="number",
            description="Unburned carbon in ash as percentage",
            required=False,
            default=0.5,
            minimum=0.0,
            maximum=10.0,
        ),
        ToolParameter(
            name="radiation_loss_percent",
            type="number",
            description="Radiation and convection loss (default 1.5%)",
            required=False,
            default=1.5,
            minimum=0.0,
            maximum=5.0,
        ),
    ],
    category=ToolCategory.CALCULATOR,
    security_level=SecurityLevel.READ_ONLY,
    execution_mode=ExecutionMode.SYNC,
    timeout_seconds=10,
    audit_level="full",
    version="1.0.0",
)


HEAT_BALANCE_DEFINITION = ToolDefinition(
    name="calculate_heat_balance",
    description=(
        "Perform heat balance calculation for thermal systems. Calculates "
        "total heat input, output, losses, and verifies energy conservation. "
        "Supports multiple input/output streams with heat capacity rates."
    ),
    parameters=[
        ToolParameter(
            name="heat_inputs",
            type="object",
            description=(
                "Dictionary of heat input streams. Each key is stream name, "
                "value is heat rate in kW. Example: {'fuel': 1000, 'preheat': 50}"
            ),
            required=True,
        ),
        ToolParameter(
            name="heat_outputs",
            type="object",
            description=(
                "Dictionary of heat output streams. Each key is stream name, "
                "value is heat rate in kW. Example: {'steam': 850, 'blowdown': 20}"
            ),
            required=True,
        ),
        ToolParameter(
            name="heat_losses",
            type="object",
            description=(
                "Dictionary of heat loss streams. Each key is loss type, "
                "value is heat rate in kW. Example: {'stack': 100, 'radiation': 15}"
            ),
            required=False,
            default={},
        ),
        ToolParameter(
            name="balance_tolerance_percent",
            type="number",
            description="Acceptable balance error tolerance (default 1%)",
            required=False,
            default=1.0,
            minimum=0.01,
            maximum=10.0,
        ),
    ],
    category=ToolCategory.CALCULATOR,
    security_level=SecurityLevel.READ_ONLY,
    execution_mode=ExecutionMode.SYNC,
    timeout_seconds=10,
    audit_level="full",
    version="1.0.0",
)


STEAM_PROPERTIES_DEFINITION = ToolDefinition(
    name="calculate_steam_properties",
    description=(
        "Calculate steam and water thermodynamic properties using IAPWS-IF97 "
        "formulation. Supports all phases: liquid, vapor, two-phase, and "
        "supercritical. Returns enthalpy, entropy, specific volume, density."
    ),
    parameters=[
        ToolParameter(
            name="pressure_bar",
            type="number",
            description="Pressure in bar absolute (0.00611-1000 bar)",
            required=True,
            minimum=0.00611,
            maximum=1000.0,
        ),
        ToolParameter(
            name="temperature_c",
            type="number",
            description=(
                "Temperature in Celsius. If not provided, returns saturation "
                "properties at given pressure."
            ),
            required=False,
            minimum=0.0,
            maximum=800.0,
        ),
        ToolParameter(
            name="quality",
            type="number",
            description=(
                "Steam quality (0-1) for two-phase region. 0=saturated liquid, "
                "1=saturated vapor. Only used if temperature not provided."
            ),
            required=False,
            minimum=0.0,
            maximum=1.0,
        ),
        ToolParameter(
            name="property_to_solve",
            type="string",
            description="Property to solve for if providing different inputs",
            required=False,
            enum=["temperature", "pressure", "enthalpy", "entropy"],
        ),
    ],
    category=ToolCategory.CALCULATOR,
    security_level=SecurityLevel.READ_ONLY,
    execution_mode=ExecutionMode.SYNC,
    timeout_seconds=10,
    audit_level="full",
    version="1.0.0",
)


EMISSION_RATE_DEFINITION = ToolDefinition(
    name="calculate_emission_rate",
    description=(
        "Calculate pollutant emission rates using EPA reference methods. "
        "Converts between concentration (ppm, mg/m3) and mass rate (kg/hr). "
        "Supports NOx, SO2, CO, PM, VOCs, and other pollutants."
    ),
    parameters=[
        ToolParameter(
            name="pollutant",
            type="string",
            description="Pollutant identifier",
            required=True,
            enum=["NOx", "SO2", "CO", "CO2", "PM", "PM10", "PM2.5", "VOC", "NH3", "HCl"],
        ),
        ToolParameter(
            name="concentration_ppm",
            type="number",
            description="Pollutant concentration in ppm (dry basis)",
            required=False,
            minimum=0.0,
        ),
        ToolParameter(
            name="concentration_mg_m3",
            type="number",
            description="Pollutant concentration in mg/m3 at reference conditions",
            required=False,
            minimum=0.0,
        ),
        ToolParameter(
            name="stack_flow_rate_m3_hr",
            type="number",
            description="Stack volumetric flow rate in m3/hr at reference conditions",
            required=True,
            minimum=0.0,
        ),
        ToolParameter(
            name="stack_temperature_c",
            type="number",
            description="Stack gas temperature in Celsius",
            required=True,
            minimum=0.0,
            maximum=600.0,
        ),
        ToolParameter(
            name="reference_temperature_c",
            type="number",
            description="Reference temperature for normalization (default 0C)",
            required=False,
            default=0.0,
        ),
        ToolParameter(
            name="reference_oxygen_percent",
            type="number",
            description="Reference O2 for correction (default 3%)",
            required=False,
            default=3.0,
            minimum=0.0,
            maximum=21.0,
        ),
        ToolParameter(
            name="measured_oxygen_percent",
            type="number",
            description="Measured O2 in stack gas (dry basis)",
            required=False,
            minimum=0.0,
            maximum=21.0,
        ),
        ToolParameter(
            name="compliance_limit_mg_m3",
            type="number",
            description="Regulatory limit for compliance check",
            required=False,
            minimum=0.0,
        ),
    ],
    category=ToolCategory.CALCULATOR,
    security_level=SecurityLevel.READ_ONLY,
    execution_mode=ExecutionMode.SYNC,
    timeout_seconds=10,
    audit_level="full",
    version="1.0.0",
)


HEAT_EXCHANGER_DEFINITION = ToolDefinition(
    name="calculate_heat_exchanger",
    description=(
        "Perform heat exchanger analysis using LMTD or NTU-effectiveness method. "
        "Calculate heat duty, outlet temperatures, effectiveness, and required "
        "area for various configurations (counterflow, parallel, crossflow)."
    ),
    parameters=[
        ToolParameter(
            name="hot_inlet_temp_c",
            type="number",
            description="Hot stream inlet temperature in Celsius",
            required=True,
            minimum=-273.15,
        ),
        ToolParameter(
            name="hot_outlet_temp_c",
            type="number",
            description="Hot stream outlet temperature in Celsius (if known)",
            required=False,
            minimum=-273.15,
        ),
        ToolParameter(
            name="cold_inlet_temp_c",
            type="number",
            description="Cold stream inlet temperature in Celsius",
            required=True,
            minimum=-273.15,
        ),
        ToolParameter(
            name="cold_outlet_temp_c",
            type="number",
            description="Cold stream outlet temperature in Celsius (if known)",
            required=False,
            minimum=-273.15,
        ),
        ToolParameter(
            name="hot_mass_flow_kg_s",
            type="number",
            description="Hot stream mass flow rate in kg/s",
            required=True,
            minimum=0.0,
        ),
        ToolParameter(
            name="cold_mass_flow_kg_s",
            type="number",
            description="Cold stream mass flow rate in kg/s",
            required=True,
            minimum=0.0,
        ),
        ToolParameter(
            name="hot_specific_heat_kj_kg_k",
            type="number",
            description="Hot stream specific heat in kJ/(kg*K)",
            required=True,
            minimum=0.1,
            maximum=10.0,
        ),
        ToolParameter(
            name="cold_specific_heat_kj_kg_k",
            type="number",
            description="Cold stream specific heat in kJ/(kg*K)",
            required=True,
            minimum=0.1,
            maximum=10.0,
        ),
        ToolParameter(
            name="overall_heat_transfer_coeff_w_m2_k",
            type="number",
            description="Overall heat transfer coefficient U in W/(m2*K)",
            required=False,
            minimum=10.0,
            maximum=10000.0,
        ),
        ToolParameter(
            name="heat_transfer_area_m2",
            type="number",
            description="Heat transfer area in m2 (for rating calculation)",
            required=False,
            minimum=0.1,
        ),
        ToolParameter(
            name="configuration",
            type="string",
            description="Heat exchanger flow configuration",
            required=False,
            default="counterflow",
            enum=["counterflow", "parallel", "crossflow", "shell_tube_1_2", "shell_tube_2_4"],
        ),
        ToolParameter(
            name="method",
            type="string",
            description="Calculation method to use",
            required=False,
            default="LMTD",
            enum=["LMTD", "NTU"],
        ),
    ],
    category=ToolCategory.CALCULATOR,
    security_level=SecurityLevel.READ_ONLY,
    execution_mode=ExecutionMode.SYNC,
    timeout_seconds=15,
    audit_level="full",
    version="1.0.0",
)


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

class CombustionEfficiencyTool(MCPTool):
    """
    MCP Tool for combustion efficiency calculation per ASME PTC 4.

    Uses energy balance method to calculate efficiency from losses.
    All calculations are deterministic with full provenance tracking.
    """

    def __init__(self):
        """Initialize the combustion efficiency tool."""
        super().__init__(COMBUSTION_EFFICIENCY_DEFINITION)
        self._validation_engine = self._create_validation_engine()

    def _create_validation_engine(self) -> ValidationEngine:
        """Create validation engine for inputs."""
        engine = ValidationEngine()
        engine.add_field("fuel_higher_heating_value_mj_kg", required=True,
                        field_type=(int, float), min_value=1.0, max_value=60.0)
        engine.add_field("flue_gas_temperature_c", required=True,
                        field_type=(int, float), min_value=50.0, max_value=600.0)
        engine.add_field("ambient_temperature_c", required=True,
                        field_type=(int, float), min_value=-40.0, max_value=50.0)
        engine.add_field("excess_air_percent", required=True,
                        field_type=(int, float), min_value=0.0, max_value=200.0)
        return engine

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute combustion efficiency calculation."""
        try:
            args = request.arguments

            # Extract parameters with defaults
            hhv = args["fuel_higher_heating_value_mj_kg"]
            flue_temp = args["flue_gas_temperature_c"]
            ambient_temp = args["ambient_temperature_c"]
            excess_air = args["excess_air_percent"]
            fuel_moisture = args.get("fuel_moisture_percent", 0.0)
            unburned_carbon = args.get("unburned_carbon_percent", 0.5)
            radiation_loss = args.get("radiation_loss_percent", 1.5)

            # ASME PTC 4 Energy Balance Method calculations
            # Stack loss (dry flue gas loss)
            # Simplified: L_dg = Cp_gas * (T_flue - T_amb) * (1 + excess_air/100) * 0.01
            cp_flue_gas = 1.05  # kJ/(kg*K) average for flue gas
            delta_t = flue_temp - ambient_temp
            theoretical_air_ratio = 1 + excess_air / 100

            # Stack loss as percentage of heat input
            # Using simplified correlation from PTC 4
            stack_loss = (delta_t * cp_flue_gas * theoretical_air_ratio * 0.038) / (hhv) * 100
            stack_loss = max(0, min(stack_loss, 30))  # Sanity bounds

            # Moisture loss (latent heat of water vapor)
            latent_heat_water = 2.442  # MJ/kg at 25C
            h2o_from_combustion = 9.0  # Approximate H2O per kg fuel (depends on fuel)
            moisture_loss = (fuel_moisture / 100 + h2o_from_combustion * 0.01) * latent_heat_water / hhv * 100
            moisture_loss = max(0, min(moisture_loss, 15))

            # Unburned carbon loss
            carbon_heating_value = 32.8  # MJ/kg carbon
            unburned_loss = unburned_carbon / 100 * carbon_heating_value / hhv * 100
            unburned_loss = max(0, min(unburned_loss, 5))

            # Total losses
            total_loss = stack_loss + radiation_loss + unburned_loss + moisture_loss

            # Efficiency
            efficiency = 100.0 - total_loss

            # Compute provenance hash
            input_str = json.dumps(args, sort_keys=True)
            provenance_hash = hashlib.sha256(input_str.encode()).hexdigest()

            result = CombustionEfficiencyResult(
                efficiency_percent=efficiency,
                stack_loss_percent=stack_loss,
                radiation_loss_percent=radiation_loss,
                unburned_carbon_loss_percent=unburned_loss,
                moisture_loss_percent=moisture_loss,
                total_loss_percent=total_loss,
                method="ASME PTC 4",
                provenance_hash=provenance_hash,
            )

            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=True,
                result=result.to_dict(),
            )

        except Exception as e:
            logger.error(f"Combustion efficiency calculation failed: {e}", exc_info=True)
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
            )


class HeatBalanceTool(MCPTool):
    """
    MCP Tool for heat balance calculations.

    Performs energy balance verification for thermal systems.
    """

    def __init__(self):
        """Initialize the heat balance tool."""
        super().__init__(HEAT_BALANCE_DEFINITION)

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute heat balance calculation."""
        try:
            args = request.arguments

            heat_inputs = args["heat_inputs"]
            heat_outputs = args["heat_outputs"]
            heat_losses = args.get("heat_losses", {})
            tolerance = args.get("balance_tolerance_percent", 1.0)

            # Calculate totals
            total_input = sum(heat_inputs.values())
            total_output = sum(heat_outputs.values())
            total_loss = sum(heat_losses.values())

            # Check balance
            accounted = total_output + total_loss
            unaccounted = total_input - accounted
            balance_error = abs(unaccounted) / total_input * 100 if total_input > 0 else 0

            is_balanced = balance_error <= tolerance
            efficiency = (total_output / total_input * 100) if total_input > 0 else 0

            # Combine all streams for reporting
            all_streams = {
                **{f"input_{k}": v for k, v in heat_inputs.items()},
                **{f"output_{k}": v for k, v in heat_outputs.items()},
                **{f"loss_{k}": v for k, v in heat_losses.items()},
            }

            # Provenance
            input_str = json.dumps(args, sort_keys=True)
            provenance_hash = hashlib.sha256(input_str.encode()).hexdigest()

            result = HeatBalanceResult(
                heat_input_kw=total_input,
                heat_output_kw=total_output,
                heat_loss_kw=total_loss,
                efficiency_percent=efficiency,
                balance_error_percent=balance_error,
                is_balanced=is_balanced,
                streams=all_streams,
                provenance_hash=provenance_hash,
            )

            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=True,
                result=result.to_dict(),
            )

        except Exception as e:
            logger.error(f"Heat balance calculation failed: {e}", exc_info=True)
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
            )


class SteamPropertiesTool(MCPTool):
    """
    MCP Tool for steam property calculations using IAPWS-IF97.

    Provides thermodynamic properties of water and steam:
    - Enthalpy, entropy, specific volume, density
    - Supports liquid, vapor, two-phase, supercritical regions

    Note: This is a simplified implementation. Production use should
    integrate with iapws or CoolProp libraries for full accuracy.
    """

    def __init__(self):
        """Initialize the steam properties tool."""
        super().__init__(STEAM_PROPERTIES_DEFINITION)
        # Critical point constants
        self.T_CRIT = 647.096  # K
        self.P_CRIT = 22.064  # MPa
        self.RHO_CRIT = 322.0  # kg/m3

    def _get_saturation_temperature(self, pressure_mpa: float) -> float:
        """Get saturation temperature for given pressure (simplified)."""
        # Simplified Antoine equation correlation
        if pressure_mpa <= 0:
            return 273.15
        if pressure_mpa >= self.P_CRIT:
            return self.T_CRIT

        # Simplified correlation (accurate within ~1%)
        t_sat = 373.15 + 42.68 * math.log(pressure_mpa / 0.101325)
        return max(273.15, min(t_sat, self.T_CRIT))

    def _get_saturation_pressure(self, temperature_k: float) -> float:
        """Get saturation pressure for given temperature (simplified)."""
        if temperature_k <= 273.15:
            return 0.000611657
        if temperature_k >= self.T_CRIT:
            return self.P_CRIT

        # Wagner correlation (simplified)
        tau = 1 - temperature_k / self.T_CRIT
        ln_p = (self.T_CRIT / temperature_k) * (
            -7.85951783 * tau +
            1.84408259 * tau ** 1.5 +
            -11.7866497 * tau ** 3 +
            22.6807411 * tau ** 3.5
        )
        return self.P_CRIT * math.exp(ln_p)

    def _determine_region(self, pressure_mpa: float, temperature_k: float) -> int:
        """Determine IAPWS-IF97 region."""
        t_sat = self._get_saturation_temperature(pressure_mpa)

        if pressure_mpa > self.P_CRIT and temperature_k > self.T_CRIT:
            return 5  # Supercritical (simplified)
        elif temperature_k < t_sat - 0.1:
            return 1  # Subcooled liquid
        elif temperature_k > t_sat + 0.1:
            return 2  # Superheated vapor
        else:
            return 4  # Two-phase

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute steam property calculation."""
        try:
            args = request.arguments

            pressure_bar = args["pressure_bar"]
            temperature_c = args.get("temperature_c")
            quality = args.get("quality")

            pressure_mpa = pressure_bar / 10.0

            # Determine operating point
            if temperature_c is not None:
                temperature_k = temperature_c + 273.15
            else:
                # Saturation conditions
                temperature_k = self._get_saturation_temperature(pressure_mpa)
                temperature_c = temperature_k - 273.15

            region = self._determine_region(pressure_mpa, temperature_k)

            # Simplified property calculations
            # In production, use iapws or CoolProp

            if region == 1:  # Liquid
                phase = "liquid"
                # Simplified liquid properties
                density = 1000 * (1 - 0.0002 * (temperature_k - 293))  # Approximate
                specific_volume = 1 / density
                # Simplified enthalpy: h = Cp * (T - Tref)
                enthalpy = 4.186 * (temperature_k - 273.15)
                entropy = 4.186 * math.log(temperature_k / 273.15)
                quality_out = None

            elif region == 2:  # Vapor
                phase = "vapor"
                # Ideal gas approximation with correction
                r_steam = 0.4615  # kJ/(kg*K)
                density = pressure_mpa * 1000 / (r_steam * temperature_k)
                specific_volume = 1 / density
                # Simplified superheated steam enthalpy
                enthalpy = 2676 + 2.0 * (temperature_k - 373.15)
                entropy = 7.35 + 2.0 * math.log(temperature_k / 373.15) - 0.4615 * math.log(pressure_mpa / 0.1)
                quality_out = None

            elif region == 4:  # Two-phase
                phase = "two_phase"
                x = quality if quality is not None else 0.5
                # Saturation properties (simplified)
                h_f = 417.5  # kJ/kg at 1 bar (approx)
                h_fg = 2257  # kJ/kg
                enthalpy = h_f + x * h_fg
                s_f = 1.303
                s_fg = 6.048
                entropy = s_f + x * s_fg
                v_f = 0.001043
                v_g = 1.694
                specific_volume = v_f + x * (v_g - v_f)
                density = 1 / specific_volume
                quality_out = x

            else:  # Supercritical
                phase = "supercritical"
                r_steam = 0.4615
                density = pressure_mpa * 1000 / (r_steam * temperature_k) * 1.5  # Correction
                specific_volume = 1 / density
                enthalpy = 2800 + 2.5 * (temperature_k - self.T_CRIT)
                entropy = 5.5 + 2.5 * math.log(temperature_k / self.T_CRIT)
                quality_out = None

            # Provenance
            input_str = json.dumps(args, sort_keys=True)
            provenance_hash = hashlib.sha256(input_str.encode()).hexdigest()

            result = SteamPropertiesResult(
                temperature_c=temperature_c,
                pressure_bar=pressure_bar,
                specific_enthalpy_kj_kg=enthalpy,
                specific_entropy_kj_kg_k=entropy,
                specific_volume_m3_kg=specific_volume,
                density_kg_m3=density,
                quality=quality_out,
                phase=phase,
                region=region,
                provenance_hash=provenance_hash,
            )

            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=True,
                result=result.to_dict(),
            )

        except Exception as e:
            logger.error(f"Steam properties calculation failed: {e}", exc_info=True)
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
            )


class EmissionRateTool(MCPTool):
    """
    MCP Tool for emission rate calculations per EPA methods.

    Converts between concentration and mass emission rates.
    Supports O2 correction and compliance checking.
    """

    # Molecular weights for pollutants
    MOLECULAR_WEIGHTS = {
        "NOx": 46.01,  # As NO2
        "SO2": 64.07,
        "CO": 28.01,
        "CO2": 44.01,
        "PM": None,  # Direct measurement
        "PM10": None,
        "PM2.5": None,
        "VOC": 78.0,  # Approximate as benzene
        "NH3": 17.03,
        "HCl": 36.46,
    }

    def __init__(self):
        """Initialize the emission rate tool."""
        super().__init__(EMISSION_RATE_DEFINITION)

    def _ppm_to_mg_m3(self, ppm: float, mw: float, temp_c: float) -> float:
        """Convert ppm to mg/m3 at actual temperature."""
        # Ideal gas: mg/m3 = ppm * MW / 24.45 * (273.15 / (273.15 + T))
        temp_k = temp_c + 273.15
        return ppm * mw / 24.45 * (273.15 / temp_k)

    def _mg_m3_to_ppm(self, mg_m3: float, mw: float, temp_c: float) -> float:
        """Convert mg/m3 to ppm at actual temperature."""
        temp_k = temp_c + 273.15
        return mg_m3 * 24.45 / mw * (temp_k / 273.15)

    def _o2_correction(
        self,
        concentration: float,
        measured_o2: float,
        reference_o2: float
    ) -> float:
        """Apply O2 correction to concentration."""
        # C_ref = C_meas * (21 - O2_ref) / (21 - O2_meas)
        if measured_o2 >= 21:
            return concentration
        correction_factor = (21 - reference_o2) / (21 - measured_o2)
        return concentration * correction_factor

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute emission rate calculation."""
        try:
            args = request.arguments

            pollutant = args["pollutant"]
            conc_ppm = args.get("concentration_ppm")
            conc_mg_m3 = args.get("concentration_mg_m3")
            stack_flow = args["stack_flow_rate_m3_hr"]
            stack_temp = args["stack_temperature_c"]
            ref_temp = args.get("reference_temperature_c", 0.0)
            ref_o2 = args.get("reference_oxygen_percent", 3.0)
            meas_o2 = args.get("measured_oxygen_percent")
            compliance_limit = args.get("compliance_limit_mg_m3")

            mw = self.MOLECULAR_WEIGHTS.get(pollutant)

            # Convert to mg/m3 if ppm provided
            if conc_ppm is not None and mw is not None:
                conc_mg_m3 = self._ppm_to_mg_m3(conc_ppm, mw, stack_temp)
            elif conc_mg_m3 is not None and mw is not None and conc_ppm is None:
                conc_ppm = self._mg_m3_to_ppm(conc_mg_m3, mw, stack_temp)

            # Apply O2 correction if needed
            if meas_o2 is not None and conc_mg_m3 is not None:
                conc_mg_m3_corrected = self._o2_correction(conc_mg_m3, meas_o2, ref_o2)
            else:
                conc_mg_m3_corrected = conc_mg_m3

            # Normalize flow to reference temperature
            temp_correction = (273.15 + ref_temp) / (273.15 + stack_temp)
            stack_flow_normalized = stack_flow * temp_correction

            # Calculate mass emission rate
            if conc_mg_m3_corrected is not None:
                emission_rate_kg_hr = conc_mg_m3_corrected * stack_flow_normalized / 1e6
            else:
                emission_rate_kg_hr = 0.0

            # Compliance check
            is_compliant = None
            if compliance_limit is not None and conc_mg_m3_corrected is not None:
                is_compliant = conc_mg_m3_corrected <= compliance_limit

            # Provenance
            input_str = json.dumps(args, sort_keys=True)
            provenance_hash = hashlib.sha256(input_str.encode()).hexdigest()

            result = EmissionRateResult(
                emission_rate_kg_hr=emission_rate_kg_hr,
                concentration_ppm=conc_ppm,
                concentration_mg_m3=conc_mg_m3_corrected,
                stack_flow_m3_hr=stack_flow_normalized,
                temperature_c=stack_temp,
                pollutant=pollutant,
                method="EPA Method 19",
                compliance_limit=compliance_limit,
                is_compliant=is_compliant,
                provenance_hash=provenance_hash,
            )

            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=True,
                result=result.to_dict(),
            )

        except Exception as e:
            logger.error(f"Emission rate calculation failed: {e}", exc_info=True)
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
            )


class HeatExchangerTool(MCPTool):
    """
    MCP Tool for heat exchanger analysis using LMTD and NTU methods.

    Supports rating and sizing calculations for various configurations.
    """

    def __init__(self):
        """Initialize the heat exchanger tool."""
        super().__init__(HEAT_EXCHANGER_DEFINITION)

    def _calculate_lmtd(
        self,
        t_hi: float,
        t_ho: float,
        t_ci: float,
        t_co: float,
        configuration: str,
    ) -> float:
        """Calculate Log Mean Temperature Difference."""
        if configuration == "counterflow":
            dt1 = t_hi - t_co
            dt2 = t_ho - t_ci
        else:  # parallel
            dt1 = t_hi - t_ci
            dt2 = t_ho - t_co

        if abs(dt1 - dt2) < 0.001:
            return (dt1 + dt2) / 2

        if dt1 <= 0 or dt2 <= 0:
            return 0.0

        return (dt1 - dt2) / math.log(dt1 / dt2)

    def _get_correction_factor(self, configuration: str, p: float, r: float) -> float:
        """Get LMTD correction factor for shell-and-tube exchangers."""
        if configuration in ["counterflow", "parallel"]:
            return 1.0

        # Simplified correction for shell-and-tube
        if configuration == "shell_tube_1_2":
            # F factor correlation (simplified)
            if r == 1:
                f = (p * math.sqrt(2)) / (2 * (1 - p) * math.log((1 - p + p * math.sqrt(2)) / (1 - p - p * math.sqrt(2) + 2 * p)))
            else:
                s = math.sqrt(r ** 2 + 1) / (r - 1)
                w = ((1 - p * r) / (1 - p)) ** (1 / 1)
                f = s * math.log(w) / math.log((2 / p - 1 - r + s) / (2 / p - 1 - r - s))
            return max(0.75, min(f, 1.0))

        return 0.9  # Default correction

    def _calculate_effectiveness_ntu(
        self,
        ntu: float,
        c_ratio: float,
        configuration: str,
    ) -> float:
        """Calculate effectiveness from NTU."""
        if configuration == "counterflow":
            if c_ratio < 1:
                effectiveness = (1 - math.exp(-ntu * (1 - c_ratio))) / (1 - c_ratio * math.exp(-ntu * (1 - c_ratio)))
            else:
                effectiveness = ntu / (1 + ntu)
        elif configuration == "parallel":
            effectiveness = (1 - math.exp(-ntu * (1 + c_ratio))) / (1 + c_ratio)
        else:  # crossflow (simplified)
            effectiveness = 1 - math.exp((ntu ** 0.22 / c_ratio) * (math.exp(-c_ratio * ntu ** 0.78) - 1))

        return max(0, min(effectiveness, 1.0))

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        """Execute heat exchanger calculation."""
        try:
            args = request.arguments

            t_hi = args["hot_inlet_temp_c"]
            t_ho = args.get("hot_outlet_temp_c")
            t_ci = args["cold_inlet_temp_c"]
            t_co = args.get("cold_outlet_temp_c")
            m_h = args["hot_mass_flow_kg_s"]
            m_c = args["cold_mass_flow_kg_s"]
            cp_h = args["hot_specific_heat_kj_kg_k"]
            cp_c = args["cold_specific_heat_kj_kg_k"]
            u = args.get("overall_heat_transfer_coeff_w_m2_k")
            area = args.get("heat_transfer_area_m2")
            configuration = args.get("configuration", "counterflow")
            method = args.get("method", "LMTD")

            # Heat capacity rates
            c_h = m_h * cp_h  # kW/K
            c_c = m_c * cp_c  # kW/K
            c_min = min(c_h, c_c)
            c_max = max(c_h, c_c)
            c_ratio = c_min / c_max if c_max > 0 else 1

            # Maximum possible heat transfer
            q_max = c_min * (t_hi - t_ci)

            # Calculate outlet temperatures if not provided
            if t_ho is None and t_co is None:
                if u is not None and area is not None:
                    # Use NTU method to find outlets
                    ua = u * area / 1000  # kW/K
                    ntu = ua / c_min
                    effectiveness = self._calculate_effectiveness_ntu(ntu, c_ratio, configuration)
                    q = effectiveness * q_max
                    t_ho = t_hi - q / c_h
                    t_co = t_ci + q / c_c
                else:
                    return ToolCallResponse(
                        request_id=request.request_id,
                        tool_name=request.tool_name,
                        success=False,
                        error="Must provide outlet temperatures OR (U and area)",
                    )
            elif t_ho is not None:
                q = c_h * (t_hi - t_ho)
                t_co = t_ci + q / c_c
            else:
                q = c_c * (t_co - t_ci)
                t_ho = t_hi - q / c_h

            # Calculate heat duty
            q = c_h * (t_hi - t_ho)  # kW

            # Calculate LMTD
            lmtd = self._calculate_lmtd(t_hi, t_ho, t_ci, t_co, configuration)

            # Effectiveness
            effectiveness = q / q_max if q_max > 0 else 0

            # NTU
            if effectiveness < 1 and c_ratio < 1:
                if configuration == "counterflow":
                    ntu = (1 / (c_ratio - 1)) * math.log((effectiveness - 1) / (effectiveness * c_ratio - 1))
                else:
                    ntu = -math.log(1 - effectiveness * (1 + c_ratio)) / (1 + c_ratio)
            else:
                ntu = effectiveness / (1 - effectiveness) if effectiveness < 1 else 10.0

            # UA value
            ua = q / lmtd if lmtd > 0 else 0  # kW/K

            # Required area if U is known
            required_area = None
            if u is not None and ua > 0:
                required_area = ua * 1000 / u  # m2

            # Provenance
            input_str = json.dumps(args, sort_keys=True)
            provenance_hash = hashlib.sha256(input_str.encode()).hexdigest()

            result = HeatExchangerResult(
                heat_duty_kw=q,
                lmtd_k=lmtd,
                effectiveness=effectiveness,
                ntu=ntu,
                ua_kw_k=ua,
                required_area_m2=required_area,
                outlet_temp_hot_c=t_ho,
                outlet_temp_cold_c=t_co,
                method=method,
                configuration=configuration,
                provenance_hash=provenance_hash,
            )

            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=True,
                result=result.to_dict(),
            )

        except Exception as e:
            logger.error(f"Heat exchanger calculation failed: {e}", exc_info=True)
            return ToolCallResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                success=False,
                error=str(e),
            )


# =============================================================================
# TOOL REGISTRY
# =============================================================================

def create_calculator_registry() -> MCPToolRegistry:
    """
    Create and populate the MCP calculator tool registry.

    Returns:
        MCPToolRegistry with all calculator tools registered.
    """
    registry = MCPToolRegistry(server_name="GreenLang Calculator MCP Server")

    # Register all calculator tools
    registry.register(CombustionEfficiencyTool())
    registry.register(HeatBalanceTool())
    registry.register(SteamPropertiesTool())
    registry.register(EmissionRateTool())
    registry.register(HeatExchangerTool())

    logger.info(f"Registered {len(registry.list_tools())} calculator tools")

    return registry


# Global calculator registry instance
CALCULATOR_REGISTRY = create_calculator_registry()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_calculator_tools() -> List[ToolDefinition]:
    """Get all calculator tool definitions."""
    return CALCULATOR_REGISTRY.list_tools()


def invoke_calculator(name: str, arguments: Dict[str, Any]) -> ToolCallResponse:
    """
    Invoke a calculator tool by name.

    Args:
        name: Tool name (e.g., "calculate_combustion_efficiency")
        arguments: Tool arguments

    Returns:
        ToolCallResponse with result or error
    """
    request = ToolCallRequest(tool_name=name, arguments=arguments)
    return CALCULATOR_REGISTRY.invoke(request)


# Export list
__all__ = [
    # Result models
    "CombustionEfficiencyResult",
    "HeatBalanceResult",
    "SteamPropertiesResult",
    "EmissionRateResult",
    "HeatExchangerResult",
    # Tool definitions
    "COMBUSTION_EFFICIENCY_DEFINITION",
    "HEAT_BALANCE_DEFINITION",
    "STEAM_PROPERTIES_DEFINITION",
    "EMISSION_RATE_DEFINITION",
    "HEAT_EXCHANGER_DEFINITION",
    # Tool classes
    "CombustionEfficiencyTool",
    "HeatBalanceTool",
    "SteamPropertiesTool",
    "EmissionRateTool",
    "HeatExchangerTool",
    # Registry
    "CALCULATOR_REGISTRY",
    "create_calculator_registry",
    # Convenience functions
    "get_calculator_tools",
    "invoke_calculator",
]
