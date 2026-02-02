"""GL-029 FUELCONDITIONER: Fuel Gas Conditioning Agent.

Controls fuel gas conditioning for optimal combustion including
pressure regulation, temperature control, and composition monitoring.

Standards: NFPA 85, API 556
"""
import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FuelGasInput(BaseModel):
    """Input for fuel gas conditioning."""

    equipment_id: str = Field(..., description="Equipment identifier")

    # Fuel gas supply
    supply_pressure_bar: float = Field(..., ge=0, description="Supply pressure (bar)")
    supply_temp_c: float = Field(..., description="Supply temperature (°C)")
    supply_flow_kg_s: float = Field(..., ge=0, description="Mass flow (kg/s)")

    # Fuel composition (mol %)
    methane_pct: float = Field(90.0, ge=0, le=100)
    ethane_pct: float = Field(5.0, ge=0, le=100)
    propane_pct: float = Field(2.0, ge=0, le=100)
    butane_pct: float = Field(1.0, ge=0, le=100)
    nitrogen_pct: float = Field(1.0, ge=0, le=100)
    co2_pct: float = Field(1.0, ge=0, le=100)
    h2s_ppm: float = Field(0.0, ge=0, description="H2S content (ppm)")
    moisture_ppm: float = Field(0.0, ge=0, description="Moisture content (ppm)")

    # Burner requirements
    required_pressure_bar: float = Field(..., ge=0, description="Required burner pressure")
    required_temp_min_c: float = Field(10, description="Minimum fuel temp (°C)")
    required_temp_max_c: float = Field(60, description="Maximum fuel temp (°C)")
    max_wobbe_variation_pct: float = Field(5.0, description="Max Wobbe index variation (%)")

    # Conditioning equipment status
    heater_available: bool = Field(True, description="Fuel heater available")
    heater_capacity_kw: float = Field(100, description="Heater capacity (kW)")
    pressure_regulator_status: str = Field("NORMAL", description="Regulator status")
    filter_dp_bar: float = Field(0.1, ge=0, description="Filter differential pressure")
    filter_dp_limit_bar: float = Field(0.5, description="Filter ΔP limit")

    # Ambient conditions
    ambient_temp_c: float = Field(20, description="Ambient temperature (°C)")
    dew_point_c: Optional[float] = Field(None, description="Fuel dew point if known")

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FuelGasOutput(BaseModel):
    """Output from fuel gas conditioning agent."""

    # Fuel properties
    higher_heating_value_mj_m3: float = Field(..., description="HHV (MJ/Nm³)")
    lower_heating_value_mj_m3: float = Field(..., description="LHV (MJ/Nm³)")
    wobbe_index_mj_m3: float = Field(..., description="Wobbe index (MJ/Nm³)")
    specific_gravity: float = Field(..., description="SG relative to air")
    calculated_dew_point_c: float = Field(..., description="Hydrocarbon dew point (°C)")

    # Conditioning recommendations
    target_pressure_bar: float = Field(..., description="Target delivery pressure")
    target_temp_c: float = Field(..., description="Target delivery temperature")
    heating_required_kw: float = Field(..., description="Heating power required")
    pressure_reduction_bar: float = Field(..., description="Pressure drop needed")

    # Quality assessment
    quality_status: str = Field(..., description="GOOD, MARGINAL, OUT_OF_SPEC")
    wobbe_deviation_pct: float = Field(..., description="Wobbe deviation from baseline")
    contaminant_status: str = Field(..., description="CLEAN, WARNING, HIGH")

    # JT cooling check
    jt_cooling_c: float = Field(..., description="Joule-Thomson cooling estimate (°C)")
    condensation_risk: str = Field(..., description="NONE, LOW, MEDIUM, HIGH")

    # Equipment status
    filter_status: str = Field(..., description="OK, ATTENTION, REPLACE")
    regulator_status: str = Field(..., description="Equipment status")

    # Combustion impact
    air_fuel_ratio_change_pct: float = Field(..., description="A/F ratio adjustment needed")
    flame_temp_impact_c: float = Field(..., description="Estimated flame temp change")

    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    calculation_hash: str
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


def calculate_heating_value(composition: Dict[str, float]) -> tuple:
    """
    Calculate fuel gas heating values from composition.

    HHV/LHV in MJ/Nm³ at standard conditions.
    """
    # Component heating values (MJ/Nm³) at 15°C, 1 atm
    hhv_values = {
        "methane": 39.82,
        "ethane": 70.29,
        "propane": 101.24,
        "butane": 134.06,
        "nitrogen": 0,
        "co2": 0
    }

    lhv_values = {
        "methane": 35.88,
        "ethane": 64.35,
        "propane": 93.18,
        "butane": 123.81,
        "nitrogen": 0,
        "co2": 0
    }

    hhv = sum(composition.get(comp, 0) / 100 * hv for comp, hv in hhv_values.items())
    lhv = sum(composition.get(comp, 0) / 100 * hv for comp, hv in lhv_values.items())

    return round(hhv, 2), round(lhv, 2)


def calculate_specific_gravity(composition: Dict[str, float]) -> float:
    """
    Calculate specific gravity relative to air.

    SG = Σ(yi × Mi) / M_air
    """
    molecular_weights = {
        "methane": 16.04,
        "ethane": 30.07,
        "propane": 44.10,
        "butane": 58.12,
        "nitrogen": 28.01,
        "co2": 44.01
    }

    m_air = 28.97
    m_gas = sum(composition.get(comp, 0) / 100 * mw for comp, mw in molecular_weights.items())

    return round(m_gas / m_air, 4)


def calculate_wobbe_index(hhv: float, sg: float) -> float:
    """
    Calculate Wobbe Index.

    WI = HHV / √SG

    Wobbe index ensures interchangeability of fuel gases.
    """
    if sg <= 0:
        return 0.0
    import math
    return round(hhv / math.sqrt(sg), 2)


def estimate_dew_point(composition: Dict[str, float], pressure_bar: float) -> float:
    """
    Estimate hydrocarbon dew point (simplified).

    Higher C3+ content = higher dew point.
    """
    c3_plus = composition.get("propane", 0) + composition.get("butane", 0)

    # Simple correlation (actual requires HYSYS/proper EOS)
    base_dp = -40  # Pure methane
    dp = base_dp + c3_plus * 2  # Each % C3+ raises dew point ~2°C

    # Pressure effect: higher pressure raises dew point
    import math
    dp += 10 * math.log10(max(1, pressure_bar))

    return round(dp, 1)


def calculate_jt_cooling(
    pressure_drop_bar: float,
    inlet_temp_c: float,
    sg: float
) -> float:
    """
    Estimate Joule-Thomson cooling effect.

    For natural gas: μJT ≈ 0.4-0.6 °C/bar at typical conditions
    """
    # JT coefficient varies with temperature and composition
    mu_jt = 0.5  # °C/bar typical

    cooling = mu_jt * pressure_drop_bar
    return round(cooling, 1)


class FuelGasConditioningAgent:
    """Fuel gas conditioning controller."""

    AGENT_ID = "GL-029"
    AGENT_NAME = "FUELCONDITIONER"
    VERSION = "1.0.0"

    # Reference Wobbe index for natural gas
    REFERENCE_WOBBE = 50.0  # MJ/Nm³

    # Contaminant limits
    H2S_LIMIT_PPM = 20
    MOISTURE_LIMIT_PPM = 100

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = FuelGasInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: FuelGasInput) -> FuelGasOutput:
        recommendations = []
        warnings = []

        # Build composition dict
        composition = {
            "methane": inp.methane_pct,
            "ethane": inp.ethane_pct,
            "propane": inp.propane_pct,
            "butane": inp.butane_pct,
            "nitrogen": inp.nitrogen_pct,
            "co2": inp.co2_pct
        }

        # Calculate fuel properties
        hhv, lhv = calculate_heating_value(composition)
        sg = calculate_specific_gravity(composition)
        wobbe = calculate_wobbe_index(hhv, sg)

        # Dew point
        if inp.dew_point_c is not None:
            dew_point = inp.dew_point_c
        else:
            dew_point = estimate_dew_point(composition, inp.supply_pressure_bar)

        # Wobbe index deviation
        wobbe_deviation = abs(wobbe - self.REFERENCE_WOBBE) / self.REFERENCE_WOBBE * 100

        if wobbe_deviation > inp.max_wobbe_variation_pct:
            warnings.append(f"Wobbe index {wobbe:.1f} deviates {wobbe_deviation:.1f}% from reference")
            quality_status = "OUT_OF_SPEC"
        elif wobbe_deviation > inp.max_wobbe_variation_pct * 0.7:
            quality_status = "MARGINAL"
        else:
            quality_status = "GOOD"

        # Contaminant check
        if inp.h2s_ppm > self.H2S_LIMIT_PPM:
            contaminant_status = "HIGH"
            warnings.append(f"H2S {inp.h2s_ppm:.0f} ppm exceeds limit {self.H2S_LIMIT_PPM}")
        elif inp.h2s_ppm > self.H2S_LIMIT_PPM * 0.7 or inp.moisture_ppm > self.MOISTURE_LIMIT_PPM * 0.7:
            contaminant_status = "WARNING"
        else:
            contaminant_status = "CLEAN"

        # Pressure conditioning
        pressure_drop = inp.supply_pressure_bar - inp.required_pressure_bar
        target_pressure = inp.required_pressure_bar

        # JT cooling estimate
        jt_cooling = calculate_jt_cooling(pressure_drop, inp.supply_temp_c, sg)
        final_temp_without_heating = inp.supply_temp_c - jt_cooling

        # Condensation risk
        condensation_margin = final_temp_without_heating - dew_point
        if condensation_margin < 5:
            condensation_risk = "HIGH"
            warnings.append("High condensation risk - preheat fuel gas")
        elif condensation_margin < 15:
            condensation_risk = "MEDIUM"
        elif condensation_margin < 30:
            condensation_risk = "LOW"
        else:
            condensation_risk = "NONE"

        # Temperature conditioning
        target_temp = max(inp.required_temp_min_c, dew_point + 15)
        target_temp = min(target_temp, inp.required_temp_max_c)

        if final_temp_without_heating < target_temp:
            heating_required = inp.supply_flow_kg_s * 2.2 * (target_temp - final_temp_without_heating)
        else:
            heating_required = 0

        if heating_required > inp.heater_capacity_kw:
            warnings.append(f"Heating required ({heating_required:.0f} kW) exceeds capacity ({inp.heater_capacity_kw:.0f} kW)")

        # Filter status
        filter_ratio = inp.filter_dp_bar / inp.filter_dp_limit_bar
        if filter_ratio > 0.9:
            filter_status = "REPLACE"
            warnings.append("Filter ΔP critical - replace immediately")
        elif filter_ratio > 0.7:
            filter_status = "ATTENTION"
            recommendations.append("Schedule filter replacement")
        else:
            filter_status = "OK"

        # Air-fuel ratio impact
        # Wobbe change affects heat input at same valve position
        af_ratio_change = wobbe_deviation * 0.5  # Simplified

        # Flame temp impact
        # Higher heating value = higher flame temp
        flame_temp_impact = (wobbe - self.REFERENCE_WOBBE) * 10  # ~10°C per MJ/Nm³

        # Recommendations
        if heating_required > 0:
            recommendations.append(f"Enable fuel heater - {heating_required:.0f} kW required")

        if jt_cooling > 10:
            recommendations.append(f"Significant JT cooling ({jt_cooling:.0f}°C) - monitor for condensation")

        if wobbe_deviation > 3:
            recommendations.append(f"Wobbe deviation {wobbe_deviation:.1f}% - may need burner adjustment")

        if inp.pressure_regulator_status != "NORMAL":
            warnings.append(f"Pressure regulator status: {inp.pressure_regulator_status}")

        # Provenance
        calc_hash = hashlib.sha256(json.dumps({
            "hhv": hhv,
            "wobbe": wobbe,
            "dew_point": dew_point
        }).encode()).hexdigest()

        return FuelGasOutput(
            higher_heating_value_mj_m3=hhv,
            lower_heating_value_mj_m3=lhv,
            wobbe_index_mj_m3=wobbe,
            specific_gravity=sg,
            calculated_dew_point_c=dew_point,
            target_pressure_bar=target_pressure,
            target_temp_c=round(target_temp, 1),
            heating_required_kw=round(heating_required, 1),
            pressure_reduction_bar=round(max(0, pressure_drop), 2),
            quality_status=quality_status,
            wobbe_deviation_pct=round(wobbe_deviation, 2),
            contaminant_status=contaminant_status,
            jt_cooling_c=jt_cooling,
            condensation_risk=condensation_risk,
            filter_status=filter_status,
            regulator_status=inp.pressure_regulator_status,
            air_fuel_ratio_change_pct=round(af_ratio_change, 2),
            flame_temp_impact_c=round(flame_temp_impact, 1),
            recommendations=recommendations,
            warnings=warnings,
            calculation_hash=calc_hash,
            agent_version=self.VERSION
        )

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "category": "Fuel Systems",
            "type": "Controller"
        }
