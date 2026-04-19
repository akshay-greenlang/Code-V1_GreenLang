"""GL-028 CONVECTION-WATCH: Convection Section Analyzer Agent.

Analyzes convection section performance and fouling in process furnaces
to optimize cleaning schedules and maintain heat transfer efficiency.

Standards: API 560, ASME PTC 4.4
"""
import hashlib
import json
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConvectionInput(BaseModel):
    """Input for convection section analysis."""

    equipment_id: str = Field(..., description="Furnace ID")

    # Gas side conditions
    flue_gas_inlet_temp_c: float = Field(..., description="Gas inlet temp (°C)")
    flue_gas_outlet_temp_c: float = Field(..., description="Gas outlet temp (°C)")
    flue_gas_flow_kg_s: float = Field(..., ge=0, description="Flue gas flow (kg/s)")
    gas_pressure_drop_mbar: float = Field(..., ge=0, description="Gas side ΔP (mbar)")
    design_gas_dp_mbar: float = Field(..., ge=0, description="Design gas ΔP (mbar)")

    # Process side (multiple services possible)
    services: List[Dict[str, float]] = Field(
        default_factory=list,
        description="List of services: {name, inlet_temp, outlet_temp, flow_kg_s, duty_mw}"
    )

    # Design parameters
    design_duty_mw: float = Field(..., description="Design heat duty (MW)")
    design_gas_temp_drop_c: float = Field(..., description="Design gas temp drop (°C)")
    design_ua_kw_k: float = Field(..., description="Design UA value (kW/K)")

    # Tube bank info
    rows: int = Field(10, ge=1, description="Number of tube rows")
    tubes_per_row: int = Field(20, ge=1, description="Tubes per row")
    fin_type: str = Field("studded", description="bare, studded, finned")
    extended_surface_ratio: float = Field(3.0, description="Extended/bare surface ratio")

    # Cleaning history
    days_since_cleaning: int = Field(0, ge=0, description="Days since last cleaning")
    cleaning_method: str = Field("water_wash", description="water_wash, air_lance, chemical")

    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConvectionOutput(BaseModel):
    """Output from convection section analyzer."""

    # Heat transfer performance
    actual_duty_mw: float = Field(..., description="Actual heat duty (MW)")
    duty_ratio: float = Field(..., description="Actual/design duty ratio")
    actual_ua_kw_k: float = Field(..., description="Current UA value (kW/K)")
    ua_degradation_pct: float = Field(..., description="UA degradation from design (%)")

    # Temperature analysis
    lmtd_c: float = Field(..., description="Log mean temp difference (°C)")
    approach_temp_c: float = Field(..., description="Cold end approach (°C)")
    gas_temp_drop_c: float = Field(..., description="Actual gas temp drop (°C)")

    # Fouling assessment
    fouling_factor_m2k_kw: float = Field(..., description="Fouling factor (m²·K/kW)")
    fouling_severity: str = Field(..., description="CLEAN, LIGHT, MODERATE, HEAVY, SEVERE")
    gas_side_fouling_pct: float = Field(..., description="Gas side contribution (%)")
    process_side_fouling_pct: float = Field(..., description="Process side contribution (%)")

    # Pressure drop analysis
    dp_ratio: float = Field(..., description="Actual/design ΔP ratio")
    dp_trend: str = Field(..., description="STABLE, INCREASING, HIGH")

    # Efficiency metrics
    heat_recovery_efficiency_pct: float = Field(..., description="Heat recovery efficiency")
    stack_loss_mw: float = Field(..., description="Energy lost to stack (MW)")
    fuel_penalty_pct: float = Field(..., description="Fuel penalty from fouling (%)")

    # Cleaning recommendations
    cleaning_urgency: str = Field(..., description="NONE, SCHEDULE, PRIORITY, IMMEDIATE")
    estimated_days_to_cleaning: int = Field(..., description="Days until cleaning needed")
    cleaning_benefit_pct: float = Field(..., description="Expected efficiency gain from cleaning")

    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    calculation_hash: str
    calculation_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_version: str = Field(default="1.0.0")


def calculate_lmtd(
    t_hot_in: float, t_hot_out: float,
    t_cold_in: float, t_cold_out: float
) -> float:
    """Calculate log mean temperature difference."""
    dt1 = t_hot_in - t_cold_out
    dt2 = t_hot_out - t_cold_in

    if dt1 <= 0 or dt2 <= 0:
        return 0.0
    if abs(dt1 - dt2) < 0.1:
        return (dt1 + dt2) / 2

    return (dt1 - dt2) / math.log(dt1 / dt2)


def calculate_actual_duty(
    gas_flow_kg_s: float,
    gas_temp_drop_c: float,
    cp_gas: float = 1.1
) -> float:
    """Calculate actual heat duty from gas side."""
    duty_kw = gas_flow_kg_s * cp_gas * gas_temp_drop_c
    return round(duty_kw / 1000, 3)  # MW


def calculate_ua(duty_kw: float, lmtd: float) -> float:
    """Calculate overall heat transfer coefficient × area."""
    if lmtd <= 0:
        return 0.0
    return round(duty_kw / lmtd, 2)


def estimate_fouling_factor(
    ua_actual: float,
    ua_design: float,
    area_m2: float
) -> float:
    """
    Estimate fouling factor from UA degradation.

    Rf = 1/UA_actual - 1/UA_design (when using same area)
    """
    if ua_actual <= 0 or ua_design <= 0 or area_m2 <= 0:
        return 0.0

    # Convert to per-area basis
    u_actual = ua_actual / area_m2
    u_design = ua_design / area_m2

    if u_actual >= u_design:
        return 0.0

    rf = (1 / u_actual) - (1 / u_design)
    return round(rf * 1000, 4)  # m²·K/kW


def classify_fouling(fouling_factor: float, fin_type: str) -> str:
    """Classify fouling severity based on factor and surface type."""
    # Finned surfaces more sensitive to fouling
    thresholds = {
        "bare": {"light": 0.05, "moderate": 0.15, "heavy": 0.30, "severe": 0.50},
        "studded": {"light": 0.03, "moderate": 0.10, "heavy": 0.20, "severe": 0.35},
        "finned": {"light": 0.02, "moderate": 0.06, "heavy": 0.12, "severe": 0.20}
    }

    levels = thresholds.get(fin_type, thresholds["bare"])

    if fouling_factor < levels["light"]:
        return "CLEAN"
    elif fouling_factor < levels["moderate"]:
        return "LIGHT"
    elif fouling_factor < levels["heavy"]:
        return "MODERATE"
    elif fouling_factor < levels["severe"]:
        return "HEAVY"
    else:
        return "SEVERE"


class ConvectionSectionAnalyzerAgent:
    """Convection section performance analyzer."""

    AGENT_ID = "GL-028"
    AGENT_NAME = "CONVECTION-WATCH"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        validated = ConvectionInput(**input_data)
        return self._process(validated).model_dump()

    async def arun(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.run(input_data)

    def _process(self, inp: ConvectionInput) -> ConvectionOutput:
        recommendations = []
        warnings = []

        # Calculate gas temperature drop
        gas_temp_drop = inp.flue_gas_inlet_temp_c - inp.flue_gas_outlet_temp_c

        # Calculate actual duty
        actual_duty = calculate_actual_duty(inp.flue_gas_flow_kg_s, gas_temp_drop)
        duty_ratio = actual_duty / inp.design_duty_mw if inp.design_duty_mw > 0 else 0

        # Estimate average process temperatures for LMTD
        if inp.services:
            t_cold_in = sum(s.get("inlet_temp", 100) for s in inp.services) / len(inp.services)
            t_cold_out = sum(s.get("outlet_temp", 200) for s in inp.services) / len(inp.services)
        else:
            t_cold_in = 100
            t_cold_out = 250

        lmtd = calculate_lmtd(
            inp.flue_gas_inlet_temp_c, inp.flue_gas_outlet_temp_c,
            t_cold_in, t_cold_out
        )

        approach_temp = inp.flue_gas_outlet_temp_c - t_cold_in

        # Calculate UA
        actual_ua = calculate_ua(actual_duty * 1000, lmtd)
        ua_degradation = (1 - actual_ua / inp.design_ua_kw_k) * 100 if inp.design_ua_kw_k > 0 else 0

        # Estimate surface area (rough)
        tube_length = 3.0  # Assumed meters
        tube_od = 0.05  # Assumed meters
        bare_area = inp.rows * inp.tubes_per_row * math.pi * tube_od * tube_length
        extended_area = bare_area * inp.extended_surface_ratio

        # Fouling factor
        fouling_factor = estimate_fouling_factor(actual_ua, inp.design_ua_kw_k, extended_area)
        fouling_severity = classify_fouling(fouling_factor, inp.fin_type)

        # Pressure drop analysis
        dp_ratio = inp.gas_pressure_drop_mbar / inp.design_gas_dp_mbar if inp.design_gas_dp_mbar > 0 else 1.0

        if dp_ratio > 1.5:
            dp_trend = "HIGH"
            warnings.append(f"Gas side ΔP {dp_ratio:.1f}x design - check for blockage")
        elif dp_ratio > 1.2:
            dp_trend = "INCREASING"
        else:
            dp_trend = "STABLE"

        # Estimate gas vs process side fouling
        if dp_ratio > 1.3:
            gas_side_fouling = 70  # High ΔP suggests gas side
            process_side_fouling = 30
        else:
            gas_side_fouling = 40
            process_side_fouling = 60

        # Heat recovery efficiency
        # Max possible recovery if gas cooled to cold inlet temp
        max_possible_duty = inp.flue_gas_flow_kg_s * 1.1 * (inp.flue_gas_inlet_temp_c - t_cold_in) / 1000
        heat_recovery_eff = (actual_duty / max_possible_duty * 100) if max_possible_duty > 0 else 0

        # Stack loss
        stack_loss = inp.flue_gas_flow_kg_s * 1.1 * (inp.flue_gas_outlet_temp_c - 150) / 1000  # vs 150°C baseline

        # Fuel penalty
        fuel_penalty = ua_degradation * 0.05  # ~0.05% penalty per 1% UA loss

        # Cleaning recommendations
        if fouling_severity in ["SEVERE", "HEAVY"]:
            cleaning_urgency = "IMMEDIATE"
            days_to_clean = 0
        elif fouling_severity == "MODERATE":
            cleaning_urgency = "PRIORITY"
            days_to_clean = 14
        elif fouling_severity == "LIGHT" or inp.days_since_cleaning > 180:
            cleaning_urgency = "SCHEDULE"
            days_to_clean = 30
        else:
            cleaning_urgency = "NONE"
            days_to_clean = max(0, 365 - inp.days_since_cleaning)

        # Cleaning benefit estimate
        cleaning_benefit = ua_degradation * 0.8  # Assume 80% recovery

        # Recommendations
        if fouling_severity not in ["CLEAN", "LIGHT"]:
            recommendations.append(f"Fouling level {fouling_severity} - cleaning will recover ~{cleaning_benefit:.1f}% performance")

        if approach_temp < 30:
            recommendations.append(f"Low approach temp ({approach_temp:.0f}°C) limits heat recovery potential")

        if gas_side_fouling > 60:
            recommendations.append(f"Gas-side fouling dominant ({gas_side_fouling}%) - recommend air lance or water wash")
        elif process_side_fouling > 70:
            recommendations.append(f"Process-side fouling indicated - check for deposits")

        if fuel_penalty > 1:
            recommendations.append(f"Fouling causing ~{fuel_penalty:.1f}% fuel penalty")

        # Provenance
        calc_hash = hashlib.sha256(json.dumps({
            "duty": actual_duty,
            "ua": actual_ua,
            "fouling": fouling_factor
        }).encode()).hexdigest()

        return ConvectionOutput(
            actual_duty_mw=actual_duty,
            duty_ratio=round(duty_ratio, 3),
            actual_ua_kw_k=actual_ua,
            ua_degradation_pct=round(ua_degradation, 1),
            lmtd_c=round(lmtd, 1),
            approach_temp_c=round(approach_temp, 1),
            gas_temp_drop_c=round(gas_temp_drop, 1),
            fouling_factor_m2k_kw=fouling_factor,
            fouling_severity=fouling_severity,
            gas_side_fouling_pct=gas_side_fouling,
            process_side_fouling_pct=process_side_fouling,
            dp_ratio=round(dp_ratio, 2),
            dp_trend=dp_trend,
            heat_recovery_efficiency_pct=round(heat_recovery_eff, 1),
            stack_loss_mw=round(stack_loss, 3),
            fuel_penalty_pct=round(fuel_penalty, 2),
            cleaning_urgency=cleaning_urgency,
            estimated_days_to_cleaning=days_to_clean,
            cleaning_benefit_pct=round(cleaning_benefit, 1),
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
            "category": "Furnaces",
            "type": "Analyzer"
        }
