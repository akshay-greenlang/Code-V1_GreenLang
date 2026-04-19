"""
GL-002 BoilerOptimizer Agent - Economizer Module

Provides economizer performance analysis and optimization.

Consolidates: GL-020 (Economizer Performance)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.shared.calculation_library import (
    ThermalIQCalculationLibrary,
    HeatExchangerInput,
)

logger = logging.getLogger(__name__)


class EconomizerInput(BaseModel):
    """Input for economizer analysis."""

    # Flue gas side
    flue_gas_inlet_temp_f: float = Field(..., description="Flue gas inlet temperature")
    flue_gas_outlet_temp_f: float = Field(..., description="Flue gas outlet temperature")
    flue_gas_flow_lb_hr: Optional[float] = Field(
        default=None,
        description="Flue gas mass flow"
    )

    # Water side
    water_inlet_temp_f: float = Field(..., description="Water inlet temperature")
    water_outlet_temp_f: float = Field(..., description="Water outlet temperature")
    water_flow_lb_hr: float = Field(..., gt=0, description="Water mass flow")

    # Operating conditions
    fuel_type: str = Field(default="natural_gas", description="Fuel type")
    fuel_sulfur_pct: float = Field(
        default=0.0,
        ge=0,
        le=5,
        description="Fuel sulfur content (%)"
    )

    # Design data
    design_duty_btu_hr: Optional[float] = Field(default=None)
    design_effectiveness: Optional[float] = Field(default=None)
    design_ua: Optional[float] = Field(default=None)


class EconomizerOutput(BaseModel):
    """Output from economizer analysis."""

    # Performance
    duty_btu_hr: float = Field(..., description="Current duty")
    effectiveness: float = Field(..., ge=0, le=1, description="Current effectiveness")
    water_temp_rise_f: float = Field(..., description="Water temperature rise")
    flue_gas_temp_drop_f: float = Field(..., description="Flue gas temperature drop")

    # Comparison to design
    duty_vs_design_pct: Optional[float] = Field(default=None)
    effectiveness_vs_design_pct: Optional[float] = Field(default=None)

    # Heat transfer
    lmtd_f: float = Field(..., description="Log mean temperature difference")
    actual_ua: float = Field(..., description="Actual UA value")
    fouling_factor: Optional[float] = Field(default=None)

    # Acid dew point
    acid_dew_point_f: float = Field(..., description="Estimated acid dew point")
    acid_dew_point_margin_f: float = Field(..., description="Margin above acid dew point")
    condensation_risk: str = Field(..., description="Condensation risk level")

    # Recommendations
    cleaning_recommended: bool = Field(default=False)
    operating_recommendations: List[str] = Field(default_factory=list)
    potential_savings_btu_hr: Optional[float] = Field(default=None)


class EconomizerOptimizer:
    """
    Economizer performance optimizer.

    Analyzes economizer operation for:
    - Heat transfer effectiveness
    - Fouling detection
    - Acid dew point protection
    - Cleaning scheduling
    """

    def __init__(
        self,
        design_duty_btu_hr: float = 5000000.0,
        design_effectiveness: float = 0.7,
        min_outlet_temp_f: float = 250.0,
    ) -> None:
        """
        Initialize economizer optimizer.

        Args:
            design_duty_btu_hr: Design heat duty
            design_effectiveness: Design effectiveness
            min_outlet_temp_f: Minimum outlet temperature
        """
        self.design_duty = design_duty_btu_hr
        self.design_effectiveness = design_effectiveness
        self.min_outlet_temp = min_outlet_temp_f

        self.calc_library = ThermalIQCalculationLibrary()

        # Performance history
        self._effectiveness_history: List[float] = []
        self._cleaning_history: List[datetime] = []

        logger.info(
            f"EconomizerOptimizer initialized: "
            f"duty={design_duty_btu_hr/1e6:.1f} MMBTU/hr, "
            f"effectiveness={design_effectiveness:.0%}"
        )

    def analyze(self, input_data: EconomizerInput) -> EconomizerOutput:
        """
        Analyze economizer performance.

        Args:
            input_data: Current operating data

        Returns:
            EconomizerOutput with analysis results
        """
        recommendations = []

        # Calculate temperature changes
        water_temp_rise = input_data.water_outlet_temp_f - input_data.water_inlet_temp_f
        flue_gas_temp_drop = (
            input_data.flue_gas_inlet_temp_f - input_data.flue_gas_outlet_temp_f
        )

        # Calculate duty from water side (more accurate)
        # Q = m * Cp * dT
        water_duty = input_data.water_flow_lb_hr * 1.0 * water_temp_rise

        # Estimate flue gas flow if not provided
        flue_gas_flow = input_data.flue_gas_flow_lb_hr
        if flue_gas_flow is None:
            # Estimate from duty and temperature drop
            # Q = m * Cp * dT, Cp_flue_gas ~ 0.24
            flue_gas_flow = water_duty / (0.24 * flue_gas_temp_drop) if flue_gas_temp_drop > 0 else 0

        # Calculate effectiveness
        # Effectiveness = actual heat transfer / maximum possible
        max_heat_transfer = min(
            input_data.water_flow_lb_hr * 1.0,  # C_water
            flue_gas_flow * 0.24 if flue_gas_flow else float('inf')  # C_flue
        ) * (input_data.flue_gas_inlet_temp_f - input_data.water_inlet_temp_f)

        effectiveness = water_duty / max_heat_transfer if max_heat_transfer > 0 else 0
        effectiveness = min(effectiveness, 1.0)  # Cap at 100%

        # Calculate LMTD (counterflow arrangement)
        dt1 = input_data.flue_gas_inlet_temp_f - input_data.water_outlet_temp_f
        dt2 = input_data.flue_gas_outlet_temp_f - input_data.water_inlet_temp_f

        if dt1 <= 0 or dt2 <= 0:
            lmtd = abs(dt1 - dt2) / 2  # Approximate if temperature cross
            recommendations.append("Temperature cross detected - check operating conditions")
        elif abs(dt1 - dt2) < 0.1:
            lmtd = dt1
        else:
            lmtd = (dt1 - dt2) / math.log(dt1 / dt2)

        # Calculate UA
        actual_ua = water_duty / lmtd if lmtd > 0 else 0

        # Calculate fouling factor if design UA available
        fouling_factor = None
        if input_data.design_ua and input_data.design_ua > 0 and actual_ua > 0:
            # R_fouling = 1/U_actual - 1/U_design
            fouling_factor = (1 / actual_ua - 1 / input_data.design_ua)
            if fouling_factor < 0:
                fouling_factor = 0  # Can't have negative fouling

        # Comparison to design
        duty_vs_design = None
        if input_data.design_duty_btu_hr:
            duty_vs_design = (water_duty / input_data.design_duty_btu_hr) * 100

        effectiveness_vs_design = None
        if input_data.design_effectiveness:
            effectiveness_vs_design = (effectiveness / input_data.design_effectiveness) * 100

        # Calculate acid dew point
        acid_dew_point = self._calculate_acid_dew_point(
            fuel_type=input_data.fuel_type,
            sulfur_pct=input_data.fuel_sulfur_pct,
        )

        # Margin above acid dew point
        adp_margin = input_data.flue_gas_outlet_temp_f - acid_dew_point

        # Condensation risk
        if adp_margin < 0:
            condensation_risk = "HIGH"
            recommendations.append(
                f"CRITICAL: Operating below acid dew point by {abs(adp_margin):.0f}F. "
                "Immediate action required to prevent corrosion."
            )
        elif adp_margin < 25:
            condensation_risk = "MODERATE"
            recommendations.append(
                f"Low margin above acid dew point ({adp_margin:.0f}F). "
                "Monitor closely and consider raising outlet temperature."
            )
        elif adp_margin < 50:
            condensation_risk = "LOW"
        else:
            condensation_risk = "NONE"

        # Cleaning recommendation
        cleaning_recommended = False
        design_eff = input_data.design_effectiveness or self.design_effectiveness

        if effectiveness < design_eff * 0.75:
            cleaning_recommended = True
            potential_recovery = (design_eff - effectiveness) * max_heat_transfer
            recommendations.append(
                f"Effectiveness at {effectiveness:.0%} vs design {design_eff:.0%}. "
                f"Cleaning could recover ~{potential_recovery/1e6:.2f} MMBTU/hr."
            )
        elif fouling_factor and fouling_factor > 0.002:
            cleaning_recommended = True
            recommendations.append(
                f"Fouling factor of {fouling_factor:.4f} hr-ft2-F/BTU indicates "
                "significant fouling. Schedule cleaning."
            )

        # Track effectiveness
        self._effectiveness_history.append(effectiveness)
        if len(self._effectiveness_history) > 100:
            self._effectiveness_history.pop(0)

        # Check for declining trend
        if len(self._effectiveness_history) >= 10:
            recent = self._effectiveness_history[-10:]
            if recent[-1] < recent[0] * 0.95:  # 5% decline
                recommendations.append(
                    "Effectiveness declining over recent readings. "
                    "Monitor for accelerated fouling."
                )

        # Potential savings
        potential_savings = None
        if cleaning_recommended and design_eff > effectiveness:
            potential_savings = (design_eff - effectiveness) * max_heat_transfer

        return EconomizerOutput(
            duty_btu_hr=round(water_duty, 0),
            effectiveness=round(effectiveness, 4),
            water_temp_rise_f=round(water_temp_rise, 1),
            flue_gas_temp_drop_f=round(flue_gas_temp_drop, 1),
            duty_vs_design_pct=round(duty_vs_design, 1) if duty_vs_design else None,
            effectiveness_vs_design_pct=(
                round(effectiveness_vs_design, 1) if effectiveness_vs_design else None
            ),
            lmtd_f=round(lmtd, 1),
            actual_ua=round(actual_ua, 2),
            fouling_factor=round(fouling_factor, 6) if fouling_factor else None,
            acid_dew_point_f=round(acid_dew_point, 1),
            acid_dew_point_margin_f=round(adp_margin, 1),
            condensation_risk=condensation_risk,
            cleaning_recommended=cleaning_recommended,
            operating_recommendations=recommendations,
            potential_savings_btu_hr=round(potential_savings, 0) if potential_savings else None,
        )

    def _calculate_acid_dew_point(
        self,
        fuel_type: str,
        sulfur_pct: float,
    ) -> float:
        """
        Calculate acid dew point based on fuel type and sulfur content.

        Uses correlation for sulfuric acid dew point.
        """
        # Base dew point for natural gas (minimal sulfur)
        base_dew_point = 180.0

        # Adjustment for sulfur content
        # Verhoff and Banchero correlation approximation
        if sulfur_pct > 0:
            # Higher sulfur = higher dew point
            # Approximate: +20F per 1% sulfur
            sulfur_adjustment = sulfur_pct * 20.0
        else:
            sulfur_adjustment = 0.0

        # Fuel type adjustments
        fuel_adjustments = {
            "natural_gas": 0,
            "no2_fuel_oil": 30,  # Higher sulfur typically
            "no6_fuel_oil": 50,
            "coal": 60,
        }

        fuel_key = fuel_type.lower().replace(" ", "_")
        fuel_adj = fuel_adjustments.get(fuel_key, 20)

        acid_dew_point = base_dew_point + sulfur_adjustment + fuel_adj

        # Typical range is 180-300F
        acid_dew_point = max(180, min(300, acid_dew_point))

        return acid_dew_point

    def record_cleaning(self) -> None:
        """Record an economizer cleaning event."""
        self._cleaning_history.append(datetime.now(timezone.utc))
        self._effectiveness_history.clear()  # Reset trend after cleaning
        logger.info("Economizer cleaning recorded")

    def get_time_since_cleaning(self) -> Optional[float]:
        """Get hours since last cleaning."""
        if not self._cleaning_history:
            return None

        last_cleaning = self._cleaning_history[-1]
        hours = (datetime.now(timezone.utc) - last_cleaning).total_seconds() / 3600
        return hours

    def get_cleaning_schedule(
        self,
        operating_hours: float,
        cleaning_interval_hours: int = 2000,
    ) -> Dict[str, Any]:
        """
        Get cleaning schedule recommendation.

        Args:
            operating_hours: Current operating hours since cleaning
            cleaning_interval_hours: Recommended interval

        Returns:
            Cleaning schedule info
        """
        hours_until_cleaning = cleaning_interval_hours - operating_hours
        overdue = hours_until_cleaning < 0

        return {
            "operating_hours": round(operating_hours, 0),
            "recommended_interval_hours": cleaning_interval_hours,
            "hours_until_cleaning": round(max(0, hours_until_cleaning), 0),
            "overdue": overdue,
            "recommendation": (
                "OVERDUE: Schedule cleaning immediately"
                if overdue
                else f"Next cleaning in ~{hours_until_cleaning:.0f} hours"
            ),
        }
