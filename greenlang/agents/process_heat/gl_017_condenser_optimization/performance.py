"""
GL-017 CONDENSYNC Agent - Condenser Performance Curves Module

This module implements condenser performance curve tracking and analysis.
It compares actual operating performance against design curves to identify
degradation and optimization opportunities.

All calculations are deterministic with zero hallucination.
Based on HEI Standards for Steam Surface Condensers.

Example:
    >>> analyzer = PerformanceAnalyzer(config)
    >>> result = analyzer.analyze_performance(
    ...     actual_backpressure=1.8,
    ...     steam_flow=450000,
    ...     cw_inlet_temp=75.0,
    ...     cw_flow=90000,
    ... )
    >>> print(f"Deviation: {result.backpressure_deviation_pct:.1f}%")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import hashlib
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_017_condenser_optimization.config import (
    PerformanceConfig,
    TubeFoulingConfig,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.schemas import (
    PerformanceResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Performance Curve Parameters
# =============================================================================

class PerformanceConstants:
    """Performance curve constants."""

    # Heat rate penalty factors
    HEAT_RATE_PENALTY_BTU_KWH_PER_INHG = 80.0
    CAPACITY_LOSS_PCT_PER_INHG = 0.75

    # Typical condenser design parameters
    TYPICAL_TTD_F = 5.0  # Terminal temperature difference
    TYPICAL_CLEANLINESS = 0.85

    # Degradation sources and typical contributions
    DEGRADATION_SOURCES = {
        "tube_fouling": 0.40,
        "air_ingress": 0.25,
        "low_cw_flow": 0.15,
        "high_cw_temp": 0.10,
        "vacuum_equipment": 0.10,
    }

    # Saturation pressure correlation
    # P_sat (inHgA) = a * exp(b * T) for T in Fahrenheit
    SAT_PRESSURE_A = 0.0001
    SAT_PRESSURE_B = 0.0575


@dataclass
class PerformanceDataPoint:
    """Historical performance data point."""
    timestamp: datetime
    backpressure_inhga: float
    expected_bp_inhga: float
    steam_flow_lb_hr: float
    cw_inlet_temp_f: float
    cw_flow_gpm: float
    load_pct: float


class PerformanceCurve:
    """
    Condenser performance curve model.

    Represents the expected backpressure as a function of:
    - Steam flow (load)
    - Cooling water inlet temperature
    - Cooling water flow rate

    The curve is built from design parameters and can be corrected
    for actual cleanliness factor.
    """

    def __init__(
        self,
        design_duty_btu_hr: float,
        design_steam_flow_lb_hr: float,
        design_backpressure_inhga: float,
        design_cw_inlet_temp_f: float,
        design_cw_outlet_temp_f: float,
        design_cw_flow_gpm: float,
        surface_area_ft2: float,
        cleanliness_factor: float = 0.85,
    ) -> None:
        """
        Initialize performance curve.

        Args:
            design_duty_btu_hr: Design heat duty (BTU/hr)
            design_steam_flow_lb_hr: Design steam flow (lb/hr)
            design_backpressure_inhga: Design backpressure (inHgA)
            design_cw_inlet_temp_f: Design CW inlet temperature (F)
            design_cw_outlet_temp_f: Design CW outlet temperature (F)
            design_cw_flow_gpm: Design CW flow (GPM)
            surface_area_ft2: Heat transfer surface area (ft2)
            cleanliness_factor: Cleanliness factor (0-1)
        """
        self.design_duty = design_duty_btu_hr
        self.design_steam_flow = design_steam_flow_lb_hr
        self.design_bp = design_backpressure_inhga
        self.design_cw_inlet = design_cw_inlet_temp_f
        self.design_cw_outlet = design_cw_outlet_temp_f
        self.design_cw_flow = design_cw_flow_gpm
        self.surface_area = surface_area_ft2
        self.cleanliness = cleanliness_factor

        # Calculate design parameters
        self.design_ttd = self._calculate_ttd(
            design_backpressure_inhga, design_cw_outlet_temp_f
        )
        self.design_range = design_cw_outlet_temp_f - design_cw_inlet_temp_f
        self.design_ua = self._calculate_ua()

        logger.debug(
            f"PerformanceCurve initialized: "
            f"TTD={self.design_ttd:.1f}F, Range={self.design_range:.1f}F"
        )

    def get_expected_backpressure(
        self,
        steam_flow_lb_hr: float,
        cw_inlet_temp_f: float,
        cw_flow_gpm: float,
        cleanliness_factor: Optional[float] = None,
    ) -> float:
        """
        Calculate expected backpressure from performance curve.

        Uses heat balance and performance curve relationships.

        Args:
            steam_flow_lb_hr: Steam flow (lb/hr)
            cw_inlet_temp_f: CW inlet temperature (F)
            cw_flow_gpm: CW flow rate (GPM)
            cleanliness_factor: Current cleanliness (optional)

        Returns:
            Expected backpressure (inHgA)
        """
        cf = cleanliness_factor or self.cleanliness

        # Calculate load ratio
        load_ratio = steam_flow_lb_hr / self.design_steam_flow

        # Calculate heat duty
        heat_duty = self.design_duty * load_ratio

        # Calculate CW temperature rise
        # Q = m_dot * Cp * dT
        # dT = Q / (m_dot * Cp)
        water_flow_lb_hr = cw_flow_gpm * 500  # Approximate conversion
        cw_rise = heat_duty / (water_flow_lb_hr * 1.0)  # Cp = 1 BTU/lb-F

        cw_outlet_temp = cw_inlet_temp_f + cw_rise

        # Calculate LMTD
        # For condenser: LMTD = Range / ln((Tsat - Tin)/(Tsat - Tout))
        # Iterative solution needed since Tsat depends on backpressure

        # Start with design TTD
        ttd = self.design_ttd

        # Correct for flow ratio
        flow_ratio = cw_flow_gpm / self.design_cw_flow
        if flow_ratio < 1.0:
            # Lower flow = higher TTD
            ttd *= (1.0 / flow_ratio) ** 0.4

        # Correct for cleanliness
        if cf < self.cleanliness:
            # Lower cleanliness = higher TTD
            ttd *= self.cleanliness / cf

        # Calculate saturation temperature
        sat_temp = cw_outlet_temp + ttd

        # Convert to backpressure
        backpressure = self._sat_temp_to_pressure(sat_temp)

        # Apply corrections
        # Inlet temperature effect
        inlet_diff = cw_inlet_temp_f - self.design_cw_inlet
        inlet_correction = inlet_diff * 0.02  # ~0.02 inHg per degree F

        backpressure += inlet_correction

        return max(0.5, min(5.0, backpressure))

    def _calculate_ttd(
        self,
        backpressure_inhga: float,
        cw_outlet_temp_f: float,
    ) -> float:
        """
        Calculate terminal temperature difference.

        TTD = Saturation temperature - CW outlet temperature

        Args:
            backpressure_inhga: Backpressure (inHgA)
            cw_outlet_temp_f: CW outlet temperature (F)

        Returns:
            TTD (F)
        """
        sat_temp = self._pressure_to_sat_temp(backpressure_inhga)
        return sat_temp - cw_outlet_temp_f

    def _calculate_ua(self) -> float:
        """
        Calculate design UA value.

        UA = Q / LMTD

        Returns:
            UA (BTU/hr-F)
        """
        sat_temp = self._pressure_to_sat_temp(self.design_bp)

        dt1 = sat_temp - self.design_cw_outlet
        dt2 = sat_temp - self.design_cw_inlet

        if dt1 <= 0 or dt2 <= 0:
            return 0.0

        if abs(dt1 - dt2) < 0.1:
            lmtd = dt1
        else:
            lmtd = (dt1 - dt2) / math.log(dt1 / dt2)

        return self.design_duty / lmtd if lmtd > 0 else 0.0

    def _sat_temp_to_pressure(self, temp_f: float) -> float:
        """Convert saturation temperature to pressure in inHgA."""
        p_psia = PerformanceConstants.SAT_PRESSURE_A * math.exp(
            PerformanceConstants.SAT_PRESSURE_B * temp_f
        )
        return p_psia * 2.036

    def _pressure_to_sat_temp(self, pressure_inhga: float) -> float:
        """Convert pressure in inHgA to saturation temperature."""
        p_psia = pressure_inhga / 2.036
        if p_psia <= 0:
            return 60.0
        temp_f = math.log(p_psia / PerformanceConstants.SAT_PRESSURE_A) / (
            PerformanceConstants.SAT_PRESSURE_B
        )
        return max(60.0, min(150.0, temp_f))


class PerformanceAnalyzer:
    """
    Condenser performance analyzer.

    Compares actual performance against design curves to identify
    degradation and quantify impacts.

    Features:
        - Performance curve tracking
        - Degradation source identification
        - Heat rate/capacity impact calculation
        - Trend analysis

    Attributes:
        config: Performance configuration
        curve: Performance curve model

    Example:
        >>> config = PerformanceConfig()
        >>> analyzer = PerformanceAnalyzer(config)
        >>> result = analyzer.analyze_performance(...)
    """

    def __init__(
        self,
        performance_config: PerformanceConfig,
        fouling_config: TubeFoulingConfig,
        surface_area_ft2: float = 150000.0,
    ) -> None:
        """
        Initialize the performance analyzer.

        Args:
            performance_config: Performance configuration
            fouling_config: Tube fouling configuration
            surface_area_ft2: Heat transfer surface area
        """
        self.config = performance_config
        self.fouling_config = fouling_config
        self._history: List[PerformanceDataPoint] = []
        self._calculation_count = 0

        # Build performance curve
        self.curve = PerformanceCurve(
            design_duty_btu_hr=performance_config.design_duty_btu_hr,
            design_steam_flow_lb_hr=performance_config.design_steam_flow_lb_hr,
            design_backpressure_inhga=performance_config.design_backpressure_inhga,
            design_cw_inlet_temp_f=performance_config.design_inlet_temp_f,
            design_cw_outlet_temp_f=performance_config.design_outlet_temp_f,
            design_cw_flow_gpm=performance_config.design_cw_flow_gpm,
            surface_area_ft2=surface_area_ft2,
            cleanliness_factor=fouling_config.design_cleanliness_factor,
        )

        logger.info("PerformanceAnalyzer initialized")

    def analyze_performance(
        self,
        actual_backpressure_inhga: float,
        steam_flow_lb_hr: float,
        cw_inlet_temp_f: float,
        cw_flow_gpm: float,
        cw_outlet_temp_f: Optional[float] = None,
        cleanliness_factor: Optional[float] = None,
        unit_capacity_mw: float = 500.0,
    ) -> PerformanceResult:
        """
        Analyze condenser performance.

        Compares actual backpressure against expected and calculates
        impacts.

        Args:
            actual_backpressure_inhga: Actual backpressure (inHgA)
            steam_flow_lb_hr: Steam flow (lb/hr)
            cw_inlet_temp_f: CW inlet temperature (F)
            cw_flow_gpm: CW flow rate (GPM)
            cw_outlet_temp_f: CW outlet temperature (optional)
            cleanliness_factor: Current cleanliness (optional)
            unit_capacity_mw: Unit capacity (MW)

        Returns:
            PerformanceResult with analysis
        """
        logger.debug(
            f"Analyzing performance: BP={actual_backpressure_inhga:.2f}, "
            f"Steam={steam_flow_lb_hr:.0f}, CW_in={cw_inlet_temp_f:.1f}F"
        )
        self._calculation_count += 1

        # Calculate load percentage
        load_pct = (
            steam_flow_lb_hr / self.config.design_steam_flow_lb_hr * 100
        )

        # Calculate expected backpressure
        expected_bp = self.curve.get_expected_backpressure(
            steam_flow_lb_hr, cw_inlet_temp_f, cw_flow_gpm, cleanliness_factor
        )

        # Calculate heat duty
        actual_duty = self._calculate_heat_duty(
            steam_flow_lb_hr, actual_backpressure_inhga
        )

        # Calculate deviations
        bp_deviation_inhg = actual_backpressure_inhga - expected_bp
        bp_deviation_pct = (
            (bp_deviation_inhg / expected_bp) * 100
            if expected_bp > 0 else 0.0
        )

        # Calculate TTD
        if cw_outlet_temp_f:
            sat_temp = self.curve._pressure_to_sat_temp(actual_backpressure_inhga)
            ttd_actual = sat_temp - cw_outlet_temp_f
        else:
            ttd_actual = self._estimate_ttd(
                actual_backpressure_inhga, steam_flow_lb_hr,
                cw_inlet_temp_f, cw_flow_gpm
            )

        ttd_design = self.curve.design_ttd
        ttd_deviation = ttd_actual - ttd_design

        # Calculate impacts
        heat_rate_impact = self._calculate_heat_rate_impact(bp_deviation_inhg)
        capacity_impact = self._calculate_capacity_impact(
            bp_deviation_inhg, unit_capacity_mw
        )
        efficiency_impact = self._calculate_efficiency_impact(bp_deviation_inhg)

        # Identify degradation source
        degradation_source, breakdown = self._identify_degradation_source(
            bp_deviation_inhg, cw_inlet_temp_f, cw_flow_gpm, cleanliness_factor
        )

        # Record data point
        self._record_data_point(
            actual_backpressure_inhga, expected_bp,
            steam_flow_lb_hr, cw_inlet_temp_f, cw_flow_gpm, load_pct
        )

        result = PerformanceResult(
            actual_duty_btu_hr=round(actual_duty, 0),
            design_duty_btu_hr=round(self.config.design_duty_btu_hr, 0),
            duty_ratio_pct=round(
                actual_duty / self.config.design_duty_btu_hr * 100, 1
            ),
            actual_backpressure_inhga=round(actual_backpressure_inhga, 3),
            expected_backpressure_inhga=round(expected_bp, 3),
            backpressure_deviation_inhg=round(bp_deviation_inhg, 3),
            backpressure_deviation_pct=round(bp_deviation_pct, 1),
            ttd_actual_f=round(ttd_actual, 2),
            ttd_design_f=round(ttd_design, 2),
            ttd_deviation_f=round(ttd_deviation, 2),
            heat_rate_impact_btu_kwh=round(heat_rate_impact, 1),
            capacity_impact_mw=round(capacity_impact, 2),
            efficiency_impact_pct=round(efficiency_impact, 2),
            degradation_source=degradation_source,
            degradation_breakdown=breakdown,
        )

        logger.info(
            f"Performance analysis complete: "
            f"deviation={bp_deviation_pct:.1f}%, source={degradation_source}"
        )

        return result

    def generate_performance_curve(
        self,
        cw_inlet_temps: List[float] = None,
        load_points: List[float] = None,
        cleanliness: float = 0.85,
    ) -> Dict[float, Dict[float, float]]:
        """
        Generate performance curve data.

        Creates expected backpressure values for various inlet temps
        and loads.

        Args:
            cw_inlet_temps: List of inlet temperatures (F)
            load_points: List of load percentages
            cleanliness: Cleanliness factor

        Returns:
            Nested dict: {load: {inlet_temp: backpressure}}
        """
        if cw_inlet_temps is None:
            cw_inlet_temps = [50, 60, 70, 80, 90, 100]
        if load_points is None:
            load_points = [40, 50, 60, 70, 80, 90, 100]

        design_cw_flow = self.config.design_cw_flow_gpm
        design_steam = self.config.design_steam_flow_lb_hr

        curves = {}
        for load in load_points:
            curves[load] = {}
            steam_flow = design_steam * (load / 100)

            for inlet_temp in cw_inlet_temps:
                bp = self.curve.get_expected_backpressure(
                    steam_flow, inlet_temp, design_cw_flow, cleanliness
                )
                curves[load][inlet_temp] = round(bp, 3)

        return curves

    def _calculate_heat_duty(
        self,
        steam_flow_lb_hr: float,
        backpressure_inhga: float,
    ) -> float:
        """
        Calculate heat duty from steam flow.

        Q = m_dot * h_fg

        Args:
            steam_flow_lb_hr: Steam flow (lb/hr)
            backpressure_inhga: Backpressure (inHgA)

        Returns:
            Heat duty (BTU/hr)
        """
        # Latent heat varies with pressure but use ~1000 BTU/lb
        h_fg = 1000.0
        return steam_flow_lb_hr * h_fg

    def _estimate_ttd(
        self,
        backpressure_inhga: float,
        steam_flow_lb_hr: float,
        cw_inlet_temp_f: float,
        cw_flow_gpm: float,
    ) -> float:
        """
        Estimate TTD when outlet temperature not available.

        Args:
            backpressure_inhga: Backpressure (inHgA)
            steam_flow_lb_hr: Steam flow (lb/hr)
            cw_inlet_temp_f: CW inlet temperature (F)
            cw_flow_gpm: CW flow (GPM)

        Returns:
            Estimated TTD (F)
        """
        # Calculate expected CW rise
        heat_duty = self._calculate_heat_duty(steam_flow_lb_hr, backpressure_inhga)
        water_flow_lb_hr = cw_flow_gpm * 500
        cw_rise = heat_duty / (water_flow_lb_hr * 1.0) if water_flow_lb_hr > 0 else 0

        cw_outlet_est = cw_inlet_temp_f + cw_rise
        sat_temp = self.curve._pressure_to_sat_temp(backpressure_inhga)

        return sat_temp - cw_outlet_est

    def _calculate_heat_rate_impact(
        self,
        bp_deviation_inhg: float,
    ) -> float:
        """
        Calculate heat rate impact.

        Args:
            bp_deviation_inhg: Backpressure deviation (inHg)

        Returns:
            Heat rate impact (BTU/kWh)
        """
        if bp_deviation_inhg <= 0:
            return 0.0

        return (
            bp_deviation_inhg *
            PerformanceConstants.HEAT_RATE_PENALTY_BTU_KWH_PER_INHG
        )

    def _calculate_capacity_impact(
        self,
        bp_deviation_inhg: float,
        unit_capacity_mw: float,
    ) -> float:
        """
        Calculate capacity impact.

        Args:
            bp_deviation_inhg: Backpressure deviation (inHg)
            unit_capacity_mw: Unit capacity (MW)

        Returns:
            Capacity impact (MW)
        """
        if bp_deviation_inhg <= 0:
            return 0.0

        capacity_loss_pct = (
            bp_deviation_inhg *
            PerformanceConstants.CAPACITY_LOSS_PCT_PER_INHG
        )
        return unit_capacity_mw * (capacity_loss_pct / 100)

    def _calculate_efficiency_impact(
        self,
        bp_deviation_inhg: float,
    ) -> float:
        """
        Calculate efficiency impact.

        Args:
            bp_deviation_inhg: Backpressure deviation (inHg)

        Returns:
            Efficiency impact (%)
        """
        heat_rate_impact = self._calculate_heat_rate_impact(bp_deviation_inhg)
        # Assume 10,000 BTU/kWh base
        return (heat_rate_impact / 10000) * 100

    def _identify_degradation_source(
        self,
        bp_deviation_inhg: float,
        cw_inlet_temp_f: float,
        cw_flow_gpm: float,
        cleanliness: Optional[float],
    ) -> Tuple[str, Dict[str, float]]:
        """
        Identify primary degradation source.

        Args:
            bp_deviation_inhg: Backpressure deviation (inHg)
            cw_inlet_temp_f: CW inlet temperature (F)
            cw_flow_gpm: CW flow (GPM)
            cleanliness: Cleanliness factor

        Returns:
            Tuple of (primary source, breakdown dict)
        """
        if bp_deviation_inhg <= 0:
            return "none", {}

        breakdown = {}

        # Check cooling water inlet temperature
        design_inlet = self.config.design_inlet_temp_f
        inlet_deviation = cw_inlet_temp_f - design_inlet
        if inlet_deviation > 5:
            temp_contribution = min(0.3, inlet_deviation * 0.03)
            breakdown["high_cw_temp"] = temp_contribution

        # Check cooling water flow
        design_flow = self.config.design_cw_flow_gpm
        flow_ratio = cw_flow_gpm / design_flow
        if flow_ratio < 0.9:
            flow_contribution = min(0.3, (1 - flow_ratio) * 0.5)
            breakdown["low_cw_flow"] = flow_contribution

        # Check cleanliness
        if cleanliness is not None:
            design_cf = self.fouling_config.design_cleanliness_factor
            if cleanliness < design_cf:
                cf_contribution = min(0.5, (design_cf - cleanliness) * 2)
                breakdown["tube_fouling"] = cf_contribution

        # Remaining attributed to vacuum/air ingress
        total_identified = sum(breakdown.values())
        if total_identified < 1.0:
            remainder = 1.0 - total_identified
            breakdown["air_ingress"] = remainder * 0.6
            breakdown["vacuum_equipment"] = remainder * 0.4

        # Normalize
        total = sum(breakdown.values())
        if total > 0:
            for key in breakdown:
                breakdown[key] = round(breakdown[key] / total * 100, 1)

        # Identify primary source
        if breakdown:
            primary = max(breakdown, key=breakdown.get)
        else:
            primary = "none"

        return primary, breakdown

    def _record_data_point(
        self,
        actual_bp: float,
        expected_bp: float,
        steam_flow: float,
        cw_inlet: float,
        cw_flow: float,
        load: float,
    ) -> None:
        """Record a performance data point."""
        dp = PerformanceDataPoint(
            timestamp=datetime.now(timezone.utc),
            backpressure_inhga=actual_bp,
            expected_bp_inhga=expected_bp,
            steam_flow_lb_hr=steam_flow,
            cw_inlet_temp_f=cw_inlet,
            cw_flow_gpm=cw_flow,
            load_pct=load,
        )
        self._history.append(dp)

        # Trim old history
        cutoff = datetime.now(timezone.utc).timestamp() - (30 * 24 * 3600)
        self._history = [
            d for d in self._history
            if d.timestamp.timestamp() > cutoff
        ]

    def get_performance_trend(
        self,
        days: int = 7,
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """
        Get performance trend data.

        Args:
            days: Days of history

        Returns:
            Dictionary of trends
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 24 * 3600)

        relevant = [
            d for d in self._history
            if d.timestamp.timestamp() > cutoff
        ]

        trends = {
            "backpressure": [],
            "deviation": [],
            "load": [],
        }

        for d in sorted(relevant, key=lambda x: x.timestamp):
            trends["backpressure"].append((d.timestamp, d.backpressure_inhga))
            trends["deviation"].append(
                (d.timestamp, d.backpressure_inhga - d.expected_bp_inhga)
            )
            trends["load"].append((d.timestamp, d.load_pct))

        return trends

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count
