"""
GL-017 CONDENSYNC Agent - Tube Fouling Detection Module

This module implements tube fouling detection from backpressure trends
for steam surface condensers. It provides deterministic algorithms for
fouling severity assessment, trend analysis, and economic impact calculations.

All calculations are zero-hallucination compliant with deterministic formulas.

Example:
    >>> detector = TubeFoulingDetector(config)
    >>> result = detector.analyze_fouling(
    ...     current_backpressure=1.8,
    ...     load_pct=85.0,
    ...     cw_inlet_temp=75.0,
    ... )
    >>> print(f"Fouling penalty: {result.backpressure_penalty_inhg:.2f} inHg")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import hashlib
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_017_condenser_optimization.config import (
    TubeFoulingConfig,
    PerformanceConfig,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.schemas import (
    TubeFoulingResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - Backpressure and Performance Correlations
# =============================================================================

class BackpressureConstants:
    """Backpressure correlation constants."""

    # Heat rate penalty per inHg backpressure deviation
    # Typical value: 60-100 BTU/kWh per inHg
    HEAT_RATE_PENALTY_BTU_KWH_PER_INHG = 80.0

    # Capacity loss per inHg backpressure
    # Typical: 0.5-1.0% per inHg
    CAPACITY_LOSS_PCT_PER_INHG = 0.75

    # Saturation pressure vs temperature (simplified)
    # P_sat (psia) at various temps (F)
    SAT_PRESSURE_TABLE = {
        60: 0.256,
        70: 0.363,
        80: 0.507,
        90: 0.699,
        100: 0.950,
        110: 1.276,
        120: 1.695,
        130: 2.226,
        140: 2.893,
    }

    # Temperature to vacuum conversion (approximate)
    # inHgA = (29.92 - P_sat * 2.036)
    @staticmethod
    def saturation_temp_to_vacuum(temp_f: float) -> float:
        """Convert saturation temperature to vacuum in inHgA."""
        # Antoine equation approximation for water
        # P_sat (psia) = 10^(A - B/(C + T))
        # Simplified correlation for 60-150F range
        p_sat_psia = 0.0001 * math.exp(0.0575 * temp_f)
        p_sat_inhg = p_sat_psia * 2.036
        vacuum_inhga = p_sat_inhg
        return vacuum_inhga

    @staticmethod
    def vacuum_to_saturation_temp(vacuum_inhga: float) -> float:
        """Convert vacuum in inHgA to saturation temperature."""
        p_sat_inhg = vacuum_inhga
        p_sat_psia = p_sat_inhg / 2.036
        # Inverse of saturation correlation
        if p_sat_psia <= 0:
            return 60.0
        temp_f = math.log(p_sat_psia / 0.0001) / 0.0575
        return max(60.0, min(150.0, temp_f))


@dataclass
class BackpressureDataPoint:
    """Historical backpressure data point."""
    timestamp: datetime
    backpressure_inhga: float
    load_pct: float
    cw_inlet_temp_f: float
    cw_flow_gpm: float


class TubeFoulingDetector:
    """
    Tube fouling detection from backpressure analysis.

    This class analyzes condenser backpressure data to detect tube fouling.
    It compares actual backpressure against expected clean-tube values
    based on operating conditions.

    Features:
        - Backpressure deviation analysis
        - Fouling trend detection
        - Heat rate penalty calculation
        - Economic impact assessment
        - Cleaning schedule optimization

    Attributes:
        fouling_config: Tube fouling configuration
        performance_config: Performance configuration
        baseline_curve: Clean-tube backpressure curve

    Example:
        >>> config = TubeFoulingConfig()
        >>> perf_config = PerformanceConfig()
        >>> detector = TubeFoulingDetector(config, perf_config)
        >>> result = detector.analyze_fouling(1.8, 85.0, 75.0, 90000.0)
    """

    def __init__(
        self,
        fouling_config: TubeFoulingConfig,
        performance_config: PerformanceConfig,
    ) -> None:
        """
        Initialize the tube fouling detector.

        Args:
            fouling_config: Tube fouling configuration
            performance_config: Performance configuration
        """
        self.fouling_config = fouling_config
        self.performance_config = performance_config
        self._history: List[BackpressureDataPoint] = []
        self._baseline_curve: Optional[Dict[float, Dict[float, float]]] = None
        self._calculation_count = 0

        # Initialize baseline curve
        self._build_baseline_curve()

        logger.info("TubeFoulingDetector initialized")

    def analyze_fouling(
        self,
        current_backpressure_inhga: float,
        load_pct: float,
        cw_inlet_temp_f: float,
        cw_flow_gpm: float,
        unit_capacity_mw: float = 500.0,
        electricity_price_usd_mwh: float = 50.0,
    ) -> TubeFoulingResult:
        """
        Analyze tube fouling from current operating conditions.

        Compares actual backpressure against expected clean-tube value
        and calculates fouling penalty and economic impact.

        Args:
            current_backpressure_inhga: Current condenser backpressure (inHgA)
            load_pct: Current unit load (%)
            cw_inlet_temp_f: Cooling water inlet temperature (F)
            cw_flow_gpm: Cooling water flow rate (GPM)
            unit_capacity_mw: Unit nameplate capacity (MW)
            electricity_price_usd_mwh: Electricity price ($/MWh)

        Returns:
            TubeFoulingResult with fouling analysis

        Raises:
            ValueError: If input parameters are invalid
        """
        logger.debug(
            f"Analyzing tube fouling: BP={current_backpressure_inhga:.2f} inHgA, "
            f"Load={load_pct:.1f}%, Tin={cw_inlet_temp_f:.1f}F"
        )
        self._calculation_count += 1

        # Validate inputs
        self._validate_inputs(
            current_backpressure_inhga, load_pct,
            cw_inlet_temp_f, cw_flow_gpm
        )

        # Calculate expected clean-tube backpressure
        expected_bp = self._calculate_expected_backpressure(
            load_pct, cw_inlet_temp_f, cw_flow_gpm
        )

        # Calculate backpressure penalty
        bp_penalty = current_backpressure_inhga - expected_bp
        bp_deviation_pct = (
            (bp_penalty / expected_bp) * 100 if expected_bp > 0 else 0.0
        )

        # Determine fouling severity
        fouling_detected = bp_penalty > self.fouling_config.backpressure_deviation_warning_inhg
        fouling_severity = self._determine_fouling_severity(bp_penalty)

        # Get fouling trend from history
        fouling_trend = self._analyze_trend()

        # Record data point
        self._record_data_point(
            current_backpressure_inhga, load_pct,
            cw_inlet_temp_f, cw_flow_gpm
        )

        # Calculate performance impacts
        heat_rate_penalty = self._calculate_heat_rate_penalty(bp_penalty)
        efficiency_loss = self._calculate_efficiency_loss(bp_penalty)
        lost_capacity = self._calculate_lost_capacity(
            bp_penalty, unit_capacity_mw
        )

        # Calculate economic impact
        daily_cost = self._calculate_daily_cost(
            lost_capacity, load_pct, electricity_price_usd_mwh
        )

        # Determine cleaning recommendation
        cleaning_recommended = self._should_recommend_cleaning(
            bp_penalty, daily_cost
        )
        cleaning_method = self._recommend_cleaning_method(fouling_severity)
        cleaning_benefit = self._estimate_cleaning_benefit(
            bp_penalty, unit_capacity_mw, electricity_price_usd_mwh
        )

        result = TubeFoulingResult(
            fouling_detected=fouling_detected,
            fouling_severity=fouling_severity,
            fouling_trend=fouling_trend,
            current_backpressure_inhga=round(current_backpressure_inhga, 3),
            expected_backpressure_inhga=round(expected_bp, 3),
            backpressure_penalty_inhg=round(bp_penalty, 3),
            backpressure_deviation_pct=round(bp_deviation_pct, 1),
            heat_rate_penalty_btu_kwh=round(heat_rate_penalty, 1),
            efficiency_loss_pct=round(efficiency_loss, 2),
            lost_capacity_mw=round(lost_capacity, 2),
            daily_cost_impact_usd=round(daily_cost, 2),
            cleaning_recommended=cleaning_recommended,
            recommended_cleaning_method=cleaning_method,
            estimated_cleaning_benefit_usd=(
                round(cleaning_benefit, 0) if cleaning_benefit else None
            ),
        )

        logger.info(
            f"Fouling analysis complete: severity={fouling_severity}, "
            f"penalty={bp_penalty:.3f} inHg"
        )

        return result

    def _validate_inputs(
        self,
        backpressure: float,
        load: float,
        inlet_temp: float,
        flow: float,
    ) -> None:
        """Validate input parameters."""
        if backpressure <= 0 or backpressure > 10:
            raise ValueError(
                f"Invalid backpressure: {backpressure} inHgA"
            )
        if load < 0 or load > 120:
            raise ValueError(f"Invalid load: {load}%")
        if inlet_temp < 32 or inlet_temp > 120:
            logger.warning(
                f"Inlet temperature {inlet_temp}F outside typical range"
            )
        if flow <= 0:
            raise ValueError(f"Invalid flow rate: {flow} GPM")

    def _build_baseline_curve(self) -> None:
        """
        Build baseline clean-tube backpressure curve.

        Creates a lookup table for expected backpressure at various
        load and inlet temperature combinations.
        """
        self._baseline_curve = {}

        # Load points (%)
        loads = [30, 40, 50, 60, 70, 80, 90, 100, 110]

        # Temperature points (F)
        temps = [50, 60, 70, 80, 90, 100]

        # Design point values
        design_bp = self.performance_config.design_backpressure_inhga
        design_inlet = self.performance_config.design_inlet_temp_f
        design_outlet = self.performance_config.design_outlet_temp_f

        for load in loads:
            self._baseline_curve[load] = {}
            for temp in temps:
                # Calculate expected backpressure
                # Based on heat balance: higher load or higher inlet = higher BP
                load_factor = load / 100.0
                temp_factor = (temp - design_inlet) / 10.0

                # Simplified correlation
                # BP increases ~0.1 inHg per 10F inlet temp increase
                # BP increases with load^0.8
                expected_bp = design_bp * (load_factor ** 0.8) + (temp_factor * 0.1)
                expected_bp = max(0.5, min(5.0, expected_bp))

                self._baseline_curve[load][temp] = expected_bp

        logger.debug("Baseline backpressure curve built")

    def _calculate_expected_backpressure(
        self,
        load_pct: float,
        cw_inlet_temp_f: float,
        cw_flow_gpm: float,
    ) -> float:
        """
        Calculate expected clean-tube backpressure.

        Uses interpolation from baseline curve with corrections
        for actual flow rate.

        Args:
            load_pct: Unit load (%)
            cw_inlet_temp_f: CW inlet temperature (F)
            cw_flow_gpm: CW flow rate (GPM)

        Returns:
            Expected backpressure (inHgA)
        """
        if self._baseline_curve is None:
            self._build_baseline_curve()

        # Interpolate load
        loads = sorted(self._baseline_curve.keys())
        load_low = max([l for l in loads if l <= load_pct], default=loads[0])
        load_high = min([l for l in loads if l >= load_pct], default=loads[-1])

        # Interpolate temperature
        temps = sorted(self._baseline_curve[load_low].keys())
        temp_low = max([t for t in temps if t <= cw_inlet_temp_f], default=temps[0])
        temp_high = min([t for t in temps if t >= cw_inlet_temp_f], default=temps[-1])

        # Bilinear interpolation
        if load_high == load_low:
            load_fraction = 0
        else:
            load_fraction = (load_pct - load_low) / (load_high - load_low)

        if temp_high == temp_low:
            temp_fraction = 0
        else:
            temp_fraction = (cw_inlet_temp_f - temp_low) / (temp_high - temp_low)

        # Get corner values
        bp_ll = self._baseline_curve[load_low][temp_low]
        bp_lh = self._baseline_curve[load_low][temp_high]
        bp_hl = self._baseline_curve[load_high][temp_low]
        bp_hh = self._baseline_curve[load_high][temp_high]

        # Interpolate
        bp_l = bp_ll + temp_fraction * (bp_lh - bp_ll)
        bp_h = bp_hl + temp_fraction * (bp_hh - bp_hl)
        expected_bp = bp_l + load_fraction * (bp_h - bp_l)

        # Flow correction
        design_flow = self.performance_config.design_cw_flow_gpm
        flow_ratio = cw_flow_gpm / design_flow if design_flow > 0 else 1.0

        # Lower flow = higher backpressure
        if flow_ratio < 1.0:
            flow_correction = 0.1 * (1.0 - flow_ratio)
            expected_bp += flow_correction

        return expected_bp

    def _determine_fouling_severity(
        self,
        bp_penalty_inhg: float,
    ) -> str:
        """
        Determine fouling severity from backpressure penalty.

        Args:
            bp_penalty_inhg: Backpressure penalty (inHg)

        Returns:
            Severity level (none, light, moderate, severe)
        """
        warning = self.fouling_config.backpressure_deviation_warning_inhg
        alarm = self.fouling_config.backpressure_deviation_alarm_inhg

        if bp_penalty_inhg < warning:
            return "none"
        elif bp_penalty_inhg < alarm:
            return "light"
        elif bp_penalty_inhg < alarm * 1.5:
            return "moderate"
        else:
            return "severe"

    def _analyze_trend(self) -> str:
        """
        Analyze fouling trend from historical data.

        Returns:
            Trend description (improving, stable, degrading)
        """
        if len(self._history) < 10:
            return "insufficient_data"

        # Get recent vs older data
        sorted_history = sorted(self._history, key=lambda x: x.timestamp)
        recent = sorted_history[-10:]
        older = sorted_history[-20:-10] if len(sorted_history) >= 20 else sorted_history[:10]

        # Compare average backpressure (normalized by conditions)
        recent_avg = sum(
            dp.backpressure_inhga for dp in recent
        ) / len(recent)
        older_avg = sum(
            dp.backpressure_inhga for dp in older
        ) / len(older)

        change = recent_avg - older_avg

        if change < -0.05:
            return "improving"
        elif change > 0.1:
            return "degrading"
        else:
            return "stable"

    def _record_data_point(
        self,
        backpressure: float,
        load: float,
        inlet_temp: float,
        flow: float,
    ) -> None:
        """Record a data point in history."""
        dp = BackpressureDataPoint(
            timestamp=datetime.now(timezone.utc),
            backpressure_inhga=backpressure,
            load_pct=load,
            cw_inlet_temp_f=inlet_temp,
            cw_flow_gpm=flow,
        )
        self._history.append(dp)

        # Trim old history (keep 30 days)
        cutoff = datetime.now(timezone.utc).timestamp() - (30 * 24 * 3600)
        self._history = [
            dp for dp in self._history
            if dp.timestamp.timestamp() > cutoff
        ]

    def _calculate_heat_rate_penalty(
        self,
        bp_penalty_inhg: float,
    ) -> float:
        """
        Calculate heat rate penalty from backpressure deviation.

        Args:
            bp_penalty_inhg: Backpressure penalty (inHg)

        Returns:
            Heat rate penalty (BTU/kWh)
        """
        if bp_penalty_inhg <= 0:
            return 0.0

        penalty = (
            bp_penalty_inhg *
            BackpressureConstants.HEAT_RATE_PENALTY_BTU_KWH_PER_INHG
        )
        return penalty

    def _calculate_efficiency_loss(
        self,
        bp_penalty_inhg: float,
    ) -> float:
        """
        Calculate efficiency loss from backpressure deviation.

        Args:
            bp_penalty_inhg: Backpressure penalty (inHg)

        Returns:
            Efficiency loss (%)
        """
        if bp_penalty_inhg <= 0:
            return 0.0

        # Heat rate penalty to efficiency loss
        # Assume 10,000 BTU/kWh baseline heat rate
        heat_rate_penalty = self._calculate_heat_rate_penalty(bp_penalty_inhg)
        efficiency_loss = (heat_rate_penalty / 10000.0) * 100.0

        return efficiency_loss

    def _calculate_lost_capacity(
        self,
        bp_penalty_inhg: float,
        unit_capacity_mw: float,
    ) -> float:
        """
        Calculate lost generation capacity.

        Args:
            bp_penalty_inhg: Backpressure penalty (inHg)
            unit_capacity_mw: Unit capacity (MW)

        Returns:
            Lost capacity (MW)
        """
        if bp_penalty_inhg <= 0:
            return 0.0

        capacity_loss_pct = (
            bp_penalty_inhg *
            BackpressureConstants.CAPACITY_LOSS_PCT_PER_INHG
        )
        lost_capacity_mw = unit_capacity_mw * (capacity_loss_pct / 100.0)

        return lost_capacity_mw

    def _calculate_daily_cost(
        self,
        lost_capacity_mw: float,
        load_pct: float,
        price_usd_mwh: float,
    ) -> float:
        """
        Calculate daily cost impact.

        Args:
            lost_capacity_mw: Lost capacity (MW)
            load_pct: Current load (%)
            price_usd_mwh: Electricity price ($/MWh)

        Returns:
            Daily cost impact ($)
        """
        if lost_capacity_mw <= 0:
            return 0.0

        # Adjust for actual operating hours (assume 24 hours at current load)
        operating_hours = 24.0 * (load_pct / 100.0)
        lost_energy_mwh = lost_capacity_mw * operating_hours
        daily_cost = lost_energy_mwh * price_usd_mwh

        return daily_cost

    def _should_recommend_cleaning(
        self,
        bp_penalty_inhg: float,
        daily_cost_usd: float,
        cleaning_cost_usd: float = 50000.0,
    ) -> bool:
        """
        Determine if cleaning should be recommended.

        Args:
            bp_penalty_inhg: Backpressure penalty (inHg)
            daily_cost_usd: Daily cost impact ($)
            cleaning_cost_usd: Estimated cleaning cost ($)

        Returns:
            True if cleaning recommended
        """
        # Recommend if payback is less than 30 days
        if daily_cost_usd <= 0:
            return False

        payback_days = cleaning_cost_usd / daily_cost_usd

        # Or if penalty exceeds alarm threshold
        alarm_threshold = self.fouling_config.backpressure_deviation_alarm_inhg

        return payback_days < 30 or bp_penalty_inhg > alarm_threshold

    def _recommend_cleaning_method(
        self,
        severity: str,
    ) -> Optional[str]:
        """
        Recommend appropriate cleaning method.

        Args:
            severity: Fouling severity

        Returns:
            Recommended cleaning method
        """
        cleaning_methods = {
            "none": None,
            "light": "online_ball_cleaning",
            "moderate": "online_ball_cleaning_intensive",
            "severe": "offline_mechanical_cleaning",
        }
        return cleaning_methods.get(severity)

    def _estimate_cleaning_benefit(
        self,
        bp_penalty_inhg: float,
        unit_capacity_mw: float,
        price_usd_mwh: float,
        days_per_year: int = 300,
    ) -> Optional[float]:
        """
        Estimate annual benefit from cleaning.

        Args:
            bp_penalty_inhg: Current backpressure penalty
            unit_capacity_mw: Unit capacity (MW)
            price_usd_mwh: Electricity price ($/MWh)
            days_per_year: Operating days per year

        Returns:
            Estimated annual benefit ($)
        """
        if bp_penalty_inhg <= 0:
            return None

        daily_cost = self._calculate_daily_cost(
            self._calculate_lost_capacity(bp_penalty_inhg, unit_capacity_mw),
            load_pct=80.0,  # Assume average load
            price_usd_mwh=price_usd_mwh,
        )

        annual_benefit = daily_cost * days_per_year

        return annual_benefit

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count

    def get_historical_penalties(
        self,
        days: int = 30,
    ) -> List[Tuple[datetime, float]]:
        """
        Get historical backpressure penalties.

        Args:
            days: Number of days of history

        Returns:
            List of (timestamp, penalty) tuples
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 24 * 3600)

        penalties = []
        for dp in self._history:
            if dp.timestamp.timestamp() > cutoff:
                expected = self._calculate_expected_backpressure(
                    dp.load_pct, dp.cw_inlet_temp_f, dp.cw_flow_gpm
                )
                penalty = dp.backpressure_inhga - expected
                penalties.append((dp.timestamp, penalty))

        return sorted(penalties, key=lambda x: x[0])
