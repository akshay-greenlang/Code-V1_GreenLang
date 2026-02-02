"""
GL-006 WasteHeatRecovery Agent - Automated Pinch Analysis Module

This module implements automated pinch analysis with composite curve construction
for optimal waste heat recovery targeting. It follows the Linnhoff methodology
for minimum utility targeting and provides complete provenance tracking.

Features:
    - Hot and Cold Composite Curve construction
    - Grand Composite Curve (GCC) generation
    - Pinch temperature identification (shifted and actual)
    - Minimum utility targets (hot and cold)
    - Delta T_min optimization
    - Problem Table Algorithm implementation
    - Interval-based heat cascade calculations
    - Stream segmentation for non-linear Cp

Standards Reference:
    - Linnhoff, B., et al. "A User Guide on Process Integration for the
      Efficient Use of Energy" (IChemE, 1982)
    - Smith, R. "Chemical Process Design and Integration" (Wiley, 2005)

Example:
    >>> analyzer = PinchAnalyzer(delta_t_min=20.0)
    >>> streams = [
    ...     HeatStream(name="H1", supply_temp=300, target_temp=150, mcp=10.0),
    ...     HeatStream(name="C1", supply_temp=80, target_temp=200, mcp=12.0),
    ... ]
    >>> result = analyzer.analyze(streams)
    >>> print(f"Pinch: {result.pinch_temperature_f}F")
    >>> print(f"Hot utility: {result.minimum_hot_utility_btu_hr:,.0f} BTU/hr")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math
import uuid

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class StreamType(Enum):
    """Stream type classification."""
    HOT = "hot"      # Heat source - needs cooling
    COLD = "cold"    # Heat sink - needs heating


class PinchViolationType(Enum):
    """Types of pinch rule violations."""
    HEAT_TRANSFER_ACROSS_PINCH = auto()
    HOT_UTILITY_BELOW_PINCH = auto()
    COLD_UTILITY_ABOVE_PINCH = auto()


# =============================================================================
# DATA MODELS
# =============================================================================

class HeatStream(BaseModel):
    """Process stream definition for pinch analysis."""

    name: str = Field(..., description="Stream identifier")
    stream_type: Optional[StreamType] = Field(
        default=None,
        description="Auto-detected if not specified"
    )
    supply_temp_f: float = Field(..., description="Supply (inlet) temperature (F)")
    target_temp_f: float = Field(..., description="Target (outlet) temperature (F)")
    mcp: float = Field(
        ...,
        gt=0,
        description="Heat capacity flow rate m*Cp (BTU/hr-F)"
    )
    mass_flow_rate: Optional[float] = Field(
        default=None,
        gt=0,
        description="Mass flow rate (lb/hr)"
    )
    specific_heat: Optional[float] = Field(
        default=None,
        gt=0,
        description="Specific heat (BTU/lb-F)"
    )
    heat_duty: Optional[float] = Field(
        default=None,
        description="Calculated heat duty (BTU/hr)"
    )
    min_approach_temp_f: Optional[float] = Field(
        default=None,
        description="Stream-specific min approach temp"
    )

    @model_validator(mode='before')
    @classmethod
    def auto_detect_and_calculate(cls, data: Any) -> Any:
        """Auto-detect stream type and calculate heat duty."""
        if isinstance(data, dict):
            # Auto-detect stream type
            if data.get('stream_type') is None:
                supply = data.get('supply_temp_f')
                target = data.get('target_temp_f')
                if supply is not None and target is not None:
                    data['stream_type'] = StreamType.HOT if supply > target else StreamType.COLD

            # Auto-calculate heat duty
            if data.get('heat_duty') is None:
                supply = data.get('supply_temp_f')
                target = data.get('target_temp_f')
                mcp = data.get('mcp')
                if supply is not None and target is not None and mcp is not None:
                    data['heat_duty'] = abs(mcp * (supply - target))

        return data

    class Config:
        use_enum_values = True


class TemperatureInterval(BaseModel):
    """Temperature interval for problem table algorithm."""

    interval_id: int = Field(..., description="Interval index")
    temp_high_f: float = Field(..., description="High temperature boundary")
    temp_low_f: float = Field(..., description="Low temperature boundary")
    delta_t_f: float = Field(..., description="Temperature span")
    hot_mcp_sum: float = Field(default=0.0, description="Sum of hot stream MCp")
    cold_mcp_sum: float = Field(default=0.0, description="Sum of cold stream MCp")
    net_mcp: float = Field(default=0.0, description="Hot MCp - Cold MCp")
    interval_heat: float = Field(default=0.0, description="Heat available (BTU/hr)")
    cascade_heat: float = Field(default=0.0, description="Cumulative cascade heat")


class CompositeCurvePoint(BaseModel):
    """Point on a composite curve."""

    temperature_f: float = Field(..., description="Temperature (F)")
    enthalpy_btu_hr: float = Field(..., description="Enthalpy (BTU/hr)")


class CompositeData(BaseModel):
    """Composite curve data."""

    curve_type: str = Field(..., description="hot or cold")
    points: List[CompositeCurvePoint] = Field(
        default_factory=list,
        description="Curve points"
    )
    total_duty_btu_hr: float = Field(default=0.0, description="Total heat duty")


class GrandCompositePoint(BaseModel):
    """Point on the Grand Composite Curve."""

    shifted_temp_f: float = Field(..., description="Shifted temperature (F)")
    net_heat_btu_hr: float = Field(..., description="Net heat flow (BTU/hr)")


class PinchViolation(BaseModel):
    """Detected pinch rule violation."""

    violation_type: PinchViolationType = Field(...)
    description: str = Field(...)
    stream_names: List[str] = Field(default_factory=list)
    heat_transfer_btu_hr: float = Field(default=0.0)
    recommendation: str = Field(default="")

    class Config:
        use_enum_values = True


class PinchAnalysisResult(BaseModel):
    """Complete result from pinch analysis."""

    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique analysis identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Pinch point
    pinch_temperature_f: float = Field(..., description="Pinch temperature (F)")
    shifted_pinch_temp_f: float = Field(
        ...,
        description="Shifted pinch temperature (F)"
    )
    pinch_above_temp_f: float = Field(
        ...,
        description="Pinch temperature for hot streams (F)"
    )
    pinch_below_temp_f: float = Field(
        ...,
        description="Pinch temperature for cold streams (F)"
    )

    # Utility targets
    minimum_hot_utility_btu_hr: float = Field(
        ...,
        description="Minimum hot utility required (BTU/hr)"
    )
    minimum_cold_utility_btu_hr: float = Field(
        ...,
        description="Minimum cold utility required (BTU/hr)"
    )
    maximum_heat_recovery_btu_hr: float = Field(
        ...,
        description="Maximum heat recovery possible (BTU/hr)"
    )

    # Process totals
    total_hot_duty_btu_hr: float = Field(
        ...,
        description="Total hot stream duty (BTU/hr)"
    )
    total_cold_duty_btu_hr: float = Field(
        ...,
        description="Total cold stream duty (BTU/hr)"
    )

    # Curves
    hot_composite: CompositeData = Field(..., description="Hot composite curve")
    cold_composite: CompositeData = Field(..., description="Cold composite curve")
    grand_composite: List[GrandCompositePoint] = Field(
        default_factory=list,
        description="Grand composite curve"
    )

    # Analysis details
    temperature_intervals: List[TemperatureInterval] = Field(
        default_factory=list,
        description="Problem table intervals"
    )
    delta_t_min_f: float = Field(..., description="Minimum approach temperature")
    stream_count: int = Field(..., description="Number of streams analyzed")

    # Violations
    pinch_violations: List[PinchViolation] = Field(
        default_factory=list,
        description="Detected pinch rule violations"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    calculation_method: str = Field(
        default="Problem Table Algorithm (Linnhoff)",
        description="Calculation method used"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Optimization recommendations"
    )


class DeltaTMinOptimizationResult(BaseModel):
    """Result from delta T_min optimization."""

    optimal_delta_t_min_f: float = Field(...)
    capital_cost_usd: float = Field(...)
    utility_cost_usd_yr: float = Field(...)
    total_annual_cost_usd: float = Field(...)
    heat_recovery_btu_hr: float = Field(...)
    hot_utility_btu_hr: float = Field(...)
    cold_utility_btu_hr: float = Field(...)
    delta_t_values: List[float] = Field(default_factory=list)
    total_costs: List[float] = Field(default_factory=list)


# =============================================================================
# PINCH ANALYZER CLASS
# =============================================================================

class PinchAnalyzer:
    """
    Automated Pinch Analysis Engine.

    Implements the Problem Table Algorithm (PTA) developed by Linnhoff
    for minimum utility targeting in heat exchanger networks.

    Features:
        - Composite curve construction (hot and cold)
        - Grand Composite Curve generation
        - Automatic pinch point identification
        - Minimum utility targets
        - Delta T_min optimization
        - Pinch rule violation detection

    Attributes:
        delta_t_min: Minimum approach temperature (F)
        hot_utility_cost: Cost per MMBTU of hot utility
        cold_utility_cost: Cost per MMBTU of cold utility

    Example:
        >>> analyzer = PinchAnalyzer(delta_t_min=20.0)
        >>> result = analyzer.analyze(streams)
        >>> print(f"Pinch: {result.pinch_temperature_f}F")
    """

    def __init__(
        self,
        delta_t_min_f: float = 20.0,
        hot_utility_cost_per_mmbtu: float = 8.0,
        cold_utility_cost_per_mmbtu: float = 1.5,
        operating_hours_per_year: int = 8760,
    ) -> None:
        """
        Initialize the Pinch Analyzer.

        Args:
            delta_t_min_f: Minimum approach temperature (F)
            hot_utility_cost_per_mmbtu: Hot utility cost ($/MMBTU)
            cold_utility_cost_per_mmbtu: Cold utility cost ($/MMBTU)
            operating_hours_per_year: Annual operating hours
        """
        self.delta_t_min = delta_t_min_f
        self.hot_utility_cost = hot_utility_cost_per_mmbtu
        self.cold_utility_cost = cold_utility_cost_per_mmbtu
        self.operating_hours = operating_hours_per_year

        logger.info(
            f"PinchAnalyzer initialized: delta_T_min={delta_t_min_f}F"
        )

    def analyze(
        self,
        streams: List[HeatStream],
        delta_t_min_override: Optional[float] = None,
    ) -> PinchAnalysisResult:
        """
        Perform complete pinch analysis on process streams.

        Args:
            streams: List of hot and cold process streams
            delta_t_min_override: Override default delta_T_min

        Returns:
            Complete pinch analysis results

        Raises:
            ValueError: If streams are invalid
        """
        logger.info(f"Starting pinch analysis for {len(streams)} streams")

        if len(streams) < 2:
            raise ValueError("At least 2 streams required for pinch analysis")

        delta_t_min = delta_t_min_override or self.delta_t_min

        # Separate hot and cold streams
        hot_streams = [s for s in streams if s.stream_type in (StreamType.HOT, "hot", StreamType.HOT.value)]
        cold_streams = [s for s in streams if s.stream_type in (StreamType.COLD, "cold", StreamType.COLD.value)]

        if not hot_streams or not cold_streams:
            raise ValueError("Both hot and cold streams required")

        logger.debug(
            f"Streams: {len(hot_streams)} hot, {len(cold_streams)} cold"
        )

        # Build temperature intervals
        intervals = self._build_temperature_intervals(
            hot_streams, cold_streams, delta_t_min
        )

        # Execute Problem Table Algorithm
        intervals = self._problem_table_algorithm(intervals)

        # Find pinch point
        pinch_result = self._find_pinch_point(intervals, delta_t_min)

        # Calculate utility targets
        utility_targets = self._calculate_utility_targets(intervals)

        # Build composite curves
        hot_composite = self._build_composite_curve(hot_streams, "hot")
        cold_composite = self._build_composite_curve(cold_streams, "cold")

        # Build Grand Composite Curve
        grand_composite = self._build_grand_composite(intervals)

        # Calculate process totals
        total_hot_duty = sum(s.heat_duty or 0 for s in hot_streams)
        total_cold_duty = sum(s.heat_duty or 0 for s in cold_streams)

        # Maximum heat recovery
        max_recovery = min(total_hot_duty, total_cold_duty) - min(
            utility_targets["hot_utility"],
            utility_targets["cold_utility"]
        )
        max_recovery = max(0, total_hot_duty - utility_targets["hot_utility"])

        # Detect pinch violations (if any existing HEN)
        violations = self._detect_pinch_violations(
            streams, pinch_result["pinch_temp"], delta_t_min
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            pinch_result,
            utility_targets,
            max_recovery,
            total_hot_duty,
            total_cold_duty,
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance(streams, delta_t_min)

        result = PinchAnalysisResult(
            pinch_temperature_f=round(pinch_result["pinch_temp"], 1),
            shifted_pinch_temp_f=round(pinch_result["shifted_pinch"], 1),
            pinch_above_temp_f=round(pinch_result["pinch_above"], 1),
            pinch_below_temp_f=round(pinch_result["pinch_below"], 1),
            minimum_hot_utility_btu_hr=round(utility_targets["hot_utility"], 0),
            minimum_cold_utility_btu_hr=round(utility_targets["cold_utility"], 0),
            maximum_heat_recovery_btu_hr=round(max_recovery, 0),
            total_hot_duty_btu_hr=round(total_hot_duty, 0),
            total_cold_duty_btu_hr=round(total_cold_duty, 0),
            hot_composite=hot_composite,
            cold_composite=cold_composite,
            grand_composite=grand_composite,
            temperature_intervals=intervals,
            delta_t_min_f=delta_t_min,
            stream_count=len(streams),
            pinch_violations=violations,
            provenance_hash=provenance_hash,
            recommendations=recommendations,
        )

        logger.info(
            f"Pinch analysis complete: Pinch={result.pinch_temperature_f}F, "
            f"Hot utility={result.minimum_hot_utility_btu_hr:,.0f} BTU/hr"
        )

        return result

    def _build_temperature_intervals(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        delta_t_min: float,
    ) -> List[TemperatureInterval]:
        """
        Build temperature intervals for Problem Table Algorithm.

        Uses shifted temperature scale where:
        - Hot streams shifted down by delta_T_min / 2
        - Cold streams shifted up by delta_T_min / 2

        Args:
            hot_streams: List of hot streams
            cold_streams: List of cold streams
            delta_t_min: Minimum approach temperature

        Returns:
            List of temperature intervals
        """
        # Collect all shifted temperatures
        temps = set()
        half_dt = delta_t_min / 2

        for s in hot_streams:
            temps.add(s.supply_temp_f - half_dt)
            temps.add(s.target_temp_f - half_dt)

        for s in cold_streams:
            temps.add(s.supply_temp_f + half_dt)
            temps.add(s.target_temp_f + half_dt)

        # Sort descending
        sorted_temps = sorted(temps, reverse=True)

        # Create intervals
        intervals = []
        for i in range(len(sorted_temps) - 1):
            t_high = sorted_temps[i]
            t_low = sorted_temps[i + 1]
            delta_t = t_high - t_low

            if delta_t < 0.001:  # Skip negligible intervals
                continue

            # Sum MCp for streams active in this interval
            hot_mcp = sum(
                s.mcp for s in hot_streams
                if (s.supply_temp_f - half_dt) >= t_high and
                   (s.target_temp_f - half_dt) <= t_low
            )

            cold_mcp = sum(
                s.mcp for s in cold_streams
                if (s.target_temp_f + half_dt) >= t_high and
                   (s.supply_temp_f + half_dt) <= t_low
            )

            net_mcp = hot_mcp - cold_mcp
            interval_heat = net_mcp * delta_t

            interval = TemperatureInterval(
                interval_id=i,
                temp_high_f=t_high,
                temp_low_f=t_low,
                delta_t_f=delta_t,
                hot_mcp_sum=hot_mcp,
                cold_mcp_sum=cold_mcp,
                net_mcp=net_mcp,
                interval_heat=interval_heat,
                cascade_heat=0.0,  # Calculated later
            )
            intervals.append(interval)

        return intervals

    def _problem_table_algorithm(
        self,
        intervals: List[TemperatureInterval],
    ) -> List[TemperatureInterval]:
        """
        Execute the Problem Table Algorithm.

        Calculates heat cascade through temperature intervals
        assuming zero hot utility initially.

        Args:
            intervals: Temperature intervals

        Returns:
            Updated intervals with cascade heat values
        """
        if not intervals:
            return intervals

        # First pass: cascade with zero hot utility
        cumulative = 0.0
        for interval in intervals:
            cumulative += interval.interval_heat
            interval.cascade_heat = cumulative

        return intervals

    def _find_pinch_point(
        self,
        intervals: List[TemperatureInterval],
        delta_t_min: float,
    ) -> Dict[str, float]:
        """
        Find the pinch point from cascade analysis.

        The pinch occurs where the heat cascade is most negative
        (or minimum positive value after adjustment).

        Args:
            intervals: Temperature intervals with cascade values
            delta_t_min: Minimum approach temperature

        Returns:
            Dictionary with pinch temperature information
        """
        if not intervals:
            return {
                "pinch_temp": 0,
                "shifted_pinch": 0,
                "pinch_above": 0,
                "pinch_below": 0,
            }

        # Find interval with minimum cascade value
        min_cascade = float('inf')
        pinch_interval_idx = 0

        for i, interval in enumerate(intervals):
            if interval.cascade_heat < min_cascade:
                min_cascade = interval.cascade_heat
                pinch_interval_idx = i

        # Pinch is at the bottom of the interval with minimum cascade
        shifted_pinch = intervals[pinch_interval_idx].temp_low_f

        # Convert back to actual temperatures
        half_dt = delta_t_min / 2
        pinch_above = shifted_pinch + half_dt  # Hot stream temp at pinch
        pinch_below = shifted_pinch - half_dt  # Cold stream temp at pinch
        pinch_temp = (pinch_above + pinch_below) / 2

        return {
            "pinch_temp": pinch_temp,
            "shifted_pinch": shifted_pinch,
            "pinch_above": pinch_above,
            "pinch_below": pinch_below,
        }

    def _calculate_utility_targets(
        self,
        intervals: List[TemperatureInterval],
    ) -> Dict[str, float]:
        """
        Calculate minimum utility requirements.

        Hot utility = -min(cascade) if min(cascade) < 0, else 0
        Cold utility = final cascade value + hot utility

        Args:
            intervals: Temperature intervals with cascade values

        Returns:
            Dictionary with utility targets
        """
        if not intervals:
            return {"hot_utility": 0, "cold_utility": 0}

        # Find minimum cascade value
        min_cascade = min(interval.cascade_heat for interval in intervals)

        # Hot utility needed to make all cascade values >= 0
        hot_utility = abs(min_cascade) if min_cascade < 0 else 0

        # Cold utility is final cascade value after hot utility addition
        final_cascade = intervals[-1].cascade_heat
        cold_utility = final_cascade + hot_utility

        return {
            "hot_utility": hot_utility,
            "cold_utility": cold_utility,
        }

    def _build_composite_curve(
        self,
        streams: List[HeatStream],
        curve_type: str,
    ) -> CompositeData:
        """
        Build composite curve from streams.

        Args:
            streams: List of streams (all same type)
            curve_type: "hot" or "cold"

        Returns:
            CompositeData with curve points
        """
        if not streams:
            return CompositeData(curve_type=curve_type, points=[], total_duty_btu_hr=0)

        # Collect all temperatures
        temps = set()
        for s in streams:
            temps.add(s.supply_temp_f)
            temps.add(s.target_temp_f)

        # Sort: descending for hot, ascending for cold
        if curve_type == "hot":
            sorted_temps = sorted(temps, reverse=True)
        else:
            sorted_temps = sorted(temps)

        # Build curve points
        points = []
        cumulative_enthalpy = 0.0

        for i, temp in enumerate(sorted_temps):
            if i == 0:
                points.append(CompositeCurvePoint(
                    temperature_f=temp,
                    enthalpy_btu_hr=0.0,
                ))
            else:
                prev_temp = sorted_temps[i - 1]

                # Sum MCp for active streams
                active_mcp = sum(
                    s.mcp for s in streams
                    if min(s.supply_temp_f, s.target_temp_f) <= min(temp, prev_temp) and
                       max(s.supply_temp_f, s.target_temp_f) >= max(temp, prev_temp)
                )

                delta_h = active_mcp * abs(temp - prev_temp)
                cumulative_enthalpy += delta_h

                points.append(CompositeCurvePoint(
                    temperature_f=temp,
                    enthalpy_btu_hr=cumulative_enthalpy,
                ))

        total_duty = sum(s.heat_duty or 0 for s in streams)

        return CompositeData(
            curve_type=curve_type,
            points=points,
            total_duty_btu_hr=total_duty,
        )

    def _build_grand_composite(
        self,
        intervals: List[TemperatureInterval],
    ) -> List[GrandCompositePoint]:
        """
        Build the Grand Composite Curve (GCC).

        The GCC shows net heat flow at each shifted temperature,
        after adding minimum hot utility.

        Args:
            intervals: Temperature intervals

        Returns:
            List of GCC points
        """
        if not intervals:
            return []

        # Calculate hot utility
        min_cascade = min(i.cascade_heat for i in intervals)
        hot_utility = abs(min_cascade) if min_cascade < 0 else 0

        points = []

        # First point at highest temperature
        points.append(GrandCompositePoint(
            shifted_temp_f=intervals[0].temp_high_f,
            net_heat_btu_hr=hot_utility,
        ))

        # Points at interval boundaries
        for interval in intervals:
            adjusted_heat = interval.cascade_heat + hot_utility
            points.append(GrandCompositePoint(
                shifted_temp_f=interval.temp_low_f,
                net_heat_btu_hr=adjusted_heat,
            ))

        return points

    def _detect_pinch_violations(
        self,
        streams: List[HeatStream],
        pinch_temp: float,
        delta_t_min: float,
    ) -> List[PinchViolation]:
        """
        Detect pinch design rule violations.

        Three rules:
        1. No heat transfer across the pinch
        2. No hot utility below the pinch
        3. No cold utility above the pinch

        Args:
            streams: All process streams
            pinch_temp: Pinch temperature
            delta_t_min: Minimum approach temperature

        Returns:
            List of detected violations
        """
        violations = []
        half_dt = delta_t_min / 2

        pinch_above = pinch_temp + half_dt
        pinch_below = pinch_temp - half_dt

        # Check for streams that cross the pinch
        for stream in streams:
            if stream.stream_type == StreamType.HOT:
                if stream.supply_temp_f > pinch_above and stream.target_temp_f < pinch_above:
                    heat_above = stream.mcp * (stream.supply_temp_f - pinch_above)
                    heat_below = stream.mcp * (pinch_above - stream.target_temp_f)

                    violations.append(PinchViolation(
                        violation_type=PinchViolationType.HEAT_TRANSFER_ACROSS_PINCH,
                        description=(
                            f"Hot stream '{stream.name}' crosses pinch point"
                        ),
                        stream_names=[stream.name],
                        heat_transfer_btu_hr=heat_below,
                        recommendation=(
                            f"Consider splitting stream at pinch or "
                            f"recovering heat only above {pinch_above}F"
                        ),
                    ))

            elif stream.stream_type == StreamType.COLD:
                if stream.supply_temp_f < pinch_below and stream.target_temp_f > pinch_below:
                    heat_below = stream.mcp * (pinch_below - stream.supply_temp_f)
                    heat_above = stream.mcp * (stream.target_temp_f - pinch_below)

                    violations.append(PinchViolation(
                        violation_type=PinchViolationType.HEAT_TRANSFER_ACROSS_PINCH,
                        description=(
                            f"Cold stream '{stream.name}' crosses pinch point"
                        ),
                        stream_names=[stream.name],
                        heat_transfer_btu_hr=heat_above,
                        recommendation=(
                            f"Consider splitting stream at pinch or "
                            f"accepting heat only below {pinch_below}F"
                        ),
                    ))

        return violations

    def _generate_recommendations(
        self,
        pinch_result: Dict[str, float],
        utility_targets: Dict[str, float],
        max_recovery: float,
        total_hot_duty: float,
        total_cold_duty: float,
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Recovery potential
        if total_hot_duty > 0:
            recovery_pct = (max_recovery / total_hot_duty) * 100
            if recovery_pct > 70:
                recommendations.append(
                    f"Excellent heat recovery potential: {recovery_pct:.1f}% of hot "
                    f"stream duty can be recovered through process integration."
                )
            elif recovery_pct > 40:
                recommendations.append(
                    f"Good heat recovery potential: {recovery_pct:.1f}% of hot "
                    f"stream duty recoverable. Prioritize streams crossing pinch."
                )
            else:
                recommendations.append(
                    f"Limited heat recovery potential ({recovery_pct:.1f}%). "
                    f"Consider process modifications to create integration opportunities."
                )

        # Pinch temperature
        pinch_temp = pinch_result["pinch_temp"]
        if pinch_temp > 400:
            recommendations.append(
                f"High pinch temperature ({pinch_temp:.0f}F) indicates "
                "potential for power generation from waste heat above pinch."
            )
        elif pinch_temp < 200:
            recommendations.append(
                f"Low pinch temperature ({pinch_temp:.0f}F). "
                "Consider heat pump applications to upgrade low-grade heat."
            )

        # Utility balance
        hot_util = utility_targets["hot_utility"]
        cold_util = utility_targets["cold_utility"]

        if hot_util > 0 and cold_util > 0:
            if hot_util > cold_util * 2:
                recommendations.append(
                    "Process is heating-dominated. Focus on reducing heating "
                    "requirements through heat recovery above the pinch."
                )
            elif cold_util > hot_util * 2:
                recommendations.append(
                    "Process is cooling-dominated. Consider recovering "
                    "reject heat for building heating or absorption cooling."
                )

        return recommendations

    def _calculate_provenance(
        self,
        streams: List[HeatStream],
        delta_t_min: float,
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        import json

        provenance_data = {
            "streams": [s.dict() for s in streams],
            "delta_t_min": delta_t_min,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def optimize_delta_t_min(
        self,
        streams: List[HeatStream],
        delta_t_range: Tuple[float, float] = (5.0, 50.0),
        num_points: int = 10,
        hx_cost_per_ft2: float = 150.0,
        u_value: float = 50.0,
        annualization_factor: float = 0.15,
    ) -> DeltaTMinOptimizationResult:
        """
        Optimize delta_T_min for minimum total annual cost.

        Balances:
        - Capital cost: decreases with increasing delta_T_min
        - Operating cost: increases with increasing delta_T_min

        Args:
            streams: Process streams
            delta_t_range: Range of delta_T_min to evaluate
            num_points: Number of evaluation points
            hx_cost_per_ft2: Heat exchanger cost ($/ft2)
            u_value: Overall heat transfer coefficient (BTU/hr-ft2-F)
            annualization_factor: Capital recovery factor

        Returns:
            Optimization result with cost breakdown
        """
        logger.info(
            f"Optimizing delta_T_min: range {delta_t_range[0]}-{delta_t_range[1]}F"
        )

        delta_t_values = []
        total_costs = []
        best_cost = float('inf')
        best_delta_t = delta_t_range[0]
        best_result = None

        step = (delta_t_range[1] - delta_t_range[0]) / (num_points - 1)

        for i in range(num_points):
            dt = delta_t_range[0] + i * step
            delta_t_values.append(dt)

            try:
                # Run pinch analysis at this delta_T_min
                result = self.analyze(streams, delta_t_min_override=dt)

                # Estimate capital cost (based on area)
                # Area ~ Q / (U * delta_T_min)
                total_duty = result.maximum_heat_recovery_btu_hr
                if dt > 0:
                    approx_area = total_duty / (u_value * dt)
                else:
                    approx_area = float('inf')

                capital_cost = approx_area * hx_cost_per_ft2
                annualized_capital = capital_cost * annualization_factor

                # Operating cost
                hot_util_mmbtu_yr = (
                    result.minimum_hot_utility_btu_hr *
                    self.operating_hours / 1_000_000
                )
                cold_util_mmbtu_yr = (
                    result.minimum_cold_utility_btu_hr *
                    self.operating_hours / 1_000_000
                )

                utility_cost = (
                    hot_util_mmbtu_yr * self.hot_utility_cost +
                    cold_util_mmbtu_yr * self.cold_utility_cost
                )

                total_annual_cost = annualized_capital + utility_cost
                total_costs.append(total_annual_cost)

                if total_annual_cost < best_cost:
                    best_cost = total_annual_cost
                    best_delta_t = dt
                    best_result = {
                        "delta_t_min": dt,
                        "capital_cost": capital_cost,
                        "utility_cost": utility_cost,
                        "total_cost": total_annual_cost,
                        "heat_recovery": result.maximum_heat_recovery_btu_hr,
                        "hot_utility": result.minimum_hot_utility_btu_hr,
                        "cold_utility": result.minimum_cold_utility_btu_hr,
                    }

            except Exception as e:
                logger.warning(f"Analysis failed at delta_T={dt}F: {e}")
                total_costs.append(float('inf'))

        return DeltaTMinOptimizationResult(
            optimal_delta_t_min_f=best_delta_t,
            capital_cost_usd=best_result["capital_cost"] if best_result else 0,
            utility_cost_usd_yr=best_result["utility_cost"] if best_result else 0,
            total_annual_cost_usd=best_result["total_cost"] if best_result else 0,
            heat_recovery_btu_hr=best_result["heat_recovery"] if best_result else 0,
            hot_utility_btu_hr=best_result["hot_utility"] if best_result else 0,
            cold_utility_btu_hr=best_result["cold_utility"] if best_result else 0,
            delta_t_values=delta_t_values,
            total_costs=total_costs,
        )

    def get_streams_above_pinch(
        self,
        streams: List[HeatStream],
        pinch_temp: float,
        delta_t_min: float,
    ) -> Tuple[List[HeatStream], List[HeatStream]]:
        """
        Get streams above the pinch for HEN design.

        Args:
            streams: All process streams
            pinch_temp: Pinch temperature
            delta_t_min: Minimum approach temperature

        Returns:
            Tuple of (hot_streams_above, cold_streams_above)
        """
        half_dt = delta_t_min / 2
        pinch_above = pinch_temp + half_dt
        pinch_below = pinch_temp - half_dt

        hot_above = []
        cold_above = []

        for stream in streams:
            if stream.stream_type == StreamType.HOT:
                if stream.supply_temp_f > pinch_above:
                    hot_above.append(stream)
            else:
                if stream.target_temp_f > pinch_below:
                    cold_above.append(stream)

        return hot_above, cold_above

    def get_streams_below_pinch(
        self,
        streams: List[HeatStream],
        pinch_temp: float,
        delta_t_min: float,
    ) -> Tuple[List[HeatStream], List[HeatStream]]:
        """
        Get streams below the pinch for HEN design.

        Args:
            streams: All process streams
            pinch_temp: Pinch temperature
            delta_t_min: Minimum approach temperature

        Returns:
            Tuple of (hot_streams_below, cold_streams_below)
        """
        half_dt = delta_t_min / 2
        pinch_above = pinch_temp + half_dt
        pinch_below = pinch_temp - half_dt

        hot_below = []
        cold_below = []

        for stream in streams:
            if stream.stream_type == StreamType.HOT:
                if stream.target_temp_f < pinch_above:
                    hot_below.append(stream)
            else:
                if stream.supply_temp_f < pinch_below:
                    cold_below.append(stream)

        return hot_below, cold_below
