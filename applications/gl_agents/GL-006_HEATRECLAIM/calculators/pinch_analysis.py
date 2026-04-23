"""
GL-006 HEATRECLAIM - Pinch Analysis Calculator

Implements deterministic pinch analysis for minimum utility targeting
and composite curve generation. All calculations are reproducible
with SHA-256 provenance tracking.

Reference: Linnhoff et al., "User Guide on Process Integration for
the Efficient Use of Energy", IChemE, 1982.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

from ..core.schemas import (
    HeatStream,
    PinchAnalysisResult,
    CompositePoint,
)
from ..core.config import StreamType

logger = logging.getLogger(__name__)


@dataclass
class TemperatureInterval:
    """Temperature interval for problem table algorithm."""

    T_hot: float  # Hot stream temperature (shifted)
    T_cold: float  # Cold stream temperature (shifted)
    delta_T: float  # Temperature span
    hot_streams: List[str]  # Hot streams in this interval
    cold_streams: List[str]  # Cold streams in this interval
    FCp_hot: float  # Sum of FCp for hot streams
    FCp_cold: float  # Sum of FCp for cold streams
    heat_deficit: float  # FCp_cold - FCp_hot


class PinchAnalysisCalculator:
    """
    Pinch analysis calculator implementing the problem table algorithm.

    Provides deterministic, reproducible calculations for:
    - Minimum hot utility (QH,min)
    - Minimum cold utility (QC,min)
    - Maximum heat recovery
    - Pinch temperature
    - Composite curves
    - Grand composite curve

    Example:
        >>> calculator = PinchAnalysisCalculator(delta_t_min=10.0)
        >>> result = calculator.calculate(hot_streams, cold_streams)
        >>> print(f"Pinch: {result.pinch_temperature_C}°C")
        >>> print(f"Min hot utility: {result.minimum_hot_utility_kW} kW")
    """

    VERSION = "1.0.0"
    FORMULA_VERSION = "PINCH_PROBLEM_TABLE_v1.0"

    def __init__(
        self,
        delta_t_min: float = 10.0,
        tolerance: float = 1e-6,
    ) -> None:
        """
        Initialize pinch analysis calculator.

        Args:
            delta_t_min: Minimum approach temperature (°C)
            tolerance: Numerical tolerance for comparisons
        """
        self.delta_t_min = delta_t_min
        self.tolerance = tolerance

    def calculate(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        delta_t_min: Optional[float] = None,
    ) -> PinchAnalysisResult:
        """
        Perform complete pinch analysis.

        Args:
            hot_streams: List of hot process streams
            cold_streams: List of cold process streams
            delta_t_min: Override default ΔTmin

        Returns:
            PinchAnalysisResult with all pinch targets and curves
        """
        dt_min = delta_t_min if delta_t_min is not None else self.delta_t_min

        # Validate inputs
        self._validate_streams(hot_streams, cold_streams)

        # Compute input hash for provenance
        input_hash = self._compute_hash({
            "hot_streams": [s.dict() for s in hot_streams],
            "cold_streams": [s.dict() for s in cold_streams],
            "delta_t_min": dt_min,
        })

        # Step 1: Create shifted temperature intervals
        intervals = self._create_temperature_intervals(
            hot_streams, cold_streams, dt_min
        )

        # Step 2: Calculate heat cascade (problem table)
        cascade, pinch_temp, qh_min, qc_min = self._calculate_heat_cascade(
            intervals, dt_min
        )

        # Step 3: Calculate maximum heat recovery
        total_hot_duty = sum(s.duty_kW for s in hot_streams)
        total_cold_duty = sum(s.duty_kW for s in cold_streams)
        max_recovery = min(total_hot_duty, total_cold_duty) - max(0, qh_min)

        # Step 4: Generate composite curves
        hot_composite = self._generate_composite_curve(hot_streams, "hot")
        cold_composite = self._generate_composite_curve(cold_streams, "cold")

        # Step 5: Generate grand composite curve
        grand_composite = self._generate_grand_composite(cascade, dt_min)

        # Build result
        result = PinchAnalysisResult(
            pinch_temperature_C=round(pinch_temp, 2),
            delta_t_min_C=dt_min,
            minimum_hot_utility_kW=round(max(0, qh_min), 2),
            minimum_cold_utility_kW=round(max(0, qc_min), 2),
            maximum_heat_recovery_kW=round(max(0, max_recovery), 2),
            hot_composite=hot_composite,
            cold_composite=cold_composite,
            grand_composite=grand_composite,
            hot_streams=[s.stream_id for s in hot_streams],
            cold_streams=[s.stream_id for s in cold_streams],
            heat_cascade=cascade,
            input_hash=input_hash,
            formula_version=self.FORMULA_VERSION,
        )

        # Compute output hash
        result.output_hash = self._compute_hash({
            "pinch_temperature_C": result.pinch_temperature_C,
            "minimum_hot_utility_kW": result.minimum_hot_utility_kW,
            "minimum_cold_utility_kW": result.minimum_cold_utility_kW,
        })

        # Validate result
        result.is_valid = self._validate_result(result, hot_streams, cold_streams)

        logger.info(
            f"Pinch analysis complete: Tpinch={result.pinch_temperature_C}°C, "
            f"QH_min={result.minimum_hot_utility_kW}kW, "
            f"QC_min={result.minimum_cold_utility_kW}kW"
        )

        return result

    def _validate_streams(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
    ) -> None:
        """Validate input streams."""
        if not hot_streams:
            raise ValueError("At least one hot stream required")
        if not cold_streams:
            raise ValueError("At least one cold stream required")

        for stream in hot_streams:
            if stream.T_supply_C <= stream.T_target_C:
                raise ValueError(
                    f"Hot stream {stream.stream_id} must cool down: "
                    f"T_supply ({stream.T_supply_C}) > T_target ({stream.T_target_C})"
                )

        for stream in cold_streams:
            if stream.T_supply_C >= stream.T_target_C:
                raise ValueError(
                    f"Cold stream {stream.stream_id} must heat up: "
                    f"T_supply ({stream.T_supply_C}) < T_target ({stream.T_target_C})"
                )

    def _create_temperature_intervals(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        delta_t_min: float,
    ) -> List[TemperatureInterval]:
        """
        Create temperature intervals using shifted temperatures.

        Hot streams are shifted down by ΔTmin/2.
        Cold streams are shifted up by ΔTmin/2.
        """
        shift = delta_t_min / 2.0

        # Collect all shifted temperatures
        temps = set()

        for s in hot_streams:
            temps.add(s.T_supply_C - shift)
            temps.add(s.T_target_C - shift)

        for s in cold_streams:
            temps.add(s.T_supply_C + shift)
            temps.add(s.T_target_C + shift)

        # Sort temperatures in descending order
        sorted_temps = sorted(temps, reverse=True)

        # Create intervals
        intervals = []
        for i in range(len(sorted_temps) - 1):
            T_high = sorted_temps[i]
            T_low = sorted_temps[i + 1]
            delta_T = T_high - T_low

            if delta_T < self.tolerance:
                continue

            # Find streams active in this interval
            hot_in_interval = []
            FCp_hot = 0.0
            for s in hot_streams:
                T_s_shifted = s.T_supply_C - shift
                T_t_shifted = s.T_target_C - shift
                if T_s_shifted >= T_high - self.tolerance and T_t_shifted <= T_low + self.tolerance:
                    if T_t_shifted <= T_high and T_s_shifted >= T_low:
                        hot_in_interval.append(s.stream_id)
                        FCp_hot += s.FCp_kW_K

            cold_in_interval = []
            FCp_cold = 0.0
            for s in cold_streams:
                T_s_shifted = s.T_supply_C + shift
                T_t_shifted = s.T_target_C + shift
                if T_s_shifted <= T_low + self.tolerance and T_t_shifted >= T_high - self.tolerance:
                    if T_s_shifted <= T_high and T_t_shifted >= T_low:
                        cold_in_interval.append(s.stream_id)
                        FCp_cold += s.FCp_kW_K

            heat_deficit = (FCp_cold - FCp_hot) * delta_T

            intervals.append(TemperatureInterval(
                T_hot=T_high + shift,
                T_cold=T_low + shift,
                delta_T=delta_T,
                hot_streams=hot_in_interval,
                cold_streams=cold_in_interval,
                FCp_hot=FCp_hot,
                FCp_cold=FCp_cold,
                heat_deficit=heat_deficit,
            ))

        return intervals

    def _calculate_heat_cascade(
        self,
        intervals: List[TemperatureInterval],
        delta_t_min: float,
    ) -> Tuple[List[Dict[str, float]], float, float, float]:
        """
        Calculate heat cascade using problem table algorithm.

        Returns:
            Tuple of (cascade_data, pinch_temperature, QH_min, QC_min)
        """
        if not intervals:
            return [], 0.0, 0.0, 0.0

        # First pass: cascade with zero utility
        cascade = []
        cumulative_heat = 0.0
        min_cascade = 0.0
        min_cascade_idx = 0

        for i, interval in enumerate(intervals):
            cumulative_heat += interval.heat_deficit
            cascade.append({
                "interval": i,
                "T_hot": interval.T_hot,
                "T_cold": interval.T_cold,
                "heat_deficit": interval.heat_deficit,
                "cumulative": cumulative_heat,
            })

            if cumulative_heat < min_cascade:
                min_cascade = cumulative_heat
                min_cascade_idx = i

        # QH_min is the negative of the minimum cascade value
        qh_min = -min_cascade if min_cascade < 0 else 0.0

        # Second pass: add QH_min to cascade
        adjusted_cascade = []
        cumulative = qh_min

        for i, interval in enumerate(intervals):
            cumulative += interval.heat_deficit
            adjusted_cascade.append({
                "interval": i,
                "T_hot": round(interval.T_hot, 2),
                "T_cold": round(interval.T_cold, 2),
                "heat_deficit": round(interval.heat_deficit, 2),
                "cumulative": round(cumulative, 2),
            })

        # QC_min is the final cascade value
        qc_min = cumulative

        # Pinch temperature is where cascade crosses zero (or is minimum)
        # Find interval where cumulative first becomes positive after being zero/negative
        pinch_temp = intervals[0].T_hot if intervals else 0.0

        cumulative = qh_min
        for i, interval in enumerate(intervals):
            prev_cumulative = cumulative
            cumulative += interval.heat_deficit

            if abs(cumulative) < self.tolerance or \
               (prev_cumulative * cumulative < 0):  # Sign change
                # Pinch is at the cold end of this interval
                pinch_temp = interval.T_cold
                break

        return adjusted_cascade, pinch_temp, qh_min, qc_min

    def _generate_composite_curve(
        self,
        streams: List[HeatStream],
        stream_type: str,
    ) -> List[CompositePoint]:
        """
        Generate composite curve for a set of streams.

        For hot streams: plot T vs cumulative heat released
        For cold streams: plot T vs cumulative heat absorbed
        """
        if not streams:
            return []

        # Collect all temperature breakpoints
        temps = set()
        for s in streams:
            temps.add(s.T_supply_C)
            temps.add(s.T_target_C)

        sorted_temps = sorted(temps, reverse=(stream_type == "hot"))

        # Build composite curve
        points = []
        cumulative_enthalpy = 0.0

        for i, T in enumerate(sorted_temps):
            # Find streams active at this temperature
            active_streams = []
            for s in streams:
                T_min = min(s.T_supply_C, s.T_target_C)
                T_max = max(s.T_supply_C, s.T_target_C)
                if T_min <= T <= T_max:
                    active_streams.append(s.stream_id)

            points.append(CompositePoint(
                temperature_C=round(T, 2),
                enthalpy_kW=round(cumulative_enthalpy, 2),
                stream_ids=active_streams,
            ))

            # Calculate enthalpy change to next temperature
            if i < len(sorted_temps) - 1:
                T_next = sorted_temps[i + 1]
                delta_T = abs(T_next - T)

                FCp_total = 0.0
                for s in streams:
                    T_min = min(s.T_supply_C, s.T_target_C)
                    T_max = max(s.T_supply_C, s.T_target_C)
                    T_mid = (T + T_next) / 2
                    if T_min <= T_mid <= T_max:
                        FCp_total += s.FCp_kW_K

                cumulative_enthalpy += FCp_total * delta_T

        # Add final point
        if points:
            points.append(CompositePoint(
                temperature_C=round(sorted_temps[-1], 2),
                enthalpy_kW=round(cumulative_enthalpy, 2),
                stream_ids=[],
            ))

        return points

    def _generate_grand_composite(
        self,
        cascade: List[Dict[str, float]],
        delta_t_min: float,
    ) -> List[CompositePoint]:
        """Generate grand composite curve from heat cascade."""
        points = []

        for entry in cascade:
            # Use shifted temperature (average of hot and cold)
            T_shifted = (entry["T_hot"] + entry["T_cold"]) / 2 - delta_t_min / 2
            points.append(CompositePoint(
                temperature_C=round(T_shifted, 2),
                enthalpy_kW=round(entry["cumulative"], 2),
                stream_ids=[],
            ))

        return points

    def _validate_result(
        self,
        result: PinchAnalysisResult,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
    ) -> bool:
        """Validate pinch analysis results."""
        messages = []

        # Check energy balance
        total_hot = sum(s.duty_kW for s in hot_streams)
        total_cold = sum(s.duty_kW for s in cold_streams)

        # Hot duty + QH = Cold duty + QC
        balance_lhs = total_hot + result.minimum_hot_utility_kW
        balance_rhs = total_cold + result.minimum_cold_utility_kW

        if abs(balance_lhs - balance_rhs) > 1.0:  # 1 kW tolerance
            messages.append(
                f"Energy balance error: {balance_lhs:.1f} != {balance_rhs:.1f}"
            )

        # Check non-negative utilities
        if result.minimum_hot_utility_kW < -self.tolerance:
            messages.append("Hot utility is negative")

        if result.minimum_cold_utility_kW < -self.tolerance:
            messages.append("Cold utility is negative")

        result.validation_messages = messages
        return len(messages) == 0

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def calculate_minimum_utilities(
    hot_streams: List[HeatStream],
    cold_streams: List[HeatStream],
    delta_t_min: float = 10.0,
) -> Tuple[float, float, float]:
    """
    Convenience function for quick pinch targeting.

    Returns:
        Tuple of (QH_min, QC_min, pinch_temperature)
    """
    calculator = PinchAnalysisCalculator(delta_t_min=delta_t_min)
    result = calculator.calculate(hot_streams, cold_streams)
    return (
        result.minimum_hot_utility_kW,
        result.minimum_cold_utility_kW,
        result.pinch_temperature_C,
    )
