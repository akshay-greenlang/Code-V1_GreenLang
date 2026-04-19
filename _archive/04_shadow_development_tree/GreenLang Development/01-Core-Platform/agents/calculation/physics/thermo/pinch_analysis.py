"""
Pinch Analysis for Heat Integration

Zero-Hallucination Process Integration Calculations

This module implements deterministic pinch analysis for:
- Composite curve generation
- Grand composite curve
- Minimum utility targets
- Heat exchanger network design
- Area targeting

References:
    - Linnhoff, B. et al. "A User Guide on Process Integration for the
      Efficient Use of Energy" (IChemE, 1982)
    - Smith, R. "Chemical Process Design and Integration" (2005)
    - Kemp, I.C. "Pinch Analysis and Process Integration" (2007)

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
import hashlib


@dataclass
class StreamData:
    """
    Process stream data for pinch analysis.

    Attributes:
        name: Stream identifier
        supply_temp_c: Supply (starting) temperature
        target_temp_c: Target (final) temperature
        heat_capacity_flow_kw_k: Mass flow * Cp (kW/K)
        h_coefficient_kw_m2k: Film heat transfer coefficient (optional)
    """
    name: str
    supply_temp_c: float
    target_temp_c: float
    heat_capacity_flow_kw_k: float
    h_coefficient_kw_m2k: Optional[float] = None

    @property
    def is_hot(self) -> bool:
        """Return True if hot stream (needs cooling)."""
        return self.supply_temp_c > self.target_temp_c

    @property
    def heat_load_kw(self) -> Decimal:
        """Calculate total heat load."""
        dt = abs(Decimal(str(self.supply_temp_c)) - Decimal(str(self.target_temp_c)))
        return Decimal(str(self.heat_capacity_flow_kw_k)) * dt


@dataclass
class PinchResult:
    """
    Pinch analysis results with complete provenance.

    All values are deterministic - same inputs produce identical outputs.
    """
    pinch_temperature_c: Decimal
    minimum_hot_utility_kw: Decimal
    minimum_cold_utility_kw: Decimal
    heat_recovery_kw: Decimal
    hot_composite_curve: List[Tuple[Decimal, Decimal]]  # (Enthalpy, Temperature)
    cold_composite_curve: List[Tuple[Decimal, Decimal]]
    grand_composite_curve: List[Tuple[Decimal, Decimal]]  # (Net heat, Temperature)
    interval_data: List[Dict]
    dt_min_c: Decimal
    provenance_hash: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "pinch_temperature_c": float(self.pinch_temperature_c),
            "minimum_hot_utility_kw": float(self.minimum_hot_utility_kw),
            "minimum_cold_utility_kw": float(self.minimum_cold_utility_kw),
            "heat_recovery_kw": float(self.heat_recovery_kw),
            "dt_min_c": float(self.dt_min_c),
            "provenance_hash": self.provenance_hash
        }


@dataclass
class AreaTarget:
    """Heat exchanger network area targets."""
    total_area_m2: Decimal
    area_above_pinch_m2: Decimal
    area_below_pinch_m2: Decimal
    number_of_units: int
    provenance_hash: str


class PinchAnalyzer:
    """
    Pinch analysis calculator for process heat integration.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations are deterministic
    - Based on established algorithms
    - Complete provenance tracking

    References:
        - Linnhoff, B. "The Pinch Design Method for Heat Exchanger Networks"
        - Smith, R. "Chemical Process Design and Integration", Chapter 16
    """

    def __init__(self, precision: int = 2):
        """Initialize pinch analyzer."""
        self.precision = precision

    def _apply_precision(self, value: Decimal) -> Decimal:
        """Apply precision rounding."""
        if self.precision == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * self.precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _calculate_provenance(self, inputs: Dict, outputs: Dict) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "method": "Pinch_Analysis_Problem_Table",
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def analyze(
        self,
        streams: List[StreamData],
        dt_min_c: float
    ) -> PinchResult:
        """
        Perform complete pinch analysis.

        ZERO-HALLUCINATION: Deterministic Problem Table Algorithm.

        Reference: Linnhoff & Flower (1978) "Synthesis of Heat Exchanger Networks"

        Args:
            streams: List of process streams
            dt_min_c: Minimum approach temperature (C)

        Returns:
            PinchResult with targets and curves
        """
        dt_min = Decimal(str(dt_min_c))

        # Separate hot and cold streams
        hot_streams = [s for s in streams if s.is_hot]
        cold_streams = [s for s in streams if not s.is_hot]

        if not hot_streams or not cold_streams:
            raise ValueError("Need at least one hot and one cold stream")

        # Step 1: Create shifted temperatures
        # Hot streams: shift down by dt_min/2
        # Cold streams: shift up by dt_min/2
        shifted_temps = set()

        for s in hot_streams:
            shifted_temps.add(Decimal(str(s.supply_temp_c)) - dt_min / Decimal("2"))
            shifted_temps.add(Decimal(str(s.target_temp_c)) - dt_min / Decimal("2"))

        for s in cold_streams:
            shifted_temps.add(Decimal(str(s.supply_temp_c)) + dt_min / Decimal("2"))
            shifted_temps.add(Decimal(str(s.target_temp_c)) + dt_min / Decimal("2"))

        # Sort temperatures in descending order
        temp_intervals = sorted(shifted_temps, reverse=True)

        # Step 2: Problem Table Algorithm
        interval_data = []
        cascade_heat = []
        cumulative_heat = Decimal("0")

        for i in range(len(temp_intervals) - 1):
            t_upper = temp_intervals[i]
            t_lower = temp_intervals[i + 1]
            dt_interval = t_upper - t_lower

            # Calculate sum of MCp for streams active in this interval
            sum_mcp_hot = Decimal("0")
            sum_mcp_cold = Decimal("0")

            for s in hot_streams:
                t_supply_shifted = Decimal(str(s.supply_temp_c)) - dt_min / Decimal("2")
                t_target_shifted = Decimal(str(s.target_temp_c)) - dt_min / Decimal("2")
                if t_supply_shifted >= t_upper and t_target_shifted <= t_lower:
                    sum_mcp_hot += Decimal(str(s.heat_capacity_flow_kw_k))

            for s in cold_streams:
                t_supply_shifted = Decimal(str(s.supply_temp_c)) + dt_min / Decimal("2")
                t_target_shifted = Decimal(str(s.target_temp_c)) + dt_min / Decimal("2")
                if t_target_shifted >= t_upper and t_supply_shifted <= t_lower:
                    sum_mcp_cold += Decimal(str(s.heat_capacity_flow_kw_k))

            # Net heat in interval (positive = surplus from hot streams)
            delta_h = (sum_mcp_hot - sum_mcp_cold) * dt_interval

            interval_data.append({
                "interval": i + 1,
                "t_upper_c": float(t_upper),
                "t_lower_c": float(t_lower),
                "dt_c": float(dt_interval),
                "sum_mcp_hot": float(sum_mcp_hot),
                "sum_mcp_cold": float(sum_mcp_cold),
                "delta_h_kw": float(delta_h)
            })

            cumulative_heat += delta_h
            cascade_heat.append(cumulative_heat)

        # Step 3: Find pinch point (where cascade heat is minimum)
        min_cascade = min(cascade_heat)
        pinch_index = cascade_heat.index(min_cascade)

        # Hot utility required (to make cascade non-negative)
        hot_utility = -min_cascade if min_cascade < Decimal("0") else Decimal("0")

        # Adjust cascade with hot utility
        adjusted_cascade = [h + hot_utility for h in cascade_heat]

        # Cold utility is final cascade value
        cold_utility = adjusted_cascade[-1] if adjusted_cascade else Decimal("0")

        # Pinch temperature (shifted)
        pinch_temp_shifted = temp_intervals[pinch_index + 1]

        # Real pinch temperature (hot side)
        pinch_temp = pinch_temp_shifted + dt_min / Decimal("2")

        # Calculate total heat loads
        total_hot_load = sum(s.heat_load_kw for s in hot_streams)
        total_cold_load = sum(s.heat_load_kw for s in cold_streams)

        # Heat recovery = min of total loads minus utilities
        heat_recovery = total_hot_load - hot_utility

        # Step 4: Generate composite curves
        hot_composite = self._generate_composite_curve(hot_streams, is_hot=True)
        cold_composite = self._generate_composite_curve(cold_streams, is_hot=False)

        # Step 5: Generate Grand Composite Curve
        gcc = self._generate_grand_composite(temp_intervals, adjusted_cascade)

        # Create provenance
        inputs = {
            "num_hot_streams": len(hot_streams),
            "num_cold_streams": len(cold_streams),
            "dt_min_c": str(dt_min),
            "stream_names": [s.name for s in streams]
        }
        outputs = {
            "pinch_temp": str(pinch_temp),
            "hot_utility": str(hot_utility),
            "cold_utility": str(cold_utility)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return PinchResult(
            pinch_temperature_c=self._apply_precision(pinch_temp),
            minimum_hot_utility_kw=self._apply_precision(hot_utility),
            minimum_cold_utility_kw=self._apply_precision(cold_utility),
            heat_recovery_kw=self._apply_precision(heat_recovery),
            hot_composite_curve=hot_composite,
            cold_composite_curve=cold_composite,
            grand_composite_curve=gcc,
            interval_data=interval_data,
            dt_min_c=self._apply_precision(dt_min),
            provenance_hash=provenance_hash
        )

    def _generate_composite_curve(
        self,
        streams: List[StreamData],
        is_hot: bool
    ) -> List[Tuple[Decimal, Decimal]]:
        """
        Generate composite curve for hot or cold streams.

        Reference: Smith, R. "Chemical Process Design", Chapter 16.3
        """
        if not streams:
            return []

        # Collect all temperature levels
        temps = set()
        for s in streams:
            temps.add(Decimal(str(s.supply_temp_c)))
            temps.add(Decimal(str(s.target_temp_c)))

        # Sort appropriately
        if is_hot:
            temp_list = sorted(temps, reverse=True)  # High to low
        else:
            temp_list = sorted(temps)  # Low to high

        # Build composite curve
        curve_points = []
        cumulative_h = Decimal("0")

        for i in range(len(temp_list)):
            if i == 0:
                curve_points.append((cumulative_h, temp_list[i]))
            else:
                t_prev = temp_list[i - 1]
                t_curr = temp_list[i]
                dt = abs(t_prev - t_curr)

                # Sum MCp of active streams
                sum_mcp = Decimal("0")
                for s in streams:
                    t_supply = Decimal(str(s.supply_temp_c))
                    t_target = Decimal(str(s.target_temp_c))

                    if is_hot:
                        if t_supply >= t_prev and t_target <= t_curr:
                            sum_mcp += Decimal(str(s.heat_capacity_flow_kw_k))
                    else:
                        if t_supply <= t_prev and t_target >= t_curr:
                            sum_mcp += Decimal(str(s.heat_capacity_flow_kw_k))

                cumulative_h += sum_mcp * dt
                curve_points.append((
                    self._apply_precision(cumulative_h),
                    self._apply_precision(t_curr)
                ))

        return curve_points

    def _generate_grand_composite(
        self,
        temp_intervals: List[Decimal],
        adjusted_cascade: List[Decimal]
    ) -> List[Tuple[Decimal, Decimal]]:
        """
        Generate Grand Composite Curve.

        Reference: Linnhoff & Hindmarsh (1983)
        """
        gcc_points = [(Decimal("0"), temp_intervals[0])]

        for i, (temp, heat) in enumerate(zip(temp_intervals[1:], adjusted_cascade)):
            gcc_points.append((
                self._apply_precision(heat),
                self._apply_precision(temp)
            ))

        return gcc_points

    def calculate_area_targets(
        self,
        streams: List[StreamData],
        dt_min_c: float,
        pinch_result: Optional[PinchResult] = None
    ) -> AreaTarget:
        """
        Calculate minimum heat exchanger network area.

        Reference: Townsend & Linnhoff (1984) "Surface Area Targets for
                   Heat Exchanger Networks"

        Uses the "Bath Formula" for area targeting.

        Args:
            streams: Process streams with h coefficients
            dt_min_c: Minimum approach temperature
            pinch_result: Optional pre-computed pinch result

        Returns:
            AreaTarget with minimum area requirements
        """
        if pinch_result is None:
            pinch_result = self.analyze(streams, dt_min_c)

        # Verify all streams have h coefficients
        for s in streams:
            if s.h_coefficient_kw_m2k is None:
                raise ValueError(f"Stream {s.name} missing heat transfer coefficient")

        pinch_temp = float(pinch_result.pinch_temperature_c)

        # Calculate area above and below pinch separately
        area_above = self._calculate_enthalpy_interval_area(
            streams, pinch_temp, float('inf'), True
        )
        area_below = self._calculate_enthalpy_interval_area(
            streams, float('-inf'), pinch_temp, False
        )

        total_area = area_above + area_below

        # Minimum number of units (Euler's theorem)
        hot_streams = [s for s in streams if s.is_hot]
        cold_streams = [s for s in streams if not s.is_hot]

        # Units above pinch
        n_above = len([s for s in hot_streams if float(s.supply_temp_c) > pinch_temp])
        n_above += len([s for s in cold_streams if float(s.target_temp_c) > pinch_temp])
        n_above += 1  # Hot utility

        # Units below pinch
        n_below = len([s for s in hot_streams if float(s.target_temp_c) < pinch_temp])
        n_below += len([s for s in cold_streams if float(s.supply_temp_c) < pinch_temp])
        n_below += 1  # Cold utility

        n_units = (n_above - 1) + (n_below - 1)

        inputs = {"num_streams": len(streams), "dt_min": str(dt_min_c)}
        outputs = {"total_area": str(total_area), "n_units": str(n_units)}
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return AreaTarget(
            total_area_m2=self._apply_precision(total_area),
            area_above_pinch_m2=self._apply_precision(area_above),
            area_below_pinch_m2=self._apply_precision(area_below),
            number_of_units=n_units,
            provenance_hash=provenance_hash
        )

    def _calculate_enthalpy_interval_area(
        self,
        streams: List[StreamData],
        t_min: float,
        t_max: float,
        above_pinch: bool
    ) -> Decimal:
        """
        Calculate area for enthalpy interval using Bath formula.

        A = Sum over intervals of: Q_interval / (LMTD * (1/h_hot + 1/h_cold))
        """
        total_area = Decimal("0")

        hot_streams = [s for s in streams if s.is_hot]
        cold_streams = [s for s in streams if not s.is_hot]

        # Simplified area calculation (vertical heat transfer)
        for hot in hot_streams:
            t_hot_in = min(float(hot.supply_temp_c), t_max)
            t_hot_out = max(float(hot.target_temp_c), t_min)

            if t_hot_in <= t_hot_out:
                continue

            q_hot = Decimal(str(hot.heat_capacity_flow_kw_k)) * Decimal(str(t_hot_in - t_hot_out))
            h_hot = Decimal(str(hot.h_coefficient_kw_m2k)) if hot.h_coefficient_kw_m2k else Decimal("1")

            for cold in cold_streams:
                t_cold_in = max(float(cold.supply_temp_c), t_min)
                t_cold_out = min(float(cold.target_temp_c), t_max)

                if t_cold_out <= t_cold_in:
                    continue

                h_cold = Decimal(str(cold.h_coefficient_kw_m2k)) if cold.h_coefficient_kw_m2k else Decimal("1")

                # Estimate LMTD
                dt1 = Decimal(str(t_hot_in - t_cold_out))
                dt2 = Decimal(str(t_hot_out - t_cold_in))

                if dt1 <= 0 or dt2 <= 0:
                    continue

                if abs(dt1 - dt2) < Decimal("0.1"):
                    lmtd = (dt1 + dt2) / Decimal("2")
                else:
                    import math
                    lmtd = (dt1 - dt2) / Decimal(str(math.log(float(dt1 / dt2))))

                # Overall U (simplified)
                u = Decimal("1") / (Decimal("1") / h_hot + Decimal("1") / h_cold)

                # Area contribution
                if lmtd > 0:
                    area = q_hot / (u * lmtd)
                    total_area += area / Decimal(str(len(cold_streams)))

        return total_area


# Convenience functions
def pinch_analysis(
    streams: List[Dict],
    dt_min_c: float
) -> PinchResult:
    """
    Perform pinch analysis on process streams.

    Example:
        >>> streams = [
        ...     {"name": "H1", "supply_temp_c": 180, "target_temp_c": 80, "mcp": 30},
        ...     {"name": "H2", "supply_temp_c": 150, "target_temp_c": 60, "mcp": 15},
        ...     {"name": "C1", "supply_temp_c": 40, "target_temp_c": 140, "mcp": 20},
        ...     {"name": "C2", "supply_temp_c": 80, "target_temp_c": 160, "mcp": 40},
        ... ]
        >>> result = pinch_analysis(streams, dt_min_c=10)
        >>> print(f"Pinch: {result.pinch_temperature_c}C")
    """
    analyzer = PinchAnalyzer()
    stream_data = [
        StreamData(
            name=s["name"],
            supply_temp_c=s["supply_temp_c"],
            target_temp_c=s["target_temp_c"],
            heat_capacity_flow_kw_k=s.get("mcp", s.get("heat_capacity_flow_kw_k")),
            h_coefficient_kw_m2k=s.get("h_coefficient_kw_m2k")
        )
        for s in streams
    ]
    return analyzer.analyze(stream_data, dt_min_c)


def minimum_utilities(
    hot_streams: List[Dict],
    cold_streams: List[Dict],
    dt_min_c: float
) -> Tuple[Decimal, Decimal]:
    """
    Calculate minimum hot and cold utility requirements.

    Returns:
        Tuple of (hot_utility_kw, cold_utility_kw)
    """
    all_streams = []
    for s in hot_streams:
        s_copy = s.copy()
        if s_copy.get("supply_temp_c", 0) <= s_copy.get("target_temp_c", 0):
            s_copy["supply_temp_c"], s_copy["target_temp_c"] = s_copy["target_temp_c"], s_copy["supply_temp_c"]
        all_streams.append(s_copy)

    for s in cold_streams:
        s_copy = s.copy()
        if s_copy.get("supply_temp_c", 0) >= s_copy.get("target_temp_c", 0):
            s_copy["supply_temp_c"], s_copy["target_temp_c"] = s_copy["target_temp_c"], s_copy["supply_temp_c"]
        all_streams.append(s_copy)

    result = pinch_analysis(all_streams, dt_min_c)
    return result.minimum_hot_utility_kw, result.minimum_cold_utility_kw
