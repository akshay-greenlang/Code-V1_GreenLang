"""
GL-006 HEATRECLAIM - Heat Exchanger Network Synthesizer

Implements HEN synthesis algorithms for designing optimal heat
exchanger networks that maximize heat recovery while respecting
pinch rules and operational constraints.

Reference: Linnhoff & Hindmarsh, "The Pinch Design Method for Heat
Exchanger Networks", Chem Eng Sci, 1983.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import json
import logging
import math

from ..core.schemas import (
    HeatStream,
    HeatExchanger,
    HENDesign,
    PinchAnalysisResult,
    UtilityCost,
)
from ..core.config import (
    StreamType,
    ExchangerType,
    FlowArrangement,
    OptimizationMode,
)
from .pinch_analysis import PinchAnalysisCalculator
from .lmtd_calculator import LMTDCalculator, NTUCalculator

logger = logging.getLogger(__name__)


@dataclass
class StreamSegment:
    """A segment of a stream above or below the pinch."""

    stream_id: str
    segment_id: str
    stream_type: StreamType
    T_start_C: float
    T_end_C: float
    FCp_kW_K: float
    duty_kW: float
    remaining_duty_kW: float
    above_pinch: bool
    matched: bool = False


@dataclass
class Match:
    """A proposed match between hot and cold streams."""

    hot_stream_id: str
    cold_stream_id: str
    duty_kW: float
    hot_inlet_T: float
    hot_outlet_T: float
    cold_inlet_T: float
    cold_outlet_T: float
    feasibility_score: float
    constraint_violations: List[str] = field(default_factory=list)


class HENSynthesizer:
    """
    Heat Exchanger Network Synthesizer.

    Implements the Pinch Design Method for HEN synthesis:
    1. Divide problem at pinch into above and below regions
    2. Apply pinch rules (no heat transfer across pinch)
    3. Match streams using FCp constraints
    4. Design heat exchangers for each match

    Example:
        >>> synthesizer = HENSynthesizer(delta_t_min=10.0)
        >>> design = synthesizer.synthesize(
        ...     hot_streams, cold_streams, pinch_result
        ... )
        >>> print(f"Heat recovered: {design.total_heat_recovered_kW} kW")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        delta_t_min: float = 10.0,
        max_exchangers: int = 50,
        default_U_W_m2K: float = 500.0,
    ) -> None:
        """
        Initialize HEN synthesizer.

        Args:
            delta_t_min: Minimum approach temperature
            max_exchangers: Maximum number of exchangers allowed
            default_U_W_m2K: Default heat transfer coefficient
        """
        self.delta_t_min = delta_t_min
        self.max_exchangers = max_exchangers
        self.default_U = default_U_W_m2K

        self.lmtd_calc = LMTDCalculator()
        self.ntu_calc = NTUCalculator()

    def synthesize(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        pinch_result: Optional[PinchAnalysisResult] = None,
        hot_utilities: Optional[List[UtilityCost]] = None,
        cold_utilities: Optional[List[UtilityCost]] = None,
        mode: OptimizationMode = OptimizationMode.GRASSROOTS,
    ) -> HENDesign:
        """
        Synthesize heat exchanger network using Pinch Design Method.

        Args:
            hot_streams: Hot process streams
            cold_streams: Cold process streams
            pinch_result: Pre-computed pinch analysis (optional)
            hot_utilities: Available hot utilities
            cold_utilities: Available cold utilities
            mode: Design mode (grassroots or retrofit)

        Returns:
            HENDesign with complete network
        """
        # Perform pinch analysis if not provided
        if pinch_result is None:
            pinch_calc = PinchAnalysisCalculator(delta_t_min=self.delta_t_min)
            pinch_result = pinch_calc.calculate(hot_streams, cold_streams)

        pinch_T = pinch_result.pinch_temperature_C

        logger.info(f"Synthesizing HEN with pinch at {pinch_T}°C")

        # Create stream segments above and below pinch
        above_hot, above_cold = self._segment_streams(
            hot_streams, cold_streams, pinch_T, above_pinch=True
        )
        below_hot, below_cold = self._segment_streams(
            hot_streams, cold_streams, pinch_T, above_pinch=False
        )

        # Design above pinch (cold end at pinch)
        above_matches = self._design_region(
            above_hot, above_cold, above_pinch=True
        )

        # Design below pinch (hot end at pinch)
        below_matches = self._design_region(
            below_hot, below_cold, above_pinch=False
        )

        # Convert matches to heat exchangers
        exchangers = []
        stream_map = {s.stream_id: s for s in hot_streams + cold_streams}

        for i, match in enumerate(above_matches + below_matches):
            hx = self._create_exchanger(
                match, stream_map, exchanger_id=f"HX-{i+1:03d}"
            )
            exchangers.append(hx)

        # Calculate remaining utility requirements
        total_heat_recovered = sum(hx.duty_kW for hx in exchangers)
        hot_utility_needed = max(0, pinch_result.minimum_hot_utility_kW)
        cold_utility_needed = max(0, pinch_result.minimum_cold_utility_kW)

        # Add utility exchangers
        utility_exchangers = self._add_utility_exchangers(
            hot_streams, cold_streams, exchangers,
            hot_utility_needed, cold_utility_needed,
            hot_utilities, cold_utilities,
            len(exchangers)
        )
        exchangers.extend(utility_exchangers)

        # Build design
        design = HENDesign(
            design_name=f"HEN-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            mode=mode,
            exchangers=exchangers,
            total_heat_recovered_kW=round(total_heat_recovered, 2),
            hot_utility_required_kW=round(hot_utility_needed, 2),
            cold_utility_required_kW=round(cold_utility_needed, 2),
            exchanger_count=len(exchangers),
            new_exchanger_count=len([hx for hx in exchangers if not hx.is_existing]),
            total_area_m2=round(sum(hx.area_m2 for hx in exchangers), 2),
        )

        # Validate design
        self._validate_design(design, pinch_result)

        # Compute hashes
        design.input_hash = self._compute_hash({
            "hot_streams": [s.stream_id for s in hot_streams],
            "cold_streams": [s.stream_id for s in cold_streams],
            "pinch_T": pinch_T,
        })

        design.output_hash = self._compute_hash({
            "exchangers": len(exchangers),
            "total_heat_recovered_kW": design.total_heat_recovered_kW,
        })

        logger.info(
            f"HEN synthesis complete: {len(exchangers)} exchangers, "
            f"{design.total_heat_recovered_kW} kW recovered"
        )

        return design

    def _segment_streams(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        pinch_T: float,
        above_pinch: bool,
    ) -> Tuple[List[StreamSegment], List[StreamSegment]]:
        """
        Segment streams above or below the pinch temperature.

        Above pinch: hot streams from supply to pinch, cold streams from pinch to target
        Below pinch: hot streams from pinch to target, cold streams from supply to pinch
        """
        hot_segments = []
        cold_segments = []

        shift = self.delta_t_min / 2

        for s in hot_streams:
            # Hot stream shifted down
            T_shifted_supply = s.T_supply_C - shift
            T_shifted_target = s.T_target_C - shift

            if above_pinch:
                # Above pinch: from supply down to pinch
                if T_shifted_supply > pinch_T > T_shifted_target:
                    # Stream crosses pinch
                    T_start = s.T_supply_C
                    T_end = pinch_T + shift
                elif T_shifted_target >= pinch_T:
                    # Entirely above pinch
                    T_start = s.T_supply_C
                    T_end = s.T_target_C
                else:
                    continue  # Entirely below pinch
            else:
                # Below pinch: from pinch down to target
                if T_shifted_supply > pinch_T > T_shifted_target:
                    # Stream crosses pinch
                    T_start = pinch_T + shift
                    T_end = s.T_target_C
                elif T_shifted_supply <= pinch_T:
                    # Entirely below pinch
                    T_start = s.T_supply_C
                    T_end = s.T_target_C
                else:
                    continue  # Entirely above pinch

            if T_start > T_end:
                duty = s.FCp_kW_K * (T_start - T_end)
                hot_segments.append(StreamSegment(
                    stream_id=s.stream_id,
                    segment_id=f"{s.stream_id}_{'above' if above_pinch else 'below'}",
                    stream_type=StreamType.HOT,
                    T_start_C=T_start,
                    T_end_C=T_end,
                    FCp_kW_K=s.FCp_kW_K,
                    duty_kW=duty,
                    remaining_duty_kW=duty,
                    above_pinch=above_pinch,
                ))

        for s in cold_streams:
            # Cold stream shifted up
            T_shifted_supply = s.T_supply_C + shift
            T_shifted_target = s.T_target_C + shift

            if above_pinch:
                # Above pinch: from pinch up to target
                if T_shifted_supply < pinch_T < T_shifted_target:
                    # Stream crosses pinch
                    T_start = pinch_T - shift
                    T_end = s.T_target_C
                elif T_shifted_supply >= pinch_T:
                    # Entirely above pinch
                    T_start = s.T_supply_C
                    T_end = s.T_target_C
                else:
                    continue  # Entirely below pinch
            else:
                # Below pinch: from supply up to pinch
                if T_shifted_supply < pinch_T < T_shifted_target:
                    # Stream crosses pinch
                    T_start = s.T_supply_C
                    T_end = pinch_T - shift
                elif T_shifted_target <= pinch_T:
                    # Entirely below pinch
                    T_start = s.T_supply_C
                    T_end = s.T_target_C
                else:
                    continue  # Entirely above pinch

            if T_end > T_start:
                duty = s.FCp_kW_K * (T_end - T_start)
                cold_segments.append(StreamSegment(
                    stream_id=s.stream_id,
                    segment_id=f"{s.stream_id}_{'above' if above_pinch else 'below'}",
                    stream_type=StreamType.COLD,
                    T_start_C=T_start,
                    T_end_C=T_end,
                    FCp_kW_K=s.FCp_kW_K,
                    duty_kW=duty,
                    remaining_duty_kW=duty,
                    above_pinch=above_pinch,
                ))

        return hot_segments, cold_segments

    def _design_region(
        self,
        hot_segments: List[StreamSegment],
        cold_segments: List[StreamSegment],
        above_pinch: bool,
    ) -> List[Match]:
        """
        Design matches for a region (above or below pinch).

        Applies pinch design rules:
        - Above pinch: FCp_hot >= FCp_cold at pinch
        - Below pinch: FCp_cold >= FCp_hot at pinch
        - Maximize heat recovery in each match
        """
        matches = []

        # Sort by temperature to prioritize pinch-adjacent matches
        if above_pinch:
            hot_sorted = sorted(hot_segments, key=lambda s: s.T_end_C)  # Lowest temp first
            cold_sorted = sorted(cold_segments, key=lambda s: s.T_start_C)  # Lowest temp first
        else:
            hot_sorted = sorted(hot_segments, key=lambda s: s.T_start_C)  # Lowest temp first
            cold_sorted = sorted(cold_segments, key=lambda s: s.T_end_C)  # Lowest temp first

        # Match streams using tick-off heuristic
        for hot_seg in hot_sorted:
            if hot_seg.remaining_duty_kW <= 0.1:
                continue

            for cold_seg in cold_sorted:
                if cold_seg.remaining_duty_kW <= 0.1:
                    continue

                # Check FCp constraint for pinch
                if above_pinch:
                    # At pinch (cold end), FCp_hot >= FCp_cold
                    if hot_seg.FCp_kW_K < cold_seg.FCp_kW_K * 0.99:
                        continue
                else:
                    # At pinch (hot end), FCp_cold >= FCp_hot
                    if cold_seg.FCp_kW_K < hot_seg.FCp_kW_K * 0.99:
                        continue

                # Check temperature feasibility
                if not self._check_feasibility(hot_seg, cold_seg):
                    continue

                # Calculate match duty
                duty = min(hot_seg.remaining_duty_kW, cold_seg.remaining_duty_kW)

                if duty < 1.0:  # Minimum 1 kW
                    continue

                # Calculate temperatures
                match = self._calculate_match(hot_seg, cold_seg, duty)

                if match and len(match.constraint_violations) == 0:
                    matches.append(match)

                    # Update remaining duties
                    hot_seg.remaining_duty_kW -= duty
                    cold_seg.remaining_duty_kW -= duty

                    if hot_seg.remaining_duty_kW <= 0.1:
                        break

        return matches

    def _check_feasibility(
        self,
        hot_seg: StreamSegment,
        cold_seg: StreamSegment,
    ) -> bool:
        """Check if match between segments is thermodynamically feasible."""
        # Hot inlet must be hotter than cold outlet + delta_T_min
        T_hot_in = hot_seg.T_start_C
        T_cold_out = cold_seg.T_end_C

        if T_hot_in < T_cold_out + self.delta_t_min:
            return False

        # Hot outlet must be hotter than cold inlet + delta_T_min
        T_hot_out = hot_seg.T_end_C
        T_cold_in = cold_seg.T_start_C

        if T_hot_out < T_cold_in + self.delta_t_min:
            return False

        return True

    def _calculate_match(
        self,
        hot_seg: StreamSegment,
        cold_seg: StreamSegment,
        duty: float,
    ) -> Optional[Match]:
        """Calculate temperatures and feasibility for a match."""
        violations = []

        # Calculate outlet temperatures
        delta_T_hot = duty / hot_seg.FCp_kW_K
        delta_T_cold = duty / cold_seg.FCp_kW_K

        T_hot_in = hot_seg.T_start_C
        T_hot_out = T_hot_in - delta_T_hot
        T_cold_in = cold_seg.T_start_C
        T_cold_out = T_cold_in + delta_T_cold

        # Check approach temperatures
        delta_T_hot_end = T_hot_in - T_cold_out
        delta_T_cold_end = T_hot_out - T_cold_in

        if delta_T_hot_end < self.delta_t_min * 0.95:
            violations.append(f"Hot end ΔT = {delta_T_hot_end:.1f}°C < ΔTmin")

        if delta_T_cold_end < self.delta_t_min * 0.95:
            violations.append(f"Cold end ΔT = {delta_T_cold_end:.1f}°C < ΔTmin")

        # Calculate feasibility score
        score = 1.0
        if delta_T_hot_end > 0 and delta_T_cold_end > 0:
            score = min(delta_T_hot_end, delta_T_cold_end) / self.delta_t_min
        else:
            score = 0.0

        return Match(
            hot_stream_id=hot_seg.stream_id,
            cold_stream_id=cold_seg.stream_id,
            duty_kW=round(duty, 2),
            hot_inlet_T=round(T_hot_in, 2),
            hot_outlet_T=round(T_hot_out, 2),
            cold_inlet_T=round(T_cold_in, 2),
            cold_outlet_T=round(T_cold_out, 2),
            feasibility_score=round(score, 3),
            constraint_violations=violations,
        )

    def _create_exchanger(
        self,
        match: Match,
        stream_map: Dict[str, HeatStream],
        exchanger_id: str,
    ) -> HeatExchanger:
        """Create heat exchanger specification from match."""
        hot_stream = stream_map.get(match.hot_stream_id)
        cold_stream = stream_map.get(match.cold_stream_id)

        # Calculate LMTD
        lmtd_result = self.lmtd_calc.calculate(
            match.hot_inlet_T, match.hot_outlet_T,
            match.cold_inlet_T, match.cold_outlet_T,
        )

        # Calculate UA and area
        if lmtd_result.effective_LMTD_C > 0:
            UA = match.duty_kW / lmtd_result.effective_LMTD_C
            U_kW_m2K = self.default_U / 1000
            area = UA / U_kW_m2K
        else:
            UA = 0.0
            area = 0.0

        return HeatExchanger(
            exchanger_id=exchanger_id,
            exchanger_name=f"{match.hot_stream_id}-{match.cold_stream_id}",
            exchanger_type=ExchangerType.SHELL_AND_TUBE,
            hot_stream_id=match.hot_stream_id,
            cold_stream_id=match.cold_stream_id,
            duty_kW=match.duty_kW,
            hot_inlet_T_C=match.hot_inlet_T,
            hot_outlet_T_C=match.hot_outlet_T,
            cold_inlet_T_C=match.cold_inlet_T,
            cold_outlet_T_C=match.cold_outlet_T,
            delta_T_hot_end_C=round(match.hot_inlet_T - match.cold_outlet_T, 2),
            delta_T_cold_end_C=round(match.hot_outlet_T - match.cold_inlet_T, 2),
            LMTD_C=lmtd_result.LMTD_C,
            UA_kW_K=round(UA, 3),
            U_W_m2K=self.default_U,
            area_m2=round(area, 2),
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
            F_correction_factor=lmtd_result.F_correction,
            hot_fouling_m2K_W=hot_stream.fouling_factor_m2K_W if hot_stream else 0.0001,
            cold_fouling_m2K_W=cold_stream.fouling_factor_m2K_W if cold_stream else 0.0001,
        )

    def _add_utility_exchangers(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        exchangers: List[HeatExchanger],
        hot_utility_needed: float,
        cold_utility_needed: float,
        hot_utilities: Optional[List[UtilityCost]],
        cold_utilities: Optional[List[UtilityCost]],
        start_index: int,
    ) -> List[HeatExchanger]:
        """Add heat exchangers for utility requirements."""
        utility_exchangers = []
        idx = start_index

        # Hot utility exchangers (heaters for cold streams)
        if hot_utility_needed > 0:
            # Find cold streams that need more heating
            for s in cold_streams:
                # Check if stream is fully satisfied
                stream_matches = [
                    hx for hx in exchangers
                    if hx.cold_stream_id == s.stream_id
                ]
                matched_duty = sum(hx.duty_kW for hx in stream_matches)
                remaining = s.duty_kW - matched_duty

                if remaining > 1.0:
                    # Need hot utility
                    T_utility = 200.0  # Default hot utility temperature
                    if hot_utilities:
                        T_utility = hot_utilities[0].T_supply_C

                    duty = min(remaining, hot_utility_needed)
                    T_out = s.T_target_C
                    T_in = T_out - duty / s.FCp_kW_K

                    hx = HeatExchanger(
                        exchanger_id=f"HX-{idx+1:03d}",
                        exchanger_name=f"HotUtility-{s.stream_id}",
                        exchanger_type=ExchangerType.SHELL_AND_TUBE,
                        hot_stream_id="HOT_UTILITY",
                        cold_stream_id=s.stream_id,
                        duty_kW=round(duty, 2),
                        hot_inlet_T_C=T_utility,
                        hot_outlet_T_C=T_utility - 10,  # Condensing utility
                        cold_inlet_T_C=round(T_in, 2),
                        cold_outlet_T_C=round(T_out, 2),
                        area_m2=round(duty / 20, 2),  # Rough estimate
                    )
                    utility_exchangers.append(hx)
                    hot_utility_needed -= duty
                    idx += 1

                    if hot_utility_needed <= 1.0:
                        break

        # Cold utility exchangers (coolers for hot streams)
        if cold_utility_needed > 0:
            for s in hot_streams:
                stream_matches = [
                    hx for hx in exchangers
                    if hx.hot_stream_id == s.stream_id
                ]
                matched_duty = sum(hx.duty_kW for hx in stream_matches)
                remaining = s.duty_kW - matched_duty

                if remaining > 1.0:
                    T_utility = 25.0  # Default cooling water
                    if cold_utilities:
                        T_utility = cold_utilities[0].T_supply_C

                    duty = min(remaining, cold_utility_needed)
                    T_out = s.T_target_C
                    T_in = T_out + duty / s.FCp_kW_K

                    hx = HeatExchanger(
                        exchanger_id=f"HX-{idx+1:03d}",
                        exchanger_name=f"ColdUtility-{s.stream_id}",
                        exchanger_type=ExchangerType.SHELL_AND_TUBE,
                        hot_stream_id=s.stream_id,
                        cold_stream_id="COLD_UTILITY",
                        duty_kW=round(duty, 2),
                        hot_inlet_T_C=round(T_in, 2),
                        hot_outlet_T_C=round(T_out, 2),
                        cold_inlet_T_C=T_utility,
                        cold_outlet_T_C=T_utility + 10,
                        area_m2=round(duty / 15, 2),
                    )
                    utility_exchangers.append(hx)
                    cold_utility_needed -= duty
                    idx += 1

                    if cold_utility_needed <= 1.0:
                        break

        return utility_exchangers

    def _validate_design(
        self,
        design: HENDesign,
        pinch_result: PinchAnalysisResult,
    ) -> None:
        """Validate the synthesized design."""
        design.constraint_details = []
        design.pinch_violations = 0
        design.temperature_violations = 0

        for hx in design.exchangers:
            # Check approach temperatures
            if hx.delta_T_hot_end_C < self.delta_t_min * 0.95:
                design.temperature_violations += 1
                design.constraint_details.append({
                    "exchanger": hx.exchanger_id,
                    "violation": f"Hot end ΔT = {hx.delta_T_hot_end_C}°C",
                })

            if hx.delta_T_cold_end_C < self.delta_t_min * 0.95:
                design.temperature_violations += 1
                design.constraint_details.append({
                    "exchanger": hx.exchanger_id,
                    "violation": f"Cold end ΔT = {hx.delta_T_cold_end_C}°C",
                })

        design.all_constraints_satisfied = (
            design.pinch_violations == 0 and
            design.temperature_violations == 0
        )

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
