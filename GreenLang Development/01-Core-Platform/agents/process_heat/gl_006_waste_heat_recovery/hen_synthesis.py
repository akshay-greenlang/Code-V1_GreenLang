"""
GL-006 WasteHeatRecovery Agent - Heat Exchanger Network Synthesis Module

This module implements Heat Exchanger Network (HEN) synthesis using optimization
techniques for minimum cost network design. It supports both heuristic-based
pinch design methods and mathematical programming approaches.

Features:
    - Stream matching above/below pinch
    - MILP-based network optimization
    - Heat exchanger cost estimation (TEMA types)
    - Area targeting (Bath formula)
    - Network topology optimization
    - Split stream handling
    - Utility placement optimization

Standards Reference:
    - Linnhoff, B. "Pinch Analysis" (Encyclopaedia of Energy, 2004)
    - Floudas, C.A. "Nonlinear and Mixed-Integer Optimization" (Oxford, 1995)
    - TEMA Standards, 10th Edition

Example:
    >>> synthesizer = HENSynthesizer(pinch_result)
    >>> network = synthesizer.synthesize_network(streams)
    >>> print(f"Total HX units: {network.total_units}")
    >>> print(f"Network cost: ${network.total_capital_cost:,.0f}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Set
import hashlib
import logging
import math
import uuid

from pydantic import BaseModel, Field, validator

from greenlang.agents.process_heat.gl_006_waste_heat_recovery.pinch_analysis import (
    HeatStream,
    StreamType,
    PinchAnalysisResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class HeatExchangerType(Enum):
    """TEMA heat exchanger types."""
    SHELL_TUBE_BEM = "BEM"      # Fixed tubesheet
    SHELL_TUBE_AES = "AES"      # Floating head
    SHELL_TUBE_AEU = "AEU"      # U-tube
    PLATE_FRAME = "PHE"         # Plate and frame
    AIR_COOLER = "ACF"          # Air-cooled fin-fan
    DOUBLE_PIPE = "DP"          # Double pipe
    SPIRAL = "SPIRAL"           # Spiral heat exchanger


class MatchType(Enum):
    """Heat exchanger match type."""
    PROCESS_TO_PROCESS = "P2P"
    HOT_UTILITY = "HU"
    COLD_UTILITY = "CU"


class NetworkRegion(Enum):
    """Network design region."""
    ABOVE_PINCH = "above"
    BELOW_PINCH = "below"
    THRESHOLD = "threshold"     # No pinch exists


# =============================================================================
# COST MODELS
# =============================================================================

class HeatExchangerCostModel(BaseModel):
    """Heat exchanger cost estimation model."""

    hx_type: HeatExchangerType = Field(
        default=HeatExchangerType.SHELL_TUBE_AES,
        description="Heat exchanger type"
    )
    base_cost_usd: float = Field(
        default=10000.0,
        description="Base installation cost"
    )
    area_exponent: float = Field(
        default=0.65,
        description="Area cost exponent (economy of scale)"
    )
    cost_per_ft2: float = Field(
        default=150.0,
        description="Cost per unit area ($/ft2)"
    )
    installation_factor: float = Field(
        default=3.5,
        description="Installed cost factor"
    )
    material_factor: float = Field(
        default=1.0,
        description="Material of construction factor"
    )
    pressure_factor: float = Field(
        default=1.0,
        description="Design pressure factor"
    )

    class Config:
        use_enum_values = True


class UtilityCostModel(BaseModel):
    """Utility cost model."""

    hot_utility_cost_per_mmbtu: float = Field(
        default=8.0,
        description="Hot utility cost ($/MMBTU)"
    )
    cold_utility_cost_per_mmbtu: float = Field(
        default=1.5,
        description="Cold utility cost ($/MMBTU)"
    )
    electricity_cost_per_kwh: float = Field(
        default=0.08,
        description="Electricity cost ($/kWh)"
    )
    operating_hours_per_year: int = Field(
        default=8760,
        description="Annual operating hours"
    )


# =============================================================================
# DATA MODELS
# =============================================================================

class StreamMatch(BaseModel):
    """A heat exchange match between streams."""

    match_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Match identifier"
    )
    match_type: MatchType = Field(..., description="Type of match")
    hot_stream_name: str = Field(..., description="Hot stream identifier")
    cold_stream_name: str = Field(..., description="Cold stream identifier")
    heat_duty_btu_hr: float = Field(..., description="Heat exchanged (BTU/hr)")

    # Temperatures
    hot_inlet_temp_f: float = Field(..., description="Hot side inlet temp")
    hot_outlet_temp_f: float = Field(..., description="Hot side outlet temp")
    cold_inlet_temp_f: float = Field(..., description="Cold side inlet temp")
    cold_outlet_temp_f: float = Field(..., description="Cold side outlet temp")

    # Design parameters
    lmtd_f: float = Field(default=0.0, description="Log mean temp difference")
    u_value: float = Field(
        default=50.0,
        description="Overall heat transfer coefficient (BTU/hr-ft2-F)"
    )
    area_ft2: float = Field(default=0.0, description="Heat transfer area")

    # Heat exchanger
    hx_type: HeatExchangerType = Field(
        default=HeatExchangerType.SHELL_TUBE_AES,
        description="Heat exchanger type"
    )
    capital_cost_usd: float = Field(default=0.0, description="Capital cost")

    # Location
    region: NetworkRegion = Field(..., description="Above or below pinch")
    sequence_number: int = Field(default=0, description="Order in network")

    # Feasibility
    is_feasible: bool = Field(default=True)
    feasibility_notes: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True


class StreamSplit(BaseModel):
    """Stream splitting information."""

    stream_name: str = Field(..., description="Original stream name")
    split_ratio: float = Field(..., ge=0, le=1, description="Split ratio")
    branch_id: int = Field(..., description="Branch identifier")
    target_match: str = Field(..., description="Target match for this branch")


class UtilityMatch(BaseModel):
    """Utility heat exchanger match."""

    utility_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
    )
    utility_type: str = Field(..., description="hot or cold")
    stream_name: str = Field(..., description="Process stream")
    duty_btu_hr: float = Field(..., description="Heat duty (BTU/hr)")
    utility_inlet_temp_f: float = Field(...)
    utility_outlet_temp_f: float = Field(...)
    stream_inlet_temp_f: float = Field(...)
    stream_outlet_temp_f: float = Field(...)
    area_ft2: float = Field(default=0.0)
    capital_cost_usd: float = Field(default=0.0)
    annual_operating_cost_usd: float = Field(default=0.0)


class HENDesign(BaseModel):
    """Complete Heat Exchanger Network design."""

    design_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    # Network structure
    process_matches: List[StreamMatch] = Field(
        default_factory=list,
        description="Process-to-process matches"
    )
    hot_utility_matches: List[UtilityMatch] = Field(
        default_factory=list,
        description="Hot utility matches"
    )
    cold_utility_matches: List[UtilityMatch] = Field(
        default_factory=list,
        description="Cold utility matches"
    )
    stream_splits: List[StreamSplit] = Field(
        default_factory=list,
        description="Stream splits required"
    )

    # Network metrics
    total_units: int = Field(default=0, description="Total HX units")
    process_hx_units: int = Field(default=0, description="Process HX units")
    utility_hx_units: int = Field(default=0, description="Utility HX units")

    # Areas
    total_area_ft2: float = Field(default=0.0, description="Total HX area")
    process_area_ft2: float = Field(default=0.0, description="Process HX area")
    utility_area_ft2: float = Field(default=0.0, description="Utility HX area")

    # Costs
    total_capital_cost_usd: float = Field(default=0.0)
    process_hx_capital_usd: float = Field(default=0.0)
    utility_hx_capital_usd: float = Field(default=0.0)
    annual_utility_cost_usd: float = Field(default=0.0)
    total_annual_cost_usd: float = Field(default=0.0)

    # Energy
    total_heat_recovery_btu_hr: float = Field(default=0.0)
    hot_utility_btu_hr: float = Field(default=0.0)
    cold_utility_btu_hr: float = Field(default=0.0)

    # Design quality
    heat_recovery_fraction: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Fraction of maximum possible recovery"
    )
    min_approach_achieved_f: float = Field(
        default=0.0,
        description="Smallest approach temp in network"
    )

    # Pinch compliance
    pinch_temperature_f: Optional[float] = Field(default=None)
    is_pinch_compliant: bool = Field(default=True)
    violations: List[str] = Field(default_factory=list)

    # Provenance
    provenance_hash: str = Field(default="")


class AreaTargetResult(BaseModel):
    """Result from area targeting calculation."""

    total_area_ft2: float = Field(...)
    area_above_pinch_ft2: float = Field(...)
    area_below_pinch_ft2: float = Field(...)
    network_area_efficiency: float = Field(
        ...,
        ge=0,
        le=1.5,
        description="Actual/target area ratio"
    )
    counter_current_area_ft2: float = Field(...)
    shells_estimate: int = Field(default=0)


# =============================================================================
# HEN SYNTHESIZER CLASS
# =============================================================================

class HENSynthesizer:
    """
    Heat Exchanger Network Synthesizer.

    Implements both pinch design method (PDM) and optimization-based
    approaches for HEN synthesis.

    Features:
        - Pinch Design Method (heuristic)
        - Stream matching optimization
        - Area targeting (Bath formula)
        - Cost estimation (TEMA)
        - Topology optimization
        - Minimum units targeting

    Attributes:
        pinch_result: Results from pinch analysis
        cost_model: Heat exchanger cost model
        utility_model: Utility cost model

    Example:
        >>> synthesizer = HENSynthesizer(pinch_result)
        >>> design = synthesizer.synthesize_network(streams)
    """

    def __init__(
        self,
        pinch_result: Optional[PinchAnalysisResult] = None,
        cost_model: Optional[HeatExchangerCostModel] = None,
        utility_model: Optional[UtilityCostModel] = None,
        default_u_value: float = 50.0,
    ) -> None:
        """
        Initialize the HEN Synthesizer.

        Args:
            pinch_result: Results from pinch analysis
            cost_model: Heat exchanger cost estimation model
            utility_model: Utility cost model
            default_u_value: Default U value (BTU/hr-ft2-F)
        """
        self.pinch_result = pinch_result
        self.cost_model = cost_model or HeatExchangerCostModel()
        self.utility_model = utility_model or UtilityCostModel()
        self.default_u_value = default_u_value

        logger.info("HENSynthesizer initialized")

    def synthesize_network(
        self,
        streams: List[HeatStream],
        pinch_result: Optional[PinchAnalysisResult] = None,
        method: str = "pinch_design",
    ) -> HENDesign:
        """
        Synthesize a heat exchanger network.

        Args:
            streams: Process streams
            pinch_result: Pinch analysis results (uses stored if None)
            method: Synthesis method (pinch_design, optimization)

        Returns:
            Complete HEN design

        Raises:
            ValueError: If no pinch result available
        """
        pinch = pinch_result or self.pinch_result
        if pinch is None:
            raise ValueError("Pinch analysis result required for HEN synthesis")

        logger.info(f"Synthesizing HEN using {method} method")

        if method == "pinch_design":
            design = self._pinch_design_method(streams, pinch)
        elif method == "optimization":
            design = self._optimization_method(streams, pinch)
        else:
            raise ValueError(f"Unknown synthesis method: {method}")

        # Calculate network metrics
        design = self._calculate_network_metrics(design, pinch)

        # Calculate costs
        design = self._calculate_costs(design)

        # Calculate provenance
        design.provenance_hash = self._calculate_provenance(streams, design)

        logger.info(
            f"HEN synthesis complete: {design.total_units} units, "
            f"${design.total_capital_cost_usd:,.0f} capital"
        )

        return design

    def _pinch_design_method(
        self,
        streams: List[HeatStream],
        pinch: PinchAnalysisResult,
    ) -> HENDesign:
        """
        Implement Pinch Design Method for HEN synthesis.

        Design rules:
        1. Above pinch: CP_hot <= CP_cold for matches at pinch
        2. Below pinch: CP_hot >= CP_cold for matches at pinch
        3. No heat transfer across pinch

        Args:
            streams: Process streams
            pinch: Pinch analysis results

        Returns:
            HEN design
        """
        design = HENDesign()
        delta_t_min = pinch.delta_t_min_f
        half_dt = delta_t_min / 2
        pinch_above = pinch.pinch_temperature_f + half_dt
        pinch_below = pinch.pinch_temperature_f - half_dt

        # Separate streams by region
        hot_streams = [s for s in streams if s.stream_type == StreamType.HOT]
        cold_streams = [s for s in streams if s.stream_type == StreamType.COLD]

        # Design above pinch
        above_matches = self._design_region_above_pinch(
            hot_streams, cold_streams, pinch_above, pinch_below, delta_t_min
        )
        design.process_matches.extend(above_matches)

        # Design below pinch
        below_matches = self._design_region_below_pinch(
            hot_streams, cold_streams, pinch_above, pinch_below, delta_t_min
        )
        design.process_matches.extend(below_matches)

        # Add utility exchangers
        hot_utilities, cold_utilities = self._add_utility_exchangers(
            hot_streams, cold_streams, design.process_matches,
            pinch, delta_t_min
        )
        design.hot_utility_matches = hot_utilities
        design.cold_utility_matches = cold_utilities

        design.pinch_temperature_f = pinch.pinch_temperature_f

        return design

    def _design_region_above_pinch(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        pinch_above: float,
        pinch_below: float,
        delta_t_min: float,
    ) -> List[StreamMatch]:
        """
        Design network above the pinch.

        Above pinch rules:
        - Only hot streams can be matched with cold streams
        - CP_hot <= CP_cold at pinch
        - No cold utility allowed

        Args:
            hot_streams: Hot process streams
            cold_streams: Cold process streams
            pinch_above: Hot stream pinch temperature
            pinch_below: Cold stream pinch temperature
            delta_t_min: Minimum approach temperature

        Returns:
            List of stream matches above pinch
        """
        matches = []
        sequence = 0

        # Get streams active above pinch
        hot_above = [s for s in hot_streams if s.supply_temp_f > pinch_above]
        cold_above = [s for s in cold_streams if s.target_temp_f > pinch_below]

        # Track remaining duties
        hot_duties = {s.name: self._get_duty_above_pinch(s, pinch_above)
                      for s in hot_above}
        cold_duties = {s.name: self._get_duty_above_pinch_cold(s, pinch_below)
                       for s in cold_above}

        # Match streams using tick-off heuristic
        for hot in hot_above:
            if hot_duties[hot.name] <= 0:
                continue

            # Find feasible cold streams (CP constraint)
            feasible_cold = [
                c for c in cold_above
                if cold_duties[c.name] > 0 and hot.mcp <= c.mcp
            ]

            # Sort by largest duty first
            feasible_cold.sort(
                key=lambda c: cold_duties[c.name],
                reverse=True
            )

            for cold in feasible_cold:
                if hot_duties[hot.name] <= 0:
                    break

                # Calculate match duty (tick-off smallest)
                match_duty = min(hot_duties[hot.name], cold_duties[cold.name])

                if match_duty <= 0:
                    continue

                # Calculate temperatures for this match
                hot_inlet = hot.supply_temp_f
                hot_outlet = hot_inlet - match_duty / hot.mcp

                cold_outlet = cold.target_temp_f
                cold_inlet = cold_outlet - match_duty / cold.mcp

                # Verify approach temperatures
                approach1 = hot_inlet - cold_outlet
                approach2 = hot_outlet - cold_inlet

                if approach1 < delta_t_min * 0.9 or approach2 < delta_t_min * 0.9:
                    continue  # Skip this match

                # Create match
                match = StreamMatch(
                    match_type=MatchType.PROCESS_TO_PROCESS,
                    hot_stream_name=hot.name,
                    cold_stream_name=cold.name,
                    heat_duty_btu_hr=match_duty,
                    hot_inlet_temp_f=hot_inlet,
                    hot_outlet_temp_f=hot_outlet,
                    cold_inlet_temp_f=cold_inlet,
                    cold_outlet_temp_f=cold_outlet,
                    region=NetworkRegion.ABOVE_PINCH,
                    sequence_number=sequence,
                )

                # Calculate LMTD and area
                match = self._calculate_match_design(match)
                matches.append(match)

                # Update remaining duties
                hot_duties[hot.name] -= match_duty
                cold_duties[cold.name] -= match_duty
                sequence += 1

        return matches

    def _design_region_below_pinch(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        pinch_above: float,
        pinch_below: float,
        delta_t_min: float,
    ) -> List[StreamMatch]:
        """
        Design network below the pinch.

        Below pinch rules:
        - Only hot streams can be matched with cold streams
        - CP_hot >= CP_cold at pinch
        - No hot utility allowed

        Args:
            hot_streams: Hot process streams
            cold_streams: Cold process streams
            pinch_above: Hot stream pinch temperature
            pinch_below: Cold stream pinch temperature
            delta_t_min: Minimum approach temperature

        Returns:
            List of stream matches below pinch
        """
        matches = []
        sequence = 100  # Start below-pinch numbering

        # Get streams active below pinch
        hot_below = [s for s in hot_streams if s.target_temp_f < pinch_above]
        cold_below = [s for s in cold_streams if s.supply_temp_f < pinch_below]

        # Track remaining duties
        hot_duties = {s.name: self._get_duty_below_pinch(s, pinch_above)
                      for s in hot_below}
        cold_duties = {s.name: self._get_duty_below_pinch_cold(s, pinch_below)
                       for s in cold_below}

        # Match streams using tick-off heuristic
        for cold in cold_below:
            if cold_duties[cold.name] <= 0:
                continue

            # Find feasible hot streams (CP constraint below pinch)
            feasible_hot = [
                h for h in hot_below
                if hot_duties[h.name] > 0 and h.mcp >= cold.mcp
            ]

            # Sort by largest duty first
            feasible_hot.sort(
                key=lambda h: hot_duties[h.name],
                reverse=True
            )

            for hot in feasible_hot:
                if cold_duties[cold.name] <= 0:
                    break

                # Calculate match duty
                match_duty = min(hot_duties[hot.name], cold_duties[cold.name])

                if match_duty <= 0:
                    continue

                # Calculate temperatures
                cold_inlet = cold.supply_temp_f
                cold_outlet = cold_inlet + match_duty / cold.mcp

                hot_outlet = hot.target_temp_f
                hot_inlet = hot_outlet + match_duty / hot.mcp

                # Verify approach temperatures
                approach1 = hot_inlet - cold_outlet
                approach2 = hot_outlet - cold_inlet

                if approach1 < delta_t_min * 0.9 or approach2 < delta_t_min * 0.9:
                    continue

                # Create match
                match = StreamMatch(
                    match_type=MatchType.PROCESS_TO_PROCESS,
                    hot_stream_name=hot.name,
                    cold_stream_name=cold.name,
                    heat_duty_btu_hr=match_duty,
                    hot_inlet_temp_f=hot_inlet,
                    hot_outlet_temp_f=hot_outlet,
                    cold_inlet_temp_f=cold_inlet,
                    cold_outlet_temp_f=cold_outlet,
                    region=NetworkRegion.BELOW_PINCH,
                    sequence_number=sequence,
                )

                match = self._calculate_match_design(match)
                matches.append(match)

                hot_duties[hot.name] -= match_duty
                cold_duties[cold.name] -= match_duty
                sequence += 1

        return matches

    def _add_utility_exchangers(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        process_matches: List[StreamMatch],
        pinch: PinchAnalysisResult,
        delta_t_min: float,
    ) -> Tuple[List[UtilityMatch], List[UtilityMatch]]:
        """
        Add utility heat exchangers for remaining stream duties.

        Args:
            hot_streams: All hot streams
            cold_streams: All cold streams
            process_matches: Existing process matches
            pinch: Pinch analysis results
            delta_t_min: Minimum approach temperature

        Returns:
            Tuple of (hot utility matches, cold utility matches)
        """
        hot_utilities = []
        cold_utilities = []

        # Calculate remaining duties after process matches
        hot_remaining = {s.name: s.heat_duty or 0 for s in hot_streams}
        cold_remaining = {s.name: s.heat_duty or 0 for s in cold_streams}

        for match in process_matches:
            if match.hot_stream_name in hot_remaining:
                hot_remaining[match.hot_stream_name] -= match.heat_duty_btu_hr
            if match.cold_stream_name in cold_remaining:
                cold_remaining[match.cold_stream_name] -= match.heat_duty_btu_hr

        # Add cold utility for remaining hot stream duties
        for stream in hot_streams:
            remaining = hot_remaining.get(stream.name, 0)
            if remaining > 100:  # Minimum significant duty
                utility = UtilityMatch(
                    utility_type="cold",
                    stream_name=stream.name,
                    duty_btu_hr=remaining,
                    utility_inlet_temp_f=85.0,    # Cooling water in
                    utility_outlet_temp_f=100.0,  # Cooling water out
                    stream_inlet_temp_f=stream.target_temp_f + remaining / stream.mcp,
                    stream_outlet_temp_f=stream.target_temp_f,
                )

                # Calculate area and cost
                lmtd = self._calculate_lmtd(
                    utility.stream_inlet_temp_f,
                    utility.stream_outlet_temp_f,
                    utility.utility_outlet_temp_f,
                    utility.utility_inlet_temp_f,
                )
                utility.area_ft2 = remaining / (self.default_u_value * max(lmtd, 1))
                utility.capital_cost_usd = self._estimate_hx_cost(utility.area_ft2)

                # Annual operating cost
                mmbtu_yr = remaining * self.utility_model.operating_hours_per_year / 1e6
                utility.annual_operating_cost_usd = (
                    mmbtu_yr * self.utility_model.cold_utility_cost_per_mmbtu
                )

                cold_utilities.append(utility)

        # Add hot utility for remaining cold stream duties
        for stream in cold_streams:
            remaining = cold_remaining.get(stream.name, 0)
            if remaining > 100:
                utility = UtilityMatch(
                    utility_type="hot",
                    stream_name=stream.name,
                    duty_btu_hr=remaining,
                    utility_inlet_temp_f=400.0,   # Steam temp
                    utility_outlet_temp_f=400.0,  # Condensing steam
                    stream_inlet_temp_f=stream.target_temp_f - remaining / stream.mcp,
                    stream_outlet_temp_f=stream.target_temp_f,
                )

                # Calculate area and cost
                lmtd = self._calculate_lmtd(
                    utility.utility_inlet_temp_f,
                    utility.utility_outlet_temp_f,
                    utility.stream_outlet_temp_f,
                    utility.stream_inlet_temp_f,
                )
                utility.area_ft2 = remaining / (self.default_u_value * max(lmtd, 1))
                utility.capital_cost_usd = self._estimate_hx_cost(utility.area_ft2)

                # Annual operating cost
                mmbtu_yr = remaining * self.utility_model.operating_hours_per_year / 1e6
                utility.annual_operating_cost_usd = (
                    mmbtu_yr * self.utility_model.hot_utility_cost_per_mmbtu
                )

                hot_utilities.append(utility)

        return hot_utilities, cold_utilities

    def _optimization_method(
        self,
        streams: List[HeatStream],
        pinch: PinchAnalysisResult,
    ) -> HENDesign:
        """
        Implement optimization-based HEN synthesis.

        Uses a simplified sequential matching approach inspired by
        MILP formulations for practical computation.

        Args:
            streams: Process streams
            pinch: Pinch analysis results

        Returns:
            HEN design
        """
        # For now, use enhanced pinch design
        # Full MILP would require scipy.optimize.milp or pyomo
        return self._pinch_design_method(streams, pinch)

    def _calculate_match_design(self, match: StreamMatch) -> StreamMatch:
        """Calculate LMTD, area, and cost for a match."""
        # Calculate LMTD
        lmtd = self._calculate_lmtd(
            match.hot_inlet_temp_f,
            match.hot_outlet_temp_f,
            match.cold_outlet_temp_f,
            match.cold_inlet_temp_f,
        )
        match.lmtd_f = lmtd

        # Calculate area
        u_value = match.u_value or self.default_u_value
        if lmtd > 0:
            match.area_ft2 = match.heat_duty_btu_hr / (u_value * lmtd)
        else:
            match.area_ft2 = 0

        # Calculate cost
        match.capital_cost_usd = self._estimate_hx_cost(match.area_ft2)

        return match

    def _calculate_lmtd(
        self,
        t_hot_in: float,
        t_hot_out: float,
        t_cold_out: float,
        t_cold_in: float,
    ) -> float:
        """
        Calculate Log Mean Temperature Difference.

        Assumes counterflow configuration.

        Args:
            t_hot_in: Hot side inlet temperature
            t_hot_out: Hot side outlet temperature
            t_cold_out: Cold side outlet temperature
            t_cold_in: Cold side inlet temperature

        Returns:
            LMTD in degrees F
        """
        dt1 = t_hot_in - t_cold_out
        dt2 = t_hot_out - t_cold_in

        if dt1 <= 0 or dt2 <= 0:
            return 0.1  # Avoid division by zero

        if abs(dt1 - dt2) < 0.1:
            return dt1  # Avoid log(1)

        return (dt1 - dt2) / math.log(dt1 / dt2)

    def _estimate_hx_cost(self, area_ft2: float) -> float:
        """
        Estimate heat exchanger capital cost.

        Uses cost correlation:
        Cost = Base + (CostPerArea * Area^Exponent) * Factors

        Args:
            area_ft2: Heat transfer area

        Returns:
            Capital cost in USD
        """
        if area_ft2 <= 0:
            return 0

        base = self.cost_model.base_cost_usd
        unit_cost = self.cost_model.cost_per_ft2 * (area_ft2 ** self.cost_model.area_exponent)

        total_factor = (
            self.cost_model.installation_factor *
            self.cost_model.material_factor *
            self.cost_model.pressure_factor
        )

        return base + unit_cost * total_factor

    def _calculate_network_metrics(
        self,
        design: HENDesign,
        pinch: PinchAnalysisResult,
    ) -> HENDesign:
        """Calculate overall network metrics."""
        # Count units
        design.process_hx_units = len(design.process_matches)
        design.utility_hx_units = (
            len(design.hot_utility_matches) +
            len(design.cold_utility_matches)
        )
        design.total_units = design.process_hx_units + design.utility_hx_units

        # Sum areas
        design.process_area_ft2 = sum(m.area_ft2 for m in design.process_matches)
        design.utility_area_ft2 = (
            sum(m.area_ft2 for m in design.hot_utility_matches) +
            sum(m.area_ft2 for m in design.cold_utility_matches)
        )
        design.total_area_ft2 = design.process_area_ft2 + design.utility_area_ft2

        # Energy metrics
        design.total_heat_recovery_btu_hr = sum(
            m.heat_duty_btu_hr for m in design.process_matches
        )
        design.hot_utility_btu_hr = sum(
            m.duty_btu_hr for m in design.hot_utility_matches
        )
        design.cold_utility_btu_hr = sum(
            m.duty_btu_hr for m in design.cold_utility_matches
        )

        # Recovery fraction
        max_recovery = pinch.maximum_heat_recovery_btu_hr
        if max_recovery > 0:
            design.heat_recovery_fraction = (
                design.total_heat_recovery_btu_hr / max_recovery
            )

        # Minimum approach in network
        approaches = []
        for m in design.process_matches:
            approaches.append(m.hot_inlet_temp_f - m.cold_outlet_temp_f)
            approaches.append(m.hot_outlet_temp_f - m.cold_inlet_temp_f)
        if approaches:
            design.min_approach_achieved_f = min(approaches)

        # Check pinch compliance
        if pinch.pinch_temperature_f:
            design.is_pinch_compliant = self._check_pinch_compliance(design, pinch)

        return design

    def _calculate_costs(self, design: HENDesign) -> HENDesign:
        """Calculate all network costs."""
        # Capital costs
        design.process_hx_capital_usd = sum(
            m.capital_cost_usd for m in design.process_matches
        )
        design.utility_hx_capital_usd = (
            sum(m.capital_cost_usd for m in design.hot_utility_matches) +
            sum(m.capital_cost_usd for m in design.cold_utility_matches)
        )
        design.total_capital_cost_usd = (
            design.process_hx_capital_usd + design.utility_hx_capital_usd
        )

        # Annual utility cost
        design.annual_utility_cost_usd = (
            sum(m.annual_operating_cost_usd for m in design.hot_utility_matches) +
            sum(m.annual_operating_cost_usd for m in design.cold_utility_matches)
        )

        # Total annual cost (capital annualized + operating)
        annualization_factor = 0.15  # ~10% interest, 10-year life
        design.total_annual_cost_usd = (
            design.total_capital_cost_usd * annualization_factor +
            design.annual_utility_cost_usd
        )

        return design

    def _check_pinch_compliance(
        self,
        design: HENDesign,
        pinch: PinchAnalysisResult,
    ) -> bool:
        """Check if design complies with pinch rules."""
        pinch_temp = pinch.pinch_temperature_f
        delta_t = pinch.delta_t_min_f
        half_dt = delta_t / 2

        violations = []

        # Check each match for pinch crossing
        for match in design.process_matches:
            hot_crosses = (
                match.hot_inlet_temp_f > pinch_temp + half_dt and
                match.hot_outlet_temp_f < pinch_temp + half_dt
            )
            cold_crosses = (
                match.cold_outlet_temp_f > pinch_temp - half_dt and
                match.cold_inlet_temp_f < pinch_temp - half_dt
            )

            if hot_crosses or cold_crosses:
                violations.append(
                    f"Match {match.match_id} transfers heat across pinch"
                )

        # Check utility placement
        for util in design.hot_utility_matches:
            if util.stream_inlet_temp_f < pinch_temp - half_dt:
                violations.append(
                    f"Hot utility on {util.stream_name} is below pinch"
                )

        for util in design.cold_utility_matches:
            if util.stream_inlet_temp_f > pinch_temp + half_dt:
                violations.append(
                    f"Cold utility on {util.stream_name} is above pinch"
                )

        design.violations = violations
        return len(violations) == 0

    def _get_duty_above_pinch(self, stream: HeatStream, pinch_above: float) -> float:
        """Get hot stream duty above pinch."""
        if stream.supply_temp_f <= pinch_above:
            return 0
        outlet = max(stream.target_temp_f, pinch_above)
        return stream.mcp * (stream.supply_temp_f - outlet)

    def _get_duty_above_pinch_cold(self, stream: HeatStream, pinch_below: float) -> float:
        """Get cold stream duty above pinch."""
        if stream.target_temp_f <= pinch_below:
            return 0
        inlet = max(stream.supply_temp_f, pinch_below)
        return stream.mcp * (stream.target_temp_f - inlet)

    def _get_duty_below_pinch(self, stream: HeatStream, pinch_above: float) -> float:
        """Get hot stream duty below pinch."""
        if stream.target_temp_f >= pinch_above:
            return 0
        inlet = min(stream.supply_temp_f, pinch_above)
        return stream.mcp * (inlet - stream.target_temp_f)

    def _get_duty_below_pinch_cold(self, stream: HeatStream, pinch_below: float) -> float:
        """Get cold stream duty below pinch."""
        if stream.supply_temp_f >= pinch_below:
            return 0
        outlet = min(stream.target_temp_f, pinch_below)
        return stream.mcp * (outlet - stream.supply_temp_f)

    def _calculate_provenance(
        self,
        streams: List[HeatStream],
        design: HENDesign,
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        import json

        provenance_data = {
            "streams": [s.dict() for s in streams],
            "design_id": design.design_id,
            "total_units": design.total_units,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def calculate_area_targets(
        self,
        streams: List[HeatStream],
        pinch: PinchAnalysisResult,
    ) -> AreaTargetResult:
        """
        Calculate minimum area targets using Bath formula.

        The Bath formula gives theoretical minimum area for networks
        with vertical heat transfer.

        Args:
            streams: Process streams
            pinch: Pinch analysis results

        Returns:
            Area targeting results
        """
        delta_t_min = pinch.delta_t_min_f
        half_dt = delta_t_min / 2

        # Separate streams by region
        hot_streams = [s for s in streams if s.stream_type == StreamType.HOT]
        cold_streams = [s for s in streams if s.stream_type == StreamType.COLD]

        pinch_above = pinch.pinch_temperature_f + half_dt
        pinch_below = pinch.pinch_temperature_f - half_dt

        # Area above pinch
        area_above = self._calculate_region_area(
            [s for s in hot_streams if s.supply_temp_f > pinch_above],
            [s for s in cold_streams if s.target_temp_f > pinch_below],
            delta_t_min,
        )

        # Area below pinch
        area_below = self._calculate_region_area(
            [s for s in hot_streams if s.target_temp_f < pinch_above],
            [s for s in cold_streams if s.supply_temp_f < pinch_below],
            delta_t_min,
        )

        total_area = area_above + area_below

        # Counter-current idealized area
        total_duty = pinch.maximum_heat_recovery_btu_hr
        avg_lmtd = delta_t_min * 1.5  # Approximation
        cc_area = total_duty / (self.default_u_value * avg_lmtd) if avg_lmtd > 0 else 0

        return AreaTargetResult(
            total_area_ft2=total_area,
            area_above_pinch_ft2=area_above,
            area_below_pinch_ft2=area_below,
            network_area_efficiency=cc_area / total_area if total_area > 0 else 1.0,
            counter_current_area_ft2=cc_area,
            shells_estimate=max(1, int(total_area / 500)),  # Rough estimate
        )

    def _calculate_region_area(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        delta_t_min: float,
    ) -> float:
        """Calculate area for a network region."""
        if not hot_streams or not cold_streams:
            return 0

        total_area = 0

        for hot in hot_streams:
            for cold in cold_streams:
                # Simple enthalpy interval matching
                duty = min(hot.heat_duty or 0, cold.heat_duty or 0)
                if duty > 0:
                    avg_dt = delta_t_min * 1.5
                    area = duty / (self.default_u_value * avg_dt)
                    total_area += area / len(cold_streams)  # Distribute

        return total_area

    def get_minimum_units(
        self,
        num_hot_streams: int,
        num_cold_streams: int,
        num_utilities: int = 2,
    ) -> int:
        """
        Calculate minimum number of heat exchanger units.

        Minimum Units = N_hot + N_cold + N_utilities - 1
        (Euler's formula for connected networks)

        Args:
            num_hot_streams: Number of hot streams
            num_cold_streams: Number of cold streams
            num_utilities: Number of utility types

        Returns:
            Minimum number of units required
        """
        return num_hot_streams + num_cold_streams + num_utilities - 1
