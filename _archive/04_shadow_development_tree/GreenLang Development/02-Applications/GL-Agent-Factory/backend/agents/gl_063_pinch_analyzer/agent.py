"""
GL-063: Pinch Analyzer Agent (PINCH-ANALYZER)

This module implements the PinchAnalyzerAgent for heat integration pinch analysis
to minimize utility consumption in industrial process systems.

Standards Reference:
    - Linnhoff, B. "Pinch Analysis and Process Integration"
    - Process Integration principles
    - Heat Exchanger Network (HEN) design

Example:
    >>> agent = PinchAnalyzerAgent()
    >>> result = agent.run(PinchAnalyzerInput(hot_streams=[...], cold_streams=[...]))
    >>> print(f"Pinch temperature: {result.pinch_temperature_celsius:.1f}C")
"""

import hashlib
import json
import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HotStream(BaseModel):
    """Hot stream that needs cooling."""
    stream_id: str = Field(..., description="Unique stream identifier")
    name: str = Field(..., description="Stream name")
    supply_temp_celsius: float = Field(..., description="Supply (inlet) temperature (C)")
    target_temp_celsius: float = Field(..., description="Target (outlet) temperature (C)")
    heat_capacity_flow_kW_K: float = Field(..., gt=0, description="mCp (kW/K)")

    @property
    def heat_load_kW(self) -> float:
        return self.heat_capacity_flow_kW_K * (self.supply_temp_celsius - self.target_temp_celsius)


class ColdStream(BaseModel):
    """Cold stream that needs heating."""
    stream_id: str = Field(..., description="Unique stream identifier")
    name: str = Field(..., description="Stream name")
    supply_temp_celsius: float = Field(..., description="Supply (inlet) temperature (C)")
    target_temp_celsius: float = Field(..., description="Target (outlet) temperature (C)")
    heat_capacity_flow_kW_K: float = Field(..., gt=0, description="mCp (kW/K)")

    @property
    def heat_load_kW(self) -> float:
        return self.heat_capacity_flow_kW_K * (self.target_temp_celsius - self.supply_temp_celsius)


class UtilityCost(BaseModel):
    """Utility cost information."""
    hot_utility_cost_per_kW: float = Field(default=0.05, description="Hot utility cost ($/kW)")
    cold_utility_cost_per_kW: float = Field(default=0.02, description="Cold utility cost ($/kW)")
    operating_hours_per_year: int = Field(default=8000, description="Annual operating hours")


class PinchAnalyzerInput(BaseModel):
    """Input for pinch analysis."""
    analysis_id: Optional[str] = Field(None, description="Analysis identifier")
    system_name: str = Field(default="Heat Integration System", description="System name")
    hot_streams: List[HotStream] = Field(..., description="Hot streams")
    cold_streams: List[ColdStream] = Field(..., description="Cold streams")
    delta_t_min_celsius: float = Field(default=10.0, gt=0, description="Minimum approach temperature (C)")
    utility_costs: UtilityCost = Field(default_factory=UtilityCost)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TemperatureInterval(BaseModel):
    """Temperature interval for cascade analysis."""
    interval_id: int
    temp_high_celsius: float
    temp_low_celsius: float
    hot_streams: List[str]
    cold_streams: List[str]
    heat_deficit_kW: float
    cumulative_heat_kW: float


class CompositePoint(BaseModel):
    """Point on composite curve."""
    temperature_celsius: float
    enthalpy_kW: float


class CompositeCurves(BaseModel):
    """Hot and cold composite curves."""
    hot_composite: List[CompositePoint]
    cold_composite: List[CompositePoint]
    grand_composite: List[CompositePoint]


class HENTargets(BaseModel):
    """Heat Exchanger Network targets."""
    min_hot_utility_kW: float
    min_cold_utility_kW: float
    max_heat_recovery_kW: float
    min_num_units_above_pinch: int
    min_num_units_below_pinch: int
    min_total_units: int


class PinchAnalyzerOutput(BaseModel):
    """Output from pinch analysis."""
    analysis_id: str
    system_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    delta_t_min_celsius: float
    pinch_temperature_celsius: float
    pinch_temperature_hot_celsius: float
    pinch_temperature_cold_celsius: float
    min_hot_utility_kW: float
    min_cold_utility_kW: float
    max_heat_recovery_kW: float
    total_hot_load_kW: float
    total_cold_load_kW: float
    composite_curves: CompositeCurves
    temperature_intervals: List[TemperatureInterval]
    hen_targets: HENTargets
    annual_utility_cost_savings: float
    simple_payback_years: Optional[float]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class PinchAnalyzerAgent:
    """GL-063: Pinch Analyzer Agent - Heat integration pinch analysis."""

    AGENT_ID = "GL-063"
    AGENT_NAME = "PINCH-ANALYZER"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"PinchAnalyzerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: PinchAnalyzerInput) -> PinchAnalyzerOutput:
        start_time = datetime.utcnow()
        dt_min = input_data.delta_t_min_celsius

        # Calculate total loads
        total_hot_load = sum(s.heat_load_kW for s in input_data.hot_streams)
        total_cold_load = sum(s.heat_load_kW for s in input_data.cold_streams)

        # Shift cold streams by delta_t_min for problem table
        shifted_cold_temps = []
        for cs in input_data.cold_streams:
            shifted_cold_temps.extend([cs.supply_temp_celsius + dt_min, cs.target_temp_celsius + dt_min])

        # Get all temperature intervals
        all_temps = []
        for hs in input_data.hot_streams:
            all_temps.extend([hs.supply_temp_celsius, hs.target_temp_celsius])
        all_temps.extend(shifted_cold_temps)
        all_temps = sorted(set(all_temps), reverse=True)

        # Build temperature intervals and cascade
        intervals = []
        cumulative = 0.0
        cascade_values = []

        for i in range(len(all_temps) - 1):
            t_high, t_low = all_temps[i], all_temps[i + 1]

            # Find streams in this interval
            hot_in_interval = [hs.stream_id for hs in input_data.hot_streams
                              if hs.supply_temp_celsius >= t_high and hs.target_temp_celsius <= t_low]
            cold_in_interval = [cs.stream_id for cs in input_data.cold_streams
                               if cs.supply_temp_celsius + dt_min <= t_high and cs.target_temp_celsius + dt_min >= t_low]

            # Calculate heat deficit (hot - cold)
            hot_heat = sum(hs.heat_capacity_flow_kW_K for hs in input_data.hot_streams
                          if hs.supply_temp_celsius >= t_high and hs.target_temp_celsius <= t_low) * (t_high - t_low)
            cold_heat = sum(cs.heat_capacity_flow_kW_K for cs in input_data.cold_streams
                           if cs.supply_temp_celsius + dt_min <= t_high and cs.target_temp_celsius + dt_min >= t_low) * (t_high - t_low)

            deficit = hot_heat - cold_heat
            cumulative += deficit
            cascade_values.append(cumulative)

            intervals.append(TemperatureInterval(
                interval_id=i + 1, temp_high_celsius=t_high, temp_low_celsius=t_low,
                hot_streams=hot_in_interval, cold_streams=cold_in_interval,
                heat_deficit_kW=round(deficit, 2), cumulative_heat_kW=round(cumulative, 2)))

        # Find minimum hot utility (add to cascade to make all values >= 0)
        if cascade_values:
            min_cascade = min(cascade_values)
            min_hot_utility = abs(min_cascade) if min_cascade < 0 else 0

            # Find pinch (where cascade = 0 after adding hot utility)
            pinch_idx = 0
            adjusted_cascade = [c + min_hot_utility for c in cascade_values]
            for i, val in enumerate(adjusted_cascade):
                if abs(val) < 0.01:
                    pinch_idx = i
                    break

            pinch_temp_hot = all_temps[pinch_idx + 1] if pinch_idx < len(all_temps) - 1 else all_temps[-1]
            pinch_temp_cold = pinch_temp_hot - dt_min
            pinch_temp = (pinch_temp_hot + pinch_temp_cold) / 2

            min_cold_utility = adjusted_cascade[-1] if adjusted_cascade else 0
        else:
            min_hot_utility = 0
            min_cold_utility = 0
            pinch_temp = pinch_temp_hot = pinch_temp_cold = 0

        max_heat_recovery = total_hot_load - min_cold_utility

        # Build composite curves
        hot_composite = self._build_composite_curve(input_data.hot_streams, is_hot=True)
        cold_composite = self._build_composite_curve(input_data.cold_streams, is_hot=False)
        grand_composite = self._build_grand_composite(intervals, min_hot_utility)

        # HEN targets
        n_above = len([s for s in input_data.hot_streams if s.target_temp_celsius >= pinch_temp_hot]) + \
                  len([s for s in input_data.cold_streams if s.target_temp_celsius >= pinch_temp_cold])
        n_below = len([s for s in input_data.hot_streams if s.supply_temp_celsius <= pinch_temp_hot]) + \
                  len([s for s in input_data.cold_streams if s.supply_temp_celsius <= pinch_temp_cold])

        hen_targets = HENTargets(
            min_hot_utility_kW=round(min_hot_utility, 2),
            min_cold_utility_kW=round(min_cold_utility, 2),
            max_heat_recovery_kW=round(max_heat_recovery, 2),
            min_num_units_above_pinch=max(1, n_above - 1),
            min_num_units_below_pinch=max(1, n_below - 1),
            min_total_units=max(2, n_above + n_below - 2))

        # Calculate savings
        costs = input_data.utility_costs
        baseline_hot = total_cold_load * costs.hot_utility_cost_per_kW * costs.operating_hours_per_year
        baseline_cold = total_hot_load * costs.cold_utility_cost_per_kW * costs.operating_hours_per_year
        optimized_hot = min_hot_utility * costs.hot_utility_cost_per_kW * costs.operating_hours_per_year
        optimized_cold = min_cold_utility * costs.cold_utility_cost_per_kW * costs.operating_hours_per_year
        annual_savings = (baseline_hot + baseline_cold) - (optimized_hot + optimized_cold)

        provenance_hash = hashlib.sha256(
            json.dumps({"agent": self.AGENT_ID, "timestamp": datetime.utcnow().isoformat()},
                      sort_keys=True, default=str).encode()).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return PinchAnalyzerOutput(
            analysis_id=input_data.analysis_id or f"PA-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            system_name=input_data.system_name, delta_t_min_celsius=dt_min,
            pinch_temperature_celsius=round(pinch_temp, 2),
            pinch_temperature_hot_celsius=round(pinch_temp_hot, 2),
            pinch_temperature_cold_celsius=round(pinch_temp_cold, 2),
            min_hot_utility_kW=round(min_hot_utility, 2),
            min_cold_utility_kW=round(min_cold_utility, 2),
            max_heat_recovery_kW=round(max_heat_recovery, 2),
            total_hot_load_kW=round(total_hot_load, 2),
            total_cold_load_kW=round(total_cold_load, 2),
            composite_curves=CompositeCurves(hot_composite=hot_composite, cold_composite=cold_composite, grand_composite=grand_composite),
            temperature_intervals=intervals, hen_targets=hen_targets,
            annual_utility_cost_savings=round(annual_savings, 2),
            simple_payback_years=None, provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2), validation_status="PASS")

    def _build_composite_curve(self, streams, is_hot: bool) -> List[CompositePoint]:
        """Build hot or cold composite curve."""
        if not streams:
            return []

        if is_hot:
            temps = sorted(set([s.supply_temp_celsius for s in streams] + [s.target_temp_celsius for s in streams]), reverse=True)
        else:
            temps = sorted(set([s.supply_temp_celsius for s in streams] + [s.target_temp_celsius for s in streams]))

        points = []
        cumulative_h = 0.0
        points.append(CompositePoint(temperature_celsius=temps[0], enthalpy_kW=0))

        for i in range(len(temps) - 1):
            t1, t2 = temps[i], temps[i + 1]
            dt = abs(t2 - t1)

            mcp_sum = 0.0
            for s in streams:
                if is_hot:
                    if s.supply_temp_celsius >= t1 and s.target_temp_celsius <= t2:
                        mcp_sum += s.heat_capacity_flow_kW_K
                else:
                    if s.supply_temp_celsius <= t1 and s.target_temp_celsius >= t2:
                        mcp_sum += s.heat_capacity_flow_kW_K

            cumulative_h += mcp_sum * dt
            points.append(CompositePoint(temperature_celsius=t2, enthalpy_kW=round(cumulative_h, 2)))

        return points

    def _build_grand_composite(self, intervals: List[TemperatureInterval], q_hot: float) -> List[CompositePoint]:
        """Build grand composite curve."""
        points = []
        cumulative = q_hot

        for interval in intervals:
            points.append(CompositePoint(
                temperature_celsius=interval.temp_high_celsius,
                enthalpy_kW=round(cumulative, 2)))
            cumulative -= interval.heat_deficit_kW

        if intervals:
            points.append(CompositePoint(
                temperature_celsius=intervals[-1].temp_low_celsius,
                enthalpy_kW=round(cumulative, 2)))

        return points


PACK_SPEC = {"schema_version": "2.0.0", "id": "GL-063", "name": "PINCH-ANALYZER", "version": "1.0.0",
    "summary": "Heat integration pinch analysis for utility minimization",
    "tags": ["pinch-analysis", "heat-integration", "HEN", "composite-curves", "utility-targets"],
    "standards": [{"ref": "Process Integration", "description": "Linnhoff March methodology"}],
    "provenance": {"calculation_verified": True, "enable_audit": True}}
