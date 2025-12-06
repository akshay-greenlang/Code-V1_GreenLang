"""
GL-006 WasteHeatRecovery Agent - Waste Heat Analyzer

Identifies waste heat recovery opportunities using thermal analysis
and economic evaluation.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.shared.calculation_library import (
    ThermalIQCalculationLibrary,
)
from greenlang.agents.process_heat.shared.base_agent import (
    BaseProcessHeatAgent,
    AgentConfig,
    SafetyLevel,
)

logger = logging.getLogger(__name__)


class WasteHeatSource(BaseModel):
    """Waste heat source definition."""

    source_id: str = Field(..., description="Source identifier")
    source_type: str = Field(
        ...,
        description="Type (exhaust_gas, hot_water, steam, process)"
    )
    temperature_f: float = Field(..., description="Source temperature")
    flow_rate: float = Field(..., gt=0, description="Mass flow rate")
    flow_unit: str = Field(default="lb/hr", description="Flow rate unit")
    specific_heat: float = Field(default=0.24, description="Specific heat")
    availability_pct: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Availability percentage"
    )
    operating_hours_yr: int = Field(
        default=8760,
        ge=0,
        le=8760,
        description="Annual operating hours"
    )
    min_discharge_temp_f: Optional[float] = Field(
        default=None,
        description="Minimum discharge temperature constraint"
    )
    acid_dew_point_f: Optional[float] = Field(
        default=None,
        description="Acid dew point if applicable"
    )


class WasteHeatSink(BaseModel):
    """Waste heat sink (demand) definition."""

    sink_id: str = Field(..., description="Sink identifier")
    sink_type: str = Field(
        ...,
        description="Type (process_heating, preheating, space_heating)"
    )
    required_temperature_f: float = Field(..., description="Required temperature")
    inlet_temperature_f: float = Field(..., description="Inlet temperature")
    flow_rate: float = Field(..., gt=0, description="Mass flow rate")
    flow_unit: str = Field(default="lb/hr", description="Flow rate unit")
    specific_heat: float = Field(default=1.0, description="Specific heat")
    current_energy_source: str = Field(
        default="natural_gas",
        description="Current energy source"
    )
    current_cost_per_mmbtu: float = Field(
        default=5.0,
        ge=0,
        description="Current energy cost ($/MMBTU)"
    )


class RecoveryOpportunity(BaseModel):
    """Identified waste heat recovery opportunity."""

    opportunity_id: str = Field(...)
    source_id: str = Field(...)
    sink_id: str = Field(...)
    recoverable_heat_btu_hr: float = Field(...)
    recoverable_heat_mmbtu_yr: float = Field(...)
    source_outlet_temp_f: float = Field(...)
    sink_outlet_temp_f: float = Field(...)
    effectiveness: float = Field(..., ge=0, le=1)
    lmtd_f: float = Field(...)
    required_ua: float = Field(...)
    estimated_hx_area_ft2: float = Field(...)
    estimated_capital_cost: float = Field(...)
    annual_savings: float = Field(...)
    simple_payback_years: float = Field(...)
    npv_10yr: float = Field(...)
    irr_pct: Optional[float] = Field(default=None)
    technical_feasibility: str = Field(...)
    implementation_complexity: str = Field(...)
    notes: List[str] = Field(default_factory=list)


class WasteHeatAnalysisOutput(BaseModel):
    """Output from waste heat analysis."""

    analysis_id: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_waste_heat_btu_hr: float = Field(...)
    total_recoverable_btu_hr: float = Field(...)
    recovery_potential_pct: float = Field(...)
    opportunities: List[RecoveryOpportunity] = Field(default_factory=list)
    pinch_temperature_f: Optional[float] = Field(default=None)
    minimum_utility_hot_btu_hr: Optional[float] = Field(default=None)
    minimum_utility_cold_btu_hr: Optional[float] = Field(default=None)
    total_annual_savings: float = Field(...)
    total_capital_cost: float = Field(...)
    portfolio_simple_payback: float = Field(...)
    recommendations: List[str] = Field(default_factory=list)


class WasteHeatAnalyzer:
    """
    Waste heat recovery opportunity analyzer.

    Identifies and evaluates waste heat recovery opportunities
    using pinch analysis principles and economic modeling.
    """

    def __init__(
        self,
        min_approach_temp_f: float = 20.0,
        discount_rate: float = 0.10,
        project_life_years: int = 10,
    ) -> None:
        """
        Initialize waste heat analyzer.

        Args:
            min_approach_temp_f: Minimum temperature approach
            discount_rate: Discount rate for NPV calculation
            project_life_years: Project evaluation period
        """
        self.min_approach_temp = min_approach_temp_f
        self.discount_rate = discount_rate
        self.project_life_years = project_life_years

        self.calc_library = ThermalIQCalculationLibrary()

        # Cost estimation factors
        self._hx_cost_per_ft2 = 150.0  # $/ft2 for shell & tube
        self._installation_factor = 1.5
        self._u_value_default = 50.0  # BTU/hr-ft2-F

        logger.info("WasteHeatAnalyzer initialized")

    def analyze(
        self,
        sources: List[WasteHeatSource],
        sinks: List[WasteHeatSink],
    ) -> WasteHeatAnalysisOutput:
        """
        Analyze waste heat recovery opportunities.

        Args:
            sources: List of waste heat sources
            sinks: List of heat sinks

        Returns:
            Analysis results with opportunities
        """
        import uuid

        logger.info(
            f"Analyzing waste heat: {len(sources)} sources, {len(sinks)} sinks"
        )

        # Calculate total waste heat
        total_waste_heat = sum(
            self._calculate_available_heat(s) for s in sources
        )

        # Match sources with sinks
        opportunities = []
        total_recoverable = 0.0

        for source in sources:
            for sink in sinks:
                opportunity = self._evaluate_opportunity(source, sink)
                if opportunity:
                    opportunities.append(opportunity)
                    total_recoverable += opportunity.recoverable_heat_btu_hr

        # Sort by economics (NPV)
        opportunities.sort(key=lambda x: x.npv_10yr, reverse=True)

        # Calculate totals
        total_savings = sum(o.annual_savings for o in opportunities)
        total_capital = sum(o.estimated_capital_cost for o in opportunities)
        portfolio_payback = total_capital / total_savings if total_savings > 0 else float('inf')

        # Perform pinch analysis if enough data
        pinch_temp = None
        min_hot = None
        min_cold = None

        if len(sources) >= 2 and len(sinks) >= 2:
            pinch_result = self._pinch_analysis(sources, sinks)
            pinch_temp = pinch_result.get("pinch_temperature")
            min_hot = pinch_result.get("minimum_hot_utility")
            min_cold = pinch_result.get("minimum_cold_utility")

        # Generate recommendations
        recommendations = self._generate_recommendations(
            sources, sinks, opportunities, total_waste_heat, total_recoverable
        )

        recovery_potential = (total_recoverable / total_waste_heat * 100) if total_waste_heat > 0 else 0

        return WasteHeatAnalysisOutput(
            analysis_id=str(uuid.uuid4())[:8],
            total_waste_heat_btu_hr=round(total_waste_heat, 0),
            total_recoverable_btu_hr=round(total_recoverable, 0),
            recovery_potential_pct=round(recovery_potential, 1),
            opportunities=opportunities,
            pinch_temperature_f=pinch_temp,
            minimum_utility_hot_btu_hr=min_hot,
            minimum_utility_cold_btu_hr=min_cold,
            total_annual_savings=round(total_savings, 0),
            total_capital_cost=round(total_capital, 0),
            portfolio_simple_payback=round(portfolio_payback, 2),
            recommendations=recommendations,
        )

    def _calculate_available_heat(self, source: WasteHeatSource) -> float:
        """Calculate available waste heat from a source."""
        min_temp = source.min_discharge_temp_f or 150  # Default minimum

        # Acid dew point constraint
        if source.acid_dew_point_f:
            min_temp = max(min_temp, source.acid_dew_point_f + 25)

        temp_drop = source.temperature_f - min_temp
        if temp_drop <= 0:
            return 0.0

        heat = source.flow_rate * source.specific_heat * temp_drop
        heat *= source.availability_pct / 100

        return heat

    def _evaluate_opportunity(
        self,
        source: WasteHeatSource,
        sink: WasteHeatSink,
    ) -> Optional[RecoveryOpportunity]:
        """Evaluate a source-sink pairing for heat recovery."""
        import uuid

        # Check temperature feasibility
        if source.temperature_f < sink.required_temperature_f + self.min_approach_temp:
            return None  # Source not hot enough

        # Calculate heat exchange
        min_discharge = source.min_discharge_temp_f or 150
        if source.acid_dew_point_f:
            min_discharge = max(min_discharge, source.acid_dew_point_f + 25)

        # Maximum recoverable from source
        source_capacity = (
            source.flow_rate * source.specific_heat *
            (source.temperature_f - min_discharge)
        )

        # Heat required by sink
        sink_demand = (
            sink.flow_rate * sink.specific_heat *
            (sink.required_temperature_f - sink.inlet_temperature_f)
        )

        # Recoverable is minimum of available and required
        recoverable = min(source_capacity, sink_demand)

        if recoverable <= 0:
            return None

        # Calculate outlet temperatures
        source_outlet = source.temperature_f - (
            recoverable / (source.flow_rate * source.specific_heat)
        )
        sink_outlet = sink.inlet_temperature_f + (
            recoverable / (sink.flow_rate * sink.specific_heat)
        )

        # Check approach temperatures
        if source_outlet < sink.inlet_temperature_f + self.min_approach_temp:
            # Recalculate with constraint
            source_outlet = sink.inlet_temperature_f + self.min_approach_temp
            recoverable = (
                source.flow_rate * source.specific_heat *
                (source.temperature_f - source_outlet)
            )
            sink_outlet = sink.inlet_temperature_f + (
                recoverable / (sink.flow_rate * sink.specific_heat)
            )

        # Calculate LMTD (counterflow)
        dt1 = source.temperature_f - sink_outlet
        dt2 = source_outlet - sink.inlet_temperature_f

        if dt1 <= 0 or dt2 <= 0:
            return None

        if abs(dt1 - dt2) < 0.1:
            lmtd = dt1
        else:
            lmtd = (dt1 - dt2) / math.log(dt1 / dt2)

        # Calculate effectiveness
        c_min = min(
            source.flow_rate * source.specific_heat,
            sink.flow_rate * sink.specific_heat
        )
        max_possible = c_min * (source.temperature_f - sink.inlet_temperature_f)
        effectiveness = recoverable / max_possible if max_possible > 0 else 0

        # Estimate heat exchanger
        required_ua = recoverable / lmtd if lmtd > 0 else 0
        hx_area = required_ua / self._u_value_default

        # Cost estimation
        capital_cost = hx_area * self._hx_cost_per_ft2 * self._installation_factor
        capital_cost = max(capital_cost, 10000)  # Minimum project cost

        # Annual savings
        operating_hours = min(source.operating_hours_yr, 8760)
        annual_heat_mmbtu = recoverable * operating_hours / 1e6
        annual_savings = annual_heat_mmbtu * sink.current_cost_per_mmbtu

        # Economics
        simple_payback = capital_cost / annual_savings if annual_savings > 0 else float('inf')

        # NPV calculation
        npv = -capital_cost
        for year in range(1, self.project_life_years + 1):
            npv += annual_savings / ((1 + self.discount_rate) ** year)

        # Technical feasibility
        if effectiveness > 0.8:
            feasibility = "challenging"
        elif effectiveness > 0.6:
            feasibility = "moderate"
        else:
            feasibility = "straightforward"

        # Implementation complexity
        complexity = "low"
        notes = []

        if source.acid_dew_point_f and source_outlet < source.acid_dew_point_f + 50:
            complexity = "medium"
            notes.append("Corrosion-resistant materials may be required")

        if hx_area > 1000:
            complexity = "high"
            notes.append("Large heat exchanger - consider multiple units")

        if effectiveness > 0.7:
            notes.append("High effectiveness requires careful design")

        return RecoveryOpportunity(
            opportunity_id=str(uuid.uuid4())[:8],
            source_id=source.source_id,
            sink_id=sink.sink_id,
            recoverable_heat_btu_hr=round(recoverable, 0),
            recoverable_heat_mmbtu_yr=round(annual_heat_mmbtu, 1),
            source_outlet_temp_f=round(source_outlet, 1),
            sink_outlet_temp_f=round(sink_outlet, 1),
            effectiveness=round(effectiveness, 3),
            lmtd_f=round(lmtd, 1),
            required_ua=round(required_ua, 2),
            estimated_hx_area_ft2=round(hx_area, 0),
            estimated_capital_cost=round(capital_cost, 0),
            annual_savings=round(annual_savings, 0),
            simple_payback_years=round(simple_payback, 2),
            npv_10yr=round(npv, 0),
            technical_feasibility=feasibility,
            implementation_complexity=complexity,
            notes=notes,
        )

    def _pinch_analysis(
        self,
        sources: List[WasteHeatSource],
        sinks: List[WasteHeatSink],
    ) -> Dict[str, Any]:
        """
        Perform simplified pinch analysis.

        Returns pinch temperature and minimum utility requirements.
        """
        # Build temperature intervals
        temps = set()
        for s in sources:
            temps.add(s.temperature_f)
            temps.add(s.min_discharge_temp_f or 150)
        for s in sinks:
            temps.add(s.inlet_temperature_f + self.min_approach_temp)
            temps.add(s.required_temperature_f + self.min_approach_temp)

        temps = sorted(temps, reverse=True)

        # Calculate cascade
        cascade = []
        cumulative = 0.0

        for i in range(len(temps) - 1):
            t_high = temps[i]
            t_low = temps[i + 1]
            dt = t_high - t_low

            # Hot streams (sources) releasing heat
            hot_cp = sum(
                s.flow_rate * s.specific_heat
                for s in sources
                if s.temperature_f >= t_high and (s.min_discharge_temp_f or 150) <= t_low
            )

            # Cold streams (sinks) absorbing heat (shifted by approach)
            cold_cp = sum(
                s.flow_rate * s.specific_heat
                for s in sinks
                if (s.required_temperature_f + self.min_approach_temp) >= t_high and
                   (s.inlet_temperature_f + self.min_approach_temp) <= t_low
            )

            interval_heat = (hot_cp - cold_cp) * dt
            cumulative += interval_heat
            cascade.append((t_low, cumulative))

        # Find pinch (where cascade is minimum)
        if cascade:
            min_cascade = min(cascade, key=lambda x: x[1])
            pinch_temp = min_cascade[0]
            min_hot = abs(min(0, min_cascade[1]))
            min_cold = abs(max(0, cascade[-1][1]))

            return {
                "pinch_temperature": pinch_temp,
                "minimum_hot_utility": min_hot,
                "minimum_cold_utility": min_cold,
            }

        return {}

    def _generate_recommendations(
        self,
        sources: List[WasteHeatSource],
        sinks: List[WasteHeatSink],
        opportunities: List[RecoveryOpportunity],
        total_waste: float,
        total_recoverable: float,
    ) -> List[str]:
        """Generate analysis recommendations."""
        recommendations = []

        # Recovery potential
        recovery_pct = (total_recoverable / total_waste * 100) if total_waste > 0 else 0
        if recovery_pct < 30:
            recommendations.append(
                f"Low recovery potential ({recovery_pct:.0f}%). "
                "Consider adding more heat sinks or process integration."
            )
        elif recovery_pct > 70:
            recommendations.append(
                f"Excellent recovery potential ({recovery_pct:.0f}%). "
                "Prioritize implementation of top opportunities."
            )

        # Best opportunities
        good_opportunities = [o for o in opportunities if o.simple_payback_years < 3]
        if good_opportunities:
            recommendations.append(
                f"{len(good_opportunities)} opportunities with <3 year payback. "
                "Consider immediate implementation."
            )

        # Challenging opportunities
        challenging = [o for o in opportunities if o.technical_feasibility == "challenging"]
        if challenging:
            recommendations.append(
                f"{len(challenging)} opportunities require detailed engineering study."
            )

        # Acid dew point concerns
        adp_sources = [s for s in sources if s.acid_dew_point_f]
        if adp_sources:
            recommendations.append(
                "Some sources have acid dew point constraints. "
                "Use corrosion-resistant materials below dew point."
            )

        return recommendations
