"""
GL-006 WasteHeatRecovery Agent - Exergy Analysis Module

This module implements comprehensive exergy (available energy) analysis for
industrial waste heat recovery systems. Exergy analysis reveals the true
thermodynamic potential and identifies improvement opportunities that
energy balance alone cannot detect.

Features:
    - Second law efficiency calculations
    - Exergy destruction by component
    - Improvement potential identification (Grassmann diagrams)
    - Dead state optimization
    - Thermoeconomic analysis (exergoeconomics)
    - Component-level exergy costing
    - Avoidable/unavoidable exergy destruction split

Standards Reference:
    - Kotas, T.J. "The Exergy Method of Thermal Plant Analysis" (Krieger, 1995)
    - Bejan, A. "Advanced Engineering Thermodynamics" (Wiley, 2016)
    - Moran, M.J. "Availability Analysis" (ASME Press, 1989)

Example:
    >>> analyzer = ExergyAnalyzer(dead_state_temp_f=77.0, dead_state_pressure_psia=14.7)
    >>> streams = [
    ...     ExergyStream(name="Exhaust", temp_f=800, mass_flow=10000, cp=0.25),
    ...     ExergyStream(name="CoolWater", temp_f=60, mass_flow=50000, cp=1.0),
    ... ]
    >>> result = analyzer.analyze_system(streams, components)
    >>> print(f"Second law efficiency: {result.second_law_efficiency_pct:.1f}%")
    >>> print(f"Total exergy destruction: {result.total_exergy_destruction_btu_hr:,.0f} BTU/hr")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Reference environment (dead state)
DEFAULT_DEAD_STATE_TEMP_F = 77.0  # 25C - ISO standard
DEFAULT_DEAD_STATE_PRESSURE_PSIA = 14.696  # 1 atm
DEFAULT_DEAD_STATE_TEMP_R = DEFAULT_DEAD_STATE_TEMP_F + 459.67  # Rankine

# Thermodynamic constants
GAS_CONSTANT_BTU_LBMOL_R = 1.986  # BTU/(lbmol-R)
STEAM_SPECIFIC_ENTROPY_BTU_LB_R = 1.7567  # at 212F, 1 atm


# =============================================================================
# ENUMS
# =============================================================================

class ComponentType(Enum):
    """Types of process components for exergy analysis."""
    HEAT_EXCHANGER = "heat_exchanger"
    BOILER = "boiler"
    TURBINE = "turbine"
    COMPRESSOR = "compressor"
    PUMP = "pump"
    VALVE = "valve"
    MIXER = "mixer"
    SEPARATOR = "separator"
    REACTOR = "reactor"
    CONDENSER = "condenser"
    EVAPORATOR = "evaporator"


class ExergyType(Enum):
    """Types of exergy."""
    PHYSICAL = "physical"      # Temperature and pressure driven
    CHEMICAL = "chemical"      # Composition driven
    KINETIC = "kinetic"        # Velocity driven
    POTENTIAL = "potential"    # Elevation driven


class DestructionCategory(Enum):
    """Categories of exergy destruction."""
    AVOIDABLE = "avoidable"          # Can be reduced with better design
    UNAVOIDABLE = "unavoidable"      # Inherent to the process
    ENDOGENOUS = "endogenous"        # Due to component itself
    EXOGENOUS = "exogenous"          # Due to other components


# =============================================================================
# DATA MODELS
# =============================================================================

class ExergyStream(BaseModel):
    """Stream definition for exergy analysis."""

    name: str = Field(..., description="Stream identifier")
    temp_f: float = Field(..., description="Stream temperature (F)")
    pressure_psia: float = Field(
        default=14.696,
        gt=0,
        description="Stream pressure (psia)"
    )
    mass_flow_lb_hr: float = Field(
        ...,
        gt=0,
        description="Mass flow rate (lb/hr)"
    )
    specific_heat_btu_lb_f: float = Field(
        default=0.24,
        gt=0,
        description="Specific heat at constant pressure (BTU/lb-F)"
    )
    specific_entropy_btu_lb_r: Optional[float] = Field(
        default=None,
        description="Specific entropy (BTU/lb-R) - calculated if not provided"
    )
    molecular_weight: float = Field(
        default=29.0,
        gt=0,
        description="Molecular weight (lb/lbmol)"
    )
    is_inlet: bool = Field(default=True, description="True if inlet stream")
    chemical_exergy_btu_lb: float = Field(
        default=0.0,
        ge=0,
        description="Specific chemical exergy (BTU/lb)"
    )

    # Calculated fields
    specific_exergy_btu_lb: Optional[float] = Field(
        default=None,
        description="Calculated specific exergy"
    )
    exergy_rate_btu_hr: Optional[float] = Field(
        default=None,
        description="Calculated exergy flow rate"
    )


class ProcessComponent(BaseModel):
    """Process component for exergy analysis."""

    component_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Component identifier"
    )
    name: str = Field(..., description="Component name")
    component_type: ComponentType = Field(..., description="Component type")
    inlet_streams: List[str] = Field(
        default_factory=list,
        description="Inlet stream names"
    )
    outlet_streams: List[str] = Field(
        default_factory=list,
        description="Outlet stream names"
    )

    # Energy/work interactions
    heat_transfer_btu_hr: float = Field(
        default=0.0,
        description="Heat transfer rate (positive into system)"
    )
    heat_transfer_temp_f: Optional[float] = Field(
        default=None,
        description="Temperature at which heat transfer occurs"
    )
    work_rate_btu_hr: float = Field(
        default=0.0,
        description="Work rate (positive out of system)"
    )

    # Cost data for thermoeconomics
    capital_cost_usd: float = Field(
        default=0.0,
        ge=0,
        description="Component capital cost"
    )
    operating_hours_yr: int = Field(
        default=8760,
        ge=0,
        le=8760,
        description="Annual operating hours"
    )
    maintenance_cost_usd_yr: float = Field(
        default=0.0,
        ge=0,
        description="Annual maintenance cost"
    )

    # Performance
    design_efficiency: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Design (first law) efficiency"
    )

    class Config:
        use_enum_values = True


class ComponentExergyResult(BaseModel):
    """Exergy analysis result for a single component."""

    component_id: str = Field(...)
    component_name: str = Field(...)
    component_type: str = Field(...)

    # Exergy flows
    exergy_input_btu_hr: float = Field(
        ...,
        description="Total exergy input"
    )
    exergy_output_btu_hr: float = Field(
        ...,
        description="Total exergy output"
    )
    exergy_destruction_btu_hr: float = Field(
        ...,
        description="Exergy destroyed (irreversibility)"
    )
    exergy_loss_btu_hr: float = Field(
        default=0.0,
        description="Exergy lost to environment"
    )

    # Efficiencies
    exergetic_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Exergetic (second law) efficiency"
    )
    first_law_efficiency_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="First law (energy) efficiency"
    )

    # Improvement potential
    improvement_potential_btu_hr: float = Field(
        ...,
        description="Potential for improvement (van Gool)"
    )
    improvement_potential_pct: float = Field(
        ...,
        description="Relative improvement potential"
    )

    # Destruction breakdown
    avoidable_destruction_btu_hr: float = Field(
        default=0.0,
        description="Avoidable exergy destruction"
    )
    unavoidable_destruction_btu_hr: float = Field(
        default=0.0,
        description="Unavoidable exergy destruction"
    )

    # Thermoeconomic metrics
    exergy_destruction_cost_usd_hr: float = Field(
        default=0.0,
        description="Cost of exergy destruction"
    )
    exergoeconomic_factor_pct: float = Field(
        default=0.0,
        description="Exergoeconomic factor (f)"
    )

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)


class SystemExergyResult(BaseModel):
    """Complete system exergy analysis result."""

    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique analysis identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Dead state
    dead_state_temp_f: float = Field(..., description="Reference temperature")
    dead_state_pressure_psia: float = Field(..., description="Reference pressure")

    # System totals
    total_exergy_input_btu_hr: float = Field(...)
    total_exergy_output_btu_hr: float = Field(...)
    total_exergy_destruction_btu_hr: float = Field(...)
    total_exergy_loss_btu_hr: float = Field(...)

    # System efficiency
    second_law_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall second law efficiency"
    )
    first_law_efficiency_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Overall first law efficiency"
    )

    # Improvement
    total_improvement_potential_btu_hr: float = Field(...)
    theoretical_minimum_exergy_btu_hr: float = Field(
        default=0.0,
        description="Theoretical minimum exergy consumption"
    )

    # Component results
    component_results: List[ComponentExergyResult] = Field(default_factory=list)

    # Ranking
    destruction_ranking: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Components ranked by exergy destruction"
    )
    improvement_ranking: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Components ranked by improvement potential"
    )

    # Thermoeconomics
    total_exergy_destruction_cost_usd_yr: float = Field(default=0.0)
    exergy_unit_cost_usd_per_btu: float = Field(default=0.0)

    # Provenance
    provenance_hash: str = Field(default="")
    calculation_method: str = Field(
        default="Kotas exergy method with thermoeconomic analysis"
    )

    # Recommendations
    system_recommendations: List[str] = Field(default_factory=list)


# =============================================================================
# EXERGY ANALYZER CLASS
# =============================================================================

class ExergyAnalyzer:
    """
    Comprehensive Exergy (Available Energy) Analyzer.

    Performs second law thermodynamic analysis to identify the true
    improvement potential in waste heat recovery systems. Exergy analysis
    reveals losses that energy balance alone cannot detect.

    Key Concepts:
        - Exergy = Available energy that can be converted to work
        - Exergy destruction = Irreversibility (entropy generation)
        - Second law efficiency = Exergy out / Exergy in
        - Improvement potential = How much better can we do?

    Attributes:
        dead_state_temp_f: Reference environment temperature
        dead_state_pressure_psia: Reference environment pressure
        fuel_exergy_cost: Cost per unit exergy of fuel

    Example:
        >>> analyzer = ExergyAnalyzer()
        >>> result = analyzer.analyze_system(streams, components)
        >>> print(result.second_law_efficiency_pct)
    """

    def __init__(
        self,
        dead_state_temp_f: float = DEFAULT_DEAD_STATE_TEMP_F,
        dead_state_pressure_psia: float = DEFAULT_DEAD_STATE_PRESSURE_PSIA,
        fuel_exergy_cost_usd_per_mmbtu: float = 8.0,
    ) -> None:
        """
        Initialize the Exergy Analyzer.

        Args:
            dead_state_temp_f: Reference (dead state) temperature (F)
            dead_state_pressure_psia: Reference pressure (psia)
            fuel_exergy_cost_usd_per_mmbtu: Cost of fuel exergy ($/MMBTU)
        """
        self.dead_state_temp_f = dead_state_temp_f
        self.dead_state_temp_r = dead_state_temp_f + 459.67  # Rankine
        self.dead_state_pressure = dead_state_pressure_psia
        self.fuel_exergy_cost = fuel_exergy_cost_usd_per_mmbtu

        logger.info(
            f"ExergyAnalyzer initialized: T0={dead_state_temp_f}F, "
            f"P0={dead_state_pressure_psia} psia"
        )

    def analyze_system(
        self,
        streams: List[ExergyStream],
        components: List[ProcessComponent],
    ) -> SystemExergyResult:
        """
        Perform complete system exergy analysis.

        Args:
            streams: List of process streams
            components: List of process components

        Returns:
            Complete system exergy analysis results

        Raises:
            ValueError: If stream/component data is invalid
        """
        logger.info(
            f"Analyzing system: {len(streams)} streams, {len(components)} components"
        )

        # Calculate stream exergies
        stream_dict = self._calculate_stream_exergies(streams)

        # Analyze each component
        component_results = []
        for component in components:
            result = self._analyze_component(component, stream_dict)
            component_results.append(result)

        # Calculate system totals
        total_destruction = sum(c.exergy_destruction_btu_hr for c in component_results)
        total_loss = sum(c.exergy_loss_btu_hr for c in component_results)
        total_input = sum(c.exergy_input_btu_hr for c in component_results)
        total_output = sum(c.exergy_output_btu_hr for c in component_results)

        # System second law efficiency
        if total_input > 0:
            second_law_eff = (1 - total_destruction / total_input) * 100
        else:
            second_law_eff = 0.0

        # Total improvement potential (van Gool method)
        total_improvement = sum(c.improvement_potential_btu_hr for c in component_results)

        # Create rankings
        destruction_ranking = self._create_destruction_ranking(component_results)
        improvement_ranking = self._create_improvement_ranking(component_results)

        # Thermoeconomic analysis
        total_destruction_cost = self._calculate_total_destruction_cost(
            component_results
        )

        # Generate system recommendations
        recommendations = self._generate_system_recommendations(
            component_results,
            second_law_eff,
            destruction_ranking,
        )

        # Calculate provenance
        provenance_hash = self._calculate_provenance(streams, components)

        result = SystemExergyResult(
            dead_state_temp_f=self.dead_state_temp_f,
            dead_state_pressure_psia=self.dead_state_pressure,
            total_exergy_input_btu_hr=round(total_input, 0),
            total_exergy_output_btu_hr=round(total_output, 0),
            total_exergy_destruction_btu_hr=round(total_destruction, 0),
            total_exergy_loss_btu_hr=round(total_loss, 0),
            second_law_efficiency_pct=round(second_law_eff, 2),
            total_improvement_potential_btu_hr=round(total_improvement, 0),
            component_results=component_results,
            destruction_ranking=destruction_ranking,
            improvement_ranking=improvement_ranking,
            total_exergy_destruction_cost_usd_yr=round(total_destruction_cost, 0),
            exergy_unit_cost_usd_per_btu=self.fuel_exergy_cost / 1e6,
            provenance_hash=provenance_hash,
            system_recommendations=recommendations,
        )

        logger.info(
            f"Exergy analysis complete: eta_II={second_law_eff:.1f}%, "
            f"destruction={total_destruction:,.0f} BTU/hr"
        )

        return result

    def calculate_physical_exergy(
        self,
        temp_f: float,
        pressure_psia: float,
        mass_flow_lb_hr: float,
        cp_btu_lb_f: float = 0.24,
    ) -> Tuple[float, float]:
        """
        Calculate physical (thermo-mechanical) exergy.

        Physical exergy consists of:
        - Thermal exergy: due to temperature difference from dead state
        - Pressure exergy: due to pressure difference from dead state

        For ideal gas: e_ph = cp*(T - T0) - T0*cp*ln(T/T0) + RT0*ln(P/P0)

        Args:
            temp_f: Stream temperature (F)
            pressure_psia: Stream pressure (psia)
            mass_flow_lb_hr: Mass flow rate (lb/hr)
            cp_btu_lb_f: Specific heat (BTU/lb-F)

        Returns:
            Tuple of (specific_exergy_btu_lb, exergy_rate_btu_hr)
        """
        temp_r = temp_f + 459.67
        t0 = self.dead_state_temp_r
        p0 = self.dead_state_pressure

        # Thermal exergy component
        # e_th = cp * (T - T0) - T0 * cp * ln(T/T0)
        if temp_r > 0 and t0 > 0:
            thermal_exergy = cp_btu_lb_f * (temp_r - t0) - t0 * cp_btu_lb_f * math.log(temp_r / t0)
        else:
            thermal_exergy = 0.0

        # Pressure exergy component (for gases)
        # e_p = R * T0 * ln(P/P0)
        if pressure_psia > 0 and p0 > 0:
            # Using cp for air: R = cp - cv, for air cp/cv = 1.4, so R = 0.4*cp
            r_gas = 0.4 * cp_btu_lb_f  # Approximate for air-like gases
            pressure_exergy = r_gas * t0 * math.log(pressure_psia / p0)
        else:
            pressure_exergy = 0.0

        specific_exergy = thermal_exergy + pressure_exergy
        exergy_rate = mass_flow_lb_hr * specific_exergy

        return specific_exergy, exergy_rate

    def calculate_heat_exergy(
        self,
        heat_rate_btu_hr: float,
        temperature_f: float,
    ) -> float:
        """
        Calculate exergy associated with heat transfer.

        Exergy of heat = Q * (1 - T0/T) [Carnot factor]

        Args:
            heat_rate_btu_hr: Heat transfer rate (BTU/hr)
            temperature_f: Temperature at which heat is transferred (F)

        Returns:
            Exergy of heat transfer (BTU/hr)
        """
        temp_r = temperature_f + 459.67
        t0 = self.dead_state_temp_r

        if temp_r <= 0:
            return 0.0

        # Carnot factor
        carnot_factor = 1.0 - (t0 / temp_r)

        return heat_rate_btu_hr * carnot_factor

    def calculate_carnot_efficiency(self, temp_hot_f: float, temp_cold_f: float) -> float:
        """
        Calculate Carnot (maximum theoretical) efficiency.

        Args:
            temp_hot_f: Hot reservoir temperature (F)
            temp_cold_f: Cold reservoir temperature (F)

        Returns:
            Carnot efficiency (0 to 1)
        """
        t_hot_r = temp_hot_f + 459.67
        t_cold_r = temp_cold_f + 459.67

        if t_hot_r <= t_cold_r or t_hot_r <= 0:
            return 0.0

        return 1.0 - (t_cold_r / t_hot_r)

    def _calculate_stream_exergies(
        self,
        streams: List[ExergyStream],
    ) -> Dict[str, ExergyStream]:
        """Calculate exergy for all streams."""
        stream_dict = {}

        for stream in streams:
            specific_ex, exergy_rate = self.calculate_physical_exergy(
                temp_f=stream.temp_f,
                pressure_psia=stream.pressure_psia,
                mass_flow_lb_hr=stream.mass_flow_lb_hr,
                cp_btu_lb_f=stream.specific_heat_btu_lb_f,
            )

            # Add chemical exergy if specified
            specific_ex += stream.chemical_exergy_btu_lb
            exergy_rate += stream.mass_flow_lb_hr * stream.chemical_exergy_btu_lb

            stream.specific_exergy_btu_lb = round(specific_ex, 4)
            stream.exergy_rate_btu_hr = round(exergy_rate, 0)

            stream_dict[stream.name] = stream

        return stream_dict

    def _analyze_component(
        self,
        component: ProcessComponent,
        stream_dict: Dict[str, ExergyStream],
    ) -> ComponentExergyResult:
        """Analyze exergy for a single component."""
        # Sum inlet exergies
        exergy_in = 0.0
        for stream_name in component.inlet_streams:
            if stream_name in stream_dict:
                stream = stream_dict[stream_name]
                if stream.exergy_rate_btu_hr:
                    exergy_in += stream.exergy_rate_btu_hr

        # Add heat exergy input (if heat is transferred into component)
        if component.heat_transfer_btu_hr > 0 and component.heat_transfer_temp_f:
            heat_exergy = self.calculate_heat_exergy(
                component.heat_transfer_btu_hr,
                component.heat_transfer_temp_f,
            )
            exergy_in += heat_exergy

        # Add work input (work is pure exergy)
        if component.work_rate_btu_hr < 0:  # Work into system is negative
            exergy_in += abs(component.work_rate_btu_hr)

        # Sum outlet exergies
        exergy_out = 0.0
        for stream_name in component.outlet_streams:
            if stream_name in stream_dict:
                stream = stream_dict[stream_name]
                if stream.exergy_rate_btu_hr:
                    exergy_out += stream.exergy_rate_btu_hr

        # Add heat exergy output
        if component.heat_transfer_btu_hr < 0 and component.heat_transfer_temp_f:
            heat_exergy = self.calculate_heat_exergy(
                abs(component.heat_transfer_btu_hr),
                component.heat_transfer_temp_f,
            )
            exergy_out += heat_exergy

        # Add work output
        if component.work_rate_btu_hr > 0:  # Work out of system is positive
            exergy_out += component.work_rate_btu_hr

        # Exergy destruction = input - output (for a control volume)
        exergy_destruction = max(0, exergy_in - exergy_out)

        # Exergy loss (to environment) - typically for non-adiabatic components
        exergy_loss = 0.0  # Would need heat loss data

        # Exergetic efficiency
        if exergy_in > 0:
            exergetic_eff = (exergy_out / exergy_in) * 100
        else:
            exergetic_eff = 0.0

        # Improvement potential (van Gool method)
        # IP = (1 - eta_II) * E_destruction
        improvement_potential = (1 - exergetic_eff / 100) * exergy_destruction

        # Improvement potential percentage
        if exergy_in > 0:
            improvement_pct = (improvement_potential / exergy_in) * 100
        else:
            improvement_pct = 0.0

        # Avoidable vs unavoidable destruction
        # Simplified: assume 30% is unavoidable for heat exchangers
        unavoidable_fraction = self._get_unavoidable_fraction(component.component_type)
        unavoidable = exergy_destruction * unavoidable_fraction
        avoidable = exergy_destruction - unavoidable

        # Thermoeconomic cost of destruction
        destruction_cost = (
            exergy_destruction *
            (self.fuel_exergy_cost / 1e6) *
            component.operating_hours_yr
        )

        # Exergoeconomic factor
        # f = Z_dot / (Z_dot + C_D_dot)
        # where Z_dot = annualized capital + O&M, C_D_dot = destruction cost
        annualized_capital = component.capital_cost_usd * 0.15  # 15% CRF
        z_dot = annualized_capital + component.maintenance_cost_usd_yr
        if (z_dot + destruction_cost) > 0:
            exergoeconomic_factor = (z_dot / (z_dot + destruction_cost)) * 100
        else:
            exergoeconomic_factor = 0.0

        # Generate recommendations
        recommendations = self._generate_component_recommendations(
            component,
            exergetic_eff,
            avoidable,
            exergoeconomic_factor,
        )

        return ComponentExergyResult(
            component_id=component.component_id,
            component_name=component.name,
            component_type=component.component_type,
            exergy_input_btu_hr=round(exergy_in, 0),
            exergy_output_btu_hr=round(exergy_out, 0),
            exergy_destruction_btu_hr=round(exergy_destruction, 0),
            exergy_loss_btu_hr=round(exergy_loss, 0),
            exergetic_efficiency_pct=round(exergetic_eff, 2),
            improvement_potential_btu_hr=round(improvement_potential, 0),
            improvement_potential_pct=round(improvement_pct, 2),
            avoidable_destruction_btu_hr=round(avoidable, 0),
            unavoidable_destruction_btu_hr=round(unavoidable, 0),
            exergy_destruction_cost_usd_hr=round(destruction_cost / 8760, 2),
            exergoeconomic_factor_pct=round(exergoeconomic_factor, 2),
            recommendations=recommendations,
        )

    def _get_unavoidable_fraction(self, component_type: ComponentType) -> float:
        """Get unavoidable exergy destruction fraction by component type."""
        unavoidable_fractions = {
            ComponentType.HEAT_EXCHANGER: 0.30,
            ComponentType.BOILER: 0.25,
            ComponentType.TURBINE: 0.15,
            ComponentType.COMPRESSOR: 0.20,
            ComponentType.PUMP: 0.20,
            ComponentType.VALVE: 0.50,  # Throttling is highly irreversible
            ComponentType.MIXER: 0.60,  # Mixing is irreversible
            ComponentType.CONDENSER: 0.35,
            ComponentType.EVAPORATOR: 0.35,
            ComponentType.REACTOR: 0.40,
        }
        return unavoidable_fractions.get(component_type, 0.30)

    def _generate_component_recommendations(
        self,
        component: ProcessComponent,
        exergetic_eff: float,
        avoidable_destruction: float,
        exergoeconomic_factor: float,
    ) -> List[str]:
        """Generate recommendations for component improvement."""
        recommendations = []

        # Low exergetic efficiency
        if exergetic_eff < 50:
            recommendations.append(
                f"Low exergetic efficiency ({exergetic_eff:.1f}%). "
                "Consider redesign or replacement with higher efficiency unit."
            )
        elif exergetic_eff < 70:
            recommendations.append(
                f"Moderate exergetic efficiency ({exergetic_eff:.1f}%). "
                "Optimization opportunities exist."
            )

        # High avoidable destruction
        if avoidable_destruction > 100000:  # > 100 MBtu/hr
            recommendations.append(
                f"High avoidable exergy destruction ({avoidable_destruction:,.0f} BTU/hr). "
                "Priority target for improvement."
            )

        # Exergoeconomic factor guidance
        if exergoeconomic_factor < 25:
            recommendations.append(
                f"Low exergoeconomic factor ({exergoeconomic_factor:.1f}%). "
                "Reduce exergy destruction to improve economics."
            )
        elif exergoeconomic_factor > 75:
            recommendations.append(
                f"High exergoeconomic factor ({exergoeconomic_factor:.1f}%). "
                "Consider cost reduction vs performance trade-off."
            )

        # Component-specific recommendations
        if component.component_type == ComponentType.HEAT_EXCHANGER:
            if exergetic_eff < 60:
                recommendations.append(
                    "Increase heat exchanger area or improve delta-T approach."
                )
        elif component.component_type == ComponentType.VALVE:
            if avoidable_destruction > 50000:
                recommendations.append(
                    "Consider replacing throttle valve with expander/turbine."
                )

        return recommendations

    def _create_destruction_ranking(
        self,
        component_results: List[ComponentExergyResult],
    ) -> List[Dict[str, Any]]:
        """Create component ranking by exergy destruction."""
        total_destruction = sum(
            c.exergy_destruction_btu_hr for c in component_results
        )

        ranking = []
        for result in sorted(
            component_results,
            key=lambda x: x.exergy_destruction_btu_hr,
            reverse=True
        ):
            pct = (result.exergy_destruction_btu_hr / total_destruction * 100) if total_destruction > 0 else 0
            ranking.append({
                "rank": len(ranking) + 1,
                "component_name": result.component_name,
                "destruction_btu_hr": result.exergy_destruction_btu_hr,
                "destruction_pct": round(pct, 1),
                "exergetic_efficiency_pct": result.exergetic_efficiency_pct,
            })

        return ranking

    def _create_improvement_ranking(
        self,
        component_results: List[ComponentExergyResult],
    ) -> List[Dict[str, Any]]:
        """Create component ranking by improvement potential."""
        total_potential = sum(
            c.improvement_potential_btu_hr for c in component_results
        )

        ranking = []
        for result in sorted(
            component_results,
            key=lambda x: x.improvement_potential_btu_hr,
            reverse=True
        ):
            pct = (result.improvement_potential_btu_hr / total_potential * 100) if total_potential > 0 else 0
            ranking.append({
                "rank": len(ranking) + 1,
                "component_name": result.component_name,
                "improvement_potential_btu_hr": result.improvement_potential_btu_hr,
                "improvement_pct": round(pct, 1),
                "avoidable_destruction_btu_hr": result.avoidable_destruction_btu_hr,
            })

        return ranking

    def _calculate_total_destruction_cost(
        self,
        component_results: List[ComponentExergyResult],
    ) -> float:
        """Calculate total annual cost of exergy destruction."""
        total = sum(
            c.exergy_destruction_btu_hr for c in component_results
        )
        # Convert to annual cost
        return total * (self.fuel_exergy_cost / 1e6) * 8760

    def _generate_system_recommendations(
        self,
        component_results: List[ComponentExergyResult],
        second_law_eff: float,
        destruction_ranking: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate system-level recommendations."""
        recommendations = []

        # Overall efficiency
        if second_law_eff < 30:
            recommendations.append(
                f"Low system second law efficiency ({second_law_eff:.1f}%). "
                "Significant thermodynamic improvement potential exists."
            )
        elif second_law_eff < 50:
            recommendations.append(
                f"Moderate system second law efficiency ({second_law_eff:.1f}%). "
                "Focus on top destruction sources for improvement."
            )
        else:
            recommendations.append(
                f"Good system second law efficiency ({second_law_eff:.1f}%). "
                "Fine-tuning opportunities may still exist."
            )

        # Top destruction sources
        if destruction_ranking:
            top_source = destruction_ranking[0]
            recommendations.append(
                f"Largest exergy destruction in {top_source['component_name']} "
                f"({top_source['destruction_pct']:.1f}% of total). "
                "Prioritize improvement efforts here."
            )

        # Check for highly inefficient components
        low_eff_components = [
            c for c in component_results
            if c.exergetic_efficiency_pct < 40
        ]
        if low_eff_components:
            names = [c.component_name for c in low_eff_components]
            recommendations.append(
                f"Components with low exergetic efficiency (<40%): {', '.join(names)}. "
                "Consider upgrade or replacement."
            )

        return recommendations

    def _calculate_provenance(
        self,
        streams: List[ExergyStream],
        components: List[ProcessComponent],
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        import json

        provenance_data = {
            "streams": [s.dict() for s in streams],
            "components": [c.dict() for c in components],
            "dead_state_temp_f": self.dead_state_temp_f,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_exergy_efficiency_comparison(
    first_law_efficiency: float,
    hot_temp_f: float,
    cold_temp_f: float,
    dead_state_temp_f: float = 77.0,
) -> Dict[str, float]:
    """
    Compare first and second law efficiencies for a heat engine.

    Args:
        first_law_efficiency: Energy efficiency (0 to 1)
        hot_temp_f: Hot reservoir temperature (F)
        cold_temp_f: Cold reservoir temperature (F)
        dead_state_temp_f: Dead state temperature (F)

    Returns:
        Dictionary with efficiency comparisons
    """
    # Convert to Rankine
    t_hot_r = hot_temp_f + 459.67
    t_cold_r = cold_temp_f + 459.67
    t0_r = dead_state_temp_f + 459.67

    # Carnot efficiency
    carnot_eff = 1.0 - (t_cold_r / t_hot_r) if t_hot_r > t_cold_r else 0

    # Second law efficiency = actual / Carnot
    second_law_eff = first_law_efficiency / carnot_eff if carnot_eff > 0 else 0

    # Exergy input factor
    exergy_factor = 1.0 - (t0_r / t_hot_r) if t_hot_r > 0 else 0

    return {
        "first_law_efficiency_pct": round(first_law_efficiency * 100, 2),
        "carnot_efficiency_pct": round(carnot_eff * 100, 2),
        "second_law_efficiency_pct": round(second_law_eff * 100, 2),
        "exergy_factor": round(exergy_factor, 4),
        "quality_of_heat_source": (
            "high" if exergy_factor > 0.6 else
            "medium" if exergy_factor > 0.3 else
            "low"
        ),
    }


def estimate_improvement_payback(
    current_destruction_btu_hr: float,
    improvement_fraction: float,
    fuel_cost_usd_per_mmbtu: float,
    retrofit_cost_usd: float,
    operating_hours_yr: int = 8760,
) -> Dict[str, float]:
    """
    Estimate payback for exergy improvement project.

    Args:
        current_destruction_btu_hr: Current exergy destruction rate
        improvement_fraction: Fraction of destruction that can be reduced
        fuel_cost_usd_per_mmbtu: Fuel cost
        retrofit_cost_usd: Cost of improvement project
        operating_hours_yr: Annual operating hours

    Returns:
        Economic analysis of improvement
    """
    # Reduction in destruction
    reduction_btu_hr = current_destruction_btu_hr * improvement_fraction

    # Annual savings
    reduction_mmbtu_yr = reduction_btu_hr * operating_hours_yr / 1e6
    annual_savings = reduction_mmbtu_yr * fuel_cost_usd_per_mmbtu

    # Simple payback
    payback_years = retrofit_cost_usd / annual_savings if annual_savings > 0 else float('inf')

    return {
        "exergy_reduction_btu_hr": round(reduction_btu_hr, 0),
        "exergy_reduction_mmbtu_yr": round(reduction_mmbtu_yr, 1),
        "annual_savings_usd": round(annual_savings, 0),
        "simple_payback_years": round(payback_years, 2),
        "recommendation": (
            "Excellent - implement immediately" if payback_years < 1 else
            "Good - prioritize implementation" if payback_years < 2 else
            "Acceptable - include in capital plan" if payback_years < 5 else
            "Marginal - requires further analysis"
        ),
    }
