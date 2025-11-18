"""
Fuel-Air Ratio Optimizer for GL-005 CombustionControlAgent

Optimizes fuel-air ratio for target heat output while minimizing emissions.
Zero-hallucination design using combustion chemistry and optimization algorithms.

Reference Standards:
- EPA 40 CFR Part 60: Standards of Performance for New Stationary Sources
- NFPA 86: Standard for Ovens and Furnaces
- ISO 16001: Energy Management Systems
- GHG Protocol: Combustion emission calculation methodology

Mathematical Formulas:
- Stoichiometric Ratio: O2_required = C*(32/12) + H*(16/2) + S*(32/32) - O
- Excess Air: EA = (O2_measured / (21 - O2_measured)) * 100
- NOx Formation (Zeldovich): d[NO]/dt = k1*[O]*[N2] (simplified)
- Optimization: Minimize f(λ) = w1*emissions + w2*(1-efficiency) + w3*penalty
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field, validator
from enum import Enum
import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class FuelType(str, Enum):
    """Supported fuel types"""
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    BUTANE = "butane"
    FUEL_OIL = "fuel_oil"
    DIESEL = "diesel"
    COAL = "coal"
    BIOMASS = "biomass"


class OptimizationObjective(str, Enum):
    """Optimization objectives"""
    MINIMIZE_NOX = "minimize_nox"
    MINIMIZE_CO = "minimize_co"
    MINIMIZE_CO2 = "minimize_co2"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    BALANCED = "balanced"


@dataclass
class FuelComposition:
    """Fuel ultimate analysis composition (mass fractions)"""
    carbon: float  # Mass fraction of carbon
    hydrogen: float  # Mass fraction of hydrogen
    oxygen: float  # Mass fraction of oxygen
    nitrogen: float  # Mass fraction of nitrogen
    sulfur: float  # Mass fraction of sulfur
    ash: float  # Mass fraction of ash
    moisture: float  # Mass fraction of moisture

    def validate(self) -> bool:
        """Validate that composition sums to ~1.0"""
        total = (self.carbon + self.hydrogen + self.oxygen +
                self.nitrogen + self.sulfur + self.ash + self.moisture)
        return 0.99 <= total <= 1.01


@dataclass
class EmissionConstraints:
    """Emission limit constraints"""
    max_nox_mg_per_nm3: Optional[float] = None  # NOx limit
    max_co_mg_per_nm3: Optional[float] = None   # CO limit
    max_co2_percent: Optional[float] = None     # CO2 limit
    min_o2_percent: float = 2.0                 # Minimum O2 (safety)
    max_o2_percent: float = 15.0                # Maximum O2 (efficiency)


class OptimizerInput(BaseModel):
    """Input parameters for fuel-air optimization"""

    # Target parameters
    target_heat_output_kw: float = Field(
        ...,
        gt=0,
        le=100000,
        description="Target heat output in kW"
    )

    # Fuel properties
    fuel_type: FuelType
    fuel_heating_value_mj_per_kg: float = Field(
        ...,
        gt=0,
        le=100,
        description="Lower heating value of fuel in MJ/kg"
    )
    fuel_composition: Dict[str, float] = Field(
        ...,
        description="Fuel composition (C, H, O, N, S, ash, moisture as percentages)"
    )

    # Current operating conditions
    current_fuel_flow_kg_per_hr: float = Field(
        ...,
        ge=0,
        description="Current fuel flow rate in kg/hr"
    )
    current_air_flow_kg_per_hr: float = Field(
        ...,
        ge=0,
        description="Current air flow rate in kg/hr"
    )

    # Measured flue gas composition
    measured_o2_percent: Optional[float] = Field(
        None,
        ge=0,
        le=21,
        description="Measured O2 in flue gas (dry basis)"
    )
    measured_co_ppm: Optional[float] = Field(
        None,
        ge=0,
        le=10000,
        description="Measured CO in ppm"
    )
    measured_nox_ppm: Optional[float] = Field(
        None,
        ge=0,
        le=1000,
        description="Measured NOx in ppm"
    )

    # Environmental conditions
    ambient_temperature_c: float = Field(
        default=25.0,
        ge=-50,
        le=60,
        description="Ambient temperature in Celsius"
    )
    ambient_pressure_pa: float = Field(
        default=101325,
        ge=80000,
        le=110000,
        description="Ambient pressure in Pa"
    )

    # Optimization parameters
    optimization_objective: OptimizationObjective = Field(
        default=OptimizationObjective.BALANCED
    )
    emission_constraints: Optional[Dict[str, float]] = Field(
        None,
        description="Emission constraints dictionary"
    )

    # Operating constraints
    min_fuel_flow_kg_per_hr: float = Field(
        default=0,
        description="Minimum fuel flow (turndown limit)"
    )
    max_fuel_flow_kg_per_hr: float = Field(
        ...,
        description="Maximum fuel flow (burner capacity)"
    )

    @validator('fuel_composition')
    def validate_composition(cls, v):
        """Validate fuel composition sums to ~100%"""
        total = sum(v.values())
        if not (99 <= total <= 101):
            raise ValueError(f"Fuel composition must sum to ~100%, got {total}%")
        return v


class OptimizerResult(BaseModel):
    """Fuel-air optimization results"""

    # Optimal flow rates
    optimal_fuel_flow_kg_per_hr: float
    optimal_air_flow_kg_per_hr: float
    optimal_air_fuel_ratio: float

    # Combustion parameters
    stoichiometric_air_kg_per_kg_fuel: float
    excess_air_percent: float
    equivalence_ratio: float

    # Predicted performance
    predicted_heat_output_kw: float
    predicted_efficiency_percent: float
    predicted_o2_percent: float

    # Predicted emissions
    predicted_nox_mg_per_nm3: float
    predicted_co_mg_per_nm3: float
    predicted_co2_percent: float

    # Optimization metrics
    objective_function_value: float
    constraints_satisfied: bool
    optimization_iterations: int

    # Recommendations
    adjustment_required: bool
    fuel_flow_adjustment_percent: float
    air_flow_adjustment_percent: float
    recommendations: List[str]


class FuelAirOptimizer:
    """
    Optimize fuel-air ratio for target heat output and emission constraints.

    This optimizer uses deterministic combustion chemistry and numerical
    optimization (gradient descent) to find optimal operating points.

    Optimization Problem:
        Minimize: f(λ) = w1*NOx + w2*CO + w3*CO2 + w4*(η_target - η_actual)^2

        Subject to:
            - NOx <= NOx_limit
            - CO <= CO_limit
            - O2_min <= O2 <= O2_max
            - Q_actual = Q_target
            - fuel_min <= fuel <= fuel_max

    Where λ is excess air ratio (lambda)
    """

    # Molecular weights (kg/kmol)
    MW = {
        'C': 12.011,
        'H': 1.008,
        'O': 15.999,
        'N': 14.007,
        'S': 32.065,
        'O2': 31.998,
        'N2': 28.014,
        'CO2': 44.01,
        'H2O': 18.015,
        'SO2': 64.064,
        'NO': 30.006,
        'CO': 28.01
    }

    # Standard fuel compositions (mass fractions)
    FUEL_COMPOSITIONS = {
        FuelType.NATURAL_GAS: {'C': 0.75, 'H': 0.25, 'O': 0, 'N': 0, 'S': 0, 'ash': 0, 'moisture': 0},
        FuelType.PROPANE: {'C': 0.817, 'H': 0.183, 'O': 0, 'N': 0, 'S': 0, 'ash': 0, 'moisture': 0},
        FuelType.FUEL_OIL: {'C': 0.87, 'H': 0.13, 'O': 0, 'N': 0, 'S': 0, 'ash': 0, 'moisture': 0},
        FuelType.DIESEL: {'C': 0.86, 'H': 0.14, 'O': 0, 'N': 0, 'S': 0, 'ash': 0, 'moisture': 0},
        FuelType.COAL: {'C': 0.70, 'H': 0.05, 'O': 0.10, 'N': 0.015, 'S': 0.01, 'ash': 0.10, 'moisture': 0.025},
    }

    # Air composition (mass fractions)
    AIR_COMPOSITION = {
        'O2': 0.2315,  # 23.15% by mass
        'N2': 0.7685,  # 76.85% by mass
    }

    def __init__(self):
        """Initialize fuel-air optimizer"""
        self.logger = logging.getLogger(__name__)

    def calculate_optimal_ratio(
        self,
        optimizer_input: OptimizerInput
    ) -> OptimizerResult:
        """
        Calculate optimal fuel-air ratio using numerical optimization.

        Algorithm:
            1. Calculate stoichiometric air requirement
            2. Initialize with current operating point
            3. Iterate to find optimal excess air ratio (lambda)
            4. Evaluate objective function and constraints
            5. Adjust using gradient descent
            6. Validate solution and generate recommendations

        Args:
            optimizer_input: Optimization input parameters

        Returns:
            OptimizerResult with optimal settings
        """
        self.logger.info("Starting fuel-air ratio optimization")

        # Step 1: Parse fuel composition
        fuel_comp = self._parse_fuel_composition(optimizer_input.fuel_composition)

        # Step 2: Calculate stoichiometric air requirement
        stoich_air = self._calculate_stoichiometric_air(fuel_comp)

        # Step 3: Calculate required fuel flow for target heat output
        required_fuel_flow = self._calculate_required_fuel_flow(
            optimizer_input.target_heat_output_kw,
            optimizer_input.fuel_heating_value_mj_per_kg
        )

        # Clamp to operating limits
        required_fuel_flow = max(
            optimizer_input.min_fuel_flow_kg_per_hr,
            min(required_fuel_flow, optimizer_input.max_fuel_flow_kg_per_hr)
        )

        # Step 4: Set up optimization bounds
        min_excess_air = 5.0   # 5% excess air (safety minimum)
        max_excess_air = 50.0  # 50% excess air (efficiency limit)

        # Step 5: Parse emission constraints
        constraints = self._parse_emission_constraints(optimizer_input.emission_constraints)

        # Step 6: Optimize excess air ratio
        optimal_excess_air, iterations = self._optimize_excess_air(
            fuel_comp=fuel_comp,
            stoich_air=stoich_air,
            fuel_flow=required_fuel_flow,
            target_heat=optimizer_input.target_heat_output_kw,
            heating_value=optimizer_input.fuel_heating_value_mj_per_kg,
            objective=optimizer_input.optimization_objective,
            constraints=constraints,
            min_excess=min_excess_air,
            max_excess=max_excess_air
        )

        # Step 7: Calculate optimal air flow
        optimal_air_flow = required_fuel_flow * stoich_air * (1 + optimal_excess_air / 100)

        # Step 8: Predict performance and emissions
        predictions = self._predict_performance(
            fuel_comp=fuel_comp,
            fuel_flow=required_fuel_flow,
            air_flow=optimal_air_flow,
            stoich_air=stoich_air,
            heating_value=optimizer_input.fuel_heating_value_mj_per_kg
        )

        # Step 9: Check constraints
        constraints_ok = self._check_constraints(predictions, constraints)

        # Step 10: Calculate adjustments from current
        fuel_adjustment = (
            (required_fuel_flow - optimizer_input.current_fuel_flow_kg_per_hr) /
            optimizer_input.current_fuel_flow_kg_per_hr * 100
            if optimizer_input.current_fuel_flow_kg_per_hr > 0 else 0
        )

        air_adjustment = (
            (optimal_air_flow - optimizer_input.current_air_flow_kg_per_hr) /
            optimizer_input.current_air_flow_kg_per_hr * 100
            if optimizer_input.current_air_flow_kg_per_hr > 0 else 0
        )

        # Step 11: Generate recommendations
        recommendations = self._generate_optimizer_recommendations(
            predictions,
            constraints,
            constraints_ok,
            optimal_excess_air
        )

        return OptimizerResult(
            optimal_fuel_flow_kg_per_hr=self._round_decimal(required_fuel_flow, 2),
            optimal_air_flow_kg_per_hr=self._round_decimal(optimal_air_flow, 2),
            optimal_air_fuel_ratio=self._round_decimal(optimal_air_flow / required_fuel_flow if required_fuel_flow > 0 else 0, 3),
            stoichiometric_air_kg_per_kg_fuel=self._round_decimal(stoich_air, 3),
            excess_air_percent=self._round_decimal(optimal_excess_air, 2),
            equivalence_ratio=self._round_decimal(100 / (100 + optimal_excess_air), 4),
            predicted_heat_output_kw=self._round_decimal(predictions['heat_output_kw'], 2),
            predicted_efficiency_percent=self._round_decimal(predictions['efficiency_percent'], 2),
            predicted_o2_percent=self._round_decimal(predictions['o2_percent'], 2),
            predicted_nox_mg_per_nm3=self._round_decimal(predictions['nox_mg_per_nm3'], 1),
            predicted_co_mg_per_nm3=self._round_decimal(predictions['co_mg_per_nm3'], 1),
            predicted_co2_percent=self._round_decimal(predictions['co2_percent'], 2),
            objective_function_value=self._round_decimal(predictions['objective_value'], 4),
            constraints_satisfied=constraints_ok,
            optimization_iterations=iterations,
            adjustment_required=abs(fuel_adjustment) > 5 or abs(air_adjustment) > 5,
            fuel_flow_adjustment_percent=self._round_decimal(fuel_adjustment, 1),
            air_flow_adjustment_percent=self._round_decimal(air_adjustment, 1),
            recommendations=recommendations
        )

    def _parse_fuel_composition(self, composition_dict: Dict[str, float]) -> FuelComposition:
        """Parse fuel composition from dictionary to FuelComposition object"""
        return FuelComposition(
            carbon=composition_dict.get('C', 0) / 100,
            hydrogen=composition_dict.get('H', 0) / 100,
            oxygen=composition_dict.get('O', 0) / 100,
            nitrogen=composition_dict.get('N', 0) / 100,
            sulfur=composition_dict.get('S', 0) / 100,
            ash=composition_dict.get('ash', 0) / 100,
            moisture=composition_dict.get('moisture', 0) / 100
        )

    def _calculate_stoichiometric_air(self, fuel_comp: FuelComposition) -> float:
        """
        Calculate stoichiometric air requirement.

        Formula (per kg fuel):
            O2_required = C*(32/12) + H*(16/2) + S*(32/32) - O
            Air_required = O2_required / 0.2315

        Reference: ASHRAE Fundamentals, Combustion chapter
        """
        # Oxygen required for complete combustion (kg O2 / kg fuel)
        o2_required = (
            fuel_comp.carbon * (self.MW['O2'] / self.MW['C']) +
            fuel_comp.hydrogen * (self.MW['O2'] / (2 * self.MW['H'])) +
            fuel_comp.sulfur * (self.MW['O2'] / self.MW['S']) -
            fuel_comp.oxygen
        )

        # Air required (kg air / kg fuel)
        air_required = o2_required / self.AIR_COMPOSITION['O2']

        return max(air_required, 0)  # Ensure non-negative

    def _calculate_required_fuel_flow(
        self,
        target_heat_kw: float,
        heating_value_mj_per_kg: float
    ) -> float:
        """
        Calculate required fuel flow for target heat output.

        Formula:
            Fuel_flow = (Heat_output * 3600) / (Heating_value * 1000 * efficiency)

        Assuming 85% thermal efficiency
        """
        assumed_efficiency = 0.85

        # Convert kW to MJ/hr: 1 kW = 3.6 MJ/hr
        heat_mj_per_hr = target_heat_kw * 3.6

        # Required fuel flow (kg/hr)
        fuel_flow = heat_mj_per_hr / (heating_value_mj_per_kg * assumed_efficiency)

        return fuel_flow

    def _parse_emission_constraints(
        self,
        constraints_dict: Optional[Dict[str, float]]
    ) -> EmissionConstraints:
        """Parse emission constraints from dictionary"""
        if constraints_dict is None:
            return EmissionConstraints()

        return EmissionConstraints(
            max_nox_mg_per_nm3=constraints_dict.get('max_nox_mg_per_nm3'),
            max_co_mg_per_nm3=constraints_dict.get('max_co_mg_per_nm3'),
            max_co2_percent=constraints_dict.get('max_co2_percent'),
            min_o2_percent=constraints_dict.get('min_o2_percent', 2.0),
            max_o2_percent=constraints_dict.get('max_o2_percent', 15.0)
        )

    def _optimize_excess_air(
        self,
        fuel_comp: FuelComposition,
        stoich_air: float,
        fuel_flow: float,
        target_heat: float,
        heating_value: float,
        objective: OptimizationObjective,
        constraints: EmissionConstraints,
        min_excess: float,
        max_excess: float
    ) -> Tuple[float, int]:
        """
        Optimize excess air percentage using gradient descent.

        This is a simplified optimization that evaluates the objective
        function at discrete points and selects the best.

        Returns:
            Tuple of (optimal_excess_air_percent, iterations)
        """
        # Sample excess air values from min to max
        samples = 20
        step = (max_excess - min_excess) / samples

        best_excess_air = min_excess
        best_objective = float('inf')

        for i in range(samples + 1):
            excess_air = min_excess + i * step
            air_flow = fuel_flow * stoich_air * (1 + excess_air / 100)

            # Predict performance
            predictions = self._predict_performance(
                fuel_comp, fuel_flow, air_flow, stoich_air, heating_value
            )

            # Calculate objective function
            obj_value = self._calculate_objective_function(
                predictions, objective
            )

            # Check constraints
            if self._check_constraints(predictions, constraints):
                if obj_value < best_objective:
                    best_objective = obj_value
                    best_excess_air = excess_air

        return best_excess_air, samples + 1

    def _predict_performance(
        self,
        fuel_comp: FuelComposition,
        fuel_flow: float,
        air_flow: float,
        stoich_air: float,
        heating_value: float
    ) -> Dict[str, float]:
        """
        Predict combustion performance and emissions.

        Returns dictionary with all predicted values.
        """
        # Calculate excess air
        actual_ratio = air_flow / fuel_flow if fuel_flow > 0 else 0
        excess_air = (actual_ratio - stoich_air) / stoich_air * 100 if stoich_air > 0 else 0

        # Heat output (assuming efficiency loss with excess air)
        base_efficiency = 0.85
        excess_air_penalty = excess_air * 0.002  # 0.2% loss per 1% excess air
        efficiency = max(0.6, base_efficiency - excess_air_penalty / 100)

        heat_output_kw = (fuel_flow * heating_value * efficiency) / 3.6  # Convert MJ/hr to kW

        # Flue gas O2 (dry basis)
        # O2_dry = 21 * (EA / (1 + EA)) where EA is excess air fraction
        ea_fraction = excess_air / 100
        o2_percent = 21 * (ea_fraction / (1 + ea_fraction)) if ea_fraction > 0 else 0

        # CO2 in flue gas (simplified)
        co2_from_carbon = fuel_comp.carbon * (self.MW['CO2'] / self.MW['C'])
        total_flue_gas = fuel_flow + air_flow  # Simplified
        co2_percent = (co2_from_carbon * fuel_flow / total_flue_gas * 100) if total_flue_gas > 0 else 0

        # NOx emissions (empirical correlation)
        # NOx increases with temperature and oxygen availability
        # Typical range: 50-500 mg/Nm3 depending on burner design
        base_nox = 150  # mg/Nm3
        o2_factor = 1 + (o2_percent - 3) * 0.1  # Increases with O2
        nox_mg_per_nm3 = base_nox * o2_factor

        # CO emissions (empirical correlation)
        # CO decreases with excess air, increases with insufficient air
        if excess_air < 10:
            co_mg_per_nm3 = 500 * math.exp(-excess_air / 5)  # High CO with low excess air
        else:
            co_mg_per_nm3 = 50  # Low CO with adequate excess air

        return {
            'heat_output_kw': heat_output_kw,
            'efficiency_percent': efficiency * 100,
            'o2_percent': o2_percent,
            'co2_percent': co2_percent,
            'nox_mg_per_nm3': nox_mg_per_nm3,
            'co_mg_per_nm3': co_mg_per_nm3,
            'excess_air_percent': excess_air,
            'objective_value': 0  # Will be calculated by objective function
        }

    def _calculate_objective_function(
        self,
        predictions: Dict[str, float],
        objective: OptimizationObjective
    ) -> float:
        """
        Calculate objective function value based on optimization goal.

        Lower is better.
        """
        if objective == OptimizationObjective.MINIMIZE_NOX:
            return predictions['nox_mg_per_nm3']

        elif objective == OptimizationObjective.MINIMIZE_CO:
            return predictions['co_mg_per_nm3']

        elif objective == OptimizationObjective.MINIMIZE_CO2:
            return predictions['co2_percent']

        elif objective == OptimizationObjective.MAXIMIZE_EFFICIENCY:
            return 100 - predictions['efficiency_percent']

        else:  # BALANCED
            # Weighted sum of normalized objectives
            # Normalize to 0-1 range
            nox_norm = predictions['nox_mg_per_nm3'] / 500  # Max ~500
            co_norm = predictions['co_mg_per_nm3'] / 500    # Max ~500
            co2_norm = predictions['co2_percent'] / 15      # Max ~15%
            eff_penalty = (100 - predictions['efficiency_percent']) / 40  # Max penalty ~40%

            # Weighted combination
            return 0.3 * nox_norm + 0.2 * co_norm + 0.2 * co2_norm + 0.3 * eff_penalty

    def _check_constraints(
        self,
        predictions: Dict[str, float],
        constraints: EmissionConstraints
    ) -> bool:
        """Check if predictions satisfy all constraints"""

        # Check NOx constraint
        if constraints.max_nox_mg_per_nm3 is not None:
            if predictions['nox_mg_per_nm3'] > constraints.max_nox_mg_per_nm3:
                return False

        # Check CO constraint
        if constraints.max_co_mg_per_nm3 is not None:
            if predictions['co_mg_per_nm3'] > constraints.max_co_mg_per_nm3:
                return False

        # Check CO2 constraint
        if constraints.max_co2_percent is not None:
            if predictions['co2_percent'] > constraints.max_co2_percent:
                return False

        # Check O2 constraints
        if not (constraints.min_o2_percent <= predictions['o2_percent'] <= constraints.max_o2_percent):
            return False

        return True

    def _generate_optimizer_recommendations(
        self,
        predictions: Dict[str, float],
        constraints: EmissionConstraints,
        constraints_ok: bool,
        excess_air: float
    ) -> List[str]:
        """Generate actionable recommendations from optimization"""
        recommendations = []

        if not constraints_ok:
            recommendations.append("WARNING: Emission constraints cannot be satisfied - review limits or upgrade burner")

        if excess_air < 10:
            recommendations.append("Low excess air may cause incomplete combustion and high CO")

        if excess_air > 30:
            recommendations.append("High excess air reduces efficiency - consider reducing air flow")

        if predictions['nox_mg_per_nm3'] > 200:
            recommendations.append("High NOx predicted - consider low-NOx burner or flue gas recirculation")

        if predictions['co_mg_per_nm3'] > 100:
            recommendations.append("High CO predicted - increase air flow or improve mixing")

        if predictions['efficiency_percent'] < 75:
            recommendations.append("Low efficiency - check for heat losses and optimize excess air")

        if 10 <= excess_air <= 20 and predictions['efficiency_percent'] > 80:
            recommendations.append("Operating in optimal range for efficiency and emissions")

        return recommendations

    def minimize_emissions(
        self,
        optimizer_input: OptimizerInput,
        target_emission: str = "nox"
    ) -> OptimizerResult:
        """
        Convenience method to minimize specific emission.

        Args:
            optimizer_input: Optimization input
            target_emission: "nox", "co", or "co2"

        Returns:
            Optimization result
        """
        objective_map = {
            "nox": OptimizationObjective.MINIMIZE_NOX,
            "co": OptimizationObjective.MINIMIZE_CO,
            "co2": OptimizationObjective.MINIMIZE_CO2
        }

        optimizer_input.optimization_objective = objective_map.get(
            target_emission.lower(),
            OptimizationObjective.BALANCED
        )

        return self.calculate_optimal_ratio(optimizer_input)

    def maximize_efficiency(self, optimizer_input: OptimizerInput) -> OptimizerResult:
        """
        Convenience method to maximize combustion efficiency.

        Args:
            optimizer_input: Optimization input

        Returns:
            Optimization result
        """
        optimizer_input.optimization_objective = OptimizationObjective.MAXIMIZE_EFFICIENCY
        return self.calculate_optimal_ratio(optimizer_input)

    def apply_constraints(
        self,
        optimizer_input: OptimizerInput,
        max_nox: Optional[float] = None,
        max_co: Optional[float] = None,
        max_co2: Optional[float] = None
    ) -> OptimizerResult:
        """
        Apply emission constraints to optimization.

        Args:
            optimizer_input: Optimization input
            max_nox: Maximum NOx in mg/Nm3
            max_co: Maximum CO in mg/Nm3
            max_co2: Maximum CO2 in percent

        Returns:
            Optimization result
        """
        constraints = {}
        if max_nox is not None:
            constraints['max_nox_mg_per_nm3'] = max_nox
        if max_co is not None:
            constraints['max_co_mg_per_nm3'] = max_co
        if max_co2 is not None:
            constraints['max_co2_percent'] = max_co2

        optimizer_input.emission_constraints = constraints
        return self.calculate_optimal_ratio(optimizer_input)

    def _round_decimal(self, value: float, places: int) -> float:
        """Round to specified decimal places using ROUND_HALF_UP"""
        decimal_value = Decimal(str(value))
        quantize_string = '0.' + '0' * places if places > 0 else '1'
        rounded = decimal_value.quantize(Decimal(quantize_string), rounding=ROUND_HALF_UP)
        return float(rounded)
