"""
GL-007 FurnacePulse - Efficiency Calculator

Deterministic calculator for furnace thermal efficiency, fuel consumption,
excess air, and stack loss calculations. All calculations follow the
zero-hallucination principle with SHA-256 provenance tracking.

Key Calculations:
    - Fuel Input Power (kW) = mass_flow * LHV
    - Specific Fuel Consumption (SFC) = fuel_input / useful_output
    - Thermal Efficiency (%) = (useful_output / fuel_input) * 100
    - Excess Air (%) from O2 readings
    - Stack Loss (Q_stack) using Siegert formula

Example:
    >>> calc = EfficiencyCalculator(agent_id="GL-007")
    >>> inputs = EfficiencyInputs(
    ...     fuel_mass_flow_kg_s=0.5,
    ...     fuel_lhv_kj_kg=42000,
    ...     useful_heat_output_kw=8000
    ... )
    >>> result = calc.calculate(inputs)
    >>> print(f"Efficiency: {result.result.thermal_efficiency_pct:.2f}%")
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum
import math
import sys
import os

# Add framework path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Framework_GreenLang', 'shared'))

from calculator_base import DeterministicCalculator, CalculationResult


class FuelType(str, Enum):
    """Supported fuel types with standard properties."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    PROPANE = "propane"
    HYDROGEN = "hydrogen"
    REFINERY_GAS = "refinery_gas"


# Standard fuel properties (can be overridden by user input)
FUEL_PROPERTIES: Dict[FuelType, Dict[str, float]] = {
    FuelType.NATURAL_GAS: {
        "lhv_kj_kg": 47100.0,  # Lower Heating Value
        "stoich_air_fuel_ratio": 17.2,  # kg air / kg fuel
        "co2_emission_factor": 2.75,  # kg CO2 / kg fuel
        "siegert_a1": 0.37,  # Siegert coefficient A1
        "siegert_a2": 0.009,  # Siegert coefficient A2
    },
    FuelType.FUEL_OIL: {
        "lhv_kj_kg": 42500.0,
        "stoich_air_fuel_ratio": 14.0,
        "co2_emission_factor": 3.15,
        "siegert_a1": 0.50,
        "siegert_a2": 0.007,
    },
    FuelType.PROPANE: {
        "lhv_kj_kg": 46350.0,
        "stoich_air_fuel_ratio": 15.7,
        "co2_emission_factor": 3.00,
        "siegert_a1": 0.42,
        "siegert_a2": 0.008,
    },
    FuelType.HYDROGEN: {
        "lhv_kj_kg": 119930.0,
        "stoich_air_fuel_ratio": 34.3,
        "co2_emission_factor": 0.0,
        "siegert_a1": 0.25,
        "siegert_a2": 0.006,
    },
    FuelType.REFINERY_GAS: {
        "lhv_kj_kg": 45000.0,
        "stoich_air_fuel_ratio": 16.0,
        "co2_emission_factor": 2.85,
        "siegert_a1": 0.40,
        "siegert_a2": 0.008,
    },
}


@dataclass
class EfficiencyInputs:
    """
    Input data for thermal efficiency calculation.

    Attributes:
        fuel_mass_flow_kg_s: Fuel mass flow rate in kg/s
        fuel_lhv_kj_kg: Fuel Lower Heating Value in kJ/kg (optional if fuel_type provided)
        useful_heat_output_kw: Useful heat absorbed by process in kW
        fuel_type: Optional fuel type for automatic LHV lookup
    """
    fuel_mass_flow_kg_s: float
    useful_heat_output_kw: float
    fuel_lhv_kj_kg: Optional[float] = None
    fuel_type: Optional[FuelType] = None

    def get_lhv(self) -> float:
        """Get LHV from explicit value or fuel type lookup."""
        if self.fuel_lhv_kj_kg is not None:
            return self.fuel_lhv_kj_kg
        if self.fuel_type is not None:
            return FUEL_PROPERTIES[self.fuel_type]["lhv_kj_kg"]
        raise ValueError("Either fuel_lhv_kj_kg or fuel_type must be provided")


@dataclass
class EfficiencyOutputs:
    """
    Output from thermal efficiency calculation.

    Attributes:
        fuel_input_kw: Total fuel energy input in kW
        specific_fuel_consumption: SFC in kJ/kJ (dimensionless)
        thermal_efficiency_pct: Thermal efficiency as percentage
        heat_loss_kw: Total heat losses in kW
    """
    fuel_input_kw: float
    specific_fuel_consumption: float
    thermal_efficiency_pct: float
    heat_loss_kw: float


@dataclass
class ExcessAirInputs:
    """
    Input data for excess air calculation from O2 readings.

    Attributes:
        o2_percent_dry: Oxygen concentration in dry flue gas (%)
        fuel_type: Fuel type for stoichiometric calculation
        co_ppm: Optional CO concentration for incomplete combustion
    """
    o2_percent_dry: float
    fuel_type: FuelType = FuelType.NATURAL_GAS
    co_ppm: Optional[float] = None


@dataclass
class ExcessAirOutputs:
    """
    Output from excess air calculation.

    Attributes:
        excess_air_pct: Excess air as percentage
        air_fuel_ratio_actual: Actual air-to-fuel ratio
        air_fuel_ratio_stoich: Stoichiometric air-to-fuel ratio
        lambda_value: Lambda (air ratio) = actual/stoich
        combustion_quality: Quality indicator (LEAN/OPTIMAL/RICH)
    """
    excess_air_pct: float
    air_fuel_ratio_actual: float
    air_fuel_ratio_stoich: float
    lambda_value: float
    combustion_quality: str


@dataclass
class StackLossInputs:
    """
    Input data for stack loss calculation (Siegert formula).

    Attributes:
        flue_gas_temp_c: Flue gas temperature at stack in Celsius
        ambient_temp_c: Ambient air temperature in Celsius
        o2_percent_dry: Oxygen concentration in dry flue gas (%)
        fuel_type: Fuel type for Siegert coefficients
        co_ppm: Optional CO concentration for incomplete combustion loss
    """
    flue_gas_temp_c: float
    ambient_temp_c: float
    o2_percent_dry: float
    fuel_type: FuelType = FuelType.NATURAL_GAS
    co_ppm: Optional[float] = None


@dataclass
class StackLossOutputs:
    """
    Output from stack loss calculation.

    Attributes:
        sensible_loss_pct: Sensible heat loss percentage
        latent_loss_pct: Latent heat loss percentage (if applicable)
        incomplete_combustion_loss_pct: CO-related loss percentage
        total_stack_loss_pct: Total stack loss percentage
        stack_efficiency_pct: Stack efficiency (100 - total_loss)
    """
    sensible_loss_pct: float
    latent_loss_pct: float
    incomplete_combustion_loss_pct: float
    total_stack_loss_pct: float
    stack_efficiency_pct: float


class EfficiencyCalculator(DeterministicCalculator[EfficiencyInputs, EfficiencyOutputs]):
    """
    Deterministic calculator for furnace thermal efficiency.

    Calculates fuel input power, specific fuel consumption, and thermal
    efficiency using deterministic formulas. No LLM or probabilistic
    methods are used in calculations.

    Formulas:
        - Fuel Input (kW) = mass_flow (kg/s) * LHV (kJ/kg)
        - SFC = Fuel Input / Useful Output (dimensionless)
        - Thermal Efficiency (%) = (Useful Output / Fuel Input) * 100

    Example:
        >>> calc = EfficiencyCalculator(agent_id="GL-007")
        >>> inputs = EfficiencyInputs(
        ...     fuel_mass_flow_kg_s=0.5,
        ...     fuel_lhv_kj_kg=42000,
        ...     useful_heat_output_kw=8000
        ... )
        >>> result = calc.calculate(inputs)
        >>> print(f"Efficiency: {result.result.thermal_efficiency_pct:.2f}%")
        Efficiency: 38.10%
    """

    NAME = "FurnaceEfficiencyCalculator"
    VERSION = "1.0.0"

    # Validation thresholds
    MIN_MASS_FLOW_KG_S = 0.0001  # 0.1 g/s minimum
    MAX_MASS_FLOW_KG_S = 1000.0  # 1000 kg/s maximum
    MIN_LHV_KJ_KG = 1000.0  # 1 MJ/kg minimum
    MAX_LHV_KJ_KG = 150000.0  # 150 MJ/kg maximum (hydrogen)
    MIN_EFFICIENCY_PCT = 1.0  # Practical minimum
    MAX_EFFICIENCY_PCT = 99.0  # Thermodynamic limit

    def _validate_inputs(self, inputs: EfficiencyInputs) -> List[str]:
        """
        Validate efficiency calculation inputs.

        Args:
            inputs: Efficiency calculation inputs

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate mass flow rate
        if inputs.fuel_mass_flow_kg_s <= 0:
            errors.append("fuel_mass_flow_kg_s must be positive")
        elif inputs.fuel_mass_flow_kg_s < self.MIN_MASS_FLOW_KG_S:
            errors.append(f"fuel_mass_flow_kg_s below minimum ({self.MIN_MASS_FLOW_KG_S} kg/s)")
        elif inputs.fuel_mass_flow_kg_s > self.MAX_MASS_FLOW_KG_S:
            errors.append(f"fuel_mass_flow_kg_s exceeds maximum ({self.MAX_MASS_FLOW_KG_S} kg/s)")

        # Validate useful heat output
        if inputs.useful_heat_output_kw <= 0:
            errors.append("useful_heat_output_kw must be positive")

        # Validate LHV availability and range
        try:
            lhv = inputs.get_lhv()
            if lhv < self.MIN_LHV_KJ_KG:
                errors.append(f"LHV below minimum ({self.MIN_LHV_KJ_KG} kJ/kg)")
            elif lhv > self.MAX_LHV_KJ_KG:
                errors.append(f"LHV exceeds maximum ({self.MAX_LHV_KJ_KG} kJ/kg)")
        except ValueError as e:
            errors.append(str(e))

        # Check for thermodynamically impossible efficiency
        if not errors:
            lhv = inputs.get_lhv()
            fuel_input = inputs.fuel_mass_flow_kg_s * lhv
            efficiency = (inputs.useful_heat_output_kw / fuel_input) * 100
            if efficiency > 100:
                errors.append(
                    f"Calculated efficiency ({efficiency:.1f}%) exceeds 100% - "
                    "check input values for errors"
                )

        return errors

    def _calculate(self, inputs: EfficiencyInputs, **kwargs: Any) -> EfficiencyOutputs:
        """
        Calculate thermal efficiency metrics.

        This is a DETERMINISTIC calculation:
        - Fuel Input (kW) = mass_flow * LHV
        - SFC = fuel_input / useful_output
        - Efficiency = (useful_output / fuel_input) * 100

        Args:
            inputs: Validated efficiency inputs

        Returns:
            EfficiencyOutputs with all calculated metrics
        """
        # Get LHV (from explicit or fuel type lookup)
        lhv = inputs.get_lhv()

        # Calculate fuel input power
        # Q_fuel (kW) = m_dot (kg/s) * LHV (kJ/kg)
        fuel_input_kw = inputs.fuel_mass_flow_kg_s * lhv

        # Calculate Specific Fuel Consumption
        # SFC = Q_fuel / Q_useful (dimensionless, kJ/kJ)
        specific_fuel_consumption = fuel_input_kw / inputs.useful_heat_output_kw

        # Calculate thermal efficiency
        # eta_th (%) = (Q_useful / Q_fuel) * 100
        thermal_efficiency_pct = (inputs.useful_heat_output_kw / fuel_input_kw) * 100

        # Calculate heat losses
        # Q_loss = Q_fuel - Q_useful
        heat_loss_kw = fuel_input_kw - inputs.useful_heat_output_kw

        return EfficiencyOutputs(
            fuel_input_kw=round(fuel_input_kw, 3),
            specific_fuel_consumption=round(specific_fuel_consumption, 6),
            thermal_efficiency_pct=round(thermal_efficiency_pct, 4),
            heat_loss_kw=round(heat_loss_kw, 3),
        )

    def calculate_excess_air(
        self,
        inputs: ExcessAirInputs,
    ) -> CalculationResult[ExcessAirOutputs]:
        """
        Calculate excess air percentage from O2 readings.

        Uses the standard O2-based excess air formula:
            excess_air_pct = O2 / (21 - O2) * 100

        Where O2 is the measured oxygen percentage in dry flue gas.

        Args:
            inputs: Excess air calculation inputs

        Returns:
            CalculationResult with ExcessAirOutputs
        """
        # Validate inputs
        errors = self._validate_excess_air_inputs(inputs)
        if errors:
            return CalculationResult(
                result=None,
                computation_hash="",
                inputs_hash=self._compute_hash(inputs),
                calculator_name=self.NAME,
                calculator_version=self.VERSION,
                is_valid=False,
                warnings=errors,
            )

        # Get stoichiometric air-fuel ratio
        stoich_afr = FUEL_PROPERTIES[inputs.fuel_type]["stoich_air_fuel_ratio"]

        # Calculate excess air percentage
        # Formula: EA% = O2 / (21 - O2) * 100
        o2 = inputs.o2_percent_dry
        excess_air_pct = (o2 / (21.0 - o2)) * 100.0

        # Calculate lambda (air ratio)
        # lambda = 1 + (EA% / 100)
        lambda_value = 1.0 + (excess_air_pct / 100.0)

        # Calculate actual air-fuel ratio
        air_fuel_ratio_actual = stoich_afr * lambda_value

        # Determine combustion quality
        if excess_air_pct < 5:
            combustion_quality = "RICH"
        elif excess_air_pct <= 25:
            combustion_quality = "OPTIMAL"
        else:
            combustion_quality = "LEAN"

        result = ExcessAirOutputs(
            excess_air_pct=round(excess_air_pct, 2),
            air_fuel_ratio_actual=round(air_fuel_ratio_actual, 3),
            air_fuel_ratio_stoich=stoich_afr,
            lambda_value=round(lambda_value, 4),
            combustion_quality=combustion_quality,
        )

        # Compute provenance
        inputs_hash = self._compute_hash(inputs)
        outputs_hash = self._compute_hash(result)
        computation_hash = self._compute_combined_hash(inputs_hash, outputs_hash, {})

        return CalculationResult(
            result=result,
            computation_hash=computation_hash,
            inputs_hash=inputs_hash,
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            is_valid=True,
        )

    def _validate_excess_air_inputs(self, inputs: ExcessAirInputs) -> List[str]:
        """Validate excess air calculation inputs."""
        errors = []

        if inputs.o2_percent_dry < 0:
            errors.append("O2 percentage cannot be negative")
        elif inputs.o2_percent_dry >= 21.0:
            errors.append("O2 percentage must be less than 21% (ambient air)")
        elif inputs.o2_percent_dry > 18.0:
            errors.append("O2 > 18% indicates extremely high excess air or measurement error")

        if inputs.co_ppm is not None and inputs.co_ppm < 0:
            errors.append("CO concentration cannot be negative")

        return errors

    def calculate_stack_loss(
        self,
        inputs: StackLossInputs,
    ) -> CalculationResult[StackLossOutputs]:
        """
        Calculate stack loss using the Siegert formula.

        The Siegert formula estimates sensible heat loss:
            Q_stack (%) = A1 * (T_flue - T_amb) / CO2%

        Or when O2 is measured instead of CO2:
            Q_stack (%) = (A1 / (21 - O2)) * (T_flue - T_amb) + A2 * O2

        Args:
            inputs: Stack loss calculation inputs

        Returns:
            CalculationResult with StackLossOutputs
        """
        # Validate inputs
        errors = self._validate_stack_loss_inputs(inputs)
        if errors:
            return CalculationResult(
                result=None,
                computation_hash="",
                inputs_hash=self._compute_hash(inputs),
                calculator_name=self.NAME,
                calculator_version=self.VERSION,
                is_valid=False,
                warnings=errors,
            )

        # Get Siegert coefficients
        props = FUEL_PROPERTIES[inputs.fuel_type]
        a1 = props["siegert_a1"]
        a2 = props["siegert_a2"]

        # Temperature difference
        delta_t = inputs.flue_gas_temp_c - inputs.ambient_temp_c

        # Calculate sensible heat loss using modified Siegert formula
        # Q_sensible (%) = A1 * delta_T / (21 - O2) + A2 * delta_T
        o2 = inputs.o2_percent_dry
        sensible_loss_pct = (a1 * delta_t / (21.0 - o2)) + (a2 * delta_t)

        # Latent heat loss (simplified - assumes all H2O exits as vapor)
        # For natural gas, typically 6-8% latent loss
        # This is a simplification; actual calculation requires fuel H/C ratio
        latent_loss_pct = 0.0  # Included in LHV basis calculation

        # Incomplete combustion loss from CO
        # Q_CO (%) = CO_ppm * 0.0001 * 126.7 / LHV_factor
        incomplete_combustion_loss_pct = 0.0
        if inputs.co_ppm is not None and inputs.co_ppm > 0:
            # Approximate loss: each 100 ppm CO ~ 0.1% loss
            incomplete_combustion_loss_pct = inputs.co_ppm * 0.001

        # Total stack loss
        total_stack_loss_pct = (
            sensible_loss_pct +
            latent_loss_pct +
            incomplete_combustion_loss_pct
        )

        # Stack efficiency
        stack_efficiency_pct = 100.0 - total_stack_loss_pct

        result = StackLossOutputs(
            sensible_loss_pct=round(sensible_loss_pct, 3),
            latent_loss_pct=round(latent_loss_pct, 3),
            incomplete_combustion_loss_pct=round(incomplete_combustion_loss_pct, 3),
            total_stack_loss_pct=round(total_stack_loss_pct, 3),
            stack_efficiency_pct=round(stack_efficiency_pct, 3),
        )

        # Compute provenance
        inputs_hash = self._compute_hash(inputs)
        outputs_hash = self._compute_hash(result)
        computation_hash = self._compute_combined_hash(inputs_hash, outputs_hash, {})

        return CalculationResult(
            result=result,
            computation_hash=computation_hash,
            inputs_hash=inputs_hash,
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            is_valid=True,
        )

    def _validate_stack_loss_inputs(self, inputs: StackLossInputs) -> List[str]:
        """Validate stack loss calculation inputs."""
        errors = []

        # Temperature validation
        if inputs.flue_gas_temp_c <= inputs.ambient_temp_c:
            errors.append("Flue gas temperature must be higher than ambient")

        if inputs.flue_gas_temp_c > 1000:
            errors.append("Flue gas temperature exceeds practical maximum (1000C)")

        if inputs.ambient_temp_c < -50 or inputs.ambient_temp_c > 60:
            errors.append("Ambient temperature out of practical range (-50 to 60C)")

        # O2 validation
        if inputs.o2_percent_dry < 0:
            errors.append("O2 percentage cannot be negative")
        elif inputs.o2_percent_dry >= 21.0:
            errors.append("O2 percentage must be less than 21%")

        # CO validation
        if inputs.co_ppm is not None:
            if inputs.co_ppm < 0:
                errors.append("CO concentration cannot be negative")
            elif inputs.co_ppm > 10000:
                errors.append("CO concentration exceeds practical maximum (10000 ppm)")

        return errors
