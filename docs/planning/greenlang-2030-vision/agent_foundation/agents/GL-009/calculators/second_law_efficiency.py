"""Second Law (Exergy/Availability) Efficiency Calculator.

This module implements thermal efficiency calculations based on the
Second Law of Thermodynamics (Exergy Analysis).

The Second Law efficiency measures how well a system converts available
work potential (exergy) rather than just energy (First Law).

Formula:
    eta_II = Ex_useful / Ex_input x 100%

Where:
    eta_II = Second Law (exergy) efficiency (%)
    Ex_useful = Useful exergy output (kW)
    Ex_input = Total exergy input (kW)

Exergy of a stream:
    Ex = m * [(h - h0) - T0 * (s - s0)]

Where:
    m = mass flow rate (kg/s)
    h = specific enthalpy (kJ/kg)
    h0 = specific enthalpy at reference state (kJ/kg)
    T0 = reference temperature (K)
    s = specific entropy (kJ/kg-K)
    s0 = specific entropy at reference state (kJ/kg-K)

Standards:
    - ASME PTC 46: Overall Plant Performance
    - ISO 50001: Energy Management Systems
    - Kotas: The Exergy Method of Thermal Plant Analysis

Author: GL-009 THERMALIQ Agent
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import hashlib
import json
from datetime import datetime
import math


class StreamType(Enum):
    """Types of exergy streams."""
    FUEL = "fuel"
    STEAM = "steam"
    HOT_WATER = "hot_water"
    FLUE_GAS = "flue_gas"
    COMBUSTION_AIR = "combustion_air"
    CONDENSATE = "condensate"
    ELECTRICITY = "electricity"
    MECHANICAL_WORK = "mechanical_work"
    HEAT_TRANSFER = "heat_transfer"
    CHEMICAL = "chemical"
    OTHER = "other"


class IrreversibilityType(Enum):
    """Types of irreversibilities (exergy destruction)."""
    COMBUSTION = "combustion"
    HEAT_TRANSFER = "heat_transfer"
    MIXING = "mixing"
    THROTTLING = "throttling"
    FRICTION = "friction"
    CHEMICAL_REACTION = "chemical_reaction"
    RADIATION = "radiation"
    UNACCOUNTED = "unaccounted"


@dataclass(frozen=True)
class ReferenceEnvironment:
    """Reference (dead state) environment conditions.

    The reference environment defines the dead state where a system
    has zero exergy. All exergy calculations are relative to this state.

    Attributes:
        temperature_k: Reference temperature (K), default 298.15 K (25C)
        pressure_kpa: Reference pressure (kPa), default 101.325 kPa
        relative_humidity_percent: Reference humidity (%)
        composition: Molar composition of reference atmosphere
    """
    temperature_k: float = 298.15  # 25 C
    pressure_kpa: float = 101.325  # 1 atm
    relative_humidity_percent: float = 60.0
    composition: Dict[str, float] = field(default_factory=lambda: {
        "N2": 0.7567,
        "O2": 0.2035,
        "H2O": 0.0303,
        "CO2": 0.0003,
        "Ar": 0.0092
    })

    @property
    def temperature_c(self) -> float:
        """Temperature in Celsius."""
        return self.temperature_k - 273.15


@dataclass(frozen=True)
class ExergyStream:
    """Represents an exergy stream in the system.

    Attributes:
        stream_type: Type of the stream
        stream_name: Descriptive name
        mass_flow_kg_s: Mass flow rate (kg/s)
        temperature_k: Stream temperature (K)
        pressure_kpa: Stream pressure (kPa)
        specific_enthalpy_kj_kg: Specific enthalpy (kJ/kg)
        specific_entropy_kj_kg_k: Specific entropy (kJ/kg-K)
        chemical_exergy_kj_kg: Chemical exergy component (kJ/kg)
        is_input: True if input stream, False if output
    """
    stream_type: StreamType
    stream_name: str
    mass_flow_kg_s: float
    temperature_k: float
    pressure_kpa: float
    specific_enthalpy_kj_kg: float
    specific_entropy_kj_kg_k: float
    chemical_exergy_kj_kg: float = 0.0
    is_input: bool = True

    def __post_init__(self) -> None:
        """Validate stream values."""
        if self.mass_flow_kg_s < 0:
            raise ValueError(f"Mass flow cannot be negative: {self.mass_flow_kg_s}")
        if self.temperature_k <= 0:
            raise ValueError(f"Temperature must be positive: {self.temperature_k}")
        if self.pressure_kpa <= 0:
            raise ValueError(f"Pressure must be positive: {self.pressure_kpa}")


@dataclass(frozen=True)
class IrreversibilityBreakdown:
    """Breakdown of exergy destruction (irreversibilities).

    Attributes:
        irreversibility_type: Type of irreversibility
        description: Description of the loss mechanism
        exergy_destruction_kw: Rate of exergy destruction (kW)
        percentage_of_input: As percentage of input exergy
    """
    irreversibility_type: IrreversibilityType
    description: str
    exergy_destruction_kw: float
    percentage_of_input: float


@dataclass
class CalculationStep:
    """Records a single calculation step for audit trail."""
    step_number: int
    description: str
    operation: str
    inputs: Dict[str, float]
    output_value: float
    output_name: str
    formula: Optional[str] = None


@dataclass
class SecondLawResult:
    """Complete result of Second Law efficiency calculation.

    Attributes:
        exergy_efficiency_percent: Second Law (exergy) efficiency (%)
        first_law_efficiency_percent: Corresponding First Law efficiency (%)
        total_exergy_input_kw: Total exergy input (kW)
        total_exergy_output_kw: Total exergy output (kW)
        total_exergy_destruction_kw: Total irreversibilities (kW)
        total_exergy_loss_kw: Exergy lost to environment (kW)
        exergy_balance_error: Relative error in exergy balance
        irreversibility_breakdown: Breakdown by mechanism
        stream_exergies: Exergy of each stream
        reference_environment: Reference state used
        calculation_steps: Ordered list of calculation steps
        provenance_hash: SHA-256 hash of calculation inputs
        calculation_timestamp: When calculation was performed
        calculator_version: Version of calculator used
        warnings: Any warnings generated
    """
    exergy_efficiency_percent: float
    first_law_efficiency_percent: float
    total_exergy_input_kw: float
    total_exergy_output_kw: float
    total_exergy_destruction_kw: float
    total_exergy_loss_kw: float
    exergy_balance_error: float
    irreversibility_breakdown: List[IrreversibilityBreakdown]
    stream_exergies: Dict[str, float]
    reference_environment: ReferenceEnvironment
    calculation_steps: List[CalculationStep]
    provenance_hash: str
    calculation_timestamp: str
    calculator_version: str = "1.0.0"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            "exergy_efficiency_percent": self.exergy_efficiency_percent,
            "first_law_efficiency_percent": self.first_law_efficiency_percent,
            "total_exergy_input_kw": self.total_exergy_input_kw,
            "total_exergy_output_kw": self.total_exergy_output_kw,
            "total_exergy_destruction_kw": self.total_exergy_destruction_kw,
            "total_exergy_loss_kw": self.total_exergy_loss_kw,
            "exergy_balance_error": self.exergy_balance_error,
            "irreversibility_breakdown": [
                {
                    "type": irr.irreversibility_type.value,
                    "description": irr.description,
                    "exergy_destruction_kw": irr.exergy_destruction_kw,
                    "percentage_of_input": irr.percentage_of_input
                }
                for irr in self.irreversibility_breakdown
            ],
            "stream_exergies": self.stream_exergies,
            "reference_environment": {
                "temperature_k": self.reference_environment.temperature_k,
                "pressure_kpa": self.reference_environment.pressure_kpa,
                "relative_humidity_percent": self.reference_environment.relative_humidity_percent
            },
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp,
            "calculator_version": self.calculator_version,
            "warnings": self.warnings
        }


class SecondLawEfficiencyCalculator:
    """Second Law (Exergy/Availability) Efficiency Calculator.

    This calculator implements exergy analysis based on the Second Law
    of Thermodynamics. It provides a more meaningful measure of
    thermodynamic performance than First Law efficiency alone.

    The Second Law efficiency is defined as:
        eta_II = Ex_useful / Ex_input x 100%

    Key Features:
        - Exergy calculation for multiple stream types
        - Irreversibility (exergy destruction) breakdown
        - Reference environment customization
        - Complete calculation audit trail
        - SHA-256 provenance hashing

    Example:
        >>> calculator = SecondLawEfficiencyCalculator()
        >>> result = calculator.calculate(
        ...     input_streams=[fuel_stream, air_stream],
        ...     output_streams=[steam_stream, flue_gas_stream],
        ...     irreversibilities={"combustion": 500.0, "heat_transfer": 200.0}
        ... )
    """

    VERSION: str = "1.0.0"
    PRECISION: int = 4

    # Reference state specific properties (for water at 25C, 101.325 kPa)
    H0_WATER_KJ_KG: float = 104.89  # Specific enthalpy at reference
    S0_WATER_KJ_KG_K: float = 0.3674  # Specific entropy at reference

    def __init__(
        self,
        reference_environment: Optional[ReferenceEnvironment] = None,
        precision: int = 4
    ) -> None:
        """Initialize the Second Law Efficiency Calculator.

        Args:
            reference_environment: Reference (dead) state conditions
            precision: Number of decimal places for rounding
        """
        self.reference = reference_environment or ReferenceEnvironment()
        self.precision = precision
        self._calculation_steps: List[CalculationStep] = []
        self._step_counter: int = 0
        self._warnings: List[str] = []

    def calculate(
        self,
        input_streams: List[ExergyStream],
        output_streams: List[ExergyStream],
        irreversibilities: Optional[Dict[str, float]] = None
    ) -> SecondLawResult:
        """Calculate Second Law (exergy) efficiency.

        Args:
            input_streams: List of input exergy streams
            output_streams: List of output exergy streams
            irreversibilities: Optional dict of known irreversibilities (kW)

        Returns:
            SecondLawResult containing efficiency and breakdown
        """
        # Reset calculation state
        self._reset_calculation_state()

        # Generate provenance hash
        provenance_hash = self._generate_provenance_hash(
            input_streams, output_streams, irreversibilities
        )

        # Step 1: Calculate exergy of each input stream
        stream_exergies: Dict[str, float] = {}
        total_input_exergy = 0.0

        for stream in input_streams:
            exergy = self._calculate_stream_exergy(stream)
            stream_exergies[stream.stream_name] = exergy
            total_input_exergy += exergy

        self._add_calculation_step(
            description="Calculate total input exergy",
            operation="sum",
            inputs={"stream_count": len(input_streams)},
            output_value=total_input_exergy,
            output_name="total_input_exergy_kw",
            formula="Sum(Ex_input_i)"
        )

        # Step 2: Calculate exergy of each output stream
        total_output_exergy = 0.0
        useful_output_exergy = 0.0
        exergy_loss = 0.0

        for stream in output_streams:
            exergy = self._calculate_stream_exergy(stream)
            stream_exergies[stream.stream_name] = exergy
            total_output_exergy += exergy

            # Classify as useful output or loss
            if stream.stream_type in [StreamType.STEAM, StreamType.HOT_WATER,
                                      StreamType.ELECTRICITY, StreamType.MECHANICAL_WORK]:
                useful_output_exergy += exergy
            else:
                exergy_loss += exergy

        self._add_calculation_step(
            description="Calculate total output exergy",
            operation="sum",
            inputs={"stream_count": len(output_streams)},
            output_value=total_output_exergy,
            output_name="total_output_exergy_kw",
            formula="Sum(Ex_output_i)"
        )

        # Step 3: Calculate or validate exergy destruction
        if irreversibilities:
            total_destruction = sum(irreversibilities.values())
        else:
            # Calculate from exergy balance
            total_destruction = total_input_exergy - total_output_exergy

        self._add_calculation_step(
            description="Calculate total exergy destruction",
            operation="subtract",
            inputs={
                "total_input_exergy_kw": total_input_exergy,
                "total_output_exergy_kw": total_output_exergy
            },
            output_value=total_destruction,
            output_name="total_exergy_destruction_kw",
            formula="Ex_destruction = Ex_input - Ex_output"
        )

        # Step 4: Validate exergy balance
        calculated_balance = total_input_exergy - useful_output_exergy - exergy_loss - total_destruction
        balance_error = abs(calculated_balance) / total_input_exergy if total_input_exergy > 0 else 0

        # Step 5: Calculate Second Law efficiency
        if total_input_exergy > 0:
            exergy_efficiency = (useful_output_exergy / total_input_exergy) * 100
        else:
            exergy_efficiency = 0.0

        self._add_calculation_step(
            description="Calculate Second Law (exergy) efficiency",
            operation="divide_multiply",
            inputs={
                "useful_exergy_output_kw": useful_output_exergy,
                "total_exergy_input_kw": total_input_exergy
            },
            output_value=exergy_efficiency,
            output_name="exergy_efficiency_percent",
            formula="eta_II = (Ex_useful / Ex_input) x 100%"
        )

        # Step 6: Calculate First Law efficiency for comparison
        total_energy_input = sum(
            s.mass_flow_kg_s * s.specific_enthalpy_kj_kg for s in input_streams
        )
        useful_energy_output = sum(
            s.mass_flow_kg_s * s.specific_enthalpy_kj_kg
            for s in output_streams
            if s.stream_type in [StreamType.STEAM, StreamType.HOT_WATER,
                                 StreamType.ELECTRICITY, StreamType.MECHANICAL_WORK]
        )
        if total_energy_input > 0:
            first_law_efficiency = (useful_energy_output / total_energy_input) * 100
        else:
            first_law_efficiency = 0.0

        # Step 7: Build irreversibility breakdown
        irreversibility_list = self._build_irreversibility_breakdown(
            irreversibilities, total_destruction, total_input_exergy
        )

        # Generate timestamp
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Add warnings
        if exergy_efficiency > 100:
            self._warnings.append(
                f"Exergy efficiency > 100% ({exergy_efficiency:.2f}%) indicates error"
            )

        return SecondLawResult(
            exergy_efficiency_percent=self._round_value(exergy_efficiency),
            first_law_efficiency_percent=self._round_value(first_law_efficiency),
            total_exergy_input_kw=self._round_value(total_input_exergy),
            total_exergy_output_kw=self._round_value(useful_output_exergy),
            total_exergy_destruction_kw=self._round_value(total_destruction),
            total_exergy_loss_kw=self._round_value(exergy_loss),
            exergy_balance_error=self._round_value(balance_error, 6),
            irreversibility_breakdown=irreversibility_list,
            stream_exergies={k: self._round_value(v) for k, v in stream_exergies.items()},
            reference_environment=self.reference,
            calculation_steps=self._calculation_steps.copy(),
            provenance_hash=provenance_hash,
            calculation_timestamp=timestamp,
            calculator_version=self.VERSION,
            warnings=self._warnings.copy()
        )

    def calculate_stream_exergy(
        self,
        mass_flow_kg_s: float,
        temperature_k: float,
        pressure_kpa: float,
        specific_enthalpy_kj_kg: float,
        specific_entropy_kj_kg_k: float,
        chemical_exergy_kj_kg: float = 0.0
    ) -> float:
        """Calculate exergy of a single stream.

        Exergy = Physical Exergy + Chemical Exergy

        Physical Exergy:
            Ex_ph = m * [(h - h0) - T0 * (s - s0)]

        Args:
            mass_flow_kg_s: Mass flow rate (kg/s)
            temperature_k: Stream temperature (K)
            pressure_kpa: Stream pressure (kPa)
            specific_enthalpy_kj_kg: Specific enthalpy (kJ/kg)
            specific_entropy_kj_kg_k: Specific entropy (kJ/kg-K)
            chemical_exergy_kj_kg: Chemical exergy (kJ/kg)

        Returns:
            Total exergy rate (kW)
        """
        # Reference state properties
        h0 = self._get_reference_enthalpy(temperature_k, pressure_kpa)
        s0 = self._get_reference_entropy(temperature_k, pressure_kpa)
        T0 = self.reference.temperature_k

        # Physical exergy: Ex_ph = m * [(h - h0) - T0 * (s - s0)]
        physical_exergy = mass_flow_kg_s * (
            (specific_enthalpy_kj_kg - h0) - T0 * (specific_entropy_kj_kg_k - s0)
        )

        # Chemical exergy
        chemical_exergy = mass_flow_kg_s * chemical_exergy_kj_kg

        # Total exergy (kW = kJ/s)
        total_exergy = physical_exergy + chemical_exergy

        return max(0.0, total_exergy)  # Exergy cannot be negative

    def calculate_fuel_exergy(
        self,
        fuel_hhv_kj_kg: float,
        mass_flow_kg_s: float,
        fuel_type: str = "natural_gas"
    ) -> float:
        """Calculate chemical exergy of fuel.

        Uses fuel exergy-to-HHV ratio (phi) for common fuels.

        Args:
            fuel_hhv_kj_kg: Higher heating value (kJ/kg)
            mass_flow_kg_s: Fuel mass flow rate (kg/s)
            fuel_type: Type of fuel

        Returns:
            Fuel exergy rate (kW)
        """
        # Exergy-to-HHV ratios (phi) for common fuels
        phi_values = {
            "natural_gas": 1.04,
            "methane": 1.04,
            "coal": 1.06,
            "oil": 1.065,
            "diesel": 1.07,
            "gasoline": 1.065,
            "biomass": 1.15,
            "hydrogen": 0.985,
            "propane": 1.05,
        }

        phi = phi_values.get(fuel_type.lower(), 1.05)  # Default 1.05

        exergy = phi * fuel_hhv_kj_kg * mass_flow_kg_s

        self._add_calculation_step(
            description=f"Calculate fuel exergy ({fuel_type})",
            operation="multiply",
            inputs={
                "phi": phi,
                "hhv_kj_kg": fuel_hhv_kj_kg,
                "mass_flow_kg_s": mass_flow_kg_s
            },
            output_value=exergy,
            output_name="fuel_exergy_kw",
            formula="Ex_fuel = phi x HHV x m_dot"
        )

        return exergy

    def calculate_heat_transfer_exergy(
        self,
        heat_rate_kw: float,
        temperature_k: float
    ) -> float:
        """Calculate exergy associated with heat transfer.

        Exergy of heat transfer (Carnot factor):
            Ex_Q = Q * (1 - T0/T)

        Args:
            heat_rate_kw: Heat transfer rate (kW)
            temperature_k: Temperature of heat source/sink (K)

        Returns:
            Exergy of heat transfer (kW)
        """
        T0 = self.reference.temperature_k

        if temperature_k <= T0:
            # Below reference temperature - cooling exergy
            carnot_factor = (T0 / temperature_k) - 1
        else:
            # Above reference temperature - heating exergy
            carnot_factor = 1 - (T0 / temperature_k)

        exergy = heat_rate_kw * carnot_factor

        self._add_calculation_step(
            description="Calculate heat transfer exergy",
            operation="multiply",
            inputs={
                "heat_rate_kw": heat_rate_kw,
                "temperature_k": temperature_k,
                "T0_k": T0,
                "carnot_factor": carnot_factor
            },
            output_value=exergy,
            output_name="heat_exergy_kw",
            formula="Ex_Q = Q x (1 - T0/T)"
        )

        return abs(exergy)

    def calculate_combustion_irreversibility(
        self,
        fuel_exergy_kw: float,
        products_exergy_kw: float,
        adiabatic_flame_temp_k: float
    ) -> float:
        """Calculate irreversibility due to combustion.

        Combustion is highly irreversible due to chemical reaction
        and temperature increase.

        Args:
            fuel_exergy_kw: Exergy of fuel input (kW)
            products_exergy_kw: Exergy of combustion products (kW)
            adiabatic_flame_temp_k: Adiabatic flame temperature (K)

        Returns:
            Combustion irreversibility (kW)
        """
        # Combustion irreversibility = Fuel exergy - Products exergy
        irreversibility = fuel_exergy_kw - products_exergy_kw

        self._add_calculation_step(
            description="Calculate combustion irreversibility",
            operation="subtract",
            inputs={
                "fuel_exergy_kw": fuel_exergy_kw,
                "products_exergy_kw": products_exergy_kw
            },
            output_value=irreversibility,
            output_name="combustion_irreversibility_kw",
            formula="I_comb = Ex_fuel - Ex_products"
        )

        return max(0.0, irreversibility)

    def calculate_heat_transfer_irreversibility(
        self,
        heat_rate_kw: float,
        hot_temp_k: float,
        cold_temp_k: float
    ) -> float:
        """Calculate irreversibility due to heat transfer.

        Heat transfer across a finite temperature difference
        is irreversible.

        Args:
            heat_rate_kw: Heat transfer rate (kW)
            hot_temp_k: Hot side temperature (K)
            cold_temp_k: Cold side temperature (K)

        Returns:
            Heat transfer irreversibility (kW)
        """
        T0 = self.reference.temperature_k

        # Irreversibility = T0 * (Q/T_cold - Q/T_hot)
        irreversibility = T0 * heat_rate_kw * ((1 / cold_temp_k) - (1 / hot_temp_k))

        self._add_calculation_step(
            description="Calculate heat transfer irreversibility",
            operation="heat_transfer_entropy",
            inputs={
                "heat_rate_kw": heat_rate_kw,
                "hot_temp_k": hot_temp_k,
                "cold_temp_k": cold_temp_k,
                "T0_k": T0
            },
            output_value=irreversibility,
            output_name="ht_irreversibility_kw",
            formula="I_HT = T0 x Q x (1/T_cold - 1/T_hot)"
        )

        return max(0.0, irreversibility)

    def _calculate_stream_exergy(self, stream: ExergyStream) -> float:
        """Calculate exergy of an ExergyStream object."""
        return self.calculate_stream_exergy(
            mass_flow_kg_s=stream.mass_flow_kg_s,
            temperature_k=stream.temperature_k,
            pressure_kpa=stream.pressure_kpa,
            specific_enthalpy_kj_kg=stream.specific_enthalpy_kj_kg,
            specific_entropy_kj_kg_k=stream.specific_entropy_kj_kg_k,
            chemical_exergy_kj_kg=stream.chemical_exergy_kj_kg
        )

    def _get_reference_enthalpy(self, temperature_k: float, pressure_kpa: float) -> float:
        """Get reference state specific enthalpy.

        For steam/water, this uses IAPWS-IF97 at reference conditions.
        Simplified implementation using constant value.
        """
        return self.H0_WATER_KJ_KG

    def _get_reference_entropy(self, temperature_k: float, pressure_kpa: float) -> float:
        """Get reference state specific entropy.

        For steam/water, this uses IAPWS-IF97 at reference conditions.
        Simplified implementation using constant value.
        """
        return self.S0_WATER_KJ_KG_K

    def _build_irreversibility_breakdown(
        self,
        irreversibilities: Optional[Dict[str, float]],
        total_destruction: float,
        total_input: float
    ) -> List[IrreversibilityBreakdown]:
        """Build detailed irreversibility breakdown."""
        breakdown = []

        if irreversibilities:
            for irr_type, value in irreversibilities.items():
                try:
                    irr_enum = IrreversibilityType(irr_type.lower())
                except ValueError:
                    irr_enum = IrreversibilityType.UNACCOUNTED

                pct = (value / total_input * 100) if total_input > 0 else 0

                breakdown.append(IrreversibilityBreakdown(
                    irreversibility_type=irr_enum,
                    description=f"{irr_type} losses",
                    exergy_destruction_kw=self._round_value(value),
                    percentage_of_input=self._round_value(pct)
                ))
        else:
            # Single unclassified destruction
            pct = (total_destruction / total_input * 100) if total_input > 0 else 0
            breakdown.append(IrreversibilityBreakdown(
                irreversibility_type=IrreversibilityType.UNACCOUNTED,
                description="Total exergy destruction (unclassified)",
                exergy_destruction_kw=self._round_value(total_destruction),
                percentage_of_input=self._round_value(pct)
            ))

        return breakdown

    def _reset_calculation_state(self) -> None:
        """Reset calculation state for new calculation."""
        self._calculation_steps = []
        self._step_counter = 0
        self._warnings = []

    def _add_calculation_step(
        self,
        description: str,
        operation: str,
        inputs: Dict[str, float],
        output_value: float,
        output_name: str,
        formula: Optional[str] = None
    ) -> None:
        """Record a calculation step for audit trail."""
        self._step_counter += 1
        step = CalculationStep(
            step_number=self._step_counter,
            description=description,
            operation=operation,
            inputs=inputs,
            output_value=output_value,
            output_name=output_name,
            formula=formula
        )
        self._calculation_steps.append(step)

    def _generate_provenance_hash(
        self,
        input_streams: List[ExergyStream],
        output_streams: List[ExergyStream],
        irreversibilities: Optional[Dict[str, float]]
    ) -> str:
        """Generate SHA-256 hash for calculation provenance."""
        provenance_data = {
            "calculator": "SecondLawEfficiencyCalculator",
            "version": self.VERSION,
            "reference_T0": self.reference.temperature_k,
            "reference_P0": self.reference.pressure_kpa,
            "input_streams": [
                {
                    "name": s.stream_name,
                    "mass_flow": s.mass_flow_kg_s,
                    "temperature": s.temperature_k,
                    "enthalpy": s.specific_enthalpy_kj_kg,
                    "entropy": s.specific_entropy_kj_kg_k
                }
                for s in input_streams
            ],
            "output_streams": [
                {
                    "name": s.stream_name,
                    "mass_flow": s.mass_flow_kg_s,
                    "temperature": s.temperature_k,
                    "enthalpy": s.specific_enthalpy_kj_kg,
                    "entropy": s.specific_entropy_kj_kg_k
                }
                for s in output_streams
            ],
            "irreversibilities": irreversibilities
        }

        json_str = json.dumps(provenance_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _round_value(self, value: float, precision: Optional[int] = None) -> float:
        """Round value to specified precision."""
        if precision is None:
            precision = self.precision

        decimal_value = Decimal(str(value))
        quantize_str = '0.' + '0' * precision
        rounded = decimal_value.quantize(
            Decimal(quantize_str),
            rounding=ROUND_HALF_UP
        )
        return float(rounded)


def calculate_second_law_efficiency(
    input_streams: List[ExergyStream],
    output_streams: List[ExergyStream],
    irreversibilities: Optional[Dict[str, float]] = None,
    reference_temperature_k: float = 298.15,
    reference_pressure_kpa: float = 101.325
) -> SecondLawResult:
    """Convenience function for Second Law efficiency calculation.

    Args:
        input_streams: List of input exergy streams
        output_streams: List of output exergy streams
        irreversibilities: Optional known irreversibilities
        reference_temperature_k: Reference temperature (K)
        reference_pressure_kpa: Reference pressure (kPa)

    Returns:
        SecondLawResult containing efficiency and breakdown
    """
    reference = ReferenceEnvironment(
        temperature_k=reference_temperature_k,
        pressure_kpa=reference_pressure_kpa
    )
    calculator = SecondLawEfficiencyCalculator(reference_environment=reference)
    return calculator.calculate(input_streams, output_streams, irreversibilities)
