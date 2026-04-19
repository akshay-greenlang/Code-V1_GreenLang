"""
Exergy Analysis (Second Law Analysis)

Zero-Hallucination Thermodynamic Efficiency Calculations

This module implements deterministic exergy calculations for:
- Physical exergy (temperature and pressure)
- Chemical exergy
- Exergy destruction and loss
- Second law efficiency
- Exergy flow analysis

References:
    - Bejan, A. "Advanced Engineering Thermodynamics", 4th Ed. (2016)
    - Kotas, T.J. "The Exergy Method of Thermal Plant Analysis" (1985)
    - Szargut, J. "Exergy Method: Technical and Ecological Applications" (2005)
    - ASME PTC 4.1: Steam Generating Units

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
import math
import hashlib


@dataclass
class ExergyState:
    """
    Thermodynamic state for exergy calculations.

    Dead state (environment) conditions are used as reference.
    """
    temperature_k: Decimal
    pressure_kpa: Decimal
    enthalpy_kj_kg: Decimal
    entropy_kj_kgk: Decimal
    mass_flow_kg_s: Optional[Decimal] = None

    def __post_init__(self):
        """Convert inputs to Decimal if needed."""
        if not isinstance(self.temperature_k, Decimal):
            self.temperature_k = Decimal(str(self.temperature_k))
        if not isinstance(self.pressure_kpa, Decimal):
            self.pressure_kpa = Decimal(str(self.pressure_kpa))
        if not isinstance(self.enthalpy_kj_kg, Decimal):
            self.enthalpy_kj_kg = Decimal(str(self.enthalpy_kj_kg))
        if not isinstance(self.entropy_kj_kgk, Decimal):
            self.entropy_kj_kgk = Decimal(str(self.entropy_kj_kgk))
        if self.mass_flow_kg_s is not None and not isinstance(self.mass_flow_kg_s, Decimal):
            self.mass_flow_kg_s = Decimal(str(self.mass_flow_kg_s))


@dataclass
class ExergyResult:
    """
    Exergy calculation results with complete provenance.

    All values are deterministic - same inputs produce identical outputs.
    """
    # Specific exergies (per kg)
    physical_exergy_kj_kg: Decimal
    thermal_exergy_kj_kg: Decimal
    mechanical_exergy_kj_kg: Decimal

    # Flow exergies (if mass flow provided)
    exergy_flow_kw: Optional[Decimal]

    # Reference state
    dead_state_temperature_k: Decimal
    dead_state_pressure_kpa: Decimal

    # Provenance
    provenance_hash: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "physical_exergy_kj_kg": float(self.physical_exergy_kj_kg),
            "thermal_exergy_kj_kg": float(self.thermal_exergy_kj_kg),
            "mechanical_exergy_kj_kg": float(self.mechanical_exergy_kj_kg),
            "exergy_flow_kw": float(self.exergy_flow_kw) if self.exergy_flow_kw else None,
            "dead_state_temperature_k": float(self.dead_state_temperature_k),
            "dead_state_pressure_kpa": float(self.dead_state_pressure_kpa),
            "provenance_hash": self.provenance_hash
        }


@dataclass
class ExergyDestructionResult:
    """Exergy destruction analysis for a process."""
    exergy_in_kw: Decimal
    exergy_out_kw: Decimal
    exergy_destruction_kw: Decimal
    exergy_loss_kw: Decimal
    exergetic_efficiency: Decimal
    irreversibility_ratio: Decimal
    provenance_hash: str


@dataclass
class ComponentExergyAnalysis:
    """Exergy analysis for a process component."""
    component_name: str
    exergy_fuel_kw: Decimal
    exergy_product_kw: Decimal
    exergy_destruction_kw: Decimal
    exergy_loss_kw: Decimal
    exergetic_efficiency: Decimal
    improvement_potential_kw: Decimal
    provenance_hash: str


class ExergyCalculator:
    """
    Exergy analysis calculator.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations are deterministic
    - Based on established thermodynamic relations
    - Complete provenance tracking

    References:
        - Bejan, A. "Advanced Engineering Thermodynamics"
        - Kotas, T.J. "The Exergy Method of Thermal Plant Analysis"
    """

    # Default dead state (ISO standard atmosphere)
    DEFAULT_T0 = Decimal("298.15")  # K (25C)
    DEFAULT_P0 = Decimal("101.325")  # kPa

    # Universal gas constant
    R_UNIVERSAL = Decimal("8.31446261815324")  # J/(mol*K)

    def __init__(
        self,
        dead_state_temp_k: float = 298.15,
        dead_state_pressure_kpa: float = 101.325,
        precision: int = 4
    ):
        """
        Initialize exergy calculator.

        Args:
            dead_state_temp_k: Reference environment temperature (K)
            dead_state_pressure_kpa: Reference environment pressure (kPa)
            precision: Decimal precision for outputs
        """
        self.t0 = Decimal(str(dead_state_temp_k))
        self.p0 = Decimal(str(dead_state_pressure_kpa))
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
            "method": "Exergy_Analysis_Second_Law",
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def physical_exergy(
        self,
        state: ExergyState,
        h0_kj_kg: float,
        s0_kj_kgk: float
    ) -> ExergyResult:
        """
        Calculate physical (thermo-mechanical) exergy.

        Reference: Bejan, "Advanced Engineering Thermodynamics", Eq. 3.21

        e_ph = (h - h0) - T0 * (s - s0)

        Physical exergy can be decomposed into:
        - Thermal exergy (temperature difference from dead state)
        - Mechanical exergy (pressure difference from dead state)

        Args:
            state: Current thermodynamic state
            h0_kj_kg: Specific enthalpy at dead state (kJ/kg)
            s0_kj_kgk: Specific entropy at dead state (kJ/kg-K)

        Returns:
            ExergyResult with physical exergy components
        """
        h = state.enthalpy_kj_kg
        s = state.entropy_kj_kgk
        t = state.temperature_k
        p = state.pressure_kpa
        h0 = Decimal(str(h0_kj_kg))
        s0 = Decimal(str(s0_kj_kgk))

        # Total physical exergy
        e_ph = (h - h0) - self.t0 * (s - s0)

        # Decomposition into thermal and mechanical components
        # Thermal exergy (at constant pressure p0)
        # e_th = (h - h_T0_p) - T0 * (s - s_T0_p) where T0,p is intermediate state

        # For ideal gas approximation:
        # e_th = Cp * (T - T0) - T0 * Cp * ln(T/T0)
        # e_mech = R * T0 * ln(p/p0)

        # Using simplified approach for general fluids:
        # Assume thermal contribution dominates for T > T0, mechanical for p > p0

        if t > self.t0:
            # Carnot efficiency approach for thermal exergy
            carnot = Decimal("1") - self.t0 / t
            q_transfer = h - h0  # Approximate heat content
            e_thermal = carnot * q_transfer if q_transfer > 0 else Decimal("0")
        else:
            e_thermal = Decimal("0")

        e_mechanical = e_ph - e_thermal
        if e_mechanical < Decimal("0"):
            e_mechanical = Decimal("0")
            e_thermal = e_ph

        # Calculate exergy flow if mass flow provided
        exergy_flow = None
        if state.mass_flow_kg_s is not None:
            exergy_flow = state.mass_flow_kg_s * e_ph

        inputs = {
            "temperature_k": str(t),
            "pressure_kpa": str(p),
            "enthalpy_kj_kg": str(h),
            "entropy_kj_kgk": str(s),
            "h0": str(h0),
            "s0": str(s0)
        }
        outputs = {
            "e_physical": str(e_ph),
            "e_thermal": str(e_thermal),
            "e_mechanical": str(e_mechanical)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return ExergyResult(
            physical_exergy_kj_kg=self._apply_precision(e_ph),
            thermal_exergy_kj_kg=self._apply_precision(e_thermal),
            mechanical_exergy_kj_kg=self._apply_precision(e_mechanical),
            exergy_flow_kw=self._apply_precision(exergy_flow) if exergy_flow else None,
            dead_state_temperature_k=self._apply_precision(self.t0),
            dead_state_pressure_kpa=self._apply_precision(self.p0),
            provenance_hash=provenance_hash
        )

    def chemical_exergy_ideal_gas(
        self,
        composition_molar: Dict[str, float],
        standard_chemical_exergies: Dict[str, float]
    ) -> Decimal:
        """
        Calculate chemical exergy for ideal gas mixture.

        Reference: Szargut, "Exergy Method", Chapter 2

        e_ch = Sum(x_i * e_ch,i) + R*T0 * Sum(x_i * ln(x_i))

        Args:
            composition_molar: Mole fractions {component: fraction}
            standard_chemical_exergies: Standard chemical exergies {component: kJ/mol}

        Returns:
            Chemical exergy in kJ/mol
        """
        # Validate composition sums to 1
        total = sum(composition_molar.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Mole fractions must sum to 1, got {total}")

        # Standard chemical exergy contribution
        e_ch_std = Decimal("0")
        for comp, x in composition_molar.items():
            if comp in standard_chemical_exergies:
                e_ch_std += Decimal(str(x)) * Decimal(str(standard_chemical_exergies[comp]))

        # Mixing term (negative contribution)
        mixing_term = Decimal("0")
        r_kj = self.R_UNIVERSAL / Decimal("1000")  # kJ/(mol*K)

        for comp, x in composition_molar.items():
            if x > 0:
                mixing_term += Decimal(str(x)) * Decimal(str(math.log(x)))

        e_ch_mix = r_kj * self.t0 * mixing_term

        e_ch_total = e_ch_std + e_ch_mix

        return self._apply_precision(e_ch_total)

    def exergy_destruction(
        self,
        exergy_in: List[Decimal],
        exergy_out: List[Decimal],
        exergy_loss: Decimal = Decimal("0")
    ) -> ExergyDestructionResult:
        """
        Calculate exergy destruction for a process.

        Reference: Kotas, "Exergy Method", Chapter 3

        Exergy balance: E_in = E_out + E_destruction + E_loss

        Args:
            exergy_in: List of input exergy flows (kW)
            exergy_out: List of output exergy flows (kW)
            exergy_loss: Exergy lost to environment (kW)

        Returns:
            ExergyDestructionResult with destruction analysis
        """
        e_in = sum(exergy_in)
        e_out = sum(exergy_out)
        e_loss = exergy_loss

        # Exergy destruction (irreversibility)
        e_destruction = e_in - e_out - e_loss

        if e_destruction < Decimal("0"):
            raise ValueError("Negative exergy destruction - check inputs")

        # Exergetic efficiency
        if e_in > Decimal("0"):
            eta_ex = e_out / e_in
            irreversibility_ratio = e_destruction / e_in
        else:
            eta_ex = Decimal("0")
            irreversibility_ratio = Decimal("0")

        inputs = {
            "exergy_in": str(e_in),
            "exergy_out": str(e_out),
            "exergy_loss": str(e_loss)
        }
        outputs = {
            "destruction": str(e_destruction),
            "efficiency": str(eta_ex)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return ExergyDestructionResult(
            exergy_in_kw=self._apply_precision(e_in),
            exergy_out_kw=self._apply_precision(e_out),
            exergy_destruction_kw=self._apply_precision(e_destruction),
            exergy_loss_kw=self._apply_precision(e_loss),
            exergetic_efficiency=self._apply_precision(eta_ex),
            irreversibility_ratio=self._apply_precision(irreversibility_ratio),
            provenance_hash=provenance_hash
        )

    def component_analysis(
        self,
        component_name: str,
        exergy_fuel: Decimal,
        exergy_product: Decimal,
        exergy_loss: Decimal = Decimal("0")
    ) -> ComponentExergyAnalysis:
        """
        Perform exergy analysis for a single component.

        Reference: Bejan, "Advanced Engineering Thermodynamics", Chapter 8

        Uses fuel-product-loss (FPL) formulation:
        E_F = E_P + E_D + E_L

        Args:
            component_name: Name of the component
            exergy_fuel: Exergy input (fuel) in kW
            exergy_product: Useful exergy output (product) in kW
            exergy_loss: Exergy lost to environment in kW

        Returns:
            ComponentExergyAnalysis with detailed breakdown
        """
        e_f = exergy_fuel
        e_p = exergy_product
        e_l = exergy_loss

        # Exergy destruction
        e_d = e_f - e_p - e_l

        if e_d < Decimal("0"):
            raise ValueError(f"Negative exergy destruction for {component_name}")

        # Exergetic efficiency
        if e_f > Decimal("0"):
            eta = e_p / e_f
        else:
            eta = Decimal("0")

        # Improvement potential (van Gool)
        # IP = (1 - eta) * E_D
        improvement_potential = (Decimal("1") - eta) * e_d

        inputs = {
            "component": component_name,
            "exergy_fuel": str(e_f),
            "exergy_product": str(e_p)
        }
        outputs = {
            "destruction": str(e_d),
            "efficiency": str(eta)
        }
        provenance_hash = self._calculate_provenance(inputs, outputs)

        return ComponentExergyAnalysis(
            component_name=component_name,
            exergy_fuel_kw=self._apply_precision(e_f),
            exergy_product_kw=self._apply_precision(e_p),
            exergy_destruction_kw=self._apply_precision(e_d),
            exergy_loss_kw=self._apply_precision(e_l),
            exergetic_efficiency=self._apply_precision(eta),
            improvement_potential_kw=self._apply_precision(improvement_potential),
            provenance_hash=provenance_hash
        )

    def heat_exergy(
        self,
        heat_duty_kw: float,
        temperature_k: float
    ) -> Decimal:
        """
        Calculate exergy associated with heat transfer.

        Reference: Bejan, Eq. 3.18

        E_Q = Q * (1 - T0/T) for T > T0 (heating)
        E_Q = Q * (T0/T - 1) for T < T0 (cooling)

        Args:
            heat_duty_kw: Heat transfer rate (kW)
            temperature_k: Temperature at which heat is transferred (K)

        Returns:
            Exergy of heat in kW
        """
        q = Decimal(str(heat_duty_kw))
        t = Decimal(str(temperature_k))

        if t <= Decimal("0"):
            raise ValueError("Temperature must be positive")

        # Carnot factor
        if t >= self.t0:
            carnot = Decimal("1") - self.t0 / t
        else:
            carnot = self.t0 / t - Decimal("1")

        e_q = abs(q) * carnot

        return self._apply_precision(e_q)

    def work_exergy(self, power_kw: float) -> Decimal:
        """
        Calculate exergy of work/power.

        Work is pure exergy (100% convertible to other forms).

        Args:
            power_kw: Power in kW

        Returns:
            Exergy in kW (equal to power)
        """
        return self._apply_precision(Decimal(str(abs(power_kw))))

    def system_exergy_balance(
        self,
        components: List[ComponentExergyAnalysis]
    ) -> Dict[str, Decimal]:
        """
        Calculate system-level exergy metrics from component analyses.

        Args:
            components: List of component analysis results

        Returns:
            Dictionary with system-level metrics
        """
        total_fuel = sum(c.exergy_fuel_kw for c in components)
        total_product = sum(c.exergy_product_kw for c in components)
        total_destruction = sum(c.exergy_destruction_kw for c in components)
        total_loss = sum(c.exergy_loss_kw for c in components)

        # System efficiency
        if total_fuel > Decimal("0"):
            system_efficiency = total_product / total_fuel
        else:
            system_efficiency = Decimal("0")

        # Destruction ratios by component
        destruction_ratios = {}
        for c in components:
            if total_destruction > Decimal("0"):
                destruction_ratios[c.component_name] = c.exergy_destruction_kw / total_destruction

        return {
            "total_exergy_fuel_kw": self._apply_precision(total_fuel),
            "total_exergy_product_kw": self._apply_precision(total_product),
            "total_exergy_destruction_kw": self._apply_precision(total_destruction),
            "total_exergy_loss_kw": self._apply_precision(total_loss),
            "system_exergetic_efficiency": self._apply_precision(system_efficiency),
            "component_destruction_ratios": {
                k: self._apply_precision(v) for k, v in destruction_ratios.items()
            }
        }


# Standard chemical exergies (kJ/mol) at 25C, 1 atm
# Reference: Szargut, "Exergy Method", Appendix
STANDARD_CHEMICAL_EXERGIES = {
    "N2": 0.72,
    "O2": 3.97,
    "H2O_vapor": 9.50,
    "H2O_liquid": 0.90,
    "CO2": 19.87,
    "CO": 275.10,
    "H2": 236.10,
    "CH4": 831.20,
    "C2H6": 1495.84,
    "C3H8": 2154.00,
    "C": 410.26,  # Graphite
    "S": 609.60,
    "SO2": 313.40,
    "NO": 88.90,
    "NO2": 55.60,
    "NH3": 337.90,
}


# Convenience functions
def calculate_physical_exergy(
    temperature_k: float,
    pressure_kpa: float,
    enthalpy_kj_kg: float,
    entropy_kj_kgk: float,
    h0_kj_kg: float,
    s0_kj_kgk: float,
    mass_flow_kg_s: Optional[float] = None
) -> ExergyResult:
    """
    Calculate physical exergy for a stream.

    Example:
        >>> result = calculate_physical_exergy(
        ...     temperature_k=500,
        ...     pressure_kpa=1000,
        ...     enthalpy_kj_kg=2800,
        ...     entropy_kj_kgk=7.0,
        ...     h0_kj_kg=105,
        ...     s0_kj_kgk=0.37
        ... )
        >>> print(f"Physical exergy: {result.physical_exergy_kj_kg} kJ/kg")
    """
    calc = ExergyCalculator()
    state = ExergyState(
        temperature_k=Decimal(str(temperature_k)),
        pressure_kpa=Decimal(str(pressure_kpa)),
        enthalpy_kj_kg=Decimal(str(enthalpy_kj_kg)),
        entropy_kj_kgk=Decimal(str(entropy_kj_kgk)),
        mass_flow_kg_s=Decimal(str(mass_flow_kg_s)) if mass_flow_kg_s else None
    )
    return calc.physical_exergy(state, h0_kj_kg, s0_kj_kgk)


def exergetic_efficiency(
    exergy_product_kw: float,
    exergy_fuel_kw: float
) -> Decimal:
    """
    Calculate exergetic efficiency.

    eta_ex = E_product / E_fuel
    """
    if exergy_fuel_kw <= 0:
        raise ValueError("Exergy fuel must be positive")

    calc = ExergyCalculator()
    return calc._apply_precision(
        Decimal(str(exergy_product_kw)) / Decimal(str(exergy_fuel_kw))
    )


def heat_to_exergy(heat_kw: float, temperature_k: float) -> Decimal:
    """
    Convert heat to exergy using Carnot factor.

    Example:
        >>> exergy = heat_to_exergy(1000, 500)  # 1000 kW at 500 K
        >>> print(f"Exergy: {exergy} kW")
    """
    calc = ExergyCalculator()
    return calc.heat_exergy(heat_kw, temperature_k)
