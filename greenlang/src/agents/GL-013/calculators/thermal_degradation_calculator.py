"""
GL-013 PREDICTMAINT - Thermal Degradation Calculator

This module implements thermal life estimation using the Arrhenius
equation and IEEE standards for insulation aging.

Key Features:
- Arrhenius equation: L = A * exp(Ea / kT)
- IEEE C57.91 transformer insulation aging
- Hot spot temperature calculation
- Thermal cycling fatigue
- Overload aging acceleration

Reference Standards:
- IEEE C57.91-2011: Transformer Loading Guide
- IEEE C57.96: Dry-Type Transformer Loading Guide
- IEC 60076-7: Power Transformer Loading
- Arrhenius, S. (1889): Reaction Rate Theory

Author: GL-CalculatorEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum, auto
import math

from .constants import (
    BOLTZMANN_CONSTANT_EV,
    KELVIN_OFFSET,
    ARRHENIUS_PARAMETERS,
    ArrheniusParameters,
    DEFAULT_DECIMAL_PRECISION,
    E,
)
from .provenance import (
    ProvenanceBuilder,
    ProvenanceRecord,
    CalculationType,
    store_provenance,
)


# =============================================================================
# ENUMS
# =============================================================================

class InsulationClass(Enum):
    """IEC/IEEE insulation temperature classes."""
    CLASS_A = auto()   # 105 C rating
    CLASS_B = auto()   # 130 C rating
    CLASS_F = auto()   # 155 C rating
    CLASS_H = auto()   # 180 C rating
    CLASS_N = auto()   # 200 C rating
    CLASS_R = auto()   # 220 C rating


class LoadingType(Enum):
    """Transformer loading patterns."""
    CONTINUOUS = auto()
    CYCLIC = auto()
    EMERGENCY = auto()


class EquipmentCategory(Enum):
    """Equipment categories for thermal analysis."""
    TRANSFORMER_OIL = auto()
    TRANSFORMER_DRY = auto()
    MOTOR = auto()
    CABLE = auto()
    CAPACITOR = auto()


# Insulation class temperature ratings (hot spot, Celsius)
INSULATION_RATINGS: Dict[InsulationClass, Decimal] = {
    InsulationClass.CLASS_A: Decimal("105"),
    InsulationClass.CLASS_B: Decimal("130"),
    InsulationClass.CLASS_F: Decimal("155"),
    InsulationClass.CLASS_H: Decimal("180"),
    InsulationClass.CLASS_N: Decimal("200"),
    InsulationClass.CLASS_R: Decimal("220"),
}

# Reference life at rated temperature (hours) per IEEE C57.91
IEEE_REFERENCE_LIFE_HOURS: Decimal = Decimal("180000")  # ~20.5 years


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class ThermalLifeResult:
    """
    Result of thermal life estimation.

    Attributes:
        remaining_life_hours: Estimated remaining life at current conditions
        remaining_life_years: Same in years
        aging_acceleration_factor: Factor by which aging is accelerated
        equivalent_aging_hours: Equivalent hours at rated temperature
        life_consumed_percent: Percentage of life already consumed
        hot_spot_temperature_c: Calculated or measured hot spot temp
        reference_temperature_c: Reference temperature for comparison
        provenance_hash: SHA-256 hash for audit
    """
    remaining_life_hours: Decimal
    remaining_life_years: Decimal
    aging_acceleration_factor: Decimal
    equivalent_aging_hours: Decimal
    life_consumed_percent: Decimal
    hot_spot_temperature_c: Decimal
    reference_temperature_c: Decimal
    insulation_class: Optional[str] = None
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "remaining_life_hours": str(self.remaining_life_hours),
            "remaining_life_years": str(self.remaining_life_years),
            "aging_acceleration_factor": str(self.aging_acceleration_factor),
            "equivalent_aging_hours": str(self.equivalent_aging_hours),
            "life_consumed_percent": str(self.life_consumed_percent),
            "hot_spot_temperature_c": str(self.hot_spot_temperature_c),
            "reference_temperature_c": str(self.reference_temperature_c),
            "insulation_class": self.insulation_class,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class AgingAccelerationResult:
    """Result of aging acceleration factor calculation."""
    acceleration_factor: Decimal
    operating_temperature_c: Decimal
    reference_temperature_c: Decimal
    activation_energy_ev: Decimal
    provenance_hash: str = ""


@dataclass(frozen=True)
class HotSpotResult:
    """Result of hot spot temperature calculation."""
    hot_spot_temperature_c: Decimal
    top_oil_temperature_c: Decimal
    ambient_temperature_c: Decimal
    hot_spot_rise: Decimal
    top_oil_rise: Decimal
    load_factor: Decimal
    provenance_hash: str = ""


@dataclass(frozen=True)
class ThermalCycleResult:
    """Result of thermal cycling analysis."""
    cumulative_damage: Decimal
    cycles_to_failure: Decimal
    current_cycles: int
    remaining_cycles: int
    fatigue_life_consumed_percent: Decimal
    temperature_range_c: Decimal
    mean_temperature_c: Decimal
    provenance_hash: str = ""


@dataclass(frozen=True)
class OverloadAssessmentResult:
    """Result of overload thermal assessment."""
    allowable_overload_percent: Decimal
    duration_hours: Decimal
    life_consumed_hours: Decimal
    hot_spot_at_overload_c: Decimal
    is_within_limits: bool
    limiting_factor: str
    provenance_hash: str = ""


# =============================================================================
# THERMAL DEGRADATION CALCULATOR
# =============================================================================

class ThermalDegradationCalculator:
    """
    Thermal degradation and life estimation calculator.

    Implements the Arrhenius equation for thermal aging and
    IEEE standards for transformer and insulation life assessment.

    The Arrhenius equation relates reaction rate to temperature:
        k = A * exp(-Ea / (k_B * T))

    For insulation life:
        L(T) = L_ref * exp(Ea/k_B * (1/T - 1/T_ref))

    Reference: IEEE C57.91-2011, Clause 7

    Example:
        >>> calc = ThermalDegradationCalculator()
        >>> result = calc.calculate_thermal_life(
        ...     operating_temperature_c=Decimal("110"),
        ...     insulation_class=InsulationClass.CLASS_A,
        ...     operating_hours=Decimal("50000")
        ... )
        >>> print(f"Remaining life: {result.remaining_life_years} years")
    """

    def __init__(
        self,
        precision: int = DEFAULT_DECIMAL_PRECISION,
        store_provenance_records: bool = True
    ):
        """
        Initialize Thermal Degradation Calculator.

        Args:
            precision: Decimal precision for calculations
            store_provenance_records: Whether to store provenance
        """
        self._precision = precision
        self._store_provenance = store_provenance_records

    # =========================================================================
    # ARRHENIUS AGING CALCULATION
    # =========================================================================

    def calculate_aging_acceleration_factor(
        self,
        operating_temperature_c: Union[Decimal, float, str],
        reference_temperature_c: Union[Decimal, float, str],
        activation_energy_ev: Union[Decimal, float, str]
    ) -> AgingAccelerationResult:
        """
        Calculate aging acceleration factor using Arrhenius equation.

        The aging acceleration factor (FAA) represents how much faster
        insulation ages at the operating temperature compared to the
        reference temperature:

            FAA = exp(Ea/k_B * (1/T_ref - 1/T_op))

        Where:
            Ea = Activation energy (eV)
            k_B = Boltzmann constant (eV/K)
            T = Temperature (Kelvin)

        Args:
            operating_temperature_c: Operating temperature in Celsius
            reference_temperature_c: Reference temperature in Celsius
            activation_energy_ev: Activation energy in electron-volts

        Returns:
            AgingAccelerationResult

        Reference:
            Arrhenius, S. (1889). "Uber die Reaktionsgeschwindigkeit"
            IEEE C57.91-2011, Equation 1

        Example:
            >>> calc = ThermalDegradationCalculator()
            >>> result = calc.calculate_aging_acceleration_factor(
            ...     operating_temperature_c="120",
            ...     reference_temperature_c="110",
            ...     activation_energy_ev="0.90"
            ... )
            >>> print(f"FAA: {result.acceleration_factor}")
        """
        builder = ProvenanceBuilder(CalculationType.THERMAL_DEGRADATION)

        # Convert inputs
        T_op_c = self._to_decimal(operating_temperature_c)
        T_ref_c = self._to_decimal(reference_temperature_c)
        Ea = self._to_decimal(activation_energy_ev)
        k_B = BOLTZMANN_CONSTANT_EV

        # Convert to Kelvin
        T_op_k = T_op_c + KELVIN_OFFSET
        T_ref_k = T_ref_c + KELVIN_OFFSET

        builder.add_input("operating_temperature_c", T_op_c)
        builder.add_input("reference_temperature_c", T_ref_c)
        builder.add_input("activation_energy_ev", Ea)

        # Step 1: Calculate temperature term
        # (1/T_ref - 1/T_op)
        temp_term = (Decimal("1") / T_ref_k) - (Decimal("1") / T_op_k)

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate temperature term",
            inputs={"T_ref_k": T_ref_k, "T_op_k": T_op_k},
            output_name="temp_term",
            output_value=temp_term,
            formula="(1/T_ref - 1/T_op)"
        )

        # Step 2: Calculate exponent
        # Ea / k_B * temp_term
        exponent = (Ea / k_B) * temp_term

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate Arrhenius exponent",
            inputs={"Ea": Ea, "k_B": k_B, "temp_term": temp_term},
            output_name="exponent",
            output_value=exponent,
            formula="Ea / k_B * (1/T_ref - 1/T_op)"
        )

        # Step 3: Calculate acceleration factor
        # FAA = exp(exponent)
        faa = self._exp(exponent)

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate aging acceleration factor",
            inputs={"exponent": exponent},
            output_name="acceleration_factor",
            output_value=faa,
            formula="FAA = exp(Ea/k_B * (1/T_ref - 1/T_op))",
            reference="Arrhenius equation"
        )

        builder.add_output("acceleration_factor", faa)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return AgingAccelerationResult(
            acceleration_factor=self._apply_precision(faa, 4),
            operating_temperature_c=T_op_c,
            reference_temperature_c=T_ref_c,
            activation_energy_ev=Ea,
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # THERMAL LIFE ESTIMATION
    # =========================================================================

    def calculate_thermal_life(
        self,
        operating_temperature_c: Union[Decimal, float, str],
        insulation_class: InsulationClass,
        operating_hours: Union[Decimal, float, int, str],
        equipment_type: Optional[str] = None,
        custom_activation_energy_ev: Optional[Union[Decimal, float, str]] = None,
        custom_reference_life_hours: Optional[Union[Decimal, float, str]] = None
    ) -> ThermalLifeResult:
        """
        Calculate remaining thermal life of insulation.

        Uses the Arrhenius equation to estimate insulation life
        based on operating temperature and time.

        The remaining life is:
            L_remaining = L_total - (FAA * t_operating)

        Where:
            L_total = Reference life at rated temperature
            FAA = Aging acceleration factor
            t_operating = Operating hours

        Args:
            operating_temperature_c: Average operating hot spot temperature
            insulation_class: Insulation temperature class
            operating_hours: Hours of operation at this temperature
            equipment_type: Optional equipment type for lookup
            custom_activation_energy_ev: Override activation energy
            custom_reference_life_hours: Override reference life

        Returns:
            ThermalLifeResult

        Reference:
            IEEE C57.91-2011, Clause 7

        Example:
            >>> calc = ThermalDegradationCalculator()
            >>> result = calc.calculate_thermal_life(
            ...     operating_temperature_c="115",
            ...     insulation_class=InsulationClass.CLASS_A,
            ...     operating_hours="50000"
            ... )
        """
        builder = ProvenanceBuilder(CalculationType.THERMAL_DEGRADATION)

        # Convert inputs
        T_op_c = self._to_decimal(operating_temperature_c)
        t_hours = self._to_decimal(operating_hours)

        # Get reference temperature from insulation class
        T_ref_c = INSULATION_RATINGS[insulation_class]

        # Get activation energy
        if custom_activation_energy_ev:
            Ea = self._to_decimal(custom_activation_energy_ev)
        elif equipment_type and equipment_type in ARRHENIUS_PARAMETERS:
            params = ARRHENIUS_PARAMETERS[equipment_type]
            Ea = params.activation_energy_ev
        else:
            # Default activation energy for insulation (IEEE C57.91)
            # Approximately 1.0 eV for cellulose insulation
            Ea = Decimal("1.0")

        # Get reference life
        if custom_reference_life_hours:
            L_ref = self._to_decimal(custom_reference_life_hours)
        elif equipment_type and equipment_type in ARRHENIUS_PARAMETERS:
            L_ref = ARRHENIUS_PARAMETERS[equipment_type].reference_life_hours
        else:
            L_ref = IEEE_REFERENCE_LIFE_HOURS

        builder.add_input("operating_temperature_c", T_op_c)
        builder.add_input("insulation_class", insulation_class.name)
        builder.add_input("operating_hours", t_hours)
        builder.add_input("activation_energy_ev", Ea)
        builder.add_input("reference_life_hours", L_ref)
        builder.add_input("reference_temperature_c", T_ref_c)

        # Step 1: Calculate aging acceleration factor
        faa_result = self.calculate_aging_acceleration_factor(
            T_op_c, T_ref_c, Ea
        )
        faa = faa_result.acceleration_factor

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate aging acceleration factor",
            inputs={
                "T_op": T_op_c,
                "T_ref": T_ref_c,
                "Ea": Ea
            },
            output_name="acceleration_factor",
            output_value=faa,
            formula="FAA = exp(Ea/k_B * (1/T_ref - 1/T_op))",
            reference="IEEE C57.91"
        )

        # Step 2: Calculate equivalent aging hours
        # Hours at reference temperature equivalent to actual operating hours
        equivalent_hours = faa * t_hours

        builder.add_step(
            step_number=2,
            operation="multiply",
            description="Calculate equivalent aging hours",
            inputs={"faa": faa, "operating_hours": t_hours},
            output_name="equivalent_hours",
            output_value=equivalent_hours,
            formula="t_equivalent = FAA * t_actual"
        )

        # Step 3: Calculate remaining life
        remaining_life_hours = L_ref - equivalent_hours
        remaining_life_hours = max(Decimal("0"), remaining_life_hours)
        remaining_life_years = remaining_life_hours / Decimal("8760")

        builder.add_step(
            step_number=3,
            operation="subtract",
            description="Calculate remaining life",
            inputs={"reference_life": L_ref, "consumed_life": equivalent_hours},
            output_name="remaining_life_hours",
            output_value=remaining_life_hours,
            formula="L_remaining = L_ref - t_equivalent"
        )

        # Step 4: Calculate life consumed percentage
        life_consumed_percent = (equivalent_hours / L_ref) * Decimal("100")
        life_consumed_percent = min(life_consumed_percent, Decimal("100"))

        builder.add_step(
            step_number=4,
            operation="calculate",
            description="Calculate life consumed percentage",
            inputs={"consumed": equivalent_hours, "total": L_ref},
            output_name="life_consumed_percent",
            output_value=life_consumed_percent,
            formula="% = (consumed / total) * 100"
        )

        # Finalize
        builder.add_output("remaining_life_hours", remaining_life_hours)
        builder.add_output("remaining_life_years", remaining_life_years)
        builder.add_output("life_consumed_percent", life_consumed_percent)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return ThermalLifeResult(
            remaining_life_hours=self._apply_precision(remaining_life_hours, 0),
            remaining_life_years=self._apply_precision(remaining_life_years, 2),
            aging_acceleration_factor=faa,
            equivalent_aging_hours=self._apply_precision(equivalent_hours, 0),
            life_consumed_percent=self._apply_precision(life_consumed_percent, 2),
            hot_spot_temperature_c=T_op_c,
            reference_temperature_c=T_ref_c,
            insulation_class=insulation_class.name,
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # HOT SPOT TEMPERATURE CALCULATION
    # =========================================================================

    def calculate_hot_spot_temperature(
        self,
        ambient_temperature_c: Union[Decimal, float, str],
        load_factor: Union[Decimal, float, str],
        rated_top_oil_rise_c: Union[Decimal, float, str] = "65",
        rated_hot_spot_rise_c: Union[Decimal, float, str] = "80",
        oil_exponent: Union[Decimal, float, str] = "0.8",
        winding_exponent: Union[Decimal, float, str] = "1.6"
    ) -> HotSpotResult:
        """
        Calculate transformer hot spot temperature.

        Uses IEEE C57.91 thermal model for oil-immersed transformers:

        Top oil rise = (Top oil rise at rated) * (load factor)^oil_exp
        Hot spot rise = (Winding rise at rated) * (load factor)^winding_exp
        Hot spot = Ambient + Top oil rise + (Hot spot rise - Top oil rise)

        Args:
            ambient_temperature_c: Ambient temperature
            load_factor: Load as fraction of rated (e.g., 1.0 = 100%)
            rated_top_oil_rise_c: Top oil rise at rated load (IEEE default: 65C)
            rated_hot_spot_rise_c: Hot spot rise at rated load (IEEE default: 80C)
            oil_exponent: Oil time constant exponent (default 0.8)
            winding_exponent: Winding time constant exponent (default 1.6)

        Returns:
            HotSpotResult

        Reference:
            IEEE C57.91-2011, Clause 7.2

        Example:
            >>> calc = ThermalDegradationCalculator()
            >>> result = calc.calculate_hot_spot_temperature(
            ...     ambient_temperature_c="30",
            ...     load_factor="1.1"
            ... )
            >>> print(f"Hot spot: {result.hot_spot_temperature_c} C")
        """
        builder = ProvenanceBuilder(CalculationType.THERMAL_DEGRADATION)

        # Convert inputs
        T_amb = self._to_decimal(ambient_temperature_c)
        K = self._to_decimal(load_factor)
        delta_TO_rated = self._to_decimal(rated_top_oil_rise_c)
        delta_HS_rated = self._to_decimal(rated_hot_spot_rise_c)
        n = self._to_decimal(oil_exponent)
        m = self._to_decimal(winding_exponent)

        builder.add_input("ambient_temperature_c", T_amb)
        builder.add_input("load_factor", K)
        builder.add_input("rated_top_oil_rise_c", delta_TO_rated)
        builder.add_input("rated_hot_spot_rise_c", delta_HS_rated)

        # Step 1: Calculate top oil rise at current load
        # delta_TO = delta_TO_rated * K^n
        delta_TO = delta_TO_rated * self._power(K, n)

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate top oil temperature rise",
            inputs={"delta_TO_rated": delta_TO_rated, "K": K, "n": n},
            output_name="delta_TO",
            output_value=delta_TO,
            formula="delta_TO = delta_TO_rated * K^n",
            reference="IEEE C57.91 Eq. 4"
        )

        # Step 2: Calculate top oil temperature
        T_TO = T_amb + delta_TO

        builder.add_step(
            step_number=2,
            operation="add",
            description="Calculate top oil temperature",
            inputs={"T_amb": T_amb, "delta_TO": delta_TO},
            output_name="T_TO",
            output_value=T_TO,
            formula="T_TO = T_ambient + delta_TO"
        )

        # Step 3: Calculate hot spot rise above top oil
        # delta_HS_TO = (delta_HS_rated - delta_TO_rated) * K^m
        delta_HS_TO_rated = delta_HS_rated - delta_TO_rated
        delta_HS_TO = delta_HS_TO_rated * self._power(K, m)

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate hot spot rise above top oil",
            inputs={"delta_HS_TO_rated": delta_HS_TO_rated, "K": K, "m": m},
            output_name="delta_HS_TO",
            output_value=delta_HS_TO,
            formula="delta_HS_TO = (delta_HS_rated - delta_TO_rated) * K^m",
            reference="IEEE C57.91 Eq. 5"
        )

        # Step 4: Calculate hot spot temperature
        T_HS = T_TO + delta_HS_TO

        builder.add_step(
            step_number=4,
            operation="add",
            description="Calculate hot spot temperature",
            inputs={"T_TO": T_TO, "delta_HS_TO": delta_HS_TO},
            output_name="T_HS",
            output_value=T_HS,
            formula="T_HS = T_TO + delta_HS_TO"
        )

        builder.add_output("hot_spot_temperature_c", T_HS)
        builder.add_output("top_oil_temperature_c", T_TO)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return HotSpotResult(
            hot_spot_temperature_c=self._apply_precision(T_HS, 1),
            top_oil_temperature_c=self._apply_precision(T_TO, 1),
            ambient_temperature_c=T_amb,
            hot_spot_rise=self._apply_precision(delta_TO + delta_HS_TO, 1),
            top_oil_rise=self._apply_precision(delta_TO, 1),
            load_factor=K,
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # THERMAL CYCLING FATIGUE
    # =========================================================================

    def calculate_thermal_cycle_damage(
        self,
        temperature_max_c: Union[Decimal, float, str],
        temperature_min_c: Union[Decimal, float, str],
        num_cycles: int,
        material_constant_c: Union[Decimal, float, str] = "0.2",
        material_constant_n: Union[Decimal, float, str] = "2.0"
    ) -> ThermalCycleResult:
        """
        Calculate cumulative damage from thermal cycling.

        Uses Coffin-Manson relationship for thermal fatigue:

            N_f = C * (delta_T)^(-n)

        Where:
            N_f = Cycles to failure
            delta_T = Temperature range
            C, n = Material constants

        Cumulative damage (Miner's rule):
            D = sum(n_i / N_f_i)

        Args:
            temperature_max_c: Maximum cycle temperature
            temperature_min_c: Minimum cycle temperature
            num_cycles: Number of cycles experienced
            material_constant_c: Material constant C (default for solder)
            material_constant_n: Material exponent n

        Returns:
            ThermalCycleResult

        Reference:
            Coffin, L.F. (1954). "A Study of the Effects of Cyclic
            Thermal Stresses on a Ductile Metal"

        Example:
            >>> calc = ThermalDegradationCalculator()
            >>> result = calc.calculate_thermal_cycle_damage(
            ...     temperature_max_c="85",
            ...     temperature_min_c="25",
            ...     num_cycles=1000
            ... )
        """
        builder = ProvenanceBuilder(CalculationType.THERMAL_DEGRADATION)

        # Convert inputs
        T_max = self._to_decimal(temperature_max_c)
        T_min = self._to_decimal(temperature_min_c)
        C = self._to_decimal(material_constant_c)
        n = self._to_decimal(material_constant_n)

        builder.add_input("temperature_max_c", T_max)
        builder.add_input("temperature_min_c", T_min)
        builder.add_input("num_cycles", num_cycles)
        builder.add_input("material_constant_c", C)
        builder.add_input("material_constant_n", n)

        # Step 1: Calculate temperature range and mean
        delta_T = T_max - T_min
        T_mean = (T_max + T_min) / Decimal("2")

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate temperature range",
            inputs={"T_max": T_max, "T_min": T_min},
            output_name="delta_T",
            output_value=delta_T,
            formula="delta_T = T_max - T_min"
        )

        # Step 2: Calculate cycles to failure using Coffin-Manson
        # N_f = C * (delta_T)^(-n)
        # Rearranged: N_f = C / (delta_T^n)
        if delta_T > Decimal("0"):
            N_f = C * Decimal("1e6") / self._power(delta_T, n)  # Scale factor for practical values
        else:
            N_f = Decimal("Infinity")

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate cycles to failure (Coffin-Manson)",
            inputs={"C": C, "delta_T": delta_T, "n": n},
            output_name="cycles_to_failure",
            output_value=N_f,
            formula="N_f = C * 1e6 / (delta_T)^n",
            reference="Coffin-Manson equation"
        )

        # Step 3: Calculate cumulative damage (Miner's rule)
        if N_f > Decimal("0") and N_f != Decimal("Infinity"):
            D = Decimal(str(num_cycles)) / N_f
        else:
            D = Decimal("0")

        builder.add_step(
            step_number=3,
            operation="divide",
            description="Calculate cumulative damage (Miner's rule)",
            inputs={"num_cycles": num_cycles, "N_f": N_f},
            output_name="cumulative_damage",
            output_value=D,
            formula="D = n / N_f",
            reference="Palmgren-Miner linear damage rule"
        )

        # Step 4: Calculate remaining cycles
        if N_f != Decimal("Infinity"):
            remaining_cycles = max(0, int(N_f) - num_cycles)
            life_consumed = D * Decimal("100")
        else:
            remaining_cycles = 999999999
            life_consumed = Decimal("0")

        builder.add_step(
            step_number=4,
            operation="calculate",
            description="Calculate remaining cycles",
            inputs={"N_f": N_f, "current_cycles": num_cycles},
            output_name="remaining_cycles",
            output_value=remaining_cycles
        )

        builder.add_output("cumulative_damage", D)
        builder.add_output("cycles_to_failure", N_f)
        builder.add_output("remaining_cycles", remaining_cycles)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return ThermalCycleResult(
            cumulative_damage=self._apply_precision(D, 6),
            cycles_to_failure=self._apply_precision(N_f, 0) if N_f != Decimal("Infinity") else N_f,
            current_cycles=num_cycles,
            remaining_cycles=remaining_cycles,
            fatigue_life_consumed_percent=self._apply_precision(min(life_consumed, Decimal("100")), 2),
            temperature_range_c=delta_T,
            mean_temperature_c=T_mean,
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # OVERLOAD ASSESSMENT
    # =========================================================================

    def assess_overload_capability(
        self,
        ambient_temperature_c: Union[Decimal, float, str],
        target_overload_percent: Union[Decimal, float, str],
        duration_hours: Union[Decimal, float, str],
        insulation_class: InsulationClass,
        rated_top_oil_rise_c: Union[Decimal, float, str] = "65",
        rated_hot_spot_rise_c: Union[Decimal, float, str] = "80",
        max_hot_spot_limit_c: Optional[Union[Decimal, float, str]] = None
    ) -> OverloadAssessmentResult:
        """
        Assess transformer overload capability.

        Evaluates whether a proposed overload is within acceptable
        limits considering:
        - Hot spot temperature limits
        - Loss of life during overload
        - Equipment design limits

        Args:
            ambient_temperature_c: Ambient temperature
            target_overload_percent: Proposed overload (e.g., 120 for 120%)
            duration_hours: Duration of overload
            insulation_class: Insulation temperature class
            rated_top_oil_rise_c: Top oil rise at rated
            rated_hot_spot_rise_c: Hot spot rise at rated
            max_hot_spot_limit_c: Maximum allowed hot spot (default: class limit + 15C)

        Returns:
            OverloadAssessmentResult

        Reference:
            IEEE C57.91-2011, Clause 8 (Loading beyond nameplate)

        Example:
            >>> calc = ThermalDegradationCalculator()
            >>> result = calc.assess_overload_capability(
            ...     ambient_temperature_c="35",
            ...     target_overload_percent="130",
            ...     duration_hours="2",
            ...     insulation_class=InsulationClass.CLASS_A
            ... )
        """
        builder = ProvenanceBuilder(CalculationType.THERMAL_DEGRADATION)

        # Convert inputs
        T_amb = self._to_decimal(ambient_temperature_c)
        overload_pct = self._to_decimal(target_overload_percent)
        duration = self._to_decimal(duration_hours)
        load_factor = overload_pct / Decimal("100")

        # Get temperature limit
        class_temp = INSULATION_RATINGS[insulation_class]
        if max_hot_spot_limit_c:
            hs_limit = self._to_decimal(max_hot_spot_limit_c)
        else:
            # IEEE allows 15C above rated for short-term overload
            hs_limit = class_temp + Decimal("15")

        builder.add_input("ambient_temperature_c", T_amb)
        builder.add_input("target_overload_percent", overload_pct)
        builder.add_input("duration_hours", duration)
        builder.add_input("insulation_class", insulation_class.name)
        builder.add_input("hot_spot_limit_c", hs_limit)

        # Step 1: Calculate hot spot at overload
        hs_result = self.calculate_hot_spot_temperature(
            ambient_temperature_c=T_amb,
            load_factor=load_factor,
            rated_top_oil_rise_c=rated_top_oil_rise_c,
            rated_hot_spot_rise_c=rated_hot_spot_rise_c
        )
        T_hs = hs_result.hot_spot_temperature_c

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate hot spot at overload",
            inputs={"load_factor": load_factor, "T_amb": T_amb},
            output_name="hot_spot_at_overload",
            output_value=T_hs,
            reference="IEEE C57.91"
        )

        # Step 2: Check against limit
        within_temp_limit = T_hs <= hs_limit

        builder.add_step(
            step_number=2,
            operation="compare",
            description="Check hot spot against limit",
            inputs={"T_hs": T_hs, "limit": hs_limit},
            output_name="within_temp_limit",
            output_value=within_temp_limit
        )

        # Step 3: Calculate life consumed during overload
        # Use rated temperature as reference
        T_ref = class_temp
        Ea = Decimal("1.0")  # Default activation energy

        faa_result = self.calculate_aging_acceleration_factor(T_hs, T_ref, Ea)
        life_consumed = faa_result.acceleration_factor * duration

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate life consumed during overload",
            inputs={"faa": faa_result.acceleration_factor, "duration": duration},
            output_name="life_consumed_hours",
            output_value=life_consumed,
            formula="life_consumed = FAA * duration"
        )

        # Step 4: Determine allowable overload
        # Maximum overload that keeps hot spot at limit
        # Solve: T_hs_limit = T_amb + rises at K
        # This is iterative, we'll approximate
        if T_hs <= hs_limit:
            allowable_overload = overload_pct
            is_within_limits = True
            limiting_factor = "None - within all limits"
        else:
            # Calculate maximum allowable
            # Approximate by scaling load factor
            temp_margin = hs_limit - T_amb
            rated_rise = self._to_decimal(rated_hot_spot_rise_c)
            max_K = self._power(temp_margin / rated_rise, Decimal("0.625"))  # Approximate
            allowable_overload = max_K * Decimal("100")
            is_within_limits = False
            limiting_factor = f"Hot spot temperature ({T_hs}C > {hs_limit}C limit)"

        builder.add_step(
            step_number=4,
            operation="assess",
            description="Determine allowable overload",
            inputs={"T_hs": T_hs, "hs_limit": hs_limit},
            output_name="allowable_overload",
            output_value=allowable_overload
        )

        builder.add_output("is_within_limits", is_within_limits)
        builder.add_output("allowable_overload_percent", allowable_overload)
        builder.add_output("life_consumed_hours", life_consumed)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return OverloadAssessmentResult(
            allowable_overload_percent=self._apply_precision(allowable_overload, 1),
            duration_hours=duration,
            life_consumed_hours=self._apply_precision(life_consumed, 2),
            hot_spot_at_overload_c=T_hs,
            is_within_limits=is_within_limits,
            limiting_factor=limiting_factor,
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # CUMULATIVE AGING WITH LOAD PROFILE
    # =========================================================================

    def calculate_cumulative_aging(
        self,
        load_profile: List[Dict[str, Union[Decimal, float, str]]],
        ambient_temperature_c: Union[Decimal, float, str],
        insulation_class: InsulationClass,
        rated_top_oil_rise_c: Union[Decimal, float, str] = "65",
        rated_hot_spot_rise_c: Union[Decimal, float, str] = "80"
    ) -> ThermalLifeResult:
        """
        Calculate cumulative aging from a varying load profile.

        Integrates aging over time for a load profile with varying
        loads and durations.

        Args:
            load_profile: List of dicts with "load_factor" and "duration_hours"
            ambient_temperature_c: Ambient temperature
            insulation_class: Insulation class
            rated_top_oil_rise_c: Top oil rise at rated
            rated_hot_spot_rise_c: Hot spot rise at rated

        Returns:
            ThermalLifeResult with cumulative aging

        Example:
            >>> calc = ThermalDegradationCalculator()
            >>> profile = [
            ...     {"load_factor": 0.8, "duration_hours": 8},
            ...     {"load_factor": 1.0, "duration_hours": 8},
            ...     {"load_factor": 0.5, "duration_hours": 8},
            ... ]
            >>> result = calc.calculate_cumulative_aging(
            ...     load_profile=profile,
            ...     ambient_temperature_c="25",
            ...     insulation_class=InsulationClass.CLASS_A
            ... )
        """
        builder = ProvenanceBuilder(CalculationType.THERMAL_DEGRADATION)

        T_amb = self._to_decimal(ambient_temperature_c)
        T_ref = INSULATION_RATINGS[insulation_class]
        Ea = Decimal("1.0")

        builder.add_input("num_load_segments", len(load_profile))
        builder.add_input("ambient_temperature_c", T_amb)
        builder.add_input("insulation_class", insulation_class.name)

        total_operating_hours = Decimal("0")
        total_equivalent_hours = Decimal("0")
        weighted_hot_spot = Decimal("0")

        for i, segment in enumerate(load_profile):
            K = self._to_decimal(segment["load_factor"])
            duration = self._to_decimal(segment["duration_hours"])

            # Calculate hot spot for this segment
            hs_result = self.calculate_hot_spot_temperature(
                T_amb, K, rated_top_oil_rise_c, rated_hot_spot_rise_c
            )
            T_hs = hs_result.hot_spot_temperature_c

            # Calculate aging for this segment
            faa_result = self.calculate_aging_acceleration_factor(T_hs, T_ref, Ea)
            segment_equivalent = faa_result.acceleration_factor * duration

            total_operating_hours += duration
            total_equivalent_hours += segment_equivalent
            weighted_hot_spot += T_hs * duration

            builder.add_step(
                step_number=i+1,
                operation="calculate",
                description=f"Calculate aging for segment {i+1}",
                inputs={"load_factor": K, "duration": duration, "T_hs": T_hs},
                output_name=f"segment_{i+1}_equivalent",
                output_value=segment_equivalent
            )

        # Calculate average hot spot
        if total_operating_hours > Decimal("0"):
            avg_hot_spot = weighted_hot_spot / total_operating_hours
        else:
            avg_hot_spot = T_ref

        # Calculate overall aging factor
        if total_operating_hours > Decimal("0"):
            overall_faa = total_equivalent_hours / total_operating_hours
        else:
            overall_faa = Decimal("1")

        # Calculate remaining life
        L_ref = IEEE_REFERENCE_LIFE_HOURS
        remaining_life = L_ref - total_equivalent_hours
        remaining_life = max(Decimal("0"), remaining_life)
        remaining_years = remaining_life / Decimal("8760")

        life_consumed = (total_equivalent_hours / L_ref) * Decimal("100")
        life_consumed = min(life_consumed, Decimal("100"))

        builder.add_output("total_operating_hours", total_operating_hours)
        builder.add_output("total_equivalent_hours", total_equivalent_hours)
        builder.add_output("remaining_life_hours", remaining_life)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return ThermalLifeResult(
            remaining_life_hours=self._apply_precision(remaining_life, 0),
            remaining_life_years=self._apply_precision(remaining_years, 2),
            aging_acceleration_factor=self._apply_precision(overall_faa, 4),
            equivalent_aging_hours=self._apply_precision(total_equivalent_hours, 0),
            life_consumed_percent=self._apply_precision(life_consumed, 2),
            hot_spot_temperature_c=self._apply_precision(avg_hot_spot, 1),
            reference_temperature_c=T_ref,
            insulation_class=insulation_class.name,
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _to_decimal(self, value: Union[Decimal, float, int, str]) -> Decimal:
        """Convert value to Decimal."""
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except InvalidOperation as e:
            raise ValueError(f"Cannot convert {value} to Decimal: {e}")

    def _apply_precision(
        self,
        value: Decimal,
        precision: Optional[int] = None
    ) -> Decimal:
        """Apply precision rounding."""
        if value == Decimal("Infinity"):
            return value
        prec = precision if precision is not None else self._precision
        if prec == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * prec
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _exp(self, x: Decimal) -> Decimal:
        """Calculate e^x."""
        if x == Decimal("0"):
            return Decimal("1")
        if x < Decimal("-700"):
            return Decimal("0")
        if x > Decimal("700"):
            raise ValueError("Exponent too large")
        return Decimal(str(math.exp(float(x))))

    def _power(self, base: Decimal, exponent: Decimal) -> Decimal:
        """Calculate base^exponent."""
        if base == Decimal("0"):
            return Decimal("0") if exponent > Decimal("0") else Decimal("1")
        if exponent == Decimal("0"):
            return Decimal("1")
        result = Decimal(str(math.pow(float(base), float(exponent))))
        return result

    def get_insulation_classes(self) -> List[str]:
        """Get list of available insulation classes."""
        return [ic.name for ic in InsulationClass]

    def get_insulation_rating(self, insulation_class: InsulationClass) -> Decimal:
        """Get temperature rating for insulation class."""
        return INSULATION_RATINGS[insulation_class]


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "InsulationClass",
    "LoadingType",
    "EquipmentCategory",

    # Constants
    "INSULATION_RATINGS",
    "IEEE_REFERENCE_LIFE_HOURS",

    # Data classes
    "ThermalLifeResult",
    "AgingAccelerationResult",
    "HotSpotResult",
    "ThermalCycleResult",
    "OverloadAssessmentResult",

    # Main class
    "ThermalDegradationCalculator",
]
