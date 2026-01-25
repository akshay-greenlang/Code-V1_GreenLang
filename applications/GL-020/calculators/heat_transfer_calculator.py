"""
GL-020 ECONOPULSE: Heat Transfer Calculator

Zero-hallucination heat transfer calculations for economizer performance
monitoring based on ASME PTC 4.3 standards.

This module provides:
- Log Mean Temperature Difference (LMTD) for counter-flow and parallel-flow
- Overall heat transfer coefficient (U-value)
- Heat duty calculations for water and gas sides
- Approach and terminal temperature differences
- Number of Transfer Units (NTU) and effectiveness (epsilon-NTU method)

All calculations are deterministic with complete provenance tracking.

Author: GL-CalculatorEngineer
Standard: ASME PTC 4.3 (Air Heaters)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, Optional, Tuple, Union

from .provenance import (
    ProvenanceTracker,
    CalculationType,
    CalculationProvenance,
    generate_calculation_hash
)
from .thermal_properties import (
    get_water_cp,
    get_flue_gas_cp,
    ValidationError,
    validate_temperature_fahrenheit,
    DEFAULT_FLUE_GAS_COMPOSITION
)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class FlowArrangement(Enum):
    """Heat exchanger flow arrangement."""
    COUNTER_FLOW = "counter_flow"
    PARALLEL_FLOW = "parallel_flow"
    CROSS_FLOW_UNMIXED = "cross_flow_unmixed"
    CROSS_FLOW_MIXED = "cross_flow_mixed"


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_positive(value: float, param_name: str) -> None:
    """
    Validate that a value is positive.

    Args:
        value: Value to validate
        param_name: Parameter name for error messages

    Raises:
        ValidationError: If value is not positive
    """
    if value <= 0:
        raise ValidationError(
            parameter=param_name,
            value=value,
            message=f"{param_name} ({value}) must be positive"
        )


def validate_non_negative(value: float, param_name: str) -> None:
    """
    Validate that a value is non-negative.

    Args:
        value: Value to validate
        param_name: Parameter name for error messages

    Raises:
        ValidationError: If value is negative
    """
    if value < 0:
        raise ValidationError(
            parameter=param_name,
            value=value,
            message=f"{param_name} ({value}) must be non-negative"
        )


def validate_heat_exchanger_temps(
    T_hot_in: float,
    T_hot_out: float,
    T_cold_in: float,
    T_cold_out: float,
    flow_arrangement: FlowArrangement
) -> None:
    """
    Validate heat exchanger temperatures are thermodynamically valid.

    Args:
        T_hot_in: Hot fluid inlet temperature
        T_hot_out: Hot fluid outlet temperature
        T_cold_in: Cold fluid inlet temperature
        T_cold_out: Cold fluid outlet temperature
        flow_arrangement: Flow configuration

    Raises:
        ValidationError: If temperatures violate thermodynamic constraints
    """
    # Hot fluid must cool down
    if T_hot_out > T_hot_in:
        raise ValidationError(
            parameter="T_hot_out",
            value=T_hot_out,
            message=f"Hot outlet ({T_hot_out}) cannot exceed hot inlet ({T_hot_in})"
        )

    # Cold fluid must heat up
    if T_cold_out < T_cold_in:
        raise ValidationError(
            parameter="T_cold_out",
            value=T_cold_out,
            message=f"Cold outlet ({T_cold_out}) cannot be less than cold inlet ({T_cold_in})"
        )

    # Check for temperature cross
    if flow_arrangement == FlowArrangement.PARALLEL_FLOW:
        # In parallel flow, outlet temperatures cannot cross
        if T_cold_out > T_hot_out:
            raise ValidationError(
                parameter="T_cold_out",
                value=T_cold_out,
                message=f"Temperature cross in parallel flow: cold outlet ({T_cold_out}) > "
                        f"hot outlet ({T_hot_out})"
            )


# =============================================================================
# LMTD CALCULATIONS
# =============================================================================

def calculate_lmtd(
    T_hot_in: float,
    T_hot_out: float,
    T_cold_in: float,
    T_cold_out: float,
    flow_arrangement: FlowArrangement = FlowArrangement.COUNTER_FLOW,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate Log Mean Temperature Difference (LMTD).

    The LMTD is the logarithmic average of temperature differences
    at each end of the heat exchanger, accounting for the exponential
    temperature profile along the exchanger length.

    Methodology (ASME PTC 4.3):
        For counter-flow:
            delta_T1 = T_hot_in - T_cold_out
            delta_T2 = T_hot_out - T_cold_in

        For parallel-flow:
            delta_T1 = T_hot_in - T_cold_in
            delta_T2 = T_hot_out - T_cold_out

        LMTD = (delta_T1 - delta_T2) / ln(delta_T1 / delta_T2)

        Special case when delta_T1 = delta_T2:
            LMTD = delta_T1 (arithmetic mean)

    Reference: ASME PTC 4.3, Incropera & DeWitt

    Args:
        T_hot_in: Hot fluid inlet temperature (F)
        T_hot_out: Hot fluid outlet temperature (F)
        T_cold_in: Cold fluid inlet temperature (F)
        T_cold_out: Cold fluid outlet temperature (F)
        flow_arrangement: Counter-flow or parallel-flow
        track_provenance: If True, return provenance record

    Returns:
        LMTD in degrees Fahrenheit, optionally with provenance

    Raises:
        ValidationError: If temperatures are invalid
        ValueError: If temperature differences result in invalid logarithm
    """
    # Validate temperatures
    validate_temperature_fahrenheit(T_hot_in, "T_hot_in")
    validate_temperature_fahrenheit(T_hot_out, "T_hot_out")
    validate_temperature_fahrenheit(T_cold_in, "T_cold_in")
    validate_temperature_fahrenheit(T_cold_out, "T_cold_out")
    validate_heat_exchanger_temps(T_hot_in, T_hot_out, T_cold_in, T_cold_out, flow_arrangement)

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.HEAT_TRANSFER,
            formula_id=f"lmtd_{flow_arrangement.value}",
            formula_version="1.0.0",
            inputs={
                "T_hot_in": T_hot_in,
                "T_hot_out": T_hot_out,
                "T_cold_in": T_cold_in,
                "T_cold_out": T_cold_out,
                "flow_arrangement": flow_arrangement.value
            }
        )

    # Calculate temperature differences based on flow arrangement
    if flow_arrangement == FlowArrangement.COUNTER_FLOW:
        delta_T1 = T_hot_in - T_cold_out
        delta_T2 = T_hot_out - T_cold_in
    else:  # Parallel flow
        delta_T1 = T_hot_in - T_cold_in
        delta_T2 = T_hot_out - T_cold_out

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate temperature difference at end 1",
            inputs={"T_hot": T_hot_in if flow_arrangement == FlowArrangement.COUNTER_FLOW else T_hot_in,
                    "T_cold": T_cold_out if flow_arrangement == FlowArrangement.COUNTER_FLOW else T_cold_in},
            output_name="delta_T1",
            output_value=delta_T1,
            formula="delta_T1 = T_hot - T_cold"
        )
        tracker.add_step(
            operation="subtract",
            description="Calculate temperature difference at end 2",
            inputs={"T_hot": T_hot_out,
                    "T_cold": T_cold_in if flow_arrangement == FlowArrangement.COUNTER_FLOW else T_cold_out},
            output_name="delta_T2",
            output_value=delta_T2,
            formula="delta_T2 = T_hot - T_cold"
        )

    # Validate temperature differences
    if delta_T1 <= 0 or delta_T2 <= 0:
        raise ValueError(
            f"Invalid temperature differences: delta_T1={delta_T1}, delta_T2={delta_T2}. "
            f"Both must be positive for heat transfer to occur."
        )

    # Calculate LMTD
    # Handle special case when delta_T1 ~ delta_T2 (avoid division by zero)
    if abs(delta_T1 - delta_T2) < 0.001:
        # Use arithmetic mean when temperatures are essentially equal
        lmtd = (delta_T1 + delta_T2) / 2

        if tracker:
            tracker.add_step(
                operation="arithmetic_mean",
                description="Calculate LMTD (special case: delta_T1 ≈ delta_T2)",
                inputs={"delta_T1": delta_T1, "delta_T2": delta_T2},
                output_name="lmtd",
                output_value=lmtd,
                formula="LMTD = (delta_T1 + delta_T2) / 2"
            )
    else:
        # Standard LMTD formula
        ratio = delta_T1 / delta_T2
        lmtd = (delta_T1 - delta_T2) / math.log(ratio)

        if tracker:
            tracker.add_step(
                operation="lmtd_formula",
                description="Calculate LMTD using logarithmic mean formula",
                inputs={"delta_T1": delta_T1, "delta_T2": delta_T2, "ratio": ratio},
                output_name="lmtd",
                output_value=lmtd,
                formula="LMTD = (delta_T1 - delta_T2) / ln(delta_T1 / delta_T2)"
            )

    lmtd_rounded = round(lmtd, 4)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=lmtd_rounded,
            output_unit="degF",
            precision=4
        )
        return lmtd_rounded, provenance

    return lmtd_rounded


# =============================================================================
# OVERALL HEAT TRANSFER COEFFICIENT
# =============================================================================

def calculate_overall_heat_transfer_coefficient(
    heat_duty: float,
    heat_transfer_area: float,
    lmtd: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate overall heat transfer coefficient (U-value) from heat duty.

    The U-value represents the total thermal resistance for heat transfer
    from hot fluid to cold fluid, including convection on both sides
    and conduction through tube walls.

    Methodology (ASME PTC 4.3):
        From the basic heat exchanger equation:
        Q = U * A * LMTD

        Therefore:
        U = Q / (A * LMTD)

    Reference: ASME PTC 4.3

    Args:
        heat_duty: Heat transfer rate (BTU/hr)
        heat_transfer_area: Total heat transfer surface area (ft2)
        lmtd: Log Mean Temperature Difference (degF)
        track_provenance: If True, return provenance record

    Returns:
        U-value in BTU/(hr-ft2-F), optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
        ZeroDivisionError: If area or LMTD is zero
    """
    validate_positive(heat_duty, "heat_duty")
    validate_positive(heat_transfer_area, "heat_transfer_area")
    validate_positive(lmtd, "lmtd")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.HEAT_TRANSFER,
            formula_id="overall_u_value",
            formula_version="1.0.0",
            inputs={
                "heat_duty": heat_duty,
                "heat_transfer_area": heat_transfer_area,
                "lmtd": lmtd
            }
        )

    # Calculate U-value
    U = heat_duty / (heat_transfer_area * lmtd)

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate U-value from Q = U * A * LMTD",
            inputs={"Q": heat_duty, "A": heat_transfer_area, "LMTD": lmtd},
            output_name="U",
            output_value=U,
            formula="U = Q / (A * LMTD)"
        )

    U_rounded = round(U, 4)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=U_rounded,
            output_unit="BTU/(hr-ft2-F)",
            precision=4
        )
        return U_rounded, provenance

    return U_rounded


def calculate_theoretical_u_value(
    h_hot: float,
    h_cold: float,
    tube_thickness: float,
    tube_thermal_conductivity: float,
    fouling_factor_hot: float = 0.0,
    fouling_factor_cold: float = 0.0,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate theoretical overall heat transfer coefficient from component resistances.

    Methodology:
        Total thermal resistance:
        1/U = 1/h_hot + Rf_hot + t/k + Rf_cold + 1/h_cold

        Where:
        - h_hot: Hot side convective heat transfer coefficient
        - h_cold: Cold side convective heat transfer coefficient
        - t: Tube wall thickness
        - k: Tube thermal conductivity
        - Rf: Fouling resistance

    Reference: ASME PTC 4.3, Incropera & DeWitt

    Args:
        h_hot: Hot side heat transfer coefficient (BTU/(hr-ft2-F))
        h_cold: Cold side heat transfer coefficient (BTU/(hr-ft2-F))
        tube_thickness: Tube wall thickness (ft)
        tube_thermal_conductivity: Tube material conductivity (BTU/(hr-ft-F))
        fouling_factor_hot: Hot side fouling resistance ((hr-ft2-F)/BTU)
        fouling_factor_cold: Cold side fouling resistance ((hr-ft2-F)/BTU)
        track_provenance: If True, return provenance record

    Returns:
        U-value in BTU/(hr-ft2-F), optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_positive(h_hot, "h_hot")
    validate_positive(h_cold, "h_cold")
    validate_positive(tube_thickness, "tube_thickness")
    validate_positive(tube_thermal_conductivity, "tube_thermal_conductivity")
    validate_non_negative(fouling_factor_hot, "fouling_factor_hot")
    validate_non_negative(fouling_factor_cold, "fouling_factor_cold")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.HEAT_TRANSFER,
            formula_id="theoretical_u_value",
            formula_version="1.0.0",
            inputs={
                "h_hot": h_hot,
                "h_cold": h_cold,
                "tube_thickness": tube_thickness,
                "tube_thermal_conductivity": tube_thermal_conductivity,
                "fouling_factor_hot": fouling_factor_hot,
                "fouling_factor_cold": fouling_factor_cold
            }
        )

    # Calculate individual resistances
    R_hot = 1 / h_hot
    R_cold = 1 / h_cold
    R_wall = tube_thickness / tube_thermal_conductivity

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate hot side convective resistance",
            inputs={"h_hot": h_hot},
            output_name="R_hot",
            output_value=R_hot,
            formula="R_hot = 1 / h_hot"
        )
        tracker.add_step(
            operation="divide",
            description="Calculate cold side convective resistance",
            inputs={"h_cold": h_cold},
            output_name="R_cold",
            output_value=R_cold,
            formula="R_cold = 1 / h_cold"
        )
        tracker.add_step(
            operation="divide",
            description="Calculate wall conductive resistance",
            inputs={"t": tube_thickness, "k": tube_thermal_conductivity},
            output_name="R_wall",
            output_value=R_wall,
            formula="R_wall = t / k"
        )

    # Total resistance
    R_total = R_hot + fouling_factor_hot + R_wall + fouling_factor_cold + R_cold

    if tracker:
        tracker.add_step(
            operation="add",
            description="Sum all thermal resistances",
            inputs={
                "R_hot": R_hot,
                "Rf_hot": fouling_factor_hot,
                "R_wall": R_wall,
                "Rf_cold": fouling_factor_cold,
                "R_cold": R_cold
            },
            output_name="R_total",
            output_value=R_total,
            formula="R_total = R_hot + Rf_hot + R_wall + Rf_cold + R_cold"
        )

    # U-value is inverse of total resistance
    U = 1 / R_total

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate U-value as inverse of total resistance",
            inputs={"R_total": R_total},
            output_name="U",
            output_value=U,
            formula="U = 1 / R_total"
        )

    U_rounded = round(U, 4)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=U_rounded,
            output_unit="BTU/(hr-ft2-F)",
            precision=4
        )
        return U_rounded, provenance

    return U_rounded


# =============================================================================
# HEAT DUTY CALCULATIONS
# =============================================================================

def calculate_heat_duty(
    mass_flow_rate: float,
    specific_heat: float,
    temp_in: float,
    temp_out: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate heat duty from mass flow rate and temperature change.

    Methodology (ASME PTC 4.3):
        Q = m_dot * Cp * (T_out - T_in)

        For heating (economizer): T_out > T_in, Q > 0
        For cooling (flue gas): T_out < T_in, Q < 0 (heat released)

    Reference: ASME PTC 4.3

    Args:
        mass_flow_rate: Mass flow rate (lbm/hr)
        specific_heat: Specific heat capacity (BTU/(lbm-F))
        temp_in: Inlet temperature (F)
        temp_out: Outlet temperature (F)
        track_provenance: If True, return provenance record

    Returns:
        Heat duty in BTU/hr (positive for heat absorbed), optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_positive(mass_flow_rate, "mass_flow_rate")
    validate_positive(specific_heat, "specific_heat")
    validate_temperature_fahrenheit(temp_in, "temp_in")
    validate_temperature_fahrenheit(temp_out, "temp_out")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.HEAT_TRANSFER,
            formula_id="heat_duty",
            formula_version="1.0.0",
            inputs={
                "mass_flow_rate": mass_flow_rate,
                "specific_heat": specific_heat,
                "temp_in": temp_in,
                "temp_out": temp_out
            }
        )

    # Calculate temperature change
    delta_T = temp_out - temp_in

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate temperature change",
            inputs={"temp_out": temp_out, "temp_in": temp_in},
            output_name="delta_T",
            output_value=delta_T,
            formula="delta_T = T_out - T_in"
        )

    # Calculate heat duty
    Q = mass_flow_rate * specific_heat * delta_T

    if tracker:
        tracker.add_step(
            operation="multiply",
            description="Calculate heat duty",
            inputs={"m_dot": mass_flow_rate, "Cp": specific_heat, "delta_T": delta_T},
            output_name="Q",
            output_value=Q,
            formula="Q = m_dot * Cp * delta_T"
        )

    Q_rounded = round(Q, 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=Q_rounded,
            output_unit="BTU/hr",
            precision=2
        )
        return Q_rounded, provenance

    return Q_rounded


def calculate_water_side_heat_duty(
    water_flow_rate: float,
    water_temp_in: float,
    water_temp_out: float,
    pressure_psia: float = 200.0,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate heat duty on the water side of the economizer.

    Uses temperature-dependent water properties from IAPWS-IF97.

    Methodology:
        Q_water = m_water * Cp_water * (T_water_out - T_water_in)

        where Cp_water is evaluated at the average temperature.

    Reference: ASME PTC 4.3, IAPWS-IF97

    Args:
        water_flow_rate: Water mass flow rate (lbm/hr)
        water_temp_in: Water inlet temperature (F)
        water_temp_out: Water outlet temperature (F)
        pressure_psia: Water pressure (psia)
        track_provenance: If True, return provenance record

    Returns:
        Heat duty in BTU/hr, optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_positive(water_flow_rate, "water_flow_rate")
    validate_temperature_fahrenheit(water_temp_in, "water_temp_in")
    validate_temperature_fahrenheit(water_temp_out, "water_temp_out")

    if water_temp_out < water_temp_in:
        raise ValidationError(
            parameter="water_temp_out",
            value=water_temp_out,
            message=f"Water outlet ({water_temp_out}F) cannot be less than inlet ({water_temp_in}F)"
        )

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.HEAT_TRANSFER,
            formula_id="water_side_heat_duty",
            formula_version="1.0.0",
            inputs={
                "water_flow_rate": water_flow_rate,
                "water_temp_in": water_temp_in,
                "water_temp_out": water_temp_out,
                "pressure_psia": pressure_psia
            }
        )

    # Calculate average temperature for property evaluation
    T_avg = (water_temp_in + water_temp_out) / 2

    if tracker:
        tracker.add_step(
            operation="average",
            description="Calculate average water temperature",
            inputs={"T_in": water_temp_in, "T_out": water_temp_out},
            output_name="T_avg",
            output_value=T_avg,
            formula="T_avg = (T_in + T_out) / 2"
        )

    # Get water specific heat at average temperature
    Cp_water = get_water_cp(T_avg, pressure_psia)

    if tracker:
        tracker.add_step(
            operation="lookup",
            description="Get water specific heat at average temperature (IAPWS-IF97)",
            inputs={"T_avg": T_avg, "P": pressure_psia},
            output_name="Cp_water",
            output_value=Cp_water,
            formula="Cp_water = f(T_avg, P) from IAPWS-IF97"
        )

    # Calculate heat duty
    delta_T = water_temp_out - water_temp_in
    Q = water_flow_rate * Cp_water * delta_T

    if tracker:
        tracker.add_step(
            operation="multiply",
            description="Calculate water side heat duty",
            inputs={"m_dot": water_flow_rate, "Cp": Cp_water, "delta_T": delta_T},
            output_name="Q_water",
            output_value=Q,
            formula="Q_water = m_dot * Cp_water * (T_out - T_in)"
        )

    Q_rounded = round(Q, 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=Q_rounded,
            output_unit="BTU/hr",
            precision=2
        )
        return Q_rounded, provenance

    return Q_rounded


def calculate_gas_side_heat_duty(
    gas_flow_rate: float,
    gas_temp_in: float,
    gas_temp_out: float,
    gas_composition: Optional[Dict[str, float]] = None,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate heat duty on the gas (flue gas) side of the economizer.

    Uses composition-dependent gas properties from JANAF tables.

    Methodology:
        Q_gas = m_gas * Cp_gas * (T_gas_in - T_gas_out)

        Note: Sign convention - heat released by gas is positive.

    Reference: ASME PTC 4.3, JANAF Tables

    Args:
        gas_flow_rate: Gas mass flow rate (lbm/hr)
        gas_temp_in: Gas inlet temperature (F)
        gas_temp_out: Gas outlet temperature (F)
        gas_composition: Flue gas composition (mass fractions)
        track_provenance: If True, return provenance record

    Returns:
        Heat duty in BTU/hr (positive), optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_positive(gas_flow_rate, "gas_flow_rate")
    validate_temperature_fahrenheit(gas_temp_in, "gas_temp_in")
    validate_temperature_fahrenheit(gas_temp_out, "gas_temp_out")

    if gas_temp_out > gas_temp_in:
        raise ValidationError(
            parameter="gas_temp_out",
            value=gas_temp_out,
            message=f"Gas outlet ({gas_temp_out}F) cannot exceed inlet ({gas_temp_in}F)"
        )

    if gas_composition is None:
        gas_composition = DEFAULT_FLUE_GAS_COMPOSITION

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.HEAT_TRANSFER,
            formula_id="gas_side_heat_duty",
            formula_version="1.0.0",
            inputs={
                "gas_flow_rate": gas_flow_rate,
                "gas_temp_in": gas_temp_in,
                "gas_temp_out": gas_temp_out,
                "gas_composition": gas_composition
            }
        )

    # Calculate average temperature for property evaluation
    T_avg = (gas_temp_in + gas_temp_out) / 2

    if tracker:
        tracker.add_step(
            operation="average",
            description="Calculate average gas temperature",
            inputs={"T_in": gas_temp_in, "T_out": gas_temp_out},
            output_name="T_avg",
            output_value=T_avg,
            formula="T_avg = (T_in + T_out) / 2"
        )

    # Get gas specific heat at average temperature
    Cp_gas = get_flue_gas_cp(T_avg, gas_composition)

    if tracker:
        tracker.add_step(
            operation="lookup",
            description="Get flue gas specific heat (JANAF)",
            inputs={"T_avg": T_avg, "composition": gas_composition},
            output_name="Cp_gas",
            output_value=Cp_gas,
            formula="Cp_gas = sum(y_i * Cp_i) from JANAF"
        )

    # Calculate heat duty (positive for heat released)
    delta_T = gas_temp_in - gas_temp_out
    Q = gas_flow_rate * Cp_gas * delta_T

    if tracker:
        tracker.add_step(
            operation="multiply",
            description="Calculate gas side heat duty (heat released)",
            inputs={"m_dot": gas_flow_rate, "Cp": Cp_gas, "delta_T": delta_T},
            output_name="Q_gas",
            output_value=Q,
            formula="Q_gas = m_dot * Cp_gas * (T_in - T_out)"
        )

    Q_rounded = round(Q, 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=Q_rounded,
            output_unit="BTU/hr",
            precision=2
        )
        return Q_rounded, provenance

    return Q_rounded


# =============================================================================
# TEMPERATURE DIFFERENCES
# =============================================================================

def calculate_approach_temperature(
    gas_temp_out: float,
    water_temp_in: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate approach temperature for the economizer.

    The approach temperature is the difference between the gas outlet
    temperature and water inlet temperature. Lower approach indicates
    better heat recovery.

    Methodology (ASME PTC 4.3):
        Approach = T_gas_out - T_water_in

    Reference: ASME PTC 4.3

    Args:
        gas_temp_out: Gas outlet temperature (F)
        water_temp_in: Water inlet temperature (F)
        track_provenance: If True, return provenance record

    Returns:
        Approach temperature in degrees F, optionally with provenance

    Raises:
        ValidationError: If temperatures are invalid
    """
    validate_temperature_fahrenheit(gas_temp_out, "gas_temp_out")
    validate_temperature_fahrenheit(water_temp_in, "water_temp_in")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.HEAT_TRANSFER,
            formula_id="approach_temperature",
            formula_version="1.0.0",
            inputs={
                "gas_temp_out": gas_temp_out,
                "water_temp_in": water_temp_in
            }
        )

    approach = gas_temp_out - water_temp_in

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate approach temperature",
            inputs={"T_gas_out": gas_temp_out, "T_water_in": water_temp_in},
            output_name="approach",
            output_value=approach,
            formula="Approach = T_gas_out - T_water_in"
        )

    approach_rounded = round(approach, 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=approach_rounded,
            output_unit="degF",
            precision=2
        )
        return approach_rounded, provenance

    return approach_rounded


def calculate_terminal_temperature_difference(
    gas_temp_in: float,
    water_temp_out: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate Terminal Temperature Difference (TTD).

    The TTD is the temperature difference between gas inlet and
    water outlet. It indicates the maximum possible heat recovery.

    Methodology (ASME PTC 4.3):
        TTD = T_gas_in - T_water_out

    Reference: ASME PTC 4.3

    Args:
        gas_temp_in: Gas inlet temperature (F)
        water_temp_out: Water outlet temperature (F)
        track_provenance: If True, return provenance record

    Returns:
        TTD in degrees F, optionally with provenance

    Raises:
        ValidationError: If temperatures are invalid
    """
    validate_temperature_fahrenheit(gas_temp_in, "gas_temp_in")
    validate_temperature_fahrenheit(water_temp_out, "water_temp_out")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.HEAT_TRANSFER,
            formula_id="terminal_temperature_difference",
            formula_version="1.0.0",
            inputs={
                "gas_temp_in": gas_temp_in,
                "water_temp_out": water_temp_out
            }
        )

    ttd = gas_temp_in - water_temp_out

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate Terminal Temperature Difference",
            inputs={"T_gas_in": gas_temp_in, "T_water_out": water_temp_out},
            output_name="TTD",
            output_value=ttd,
            formula="TTD = T_gas_in - T_water_out"
        )

    ttd_rounded = round(ttd, 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=ttd_rounded,
            output_unit="degF",
            precision=2
        )
        return ttd_rounded, provenance

    return ttd_rounded


# =============================================================================
# NTU-EFFECTIVENESS METHOD
# =============================================================================

def calculate_ntu(
    U: float,
    A: float,
    C_min: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate Number of Transfer Units (NTU).

    NTU is a dimensionless parameter that characterizes the size
    of the heat exchanger relative to its heat capacity rate.

    Methodology:
        NTU = U * A / C_min

        Where:
        - U: Overall heat transfer coefficient
        - A: Heat transfer area
        - C_min: Minimum heat capacity rate = min(m_hot*Cp_hot, m_cold*Cp_cold)

    Reference: ASME PTC 4.3, Kays & London

    Args:
        U: Overall heat transfer coefficient (BTU/(hr-ft2-F))
        A: Heat transfer area (ft2)
        C_min: Minimum heat capacity rate (BTU/(hr-F))
        track_provenance: If True, return provenance record

    Returns:
        NTU (dimensionless), optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_positive(U, "U")
    validate_positive(A, "A")
    validate_positive(C_min, "C_min")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.HEAT_TRANSFER,
            formula_id="ntu",
            formula_version="1.0.0",
            inputs={
                "U": U,
                "A": A,
                "C_min": C_min
            }
        )

    ntu = (U * A) / C_min

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate NTU",
            inputs={"U": U, "A": A, "C_min": C_min},
            output_name="NTU",
            output_value=ntu,
            formula="NTU = U * A / C_min"
        )

    ntu_rounded = round(ntu, 4)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=ntu_rounded,
            output_unit="dimensionless",
            precision=4
        )
        return ntu_rounded, provenance

    return ntu_rounded


def calculate_heat_capacity_ratio(
    C_min: float,
    C_max: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate heat capacity ratio (Cr).

    The heat capacity ratio is a key parameter in the epsilon-NTU method.

    Methodology:
        Cr = C_min / C_max

        Where C = m_dot * Cp

        Note: Cr is always <= 1 by definition.

    Reference: Kays & London

    Args:
        C_min: Minimum heat capacity rate (BTU/(hr-F))
        C_max: Maximum heat capacity rate (BTU/(hr-F))
        track_provenance: If True, return provenance record

    Returns:
        Heat capacity ratio (0 to 1), optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_positive(C_min, "C_min")
    validate_positive(C_max, "C_max")

    if C_min > C_max:
        raise ValidationError(
            parameter="C_min",
            value=C_min,
            message=f"C_min ({C_min}) cannot exceed C_max ({C_max})"
        )

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.HEAT_TRANSFER,
            formula_id="heat_capacity_ratio",
            formula_version="1.0.0",
            inputs={
                "C_min": C_min,
                "C_max": C_max
            }
        )

    Cr = C_min / C_max

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate heat capacity ratio",
            inputs={"C_min": C_min, "C_max": C_max},
            output_name="Cr",
            output_value=Cr,
            formula="Cr = C_min / C_max"
        )

    Cr_rounded = round(Cr, 4)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=Cr_rounded,
            output_unit="dimensionless",
            precision=4
        )
        return Cr_rounded, provenance

    return Cr_rounded


def calculate_effectiveness(
    NTU: float,
    Cr: float,
    flow_arrangement: FlowArrangement = FlowArrangement.COUNTER_FLOW,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate heat exchanger effectiveness using epsilon-NTU method.

    Effectiveness (epsilon) represents the ratio of actual heat transfer
    to the maximum possible heat transfer.

    Methodology:
        For counter-flow (Cr < 1):
            epsilon = [1 - exp(-NTU*(1-Cr))] / [1 - Cr*exp(-NTU*(1-Cr))]

        For counter-flow (Cr = 1):
            epsilon = NTU / (1 + NTU)

        For parallel-flow:
            epsilon = [1 - exp(-NTU*(1+Cr))] / (1 + Cr)

    Reference: ASME PTC 4.3, Kays & London

    Args:
        NTU: Number of Transfer Units (dimensionless)
        Cr: Heat capacity ratio (0 to 1)
        flow_arrangement: Counter-flow or parallel-flow
        track_provenance: If True, return provenance record

    Returns:
        Effectiveness (0 to 1), optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_non_negative(NTU, "NTU")
    validate_non_negative(Cr, "Cr")

    if Cr > 1:
        raise ValidationError(
            parameter="Cr",
            value=Cr,
            message=f"Heat capacity ratio ({Cr}) cannot exceed 1.0"
        )

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.HEAT_TRANSFER,
            formula_id=f"effectiveness_{flow_arrangement.value}",
            formula_version="1.0.0",
            inputs={
                "NTU": NTU,
                "Cr": Cr,
                "flow_arrangement": flow_arrangement.value
            }
        )

    if flow_arrangement == FlowArrangement.COUNTER_FLOW:
        if abs(Cr - 1.0) < 1e-6:
            # Special case: Cr = 1
            epsilon = NTU / (1 + NTU)
            formula = "epsilon = NTU / (1 + NTU)"
        else:
            # General case
            exp_term = math.exp(-NTU * (1 - Cr))
            numerator = 1 - exp_term
            denominator = 1 - Cr * exp_term
            epsilon = numerator / denominator
            formula = "epsilon = [1 - exp(-NTU*(1-Cr))] / [1 - Cr*exp(-NTU*(1-Cr))]"

    elif flow_arrangement == FlowArrangement.PARALLEL_FLOW:
        exp_term = math.exp(-NTU * (1 + Cr))
        epsilon = (1 - exp_term) / (1 + Cr)
        formula = "epsilon = [1 - exp(-NTU*(1+Cr))] / (1 + Cr)"

    elif flow_arrangement == FlowArrangement.CROSS_FLOW_UNMIXED:
        # Cross-flow, both fluids unmixed (approximate)
        exp_term1 = math.exp(-NTU)
        exp_term2 = math.exp(-Cr * NTU * exp_term1) - 1
        epsilon = 1 - math.exp(exp_term2 / Cr) if Cr > 0 else 1 - math.exp(-NTU)
        formula = "epsilon (cross-flow unmixed approximation)"

    else:  # CROSS_FLOW_MIXED
        # Cross-flow, one fluid mixed (C_max fluid mixed)
        exp_term = math.exp(-Cr * (1 - math.exp(-NTU)))
        epsilon = (1 - exp_term) / Cr if Cr > 0 else 1 - math.exp(-NTU)
        formula = "epsilon (cross-flow mixed approximation)"

    if tracker:
        tracker.add_step(
            operation="effectiveness_formula",
            description=f"Calculate effectiveness for {flow_arrangement.value}",
            inputs={"NTU": NTU, "Cr": Cr},
            output_name="epsilon",
            output_value=epsilon,
            formula=formula
        )

    # Clamp to valid range [0, 1]
    epsilon = max(0.0, min(1.0, epsilon))

    epsilon_rounded = round(epsilon, 4)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=epsilon_rounded,
            output_unit="dimensionless",
            precision=4
        )
        return epsilon_rounded, provenance

    return epsilon_rounded


def calculate_effectiveness_from_temperatures(
    T_hot_in: float,
    T_hot_out: float,
    T_cold_in: float,
    T_cold_out: float,
    C_hot: float,
    C_cold: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate effectiveness directly from temperatures and heat capacities.

    Methodology:
        epsilon = Q_actual / Q_max

        Q_actual = C_cold * (T_cold_out - T_cold_in)
                 = C_hot * (T_hot_in - T_hot_out)

        Q_max = C_min * (T_hot_in - T_cold_in)

    Reference: ASME PTC 4.3

    Args:
        T_hot_in: Hot fluid inlet temperature (F)
        T_hot_out: Hot fluid outlet temperature (F)
        T_cold_in: Cold fluid inlet temperature (F)
        T_cold_out: Cold fluid outlet temperature (F)
        C_hot: Hot fluid heat capacity rate (BTU/(hr-F))
        C_cold: Cold fluid heat capacity rate (BTU/(hr-F))
        track_provenance: If True, return provenance record

    Returns:
        Effectiveness (0 to 1), optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_temperature_fahrenheit(T_hot_in, "T_hot_in")
    validate_temperature_fahrenheit(T_hot_out, "T_hot_out")
    validate_temperature_fahrenheit(T_cold_in, "T_cold_in")
    validate_temperature_fahrenheit(T_cold_out, "T_cold_out")
    validate_positive(C_hot, "C_hot")
    validate_positive(C_cold, "C_cold")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.HEAT_TRANSFER,
            formula_id="effectiveness_from_temperatures",
            formula_version="1.0.0",
            inputs={
                "T_hot_in": T_hot_in,
                "T_hot_out": T_hot_out,
                "T_cold_in": T_cold_in,
                "T_cold_out": T_cold_out,
                "C_hot": C_hot,
                "C_cold": C_cold
            }
        )

    # Determine C_min
    C_min = min(C_hot, C_cold)

    if tracker:
        tracker.add_step(
            operation="min",
            description="Determine minimum heat capacity rate",
            inputs={"C_hot": C_hot, "C_cold": C_cold},
            output_name="C_min",
            output_value=C_min,
            formula="C_min = min(C_hot, C_cold)"
        )

    # Calculate actual heat transfer (use cold side)
    Q_actual = C_cold * (T_cold_out - T_cold_in)

    if tracker:
        tracker.add_step(
            operation="multiply",
            description="Calculate actual heat transfer",
            inputs={"C_cold": C_cold, "delta_T_cold": T_cold_out - T_cold_in},
            output_name="Q_actual",
            output_value=Q_actual,
            formula="Q_actual = C_cold * (T_cold_out - T_cold_in)"
        )

    # Calculate maximum possible heat transfer
    Q_max = C_min * (T_hot_in - T_cold_in)

    if tracker:
        tracker.add_step(
            operation="multiply",
            description="Calculate maximum possible heat transfer",
            inputs={"C_min": C_min, "delta_T_max": T_hot_in - T_cold_in},
            output_name="Q_max",
            output_value=Q_max,
            formula="Q_max = C_min * (T_hot_in - T_cold_in)"
        )

    # Calculate effectiveness
    if Q_max == 0:
        epsilon = 0.0
    else:
        epsilon = Q_actual / Q_max

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate effectiveness",
            inputs={"Q_actual": Q_actual, "Q_max": Q_max},
            output_name="epsilon",
            output_value=epsilon,
            formula="epsilon = Q_actual / Q_max"
        )

    # Clamp to valid range
    epsilon = max(0.0, min(1.0, epsilon))

    epsilon_rounded = round(epsilon, 4)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=epsilon_rounded,
            output_unit="dimensionless",
            precision=4
        )
        return epsilon_rounded, provenance

    return epsilon_rounded
