"""
GL-020 ECONOPULSE: Economizer Efficiency Calculator

Zero-hallucination efficiency calculations for economizer performance
monitoring based on ASME PTC 4.3 standards.

This module provides:
- Economizer effectiveness calculation
- Heat recovery ratio
- Gas-side and water-side efficiency
- Overall efficiency metrics
- Design deviation analysis
- Performance trending

All calculations are deterministic with complete provenance tracking.

Author: GL-CalculatorEngineer
Standard: ASME PTC 4.3 (Air Heaters)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

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
from .heat_transfer_calculator import (
    calculate_water_side_heat_duty,
    calculate_gas_side_heat_duty,
    calculate_lmtd,
    FlowArrangement,
    validate_positive,
    validate_non_negative
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EconomizerOperatingPoint:
    """Complete operating point data for economizer."""
    # Gas side (required fields)
    gas_temp_in: float      # F
    gas_temp_out: float     # F
    gas_flow_rate: float    # lbm/hr

    # Water side (required fields)
    water_temp_in: float    # F
    water_temp_out: float   # F
    water_flow_rate: float  # lbm/hr

    # Optional fields with defaults (must come after required fields)
    gas_composition: Optional[Dict[str, float]] = None
    water_pressure: float = 200.0  # psia

    # Heat transfer
    heat_transfer_area: float = 0.0  # ft2

    def validate(self) -> None:
        """Validate all operating point parameters."""
        validate_temperature_fahrenheit(self.gas_temp_in, "gas_temp_in")
        validate_temperature_fahrenheit(self.gas_temp_out, "gas_temp_out")
        validate_temperature_fahrenheit(self.water_temp_in, "water_temp_in")
        validate_temperature_fahrenheit(self.water_temp_out, "water_temp_out")
        validate_positive(self.gas_flow_rate, "gas_flow_rate")
        validate_positive(self.water_flow_rate, "water_flow_rate")

        # Thermodynamic validity checks
        if self.gas_temp_out > self.gas_temp_in:
            raise ValidationError(
                parameter="gas_temp_out",
                value=self.gas_temp_out,
                message=f"Gas outlet ({self.gas_temp_out}F) cannot exceed inlet ({self.gas_temp_in}F)"
            )

        if self.water_temp_out < self.water_temp_in:
            raise ValidationError(
                parameter="water_temp_out",
                value=self.water_temp_out,
                message=f"Water outlet ({self.water_temp_out}F) cannot be less than inlet ({self.water_temp_in}F)"
            )


@dataclass
class DesignConditions:
    """Design point specifications for economizer."""
    gas_temp_in_design: float      # F
    gas_temp_out_design: float     # F
    water_temp_in_design: float    # F
    water_temp_out_design: float   # F
    gas_flow_rate_design: float    # lbm/hr
    water_flow_rate_design: float  # lbm/hr
    heat_duty_design: float        # BTU/hr
    U_clean: float                 # BTU/(hr-ft2-F)
    heat_transfer_area: float      # ft2
    effectiveness_design: float = 0.85  # Design effectiveness


@dataclass
class EfficiencyResult:
    """Complete efficiency calculation results."""
    economizer_effectiveness: float
    heat_recovery_ratio: float
    gas_side_efficiency: float
    water_side_efficiency: float
    overall_efficiency: float
    heat_duty_actual: float  # BTU/hr
    heat_duty_theoretical_max: float  # BTU/hr
    provenance_hash: str = ""


# =============================================================================
# ECONOMIZER EFFECTIVENESS
# =============================================================================

def calculate_economizer_effectiveness(
    water_temp_out: float,
    water_temp_in: float,
    gas_temp_in: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate economizer effectiveness based on temperature rise.

    Effectiveness is the ratio of actual water temperature rise to
    the maximum possible temperature rise (if water could reach gas inlet temperature).

    Methodology (ASME PTC 4.3):
        epsilon = (T_water_out - T_water_in) / (T_gas_in - T_water_in)

        This represents the water-side thermal effectiveness, measuring
        how effectively the economizer heats the feedwater.

    Reference: ASME PTC 4.3

    Args:
        water_temp_out: Water outlet temperature (F)
        water_temp_in: Water inlet temperature (F)
        gas_temp_in: Gas inlet temperature (F)
        track_provenance: If True, return provenance record

    Returns:
        Effectiveness (0 to 1), optionally with provenance

    Raises:
        ValidationError: If temperatures are invalid
    """
    validate_temperature_fahrenheit(water_temp_out, "water_temp_out")
    validate_temperature_fahrenheit(water_temp_in, "water_temp_in")
    validate_temperature_fahrenheit(gas_temp_in, "gas_temp_in")

    if water_temp_out < water_temp_in:
        raise ValidationError(
            parameter="water_temp_out",
            value=water_temp_out,
            message=f"Water outlet ({water_temp_out}F) cannot be less than inlet ({water_temp_in}F)"
        )

    if gas_temp_in < water_temp_in:
        raise ValidationError(
            parameter="gas_temp_in",
            value=gas_temp_in,
            message=f"Gas inlet ({gas_temp_in}F) must be higher than water inlet ({water_temp_in}F)"
        )

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.EFFICIENCY,
            formula_id="economizer_effectiveness",
            formula_version="1.0.0",
            inputs={
                "water_temp_out": water_temp_out,
                "water_temp_in": water_temp_in,
                "gas_temp_in": gas_temp_in
            }
        )

    # Calculate actual temperature rise
    actual_rise = water_temp_out - water_temp_in

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate actual water temperature rise",
            inputs={"T_water_out": water_temp_out, "T_water_in": water_temp_in},
            output_name="actual_rise",
            output_value=actual_rise,
            formula="actual_rise = T_water_out - T_water_in"
        )

    # Calculate maximum possible temperature rise
    max_rise = gas_temp_in - water_temp_in

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate maximum possible temperature rise",
            inputs={"T_gas_in": gas_temp_in, "T_water_in": water_temp_in},
            output_name="max_rise",
            output_value=max_rise,
            formula="max_rise = T_gas_in - T_water_in"
        )

    # Calculate effectiveness
    if max_rise == 0:
        epsilon = 0.0
    else:
        epsilon = actual_rise / max_rise

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate economizer effectiveness",
            inputs={"actual_rise": actual_rise, "max_rise": max_rise},
            output_name="epsilon",
            output_value=epsilon,
            formula="epsilon = (T_water_out - T_water_in) / (T_gas_in - T_water_in)"
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


# =============================================================================
# HEAT RECOVERY RATIO
# =============================================================================

def calculate_heat_recovery_ratio(
    operating_point: EconomizerOperatingPoint,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate heat recovery ratio comparing actual to theoretical heat recovery.

    The heat recovery ratio indicates how effectively the economizer
    extracts available heat from the flue gas.

    Methodology:
        Q_actual = m_water * Cp_water * (T_water_out - T_water_in)
        Q_available = m_gas * Cp_gas * (T_gas_in - T_water_in)

        Heat Recovery Ratio = Q_actual / Q_available

    Reference: ASME PTC 4.3

    Args:
        operating_point: Complete economizer operating data
        track_provenance: If True, return provenance record

    Returns:
        Heat recovery ratio (0 to 1), optionally with provenance

    Raises:
        ValidationError: If operating point data is invalid
    """
    operating_point.validate()

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.EFFICIENCY,
            formula_id="heat_recovery_ratio",
            formula_version="1.0.0",
            inputs={
                "gas_temp_in": operating_point.gas_temp_in,
                "gas_temp_out": operating_point.gas_temp_out,
                "water_temp_in": operating_point.water_temp_in,
                "water_temp_out": operating_point.water_temp_out,
                "gas_flow_rate": operating_point.gas_flow_rate,
                "water_flow_rate": operating_point.water_flow_rate
            }
        )

    # Calculate actual heat recovered (water side)
    Q_actual = calculate_water_side_heat_duty(
        water_flow_rate=operating_point.water_flow_rate,
        water_temp_in=operating_point.water_temp_in,
        water_temp_out=operating_point.water_temp_out,
        pressure_psia=operating_point.water_pressure
    )

    if tracker:
        tracker.add_step(
            operation="heat_duty",
            description="Calculate actual heat recovered by water",
            inputs={
                "m_water": operating_point.water_flow_rate,
                "T_in": operating_point.water_temp_in,
                "T_out": operating_point.water_temp_out
            },
            output_name="Q_actual",
            output_value=Q_actual,
            formula="Q_actual = m_water * Cp_water * (T_out - T_in)"
        )

    # Calculate theoretical maximum available heat
    # (if flue gas cooled to water inlet temperature)
    T_avg_gas = (operating_point.gas_temp_in + operating_point.water_temp_in) / 2
    Cp_gas = get_flue_gas_cp(T_avg_gas, operating_point.gas_composition)

    Q_available = operating_point.gas_flow_rate * Cp_gas * (
        operating_point.gas_temp_in - operating_point.water_temp_in
    )

    if tracker:
        tracker.add_step(
            operation="heat_duty",
            description="Calculate theoretical maximum available heat",
            inputs={
                "m_gas": operating_point.gas_flow_rate,
                "Cp_gas": Cp_gas,
                "T_gas_in": operating_point.gas_temp_in,
                "T_water_in": operating_point.water_temp_in
            },
            output_name="Q_available",
            output_value=Q_available,
            formula="Q_available = m_gas * Cp_gas * (T_gas_in - T_water_in)"
        )

    # Calculate heat recovery ratio
    if Q_available == 0:
        ratio = 0.0
    else:
        ratio = Q_actual / Q_available

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate heat recovery ratio",
            inputs={"Q_actual": Q_actual, "Q_available": Q_available},
            output_name="ratio",
            output_value=ratio,
            formula="Heat Recovery Ratio = Q_actual / Q_available"
        )

    ratio = max(0.0, min(1.0, ratio))
    ratio_rounded = round(ratio, 4)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=ratio_rounded,
            output_unit="dimensionless",
            precision=4
        )
        return ratio_rounded, provenance

    return ratio_rounded


# =============================================================================
# GAS-SIDE AND WATER-SIDE EFFICIENCY
# =============================================================================

def calculate_gas_side_efficiency(
    gas_temp_in: float,
    gas_temp_out: float,
    water_temp_in: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate gas-side (flue gas cooling) efficiency.

    This metric indicates how effectively the economizer cools
    the flue gas, extracting heat for water heating.

    Methodology:
        eta_gas = (T_gas_in - T_gas_out) / (T_gas_in - T_water_in) * 100%

        The denominator represents the maximum possible temperature drop
        (if gas could be cooled to water inlet temperature).

    Reference: ASME PTC 4.3

    Args:
        gas_temp_in: Flue gas inlet temperature (F)
        gas_temp_out: Flue gas outlet temperature (F)
        water_temp_in: Water inlet temperature (F)
        track_provenance: If True, return provenance record

    Returns:
        Gas-side efficiency as percentage (0-100), optionally with provenance

    Raises:
        ValidationError: If temperatures are invalid
    """
    validate_temperature_fahrenheit(gas_temp_in, "gas_temp_in")
    validate_temperature_fahrenheit(gas_temp_out, "gas_temp_out")
    validate_temperature_fahrenheit(water_temp_in, "water_temp_in")

    if gas_temp_out > gas_temp_in:
        raise ValidationError(
            parameter="gas_temp_out",
            value=gas_temp_out,
            message=f"Gas outlet ({gas_temp_out}F) cannot exceed inlet ({gas_temp_in}F)"
        )

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.EFFICIENCY,
            formula_id="gas_side_efficiency",
            formula_version="1.0.0",
            inputs={
                "gas_temp_in": gas_temp_in,
                "gas_temp_out": gas_temp_out,
                "water_temp_in": water_temp_in
            }
        )

    # Calculate actual gas temperature drop
    actual_drop = gas_temp_in - gas_temp_out

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate actual gas temperature drop",
            inputs={"T_gas_in": gas_temp_in, "T_gas_out": gas_temp_out},
            output_name="actual_drop",
            output_value=actual_drop,
            formula="actual_drop = T_gas_in - T_gas_out"
        )

    # Calculate maximum possible drop
    max_drop = gas_temp_in - water_temp_in

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate maximum possible temperature drop",
            inputs={"T_gas_in": gas_temp_in, "T_water_in": water_temp_in},
            output_name="max_drop",
            output_value=max_drop,
            formula="max_drop = T_gas_in - T_water_in"
        )

    # Calculate efficiency
    if max_drop == 0:
        eta_gas = 0.0
    else:
        eta_gas = (actual_drop / max_drop) * 100

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate gas-side efficiency",
            inputs={"actual_drop": actual_drop, "max_drop": max_drop},
            output_name="eta_gas",
            output_value=eta_gas,
            formula="eta_gas = (actual_drop / max_drop) * 100%"
        )

    eta_gas = max(0.0, min(100.0, eta_gas))
    eta_rounded = round(eta_gas, 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=eta_rounded,
            output_unit="%",
            precision=2
        )
        return eta_rounded, provenance

    return eta_rounded


def calculate_water_side_efficiency(
    water_temp_in: float,
    water_temp_out: float,
    gas_temp_in: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate water-side (feedwater heating) efficiency.

    This metric indicates how effectively the feedwater is heated
    relative to the available temperature potential.

    Methodology:
        eta_water = (T_water_out - T_water_in) / (T_gas_in - T_water_in) * 100%

        This is equivalent to economizer effectiveness expressed as percentage.

    Reference: ASME PTC 4.3

    Args:
        water_temp_in: Water inlet temperature (F)
        water_temp_out: Water outlet temperature (F)
        gas_temp_in: Flue gas inlet temperature (F)
        track_provenance: If True, return provenance record

    Returns:
        Water-side efficiency as percentage (0-100), optionally with provenance

    Raises:
        ValidationError: If temperatures are invalid
    """
    validate_temperature_fahrenheit(water_temp_in, "water_temp_in")
    validate_temperature_fahrenheit(water_temp_out, "water_temp_out")
    validate_temperature_fahrenheit(gas_temp_in, "gas_temp_in")

    if water_temp_out < water_temp_in:
        raise ValidationError(
            parameter="water_temp_out",
            value=water_temp_out,
            message=f"Water outlet ({water_temp_out}F) cannot be less than inlet ({water_temp_in}F)"
        )

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.EFFICIENCY,
            formula_id="water_side_efficiency",
            formula_version="1.0.0",
            inputs={
                "water_temp_in": water_temp_in,
                "water_temp_out": water_temp_out,
                "gas_temp_in": gas_temp_in
            }
        )

    # Calculate actual water temperature rise
    actual_rise = water_temp_out - water_temp_in

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate actual water temperature rise",
            inputs={"T_water_out": water_temp_out, "T_water_in": water_temp_in},
            output_name="actual_rise",
            output_value=actual_rise,
            formula="actual_rise = T_water_out - T_water_in"
        )

    # Calculate maximum possible rise
    max_rise = gas_temp_in - water_temp_in

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate maximum possible temperature rise",
            inputs={"T_gas_in": gas_temp_in, "T_water_in": water_temp_in},
            output_name="max_rise",
            output_value=max_rise,
            formula="max_rise = T_gas_in - T_water_in"
        )

    # Calculate efficiency
    if max_rise == 0:
        eta_water = 0.0
    else:
        eta_water = (actual_rise / max_rise) * 100

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate water-side efficiency",
            inputs={"actual_rise": actual_rise, "max_rise": max_rise},
            output_name="eta_water",
            output_value=eta_water,
            formula="eta_water = (actual_rise / max_rise) * 100%"
        )

    eta_water = max(0.0, min(100.0, eta_water))
    eta_rounded = round(eta_water, 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=eta_rounded,
            output_unit="%",
            precision=2
        )
        return eta_rounded, provenance

    return eta_rounded


# =============================================================================
# OVERALL EFFICIENCY
# =============================================================================

def calculate_overall_efficiency(
    operating_point: EconomizerOperatingPoint,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate overall economizer efficiency combining multiple metrics.

    This provides a single efficiency value that accounts for both
    gas-side and water-side performance.

    Methodology:
        The overall efficiency is calculated as the geometric mean of
        heat recovery ratio and temperature-based effectiveness:

        eta_overall = sqrt(HRR * epsilon) * 100%

        Where:
        - HRR = Heat Recovery Ratio
        - epsilon = (T_water_out - T_water_in) / (T_gas_in - T_water_in)

    Reference: ASME PTC 4.3

    Args:
        operating_point: Complete economizer operating data
        track_provenance: If True, return provenance record

    Returns:
        Overall efficiency as percentage (0-100), optionally with provenance

    Raises:
        ValidationError: If operating point data is invalid
    """
    operating_point.validate()

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.EFFICIENCY,
            formula_id="overall_efficiency",
            formula_version="1.0.0",
            inputs={
                "gas_temp_in": operating_point.gas_temp_in,
                "gas_temp_out": operating_point.gas_temp_out,
                "water_temp_in": operating_point.water_temp_in,
                "water_temp_out": operating_point.water_temp_out,
                "gas_flow_rate": operating_point.gas_flow_rate,
                "water_flow_rate": operating_point.water_flow_rate
            }
        )

    # Calculate heat recovery ratio
    hrr = calculate_heat_recovery_ratio(operating_point)

    if tracker:
        tracker.add_step(
            operation="calculate",
            description="Calculate heat recovery ratio",
            inputs={
                "Q_actual": "calculated",
                "Q_available": "calculated"
            },
            output_name="HRR",
            output_value=hrr,
            formula="HRR = Q_actual / Q_available"
        )

    # Calculate temperature effectiveness
    epsilon = calculate_economizer_effectiveness(
        water_temp_out=operating_point.water_temp_out,
        water_temp_in=operating_point.water_temp_in,
        gas_temp_in=operating_point.gas_temp_in
    )

    if tracker:
        tracker.add_step(
            operation="calculate",
            description="Calculate temperature effectiveness",
            inputs={
                "T_water_out": operating_point.water_temp_out,
                "T_water_in": operating_point.water_temp_in,
                "T_gas_in": operating_point.gas_temp_in
            },
            output_name="epsilon",
            output_value=epsilon,
            formula="epsilon = (T_water_out - T_water_in) / (T_gas_in - T_water_in)"
        )

    # Calculate overall efficiency as geometric mean
    eta_overall = math.sqrt(hrr * epsilon) * 100

    if tracker:
        tracker.add_step(
            operation="geometric_mean",
            description="Calculate overall efficiency as geometric mean",
            inputs={"HRR": hrr, "epsilon": epsilon},
            output_name="eta_overall",
            output_value=eta_overall,
            formula="eta_overall = sqrt(HRR * epsilon) * 100%"
        )

    eta_overall = max(0.0, min(100.0, eta_overall))
    eta_rounded = round(eta_overall, 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=eta_rounded,
            output_unit="%",
            precision=2
        )
        return eta_rounded, provenance

    return eta_rounded


# =============================================================================
# DESIGN DEVIATION ANALYSIS
# =============================================================================

def calculate_design_deviation(
    operating_point: EconomizerOperatingPoint,
    design_conditions: DesignConditions,
    track_provenance: bool = False
) -> Union[Dict[str, float], Tuple[Dict[str, float], CalculationProvenance]]:
    """
    Calculate deviations from design performance.

    This analysis identifies how current performance differs from
    design specifications, helping to quantify degradation.

    Methodology:
        For each parameter:
        Deviation = ((Actual - Design) / Design) * 100%

        Key metrics:
        - Heat duty deviation
        - Effectiveness deviation
        - Temperature deviation
        - Flow rate deviation

    Reference: ASME PTC 4.3

    Args:
        operating_point: Current operating data
        design_conditions: Design specifications
        track_provenance: If True, return provenance record

    Returns:
        Dictionary of deviations (percentage), optionally with provenance

    Raises:
        ValidationError: If data is invalid
    """
    operating_point.validate()

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.EFFICIENCY,
            formula_id="design_deviation",
            formula_version="1.0.0",
            inputs={
                "operating_point": "EconomizerOperatingPoint",
                "design_conditions": "DesignConditions"
            }
        )

    # Calculate actual heat duty
    Q_actual = calculate_water_side_heat_duty(
        water_flow_rate=operating_point.water_flow_rate,
        water_temp_in=operating_point.water_temp_in,
        water_temp_out=operating_point.water_temp_out,
        pressure_psia=operating_point.water_pressure
    )

    # Calculate heat duty deviation
    heat_duty_deviation = ((Q_actual - design_conditions.heat_duty_design) /
                           design_conditions.heat_duty_design) * 100

    if tracker:
        tracker.add_step(
            operation="deviation",
            description="Calculate heat duty deviation from design",
            inputs={"Q_actual": Q_actual, "Q_design": design_conditions.heat_duty_design},
            output_name="heat_duty_deviation",
            output_value=heat_duty_deviation,
            formula="deviation = ((actual - design) / design) * 100%"
        )

    # Calculate effectiveness deviation
    epsilon_actual = calculate_economizer_effectiveness(
        water_temp_out=operating_point.water_temp_out,
        water_temp_in=operating_point.water_temp_in,
        gas_temp_in=operating_point.gas_temp_in
    )

    effectiveness_deviation = ((epsilon_actual - design_conditions.effectiveness_design) /
                               design_conditions.effectiveness_design) * 100

    if tracker:
        tracker.add_step(
            operation="deviation",
            description="Calculate effectiveness deviation from design",
            inputs={
                "epsilon_actual": epsilon_actual,
                "epsilon_design": design_conditions.effectiveness_design
            },
            output_name="effectiveness_deviation",
            output_value=effectiveness_deviation,
            formula="deviation = ((actual - design) / design) * 100%"
        )

    # Calculate water temperature rise deviation
    actual_water_rise = operating_point.water_temp_out - operating_point.water_temp_in
    design_water_rise = (design_conditions.water_temp_out_design -
                         design_conditions.water_temp_in_design)

    water_rise_deviation = ((actual_water_rise - design_water_rise) /
                            design_water_rise) * 100 if design_water_rise != 0 else 0

    if tracker:
        tracker.add_step(
            operation="deviation",
            description="Calculate water temperature rise deviation",
            inputs={"actual_rise": actual_water_rise, "design_rise": design_water_rise},
            output_name="water_rise_deviation",
            output_value=water_rise_deviation,
            formula="deviation = ((actual - design) / design) * 100%"
        )

    # Calculate gas temperature drop deviation
    actual_gas_drop = operating_point.gas_temp_in - operating_point.gas_temp_out
    design_gas_drop = (design_conditions.gas_temp_in_design -
                       design_conditions.gas_temp_out_design)

    gas_drop_deviation = ((actual_gas_drop - design_gas_drop) /
                          design_gas_drop) * 100 if design_gas_drop != 0 else 0

    if tracker:
        tracker.add_step(
            operation="deviation",
            description="Calculate gas temperature drop deviation",
            inputs={"actual_drop": actual_gas_drop, "design_drop": design_gas_drop},
            output_name="gas_drop_deviation",
            output_value=gas_drop_deviation,
            formula="deviation = ((actual - design) / design) * 100%"
        )

    # Flow rate deviations
    water_flow_deviation = ((operating_point.water_flow_rate -
                             design_conditions.water_flow_rate_design) /
                            design_conditions.water_flow_rate_design) * 100

    gas_flow_deviation = ((operating_point.gas_flow_rate -
                           design_conditions.gas_flow_rate_design) /
                          design_conditions.gas_flow_rate_design) * 100

    result = {
        "heat_duty_deviation_percent": round(heat_duty_deviation, 2),
        "effectiveness_deviation_percent": round(effectiveness_deviation, 2),
        "water_rise_deviation_percent": round(water_rise_deviation, 2),
        "gas_drop_deviation_percent": round(gas_drop_deviation, 2),
        "water_flow_deviation_percent": round(water_flow_deviation, 2),
        "gas_flow_deviation_percent": round(gas_flow_deviation, 2),
        "actual_heat_duty_btu_hr": round(Q_actual, 2),
        "actual_effectiveness": round(epsilon_actual, 4)
    }

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=heat_duty_deviation,
            output_unit="%",
            precision=2
        )
        return result, provenance

    return result


# =============================================================================
# COMPREHENSIVE EFFICIENCY REPORT
# =============================================================================

def calculate_comprehensive_efficiency(
    operating_point: EconomizerOperatingPoint,
    design_conditions: Optional[DesignConditions] = None,
    track_provenance: bool = False
) -> Union[EfficiencyResult, Tuple[EfficiencyResult, CalculationProvenance]]:
    """
    Calculate comprehensive efficiency metrics for the economizer.

    This function provides a complete efficiency assessment combining
    all efficiency metrics into a single result object.

    Args:
        operating_point: Complete economizer operating data
        design_conditions: Optional design specifications for comparison
        track_provenance: If True, return provenance record

    Returns:
        EfficiencyResult with all metrics, optionally with provenance

    Raises:
        ValidationError: If operating point data is invalid
    """
    operating_point.validate()

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.EFFICIENCY,
            formula_id="comprehensive_efficiency",
            formula_version="1.0.0",
            inputs={
                "gas_temp_in": operating_point.gas_temp_in,
                "gas_temp_out": operating_point.gas_temp_out,
                "water_temp_in": operating_point.water_temp_in,
                "water_temp_out": operating_point.water_temp_out,
                "gas_flow_rate": operating_point.gas_flow_rate,
                "water_flow_rate": operating_point.water_flow_rate
            }
        )

    # Calculate all efficiency metrics
    economizer_effectiveness = calculate_economizer_effectiveness(
        water_temp_out=operating_point.water_temp_out,
        water_temp_in=operating_point.water_temp_in,
        gas_temp_in=operating_point.gas_temp_in
    )

    heat_recovery_ratio = calculate_heat_recovery_ratio(operating_point)

    gas_side_efficiency = calculate_gas_side_efficiency(
        gas_temp_in=operating_point.gas_temp_in,
        gas_temp_out=operating_point.gas_temp_out,
        water_temp_in=operating_point.water_temp_in
    )

    water_side_efficiency = calculate_water_side_efficiency(
        water_temp_in=operating_point.water_temp_in,
        water_temp_out=operating_point.water_temp_out,
        gas_temp_in=operating_point.gas_temp_in
    )

    overall_efficiency = calculate_overall_efficiency(operating_point)

    # Calculate actual heat duty
    heat_duty_actual = calculate_water_side_heat_duty(
        water_flow_rate=operating_point.water_flow_rate,
        water_temp_in=operating_point.water_temp_in,
        water_temp_out=operating_point.water_temp_out,
        pressure_psia=operating_point.water_pressure
    )

    # Calculate theoretical maximum heat duty
    T_avg_gas = (operating_point.gas_temp_in + operating_point.water_temp_in) / 2
    Cp_gas = get_flue_gas_cp(T_avg_gas, operating_point.gas_composition)
    heat_duty_max = operating_point.gas_flow_rate * Cp_gas * (
        operating_point.gas_temp_in - operating_point.water_temp_in
    )

    if tracker:
        tracker.add_step(
            operation="aggregate",
            description="Aggregate all efficiency metrics",
            inputs={
                "economizer_effectiveness": economizer_effectiveness,
                "heat_recovery_ratio": heat_recovery_ratio,
                "gas_side_efficiency": gas_side_efficiency,
                "water_side_efficiency": water_side_efficiency,
                "overall_efficiency": overall_efficiency
            },
            output_name="efficiency_result",
            output_value=overall_efficiency,
            formula="Comprehensive efficiency calculation"
        )

    result = EfficiencyResult(
        economizer_effectiveness=economizer_effectiveness,
        heat_recovery_ratio=heat_recovery_ratio,
        gas_side_efficiency=gas_side_efficiency,
        water_side_efficiency=water_side_efficiency,
        overall_efficiency=overall_efficiency,
        heat_duty_actual=round(heat_duty_actual, 2),
        heat_duty_theoretical_max=round(heat_duty_max, 2)
    )

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=overall_efficiency,
            output_unit="%",
            precision=2
        )
        result.provenance_hash = provenance.provenance_hash
        return result, provenance

    return result


# =============================================================================
# PERFORMANCE TRENDING
# =============================================================================

def calculate_efficiency_trend(
    efficiency_history: List[Tuple[float, float]],
    window_hours: float = 168.0
) -> Dict[str, float]:
    """
    Calculate efficiency trend from historical data.

    Args:
        efficiency_history: List of (timestamp_hours, efficiency_percent) tuples
        window_hours: Time window for trend calculation (default: 1 week)

    Returns:
        Dictionary with trend metrics (slope, r_squared, predicted_next)
    """
    if len(efficiency_history) < 2:
        return {
            "slope_percent_per_hour": 0.0,
            "r_squared": 0.0,
            "predicted_next_efficiency": efficiency_history[-1][1] if efficiency_history else 0.0,
            "trend_direction": "stable"
        }

    # Filter to window
    max_time = max(t for t, _ in efficiency_history)
    window_data = [(t, e) for t, e in efficiency_history if t >= max_time - window_hours]

    if len(window_data) < 2:
        window_data = efficiency_history[-2:]

    # Linear regression
    n = len(window_data)
    times = [t for t, _ in window_data]
    efficiencies = [e for _, e in window_data]

    sum_t = sum(times)
    sum_e = sum(efficiencies)
    sum_t2 = sum(t**2 for t in times)
    sum_te = sum(t * e for t, e in window_data)
    sum_e2 = sum(e**2 for e in efficiencies)

    denominator = n * sum_t2 - sum_t**2

    if abs(denominator) < 1e-10:
        slope = 0.0
        intercept = sum_e / n if n > 0 else 0.0
    else:
        slope = (n * sum_te - sum_t * sum_e) / denominator
        intercept = (sum_e - slope * sum_t) / n

    # Calculate R-squared
    mean_e = sum_e / n
    ss_tot = sum((e - mean_e)**2 for e in efficiencies)
    ss_res = sum((e - (slope * t + intercept))**2 for t, e in window_data)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Predict next efficiency (1 hour ahead)
    predicted_next = slope * (max_time + 1) + intercept

    # Determine trend direction
    if slope > 0.001:
        trend_direction = "improving"
    elif slope < -0.001:
        trend_direction = "degrading"
    else:
        trend_direction = "stable"

    return {
        "slope_percent_per_hour": round(slope, 6),
        "r_squared": round(r_squared, 4),
        "predicted_next_efficiency": round(predicted_next, 2),
        "trend_direction": trend_direction,
        "intercept": round(intercept, 2)
    }
