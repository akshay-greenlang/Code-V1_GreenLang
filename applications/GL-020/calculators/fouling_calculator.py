"""
GL-020 ECONOPULSE: Fouling Calculator

Zero-hallucination fouling analysis calculations for economizer performance
monitoring based on ASME PTC 4.3 and TEMA standards.

This module provides:
- Fouling factor (Rf) calculation from U-values
- Fouling rate trending (dRf/dt)
- Cleaning time prediction
- Cleanliness factor assessment
- Efficiency loss quantification
- Fuel penalty estimation

Supports both gas-side fouling (soot, ash) and water-side fouling (scale).

Author: GL-CalculatorEngineer
Standards: ASME PTC 4.3, TEMA
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from .provenance import (
    ProvenanceTracker,
    CalculationType,
    CalculationProvenance,
    generate_calculation_hash
)
from .thermal_properties import ValidationError


# =============================================================================
# ENUMERATIONS AND CONSTANTS
# =============================================================================

class FoulingType(Enum):
    """Type of fouling deposit."""
    GAS_SIDE_SOOT = "gas_side_soot"
    GAS_SIDE_ASH = "gas_side_ash"
    GAS_SIDE_SULFATE = "gas_side_sulfate"
    WATER_SIDE_SCALE = "water_side_scale"
    WATER_SIDE_DEPOSIT = "water_side_deposit"
    COMBINED = "combined"


class FoulingTrendModel(Enum):
    """Mathematical model for fouling rate prediction."""
    LINEAR = "linear"
    ASYMPTOTIC = "asymptotic"
    FALLING_RATE = "falling_rate"


# TEMA fouling factor recommendations ((hr-ft2-F)/BTU)
TEMA_FOULING_FACTORS = {
    "boiler_feedwater_treated": 0.0005,
    "boiler_feedwater_untreated": 0.001,
    "flue_gas_clean": 0.001,
    "flue_gas_coal": 0.005,
    "flue_gas_oil": 0.003,
    "flue_gas_natural_gas": 0.001,
    "steam_clean": 0.0005,
}

# Typical fouling thresholds for cleaning decision
FOULING_THRESHOLDS = {
    "critical": 0.010,  # (hr-ft2-F)/BTU - Requires immediate cleaning
    "high": 0.005,      # Plan cleaning within 1 week
    "moderate": 0.003,  # Monitor closely
    "normal": 0.001,    # Acceptable
}


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_positive(value: float, param_name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValidationError(
            parameter=param_name,
            value=value,
            message=f"{param_name} ({value}) must be positive"
        )


def validate_non_negative(value: float, param_name: str) -> None:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ValidationError(
            parameter=param_name,
            value=value,
            message=f"{param_name} ({value}) must be non-negative"
        )


def validate_u_value_relationship(U_fouled: float, U_clean: float) -> None:
    """Validate that fouled U-value is less than or equal to clean U-value."""
    if U_fouled > U_clean:
        raise ValidationError(
            parameter="U_fouled",
            value=U_fouled,
            message=f"Fouled U-value ({U_fouled}) cannot exceed clean U-value ({U_clean})"
        )


# =============================================================================
# FOULING FACTOR CALCULATIONS
# =============================================================================

def calculate_fouling_factor(
    U_fouled: float,
    U_clean: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate total fouling factor (Rf) from clean and fouled U-values.

    The fouling factor represents the additional thermal resistance
    introduced by deposits on heat transfer surfaces.

    Methodology (TEMA):
        Total thermal resistance:
        1/U_fouled = 1/U_clean + Rf

        Therefore:
        Rf = (1/U_fouled) - (1/U_clean)

    Reference: TEMA Standards, ASME PTC 4.3

    Args:
        U_fouled: Current (fouled) overall heat transfer coefficient (BTU/(hr-ft2-F))
        U_clean: Clean (baseline) overall heat transfer coefficient (BTU/(hr-ft2-F))
        track_provenance: If True, return provenance record

    Returns:
        Fouling factor in (hr-ft2-F)/BTU, optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_positive(U_fouled, "U_fouled")
    validate_positive(U_clean, "U_clean")
    validate_u_value_relationship(U_fouled, U_clean)

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.FOULING,
            formula_id="fouling_factor",
            formula_version="1.0.0",
            inputs={
                "U_fouled": U_fouled,
                "U_clean": U_clean
            }
        )

    # Calculate thermal resistances
    R_fouled = 1 / U_fouled
    R_clean = 1 / U_clean

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate fouled thermal resistance",
            inputs={"U_fouled": U_fouled},
            output_name="R_fouled",
            output_value=R_fouled,
            formula="R_fouled = 1 / U_fouled"
        )
        tracker.add_step(
            operation="divide",
            description="Calculate clean thermal resistance",
            inputs={"U_clean": U_clean},
            output_name="R_clean",
            output_value=R_clean,
            formula="R_clean = 1 / U_clean"
        )

    # Calculate fouling factor
    Rf = R_fouled - R_clean

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate fouling factor as difference in resistances",
            inputs={"R_fouled": R_fouled, "R_clean": R_clean},
            output_name="Rf",
            output_value=Rf,
            formula="Rf = (1/U_fouled) - (1/U_clean)"
        )

    Rf_rounded = round(Rf, 6)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=Rf_rounded,
            output_unit="(hr-ft2-F)/BTU",
            precision=6
        )
        return Rf_rounded, provenance

    return Rf_rounded


def calculate_fouling_factor_components(
    U_fouled: float,
    U_clean: float,
    gas_side_fraction: float = 0.7,
    track_provenance: bool = False
) -> Union[Dict[str, float], Tuple[Dict[str, float], CalculationProvenance]]:
    """
    Calculate gas-side and water-side fouling factor components.

    In economizers, fouling typically occurs on both gas and water sides.
    This function partitions the total fouling based on typical distributions.

    Methodology:
        Rf_total = Rf_gas + Rf_water
        Rf_gas = Rf_total * gas_side_fraction
        Rf_water = Rf_total * (1 - gas_side_fraction)

    Note: Actual partitioning should be based on inspection data when available.

    Args:
        U_fouled: Current overall heat transfer coefficient (BTU/(hr-ft2-F))
        U_clean: Clean overall heat transfer coefficient (BTU/(hr-ft2-F))
        gas_side_fraction: Fraction of fouling on gas side (default 0.7)
        track_provenance: If True, return provenance record

    Returns:
        Dictionary with Rf_total, Rf_gas, Rf_water, optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_positive(U_fouled, "U_fouled")
    validate_positive(U_clean, "U_clean")
    validate_u_value_relationship(U_fouled, U_clean)

    if not 0 <= gas_side_fraction <= 1:
        raise ValidationError(
            parameter="gas_side_fraction",
            value=gas_side_fraction,
            message="Gas side fraction must be between 0 and 1"
        )

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.FOULING,
            formula_id="fouling_factor_components",
            formula_version="1.0.0",
            inputs={
                "U_fouled": U_fouled,
                "U_clean": U_clean,
                "gas_side_fraction": gas_side_fraction
            }
        )

    # Calculate total fouling factor
    Rf_total = (1 / U_fouled) - (1 / U_clean)

    if tracker:
        tracker.add_step(
            operation="subtract",
            description="Calculate total fouling factor",
            inputs={"U_fouled": U_fouled, "U_clean": U_clean},
            output_name="Rf_total",
            output_value=Rf_total,
            formula="Rf_total = (1/U_fouled) - (1/U_clean)"
        )

    # Partition between gas and water sides
    Rf_gas = Rf_total * gas_side_fraction
    Rf_water = Rf_total * (1 - gas_side_fraction)

    if tracker:
        tracker.add_step(
            operation="multiply",
            description="Calculate gas-side fouling factor",
            inputs={"Rf_total": Rf_total, "gas_side_fraction": gas_side_fraction},
            output_name="Rf_gas",
            output_value=Rf_gas,
            formula="Rf_gas = Rf_total * gas_side_fraction"
        )
        tracker.add_step(
            operation="multiply",
            description="Calculate water-side fouling factor",
            inputs={"Rf_total": Rf_total, "water_side_fraction": 1 - gas_side_fraction},
            output_name="Rf_water",
            output_value=Rf_water,
            formula="Rf_water = Rf_total * (1 - gas_side_fraction)"
        )

    result = {
        "Rf_total": round(Rf_total, 6),
        "Rf_gas": round(Rf_gas, 6),
        "Rf_water": round(Rf_water, 6)
    }

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=Rf_total,
            output_unit="(hr-ft2-F)/BTU",
            precision=6
        )
        return result, provenance

    return result


# =============================================================================
# FOULING RATE CALCULATIONS
# =============================================================================

@dataclass
class FoulingDataPoint:
    """Single fouling measurement data point."""
    timestamp: datetime
    fouling_factor: float
    U_value: float


def calculate_fouling_rate(
    fouling_data: List[FoulingDataPoint],
    model: FoulingTrendModel = FoulingTrendModel.LINEAR,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate fouling rate (dRf/dt) from historical data.

    The fouling rate indicates how quickly deposits are accumulating,
    which is essential for predicting cleaning intervals.

    Methodology:
        Linear model:
            Rf(t) = Rf0 + k*t
            dRf/dt = k (constant rate)

        Asymptotic model:
            Rf(t) = Rf_max * (1 - exp(-k*t))
            dRf/dt varies with time

        Falling rate model:
            Rf(t) = Rf0 + k*t^0.5
            dRf/dt = 0.5*k/sqrt(t)

    Reference: Kern & Seaton fouling models

    Args:
        fouling_data: List of FoulingDataPoint objects (minimum 2 points)
        model: Fouling trend model to use
        track_provenance: If True, return provenance record

    Returns:
        Fouling rate in (hr-ft2-F)/(BTU-hr), optionally with provenance

    Raises:
        ValueError: If insufficient data points
        ValidationError: If data is invalid
    """
    if len(fouling_data) < 2:
        raise ValueError("At least 2 data points required to calculate fouling rate")

    # Sort by timestamp
    sorted_data = sorted(fouling_data, key=lambda x: x.timestamp)

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.FOULING,
            formula_id=f"fouling_rate_{model.value}",
            formula_version="1.0.0",
            inputs={
                "n_points": len(fouling_data),
                "model": model.value,
                "start_time": sorted_data[0].timestamp.isoformat(),
                "end_time": sorted_data[-1].timestamp.isoformat()
            }
        )

    if model == FoulingTrendModel.LINEAR:
        # Linear regression for dRf/dt
        # Use least squares: Rf = a + b*t
        n = len(sorted_data)
        t0 = sorted_data[0].timestamp

        # Calculate time differences in hours
        times = [(d.timestamp - t0).total_seconds() / 3600 for d in sorted_data]
        rf_values = [d.fouling_factor for d in sorted_data]

        # Calculate sums for linear regression
        sum_t = sum(times)
        sum_rf = sum(rf_values)
        sum_t2 = sum(t**2 for t in times)
        sum_t_rf = sum(t * rf for t, rf in zip(times, rf_values))

        # Linear regression slope (fouling rate)
        denominator = n * sum_t2 - sum_t**2

        if abs(denominator) < 1e-10:
            # All points at same time - cannot calculate rate
            fouling_rate = 0.0
        else:
            fouling_rate = (n * sum_t_rf - sum_t * sum_rf) / denominator

        if tracker:
            tracker.add_step(
                operation="linear_regression",
                description="Calculate fouling rate using linear regression",
                inputs={
                    "n": n,
                    "sum_t": sum_t,
                    "sum_rf": sum_rf,
                    "sum_t2": sum_t2,
                    "sum_t_rf": sum_t_rf
                },
                output_name="fouling_rate",
                output_value=fouling_rate,
                formula="dRf/dt = (n*sum(t*Rf) - sum(t)*sum(Rf)) / (n*sum(t^2) - sum(t)^2)"
            )

    elif model == FoulingTrendModel.ASYMPTOTIC:
        # Asymptotic model fitting (simplified)
        # Use most recent rate as approximation
        n = len(sorted_data)
        if n >= 3:
            dt = (sorted_data[-1].timestamp - sorted_data[-2].timestamp).total_seconds() / 3600
            dRf = sorted_data[-1].fouling_factor - sorted_data[-2].fouling_factor

            if dt > 0:
                fouling_rate = dRf / dt
            else:
                fouling_rate = 0.0
        else:
            dt = (sorted_data[-1].timestamp - sorted_data[0].timestamp).total_seconds() / 3600
            dRf = sorted_data[-1].fouling_factor - sorted_data[0].fouling_factor

            if dt > 0:
                fouling_rate = dRf / dt
            else:
                fouling_rate = 0.0

        if tracker:
            tracker.add_step(
                operation="asymptotic_rate",
                description="Calculate instantaneous fouling rate (asymptotic model)",
                inputs={"dRf": dRf, "dt": dt},
                output_name="fouling_rate",
                output_value=fouling_rate,
                formula="dRf/dt = (Rf_n - Rf_n-1) / (t_n - t_n-1)"
            )

    else:  # FALLING_RATE
        # Falling rate model: dRf/dt decreases over time
        t0 = sorted_data[0].timestamp
        total_hours = (sorted_data[-1].timestamp - t0).total_seconds() / 3600
        total_dRf = sorted_data[-1].fouling_factor - sorted_data[0].fouling_factor

        if total_hours > 0:
            # k = 2 * total_dRf / sqrt(total_hours)
            k = 2 * total_dRf / math.sqrt(total_hours) if total_hours > 0 else 0
            # Current rate at latest time
            fouling_rate = 0.5 * k / math.sqrt(total_hours) if total_hours > 0 else 0
        else:
            fouling_rate = 0.0

        if tracker:
            tracker.add_step(
                operation="falling_rate",
                description="Calculate fouling rate using falling rate model",
                inputs={"total_dRf": total_dRf, "total_hours": total_hours},
                output_name="fouling_rate",
                output_value=fouling_rate,
                formula="dRf/dt = 0.5 * k / sqrt(t), k = 2*Rf/sqrt(t)"
            )

    rate_rounded = round(fouling_rate, 9)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=rate_rounded,
            output_unit="(hr-ft2-F)/(BTU-hr)",
            precision=9
        )
        return rate_rounded, provenance

    return rate_rounded


# =============================================================================
# CLEANING PREDICTION
# =============================================================================

def predict_cleaning_time(
    current_fouling_factor: float,
    fouling_rate: float,
    threshold_fouling_factor: float = 0.005,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Predict time until fouling factor reaches cleaning threshold.

    This calculation enables proactive maintenance scheduling by
    predicting when cleaning will be required.

    Methodology:
        Assuming linear fouling:
        Rf(t) = Rf_current + fouling_rate * t
        Rf_threshold = Rf_current + fouling_rate * t_cleaning

        Therefore:
        t_cleaning = (Rf_threshold - Rf_current) / fouling_rate

    Reference: EPRI fouling management guidelines

    Args:
        current_fouling_factor: Current Rf ((hr-ft2-F)/BTU)
        fouling_rate: Rate of fouling accumulation ((hr-ft2-F)/(BTU-hr))
        threshold_fouling_factor: Rf threshold for cleaning ((hr-ft2-F)/BTU)
        track_provenance: If True, return provenance record

    Returns:
        Time to cleaning in hours (negative if already past threshold),
        optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_non_negative(current_fouling_factor, "current_fouling_factor")
    validate_non_negative(threshold_fouling_factor, "threshold_fouling_factor")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.FOULING,
            formula_id="predict_cleaning_time",
            formula_version="1.0.0",
            inputs={
                "current_fouling_factor": current_fouling_factor,
                "fouling_rate": fouling_rate,
                "threshold_fouling_factor": threshold_fouling_factor
            }
        )

    # Check if already past threshold
    if current_fouling_factor >= threshold_fouling_factor:
        time_to_cleaning = 0.0
        if tracker:
            tracker.add_step(
                operation="comparison",
                description="Current fouling already exceeds threshold",
                inputs={
                    "Rf_current": current_fouling_factor,
                    "Rf_threshold": threshold_fouling_factor
                },
                output_name="time_to_cleaning",
                output_value=0.0,
                formula="Rf_current >= Rf_threshold => immediate cleaning needed"
            )

    elif fouling_rate <= 0:
        # No fouling accumulation or cleaning in progress
        time_to_cleaning = float('inf')
        if tracker:
            tracker.add_step(
                operation="check_rate",
                description="Fouling rate is zero or negative - no cleaning predicted",
                inputs={"fouling_rate": fouling_rate},
                output_name="time_to_cleaning",
                output_value=float('inf'),
                formula="dRf/dt <= 0 => no cleaning needed"
            )

    else:
        # Calculate time to threshold
        delta_Rf = threshold_fouling_factor - current_fouling_factor
        time_to_cleaning = delta_Rf / fouling_rate

        if tracker:
            tracker.add_step(
                operation="divide",
                description="Calculate time to reach cleaning threshold",
                inputs={
                    "delta_Rf": delta_Rf,
                    "fouling_rate": fouling_rate
                },
                output_name="time_to_cleaning",
                output_value=time_to_cleaning,
                formula="t_cleaning = (Rf_threshold - Rf_current) / fouling_rate"
            )

    if time_to_cleaning == float('inf'):
        result = float('inf')
    else:
        result = round(time_to_cleaning, 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=result if result != float('inf') else Decimal('999999'),
            output_unit="hours",
            precision=2
        )
        return result, provenance

    return result


def predict_cleaning_date(
    current_fouling_factor: float,
    fouling_rate: float,
    threshold_fouling_factor: float = 0.005,
    reference_date: Optional[datetime] = None
) -> Optional[datetime]:
    """
    Predict the date when cleaning will be required.

    Args:
        current_fouling_factor: Current Rf ((hr-ft2-F)/BTU)
        fouling_rate: Rate of fouling accumulation ((hr-ft2-F)/(BTU-hr))
        threshold_fouling_factor: Rf threshold for cleaning ((hr-ft2-F)/BTU)
        reference_date: Reference date for calculation (default: now)

    Returns:
        Predicted cleaning date, or None if cleaning not predicted

    Raises:
        ValidationError: If inputs are invalid
    """
    time_to_cleaning = predict_cleaning_time(
        current_fouling_factor,
        fouling_rate,
        threshold_fouling_factor
    )

    if time_to_cleaning == float('inf'):
        return None

    if reference_date is None:
        reference_date = datetime.now()

    return reference_date + timedelta(hours=time_to_cleaning)


# =============================================================================
# CLEANLINESS FACTOR
# =============================================================================

def calculate_cleanliness_factor(
    U_actual: float,
    U_clean: float,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate cleanliness factor as percentage of clean performance.

    The cleanliness factor indicates the current heat transfer
    performance relative to clean (design) conditions.

    Methodology:
        CF = (U_actual / U_clean) * 100%

        Where:
        - CF = 100%: Perfectly clean
        - CF < 100%: Some fouling present
        - CF < 80%: Significant fouling, consider cleaning

    Reference: TEMA Standards

    Args:
        U_actual: Current overall heat transfer coefficient (BTU/(hr-ft2-F))
        U_clean: Clean (design) overall heat transfer coefficient (BTU/(hr-ft2-F))
        track_provenance: If True, return provenance record

    Returns:
        Cleanliness factor as percentage (0-100+), optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_positive(U_actual, "U_actual")
    validate_positive(U_clean, "U_clean")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.FOULING,
            formula_id="cleanliness_factor",
            formula_version="1.0.0",
            inputs={
                "U_actual": U_actual,
                "U_clean": U_clean
            }
        )

    CF = (U_actual / U_clean) * 100

    if tracker:
        tracker.add_step(
            operation="divide_multiply",
            description="Calculate cleanliness factor",
            inputs={"U_actual": U_actual, "U_clean": U_clean},
            output_name="CF",
            output_value=CF,
            formula="CF = (U_actual / U_clean) * 100%"
        )

    CF_rounded = round(CF, 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=CF_rounded,
            output_unit="%",
            precision=2
        )
        return CF_rounded, provenance

    return CF_rounded


# =============================================================================
# EFFICIENCY LOSS CALCULATIONS
# =============================================================================

def calculate_efficiency_loss_from_fouling(
    U_fouled: float,
    U_clean: float,
    design_effectiveness: float = 0.85,
    track_provenance: bool = False
) -> Union[float, Tuple[float, CalculationProvenance]]:
    """
    Calculate heat transfer efficiency loss due to fouling.

    This quantifies the reduction in heat exchanger performance
    caused by fouling deposits.

    Methodology:
        For a given NTU, effectiveness decreases as U decreases.
        Approximate efficiency loss:

        eta_loss = (1 - U_fouled/U_clean) * design_effectiveness * 100%

        This simplified formula provides a conservative estimate
        of efficiency degradation.

    Reference: ASME PTC 4.3

    Args:
        U_fouled: Current (fouled) U-value (BTU/(hr-ft2-F))
        U_clean: Clean (design) U-value (BTU/(hr-ft2-F))
        design_effectiveness: Design point effectiveness (0-1)
        track_provenance: If True, return provenance record

    Returns:
        Efficiency loss as percentage points, optionally with provenance

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_positive(U_fouled, "U_fouled")
    validate_positive(U_clean, "U_clean")

    if not 0 < design_effectiveness <= 1:
        raise ValidationError(
            parameter="design_effectiveness",
            value=design_effectiveness,
            message="Design effectiveness must be between 0 and 1"
        )

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.FOULING,
            formula_id="efficiency_loss_fouling",
            formula_version="1.0.0",
            inputs={
                "U_fouled": U_fouled,
                "U_clean": U_clean,
                "design_effectiveness": design_effectiveness
            }
        )

    # Calculate U-value ratio
    U_ratio = U_fouled / U_clean

    if tracker:
        tracker.add_step(
            operation="divide",
            description="Calculate U-value ratio",
            inputs={"U_fouled": U_fouled, "U_clean": U_clean},
            output_name="U_ratio",
            output_value=U_ratio,
            formula="U_ratio = U_fouled / U_clean"
        )

    # Calculate efficiency loss
    eta_loss = (1 - U_ratio) * design_effectiveness * 100

    if tracker:
        tracker.add_step(
            operation="calculate",
            description="Calculate efficiency loss percentage",
            inputs={"U_ratio": U_ratio, "design_effectiveness": design_effectiveness},
            output_name="eta_loss",
            output_value=eta_loss,
            formula="eta_loss = (1 - U_ratio) * design_effectiveness * 100%"
        )

    eta_loss_rounded = round(max(0, eta_loss), 2)

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=eta_loss_rounded,
            output_unit="percentage_points",
            precision=2
        )
        return eta_loss_rounded, provenance

    return eta_loss_rounded


# =============================================================================
# FUEL PENALTY CALCULATIONS
# =============================================================================

def estimate_fuel_penalty(
    efficiency_loss_percent: float,
    boiler_heat_input: float,
    fuel_cost_per_mmbtu: float = 5.0,
    operating_hours_per_year: float = 8000,
    track_provenance: bool = False
) -> Union[Dict[str, float], Tuple[Dict[str, float], CalculationProvenance]]:
    """
    Estimate fuel penalty due to economizer fouling.

    When the economizer is fouled, less heat is recovered from flue gas,
    resulting in higher fuel consumption to maintain steam output.

    Methodology:
        Additional fuel required to compensate for reduced heat recovery:

        Fuel_penalty (MMBtu/hr) = Heat_input * (eta_loss / 100)
        Annual_cost ($) = Fuel_penalty * fuel_cost * operating_hours

    Reference: EPRI Guidelines

    Args:
        efficiency_loss_percent: Economizer efficiency loss (percentage points)
        boiler_heat_input: Boiler heat input (MMBtu/hr)
        fuel_cost_per_mmbtu: Fuel cost ($/MMBtu)
        operating_hours_per_year: Annual operating hours
        track_provenance: If True, return provenance record

    Returns:
        Dictionary with fuel_penalty_mmbtu_hr, cost_per_hour, annual_cost

    Raises:
        ValidationError: If inputs are invalid
    """
    validate_non_negative(efficiency_loss_percent, "efficiency_loss_percent")
    validate_positive(boiler_heat_input, "boiler_heat_input")
    validate_positive(fuel_cost_per_mmbtu, "fuel_cost_per_mmbtu")
    validate_positive(operating_hours_per_year, "operating_hours_per_year")

    tracker = ProvenanceTracker() if track_provenance else None
    if tracker:
        tracker.start_calculation(
            calculation_type=CalculationType.FOULING,
            formula_id="fuel_penalty",
            formula_version="1.0.0",
            inputs={
                "efficiency_loss_percent": efficiency_loss_percent,
                "boiler_heat_input": boiler_heat_input,
                "fuel_cost_per_mmbtu": fuel_cost_per_mmbtu,
                "operating_hours_per_year": operating_hours_per_year
            }
        )

    # Calculate fuel penalty (MMBtu/hr)
    fuel_penalty_mmbtu_hr = boiler_heat_input * (efficiency_loss_percent / 100)

    if tracker:
        tracker.add_step(
            operation="multiply",
            description="Calculate fuel penalty rate",
            inputs={
                "heat_input": boiler_heat_input,
                "efficiency_loss": efficiency_loss_percent
            },
            output_name="fuel_penalty_mmbtu_hr",
            output_value=fuel_penalty_mmbtu_hr,
            formula="Fuel_penalty = Heat_input * (eta_loss / 100)"
        )

    # Calculate hourly cost
    cost_per_hour = fuel_penalty_mmbtu_hr * fuel_cost_per_mmbtu

    if tracker:
        tracker.add_step(
            operation="multiply",
            description="Calculate hourly fuel cost penalty",
            inputs={
                "fuel_penalty": fuel_penalty_mmbtu_hr,
                "fuel_cost": fuel_cost_per_mmbtu
            },
            output_name="cost_per_hour",
            output_value=cost_per_hour,
            formula="Cost_hr = Fuel_penalty * fuel_cost"
        )

    # Calculate annual cost
    annual_cost = cost_per_hour * operating_hours_per_year

    if tracker:
        tracker.add_step(
            operation="multiply",
            description="Calculate annual fuel cost penalty",
            inputs={
                "cost_per_hour": cost_per_hour,
                "operating_hours": operating_hours_per_year
            },
            output_name="annual_cost",
            output_value=annual_cost,
            formula="Annual_cost = Cost_hr * operating_hours"
        )

    result = {
        "fuel_penalty_mmbtu_hr": round(fuel_penalty_mmbtu_hr, 4),
        "cost_per_hour": round(cost_per_hour, 2),
        "annual_cost": round(annual_cost, 2)
    }

    if track_provenance:
        provenance = tracker.complete_calculation(
            output_value=annual_cost,
            output_unit="$/year",
            precision=2
        )
        return result, provenance

    return result


# =============================================================================
# FOULING ASSESSMENT
# =============================================================================

def assess_fouling_severity(
    fouling_factor: float,
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Union[str, float, bool]]:
    """
    Assess fouling severity and recommend action.

    Args:
        fouling_factor: Current fouling factor ((hr-ft2-F)/BTU)
        thresholds: Custom thresholds (default: FOULING_THRESHOLDS)

    Returns:
        Dictionary with severity assessment and recommendations
    """
    if thresholds is None:
        thresholds = FOULING_THRESHOLDS

    if fouling_factor >= thresholds.get("critical", 0.010):
        severity = "CRITICAL"
        recommendation = "Immediate cleaning required"
        cleaning_needed = True
    elif fouling_factor >= thresholds.get("high", 0.005):
        severity = "HIGH"
        recommendation = "Schedule cleaning within 1 week"
        cleaning_needed = True
    elif fouling_factor >= thresholds.get("moderate", 0.003):
        severity = "MODERATE"
        recommendation = "Monitor closely, plan cleaning"
        cleaning_needed = False
    elif fouling_factor >= thresholds.get("normal", 0.001):
        severity = "NORMAL"
        recommendation = "Continue normal monitoring"
        cleaning_needed = False
    else:
        severity = "CLEAN"
        recommendation = "Excellent condition"
        cleaning_needed = False

    return {
        "fouling_factor": fouling_factor,
        "severity": severity,
        "recommendation": recommendation,
        "cleaning_needed": cleaning_needed,
        "percent_of_critical": round(fouling_factor / thresholds.get("critical", 0.010) * 100, 1)
    }
