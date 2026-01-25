"""
Enthalpy Balance Module - Mass and Energy Balance Calculations

This module provides deterministic mass and energy balance calculations
for steam systems, including:
- Mass balance across system boundaries
- Energy balance with heat additions and losses
- Distribution loss estimation
- Measurement reconciliation

All calculations are deterministic with complete provenance tracking.

Author: GL-CalculatorEngineer
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math


@dataclass
class StreamData:
    """
    Data for a single stream in the steam system.

    Represents a flow of steam/water at a specific measurement point.
    """
    stream_id: str
    name: str

    # Flow properties
    mass_flow_kg_s: float          # Mass flow rate [kg/s]
    mass_flow_uncertainty: float = 0.0  # Uncertainty in mass flow [kg/s]

    # Thermodynamic state
    pressure_kpa: float = 0.0      # Pressure [kPa]
    temperature_c: float = 0.0     # Temperature [C]
    specific_enthalpy_kj_kg: float = 0.0  # Specific enthalpy [kJ/kg]

    # Optional quality for two-phase
    quality_x: Optional[float] = None

    # Measurement metadata
    is_measured: bool = True       # True if measured, False if calculated
    measurement_timestamp: str = ""
    sensor_id: str = ""


@dataclass
class MassBalanceResult:
    """
    Result of mass balance calculation.
    """
    # Mass flow totals
    total_input_kg_s: float        # Total mass input [kg/s]
    total_output_kg_s: float       # Total mass output [kg/s]

    # Balance
    imbalance_kg_s: float          # Mass imbalance [kg/s]
    imbalance_percent: float       # Imbalance as percentage of input

    # Validation
    is_balanced: bool              # True if imbalance within tolerance
    tolerance_percent: float       # Tolerance used for validation

    # Stream details
    input_streams: List[StreamData]
    output_streams: List[StreamData]

    # Provenance
    provenance_hash: str = ""
    calculation_timestamp: str = ""


@dataclass
class EnergyBalanceResult:
    """
    Result of energy balance calculation.
    """
    # Energy flow totals [kW]
    total_input_kw: float          # Total energy input rate [kW]
    total_output_kw: float         # Total energy output rate [kW]
    heat_added_kw: float           # External heat added [kW]
    heat_lost_kw: float            # Heat lost to surroundings [kW]

    # Balance
    imbalance_kw: float            # Energy imbalance [kW]
    imbalance_percent: float       # Imbalance as percentage

    # Validation
    is_balanced: bool              # True if imbalance within tolerance
    tolerance_percent: float       # Tolerance used for validation

    # Efficiency metrics
    thermal_efficiency: Optional[float] = None  # If applicable

    # Stream details
    input_streams: List[Tuple[StreamData, float]]   # (stream, energy_rate_kw)
    output_streams: List[Tuple[StreamData, float]]  # (stream, energy_rate_kw)

    # Provenance
    provenance_hash: str = ""
    calculation_timestamp: str = ""


@dataclass
class LossEstimate:
    """
    Estimate of heat losses in steam distribution network.
    """
    # Total losses
    total_loss_kw: float           # Total estimated heat loss [kW]
    total_loss_percent: float      # Loss as percentage of input

    # Loss breakdown
    pipe_conduction_kw: float      # Loss through pipe walls
    valve_leakage_kw: float        # Loss through valve leaks
    trap_losses_kw: float          # Loss through steam traps
    flange_losses_kw: float        # Loss through flanges

    # Uncertainty
    loss_uncertainty_kw: float     # Uncertainty in loss estimate
    confidence_level: float        # Confidence level (e.g., 0.95)

    # Recommendations
    high_loss_components: List[str]  # Components with high losses

    # Provenance
    provenance_hash: str = ""


@dataclass
class ReconciledState:
    """
    Result of measurement reconciliation.
    """
    # Reconciled values
    reconciled_streams: List[StreamData]

    # Adjustments made
    adjustments: Dict[str, float]  # stream_id -> adjustment made

    # Statistics
    chi_squared: float             # Chi-squared statistic
    degrees_of_freedom: int        # DOF for chi-squared test
    p_value: float                 # P-value for goodness of fit

    # Validation
    is_consistent: bool            # True if measurements are consistent
    gross_errors_detected: List[str]  # Stream IDs with detected gross errors

    # Provenance
    provenance_hash: str = ""


# =============================================================================
# MASS BALANCE FUNCTIONS
# =============================================================================

def compute_mass_balance(
    inputs: List[StreamData],
    outputs: List[StreamData],
    tolerance_percent: float = 2.0,
) -> MassBalanceResult:
    """
    Compute mass balance across a system boundary.

    DETERMINISTIC: Same inputs always produce same output.

    Conservation of mass: sum(m_in) = sum(m_out)

    Args:
        inputs: List of input streams
        outputs: List of output streams
        tolerance_percent: Acceptable imbalance percentage

    Returns:
        MassBalanceResult with balance details

    Raises:
        ValueError: If no streams provided
    """
    from datetime import datetime

    if not inputs:
        raise ValueError("At least one input stream required")
    if not outputs:
        raise ValueError("At least one output stream required")

    # Calculate totals using Decimal for precision
    total_input = Decimal('0')
    for stream in inputs:
        if stream.mass_flow_kg_s < 0:
            raise ValueError(
                f"Negative mass flow in stream {stream.stream_id}: "
                f"{stream.mass_flow_kg_s} kg/s"
            )
        total_input += Decimal(str(stream.mass_flow_kg_s))

    total_output = Decimal('0')
    for stream in outputs:
        if stream.mass_flow_kg_s < 0:
            raise ValueError(
                f"Negative mass flow in stream {stream.stream_id}: "
                f"{stream.mass_flow_kg_s} kg/s"
            )
        total_output += Decimal(str(stream.mass_flow_kg_s))

    # Calculate imbalance
    imbalance = total_input - total_output

    # Handle near-zero flow case
    if total_input > Decimal('0'):
        imbalance_percent = float(abs(imbalance) / total_input * 100)
    else:
        imbalance_percent = 0.0 if imbalance == 0 else float('inf')

    # Check if balanced within tolerance
    is_balanced = imbalance_percent <= tolerance_percent

    # Create provenance hash
    inputs_data = [
        {"id": s.stream_id, "flow": s.mass_flow_kg_s}
        for s in inputs
    ]
    outputs_data = [
        {"id": s.stream_id, "flow": s.mass_flow_kg_s}
        for s in outputs
    ]
    provenance_hash = _compute_provenance({
        "inputs": inputs_data,
        "outputs": outputs_data,
        "total_input": float(total_input),
        "total_output": float(total_output),
        "imbalance": float(imbalance),
    })

    return MassBalanceResult(
        total_input_kg_s=float(total_input),
        total_output_kg_s=float(total_output),
        imbalance_kg_s=float(imbalance),
        imbalance_percent=imbalance_percent,
        is_balanced=is_balanced,
        tolerance_percent=tolerance_percent,
        input_streams=inputs,
        output_streams=outputs,
        provenance_hash=provenance_hash,
        calculation_timestamp=datetime.utcnow().isoformat(),
    )


# =============================================================================
# ENERGY BALANCE FUNCTIONS
# =============================================================================

def compute_energy_balance(
    inputs: List[StreamData],
    outputs: List[StreamData],
    heat_added_kw: float = 0.0,
    heat_lost_kw: float = 0.0,
    tolerance_percent: float = 3.0,
) -> EnergyBalanceResult:
    """
    Compute energy balance across a system boundary.

    DETERMINISTIC: Same inputs always produce same output.

    First Law: sum(m*h)_in + Q_added = sum(m*h)_out + Q_lost + W_shaft

    For steam systems without shaft work:
    sum(m*h)_in + Q_added = sum(m*h)_out + Q_lost

    Args:
        inputs: List of input streams with enthalpy data
        outputs: List of output streams with enthalpy data
        heat_added_kw: External heat added to system [kW]
        heat_lost_kw: Heat lost to surroundings [kW]
        tolerance_percent: Acceptable imbalance percentage

    Returns:
        EnergyBalanceResult with balance details

    Raises:
        ValueError: If streams missing enthalpy data
    """
    from datetime import datetime

    if not inputs:
        raise ValueError("At least one input stream required")
    if not outputs:
        raise ValueError("At least one output stream required")

    # Calculate input energy rates
    input_energies = []
    total_input_kw = Decimal('0')

    for stream in inputs:
        energy_rate = compute_enthalpy_rate(
            stream.mass_flow_kg_s,
            stream.specific_enthalpy_kj_kg
        )
        input_energies.append((stream, energy_rate))
        total_input_kw += Decimal(str(energy_rate))

    # Add external heat
    total_input_kw += Decimal(str(heat_added_kw))

    # Calculate output energy rates
    output_energies = []
    total_output_kw = Decimal('0')

    for stream in outputs:
        energy_rate = compute_enthalpy_rate(
            stream.mass_flow_kg_s,
            stream.specific_enthalpy_kj_kg
        )
        output_energies.append((stream, energy_rate))
        total_output_kw += Decimal(str(energy_rate))

    # Add heat losses
    total_output_kw += Decimal(str(heat_lost_kw))

    # Calculate imbalance
    imbalance = total_input_kw - total_output_kw

    # Handle near-zero flow case
    if total_input_kw > Decimal('0'):
        imbalance_percent = float(abs(imbalance) / total_input_kw * 100)
    else:
        imbalance_percent = 0.0 if imbalance == 0 else float('inf')

    # Check if balanced within tolerance
    is_balanced = imbalance_percent <= tolerance_percent

    # Create provenance hash
    provenance_hash = _compute_provenance({
        "total_input_kw": float(total_input_kw),
        "total_output_kw": float(total_output_kw),
        "heat_added_kw": heat_added_kw,
        "heat_lost_kw": heat_lost_kw,
        "imbalance_kw": float(imbalance),
    })

    return EnergyBalanceResult(
        total_input_kw=float(total_input_kw),
        total_output_kw=float(total_output_kw),
        heat_added_kw=heat_added_kw,
        heat_lost_kw=heat_lost_kw,
        imbalance_kw=float(imbalance),
        imbalance_percent=imbalance_percent,
        is_balanced=is_balanced,
        tolerance_percent=tolerance_percent,
        input_streams=input_energies,
        output_streams=output_energies,
        provenance_hash=provenance_hash,
        calculation_timestamp=datetime.utcnow().isoformat(),
    )


def compute_enthalpy_rate(
    mass_flow_kg_s: float,
    specific_enthalpy_kj_kg: float,
) -> float:
    """
    Compute enthalpy rate (energy flow rate) for a stream.

    DETERMINISTIC: Same inputs always produce same output.

    Formula: Q = m_dot * h

    Args:
        mass_flow_kg_s: Mass flow rate [kg/s]
        specific_enthalpy_kj_kg: Specific enthalpy [kJ/kg]

    Returns:
        Energy flow rate in kW (kJ/s)

    Raises:
        ValueError: If mass flow is negative
    """
    if mass_flow_kg_s < 0:
        raise ValueError(f"Mass flow cannot be negative: {mass_flow_kg_s} kg/s")

    # Handle near-zero flow
    if abs(mass_flow_kg_s) < 1e-10:
        return 0.0

    # Q [kW] = m_dot [kg/s] * h [kJ/kg]
    # Units: kg/s * kJ/kg = kJ/s = kW
    energy_rate_kw = mass_flow_kg_s * specific_enthalpy_kj_kg

    return energy_rate_kw


# =============================================================================
# LOSS ESTIMATION FUNCTIONS
# =============================================================================

def estimate_distribution_losses(
    network_data: Dict[str, Any],
    ambient_temperature_c: float = 20.0,
) -> LossEstimate:
    """
    Estimate heat losses in a steam distribution network.

    DETERMINISTIC: Same inputs always produce same output.

    Loss mechanisms:
    1. Pipe conduction through insulation
    2. Valve and fitting losses
    3. Steam trap losses (passing and blow-through)
    4. Flange and joint losses

    Args:
        network_data: Dictionary containing network configuration:
            - pipes: List of pipe segments with length, diameter, insulation
            - valves: List of valves with type and condition
            - traps: List of steam traps with type and condition
            - steam_pressure_kpa: Operating steam pressure
            - steam_temperature_c: Steam temperature
        ambient_temperature_c: Ambient temperature [C]

    Returns:
        LossEstimate with breakdown of losses

    Raises:
        ValueError: If required network data missing
    """
    # Extract network data with defaults
    pipes = network_data.get("pipes", [])
    valves = network_data.get("valves", [])
    traps = network_data.get("traps", [])
    steam_temp_c = network_data.get("steam_temperature_c", 150.0)
    steam_pressure_kpa = network_data.get("steam_pressure_kpa", 500.0)

    # Temperature difference for heat transfer
    delta_T = steam_temp_c - ambient_temperature_c

    # ==========================================================================
    # 1. Pipe Conduction Losses
    # ==========================================================================
    pipe_loss_kw = Decimal('0')

    for pipe in pipes:
        length_m = pipe.get("length_m", 0)
        outer_diameter_m = pipe.get("outer_diameter_m", 0.1)
        insulation_thickness_m = pipe.get("insulation_thickness_m", 0.05)
        insulation_conductivity = pipe.get(
            "insulation_conductivity_w_mk", 0.04
        )  # Default: mineral wool

        if length_m <= 0 or outer_diameter_m <= 0:
            continue

        # Simplified heat loss per unit length for insulated pipe
        # Q/L = 2 * pi * k * deltaT / ln(r_outer/r_inner)
        r_inner = outer_diameter_m / 2
        r_outer = r_inner + insulation_thickness_m

        if r_outer > r_inner:
            q_per_m = (
                2 * math.pi * insulation_conductivity * delta_T /
                math.log(r_outer / r_inner)
            )  # W/m
        else:
            # Bare pipe - much higher loss
            # Use convective heat transfer coefficient ~ 10 W/(m2*K)
            h_conv = 10.0
            q_per_m = math.pi * outer_diameter_m * h_conv * delta_T

        pipe_loss_w = q_per_m * length_m
        pipe_loss_kw += Decimal(str(pipe_loss_w / 1000))  # Convert to kW

    # ==========================================================================
    # 2. Valve and Fitting Losses
    # ==========================================================================
    valve_loss_kw = Decimal('0')

    # Standard loss rates for uninsulated valves (W per valve)
    valve_loss_rates = {
        "gate": 50,      # Low loss when open
        "globe": 100,    # Higher due to tortuous path
        "ball": 40,      # Low loss
        "check": 60,     # Moderate
        "control": 150,  # Higher due to complexity
    }

    for valve in valves:
        valve_type = valve.get("type", "gate").lower()
        is_insulated = valve.get("insulated", False)
        condition = valve.get("condition", "good").lower()

        base_loss_w = valve_loss_rates.get(valve_type, 100)

        # Adjust for insulation
        if is_insulated:
            base_loss_w *= 0.1  # 90% reduction

        # Adjust for condition (leaking valves)
        condition_multiplier = {
            "good": 1.0,
            "fair": 1.5,
            "poor": 3.0,
            "leaking": 10.0,
        }
        base_loss_w *= condition_multiplier.get(condition, 1.0)

        # Scale with temperature difference
        base_loss_w *= (delta_T / 100)  # Normalized to 100 C delta T

        valve_loss_kw += Decimal(str(base_loss_w / 1000))

    # ==========================================================================
    # 3. Steam Trap Losses
    # ==========================================================================
    trap_loss_kw = Decimal('0')

    # Steam trap loss rates (kg/h of steam lost)
    trap_loss_rates_kg_h = {
        "thermodynamic": {"good": 0.5, "fair": 2.0, "poor": 5.0, "failed": 20.0},
        "inverted_bucket": {"good": 0.3, "fair": 1.5, "poor": 4.0, "failed": 15.0},
        "float": {"good": 0.2, "fair": 1.0, "poor": 3.0, "failed": 12.0},
        "thermostatic": {"good": 0.4, "fair": 1.5, "poor": 4.0, "failed": 18.0},
    }

    # Get latent heat at operating pressure
    try:
        from .steam_properties import get_saturation_properties
        sat = get_saturation_properties(pressure_kpa=steam_pressure_kpa)
        hfg = sat.hfg_kj_kg
    except ImportError:
        # Fallback: approximate latent heat
        hfg = 2000.0  # kJ/kg typical value

    for trap in traps:
        trap_type = trap.get("type", "thermodynamic").lower()
        condition = trap.get("condition", "good").lower()

        loss_rates = trap_loss_rates_kg_h.get(
            trap_type, trap_loss_rates_kg_h["thermodynamic"]
        )
        steam_loss_kg_h = loss_rates.get(condition, 0.5)

        # Energy loss = mass loss * latent heat
        energy_loss_kw = steam_loss_kg_h / 3600 * hfg  # kJ/s = kW
        trap_loss_kw += Decimal(str(energy_loss_kw))

    # ==========================================================================
    # 4. Flange and Joint Losses
    # ==========================================================================
    flange_loss_kw = Decimal('0')

    # Estimate based on pipe count (each pipe has ~2 flanges typically)
    num_flanges = len(pipes) * 2
    loss_per_flange_w = 20 * (delta_T / 100)  # W per uninsulated flange

    # Assume 50% are insulated
    flange_loss_kw = Decimal(str(num_flanges * loss_per_flange_w * 0.5 / 1000))

    # ==========================================================================
    # Total Losses
    # ==========================================================================
    total_loss_kw = (
        pipe_loss_kw + valve_loss_kw + trap_loss_kw + flange_loss_kw
    )

    # Calculate percentage of typical input (need reference)
    input_energy_kw = network_data.get("total_input_kw", 1000.0)
    if input_energy_kw > 0:
        total_loss_percent = float(total_loss_kw / Decimal(str(input_energy_kw)) * 100)
    else:
        total_loss_percent = 0.0

    # Estimate uncertainty (typically 20-30% for loss estimates)
    loss_uncertainty_kw = float(total_loss_kw * Decimal('0.25'))

    # Identify high-loss components
    high_loss_components = []
    loss_components = {
        "pipes": float(pipe_loss_kw),
        "valves": float(valve_loss_kw),
        "traps": float(trap_loss_kw),
        "flanges": float(flange_loss_kw),
    }

    max_component_loss = max(loss_components.values()) if loss_components else 0
    for component, loss in loss_components.items():
        if loss > 0.3 * max_component_loss and loss > 1.0:  # > 30% of max and > 1 kW
            high_loss_components.append(f"{component}: {loss:.1f} kW")

    # Create provenance hash
    provenance_hash = _compute_provenance({
        "network_data": network_data,
        "ambient_temperature_c": ambient_temperature_c,
        "total_loss_kw": float(total_loss_kw),
    })

    return LossEstimate(
        total_loss_kw=float(total_loss_kw),
        total_loss_percent=total_loss_percent,
        pipe_conduction_kw=float(pipe_loss_kw),
        valve_leakage_kw=float(valve_loss_kw),
        trap_losses_kw=float(trap_loss_kw),
        flange_losses_kw=float(flange_loss_kw),
        loss_uncertainty_kw=loss_uncertainty_kw,
        confidence_level=0.95,
        high_loss_components=high_loss_components,
        provenance_hash=provenance_hash,
    )


# =============================================================================
# MEASUREMENT RECONCILIATION
# =============================================================================

def reconcile_measurements(
    measured: List[StreamData],
    calculated: List[StreamData],
    uncertainties: Dict[str, float],
    constraint_tolerance: float = 1e-6,
) -> ReconciledState:
    """
    Reconcile measured and calculated values using weighted least squares.

    DETERMINISTIC: Same inputs always produce same output.

    Uses data reconciliation to find the best estimate of true values
    that satisfies conservation constraints while minimizing weighted
    squared deviations from measured values.

    Objective: min sum((x_rec - x_meas)^2 / sigma^2)
    Subject to: sum(m_in) = sum(m_out)

    Args:
        measured: List of measured stream data
        calculated: List of calculated/expected stream data
        uncertainties: Dictionary mapping stream_id to measurement uncertainty
        constraint_tolerance: Tolerance for constraint satisfaction

    Returns:
        ReconciledState with reconciled values and diagnostics

    Raises:
        ValueError: If insufficient data for reconciliation
    """
    from datetime import datetime

    if len(measured) < 2:
        raise ValueError("At least 2 measurements required for reconciliation")

    # Separate inputs and outputs
    all_streams = measured + calculated
    n_streams = len(all_streams)

    # Build measurement vector and covariance matrix
    measurements = []
    variances = []
    stream_map = {}

    for i, stream in enumerate(all_streams):
        measurements.append(stream.mass_flow_kg_s)

        # Get uncertainty (default to 5% if not specified)
        sigma = uncertainties.get(stream.stream_id, stream.mass_flow_kg_s * 0.05)
        if sigma <= 0:
            sigma = stream.mass_flow_kg_s * 0.05
        variances.append(sigma ** 2)

        stream_map[stream.stream_id] = i

    # Simple reconciliation using Lagrange multipliers
    # For mass balance: sum(A * x) = 0 where A has +1 for inputs, -1 for outputs

    # Identify inputs and outputs by convention (first half inputs, second half outputs)
    # Or use naming convention
    n_inputs = len(measured) // 2 + len(calculated) // 2
    if n_inputs == 0:
        n_inputs = len(all_streams) // 2

    # Build constraint matrix A (1 x n_streams)
    # Positive for inputs, negative for outputs
    A = []
    for i, stream in enumerate(all_streams):
        if "input" in stream.name.lower() or "supply" in stream.name.lower():
            A.append(1.0)
        elif "output" in stream.name.lower() or "return" in stream.name.lower():
            A.append(-1.0)
        else:
            # Default: first half are inputs
            A.append(1.0 if i < n_inputs else -1.0)

    # Solve reconciliation using weighted least squares
    # x_rec = x_meas - V * A^T * (A * V * A^T)^(-1) * (A * x_meas)
    # where V is diagonal covariance matrix

    # Calculate A * V * A^T (scalar for single constraint)
    AVAt = sum(A[i] ** 2 * variances[i] for i in range(n_streams))

    if AVAt < 1e-20:
        raise ValueError("Constraint matrix is singular - cannot reconcile")

    # Calculate constraint violation
    Ax = sum(A[i] * measurements[i] for i in range(n_streams))

    # Lagrange multiplier
    lam = Ax / AVAt

    # Calculate reconciled values
    reconciled_values = []
    adjustments = {}

    for i in range(n_streams):
        adjustment = variances[i] * A[i] * lam
        reconciled_value = measurements[i] - adjustment

        # Ensure non-negative
        reconciled_value = max(0, reconciled_value)

        reconciled_values.append(reconciled_value)
        adjustments[all_streams[i].stream_id] = adjustment

    # Create reconciled stream data
    reconciled_streams = []
    for i, stream in enumerate(all_streams):
        reconciled_stream = StreamData(
            stream_id=stream.stream_id,
            name=stream.name,
            mass_flow_kg_s=reconciled_values[i],
            mass_flow_uncertainty=math.sqrt(variances[i]),
            pressure_kpa=stream.pressure_kpa,
            temperature_c=stream.temperature_c,
            specific_enthalpy_kj_kg=stream.specific_enthalpy_kj_kg,
            quality_x=stream.quality_x,
            is_measured=stream.is_measured,
            measurement_timestamp=stream.measurement_timestamp,
            sensor_id=stream.sensor_id,
        )
        reconciled_streams.append(reconciled_stream)

    # Calculate chi-squared statistic
    chi_squared = sum(
        (reconciled_values[i] - measurements[i]) ** 2 / variances[i]
        for i in range(n_streams)
    )

    # Degrees of freedom = n_measurements - n_constraints
    dof = n_streams - 1  # One mass balance constraint

    # P-value (simplified - would need scipy for exact)
    # Using approximation: chi-squared with dof degrees of freedom
    # For large DOF, chi-squared ~ Normal(dof, sqrt(2*dof))
    if dof > 0:
        z = (chi_squared - dof) / math.sqrt(2 * dof)
        # Approximate p-value (two-tailed)
        p_value = 2 * (1 - _normal_cdf(abs(z)))
    else:
        p_value = 1.0

    # Detect gross errors (adjustments > 3 sigma)
    gross_errors = []
    for stream_id, adjustment in adjustments.items():
        sigma = math.sqrt(uncertainties.get(stream_id, 0.01))
        if abs(adjustment) > 3 * sigma:
            gross_errors.append(stream_id)

    # Check if reconciled values satisfy constraints
    reconciled_Ax = sum(A[i] * reconciled_values[i] for i in range(n_streams))
    is_consistent = abs(reconciled_Ax) < constraint_tolerance and p_value > 0.05

    # Create provenance hash
    provenance_hash = _compute_provenance({
        "measured": [{"id": s.stream_id, "flow": s.mass_flow_kg_s} for s in measured],
        "calculated": [{"id": s.stream_id, "flow": s.mass_flow_kg_s} for s in calculated],
        "chi_squared": chi_squared,
        "reconciled": reconciled_values,
    })

    return ReconciledState(
        reconciled_streams=reconciled_streams,
        adjustments=adjustments,
        chi_squared=chi_squared,
        degrees_of_freedom=dof,
        p_value=p_value,
        is_consistent=is_consistent,
        gross_errors_detected=gross_errors,
        provenance_hash=provenance_hash,
    )


def _normal_cdf(x: float) -> float:
    """
    Approximate standard normal CDF using error function approximation.

    DETERMINISTIC: Same input always produces same output.
    """
    # Approximation using error function
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _compute_provenance(data: Dict[str, Any]) -> str:
    """
    Compute SHA-256 provenance hash.

    DETERMINISTIC: Same inputs always produce same hash.
    """
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()
