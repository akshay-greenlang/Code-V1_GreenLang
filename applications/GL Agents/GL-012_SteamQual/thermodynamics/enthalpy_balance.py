"""
Enthalpy Balance Module for GL-012_SteamQual

This module provides enthalpy balance calculations across steam system components
for the SteamQual agent. It supports:
- Component-level enthalpy tracking
- System-wide energy balance
- Loss estimation and reconciliation
- Quality impact on enthalpy flows

All calculations are DETERMINISTIC with complete provenance tracking for
zero-hallucination compliance.

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import hashlib
import json
import math
import logging
from datetime import datetime

from .steam_properties import (
    get_saturation_properties,
    get_saturation_temperature,
    compute_steam_properties,
    SteamState,
)
from .iapws_wrapper import (
    kpa_to_mpa,
    mpa_to_kpa,
    celsius_to_kelvin,
    kelvin_to_celsius,
    compute_provenance_hash,
    region4_mixture_enthalpy,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ComponentType(Enum):
    """Steam system component types."""
    BOILER = "boiler"
    SUPERHEATER = "superheater"
    DESUPERHEATER = "desuperheater"
    STEAM_HEADER = "steam_header"
    PRESSURE_REDUCING_VALVE = "prv"
    HEAT_EXCHANGER = "heat_exchanger"
    CONDENSER = "condenser"
    FLASH_TANK = "flash_tank"
    STEAM_TRAP = "steam_trap"
    DEAERATOR = "deaerator"
    ECONOMIZER = "economizer"
    TURBINE = "turbine"


class BalanceStatus(Enum):
    """Status of balance calculation."""
    BALANCED = "balanced"
    IMBALANCED = "imbalanced"
    WARNING = "warning"
    ERROR = "error"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class StreamState:
    """
    Thermodynamic state of a steam/water stream.

    Attributes:
        stream_id: Unique identifier for the stream
        name: Human-readable stream name
        mass_flow_kg_s: Mass flow rate in kg/s
        pressure_kpa: Pressure in kPa
        temperature_c: Temperature in Celsius
        specific_enthalpy_kj_kg: Specific enthalpy in kJ/kg
        quality_x: Steam quality (0-1, None if single phase)
        is_measured: True if values are measured, False if calculated
    """
    stream_id: str
    name: str
    mass_flow_kg_s: float
    pressure_kpa: float
    temperature_c: float
    specific_enthalpy_kj_kg: float
    quality_x: Optional[float] = None
    is_measured: bool = True
    uncertainty_flow: float = 0.0
    uncertainty_enthalpy: float = 0.0


@dataclass
class ComponentBalance:
    """
    Enthalpy balance for a single component.

    Attributes:
        component_id: Unique component identifier
        component_type: Type of component
        input_streams: List of input streams
        output_streams: List of output streams
        heat_input_kw: External heat input (e.g., combustion)
        heat_loss_kw: Heat loss to surroundings
        work_output_kw: Shaft work output (e.g., turbine)
        enthalpy_in_kw: Total input enthalpy rate
        enthalpy_out_kw: Total output enthalpy rate
        imbalance_kw: Energy imbalance
        imbalance_percent: Imbalance as percentage
        status: Balance status
        provenance_hash: SHA-256 hash for audit trail
    """
    component_id: str
    component_type: ComponentType
    input_streams: List[StreamState]
    output_streams: List[StreamState]
    heat_input_kw: float
    heat_loss_kw: float
    work_output_kw: float
    enthalpy_in_kw: float
    enthalpy_out_kw: float
    imbalance_kw: float
    imbalance_percent: float
    status: BalanceStatus
    provenance_hash: str = ""
    calculation_timestamp: str = ""


@dataclass
class SystemBalance:
    """
    System-wide enthalpy balance result.

    Attributes:
        component_balances: List of component-level balances
        total_mass_in_kg_s: Total mass input to system
        total_mass_out_kg_s: Total mass output from system
        total_enthalpy_in_kw: Total enthalpy input rate
        total_enthalpy_out_kw: Total enthalpy output rate
        total_heat_input_kw: Total external heat input
        total_heat_loss_kw: Total heat loss
        total_work_output_kw: Total work output
        system_imbalance_kw: Overall system imbalance
        system_imbalance_percent: Imbalance percentage
        status: Overall system status
        recommendations: List of recommendations
        provenance_hash: SHA-256 hash for audit trail
    """
    component_balances: List[ComponentBalance]
    total_mass_in_kg_s: float
    total_mass_out_kg_s: float
    total_enthalpy_in_kw: float
    total_enthalpy_out_kw: float
    total_heat_input_kw: float
    total_heat_loss_kw: float
    total_work_output_kw: float
    system_imbalance_kw: float
    system_imbalance_percent: float
    status: BalanceStatus
    recommendations: List[str] = field(default_factory=list)
    provenance_hash: str = ""
    calculation_timestamp: str = ""


@dataclass
class QualityImpact:
    """
    Impact of steam quality on enthalpy balance.

    Attributes:
        stream_id: Stream identifier
        actual_quality: Actual steam quality
        design_quality: Design/expected quality
        enthalpy_deviation_kj_kg: Enthalpy deviation per kg
        enthalpy_deviation_kw: Total enthalpy deviation rate
        cost_impact_per_hour: Estimated cost impact
    """
    stream_id: str
    actual_quality: float
    design_quality: float
    enthalpy_deviation_kj_kg: float
    enthalpy_deviation_kw: float
    cost_impact_per_hour: float = 0.0


# =============================================================================
# ENTHALPY CALCULATION FUNCTIONS
# =============================================================================

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

    if abs(mass_flow_kg_s) < 1e-10:
        return 0.0

    energy_rate_kw = mass_flow_kg_s * specific_enthalpy_kj_kg
    return energy_rate_kw


def compute_stream_enthalpy(stream: StreamState) -> float:
    """
    Compute enthalpy rate for a stream.

    DETERMINISTIC: Same input produces same output.

    Args:
        stream: StreamState with flow and enthalpy data

    Returns:
        Enthalpy rate in kW
    """
    return compute_enthalpy_rate(
        stream.mass_flow_kg_s,
        stream.specific_enthalpy_kj_kg,
    )


def estimate_specific_enthalpy(
    pressure_kpa: float,
    temperature_c: float,
    quality_x: Optional[float] = None,
) -> float:
    """
    Estimate specific enthalpy from P, T, and optionally quality.

    DETERMINISTIC: Same inputs produce same output.

    Args:
        pressure_kpa: Pressure in kPa
        temperature_c: Temperature in Celsius
        quality_x: Steam quality (if two-phase)

    Returns:
        Specific enthalpy in kJ/kg
    """
    props = compute_steam_properties(
        pressure_kpa=pressure_kpa,
        temperature_c=temperature_c,
        quality_x=quality_x,
    )
    return props.specific_enthalpy_kj_kg


def compute_enthalpy_change(
    inlet_stream: StreamState,
    outlet_stream: StreamState,
) -> float:
    """
    Compute enthalpy change across a component.

    DETERMINISTIC: Same inputs produce same output.

    Args:
        inlet_stream: Inlet stream state
        outlet_stream: Outlet stream state

    Returns:
        Enthalpy change rate in kW (positive = heat addition)
    """
    h_in = compute_stream_enthalpy(inlet_stream)
    h_out = compute_stream_enthalpy(outlet_stream)
    return h_out - h_in


# =============================================================================
# COMPONENT BALANCE FUNCTIONS
# =============================================================================

def compute_component_balance(
    component_id: str,
    component_type: ComponentType,
    input_streams: List[StreamState],
    output_streams: List[StreamState],
    heat_input_kw: float = 0.0,
    heat_loss_kw: float = 0.0,
    work_output_kw: float = 0.0,
    tolerance_percent: float = 3.0,
) -> ComponentBalance:
    """
    Compute enthalpy balance for a single component.

    DETERMINISTIC: Same inputs always produce same output.

    First Law for open system:
        sum(m*h)_in + Q_in = sum(m*h)_out + Q_loss + W_out

    Args:
        component_id: Unique component identifier
        component_type: Type of component
        input_streams: List of input stream states
        output_streams: List of output stream states
        heat_input_kw: External heat input (e.g., combustion) [kW]
        heat_loss_kw: Heat loss to surroundings [kW]
        work_output_kw: Shaft work output (turbine) [kW]
        tolerance_percent: Acceptable imbalance tolerance

    Returns:
        ComponentBalance with balance details
    """
    # Calculate input enthalpy rate
    enthalpy_in = Decimal('0')
    for stream in input_streams:
        h_rate = compute_stream_enthalpy(stream)
        enthalpy_in += Decimal(str(h_rate))

    # Add external heat input
    enthalpy_in += Decimal(str(heat_input_kw))

    # Calculate output enthalpy rate
    enthalpy_out = Decimal('0')
    for stream in output_streams:
        h_rate = compute_stream_enthalpy(stream)
        enthalpy_out += Decimal(str(h_rate))

    # Add heat loss and work output
    enthalpy_out += Decimal(str(heat_loss_kw))
    enthalpy_out += Decimal(str(work_output_kw))

    # Calculate imbalance
    imbalance = enthalpy_in - enthalpy_out

    # Calculate percentage (relative to input)
    if enthalpy_in > Decimal('0'):
        imbalance_percent = float(abs(imbalance) / enthalpy_in * 100)
    else:
        imbalance_percent = 0.0 if imbalance == 0 else float('inf')

    # Determine status
    if imbalance_percent <= tolerance_percent:
        status = BalanceStatus.BALANCED
    elif imbalance_percent <= tolerance_percent * 2:
        status = BalanceStatus.WARNING
    else:
        status = BalanceStatus.IMBALANCED

    # Create provenance hash
    inputs = {
        "component_id": component_id,
        "input_streams": [s.stream_id for s in input_streams],
        "output_streams": [s.stream_id for s in output_streams],
        "heat_input_kw": heat_input_kw,
        "heat_loss_kw": heat_loss_kw,
        "work_output_kw": work_output_kw,
    }
    outputs = {
        "enthalpy_in_kw": float(enthalpy_in),
        "enthalpy_out_kw": float(enthalpy_out),
        "imbalance_kw": float(imbalance),
    }
    provenance = compute_provenance_hash(inputs, outputs, "component_balance")

    logger.info(
        f"Component {component_id} balance: "
        f"In={float(enthalpy_in):.1f} kW, Out={float(enthalpy_out):.1f} kW, "
        f"Imbalance={float(imbalance):.1f} kW ({imbalance_percent:.2f}%)"
    )

    return ComponentBalance(
        component_id=component_id,
        component_type=component_type,
        input_streams=input_streams,
        output_streams=output_streams,
        heat_input_kw=heat_input_kw,
        heat_loss_kw=heat_loss_kw,
        work_output_kw=work_output_kw,
        enthalpy_in_kw=float(enthalpy_in),
        enthalpy_out_kw=float(enthalpy_out),
        imbalance_kw=float(imbalance),
        imbalance_percent=imbalance_percent,
        status=status,
        provenance_hash=provenance,
        calculation_timestamp=datetime.utcnow().isoformat(),
    )


def compute_system_balance(
    component_balances: List[ComponentBalance],
    boundary_inputs: List[StreamState],
    boundary_outputs: List[StreamState],
    tolerance_percent: float = 5.0,
) -> SystemBalance:
    """
    Compute system-wide enthalpy balance.

    DETERMINISTIC: Same inputs always produce same output.

    Aggregates component balances and checks overall system energy balance.

    Args:
        component_balances: List of component-level balances
        boundary_inputs: Streams entering system boundary
        boundary_outputs: Streams leaving system boundary
        tolerance_percent: Acceptable system imbalance tolerance

    Returns:
        SystemBalance with system-level details
    """
    # Calculate boundary totals
    total_mass_in = sum(s.mass_flow_kg_s for s in boundary_inputs)
    total_mass_out = sum(s.mass_flow_kg_s for s in boundary_outputs)

    total_h_in = sum(compute_stream_enthalpy(s) for s in boundary_inputs)
    total_h_out = sum(compute_stream_enthalpy(s) for s in boundary_outputs)

    # Aggregate component contributions
    total_heat_input = sum(c.heat_input_kw for c in component_balances)
    total_heat_loss = sum(c.heat_loss_kw for c in component_balances)
    total_work_output = sum(c.work_output_kw for c in component_balances)

    # System energy balance
    # Input = Output + Loss + Work
    total_input = total_h_in + total_heat_input
    total_output = total_h_out + total_heat_loss + total_work_output

    system_imbalance = total_input - total_output

    if total_input > 0:
        imbalance_percent = abs(system_imbalance) / total_input * 100
    else:
        imbalance_percent = 0.0 if system_imbalance == 0 else float('inf')

    # Determine status
    if imbalance_percent <= tolerance_percent:
        status = BalanceStatus.BALANCED
    elif imbalance_percent <= tolerance_percent * 2:
        status = BalanceStatus.WARNING
    else:
        status = BalanceStatus.IMBALANCED

    # Generate recommendations
    recommendations = []
    if status != BalanceStatus.BALANCED:
        recommendations.append(
            f"System imbalance of {system_imbalance:.1f} kW detected. "
            "Check for unmeasured losses or measurement errors."
        )

    for cb in component_balances:
        if cb.status == BalanceStatus.IMBALANCED:
            recommendations.append(
                f"Component {cb.component_id} has significant imbalance "
                f"({cb.imbalance_percent:.1f}%). Investigate further."
            )

    # Check mass balance
    mass_imbalance_percent = abs(total_mass_in - total_mass_out) / max(total_mass_in, 0.001) * 100
    if mass_imbalance_percent > 2.0:
        recommendations.append(
            f"Mass imbalance of {mass_imbalance_percent:.1f}% detected. "
            "Check for leaks or unmeasured flows."
        )

    # Provenance
    inputs = {
        "n_components": len(component_balances),
        "boundary_input_streams": [s.stream_id for s in boundary_inputs],
        "boundary_output_streams": [s.stream_id for s in boundary_outputs],
    }
    outputs = {
        "total_h_in_kw": total_h_in,
        "total_h_out_kw": total_h_out,
        "system_imbalance_kw": system_imbalance,
    }
    provenance = compute_provenance_hash(inputs, outputs, "system_balance")

    logger.info(
        f"System balance: In={total_input:.1f} kW, Out={total_output:.1f} kW, "
        f"Imbalance={system_imbalance:.1f} kW ({imbalance_percent:.2f}%)"
    )

    return SystemBalance(
        component_balances=component_balances,
        total_mass_in_kg_s=total_mass_in,
        total_mass_out_kg_s=total_mass_out,
        total_enthalpy_in_kw=total_h_in + total_heat_input,
        total_enthalpy_out_kw=total_h_out + total_heat_loss + total_work_output,
        total_heat_input_kw=total_heat_input,
        total_heat_loss_kw=total_heat_loss,
        total_work_output_kw=total_work_output,
        system_imbalance_kw=system_imbalance,
        system_imbalance_percent=imbalance_percent,
        status=status,
        recommendations=recommendations,
        provenance_hash=provenance,
        calculation_timestamp=datetime.utcnow().isoformat(),
    )


# =============================================================================
# QUALITY IMPACT ANALYSIS
# =============================================================================

def compute_quality_impact(
    stream: StreamState,
    design_quality: float,
) -> QualityImpact:
    """
    Compute the impact of actual vs design steam quality on enthalpy.

    DETERMINISTIC: Same inputs produce same output.

    Args:
        stream: Stream state with actual quality
        design_quality: Design/expected quality

    Returns:
        QualityImpact with deviation analysis
    """
    if stream.quality_x is None:
        raise ValueError(f"Stream {stream.stream_id} has no quality specified")

    actual_x = stream.quality_x
    design_x = design_quality

    # Get saturation properties at stream pressure
    hf, hg, hfg, _, _, _, _, _ = get_saturation_properties(stream.pressure_kpa)

    # Calculate enthalpies at actual and design quality
    h_actual = hf + actual_x * hfg
    h_design = hf + design_x * hfg

    # Enthalpy deviation per kg
    delta_h = h_actual - h_design

    # Total deviation rate
    delta_h_rate = delta_h * stream.mass_flow_kg_s

    # Cost impact (assuming steam cost of $0.03/kWh)
    steam_cost_per_kwh = 0.03
    cost_per_hour = abs(delta_h_rate) * steam_cost_per_kwh

    logger.info(
        f"Quality impact for {stream.stream_id}: "
        f"Actual x={actual_x:.3f}, Design x={design_x:.3f}, "
        f"dH={delta_h:.1f} kJ/kg, dH_rate={delta_h_rate:.1f} kW"
    )

    return QualityImpact(
        stream_id=stream.stream_id,
        actual_quality=actual_x,
        design_quality=design_x,
        enthalpy_deviation_kj_kg=delta_h,
        enthalpy_deviation_kw=delta_h_rate,
        cost_impact_per_hour=cost_per_hour,
    )


def analyze_quality_impacts(
    streams: List[Tuple[StreamState, float]],
) -> List[QualityImpact]:
    """
    Analyze quality impacts for multiple streams.

    DETERMINISTIC: Same inputs produce same output.

    Args:
        streams: List of (StreamState, design_quality) tuples

    Returns:
        List of QualityImpact results
    """
    impacts = []
    for stream, design_quality in streams:
        if stream.quality_x is not None:
            impact = compute_quality_impact(stream, design_quality)
            impacts.append(impact)

    return impacts


# =============================================================================
# SPECIAL COMPONENT BALANCES
# =============================================================================

def compute_flash_tank_balance(
    inlet: StreamState,
    flash_pressure_kpa: float,
    separator_efficiency: float = 0.98,
) -> Tuple[StreamState, StreamState, ComponentBalance]:
    """
    Compute flash tank balance (isenthalpic flash).

    DETERMINISTIC: Same inputs produce same output.

    Flash process is isenthalpic: h_in = h_out_vapor * x + h_out_liquid * (1-x)

    Args:
        inlet: Inlet stream (typically condensate)
        flash_pressure_kpa: Flash pressure in kPa
        separator_efficiency: Vapor/liquid separation efficiency

    Returns:
        Tuple of (vapor_stream, liquid_stream, balance)
    """
    h_in = inlet.specific_enthalpy_kj_kg
    m_in = inlet.mass_flow_kg_s

    # Get saturation properties at flash pressure
    hf, hg, hfg, _, _, _, _, _ = get_saturation_properties(flash_pressure_kpa)
    T_sat = get_saturation_temperature(flash_pressure_kpa)

    # Calculate flash quality (isenthalpic)
    if h_in <= hf:
        # All liquid - no flash
        x_flash = 0.0
    elif h_in >= hg:
        # All vapor - rare case
        x_flash = 1.0
    else:
        # Two-phase flash
        x_flash = (h_in - hf) / hfg

    # Apply separator efficiency
    vapor_fraction = x_flash * separator_efficiency
    liquid_fraction = 1 - vapor_fraction

    # Create output streams
    vapor_stream = StreamState(
        stream_id=f"{inlet.stream_id}_flash_vapor",
        name=f"{inlet.name} Flash Vapor",
        mass_flow_kg_s=m_in * vapor_fraction,
        pressure_kpa=flash_pressure_kpa,
        temperature_c=T_sat,
        specific_enthalpy_kj_kg=hg,
        quality_x=1.0,
        is_measured=False,
    )

    liquid_stream = StreamState(
        stream_id=f"{inlet.stream_id}_flash_liquid",
        name=f"{inlet.name} Flash Liquid",
        mass_flow_kg_s=m_in * liquid_fraction,
        pressure_kpa=flash_pressure_kpa,
        temperature_c=T_sat,
        specific_enthalpy_kj_kg=hf,
        quality_x=0.0,
        is_measured=False,
    )

    # Compute balance
    balance = compute_component_balance(
        component_id=f"flash_tank_{inlet.stream_id}",
        component_type=ComponentType.FLASH_TANK,
        input_streams=[inlet],
        output_streams=[vapor_stream, liquid_stream],
        heat_input_kw=0.0,
        heat_loss_kw=0.0,
        work_output_kw=0.0,
    )

    return vapor_stream, liquid_stream, balance


def compute_prv_balance(
    inlet: StreamState,
    outlet_pressure_kpa: float,
) -> Tuple[StreamState, ComponentBalance]:
    """
    Compute pressure reducing valve balance (isenthalpic process).

    DETERMINISTIC: Same inputs produce same output.

    PRV process is isenthalpic: h_out = h_in

    Args:
        inlet: Inlet stream at high pressure
        outlet_pressure_kpa: Target outlet pressure

    Returns:
        Tuple of (outlet_stream, balance)
    """
    h_out = inlet.specific_enthalpy_kj_kg  # Isenthalpic

    # Determine outlet state
    hf, hg, hfg, _, _, _, _, _ = get_saturation_properties(outlet_pressure_kpa)
    T_sat = get_saturation_temperature(outlet_pressure_kpa)

    if h_out <= hf:
        # Subcooled liquid (unusual for PRV)
        T_out = T_sat - 5  # Approximate
        x_out = None
    elif h_out >= hg:
        # Superheated vapor
        x_out = None
        # Approximate superheat temperature
        T_out = T_sat + (h_out - hg) / 2.0  # Rough Cp of 2 kJ/(kg*K)
    else:
        # Wet steam
        x_out = (h_out - hf) / hfg
        T_out = T_sat

    outlet = StreamState(
        stream_id=f"{inlet.stream_id}_prv_out",
        name=f"{inlet.name} After PRV",
        mass_flow_kg_s=inlet.mass_flow_kg_s,
        pressure_kpa=outlet_pressure_kpa,
        temperature_c=T_out,
        specific_enthalpy_kj_kg=h_out,
        quality_x=x_out,
        is_measured=False,
    )

    balance = compute_component_balance(
        component_id=f"prv_{inlet.stream_id}",
        component_type=ComponentType.PRESSURE_REDUCING_VALVE,
        input_streams=[inlet],
        output_streams=[outlet],
        heat_input_kw=0.0,
        heat_loss_kw=0.0,
        work_output_kw=0.0,
    )

    return outlet, balance


def compute_desuperheater_balance(
    steam_inlet: StreamState,
    water_inlet: StreamState,
    target_temperature_c: float,
) -> Tuple[StreamState, ComponentBalance]:
    """
    Compute desuperheater balance (adiabatic mixing).

    DETERMINISTIC: Same inputs produce same output.

    Energy balance: m_steam * h_steam + m_water * h_water = m_out * h_out

    Args:
        steam_inlet: Superheated steam inlet
        water_inlet: Spray water inlet
        target_temperature_c: Target outlet temperature

    Returns:
        Tuple of (outlet_stream, balance)
    """
    # Total mass
    m_total = steam_inlet.mass_flow_kg_s + water_inlet.mass_flow_kg_s

    # Energy balance to find outlet enthalpy
    h_out = (
        steam_inlet.mass_flow_kg_s * steam_inlet.specific_enthalpy_kj_kg +
        water_inlet.mass_flow_kg_s * water_inlet.specific_enthalpy_kj_kg
    ) / m_total

    # Use average pressure (should be similar)
    P_out = (steam_inlet.pressure_kpa + water_inlet.pressure_kpa) / 2

    outlet = StreamState(
        stream_id=f"desuper_{steam_inlet.stream_id}_out",
        name=f"Desuperheated {steam_inlet.name}",
        mass_flow_kg_s=m_total,
        pressure_kpa=P_out,
        temperature_c=target_temperature_c,
        specific_enthalpy_kj_kg=h_out,
        quality_x=None,  # Determine from state
        is_measured=False,
    )

    balance = compute_component_balance(
        component_id=f"desuperheater_{steam_inlet.stream_id}",
        component_type=ComponentType.DESUPERHEATER,
        input_streams=[steam_inlet, water_inlet],
        output_streams=[outlet],
        heat_input_kw=0.0,
        heat_loss_kw=0.0,
        work_output_kw=0.0,
    )

    return outlet, balance


def compute_heat_exchanger_balance(
    hot_inlet: StreamState,
    hot_outlet: StreamState,
    cold_inlet: StreamState,
    cold_outlet: StreamState,
    heat_loss_fraction: float = 0.02,
) -> ComponentBalance:
    """
    Compute heat exchanger balance.

    DETERMINISTIC: Same inputs produce same output.

    Energy balance: Q_hot = Q_cold + Q_loss

    Args:
        hot_inlet: Hot side inlet stream
        hot_outlet: Hot side outlet stream
        cold_inlet: Cold side inlet stream
        cold_outlet: Cold side outlet stream
        heat_loss_fraction: Fraction of heat lost to surroundings

    Returns:
        ComponentBalance for heat exchanger
    """
    # Calculate heat duties
    Q_hot = compute_stream_enthalpy(hot_inlet) - compute_stream_enthalpy(hot_outlet)
    Q_cold = compute_stream_enthalpy(cold_outlet) - compute_stream_enthalpy(cold_inlet)

    # Estimate heat loss
    Q_loss = abs(Q_hot) * heat_loss_fraction

    balance = compute_component_balance(
        component_id=f"hx_{hot_inlet.stream_id}_{cold_inlet.stream_id}",
        component_type=ComponentType.HEAT_EXCHANGER,
        input_streams=[hot_inlet, cold_inlet],
        output_streams=[hot_outlet, cold_outlet],
        heat_input_kw=0.0,
        heat_loss_kw=Q_loss,
        work_output_kw=0.0,
    )

    return balance


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_stream_from_conditions(
    stream_id: str,
    name: str,
    mass_flow_kg_s: float,
    pressure_kpa: float,
    temperature_c: Optional[float] = None,
    quality_x: Optional[float] = None,
    enthalpy_kj_kg: Optional[float] = None,
) -> StreamState:
    """
    Create a StreamState from various input conditions.

    DETERMINISTIC: Same inputs produce same output.

    Calculates missing properties from available data.

    Args:
        stream_id: Unique stream identifier
        name: Human-readable name
        mass_flow_kg_s: Mass flow rate
        pressure_kpa: Pressure
        temperature_c: Temperature (optional)
        quality_x: Steam quality (optional)
        enthalpy_kj_kg: Specific enthalpy (optional)

    Returns:
        Complete StreamState
    """
    if enthalpy_kj_kg is not None:
        h = enthalpy_kj_kg
        if temperature_c is None:
            # Estimate temperature from enthalpy
            T_sat = get_saturation_temperature(pressure_kpa)
            hf, hg, _, _, _, _, _, _ = get_saturation_properties(pressure_kpa)
            if hf <= h <= hg:
                temperature_c = T_sat
            else:
                temperature_c = T_sat  # Approximation

    elif quality_x is not None:
        # Two-phase
        P_mpa = kpa_to_mpa(pressure_kpa)
        h = region4_mixture_enthalpy(P_mpa, quality_x)
        temperature_c = get_saturation_temperature(pressure_kpa)

    elif temperature_c is not None:
        # Single phase - get enthalpy
        props = compute_steam_properties(pressure_kpa, temperature_c)
        h = props.specific_enthalpy_kj_kg

    else:
        raise ValueError("Must provide temperature_c, quality_x, or enthalpy_kj_kg")

    return StreamState(
        stream_id=stream_id,
        name=name,
        mass_flow_kg_s=mass_flow_kg_s,
        pressure_kpa=pressure_kpa,
        temperature_c=temperature_c,
        specific_enthalpy_kj_kg=h,
        quality_x=quality_x,
        is_measured=True,
    )
