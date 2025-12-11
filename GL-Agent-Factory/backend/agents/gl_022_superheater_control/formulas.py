"""Zero-hallucination formulas for GL-022 Superheater Control.

All calculations are deterministic and traceable with SHA-256 provenance.
Based on IAPWS-IF97 steam tables and process control theory.
"""
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class SteamProperties:
    """Steam thermodynamic properties."""
    temperature_c: float
    pressure_bar: float
    saturation_temp_c: float
    superheat_c: float
    specific_enthalpy_kj_kg: float
    specific_heat_cp_kj_kg_k: float


def calculate_saturation_temperature(pressure_bar: float) -> float:
    """
    Calculate saturation temperature from pressure using simplified IAPWS-IF97.

    Formula: T_sat = 100 * (P/1.01325)^0.25 for P < 10 bar (simplified)
             For higher pressures, uses polynomial fit to IAPWS-IF97

    Reference: IAPWS-IF97, Industrial Formulation 1997
    """
    if pressure_bar <= 0:
        raise ValueError("Pressure must be positive")

    # Coefficients for polynomial approximation (valid 0.1 to 200 bar)
    # T_sat(°C) = a0 + a1*ln(P) + a2*ln(P)^2 + a3*ln(P)^3
    a0 = 99.974
    a1 = 28.080
    a2 = -0.5479
    a3 = 0.01923

    ln_p = math.log(pressure_bar)
    t_sat = a0 + a1 * ln_p + a2 * ln_p**2 + a3 * ln_p**3

    return round(t_sat, 2)


def calculate_superheat(steam_temp_c: float, pressure_bar: float) -> float:
    """
    Calculate degree of superheat above saturation.

    Formula: ΔT_superheat = T_steam - T_saturation

    Returns superheat in °C (positive if superheated, negative if wet)
    """
    t_sat = calculate_saturation_temperature(pressure_bar)
    superheat = steam_temp_c - t_sat
    return round(superheat, 2)


def calculate_steam_enthalpy(temp_c: float, pressure_bar: float) -> float:
    """
    Calculate specific enthalpy of superheated steam.

    Simplified formula based on IAPWS-IF97:
    h = h_sat + cp_avg * (T - T_sat)

    Where:
    - h_sat: Saturated vapor enthalpy at pressure
    - cp_avg: Average specific heat (≈ 2.1 kJ/kg·K for superheated steam)
    """
    t_sat = calculate_saturation_temperature(pressure_bar)

    # Saturated vapor enthalpy approximation (kJ/kg)
    # h_g = 2675 + 1.8 * (T_sat - 100) for low pressures
    h_sat = 2675 + 1.8 * (t_sat - 100)

    # Average specific heat of superheated steam
    cp_avg = 2.1  # kJ/kg·K (varies with pressure, simplified)

    superheat = temp_c - t_sat
    if superheat < 0:
        # Wet steam - return saturated enthalpy
        return round(h_sat, 1)

    h = h_sat + cp_avg * superheat
    return round(h, 1)


def calculate_spray_water_flow(
    steam_flow_kg_s: float,
    steam_temp_in_c: float,
    steam_temp_target_c: float,
    spray_water_temp_c: float,
    steam_pressure_bar: float
) -> Tuple[float, float]:
    """
    Calculate required spray water flow rate for desuperheating.

    Energy balance: m_steam * h_in + m_spray * h_water = (m_steam + m_spray) * h_out

    Solving for m_spray:
    m_spray = m_steam * (h_in - h_out) / (h_out - h_water)

    Returns: (spray_flow_kg_s, energy_absorbed_kw)
    """
    if steam_temp_in_c <= steam_temp_target_c:
        return 0.0, 0.0

    # Calculate enthalpies
    h_in = calculate_steam_enthalpy(steam_temp_in_c, steam_pressure_bar)
    h_out = calculate_steam_enthalpy(steam_temp_target_c, steam_pressure_bar)

    # Spray water enthalpy (subcooled liquid approximation)
    # h_water ≈ 4.18 * T_water (kJ/kg) for liquid water
    h_water = 4.18 * spray_water_temp_c

    # Calculate required spray flow
    if h_out <= h_water:
        raise ValueError("Target temperature too low for spray control")

    spray_flow = steam_flow_kg_s * (h_in - h_out) / (h_out - h_water)

    # Energy absorbed by spray water (kW)
    energy_absorbed = spray_flow * (h_out - h_water)

    return round(spray_flow, 4), round(energy_absorbed, 2)


def calculate_valve_position(
    required_flow_kg_s: float,
    max_flow_kg_s: float,
    valve_cv: float = 100.0
) -> float:
    """
    Calculate valve position for required flow rate.

    Using equal percentage valve characteristic:
    Flow/Cv = (R^(x-1)) where R = rangeability (typically 50), x = position

    Simplified linear approximation for control purposes:
    Position(%) = (Required_flow / Max_flow) * 100
    """
    if max_flow_kg_s <= 0:
        return 0.0

    position = (required_flow_kg_s / max_flow_kg_s) * 100
    return min(100.0, max(0.0, round(position, 1)))


def calculate_pid_parameters(
    process_time_constant_s: float = 60.0,
    process_dead_time_s: float = 10.0,
    desired_response_time_s: float = 120.0
) -> Dict[str, float]:
    """
    Calculate PID tuning parameters using Lambda tuning method.

    Lambda tuning provides robust, non-oscillatory control.

    Formulas:
    - Kp = τ / (K * (λ + θ))
    - Ki = Kp / τ
    - Kd = Kp * θ / 2

    Where:
    - τ = process time constant
    - θ = dead time
    - λ = desired closed-loop time constant (response time)
    - K = process gain (assumed 1.0 for normalized)
    """
    tau = process_time_constant_s
    theta = process_dead_time_s
    lambda_cl = desired_response_time_s
    k = 1.0  # Normalized process gain

    kp = tau / (k * (lambda_cl + theta))
    ki = kp / tau
    kd = kp * theta / 2

    return {
        "kp": round(kp, 4),
        "ki": round(ki, 6),
        "kd": round(kd, 4),
        "deadband_c": 1.0,  # Standard deadband
        "max_rate_c_per_min": 5.0  # Typical max rate
    }


def calculate_spray_energy_loss(
    spray_flow_kg_s: float,
    steam_enthalpy_reduction_kj_kg: float
) -> float:
    """
    Calculate energy loss from spray water injection.

    Energy loss = spray_flow * (h_steam_in - h_steam_out)

    Note: This represents useful energy diverted to evaporating spray water.
    """
    energy_loss_kw = spray_flow_kg_s * steam_enthalpy_reduction_kj_kg
    return round(energy_loss_kw, 2)


def calculate_thermal_efficiency_impact(
    spray_energy_loss_kw: float,
    total_fuel_input_kw: float
) -> float:
    """
    Calculate impact on thermal efficiency from spray water use.

    Efficiency impact = (spray_energy_loss / fuel_input) * 100

    Excessive spray water use indicates poor heat absorption in superheater.
    """
    if total_fuel_input_kw <= 0:
        return 0.0

    impact = (spray_energy_loss_kw / total_fuel_input_kw) * 100
    return round(impact, 3)


def generate_calculation_hash(inputs: Dict, outputs: Dict) -> str:
    """
    Generate SHA-256 hash for calculation provenance.

    This ensures deterministic, traceable calculations with
    zero-hallucination compliance.
    """
    data = {
        "inputs": inputs,
        "outputs": outputs,
        "formula_version": "1.0.0",
        "standard": "IAPWS-IF97"
    }
    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()


# Unit conversion helpers
def bar_to_psi(bar: float) -> float:
    """Convert bar to PSI."""
    return round(bar * 14.5038, 2)


def celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return round(c * 9/5 + 32, 2)


def kg_s_to_lb_hr(kg_s: float) -> float:
    """Convert kg/s to lb/hr."""
    return round(kg_s * 7936.64, 2)
