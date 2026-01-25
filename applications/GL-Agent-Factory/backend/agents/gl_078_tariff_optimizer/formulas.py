"""
GL-078: Tariff Optimizer Formulas Module

This module contains all deterministic calculation formulas for utility
tariff analysis and cost optimization.

Formula Sources:
- Standard utility rate calculation methods
- TOU (Time-of-Use) pricing structures
- Demand charge methodologies
- Load factor calculations

Example:
    >>> from formulas import calculate_tou_cost
    >>> cost = calculate_tou_cost(hourly_kwh, rate_schedule)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TOUCostResult:
    """Result of TOU cost calculation."""
    on_peak_cost: float
    mid_peak_cost: float
    off_peak_cost: float
    total_energy_cost: float
    on_peak_kwh: float
    mid_peak_kwh: float
    off_peak_kwh: float


@dataclass
class DemandChargeResult:
    """Result of demand charge calculation."""
    facility_demand_charge: float
    tou_demand_charge: float
    total_demand_charge: float
    peak_demand_kw: float
    on_peak_demand_kw: float


@dataclass
class LoadShiftResult:
    """Result of load shift calculation."""
    original_cost: float
    shifted_cost: float
    savings: float
    shifted_kwh: float
    effective_rate_reduction: float


@dataclass
class AnnualCostResult:
    """Result of annual cost calculation."""
    energy_cost: float
    demand_cost: float
    fixed_cost: float
    total_cost: float
    average_rate_per_kwh: float


# =============================================================================
# TIME-OF-USE COST CALCULATIONS
# =============================================================================

def calculate_tou_cost(
    hourly_kwh: List[float],
    on_peak_rate: float,
    mid_peak_rate: float,
    off_peak_rate: float,
    on_peak_hours: List[int] = None,
    mid_peak_hours: List[int] = None,
) -> TOUCostResult:
    """
    Calculate time-of-use energy costs.

    ZERO-HALLUCINATION: Standard TOU rate calculation

    Args:
        hourly_kwh: List of hourly kWh values (24 or 8760 hours)
        on_peak_rate: $/kWh for on-peak period
        mid_peak_rate: $/kWh for mid-peak period
        off_peak_rate: $/kWh for off-peak period
        on_peak_hours: Hours (0-23) for on-peak period
        mid_peak_hours: Hours (0-23) for mid-peak period

    Returns:
        TOUCostResult with cost breakdown

    Formula:
        Energy_Cost = SUM(kWh_h * Rate_h) for each hour h
    """
    # Default TOU periods (typical summer)
    if on_peak_hours is None:
        on_peak_hours = list(range(16, 21))  # 4pm-9pm
    if mid_peak_hours is None:
        mid_peak_hours = list(range(12, 16)) + list(range(21, 24))

    # All other hours are off-peak
    off_peak_hours = [h for h in range(24) if h not in on_peak_hours and h not in mid_peak_hours]

    on_peak_kwh = 0.0
    mid_peak_kwh = 0.0
    off_peak_kwh = 0.0

    num_days = len(hourly_kwh) // 24 if len(hourly_kwh) >= 24 else 1

    for day in range(num_days):
        for hour in range(24):
            idx = day * 24 + hour
            if idx >= len(hourly_kwh):
                break
            kwh = hourly_kwh[idx]

            if hour in on_peak_hours:
                on_peak_kwh += kwh
            elif hour in mid_peak_hours:
                mid_peak_kwh += kwh
            else:
                off_peak_kwh += kwh

    on_peak_cost = on_peak_kwh * on_peak_rate
    mid_peak_cost = mid_peak_kwh * mid_peak_rate
    off_peak_cost = off_peak_kwh * off_peak_rate

    return TOUCostResult(
        on_peak_cost=round(on_peak_cost, 2),
        mid_peak_cost=round(mid_peak_cost, 2),
        off_peak_cost=round(off_peak_cost, 2),
        total_energy_cost=round(on_peak_cost + mid_peak_cost + off_peak_cost, 2),
        on_peak_kwh=round(on_peak_kwh, 2),
        mid_peak_kwh=round(mid_peak_kwh, 2),
        off_peak_kwh=round(off_peak_kwh, 2),
    )


def calculate_flat_rate_cost(
    total_kwh: float,
    flat_rate: float,
) -> float:
    """
    Calculate flat-rate energy cost.

    ZERO-HALLUCINATION: Simple multiplication

    Args:
        total_kwh: Total energy consumption
        flat_rate: $/kWh flat rate

    Returns:
        Total energy cost

    Formula:
        Energy_Cost = Total_kWh * Rate
    """
    return round(total_kwh * flat_rate, 2)


def calculate_tiered_cost(
    total_kwh: float,
    tier_limits: List[float],
    tier_rates: List[float],
) -> float:
    """
    Calculate tiered energy cost.

    ZERO-HALLUCINATION: Standard tiered rate calculation

    Args:
        total_kwh: Total energy consumption
        tier_limits: Upper limits for each tier (excluding last tier)
        tier_rates: Rate for each tier (one more than limits)

    Returns:
        Total energy cost

    Formula:
        Cost = SUM(min(remaining, tier_limit) * tier_rate) for each tier
    """
    if len(tier_rates) != len(tier_limits) + 1:
        raise ValueError("tier_rates must have one more element than tier_limits")

    total_cost = 0.0
    remaining_kwh = total_kwh

    for i, rate in enumerate(tier_rates):
        if i < len(tier_limits):
            tier_kwh = min(remaining_kwh, tier_limits[i])
            if i > 0:
                tier_kwh = min(remaining_kwh, tier_limits[i] - tier_limits[i-1])
        else:
            tier_kwh = remaining_kwh

        total_cost += tier_kwh * rate
        remaining_kwh -= tier_kwh

        if remaining_kwh <= 0:
            break

    return round(total_cost, 2)


# =============================================================================
# DEMAND CHARGE CALCULATIONS
# =============================================================================

def calculate_demand_charge(
    peak_demand_kw: float,
    facility_demand_rate: float = 0.0,
    on_peak_demand_kw: Optional[float] = None,
    on_peak_demand_rate: float = 0.0,
    mid_peak_demand_kw: Optional[float] = None,
    mid_peak_demand_rate: float = 0.0,
) -> DemandChargeResult:
    """
    Calculate monthly demand charges.

    ZERO-HALLUCINATION: Standard demand charge calculation

    Args:
        peak_demand_kw: Maximum demand in billing period (kW)
        facility_demand_rate: $/kW for facility demand charge
        on_peak_demand_kw: Maximum demand during on-peak hours
        on_peak_demand_rate: $/kW for on-peak demand
        mid_peak_demand_kw: Maximum demand during mid-peak hours
        mid_peak_demand_rate: $/kW for mid-peak demand

    Returns:
        DemandChargeResult with charge breakdown

    Formula:
        Demand_Charge = Peak_kW * Facility_Rate + OnPeak_kW * OnPeak_Rate
    """
    facility_charge = peak_demand_kw * facility_demand_rate

    on_peak_kw = on_peak_demand_kw or (peak_demand_kw * 0.9)
    mid_peak_kw = mid_peak_demand_kw or (peak_demand_kw * 0.7)

    tou_charge = on_peak_kw * on_peak_demand_rate + mid_peak_kw * mid_peak_demand_rate

    return DemandChargeResult(
        facility_demand_charge=round(facility_charge, 2),
        tou_demand_charge=round(tou_charge, 2),
        total_demand_charge=round(facility_charge + tou_charge, 2),
        peak_demand_kw=round(peak_demand_kw, 2),
        on_peak_demand_kw=round(on_peak_kw, 2),
    )


def calculate_ratcheted_demand(
    current_peak_kw: float,
    historical_peaks: List[float],
    ratchet_percent: float = 0.80,
    lookback_months: int = 11,
) -> float:
    """
    Calculate ratcheted demand for billing.

    ZERO-HALLUCINATION: Standard demand ratchet calculation

    Args:
        current_peak_kw: Current month's peak demand
        historical_peaks: Previous months' peak demands
        ratchet_percent: Ratchet percentage (typically 80%)
        lookback_months: Number of months to look back

    Returns:
        Billing demand (max of current or ratcheted historical)

    Formula:
        Billing_Demand = MAX(Current_Peak, Ratchet% * MAX(Historical_Peaks))
    """
    if not historical_peaks:
        return current_peak_kw

    relevant_peaks = historical_peaks[-lookback_months:]
    ratcheted_peak = max(relevant_peaks) * ratchet_percent

    return round(max(current_peak_kw, ratcheted_peak), 2)


# =============================================================================
# LOAD SHIFTING CALCULATIONS
# =============================================================================

def calculate_optimal_shift(
    on_peak_kwh: float,
    mid_peak_kwh: float,
    off_peak_kwh: float,
    on_peak_rate: float,
    mid_peak_rate: float,
    off_peak_rate: float,
    max_shiftable_percent: float = 0.30,
) -> LoadShiftResult:
    """
    Calculate optimal load shift from on-peak to off-peak.

    ZERO-HALLUCINATION: Deterministic shift calculation

    Args:
        on_peak_kwh: Current on-peak consumption
        mid_peak_kwh: Current mid-peak consumption
        off_peak_kwh: Current off-peak consumption
        on_peak_rate: $/kWh on-peak
        mid_peak_rate: $/kWh mid-peak
        off_peak_rate: $/kWh off-peak
        max_shiftable_percent: Maximum percentage that can be shifted

    Returns:
        LoadShiftResult with savings analysis

    Formula:
        Savings = Shifted_kWh * (OnPeak_Rate - OffPeak_Rate)
    """
    # Calculate original cost
    original_cost = (
        on_peak_kwh * on_peak_rate +
        mid_peak_kwh * mid_peak_rate +
        off_peak_kwh * off_peak_rate
    )

    # Calculate shiftable amount
    shiftable_kwh = on_peak_kwh * max_shiftable_percent

    # Calculate shifted cost
    new_on_peak = on_peak_kwh - shiftable_kwh
    new_off_peak = off_peak_kwh + shiftable_kwh

    shifted_cost = (
        new_on_peak * on_peak_rate +
        mid_peak_kwh * mid_peak_rate +
        new_off_peak * off_peak_rate
    )

    savings = original_cost - shifted_cost
    rate_reduction = (on_peak_rate - off_peak_rate) if on_peak_kwh > 0 else 0

    return LoadShiftResult(
        original_cost=round(original_cost, 2),
        shifted_cost=round(shifted_cost, 2),
        savings=round(savings, 2),
        shifted_kwh=round(shiftable_kwh, 2),
        effective_rate_reduction=round(rate_reduction, 4),
    )


def calculate_battery_shift_value(
    battery_capacity_kwh: float,
    on_peak_rate: float,
    off_peak_rate: float,
    charge_efficiency: float = 0.95,
    discharge_efficiency: float = 0.95,
    cycles_per_day: float = 1.0,
    days_per_year: int = 250,
) -> Tuple[float, float]:
    """
    Calculate annual value of battery for load shifting.

    ZERO-HALLUCINATION: Standard battery economics calculation

    Args:
        battery_capacity_kwh: Usable battery capacity
        on_peak_rate: $/kWh on-peak
        off_peak_rate: $/kWh off-peak
        charge_efficiency: Charging efficiency
        discharge_efficiency: Discharging efficiency
        cycles_per_day: Number of cycles per day
        days_per_year: Operating days per year

    Returns:
        Tuple of (annual_savings, effective_rate_spread)

    Formula:
        Round_Trip_Efficiency = Charge_Eff * Discharge_Eff
        Effective_Spread = OnPeak - OffPeak / RTE
        Annual_Savings = Capacity * Spread * Cycles * Days
    """
    round_trip_efficiency = charge_efficiency * discharge_efficiency

    # Cost to charge (off-peak) accounting for efficiency losses
    charge_cost_per_kwh = off_peak_rate / charge_efficiency

    # Revenue from discharge (on-peak)
    discharge_value_per_kwh = on_peak_rate * discharge_efficiency

    # Net value per kWh cycled
    effective_spread = discharge_value_per_kwh - charge_cost_per_kwh

    # Annual savings
    annual_cycles = cycles_per_day * days_per_year
    annual_savings = battery_capacity_kwh * effective_spread * annual_cycles

    return round(annual_savings, 2), round(effective_spread, 4)


# =============================================================================
# ANNUAL SAVINGS CALCULATIONS
# =============================================================================

def calculate_annual_savings(
    current_annual_cost: float,
    proposed_annual_cost: float,
) -> Tuple[float, float]:
    """
    Calculate annual savings and percentage.

    ZERO-HALLUCINATION: Simple arithmetic

    Args:
        current_annual_cost: Current annual electricity cost
        proposed_annual_cost: Proposed annual electricity cost

    Returns:
        Tuple of (absolute_savings, percent_savings)

    Formula:
        Savings = Current - Proposed
        Percent = Savings / Current * 100
    """
    savings = current_annual_cost - proposed_annual_cost
    percent = (savings / current_annual_cost * 100) if current_annual_cost > 0 else 0

    return round(savings, 2), round(percent, 1)


def calculate_combined_annual_cost(
    energy_cost: float,
    demand_cost: float,
    fixed_cost: float,
    total_kwh: float,
) -> AnnualCostResult:
    """
    Calculate combined annual cost components.

    ZERO-HALLUCINATION: Simple summation

    Args:
        energy_cost: Annual energy charges
        demand_cost: Annual demand charges
        fixed_cost: Annual fixed charges
        total_kwh: Total annual consumption

    Returns:
        AnnualCostResult with all components

    Formula:
        Total = Energy + Demand + Fixed
        Avg_Rate = Total / Total_kWh
    """
    total = energy_cost + demand_cost + fixed_cost
    avg_rate = total / total_kwh if total_kwh > 0 else 0

    return AnnualCostResult(
        energy_cost=round(energy_cost, 2),
        demand_cost=round(demand_cost, 2),
        fixed_cost=round(fixed_cost, 2),
        total_cost=round(total, 2),
        average_rate_per_kwh=round(avg_rate, 4),
    )


# =============================================================================
# PEAK SHAVING CALCULATIONS
# =============================================================================

def calculate_peak_shaving_benefit(
    current_peak_kw: float,
    reduced_peak_kw: float,
    demand_charge_rate: float,
    months_per_year: int = 12,
) -> Tuple[float, float]:
    """
    Calculate benefit from peak demand reduction.

    ZERO-HALLUCINATION: Standard demand reduction calculation

    Args:
        current_peak_kw: Current peak demand (kW)
        reduced_peak_kw: Target reduced peak demand (kW)
        demand_charge_rate: $/kW monthly demand charge
        months_per_year: Number of billing months

    Returns:
        Tuple of (annual_savings, peak_reduction_kw)

    Formula:
        Annual_Savings = (Current - Reduced) * Rate * 12
    """
    peak_reduction = current_peak_kw - reduced_peak_kw
    annual_savings = peak_reduction * demand_charge_rate * months_per_year

    return round(annual_savings, 2), round(peak_reduction, 2)


def calculate_power_factor_penalty(
    measured_pf: float,
    target_pf: float,
    kw_demand: float,
    demand_charge_rate: float,
) -> Tuple[float, float]:
    """
    Calculate power factor penalty or correction benefit.

    ZERO-HALLUCINATION: Standard PF penalty calculation

    Args:
        measured_pf: Measured power factor (0-1)
        target_pf: Utility target power factor (typically 0.90)
        kw_demand: Current kW demand
        demand_charge_rate: $/kW demand charge

    Returns:
        Tuple of (monthly_penalty, kvar_needed_for_correction)

    Formula:
        Penalty = (Target_PF/Measured_PF - 1) * kW * Rate
        kVAR_needed = kW * (tan(acos(measured)) - tan(acos(target)))
    """
    import math

    if measured_pf >= target_pf:
        return 0.0, 0.0

    # Calculate billing demand multiplier
    pf_multiplier = target_pf / measured_pf
    adjusted_demand = kw_demand * pf_multiplier
    penalty = (adjusted_demand - kw_demand) * demand_charge_rate

    # Calculate kVAR needed for correction
    theta_measured = math.acos(measured_pf)
    theta_target = math.acos(target_pf)
    kvar_needed = kw_demand * (math.tan(theta_measured) - math.tan(theta_target))

    return round(penalty, 2), round(kvar_needed, 2)


# =============================================================================
# LOAD FACTOR CALCULATIONS
# =============================================================================

def calculate_load_factor(
    total_kwh: float,
    peak_demand_kw: float,
    hours_in_period: int,
) -> float:
    """
    Calculate load factor.

    ZERO-HALLUCINATION: Standard load factor formula

    Args:
        total_kwh: Total energy consumption in period
        peak_demand_kw: Maximum demand in period
        hours_in_period: Number of hours in period

    Returns:
        Load factor (0-1)

    Formula:
        Load_Factor = Total_kWh / (Peak_kW * Hours)
    """
    if peak_demand_kw <= 0 or hours_in_period <= 0:
        return 0.0

    avg_demand = total_kwh / hours_in_period
    load_factor = avg_demand / peak_demand_kw

    return round(min(load_factor, 1.0), 4)


def calculate_coincident_peak_factor(
    facility_peak_kw: float,
    system_peak_kw: float,
    facility_demand_at_system_peak_kw: float,
) -> float:
    """
    Calculate coincident peak factor.

    ZERO-HALLUCINATION: Standard coincidence factor

    Args:
        facility_peak_kw: Facility's individual peak demand
        system_peak_kw: System peak demand (not used, for reference)
        facility_demand_at_system_peak_kw: Facility demand at system peak

    Returns:
        Coincident peak factor (0-1)

    Formula:
        CPF = Demand_at_System_Peak / Facility_Peak
    """
    if facility_peak_kw <= 0:
        return 0.0

    cpf = facility_demand_at_system_peak_kw / facility_peak_kw
    return round(min(cpf, 1.0), 4)
