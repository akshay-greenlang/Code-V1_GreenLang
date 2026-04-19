"""
GL-019 HEATSCHEDULER - Energy Cost Calculator

Zero-hallucination, deterministic calculations for energy cost analysis
supporting multiple tariff structures for heating operation scheduling.

This module provides:
- Time-of-Use (ToU) tariff calculations
- Demand charge calculations based on peak demand windows
- Real-time pricing (RTP) support
- Tiered rate calculations
- Total energy cost computation for heating operations

Standards Reference:
- ISO 50001 - Energy Management Systems
- ISO 50006 - Measuring Energy Performance Using Baselines
- ASHRAE Guideline 14 - Measurement of Energy and Demand Savings

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import math

from .provenance import ProvenanceTracker, ProvenanceRecord


# =============================================================================
# CONSTANTS AND REFERENCE DATA
# =============================================================================

# Default time period definitions (hour of day, 0-23)
DEFAULT_PEAK_HOURS = list(range(14, 20))  # 2 PM - 8 PM
DEFAULT_OFF_PEAK_HOURS = list(range(0, 6)) + list(range(22, 24))  # 10 PM - 6 AM
DEFAULT_SHOULDER_HOURS = list(range(6, 14)) + list(range(20, 22))  # 6 AM - 2 PM, 8 PM - 10 PM

# Demand charge windows (typical 15-minute intervals)
DEMAND_INTERVAL_MINUTES = 15

# Currency precision
CURRENCY_PRECISION = 2


class TariffType(str, Enum):
    """Enumeration of supported tariff types."""
    FIXED = "fixed"
    TIERED = "tiered"
    TIME_OF_USE = "time_of_use"
    DEMAND = "demand"
    REAL_TIME = "real_time"


class TimePeriod(str, Enum):
    """Time-of-use periods."""
    PEAK = "peak"
    SHOULDER = "shoulder"
    OFF_PEAK = "off_peak"


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class TariffRate:
    """
    Single tariff rate definition.

    Attributes:
        rate_per_kwh: Energy rate (currency/kWh)
        period: Time period (for ToU tariffs)
        tier_start_kwh: Start of tier (for tiered tariffs)
        tier_end_kwh: End of tier (for tiered tariffs)
        demand_rate_per_kw: Demand charge rate (currency/kW)
    """
    rate_per_kwh: float
    period: Optional[TimePeriod] = None
    tier_start_kwh: Optional[float] = None
    tier_end_kwh: Optional[float] = None
    demand_rate_per_kw: Optional[float] = None


@dataclass(frozen=True)
class TariffStructure:
    """
    Complete tariff structure definition.

    Attributes:
        tariff_type: Type of tariff (fixed, ToU, tiered, etc.)
        rates: List of rates applicable to this tariff
        base_charge_per_month: Fixed monthly base charge
        peak_hours: Hours classified as peak (0-23)
        shoulder_hours: Hours classified as shoulder
        off_peak_hours: Hours classified as off-peak
        demand_window_minutes: Duration for demand measurement
        currency: Currency code (USD, EUR, etc.)
    """
    tariff_type: TariffType
    rates: List[TariffRate]
    base_charge_per_month: float = 0.0
    peak_hours: List[int] = None
    shoulder_hours: List[int] = None
    off_peak_hours: List[int] = None
    demand_window_minutes: int = DEMAND_INTERVAL_MINUTES
    currency: str = "USD"


@dataclass(frozen=True)
class HourlyLoad:
    """
    Hourly energy load data.

    Attributes:
        hour: Hour of day (0-23)
        energy_kwh: Energy consumption (kWh)
        peak_demand_kw: Peak demand during this hour (kW)
        date: Date string (YYYY-MM-DD)
    """
    hour: int
    energy_kwh: float
    peak_demand_kw: float
    date: str


@dataclass(frozen=True)
class EnergyCostInput:
    """
    Input parameters for energy cost calculations.

    Attributes:
        tariff: Tariff structure to apply
        hourly_loads: List of hourly load data
        billing_period_days: Number of days in billing period
    """
    tariff: TariffStructure
    hourly_loads: List[HourlyLoad]
    billing_period_days: int = 30


@dataclass(frozen=True)
class EnergyCostOutput:
    """
    Output results from energy cost calculations.

    Attributes:
        total_cost: Total energy cost (currency)
        energy_cost: Cost from energy consumption (currency)
        demand_cost: Cost from demand charges (currency)
        base_cost: Fixed base charge (currency)
        total_energy_kwh: Total energy consumed (kWh)
        peak_demand_kw: Peak demand recorded (kW)
        average_rate_per_kwh: Effective average rate (currency/kWh)
        cost_by_period: Cost breakdown by time period
        daily_costs: Cost breakdown by day
        peak_energy_kwh: Energy consumed during peak hours (kWh)
        off_peak_energy_kwh: Energy consumed during off-peak hours (kWh)
        shoulder_energy_kwh: Energy consumed during shoulder hours (kWh)
    """
    total_cost: float
    energy_cost: float
    demand_cost: float
    base_cost: float
    total_energy_kwh: float
    peak_demand_kw: float
    average_rate_per_kwh: float
    cost_by_period: Dict[str, float]
    daily_costs: Dict[str, float]
    peak_energy_kwh: float
    off_peak_energy_kwh: float
    shoulder_energy_kwh: float


# =============================================================================
# ENERGY COST CALCULATOR CLASS
# =============================================================================

class EnergyCostCalculator:
    """
    Zero-hallucination energy cost calculator.

    Implements deterministic calculations for energy costs under
    various tariff structures. All calculations produce bit-perfect
    reproducible results with complete provenance.

    Supports:
    - Fixed rate tariffs
    - Tiered rate tariffs
    - Time-of-Use (ToU) tariffs
    - Demand charges
    - Real-time pricing

    Guarantees:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> calculator = EnergyCostCalculator()
        >>> tariff = TariffStructure(
        ...     tariff_type=TariffType.TIME_OF_USE,
        ...     rates=[
        ...         TariffRate(rate_per_kwh=0.25, period=TimePeriod.PEAK),
        ...         TariffRate(rate_per_kwh=0.15, period=TimePeriod.SHOULDER),
        ...         TariffRate(rate_per_kwh=0.08, period=TimePeriod.OFF_PEAK),
        ...     ],
        ...     peak_hours=list(range(14, 20)),
        ...     shoulder_hours=list(range(6, 14)) + list(range(20, 22)),
        ...     off_peak_hours=list(range(0, 6)) + list(range(22, 24))
        ... )
        >>> loads = [HourlyLoad(hour=h, energy_kwh=100.0, peak_demand_kw=25.0, date="2024-01-01") for h in range(24)]
        >>> inputs = EnergyCostInput(tariff=tariff, hourly_loads=loads)
        >>> result, provenance = calculator.calculate(inputs)
        >>> print(f"Total Cost: ${result.total_cost:.2f}")
    """

    VERSION = "1.0.0"
    NAME = "EnergyCostCalculator"

    def __init__(self):
        """Initialize the energy cost calculator."""
        self._tracker: Optional[ProvenanceTracker] = None
        self._step_counter = 0

    def calculate(
        self,
        inputs: EnergyCostInput
    ) -> Tuple[EnergyCostOutput, ProvenanceRecord]:
        """
        Calculate total energy costs for heating operations.

        Args:
            inputs: EnergyCostInput with tariff and load data

        Returns:
            Tuple of (EnergyCostOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": ["ISO 50001", "ISO 50006", "ASHRAE Guideline 14"],
                "domain": "Energy Cost Analysis"
            }
        )
        self._step_counter = 0

        # Prepare inputs for provenance
        input_dict = {
            "tariff_type": inputs.tariff.tariff_type.value,
            "num_rates": len(inputs.tariff.rates),
            "base_charge_per_month": inputs.tariff.base_charge_per_month,
            "num_hourly_loads": len(inputs.hourly_loads),
            "billing_period_days": inputs.billing_period_days,
            "currency": inputs.tariff.currency
        }
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Calculate energy costs based on tariff type
        if inputs.tariff.tariff_type == TariffType.FIXED:
            energy_cost, cost_by_period = self._calculate_fixed_rate(inputs)
        elif inputs.tariff.tariff_type == TariffType.TIERED:
            energy_cost, cost_by_period = self._calculate_tiered_rate(inputs)
        elif inputs.tariff.tariff_type == TariffType.TIME_OF_USE:
            energy_cost, cost_by_period = self._calculate_tou_rate(inputs)
        elif inputs.tariff.tariff_type == TariffType.REAL_TIME:
            energy_cost, cost_by_period = self._calculate_rtp_rate(inputs)
        else:
            energy_cost, cost_by_period = self._calculate_fixed_rate(inputs)

        # Calculate demand charges
        demand_cost, peak_demand = self._calculate_demand_charges(inputs)

        # Calculate base charge
        base_cost = self._calculate_base_charge(inputs)

        # Calculate total cost
        total_cost = energy_cost + demand_cost + base_cost
        self._add_step(
            "Calculate total cost",
            "add",
            {"energy_cost": energy_cost, "demand_cost": demand_cost, "base_cost": base_cost},
            total_cost,
            "total_cost",
            "Total = Energy + Demand + Base"
        )

        # Calculate energy totals by period
        peak_energy, shoulder_energy, off_peak_energy = self._calculate_energy_by_period(inputs)

        # Calculate total energy
        total_energy = sum(load.energy_kwh for load in inputs.hourly_loads)
        self._add_step(
            "Sum total energy consumption",
            "sum",
            {"num_loads": len(inputs.hourly_loads)},
            total_energy,
            "total_energy_kwh",
            "Total = sum(hourly_energy)"
        )

        # Calculate average rate
        average_rate = total_cost / total_energy if total_energy > 0 else 0.0
        self._add_step(
            "Calculate average rate per kWh",
            "divide",
            {"total_cost": total_cost, "total_energy_kwh": total_energy},
            average_rate,
            "average_rate_per_kwh",
            "Avg Rate = Total Cost / Total Energy"
        )

        # Calculate daily costs
        daily_costs = self._calculate_daily_costs(inputs)

        # Create output
        output = EnergyCostOutput(
            total_cost=round(total_cost, CURRENCY_PRECISION),
            energy_cost=round(energy_cost, CURRENCY_PRECISION),
            demand_cost=round(demand_cost, CURRENCY_PRECISION),
            base_cost=round(base_cost, CURRENCY_PRECISION),
            total_energy_kwh=round(total_energy, 2),
            peak_demand_kw=round(peak_demand, 2),
            average_rate_per_kwh=round(average_rate, 4),
            cost_by_period={k: round(v, CURRENCY_PRECISION) for k, v in cost_by_period.items()},
            daily_costs={k: round(v, CURRENCY_PRECISION) for k, v in daily_costs.items()},
            peak_energy_kwh=round(peak_energy, 2),
            off_peak_energy_kwh=round(off_peak_energy, 2),
            shoulder_energy_kwh=round(shoulder_energy, 2)
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs({
            "total_cost": output.total_cost,
            "energy_cost": output.energy_cost,
            "demand_cost": output.demand_cost,
            "base_cost": output.base_cost,
            "total_energy_kwh": output.total_energy_kwh,
            "peak_demand_kw": output.peak_demand_kw,
            "average_rate_per_kwh": output.average_rate_per_kwh,
            "peak_energy_kwh": output.peak_energy_kwh,
            "off_peak_energy_kwh": output.off_peak_energy_kwh,
            "shoulder_energy_kwh": output.shoulder_energy_kwh
        })

        provenance = self._tracker.finalize()
        return output, provenance

    def _add_step(
        self,
        description: str,
        operation: str,
        inputs: Dict[str, Any],
        output_value: Union[float, str, Dict, List],
        output_name: str,
        formula: str = ""
    ) -> None:
        """Add a calculation step to provenance tracking."""
        self._step_counter += 1
        self._tracker.add_step(
            step_number=self._step_counter,
            description=description,
            operation=operation,
            inputs=inputs,
            output_value=output_value,
            output_name=output_name,
            formula=formula
        )

    def _validate_inputs(self, inputs: EnergyCostInput) -> None:
        """Validate input parameters."""
        if not inputs.tariff.rates:
            raise ValueError("Tariff must have at least one rate defined")

        if not inputs.hourly_loads:
            raise ValueError("At least one hourly load must be provided")

        for load in inputs.hourly_loads:
            if load.hour < 0 or load.hour > 23:
                raise ValueError(f"Invalid hour: {load.hour}. Must be 0-23")
            if load.energy_kwh < 0:
                raise ValueError(f"Energy cannot be negative: {load.energy_kwh}")
            if load.peak_demand_kw < 0:
                raise ValueError(f"Peak demand cannot be negative: {load.peak_demand_kw}")

        for rate in inputs.tariff.rates:
            if rate.rate_per_kwh < 0:
                raise ValueError("Rate cannot be negative")

    def _calculate_fixed_rate(
        self,
        inputs: EnergyCostInput
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate energy cost using fixed rate.

        Formula:
            Energy_Cost = Total_Energy_kWh * Rate_per_kWh

        Args:
            inputs: Cost calculation inputs

        Returns:
            Tuple of (total energy cost, cost breakdown by period)
        """
        rate = inputs.tariff.rates[0].rate_per_kwh
        total_energy = sum(load.energy_kwh for load in inputs.hourly_loads)

        energy_cost = total_energy * rate

        self._add_step(
            "Calculate fixed rate energy cost",
            "multiply",
            {"total_energy_kwh": total_energy, "rate_per_kwh": rate},
            energy_cost,
            "energy_cost_fixed",
            "Cost = Energy * Rate"
        )

        return energy_cost, {"fixed": energy_cost}

    def _calculate_tiered_rate(
        self,
        inputs: EnergyCostInput
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate energy cost using tiered rates.

        Each tier has a different rate per kWh applied to
        consumption within that tier's boundaries.

        Formula:
            Energy_Cost = sum(Tier_Energy_i * Tier_Rate_i)

        Args:
            inputs: Cost calculation inputs

        Returns:
            Tuple of (total energy cost, cost breakdown by tier)
        """
        total_energy = sum(load.energy_kwh for load in inputs.hourly_loads)
        remaining_energy = total_energy
        total_cost = 0.0
        cost_by_tier = {}

        # Sort rates by tier start
        sorted_rates = sorted(
            [r for r in inputs.tariff.rates if r.tier_start_kwh is not None],
            key=lambda r: r.tier_start_kwh or 0
        )

        for i, rate in enumerate(sorted_rates):
            tier_start = rate.tier_start_kwh or 0
            tier_end = rate.tier_end_kwh or float('inf')
            tier_size = tier_end - tier_start

            # Calculate energy in this tier
            tier_energy = min(remaining_energy, tier_size)
            if tier_energy <= 0:
                break

            tier_cost = tier_energy * rate.rate_per_kwh
            total_cost += tier_cost
            cost_by_tier[f"tier_{i + 1}"] = tier_cost
            remaining_energy -= tier_energy

            self._add_step(
                f"Calculate tier {i + 1} cost",
                "tier_multiply",
                {
                    "tier_energy_kwh": tier_energy,
                    "tier_rate": rate.rate_per_kwh,
                    "tier_start": tier_start,
                    "tier_end": tier_end
                },
                tier_cost,
                f"tier_{i + 1}_cost",
                f"Tier Cost = {tier_energy:.2f} kWh * ${rate.rate_per_kwh:.4f}/kWh"
            )

        self._add_step(
            "Sum tiered costs",
            "sum",
            {"num_tiers": len(cost_by_tier)},
            total_cost,
            "energy_cost_tiered",
            "Total = sum(Tier Costs)"
        )

        return total_cost, cost_by_tier

    def _calculate_tou_rate(
        self,
        inputs: EnergyCostInput
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate energy cost using Time-of-Use rates.

        Different rates apply based on the time period (peak, shoulder, off-peak).

        Formula:
            Energy_Cost = sum(Period_Energy_i * Period_Rate_i)

        Args:
            inputs: Cost calculation inputs

        Returns:
            Tuple of (total energy cost, cost breakdown by period)
        """
        # Get rate for each period
        peak_rate = next(
            (r.rate_per_kwh for r in inputs.tariff.rates
             if r.period == TimePeriod.PEAK), 0.0
        )
        shoulder_rate = next(
            (r.rate_per_kwh for r in inputs.tariff.rates
             if r.period == TimePeriod.SHOULDER), 0.0
        )
        off_peak_rate = next(
            (r.rate_per_kwh for r in inputs.tariff.rates
             if r.period == TimePeriod.OFF_PEAK), 0.0
        )

        # Get hour classifications
        peak_hours = set(inputs.tariff.peak_hours or DEFAULT_PEAK_HOURS)
        shoulder_hours = set(inputs.tariff.shoulder_hours or DEFAULT_SHOULDER_HOURS)
        off_peak_hours = set(inputs.tariff.off_peak_hours or DEFAULT_OFF_PEAK_HOURS)

        # Calculate energy and cost for each period
        peak_energy = 0.0
        shoulder_energy = 0.0
        off_peak_energy = 0.0

        for load in inputs.hourly_loads:
            if load.hour in peak_hours:
                peak_energy += load.energy_kwh
            elif load.hour in shoulder_hours:
                shoulder_energy += load.energy_kwh
            elif load.hour in off_peak_hours:
                off_peak_energy += load.energy_kwh

        peak_cost = peak_energy * peak_rate
        shoulder_cost = shoulder_energy * shoulder_rate
        off_peak_cost = off_peak_energy * off_peak_rate

        self._add_step(
            "Calculate peak period cost",
            "multiply",
            {"peak_energy_kwh": peak_energy, "peak_rate": peak_rate},
            peak_cost,
            "peak_cost",
            f"Peak Cost = {peak_energy:.2f} kWh * ${peak_rate:.4f}/kWh"
        )

        self._add_step(
            "Calculate shoulder period cost",
            "multiply",
            {"shoulder_energy_kwh": shoulder_energy, "shoulder_rate": shoulder_rate},
            shoulder_cost,
            "shoulder_cost",
            f"Shoulder Cost = {shoulder_energy:.2f} kWh * ${shoulder_rate:.4f}/kWh"
        )

        self._add_step(
            "Calculate off-peak period cost",
            "multiply",
            {"off_peak_energy_kwh": off_peak_energy, "off_peak_rate": off_peak_rate},
            off_peak_cost,
            "off_peak_cost",
            f"Off-Peak Cost = {off_peak_energy:.2f} kWh * ${off_peak_rate:.4f}/kWh"
        )

        total_cost = peak_cost + shoulder_cost + off_peak_cost

        self._add_step(
            "Sum ToU costs",
            "sum",
            {"peak_cost": peak_cost, "shoulder_cost": shoulder_cost, "off_peak_cost": off_peak_cost},
            total_cost,
            "energy_cost_tou",
            "Total = Peak + Shoulder + Off-Peak"
        )

        return total_cost, {
            "peak": peak_cost,
            "shoulder": shoulder_cost,
            "off_peak": off_peak_cost
        }

    def _calculate_rtp_rate(
        self,
        inputs: EnergyCostInput
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate energy cost using real-time pricing.

        Each hour may have a different rate based on market conditions.
        This method expects rates to be provided per hour.

        Formula:
            Energy_Cost = sum(Hourly_Energy_i * Hourly_Rate_i)

        Args:
            inputs: Cost calculation inputs

        Returns:
            Tuple of (total energy cost, cost breakdown by hour)
        """
        # For RTP, we need hourly rates - use first rate as default
        default_rate = inputs.tariff.rates[0].rate_per_kwh if inputs.tariff.rates else 0.0

        total_cost = 0.0
        cost_by_hour = {}

        for load in inputs.hourly_loads:
            # In production, hourly rates would be looked up from a database
            # Here we use the default rate
            hourly_rate = default_rate
            hourly_cost = load.energy_kwh * hourly_rate
            total_cost += hourly_cost
            cost_by_hour[f"hour_{load.hour}"] = hourly_cost

        self._add_step(
            "Calculate real-time pricing cost",
            "sum_hourly",
            {"num_hours": len(inputs.hourly_loads), "default_rate": default_rate},
            total_cost,
            "energy_cost_rtp",
            "Total = sum(Hourly Energy * Hourly Rate)"
        )

        return total_cost, cost_by_hour

    def _calculate_demand_charges(
        self,
        inputs: EnergyCostInput
    ) -> Tuple[float, float]:
        """
        Calculate demand charges based on peak demand.

        Demand charges are based on the highest demand recorded
        during the billing period, typically measured in 15-minute intervals.

        Formula:
            Demand_Cost = Peak_Demand_kW * Demand_Rate_per_kW

        Args:
            inputs: Cost calculation inputs

        Returns:
            Tuple of (demand cost, peak demand kW)
        """
        # Find peak demand across all hourly loads
        peak_demand = max((load.peak_demand_kw for load in inputs.hourly_loads), default=0.0)

        # Get demand rate (if specified)
        demand_rate = next(
            (r.demand_rate_per_kw for r in inputs.tariff.rates
             if r.demand_rate_per_kw is not None), 0.0
        )

        demand_cost = peak_demand * demand_rate

        self._add_step(
            "Calculate demand charge",
            "multiply",
            {"peak_demand_kw": peak_demand, "demand_rate_per_kw": demand_rate},
            demand_cost,
            "demand_cost",
            f"Demand Cost = {peak_demand:.2f} kW * ${demand_rate:.2f}/kW"
        )

        return demand_cost, peak_demand

    def _calculate_base_charge(self, inputs: EnergyCostInput) -> float:
        """
        Calculate monthly base/service charge.

        Formula:
            Base_Cost = Monthly_Base_Charge * (Billing_Days / 30)

        Args:
            inputs: Cost calculation inputs

        Returns:
            Prorated base charge for billing period
        """
        monthly_base = inputs.tariff.base_charge_per_month
        proration_factor = inputs.billing_period_days / 30.0
        base_cost = monthly_base * proration_factor

        self._add_step(
            "Calculate prorated base charge",
            "multiply",
            {
                "monthly_base_charge": monthly_base,
                "billing_days": inputs.billing_period_days,
                "proration_factor": proration_factor
            },
            base_cost,
            "base_cost",
            f"Base = ${monthly_base:.2f} * ({inputs.billing_period_days}/30)"
        )

        return base_cost

    def _calculate_energy_by_period(
        self,
        inputs: EnergyCostInput
    ) -> Tuple[float, float, float]:
        """
        Calculate total energy consumption by time period.

        Args:
            inputs: Cost calculation inputs

        Returns:
            Tuple of (peak_energy, shoulder_energy, off_peak_energy) in kWh
        """
        peak_hours = set(inputs.tariff.peak_hours or DEFAULT_PEAK_HOURS)
        shoulder_hours = set(inputs.tariff.shoulder_hours or DEFAULT_SHOULDER_HOURS)
        off_peak_hours = set(inputs.tariff.off_peak_hours or DEFAULT_OFF_PEAK_HOURS)

        peak_energy = sum(
            load.energy_kwh for load in inputs.hourly_loads
            if load.hour in peak_hours
        )
        shoulder_energy = sum(
            load.energy_kwh for load in inputs.hourly_loads
            if load.hour in shoulder_hours
        )
        off_peak_energy = sum(
            load.energy_kwh for load in inputs.hourly_loads
            if load.hour in off_peak_hours
        )

        return peak_energy, shoulder_energy, off_peak_energy

    def _calculate_daily_costs(
        self,
        inputs: EnergyCostInput
    ) -> Dict[str, float]:
        """
        Calculate energy costs by day.

        Args:
            inputs: Cost calculation inputs

        Returns:
            Dictionary mapping date to daily cost
        """
        daily_costs = {}

        # Group loads by date
        loads_by_date: Dict[str, List[HourlyLoad]] = {}
        for load in inputs.hourly_loads:
            if load.date not in loads_by_date:
                loads_by_date[load.date] = []
            loads_by_date[load.date].append(load)

        # Calculate cost for each day (simplified - uses average rate)
        avg_rate = inputs.tariff.rates[0].rate_per_kwh if inputs.tariff.rates else 0.0

        for date, loads in loads_by_date.items():
            daily_energy = sum(load.energy_kwh for load in loads)
            daily_costs[date] = daily_energy * avg_rate

        return daily_costs


# =============================================================================
# STANDALONE CALCULATION FUNCTIONS
# =============================================================================

def calculate_simple_energy_cost(
    energy_kwh: float,
    rate_per_kwh: float
) -> float:
    """
    Calculate energy cost using simple fixed rate.

    Formula:
        Cost = Energy_kWh * Rate_per_kWh

    Args:
        energy_kwh: Energy consumption (kWh)
        rate_per_kwh: Energy rate (currency/kWh)

    Returns:
        Energy cost (currency)

    Example:
        >>> cost = calculate_simple_energy_cost(1000.0, 0.12)
        >>> print(f"Cost: ${cost:.2f}")  # $120.00
    """
    return energy_kwh * rate_per_kwh


def calculate_demand_charge(
    peak_demand_kw: float,
    demand_rate_per_kw: float
) -> float:
    """
    Calculate demand charge based on peak demand.

    Formula:
        Demand_Charge = Peak_Demand_kW * Rate_per_kW

    Args:
        peak_demand_kw: Peak demand (kW)
        demand_rate_per_kw: Demand rate (currency/kW)

    Returns:
        Demand charge (currency)

    Example:
        >>> charge = calculate_demand_charge(500.0, 15.00)
        >>> print(f"Demand Charge: ${charge:.2f}")  # $7,500.00
    """
    return peak_demand_kw * demand_rate_per_kw


def calculate_tou_cost(
    peak_energy_kwh: float,
    shoulder_energy_kwh: float,
    off_peak_energy_kwh: float,
    peak_rate: float,
    shoulder_rate: float,
    off_peak_rate: float
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate Time-of-Use energy cost.

    Formula:
        Cost = (Peak_kWh * Peak_Rate) + (Shoulder_kWh * Shoulder_Rate)
               + (OffPeak_kWh * OffPeak_Rate)

    Args:
        peak_energy_kwh: Energy during peak hours (kWh)
        shoulder_energy_kwh: Energy during shoulder hours (kWh)
        off_peak_energy_kwh: Energy during off-peak hours (kWh)
        peak_rate: Peak rate (currency/kWh)
        shoulder_rate: Shoulder rate (currency/kWh)
        off_peak_rate: Off-peak rate (currency/kWh)

    Returns:
        Tuple of (total cost, breakdown by period)

    Example:
        >>> total, breakdown = calculate_tou_cost(
        ...     300, 400, 300,  # Energy by period
        ...     0.25, 0.15, 0.08  # Rates by period
        ... )
        >>> print(f"Total: ${total:.2f}")  # $159.00
    """
    peak_cost = peak_energy_kwh * peak_rate
    shoulder_cost = shoulder_energy_kwh * shoulder_rate
    off_peak_cost = off_peak_energy_kwh * off_peak_rate

    total_cost = peak_cost + shoulder_cost + off_peak_cost

    return total_cost, {
        "peak": peak_cost,
        "shoulder": shoulder_cost,
        "off_peak": off_peak_cost
    }


def classify_hour(
    hour: int,
    peak_hours: List[int] = None,
    shoulder_hours: List[int] = None,
    off_peak_hours: List[int] = None
) -> TimePeriod:
    """
    Classify an hour into a time-of-use period.

    Args:
        hour: Hour of day (0-23)
        peak_hours: Hours classified as peak
        shoulder_hours: Hours classified as shoulder
        off_peak_hours: Hours classified as off-peak

    Returns:
        TimePeriod classification

    Example:
        >>> period = classify_hour(15)  # 3 PM
        >>> print(period.value)  # "peak"
    """
    peak_set = set(peak_hours or DEFAULT_PEAK_HOURS)
    shoulder_set = set(shoulder_hours or DEFAULT_SHOULDER_HOURS)
    off_peak_set = set(off_peak_hours or DEFAULT_OFF_PEAK_HOURS)

    if hour in peak_set:
        return TimePeriod.PEAK
    elif hour in shoulder_set:
        return TimePeriod.SHOULDER
    elif hour in off_peak_set:
        return TimePeriod.OFF_PEAK
    else:
        return TimePeriod.SHOULDER  # Default


def calculate_average_rate(
    total_cost: float,
    total_energy_kwh: float
) -> float:
    """
    Calculate effective average rate per kWh.

    Formula:
        Average_Rate = Total_Cost / Total_Energy

    Args:
        total_cost: Total energy cost (currency)
        total_energy_kwh: Total energy consumed (kWh)

    Returns:
        Average rate (currency/kWh)

    Example:
        >>> avg = calculate_average_rate(150.0, 1000.0)
        >>> print(f"Average: ${avg:.4f}/kWh")  # $0.1500/kWh
    """
    if total_energy_kwh <= 0:
        return 0.0
    return total_cost / total_energy_kwh
