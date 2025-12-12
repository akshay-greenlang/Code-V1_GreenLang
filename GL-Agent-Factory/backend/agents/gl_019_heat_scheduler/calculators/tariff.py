"""
Tariff Analysis Calculator for GL-019 HEATSCHEDULER

This module implements Time-of-Use (TOU) tariff analysis, demand charge
calculations, and cost optimization support for process heating scheduling.

The tariff analysis follows zero-hallucination principles:
- All calculations are deterministic rate lookups and arithmetic
- No ML/LLM in the cost calculation path
- Rates are applied exactly as defined in tariff structures
- Results are mathematically verifiable

Standards:
- OpenADR 2.0 Demand Response Standards
- NAESB WEQ Business Practices (utility tariffs)
- IEEE 2030.5 Smart Energy Profile

Tariff Types Supported:
1. Flat Rate - Single rate for all hours
2. Time-of-Use (TOU) - Different rates by time period
3. Real-Time Pricing (RTP) - Hourly varying rates
4. Critical Peak Pricing (CPP) - Elevated rates during grid stress
5. Demand Charges - Monthly peak demand charges

Example:
    >>> analyzer = TariffAnalyzer(tariff_structure)
    >>> cost = analyzer.calculate_cost(
    ...     demand_profile=hourly_demand,
    ...     start_time=datetime.now()
    ... )
"""

from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

class RatePeriod(str, Enum):
    """Standard rate period classifications."""
    ON_PEAK = "on_peak"
    MID_PEAK = "mid_peak"
    OFF_PEAK = "off_peak"
    SUPER_OFF_PEAK = "super_off_peak"
    CRITICAL_PEAK = "critical_peak"


@dataclass
class TariffPeriodDef:
    """Definition of a tariff period."""
    period_name: str
    period_type: RatePeriod
    start_hour: int  # 0-23
    end_hour: int  # 0-23
    days_of_week: List[int]  # 0=Monday, 6=Sunday
    energy_rate_per_kwh: float
    demand_rate_per_kw: float = 0.0
    months: Optional[List[int]] = None  # None = all months


@dataclass
class DemandChargeStructure:
    """Demand charge structure definition."""
    monthly_rate_per_kw: float
    peak_hours_only: bool = False  # If True, only peak period counts
    ratchet_percentage: float = 0.0  # % of historical peak
    ratchet_months: int = 12  # How many months to look back
    coincident_peak: bool = False  # If True, based on system peak


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""
    total_cost: float
    energy_cost: float
    demand_cost: float
    carbon_cost: float

    # By period
    on_peak_energy_kwh: float = 0.0
    on_peak_cost: float = 0.0
    mid_peak_energy_kwh: float = 0.0
    mid_peak_cost: float = 0.0
    off_peak_energy_kwh: float = 0.0
    off_peak_cost: float = 0.0

    # Peak demand
    peak_demand_kw: float = 0.0
    on_peak_demand_kw: float = 0.0


# =============================================================================
# Rate Lookup Functions
# =============================================================================

def get_period_type(
    timestamp: datetime,
    periods: List[TariffPeriodDef]
) -> Tuple[str, RatePeriod, float, float]:
    """
    Get the tariff period for a given timestamp.

    Args:
        timestamp: Datetime to look up.
        periods: List of tariff period definitions.

    Returns:
        Tuple of (period_name, period_type, energy_rate, demand_rate).

    Example:
        >>> periods = [TariffPeriodDef('peak', RatePeriod.ON_PEAK, 12, 18, [0,1,2,3,4], 0.25)]
        >>> name, ptype, rate, _ = get_period_type(datetime(2024, 6, 15, 14, 0), periods)
        >>> print(f"Period: {name}, Rate: ${rate}/kWh")
    """
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    month = timestamp.month

    for period in periods:
        # Check day of week
        if day_of_week not in period.days_of_week:
            continue

        # Check month (if specified)
        if period.months and month not in period.months:
            continue

        # Check hour range
        if period.start_hour <= period.end_hour:
            # Normal range (e.g., 8-18)
            if period.start_hour <= hour < period.end_hour:
                return (period.period_name, period.period_type,
                        period.energy_rate_per_kwh, period.demand_rate_per_kw)
        else:
            # Wrap-around range (e.g., 22-6)
            if hour >= period.start_hour or hour < period.end_hour:
                return (period.period_name, period.period_type,
                        period.energy_rate_per_kwh, period.demand_rate_per_kw)

    # Default to last period if no match (should have a catch-all)
    if periods:
        return (periods[-1].period_name, periods[-1].period_type,
                periods[-1].energy_rate_per_kwh, periods[-1].demand_rate_per_kw)

    # No periods defined - return default
    return ("default", RatePeriod.OFF_PEAK, 0.10, 0.0)


def create_standard_tou_periods(
    peak_rate: float = 0.25,
    mid_peak_rate: float = 0.15,
    off_peak_rate: float = 0.08,
    summer_months: List[int] = None
) -> List[TariffPeriodDef]:
    """
    Create standard TOU tariff periods.

    Standard industrial TOU structure:
    - On-Peak: 12:00-18:00 weekdays
    - Mid-Peak: 08:00-12:00, 18:00-21:00 weekdays
    - Off-Peak: 21:00-08:00 and weekends

    Args:
        peak_rate: On-peak energy rate ($/kWh).
        mid_peak_rate: Mid-peak energy rate ($/kWh).
        off_peak_rate: Off-peak energy rate ($/kWh).
        summer_months: Summer months for higher rates (default: June-September).

    Returns:
        List of TariffPeriodDef objects.

    Example:
        >>> periods = create_standard_tou_periods(0.30, 0.18, 0.10)
        >>> for p in periods:
        ...     print(f"{p.period_name}: ${p.energy_rate_per_kwh}/kWh")
    """
    if summer_months is None:
        summer_months = [6, 7, 8, 9]  # June-September

    weekdays = [0, 1, 2, 3, 4]  # Monday-Friday
    weekend = [5, 6]  # Saturday-Sunday

    periods = [
        # Weekday periods
        TariffPeriodDef(
            period_name="on_peak",
            period_type=RatePeriod.ON_PEAK,
            start_hour=12,
            end_hour=18,
            days_of_week=weekdays,
            energy_rate_per_kwh=peak_rate,
            demand_rate_per_kw=0.0,
            months=summer_months
        ),
        TariffPeriodDef(
            period_name="mid_peak_morning",
            period_type=RatePeriod.MID_PEAK,
            start_hour=8,
            end_hour=12,
            days_of_week=weekdays,
            energy_rate_per_kwh=mid_peak_rate,
            demand_rate_per_kw=0.0,
            months=None
        ),
        TariffPeriodDef(
            period_name="mid_peak_evening",
            period_type=RatePeriod.MID_PEAK,
            start_hour=18,
            end_hour=21,
            days_of_week=weekdays,
            energy_rate_per_kwh=mid_peak_rate,
            demand_rate_per_kw=0.0,
            months=None
        ),
        TariffPeriodDef(
            period_name="off_peak_night",
            period_type=RatePeriod.OFF_PEAK,
            start_hour=21,
            end_hour=8,
            days_of_week=weekdays,
            energy_rate_per_kwh=off_peak_rate,
            demand_rate_per_kw=0.0,
            months=None
        ),
        # Weekend (all off-peak)
        TariffPeriodDef(
            period_name="off_peak_weekend",
            period_type=RatePeriod.OFF_PEAK,
            start_hour=0,
            end_hour=24,
            days_of_week=weekend,
            energy_rate_per_kwh=off_peak_rate,
            demand_rate_per_kw=0.0,
            months=None
        ),
    ]

    return periods


# =============================================================================
# Cost Calculation Functions
# =============================================================================

def calculate_energy_cost(
    timestamps: List[datetime],
    demands_kw: List[float],
    periods: List[TariffPeriodDef],
    resolution_minutes: int = 60
) -> CostBreakdown:
    """
    Calculate energy cost for a demand profile.

    Args:
        timestamps: List of timestamps for each demand point.
        demands_kw: Demand in kW at each timestamp.
        periods: Tariff period definitions.
        resolution_minutes: Time resolution in minutes.

    Returns:
        CostBreakdown with detailed cost analysis.

    Example:
        >>> timestamps = [datetime(2024, 6, 15, h, 0) for h in range(24)]
        >>> demands = [100] * 24  # Constant 100 kW
        >>> periods = create_standard_tou_periods()
        >>> cost = calculate_energy_cost(timestamps, demands, periods)
        >>> print(f"Total cost: ${cost.total_cost:.2f}")
    """
    if len(timestamps) != len(demands_kw):
        raise ValueError("timestamps and demands_kw must have same length")

    hours_per_slot = resolution_minutes / 60.0

    # Initialize accumulators
    on_peak_energy = 0.0
    on_peak_cost = 0.0
    mid_peak_energy = 0.0
    mid_peak_cost = 0.0
    off_peak_energy = 0.0
    off_peak_cost = 0.0

    peak_demand = 0.0
    on_peak_demand = 0.0

    for ts, demand in zip(timestamps, demands_kw):
        _, period_type, rate, _ = get_period_type(ts, periods)

        energy_kwh = demand * hours_per_slot
        cost = energy_kwh * rate

        # Track peak demand
        if demand > peak_demand:
            peak_demand = demand

        # Accumulate by period
        if period_type == RatePeriod.ON_PEAK or period_type == RatePeriod.CRITICAL_PEAK:
            on_peak_energy += energy_kwh
            on_peak_cost += cost
            if demand > on_peak_demand:
                on_peak_demand = demand

        elif period_type == RatePeriod.MID_PEAK:
            mid_peak_energy += energy_kwh
            mid_peak_cost += cost

        else:  # OFF_PEAK, SUPER_OFF_PEAK
            off_peak_energy += energy_kwh
            off_peak_cost += cost

    total_energy_cost = on_peak_cost + mid_peak_cost + off_peak_cost

    return CostBreakdown(
        total_cost=total_energy_cost,
        energy_cost=total_energy_cost,
        demand_cost=0.0,  # Calculated separately
        carbon_cost=0.0,  # Calculated separately
        on_peak_energy_kwh=on_peak_energy,
        on_peak_cost=on_peak_cost,
        mid_peak_energy_kwh=mid_peak_energy,
        mid_peak_cost=mid_peak_cost,
        off_peak_energy_kwh=off_peak_energy,
        off_peak_cost=off_peak_cost,
        peak_demand_kw=peak_demand,
        on_peak_demand_kw=on_peak_demand
    )


def calculate_demand_charge(
    peak_demand_kw: float,
    demand_structure: DemandChargeStructure,
    historical_peaks: Optional[List[float]] = None
) -> float:
    """
    Calculate monthly demand charge.

    Args:
        peak_demand_kw: Current month peak demand (kW).
        demand_structure: Demand charge structure.
        historical_peaks: Historical peak demands for ratchet.

    Returns:
        Demand charge ($).

    Example:
        >>> structure = DemandChargeStructure(monthly_rate_per_kw=15.0, ratchet_percentage=50)
        >>> charge = calculate_demand_charge(500, structure, [600, 550, 580])
        >>> print(f"Demand charge: ${charge:.2f}")
    """
    billing_demand = peak_demand_kw

    # Apply ratchet if specified
    if demand_structure.ratchet_percentage > 0 and historical_peaks:
        # Get historical peak
        historical_peak = max(historical_peaks) if historical_peaks else 0

        # Ratchet demand
        ratchet_demand = historical_peak * (demand_structure.ratchet_percentage / 100)

        # Billing demand is max of current and ratchet
        billing_demand = max(peak_demand_kw, ratchet_demand)

        logger.debug(
            f"Demand ratchet: current={peak_demand_kw:.1f}, "
            f"historical_peak={historical_peak:.1f}, "
            f"ratchet={ratchet_demand:.1f}, billing={billing_demand:.1f}"
        )

    charge = billing_demand * demand_structure.monthly_rate_per_kw

    return charge


def calculate_carbon_cost(
    energy_kwh: float,
    carbon_intensity_kg_per_kwh: float,
    carbon_price_per_tonne: float
) -> float:
    """
    Calculate carbon cost for energy consumption.

    Args:
        energy_kwh: Energy consumption (kWh).
        carbon_intensity_kg_per_kwh: Grid carbon intensity (kg CO2/kWh).
        carbon_price_per_tonne: Carbon price ($/tonne CO2).

    Returns:
        Carbon cost ($).

    Example:
        >>> cost = calculate_carbon_cost(1000, 0.4, 50)  # 1000 kWh, 0.4 kg/kWh, $50/tonne
        >>> print(f"Carbon cost: ${cost:.2f}")  # $20.00
    """
    emissions_kg = energy_kwh * carbon_intensity_kg_per_kwh
    emissions_tonnes = emissions_kg / 1000
    carbon_cost = emissions_tonnes * carbon_price_per_tonne

    return carbon_cost


# =============================================================================
# Tariff Analyzer Class
# =============================================================================

class TariffAnalyzer:
    """
    Tariff analysis for process heating scheduling.

    This class provides comprehensive tariff analysis including
    TOU rates, demand charges, and carbon costs.

    Zero-Hallucination Approach:
    - All calculations are deterministic rate lookups
    - No ML/LLM in cost calculations
    - Rates applied exactly as defined
    - Results are mathematically verifiable

    Attributes:
        periods: Tariff period definitions.
        demand_structure: Demand charge structure.
        carbon_intensity: Grid carbon intensity (kg CO2/kWh).
        carbon_price: Carbon price ($/tonne CO2).

    Example:
        >>> periods = create_standard_tou_periods(0.25, 0.15, 0.08)
        >>> analyzer = TariffAnalyzer(periods)
        >>> cost = analyzer.analyze_cost(timestamps, demands)
    """

    def __init__(
        self,
        periods: List[TariffPeriodDef],
        demand_structure: Optional[DemandChargeStructure] = None,
        carbon_intensity: float = 0.4,
        carbon_price: float = 0.0
    ):
        """
        Initialize TariffAnalyzer.

        Args:
            periods: Tariff period definitions.
            demand_structure: Optional demand charge structure.
            carbon_intensity: Grid carbon intensity (kg CO2/kWh).
            carbon_price: Carbon price ($/tonne CO2).
        """
        self.periods = periods
        self.demand_structure = demand_structure
        self.carbon_intensity = carbon_intensity
        self.carbon_price = carbon_price

        logger.info(
            f"Initialized TariffAnalyzer with {len(periods)} periods, "
            f"carbon_intensity={carbon_intensity} kg/kWh"
        )

    def get_rate(self, timestamp: datetime) -> Tuple[str, RatePeriod, float]:
        """
        Get energy rate for a timestamp.

        Args:
            timestamp: Datetime to look up.

        Returns:
            Tuple of (period_name, period_type, rate_per_kwh).
        """
        name, ptype, rate, _ = get_period_type(timestamp, self.periods)
        return name, ptype, rate

    def get_rate_schedule(
        self,
        start_time: datetime,
        hours: int,
        resolution_minutes: int = 60
    ) -> List[Tuple[datetime, str, float]]:
        """
        Get rate schedule for a time period.

        Args:
            start_time: Schedule start time.
            hours: Number of hours.
            resolution_minutes: Time resolution.

        Returns:
            List of (timestamp, period_name, rate) tuples.

        Example:
            >>> schedule = analyzer.get_rate_schedule(datetime.now(), 24)
            >>> for ts, period, rate in schedule:
            ...     print(f"{ts}: {period} @ ${rate}/kWh")
        """
        schedule = []
        n_slots = hours * 60 // resolution_minutes

        for i in range(n_slots):
            ts = start_time + timedelta(minutes=i * resolution_minutes)
            name, _, rate = self.get_rate(ts)
            schedule.append((ts, name, rate))

        return schedule

    def analyze_cost(
        self,
        timestamps: List[datetime],
        demands_kw: List[float],
        resolution_minutes: int = 60,
        historical_peaks: Optional[List[float]] = None
    ) -> CostBreakdown:
        """
        Analyze complete cost for a demand profile.

        Args:
            timestamps: Timestamps for demand values.
            demands_kw: Demand values in kW.
            resolution_minutes: Time resolution in minutes.
            historical_peaks: Historical peaks for demand ratchet.

        Returns:
            CostBreakdown with complete analysis.

        Example:
            >>> cost = analyzer.analyze_cost(timestamps, demands)
            >>> print(f"Total: ${cost.total_cost:.2f}")
            >>> print(f"Energy: ${cost.energy_cost:.2f}")
            >>> print(f"Demand: ${cost.demand_cost:.2f}")
        """
        # Calculate energy cost
        cost = calculate_energy_cost(
            timestamps, demands_kw, self.periods, resolution_minutes
        )

        # Calculate demand charge
        demand_cost = 0.0
        if self.demand_structure:
            # Use on-peak demand if peak_hours_only
            peak_for_charge = (cost.on_peak_demand_kw
                               if self.demand_structure.peak_hours_only
                               else cost.peak_demand_kw)
            demand_cost = calculate_demand_charge(
                peak_for_charge, self.demand_structure, historical_peaks
            )

        # Calculate carbon cost
        total_energy = (cost.on_peak_energy_kwh +
                        cost.mid_peak_energy_kwh +
                        cost.off_peak_energy_kwh)
        carbon_cost = calculate_carbon_cost(
            total_energy, self.carbon_intensity, self.carbon_price
        )

        # Update cost breakdown
        cost.demand_cost = demand_cost
        cost.carbon_cost = carbon_cost
        cost.total_cost = cost.energy_cost + demand_cost + carbon_cost

        return cost

    def calculate_arbitrage_potential(
        self,
        hours: int = 24,
        storage_capacity_kwh: float = 1000,
        storage_efficiency: float = 0.85
    ) -> Dict[str, float]:
        """
        Calculate potential arbitrage value from rate differentials.

        Args:
            hours: Analysis period in hours.
            storage_capacity_kwh: Storage capacity for arbitrage.
            storage_efficiency: Round-trip efficiency.

        Returns:
            Dictionary with arbitrage analysis.

        Example:
            >>> potential = analyzer.calculate_arbitrage_potential(24, 500)
            >>> print(f"Daily arbitrage: ${potential['daily_value']:.2f}")
        """
        # Get rates for the period
        schedule = self.get_rate_schedule(datetime.now(), hours)
        rates = [rate for _, _, rate in schedule]

        if not rates:
            return {'max_rate': 0, 'min_rate': 0, 'spread': 0, 'daily_value': 0}

        max_rate = max(rates)
        min_rate = min(rates)
        spread = max_rate - min_rate

        # Calculate arbitrage value
        # Value = spread * capacity * efficiency (per cycle)
        # Assume 1 cycle per day
        daily_value = spread * storage_capacity_kwh * storage_efficiency

        return {
            'max_rate': max_rate,
            'min_rate': min_rate,
            'spread': spread,
            'daily_value': daily_value,
            'monthly_value': daily_value * 30,
            'annual_value': daily_value * 365,
        }

    def identify_optimal_windows(
        self,
        start_time: datetime,
        hours: int,
        min_duration_hours: int = 2
    ) -> Dict[str, List[Tuple[datetime, datetime]]]:
        """
        Identify optimal charging and discharging windows.

        Args:
            start_time: Analysis start time.
            hours: Analysis period.
            min_duration_hours: Minimum window duration.

        Returns:
            Dict with 'charge_windows' and 'discharge_windows'.

        Example:
            >>> windows = analyzer.identify_optimal_windows(datetime.now(), 24)
            >>> for start, end in windows['charge_windows']:
            ...     print(f"Charge: {start} to {end}")
        """
        schedule = self.get_rate_schedule(start_time, hours, resolution_minutes=60)

        if not schedule:
            return {'charge_windows': [], 'discharge_windows': []}

        rates = [rate for _, _, rate in schedule]
        avg_rate = sum(rates) / len(rates)

        # Thresholds for windows
        low_threshold = avg_rate * 0.85  # 15% below average
        high_threshold = avg_rate * 1.15  # 15% above average

        charge_windows = []
        discharge_windows = []

        # Find contiguous low-rate periods (charge windows)
        in_low = False
        low_start = None

        for i, (ts, _, rate) in enumerate(schedule):
            if rate <= low_threshold:
                if not in_low:
                    in_low = True
                    low_start = ts
            else:
                if in_low:
                    # End of low period
                    duration = (ts - low_start).total_seconds() / 3600
                    if duration >= min_duration_hours:
                        charge_windows.append((low_start, ts))
                    in_low = False

        # Handle window extending to end
        if in_low and low_start:
            end_ts = schedule[-1][0] + timedelta(hours=1)
            duration = (end_ts - low_start).total_seconds() / 3600
            if duration >= min_duration_hours:
                charge_windows.append((low_start, end_ts))

        # Find contiguous high-rate periods (discharge windows)
        in_high = False
        high_start = None

        for i, (ts, _, rate) in enumerate(schedule):
            if rate >= high_threshold:
                if not in_high:
                    in_high = True
                    high_start = ts
            else:
                if in_high:
                    duration = (ts - high_start).total_seconds() / 3600
                    if duration >= min_duration_hours:
                        discharge_windows.append((high_start, ts))
                    in_high = False

        if in_high and high_start:
            end_ts = schedule[-1][0] + timedelta(hours=1)
            duration = (end_ts - high_start).total_seconds() / 3600
            if duration >= min_duration_hours:
                discharge_windows.append((high_start, end_ts))

        return {
            'charge_windows': charge_windows,
            'discharge_windows': discharge_windows,
            'avg_rate': avg_rate,
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
        }


# =============================================================================
# Real-Time Pricing Support
# =============================================================================

def interpolate_rtp_rates(
    known_rates: List[Tuple[datetime, float]],
    target_timestamps: List[datetime]
) -> List[float]:
    """
    Interpolate real-time pricing rates for target timestamps.

    Args:
        known_rates: List of (timestamp, rate) tuples with known rates.
        target_timestamps: Timestamps to interpolate rates for.

    Returns:
        List of interpolated rates.

    Example:
        >>> known = [(datetime(2024,6,15,0,0), 0.08), (datetime(2024,6,15,12,0), 0.20)]
        >>> targets = [datetime(2024,6,15,h,0) for h in range(24)]
        >>> rates = interpolate_rtp_rates(known, targets)
    """
    if not known_rates:
        return [0.10] * len(target_timestamps)  # Default rate

    # Sort known rates by timestamp
    sorted_rates = sorted(known_rates, key=lambda x: x[0])

    interpolated = []

    for target in target_timestamps:
        # Find surrounding known rates
        before = None
        after = None

        for ts, rate in sorted_rates:
            if ts <= target:
                before = (ts, rate)
            if ts >= target and after is None:
                after = (ts, rate)

        if before is None and after is None:
            # No known rates
            interpolated.append(0.10)
        elif before is None:
            # Before first known rate
            interpolated.append(after[1])
        elif after is None:
            # After last known rate
            interpolated.append(before[1])
        elif before[0] == after[0]:
            # Exact match
            interpolated.append(before[1])
        else:
            # Linear interpolation
            total_seconds = (after[0] - before[0]).total_seconds()
            elapsed_seconds = (target - before[0]).total_seconds()
            ratio = elapsed_seconds / total_seconds
            rate = before[1] + ratio * (after[1] - before[1])
            interpolated.append(rate)

    return interpolated


def apply_critical_peak_event(
    base_rates: List[float],
    event_start_hour: int,
    event_end_hour: int,
    multiplier: float = 3.0
) -> List[float]:
    """
    Apply critical peak pricing event to base rates.

    Args:
        base_rates: List of 24 hourly base rates.
        event_start_hour: Event start hour (0-23).
        event_end_hour: Event end hour (0-23).
        multiplier: Rate multiplier during event.

    Returns:
        Modified rates with CPP applied.

    Example:
        >>> base = [0.10] * 24
        >>> cpp_rates = apply_critical_peak_event(base, 14, 18, 5.0)
        >>> print(cpp_rates[14])  # Should be 0.50
    """
    if len(base_rates) != 24:
        logger.warning(f"Expected 24 hourly rates, got {len(base_rates)}")

    modified = base_rates.copy()

    for hour in range(24):
        if event_start_hour <= hour < event_end_hour:
            modified[hour] = base_rates[hour] * multiplier

    return modified
