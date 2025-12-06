"""
GL-019 HEATSCHEDULER - Demand Charge Optimization Module

Demand charge optimization including peak demand reduction, load shifting,
and real-time pricing response.

Key Features:
    - Peak demand reduction strategies
    - Load shifting to off-peak periods
    - Real-time pricing (RTP) response
    - Demand response (DR) event handling
    - Ratchet clause management
    - Zero-hallucination: Deterministic optimization algorithms

Author: GreenLang Team
Version: 1.0.0
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import statistics

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_019_heat_scheduler.config import (
    DemandChargeConfiguration,
    TariffConfiguration,
)
from greenlang.agents.process_heat.gl_019_heat_scheduler.schemas import (
    DemandChargeResult,
    DemandPeriod,
    DemandAlertLevel,
    LoadForecastResult,
    LoadForecastPoint,
    ScheduleActionItem,
    ScheduleAction,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DEMAND PERIOD ANALYZER
# =============================================================================

class DemandPeriodAnalyzer:
    """
    Analyzes demand periods from load profiles.

    Identifies peak demand periods, calculates rolling averages,
    and determines billing demand based on utility rules.
    """

    def __init__(
        self,
        demand_interval_minutes: int = 15,
        use_rolling_average: bool = True,
    ) -> None:
        """
        Initialize demand period analyzer.

        Args:
            demand_interval_minutes: Demand measurement interval
            use_rolling_average: Use rolling average for demand calculation
        """
        self._interval_minutes = demand_interval_minutes
        self._use_rolling = use_rolling_average

        logger.info(
            f"DemandPeriodAnalyzer initialized: "
            f"interval={demand_interval_minutes}min, rolling={use_rolling_average}"
        )

    def analyze_periods(
        self,
        load_forecast: LoadForecastResult,
        tariff_config: Optional[TariffConfiguration] = None,
    ) -> List[DemandPeriod]:
        """
        Analyze load forecast to identify demand periods.

        Args:
            load_forecast: Load forecast data
            tariff_config: Tariff configuration for peak periods

        Returns:
            List of DemandPeriod objects
        """
        if not load_forecast.forecast_points:
            return []

        periods: List[DemandPeriod] = []
        peak_hours = self._get_peak_hours(tariff_config)

        # Group forecast points by demand interval
        interval_points: Dict[datetime, List[LoadForecastPoint]] = {}

        for point in load_forecast.forecast_points:
            # Round down to interval boundary
            interval_start = self._round_to_interval(point.timestamp)
            if interval_start not in interval_points:
                interval_points[interval_start] = []
            interval_points[interval_start].append(point)

        # Calculate demand for each interval
        for interval_start, points in sorted(interval_points.items()):
            loads = [p.load_kw for p in points]
            avg_demand = statistics.mean(loads)
            peak_demand = max(loads)

            is_peak = interval_start.hour in peak_hours

            period = DemandPeriod(
                period_start=interval_start,
                period_end=interval_start + timedelta(minutes=self._interval_minutes),
                avg_demand_kw=round(avg_demand, 2),
                peak_demand_kw=round(peak_demand, 2),
                is_on_peak=is_peak,
            )
            periods.append(period)

        return periods

    def find_peak_period(
        self,
        periods: List[DemandPeriod],
        on_peak_only: bool = True,
    ) -> Optional[DemandPeriod]:
        """
        Find the period with highest demand.

        Args:
            periods: List of demand periods
            on_peak_only: Only consider on-peak periods

        Returns:
            Peak demand period or None
        """
        if not periods:
            return None

        filtered = periods
        if on_peak_only:
            filtered = [p for p in periods if p.is_on_peak]

        if not filtered:
            filtered = periods  # Fallback to all periods

        return max(filtered, key=lambda p: p.peak_demand_kw)

    def calculate_billing_demand(
        self,
        periods: List[DemandPeriod],
        ratchet_percentage: float = 0.0,
        annual_peak_kw: float = 0.0,
    ) -> float:
        """
        Calculate billing demand considering ratchet clause.

        Args:
            periods: Demand periods for billing cycle
            ratchet_percentage: Ratchet percentage (0-100)
            annual_peak_kw: Annual peak demand (for ratchet)

        Returns:
            Billing demand (kW)
        """
        if not periods:
            return 0.0

        # Current period peak
        current_peak = max(p.peak_demand_kw for p in periods)

        # Apply ratchet
        if ratchet_percentage > 0 and annual_peak_kw > 0:
            ratchet_demand = annual_peak_kw * (ratchet_percentage / 100.0)
            return max(current_peak, ratchet_demand)

        return current_peak

    def _round_to_interval(self, dt: datetime) -> datetime:
        """Round datetime down to interval boundary."""
        minutes = (dt.minute // self._interval_minutes) * self._interval_minutes
        return dt.replace(minute=minutes, second=0, microsecond=0)

    def _get_peak_hours(
        self,
        tariff_config: Optional[TariffConfiguration],
    ) -> set:
        """Get peak hours from tariff config."""
        if tariff_config is None:
            return {14, 15, 16, 17, 18, 19}  # Default

        start = tariff_config.peak_hours_start
        end = tariff_config.peak_hours_end

        if end > start:
            return set(range(start, end + 1))
        else:
            return set(range(start, 24)) | set(range(0, end + 1))


# =============================================================================
# LOAD SHIFTER
# =============================================================================

class LoadShifter:
    """
    Optimizes load shifting for demand charge reduction.

    Identifies shiftable loads and determines optimal timing
    to minimize peak demand and energy costs.
    """

    def __init__(
        self,
        max_shift_hours: int = 4,
        min_savings_threshold_usd: float = 5.0,
    ) -> None:
        """
        Initialize load shifter.

        Args:
            max_shift_hours: Maximum hours a load can be shifted
            min_savings_threshold_usd: Minimum savings to trigger shift
        """
        self._max_shift_hours = max_shift_hours
        self._min_savings = min_savings_threshold_usd

        logger.info(
            f"LoadShifter initialized: max_shift={max_shift_hours}h, "
            f"min_savings=${min_savings_threshold_usd}"
        )

    def identify_shift_opportunities(
        self,
        load_forecast: LoadForecastResult,
        demand_limit_kw: float,
        tariff_config: Optional[TariffConfiguration] = None,
    ) -> List[Dict[str, Any]]:
        """
        Identify load shifting opportunities.

        Args:
            load_forecast: Load forecast
            demand_limit_kw: Target demand limit
            tariff_config: Tariff configuration

        Returns:
            List of shift opportunities
        """
        opportunities: List[Dict[str, Any]] = []

        if not load_forecast.forecast_points:
            return opportunities

        # Find periods exceeding limit
        peak_periods: List[LoadForecastPoint] = []
        valley_periods: List[LoadForecastPoint] = []

        for point in load_forecast.forecast_points:
            if point.load_kw > demand_limit_kw:
                peak_periods.append(point)
            elif point.load_kw < demand_limit_kw * 0.7:
                valley_periods.append(point)

        # Match peaks with valleys for shifting
        for peak in peak_periods:
            excess = peak.load_kw - demand_limit_kw

            # Find best valley within shift window
            best_valley = None
            best_savings = 0.0

            for valley in valley_periods:
                time_diff = abs((valley.timestamp - peak.timestamp).total_seconds() / 3600)

                if time_diff <= self._max_shift_hours:
                    available_capacity = demand_limit_kw * 0.7 - valley.load_kw
                    shiftable = min(excess, available_capacity)

                    if shiftable > 0:
                        # Calculate savings
                        savings = self._calculate_shift_savings(
                            shiftable,
                            peak.timestamp,
                            valley.timestamp,
                            tariff_config,
                        )

                        if savings > best_savings and savings >= self._min_savings:
                            best_savings = savings
                            best_valley = valley

            if best_valley is not None:
                opportunities.append({
                    "from_time": peak.timestamp,
                    "to_time": best_valley.timestamp,
                    "load_kw": min(excess, demand_limit_kw * 0.3),
                    "savings_usd": round(best_savings, 2),
                    "shift_hours": abs(
                        (best_valley.timestamp - peak.timestamp).total_seconds() / 3600
                    ),
                })

        return opportunities

    def generate_shift_actions(
        self,
        opportunities: List[Dict[str, Any]],
    ) -> List[ScheduleActionItem]:
        """
        Generate schedule actions for load shifts.

        Args:
            opportunities: Identified shift opportunities

        Returns:
            List of schedule actions
        """
        actions: List[ScheduleActionItem] = []

        for opp in opportunities:
            # Reduction action at peak time
            actions.append(ScheduleActionItem(
                timestamp=opp["from_time"],
                action_type=ScheduleAction.LOAD_SHIFT,
                power_setpoint_kw=-opp["load_kw"],  # Reduction
                duration_minutes=15,
                reason=f"Peak reduction: shift {opp['load_kw']:.0f}kW",
                expected_savings_usd=opp["savings_usd"],
                priority=8,
            ))

            # Addition action at valley time
            actions.append(ScheduleActionItem(
                timestamp=opp["to_time"],
                action_type=ScheduleAction.LOAD_SHIFT,
                power_setpoint_kw=opp["load_kw"],  # Addition
                duration_minutes=15,
                reason=f"Shifted load from {opp['from_time'].strftime('%H:%M')}",
                expected_savings_usd=0.0,
                priority=5,
            ))

        return actions

    def _calculate_shift_savings(
        self,
        load_kw: float,
        from_time: datetime,
        to_time: datetime,
        tariff_config: Optional[TariffConfiguration],
    ) -> float:
        """Calculate savings from shifting load."""
        if tariff_config is None:
            return load_kw * 0.05  # Default $0.05/kW shifted

        # Get rates
        from_rate = self._get_rate_at_time(from_time, tariff_config)
        to_rate = self._get_rate_at_time(to_time, tariff_config)

        # Energy savings (per 15-minute interval)
        energy_kwh = load_kw * 0.25
        energy_savings = energy_kwh * (from_rate - to_rate)

        # Demand savings (if reduces peak)
        demand_savings = load_kw * tariff_config.demand_charge_per_kw / 720  # Monthly amortized

        return max(0, energy_savings + demand_savings)

    def _get_rate_at_time(
        self,
        dt: datetime,
        tariff: TariffConfiguration,
    ) -> float:
        """Get energy rate at given time."""
        hour = dt.hour
        if tariff.peak_hours_start <= hour < tariff.peak_hours_end:
            return tariff.peak_rate_per_kwh
        return tariff.off_peak_rate_per_kwh


# =============================================================================
# DEMAND RESPONSE HANDLER
# =============================================================================

class DemandResponseHandler:
    """
    Handles demand response events and curtailment.

    Manages participation in utility demand response programs
    with proper notification and curtailment execution.
    """

    def __init__(
        self,
        config: DemandChargeConfiguration,
    ) -> None:
        """
        Initialize demand response handler.

        Args:
            config: Demand charge configuration
        """
        self._enabled = config.enable_demand_response
        self._threshold_kw = config.demand_response_threshold_kw
        self._max_curtailment_pct = config.max_demand_curtailment_pct
        self._notification_lead_time = config.dr_notification_lead_time_minutes

        self._active_events: List[Dict[str, Any]] = []

        logger.info(
            f"DemandResponseHandler initialized: enabled={self._enabled}, "
            f"threshold={self._threshold_kw}kW"
        )

    @property
    def is_enabled(self) -> bool:
        """Check if DR is enabled."""
        return self._enabled

    def register_dr_event(
        self,
        event_id: str,
        start_time: datetime,
        end_time: datetime,
        curtailment_target_kw: float,
    ) -> None:
        """
        Register a demand response event.

        Args:
            event_id: Event identifier
            start_time: Event start time
            end_time: Event end time
            curtailment_target_kw: Target curtailment (kW)
        """
        event = {
            "event_id": event_id,
            "start_time": start_time,
            "end_time": end_time,
            "target_kw": curtailment_target_kw,
            "status": "scheduled",
            "registered_at": datetime.now(timezone.utc),
        }
        self._active_events.append(event)

        logger.info(
            f"DR event registered: {event_id}, "
            f"target={curtailment_target_kw}kW, "
            f"start={start_time}"
        )

    def get_curtailment_for_time(
        self,
        target_time: datetime,
        baseline_load_kw: float,
    ) -> Tuple[float, Optional[str]]:
        """
        Get required curtailment for a given time.

        Args:
            target_time: Time to check
            baseline_load_kw: Baseline load at that time

        Returns:
            Tuple of (curtailment_kw, event_id or None)
        """
        if not self._enabled:
            return (0.0, None)

        for event in self._active_events:
            if event["start_time"] <= target_time <= event["end_time"]:
                # Event is active
                max_curtailment = baseline_load_kw * (self._max_curtailment_pct / 100.0)
                curtailment = min(event["target_kw"], max_curtailment)
                return (curtailment, event["event_id"])

        return (0.0, None)

    def generate_dr_actions(
        self,
        event_id: str,
    ) -> List[ScheduleActionItem]:
        """
        Generate schedule actions for a DR event.

        Args:
            event_id: DR event identifier

        Returns:
            List of schedule actions
        """
        actions: List[ScheduleActionItem] = []

        event = None
        for e in self._active_events:
            if e["event_id"] == event_id:
                event = e
                break

        if event is None:
            return actions

        # Pre-event notification action
        notification_time = event["start_time"] - timedelta(
            minutes=self._notification_lead_time
        )
        actions.append(ScheduleActionItem(
            timestamp=notification_time,
            action_type=ScheduleAction.RAMP_DOWN,
            power_setpoint_kw=event["target_kw"],
            duration_minutes=self._notification_lead_time,
            reason=f"DR event {event_id} preparation",
            expected_savings_usd=0.0,
            priority=10,
            is_mandatory=True,
        ))

        # Event start action
        actions.append(ScheduleActionItem(
            timestamp=event["start_time"],
            action_type=ScheduleAction.LOAD_SHIFT,
            power_setpoint_kw=-event["target_kw"],
            duration_minutes=int(
                (event["end_time"] - event["start_time"]).total_seconds() / 60
            ),
            reason=f"DR event {event_id} curtailment",
            expected_savings_usd=0.0,  # DR payments handled separately
            priority=10,
            is_mandatory=True,
        ))

        return actions


# =============================================================================
# DEMAND CHARGE OPTIMIZER
# =============================================================================

class DemandChargeOptimizer:
    """
    Main optimizer for demand charge management.

    Coordinates peak reduction, load shifting, and demand response
    to minimize demand charges while meeting operational constraints.

    All optimization is DETERMINISTIC with no ML/LLM inference.
    """

    def __init__(
        self,
        config: DemandChargeConfiguration,
        tariff_config: Optional[TariffConfiguration] = None,
    ) -> None:
        """
        Initialize demand charge optimizer.

        Args:
            config: Demand charge configuration
            tariff_config: Tariff configuration
        """
        self._config = config
        self._tariff = tariff_config

        # Initialize components
        self._analyzer = DemandPeriodAnalyzer(
            demand_interval_minutes=config.demand_interval_minutes,
            use_rolling_average=config.rolling_demand_average,
        )

        self._shifter = LoadShifter(
            max_shift_hours=config.max_shift_hours,
            min_savings_threshold_usd=config.min_shift_savings_threshold_usd,
        )

        self._dr_handler = DemandResponseHandler(config)

        # Track state
        self._annual_peak_kw = config.annual_ratchet_peak_kw
        self._monthly_peak_kw = 0.0

        logger.info(
            f"DemandChargeOptimizer initialized: "
            f"limit={config.peak_demand_limit_kw}kW, "
            f"load_shifting={config.enable_load_shifting}"
        )

    def optimize(
        self,
        load_forecast: LoadForecastResult,
        storage_capacity_kw: float = 0.0,
    ) -> DemandChargeResult:
        """
        Optimize demand charges for the forecast horizon.

        Args:
            load_forecast: Load forecast
            storage_capacity_kw: Available storage discharge capacity

        Returns:
            DemandChargeResult with optimization results
        """
        start_time = datetime.now(timezone.utc)
        logger.info("Optimizing demand charges")

        # Analyze demand periods
        periods = self._analyzer.analyze_periods(load_forecast, self._tariff)

        # Find baseline peak
        baseline_peak_period = self._analyzer.find_peak_period(periods, on_peak_only=True)
        baseline_peak_kw = baseline_peak_period.peak_demand_kw if baseline_peak_period else 0.0
        baseline_time = baseline_peak_period.period_start if baseline_peak_period else None

        # Calculate potential reductions
        optimized_peak_kw = baseline_peak_kw
        peak_reduction_kw = 0.0
        load_shifted_kwh = 0.0
        load_shift_savings = 0.0

        # Strategy 1: Storage discharge during peak
        if storage_capacity_kw > 0:
            storage_reduction = min(
                storage_capacity_kw,
                baseline_peak_kw - self._config.peak_demand_limit_kw
            )
            if storage_reduction > 0:
                optimized_peak_kw -= storage_reduction
                peak_reduction_kw += storage_reduction
                logger.debug(f"Storage reduction: {storage_reduction:.0f}kW")

        # Strategy 2: Load shifting
        if self._config.enable_load_shifting and optimized_peak_kw > self._config.peak_demand_limit_kw:
            opportunities = self._shifter.identify_shift_opportunities(
                load_forecast,
                self._config.peak_demand_limit_kw,
                self._tariff,
            )

            for opp in opportunities:
                shift_reduction = min(
                    opp["load_kw"],
                    optimized_peak_kw - self._config.peak_demand_limit_kw
                )
                if shift_reduction > 0:
                    optimized_peak_kw -= shift_reduction
                    peak_reduction_kw += shift_reduction
                    load_shifted_kwh += shift_reduction * 0.25  # 15-min interval
                    load_shift_savings += opp["savings_usd"]

        # Calculate costs
        baseline_demand_charge = self._calculate_demand_charge(baseline_peak_kw)
        optimized_demand_charge = self._calculate_demand_charge(optimized_peak_kw)
        demand_savings = baseline_demand_charge - optimized_demand_charge

        # Calculate ratchet impact
        ratchet_impact = 0.0
        if optimized_peak_kw > self._annual_peak_kw:
            # New annual peak - calculate increased ratchet costs
            ratchet_impact = self._calculate_ratchet_impact(
                optimized_peak_kw,
                self._annual_peak_kw,
            )
            self._annual_peak_kw = optimized_peak_kw

        # Update monthly peak
        if optimized_peak_kw > self._monthly_peak_kw:
            self._monthly_peak_kw = optimized_peak_kw

        # Check alerts
        alert_level = None
        alert_message = None
        peak_exceeded = optimized_peak_kw > self._config.peak_demand_limit_kw

        if peak_exceeded:
            excess_pct = (
                (optimized_peak_kw - self._config.peak_demand_limit_kw) /
                self._config.peak_demand_limit_kw * 100
            )
            if excess_pct > 20:
                alert_level = DemandAlertLevel.CRITICAL
                alert_message = f"Peak demand {excess_pct:.0f}% over limit"
            elif excess_pct > 10:
                alert_level = DemandAlertLevel.WARNING
                alert_message = f"Peak demand {excess_pct:.0f}% over limit"
            else:
                alert_level = DemandAlertLevel.INFO
                alert_message = f"Peak demand slightly over limit"

        # Find optimized peak time
        optimized_peak_time = baseline_time
        if peak_reduction_kw > 0 and load_forecast.forecast_points:
            # Find new peak after reductions
            max_load = 0.0
            for point in load_forecast.forecast_points:
                adjusted_load = point.load_kw
                # Simple approximation of redistributed load
                if adjusted_load > max_load:
                    max_load = adjusted_load
                    optimized_peak_time = point.timestamp

        result = DemandChargeResult(
            timestamp=datetime.now(timezone.utc),
            baseline_peak_kw=round(baseline_peak_kw, 2),
            optimized_peak_kw=round(optimized_peak_kw, 2),
            peak_reduction_kw=round(peak_reduction_kw, 2),
            peak_reduction_pct=round(
                peak_reduction_kw / baseline_peak_kw * 100 if baseline_peak_kw > 0 else 0,
                1
            ),
            peak_time_baseline=baseline_time,
            peak_time_optimized=optimized_peak_time,
            baseline_demand_charge_usd=round(baseline_demand_charge, 2),
            optimized_demand_charge_usd=round(optimized_demand_charge, 2),
            demand_charge_savings_usd=round(demand_savings, 2),
            annual_ratchet_peak_kw=round(self._annual_peak_kw, 2),
            ratchet_impact_usd=round(ratchet_impact, 2),
            load_shifted_kwh=round(load_shifted_kwh, 2),
            load_shift_savings_usd=round(load_shift_savings, 2),
            peak_limit_exceeded=peak_exceeded,
            alert_level=alert_level,
            alert_message=alert_message,
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            f"Demand optimization complete: "
            f"baseline={baseline_peak_kw:.0f}kW, optimized={optimized_peak_kw:.0f}kW, "
            f"savings=${demand_savings:.2f} ({processing_time:.1f}ms)"
        )

        return result

    def _calculate_demand_charge(self, peak_kw: float) -> float:
        """Calculate demand charge for given peak."""
        if self._tariff is None:
            return peak_kw * 15.0  # Default $15/kW

        charge = peak_kw * self._tariff.demand_charge_per_kw

        # Add peak demand charge if applicable
        if self._tariff.peak_demand_charge_per_kw > 0:
            charge += peak_kw * self._tariff.peak_demand_charge_per_kw

        return charge

    def _calculate_ratchet_impact(
        self,
        new_peak_kw: float,
        previous_peak_kw: float,
    ) -> float:
        """Calculate cost impact of new ratchet peak."""
        if self._config.ratchet_percentage == 0:
            return 0.0

        # Estimate annual impact
        monthly_increase = (new_peak_kw - previous_peak_kw) * (
            self._config.ratchet_percentage / 100.0
        )
        monthly_cost = monthly_increase * self._tariff.demand_charge_per_kw if self._tariff else monthly_increase * 15.0

        # 12 months impact
        return monthly_cost * 12

    def register_dr_event(
        self,
        event_id: str,
        start_time: datetime,
        end_time: datetime,
        target_kw: float,
    ) -> None:
        """Register a demand response event."""
        self._dr_handler.register_dr_event(
            event_id, start_time, end_time, target_kw
        )

    def get_dr_actions(self, event_id: str) -> List[ScheduleActionItem]:
        """Get schedule actions for a DR event."""
        return self._dr_handler.generate_dr_actions(event_id)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DemandPeriodAnalyzer",
    "LoadShifter",
    "DemandResponseHandler",
    "DemandChargeOptimizer",
]
