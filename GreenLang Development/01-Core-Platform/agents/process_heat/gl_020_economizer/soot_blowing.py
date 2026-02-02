"""
GL-020 ECONOPULSE - Soot Blower Optimizer

Optimizes soot blower operation to:
- Minimize steam consumption
- Maximize cleanliness (heat transfer effectiveness)
- Provide intelligent scheduling based on fouling indicators
- Track blowing effectiveness

Standards Reference:
    - ASME PTC 4.3 Air Heater Test Code
    - ASME PTC 4.1 Steam Generating Units

Zero-Hallucination: All calculations use deterministic formulas with full provenance.
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Steam enthalpy at typical soot blowing conditions (BTU/lb)
# Assuming 200 psig saturated steam
SOOT_BLOWING_STEAM_ENTHALPY = 1199.0

# Typical steam cost ($/1000 lb)
STEAM_COST_PER_KLBS = 8.0

# Natural gas cost for efficiency calculations ($/MMBtu)
FUEL_COST_PER_MMBTU = 5.0


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SootBlowerConfig:
    """Configuration for soot blower optimization."""

    # Soot blower inventory
    num_blowers: int = 4
    steam_per_blower_lb: float = 500.0
    blowing_duration_s: int = 90
    steam_pressure_psig: float = 200.0

    # Scheduling parameters
    fixed_schedule_enabled: bool = False
    fixed_interval_hours: float = 8.0
    min_interval_hours: float = 2.0
    max_interval_hours: float = 12.0

    # Trigger thresholds
    dp_trigger_ratio: float = 1.2
    effectiveness_trigger_ratio: float = 0.95
    exit_temp_rise_trigger_f: float = 20.0

    # Economic parameters
    steam_cost_per_klbs: float = STEAM_COST_PER_KLBS
    fuel_cost_per_mmbtu: float = FUEL_COST_PER_MMBTU
    economizer_duty_btu_hr: float = 20_000_000.0


@dataclass
class SootBlowerInput:
    """Input data for soot blower optimization."""

    timestamp: datetime

    # Current fouling state
    gas_side_dp_ratio: float  # actual/design
    effectiveness_ratio: float  # actual/design

    # Operating conditions
    boiler_load_pct: float

    # Optional fields with defaults
    gas_outlet_temp_deviation_f: float = 0.0
    steam_available: bool = True
    steam_pressure_psig: Optional[float] = None

    # History
    hours_since_last_blow: float = 0.0
    last_blow_timestamp: Optional[datetime] = None

    # Pre/post blow data for effectiveness tracking
    pre_blow_effectiveness: Optional[float] = None
    post_blow_effectiveness: Optional[float] = None
    pre_blow_dp_ratio: Optional[float] = None
    post_blow_dp_ratio: Optional[float] = None


@dataclass
class BlowEffectivenessRecord:
    """Record of a single blow cycle effectiveness."""
    timestamp: datetime
    pre_blow_dp_ratio: float
    post_blow_dp_ratio: float
    pre_blow_effectiveness: float
    post_blow_effectiveness: float
    dp_improvement_pct: float
    effectiveness_gain: float
    steam_used_lb: float


@dataclass
class SootBlowerResult:
    """Result of soot blower optimization analysis."""

    # Recommendation
    blowing_recommended: bool
    blowing_status: str  # idle, scheduled, in_progress, completed, bypassed

    # Timing
    hours_since_last_blow: float
    recommended_next_blow_hours: float
    optimal_blow_interval_hours: float

    # Trigger analysis
    dp_trigger_active: bool
    effectiveness_trigger_active: bool
    time_trigger_active: bool
    trigger_reason: str

    # Steam consumption
    estimated_steam_per_cycle_lb: float
    steam_savings_vs_fixed_pct: float
    daily_steam_consumption_lb: float

    # Effectiveness tracking
    pre_blow_effectiveness: Optional[float]
    post_blow_effectiveness: Optional[float]
    blow_effectiveness_gain: Optional[float]

    # Optimization metrics
    blowing_efficiency_score: float  # 0-1

    # Economics
    estimated_daily_steam_cost: float
    estimated_efficiency_loss_cost: float
    net_economic_impact: float

    # Provenance
    calculation_method: str
    provenance_hash: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# SOOT BLOWER OPTIMIZER
# =============================================================================

class SootBlowerOptimizer:
    """
    Optimizes soot blower operation for economizers.

    Soot blowing removes ash and soot deposits from gas-side surfaces,
    restoring heat transfer effectiveness. However, excessive blowing:
    - Wastes steam
    - Can cause tube erosion
    - Reduces availability

    Intelligent optimization balances:
    1. Fouling penalty (efficiency loss from deposits)
    2. Blowing cost (steam consumption)
    3. Erosion risk (minimizing unnecessary cycles)

    Optimization approach:
    - Monitor DP ratio, effectiveness, and outlet temp
    - Trigger blowing when indicators exceed thresholds
    - Track effectiveness of each blow cycle
    - Adjust optimal interval based on fouling rate
    - Compare against fixed schedule for savings calculation

    Reference: ASME PTC 4.3 Air Heater Test Code
    """

    def __init__(self, config: SootBlowerConfig):
        """
        Initialize soot blower optimizer.

        Args:
            config: Soot blower configuration
        """
        self.config = config
        self.blow_history: List[BlowEffectivenessRecord] = []
        self._adaptive_interval_hours = config.fixed_interval_hours
        logger.info(
            f"SootBlowerOptimizer initialized: {config.num_blowers} blowers, "
            f"interval={config.fixed_interval_hours}h"
        )

    def check_triggers(
        self,
        dp_ratio: float,
        effectiveness_ratio: float,
        outlet_temp_deviation_f: float,
        hours_since_blow: float,
    ) -> Tuple[bool, bool, bool, str]:
        """
        Check if any blowing triggers are active.

        Triggers:
        1. DP ratio exceeds threshold (deposit buildup)
        2. Effectiveness drops below threshold (heat transfer degraded)
        3. Maximum time interval exceeded (preventive)

        Args:
            dp_ratio: Current DP ratio (actual/design)
            effectiveness_ratio: Current effectiveness ratio
            outlet_temp_deviation_f: Gas outlet temp deviation
            hours_since_blow: Hours since last blow

        Returns:
            Tuple of (dp_trigger, effectiveness_trigger, time_trigger, reason)
        """
        dp_trigger = dp_ratio >= self.config.dp_trigger_ratio
        effectiveness_trigger = effectiveness_ratio <= self.config.effectiveness_trigger_ratio
        time_trigger = hours_since_blow >= self.config.max_interval_hours

        # Also check outlet temp rise
        temp_trigger = outlet_temp_deviation_f >= self.config.exit_temp_rise_trigger_f

        # Build reason string
        reasons = []
        if dp_trigger:
            reasons.append(f"DP ratio {dp_ratio:.2f} >= {self.config.dp_trigger_ratio}")
        if effectiveness_trigger:
            reasons.append(
                f"Effectiveness {effectiveness_ratio:.2f} <= "
                f"{self.config.effectiveness_trigger_ratio}"
            )
        if temp_trigger:
            reasons.append(f"Outlet temp +{outlet_temp_deviation_f:.1f}F")
        if time_trigger:
            reasons.append(f"Time {hours_since_blow:.1f}h >= max {self.config.max_interval_hours}h")

        reason = "; ".join(reasons) if reasons else "No triggers active"

        # Combine effectiveness and temp triggers
        effectiveness_or_temp = effectiveness_trigger or temp_trigger

        return dp_trigger, effectiveness_or_temp, time_trigger, reason

    def calculate_optimal_interval(
        self,
        fouling_rate_pct_per_hour: float,
        blow_effectiveness: float,
    ) -> float:
        """
        Calculate optimal blowing interval based on fouling rate.

        The optimal interval balances:
        - Cost of fouling (efficiency loss between blows)
        - Cost of blowing (steam consumption)

        Simple model:
        - Efficiency loss accumulates linearly with time
        - Each blow has fixed cost
        - Optimize interval to minimize total cost

        Economic optimal interval:
        T_opt = sqrt(2 * C_blow / (C_fuel * fouling_rate))

        Args:
            fouling_rate_pct_per_hour: Fouling rate (% efficiency loss per hour)
            blow_effectiveness: Effectiveness of blowing (0-1)

        Returns:
            Optimal interval in hours
        """
        if fouling_rate_pct_per_hour <= 0:
            return self.config.max_interval_hours

        # Cost per blow (steam cost)
        steam_per_cycle = self.config.num_blowers * self.config.steam_per_blower_lb
        blow_cost = steam_per_cycle * self.config.steam_cost_per_klbs / 1000

        # Cost of efficiency loss (fuel cost per % per hour)
        # Assuming efficiency loss leads to proportional fuel waste
        efficiency_cost_per_pct = (
            self.config.economizer_duty_btu_hr *
            self.config.fuel_cost_per_mmbtu /
            1_000_000 / 100 * blow_effectiveness
        )

        if efficiency_cost_per_pct <= 0:
            return self.config.max_interval_hours

        # Optimal interval (square root formula for economic optimization)
        # T_opt minimizes: C_blow/T + C_fuel * rate * T / 2
        optimal = math.sqrt(2 * blow_cost / (efficiency_cost_per_pct * fouling_rate_pct_per_hour))

        # Clamp to configured limits
        optimal = max(self.config.min_interval_hours, min(optimal, self.config.max_interval_hours))

        return optimal

    def estimate_fouling_rate(
        self,
        blow_history: List[BlowEffectivenessRecord],
    ) -> float:
        """
        Estimate fouling rate from historical blow data.

        Uses the effectiveness degradation between blows to estimate
        the fouling rate in % per hour.

        Args:
            blow_history: List of blow effectiveness records

        Returns:
            Estimated fouling rate (% effectiveness loss per hour)
        """
        if len(blow_history) < 2:
            # Default fouling rate assumption
            return 0.5  # 0.5% per hour

        # Calculate average effectiveness loss rate between blows
        rates = []

        for i in range(1, len(blow_history)):
            prev = blow_history[i - 1]
            curr = blow_history[i]

            # Time between blows
            hours = (curr.timestamp - prev.timestamp).total_seconds() / 3600

            if hours > 0:
                # Effectiveness loss (post-blow to pre-blow of next cycle)
                loss = (prev.post_blow_effectiveness - curr.pre_blow_effectiveness) * 100
                rate = loss / hours
                if rate > 0:
                    rates.append(rate)

        if rates:
            return sum(rates) / len(rates)
        else:
            return 0.5  # Default

    def calculate_blow_effectiveness(
        self,
        pre_blow_dp_ratio: float,
        post_blow_dp_ratio: float,
        pre_blow_effectiveness: float,
        post_blow_effectiveness: float,
    ) -> Tuple[float, float]:
        """
        Calculate the effectiveness of a blow cycle.

        Args:
            pre_blow_dp_ratio: DP ratio before blowing
            post_blow_dp_ratio: DP ratio after blowing
            pre_blow_effectiveness: Heat transfer effectiveness before
            post_blow_effectiveness: Heat transfer effectiveness after

        Returns:
            Tuple of (dp_improvement_pct, effectiveness_gain)
        """
        # DP improvement (% reduction)
        if pre_blow_dp_ratio > 0:
            dp_improvement_pct = (
                (pre_blow_dp_ratio - post_blow_dp_ratio) / pre_blow_dp_ratio * 100
            )
        else:
            dp_improvement_pct = 0.0

        # Effectiveness gain
        effectiveness_gain = post_blow_effectiveness - pre_blow_effectiveness

        return dp_improvement_pct, effectiveness_gain

    def record_blow_cycle(
        self,
        timestamp: datetime,
        pre_blow_dp_ratio: float,
        post_blow_dp_ratio: float,
        pre_blow_effectiveness: float,
        post_blow_effectiveness: float,
    ) -> BlowEffectivenessRecord:
        """
        Record a blow cycle for effectiveness tracking.

        Args:
            timestamp: Blow cycle timestamp
            pre_blow_dp_ratio: DP ratio before blowing
            post_blow_dp_ratio: DP ratio after blowing
            pre_blow_effectiveness: Effectiveness before
            post_blow_effectiveness: Effectiveness after

        Returns:
            BlowEffectivenessRecord
        """
        dp_improvement, eff_gain = self.calculate_blow_effectiveness(
            pre_blow_dp_ratio,
            post_blow_dp_ratio,
            pre_blow_effectiveness,
            post_blow_effectiveness,
        )

        steam_used = self.config.num_blowers * self.config.steam_per_blower_lb

        record = BlowEffectivenessRecord(
            timestamp=timestamp,
            pre_blow_dp_ratio=pre_blow_dp_ratio,
            post_blow_dp_ratio=post_blow_dp_ratio,
            pre_blow_effectiveness=pre_blow_effectiveness,
            post_blow_effectiveness=post_blow_effectiveness,
            dp_improvement_pct=dp_improvement,
            effectiveness_gain=eff_gain,
            steam_used_lb=steam_used,
        )

        self.blow_history.append(record)

        # Keep only last 100 records
        if len(self.blow_history) > 100:
            self.blow_history = self.blow_history[-100:]

        logger.info(
            f"Blow cycle recorded: DP improvement={dp_improvement:.1f}%, "
            f"effectiveness gain={eff_gain:.3f}"
        )

        return record

    def calculate_steam_savings(
        self,
        optimal_interval_hours: float,
        fixed_interval_hours: float,
    ) -> Tuple[float, float, float]:
        """
        Calculate steam savings vs fixed schedule.

        Args:
            optimal_interval_hours: Optimized blowing interval
            fixed_interval_hours: Fixed schedule interval

        Returns:
            Tuple of (savings_pct, daily_steam_optimal, daily_steam_fixed)
        """
        steam_per_cycle = self.config.num_blowers * self.config.steam_per_blower_lb

        # Daily cycles
        cycles_fixed = 24.0 / fixed_interval_hours
        cycles_optimal = 24.0 / optimal_interval_hours

        # Daily steam consumption
        daily_steam_fixed = cycles_fixed * steam_per_cycle
        daily_steam_optimal = cycles_optimal * steam_per_cycle

        # Savings percentage
        if daily_steam_fixed > 0:
            savings_pct = (daily_steam_fixed - daily_steam_optimal) / daily_steam_fixed * 100
        else:
            savings_pct = 0.0

        return savings_pct, daily_steam_optimal, daily_steam_fixed

    def calculate_blowing_efficiency_score(
        self,
        blow_history: List[BlowEffectivenessRecord],
    ) -> float:
        """
        Calculate blowing efficiency score (0-1).

        Score based on:
        - Consistency of blow effectiveness
        - DP improvement achieved
        - Timing optimization

        Args:
            blow_history: List of blow records

        Returns:
            Efficiency score (0-1)
        """
        if not blow_history:
            return 0.5  # No data, neutral score

        # Calculate average effectiveness gain
        eff_gains = [r.effectiveness_gain for r in blow_history[-10:]]
        avg_eff_gain = sum(eff_gains) / len(eff_gains) if eff_gains else 0

        # Calculate average DP improvement
        dp_improvements = [r.dp_improvement_pct for r in blow_history[-10:]]
        avg_dp_improvement = sum(dp_improvements) / len(dp_improvements) if dp_improvements else 0

        # Score components
        # Good effectiveness gain: 0.03 to 0.08 is typical
        eff_score = min(1.0, avg_eff_gain / 0.05) if avg_eff_gain > 0 else 0.3

        # Good DP improvement: 10-20% is typical
        dp_score = min(1.0, avg_dp_improvement / 15) if avg_dp_improvement > 0 else 0.3

        # Combined score (weighted average)
        score = 0.4 * eff_score + 0.4 * dp_score + 0.2

        return min(1.0, max(0.0, score))

    def calculate_economics(
        self,
        daily_steam_lb: float,
        effectiveness_ratio: float,
    ) -> Tuple[float, float, float]:
        """
        Calculate economic impact of soot blowing strategy.

        Args:
            daily_steam_lb: Daily steam consumption (lb)
            effectiveness_ratio: Current effectiveness ratio

        Returns:
            Tuple of (daily_steam_cost, daily_efficiency_loss_cost, net_impact)
        """
        # Steam cost
        daily_steam_cost = daily_steam_lb * self.config.steam_cost_per_klbs / 1000

        # Efficiency loss cost
        # Effectiveness ratio of 0.95 means 5% below design
        efficiency_loss_pct = (1.0 - effectiveness_ratio) * 100

        # Fuel waste from efficiency loss
        daily_duty = self.config.economizer_duty_btu_hr * 24
        fuel_waste_btu = daily_duty * (efficiency_loss_pct / 100)
        efficiency_loss_cost = fuel_waste_btu * self.config.fuel_cost_per_mmbtu / 1_000_000

        # Net impact (positive = saving, negative = cost)
        net_impact = efficiency_loss_cost - daily_steam_cost

        return daily_steam_cost, efficiency_loss_cost, net_impact

    def optimize(self, input_data: SootBlowerInput) -> SootBlowerResult:
        """
        Perform soot blower optimization analysis.

        Args:
            input_data: SootBlowerInput with current conditions

        Returns:
            SootBlowerResult with recommendations
        """
        # Check triggers
        dp_trigger, eff_trigger, time_trigger, reason = self.check_triggers(
            input_data.gas_side_dp_ratio,
            input_data.effectiveness_ratio,
            input_data.gas_outlet_temp_deviation_f,
            input_data.hours_since_last_blow,
        )

        # Any trigger active?
        any_trigger = dp_trigger or eff_trigger or time_trigger

        # Determine if blowing is recommended
        blowing_recommended = any_trigger and input_data.steam_available

        # Check minimum interval
        if input_data.hours_since_last_blow < self.config.min_interval_hours:
            blowing_recommended = False
            if any_trigger:
                reason += f" (blocked: min interval {self.config.min_interval_hours}h)"

        # Estimate fouling rate
        fouling_rate = self.estimate_fouling_rate(self.blow_history)

        # Calculate average blow effectiveness
        if self.blow_history:
            recent = self.blow_history[-5:]
            avg_blow_eff = sum(r.effectiveness_gain for r in recent) / len(recent)
            avg_blow_eff = max(0.01, avg_blow_eff)  # Ensure positive
        else:
            avg_blow_eff = 0.05  # Default assumption

        # Calculate optimal interval
        optimal_interval = self.calculate_optimal_interval(fouling_rate, avg_blow_eff)
        self._adaptive_interval_hours = optimal_interval

        # Calculate recommended next blow time
        if blowing_recommended:
            recommended_next = 0.0
        else:
            remaining = optimal_interval - input_data.hours_since_last_blow
            recommended_next = max(0.0, remaining)

        # Calculate steam consumption
        steam_per_cycle = self.config.num_blowers * self.config.steam_per_blower_lb

        savings_pct, daily_steam_optimal, daily_steam_fixed = self.calculate_steam_savings(
            optimal_interval,
            self.config.fixed_interval_hours,
        )

        # Calculate blowing efficiency score
        efficiency_score = self.calculate_blowing_efficiency_score(self.blow_history)

        # Calculate economics
        daily_steam_cost, eff_loss_cost, net_impact = self.calculate_economics(
            daily_steam_optimal,
            input_data.effectiveness_ratio,
        )

        # Determine blowing status
        if blowing_recommended:
            blowing_status = "scheduled"
        elif not input_data.steam_available:
            blowing_status = "bypassed"
        else:
            blowing_status = "idle"

        # Record blow cycle if we have pre/post data
        blow_effectiveness_gain = None
        if (
            input_data.pre_blow_effectiveness is not None and
            input_data.post_blow_effectiveness is not None and
            input_data.pre_blow_dp_ratio is not None and
            input_data.post_blow_dp_ratio is not None
        ):
            record = self.record_blow_cycle(
                input_data.timestamp,
                input_data.pre_blow_dp_ratio,
                input_data.post_blow_dp_ratio,
                input_data.pre_blow_effectiveness,
                input_data.post_blow_effectiveness,
            )
            blow_effectiveness_gain = record.effectiveness_gain

        # Build result
        result_data = {
            "blowing_recommended": blowing_recommended,
            "blowing_status": blowing_status,
            "hours_since_last_blow": round(input_data.hours_since_last_blow, 2),
            "recommended_next_blow_hours": round(recommended_next, 2),
            "optimal_blow_interval_hours": round(optimal_interval, 2),
            "dp_trigger_active": dp_trigger,
            "effectiveness_trigger_active": eff_trigger,
            "time_trigger_active": time_trigger,
            "trigger_reason": reason,
            "estimated_steam_per_cycle_lb": round(steam_per_cycle, 0),
            "steam_savings_vs_fixed_pct": round(savings_pct, 1),
            "daily_steam_consumption_lb": round(daily_steam_optimal, 0),
            "pre_blow_effectiveness": input_data.pre_blow_effectiveness,
            "post_blow_effectiveness": input_data.post_blow_effectiveness,
            "blow_effectiveness_gain": round(blow_effectiveness_gain, 4) if blow_effectiveness_gain else None,
            "blowing_efficiency_score": round(efficiency_score, 3),
            "estimated_daily_steam_cost": round(daily_steam_cost, 2),
            "estimated_efficiency_loss_cost": round(eff_loss_cost, 2),
            "net_economic_impact": round(net_impact, 2),
            "calculation_method": "ADAPTIVE_OPTIMIZATION",
        }

        # Calculate provenance hash
        provenance_hash = hashlib.sha256(
            json.dumps(result_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        result = SootBlowerResult(
            **result_data,
            provenance_hash=provenance_hash,
        )

        logger.info(
            f"Soot blower optimization: recommended={blowing_recommended}, "
            f"optimal_interval={optimal_interval:.1f}h, savings={savings_pct:.1f}%"
        )

        return result


def create_soot_blower_optimizer(
    config: Optional[SootBlowerConfig] = None,
) -> SootBlowerOptimizer:
    """Factory function to create SootBlowerOptimizer."""
    if config is None:
        config = SootBlowerConfig()
    return SootBlowerOptimizer(config)
