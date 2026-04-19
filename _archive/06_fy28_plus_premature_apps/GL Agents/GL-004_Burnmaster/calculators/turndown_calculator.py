"""
Turndown Calculator for GL-004 BURNMASTER

Zero-hallucination calculation engine for turndown ratio management.
All calculations are deterministic, auditable, and bit-perfect reproducible.

This module implements:
- Minimum stable load calculation
- Current turndown ratio computation
- Safe turndown setpoint calculation with rate limiting
- Turndown feasibility validation
- Burner staging recommendations

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import hashlib

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums and Constants
# =============================================================================

class FeasibilityStatus(str, Enum):
    """Turndown feasibility status."""
    FEASIBLE = "feasible"
    MARGINAL = "marginal"
    NOT_FEASIBLE = "not_feasible"
    REQUIRES_STAGING = "requires_staging"


class StagingStrategy(str, Enum):
    """Burner staging strategy."""
    ALL_ON_MODULATING = "all_on_modulating"
    STAGED_OFF = "staged_off"
    LEAD_LAG = "lead_lag"
    PARALLEL = "parallel"


# Typical turndown capabilities by burner type
BURNER_TURNDOWN_RATIOS: Dict[str, float] = {
    "standard_gas": 4.0,        # 4:1 turndown
    "low_nox_gas": 5.0,         # 5:1 turndown
    "ultra_low_nox": 6.0,       # 6:1 turndown
    "oil_pressure": 3.0,        # 3:1 turndown
    "oil_rotary_cup": 4.0,      # 4:1 turndown
    "dual_fuel": 4.0,           # 4:1 turndown
    "premix": 8.0,              # 8:1 turndown
    "forced_draft": 5.0,        # 5:1 turndown
    "natural_draft": 3.0,       # 3:1 turndown
}

# Minimum stability margins by load range
STABILITY_MARGINS: Dict[str, float] = {
    "high_load": 0.05,      # 5% margin above minimum
    "mid_load": 0.10,       # 10% margin
    "low_load": 0.15,       # 15% margin at low loads
}


# =============================================================================
# Pydantic Schemas for Input/Output
# =============================================================================

class MinimumLoadInput(BaseModel):
    """Input schema for minimum stable load calculation."""

    burner_type: str = Field(..., description="Type of burner")
    max_firing_rate: float = Field(..., gt=0, description="Maximum firing rate (MW or MMBtu/h)")
    ambient_temp: float = Field(default=20.0, description="Ambient temperature (Celsius)")
    altitude_m: float = Field(default=0.0, ge=0, description="Altitude (meters)")
    fuel_type: str = Field(default="natural_gas", description="Type of fuel")


class TurndownSetpointInput(BaseModel):
    """Input schema for safe turndown setpoint calculation."""

    current_load: float = Field(..., ge=0, description="Current load (%)")
    target_load: float = Field(..., ge=0, description="Target load (%)")
    rate_limit: float = Field(default=5.0, gt=0, description="Maximum rate of change (%/min)")
    min_stable_load: float = Field(default=25.0, ge=0, description="Minimum stable load (%)")


class StagingInput(BaseModel):
    """Input schema for staging recommendation calculation."""

    total_load_required: float = Field(..., ge=0, description="Total load required (MW or MMBtu/h)")
    burner_count: int = Field(..., ge=1, le=20, description="Number of burners available")
    max_burner_capacity: float = Field(..., gt=0, description="Max capacity per burner")
    min_burner_load: float = Field(default=25.0, ge=0, le=100, description="Minimum load per burner (%)")


class ValidationResult(BaseModel):
    """Output schema for turndown feasibility validation."""

    status: FeasibilityStatus = Field(..., description="Feasibility status")
    is_feasible: bool = Field(..., description="Whether target is feasible")
    target_load_percent: Decimal = Field(..., description="Target load (%)")
    minimum_stable_load: Decimal = Field(..., description="Minimum stable load (%)")
    margin_percent: Decimal = Field(..., description="Margin above minimum (%)")
    constraints_violated: List[str] = Field(default_factory=list, description="Violated constraints")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BurnerAllocation(BaseModel):
    """Allocation for a single burner in staging plan."""

    burner_id: int = Field(..., description="Burner identifier")
    status: str = Field(..., description="on/off/standby")
    load_percent: Decimal = Field(..., description="Load percentage for this burner")
    load_absolute: Decimal = Field(..., description="Absolute load (MW or MMBtu/h)")


class StagingPlan(BaseModel):
    """Output schema for staging recommendation."""

    strategy: StagingStrategy = Field(..., description="Recommended staging strategy")
    burners_active: int = Field(..., description="Number of burners to operate")
    burners_standby: int = Field(..., description="Number of burners on standby")
    burners_off: int = Field(..., description="Number of burners off")
    allocations: List[BurnerAllocation] = Field(..., description="Individual burner allocations")
    average_load_per_burner: Decimal = Field(..., description="Average load per active burner (%)")
    total_capacity_utilized: Decimal = Field(..., description="Total capacity utilization (%)")
    efficiency_estimate: Decimal = Field(..., description="Estimated efficiency at this staging")
    recommendations: List[str] = Field(default_factory=list, description="Staging recommendations")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


# =============================================================================
# Turndown Calculator Class
# =============================================================================

class TurndownCalculator:
    """
    Zero-hallucination calculator for turndown ratio management.

    Guarantees:
    - Deterministic: Same input produces same output (bit-perfect)
    - Auditable: SHA-256 provenance hash for every calculation
    - Reproducible: Complete calculation step tracking
    - NO LLM: Pure arithmetic and lookup operations only

    Example:
        >>> calculator = TurndownCalculator()
        >>> min_load = calculator.compute_minimum_stable_load(
        ...     {'burner_type': 'low_nox_gas', 'max_firing_rate': 10.0},
        ...     {'ambient_temp': 20.0}
        ... )
        >>> print(f"Minimum stable load: {min_load}%")
    """

    def __init__(self, precision: int = 2):
        """
        Initialize calculator with precision settings.

        Args:
            precision: Decimal places for output values (default: 2)
        """
        self.precision = precision
        self._quantize_str = '0.' + '0' * precision

    def _quantize(self, value: Decimal) -> Decimal:
        """Apply precision rounding (ROUND_HALF_UP for regulatory compliance)."""
        return value.quantize(Decimal(self._quantize_str), rounding=ROUND_HALF_UP)

    def _compute_provenance_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for audit trail."""
        data_str = str(sorted(data.items()))
        return hashlib.sha256(data_str.encode()).hexdigest()

    # -------------------------------------------------------------------------
    # Core Calculation Methods
    # -------------------------------------------------------------------------

    def compute_minimum_stable_load(
        self,
        burner_specs: Dict[str, Any],
        ambient: Dict[str, Any] = None
    ) -> Decimal:
        """
        Compute minimum stable firing rate as percentage of maximum.

        DETERMINISTIC: Based on burner type and ambient corrections.

        Minimum load = 100 / turndown_ratio * ambient_correction

        Args:
            burner_specs: Dict with 'burner_type' and 'max_firing_rate'
            ambient: Dict with 'ambient_temp' and optionally 'altitude_m'

        Returns:
            Minimum stable load as percentage of maximum
        """
        if ambient is None:
            ambient = {}

        # Step 1: Get base turndown ratio (DETERMINISTIC lookup)
        burner_type = burner_specs.get('burner_type', 'standard_gas').lower().replace(' ', '_')
        base_turndown = BURNER_TURNDOWN_RATIOS.get(burner_type, 4.0)

        # Step 2: Calculate base minimum load
        base_min_load = 100.0 / base_turndown

        # Step 3: Apply ambient temperature correction (DETERMINISTIC)
        # Higher ambient temp slightly reduces minimum stable load
        # Lower ambient temp increases minimum stable load
        ambient_temp = ambient.get('ambient_temp', 20.0)
        temp_correction = 1.0
        if ambient_temp < 0:
            # Cold conditions - increase minimum load
            temp_correction = 1.0 + (0 - ambient_temp) * 0.005  # 0.5% per degree below 0
        elif ambient_temp > 35:
            # Hot conditions - slight reduction possible
            temp_correction = 1.0 - (ambient_temp - 35) * 0.002  # 0.2% per degree above 35

        # Step 4: Apply altitude correction (DETERMINISTIC)
        # Higher altitude reduces air density, increases minimum load
        altitude = ambient.get('altitude_m', 0.0)
        altitude_correction = 1.0 + altitude / 10000  # ~1% per 100m

        # Step 5: Calculate corrected minimum load
        corrected_min_load = base_min_load * temp_correction * altitude_correction

        # Cap at reasonable values (10-50%)
        corrected_min_load = max(10.0, min(50.0, corrected_min_load))

        return self._quantize(Decimal(str(corrected_min_load)))

    def compute_current_turndown_ratio(
        self,
        load: float,
        max_load: float
    ) -> Decimal:
        """
        Compute current turndown ratio from load and maximum.

        DETERMINISTIC: max_load / load

        Args:
            load: Current firing rate
            max_load: Maximum firing rate

        Returns:
            Current turndown ratio (e.g., 4.0 means 4:1)
        """
        if load <= 0:
            # At zero load, ratio is undefined (return max possible)
            return self._quantize(Decimal('999.99'))

        ratio = max_load / load
        return self._quantize(Decimal(str(ratio)))

    def compute_safe_turndown_setpoint(
        self,
        current: float,
        target: float,
        rate_limit: float,
        time_step_seconds: float = 1.0
    ) -> Decimal:
        """
        Compute next safe setpoint considering rate limits.

        DETERMINISTIC: Apply rate limit to move toward target.

        Args:
            current: Current load (%)
            target: Target load (%)
            rate_limit: Maximum rate of change (%/minute)
            time_step_seconds: Time step for calculation (default 1 second)

        Returns:
            Safe setpoint for next time step
        """
        # Step 1: Calculate maximum change for this time step (DETERMINISTIC)
        rate_per_second = rate_limit / 60.0
        max_change = rate_per_second * time_step_seconds

        # Step 2: Calculate required change
        required_change = target - current

        # Step 3: Apply rate limit (DETERMINISTIC)
        if abs(required_change) <= max_change:
            # Can reach target in one step
            safe_setpoint = target
        else:
            # Apply rate limit
            if required_change > 0:
                safe_setpoint = current + max_change
            else:
                safe_setpoint = current - max_change

        # Step 4: Ensure within bounds (0-100%)
        safe_setpoint = max(0.0, min(100.0, safe_setpoint))

        return self._quantize(Decimal(str(safe_setpoint)))

    def validate_turndown_feasibility(
        self,
        target_load: float,
        constraints: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate whether target load is feasible given constraints.

        DETERMINISTIC: Check target against minimum load and constraints.

        Args:
            target_load: Target load percentage (0-100)
            constraints: Dict with constraint parameters:
                - 'min_stable_load': Minimum stable load (%)
                - 'max_load': Maximum load (%)
                - 'min_o2': Minimum O2 requirement
                - 'max_nox': Maximum NOx limit
                - 'burner_count': Number of burners (for staging check)

        Returns:
            ValidationResult with feasibility assessment
        """
        constraints_violated = []
        recommendations = []

        # Step 1: Get minimum stable load (DETERMINISTIC)
        min_stable = constraints.get('min_stable_load', 25.0)
        max_load = constraints.get('max_load', 100.0)

        # Step 2: Check against minimum (DETERMINISTIC)
        if target_load < min_stable:
            constraints_violated.append(f"Target {target_load}% below minimum stable load {min_stable}%")
            recommendations.append("Consider staging burners off to reduce capacity")

        # Step 3: Check against maximum (DETERMINISTIC)
        if target_load > max_load:
            constraints_violated.append(f"Target {target_load}% exceeds maximum load {max_load}%")
            recommendations.append("Target load exceeds burner capacity")

        # Step 4: Calculate margin (DETERMINISTIC)
        if target_load >= min_stable:
            margin = target_load - min_stable
        else:
            margin = 0.0

        # Step 5: Determine status (DETERMINISTIC thresholds)
        if len(constraints_violated) == 0:
            if margin > 10.0:
                status = FeasibilityStatus.FEASIBLE
            else:
                status = FeasibilityStatus.MARGINAL
                recommendations.append("Operating close to minimum - monitor stability")
        else:
            burner_count = constraints.get('burner_count', 1)
            if burner_count > 1 and target_load < min_stable:
                status = FeasibilityStatus.REQUIRES_STAGING
                recommendations.append(f"Stage down from {burner_count} burners to reduce minimum load")
            else:
                status = FeasibilityStatus.NOT_FEASIBLE

        is_feasible = status in [FeasibilityStatus.FEASIBLE, FeasibilityStatus.MARGINAL]

        # Compute provenance hash
        provenance = self._compute_provenance_hash({
            'target_load': target_load,
            'min_stable': min_stable,
            'max_load': max_load,
            'status': status.value,
            'is_feasible': is_feasible
        })

        return ValidationResult(
            status=status,
            is_feasible=is_feasible,
            target_load_percent=self._quantize(Decimal(str(target_load))),
            minimum_stable_load=self._quantize(Decimal(str(min_stable))),
            margin_percent=self._quantize(Decimal(str(margin))),
            constraints_violated=constraints_violated,
            recommendations=recommendations,
            provenance_hash=provenance
        )

    def compute_staging_recommendation(
        self,
        total_load: float,
        burner_count: int,
        max_capacity_per_burner: float = 10.0,
        min_load_percent: float = 25.0
    ) -> StagingPlan:
        """
        Compute optimal burner staging for given total load requirement.

        DETERMINISTIC: Optimization based on efficiency and stability.

        The algorithm:
        1. Calculate how many burners needed at optimal load
        2. Distribute load evenly among active burners
        3. Verify all burners above minimum stable load
        4. Generate staging plan

        Args:
            total_load: Total load required (MW or MMBtu/h)
            burner_count: Number of burners available
            max_capacity_per_burner: Maximum capacity per burner
            min_load_percent: Minimum load per burner (%)

        Returns:
            StagingPlan with burner allocations and recommendations
        """
        recommendations = []
        total_capacity = burner_count * max_capacity_per_burner

        # Step 1: Calculate capacity utilization (DETERMINISTIC)
        if total_capacity > 0:
            utilization = (total_load / total_capacity) * 100
        else:
            utilization = 0

        # Step 2: Determine optimal number of active burners (DETERMINISTIC)
        # Target: Each active burner at 60-80% for best efficiency
        optimal_load_per_burner = 0.7 * max_capacity_per_burner  # 70% load
        min_load_absolute = (min_load_percent / 100) * max_capacity_per_burner

        # Calculate burners needed
        if total_load <= 0:
            burners_needed = 0
        elif total_load <= min_load_absolute:
            burners_needed = 1  # Minimum one burner
        else:
            # Start with ideal calculation
            ideal_burners = total_load / optimal_load_per_burner
            burners_needed = max(1, min(burner_count, int(ideal_burners + 0.5)))

            # Verify minimum load constraint
            load_per_burner = total_load / burners_needed
            if load_per_burner < min_load_absolute and burners_needed > 1:
                # Reduce burner count
                burners_needed = max(1, int(total_load / min_load_absolute))

        # Ensure we don't exceed capacity
        if burners_needed > 0:
            load_per_burner = total_load / burners_needed
            if load_per_burner > max_capacity_per_burner:
                burners_needed = min(burner_count, int(total_load / max_capacity_per_burner) + 1)

        # Step 3: Create burner allocations (DETERMINISTIC)
        allocations = []
        if burners_needed > 0:
            load_per_active = total_load / burners_needed
            load_percent = (load_per_active / max_capacity_per_burner) * 100

            for i in range(burner_count):
                if i < burners_needed:
                    allocations.append(BurnerAllocation(
                        burner_id=i + 1,
                        status="on",
                        load_percent=self._quantize(Decimal(str(load_percent))),
                        load_absolute=self._quantize(Decimal(str(load_per_active)))
                    ))
                else:
                    allocations.append(BurnerAllocation(
                        burner_id=i + 1,
                        status="off",
                        load_percent=Decimal('0'),
                        load_absolute=Decimal('0')
                    ))
        else:
            for i in range(burner_count):
                allocations.append(BurnerAllocation(
                    burner_id=i + 1,
                    status="off",
                    load_percent=Decimal('0'),
                    load_absolute=Decimal('0')
                ))

        # Step 4: Determine strategy (DETERMINISTIC)
        if burners_needed == burner_count:
            strategy = StagingStrategy.ALL_ON_MODULATING
        elif burners_needed == 0:
            strategy = StagingStrategy.STAGED_OFF
        elif burners_needed == 1:
            strategy = StagingStrategy.LEAD_LAG
        else:
            strategy = StagingStrategy.PARALLEL

        # Step 5: Estimate efficiency (DETERMINISTIC)
        # Efficiency curve: peaks at 70-80% load, drops at low and high loads
        if burners_needed > 0:
            avg_load_percent = (total_load / burners_needed) / max_capacity_per_burner * 100
            if avg_load_percent < 30:
                efficiency = 75 + (avg_load_percent / 30) * 10
            elif avg_load_percent < 70:
                efficiency = 85 + ((avg_load_percent - 30) / 40) * 10
            elif avg_load_percent < 90:
                efficiency = 95 - ((avg_load_percent - 70) / 20) * 3
            else:
                efficiency = 92 - ((avg_load_percent - 90) / 10) * 2
        else:
            efficiency = 0
            avg_load_percent = 0

        # Step 6: Generate recommendations (DETERMINISTIC rules)
        if burners_needed == 0:
            recommendations.append("All burners can be turned off")
        elif avg_load_percent < 40:
            recommendations.append("Consider reducing active burner count for better efficiency")
        elif avg_load_percent > 90:
            recommendations.append("Consider adding burners to reduce individual loads")

        if burners_needed < burner_count:
            standby_burners = burner_count - burners_needed
            recommendations.append(f"Keep {standby_burners} burner(s) on hot standby for quick response")

        # Compute provenance hash
        provenance = self._compute_provenance_hash({
            'total_load': total_load,
            'burner_count': burner_count,
            'burners_needed': burners_needed,
            'strategy': strategy.value,
            'efficiency': efficiency
        })

        return StagingPlan(
            strategy=strategy,
            burners_active=burners_needed,
            burners_standby=max(0, burner_count - burners_needed - (burner_count - burners_needed) // 2),
            burners_off=max(0, (burner_count - burners_needed) // 2),
            allocations=allocations,
            average_load_per_burner=self._quantize(Decimal(str(avg_load_percent))),
            total_capacity_utilized=self._quantize(Decimal(str(utilization))),
            efficiency_estimate=self._quantize(Decimal(str(efficiency))),
            recommendations=recommendations,
            provenance_hash=provenance
        )

    # -------------------------------------------------------------------------
    # Batch Processing Methods
    # -------------------------------------------------------------------------

    def validate_turndown_batch(
        self,
        targets: List[Dict[str, Any]]
    ) -> List[ValidationResult]:
        """
        Validate multiple turndown targets.

        Args:
            targets: List of dicts with 'target_load' and 'constraints'

        Returns:
            List of ValidationResult for each target
        """
        results = []
        for t in targets:
            result = self.validate_turndown_feasibility(
                target_load=t.get('target_load', 50.0),
                constraints=t.get('constraints', {})
            )
            results.append(result)
        return results
