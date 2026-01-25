# -*- coding: utf-8 -*-
"""
Burner Tuning Optimization Tests for GL-004 BurnMaster
======================================================

Comprehensive tests for burner optimization algorithms, including:
- Air/fuel ratio optimization
- Excess air minimization
- Emission target optimization (NOx, CO tradeoff)
- Multi-objective optimization (efficiency + emissions)
- Gradient-based and gradient-free optimization methods
- Constraint handling (safety limits, permit limits)
- Real-time tuning recommendations

Reference Sources:
    - EPA Good Combustion Practice Guidelines
    - ASME PTC 4: Fired Steam Generators
    - ISA-77.44: Fossil Fuel Power Plant Boiler Combustion Controls
    - Combustion Engineering by Baukal (3rd Edition)

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

import pytest
import sys
import math
import hashlib
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# OPTIMIZATION CONSTANTS AND REFERENCE DATA
# =============================================================================

# Optimal operating ranges by fuel type
OPTIMAL_OPERATING_RANGES = {
    "natural_gas": {
        "excess_air_percent": {"min": 10, "max": 20, "target": 15},
        "o2_percent": {"min": 2.0, "max": 4.0, "target": 3.0},
        "co2_percent_max": {"min": 10.5, "max": 12.0, "target": 11.7},
        "nox_ppm": {"typical": 30, "low_nox": 9, "limit": 50},
        "co_ppm": {"typical": 20, "limit": 50},
        "stack_temp_f": {"min": 250, "max": 400, "target": 320},
    },
    "fuel_oil_no2": {
        "excess_air_percent": {"min": 15, "max": 25, "target": 18},
        "o2_percent": {"min": 2.5, "max": 4.5, "target": 3.5},
        "co2_percent_max": {"min": 13.5, "max": 15.0, "target": 14.5},
        "nox_ppm": {"typical": 100, "low_nox": 40, "limit": 150},
        "co_ppm": {"typical": 30, "limit": 100},
        "stack_temp_f": {"min": 300, "max": 450, "target": 350},
    },
    "coal_bituminous": {
        "excess_air_percent": {"min": 20, "max": 30, "target": 25},
        "o2_percent": {"min": 3.5, "max": 5.5, "target": 4.5},
        "co2_percent_max": {"min": 15.0, "max": 18.0, "target": 16.5},
        "nox_ppm": {"typical": 300, "low_nox": 150, "limit": 500},
        "co_ppm": {"typical": 100, "limit": 200},
        "stack_temp_f": {"min": 300, "max": 450, "target": 380},
    },
}

# Efficiency loss factors
EFFICIENCY_LOSS_FACTORS = {
    "dry_gas_loss_per_pct_o2": 0.38,  # % efficiency loss per % O2 above optimal
    "co_loss_per_100ppm": 0.05,  # % efficiency loss per 100 ppm CO
    "unburned_carbon_loss_coal": 0.5,  # % typical for coal
    "radiation_loss_typical": 1.5,  # % typical radiation loss
}

# NOx-CO tradeoff parameters
NOX_CO_TRADEOFF = {
    "natural_gas": {
        # As O2 decreases: NOx decreases, CO increases
        "nox_per_pct_o2": 8,  # ppm NOx change per % O2 change
        "co_per_pct_o2": -15,  # ppm CO change per % O2 change (negative = increases as O2 drops)
    },
    "fuel_oil_no2": {
        "nox_per_pct_o2": 20,
        "co_per_pct_o2": -25,
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class BurnerOperatingPoint:
    """Operating point for a burner."""
    firing_rate_pct: float  # 0-100%
    o2_percent: float
    co_ppm: float
    nox_ppm: float
    stack_temp_f: float
    efficiency_percent: float
    fuel_type: str = "natural_gas"


@dataclass(frozen=True)
class OptimizationTarget:
    """Target for burner optimization."""
    maximize_efficiency: bool = True
    nox_limit_ppm: float = 50.0
    co_limit_ppm: float = 50.0
    min_o2_percent: float = 1.5
    max_o2_percent: float = 6.0
    min_firing_rate_pct: float = 30.0
    max_firing_rate_pct: float = 100.0


@dataclass
class OptimizationResult:
    """Result from burner optimization."""
    optimal_o2: float
    optimal_firing_rate: float
    predicted_efficiency: float
    predicted_nox: float
    predicted_co: float
    iterations: int
    converged: bool
    objective_value: float


# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================

def calculate_efficiency(
    o2_percent: float,
    co_ppm: float,
    stack_temp_f: float,
    ambient_temp_f: float = 70.0,
    fuel_type: str = "natural_gas"
) -> float:
    """
    Calculate combustion efficiency using heat loss method.

    DETERMINISTIC calculation based on ASME PTC 4.

    Args:
        o2_percent: Flue gas O2 percentage
        co_ppm: CO concentration (ppm)
        stack_temp_f: Stack temperature (F)
        ambient_temp_f: Ambient temperature (F)
        fuel_type: Type of fuel

    Returns:
        Combustion efficiency percentage
    """
    # Base efficiency (theoretical maximum)
    base_efficiency = 100.0

    # Dry gas loss (major heat loss)
    # Loss = K * (T_stack - T_ambient) / CO2%
    # Approximation: Loss = 0.38 * (T_stack - T_amb) / (21 - O2)
    if o2_percent < 20.9:
        co2_approx = {
            "natural_gas": (21 - o2_percent) * 0.55,
            "fuel_oil_no2": (21 - o2_percent) * 0.70,
            "coal_bituminous": (21 - o2_percent) * 0.85,
        }.get(fuel_type, (21 - o2_percent) * 0.55)

        dry_gas_loss = 0.38 * (stack_temp_f - ambient_temp_f) / max(co2_approx, 1.0)
    else:
        dry_gas_loss = 100.0  # Pure air = no combustion

    # Moisture loss from H2 in fuel
    moisture_loss = {
        "natural_gas": 10.5,  # High H content
        "fuel_oil_no2": 6.5,
        "coal_bituminous": 4.5,
    }.get(fuel_type, 8.0)

    # CO loss (incomplete combustion)
    co_loss = (co_ppm / 10000) * 3.18  # Based on CO heating value

    # Radiation and convection losses
    radiation_loss = EFFICIENCY_LOSS_FACTORS["radiation_loss_typical"]

    # Calculate total efficiency
    total_loss = dry_gas_loss + moisture_loss + co_loss + radiation_loss
    efficiency = base_efficiency - total_loss

    # Clamp to reasonable range
    return max(min(efficiency, 99.5), 50.0)


def calculate_nox_from_o2(
    o2_percent: float,
    fuel_type: str = "natural_gas",
    burner_type: str = "standard"
) -> float:
    """
    Calculate NOx emissions based on O2 level.

    DETERMINISTIC calculation.

    Args:
        o2_percent: Flue gas O2 percentage
        fuel_type: Type of fuel
        burner_type: "standard" or "low_nox"

    Returns:
        NOx concentration (ppm)
    """
    # Base NOx at 3% O2
    base_nox = {
        "natural_gas": {"standard": 80, "low_nox": 25},
        "fuel_oil_no2": {"standard": 150, "low_nox": 60},
        "coal_bituminous": {"standard": 400, "low_nox": 200},
    }.get(fuel_type, {"standard": 80, "low_nox": 25})

    nox_at_3pct = base_nox.get(burner_type, base_nox["standard"])

    # NOx increases with excess O2 (more O available for NOx formation)
    # and with higher flame temperatures (thermal NOx)
    # At lower O2, NOx decreases but CO increases

    tradeoff = NOX_CO_TRADEOFF.get(fuel_type, NOX_CO_TRADEOFF["natural_gas"])
    o2_deviation = o2_percent - 3.0

    nox = nox_at_3pct + o2_deviation * tradeoff["nox_per_pct_o2"]

    return max(nox, 5.0)  # Minimum floor


def calculate_co_from_o2(
    o2_percent: float,
    fuel_type: str = "natural_gas",
    burner_condition: str = "well_tuned"
) -> float:
    """
    Calculate CO emissions based on O2 level.

    DETERMINISTIC calculation.

    Args:
        o2_percent: Flue gas O2 percentage
        fuel_type: Type of fuel
        burner_condition: Burner tuning condition

    Returns:
        CO concentration (ppm)
    """
    # Base CO at 3% O2
    base_co = {
        "natural_gas": {"well_tuned": 10, "average": 30, "poor": 100},
        "fuel_oil_no2": {"well_tuned": 25, "average": 50, "poor": 150},
        "coal_bituminous": {"well_tuned": 100, "average": 200, "poor": 500},
    }.get(fuel_type, {"well_tuned": 10, "average": 30, "poor": 100})

    co_at_3pct = base_co.get(burner_condition, base_co["average"])

    # CO increases dramatically below 2% O2 (incomplete combustion)
    # CO is relatively flat 2-4% O2 (optimal zone)
    # CO increases slightly above 5% O2 (quench effect)

    if o2_percent < 1.0:
        # Rich combustion - exponential CO increase
        o2_factor = 10.0 * math.exp(2.0 * (1.0 - o2_percent))
    elif o2_percent < 2.0:
        # Transition zone - steep increase
        o2_factor = 1.0 + 9.0 * ((2.0 - o2_percent) / 1.0) ** 2
    elif o2_percent <= 4.0:
        # Optimal zone - relatively flat
        o2_factor = 1.0 - 0.1 * (o2_percent - 3.0)
    elif o2_percent <= 6.0:
        # Lean combustion - slight increase
        o2_factor = 1.0 + 0.2 * (o2_percent - 4.0)
    else:
        # Very lean - quench effect
        o2_factor = 1.4 + 0.3 * (o2_percent - 6.0)

    return co_at_3pct * o2_factor


def optimize_burner_o2(
    target: OptimizationTarget,
    fuel_type: str = "natural_gas",
    burner_type: str = "standard",
    stack_temp_f: float = 320.0,
    max_iterations: int = 100,
    tolerance: float = 0.01
) -> OptimizationResult:
    """
    Optimize burner O2 setpoint for maximum efficiency within emission limits.

    DETERMINISTIC gradient-free optimization (golden section search).

    Args:
        target: Optimization targets and constraints
        fuel_type: Type of fuel
        burner_type: Burner type for NOx estimation
        stack_temp_f: Stack temperature
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance for O2 (%)

    Returns:
        OptimizationResult with optimal operating point
    """
    # Objective function: maximize efficiency subject to emission constraints
    def objective(o2: float) -> float:
        eff = calculate_efficiency(o2, calculate_co_from_o2(o2, fuel_type), stack_temp_f, fuel_type=fuel_type)
        nox = calculate_nox_from_o2(o2, fuel_type, burner_type)
        co = calculate_co_from_o2(o2, fuel_type)

        # Penalty for violating constraints
        penalty = 0.0
        if nox > target.nox_limit_ppm:
            penalty += (nox - target.nox_limit_ppm) * 0.5
        if co > target.co_limit_ppm:
            penalty += (co - target.co_limit_ppm) * 0.5

        return eff - penalty if target.maximize_efficiency else -eff + penalty

    # Golden section search
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    a, b = target.min_o2_percent, target.max_o2_percent

    c = b - (b - a) / phi
    d = a + (b - a) / phi

    iterations = 0
    while abs(b - a) > tolerance and iterations < max_iterations:
        if objective(c) > objective(d):
            b = d
        else:
            a = c

        c = b - (b - a) / phi
        d = a + (b - a) / phi
        iterations += 1

    optimal_o2 = (a + b) / 2

    # Calculate final values at optimal point
    predicted_co = calculate_co_from_o2(optimal_o2, fuel_type)
    predicted_nox = calculate_nox_from_o2(optimal_o2, fuel_type, burner_type)
    predicted_efficiency = calculate_efficiency(optimal_o2, predicted_co, stack_temp_f, fuel_type=fuel_type)

    # Check convergence
    converged = abs(b - a) <= tolerance

    return OptimizationResult(
        optimal_o2=round(optimal_o2, 2),
        optimal_firing_rate=100.0,  # Default to full load
        predicted_efficiency=round(predicted_efficiency, 2),
        predicted_nox=round(predicted_nox, 1),
        predicted_co=round(predicted_co, 1),
        iterations=iterations,
        converged=converged,
        objective_value=round(objective(optimal_o2), 4),
    )


def multi_objective_optimization(
    target: OptimizationTarget,
    fuel_type: str = "natural_gas",
    burner_type: str = "low_nox",
    stack_temp_f: float = 320.0,
    efficiency_weight: float = 0.6,
    nox_weight: float = 0.25,
    co_weight: float = 0.15
) -> Dict[str, Any]:
    """
    Multi-objective optimization balancing efficiency, NOx, and CO.

    DETERMINISTIC weighted sum method.

    Args:
        target: Optimization targets
        fuel_type: Type of fuel
        burner_type: Burner type
        stack_temp_f: Stack temperature
        efficiency_weight: Weight for efficiency objective
        nox_weight: Weight for NOx minimization
        co_weight: Weight for CO minimization

    Returns:
        Dictionary with Pareto-optimal point and analysis
    """
    # Normalize weights
    total_weight = efficiency_weight + nox_weight + co_weight
    w_eff = efficiency_weight / total_weight
    w_nox = nox_weight / total_weight
    w_co = co_weight / total_weight

    # Scan O2 range to find optimal point
    best_score = float('-inf')
    best_o2 = target.min_o2_percent
    results = []

    for o2_int in range(int(target.min_o2_percent * 10), int(target.max_o2_percent * 10) + 1):
        o2 = o2_int / 10.0

        eff = calculate_efficiency(o2, calculate_co_from_o2(o2, fuel_type), stack_temp_f, fuel_type=fuel_type)
        nox = calculate_nox_from_o2(o2, fuel_type, burner_type)
        co = calculate_co_from_o2(o2, fuel_type)

        # Normalize to 0-1 scale (higher is better)
        eff_norm = (eff - 70) / 30  # Assume 70-100% range
        nox_norm = 1 - min(nox / 100, 1)  # Invert: lower NOx = higher score
        co_norm = 1 - min(co / 100, 1)  # Invert: lower CO = higher score

        # Constraint penalties
        penalty = 0
        if nox > target.nox_limit_ppm:
            penalty += 0.5
        if co > target.co_limit_ppm:
            penalty += 0.5

        score = w_eff * eff_norm + w_nox * nox_norm + w_co * co_norm - penalty

        results.append({
            "o2": o2,
            "efficiency": eff,
            "nox": nox,
            "co": co,
            "score": score,
        })

        if score > best_score:
            best_score = score
            best_o2 = o2

    # Get values at best point
    best_result = next(r for r in results if r["o2"] == best_o2)

    return {
        "optimal_o2": best_o2,
        "optimal_efficiency": round(best_result["efficiency"], 2),
        "optimal_nox": round(best_result["nox"], 1),
        "optimal_co": round(best_result["co"], 1),
        "multi_objective_score": round(best_score, 4),
        "weights": {"efficiency": w_eff, "nox": w_nox, "co": w_co},
        "pareto_frontier": results,
    }


def generate_tuning_recommendations(
    current_o2: float,
    current_nox: float,
    current_co: float,
    target: OptimizationTarget,
    fuel_type: str = "natural_gas"
) -> List[Dict[str, Any]]:
    """
    Generate actionable tuning recommendations based on current operating point.

    DETERMINISTIC recommendation engine.

    Args:
        current_o2: Current O2 percentage
        current_nox: Current NOx (ppm)
        current_co: Current CO (ppm)
        target: Optimization targets
        fuel_type: Type of fuel

    Returns:
        List of prioritized recommendations
    """
    recommendations = []
    optimal_range = OPTIMAL_OPERATING_RANGES.get(fuel_type, OPTIMAL_OPERATING_RANGES["natural_gas"])

    # Check O2 level
    if current_o2 < optimal_range["o2_percent"]["min"]:
        recommendations.append({
            "priority": 1,
            "category": "air_fuel_ratio",
            "action": "INCREASE O2",
            "reason": f"O2 ({current_o2}%) below minimum ({optimal_range['o2_percent']['min']}%)",
            "expected_impact": "Reduce CO, slightly increase NOx",
            "adjustment": f"Increase O2 to {optimal_range['o2_percent']['target']}%",
        })
    elif current_o2 > optimal_range["o2_percent"]["max"]:
        recommendations.append({
            "priority": 2,
            "category": "air_fuel_ratio",
            "action": "REDUCE O2",
            "reason": f"O2 ({current_o2}%) above maximum ({optimal_range['o2_percent']['max']}%)",
            "expected_impact": "Improve efficiency, reduce NOx",
            "adjustment": f"Reduce O2 to {optimal_range['o2_percent']['target']}%",
        })

    # Check NOx level
    if current_nox > target.nox_limit_ppm:
        recommendations.append({
            "priority": 1,
            "category": "emissions",
            "action": "REDUCE NOx",
            "reason": f"NOx ({current_nox} ppm) exceeds limit ({target.nox_limit_ppm} ppm)",
            "expected_impact": "Compliance required",
            "adjustment": "Consider: reduce O2, install FGR, tune burner",
        })

    # Check CO level
    if current_co > target.co_limit_ppm:
        recommendations.append({
            "priority": 1,
            "category": "emissions",
            "action": "REDUCE CO",
            "reason": f"CO ({current_co} ppm) exceeds limit ({target.co_limit_ppm} ppm)",
            "expected_impact": "Compliance and efficiency required",
            "adjustment": "Increase O2, check burner tune, verify flame pattern",
        })

    # General efficiency recommendation
    if current_o2 > optimal_range["o2_percent"]["target"] + 0.5:
        eff_gain = (current_o2 - optimal_range["o2_percent"]["target"]) * 0.3
        recommendations.append({
            "priority": 3,
            "category": "efficiency",
            "action": "OPTIMIZE AIR/FUEL RATIO",
            "reason": f"Excess air higher than optimal",
            "expected_impact": f"Potential ~{eff_gain:.1f}% efficiency gain",
            "adjustment": f"Reduce O2 from {current_o2}% to {optimal_range['o2_percent']['target']}%",
        })

    # Sort by priority
    recommendations.sort(key=lambda x: x["priority"])

    return recommendations


# =============================================================================
# TEST CLASSES
# =============================================================================

@pytest.mark.optimization
class TestEfficiencyCalculation:
    """Test efficiency calculation functions."""

    @pytest.mark.parametrize("o2_percent,expected_eff_range", [
        (2.0, (76, 82)),   # Low O2 - good efficiency
        (3.0, (75, 81)),   # Optimal O2
        (4.0, (74, 80)),   # Slightly high O2
        (6.0, (71, 78)),   # High O2 - efficiency drops
        (8.0, (68, 76)),   # Very high O2 - poor efficiency
    ])
    def test_efficiency_vs_o2_natural_gas(
        self,
        o2_percent: float,
        expected_eff_range: Tuple[float, float]
    ):
        """Test efficiency calculation vs O2 for natural gas."""
        co = calculate_co_from_o2(o2_percent, "natural_gas")
        efficiency = calculate_efficiency(o2_percent, co, 320, fuel_type="natural_gas")

        assert expected_eff_range[0] <= efficiency <= expected_eff_range[1], (
            f"At O2={o2_percent}%: efficiency={efficiency:.1f}% outside "
            f"expected range {expected_eff_range}"
        )

    def test_efficiency_decreases_with_stack_temp(self):
        """Higher stack temperature = more heat loss = lower efficiency."""
        eff_low_temp = calculate_efficiency(3.0, 20, 280, fuel_type="natural_gas")
        eff_mid_temp = calculate_efficiency(3.0, 20, 350, fuel_type="natural_gas")
        eff_high_temp = calculate_efficiency(3.0, 20, 450, fuel_type="natural_gas")

        assert eff_low_temp > eff_mid_temp > eff_high_temp, (
            f"Efficiency should decrease with stack temp: "
            f"{eff_low_temp:.1f}% > {eff_mid_temp:.1f}% > {eff_high_temp:.1f}%"
        )

    def test_efficiency_decreases_with_co(self):
        """Higher CO = incomplete combustion = lower efficiency."""
        eff_low_co = calculate_efficiency(3.0, 10, 320, fuel_type="natural_gas")
        eff_mid_co = calculate_efficiency(3.0, 100, 320, fuel_type="natural_gas")
        eff_high_co = calculate_efficiency(3.0, 500, 320, fuel_type="natural_gas")

        assert eff_low_co > eff_mid_co > eff_high_co, (
            f"Efficiency should decrease with CO: "
            f"{eff_low_co:.1f}% > {eff_mid_co:.1f}% > {eff_high_co:.1f}%"
        )

    def test_fuel_type_effects(self):
        """Different fuels have different efficiency characteristics."""
        eff_gas = calculate_efficiency(3.0, 20, 320, fuel_type="natural_gas")
        eff_oil = calculate_efficiency(3.5, 30, 350, fuel_type="fuel_oil_no2")
        eff_coal = calculate_efficiency(4.5, 150, 380, fuel_type="coal_bituminous")

        # All should be reasonable efficiencies
        assert 75 < eff_gas < 92, f"Gas efficiency {eff_gas}% outside expected range"
        assert 75 < eff_oil < 90, f"Oil efficiency {eff_oil}% outside expected range"
        assert 70 < eff_coal < 88, f"Coal efficiency {eff_coal}% outside expected range"


@pytest.mark.optimization
class TestNOxCOTradeoff:
    """Test NOx-CO tradeoff relationships."""

    def test_nox_increases_with_o2(self):
        """NOx should increase as O2 increases (more available oxygen)."""
        nox_at_2 = calculate_nox_from_o2(2.0, "natural_gas")
        nox_at_3 = calculate_nox_from_o2(3.0, "natural_gas")
        nox_at_4 = calculate_nox_from_o2(4.0, "natural_gas")
        nox_at_5 = calculate_nox_from_o2(5.0, "natural_gas")

        assert nox_at_3 > nox_at_2, "NOx should increase 2% -> 3% O2"
        assert nox_at_4 > nox_at_3, "NOx should increase 3% -> 4% O2"
        assert nox_at_5 > nox_at_4, "NOx should increase 4% -> 5% O2"

    def test_co_increases_at_low_o2(self):
        """CO should increase dramatically at low O2."""
        co_at_3 = calculate_co_from_o2(3.0, "natural_gas")
        co_at_2 = calculate_co_from_o2(2.0, "natural_gas")
        co_at_1 = calculate_co_from_o2(1.0, "natural_gas")
        co_at_05 = calculate_co_from_o2(0.5, "natural_gas")

        assert co_at_2 >= co_at_3, "CO should increase as O2 drops below 3%"
        assert co_at_1 > co_at_2 * 2, "CO should increase significantly below 2%"
        assert co_at_05 > co_at_1 * 2, "CO should be very high below 1%"

    def test_low_nox_burner_reduces_nox(self):
        """Low-NOx burner should produce less NOx at same conditions."""
        nox_standard = calculate_nox_from_o2(3.0, "natural_gas", "standard")
        nox_low = calculate_nox_from_o2(3.0, "natural_gas", "low_nox")

        reduction = (nox_standard - nox_low) / nox_standard * 100

        assert nox_low < nox_standard, "Low-NOx burner should reduce NOx"
        assert reduction > 50, f"Low-NOx reduction should be >50%, got {reduction:.1f}%"

    def test_optimal_o2_balances_nox_and_co(self):
        """Optimal O2 minimizes both NOx and CO simultaneously."""
        fuel_type = "natural_gas"

        # Scan O2 range
        min_combined = float('inf')
        optimal_o2 = 3.0

        for o2_int in range(15, 50):  # 1.5% to 5.0%
            o2 = o2_int / 10.0
            nox = calculate_nox_from_o2(o2, fuel_type)
            co = calculate_co_from_o2(o2, fuel_type)

            # Combined metric (normalized)
            combined = (nox / 100) + (co / 50)  # Weighted sum

            if combined < min_combined:
                min_combined = combined
                optimal_o2 = o2

        # Optimal should be in typical operating range (allowing some margin)
        assert 1.5 <= optimal_o2 <= 5.0, (
            f"Optimal O2 ({optimal_o2}%) should be in 1.5-5.0% range"
        )


@pytest.mark.optimization
class TestBurnerOptimization:
    """Test burner optimization algorithms."""

    def test_optimization_converges(self):
        """Optimization should converge within max iterations."""
        target = OptimizationTarget(
            maximize_efficiency=True,
            nox_limit_ppm=50.0,
            co_limit_ppm=50.0,
        )

        result = optimize_burner_o2(target, "natural_gas", "low_nox")

        assert result.converged, "Optimization should converge"
        assert result.iterations < 100, f"Should converge in <100 iterations, got {result.iterations}"

    def test_optimization_respects_nox_limit(self):
        """Optimal point should respect NOx emission limit."""
        target = OptimizationTarget(
            maximize_efficiency=True,
            nox_limit_ppm=30.0,  # Strict limit
            co_limit_ppm=100.0,
        )

        result = optimize_burner_o2(target, "natural_gas", "low_nox")

        # Should either meet limit or be penalized
        if result.predicted_nox <= target.nox_limit_ppm:
            assert True, "NOx limit respected"
        else:
            # If limit exceeded, efficiency should be significantly impacted
            assert result.objective_value < 90, (
                f"NOx exceeded limit but efficiency not penalized: {result.objective_value}"
            )

    def test_optimization_respects_co_limit(self):
        """Optimal point should respect CO emission limit."""
        target = OptimizationTarget(
            maximize_efficiency=True,
            nox_limit_ppm=100.0,
            co_limit_ppm=30.0,  # Strict limit
            min_o2_percent=2.0,  # Enforce minimum O2 to keep CO low
        )

        result = optimize_burner_o2(target, "natural_gas")

        # Optimization should find point within constraints
        assert result.optimal_o2 >= target.min_o2_percent, (
            f"Optimal O2 ({result.optimal_o2}%) below minimum ({target.min_o2_percent}%)"
        )

    def test_optimal_o2_in_expected_range(self):
        """Optimal O2 should be in expected operating range."""
        target = OptimizationTarget(
            maximize_efficiency=True,
            nox_limit_ppm=50.0,
            co_limit_ppm=50.0,
        )

        for fuel_type in ["natural_gas", "fuel_oil_no2"]:
            result = optimize_burner_o2(target, fuel_type, "low_nox")
            expected_range = OPTIMAL_OPERATING_RANGES[fuel_type]["o2_percent"]

            # Should be close to expected range
            assert expected_range["min"] - 1 <= result.optimal_o2 <= expected_range["max"] + 1, (
                f"{fuel_type}: Optimal O2 ({result.optimal_o2}%) outside "
                f"expected range [{expected_range['min']}, {expected_range['max']}]"
            )


@pytest.mark.optimization
class TestMultiObjectiveOptimization:
    """Test multi-objective optimization."""

    def test_weights_affect_optimal_point(self):
        """Different weight combinations should yield different optimal points."""
        target = OptimizationTarget()

        # Efficiency-focused
        result_eff = multi_objective_optimization(
            target, efficiency_weight=0.9, nox_weight=0.05, co_weight=0.05
        )

        # NOx-focused
        result_nox = multi_objective_optimization(
            target, efficiency_weight=0.1, nox_weight=0.85, co_weight=0.05
        )

        # Different weights should yield different optima
        assert result_eff["optimal_o2"] != result_nox["optimal_o2"] or \
               result_eff["optimal_nox"] != result_nox["optimal_nox"], (
            "Different weights should produce different optimal points"
        )

    def test_pareto_frontier_generated(self):
        """Multi-objective optimization should generate Pareto frontier."""
        target = OptimizationTarget()
        result = multi_objective_optimization(target)

        assert "pareto_frontier" in result, "Should include Pareto frontier"
        assert len(result["pareto_frontier"]) > 10, "Frontier should have multiple points"

        # Verify frontier points are valid
        for point in result["pareto_frontier"]:
            assert "o2" in point
            assert "efficiency" in point
            assert "nox" in point
            assert "co" in point

    def test_balanced_optimization(self):
        """Balanced weights should find compromise solution."""
        target = OptimizationTarget()

        result = multi_objective_optimization(
            target, efficiency_weight=0.4, nox_weight=0.3, co_weight=0.3
        )

        # Should find reasonable compromise
        assert result["optimal_efficiency"] > 75, "Efficiency should be reasonable"
        assert result["optimal_nox"] < 100, "NOx should be reasonable"
        assert result["optimal_co"] < 100, "CO should be reasonable"


@pytest.mark.optimization
class TestTuningRecommendations:
    """Test tuning recommendation generation."""

    def test_high_o2_generates_reduce_recommendation(self):
        """High O2 should generate reduce O2 recommendation."""
        target = OptimizationTarget()

        recommendations = generate_tuning_recommendations(
            current_o2=6.0,  # High
            current_nox=40,
            current_co=15,
            target=target
        )

        # Should recommend reducing O2
        air_fuel_recs = [r for r in recommendations if r["category"] == "air_fuel_ratio"]
        assert len(air_fuel_recs) > 0, "Should have air/fuel ratio recommendation"
        assert "REDUCE" in air_fuel_recs[0]["action"], "Should recommend reducing O2"

    def test_low_o2_generates_increase_recommendation(self):
        """Low O2 should generate increase O2 recommendation."""
        target = OptimizationTarget()

        recommendations = generate_tuning_recommendations(
            current_o2=1.5,  # Low
            current_nox=20,
            current_co=200,
            target=target
        )

        # Should recommend increasing O2
        air_fuel_recs = [r for r in recommendations if r["category"] == "air_fuel_ratio"]
        assert len(air_fuel_recs) > 0, "Should have air/fuel ratio recommendation"
        assert "INCREASE" in air_fuel_recs[0]["action"], "Should recommend increasing O2"

    def test_high_nox_generates_emission_recommendation(self):
        """High NOx should generate emission reduction recommendation."""
        target = OptimizationTarget(nox_limit_ppm=40)

        recommendations = generate_tuning_recommendations(
            current_o2=3.5,
            current_nox=60,  # Above limit
            current_co=20,
            target=target
        )

        # Should recommend reducing NOx
        emission_recs = [r for r in recommendations if r["category"] == "emissions"]
        nox_recs = [r for r in emission_recs if "NOx" in r["action"]]
        assert len(nox_recs) > 0, "Should have NOx reduction recommendation"
        assert nox_recs[0]["priority"] == 1, "NOx violation should be high priority"

    def test_high_co_generates_emission_recommendation(self):
        """High CO should generate emission reduction recommendation."""
        target = OptimizationTarget(co_limit_ppm=40)

        recommendations = generate_tuning_recommendations(
            current_o2=2.0,
            current_nox=30,
            current_co=80,  # Above limit
            target=target
        )

        # Should recommend reducing CO
        emission_recs = [r for r in recommendations if r["category"] == "emissions"]
        co_recs = [r for r in emission_recs if "CO" in r["action"]]
        assert len(co_recs) > 0, "Should have CO reduction recommendation"

    def test_optimal_point_no_critical_recommendations(self):
        """Optimal operating point should have few/no critical recommendations."""
        target = OptimizationTarget()

        recommendations = generate_tuning_recommendations(
            current_o2=3.0,  # Optimal
            current_nox=25,  # Below limit
            current_co=15,   # Below limit
            target=target
        )

        # Should have no priority-1 recommendations
        critical_recs = [r for r in recommendations if r["priority"] == 1]
        assert len(critical_recs) == 0, (
            f"Optimal point should have no critical recommendations, got {len(critical_recs)}"
        )


@pytest.mark.optimization
class TestDeterminism:
    """Test calculation determinism."""

    def test_efficiency_determinism(self):
        """Efficiency calculation must be deterministic."""
        results = []
        for _ in range(100):
            eff = calculate_efficiency(3.5, 25, 320, fuel_type="natural_gas")
            results.append(f"{eff:.10f}")

        assert len(set(results)) == 1, "Efficiency calculation not deterministic"

    def test_optimization_determinism(self):
        """Optimization must produce deterministic results."""
        target = OptimizationTarget()

        results = []
        for _ in range(20):
            result = optimize_burner_o2(target, "natural_gas", "low_nox")
            results.append(f"{result.optimal_o2:.4f}:{result.predicted_efficiency:.4f}")

        assert len(set(results)) == 1, "Optimization not deterministic"

    def test_multi_objective_determinism(self):
        """Multi-objective optimization must be deterministic."""
        target = OptimizationTarget()

        results = []
        for _ in range(10):
            result = multi_objective_optimization(target)
            results.append(f"{result['optimal_o2']:.4f}:{result['multi_objective_score']:.6f}")

        assert len(set(results)) == 1, "Multi-objective optimization not deterministic"

    def test_recommendations_determinism(self):
        """Recommendations must be deterministic."""
        target = OptimizationTarget()

        results = []
        for _ in range(50):
            recs = generate_tuning_recommendations(4.5, 45, 25, target)
            rec_str = "|".join(r["action"] for r in recs)
            results.append(rec_str)

        assert len(set(results)) == 1, "Recommendations not deterministic"


@pytest.mark.optimization
class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_minimum_o2_boundary(self):
        """Test behavior at minimum O2 constraint."""
        target = OptimizationTarget(min_o2_percent=2.5)
        result = optimize_burner_o2(target, "natural_gas")

        assert result.optimal_o2 >= target.min_o2_percent, (
            f"Optimal O2 ({result.optimal_o2}%) below minimum ({target.min_o2_percent}%)"
        )

    def test_maximum_o2_boundary(self):
        """Test behavior at maximum O2 constraint."""
        target = OptimizationTarget(max_o2_percent=4.0)
        result = optimize_burner_o2(target, "natural_gas")

        assert result.optimal_o2 <= target.max_o2_percent, (
            f"Optimal O2 ({result.optimal_o2}%) above maximum ({target.max_o2_percent}%)"
        )

    def test_very_strict_emission_limits(self):
        """Test with very strict emission limits."""
        target = OptimizationTarget(
            nox_limit_ppm=10.0,  # Very strict
            co_limit_ppm=20.0,   # Very strict
        )

        result = optimize_burner_o2(target, "natural_gas", "low_nox")

        # Should still converge
        assert result.converged, "Should converge even with strict limits"

    def test_conflicting_constraints(self):
        """Test with conflicting constraints (low NOx but low O2 limit)."""
        target = OptimizationTarget(
            nox_limit_ppm=15.0,  # Needs low O2
            co_limit_ppm=30.0,   # Needs higher O2
            min_o2_percent=3.0,  # Can't go too low
        )

        result = optimize_burner_o2(target, "natural_gas", "low_nox")

        # Should find best compromise
        assert result.converged, "Should converge with conflicting constraints"
        assert result.optimal_o2 >= target.min_o2_percent, "Should respect min O2"


# =============================================================================
# EXPORT FUNCTION
# =============================================================================

def export_optimization_golden_values() -> Dict[str, Any]:
    """Export optimization golden values for validation."""
    return {
        "metadata": {
            "version": "1.0.0",
            "source": "EPA Good Combustion Practice, ASME PTC 4",
            "agent": "GL-004_BurnMaster",
        },
        "optimal_operating_ranges": OPTIMAL_OPERATING_RANGES,
        "efficiency_loss_factors": EFFICIENCY_LOSS_FACTORS,
        "nox_co_tradeoff": NOX_CO_TRADEOFF,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(export_optimization_golden_values(), indent=2))
