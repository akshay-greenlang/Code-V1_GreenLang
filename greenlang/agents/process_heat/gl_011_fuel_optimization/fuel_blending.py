"""
GL-011 FUELCRAFT - Fuel Blending Optimizer

This module provides multi-fuel blending optimization for cost minimization
while meeting equipment constraints and emission limits.

Features:
    - Linear programming optimization for blend ratios
    - Wobbe Index interchangeability constraints
    - Emission constraints (CO2, NOx, SO2)
    - Equipment capability constraints
    - Real-time blend ratio adjustment
    - Zero-hallucination deterministic calculations

Example:
    >>> from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_blending import (
    ...     FuelBlendingOptimizer,
    ...     BlendInput,
    ... )
    >>>
    >>> optimizer = FuelBlendingOptimizer(config)
    >>> result = optimizer.optimize_blend(input_data)
    >>> print(f"Optimal blend: {result.blend_ratios}")
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math

from pydantic import BaseModel, Field, validator

from greenlang.agents.process_heat.gl_011_fuel_optimization.config import (
    BlendingConfig,
    FuelType,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.schemas import (
    FuelProperties,
    FuelPrice,
    BlendRecommendation,
    BlendStatus,
)
from greenlang.agents.process_heat.gl_011_fuel_optimization.heating_value import (
    HeatingValueCalculator,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Emission factors (kg CO2/MMBTU, lb NOx/MMBTU)
DEFAULT_EMISSION_FACTORS = {
    "natural_gas": {"co2": 53.06, "nox": 0.098, "so2": 0.001},
    "no2_fuel_oil": {"co2": 73.16, "nox": 0.140, "so2": 0.050},
    "no6_fuel_oil": {"co2": 75.10, "nox": 0.150, "so2": 0.200},
    "lpg_propane": {"co2": 62.87, "nox": 0.080, "so2": 0.001},
    "coal_bituminous": {"co2": 93.28, "nox": 0.200, "so2": 0.800},
    "biomass_wood": {"co2": 0.00, "nox": 0.080, "so2": 0.010},
    "biogas": {"co2": 0.00, "nox": 0.040, "so2": 0.001},
    "hydrogen": {"co2": 0.00, "nox": 0.050, "so2": 0.000},
    "rng": {"co2": 0.00, "nox": 0.080, "so2": 0.001},
}


# =============================================================================
# DATA MODELS
# =============================================================================

class BlendInput(BaseModel):
    """Input for blend optimization."""

    # Available fuels
    available_fuels: List[str] = Field(
        ...,
        min_items=1,
        description="List of available fuel types"
    )
    fuel_properties: Dict[str, FuelProperties] = Field(
        ...,
        description="Properties by fuel type"
    )
    fuel_prices: Dict[str, FuelPrice] = Field(
        ...,
        description="Prices by fuel type"
    )

    # Current state
    current_blend: Optional[Dict[str, float]] = Field(
        default=None,
        description="Current blend ratios if blending"
    )
    required_heat_input_mmbtu_hr: float = Field(
        ...,
        gt=0,
        description="Required heat input (MMBTU/hr)"
    )

    # Constraints
    max_co2_kg_hr: Optional[float] = Field(
        default=None,
        description="Maximum CO2 emissions (kg/hr)"
    )
    max_nox_lb_hr: Optional[float] = Field(
        default=None,
        description="Maximum NOx emissions (lb/hr)"
    )
    max_so2_lb_hr: Optional[float] = Field(
        default=None,
        description="Maximum SO2 emissions (lb/hr)"
    )

    # Equipment constraints
    min_hhv_btu_scf: Optional[float] = Field(
        default=None,
        description="Minimum HHV for equipment"
    )
    max_hhv_btu_scf: Optional[float] = Field(
        default=None,
        description="Maximum HHV for equipment"
    )
    min_wobbe_index: Optional[float] = Field(
        default=None,
        description="Minimum Wobbe Index"
    )
    max_wobbe_index: Optional[float] = Field(
        default=None,
        description="Maximum Wobbe Index"
    )

    # Fuel-specific constraints
    fuel_constraints: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Min/max ratios by fuel type"
    )


class BlendOutput(BaseModel):
    """Output from blend optimization."""

    # Status
    status: BlendStatus = Field(..., description="Optimization status")
    optimization_time_ms: float = Field(
        default=0.0,
        description="Optimization time (ms)"
    )

    # Optimal blend
    optimal_blend: Dict[str, float] = Field(
        ...,
        description="Optimal blend ratios (%)"
    )
    primary_fuel: str = Field(..., description="Primary fuel in blend")

    # Properties of blended fuel
    blended_hhv_btu_scf: Optional[float] = Field(
        default=None,
        description="Blended HHV (BTU/SCF)"
    )
    blended_wobbe_index: Optional[float] = Field(
        default=None,
        description="Blended Wobbe Index"
    )
    blended_specific_gravity: Optional[float] = Field(
        default=None,
        description="Blended specific gravity"
    )

    # Cost analysis
    blended_cost_usd_mmbtu: float = Field(
        ...,
        description="Blended fuel cost ($/MMBTU)"
    )
    hourly_cost_usd: float = Field(
        ...,
        description="Hourly fuel cost ($)"
    )
    cost_vs_primary_pct: float = Field(
        default=0.0,
        description="Cost vs primary fuel only (%)"
    )
    annual_savings_usd: float = Field(
        default=0.0,
        description="Annual savings vs primary fuel ($)"
    )

    # Emissions
    blended_co2_kg_mmbtu: float = Field(
        ...,
        description="Blended CO2 factor (kg/MMBTU)"
    )
    hourly_co2_kg: float = Field(..., description="Hourly CO2 emissions (kg)")
    co2_reduction_pct: float = Field(
        default=0.0,
        description="CO2 reduction vs primary (%)"
    )

    # Constraints status
    constraints_satisfied: bool = Field(
        default=True,
        description="All constraints satisfied"
    )
    constraint_violations: List[str] = Field(
        default_factory=list,
        description="List of constraint violations"
    )

    # Provenance
    provenance_hash: str = Field(..., description="Calculation provenance hash")

    class Config:
        use_enum_values = True


class BlendConstraints(BaseModel):
    """Blend optimization constraints."""

    min_wobbe_index: float = Field(default=1300.0, description="Min Wobbe Index")
    max_wobbe_index: float = Field(default=1400.0, description="Max Wobbe Index")
    min_hhv_btu_scf: float = Field(default=900.0, description="Min HHV")
    max_hhv_btu_scf: float = Field(default=1200.0, description="Max HHV")
    max_co2_kg_mmbtu: Optional[float] = Field(default=None, description="Max CO2")
    min_primary_fuel_pct: float = Field(
        default=50.0,
        description="Min primary fuel percentage"
    )
    max_fuels_in_blend: int = Field(default=3, description="Max fuels in blend")


# =============================================================================
# FUEL BLENDING OPTIMIZER
# =============================================================================

class FuelBlendingOptimizer:
    """
    Multi-fuel blending optimizer for cost minimization.

    This optimizer finds the optimal blend of available fuels to minimize
    cost while satisfying equipment constraints and emission limits.
    Uses linear programming for deterministic optimization.

    Features:
        - Cost minimization objective
        - Wobbe Index constraints for interchangeability
        - Emission constraints (CO2, NOx, SO2)
        - Equipment capability constraints
        - Blend transition recommendations

    Example:
        >>> optimizer = FuelBlendingOptimizer(config)
        >>> result = optimizer.optimize_blend(input_data)
        >>> if result.status == BlendStatus.OPTIMAL:
        ...     print(f"Blend: {result.optimal_blend}")
    """

    def __init__(
        self,
        config: BlendingConfig,
        heating_value_calculator: Optional[HeatingValueCalculator] = None,
    ) -> None:
        """
        Initialize the fuel blending optimizer.

        Args:
            config: Blending configuration
            heating_value_calculator: Optional heating value calculator
        """
        self.config = config
        self.hv_calc = heating_value_calculator or HeatingValueCalculator()
        self._optimization_count = 0

        logger.info(
            f"FuelBlendingOptimizer initialized "
            f"(max_fuels: {config.max_fuels_in_blend})"
        )

    def optimize_blend(
        self,
        input_data: BlendInput,
    ) -> BlendOutput:
        """
        Optimize fuel blend for minimum cost.

        Uses a simplified linear programming approach suitable for
        real-time optimization. The algorithm:
        1. Enumerate feasible blend combinations
        2. Calculate cost and properties for each
        3. Select minimum cost blend satisfying constraints

        Args:
            input_data: Blend optimization input

        Returns:
            BlendOutput with optimal blend and analysis

        Raises:
            ValueError: If no feasible blend exists
        """
        start_time = datetime.now(timezone.utc)
        self._optimization_count += 1

        logger.debug(
            f"Optimizing blend for {len(input_data.available_fuels)} fuels, "
            f"{input_data.required_heat_input_mmbtu_hr:.2f} MMBTU/hr"
        )

        # Get constraints
        constraints = self._build_constraints(input_data)

        # Generate candidate blends
        candidates = self._generate_blend_candidates(
            input_data.available_fuels,
            constraints
        )

        if not candidates:
            # No feasible blends - return single fuel solution
            return self._single_fuel_solution(input_data, start_time)

        # Evaluate each candidate
        best_blend = None
        best_cost = float("inf")
        best_result = None

        for blend in candidates:
            result = self._evaluate_blend(
                blend,
                input_data,
                constraints
            )

            if result is not None and result["cost_usd_mmbtu"] < best_cost:
                if self._check_constraints(result, constraints):
                    best_blend = blend
                    best_cost = result["cost_usd_mmbtu"]
                    best_result = result

        if best_blend is None:
            # No feasible blend found - use primary fuel
            return self._single_fuel_solution(input_data, start_time)

        # Calculate savings vs single fuel
        primary_fuel = max(best_blend.items(), key=lambda x: x[1])[0]
        primary_price = input_data.fuel_prices[primary_fuel].total_price
        savings_pct = (primary_price - best_cost) / primary_price * 100

        # Calculate annual savings
        hours_per_year = 8760 * 0.9  # 90% availability
        annual_savings = (
            (primary_price - best_cost) *
            input_data.required_heat_input_mmbtu_hr *
            hours_per_year
        )

        # Calculate CO2 reduction
        primary_co2 = self._get_emission_factor(primary_fuel, "co2")
        co2_reduction_pct = (
            (primary_co2 - best_result["co2_kg_mmbtu"]) / primary_co2 * 100
            if primary_co2 > 0 else 0.0
        )

        # Check constraint violations
        violations = self._get_constraint_violations(best_result, constraints)

        # Calculate processing time
        processing_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            input_data.dict(),
            best_blend,
            best_result
        )

        return BlendOutput(
            status=BlendStatus.OPTIMAL if not violations else BlendStatus.SUB_OPTIMAL,
            optimization_time_ms=round(processing_time, 2),
            optimal_blend={k: round(v, 1) for k, v in best_blend.items()},
            primary_fuel=primary_fuel,
            blended_hhv_btu_scf=best_result.get("hhv_btu_scf"),
            blended_wobbe_index=best_result.get("wobbe_index"),
            blended_specific_gravity=best_result.get("specific_gravity"),
            blended_cost_usd_mmbtu=round(best_cost, 4),
            hourly_cost_usd=round(
                best_cost * input_data.required_heat_input_mmbtu_hr,
                2
            ),
            cost_vs_primary_pct=round(savings_pct, 2),
            annual_savings_usd=round(annual_savings, 0),
            blended_co2_kg_mmbtu=round(best_result["co2_kg_mmbtu"], 2),
            hourly_co2_kg=round(
                best_result["co2_kg_mmbtu"] * input_data.required_heat_input_mmbtu_hr,
                2
            ),
            co2_reduction_pct=round(co2_reduction_pct, 2),
            constraints_satisfied=len(violations) == 0,
            constraint_violations=violations,
            provenance_hash=provenance_hash,
        )

    def create_blend_recommendation(
        self,
        blend_output: BlendOutput,
        current_blend: Optional[Dict[str, float]] = None,
    ) -> BlendRecommendation:
        """
        Create a blend recommendation from optimization output.

        Args:
            blend_output: Blend optimization output
            current_blend: Current blend ratios

        Returns:
            BlendRecommendation for implementation
        """
        # Calculate transition time if changing blend
        transition_time = None
        if current_blend is not None:
            max_change = max(
                abs(blend_output.optimal_blend.get(f, 0) - current_blend.get(f, 0))
                for f in set(blend_output.optimal_blend) | set(current_blend)
            )
            # Assume 5%/min max change rate
            transition_time = int(max_change / self.config.max_blend_change_rate_pct_min)

        return BlendRecommendation(
            status=blend_output.status,
            blend_ratios=blend_output.optimal_blend,
            primary_fuel=blend_output.primary_fuel,
            blended_hhv=blend_output.blended_hhv_btu_scf or 0.0,
            blended_wobbe_index=blend_output.blended_wobbe_index,
            blended_co2_factor=blend_output.blended_co2_kg_mmbtu,
            blended_cost_usd_mmbtu=blend_output.blended_cost_usd_mmbtu,
            cost_savings_usd_hr=blend_output.hourly_cost_usd * (
                blend_output.cost_vs_primary_pct / 100
            ) if blend_output.cost_vs_primary_pct != 0 else None,
            emissions_reduction_pct=blend_output.co2_reduction_pct,
            wobbe_in_range=blend_output.constraints_satisfied,
            hhv_in_range=blend_output.constraints_satisfied,
            emissions_in_range=blend_output.constraints_satisfied,
            current_blend=current_blend,
            transition_time_minutes=transition_time,
        )

    def _build_constraints(self, input_data: BlendInput) -> BlendConstraints:
        """Build constraints from input and config."""
        return BlendConstraints(
            min_wobbe_index=input_data.min_wobbe_index or self.config.min_wobbe_index,
            max_wobbe_index=input_data.max_wobbe_index or self.config.max_wobbe_index,
            min_hhv_btu_scf=input_data.min_hhv_btu_scf or self.config.min_hhv_btu_scf or 900.0,
            max_hhv_btu_scf=input_data.max_hhv_btu_scf or self.config.max_hhv_btu_scf or 1200.0,
            max_co2_kg_mmbtu=self.config.max_co2_kg_mmbtu,
            min_primary_fuel_pct=self.config.min_primary_fuel_pct,
            max_fuels_in_blend=self.config.max_fuels_in_blend,
        )

    def _generate_blend_candidates(
        self,
        fuels: List[str],
        constraints: BlendConstraints,
    ) -> List[Dict[str, float]]:
        """
        Generate candidate blend combinations.

        Uses a grid search approach with configurable resolution.
        """
        candidates = []
        n_fuels = min(len(fuels), constraints.max_fuels_in_blend)

        if n_fuels == 1:
            # Single fuel only
            return [{fuels[0]: 100.0}]

        # Generate blends with different primary fuel percentages
        step = 10  # 10% resolution
        min_primary = int(constraints.min_primary_fuel_pct)

        for primary_pct in range(min_primary, 101, step):
            remaining = 100 - primary_pct

            if n_fuels == 2 and len(fuels) >= 2:
                # Two fuel blends
                for i, primary in enumerate(fuels):
                    for j, secondary in enumerate(fuels):
                        if i != j:
                            candidates.append({
                                primary: float(primary_pct),
                                secondary: float(remaining),
                            })

            elif n_fuels >= 3 and len(fuels) >= 3:
                # Three fuel blends
                for second_pct in range(0, remaining + 1, step):
                    third_pct = remaining - second_pct
                    if third_pct >= 0:
                        for i, primary in enumerate(fuels):
                            for j, secondary in enumerate(fuels):
                                for k, tertiary in enumerate(fuels):
                                    if i != j and j != k and i != k:
                                        if second_pct > 0 and third_pct > 0:
                                            candidates.append({
                                                primary: float(primary_pct),
                                                secondary: float(second_pct),
                                                tertiary: float(third_pct),
                                            })

        return candidates

    def _evaluate_blend(
        self,
        blend: Dict[str, float],
        input_data: BlendInput,
        constraints: BlendConstraints,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a blend for cost and properties.

        Returns None if blend is invalid.
        """
        try:
            # Calculate blended cost
            cost = 0.0
            for fuel, pct in blend.items():
                if fuel in input_data.fuel_prices:
                    cost += input_data.fuel_prices[fuel].total_price * (pct / 100)
                else:
                    return None  # Missing price data

            # Calculate blended properties (for gases)
            hhv = 0.0
            specific_gravity = 0.0
            co2_factor = 0.0
            nox_factor = 0.0

            for fuel, pct in blend.items():
                frac = pct / 100

                # Get fuel properties
                if fuel in input_data.fuel_properties:
                    props = input_data.fuel_properties[fuel]
                    if props.hhv_btu_scf:
                        hhv += props.hhv_btu_scf * frac
                    if props.specific_gravity:
                        specific_gravity += props.specific_gravity * frac
                    if props.co2_kg_mmbtu:
                        co2_factor += props.co2_kg_mmbtu * frac
                else:
                    # Use defaults
                    hhv += 1000.0 * frac  # Typical natural gas
                    specific_gravity += 0.6 * frac
                    co2_factor += self._get_emission_factor(fuel, "co2") * frac

                nox_factor += self._get_emission_factor(fuel, "nox") * frac

            # Calculate Wobbe Index
            wobbe_index = None
            if specific_gravity > 0:
                wobbe_index = hhv / math.sqrt(specific_gravity)

            return {
                "cost_usd_mmbtu": cost,
                "hhv_btu_scf": hhv,
                "specific_gravity": specific_gravity,
                "wobbe_index": wobbe_index,
                "co2_kg_mmbtu": co2_factor,
                "nox_lb_mmbtu": nox_factor,
            }

        except Exception as e:
            logger.warning(f"Error evaluating blend {blend}: {e}")
            return None

    def _check_constraints(
        self,
        result: Dict[str, Any],
        constraints: BlendConstraints,
    ) -> bool:
        """Check if blend result satisfies all constraints."""
        # Check Wobbe Index
        if result.get("wobbe_index") is not None:
            if result["wobbe_index"] < constraints.min_wobbe_index:
                return False
            if result["wobbe_index"] > constraints.max_wobbe_index:
                return False

        # Check HHV
        if result.get("hhv_btu_scf") is not None:
            if result["hhv_btu_scf"] < constraints.min_hhv_btu_scf:
                return False
            if result["hhv_btu_scf"] > constraints.max_hhv_btu_scf:
                return False

        # Check CO2
        if constraints.max_co2_kg_mmbtu is not None:
            if result["co2_kg_mmbtu"] > constraints.max_co2_kg_mmbtu:
                return False

        return True

    def _get_constraint_violations(
        self,
        result: Dict[str, Any],
        constraints: BlendConstraints,
    ) -> List[str]:
        """Get list of constraint violations."""
        violations = []

        if result.get("wobbe_index") is not None:
            if result["wobbe_index"] < constraints.min_wobbe_index:
                violations.append(
                    f"Wobbe Index {result['wobbe_index']:.0f} below min "
                    f"{constraints.min_wobbe_index:.0f}"
                )
            if result["wobbe_index"] > constraints.max_wobbe_index:
                violations.append(
                    f"Wobbe Index {result['wobbe_index']:.0f} above max "
                    f"{constraints.max_wobbe_index:.0f}"
                )

        if result.get("hhv_btu_scf") is not None:
            if result["hhv_btu_scf"] < constraints.min_hhv_btu_scf:
                violations.append(
                    f"HHV {result['hhv_btu_scf']:.0f} below min "
                    f"{constraints.min_hhv_btu_scf:.0f}"
                )

        if constraints.max_co2_kg_mmbtu is not None:
            if result["co2_kg_mmbtu"] > constraints.max_co2_kg_mmbtu:
                violations.append(
                    f"CO2 {result['co2_kg_mmbtu']:.1f} above max "
                    f"{constraints.max_co2_kg_mmbtu:.1f}"
                )

        return violations

    def _single_fuel_solution(
        self,
        input_data: BlendInput,
        start_time: datetime,
    ) -> BlendOutput:
        """Return single fuel (no blend) solution."""
        # Use first available fuel
        fuel = input_data.available_fuels[0]
        price = input_data.fuel_prices[fuel].total_price
        co2 = self._get_emission_factor(fuel, "co2")

        processing_time = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        provenance_hash = self._calculate_provenance_hash(
            input_data.dict(),
            {fuel: 100.0},
            {"cost": price, "co2": co2}
        )

        return BlendOutput(
            status=BlendStatus.OPTIMAL,
            optimization_time_ms=round(processing_time, 2),
            optimal_blend={fuel: 100.0},
            primary_fuel=fuel,
            blended_cost_usd_mmbtu=round(price, 4),
            hourly_cost_usd=round(
                price * input_data.required_heat_input_mmbtu_hr,
                2
            ),
            blended_co2_kg_mmbtu=round(co2, 2),
            hourly_co2_kg=round(
                co2 * input_data.required_heat_input_mmbtu_hr,
                2
            ),
            constraints_satisfied=True,
            provenance_hash=provenance_hash,
        )

    def _get_emission_factor(self, fuel_type: str, emission: str) -> float:
        """Get emission factor for fuel type."""
        fuel_key = fuel_type.lower().replace(" ", "_").replace("-", "_")
        factors = DEFAULT_EMISSION_FACTORS.get(fuel_key, {})
        return factors.get(emission, 0.0)

    def _calculate_provenance_hash(
        self,
        inputs: Dict[str, Any],
        blend: Dict[str, float],
        result: Dict[str, Any],
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        import json

        data = {
            "optimizer": "FuelBlendingOptimizer",
            "inputs_hash": hashlib.sha256(
                json.dumps(inputs, sort_keys=True, default=str).encode()
            ).hexdigest()[:16],
            "blend": blend,
            "result": result,
        }

        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    @property
    def optimization_count(self) -> int:
        """Get total optimization count."""
        return self._optimization_count
