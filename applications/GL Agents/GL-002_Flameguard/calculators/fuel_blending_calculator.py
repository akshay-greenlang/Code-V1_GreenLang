"""
GL-002 FLAMEGUARD - Fuel Blending Calculator

Optimizes fuel blending for cost, efficiency, and emissions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class FuelBlendInput:
    """Single fuel for blending."""
    fuel_id: str
    fuel_type: str
    available_fraction: float  # 0-1
    cost_per_mmbtu: float
    hhv_btu_lb: float
    carbon_percent: float
    sulfur_percent: float
    stoich_air_ratio: float


@dataclass
class BlendResult:
    """Fuel blend optimization result."""
    blend_fractions: Dict[str, float]
    blended_hhv: float
    blended_cost: float
    blended_carbon: float
    blended_sulfur: float
    blended_stoich_ratio: float
    cost_savings_percent: float
    emissions_impact_percent: float
    feasible: bool
    constraints_met: List[str]


class FuelBlendingCalculator:
    """Fuel blending optimizer."""

    def __init__(self, max_sulfur_percent: float = 0.5) -> None:
        self.max_sulfur = max_sulfur_percent

    def optimize_blend(
        self,
        fuels: List[FuelBlendInput],
        objective: str = "cost",
        constraints: Optional[Dict] = None,
    ) -> BlendResult:
        """
        Optimize fuel blend.

        Args:
            fuels: Available fuels
            objective: cost, emissions, or balanced
            constraints: Optional constraints

        Returns:
            Optimal blend result
        """
        if not fuels:
            raise ValueError("No fuels provided")

        # Simple optimization: use lowest cost fuel that meets constraints
        # In production, use scipy.optimize or MILP solver

        # Sort by cost
        sorted_fuels = sorted(fuels, key=lambda f: f.cost_per_mmbtu)

        # Start with cheapest fuel
        blend = {f.fuel_id: 0.0 for f in fuels}
        remaining = 1.0

        for fuel in sorted_fuels:
            if remaining <= 0:
                break

            # Check sulfur constraint
            use_fraction = min(remaining, fuel.available_fraction)

            # Calculate resulting sulfur
            current_sulfur = sum(
                blend[f.fuel_id] * f.sulfur_percent
                for f in fuels
            )
            potential_sulfur = current_sulfur + use_fraction * fuel.sulfur_percent

            if potential_sulfur <= self.max_sulfur:
                blend[fuel.fuel_id] = use_fraction
                remaining -= use_fraction

        # Calculate blended properties
        total = sum(blend.values())
        if total == 0:
            total = 1.0
            blend[fuels[0].fuel_id] = 1.0

        blended_hhv = sum(
            blend[f.fuel_id] * f.hhv_btu_lb
            for f in fuels
        ) / total

        blended_cost = sum(
            blend[f.fuel_id] * f.cost_per_mmbtu
            for f in fuels
        ) / total

        blended_carbon = sum(
            blend[f.fuel_id] * f.carbon_percent
            for f in fuels
        ) / total

        blended_sulfur = sum(
            blend[f.fuel_id] * f.sulfur_percent
            for f in fuels
        ) / total

        blended_stoich = sum(
            blend[f.fuel_id] * f.stoich_air_ratio
            for f in fuels
        ) / total

        # Calculate savings vs primary fuel
        primary_cost = fuels[0].cost_per_mmbtu
        savings = (primary_cost - blended_cost) / primary_cost * 100 if primary_cost > 0 else 0

        return BlendResult(
            blend_fractions=blend,
            blended_hhv=round(blended_hhv, 0),
            blended_cost=round(blended_cost, 2),
            blended_carbon=round(blended_carbon, 1),
            blended_sulfur=round(blended_sulfur, 3),
            blended_stoich_ratio=round(blended_stoich, 1),
            cost_savings_percent=round(savings, 1),
            emissions_impact_percent=0.0,
            feasible=remaining <= 0.01,
            constraints_met=["sulfur_limit"] if blended_sulfur <= self.max_sulfur else [],
        )
