# -*- coding: utf-8 -*-
"""
Marginal Abatement Cost Curve Generator
GL-VCCI Scope 3 Platform

Generate MACC (Marginal Abatement Cost Curve) for emission reduction initiatives.
Visualizes cost-effectiveness of reduction opportunities.

Version: 1.0.0
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""

import logging
from typing import List, Dict, Any

from ..models import Initiative, AbatementCurve, AbatementCurvePoint
from ..config import ROIConfig
from ..exceptions import AbatementCurveError
from .roi_calculator import ROICalculator

logger = logging.getLogger(__name__)


class AbatementCurveGenerator:
    """
    Marginal Abatement Cost Curve (MACC) generator.

    Creates cost curve visualization showing initiatives sorted by
    cost-effectiveness (cost per tCO2e reduced).
    """

    def __init__(self, roi_config: ROIConfig = None):
        """
        Initialize abatement curve generator.

        Args:
            roi_config: ROI configuration for calculations
        """
        self.roi_calculator = ROICalculator(roi_config)
        logger.info("Initialized AbatementCurveGenerator")

    def generate(self, initiatives: List[Initiative]) -> AbatementCurve:
        """
        Generate marginal abatement cost curve.

        Args:
            initiatives: List of emission reduction initiatives

        Returns:
            AbatementCurve with sorted initiatives

        Raises:
            AbatementCurveError: If generation fails
        """
        try:
            logger.info(f"Generating abatement curve for {len(initiatives)} initiatives")

            if not initiatives:
                raise AbatementCurveError("No initiatives provided")

            # Calculate ROI for each initiative
            initiative_rois = []
            for initiative in initiatives:
                try:
                    roi_analysis = self.roi_calculator.calculate(initiative)
                    initiative_rois.append({
                        "initiative": initiative,
                        "roi": roi_analysis
                    })
                except Exception as e:
                    logger.warning(f"Failed to calculate ROI for {initiative.name}: {e}")
                    continue

            if not initiative_rois:
                raise AbatementCurveError("No valid initiatives after ROI calculation")

            # Sort by cost-effectiveness (ascending cost per tCO2e)
            # Negative costs (savings) come first
            initiative_rois.sort(key=lambda x: x["roi"].roi_usd_per_tco2e)

            # Build curve points
            curve_points = []
            cumulative_reduction = 0.0
            cumulative_cost = 0.0

            for item in initiative_rois:
                initiative = item["initiative"]
                roi = item["roi"]

                cumulative_reduction += initiative.reduction_potential_tco2e
                cumulative_cost += initiative.implementation_cost_usd

                point = AbatementCurvePoint(
                    initiative_name=initiative.name,
                    reduction_tco2e=round(initiative.reduction_potential_tco2e, 2),
                    cost_per_tco2e=round(roi.roi_usd_per_tco2e, 2),
                    cumulative_reduction=round(cumulative_reduction, 2),
                    cumulative_cost=round(cumulative_cost, 2)
                )
                curve_points.append(point)

            # Calculate summary statistics
            total_reduction = cumulative_reduction
            total_cost = cumulative_cost

            weighted_avg_cost = (
                total_cost / total_reduction
                if total_reduction > 0 else 0
            )

            # Count negative/positive cost initiatives
            n_negative_cost = sum(
                1 for p in curve_points
                if p.cost_per_tco2e < 0
            )
            n_positive_cost = len(curve_points) - n_negative_cost

            # Generate chart data
            chart_data = self._generate_chart_data(curve_points)

            result = AbatementCurve(
                initiatives=curve_points,
                total_reduction_potential_tco2e=round(total_reduction, 2),
                total_cost_usd=round(total_cost, 2),
                weighted_average_cost_per_tco2e=round(weighted_avg_cost, 2),
                n_negative_cost=n_negative_cost,
                n_positive_cost=n_positive_cost,
                chart_data=chart_data
            )

            logger.info(
                f"Abatement curve generated: {len(curve_points)} initiatives, "
                f"total reduction={total_reduction:.1f} tCO2e, "
                f"avg cost=${weighted_avg_cost:.2f}/tCO2e, "
                f"{n_negative_cost} with savings"
            )

            return result

        except AbatementCurveError:
            raise
        except Exception as e:
            logger.error(f"Abatement curve generation failed: {e}", exc_info=True)
            raise AbatementCurveError(f"Curve generation failed: {e}") from e

    def identify_priority_initiatives(
        self,
        curve: AbatementCurve,
        max_cost_per_tco2e: float = 100.0
    ) -> List[AbatementCurvePoint]:
        """
        Identify priority initiatives based on cost threshold.

        Args:
            curve: Abatement curve
            max_cost_per_tco2e: Maximum acceptable cost per tCO2e

        Returns:
            List of priority initiatives
        """
        priority = [
            point for point in curve.initiatives
            if point.cost_per_tco2e <= max_cost_per_tco2e
        ]

        logger.info(
            f"Identified {len(priority)} priority initiatives "
            f"(cost <= ${max_cost_per_tco2e}/tCO2e)"
        )

        return priority

    def calculate_reduction_target(
        self,
        curve: AbatementCurve,
        budget_usd: float
    ) -> Dict[str, Any]:
        """
        Calculate achievable reduction given budget constraint.

        Args:
            curve: Abatement curve
            budget_usd: Available budget

        Returns:
            Achievable reduction analysis
        """
        achievable_reduction = 0.0
        achievable_initiatives = []
        remaining_budget = budget_usd

        for point in curve.initiatives:
            # Cost of this initiative
            initiative_cost = point.reduction_tco2e * point.cost_per_tco2e

            if initiative_cost <= remaining_budget:
                achievable_reduction += point.reduction_tco2e
                achievable_initiatives.append(point.initiative_name)
                remaining_budget -= initiative_cost
            else:
                # Partial implementation possible
                if point.cost_per_tco2e > 0:
                    partial_reduction = remaining_budget / point.cost_per_tco2e
                    if partial_reduction > 0:
                        achievable_reduction += partial_reduction
                        achievable_initiatives.append(f"{point.initiative_name} (Partial)")
                        remaining_budget = 0
                break

            if remaining_budget <= 0:
                break

        pct_of_total = (
            (achievable_reduction / curve.total_reduction_potential_tco2e * 100)
            if curve.total_reduction_potential_tco2e > 0 else 0
        )

        return {
            "budget_usd": budget_usd,
            "achievable_reduction_tco2e": round(achievable_reduction, 2),
            "percent_of_total_potential": round(pct_of_total, 2),
            "n_initiatives": len(achievable_initiatives),
            "initiatives": achievable_initiatives,
            "budget_utilized_usd": round(budget_usd - remaining_budget, 2),
            "budget_remaining_usd": round(remaining_budget, 2)
        }

    def _generate_chart_data(
        self,
        curve_points: List[AbatementCurvePoint]
    ) -> Dict[str, Any]:
        """
        Generate data for MACC visualization.

        Args:
            curve_points: Abatement curve points

        Returns:
            Chart data
        """
        return {
            "chart_type": "macc",
            "title": "Marginal Abatement Cost Curve",
            "data": [
                {
                    "initiative": point.initiative_name,
                    "reduction_tco2e": point.reduction_tco2e,
                    "cost_per_tco2e": point.cost_per_tco2e,
                    "cumulative_reduction": point.cumulative_reduction,
                    "cumulative_cost": point.cumulative_cost,
                    "color": "green" if point.cost_per_tco2e < 0 else "orange"
                }
                for point in curve_points
            ],
            "x_axis": "Cumulative Emissions Reduction (tCO2e)",
            "y_axis": "Cost per tCO2e (USD)",
            "zero_line": True,
            "description": (
                "MACC shows initiatives sorted by cost-effectiveness. "
                "Initiatives below zero line generate savings."
            )
        }


__all__ = ["AbatementCurveGenerator"]
