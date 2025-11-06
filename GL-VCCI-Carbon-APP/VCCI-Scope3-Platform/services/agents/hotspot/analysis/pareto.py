"""
Pareto Analysis Engine
GL-VCCI Scope 3 Platform

Implements Pareto principle (80/20 rule) analysis for emissions data.
Identifies top 20% contributors responsible for 80% of emissions.

Version: 1.0.0
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..models import EmissionRecord, ParetoAnalysis, ParetoItem
from ..config import ParetoConfig, DIMENSION_FIELD_MAP
from ..exceptions import ParetoAnalysisError, InsufficientDataError, DataValidationError

logger = logging.getLogger(__name__)


class ParetoAnalyzer:
    """
    Pareto analysis engine for emissions hotspot identification.

    Identifies the top 20% of contributors (suppliers, categories, etc.)
    responsible for 80% of total emissions.
    """

    def __init__(self, config: Optional[ParetoConfig] = None):
        """
        Initialize Pareto analyzer.

        Args:
            config: Pareto analysis configuration
        """
        self.config = config or ParetoConfig()
        logger.info(
            f"Initialized ParetoAnalyzer with threshold={self.config.pareto_threshold}, "
            f"top_n_percent={self.config.top_n_percent}"
        )

    def analyze(
        self,
        emissions_data: List[Dict[str, Any]],
        dimension: str = "supplier_name"
    ) -> ParetoAnalysis:
        """
        Perform Pareto analysis on emissions data.

        Args:
            emissions_data: List of emission records
            dimension: Dimension to analyze (supplier_name, scope3_category, etc.)

        Returns:
            ParetoAnalysis with top contributors

        Raises:
            ParetoAnalysisError: If analysis fails
            InsufficientDataError: If not enough data
        """
        try:
            logger.info(
                f"Starting Pareto analysis on {len(emissions_data)} records, "
                f"dimension={dimension}"
            )

            # Validate input
            self._validate_input(emissions_data, dimension)

            # Aggregate by dimension
            aggregated = self._aggregate_by_dimension(emissions_data, dimension)

            # Calculate total emissions
            total_emissions = sum(item["emissions_tco2e"] for item in aggregated)

            if total_emissions == 0:
                raise DataValidationError("Total emissions is zero")

            # Sort by emissions (descending)
            sorted_data = sorted(
                aggregated,
                key=lambda x: x["emissions_tco2e"],
                reverse=True
            )

            # Calculate cumulative percentages
            cumulative_emissions = 0
            pareto_items = []

            for rank, item in enumerate(sorted_data, start=1):
                cumulative_emissions += item["emissions_tco2e"]
                percent_of_total = (item["emissions_tco2e"] / total_emissions) * 100
                cumulative_percent = (cumulative_emissions / total_emissions) * 100

                pareto_items.append(
                    ParetoItem(
                        rank=rank,
                        entity_name=item["entity_name"],
                        emissions_tco2e=item["emissions_tco2e"],
                        percent_of_total=round(percent_of_total, 2),
                        cumulative_percent=round(cumulative_percent, 2)
                    )
                )

            # Identify top N percent
            n_top = max(1, int(len(sorted_data) * self.config.top_n_percent))
            top_items = pareto_items[:n_top]

            # Calculate Pareto efficiency (cumulative at top N%)
            pareto_efficiency = top_items[-1].cumulative_percent / 100 if top_items else 0

            # Check if Pareto rule is achieved
            pareto_achieved = pareto_efficiency >= self.config.pareto_threshold

            # Generate chart data
            chart_data = self._generate_chart_data(pareto_items, self.config.pareto_threshold)

            result = ParetoAnalysis(
                dimension=dimension,
                total_emissions_tco2e=round(total_emissions, 2),
                total_entities=len(sorted_data),
                top_20_percent=top_items,
                n_entities_in_top_20=n_top,
                pareto_threshold=self.config.pareto_threshold,
                pareto_efficiency=round(pareto_efficiency, 4),
                pareto_achieved=pareto_achieved,
                chart_data=chart_data
            )

            logger.info(
                f"Pareto analysis complete: {n_top}/{len(sorted_data)} entities "
                f"account for {pareto_efficiency*100:.1f}% of emissions "
                f"(target: {self.config.pareto_threshold*100:.0f}%)"
            )

            return result

        except (InsufficientDataError, DataValidationError):
            raise
        except Exception as e:
            logger.error(f"Pareto analysis failed: {e}", exc_info=True)
            raise ParetoAnalysisError(f"Pareto analysis failed: {e}") from e

    def _validate_input(
        self,
        emissions_data: List[Dict[str, Any]],
        dimension: str
    ) -> None:
        """
        Validate input data.

        Args:
            emissions_data: Emission records
            dimension: Dimension field

        Raises:
            InsufficientDataError: If not enough data
            DataValidationError: If data is invalid
        """
        if not emissions_data:
            raise InsufficientDataError("No emission data provided")

        if len(emissions_data) < self.config.min_records:
            raise InsufficientDataError(
                f"Need at least {self.config.min_records} records, "
                f"got {len(emissions_data)}"
            )

        # Check required fields
        first_record = emissions_data[0]
        if "emissions_tco2e" not in first_record:
            raise DataValidationError("Missing required field: emissions_tco2e")

        if dimension not in first_record and dimension != "scope3_category":
            raise DataValidationError(f"Dimension field not found: {dimension}")

    def _aggregate_by_dimension(
        self,
        emissions_data: List[Dict[str, Any]],
        dimension: str
    ) -> List[Dict[str, Any]]:
        """
        Aggregate emissions by dimension.

        Args:
            emissions_data: Emission records
            dimension: Dimension to aggregate by

        Returns:
            Aggregated data
        """
        aggregation: Dict[str, float] = {}

        for record in emissions_data:
            # Get dimension value
            if dimension == "scope3_category":
                key = f"Category {record.get(dimension, 'Unknown')}"
            else:
                key = record.get(dimension) or "Unknown"

            # Aggregate emissions
            emissions = record.get("emissions_tco2e", 0)
            aggregation[key] = aggregation.get(key, 0) + emissions

        # Convert to list
        result = [
            {
                "entity_name": name,
                "emissions_tco2e": emissions
            }
            for name, emissions in aggregation.items()
        ]

        return result

    def _generate_chart_data(
        self,
        pareto_items: List[ParetoItem],
        threshold: float
    ) -> Dict[str, Any]:
        """
        Generate data for Pareto chart visualization.

        Args:
            pareto_items: Pareto items
            threshold: Pareto threshold (e.g., 0.80)

        Returns:
            Chart data
        """
        # Limit to top 20 for readability
        display_items = pareto_items[:20]

        return {
            "chart_type": "pareto",
            "title": "Pareto Analysis - Top Contributors",
            "data": [
                {
                    "label": item.entity_name,
                    "value": item.emissions_tco2e,
                    "percent": item.percent_of_total,
                    "cumulative": item.cumulative_percent
                }
                for item in display_items
            ],
            "threshold_line": threshold * 100,
            "x_axis": "Entity",
            "y_axis_left": "Emissions (tCO2e)",
            "y_axis_right": "Cumulative %"
        }


__all__ = ["ParetoAnalyzer"]
