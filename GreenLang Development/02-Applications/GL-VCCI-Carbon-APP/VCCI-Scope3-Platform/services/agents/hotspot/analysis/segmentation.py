# -*- coding: utf-8 -*-
"""
Segmentation Analysis Engine
GL-VCCI Scope 3 Platform

Multi-dimensional segmentation of emissions data.
Analyzes emissions by supplier, category, product, region, facility, etc.

Version: 1.0.0
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

from ..models import EmissionRecord, SegmentationAnalysis, Segment
from ..config import SegmentationConfig, AnalysisDimension, DIMENSION_FIELD_MAP
from ..exceptions import SegmentationError, InvalidDimensionError, DataValidationError

logger = logging.getLogger(__name__)


class SegmentationAnalyzer:
    """
    Multi-dimensional segmentation analyzer for emissions data.

    Segments emissions by various dimensions (supplier, category, product, etc.)
    and provides detailed breakdowns with quality metrics.
    """

    def __init__(self, config: Optional[SegmentationConfig] = None):
        """
        Initialize segmentation analyzer.

        Args:
            config: Segmentation configuration
        """
        self.config = config or SegmentationConfig()
        logger.info(f"Initialized SegmentationAnalyzer with config: {self.config}")

    def analyze(
        self,
        emissions_data: List[Dict[str, Any]],
        dimension: AnalysisDimension
    ) -> SegmentationAnalysis:
        """
        Perform segmentation analysis on emissions data.

        Args:
            emissions_data: List of emission records
            dimension: Dimension to segment by

        Returns:
            SegmentationAnalysis with segments

        Raises:
            SegmentationError: If analysis fails
            InvalidDimensionError: If dimension is invalid
        """
        try:
            logger.info(
                f"Starting segmentation analysis on {len(emissions_data)} records, "
                f"dimension={dimension.value}"
            )

            # Validate input
            self._validate_input(emissions_data, dimension)

            # Get dimension field name
            dimension_field = DIMENSION_FIELD_MAP.get(dimension)
            if not dimension_field:
                raise InvalidDimensionError(f"No field mapping for dimension: {dimension}")

            # Aggregate by dimension
            segments_data = self._aggregate_by_dimension(emissions_data, dimension_field)

            # Calculate total emissions
            total_emissions = sum(s["emissions_tco2e"] for s in segments_data)
            total_records = len(emissions_data)

            if total_emissions == 0:
                raise DataValidationError("Total emissions is zero")

            # Create segment objects
            segments = []
            for seg_data in segments_data:
                percent_of_total = (seg_data["emissions_tco2e"] / total_emissions) * 100

                segment = Segment(
                    segment_name=seg_data["name"],
                    emissions_tco2e=round(seg_data["emissions_tco2e"], 2),
                    percent_of_total=round(percent_of_total, 2),
                    record_count=seg_data["record_count"],
                    avg_dqi_score=seg_data.get("avg_dqi_score"),
                    avg_uncertainty_pct=seg_data.get("avg_uncertainty_pct"),
                    total_spend_usd=seg_data.get("total_spend_usd"),
                    metadata=seg_data.get("metadata", {})
                )
                segments.append(segment)

            # Sort by emissions (descending)
            segments.sort(key=lambda s: s.emissions_tco2e, reverse=True)

            # Apply segment limit and aggregation
            if len(segments) > self.config.max_segments_per_dimension:
                if self.config.aggregate_small_segments:
                    segments = self._aggregate_small_segments(
                        segments,
                        self.config.max_segments_per_dimension
                    )
                else:
                    segments = segments[:self.config.max_segments_per_dimension]

            # Get top 10
            top_10 = segments[:10]

            # Calculate concentration (top 3)
            top_3_concentration = sum(s.percent_of_total for s in segments[:3])

            # Generate chart data
            chart_data = self._generate_chart_data(segments, dimension)

            result = SegmentationAnalysis(
                dimension=dimension,
                total_emissions_tco2e=round(total_emissions, 2),
                total_records=total_records,
                segments=segments,
                top_10_segments=top_10,
                n_segments=len(segments),
                top_3_concentration=round(top_3_concentration, 2),
                chart_data=chart_data
            )

            logger.info(
                f"Segmentation complete: {len(segments)} segments, "
                f"top 3 concentration={top_3_concentration:.1f}%"
            )

            return result

        except (InvalidDimensionError, DataValidationError):
            raise
        except Exception as e:
            logger.error(f"Segmentation analysis failed: {e}", exc_info=True)
            raise SegmentationError(f"Segmentation analysis failed: {e}") from e

    def analyze_multiple_dimensions(
        self,
        emissions_data: List[Dict[str, Any]],
        dimensions: List[AnalysisDimension]
    ) -> Dict[AnalysisDimension, SegmentationAnalysis]:
        """
        Perform segmentation across multiple dimensions.

        Args:
            emissions_data: Emission records
            dimensions: List of dimensions to analyze

        Returns:
            Dictionary mapping dimension to analysis result
        """
        results = {}

        for dimension in dimensions:
            try:
                result = self.analyze(emissions_data, dimension)
                results[dimension] = result
            except Exception as e:
                logger.warning(
                    f"Failed to analyze dimension {dimension.value}: {e}"
                )
                continue

        return results

    def _validate_input(
        self,
        emissions_data: List[Dict[str, Any]],
        dimension: AnalysisDimension
    ) -> None:
        """
        Validate input data.

        Args:
            emissions_data: Emission records
            dimension: Dimension to validate

        Raises:
            DataValidationError: If data is invalid
        """
        if not emissions_data:
            raise DataValidationError("No emission data provided")

        # Check required fields
        first_record = emissions_data[0]
        if "emissions_tco2e" not in first_record:
            raise DataValidationError("Missing required field: emissions_tco2e")

    def _aggregate_by_dimension(
        self,
        emissions_data: List[Dict[str, Any]],
        dimension_field: str
    ) -> List[Dict[str, Any]]:
        """
        Aggregate emissions by dimension with quality metrics.

        Args:
            emissions_data: Emission records
            dimension_field: Field name to aggregate by

        Returns:
            Aggregated segment data
        """
        aggregation = defaultdict(lambda: {
            "emissions_tco2e": 0,
            "record_count": 0,
            "dqi_scores": [],
            "uncertainty_pcts": [],
            "spend_usd": 0
        })

        for record in emissions_data:
            # Get dimension value
            if dimension_field == "scope3_category":
                key = f"Category {record.get(dimension_field, 'Unknown')}"
            else:
                key = record.get(dimension_field) or "Unknown"

            # Aggregate emissions
            aggregation[key]["emissions_tco2e"] += record.get("emissions_tco2e", 0)
            aggregation[key]["record_count"] += 1

            # Collect quality metrics
            if "dqi_score" in record and record["dqi_score"] is not None:
                aggregation[key]["dqi_scores"].append(record["dqi_score"])

            if "uncertainty_pct" in record and record["uncertainty_pct"] is not None:
                aggregation[key]["uncertainty_pcts"].append(record["uncertainty_pct"])

            if "spend_usd" in record and record["spend_usd"] is not None:
                aggregation[key]["spend_usd"] += record["spend_usd"]

        # Convert to list with averages
        result = []
        for name, data in aggregation.items():
            # Skip if below threshold
            if data["emissions_tco2e"] < self.config.min_emission_threshold_tco2e:
                continue

            segment_data = {
                "name": name,
                "emissions_tco2e": data["emissions_tco2e"],
                "record_count": data["record_count"],
                "avg_dqi_score": (
                    round(sum(data["dqi_scores"]) / len(data["dqi_scores"]), 1)
                    if data["dqi_scores"] else None
                ),
                "avg_uncertainty_pct": (
                    round(sum(data["uncertainty_pcts"]) / len(data["uncertainty_pcts"]), 1)
                    if data["uncertainty_pcts"] else None
                ),
                "total_spend_usd": data["spend_usd"] if data["spend_usd"] > 0 else None
            }
            result.append(segment_data)

        return result

    def _aggregate_small_segments(
        self,
        segments: List[Segment],
        max_segments: int
    ) -> List[Segment]:
        """
        Aggregate small segments into 'Other' category.

        Args:
            segments: All segments (sorted by emissions desc)
            max_segments: Maximum segments to keep

        Returns:
            Segments with small ones aggregated
        """
        if len(segments) <= max_segments:
            return segments

        # Keep top segments
        top_segments = segments[:max_segments - 1]

        # Aggregate remaining
        other_emissions = sum(s.emissions_tco2e for s in segments[max_segments - 1:])
        other_records = sum(s.record_count for s in segments[max_segments - 1:])
        other_spend = sum(
            s.total_spend_usd for s in segments[max_segments - 1:]
            if s.total_spend_usd is not None
        )

        # Calculate total for percentage
        total_emissions = sum(s.emissions_tco2e for s in segments)
        other_percent = (other_emissions / total_emissions) * 100 if total_emissions > 0 else 0

        # Create 'Other' segment
        other_segment = Segment(
            segment_name="Other",
            emissions_tco2e=round(other_emissions, 2),
            percent_of_total=round(other_percent, 2),
            record_count=other_records,
            total_spend_usd=other_spend if other_spend > 0 else None,
            metadata={"aggregated": True, "n_segments": len(segments) - (max_segments - 1)}
        )

        return top_segments + [other_segment]

    def _generate_chart_data(
        self,
        segments: List[Segment],
        dimension: AnalysisDimension
    ) -> Dict[str, Any]:
        """
        Generate data for segmentation chart visualization.

        Args:
            segments: Segments
            dimension: Analysis dimension

        Returns:
            Chart data
        """
        # Top 10 for bar chart
        top_10 = segments[:10]

        return {
            "chart_type": "segmentation",
            "title": f"Segmentation by {dimension.value.replace('_', ' ').title()}",
            "bar_chart": {
                "data": [
                    {
                        "label": seg.segment_name,
                        "value": seg.emissions_tco2e,
                        "percent": seg.percent_of_total
                    }
                    for seg in top_10
                ],
                "x_axis": dimension.value.replace("_", " ").title(),
                "y_axis": "Emissions (tCO2e)"
            },
            "pie_chart": {
                "data": [
                    {
                        "label": seg.segment_name,
                        "value": seg.emissions_tco2e,
                        "percent": seg.percent_of_total
                    }
                    for seg in segments[:8]  # Top 8 for pie chart
                ]
            }
        }


__all__ = ["SegmentationAnalyzer"]
