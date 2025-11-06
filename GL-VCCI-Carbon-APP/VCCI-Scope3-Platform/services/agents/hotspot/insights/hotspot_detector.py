"""
Hotspot Detector
GL-VCCI Scope 3 Platform

Automatically identifies emissions hotspots using configurable criteria.
Flags high-emission entities, poor data quality, and concentration risks.

Version: 1.0.0
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""

import logging
from typing import List, Dict, Any
from collections import defaultdict
import uuid

from ..models import Hotspot, HotspotReport
from ..config import HotspotCriteria, InsightPriority
from ..exceptions import HotspotDetectionError, DataValidationError

logger = logging.getLogger(__name__)


class HotspotDetector:
    """
    Emissions hotspot detector.

    Identifies hotspots based on:
    - Absolute emission thresholds
    - Percentage of total emissions
    - Data quality issues
    - Concentration risks
    """

    def __init__(self, criteria: HotspotCriteria = None):
        """
        Initialize hotspot detector.

        Args:
            criteria: Hotspot identification criteria
        """
        self.criteria = criteria or HotspotCriteria()
        logger.info(f"Initialized HotspotDetector with criteria: {self.criteria}")

    def detect(
        self,
        emissions_data: List[Dict[str, Any]],
        dimensions: List[str] = None
    ) -> HotspotReport:
        """
        Detect emissions hotspots in data.

        Args:
            emissions_data: Emission records
            dimensions: Dimensions to analyze (default: supplier, category)

        Returns:
            HotspotReport with identified hotspots

        Raises:
            HotspotDetectionError: If detection fails
        """
        try:
            logger.info(f"Detecting hotspots in {len(emissions_data)} records")

            if not emissions_data:
                raise DataValidationError("No emission data provided")

            # Default dimensions
            if dimensions is None:
                dimensions = ["supplier_name", "scope3_category"]

            # Calculate total emissions
            total_emissions = sum(r.get("emissions_tco2e", 0) for r in emissions_data)

            if total_emissions == 0:
                raise DataValidationError("Total emissions is zero")

            # Detect hotspots by dimension
            all_hotspots = []

            for dimension in dimensions:
                dimension_hotspots = self._detect_by_dimension(
                    emissions_data,
                    dimension,
                    total_emissions
                )
                all_hotspots.extend(dimension_hotspots)

            # Sort by priority and emissions
            all_hotspots.sort(
                key=lambda h: (
                    self._priority_score(h.priority),
                    -h.emissions_tco2e
                )
            )

            # Categorize by priority
            critical = [h for h in all_hotspots if h.priority == InsightPriority.CRITICAL]
            high = [h for h in all_hotspots if h.priority == InsightPriority.HIGH]

            # Calculate hotspot coverage
            hotspot_emissions = sum(h.emissions_tco2e for h in all_hotspots)
            hotspot_coverage_pct = (hotspot_emissions / total_emissions * 100) if total_emissions > 0 else 0

            report = HotspotReport(
                total_emissions_tco2e=round(total_emissions, 2),
                total_records=len(emissions_data),
                hotspots=all_hotspots,
                n_hotspots=len(all_hotspots),
                critical_hotspots=critical,
                high_hotspots=high,
                hotspot_emissions_tco2e=round(hotspot_emissions, 2),
                hotspot_coverage_pct=round(hotspot_coverage_pct, 2),
                criteria_used=self.criteria.model_dump()
            )

            logger.info(
                f"Hotspot detection complete: {len(all_hotspots)} hotspots found, "
                f"{len(critical)} critical, {len(high)} high, "
                f"coverage={hotspot_coverage_pct:.1f}%"
            )

            return report

        except DataValidationError:
            raise
        except Exception as e:
            logger.error(f"Hotspot detection failed: {e}", exc_info=True)
            raise HotspotDetectionError(f"Hotspot detection failed: {e}") from e

    def _detect_by_dimension(
        self,
        emissions_data: List[Dict[str, Any]],
        dimension: str,
        total_emissions: float
    ) -> List[Hotspot]:
        """
        Detect hotspots for specific dimension.

        Args:
            emissions_data: Emission records
            dimension: Dimension field
            total_emissions: Total emissions

        Returns:
            List of hotspots
        """
        # Aggregate by dimension
        aggregation = defaultdict(lambda: {
            "emissions_tco2e": 0,
            "record_count": 0,
            "dqi_scores": [],
            "tiers": []
        })

        for record in emissions_data:
            # Get dimension value
            if dimension == "scope3_category":
                key = f"Category {record.get(dimension, 'Unknown')}"
            else:
                key = record.get(dimension) or "Unknown"

            aggregation[key]["emissions_tco2e"] += record.get("emissions_tco2e", 0)
            aggregation[key]["record_count"] += 1

            # Collect quality metrics
            if "dqi_score" in record and record["dqi_score"] is not None:
                aggregation[key]["dqi_scores"].append(record["dqi_score"])

            if "tier" in record and record["tier"] is not None:
                aggregation[key]["tiers"].append(record["tier"])

        # Identify hotspots
        hotspots = []

        for entity_name, data in aggregation.items():
            emissions = data["emissions_tco2e"]
            percent_of_total = (emissions / total_emissions * 100) if total_emissions > 0 else 0

            # Calculate average quality metrics
            avg_dqi = (
                sum(data["dqi_scores"]) / len(data["dqi_scores"])
                if data["dqi_scores"] else None
            )
            avg_tier = (
                sum(data["tiers"]) / len(data["tiers"])
                if data["tiers"] else None
            )

            # Check criteria
            triggered_rules = []
            priority = InsightPriority.LOW

            # High emissions (absolute)
            if emissions >= self.criteria.emission_threshold_tco2e:
                triggered_rules.append(
                    f"Emissions >= {self.criteria.emission_threshold_tco2e} tCO2e"
                )
                priority = self._escalate_priority(priority, InsightPriority.HIGH)

            # High emissions (percentage)
            if percent_of_total >= self.criteria.percent_threshold:
                triggered_rules.append(
                    f"Emissions >= {self.criteria.percent_threshold}% of total"
                )
                priority = self._escalate_priority(priority, InsightPriority.HIGH)

            # Concentration risk
            if percent_of_total >= self.criteria.concentration_threshold:
                triggered_rules.append(
                    f"Concentration risk: {percent_of_total:.1f}% of total emissions"
                )
                priority = self._escalate_priority(priority, InsightPriority.CRITICAL)

            # Low data quality
            data_quality_flag = False
            if avg_dqi is not None and avg_dqi < self.criteria.dqi_threshold:
                triggered_rules.append(
                    f"Low data quality: DQI={avg_dqi:.1f} < {self.criteria.dqi_threshold}"
                )
                data_quality_flag = True
                priority = self._escalate_priority(priority, InsightPriority.MEDIUM)

            # Low tier data
            if avg_tier is not None and avg_tier >= self.criteria.tier_threshold:
                triggered_rules.append(
                    f"Low tier data: Tier {avg_tier:.1f} (spend-based)"
                )
                data_quality_flag = True
                priority = self._escalate_priority(priority, InsightPriority.MEDIUM)

            # Create hotspot if any rules triggered
            if triggered_rules:
                hotspot = Hotspot(
                    hotspot_id=str(uuid.uuid4())[:8],
                    hotspot_type=dimension,
                    entity_name=entity_name,
                    emissions_tco2e=round(emissions, 2),
                    percent_of_total=round(percent_of_total, 2),
                    triggered_rules=triggered_rules,
                    dqi_score=round(avg_dqi, 1) if avg_dqi else None,
                    tier=int(round(avg_tier)) if avg_tier else None,
                    data_quality_flag=data_quality_flag,
                    priority=priority,
                    record_count=data["record_count"]
                )
                hotspots.append(hotspot)

        return hotspots

    def _priority_score(self, priority: InsightPriority) -> int:
        """
        Convert priority to numeric score for sorting.

        Args:
            priority: Priority level

        Returns:
            Numeric score (lower = higher priority)
        """
        scores = {
            InsightPriority.CRITICAL: 0,
            InsightPriority.HIGH: 1,
            InsightPriority.MEDIUM: 2,
            InsightPriority.LOW: 3
        }
        return scores.get(priority, 4)

    def _escalate_priority(
        self,
        current: InsightPriority,
        new: InsightPriority
    ) -> InsightPriority:
        """
        Escalate priority to higher level.

        Args:
            current: Current priority
            new: New priority to consider

        Returns:
            Higher priority
        """
        if self._priority_score(new) < self._priority_score(current):
            return new
        return current


__all__ = ["HotspotDetector"]
