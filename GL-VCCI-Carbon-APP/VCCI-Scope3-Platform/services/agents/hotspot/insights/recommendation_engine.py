# -*- coding: utf-8 -*-
"""
Recommendation Engine
GL-VCCI Scope 3 Platform

Generates actionable insights and recommendations from analysis results.
Provides context-aware guidance for emission reduction actions.

Version: 1.0.0
Phase: 3 (Weeks 14-16)
Date: 2025-10-30
"""

import logging
from typing import List, Dict, Any
import uuid

from ..models import (
from greenlang.determinism import deterministic_uuid, DeterministicClock
    Insight,
    InsightReport,
    HotspotReport,
    ParetoAnalysis,
    SegmentationAnalysis,
    AbatementCurve
)
from ..config import InsightPriority, InsightType
from ..exceptions import InsightGenerationError

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Actionable recommendation generator.

    Analyzes hotspots, Pareto analysis, and segmentation results
    to generate prioritized, actionable insights.
    """

    def __init__(self):
        """Initialize recommendation engine."""
        logger.info("Initialized RecommendationEngine")

    def generate_insights(
        self,
        hotspot_report: HotspotReport = None,
        pareto_analysis: ParetoAnalysis = None,
        segmentation_analysis: SegmentationAnalysis = None,
        abatement_curve: AbatementCurve = None
    ) -> InsightReport:
        """
        Generate comprehensive insights from analysis results.

        Args:
            hotspot_report: Hotspot detection results
            pareto_analysis: Pareto analysis results
            segmentation_analysis: Segmentation analysis results
            abatement_curve: Abatement curve results

        Returns:
            InsightReport with prioritized recommendations

        Raises:
            InsightGenerationError: If generation fails
        """
        try:
            logger.info("Generating actionable insights")

            insights = []

            # Generate insights from hotspots
            if hotspot_report:
                insights.extend(self._insights_from_hotspots(hotspot_report))

            # Generate insights from Pareto analysis
            if pareto_analysis:
                insights.extend(self._insights_from_pareto(pareto_analysis))

            # Generate insights from segmentation
            if segmentation_analysis:
                insights.extend(self._insights_from_segmentation(segmentation_analysis))

            # Generate insights from abatement curve
            if abatement_curve:
                insights.extend(self._insights_from_abatement_curve(abatement_curve))

            # Sort by priority
            insights.sort(
                key=lambda i: self._priority_score(i.priority)
            )

            # Categorize by priority
            critical = [i for i in insights if i.priority == InsightPriority.CRITICAL]
            high = [i for i in insights if i.priority == InsightPriority.HIGH]
            medium = [i for i in insights if i.priority == InsightPriority.MEDIUM]
            low = [i for i in insights if i.priority == InsightPriority.LOW]

            # Generate summary
            summary = self._generate_summary(insights)

            # Extract top recommendations
            top_recommendations = [i.recommendation for i in insights[:5]]

            report = InsightReport(
                total_insights=len(insights),
                critical_insights=critical,
                high_insights=high,
                medium_insights=medium,
                low_insights=low,
                all_insights=insights,
                summary=summary,
                top_recommendations=top_recommendations
            )

            logger.info(
                f"Generated {len(insights)} insights: "
                f"{len(critical)} critical, {len(high)} high, "
                f"{len(medium)} medium, {len(low)} low"
            )

            return report

        except Exception as e:
            logger.error(f"Insight generation failed: {e}", exc_info=True)
            raise InsightGenerationError(f"Insight generation failed: {e}") from e

    def _insights_from_hotspots(
        self,
        hotspot_report: HotspotReport
    ) -> List[Insight]:
        """
        Generate insights from hotspot detection.

        Args:
            hotspot_report: Hotspot report

        Returns:
            List of insights
        """
        insights = []

        for hotspot in hotspot_report.hotspots:
            # High emissions supplier
            if hotspot.hotspot_type == "supplier_name" and hotspot.emissions_tco2e >= 1000:
                insight = Insight(
                    insight_id=str(deterministic_uuid(__name__, str(DeterministicClock.now())))[:8],
                    insight_type=InsightType.HIGH_EMISSIONS_SUPPLIER,
                    priority=hotspot.priority,
                    title=f"High Emissions from {hotspot.entity_name}",
                    description=(
                        f"{hotspot.entity_name} contributes {hotspot.emissions_tco2e:,.0f} tCO2e "
                        f"({hotspot.percent_of_total:.1f}% of total emissions). "
                        f"This represents a significant emission source."
                    ),
                    recommendation=(
                        f"Engage {hotspot.entity_name} for primary data collection and "
                        "explore emission reduction opportunities through supplier collaboration programs."
                    ),
                    affected_entity=hotspot.entity_name,
                    emissions_tco2e=hotspot.emissions_tco2e,
                    percent_of_total=hotspot.percent_of_total,
                    estimated_impact=(
                        f"Upgrading to primary data could improve DQI by 30-40 points. "
                        f"Supplier engagement could reduce emissions by 10-20%."
                    ),
                    potential_reduction_tco2e=hotspot.emissions_tco2e * 0.15,
                    metrics={
                        "current_dqi": hotspot.dqi_score,
                        "current_tier": hotspot.tier,
                        "record_count": hotspot.record_count
                    }
                )
                insights.append(insight)

            # Low data quality
            if hotspot.data_quality_flag:
                insight = Insight(
                    insight_id=str(deterministic_uuid(__name__, str(DeterministicClock.now())))[:8],
                    insight_type=InsightType.LOW_DATA_QUALITY,
                    priority=InsightPriority.MEDIUM,
                    title=f"Low Data Quality for {hotspot.entity_name}",
                    description=(
                        f"{hotspot.entity_name} has low data quality "
                        f"(DQI={hotspot.dqi_score:.1f}, Tier {hotspot.tier}). "
                        f"Improving data quality will reduce uncertainty."
                    ),
                    recommendation=(
                        f"Prioritize primary data collection for {hotspot.entity_name}. "
                        "Request product carbon footprints or activity-based data."
                    ),
                    affected_entity=hotspot.entity_name,
                    emissions_tco2e=hotspot.emissions_tco2e,
                    estimated_impact="Upgrade from Tier 3 to Tier 1 data",
                    metrics={
                        "current_dqi": hotspot.dqi_score,
                        "current_tier": hotspot.tier
                    }
                )
                insights.append(insight)

            # Concentration risk
            if hotspot.percent_of_total >= 30:
                insight = Insight(
                    insight_id=str(deterministic_uuid(__name__, str(DeterministicClock.now())))[:8],
                    insight_type=InsightType.CONCENTRATION_RISK,
                    priority=InsightPriority.CRITICAL,
                    title=f"Concentration Risk: {hotspot.entity_name}",
                    description=(
                        f"{hotspot.entity_name} represents {hotspot.percent_of_total:.1f}% "
                        "of total emissions. High concentration creates risk."
                    ),
                    recommendation=(
                        "Diversify supply chain to reduce concentration risk. "
                        "Develop alternative suppliers and engage current supplier "
                        "for emission reduction commitments."
                    ),
                    affected_entity=hotspot.entity_name,
                    emissions_tco2e=hotspot.emissions_tco2e,
                    percent_of_total=hotspot.percent_of_total,
                    metrics={
                        "concentration_pct": hotspot.percent_of_total
                    }
                )
                insights.append(insight)

        return insights

    def _insights_from_pareto(
        self,
        pareto_analysis: ParetoAnalysis
    ) -> List[Insight]:
        """
        Generate insights from Pareto analysis.

        Args:
            pareto_analysis: Pareto analysis

        Returns:
            List of insights
        """
        insights = []

        # Pareto efficiency insight
        if pareto_analysis.pareto_achieved:
            insight = Insight(
                insight_id=str(deterministic_uuid(__name__, str(DeterministicClock.now())))[:8],
                insight_type=InsightType.CONCENTRATION_RISK,
                priority=InsightPriority.HIGH,
                title="Strong Pareto Pattern Detected",
                description=(
                    f"Top {pareto_analysis.n_entities_in_top_20} entities "
                    f"({pareto_analysis.n_entities_in_top_20 / pareto_analysis.total_entities * 100:.0f}% of total) "
                    f"contribute {pareto_analysis.pareto_efficiency * 100:.0f}% of emissions. "
                    "Focus on these key contributors for maximum impact."
                ),
                recommendation=(
                    "Prioritize engagement with top 20% contributors. "
                    "Implement targeted reduction programs for these entities."
                ),
                estimated_impact=f"Targeting top 20% addresses {pareto_analysis.pareto_efficiency * 100:.0f}% of emissions",
                metrics={
                    "pareto_efficiency": pareto_analysis.pareto_efficiency,
                    "n_top_entities": pareto_analysis.n_entities_in_top_20
                }
            )
            insights.append(insight)

        return insights

    def _insights_from_segmentation(
        self,
        segmentation_analysis: SegmentationAnalysis
    ) -> List[Insight]:
        """
        Generate insights from segmentation analysis.

        Args:
            segmentation_analysis: Segmentation analysis

        Returns:
            List of insights
        """
        insights = []

        # Top segment insight
        if segmentation_analysis.top_10_segments:
            top_segment = segmentation_analysis.top_10_segments[0]

            insight = Insight(
                insight_id=str(deterministic_uuid(__name__, str(DeterministicClock.now())))[:8],
                insight_type=InsightType.HIGH_EMISSIONS_CATEGORY,
                priority=InsightPriority.HIGH,
                title=f"Highest Emissions: {top_segment.segment_name}",
                description=(
                    f"{top_segment.segment_name} is the largest emission source, "
                    f"contributing {top_segment.emissions_tco2e:,.0f} tCO2e "
                    f"({top_segment.percent_of_total:.1f}% of total)."
                ),
                recommendation=(
                    f"Develop targeted reduction strategy for {top_segment.segment_name}. "
                    "Conduct detailed analysis of reduction opportunities."
                ),
                affected_entity=top_segment.segment_name,
                emissions_tco2e=top_segment.emissions_tco2e,
                percent_of_total=top_segment.percent_of_total,
                metrics={
                    "record_count": top_segment.record_count,
                    "avg_dqi": top_segment.avg_dqi_score
                }
            )
            insights.append(insight)

        return insights

    def _insights_from_abatement_curve(
        self,
        abatement_curve: AbatementCurve
    ) -> List[Insight]:
        """
        Generate insights from abatement curve.

        Args:
            abatement_curve: Abatement curve

        Returns:
            List of insights
        """
        insights = []

        # Quick wins (negative cost)
        if abatement_curve.n_negative_cost > 0:
            negative_cost_initiatives = [
                i for i in abatement_curve.initiatives
                if i.cost_per_tco2e < 0
            ]
            total_reduction = sum(i.reduction_tco2e for i in negative_cost_initiatives)
            total_savings = sum(abs(i.cost_per_tco2e * i.reduction_tco2e) for i in negative_cost_initiatives)

            insight = Insight(
                insight_id=str(deterministic_uuid(__name__, str(DeterministicClock.now())))[:8],
                insight_type=InsightType.QUICK_WIN,
                priority=InsightPriority.CRITICAL,
                title=f"Quick Wins: {abatement_curve.n_negative_cost} Cost-Saving Opportunities",
                description=(
                    f"Identified {abatement_curve.n_negative_cost} initiatives with negative costs "
                    f"(generate savings). Combined reduction potential: {total_reduction:,.0f} tCO2e "
                    f"with ${total_savings:,.0f} in savings."
                ),
                recommendation=(
                    "Implement all negative-cost initiatives immediately. "
                    "These provide both emissions reductions and financial benefits."
                ),
                potential_reduction_tco2e=total_reduction,
                metrics={
                    "n_initiatives": abatement_curve.n_negative_cost,
                    "total_savings_usd": total_savings
                }
            )
            insights.append(insight)

        # Cost-effective opportunities
        low_cost_initiatives = [
            i for i in abatement_curve.initiatives
            if 0 <= i.cost_per_tco2e <= 50
        ]
        if low_cost_initiatives:
            total_reduction = sum(i.reduction_tco2e for i in low_cost_initiatives)

            insight = Insight(
                insight_id=str(deterministic_uuid(__name__, str(DeterministicClock.now())))[:8],
                insight_type=InsightType.COST_EFFECTIVE_REDUCTION,
                priority=InsightPriority.HIGH,
                title=f"Cost-Effective Opportunities: {len(low_cost_initiatives)} Initiatives",
                description=(
                    f"Identified {len(low_cost_initiatives)} initiatives with cost < $50/tCO2e. "
                    f"Combined reduction potential: {total_reduction:,.0f} tCO2e."
                ),
                recommendation=(
                    "Prioritize low-cost initiatives ($0-50/tCO2e) for implementation. "
                    "These provide strong ROI."
                ),
                potential_reduction_tco2e=total_reduction,
                metrics={
                    "n_initiatives": len(low_cost_initiatives),
                    "avg_cost_per_tco2e": sum(i.cost_per_tco2e for i in low_cost_initiatives) / len(low_cost_initiatives)
                }
            )
            insights.append(insight)

        return insights

    def _generate_summary(self, insights: List[Insight]) -> str:
        """
        Generate executive summary of insights.

        Args:
            insights: All insights

        Returns:
            Summary text
        """
        n_critical = sum(1 for i in insights if i.priority == InsightPriority.CRITICAL)
        n_high = sum(1 for i in insights if i.priority == InsightPriority.HIGH)

        total_reduction_potential = sum(
            i.potential_reduction_tco2e for i in insights
            if i.potential_reduction_tco2e is not None
        )

        summary = (
            f"Analysis identified {len(insights)} actionable insights "
            f"({n_critical} critical, {n_high} high priority). "
        )

        if total_reduction_potential > 0:
            summary += (
                f"Total reduction potential: {total_reduction_potential:,.0f} tCO2e. "
            )

        summary += (
            "Focus on high-emission hotspots, data quality improvement, "
            "and cost-effective reduction initiatives for maximum impact."
        )

        return summary

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


__all__ = ["RecommendationEngine"]
