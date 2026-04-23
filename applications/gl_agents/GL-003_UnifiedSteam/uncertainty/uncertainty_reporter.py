"""
Uncertainty Reporting for GL-003 UNIFIEDSTEAM SteamSystemOptimizer.

This module provides user-friendly uncertainty presentation for UI and APIs,
including formatted output with confidence intervals, breakdown analysis,
and instrumentation improvement recommendations.

Zero-Hallucination Guarantee:
- All formatting and analysis is deterministic
- Recommendations based on quantitative analysis, not LLM inference
- Complete provenance for all reported values
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math
import logging

from .uncertainty_models import (
    UncertainValue,
    PropagatedUncertainty,
    PropertyUncertainty,
    UncertaintyBreakdown,
    UncertaintySource,
    ConfidenceLevel
)


logger = logging.getLogger(__name__)


class FormatStyle(Enum):
    """Output format styles for uncertainty values."""
    PLUS_MINUS = "plus_minus"        # 85.2 +/- 1.3
    PARENTHETICAL = "parenthetical"  # 85.2 (1.3)
    INTERVAL = "interval"            # [83.9, 86.5]
    PERCENT = "percent"              # 85.2 +/- 1.5%
    FULL = "full"                    # 85.2 +/- 1.3 (95% CI: 83.9-86.5)


@dataclass
class InstrumentationRecommendation:
    """
    Recommendation for improving instrumentation quality.

    Attributes:
        recommendation_id: Unique identifier
        sensor_id: Target sensor
        current_uncertainty: Current uncertainty level (%)
        target_uncertainty: Recommended target uncertainty (%)
        improvement_type: Type of improvement
        estimated_cost: Cost estimate category
        expected_benefit: Expected uncertainty reduction (%)
        priority: Recommendation priority (1-5, 1=highest)
        justification: Quantitative justification
        implementation_notes: Implementation guidance
    """
    recommendation_id: str
    sensor_id: str
    current_uncertainty: float
    target_uncertainty: float
    improvement_type: str
    estimated_cost: str
    expected_benefit: float
    priority: int
    justification: str
    implementation_notes: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class UncertaintyReporter:
    """
    Generates user-friendly uncertainty reports and recommendations.

    Provides formatted output for UI display, detailed breakdown analysis,
    and quantitative recommendations for reducing uncertainty.

    Example:
        reporter = UncertaintyReporter()

        # Format value with uncertainty
        formatted = reporter.format_with_bounds(
            value=85.2,
            uncertainty=1.3,
            confidence_level=ConfidenceLevel.CI_95
        )
        # Returns: "85.2 +/- 1.3 (95% CI)"

        # Get uncertainty breakdown
        breakdown = reporter.generate_uncertainty_breakdown(propagated_result)

        # Get recommendations
        recommendations = reporter.recommend_instrumentation_improvements(
            uncertainty_analysis
        )
    """

    def __init__(
        self,
        default_precision: int = 2,
        default_style: FormatStyle = FormatStyle.PLUS_MINUS
    ):
        """
        Initialize uncertainty reporter.

        Args:
            default_precision: Default decimal places for formatting
            default_style: Default output format style
        """
        self.default_precision = default_precision
        self.default_style = default_style

    def format_with_bounds(
        self,
        value: float,
        uncertainty: float,
        confidence_level: ConfidenceLevel = ConfidenceLevel.CI_95,
        unit: str = "",
        style: Optional[FormatStyle] = None,
        precision: Optional[int] = None
    ) -> str:
        """
        Format a value with uncertainty bounds for display.

        Args:
            value: The measured/computed value
            uncertainty: Absolute uncertainty (1-sigma)
            confidence_level: Confidence level for reported interval
            unit: Optional unit string
            style: Output format style
            precision: Decimal precision

        Returns:
            Formatted string like "85.2 +/- 1.3 (95% CI)"

        Example:
            >>> format_with_bounds(85.2, 1.3, ConfidenceLevel.CI_95)
            "85.2 +/- 2.5 (95% CI)"
        """
        style = style or self.default_style
        precision = precision if precision is not None else self.default_precision

        # Z-score for confidence level
        z_scores = {
            ConfidenceLevel.CI_68: 1.0,
            ConfidenceLevel.CI_90: 1.645,
            ConfidenceLevel.CI_95: 1.96,
            ConfidenceLevel.CI_99: 2.576,
            ConfidenceLevel.CI_99_7: 3.0
        }
        z = z_scores.get(confidence_level, 1.96)

        # Calculate interval half-width
        interval_half = uncertainty * z
        lower_bound = value - interval_half
        upper_bound = value + interval_half

        # Calculate relative uncertainty
        if abs(value) > 1e-10:
            relative_percent = (uncertainty / abs(value)) * 100 * z
        else:
            relative_percent = 0.0

        # Format based on style
        unit_str = f" {unit}" if unit else ""
        ci_label = f"{confidence_level.value:.0f}% CI"

        if style == FormatStyle.PLUS_MINUS:
            return f"{value:.{precision}f} +/- {interval_half:.{precision}f}{unit_str} ({ci_label})"

        elif style == FormatStyle.PARENTHETICAL:
            return f"{value:.{precision}f} ({interval_half:.{precision}f}){unit_str}"

        elif style == FormatStyle.INTERVAL:
            return f"[{lower_bound:.{precision}f}, {upper_bound:.{precision}f}]{unit_str}"

        elif style == FormatStyle.PERCENT:
            return f"{value:.{precision}f}{unit_str} +/- {relative_percent:.1f}%"

        elif style == FormatStyle.FULL:
            return (
                f"{value:.{precision}f} +/- {interval_half:.{precision}f}{unit_str} "
                f"({ci_label}: {lower_bound:.{precision}f}-{upper_bound:.{precision}f})"
            )

        else:
            return f"{value:.{precision}f} +/- {interval_half:.{precision}f}{unit_str}"

    def format_uncertain_value(
        self,
        uv: UncertainValue,
        style: Optional[FormatStyle] = None,
        precision: Optional[int] = None
    ) -> str:
        """
        Format an UncertainValue object for display.

        Args:
            uv: UncertainValue to format
            style: Output format style
            precision: Decimal precision

        Returns:
            Formatted string
        """
        return self.format_with_bounds(
            value=uv.mean,
            uncertainty=uv.std,
            confidence_level=ConfidenceLevel.CI_95,
            unit=uv.unit,
            style=style,
            precision=precision
        )

    def generate_uncertainty_breakdown(
        self,
        computed_value: PropagatedUncertainty
    ) -> UncertaintyBreakdown:
        """
        Generate detailed breakdown of uncertainty contributions.

        Analyzes a propagated uncertainty result and identifies
        the contribution of each input to the total uncertainty.

        Args:
            computed_value: PropagatedUncertainty result

        Returns:
            UncertaintyBreakdown with contribution analysis
        """
        # Calculate variance contributions
        contributions = computed_value.get_contribution_breakdown()

        # Sort by contribution
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Identify dominant sources (>10% contribution)
        dominant_sources = [
            name for name, contrib in sorted_contributions
            if contrib >= 10.0
        ]

        # Generate recommendations based on contributions
        recommendations = []

        for name, contrib in sorted_contributions[:3]:
            if contrib >= 30.0:
                recommendations.append(
                    f"PRIORITY: Reduce uncertainty in {name} "
                    f"(contributes {contrib:.1f}% of total uncertainty)"
                )
            elif contrib >= 15.0:
                recommendations.append(
                    f"Consider improving measurement of {name} "
                    f"({contrib:.1f}% contribution)"
                )

        if not recommendations:
            recommendations.append(
                "Uncertainty is well-distributed across inputs. "
                "Improvement requires addressing multiple sources."
            )

        # Prepare visualization data
        visualization_data = {
            "pie_chart": {
                "labels": [name for name, _ in sorted_contributions],
                "values": [contrib for _, contrib in sorted_contributions]
            },
            "bar_chart": {
                "inputs": [name for name, _ in sorted_contributions[:10]],
                "contributions": [contrib for _, contrib in sorted_contributions[:10]]
            },
            "dominant_count": len(dominant_sources),
            "total_inputs": len(contributions)
        }

        return UncertaintyBreakdown(
            total_uncertainty=computed_value.uncertainty,
            total_uncertainty_percent=computed_value.relative_uncertainty_percent(),
            contributions=dict(contributions),
            dominant_sources=dominant_sources,
            recommendations=recommendations,
            visualization_data=visualization_data
        )

    def identify_dominant_uncertainty_sources(
        self,
        propagation_result: PropagatedUncertainty,
        threshold_percent: float = 15.0
    ) -> List[UncertaintySource]:
        """
        Identify inputs contributing most to output uncertainty.

        Args:
            propagation_result: Propagated uncertainty result
            threshold_percent: Minimum contribution to be considered dominant

        Returns:
            List of UncertaintySource objects, sorted by contribution
        """
        # Get contribution breakdown
        contributions = propagation_result.get_contribution_breakdown()

        sources = []

        for input_name, contrib_percent in contributions.items():
            if contrib_percent < threshold_percent:
                continue

            input_value = propagation_result.contributing_inputs.get(input_name)
            if input_value is None:
                continue

            current_uncertainty = input_value.relative_uncertainty()

            # Determine reducibility based on source characteristics
            # (This is deterministic logic, not LLM inference)
            if current_uncertainty > 10.0:
                reducibility = "high"
                improvement_cost = "medium"
                improvement_benefit = contrib_percent * 0.5  # 50% reduction possible
            elif current_uncertainty > 5.0:
                reducibility = "medium"
                improvement_cost = "medium"
                improvement_benefit = contrib_percent * 0.3  # 30% reduction
            elif current_uncertainty > 2.0:
                reducibility = "low"
                improvement_cost = "high"
                improvement_benefit = contrib_percent * 0.2  # 20% reduction
            else:
                reducibility = "fixed"
                improvement_cost = "high"
                improvement_benefit = 0.0

            source = UncertaintySource(
                source_id=input_name,
                source_type="measurement",
                contribution_percent=contrib_percent,
                current_uncertainty=current_uncertainty,
                reducibility=reducibility,
                improvement_cost=improvement_cost,
                improvement_benefit=improvement_benefit
            )

            sources.append(source)

        # Sort by contribution (highest first)
        sources.sort(key=lambda s: s.contribution_percent, reverse=True)

        return sources

    def recommend_instrumentation_improvements(
        self,
        uncertainty_analysis: UncertaintyBreakdown,
        sensor_data: Optional[Dict[str, Dict[str, Any]]] = None,
        budget_constraint: Optional[str] = None
    ) -> List[InstrumentationRecommendation]:
        """
        Generate recommendations for reducing uncertainty.

        Analyzes the uncertainty breakdown and provides quantitative
        recommendations for instrumentation improvements.

        Args:
            uncertainty_analysis: Uncertainty breakdown to analyze
            sensor_data: Optional sensor metadata for context
            budget_constraint: Optional budget level ("low", "medium", "high")

        Returns:
            List of InstrumentationRecommendation objects
        """
        recommendations = []
        sensor_data = sensor_data or {}

        # Get Pareto sources (80% cumulative contribution)
        pareto_sources = uncertainty_analysis.get_pareto_sources(80.0)

        for idx, source_id in enumerate(pareto_sources[:5]):  # Top 5
            contribution = uncertainty_analysis.contributions.get(source_id, 0)

            if contribution < 5.0:
                continue  # Not significant enough

            # Get sensor info if available
            sensor_info = sensor_data.get(source_id, {})
            current_accuracy = sensor_info.get("accuracy", 2.0)
            sensor_type = sensor_info.get("type", "measurement")

            # Determine improvement strategy based on contribution level
            if contribution >= 40.0:
                # Major contributor - significant improvement needed
                improvement_type = "upgrade_sensor"
                target_uncertainty = current_accuracy * 0.5  # 50% reduction
                estimated_cost = "high"
                expected_benefit = contribution * 0.4
                priority = 1
                justification = (
                    f"Sensor {source_id} contributes {contribution:.1f}% of total "
                    f"uncertainty. Upgrading to higher-accuracy sensor can reduce "
                    f"overall uncertainty by approximately {expected_benefit:.1f}%."
                )
                implementation_notes = (
                    f"Consider upgrading from {current_accuracy:.2f}% accuracy to "
                    f"{target_uncertainty:.2f}% accuracy class. Evaluate "
                    f"manufacturers: Rosemount, Endress+Hauser, Yokogawa."
                )

            elif contribution >= 20.0:
                # Significant contributor - calibration or minor upgrade
                improvement_type = "recalibrate_or_upgrade"
                target_uncertainty = current_accuracy * 0.7
                estimated_cost = "medium"
                expected_benefit = contribution * 0.3
                priority = 2
                justification = (
                    f"Sensor {source_id} contributes {contribution:.1f}% of total "
                    f"uncertainty. Recalibration or targeted upgrade can reduce "
                    f"overall uncertainty by approximately {expected_benefit:.1f}%."
                )
                implementation_notes = (
                    f"First option: Schedule precision recalibration to tighten "
                    f"accuracy to {target_uncertainty:.2f}%. Second option: "
                    f"Upgrade sensor class."
                )

            else:
                # Moderate contributor - calibration improvement
                improvement_type = "recalibrate"
                target_uncertainty = current_accuracy * 0.8
                estimated_cost = "low"
                expected_benefit = contribution * 0.2
                priority = 3
                justification = (
                    f"Sensor {source_id} contributes {contribution:.1f}% of total "
                    f"uncertainty. Improved calibration frequency can help."
                )
                implementation_notes = (
                    f"Reduce calibration interval or improve calibration procedure "
                    f"to achieve {target_uncertainty:.2f}% accuracy."
                )

            # Apply budget constraint
            if budget_constraint == "low" and estimated_cost == "high":
                # Adjust to lower-cost alternative
                improvement_type = "recalibrate"
                target_uncertainty = current_accuracy * 0.85
                estimated_cost = "low"
                expected_benefit = expected_benefit * 0.5
                implementation_notes = (
                    f"Budget-constrained option: Focus on calibration improvement "
                    f"rather than sensor upgrade."
                )

            rec = InstrumentationRecommendation(
                recommendation_id=f"rec_{source_id}_{idx+1}",
                sensor_id=source_id,
                current_uncertainty=current_accuracy,
                target_uncertainty=target_uncertainty,
                improvement_type=improvement_type,
                estimated_cost=estimated_cost,
                expected_benefit=expected_benefit,
                priority=priority,
                justification=justification,
                implementation_notes=implementation_notes
            )

            recommendations.append(rec)

        # Sort by priority
        recommendations.sort(key=lambda r: (r.priority, -r.expected_benefit))

        return recommendations

    def generate_summary_report(
        self,
        propagated_results: Dict[str, PropagatedUncertainty],
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive uncertainty summary report.

        Args:
            propagated_results: Dictionary of output name to PropagatedUncertainty
            include_recommendations: Whether to include improvement recommendations

        Returns:
            Dictionary containing complete uncertainty report
        """
        report = {
            "report_timestamp": datetime.utcnow().isoformat(),
            "summary": {},
            "detailed_results": {},
            "overall_assessment": "",
            "recommendations": []
        }

        # Process each result
        total_uncertainty = 0.0
        max_uncertainty = 0.0
        worst_output = ""

        for output_name, result in propagated_results.items():
            rel_uncertainty = result.relative_uncertainty_percent()
            total_uncertainty += rel_uncertainty

            if rel_uncertainty > max_uncertainty:
                max_uncertainty = rel_uncertainty
                worst_output = output_name

            # Format for display
            formatted = self.format_with_bounds(
                result.value,
                result.uncertainty,
                ConfidenceLevel.CI_95
            )

            # Get breakdown
            breakdown = self.generate_uncertainty_breakdown(result)

            report["detailed_results"][output_name] = {
                "formatted_value": formatted,
                "value": result.value,
                "uncertainty": result.uncertainty,
                "relative_uncertainty_percent": rel_uncertainty,
                "confidence_interval_95": result.confidence_interval_95,
                "dominant_contributor": result.dominant_contributor,
                "contributions": breakdown.contributions,
                "dominant_sources": breakdown.dominant_sources
            }

        # Summary statistics
        n_outputs = len(propagated_results)
        avg_uncertainty = total_uncertainty / n_outputs if n_outputs > 0 else 0

        report["summary"] = {
            "total_outputs": n_outputs,
            "average_uncertainty_percent": avg_uncertainty,
            "maximum_uncertainty_percent": max_uncertainty,
            "worst_output": worst_output,
            "outputs_exceeding_10_percent": sum(
                1 for r in propagated_results.values()
                if r.relative_uncertainty_percent() > 10.0
            )
        }

        # Overall assessment
        if max_uncertainty > 20.0:
            assessment = (
                "CRITICAL: High uncertainty levels detected. Some outputs have "
                "uncertainty exceeding 20%, which may significantly affect "
                "decision reliability. Immediate action recommended."
            )
        elif max_uncertainty > 10.0:
            assessment = (
                "WARNING: Elevated uncertainty levels detected. Some outputs have "
                "uncertainty between 10-20%. Consider addressing dominant "
                "uncertainty sources."
            )
        elif max_uncertainty > 5.0:
            assessment = (
                "ACCEPTABLE: Uncertainty levels are within typical operating range. "
                "Monitor for drift and maintain calibration schedules."
            )
        else:
            assessment = (
                "GOOD: Uncertainty levels are low across all outputs. "
                "Continue current maintenance practices."
            )

        report["overall_assessment"] = assessment

        # Generate recommendations if requested
        if include_recommendations and n_outputs > 0:
            # Use the result with highest uncertainty for recommendations
            if worst_output:
                worst_result = propagated_results[worst_output]
                breakdown = self.generate_uncertainty_breakdown(worst_result)
                recommendations = self.recommend_instrumentation_improvements(
                    breakdown
                )

                report["recommendations"] = [
                    {
                        "sensor_id": rec.sensor_id,
                        "priority": rec.priority,
                        "improvement_type": rec.improvement_type,
                        "expected_benefit": rec.expected_benefit,
                        "estimated_cost": rec.estimated_cost,
                        "justification": rec.justification
                    }
                    for rec in recommendations[:5]  # Top 5
                ]

        return report

    def format_for_api(
        self,
        value: float,
        uncertainty: float,
        confidence_level: ConfidenceLevel = ConfidenceLevel.CI_95,
        unit: str = ""
    ) -> Dict[str, Any]:
        """
        Format uncertainty data for API response.

        Returns structured data suitable for JSON API response.

        Args:
            value: The value
            uncertainty: 1-sigma uncertainty
            confidence_level: Confidence level
            unit: Unit string

        Returns:
            Dictionary with structured uncertainty data
        """
        z = {
            ConfidenceLevel.CI_68: 1.0,
            ConfidenceLevel.CI_90: 1.645,
            ConfidenceLevel.CI_95: 1.96,
            ConfidenceLevel.CI_99: 2.576,
            ConfidenceLevel.CI_99_7: 3.0
        }.get(confidence_level, 1.96)

        interval_half = uncertainty * z
        lower_bound = value - interval_half
        upper_bound = value + interval_half

        return {
            "value": value,
            "uncertainty_1sigma": uncertainty,
            "uncertainty_at_confidence": interval_half,
            "confidence_level": confidence_level.value,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "relative_uncertainty_percent": (
                (uncertainty / abs(value)) * 100 if abs(value) > 1e-10 else None
            ),
            "unit": unit,
            "formatted": self.format_with_bounds(
                value, uncertainty, confidence_level, unit
            )
        }

    def format_table_row(
        self,
        name: str,
        uv: UncertainValue,
        precision: int = 2
    ) -> Dict[str, str]:
        """
        Format uncertainty data for table display.

        Args:
            name: Row name/label
            uv: UncertainValue to format
            precision: Decimal precision

        Returns:
            Dictionary with formatted table columns
        """
        return {
            "name": name,
            "value": f"{uv.mean:.{precision}f}",
            "uncertainty": f"+/- {uv.std * 1.96:.{precision}f}",
            "lower_95": f"{uv.lower_95:.{precision}f}",
            "upper_95": f"{uv.upper_95:.{precision}f}",
            "relative": f"{uv.relative_uncertainty():.1f}%",
            "unit": uv.unit
        }
