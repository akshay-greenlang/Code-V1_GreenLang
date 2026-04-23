# -*- coding: utf-8 -*-
"""
GL-002 FLAMEGUARD - Efficiency Explainer with Recommendations

Provides natural language explanations for efficiency losses and
generates actionable efficiency improvement recommendations.

Features:
- Natural language efficiency loss explanations
- Root cause analysis for low efficiency
- Prioritized improvement recommendations
- Economic impact analysis
- Optimization strategy generation

Reference: ASME PTC 4.1, Industry Best Practices
Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging

logger = logging.getLogger(__name__)


class EfficiencyFactor(str, Enum):
    """Factors affecting boiler efficiency."""
    EXCESS_AIR = "excess_air"
    STACK_TEMPERATURE = "stack_temperature"
    FUEL_MOISTURE = "fuel_moisture"
    UNBURNED_CARBON = "unburned_carbon"
    RADIATION_LOSS = "radiation_loss"
    BLOWDOWN = "blowdown"
    COMBUSTION_QUALITY = "combustion_quality"
    HEAT_RECOVERY = "heat_recovery"
    FOULING = "fouling"
    INSULATION = "insulation"


class RecommendationPriority(str, Enum):
    """Priority levels for recommendations."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationCategory(str, Enum):
    """Categories of efficiency improvements."""
    COMBUSTION = "combustion_optimization"
    HEAT_RECOVERY = "heat_recovery"
    MAINTENANCE = "maintenance"
    OPERATING_PRACTICE = "operating_practice"
    CAPITAL_IMPROVEMENT = "capital_improvement"


@dataclass
class EfficiencyLossExplanation:
    """Explanation of a single efficiency loss component."""
    loss_type: str
    loss_percent: float
    description: str
    cause: str
    typical_range: Tuple[float, float]
    is_excessive: bool
    contributing_factors: List[str] = field(default_factory=list)


@dataclass
class EfficiencyRecommendation:
    """Actionable efficiency improvement recommendation."""
    recommendation_id: str
    title: str
    description: str
    category: RecommendationCategory
    priority: RecommendationPriority
    estimated_savings_percent: float
    implementation_effort: str  # "low", "medium", "high"
    payback_period: Optional[str] = None
    steps: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class EfficiencyAnalysisReport:
    """Complete efficiency analysis with explanations and recommendations."""
    report_id: str
    boiler_id: str
    timestamp: datetime
    current_efficiency: float
    target_efficiency: float
    efficiency_gap: float
    loss_explanations: List[EfficiencyLossExplanation]
    recommendations: List[EfficiencyRecommendation]
    total_potential_savings: float
    executive_summary: str
    natural_language_explanation: str
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            content = f"{self.report_id}|{self.current_efficiency}|{self.timestamp.isoformat()}"
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


class EfficiencyExplainer:
    """
    Efficiency Explainer with Natural Language Recommendations.

    Analyzes boiler efficiency results and generates:
    - Human-readable explanations of losses
    - Prioritized improvement recommendations
    - Economic impact analysis
    """

    VERSION = "1.0.0"

    # Industry benchmarks and thresholds
    LOSS_THRESHOLDS = {
        "dry_flue_gas": {"typical": (4.0, 6.0), "warning": 7.0, "critical": 10.0},
        "hydrogen_moisture": {"typical": (10.0, 12.0), "warning": 13.0, "critical": 15.0},
        "moisture_in_fuel": {"typical": (0.0, 0.5), "warning": 1.0, "critical": 2.0},
        "radiation": {"typical": (0.5, 1.5), "warning": 2.0, "critical": 3.0},
        "blowdown": {"typical": (0.3, 0.8), "warning": 1.5, "critical": 3.0},
        "co_loss": {"typical": (0.0, 0.05), "warning": 0.1, "critical": 0.2},
        "unburned_carbon": {"typical": (0.0, 0.3), "warning": 0.5, "critical": 1.0},
    }

    # Optimal O2 by fuel type
    OPTIMAL_O2 = {
        "natural_gas": (2.0, 3.0),
        "fuel_oil_no2": (2.5, 4.0),
        "coal_bituminous": (3.5, 5.0),
    }

    def __init__(self, boiler_id: str) -> None:
        """Initialize explainer for a specific boiler."""
        self.boiler_id = boiler_id
        self._report_history: List[EfficiencyAnalysisReport] = []
        logger.info(f"EfficiencyExplainer initialized for {boiler_id}")

    def explain_efficiency_result(
        self,
        efficiency_result: Dict[str, Any],
        target_efficiency: float = 85.0,
    ) -> EfficiencyAnalysisReport:
        """
        Generate comprehensive efficiency analysis with explanations.

        Args:
            efficiency_result: Results from EfficiencyCalculator
            target_efficiency: Target efficiency percentage

        Returns:
            Complete analysis report with explanations and recommendations
        """
        current_eff = efficiency_result.get("efficiency_hhv_percent", 0)
        efficiency_gap = target_efficiency - current_eff

        # Generate loss explanations
        loss_explanations = self._generate_loss_explanations(efficiency_result)

        # Generate recommendations based on analysis
        recommendations = self._generate_recommendations(
            efficiency_result, loss_explanations, efficiency_gap
        )

        # Calculate total potential savings
        total_savings = sum(r.estimated_savings_percent for r in recommendations)

        # Generate natural language explanation
        nl_explanation = self._generate_natural_language_explanation(
            efficiency_result, loss_explanations
        )

        # Generate executive summary
        exec_summary = self._generate_executive_summary(
            current_eff, target_efficiency, recommendations
        )

        report = EfficiencyAnalysisReport(
            report_id=f"EFF-{self.boiler_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            boiler_id=self.boiler_id,
            timestamp=datetime.now(timezone.utc),
            current_efficiency=current_eff,
            target_efficiency=target_efficiency,
            efficiency_gap=efficiency_gap,
            loss_explanations=loss_explanations,
            recommendations=recommendations,
            total_potential_savings=total_savings,
            executive_summary=exec_summary,
            natural_language_explanation=nl_explanation,
        )

        self._report_history.append(report)
        return report

    def _generate_loss_explanations(
        self,
        result: Dict[str, Any],
    ) -> List[EfficiencyLossExplanation]:
        """Generate natural language explanations for each loss component."""
        explanations = []

        # Dry Flue Gas Loss
        dry_gas = result.get("dry_flue_gas_loss_percent", 0)
        thresholds = self.LOSS_THRESHOLDS["dry_flue_gas"]
        explanations.append(EfficiencyLossExplanation(
            loss_type="Dry Flue Gas Loss",
            loss_percent=dry_gas,
            description=(
                "Heat carried away by the dry combustion gases (primarily N2, CO2, O2) "
                "as they exit through the stack."
            ),
            cause=(
                "This loss is proportional to stack temperature and excess air. "
                f"At {result.get('flue_gas_temperature_f', 0):.0f}°F stack temperature "
                f"and {result.get('excess_air_percent', 0):.0f}% excess air, "
                "the hot gases carry significant heat out of the boiler."
            ),
            typical_range=thresholds["typical"],
            is_excessive=dry_gas > thresholds["warning"],
            contributing_factors=[
                "High stack temperature",
                "Excessive excess air",
                "Poor economizer performance",
            ] if dry_gas > thresholds["warning"] else [],
        ))

        # Hydrogen Moisture Loss
        h2_moisture = result.get("hydrogen_combustion_loss_percent", 0)
        thresholds = self.LOSS_THRESHOLDS["hydrogen_moisture"]
        explanations.append(EfficiencyLossExplanation(
            loss_type="Hydrogen Combustion Moisture Loss",
            loss_percent=h2_moisture,
            description=(
                "Heat absorbed to vaporize water formed from hydrogen combustion. "
                "Every pound of hydrogen produces 9 pounds of water vapor."
            ),
            cause=(
                "This is an inherent loss based on fuel hydrogen content. "
                "Natural gas has high hydrogen content (~25%), resulting in higher moisture loss. "
                "This loss is unavoidable but can be partially recovered with condensing technology."
            ),
            typical_range=thresholds["typical"],
            is_excessive=h2_moisture > thresholds["warning"],
            contributing_factors=[
                "High hydrogen content fuel",
                "High stack temperature",
                "No condensing heat recovery",
            ] if h2_moisture > thresholds["warning"] else [],
        ))

        # Radiation Loss
        radiation = result.get("radiation_loss_percent", 0)
        thresholds = self.LOSS_THRESHOLDS["radiation"]
        explanations.append(EfficiencyLossExplanation(
            loss_type="Radiation and Convection Loss",
            loss_percent=radiation,
            description=(
                "Heat lost from the boiler's external surfaces to the surrounding environment "
                "through radiation and convection."
            ),
            cause=(
                "Related to boiler surface area, insulation condition, and ambient temperature. "
                "Small boilers have higher percentage loss due to surface area to volume ratio."
            ),
            typical_range=thresholds["typical"],
            is_excessive=radiation > thresholds["warning"],
            contributing_factors=[
                "Damaged insulation",
                "Missing lagging",
                "High boiler surface temperature",
            ] if radiation > thresholds["warning"] else [],
        ))

        # Blowdown Loss
        blowdown = result.get("blowdown_loss_percent", 0)
        thresholds = self.LOSS_THRESHOLDS["blowdown"]
        explanations.append(EfficiencyLossExplanation(
            loss_type="Blowdown Loss",
            loss_percent=blowdown,
            description=(
                "Heat lost when water is discharged from the boiler to control "
                "dissolved solids concentration."
            ),
            cause=(
                f"At {result.get('blowdown_rate_percent', 0):.1f}% blowdown rate, "
                "hot water is discharged and replaced with cooler makeup water. "
                "Higher makeup water TDS or loose blowdown control increases this loss."
            ),
            typical_range=thresholds["typical"],
            is_excessive=blowdown > thresholds["warning"],
            contributing_factors=[
                "High blowdown rate",
                "Poor water treatment",
                "No blowdown heat recovery",
            ] if blowdown > thresholds["warning"] else [],
        ))

        # CO Loss (combustion quality indicator)
        co_loss = result.get("co_loss_percent", 0)
        thresholds = self.LOSS_THRESHOLDS["co_loss"]
        explanations.append(EfficiencyLossExplanation(
            loss_type="CO Loss (Incomplete Combustion)",
            loss_percent=co_loss,
            description=(
                "Heat lost due to incomplete combustion of fuel, indicated by "
                "carbon monoxide in the flue gas."
            ),
            cause=(
                f"CO at {result.get('flue_gas_co_ppm', 0):.0f} ppm indicates "
                "combustion quality. High CO suggests insufficient air mixing, "
                "burner fouling, or improper fuel/air ratio."
            ),
            typical_range=thresholds["typical"],
            is_excessive=co_loss > thresholds["warning"],
            contributing_factors=[
                "Poor burner condition",
                "Insufficient combustion air",
                "Improper fuel/air mixing",
            ] if co_loss > thresholds["warning"] else [],
        ))

        return explanations

    def _generate_recommendations(
        self,
        result: Dict[str, Any],
        explanations: List[EfficiencyLossExplanation],
        efficiency_gap: float,
    ) -> List[EfficiencyRecommendation]:
        """Generate prioritized efficiency improvement recommendations."""
        recommendations = []
        rec_id = 1

        excess_air = result.get("excess_air_percent", 0)
        stack_temp = result.get("flue_gas_temperature_f", 0)
        fuel_type = result.get("fuel_type", "natural_gas")
        optimal_o2 = self.OPTIMAL_O2.get(fuel_type, (2.0, 4.0))

        # Recommendation 1: Reduce Excess Air
        if excess_air > 25:
            savings = min((excess_air - 15) * 0.3, 3.0)  # ~0.3% per 1% excess air reduction
            recommendations.append(EfficiencyRecommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                title="Optimize Excess Air / O2 Trim",
                description=(
                    f"Current excess air at {excess_air:.0f}% is above optimal. "
                    f"Reducing O2 to {optimal_o2[0]:.1f}-{optimal_o2[1]:.1f}% can improve efficiency."
                ),
                category=RecommendationCategory.COMBUSTION,
                priority=RecommendationPriority.HIGH if excess_air > 40 else RecommendationPriority.MEDIUM,
                estimated_savings_percent=round(savings, 2),
                implementation_effort="low",
                payback_period="< 1 month",
                steps=[
                    "Verify O2 analyzer calibration",
                    "Review combustion controller O2 setpoint",
                    "Tune burner air/fuel ratio",
                    "Implement O2 trim control if not present",
                    "Verify with flue gas analysis",
                ],
            ))
            rec_id += 1

        # Recommendation 2: Reduce Stack Temperature
        if stack_temp > 400:
            savings_per_40f = 1.0
            temp_reduction = min(stack_temp - 350, 150)
            savings = (temp_reduction / 40) * savings_per_40f
            recommendations.append(EfficiencyRecommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                title="Reduce Stack Temperature",
                description=(
                    f"Stack temperature at {stack_temp:.0f}°F is above optimal. "
                    "Each 40°F reduction improves efficiency by ~1%."
                ),
                category=RecommendationCategory.HEAT_RECOVERY,
                priority=RecommendationPriority.HIGH if stack_temp > 500 else RecommendationPriority.MEDIUM,
                estimated_savings_percent=round(savings, 2),
                implementation_effort="medium" if stack_temp < 500 else "high",
                payback_period="6-18 months",
                steps=[
                    "Clean and inspect economizer",
                    "Check for air heater fouling",
                    "Evaluate soot blowing effectiveness",
                    "Consider economizer upgrade or addition",
                    "Review feedwater temperature",
                ],
            ))
            rec_id += 1

        # Recommendation 3: Blowdown Optimization
        blowdown = result.get("blowdown_rate_percent", 0)
        if blowdown > 3:
            savings = (blowdown - 2) * 0.25
            recommendations.append(EfficiencyRecommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                title="Optimize Blowdown Rate",
                description=(
                    f"Blowdown rate at {blowdown:.1f}% is above optimal. "
                    "Reducing to 2-3% through water treatment optimization can save energy."
                ),
                category=RecommendationCategory.OPERATING_PRACTICE,
                priority=RecommendationPriority.MEDIUM,
                estimated_savings_percent=round(savings, 2),
                implementation_effort="low",
                payback_period="< 3 months",
                steps=[
                    "Review water treatment program",
                    "Implement automatic TDS control",
                    "Consider blowdown heat recovery",
                    "Optimize makeup water quality",
                    "Monitor and trend blowdown rate",
                ],
                prerequisites=["Water chemistry testing capability"],
            ))
            rec_id += 1

        # Recommendation 4: Combustion Quality (CO reduction)
        co_ppm = result.get("flue_gas_co_ppm", 0)
        if co_ppm > 100:
            recommendations.append(EfficiencyRecommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                title="Improve Combustion Quality",
                description=(
                    f"CO at {co_ppm:.0f} ppm indicates incomplete combustion. "
                    "Burner tuning and maintenance can reduce CO to <100 ppm."
                ),
                category=RecommendationCategory.MAINTENANCE,
                priority=RecommendationPriority.HIGH,
                estimated_savings_percent=0.5,
                implementation_effort="medium",
                payback_period="1-3 months",
                steps=[
                    "Inspect burner for fouling or damage",
                    "Clean fuel nozzles and air registers",
                    "Verify fuel pressure and flow",
                    "Tune burner across firing range",
                    "Verify combustion air damper operation",
                ],
            ))
            rec_id += 1

        # Recommendation 5: Heat Recovery
        if stack_temp > 350 and not result.get("economizer_installed", True):
            recommendations.append(EfficiencyRecommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                title="Install/Upgrade Economizer",
                description=(
                    "An economizer can recover heat from flue gases to preheat feedwater, "
                    "reducing stack temperature and improving efficiency 3-5%."
                ),
                category=RecommendationCategory.CAPITAL_IMPROVEMENT,
                priority=RecommendationPriority.MEDIUM,
                estimated_savings_percent=4.0,
                implementation_effort="high",
                payback_period="1-3 years",
                steps=[
                    "Conduct engineering study",
                    "Size economizer for heat recovery",
                    "Plan installation during outage",
                    "Install with bypass for startup",
                    "Implement soot blowing if needed",
                ],
            ))
            rec_id += 1

        # Sort by priority and estimated savings
        priority_order = {
            RecommendationPriority.CRITICAL: 0,
            RecommendationPriority.HIGH: 1,
            RecommendationPriority.MEDIUM: 2,
            RecommendationPriority.LOW: 3,
        }
        recommendations.sort(
            key=lambda r: (priority_order[r.priority], -r.estimated_savings_percent)
        )

        return recommendations

    def _generate_natural_language_explanation(
        self,
        result: Dict[str, Any],
        explanations: List[EfficiencyLossExplanation],
    ) -> str:
        """Generate a natural language explanation of efficiency losses."""
        efficiency = result.get("efficiency_hhv_percent", 0)
        total_losses = result.get("total_losses_percent", 0)

        # Find the largest losses
        excessive_losses = [e for e in explanations if e.is_excessive]
        largest_losses = sorted(
            explanations,
            key=lambda e: e.loss_percent,
            reverse=True
        )[:3]

        explanation = f"The boiler is operating at {efficiency:.1f}% efficiency (HHV basis), "
        explanation += f"with total losses of {total_losses:.1f}%.\n\n"

        # Main losses explanation
        explanation += "Primary heat losses:\n"
        for i, loss in enumerate(largest_losses, 1):
            explanation += f"{i}. {loss.loss_type}: {loss.loss_percent:.1f}% - {loss.description}\n"

        # Problem areas
        if excessive_losses:
            explanation += "\nAreas requiring attention:\n"
            for loss in excessive_losses:
                explanation += f"- {loss.loss_type} is above normal range. {loss.cause}\n"

        # Key metrics
        explanation += f"\nKey operating parameters:\n"
        explanation += f"- Excess air: {result.get('excess_air_percent', 0):.0f}%\n"
        explanation += f"- Stack temperature: {result.get('flue_gas_temperature_f', 0):.0f}°F\n"
        explanation += f"- Flue gas O2: {result.get('flue_gas_o2_percent', 0):.1f}%\n"

        return explanation

    def _generate_executive_summary(
        self,
        current_eff: float,
        target_eff: float,
        recommendations: List[EfficiencyRecommendation],
    ) -> str:
        """Generate executive summary."""
        gap = target_eff - current_eff
        total_savings = sum(r.estimated_savings_percent for r in recommendations)
        high_priority = [r for r in recommendations if r.priority in [RecommendationPriority.CRITICAL, RecommendationPriority.HIGH]]

        summary = f"Boiler Efficiency Analysis Summary\n"
        summary += f"{'='*40}\n\n"
        summary += f"Current Efficiency: {current_eff:.1f}%\n"
        summary += f"Target Efficiency: {target_eff:.1f}%\n"
        summary += f"Improvement Needed: {gap:.1f}%\n\n"

        if gap <= 0:
            summary += "Status: MEETING TARGET - Continue current operating practices.\n"
        elif gap <= 3:
            summary += "Status: NEAR TARGET - Minor optimizations recommended.\n"
        else:
            summary += "Status: BELOW TARGET - Significant improvements available.\n"

        summary += f"\nTotal Recommendations: {len(recommendations)}\n"
        summary += f"High Priority Actions: {len(high_priority)}\n"
        summary += f"Potential Savings: Up to {total_savings:.1f}%\n\n"

        if high_priority:
            summary += "Priority Actions:\n"
            for r in high_priority[:3]:
                summary += f"  - {r.title} ({r.estimated_savings_percent:.1f}% savings)\n"

        return summary

    def get_recommendation_by_category(
        self,
        report: EfficiencyAnalysisReport,
        category: RecommendationCategory,
    ) -> List[EfficiencyRecommendation]:
        """Filter recommendations by category."""
        return [r for r in report.recommendations if r.category == category]

    def get_quick_wins(
        self,
        report: EfficiencyAnalysisReport,
    ) -> List[EfficiencyRecommendation]:
        """Get low-effort, high-impact recommendations."""
        return [
            r for r in report.recommendations
            if r.implementation_effort == "low" and r.estimated_savings_percent >= 0.5
        ]


__all__ = [
    "EfficiencyFactor",
    "RecommendationPriority",
    "RecommendationCategory",
    "EfficiencyLossExplanation",
    "EfficiencyRecommendation",
    "EfficiencyAnalysisReport",
    "EfficiencyExplainer",
]
