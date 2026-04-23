"""
GL-002 FLAMEGUARD - Optimization Explainer

Detailed explanations for optimization recommendations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class OptimizationFactor:
    """Single factor in optimization decision."""
    name: str
    category: str  # efficiency, emissions, cost, safety
    weight: float
    current_value: float
    optimal_value: float
    improvement: float
    explanation: str

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "category": self.category,
            "weight": self.weight,
            "current_value": self.current_value,
            "optimal_value": self.optimal_value,
            "improvement": self.improvement,
            "explanation": self.explanation,
        }


@dataclass
class TradeoffAnalysis:
    """Analysis of optimization tradeoffs."""
    tradeoff_name: str
    option_a: str
    option_b: str
    chosen: str
    rationale: str
    impact: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "tradeoff_name": self.tradeoff_name,
            "option_a": self.option_a,
            "option_b": self.option_b,
            "chosen": self.chosen,
            "rationale": self.rationale,
            "impact": self.impact,
        }


@dataclass
class OptimizationExplanation:
    """Complete explanation for optimization recommendation."""
    explanation_id: str
    timestamp: datetime
    boiler_id: str
    optimization_mode: str

    # Current and recommended state
    current_state: Dict[str, float]
    recommended_state: Dict[str, float]

    # Factors considered
    factors: List[OptimizationFactor]

    # Tradeoffs
    tradeoffs: List[TradeoffAnalysis]

    # Expected improvements
    expected_improvements: Dict[str, float]

    # Constraints applied
    constraints_applied: List[str]
    constraints_limiting: List[str]

    # Natural language summary
    summary: str = ""
    detailed_rationale: str = ""

    # Confidence and uncertainty
    confidence: float = 0.0
    uncertainty_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "explanation_id": self.explanation_id,
            "timestamp": self.timestamp.isoformat(),
            "boiler_id": self.boiler_id,
            "optimization_mode": self.optimization_mode,
            "current_state": self.current_state,
            "recommended_state": self.recommended_state,
            "factors": [f.to_dict() for f in self.factors],
            "tradeoffs": [t.to_dict() for t in self.tradeoffs],
            "expected_improvements": self.expected_improvements,
            "constraints_applied": self.constraints_applied,
            "constraints_limiting": self.constraints_limiting,
            "summary": self.summary,
            "detailed_rationale": self.detailed_rationale,
            "confidence": self.confidence,
            "uncertainty_factors": self.uncertainty_factors,
        }


class OptimizationExplainer:
    """
    Explains optimization recommendations in detail.

    Features:
    - Factor-by-factor analysis
    - Tradeoff explanation
    - Constraint impact analysis
    - Natural language summaries
    """

    def __init__(self, agent_id: str = "GL-002") -> None:
        self.agent_id = agent_id
        self._explanations: Dict[str, OptimizationExplanation] = {}

        # Factor descriptions
        self._factor_descriptions = {
            "o2_setpoint": {
                "name": "O2 Setpoint",
                "category": "efficiency",
                "description": "Optimal oxygen concentration for combustion efficiency",
            },
            "excess_air": {
                "name": "Excess Air",
                "category": "efficiency",
                "description": "Air above stoichiometric requirement",
            },
            "load_allocation": {
                "name": "Load Allocation",
                "category": "efficiency",
                "description": "Distribution of steam load across boilers",
            },
            "co_limit": {
                "name": "CO Limit",
                "category": "emissions",
                "description": "Carbon monoxide emission constraint",
            },
            "nox_limit": {
                "name": "NOx Limit",
                "category": "emissions",
                "description": "Nitrogen oxide emission constraint",
            },
        }

        logger.info(f"OptimizationExplainer initialized: {agent_id}")

    def explain_combustion_optimization(
        self,
        boiler_id: str,
        mode: str,
        current: Dict[str, float],
        recommended: Dict[str, float],
        constraints: Dict[str, Any],
    ) -> OptimizationExplanation:
        """Generate explanation for combustion optimization."""

        factors = self._analyze_combustion_factors(current, recommended)
        tradeoffs = self._analyze_combustion_tradeoffs(mode, current, recommended)
        improvements = self._calculate_improvements(current, recommended)
        active_constraints, limiting = self._analyze_constraints(constraints, recommended)

        explanation = OptimizationExplanation(
            explanation_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            boiler_id=boiler_id,
            optimization_mode=mode,
            current_state=current,
            recommended_state=recommended,
            factors=factors,
            tradeoffs=tradeoffs,
            expected_improvements=improvements,
            constraints_applied=active_constraints,
            constraints_limiting=limiting,
        )

        self._generate_summary(explanation)
        self._generate_detailed_rationale(explanation)
        self._calculate_confidence(explanation)

        self._explanations[explanation.explanation_id] = explanation

        return explanation

    def _analyze_combustion_factors(
        self,
        current: Dict[str, float],
        recommended: Dict[str, float],
    ) -> List[OptimizationFactor]:
        """Analyze factors contributing to recommendation."""
        factors = []

        # O2 setpoint analysis
        if "o2_setpoint" in recommended:
            current_o2 = current.get("o2_percent", 3.5)
            recommended_o2 = recommended["o2_setpoint"]
            improvement = (current_o2 - recommended_o2) * 0.3  # ~0.3% efficiency per 1% O2

            factors.append(OptimizationFactor(
                name="O2 Setpoint",
                category="efficiency",
                weight=0.4,
                current_value=current_o2,
                optimal_value=recommended_o2,
                improvement=improvement,
                explanation=self._explain_o2_change(current_o2, recommended_o2),
            ))

        # Excess air analysis
        if "excess_air" in recommended:
            current_ea = current.get("excess_air_percent", 20.0)
            recommended_ea = recommended["excess_air"]
            improvement = (current_ea - recommended_ea) * 0.02

            factors.append(OptimizationFactor(
                name="Excess Air",
                category="efficiency",
                weight=0.3,
                current_value=current_ea,
                optimal_value=recommended_ea,
                improvement=improvement,
                explanation=self._explain_excess_air_change(current_ea, recommended_ea),
            ))

        # Load analysis
        if "load_percent" in recommended:
            current_load = current.get("load_percent", 75.0)
            recommended_load = recommended["load_percent"]

            factors.append(OptimizationFactor(
                name="Load Point",
                category="efficiency",
                weight=0.2,
                current_value=current_load,
                optimal_value=recommended_load,
                improvement=self._calculate_load_impact(current_load, recommended_load),
                explanation=self._explain_load_change(current_load, recommended_load),
            ))

        return factors

    def _explain_o2_change(self, current: float, recommended: float) -> str:
        """Generate explanation for O2 setpoint change."""
        delta = recommended - current

        if abs(delta) < 0.1:
            return "O2 setpoint is near optimal - minimal adjustment needed."
        elif delta < 0:
            return (
                f"Reducing O2 setpoint from {current:.1f}% to {recommended:.1f}% "
                f"will decrease excess air and improve efficiency. "
                f"Ensure CO stays below limits during transition."
            )
        else:
            return (
                f"Increasing O2 setpoint from {current:.1f}% to {recommended:.1f}% "
                f"provides additional safety margin for complete combustion. "
                f"This may slightly reduce efficiency but ensures stable operation."
            )

    def _explain_excess_air_change(self, current: float, recommended: float) -> str:
        """Generate explanation for excess air change."""
        delta = recommended - current

        if abs(delta) < 1.0:
            return "Excess air is at an acceptable level."
        elif delta < 0:
            return (
                f"Reducing excess air from {current:.0f}% to {recommended:.0f}% "
                f"minimizes stack losses while maintaining combustion quality."
            )
        else:
            return (
                f"Increasing excess air from {current:.0f}% to {recommended:.0f}% "
                f"ensures complete combustion and reduces CO emissions."
            )

    def _explain_load_change(self, current: float, recommended: float) -> str:
        """Generate explanation for load change."""
        if abs(recommended - current) < 2.0:
            return "Operating near optimal load point."
        elif recommended > current:
            return (
                f"Increasing load to {recommended:.0f}% improves efficiency "
                f"by operating closer to design point."
            )
        else:
            return (
                f"Reducing load to {recommended:.0f}% matches current demand "
                f"while maintaining stable operation."
            )

    def _calculate_load_impact(self, current: float, recommended: float) -> float:
        """Calculate efficiency impact of load change."""
        # Efficiency typically peaks around 70-80% load
        optimal_load = 75.0
        current_penalty = ((current - optimal_load) / 50) ** 2
        recommended_penalty = ((recommended - optimal_load) / 50) ** 2
        return (current_penalty - recommended_penalty) * 2.0

    def _analyze_combustion_tradeoffs(
        self,
        mode: str,
        current: Dict[str, float],
        recommended: Dict[str, float],
    ) -> List[TradeoffAnalysis]:
        """Analyze optimization tradeoffs."""
        tradeoffs = []

        # Efficiency vs Emissions tradeoff
        if mode == "efficiency":
            tradeoffs.append(TradeoffAnalysis(
                tradeoff_name="Efficiency vs NOx",
                option_a="Lower O2 for higher efficiency",
                option_b="Higher O2 for lower NOx",
                chosen="Lower O2 (efficiency mode)",
                rationale="In efficiency mode, prioritizing combustion efficiency over NOx reduction.",
                impact={
                    "efficiency_gain": 0.5,
                    "nox_increase": 5.0,
                },
            ))
        elif mode == "emissions":
            tradeoffs.append(TradeoffAnalysis(
                tradeoff_name="Efficiency vs NOx",
                option_a="Lower O2 for higher efficiency",
                option_b="Higher O2 for lower NOx",
                chosen="Higher O2 (emissions mode)",
                rationale="In emissions mode, prioritizing NOx reduction over efficiency.",
                impact={
                    "efficiency_loss": 0.3,
                    "nox_reduction": 10.0,
                },
            ))

        # CO vs Efficiency tradeoff
        current_o2 = current.get("o2_percent", 3.5)
        if current_o2 < 2.5:
            tradeoffs.append(TradeoffAnalysis(
                tradeoff_name="Efficiency vs CO",
                option_a="Minimum O2 for max efficiency",
                option_b="Higher O2 for CO margin",
                chosen="Increased O2 for safety margin",
                rationale="O2 is too low - risk of CO breakthrough. Increasing O2 slightly.",
                impact={
                    "efficiency_loss": 0.2,
                    "co_margin_gain": 100.0,
                },
            ))

        return tradeoffs

    def _calculate_improvements(
        self,
        current: Dict[str, float],
        recommended: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate expected improvements."""
        improvements = {}

        # Efficiency improvement
        current_eff = current.get("efficiency_percent", 82.0)
        predicted_eff = recommended.get("efficiency_percent", current_eff + 0.5)
        improvements["efficiency_percent"] = predicted_eff - current_eff

        # Emissions reduction
        current_nox = current.get("nox_ppm", 50.0)
        predicted_nox = recommended.get("nox_ppm", current_nox * 0.95)
        improvements["nox_reduction_percent"] = (current_nox - predicted_nox) / current_nox * 100

        # Cost savings
        fuel_rate = current.get("fuel_flow_scfh", 25000)
        efficiency_gain = improvements["efficiency_percent"] / 100
        improvements["fuel_savings_scfh"] = fuel_rate * efficiency_gain

        return improvements

    def _analyze_constraints(
        self,
        constraints: Dict[str, Any],
        recommended: Dict[str, float],
    ) -> tuple:
        """Analyze which constraints are active and limiting."""
        active = []
        limiting = []

        # O2 limits
        o2_min = constraints.get("o2_min", 1.5)
        o2_max = constraints.get("o2_max", 6.0)
        recommended_o2 = recommended.get("o2_setpoint", 3.0)

        active.append(f"O2 range: {o2_min}-{o2_max}%")

        if abs(recommended_o2 - o2_min) < 0.1:
            limiting.append("O2 minimum limit is active")
        if abs(recommended_o2 - o2_max) < 0.1:
            limiting.append("O2 maximum limit is active")

        # CO limit
        co_max = constraints.get("co_max", 400)
        active.append(f"CO limit: {co_max} ppm")

        # Load limits
        load_min = constraints.get("load_min", 25)
        load_max = constraints.get("load_max", 100)
        active.append(f"Load range: {load_min}-{load_max}%")

        return active, limiting

    def _generate_summary(self, explanation: OptimizationExplanation) -> None:
        """Generate natural language summary."""
        mode = explanation.optimization_mode
        improvements = explanation.expected_improvements

        parts = [f"Optimization ({mode} mode):"]

        if "efficiency_percent" in improvements:
            eff_change = improvements["efficiency_percent"]
            if eff_change > 0:
                parts.append(f"+{eff_change:.2f}% efficiency expected.")
            elif eff_change < 0:
                parts.append(f"{eff_change:.2f}% efficiency tradeoff accepted.")

        if "nox_reduction_percent" in improvements and improvements["nox_reduction_percent"] > 0:
            parts.append(f"{improvements['nox_reduction_percent']:.1f}% NOx reduction.")

        if explanation.constraints_limiting:
            parts.append(f"Limited by: {explanation.constraints_limiting[0]}")

        explanation.summary = " ".join(parts)

    def _generate_detailed_rationale(self, explanation: OptimizationExplanation) -> None:
        """Generate detailed rationale."""
        lines = []

        lines.append(f"Optimization Mode: {explanation.optimization_mode}")
        lines.append("")

        lines.append("Key Factors:")
        for factor in explanation.factors[:3]:
            lines.append(f"- {factor.name}: {factor.explanation}")

        if explanation.tradeoffs:
            lines.append("")
            lines.append("Tradeoffs Considered:")
            for tradeoff in explanation.tradeoffs:
                lines.append(f"- {tradeoff.tradeoff_name}: {tradeoff.rationale}")

        if explanation.constraints_limiting:
            lines.append("")
            lines.append("Active Constraints:")
            for constraint in explanation.constraints_limiting:
                lines.append(f"- {constraint}")

        explanation.detailed_rationale = "\n".join(lines)

    def _calculate_confidence(self, explanation: OptimizationExplanation) -> None:
        """Calculate confidence in recommendation."""
        confidence = 0.8  # Base confidence

        # Reduce confidence if many constraints are limiting
        if len(explanation.constraints_limiting) > 2:
            confidence -= 0.1

        # Reduce confidence if recommendations are at bounds
        for factor in explanation.factors:
            if abs(factor.improvement) < 0.1:
                confidence -= 0.05

        # Add uncertainty factors
        if explanation.current_state.get("load_percent", 75) < 30:
            explanation.uncertainty_factors.append("Low load operation")
            confidence -= 0.1

        explanation.confidence = max(0.3, min(1.0, confidence))

    def get_explanation(self, explanation_id: str) -> Optional[OptimizationExplanation]:
        """Get explanation by ID."""
        return self._explanations.get(explanation_id)
