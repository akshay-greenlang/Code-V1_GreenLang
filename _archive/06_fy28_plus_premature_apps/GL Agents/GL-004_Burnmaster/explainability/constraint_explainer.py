"""
ConstraintExplainer - Explain optimization constraints for combustion systems.

This module provides explanations for optimization constraints, including
binding constraints, constraint violations, margins to limits, and
suggestions for constraint relaxation.

Example:
    >>> explainer = ConstraintExplainer(config)
    >>> binding = explainer.explain_binding_constraints(opt_result)
    >>> for constraint in binding:
    ...     print(f"{constraint.constraint_name}: {constraint.summary}")
"""

import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .explainability_payload import (
    ConstraintExplanation,
    ConstraintViolation,
    ViolationExplanation,
    MarginExplanation,
    RelaxationSuggestion,
    FeatureContribution,
    ConstraintStatus,
    ImpactDirection,
)

logger = logging.getLogger(__name__)


# Constraint type definitions for combustion optimization
CONSTRAINT_DEFINITIONS = {
    "o2_min": {
        "name": "Minimum O2 Concentration",
        "type": "inequality_lower",
        "unit": "%",
        "physical_meaning": "Minimum oxygen in flue gas for complete combustion",
        "safety_critical": True,
    },
    "o2_max": {
        "name": "Maximum O2 Concentration",
        "type": "inequality_upper",
        "unit": "%",
        "physical_meaning": "Maximum oxygen to limit stack losses",
        "safety_critical": False,
    },
    "co_max": {
        "name": "Maximum CO Emissions",
        "type": "inequality_upper",
        "unit": "ppm",
        "physical_meaning": "Carbon monoxide limit for complete combustion",
        "safety_critical": True,
    },
    "nox_max": {
        "name": "Maximum NOx Emissions",
        "type": "inequality_upper",
        "unit": "ppm",
        "physical_meaning": "Nitrogen oxide regulatory limit",
        "safety_critical": True,
    },
    "t_stack_max": {
        "name": "Maximum Stack Temperature",
        "type": "inequality_upper",
        "unit": "degC",
        "physical_meaning": "Stack temperature limit for efficiency",
        "safety_critical": False,
    },
    "load_min": {
        "name": "Minimum Load",
        "type": "inequality_lower",
        "unit": "%",
        "physical_meaning": "Minimum turndown for stable combustion",
        "safety_critical": True,
    },
    "load_max": {
        "name": "Maximum Load",
        "type": "inequality_upper",
        "unit": "%",
        "physical_meaning": "Maximum firing rate capacity",
        "safety_critical": False,
    },
    "stability_min": {
        "name": "Minimum Stability Index",
        "type": "inequality_lower",
        "unit": "index",
        "physical_meaning": "Minimum flame stability for safe operation",
        "safety_critical": True,
    },
}


class ConstraintExplainerConfig:
    """Configuration for ConstraintExplainer."""

    def __init__(
        self,
        binding_threshold: float = 0.01,
        active_threshold: float = 0.10,
        warning_margin_percent: float = 10.0,
        critical_margin_percent: float = 5.0,
        max_relaxation_percent: float = 20.0,
    ):
        """
        Initialize ConstraintExplainer configuration.

        Args:
            binding_threshold: Threshold for considering constraint binding (% of limit)
            active_threshold: Threshold for considering constraint active (% of limit)
            warning_margin_percent: Warning threshold for margin (%)
            critical_margin_percent: Critical threshold for margin (%)
            max_relaxation_percent: Maximum constraint relaxation to suggest (%)
        """
        self.binding_threshold = binding_threshold
        self.active_threshold = active_threshold
        self.warning_margin_percent = warning_margin_percent
        self.critical_margin_percent = critical_margin_percent
        self.max_relaxation_percent = max_relaxation_percent


class ConstraintExplainer:
    """
    Explainer for optimization constraints in combustion systems.

    This class explains binding constraints, constraint violations,
    margins to limits, and suggests constraint relaxations when appropriate.

    Attributes:
        config: Configuration parameters
        constraint_definitions: Definitions for known constraint types

    Example:
        >>> config = ConstraintExplainerConfig()
        >>> explainer = ConstraintExplainer(config)
        >>> explanations = explainer.explain_binding_constraints(opt_result)
    """

    def __init__(
        self,
        config: Optional[ConstraintExplainerConfig] = None,
        custom_constraints: Optional[Dict[str, Dict]] = None,
    ):
        """
        Initialize ConstraintExplainer.

        Args:
            config: Configuration parameters. Uses defaults if not provided.
            custom_constraints: Custom constraint definitions to add.
        """
        self.config = config or ConstraintExplainerConfig()
        self.constraint_definitions = CONSTRAINT_DEFINITIONS.copy()

        if custom_constraints:
            self.constraint_definitions.update(custom_constraints)

        logger.info("ConstraintExplainer initialized")

    def explain_binding_constraints(
        self,
        optimization_result: Dict[str, Any],
    ) -> List[ConstraintExplanation]:
        """
        Explain binding constraints from optimization result.

        Identifies constraints that are at their limits and affecting
        the optimal solution, providing physical interpretation.

        Args:
            optimization_result: Dictionary containing:
                - constraints: Dict of constraint values
                - limits: Dict of constraint limits
                - shadow_prices: Dict of Lagrange multipliers (optional)
                - sensitivities: Dict of sensitivities (optional)

        Returns:
            List of ConstraintExplanation for binding constraints

        Example:
            >>> opt_result = {
            ...     "constraints": {"o2_min": 2.0, "co_max": 45},
            ...     "limits": {"o2_min": 2.0, "co_max": 50}
            ... }
            >>> binding = explainer.explain_binding_constraints(opt_result)
        """
        start_time = datetime.now()
        logger.info("Explaining binding constraints")

        explanations = []
        constraints = optimization_result.get("constraints", {})
        limits = optimization_result.get("limits", {})
        shadow_prices = optimization_result.get("shadow_prices", {})
        sensitivities = optimization_result.get("sensitivities", {})

        for constraint_name, current_value in constraints.items():
            limit_value = limits.get(constraint_name)
            if limit_value is None:
                continue

            # Determine constraint status
            status, margin, margin_pct = self._determine_constraint_status(
                constraint_name, current_value, limit_value
            )

            if status in [ConstraintStatus.BINDING, ConstraintStatus.ACTIVE]:
                # Get constraint definition
                defn = self.constraint_definitions.get(
                    constraint_name,
                    {"name": constraint_name, "type": "inequality", "unit": "", "physical_meaning": ""}
                )

                # Get shadow price if available
                shadow_price = shadow_prices.get(constraint_name)
                sensitivity = sensitivities.get(constraint_name)

                # Generate explanations
                summary = self._generate_constraint_summary(
                    constraint_name, current_value, limit_value, status, defn
                )
                engineering_detail = self._generate_constraint_engineering_detail(
                    constraint_name, current_value, limit_value, status,
                    shadow_price, sensitivity, defn
                )

                explanations.append(
                    ConstraintExplanation(
                        constraint_name=defn.get("name", constraint_name),
                        constraint_type=defn.get("type", "inequality"),
                        current_value=round(current_value, 4),
                        limit_value=round(limit_value, 4),
                        status=status,
                        margin_to_limit=round(margin, 4),
                        margin_percent=round(margin_pct, 2),
                        shadow_price=round(shadow_price, 6) if shadow_price else None,
                        sensitivity=round(sensitivity, 6) if sensitivity else None,
                        unit=defn.get("unit", ""),
                        physical_meaning=defn.get("physical_meaning", ""),
                        summary=summary,
                        engineering_detail=engineering_detail,
                    )
                )

        # Sort by shadow price (most impactful first)
        explanations.sort(
            key=lambda x: abs(x.shadow_price) if x.shadow_price else 0,
            reverse=True
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Explained {len(explanations)} binding constraints in {processing_time:.1f}ms")

        return explanations

    def explain_constraint_violation(
        self,
        violation: ConstraintViolation,
    ) -> ViolationExplanation:
        """
        Explain a constraint violation in detail.

        Provides root cause analysis, contributing factors, and
        recommended actions for addressing the violation.

        Args:
            violation: ConstraintViolation details

        Returns:
            ViolationExplanation with detailed analysis

        Example:
            >>> violation = ConstraintViolation(
            ...     constraint_name="co_max",
            ...     constraint_type="inequality_upper",
            ...     current_value=55,
            ...     limit_value=50,
            ...     violation_amount=5,
            ...     violation_percent=10,
            ...     unit="ppm"
            ... )
            >>> explanation = explainer.explain_constraint_violation(violation)
        """
        start_time = datetime.now()
        logger.info(f"Explaining violation: {violation.constraint_name}")

        # Determine root cause based on constraint type
        root_cause = self._identify_root_cause(violation)

        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(violation)

        # Generate immediate actions
        immediate_actions = self._generate_immediate_actions(violation)

        # Generate long-term solutions
        long_term_solutions = self._generate_long_term_solutions(violation)

        # Assess safety implications
        defn = self.constraint_definitions.get(violation.constraint_name, {})
        safety_implications = None
        if defn.get("safety_critical", False):
            safety_implications = self._assess_safety_implications(violation)

        # Generate summary
        summary = self._generate_violation_summary(violation, root_cause)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.debug(f"Violation explanation generated in {processing_time:.1f}ms")

        return ViolationExplanation(
            violation=violation,
            root_cause=root_cause,
            contributing_factors=contributing_factors,
            immediate_actions=immediate_actions,
            long_term_solutions=long_term_solutions,
            safety_implications=safety_implications,
            summary=summary,
        )

    def explain_margin_to_limit(
        self,
        value: float,
        limit: float,
        parameter_name: str = "parameter",
        trend: Optional[float] = None,
    ) -> MarginExplanation:
        """
        Explain margin between current value and limit.

        Provides context about how close the parameter is to its limit
        and the risk level based on margin.

        Args:
            value: Current parameter value
            limit: Limit value
            parameter_name: Name of the parameter
            trend: Rate of change (units per hour), optional

        Returns:
            MarginExplanation with margin analysis

        Example:
            >>> margin = explainer.explain_margin_to_limit(
            ...     value=45, limit=50, parameter_name="NOx", trend=2.5
            ... )
            >>> print(f"Risk level: {margin.risk_level}")
        """
        logger.debug(f"Explaining margin for {parameter_name}")

        # Calculate margins
        margin_absolute = limit - value
        margin_percent = (margin_absolute / limit * 100) if limit != 0 else 0

        # Determine trend direction
        if trend is not None:
            if trend > 0.1:
                trend_direction = ImpactDirection.INCREASE
            elif trend < -0.1:
                trend_direction = ImpactDirection.DECREASE
            else:
                trend_direction = ImpactDirection.NO_CHANGE
        else:
            trend_direction = ImpactDirection.UNCERTAIN

        # Calculate time to limit
        time_to_limit = None
        if trend is not None and trend > 0 and margin_absolute > 0:
            time_to_limit = margin_absolute / trend

        # Determine risk level
        risk_level = self._determine_risk_level(margin_percent, trend_direction)

        # Generate recommendations
        recommendations = self._generate_margin_recommendations(
            parameter_name, margin_percent, risk_level, trend_direction
        )

        # Generate summary
        summary = self._generate_margin_summary(
            parameter_name, value, limit, margin_percent, risk_level, time_to_limit
        )

        return MarginExplanation(
            parameter_name=parameter_name,
            current_value=round(value, 4),
            limit_value=round(limit, 4),
            margin_absolute=round(margin_absolute, 4),
            margin_percent=round(margin_percent, 2),
            trend=trend_direction,
            time_to_limit=round(time_to_limit, 2) if time_to_limit else None,
            risk_level=risk_level,
            recommendations=recommendations,
            summary=summary,
        )

    def suggest_constraint_relaxation(
        self,
        binding: List[ConstraintExplanation],
    ) -> List[RelaxationSuggestion]:
        """
        Suggest relaxation for binding constraints.

        Analyzes binding constraints and suggests safe relaxations
        that could improve the optimization objective.

        Args:
            binding: List of binding constraint explanations

        Returns:
            List of RelaxationSuggestion for feasible relaxations

        Example:
            >>> binding = explainer.explain_binding_constraints(opt_result)
            >>> suggestions = explainer.suggest_constraint_relaxation(binding)
            >>> for s in suggestions:
            ...     print(f"{s.constraint_name}: {s.summary}")
        """
        start_time = datetime.now()
        logger.info("Generating constraint relaxation suggestions")

        suggestions = []

        for constraint in binding:
            # Skip safety-critical constraints
            defn = self.constraint_definitions.get(constraint.constraint_name, {})
            if defn.get("safety_critical", False):
                logger.debug(f"Skipping safety-critical constraint: {constraint.constraint_name}")
                continue

            # Check if relaxation would be beneficial
            if constraint.shadow_price is None or abs(constraint.shadow_price) < 0.001:
                continue

            # Calculate suggested relaxation
            suggestion = self._calculate_relaxation_suggestion(constraint)
            if suggestion:
                suggestions.append(suggestion)

        # Sort by expected benefit
        suggestions.sort(key=lambda x: x.expected_benefit, reverse=True)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Generated {len(suggestions)} relaxation suggestions in {processing_time:.1f}ms")

        return suggestions

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _determine_constraint_status(
        self,
        name: str,
        value: float,
        limit: float,
    ) -> Tuple[ConstraintStatus, float, float]:
        """Determine constraint status based on value and limit."""
        defn = self.constraint_definitions.get(name, {"type": "inequality_upper"})
        constraint_type = defn.get("type", "inequality_upper")

        if "lower" in constraint_type:
            margin = value - limit
            margin_pct = (margin / abs(limit) * 100) if limit != 0 else 0
        else:  # upper
            margin = limit - value
            margin_pct = (margin / abs(limit) * 100) if limit != 0 else 0

        # Determine status
        if margin < 0:
            status = ConstraintStatus.VIOLATED
        elif margin_pct < self.config.binding_threshold * 100:
            status = ConstraintStatus.BINDING
        elif margin_pct < self.config.active_threshold * 100:
            status = ConstraintStatus.ACTIVE
        else:
            status = ConstraintStatus.INACTIVE

        return status, margin, margin_pct

    def _generate_constraint_summary(
        self,
        name: str,
        value: float,
        limit: float,
        status: ConstraintStatus,
        defn: Dict,
    ) -> str:
        """Generate plain language summary for constraint."""
        status_text = {
            ConstraintStatus.BINDING: "is at its limit and affecting the solution",
            ConstraintStatus.ACTIVE: "is close to its limit",
            ConstraintStatus.VIOLATED: "has exceeded its limit",
            ConstraintStatus.INACTIVE: "is within safe margin",
        }

        return (
            f"{defn.get('name', name)} {status_text[status]}. "
            f"Current value: {value:.2f} {defn.get('unit', '')}, "
            f"Limit: {limit:.2f} {defn.get('unit', '')}."
        )

    def _generate_constraint_engineering_detail(
        self,
        name: str,
        value: float,
        limit: float,
        status: ConstraintStatus,
        shadow_price: Optional[float],
        sensitivity: Optional[float],
        defn: Dict,
    ) -> str:
        """Generate engineering detail for constraint."""
        detail = []
        detail.append(f"Constraint: {defn.get('name', name)}\n")
        detail.append(f"Type: {defn.get('type', 'inequality')}\n")
        detail.append(f"Current value: {value:.4f} {defn.get('unit', '')}\n")
        detail.append(f"Limit: {limit:.4f} {defn.get('unit', '')}\n")
        detail.append(f"Status: {status.value}\n")

        if shadow_price is not None:
            detail.append(f"Shadow price (Lagrange multiplier): {shadow_price:.6f}\n")
            detail.append(f"  (Impact on objective per unit relaxation)\n")

        if sensitivity is not None:
            detail.append(f"Sensitivity: {sensitivity:.6f}\n")

        detail.append(f"\nPhysical meaning: {defn.get('physical_meaning', 'N/A')}\n")

        return "".join(detail)

    def _identify_root_cause(self, violation: ConstraintViolation) -> str:
        """Identify root cause of constraint violation."""
        name = violation.constraint_name.lower()

        if "co" in name:
            return (
                "Incomplete combustion due to insufficient air or poor mixing. "
                "CO forms when carbon cannot fully oxidize to CO2."
            )
        elif "nox" in name:
            return (
                "High flame temperature or extended residence time at high temperature. "
                "Thermal NOx formation increases exponentially with temperature."
            )
        elif "o2" in name and "min" in violation.constraint_type:
            return (
                "Insufficient combustion air supply. "
                "May indicate fan limitation, damper issue, or air leaks."
            )
        elif "o2" in name and "max" in violation.constraint_type:
            return (
                "Excessive air supply leading to efficiency loss. "
                "May indicate control tuning issue or air infiltration."
            )
        elif "stability" in name:
            return (
                "Operating outside stable combustion envelope. "
                "May be due to low load, fuel quality variation, or burner issues."
            )
        else:
            return (
                f"Operating limit exceeded for {violation.constraint_name}. "
                "Review operating conditions and control parameters."
            )

    def _identify_contributing_factors(
        self,
        violation: ConstraintViolation,
    ) -> List[FeatureContribution]:
        """Identify factors contributing to violation."""
        factors = []
        name = violation.constraint_name.lower()

        if "co" in name:
            factors.append(
                FeatureContribution(
                    feature_name="O2 concentration",
                    feature_value=0.0,  # Would be actual value
                    contribution=0.5,
                    contribution_percent=50,
                    direction=ImpactDirection.DECREASE,
                    unit="%",
                    description="Low O2 limits CO oxidation to CO2",
                )
            )
            factors.append(
                FeatureContribution(
                    feature_name="Fuel-air mixing",
                    feature_value=0.0,
                    contribution=0.3,
                    contribution_percent=30,
                    direction=ImpactDirection.DECREASE,
                    unit="quality",
                    description="Poor mixing creates fuel-rich zones",
                )
            )
        elif "nox" in name:
            factors.append(
                FeatureContribution(
                    feature_name="Flame temperature",
                    feature_value=0.0,
                    contribution=0.6,
                    contribution_percent=60,
                    direction=ImpactDirection.INCREASE,
                    unit="degC",
                    description="High temperature drives thermal NOx formation",
                )
            )
            factors.append(
                FeatureContribution(
                    feature_name="Excess air",
                    feature_value=0.0,
                    contribution=0.25,
                    contribution_percent=25,
                    direction=ImpactDirection.INCREASE,
                    unit="%",
                    description="Higher O2 availability increases NOx",
                )
            )

        return factors

    def _generate_immediate_actions(
        self,
        violation: ConstraintViolation,
    ) -> List[str]:
        """Generate immediate actions for violation."""
        actions = []
        name = violation.constraint_name.lower()

        if "co" in name:
            actions.append("Increase combustion air immediately")
            actions.append("Check air damper position and operation")
            actions.append("Verify forced draft fan operation")
            actions.append("Reduce firing rate if necessary")
        elif "nox" in name:
            actions.append("Reduce firing rate to lower flame temperature")
            actions.append("Increase flue gas recirculation if available")
            actions.append("Reduce excess air (cautiously, monitor CO)")
        elif "stability" in name:
            actions.append("Increase firing rate above minimum stable load")
            actions.append("Check flame detector signals")
            actions.append("Verify fuel supply stability")

        if not actions:
            actions.append(f"Review {violation.constraint_name} operating conditions")
            actions.append("Check related control loops for proper operation")

        return actions

    def _generate_long_term_solutions(
        self,
        violation: ConstraintViolation,
    ) -> List[str]:
        """Generate long-term solutions for violation."""
        solutions = []
        name = violation.constraint_name.lower()

        if "co" in name:
            solutions.append("Tune combustion controls for optimal air-fuel ratio")
            solutions.append("Consider burner maintenance or upgrade")
            solutions.append("Implement continuous CO monitoring with feedback control")
        elif "nox" in name:
            solutions.append("Consider low-NOx burner installation")
            solutions.append("Implement or upgrade flue gas recirculation system")
            solutions.append("Review and optimize combustion staging")
        elif "stability" in name:
            solutions.append("Evaluate burner turndown capability")
            solutions.append("Consider multiple burner installation for better modulation")
            solutions.append("Implement advanced combustion control")

        if not solutions:
            solutions.append("Review constraint limits against current equipment capability")
            solutions.append("Consider process modifications to expand operating envelope")

        return solutions

    def _assess_safety_implications(
        self,
        violation: ConstraintViolation,
    ) -> str:
        """Assess safety implications of violation."""
        name = violation.constraint_name.lower()

        if "co" in name:
            return (
                "HIGH CO levels indicate incomplete combustion and potential for "
                "toxic gas accumulation. CO is poisonous and can be explosive "
                "in certain concentrations. Immediate action required."
            )
        elif "stability" in name:
            return (
                "Flame instability can lead to flame-out and potential for "
                "unburned fuel accumulation. Risk of delayed ignition and explosion. "
                "Burner safety controls should be verified."
            )
        elif "o2" in name and "min" in violation.constraint_type:
            return (
                "Low O2 indicates substoichiometric combustion. Risk of "
                "incomplete combustion, CO formation, and potential flame instability."
            )
        else:
            return (
                "Constraint violation may impact safe operation. "
                "Review safety implications specific to this parameter."
            )

    def _generate_violation_summary(
        self,
        violation: ConstraintViolation,
        root_cause: str,
    ) -> str:
        """Generate summary for violation explanation."""
        return (
            f"{violation.constraint_name} exceeded by {violation.violation_amount:.2f} "
            f"{violation.unit} ({violation.violation_percent:.1f}%). "
            f"{root_cause.split('.')[0]}."
        )

    def _determine_risk_level(
        self,
        margin_percent: float,
        trend: ImpactDirection,
    ) -> str:
        """Determine risk level based on margin and trend."""
        # Negative margin means violated
        if margin_percent < 0:
            return "critical"

        # Low margin
        if margin_percent < self.config.critical_margin_percent:
            if trend == ImpactDirection.INCREASE:
                return "critical"
            return "high"

        # Moderate margin
        if margin_percent < self.config.warning_margin_percent:
            if trend == ImpactDirection.INCREASE:
                return "high"
            return "medium"

        # Good margin
        return "low"

    def _generate_margin_recommendations(
        self,
        name: str,
        margin_pct: float,
        risk_level: str,
        trend: ImpactDirection,
    ) -> List[str]:
        """Generate recommendations based on margin analysis."""
        recommendations = []

        if risk_level == "critical":
            recommendations.append(f"Immediate action required: {name} at risk")
            recommendations.append("Reduce parameter or increase limit if safe")
        elif risk_level == "high":
            recommendations.append(f"Close monitoring required for {name}")
            if trend == ImpactDirection.INCREASE:
                recommendations.append("Take corrective action to reverse trend")
        elif risk_level == "medium":
            recommendations.append(f"Monitor {name} for changes")
            recommendations.append("Consider preventive action if trend continues")
        else:
            recommendations.append(f"{name} within safe operating margin")

        return recommendations

    def _generate_margin_summary(
        self,
        name: str,
        value: float,
        limit: float,
        margin_pct: float,
        risk_level: str,
        time_to_limit: Optional[float],
    ) -> str:
        """Generate summary for margin explanation."""
        summary = (
            f"{name} at {value:.2f} with {margin_pct:.1f}% margin to limit ({limit:.2f}). "
            f"Risk level: {risk_level}."
        )

        if time_to_limit is not None and time_to_limit > 0:
            summary += f" At current trend, limit reached in {time_to_limit:.1f} hours."

        return summary

    def _calculate_relaxation_suggestion(
        self,
        constraint: ConstraintExplanation,
    ) -> Optional[RelaxationSuggestion]:
        """Calculate relaxation suggestion for a constraint."""
        # Determine relaxation amount (limited by configuration)
        max_relaxation = abs(constraint.limit_value * self.config.max_relaxation_percent / 100)
        suggested_relaxation = min(
            abs(constraint.shadow_price) * 10 if constraint.shadow_price else max_relaxation / 2,
            max_relaxation
        )

        # Determine new limit
        if "upper" in constraint.constraint_type:
            suggested_limit = constraint.limit_value + suggested_relaxation
        else:
            suggested_limit = constraint.limit_value - suggested_relaxation

        # Calculate expected benefit
        expected_benefit = suggested_relaxation * abs(constraint.shadow_price) if constraint.shadow_price else 0

        # Assess feasibility
        feasibility = self._assess_relaxation_feasibility(constraint, suggested_relaxation)

        # Identify risks
        risks = self._identify_relaxation_risks(constraint, suggested_relaxation)

        # Determine required approvals
        approvals = self._determine_relaxation_approvals(constraint)

        # Generate summary
        summary = (
            f"Relaxing {constraint.constraint_name} from {constraint.limit_value:.2f} to "
            f"{suggested_limit:.2f} {constraint.unit} could improve efficiency by "
            f"{expected_benefit:.3f}. Feasibility: {feasibility}."
        )

        return RelaxationSuggestion(
            constraint_name=constraint.constraint_name,
            current_limit=constraint.limit_value,
            suggested_limit=round(suggested_limit, 4),
            relaxation_amount=round(suggested_relaxation, 4),
            expected_benefit=round(expected_benefit, 4),
            benefit_per_unit_relaxation=abs(constraint.shadow_price) if constraint.shadow_price else 0,
            feasibility=feasibility,
            risks=risks,
            approvals_required=approvals,
            summary=summary,
        )

    def _assess_relaxation_feasibility(
        self,
        constraint: ConstraintExplanation,
        relaxation: float,
    ) -> str:
        """Assess feasibility of constraint relaxation."""
        # Check if relaxation is within reasonable bounds
        relaxation_pct = abs(relaxation / constraint.limit_value * 100) if constraint.limit_value else 0

        if relaxation_pct > 15:
            return "not_recommended"
        elif relaxation_pct > 10:
            return "needs_review"
        else:
            return "safe"

    def _identify_relaxation_risks(
        self,
        constraint: ConstraintExplanation,
        relaxation: float,
    ) -> List[str]:
        """Identify risks associated with relaxation."""
        risks = []
        name = constraint.constraint_name.lower()

        if "o2" in name and "min" in constraint.constraint_type:
            risks.append("Risk of incomplete combustion")
            risks.append("Potential CO increase")
        elif "nox" in name or "co" in name:
            risks.append("Risk of regulatory non-compliance")
            risks.append("May require environmental permit modification")
        elif "stability" in name:
            risks.append("Risk of flame instability")
            risks.append("May affect safe operation envelope")

        if not risks:
            risks.append("Review impact on overall process performance")

        return risks

    def _determine_relaxation_approvals(
        self,
        constraint: ConstraintExplanation,
    ) -> List[str]:
        """Determine approvals required for relaxation."""
        approvals = []
        name = constraint.constraint_name.lower()

        if "nox" in name or "co" in name:
            approvals.append("Environmental compliance review")
            approvals.append("Regulatory approval may be required")
        elif "safety" in name or "stability" in name:
            approvals.append("Safety review required")
            approvals.append("Operations manager approval")

        approvals.append("Process engineering review")

        return approvals
