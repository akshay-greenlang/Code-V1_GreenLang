"""
PhysicsExplainer - Physics-based explanations for combustion optimization.

This module provides physics-based explanations for combustion processes,
including stoichiometry, efficiency changes, emission formation, and
flame stability. All explanations are grounded in combustion physics
and thermodynamics principles.

Example:
    >>> explainer = PhysicsExplainer(config)
    >>> stoich = explainer.explain_stoichiometry(lambda_val=1.15, excess_air=15.0)
    >>> print(stoich.summary)
"""

import hashlib
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import uuid

from .explainability_payload import (
    PhysicsExplanation,
    StoichiometryExplanation,
    EfficiencyExplanation,
    EmissionExplanation,
    StabilityExplanation,
    FeatureContribution,
    ImpactDirection,
    ExplanationType,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


class PhysicsExplainerConfig:
    """Configuration for PhysicsExplainer."""

    def __init__(
        self,
        stoichiometric_air_natural_gas: float = 17.2,
        stoichiometric_air_fuel_oil: float = 14.5,
        adiabatic_flame_temp_ng: float = 1960.0,
        nox_temp_threshold: float = 1500.0,
        co_o2_threshold: float = 1.5,
        stability_margin_warning: float = 15.0,
        stability_margin_critical: float = 5.0,
    ):
        """
        Initialize PhysicsExplainer configuration.

        Args:
            stoichiometric_air_natural_gas: Stoichiometric air for natural gas (kg/kg)
            stoichiometric_air_fuel_oil: Stoichiometric air for fuel oil (kg/kg)
            adiabatic_flame_temp_ng: Adiabatic flame temperature for natural gas (degC)
            nox_temp_threshold: Temperature threshold for thermal NOx formation (degC)
            co_o2_threshold: O2 threshold below which CO increases rapidly (%)
            stability_margin_warning: Warning threshold for stability margin (%)
            stability_margin_critical: Critical threshold for stability margin (%)
        """
        self.stoichiometric_air_natural_gas = stoichiometric_air_natural_gas
        self.stoichiometric_air_fuel_oil = stoichiometric_air_fuel_oil
        self.adiabatic_flame_temp_ng = adiabatic_flame_temp_ng
        self.nox_temp_threshold = nox_temp_threshold
        self.co_o2_threshold = co_o2_threshold
        self.stability_margin_warning = stability_margin_warning
        self.stability_margin_critical = stability_margin_critical


class PhysicsExplainer:
    """
    Physics-based explainer for combustion optimization.

    This class provides explanations grounded in combustion physics,
    thermodynamics, and empirical correlations. All explanations follow
    zero-hallucination principles by using deterministic calculations.

    Attributes:
        config: Configuration parameters for physics calculations

    Example:
        >>> config = PhysicsExplainerConfig()
        >>> explainer = PhysicsExplainer(config)
        >>> stoich = explainer.explain_stoichiometry(1.15, 15.0)
    """

    def __init__(self, config: Optional[PhysicsExplainerConfig] = None):
        """
        Initialize PhysicsExplainer.

        Args:
            config: Configuration parameters. Uses defaults if not provided.
        """
        self.config = config or PhysicsExplainerConfig()
        logger.info("PhysicsExplainer initialized")

    def explain_stoichiometry(
        self,
        lambda_val: float,
        excess_air: float,
        fuel_type: str = "natural_gas",
    ) -> StoichiometryExplanation:
        """
        Explain combustion stoichiometry.

        Provides a physics-based explanation of the air-fuel ratio,
        excess air percentage, and expected combustion completeness.

        Args:
            lambda_val: Air-fuel equivalence ratio (lambda)
            excess_air: Excess air percentage
            fuel_type: Type of fuel ("natural_gas" or "fuel_oil")

        Returns:
            StoichiometryExplanation with physics-based insights

        Example:
            >>> stoich = explainer.explain_stoichiometry(1.15, 15.0)
            >>> print(stoich.oxygen_percent)
            3.15
        """
        start_time = datetime.now()
        logger.info(f"Explaining stoichiometry: lambda={lambda_val}, excess_air={excess_air}%")

        # Determine stoichiometric air requirement based on fuel type
        if fuel_type == "natural_gas":
            stoich_air = self.config.stoichiometric_air_natural_gas
        else:
            stoich_air = self.config.stoichiometric_air_fuel_oil

        # Calculate actual air supplied
        actual_air = stoich_air * lambda_val

        # Calculate expected O2 in flue gas (empirical correlation)
        # O2 = 21 * (lambda - 1) / lambda (approximately)
        oxygen_percent = min(21.0 * (lambda_val - 1) / lambda_val, 21.0)
        oxygen_percent = max(0.0, oxygen_percent)

        # Estimate combustion completeness
        # Completeness decreases with insufficient air (lambda < 1)
        # and with very high excess air (flame cooling)
        if lambda_val < 1.0:
            combustion_completeness = 0.8 + 0.2 * lambda_val
        elif lambda_val > 1.5:
            combustion_completeness = 1.0 - 0.05 * (lambda_val - 1.5)
        else:
            combustion_completeness = 0.99

        combustion_completeness = max(0.7, min(1.0, combustion_completeness))

        # Generate recommendations
        recommendations = self._generate_stoich_recommendations(
            lambda_val, excess_air, oxygen_percent
        )

        # Generate plain language summary
        if lambda_val < 1.0:
            air_status = "fuel-rich (substoichiometric)"
            risk = "Risk of incomplete combustion and elevated CO."
        elif excess_air > 30:
            air_status = "lean with high excess air"
            risk = "Stack losses increase due to excess air heating."
        elif 10 <= excess_air <= 20:
            air_status = "near optimal"
            risk = "Good balance between completeness and efficiency."
        else:
            air_status = "lean"
            risk = "Monitor for stable combustion."

        summary = (
            f"Operating with {excess_air:.1f}% excess air (lambda = {lambda_val:.2f}). "
            f"Combustion is {air_status}. {risk}"
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.debug(f"Stoichiometry explanation completed in {processing_time:.1f}ms")

        return StoichiometryExplanation(
            lambda_value=lambda_val,
            excess_air_percent=excess_air,
            oxygen_percent=round(oxygen_percent, 2),
            stoichiometric_air=stoich_air,
            actual_air=round(actual_air, 2),
            combustion_completeness=round(combustion_completeness, 3),
            summary=summary,
            recommendations=recommendations,
        )

    def explain_efficiency_change(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
    ) -> EfficiencyExplanation:
        """
        Explain efficiency changes between two operating states.

        Analyzes the change in boiler efficiency and attributes it to
        specific factors based on combustion physics.

        Args:
            before: Operating state before change (efficiency, O2, T_stack, etc.)
            after: Operating state after change

        Returns:
            EfficiencyExplanation with physics-based analysis

        Example:
            >>> before = {"efficiency": 0.85, "o2_percent": 5.0, "t_stack": 200}
            >>> after = {"efficiency": 0.87, "o2_percent": 3.5, "t_stack": 185}
            >>> explanation = explainer.explain_efficiency_change(before, after)
        """
        start_time = datetime.now()
        logger.info("Explaining efficiency change")

        before_eff = before.get("efficiency", 0.85)
        after_eff = after.get("efficiency", 0.85)
        efficiency_delta = (after_eff - before_eff) * 100  # Convert to percentage points

        # Calculate loss breakdown
        loss_breakdown = self._calculate_loss_breakdown(before, after)

        # Determine primary driver
        primary_driver, contributing_factors = self._identify_efficiency_drivers(
            before, after, loss_breakdown
        )

        # Calculate fuel savings
        # Fuel savings = delta_efficiency / after_efficiency (approximately)
        fuel_savings_percent = (efficiency_delta / (after_eff * 100)) * 100 if after_eff > 0 else 0

        # Generate summaries
        direction = "increased" if efficiency_delta > 0 else "decreased"
        summary = (
            f"Boiler efficiency {direction} by {abs(efficiency_delta):.2f} percentage points "
            f"(from {before_eff*100:.1f}% to {after_eff*100:.1f}%). "
            f"Primary driver: {primary_driver}."
        )

        engineering_detail = self._generate_efficiency_engineering_detail(
            before, after, loss_breakdown, efficiency_delta
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.debug(f"Efficiency explanation completed in {processing_time:.1f}ms")

        return EfficiencyExplanation(
            before_efficiency=before_eff,
            after_efficiency=after_eff,
            efficiency_delta=round(efficiency_delta, 2),
            primary_driver=primary_driver,
            loss_breakdown=loss_breakdown,
            contributing_factors=contributing_factors,
            fuel_savings_percent=round(fuel_savings_percent, 2),
            summary=summary,
            engineering_detail=engineering_detail,
        )

    def explain_emission_formation(
        self,
        conditions: Dict[str, float],
    ) -> EmissionExplanation:
        """
        Explain emission formation mechanisms based on operating conditions.

        Provides physics-based explanation of CO and NOx formation,
        including sensitivities to key parameters.

        Args:
            conditions: Operating conditions including:
                - emission_type: "CO" or "NOx"
                - current_level_ppm: Current emission level
                - o2_percent: Flue gas oxygen percentage
                - t_flame: Flame temperature (degC)
                - regulatory_limit_ppm: Applicable limit

        Returns:
            EmissionExplanation with formation mechanism analysis

        Example:
            >>> conditions = {
            ...     "emission_type": "NOx",
            ...     "current_level_ppm": 45,
            ...     "o2_percent": 3.0,
            ...     "t_flame": 1700,
            ...     "regulatory_limit_ppm": 50
            ... }
            >>> explanation = explainer.explain_emission_formation(conditions)
        """
        start_time = datetime.now()
        emission_type = conditions.get("emission_type", "NOx")
        logger.info(f"Explaining {emission_type} formation")

        current_level = conditions.get("current_level_ppm", 0)
        o2_percent = conditions.get("o2_percent", 3.0)
        t_flame = conditions.get("t_flame", 1500)
        regulatory_limit = conditions.get("regulatory_limit_ppm", 100)

        # Calculate margin to limit
        margin_to_limit = ((regulatory_limit - current_level) / regulatory_limit) * 100

        # Determine formation mechanism and sensitivities
        if emission_type.upper() == "NOX":
            formation_mechanism, temp_sens, o2_sens = self._analyze_nox_formation(
                t_flame, o2_percent
            )
            key_drivers = self._get_nox_drivers(t_flame, o2_percent, current_level)
            reduction_strategies = self._get_nox_reduction_strategies(t_flame, o2_percent)
        else:  # CO
            formation_mechanism, temp_sens, o2_sens = self._analyze_co_formation(
                t_flame, o2_percent
            )
            key_drivers = self._get_co_drivers(t_flame, o2_percent, current_level)
            reduction_strategies = self._get_co_reduction_strategies(o2_percent)

        # Generate summaries
        summary = self._generate_emission_summary(
            emission_type, current_level, regulatory_limit, margin_to_limit, formation_mechanism
        )
        engineering_detail = self._generate_emission_engineering_detail(
            emission_type, formation_mechanism, t_flame, o2_percent, temp_sens, o2_sens
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.debug(f"Emission explanation completed in {processing_time:.1f}ms")

        return EmissionExplanation(
            emission_type=emission_type,
            formation_mechanism=formation_mechanism,
            current_level_ppm=current_level,
            regulatory_limit_ppm=regulatory_limit,
            margin_to_limit_percent=round(margin_to_limit, 1),
            temperature_sensitivity=round(temp_sens, 3),
            o2_sensitivity=round(o2_sens, 2),
            key_drivers=key_drivers,
            reduction_strategies=reduction_strategies,
            summary=summary,
            engineering_detail=engineering_detail,
        )

    def explain_stability_risk(
        self,
        factors: Dict[str, float],
    ) -> StabilityExplanation:
        """
        Explain combustion stability risks.

        Analyzes flame stability based on operating conditions and
        provides risk assessments for common instability modes.

        Args:
            factors: Stability factors including:
                - air_fuel_ratio: Current air-fuel ratio
                - firing_rate_percent: Current firing rate (% of max)
                - flame_signal: Flame detector signal strength
                - pressure_fluctuation: Combustion pressure fluctuation (%)
                - swirl_number: Burner swirl number

        Returns:
            StabilityExplanation with risk analysis

        Example:
            >>> factors = {
            ...     "air_fuel_ratio": 1.05,
            ...     "firing_rate_percent": 30,
            ...     "flame_signal": 0.7,
            ...     "pressure_fluctuation": 5.0
            ... }
            >>> stability = explainer.explain_stability_risk(factors)
        """
        start_time = datetime.now()
        logger.info("Explaining stability risk")

        # Extract factors
        air_fuel_ratio = factors.get("air_fuel_ratio", 1.15)
        firing_rate = factors.get("firing_rate_percent", 50)
        flame_signal = factors.get("flame_signal", 1.0)
        pressure_fluct = factors.get("pressure_fluctuation", 2.0)
        swirl = factors.get("swirl_number", 0.6)

        # Calculate individual risks
        pulsation_risk = self._calculate_pulsation_risk(pressure_fluct, firing_rate)
        flashback_risk = self._calculate_flashback_risk(air_fuel_ratio, firing_rate)
        blowout_risk = self._calculate_blowout_risk(air_fuel_ratio, firing_rate, flame_signal)

        # Calculate overall stability index (0 = unstable, 1 = stable)
        stability_index = 1.0 - max(pulsation_risk, flashback_risk, blowout_risk)
        stability_index = max(0.0, min(1.0, stability_index))

        # Calculate flame stability margin
        flame_stability_margin = self._calculate_flame_margin(
            air_fuel_ratio, firing_rate, flame_signal
        )

        # Get risk factors
        risk_factors = self._get_stability_risk_factors(
            air_fuel_ratio, firing_rate, flame_signal, pressure_fluct
        )

        # Calculate safe operating envelope
        safe_envelope = self._calculate_safe_envelope(air_fuel_ratio, firing_rate)

        # Generate warnings
        warnings = self._generate_stability_warnings(
            stability_index, flame_stability_margin, pulsation_risk, flashback_risk, blowout_risk
        )

        # Generate summaries
        summary = self._generate_stability_summary(stability_index, flame_stability_margin, warnings)
        engineering_detail = self._generate_stability_engineering_detail(
            air_fuel_ratio, firing_rate, pulsation_risk, flashback_risk, blowout_risk
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.debug(f"Stability explanation completed in {processing_time:.1f}ms")

        return StabilityExplanation(
            stability_index=round(stability_index, 3),
            flame_stability_margin=round(flame_stability_margin, 1),
            pulsation_risk=round(pulsation_risk, 3),
            flashback_risk=round(flashback_risk, 3),
            blowout_risk=round(blowout_risk, 3),
            risk_factors=risk_factors,
            safe_operating_envelope=safe_envelope,
            summary=summary,
            engineering_detail=engineering_detail,
            warnings=warnings,
        )

    def generate_engineering_narrative(
        self,
        context: Dict[str, Any],
    ) -> str:
        """
        Generate a comprehensive engineering narrative.

        Creates a detailed narrative combining all physics-based
        explanations into a coherent technical story.

        Args:
            context: Context containing operating data and explanations

        Returns:
            Engineering narrative string

        Example:
            >>> context = {
            ...     "stoichiometry": stoich_explanation,
            ...     "efficiency": eff_explanation,
            ...     "emissions": [nox_explanation, co_explanation],
            ...     "stability": stability_explanation
            ... }
            >>> narrative = explainer.generate_engineering_narrative(context)
        """
        logger.info("Generating engineering narrative")

        sections = []

        # Overview
        sections.append("## Combustion Analysis Summary\n")

        # Stoichiometry section
        if "stoichiometry" in context:
            stoich = context["stoichiometry"]
            sections.append(
                f"### Air-Fuel Ratio Analysis\n"
                f"The burner is operating at lambda = {stoich.lambda_value:.2f} "
                f"({stoich.excess_air_percent:.1f}% excess air). "
                f"Expected flue gas O2 is {stoich.oxygen_percent:.1f}%. "
                f"Combustion completeness is estimated at {stoich.combustion_completeness*100:.1f}%.\n"
            )

        # Efficiency section
        if "efficiency" in context:
            eff = context["efficiency"]
            direction = "improvement" if eff.efficiency_delta > 0 else "reduction"
            sections.append(
                f"### Efficiency Analysis\n"
                f"Efficiency {direction} of {abs(eff.efficiency_delta):.2f} percentage points observed. "
                f"Primary driver: {eff.primary_driver}. "
                f"This corresponds to approximately {abs(eff.fuel_savings_percent):.1f}% fuel savings.\n"
            )
            if eff.loss_breakdown:
                sections.append("Loss breakdown:\n")
                for loss_type, value in eff.loss_breakdown.items():
                    sections.append(f"  - {loss_type}: {value:.2f}%\n")

        # Emissions section
        if "emissions" in context:
            sections.append("### Emissions Analysis\n")
            for emission in context["emissions"]:
                margin_status = "within" if emission.margin_to_limit_percent > 10 else "approaching"
                sections.append(
                    f"**{emission.emission_type}**: {emission.current_level_ppm:.0f} ppm "
                    f"({margin_status} limit of {emission.regulatory_limit_ppm:.0f} ppm). "
                    f"Formation mechanism: {emission.formation_mechanism}.\n"
                )

        # Stability section
        if "stability" in context:
            stab = context["stability"]
            stability_status = (
                "stable" if stab.stability_index > 0.8
                else "marginal" if stab.stability_index > 0.5
                else "at risk"
            )
            sections.append(
                f"### Stability Analysis\n"
                f"Combustion is {stability_status} (stability index: {stab.stability_index:.2f}). "
                f"Flame stability margin: {stab.flame_stability_margin:.1f}%.\n"
            )
            if stab.warnings:
                sections.append("Active warnings:\n")
                for warning in stab.warnings:
                    sections.append(f"  - {warning}\n")

        return "".join(sections)

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _generate_stoich_recommendations(
        self,
        lambda_val: float,
        excess_air: float,
        o2_percent: float,
    ) -> List[str]:
        """Generate recommendations based on stoichiometry analysis."""
        recommendations = []

        if lambda_val < 1.0:
            recommendations.append("Increase air supply to achieve complete combustion")
            recommendations.append("Check for air damper restrictions or fan issues")
        elif excess_air > 25:
            recommendations.append("Reduce excess air to improve efficiency")
            recommendations.append(f"Target O2 setpoint of 2.5-3.5% (currently {o2_percent:.1f}%)")
        elif excess_air < 10:
            recommendations.append("Monitor CO levels closely at low excess air")
            recommendations.append("Consider increasing excess air margin for stability")
        else:
            recommendations.append("Maintain current air-fuel ratio")
            recommendations.append("Continue monitoring for optimal performance")

        return recommendations

    def _calculate_loss_breakdown(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate efficiency loss breakdown."""
        losses = {}

        # Dry flue gas loss (empirical: ~0.5% per 1% excess O2)
        o2_before = before.get("o2_percent", 4.0)
        o2_after = after.get("o2_percent", 4.0)
        losses["dry_flue_gas_change"] = round((o2_before - o2_after) * 0.5, 2)

        # Stack loss (empirical: ~0.05% per degC stack temperature)
        t_stack_before = before.get("t_stack", 200)
        t_stack_after = after.get("t_stack", 200)
        losses["stack_loss_change"] = round((t_stack_before - t_stack_after) * 0.05, 2)

        # Moisture loss (relatively constant, small change)
        losses["moisture_loss_change"] = round(
            (before.get("moisture_loss", 5.0) - after.get("moisture_loss", 5.0)), 2
        )

        # Radiation loss (relatively constant)
        losses["radiation_loss_change"] = round(
            (before.get("radiation_loss", 0.5) - after.get("radiation_loss", 0.5)), 2
        )

        return losses

    def _identify_efficiency_drivers(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
        loss_breakdown: Dict[str, float],
    ) -> Tuple[str, List[FeatureContribution]]:
        """Identify primary driver and contributing factors for efficiency change."""
        contributions = []

        # Find the largest contributor
        max_contribution = 0
        primary_driver = "Operating conditions"

        for loss_type, change in loss_breakdown.items():
            abs_change = abs(change)
            if abs_change > max_contribution:
                max_contribution = abs_change
                primary_driver = loss_type.replace("_change", "").replace("_", " ").title()

            if abs_change > 0.01:
                direction = ImpactDirection.DECREASE if change > 0 else ImpactDirection.INCREASE
                contributions.append(
                    FeatureContribution(
                        feature_name=loss_type,
                        feature_value=change,
                        contribution=change,
                        contribution_percent=abs_change * 10,  # Approximate
                        direction=direction,
                        unit="%",
                        description=f"{loss_type.replace('_', ' ')} changed by {change:.2f}%",
                    )
                )

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)

        return primary_driver, contributions

    def _generate_efficiency_engineering_detail(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
        loss_breakdown: Dict[str, float],
        efficiency_delta: float,
    ) -> str:
        """Generate engineering detail for efficiency explanation."""
        detail = []
        detail.append(f"Efficiency change: {efficiency_delta:+.2f} percentage points\n")
        detail.append(f"Before state: eta={before.get('efficiency', 0)*100:.1f}%, ")
        detail.append(f"O2={before.get('o2_percent', 0):.1f}%, ")
        detail.append(f"T_stack={before.get('t_stack', 0):.0f}C\n")
        detail.append(f"After state: eta={after.get('efficiency', 0)*100:.1f}%, ")
        detail.append(f"O2={after.get('o2_percent', 0):.1f}%, ")
        detail.append(f"T_stack={after.get('t_stack', 0):.0f}C\n")
        detail.append("Loss analysis (indirect method):\n")
        for loss_type, change in loss_breakdown.items():
            detail.append(f"  {loss_type}: {change:+.2f}%\n")

        return "".join(detail)

    def _analyze_nox_formation(
        self,
        t_flame: float,
        o2_percent: float,
    ) -> Tuple[str, float, float]:
        """Analyze NOx formation mechanism."""
        # Thermal NOx dominates above 1500C
        if t_flame > self.config.nox_temp_threshold:
            mechanism = (
                "Thermal NOx (Zeldovich mechanism) - dominant at high flame temperatures. "
                "NOx formation increases exponentially with temperature above 1500C."
            )
            # Temperature sensitivity: approximately doubles every 90C above 1500C
            temp_sensitivity = 0.5 * math.exp((t_flame - 1500) / 200)
        else:
            mechanism = (
                "Prompt NOx (Fenimore mechanism) - dominant at lower temperatures. "
                "Formed in fuel-rich flame zones through hydrocarbon radical reactions."
            )
            temp_sensitivity = 0.1

        # O2 sensitivity: higher O2 increases NOx
        o2_sensitivity = 5.0 if o2_percent > 3.0 else 2.0

        return mechanism, temp_sensitivity, o2_sensitivity

    def _analyze_co_formation(
        self,
        t_flame: float,
        o2_percent: float,
    ) -> Tuple[str, float, float]:
        """Analyze CO formation mechanism."""
        if o2_percent < self.config.co_o2_threshold:
            mechanism = (
                "Incomplete combustion due to insufficient oxygen. "
                "CO is an intermediate product that cannot oxidize to CO2 without adequate O2."
            )
            o2_sensitivity = -50.0  # CO increases rapidly as O2 decreases
        elif t_flame < 800:
            mechanism = (
                "Quenching of CO oxidation due to low temperature. "
                "CO to CO2 conversion requires sufficient residence time at high temperature."
            )
            o2_sensitivity = -10.0
        else:
            mechanism = (
                "Normal CO formation as combustion intermediate. "
                "CO levels are low with adequate O2 and temperature."
            )
            o2_sensitivity = -5.0

        # Temperature sensitivity
        temp_sensitivity = -0.1 if t_flame > 1000 else 0.2

        return mechanism, temp_sensitivity, o2_sensitivity

    def _get_nox_drivers(
        self,
        t_flame: float,
        o2_percent: float,
        current_level: float,
    ) -> List[FeatureContribution]:
        """Get key drivers for NOx formation."""
        drivers = []

        # Temperature driver
        temp_contribution = 0.6 if t_flame > 1500 else 0.3
        drivers.append(
            FeatureContribution(
                feature_name="flame_temperature",
                feature_value=t_flame,
                contribution=temp_contribution * current_level,
                contribution_percent=temp_contribution * 100,
                direction=ImpactDirection.INCREASE,
                unit="degC",
                description=f"Flame temperature of {t_flame:.0f}C drives thermal NOx formation",
            )
        )

        # O2 driver
        o2_contribution = 0.3 if o2_percent > 3.0 else 0.15
        drivers.append(
            FeatureContribution(
                feature_name="oxygen_concentration",
                feature_value=o2_percent,
                contribution=o2_contribution * current_level,
                contribution_percent=o2_contribution * 100,
                direction=ImpactDirection.INCREASE,
                unit="%",
                description=f"Oxygen at {o2_percent:.1f}% provides oxidizer for NOx formation",
            )
        )

        return drivers

    def _get_co_drivers(
        self,
        t_flame: float,
        o2_percent: float,
        current_level: float,
    ) -> List[FeatureContribution]:
        """Get key drivers for CO formation."""
        drivers = []

        # O2 driver (inverse relationship)
        o2_contribution = 0.7 if o2_percent < 2.0 else 0.3
        drivers.append(
            FeatureContribution(
                feature_name="oxygen_concentration",
                feature_value=o2_percent,
                contribution=o2_contribution * current_level,
                contribution_percent=o2_contribution * 100,
                direction=ImpactDirection.DECREASE if o2_percent > 2.0 else ImpactDirection.INCREASE,
                unit="%",
                description=f"Oxygen at {o2_percent:.1f}% - {'adequate' if o2_percent > 2.0 else 'low'} for complete combustion",
            )
        )

        return drivers

    def _get_nox_reduction_strategies(
        self,
        t_flame: float,
        o2_percent: float,
    ) -> List[str]:
        """Get NOx reduction strategies."""
        strategies = []

        if t_flame > 1600:
            strategies.append("Reduce flame temperature through staged combustion or FGR")
            strategies.append("Consider low-NOx burner technology")
        if o2_percent > 3.5:
            strategies.append("Reduce excess air to lower O2 (target 2.5-3.0%)")
        strategies.append("Optimize fuel-air mixing to reduce peak temperatures")
        strategies.append("Consider flue gas recirculation (FGR) for dilution")

        return strategies

    def _get_co_reduction_strategies(
        self,
        o2_percent: float,
    ) -> List[str]:
        """Get CO reduction strategies."""
        strategies = []

        if o2_percent < 2.0:
            strategies.append("Increase combustion air to raise O2 above 2%")
            strategies.append("Check for air leaks in fuel supply or damper issues")
        strategies.append("Ensure proper fuel-air mixing in burner")
        strategies.append("Verify adequate residence time in combustion zone")
        strategies.append("Check for cold spots or flame impingement")

        return strategies

    def _generate_emission_summary(
        self,
        emission_type: str,
        current_level: float,
        regulatory_limit: float,
        margin: float,
        mechanism: str,
    ) -> str:
        """Generate emission summary."""
        status = "within" if margin > 10 else "approaching"
        return (
            f"{emission_type} at {current_level:.0f} ppm ({status} regulatory limit of "
            f"{regulatory_limit:.0f} ppm with {margin:.0f}% margin). "
            f"{mechanism.split('.')[0]}."
        )

    def _generate_emission_engineering_detail(
        self,
        emission_type: str,
        mechanism: str,
        t_flame: float,
        o2_percent: float,
        temp_sens: float,
        o2_sens: float,
    ) -> str:
        """Generate engineering detail for emission explanation."""
        return (
            f"{emission_type} Formation Analysis:\n"
            f"Mechanism: {mechanism}\n"
            f"Operating conditions: T_flame = {t_flame:.0f}C, O2 = {o2_percent:.1f}%\n"
            f"Sensitivities:\n"
            f"  - Temperature: {temp_sens:.3f} ppm/degC\n"
            f"  - Oxygen: {o2_sens:.2f} ppm/%O2"
        )

    def _calculate_pulsation_risk(
        self,
        pressure_fluct: float,
        firing_rate: float,
    ) -> float:
        """Calculate combustion pulsation risk."""
        # Risk increases with pressure fluctuation and low firing rates
        base_risk = min(pressure_fluct / 10.0, 1.0)
        low_fire_factor = max(0, (50 - firing_rate) / 50) * 0.3
        return min(base_risk + low_fire_factor, 1.0)

    def _calculate_flashback_risk(
        self,
        air_fuel_ratio: float,
        firing_rate: float,
    ) -> float:
        """Calculate flame flashback risk."""
        # Flashback risk increases with fuel-rich mixtures and high firing rates
        if air_fuel_ratio < 1.0:
            base_risk = (1.0 - air_fuel_ratio) * 2.0
        else:
            base_risk = 0.05
        high_fire_factor = max(0, (firing_rate - 80) / 20) * 0.3
        return min(base_risk + high_fire_factor, 1.0)

    def _calculate_blowout_risk(
        self,
        air_fuel_ratio: float,
        firing_rate: float,
        flame_signal: float,
    ) -> float:
        """Calculate flame blowout risk."""
        # Blowout risk increases with lean mixtures, low firing, and weak flame
        if air_fuel_ratio > 1.5:
            lean_risk = (air_fuel_ratio - 1.5) * 0.5
        else:
            lean_risk = 0.0
        low_fire_risk = max(0, (30 - firing_rate) / 30) * 0.4
        weak_flame_risk = max(0, (0.5 - flame_signal)) * 0.5 if flame_signal < 0.5 else 0
        return min(lean_risk + low_fire_risk + weak_flame_risk, 1.0)

    def _calculate_flame_margin(
        self,
        air_fuel_ratio: float,
        firing_rate: float,
        flame_signal: float,
    ) -> float:
        """Calculate flame stability margin percentage."""
        # Base margin from operating point
        base_margin = 50.0

        # Adjust for air-fuel ratio
        if air_fuel_ratio < 1.0:
            base_margin -= (1.0 - air_fuel_ratio) * 100
        elif air_fuel_ratio > 1.3:
            base_margin -= (air_fuel_ratio - 1.3) * 50

        # Adjust for firing rate
        if firing_rate < 30:
            base_margin -= (30 - firing_rate)

        # Adjust for flame signal
        if flame_signal < 0.8:
            base_margin -= (0.8 - flame_signal) * 50

        return max(0, min(100, base_margin))

    def _get_stability_risk_factors(
        self,
        air_fuel_ratio: float,
        firing_rate: float,
        flame_signal: float,
        pressure_fluct: float,
    ) -> List[FeatureContribution]:
        """Get stability risk factors."""
        factors = []

        # Air-fuel ratio factor
        afr_direction = ImpactDirection.INCREASE if air_fuel_ratio < 1.0 or air_fuel_ratio > 1.4 else ImpactDirection.NO_CHANGE
        factors.append(
            FeatureContribution(
                feature_name="air_fuel_ratio",
                feature_value=air_fuel_ratio,
                contribution=abs(air_fuel_ratio - 1.15) * 0.5,
                contribution_percent=abs(air_fuel_ratio - 1.15) * 50,
                direction=afr_direction,
                unit="ratio",
                description=f"Air-fuel ratio at {air_fuel_ratio:.2f} (optimal ~1.15)",
            )
        )

        # Firing rate factor
        if firing_rate < 30:
            factors.append(
                FeatureContribution(
                    feature_name="firing_rate",
                    feature_value=firing_rate,
                    contribution=0.3,
                    contribution_percent=30,
                    direction=ImpactDirection.INCREASE,
                    unit="%",
                    description=f"Low firing rate ({firing_rate:.0f}%) increases instability risk",
                )
            )

        # Pressure fluctuation factor
        if pressure_fluct > 3:
            factors.append(
                FeatureContribution(
                    feature_name="pressure_fluctuation",
                    feature_value=pressure_fluct,
                    contribution=pressure_fluct / 10,
                    contribution_percent=pressure_fluct * 10,
                    direction=ImpactDirection.INCREASE,
                    unit="%",
                    description=f"Pressure fluctuation of {pressure_fluct:.1f}% indicates pulsation",
                )
            )

        return factors

    def _calculate_safe_envelope(
        self,
        air_fuel_ratio: float,
        firing_rate: float,
    ) -> Dict[str, Dict[str, float]]:
        """Calculate safe operating envelope."""
        return {
            "air_fuel_ratio": {"min": 1.02, "max": 1.40, "optimal": 1.15},
            "firing_rate": {"min": 20.0, "max": 100.0, "optimal": 70.0},
            "o2_percent": {"min": 1.5, "max": 6.0, "optimal": 3.0},
            "flame_signal": {"min": 0.5, "max": 1.0, "optimal": 0.9},
        }

    def _generate_stability_warnings(
        self,
        stability_index: float,
        flame_margin: float,
        pulsation_risk: float,
        flashback_risk: float,
        blowout_risk: float,
    ) -> List[str]:
        """Generate stability warnings."""
        warnings = []

        if stability_index < 0.5:
            warnings.append("CRITICAL: Combustion stability is compromised")
        elif stability_index < 0.7:
            warnings.append("WARNING: Combustion stability is marginal")

        if flame_margin < self.config.stability_margin_critical:
            warnings.append("CRITICAL: Flame stability margin below safe threshold")
        elif flame_margin < self.config.stability_margin_warning:
            warnings.append("WARNING: Flame stability margin approaching limit")

        if pulsation_risk > 0.5:
            warnings.append("WARNING: Elevated combustion pulsation risk")
        if flashback_risk > 0.3:
            warnings.append("WARNING: Elevated flame flashback risk")
        if blowout_risk > 0.3:
            warnings.append("WARNING: Elevated flame blowout risk")

        return warnings

    def _generate_stability_summary(
        self,
        stability_index: float,
        flame_margin: float,
        warnings: List[str],
    ) -> str:
        """Generate stability summary."""
        if stability_index > 0.8:
            status = "stable with good margins"
        elif stability_index > 0.5:
            status = "marginally stable - monitoring recommended"
        else:
            status = "at risk - corrective action needed"

        warning_text = f" Active warnings: {len(warnings)}" if warnings else ""

        return (
            f"Combustion is {status}. "
            f"Stability index: {stability_index:.2f}, flame margin: {flame_margin:.0f}%.{warning_text}"
        )

    def _generate_stability_engineering_detail(
        self,
        air_fuel_ratio: float,
        firing_rate: float,
        pulsation_risk: float,
        flashback_risk: float,
        blowout_risk: float,
    ) -> str:
        """Generate engineering detail for stability explanation."""
        return (
            f"Stability Analysis:\n"
            f"Operating point: AFR = {air_fuel_ratio:.2f}, Firing = {firing_rate:.0f}%\n"
            f"Risk assessment:\n"
            f"  - Pulsation: {pulsation_risk*100:.0f}%\n"
            f"  - Flashback: {flashback_risk*100:.0f}%\n"
            f"  - Blowout: {blowout_risk*100:.0f}%\n"
            f"Stability limits based on burner design and fuel characteristics."
        )

    def _calculate_provenance_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _generate_explanation_id(self) -> str:
        """Generate unique explanation ID."""
        return f"physics-{uuid.uuid4().hex[:12]}"
