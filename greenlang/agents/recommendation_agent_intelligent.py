# -*- coding: utf-8 -*-
"""
Intelligent Recommendation Agent - AI-Native Building Optimization Advisor

This is the RETROFITTED version of RecommendationAgent with full LLM intelligence.
It demonstrates the pattern for upgrading any existing agent to be AI-native.

BEFORE (recommendation_agent.py):
    - Generates recommendations from static database
    - Returns rule-based suggestions
    - No contextual explanations or personalization

AFTER (this file):
    - Same deterministic recommendation matching (zero-hallucination)
    - AI-generated explanations tailored to building context
    - AI-powered prioritization based on specific circumstances
    - Anomaly detection for unusual building metrics
    - Regulatory context awareness (CSRD, EU Taxonomy, IRA incentives)

Migration Pattern:
    The original RecommendationAgent rule-based logic is UNCHANGED.
    Intelligence is ADDED on top via IntelligentAgentBase.

Usage:
    from greenlang.agents.recommendation_agent_intelligent import IntelligentRecommendationAgent

    agent = IntelligentRecommendationAgent()
    result = agent.run({
        "emissions_by_source": {"electricity": 5000, "natural_gas": 3000},
        "building_type": "commercial_office",
        "building_age": 25,
        "performance_rating": "Below Average"
    })

    # Result now includes AI-generated content:
    print(result.data["explanation"])  # "Based on your building's profile..."
    print(result.data["ai_prioritized_recommendations"])  # [{...}]

Author: GreenLang Intelligence Framework
Date: December 2025
Status: Production Ready
"""

from __future__ import annotations
import logging
import re
from typing import Any, Dict, List, Optional

from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult
from greenlang.agents.intelligent_base import (
    IntelligentAgentBase,
    IntelligentAgentConfig,
    IntelligenceLevel,
    Recommendation,
    Anomaly,
)
from greenlang.agents.intelligence_interface import (
    IntelligenceCapabilities,
    require_intelligence,
)
from greenlang.determinism import DeterministicClock
from greenlang.templates.agent_monitoring import OperationalMonitoringMixin

logger = logging.getLogger(__name__)


class IntelligentRecommendationAgentConfig(IntelligentAgentConfig):
    """Configuration for Intelligent Recommendation Agent."""

    class Config:
        extra = "allow"


@require_intelligence
class IntelligentRecommendationAgent(OperationalMonitoringMixin, IntelligentAgentBase):
    """
    AI-Native Building Optimization Recommendation Agent.

    This agent provides optimization recommendations with FULL LLM INTELLIGENCE:

    1. DETERMINISTIC OPERATIONS (Zero-Hallucination):
       - Rule-based recommendation matching
       - Priority calculation
       - Payback period estimation
       - Savings potential calculation

    2. AI-POWERED INTELLIGENCE:
       - Contextual explanations tailored to building profile
       - AI-powered prioritization based on circumstances
       - Investment roadmap with justifications
       - Regulatory incentive awareness
       - Anomaly detection for unusual metrics

    Intelligence Level: ADVANCED (explanations + recommendations + reasoning)
    """

    def __init__(self, config: Optional[IntelligentRecommendationAgentConfig] = None):
        """Initialize the intelligent recommendation agent."""
        if config is None:
            config = IntelligentRecommendationAgentConfig(
                name="IntelligentRecommendationAgent",
                description="AI-native building optimization advisor with LLM intelligence",
                intelligence_level=IntelligenceLevel.ADVANCED,
                enable_explanations=True,
                enable_recommendations=True,
                enable_anomaly_detection=True,
                regulatory_context="CSRD ESRS E1, EU Taxonomy, IRA Tax Credits, ENERGY STAR",
                domain_context="building energy efficiency and decarbonization"
            )

        super().__init__(config)
        self.setup_monitoring(agent_name="intelligent_recommendation_agent")

        # Load recommendations database (same as original)
        self.recommendations_db = self._load_recommendations()

        logger.info(
            f"Initialized IntelligentRecommendationAgent with intelligence level: "
            f"{config.intelligence_level.value}"
        )

    def _load_recommendations(self) -> Dict:
        """Load recommendations database."""
        return {
            "hvac": {
                "high_consumption": [
                    {
                        "action": "Upgrade to high-efficiency HVAC system",
                        "impact": "20-30% reduction in HVAC energy",
                        "cost": "High",
                        "payback": "5-7 years",
                        "priority": "High",
                    },
                    {
                        "action": "Install smart thermostats and zone controls",
                        "impact": "10-15% reduction in HVAC energy",
                        "cost": "Low",
                        "payback": "1-2 years",
                        "priority": "High",
                    },
                    {
                        "action": "Implement demand-controlled ventilation",
                        "impact": "5-10% reduction in HVAC energy",
                        "cost": "Medium",
                        "payback": "3-4 years",
                        "priority": "Medium",
                    },
                ],
                "poor_efficiency": [
                    {
                        "action": "Replace old HVAC units (>15 years)",
                        "impact": "25-35% reduction in HVAC energy",
                        "cost": "High",
                        "payback": "4-6 years",
                        "priority": "High",
                    },
                    {
                        "action": "Regular maintenance and filter replacement",
                        "impact": "5-10% improvement in efficiency",
                        "cost": "Low",
                        "payback": "Immediate",
                        "priority": "High",
                    },
                ],
            },
            "lighting": {
                "high_consumption": [
                    {
                        "action": "Convert to LED lighting",
                        "impact": "50-70% reduction in lighting energy",
                        "cost": "Medium",
                        "payback": "2-3 years",
                        "priority": "High",
                    },
                    {
                        "action": "Install occupancy sensors",
                        "impact": "20-30% reduction in lighting energy",
                        "cost": "Low",
                        "payback": "1-2 years",
                        "priority": "High",
                    },
                    {
                        "action": "Implement daylight harvesting",
                        "impact": "15-25% reduction in lighting energy",
                        "cost": "Medium",
                        "payback": "3-4 years",
                        "priority": "Medium",
                    },
                ]
            },
            "envelope": {
                "poor_insulation": [
                    {
                        "action": "Add or upgrade wall insulation",
                        "impact": "10-20% reduction in heating/cooling",
                        "cost": "High",
                        "payback": "7-10 years",
                        "priority": "Medium",
                    },
                    {
                        "action": "Upgrade to double/triple glazed windows",
                        "impact": "5-15% reduction in HVAC load",
                        "cost": "High",
                        "payback": "10-15 years",
                        "priority": "Low",
                    },
                    {
                        "action": "Seal air leaks and improve weatherstripping",
                        "impact": "5-10% reduction in HVAC load",
                        "cost": "Low",
                        "payback": "1 year",
                        "priority": "High",
                    },
                ]
            },
            "renewable": {
                "high_electricity": [
                    {
                        "action": "Install rooftop solar PV system",
                        "impact": "30-70% reduction in grid electricity",
                        "cost": "High",
                        "payback": "5-8 years",
                        "priority": "High",
                    },
                    {
                        "action": "Purchase renewable energy certificates (RECs)",
                        "impact": "100% carbon neutral electricity",
                        "cost": "Low",
                        "payback": "N/A",
                        "priority": "Medium",
                    },
                    {
                        "action": "Switch to green power purchase agreement",
                        "impact": "50-100% renewable electricity",
                        "cost": "Medium",
                        "payback": "Immediate",
                        "priority": "High",
                    },
                ]
            },
            "operations": {
                "general": [
                    {
                        "action": "Implement energy management system (EMS)",
                        "impact": "10-20% overall energy reduction",
                        "cost": "Medium",
                        "payback": "2-3 years",
                        "priority": "High",
                    },
                    {
                        "action": "Conduct regular energy audits",
                        "impact": "Identifies 15-30% savings opportunities",
                        "cost": "Low",
                        "payback": "Immediate",
                        "priority": "High",
                    },
                    {
                        "action": "Train staff on energy conservation",
                        "impact": "5-10% reduction through behavior change",
                        "cost": "Low",
                        "payback": "Immediate",
                        "priority": "Medium",
                    },
                ]
            },
        }

    # =========================================================================
    # MANDATORY: Intelligence Interface Implementation
    # =========================================================================

    def get_intelligence_level(self) -> IntelligenceLevel:
        """Return the agent's intelligence level."""
        return self.intelligent_config.intelligence_level

    def get_intelligence_capabilities(self) -> IntelligenceCapabilities:
        """Return the agent's intelligence capabilities."""
        return IntelligenceCapabilities(
            can_explain=True,
            can_recommend=True,
            can_detect_anomalies=True,
            can_reason=True,
            can_validate=True,
            uses_rag=False,
            uses_tools=False
        )

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data structure."""
        # Flexible input - accepts various building analysis data
        return True

    # =========================================================================
    # CORE EXECUTION
    # =========================================================================

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute recommendation generation with AI intelligence.

        This method:
        1. Generates rule-based recommendations deterministically
        2. Uses AI to explain and prioritize recommendations
        3. Creates AI-powered implementation roadmap
        4. Detects anomalies in building metrics

        Args:
            input_data: Dictionary with building analysis data

        Returns:
            AgentResult with recommendations and AI intelligence
        """
        with self.track_execution(input_data) as tracker:
            # =========================================================
            # STEP 1: DETERMINISTIC RECOMMENDATION MATCHING
            # =========================================================
            rule_based_result = self._generate_rule_based_recommendations(input_data)
            analysis_steps = self._get_analysis_steps(input_data, rule_based_result)

            # =========================================================
            # STEP 2: AI-POWERED EXPLANATION
            # =========================================================
            explanation = self.generate_explanation(
                input_data=input_data,
                output_data=rule_based_result,
                calculation_steps=analysis_steps
            )

            # =========================================================
            # STEP 3: AI-POWERED RECOMMENDATION ENHANCEMENT
            # =========================================================
            ai_recommendations = []
            if self.intelligent_config.enable_recommendations:
                ai_recommendations = self.generate_recommendations(
                    analysis={
                        **rule_based_result,
                        "building_context": self._get_building_context(input_data),
                        "regulatory_incentives": self._get_regulatory_incentives(input_data.get("country", "US")),
                        "climate_zone": self._determine_climate_zone(input_data),
                        "building_vintage": self._get_building_vintage(input_data.get("building_age", 10))
                    },
                    max_recommendations=10,
                    focus_areas=[
                        "quick_wins",
                        "high_roi",
                        "regulatory_compliance",
                        "decarbonization_pathway",
                        "operational_excellence"
                    ]
                )

            # =========================================================
            # STEP 4: ANOMALY DETECTION
            # =========================================================
            anomalies = []
            if self.intelligent_config.enable_anomaly_detection:
                anomalies = self.detect_anomalies(
                    data=input_data,
                    expected_ranges=self._get_expected_building_ranges(input_data.get("building_type"))
                )

            # =========================================================
            # STEP 5: AI-POWERED REASONING FOR ROADMAP
            # =========================================================
            investment_rationale = ""
            if self.intelligent_config.intelligence_level in [IntelligenceLevel.ADVANCED, IntelligenceLevel.FULL]:
                investment_rationale = self.reason_about(
                    question="What is the optimal investment sequence for these recommendations?",
                    context={
                        "recommendations": rule_based_result.get("recommendations", [])[:5],
                        "building_profile": {
                            "type": input_data.get("building_type"),
                            "age": input_data.get("building_age"),
                            "performance": input_data.get("performance_rating")
                        },
                        "budget_constraints": input_data.get("budget_constraints", "moderate"),
                        "regulatory_deadlines": self._get_regulatory_deadlines(input_data.get("country", "US"))
                    },
                    chain_of_thought=True
                )

            # =========================================================
            # STEP 6: BUILD INTELLIGENT RESULT
            # =========================================================
            result_data = {
                **rule_based_result,
                "explanation": explanation,
                "ai_prioritized_recommendations": [r.dict() if hasattr(r, 'dict') else r for r in ai_recommendations],
                "anomalies": [a.dict() if hasattr(a, 'dict') else a for a in anomalies],
                "investment_rationale": investment_rationale,
                "intelligence_level": self.get_intelligence_level().value,
            }

            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent": "IntelligentRecommendationAgent",
                    "analysis_basis": {
                        "building_type": input_data.get("building_type", "commercial_office"),
                        "performance_rating": input_data.get("performance_rating", "Average"),
                        "building_age": input_data.get("building_age", 10),
                    },
                    "intelligence_metrics": self.get_intelligence_metrics().dict(),
                    "regulatory_context": self.intelligent_config.regulatory_context,
                },
                timestamp=DeterministicClock.now()
            )

    # =========================================================================
    # DETERMINISTIC RECOMMENDATION METHODS (Zero-Hallucination)
    # =========================================================================

    def _generate_rule_based_recommendations(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations using rule-based matching - FULLY DETERMINISTIC.

        This method contains ZERO LLM calls. All matching is rule-based.
        """
        recommendations = []

        emissions_by_source = input_data.get("emissions_by_source", {})
        intensity = input_data.get("intensity", {})
        building_type = input_data.get("building_type", "commercial_office")
        building_age = input_data.get("building_age", 10)
        performance_rating = input_data.get("performance_rating", "Average")
        load_breakdown = input_data.get("load_breakdown", {})
        country = input_data.get("country", "US")

        # Analyze electricity consumption
        if "electricity" in emissions_by_source:
            elec_emissions = emissions_by_source["electricity"]
            total_emissions = sum(emissions_by_source.values())
            elec_percentage = (elec_emissions / total_emissions * 100) if total_emissions > 0 else 0

            if elec_percentage > 60:
                recommendations.extend(self._get_recommendations("renewable", "high_electricity"))
                recommendations.extend(self._get_recommendations("lighting", "high_consumption"))

        # Analyze HVAC load
        hvac_load = load_breakdown.get("hvac_load", 0)
        if hvac_load > 0.4:
            recommendations.extend(self._get_recommendations("hvac", "high_consumption"))
            if building_age > 15:
                recommendations.extend(self._get_recommendations("hvac", "poor_efficiency"))

        # Building envelope for older buildings
        if building_age > 20:
            recommendations.extend(self._get_recommendations("envelope", "poor_insulation"))

        # Performance-based recommendations
        if performance_rating in ["Below Average", "Poor"]:
            recommendations.extend(self._get_recommendations("operations", "general"))

        # Country-specific recommendations
        country_specific = self._get_country_specific_recommendations(country)
        recommendations.extend(country_specific)

        # Prioritize and deduplicate
        unique_recommendations = self._prioritize_recommendations(recommendations)

        # Calculate potential savings
        total_potential_savings = self._calculate_savings_potential(unique_recommendations, emissions_by_source)

        # Group by category
        grouped = self._group_recommendations(unique_recommendations)

        # Create roadmap
        roadmap = self._create_roadmap(unique_recommendations)

        return {
            "recommendations": unique_recommendations[:10],
            "grouped_recommendations": grouped,
            "total_recommendations": len(unique_recommendations),
            "potential_emissions_reduction": total_potential_savings,
            "implementation_roadmap": roadmap,
            "quick_wins": [r for r in unique_recommendations if r.get("cost") == "Low"][:3],
            "high_impact": [r for r in unique_recommendations if "20%" in r.get("impact", "")][:3],
        }

    def _get_recommendations(self, category: str, subcategory: str) -> List[Dict]:
        """Get recommendations from database."""
        if category in self.recommendations_db:
            if subcategory in self.recommendations_db[category]:
                return self.recommendations_db[category][subcategory].copy()
        return []

    def _get_country_specific_recommendations(self, country: str) -> List[Dict]:
        """Get country-specific recommendations."""
        country_recommendations = {
            "IN": [
                {
                    "action": "Install solar rooftop under government subsidy schemes",
                    "impact": "40-60% reduction in grid dependency",
                    "cost": "Medium (with subsidies)",
                    "payback": "3-5 years",
                    "priority": "High",
                },
            ],
            "US": [
                {
                    "action": "Apply for IRA tax credits for efficiency upgrades",
                    "impact": "30% cost reduction on improvements",
                    "cost": "N/A",
                    "payback": "Immediate",
                    "priority": "High",
                },
                {
                    "action": "Pursue ENERGY STAR certification",
                    "impact": "10-20% energy reduction, market recognition",
                    "cost": "Low",
                    "payback": "1-2 years",
                    "priority": "High",
                },
            ],
            "EU": [
                {
                    "action": "Comply with EU Taxonomy requirements",
                    "impact": "Access to green financing",
                    "cost": "Low",
                    "payback": "Immediate",
                    "priority": "High",
                },
            ],
        }
        return country_recommendations.get(country, [])

    def _prioritize_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Prioritize and deduplicate recommendations."""
        seen = set()
        unique = []
        for rec in recommendations:
            action = rec.get("action", "")
            if action not in seen:
                seen.add(action)
                unique.append(rec)

        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        unique.sort(
            key=lambda x: (
                priority_order.get(x.get("priority", "Low"), 3),
                self._extract_payback_years(x.get("payback", "10 years")),
            )
        )

        return unique

    def _extract_payback_years(self, payback_str: str) -> float:
        """Extract numeric payback period."""
        if "Immediate" in payback_str:
            return 0
        if "N/A" in payback_str:
            return 999

        try:
            numbers = re.findall(r"\d+", payback_str)
            if numbers:
                return float(numbers[0])
        except Exception:
            pass

        return 10

    def _calculate_savings_potential(
        self, recommendations: List[Dict], emissions_by_source: Dict
    ) -> Dict:
        """Calculate potential emissions savings."""
        total_emissions = sum(emissions_by_source.values())

        min_savings = 0
        max_savings = 0

        for rec in recommendations[:5]:
            impact = rec.get("impact", "")
            if "%" in impact:
                percentages = re.findall(r"(\d+)%", impact)
                if percentages:
                    avg_percentage = sum(map(int, percentages)) / len(percentages)
                    min_savings += total_emissions * (avg_percentage * 0.5 / 100)
                    max_savings += total_emissions * (avg_percentage / 100)

        if total_emissions > 0:
            percentage_range = f"{round(min_savings/total_emissions*100, 1)}-{round(max_savings/total_emissions*100, 1)}%"
        else:
            percentage_range = "0.0-0.0%"

        return {
            "minimum_kg_co2e": round(min_savings, 1),
            "maximum_kg_co2e": round(max_savings, 1),
            "percentage_range": percentage_range,
        }

    def _group_recommendations(self, recommendations: List[Dict]) -> Dict:
        """Group recommendations by timeframe."""
        grouped = {
            "Immediate Actions": [],
            "Short Term (1-2 years)": [],
            "Medium Term (3-5 years)": [],
            "Long Term (5+ years)": [],
        }

        for rec in recommendations:
            payback = self._extract_payback_years(rec.get("payback", ""))
            if payback == 0:
                grouped["Immediate Actions"].append(rec)
            elif payback <= 2:
                grouped["Short Term (1-2 years)"].append(rec)
            elif payback <= 5:
                grouped["Medium Term (3-5 years)"].append(rec)
            else:
                grouped["Long Term (5+ years)"].append(rec)

        return {k: v for k, v in grouped.items() if v}

    def _create_roadmap(self, recommendations: List[Dict]) -> List[Dict]:
        """Create implementation roadmap."""
        roadmap = []

        phase1 = [r for r in recommendations if r.get("cost") == "Low" and r.get("priority") == "High"]
        if phase1:
            roadmap.append({
                "phase": "Phase 1: Quick Wins (0-6 months)",
                "actions": phase1[:3],
                "estimated_cost": "Low",
                "expected_impact": "5-15% reduction",
            })

        phase2 = [r for r in recommendations if r.get("cost") == "Medium"]
        if phase2:
            roadmap.append({
                "phase": "Phase 2: Strategic Improvements (6-18 months)",
                "actions": phase2[:3],
                "estimated_cost": "Medium",
                "expected_impact": "15-30% reduction",
            })

        phase3 = [r for r in recommendations if r.get("cost") == "High" and r.get("priority") == "High"]
        if phase3:
            roadmap.append({
                "phase": "Phase 3: Major Upgrades (18-36 months)",
                "actions": phase3[:2],
                "estimated_cost": "High",
                "expected_impact": "20-40% reduction",
            })

        return roadmap

    def _get_analysis_steps(
        self,
        input_data: Dict[str, Any],
        result: Dict[str, Any]
    ) -> List[str]:
        """Document analysis steps for explainability."""
        steps = [
            f"Analyzed building type: {input_data.get('building_type', 'commercial_office')}",
            f"Building age: {input_data.get('building_age', 'unknown')} years",
            f"Performance rating: {input_data.get('performance_rating', 'Average')}",
        ]

        emissions = input_data.get("emissions_by_source", {})
        if emissions:
            total = sum(emissions.values())
            steps.append(f"Total emissions analyzed: {total} kg CO2e")
            for source, value in emissions.items():
                pct = (value / total * 100) if total > 0 else 0
                steps.append(f"  - {source}: {value} kg ({pct:.1f}%)")

        steps.append(f"Generated {result.get('total_recommendations', 0)} recommendations")
        steps.append(f"Identified {len(result.get('quick_wins', []))} quick wins")

        return steps

    def _get_building_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get building context for AI analysis."""
        return {
            "type": input_data.get("building_type", "commercial_office"),
            "age": input_data.get("building_age", 10),
            "area_sqft": input_data.get("area_sqft"),
            "occupancy": input_data.get("occupancy"),
            "operating_hours": input_data.get("operating_hours", "standard"),
            "climate_zone": self._determine_climate_zone(input_data),
        }

    def _determine_climate_zone(self, input_data: Dict[str, Any]) -> str:
        """Determine climate zone from location."""
        country = input_data.get("country", "US")
        state = input_data.get("state", "")

        if country == "US":
            hot_states = ["TX", "FL", "AZ", "NV", "CA"]
            cold_states = ["MN", "WI", "MI", "ME", "VT"]
            if state in hot_states:
                return "hot"
            elif state in cold_states:
                return "cold"
        return "mixed"

    def _get_building_vintage(self, age: int) -> str:
        """Categorize building by vintage."""
        if age < 10:
            return "modern"
        elif age < 25:
            return "mid-age"
        elif age < 50:
            return "aging"
        else:
            return "historic"

    def _get_regulatory_incentives(self, country: str) -> List[str]:
        """Get available regulatory incentives."""
        incentives = {
            "US": [
                "IRA Section 179D Commercial Building Deduction",
                "IRA Investment Tax Credit (ITC) for solar",
                "State-level utility rebates",
                "ENERGY STAR certification benefits"
            ],
            "EU": [
                "EU Taxonomy alignment benefits",
                "CSRD reporting requirements",
                "National renovation subsidies",
                "Green bond eligibility"
            ],
            "IN": [
                "GRIHA certification incentives",
                "Solar rooftop subsidies",
                "PAT scheme benefits",
                "State-level incentives"
            ],
        }
        return incentives.get(country, [])

    def _get_regulatory_deadlines(self, country: str) -> Dict[str, str]:
        """Get regulatory deadlines."""
        deadlines = {
            "US": {
                "SB253 (California)": "2026-2027",
                "Local Building Performance Standards": "Varies by city"
            },
            "EU": {
                "CSRD Reporting": "2024-2026",
                "EU Taxonomy Disclosure": "2024",
                "EPBD Renovation Requirements": "2030"
            },
        }
        return deadlines.get(country, {})

    def _get_expected_building_ranges(self, building_type: str) -> Dict[str, tuple]:
        """Define expected ranges for anomaly detection."""
        ranges = {
            "commercial_office": {
                "energy_intensity_kwh_sqft": (10, 50),
                "emissions_intensity_kgco2e_sqft": (5, 25)
            },
            "retail": {
                "energy_intensity_kwh_sqft": (15, 60),
                "emissions_intensity_kgco2e_sqft": (7, 30)
            },
            "warehouse": {
                "energy_intensity_kwh_sqft": (5, 20),
                "emissions_intensity_kgco2e_sqft": (2, 10)
            },
        }
        return ranges.get(building_type, {"energy_intensity_kwh_sqft": (5, 100)})


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_intelligent_recommendation_agent(
    intelligence_level: IntelligenceLevel = IntelligenceLevel.ADVANCED,
    regulatory_context: str = "CSRD ESRS E1, EU Taxonomy, IRA Tax Credits",
    **kwargs
) -> IntelligentRecommendationAgent:
    """
    Factory function to create IntelligentRecommendationAgent.

    Args:
        intelligence_level: Level of LLM intelligence
        regulatory_context: Regulatory context for prompts
        **kwargs: Additional configuration

    Returns:
        Configured IntelligentRecommendationAgent
    """
    config = IntelligentRecommendationAgentConfig(
        name="IntelligentRecommendationAgent",
        description="AI-native building optimization advisor",
        intelligence_level=intelligence_level,
        regulatory_context=regulatory_context,
        **kwargs
    )

    return IntelligentRecommendationAgent(config)


# =============================================================================
# BACKWARD COMPATIBILITY: Retrofit existing RecommendationAgent
# =============================================================================

def retrofit_recommendation_agent():
    """
    Retrofit the existing RecommendationAgent with intelligence.
    """
    from greenlang.agents.recommendation_agent import RecommendationAgent
    from greenlang.agents.intelligence_mixin import retrofit_agent_class

    IntelligentRecommendationAgentRetrofit = retrofit_agent_class(RecommendationAgent)
    return IntelligentRecommendationAgentRetrofit


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """Example usage of IntelligentRecommendationAgent."""

    agent = create_intelligent_recommendation_agent()

    example_input = {
        "emissions_by_source": {
            "electricity": 50000,
            "natural_gas": 30000,
            "district_heating": 10000
        },
        "building_type": "commercial_office",
        "building_age": 25,
        "performance_rating": "Below Average",
        "load_breakdown": {"hvac_load": 0.45, "lighting_load": 0.25, "plug_load": 0.30},
        "country": "US"
    }

    result = agent.run(example_input)

    print("=" * 60)
    print("INTELLIGENT RECOMMENDATION AGENT RESULTS")
    print("=" * 60)
    print(f"\nSuccess: {result.success}")
    print(f"\n--- DETERMINISTIC DATA ---")
    print(f"Total Recommendations: {result.data['total_recommendations']}")
    print(f"Quick Wins: {len(result.data['quick_wins'])}")
    print(f"\n--- AI-GENERATED EXPLANATION ---")
    print(result.data.get("explanation", "N/A"))
    print(f"\n--- AI-PRIORITIZED RECOMMENDATIONS ---")
    for rec in result.data.get("ai_prioritized_recommendations", [])[:5]:
        title = rec.get("title", "Unknown") if isinstance(rec, dict) else "Unknown"
        print(f"  - {title}")
    print(f"\n--- INVESTMENT RATIONALE ---")
    print(result.data.get("investment_rationale", "N/A")[:500])
