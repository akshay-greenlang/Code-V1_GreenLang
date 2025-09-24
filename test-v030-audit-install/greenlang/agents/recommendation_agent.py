from typing import Dict, Any, Optional, List
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
import logging

logger = logging.getLogger(__name__)


class RecommendationAgent(BaseAgent):
    """Agent for providing optimization recommendations based on building analysis"""

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            config
            or AgentConfig(
                name="RecommendationAgent",
                description="Provides actionable recommendations for reducing emissions",
            )
        )
        self.recommendations_db = self._load_recommendations()

    def _load_recommendations(self) -> Dict:
        """Load recommendations database"""
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
            "water": {
                "high_consumption": [
                    {
                        "action": "Install low-flow fixtures",
                        "impact": "20-30% reduction in water use",
                        "cost": "Low",
                        "payback": "1 year",
                        "priority": "High",
                    },
                    {
                        "action": "Implement rainwater harvesting",
                        "impact": "30-50% reduction in water demand",
                        "cost": "Medium",
                        "payback": "5-7 years",
                        "priority": "Medium",
                    },
                ]
            },
        }

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input has required fields"""
        return True  # Flexible input

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Generate recommendations based on building analysis"""
        try:
            recommendations = []

            # Extract relevant metrics
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
                elec_percentage = (
                    (elec_emissions / total_emissions * 100)
                    if total_emissions > 0
                    else 0
                )

                if elec_percentage > 60:
                    recommendations.extend(
                        self._get_recommendations("renewable", "high_electricity")
                    )
                    recommendations.extend(
                        self._get_recommendations("lighting", "high_consumption")
                    )

            # Analyze HVAC load
            hvac_load = load_breakdown.get("hvac_load", 0)
            if hvac_load > 0.4:  # HVAC is >40% of load
                recommendations.extend(
                    self._get_recommendations("hvac", "high_consumption")
                )
                if building_age > 15:
                    recommendations.extend(
                        self._get_recommendations("hvac", "poor_efficiency")
                    )

            # Building envelope recommendations for older buildings
            if building_age > 20:
                recommendations.extend(
                    self._get_recommendations("envelope", "poor_insulation")
                )

            # Performance-based recommendations
            if performance_rating in ["Below Average", "Poor"]:
                recommendations.extend(
                    self._get_recommendations("operations", "general")
                )

            # Country-specific recommendations
            country_specific = self._get_country_specific_recommendations(country)
            recommendations.extend(country_specific)

            # Prioritize and deduplicate
            unique_recommendations = self._prioritize_recommendations(recommendations)

            # Calculate potential impact
            total_potential_savings = self._calculate_savings_potential(
                unique_recommendations, emissions_by_source
            )

            # Group by category
            grouped = self._group_recommendations(unique_recommendations)

            return AgentResult(
                success=True,
                data={
                    "recommendations": unique_recommendations[:10],  # Top 10
                    "grouped_recommendations": grouped,
                    "total_recommendations": len(unique_recommendations),
                    "potential_emissions_reduction": total_potential_savings,
                    "implementation_roadmap": self._create_roadmap(
                        unique_recommendations
                    ),
                    "quick_wins": [
                        r for r in unique_recommendations if r.get("cost") == "Low"
                    ][:3],
                    "high_impact": [
                        r
                        for r in unique_recommendations
                        if "20%" in r.get("impact", "")
                    ][:3],
                },
                metadata={
                    "analysis_basis": {
                        "building_type": building_type,
                        "performance_rating": performance_rating,
                        "building_age": building_age,
                    }
                },
            )

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return AgentResult(success=False, error=str(e))

    def _get_recommendations(self, category: str, subcategory: str) -> List[Dict]:
        """Get recommendations from database"""
        if category in self.recommendations_db:
            if subcategory in self.recommendations_db[category]:
                return self.recommendations_db[category][subcategory].copy()
        return []

    def _get_country_specific_recommendations(self, country: str) -> List[Dict]:
        """Get country-specific recommendations"""
        country_recommendations = {
            "IN": [
                {
                    "action": "Install solar rooftop under government subsidy schemes",
                    "impact": "40-60% reduction in grid dependency",
                    "cost": "Medium (with subsidies)",
                    "payback": "3-5 years",
                    "priority": "High",
                },
                {
                    "action": "Participate in PAT (Perform, Achieve, Trade) scheme",
                    "impact": "Carbon credits and compliance",
                    "cost": "Low",
                    "payback": "Immediate",
                    "priority": "Medium",
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
                {
                    "action": "Connect to district heating network",
                    "impact": "20-40% reduction in heating emissions",
                    "cost": "Medium",
                    "payback": "5-7 years",
                    "priority": "Medium",
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
        }

        return country_recommendations.get(country, [])

    def _prioritize_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Prioritize and deduplicate recommendations"""
        # Remove duplicates
        seen = set()
        unique = []
        for rec in recommendations:
            action = rec.get("action", "")
            if action not in seen:
                seen.add(action)
                unique.append(rec)

        # Sort by priority and payback
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        unique.sort(
            key=lambda x: (
                priority_order.get(x.get("priority", "Low"), 3),
                self._extract_payback_years(x.get("payback", "10 years")),
            )
        )

        return unique

    def _extract_payback_years(self, payback_str: str) -> float:
        """Extract numeric payback period"""
        if "Immediate" in payback_str:
            return 0
        if "N/A" in payback_str:
            return 999

        try:
            # Extract first number
            import re

            numbers = re.findall(r"\d+", payback_str)
            if numbers:
                return float(numbers[0])
        except:
            pass

        return 10

    def _calculate_savings_potential(
        self, recommendations: List[Dict], emissions_by_source: Dict
    ) -> Dict:
        """Calculate potential emissions savings"""
        total_emissions = sum(emissions_by_source.values())

        # Estimate based on impact strings (simplified)
        min_savings = 0
        max_savings = 0

        for rec in recommendations[:5]:  # Top 5 recommendations
            impact = rec.get("impact", "")
            if "%" in impact:
                import re

                percentages = re.findall(r"(\d+)%", impact)
                if percentages:
                    avg_percentage = sum(map(int, percentages)) / len(percentages)
                    min_savings += total_emissions * (avg_percentage * 0.5 / 100)
                    max_savings += total_emissions * (avg_percentage / 100)

        return {
            "minimum_kg_co2e": round(min_savings, 1),
            "maximum_kg_co2e": round(max_savings, 1),
            "percentage_range": f"{round(min_savings/total_emissions*100, 1)}-{round(max_savings/total_emissions*100, 1)}%",
        }

    def _group_recommendations(self, recommendations: List[Dict]) -> Dict:
        """Group recommendations by category"""
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

        # Remove empty groups
        return {k: v for k, v in grouped.items() if v}

    def _create_roadmap(self, recommendations: List[Dict]) -> List[Dict]:
        """Create implementation roadmap"""
        roadmap = []

        # Phase 1: Quick wins
        phase1 = [
            r
            for r in recommendations
            if r.get("cost") == "Low" and r.get("priority") == "High"
        ]
        if phase1:
            roadmap.append(
                {
                    "phase": "Phase 1: Quick Wins (0-6 months)",
                    "actions": phase1[:3],
                    "estimated_cost": "Low",
                    "expected_impact": "5-15% reduction",
                }
            )

        # Phase 2: Medium investments
        phase2 = [r for r in recommendations if r.get("cost") == "Medium"]
        if phase2:
            roadmap.append(
                {
                    "phase": "Phase 2: Strategic Improvements (6-18 months)",
                    "actions": phase2[:3],
                    "estimated_cost": "Medium",
                    "expected_impact": "15-30% reduction",
                }
            )

        # Phase 3: Major upgrades
        phase3 = [
            r
            for r in recommendations
            if r.get("cost") == "High" and r.get("priority") == "High"
        ]
        if phase3:
            roadmap.append(
                {
                    "phase": "Phase 3: Major Upgrades (18-36 months)",
                    "actions": phase3[:2],
                    "estimated_cost": "High",
                    "expected_impact": "20-40% reduction",
                }
            )

        return roadmap
