"""
AI-Driven Recommendation Engine -- Improvement suggestions based on gaps and best practices.

This module implements the ``RecommendationEngine`` for GL-TCFD-APP v1.0.
It generates prioritized improvement recommendations based on gap analysis
results, sector-specific best practices, and regulatory requirements.
Recommendations are scored using a composite priority formula:
priority = (regulatory_urgency * impact) / effort, enabling organizations
to focus on the highest-value improvements first.

The engine maintains a library of best practices organized by sector and
pillar, and a recommendation template library that maps gap types to
specific, actionable guidance with implementation steps.

Reference:
    - TCFD Good Practice Handbook (2021)
    - TCFD Status Report (2023)
    - IFRS S2 Implementation Guidance (2023)

Example:
    >>> from services.config import TCFDAppConfig
    >>> from services.gap_analysis_engine import GapAnalysisEngine
    >>> gap_engine = GapAnalysisEngine(TCFDAppConfig())
    >>> rec_engine = RecommendationEngine(TCFDAppConfig())
    >>> assessment = gap_engine.assess_maturity("org-1")
    >>> recs = rec_engine.generate_recommendations("org-1", assessment, "energy")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import TCFDAppConfig, GovernanceMaturityLevel
from .models import GapAssessment, Recommendation, _new_id, _now, _sha256

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector best practices (top 5 sectors)
# ---------------------------------------------------------------------------

SECTOR_BEST_PRACTICES: Dict[str, List[Dict[str, Any]]] = {
    "energy": [
        {
            "title": "Quantitative Scenario Analysis with IEA and NGFS Pathways",
            "description": "Leading energy companies use 3+ scenarios (IEA NZE, APS, STEPS) with quantified financial impacts on asset portfolios.",
            "pillar": "strategy",
            "maturity_impact": 1.5,
            "example_companies": ["Shell", "TotalEnergies", "BP"],
        },
        {
            "title": "Stranding Risk Assessment for Fossil Fuel Assets",
            "description": "Comprehensive asset-level stranding probability assessment under accelerated transition scenarios.",
            "pillar": "strategy",
            "maturity_impact": 1.2,
            "example_companies": ["Equinor", "Repsol"],
        },
        {
            "title": "Scope 3 Category 11 (Use of Sold Products) Reporting",
            "description": "Full Scope 3 Cat 11 disclosure with well-to-wheel lifecycle methodology for sold fuels and products.",
            "pillar": "metrics_targets",
            "maturity_impact": 1.3,
            "example_companies": ["Shell", "TotalEnergies", "Eni"],
        },
        {
            "title": "Net-Zero Transition Plan with Interim Milestones",
            "description": "Published net-zero transition plan with 2030 interim targets, capex allocation, and technology roadmap.",
            "pillar": "strategy",
            "maturity_impact": 1.4,
            "example_companies": ["Orsted", "Iberdrola", "EDP"],
        },
        {
            "title": "Climate-Linked Executive Remuneration",
            "description": "15-30% of executive short-term incentive tied to emissions reduction KPIs and energy transition metrics.",
            "pillar": "governance",
            "maturity_impact": 1.1,
            "example_companies": ["Shell", "BP", "TotalEnergies"],
        },
    ],
    "banking": [
        {
            "title": "PCAF-Aligned Financed Emissions Reporting",
            "description": "Full PCAF methodology for calculating financed emissions across all major asset classes (equity, bonds, project finance, CRE, mortgages).",
            "pillar": "metrics_targets",
            "maturity_impact": 1.5,
            "example_companies": ["ING", "ABN AMRO", "NatWest"],
        },
        {
            "title": "Portfolio Alignment and WACI Reporting",
            "description": "Temperature alignment score and Weighted Average Carbon Intensity (WACI) for lending and investment portfolios.",
            "pillar": "metrics_targets",
            "maturity_impact": 1.3,
            "example_companies": ["HSBC", "Barclays", "Deutsche Bank"],
        },
        {
            "title": "Climate Stress Testing (NGFS Framework)",
            "description": "Climate stress testing using NGFS scenarios for credit risk, market risk, and operational risk across portfolios.",
            "pillar": "risk_management",
            "maturity_impact": 1.4,
            "example_companies": ["BNP Paribas", "Societe Generale", "UniCredit"],
        },
        {
            "title": "Sector-Specific Decarbonization Pathways",
            "description": "Published sector-specific financing policies aligned with IEA NZE pathways for high-emitting sectors.",
            "pillar": "strategy",
            "maturity_impact": 1.2,
            "example_companies": ["Standard Chartered", "Citi", "JPMorgan"],
        },
        {
            "title": "Board-Level Climate Risk Committee",
            "description": "Dedicated board-level committee for climate risk with quarterly reviews, external expert advisors, and direct CEO reporting line.",
            "pillar": "governance",
            "maturity_impact": 1.0,
            "example_companies": ["HSBC", "Lloyds Banking Group"],
        },
    ],
    "materials": [
        {
            "title": "Process Emissions Reduction Roadmap",
            "description": "Technology roadmap for hard-to-abate process emissions (cement, steel, chemicals) including CCUS, hydrogen, and electrification.",
            "pillar": "strategy",
            "maturity_impact": 1.4,
            "example_companies": ["HeidelbergCement", "ArcelorMittal", "BASF"],
        },
        {
            "title": "Scope 3 Supply Chain Engagement Program",
            "description": "Structured supplier engagement covering 80%+ of procurement spend with emissions reduction targets.",
            "pillar": "metrics_targets",
            "maturity_impact": 1.2,
            "example_companies": ["LafargeHolcim", "Dow"],
        },
        {
            "title": "Circular Economy Integration",
            "description": "Climate-positive circularity strategy integrating recycled content targets, waste-to-resource programs, and lifecycle emissions reporting.",
            "pillar": "strategy",
            "maturity_impact": 1.1,
            "example_companies": ["BASF", "Covestro"],
        },
        {
            "title": "Internal Carbon Pricing Mechanism",
            "description": "Shadow carbon price ($50-150/tCO2e) integrated into all capital investment decisions above threshold.",
            "pillar": "metrics_targets",
            "maturity_impact": 1.0,
            "example_companies": ["Saint-Gobain", "Solvay"],
        },
        {
            "title": "Physical Risk Assessment for Manufacturing Sites",
            "description": "Asset-level physical risk screening for all manufacturing sites using RCP8.5/SSP5-8.5 scenarios.",
            "pillar": "risk_management",
            "maturity_impact": 1.3,
            "example_companies": ["ArcelorMittal", "Glencore"],
        },
    ],
    "technology": [
        {
            "title": "Data Center Renewable Energy Procurement",
            "description": "100% renewable energy for data centers through PPAs, with hourly matching and additionality proof.",
            "pillar": "metrics_targets",
            "maturity_impact": 1.3,
            "example_companies": ["Google", "Microsoft", "Apple"],
        },
        {
            "title": "Value Chain Scope 3 Quantification",
            "description": "Comprehensive Scope 3 reporting covering purchased goods (Cat 1), business travel (Cat 6), use of sold products (Cat 11).",
            "pillar": "metrics_targets",
            "maturity_impact": 1.2,
            "example_companies": ["Microsoft", "Salesforce", "SAP"],
        },
        {
            "title": "AI-Enabled Climate Risk Assessment",
            "description": "Machine learning models for physical climate risk assessment of global operations and supply chain.",
            "pillar": "risk_management",
            "maturity_impact": 1.1,
            "example_companies": ["Google", "IBM"],
        },
        {
            "title": "Climate Solution Revenue Disclosure",
            "description": "Revenue attribution framework for products and services that enable customer emissions reductions.",
            "pillar": "strategy",
            "maturity_impact": 1.0,
            "example_companies": ["Microsoft", "Schneider Electric"],
        },
        {
            "title": "SBTi-Validated Net-Zero Target",
            "description": "Science-based net-zero target validated by SBTi with near-term and long-term milestones.",
            "pillar": "metrics_targets",
            "maturity_impact": 1.4,
            "example_companies": ["Microsoft", "Apple", "HP"],
        },
    ],
    "insurance": [
        {
            "title": "Nat-Cat Modeling with Climate Projections",
            "description": "Integration of climate change projections into natural catastrophe models for underwriting.",
            "pillar": "risk_management",
            "maturity_impact": 1.5,
            "example_companies": ["Swiss Re", "Munich Re", "Zurich"],
        },
        {
            "title": "Investment Portfolio Decarbonization",
            "description": "Decarbonization targets for investment portfolio aligned with Net-Zero Asset Owner Alliance.",
            "pillar": "metrics_targets",
            "maturity_impact": 1.3,
            "example_companies": ["Allianz", "AXA", "Aviva"],
        },
        {
            "title": "Climate Scenario Impact on Reserves",
            "description": "Quantified impact of climate scenarios on claims reserves and loss ratios.",
            "pillar": "strategy",
            "maturity_impact": 1.4,
            "example_companies": ["Swiss Re", "Lloyd's"],
        },
        {
            "title": "Transition Risk Underwriting Policies",
            "description": "Sector-specific underwriting restrictions for coal, oil sands, and Arctic drilling.",
            "pillar": "strategy",
            "maturity_impact": 1.1,
            "example_companies": ["AXA", "Zurich", "Swiss Re"],
        },
        {
            "title": "Physical Risk Dashboard for All Operations",
            "description": "Real-time physical risk monitoring dashboard covering all insured assets and own operations.",
            "pillar": "risk_management",
            "maturity_impact": 1.2,
            "example_companies": ["Munich Re", "Swiss Re"],
        },
    ],
}


# ---------------------------------------------------------------------------
# Recommendation templates per pillar
# ---------------------------------------------------------------------------

RECOMMENDATION_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "governance": [
        {
            "id": "REC-GOV-001",
            "title": "Establish Dedicated Board Climate Committee",
            "description": "Create a board-level sustainability/climate risk committee with formal terms of reference, quarterly meetings, and external expert advisors.",
            "impact": "high",
            "effort": "medium",
            "effort_days": 15,
            "regulatory_urgency": 3,
            "applicable_when": "no_committee",
        },
        {
            "id": "REC-GOV-002",
            "title": "Link Executive Remuneration to Climate KPIs",
            "description": "Integrate emissions reduction targets, transition plan milestones, and climate risk metrics into short-term and long-term incentive plans (10-20% weighting).",
            "impact": "high",
            "effort": "medium",
            "effort_days": 20,
            "regulatory_urgency": 2,
            "applicable_when": "no_remuneration_link",
        },
        {
            "id": "REC-GOV-003",
            "title": "Implement Board Climate Competency Program",
            "description": "Structured climate education program for all board members covering science, regulation, scenario analysis, and financial impacts.",
            "impact": "medium",
            "effort": "low",
            "effort_days": 8,
            "regulatory_urgency": 2,
            "applicable_when": "low_competency",
        },
        {
            "id": "REC-GOV-004",
            "title": "Increase Board Climate Review Frequency",
            "description": "Move from annual/ad-hoc to quarterly board reviews of climate risks, targets progress, and regulatory developments.",
            "impact": "medium",
            "effort": "low",
            "effort_days": 5,
            "regulatory_urgency": 2,
            "applicable_when": "low_review_frequency",
        },
    ],
    "strategy": [
        {
            "id": "REC-STR-001",
            "title": "Conduct Quantitative Scenario Analysis",
            "description": "Perform quantitative scenario analysis using at least 3 scenarios (1.5C, 2C, 3C+) with financial impact quantification on revenue, costs, and asset values.",
            "impact": "high",
            "effort": "high",
            "effort_days": 40,
            "regulatory_urgency": 4,
            "applicable_when": "no_scenario_analysis",
        },
        {
            "id": "REC-STR-002",
            "title": "Develop Climate Transition Plan",
            "description": "Create a comprehensive transition plan aligned with 1.5C pathway, including interim targets, technology roadmap, capex allocation, and governance.",
            "impact": "high",
            "effort": "high",
            "effort_days": 50,
            "regulatory_urgency": 4,
            "applicable_when": "no_transition_plan",
        },
        {
            "id": "REC-STR-003",
            "title": "Quantify Financial Impacts of Climate Risks",
            "description": "Quantify climate risk financial impacts on income statement, balance sheet, and cash flows using deterministic and probabilistic methods.",
            "impact": "high",
            "effort": "medium",
            "effort_days": 30,
            "regulatory_urgency": 3,
            "applicable_when": "no_financial_quantification",
        },
        {
            "id": "REC-STR-004",
            "title": "Map Value Chain Climate Exposure",
            "description": "Conduct systematic assessment of climate risks and opportunities across upstream, direct operations, and downstream value chain segments.",
            "impact": "medium",
            "effort": "medium",
            "effort_days": 25,
            "regulatory_urgency": 2,
            "applicable_when": "no_value_chain_assessment",
        },
    ],
    "risk_management": [
        {
            "id": "REC-RM-001",
            "title": "Integrate Climate into Enterprise Risk Management",
            "description": "Formally integrate climate risks into the ERM framework with defined risk appetite, assessment criteria, and reporting thresholds.",
            "impact": "high",
            "effort": "medium",
            "effort_days": 25,
            "regulatory_urgency": 3,
            "applicable_when": "no_erm_integration",
        },
        {
            "id": "REC-RM-002",
            "title": "Establish Climate Risk Register",
            "description": "Create and maintain a comprehensive climate risk register with likelihood/impact scoring, risk owners, response strategies, and review dates.",
            "impact": "high",
            "effort": "medium",
            "effort_days": 20,
            "regulatory_urgency": 3,
            "applicable_when": "no_risk_register",
        },
        {
            "id": "REC-RM-003",
            "title": "Conduct Physical Risk Assessment",
            "description": "Perform asset-level physical risk assessment covering acute (flood, wildfire, cyclone) and chronic (temperature, water stress) hazards under RCP4.5 and RCP8.5.",
            "impact": "high",
            "effort": "high",
            "effort_days": 35,
            "regulatory_urgency": 3,
            "applicable_when": "no_physical_risk_assessment",
        },
        {
            "id": "REC-RM-004",
            "title": "Implement Climate Risk Monitoring Dashboard",
            "description": "Deploy automated monitoring of climate risk indicators including regulatory changes, physical hazard alerts, and transition risk triggers.",
            "impact": "medium",
            "effort": "medium",
            "effort_days": 20,
            "regulatory_urgency": 2,
            "applicable_when": "no_monitoring",
        },
    ],
    "metrics_targets": [
        {
            "id": "REC-MT-001",
            "title": "Complete Scope 3 GHG Inventory",
            "description": "Calculate and report Scope 3 emissions across all 15 GHG Protocol categories, with at minimum Categories 1, 2, 3, 4, 5, 6, 7 and 11.",
            "impact": "high",
            "effort": "high",
            "effort_days": 45,
            "regulatory_urgency": 4,
            "applicable_when": "incomplete_scope3",
        },
        {
            "id": "REC-MT-002",
            "title": "Set Science-Based Targets (SBTi)",
            "description": "Develop and submit near-term and (if applicable) net-zero targets to the Science Based Targets initiative for validation.",
            "impact": "high",
            "effort": "high",
            "effort_days": 40,
            "regulatory_urgency": 3,
            "applicable_when": "no_sbti",
        },
        {
            "id": "REC-MT-003",
            "title": "Implement ISSB 7 Cross-Industry Metrics",
            "description": "Report all 7 IFRS S2 cross-industry metrics: GHG emissions, transition risk assets, physical risk assets, opportunity revenue, capex, internal carbon price, remuneration.",
            "impact": "high",
            "effort": "medium",
            "effort_days": 30,
            "regulatory_urgency": 3,
            "applicable_when": "missing_cross_industry_metrics",
        },
        {
            "id": "REC-MT-004",
            "title": "Obtain Third-Party Assurance on GHG Data",
            "description": "Engage an accredited verification body for limited or reasonable assurance of Scope 1, 2, and material Scope 3 emissions.",
            "impact": "medium",
            "effort": "medium",
            "effort_days": 20,
            "regulatory_urgency": 3,
            "applicable_when": "no_assurance",
        },
    ],
}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PrioritizedRecommendation(BaseModel):
    """A recommendation with composite priority score."""
    id: str = Field(default_factory=_new_id)
    title: str = Field(...)
    pillar: str = Field(default="")
    description: str = Field(default="")
    impact: str = Field(default="medium")
    effort: str = Field(default="medium")
    effort_days: int = Field(default=10)
    regulatory_urgency: int = Field(default=2, ge=1, le=5)
    priority_score: float = Field(default=0.0)
    estimated_maturity_improvement: float = Field(default=0.0)
    status: str = Field(default="open")
    implementation_guide: List[str] = Field(default_factory=list)


class RecommendationStatus(BaseModel):
    """Status of all recommendations for an organization."""
    org_id: str = Field(...)
    total_recommendations: int = Field(default=0)
    completed: int = Field(default=0)
    in_progress: int = Field(default=0)
    open: int = Field(default=0)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    overall_progress_pct: float = Field(default=0.0)
    generated_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# RecommendationEngine
# ---------------------------------------------------------------------------

class RecommendationEngine:
    """
    AI-Driven Recommendation Engine for TCFD disclosure improvement.

    Generates, prioritizes, and tracks improvement recommendations based on
    gap analysis results, sector best practices, and regulatory requirements.

    Attributes:
        config: Application configuration.
        _org_recommendations: Stored recommendations per organization.

    Example:
        >>> engine = RecommendationEngine(TCFDAppConfig())
        >>> recs = engine.generate_recommendations("org-1", assessment, "energy")
        >>> ranked = engine.prioritize_recommendations(recs)
    """

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        """
        Initialize the RecommendationEngine.

        Args:
            config: Optional application configuration.
        """
        self.config = config or TCFDAppConfig()
        self._org_recommendations: Dict[str, List[PrioritizedRecommendation]] = {}
        logger.info("RecommendationEngine initialized")

    def generate_recommendations(
        self,
        org_id: str,
        gap_assessment: GapAssessment,
        sector: str = "",
    ) -> List[PrioritizedRecommendation]:
        """
        Generate recommendations based on gap assessment and sector.

        Args:
            org_id: Organization ID.
            gap_assessment: Gap analysis result.
            sector: Organization sector for best-practice matching.

        Returns:
            List of PrioritizedRecommendation objects.
        """
        recommendations: List[PrioritizedRecommendation] = []

        # Identify weak pillars from assessment
        # pillar_scores is List[MaturityScore] with .pillar (TCFDPillar) and .score (int)
        weak_pillars = [
            ms.pillar.value for ms in gap_assessment.pillar_scores
            if ms.score < 3
        ]
        if not weak_pillars:
            weak_pillars = [ms.pillar.value for ms in gap_assessment.pillar_scores]

        # Generate from templates for weak pillars
        for pillar in weak_pillars:
            templates = RECOMMENDATION_TEMPLATES.get(pillar, [])
            for template in templates:
                impact_score = {"high": 3, "medium": 2, "low": 1}.get(
                    template["impact"], 2
                )
                effort_score = {"high": 3, "medium": 2, "low": 1}.get(
                    template["effort"], 2
                )
                urgency = template.get("regulatory_urgency", 2)
                priority = round(
                    (urgency * impact_score) / max(effort_score, 1), 2
                )

                guide = self.create_implementation_guide_steps(template)

                rec = PrioritizedRecommendation(
                    title=template["title"],
                    pillar=pillar,
                    description=template["description"],
                    impact=template["impact"],
                    effort=template["effort"],
                    effort_days=template.get("effort_days", 10),
                    regulatory_urgency=urgency,
                    priority_score=priority,
                    implementation_guide=guide,
                )
                recommendations.append(rec)

        # Add sector-specific best practices
        sector_key = sector.lower().replace(" ", "_")
        best_practices = SECTOR_BEST_PRACTICES.get(sector_key, [])
        for bp in best_practices:
            if bp["pillar"] in weak_pillars:
                rec = PrioritizedRecommendation(
                    title=f"Best Practice: {bp['title']}",
                    pillar=bp["pillar"],
                    description=bp["description"],
                    impact="high",
                    effort="medium",
                    effort_days=25,
                    regulatory_urgency=2,
                    priority_score=round(bp.get("maturity_impact", 1.0) * 2, 2),
                    estimated_maturity_improvement=bp.get("maturity_impact", 0.5),
                    implementation_guide=[
                        f"Study approach used by: {', '.join(bp.get('example_companies', []))}",
                        "Assess applicability to organization context",
                        "Develop implementation plan with 90-day milestones",
                        "Assign dedicated project team with executive sponsor",
                    ],
                )
                recommendations.append(rec)

        # Store
        self._org_recommendations[org_id] = recommendations
        logger.info(
            "Generated %d recommendations for org %s (sector=%s, weak_pillars=%s)",
            len(recommendations), org_id, sector, weak_pillars,
        )
        return recommendations

    def prioritize_recommendations(
        self,
        recommendations: List[PrioritizedRecommendation],
    ) -> List[PrioritizedRecommendation]:
        """
        Prioritize recommendations by composite score.

        Formula: priority = (regulatory_urgency * impact_score) / effort_score

        Args:
            recommendations: List of recommendations to prioritize.

        Returns:
            Recommendations sorted by priority_score descending.
        """
        for rec in recommendations:
            impact_val = {"high": 3, "medium": 2, "low": 1}.get(rec.impact, 2)
            effort_val = {"high": 3, "medium": 2, "low": 1}.get(rec.effort, 2)
            rec.priority_score = round(
                (rec.regulatory_urgency * impact_val) / max(effort_val, 1), 2
            )

        sorted_recs = sorted(
            recommendations, key=lambda r: r.priority_score, reverse=True,
        )
        logger.info(
            "Prioritized %d recommendations, top score=%.2f",
            len(sorted_recs),
            sorted_recs[0].priority_score if sorted_recs else 0.0,
        )
        return sorted_recs

    def get_sector_best_practices(self, sector: str) -> List[Dict[str, Any]]:
        """
        Get best practice examples for a sector.

        Args:
            sector: Sector key (e.g. "energy", "banking").

        Returns:
            List of best practice dictionaries.
        """
        sector_key = sector.lower().replace(" ", "_")
        practices = SECTOR_BEST_PRACTICES.get(sector_key, [])
        if not practices:
            logger.warning(
                "No best practices found for sector '%s'. Available: %s",
                sector, list(SECTOR_BEST_PRACTICES.keys()),
            )
        return practices

    def estimate_improvement_impact(
        self,
        recommendation: PrioritizedRecommendation,
    ) -> float:
        """
        Estimate the maturity score improvement from implementing a recommendation.

        Args:
            recommendation: Recommendation to estimate.

        Returns:
            Estimated maturity score improvement (0-2.0 scale on 5-point maturity).
        """
        if recommendation.estimated_maturity_improvement > 0:
            return recommendation.estimated_maturity_improvement

        impact_val = {"high": 1.2, "medium": 0.7, "low": 0.3}.get(
            recommendation.impact, 0.5
        )
        urgency_bonus = recommendation.regulatory_urgency * 0.1
        improvement = round(min(impact_val + urgency_bonus, 2.0), 2)

        logger.info(
            "Estimated improvement for '%s': %.2f maturity points",
            recommendation.title[:30], improvement,
        )
        return improvement

    def create_implementation_guide(
        self,
        recommendation: PrioritizedRecommendation,
    ) -> List[str]:
        """
        Create a step-by-step implementation guide for a recommendation.

        Args:
            recommendation: Recommendation to create guide for.

        Returns:
            List of implementation steps.
        """
        if recommendation.implementation_guide:
            return recommendation.implementation_guide

        # Generic guide based on pillar
        pillar_guides: Dict[str, List[str]] = {
            "governance": [
                "1. Conduct current-state assessment of governance structures",
                "2. Draft terms of reference / policy updates",
                "3. Obtain board approval for proposed changes",
                "4. Implement organizational changes (committee, roles, reporting)",
                "5. Establish review cadence and KPIs",
                "6. Document in annual TCFD disclosure",
            ],
            "strategy": [
                "1. Define scope and objectives of the analysis",
                "2. Select climate scenarios and time horizons",
                "3. Gather operational and financial data inputs",
                "4. Run quantitative/qualitative analysis",
                "5. Interpret results and identify strategic implications",
                "6. Present findings to board and update strategy",
                "7. Document in TCFD Strategy disclosure sections",
            ],
            "risk_management": [
                "1. Map current risk management processes",
                "2. Identify integration points for climate risks",
                "3. Define climate risk assessment methodology",
                "4. Update risk register and governance framework",
                "5. Train risk management teams",
                "6. Implement monitoring and reporting",
                "7. Document in TCFD Risk Management disclosures",
            ],
            "metrics_targets": [
                "1. Inventory current climate metrics and data sources",
                "2. Identify gaps against TCFD/ISSB requirements",
                "3. Establish data collection processes",
                "4. Calculate and validate metrics",
                "5. Set targets with base year and milestones",
                "6. Implement performance tracking systems",
                "7. Obtain third-party assurance if applicable",
                "8. Document in TCFD Metrics & Targets disclosures",
            ],
        }

        return pillar_guides.get(recommendation.pillar, [
            "1. Assess current state",
            "2. Define target state and gap",
            "3. Develop implementation plan",
            "4. Execute with assigned resources",
            "5. Monitor progress and adjust",
            "6. Document in TCFD disclosure",
        ])

    def create_implementation_guide_steps(
        self,
        template: Dict[str, Any],
    ) -> List[str]:
        """
        Create implementation steps from a recommendation template.

        Args:
            template: Recommendation template dictionary.

        Returns:
            List of implementation steps.
        """
        effort = template.get("effort", "medium")
        pillar = template.get("pillar", "")

        base_steps = [
            f"1. Assess current state for: {template.get('title', '')}",
            "2. Define target outcomes and success criteria",
            f"3. Allocate resources ({template.get('effort_days', 10)} person-days estimated)",
        ]

        if effort == "high":
            base_steps.extend([
                "4. Establish dedicated project team with executive sponsor",
                "5. Develop detailed project plan with 30-60-90 day milestones",
                "6. Execute implementation in phased approach",
                "7. Conduct mid-point review and adjust",
                "8. Validate outputs and document results",
                "9. Integrate into TCFD disclosure narrative",
            ])
        elif effort == "medium":
            base_steps.extend([
                "4. Assign responsible owner and timeline",
                "5. Execute implementation within 60-day window",
                "6. Review and validate results",
                "7. Integrate into TCFD disclosure narrative",
            ])
        else:
            base_steps.extend([
                "4. Execute within 30-day window",
                "5. Document and disclose",
            ])

        return base_steps

    def track_recommendation_status(self, org_id: str) -> RecommendationStatus:
        """
        Track status of all recommendations for an organization.

        Args:
            org_id: Organization ID.

        Returns:
            RecommendationStatus with progress metrics.
        """
        recs = self._org_recommendations.get(org_id, [])

        completed = sum(1 for r in recs if r.status == "completed")
        in_progress = sum(1 for r in recs if r.status == "in_progress")
        open_count = sum(1 for r in recs if r.status == "open")
        total = len(recs)

        progress_pct = round(completed / max(total, 1) * 100, 1)

        rec_dicts: List[Dict[str, Any]] = [
            {
                "id": r.id,
                "title": r.title,
                "pillar": r.pillar,
                "priority_score": r.priority_score,
                "effort_days": r.effort_days,
                "status": r.status,
            }
            for r in recs
        ]

        return RecommendationStatus(
            org_id=org_id,
            total_recommendations=total,
            completed=completed,
            in_progress=in_progress,
            open=open_count,
            recommendations=rec_dicts,
            overall_progress_pct=progress_pct,
        )
