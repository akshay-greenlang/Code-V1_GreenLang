"""
Opportunity Prioritizer for GL-006 HeatRecoveryMaximizer

Multi-criteria decision analysis for prioritizing heat recovery opportunities.
Ranks opportunities based on:
- Financial metrics (ROI, payback, NPV)
- Technical feasibility
- Implementation complexity
- Environmental impact
- Strategic alignment

Uses weighted scoring and Pareto frontier analysis.

References:
- Multi-Criteria Decision Analysis (Belton & Stewart)
- Analytic Hierarchy Process (Saaty)
- Project Portfolio Management

Author: GreenLang AI Agent Factory
Created: 2025-11-19
Version: 1.0.0
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PriorityLevel(str, Enum):
    """Priority levels"""
    CRITICAL = "critical"  # Must do immediately
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEFERRED = "deferred"  # Not currently viable


class RiskLevel(str, Enum):
    """Implementation risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class OpportunityInput(BaseModel):
    """Input for a single heat recovery opportunity"""
    opportunity_id: str = Field(..., description="Unique ID")
    opportunity_name: str = Field(..., description="Descriptive name")

    # Financial metrics
    capital_cost_usd: float = Field(..., gt=0, description="Capital cost $")
    annual_savings_usd: float = Field(..., gt=0, description="Annual savings $")
    roi_percent: float = Field(..., description="ROI %")
    payback_years: float = Field(..., gt=0, description="Payback years")
    npv_usd: float = Field(..., description="NPV $")
    irr_percent: float = Field(..., description="IRR %")

    # Technical metrics
    heat_recovery_kw: float = Field(..., gt=0, description="Heat recovery kW")
    technical_feasibility_score: float = Field(..., ge=0, le=100, description="Feasibility 0-100")
    implementation_complexity: str = Field("medium", description="low/medium/high/very_high")

    # Environmental metrics
    annual_co2_reduction_tonnes: float = Field(..., ge=0, description="CO2 reduction tonnes/year")
    annual_energy_reduction_mwh: float = Field(..., ge=0, description="Energy reduction MWh/year")

    # Strategic metrics
    strategic_alignment_score: float = Field(50, ge=0, le=100, description="Strategic alignment 0-100")
    enables_other_projects: bool = Field(False, description="Enables future projects")
    regulatory_driver: bool = Field(False, description="Regulatory compliance driver")

    # Implementation metrics
    estimated_implementation_months: int = Field(..., gt=0, description="Implementation time months")
    requires_shutdown: bool = Field(False, description="Requires plant shutdown")
    dependencies: List[str] = Field(default_factory=list, description="Dependent opportunity IDs")


class PrioritizationWeights(BaseModel):
    """Weights for multi-criteria scoring (must sum to 1.0)"""
    financial_weight: float = Field(0.40, ge=0, le=1, description="Financial metrics weight")
    technical_weight: float = Field(0.20, ge=0, le=1, description="Technical feasibility weight")
    environmental_weight: float = Field(0.15, ge=0, le=1, description="Environmental impact weight")
    strategic_weight: float = Field(0.15, ge=0, le=1, description="Strategic alignment weight")
    implementation_weight: float = Field(0.10, ge=0, le=1, description="Implementation ease weight")

    @property
    def total_weight(self) -> float:
        return (self.financial_weight + self.technical_weight +
                self.environmental_weight + self.strategic_weight +
                self.implementation_weight)


class PrioritizedOpportunity(BaseModel):
    """Prioritization result for a single opportunity"""
    opportunity_id: str
    opportunity_name: str

    # Scores (0-100)
    financial_score: float = Field(..., ge=0, le=100)
    technical_score: float = Field(..., ge=0, le=100)
    environmental_score: float = Field(..., ge=0, le=100)
    strategic_score: float = Field(..., ge=0, le=100)
    implementation_score: float = Field(..., ge=0, le=100)

    # Overall
    total_score: float = Field(..., ge=0, le=100, description="Weighted total score")
    priority_level: PriorityLevel
    risk_level: RiskLevel

    # Ranking
    rank: int = Field(..., gt=0, description="Priority rank (1=highest)")
    percentile: float = Field(..., ge=0, le=100, description="Score percentile")

    # Recommendations
    recommended_phase: str = Field(..., description="immediate/short_term/medium_term/long_term")
    should_proceed: bool = Field(..., description="Proceed with implementation")
    blocking_issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class PortfolioAnalysis(BaseModel):
    """Portfolio-level analysis results"""
    total_opportunities: int
    total_capital_required_usd: float
    total_annual_savings_usd: float
    portfolio_roi_percent: float
    portfolio_payback_years: float

    # Recommended subset
    recommended_opportunities: List[str] = Field(default_factory=list)
    recommended_capital_usd: float = Field(0)
    recommended_savings_usd: float = Field(0)

    # Pareto analysis
    pareto_frontier: List[str] = Field(default_factory=list, description="Pareto-optimal opportunities")

    # Risk distribution
    low_risk_count: int = Field(0)
    medium_risk_count: int = Field(0)
    high_risk_count: int = Field(0)


class OpportunityPrioritizer:
    """
    Prioritize heat recovery opportunities using multi-criteria analysis.

    Zero-hallucination approach:
    - Transparent scoring algorithms
    - Configurable weights
    - Explainable decisions
    """

    def __init__(self, weights: Optional[PrioritizationWeights] = None):
        """
        Initialize prioritizer.

        Args:
            weights: Custom prioritization weights (defaults to balanced)
        """
        self.weights = weights or PrioritizationWeights()
        self.logger = logging.getLogger(__name__)

        # Validate weights sum to 1.0
        if abs(self.weights.total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {self.weights.total_weight}")

    def prioritize(
        self,
        opportunities: List[OpportunityInput],
        budget_constraint_usd: Optional[float] = None
    ) -> Tuple[List[PrioritizedOpportunity], PortfolioAnalysis]:
        """
        Prioritize list of opportunities.

        Args:
            opportunities: List of opportunities to prioritize
            budget_constraint_usd: Optional budget constraint

        Returns:
            Tuple of (prioritized opportunities, portfolio analysis)
        """
        self.logger.info(f"Prioritizing {len(opportunities)} opportunities")

        # 1. Score each opportunity
        scored_opportunities = []
        for opp in opportunities:
            scored = self._score_opportunity(opp)
            scored_opportunities.append(scored)

        # 2. Rank by total score
        scored_opportunities.sort(key=lambda x: x.total_score, reverse=True)

        # 3. Assign ranks and percentiles
        for i, opp in enumerate(scored_opportunities):
            opp.rank = i + 1
            opp.percentile = (1 - i / len(scored_opportunities)) * 100

        # 4. Identify Pareto frontier
        pareto_frontier = self._identify_pareto_frontier(opportunities)

        # 5. Portfolio analysis
        portfolio = self._analyze_portfolio(
            opportunities,
            scored_opportunities,
            pareto_frontier,
            budget_constraint_usd
        )

        return scored_opportunities, portfolio

    def _score_opportunity(self, opp: OpportunityInput) -> PrioritizedOpportunity:
        """Score a single opportunity across all criteria"""

        # 1. Financial score (0-100)
        financial_score = self._calculate_financial_score(
            opp.roi_percent,
            opp.payback_years,
            opp.npv_usd
        )

        # 2. Technical score
        technical_score = opp.technical_feasibility_score

        # 3. Environmental score
        environmental_score = self._calculate_environmental_score(
            opp.annual_co2_reduction_tonnes,
            opp.annual_energy_reduction_mwh
        )

        # 4. Strategic score
        strategic_score = self._calculate_strategic_score(
            opp.strategic_alignment_score,
            opp.enables_other_projects,
            opp.regulatory_driver
        )

        # 5. Implementation score
        implementation_score = self._calculate_implementation_score(
            opp.implementation_complexity,
            opp.estimated_implementation_months,
            opp.requires_shutdown,
            len(opp.dependencies)
        )

        # 6. Calculate weighted total
        total_score = (
            financial_score * self.weights.financial_weight +
            technical_score * self.weights.technical_weight +
            environmental_score * self.weights.environmental_weight +
            strategic_score * self.weights.strategic_weight +
            implementation_score * self.weights.implementation_weight
        )

        # 7. Determine priority level
        priority_level = self._determine_priority_level(total_score, opp)

        # 8. Determine risk level
        risk_level = self._determine_risk_level(opp)

        # 9. Generate recommendations
        should_proceed, blocking_issues, recommendations = self._generate_recommendations(
            opp, total_score, technical_score, financial_score
        )

        # 10. Determine recommended phase
        recommended_phase = self._determine_phase(opp, priority_level)

        return PrioritizedOpportunity(
            opportunity_id=opp.opportunity_id,
            opportunity_name=opp.opportunity_name,
            financial_score=financial_score,
            technical_score=technical_score,
            environmental_score=environmental_score,
            strategic_score=strategic_score,
            implementation_score=implementation_score,
            total_score=total_score,
            priority_level=priority_level,
            risk_level=risk_level,
            rank=0,  # Will be set in prioritize()
            percentile=0,  # Will be set in prioritize()
            recommended_phase=recommended_phase,
            should_proceed=should_proceed,
            blocking_issues=blocking_issues,
            recommendations=recommendations
        )

    def _calculate_financial_score(
        self,
        roi: float,
        payback: float,
        npv: float
    ) -> float:
        """Calculate financial score (0-100)"""

        # ROI component (0-50 points)
        if roi >= 50:
            roi_points = 50
        elif roi >= 30:
            roi_points = 40
        elif roi >= 20:
            roi_points = 30
        elif roi >= 10:
            roi_points = 20
        else:
            roi_points = max(0, roi / 10 * 20)

        # Payback component (0-30 points)
        if payback <= 2:
            payback_points = 30
        elif payback <= 3:
            payback_points = 25
        elif payback <= 5:
            payback_points = 15
        elif payback <= 10:
            payback_points = 5
        else:
            payback_points = 0

        # NPV component (0-20 points)
        if npv > 1000000:
            npv_points = 20
        elif npv > 500000:
            npv_points = 15
        elif npv > 100000:
            npv_points = 10
        elif npv > 0:
            npv_points = 5
        else:
            npv_points = 0

        return min(100, roi_points + payback_points + npv_points)

    def _calculate_environmental_score(
        self,
        co2_reduction: float,
        energy_reduction: float
    ) -> float:
        """Calculate environmental impact score (0-100)"""

        # CO2 reduction (0-60 points)
        if co2_reduction >= 1000:
            co2_points = 60
        elif co2_reduction >= 500:
            co2_points = 50
        elif co2_reduction >= 100:
            co2_points = 40
        elif co2_reduction >= 50:
            co2_points = 30
        elif co2_reduction >= 10:
            co2_points = 20
        else:
            co2_points = max(0, co2_reduction / 10 * 20)

        # Energy reduction (0-40 points)
        if energy_reduction >= 5000:
            energy_points = 40
        elif energy_reduction >= 2000:
            energy_points = 30
        elif energy_reduction >= 1000:
            energy_points = 20
        elif energy_reduction >= 500:
            energy_points = 10
        else:
            energy_points = max(0, energy_reduction / 500 * 10)

        return min(100, co2_points + energy_points)

    def _calculate_strategic_score(
        self,
        alignment: float,
        enables_others: bool,
        regulatory: bool
    ) -> float:
        """Calculate strategic alignment score (0-100)"""

        # Base alignment score (0-70 points)
        strategic_score = alignment * 0.7

        # Bonus for enabling other projects
        if enables_others:
            strategic_score += 15

        # Bonus for regulatory driver
        if regulatory:
            strategic_score += 15

        return min(100, strategic_score)

    def _calculate_implementation_score(
        self,
        complexity: str,
        months: int,
        shutdown_required: bool,
        num_dependencies: int
    ) -> float:
        """Calculate implementation ease score (0-100)"""

        # Complexity factor (0-40 points)
        complexity_map = {
            'low': 40,
            'medium': 25,
            'high': 10,
            'very_high': 0
        }
        complexity_points = complexity_map.get(complexity.lower(), 25)

        # Timeline factor (0-30 points)
        if months <= 3:
            timeline_points = 30
        elif months <= 6:
            timeline_points = 25
        elif months <= 12:
            timeline_points = 15
        elif months <= 24:
            timeline_points = 5
        else:
            timeline_points = 0

        # Shutdown penalty (-20 points)
        shutdown_penalty = -20 if shutdown_required else 0

        # Dependencies penalty (-5 points each)
        dependency_penalty = -min(20, num_dependencies * 5)

        score = complexity_points + timeline_points + shutdown_penalty + dependency_penalty
        return max(0, min(100, score))

    def _determine_priority_level(
        self,
        total_score: float,
        opp: OpportunityInput
    ) -> PriorityLevel:
        """Determine priority level based on score and other factors"""

        # Critical: Very high ROI + high score
        if total_score >= 80 and opp.roi_percent >= 40:
            return PriorityLevel.CRITICAL

        # High: Good score + reasonable financials
        elif total_score >= 65:
            return PriorityLevel.HIGH

        # Medium: Moderate score
        elif total_score >= 50:
            return PriorityLevel.MEDIUM

        # Low: Lower score but positive
        elif total_score >= 35:
            return PriorityLevel.LOW

        # Deferred: Very low score or negative NPV
        else:
            return PriorityLevel.DEFERRED

    def _determine_risk_level(self, opp: OpportunityInput) -> RiskLevel:
        """Determine implementation risk level"""

        risk_factors = 0

        # High complexity
        if opp.implementation_complexity in ['high', 'very_high']:
            risk_factors += 2

        # Low feasibility
        if opp.technical_feasibility_score < 60:
            risk_factors += 2

        # Long payback
        if opp.payback_years > 5:
            risk_factors += 1

        # Shutdown required
        if opp.requires_shutdown:
            risk_factors += 1

        # Dependencies
        if len(opp.dependencies) > 2:
            risk_factors += 1

        # Classify risk
        if risk_factors >= 5:
            return RiskLevel.VERY_HIGH
        elif risk_factors >= 3:
            return RiskLevel.HIGH
        elif risk_factors >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _generate_recommendations(
        self,
        opp: OpportunityInput,
        total_score: float,
        technical_score: float,
        financial_score: float
    ) -> Tuple[bool, List[str], List[str]]:
        """Generate proceed decision, blocking issues, and recommendations"""

        blocking_issues = []
        recommendations = []
        should_proceed = True

        # Check blocking issues
        if opp.technical_feasibility_score < 50:
            blocking_issues.append("Technical feasibility too low (<50)")
            should_proceed = False

        if opp.npv_usd < 0:
            blocking_issues.append("Negative NPV")
            should_proceed = False

        if opp.payback_years > 15:
            blocking_issues.append("Payback period too long (>15 years)")
            should_proceed = False

        # Generate recommendations
        if financial_score < 50:
            recommendations.append("Consider cost optimization or energy price escalation")

        if technical_score < 70:
            recommendations.append("Conduct detailed technical study before proceeding")

        if opp.requires_shutdown:
            recommendations.append("Schedule during planned shutdown to minimize impact")

        if len(opp.dependencies) > 0:
            recommendations.append(f"Coordinate with {len(opp.dependencies)} dependent projects")

        if opp.roi_percent > 40:
            recommendations.append("High ROI - prioritize for immediate implementation")

        return should_proceed, blocking_issues, recommendations

    def _determine_phase(
        self,
        opp: OpportunityInput,
        priority: PriorityLevel
    ) -> str:
        """Determine recommended implementation phase"""

        if priority == PriorityLevel.CRITICAL:
            return "immediate"
        elif priority == PriorityLevel.HIGH:
            if opp.payback_years <= 2:
                return "immediate"
            else:
                return "short_term"
        elif priority == PriorityLevel.MEDIUM:
            return "medium_term"
        elif priority == PriorityLevel.LOW:
            return "long_term"
        else:
            return "deferred"

    def _identify_pareto_frontier(
        self,
        opportunities: List[OpportunityInput]
    ) -> List[str]:
        """
        Identify Pareto-optimal opportunities (ROI vs. Capital Cost).

        An opportunity is Pareto-optimal if no other opportunity has both
        higher ROI and lower capital cost.
        """
        pareto = []

        for opp in opportunities:
            is_dominated = False

            for other in opportunities:
                if other.opportunity_id == opp.opportunity_id:
                    continue

                # Check if 'other' dominates 'opp'
                if (other.roi_percent >= opp.roi_percent and
                    other.capital_cost_usd <= opp.capital_cost_usd and
                    (other.roi_percent > opp.roi_percent or
                     other.capital_cost_usd < opp.capital_cost_usd)):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto.append(opp.opportunity_id)

        return pareto

    def _analyze_portfolio(
        self,
        opportunities: List[OpportunityInput],
        scored: List[PrioritizedOpportunity],
        pareto: List[str],
        budget: Optional[float]
    ) -> PortfolioAnalysis:
        """Analyze overall portfolio"""

        total_capital = sum(o.capital_cost_usd for o in opportunities)
        total_savings = sum(o.annual_savings_usd for o in opportunities)

        portfolio_roi = (total_savings / total_capital * 100) if total_capital > 0 else 0
        portfolio_payback = total_capital / total_savings if total_savings > 0 else 999

        # Recommended subset (proceed = True and within budget)
        recommended = []
        recommended_capital = 0
        recommended_savings = 0

        for scored_opp in scored:
            if not scored_opp.should_proceed:
                continue

            # Find original opportunity
            orig_opp = next((o for o in opportunities if o.opportunity_id == scored_opp.opportunity_id), None)
            if not orig_opp:
                continue

            # Budget constraint check
            if budget is not None:
                if recommended_capital + orig_opp.capital_cost_usd > budget:
                    continue

            recommended.append(scored_opp.opportunity_id)
            recommended_capital += orig_opp.capital_cost_usd
            recommended_savings += orig_opp.annual_savings_usd

        # Risk distribution
        low_risk = sum(1 for s in scored if s.risk_level == RiskLevel.LOW)
        medium_risk = sum(1 for s in scored if s.risk_level == RiskLevel.MEDIUM)
        high_risk = sum(1 for s in scored if s.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH])

        return PortfolioAnalysis(
            total_opportunities=len(opportunities),
            total_capital_required_usd=total_capital,
            total_annual_savings_usd=total_savings,
            portfolio_roi_percent=portfolio_roi,
            portfolio_payback_years=portfolio_payback,
            recommended_opportunities=recommended,
            recommended_capital_usd=recommended_capital,
            recommended_savings_usd=recommended_savings,
            pareto_frontier=pareto,
            low_risk_count=low_risk,
            medium_risk_count=medium_risk,
            high_risk_count=high_risk
        )


# Example usage
if __name__ == "__main__":
    # Custom weights (financial focus)
    weights = PrioritizationWeights(
        financial_weight=0.50,
        technical_weight=0.20,
        environmental_weight=0.10,
        strategic_weight=0.10,
        implementation_weight=0.10
    )

    prioritizer = OpportunityPrioritizer(weights)

    # Example opportunities
    opps = [
        OpportunityInput(
            opportunity_id="HX-001",
            opportunity_name="Flue Gas to Feedwater HX",
            capital_cost_usd=150000,
            annual_savings_usd=75000,
            roi_percent=50,
            payback_years=2.0,
            npv_usd=450000,
            irr_percent=48,
            heat_recovery_kw=500,
            technical_feasibility_score=85,
            implementation_complexity="medium",
            annual_co2_reduction_tonnes=200,
            annual_energy_reduction_mwh=4000,
            estimated_implementation_months=4
        ),
        OpportunityInput(
            opportunity_id="HX-002",
            opportunity_name="Process to Process HX",
            capital_cost_usd=80000,
            annual_savings_usd=25000,
            roi_percent=31,
            payback_years=3.2,
            npv_usd=120000,
            irr_percent=28,
            heat_recovery_kw=200,
            technical_feasibility_score=90,
            implementation_complexity="low",
            annual_co2_reduction_tonnes=80,
            annual_energy_reduction_mwh=1600,
            estimated_implementation_months=2
        )
    ]

    prioritized, portfolio = prioritizer.prioritize(opps, budget_constraint_usd=200000)

    print("Prioritization Results:")
    for p in prioritized:
        print(f"\n{p.rank}. {p.opportunity_name} ({p.opportunity_id})")
        print(f"   Total Score: {p.total_score:.1f}/100 (Top {p.percentile:.0f}%)")
        print(f"   Priority: {p.priority_level.value.upper()}, Risk: {p.risk_level.value}")
        print(f"   Phase: {p.recommended_phase}")
        print(f"   Proceed: {p.should_proceed}")
        if p.recommendations:
            print(f"   Recommendations: {', '.join(p.recommendations)}")

    print(f"\nPortfolio Analysis:")
    print(f"  Total Opportunities: {portfolio.total_opportunities}")
    print(f"  Total Capital: ${portfolio.total_capital_required_usd:,.0f}")
    print(f"  Total Savings: ${portfolio.total_annual_savings_usd:,.0f}/year")
    print(f"  Portfolio ROI: {portfolio.portfolio_roi_percent:.1f}%")
    print(f"  Recommended (within budget): {len(portfolio.recommended_opportunities)} projects")
    print(f"  Pareto Frontier: {len(portfolio.pareto_frontier)} projects")
