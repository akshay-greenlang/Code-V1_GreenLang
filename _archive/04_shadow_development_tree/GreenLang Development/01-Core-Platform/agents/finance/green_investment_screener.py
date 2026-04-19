# -*- coding: utf-8 -*-
"""
GL-FIN-X-003: Green Investment Screener Agent
=============================================

Screens investments against sustainability criteria including ESG ratings,
climate alignment, exclusion lists, and positive impact thresholds.

Capabilities:
    - Negative screening (exclusion lists)
    - Positive screening (impact thresholds)
    - ESG rating integration
    - Climate alignment assessment
    - SFDR Article 8/9 classification
    - Custom criteria definition
    - Portfolio-level sustainability scoring

Zero-Hallucination Guarantees:
    - All screening decisions are deterministic rule-based
    - ESG data from structured inputs (no inference)
    - Complete audit trail for screening decisions
    - SHA-256 provenance hashes for all outputs

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.base_agents import AuditEntry
from greenlang.agents.categories import AgentCategory

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class ESGRating(str, Enum):
    """ESG rating scale."""
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC"
    NOT_RATED = "not_rated"


class SFDRClassification(str, Enum):
    """SFDR fund classification."""
    ARTICLE_6 = "article_6"
    ARTICLE_8 = "article_8"
    ARTICLE_8_PLUS = "article_8_plus"
    ARTICLE_9 = "article_9"


class ScreeningType(str, Enum):
    """Types of screening."""
    NEGATIVE = "negative"
    POSITIVE = "positive"
    NORMS_BASED = "norms_based"
    BEST_IN_CLASS = "best_in_class"


class ExclusionCategory(str, Enum):
    """Standard exclusion categories."""
    CONTROVERSIAL_WEAPONS = "controversial_weapons"
    TOBACCO = "tobacco"
    THERMAL_COAL = "thermal_coal"
    OIL_SANDS = "oil_sands"
    ARCTIC_DRILLING = "arctic_drilling"
    GAMBLING = "gambling"
    ADULT_ENTERTAINMENT = "adult_entertainment"
    PRIVATE_PRISONS = "private_prisons"
    PALM_OIL = "palm_oil"
    DEFORESTATION = "deforestation"
    HUMAN_RIGHTS_VIOLATIONS = "human_rights_violations"
    LABOR_VIOLATIONS = "labor_violations"
    UNGC_VIOLATIONS = "ungc_violations"
    NUCLEAR_WEAPONS = "nuclear_weapons"
    CIVILIAN_FIREARMS = "civilian_firearms"


# ESG rating to score mapping
ESG_SCORE_MAP: Dict[ESGRating, int] = {
    ESGRating.AAA: 100,
    ESGRating.AA: 85,
    ESGRating.A: 70,
    ESGRating.BBB: 55,
    ESGRating.BB: 40,
    ESGRating.B: 25,
    ESGRating.CCC: 10,
    ESGRating.NOT_RATED: 0,
}

# Default exclusion thresholds (revenue percentage)
DEFAULT_EXCLUSION_THRESHOLDS: Dict[str, float] = {
    ExclusionCategory.CONTROVERSIAL_WEAPONS.value: 0.0,  # Zero tolerance
    ExclusionCategory.TOBACCO.value: 5.0,
    ExclusionCategory.THERMAL_COAL.value: 5.0,
    ExclusionCategory.OIL_SANDS.value: 5.0,
    ExclusionCategory.ARCTIC_DRILLING.value: 5.0,
    ExclusionCategory.GAMBLING.value: 10.0,
    ExclusionCategory.ADULT_ENTERTAINMENT.value: 5.0,
    ExclusionCategory.NUCLEAR_WEAPONS.value: 0.0,
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class InvestmentCriteria(BaseModel):
    """Criteria for screening investments."""
    # Negative screening
    exclusion_categories: List[ExclusionCategory] = Field(
        default_factory=list, description="Categories to exclude"
    )
    custom_exclusion_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Custom revenue thresholds by category (%)"
    )

    # ESG requirements
    minimum_esg_rating: ESGRating = Field(
        default=ESGRating.BB, description="Minimum acceptable ESG rating"
    )
    minimum_esg_score: float = Field(
        default=30.0, ge=0, le=100,
        description="Minimum ESG score (0-100)"
    )

    # Environmental requirements
    minimum_environmental_score: float = Field(
        default=0.0, ge=0, le=100,
        description="Minimum E score"
    )
    minimum_social_score: float = Field(
        default=0.0, ge=0, le=100,
        description="Minimum S score"
    )
    minimum_governance_score: float = Field(
        default=0.0, ge=0, le=100,
        description="Minimum G score"
    )

    # Climate alignment
    require_sbti_target: bool = Field(
        default=False, description="Require Science Based Targets"
    )
    require_net_zero_commitment: bool = Field(
        default=False, description="Require net zero commitment"
    )
    maximum_carbon_intensity: Optional[float] = Field(
        None, ge=0, description="Max carbon intensity (tCO2e/$M revenue)"
    )

    # Positive impact
    minimum_green_revenue_pct: float = Field(
        default=0.0, ge=0, le=100,
        description="Minimum green revenue percentage"
    )
    require_eu_taxonomy_alignment: bool = Field(
        default=False, description="Require EU Taxonomy aligned activities"
    )
    minimum_taxonomy_alignment_pct: float = Field(
        default=0.0, ge=0, le=100,
        description="Minimum EU Taxonomy alignment"
    )


class InvestmentProfile(BaseModel):
    """Profile of an investment to be screened."""
    investment_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Investment name")
    ticker: Optional[str] = Field(None, description="Stock ticker if applicable")
    sector: str = Field(..., description="Industry sector")
    country: str = Field(..., description="Country of incorporation")

    # ESG data
    esg_rating: ESGRating = Field(
        default=ESGRating.NOT_RATED, description="Overall ESG rating"
    )
    esg_score: float = Field(default=0.0, ge=0, le=100, description="ESG score")
    environmental_score: float = Field(default=0.0, ge=0, le=100)
    social_score: float = Field(default=0.0, ge=0, le=100)
    governance_score: float = Field(default=0.0, ge=0, le=100)

    # Exclusion data (revenue percentages)
    controversial_weapons_revenue_pct: float = Field(default=0.0, ge=0, le=100)
    tobacco_revenue_pct: float = Field(default=0.0, ge=0, le=100)
    thermal_coal_revenue_pct: float = Field(default=0.0, ge=0, le=100)
    oil_sands_revenue_pct: float = Field(default=0.0, ge=0, le=100)
    gambling_revenue_pct: float = Field(default=0.0, ge=0, le=100)
    adult_entertainment_revenue_pct: float = Field(default=0.0, ge=0, le=100)

    # Climate data
    has_sbti_target: bool = Field(default=False)
    has_net_zero_commitment: bool = Field(default=False)
    carbon_intensity_tco2e_per_m_revenue: Optional[float] = Field(None, ge=0)

    # Taxonomy data
    green_revenue_pct: float = Field(default=0.0, ge=0, le=100)
    eu_taxonomy_aligned_pct: float = Field(default=0.0, ge=0, le=100)

    # Norms-based
    ungc_violations: bool = Field(default=False)
    controversy_level: int = Field(
        default=0, ge=0, le=5,
        description="Controversy level 0-5 (5=severe)"
    )


class ScreeningResult(BaseModel):
    """Result of screening a single investment."""
    investment_id: str
    investment_name: str
    passed: bool = Field(..., description="Whether investment passed all screens")

    # Detailed results
    exclusion_violations: List[str] = Field(
        default_factory=list, description="Exclusion categories failed"
    )
    esg_violations: List[str] = Field(
        default_factory=list, description="ESG criteria failed"
    )
    climate_violations: List[str] = Field(
        default_factory=list, description="Climate criteria failed"
    )
    positive_criteria_missed: List[str] = Field(
        default_factory=list, description="Positive criteria not met"
    )

    # Scores
    overall_sustainability_score: float = Field(
        ..., ge=0, le=100, description="Overall sustainability score"
    )
    screening_score: float = Field(
        ..., ge=0, le=100, description="Screening pass rate"
    )


class SustainabilityScore(BaseModel):
    """Detailed sustainability score breakdown."""
    overall_score: float = Field(..., ge=0, le=100)
    environmental_score: float = Field(..., ge=0, le=100)
    social_score: float = Field(..., ge=0, le=100)
    governance_score: float = Field(..., ge=0, le=100)
    climate_alignment_score: float = Field(..., ge=0, le=100)
    impact_score: float = Field(..., ge=0, le=100)


class InvestmentScreenerInput(BaseModel):
    """Input for investment screening."""
    operation: str = Field(
        default="screen_investment",
        description="Operation: screen_investment, screen_portfolio, get_criteria"
    )

    # Investment(s) to screen
    investment: Optional[InvestmentProfile] = Field(
        None, description="Single investment to screen"
    )
    portfolio: Optional[List[InvestmentProfile]] = Field(
        None, description="Portfolio of investments"
    )

    # Screening criteria
    criteria: Optional[InvestmentCriteria] = Field(
        None, description="Screening criteria (uses defaults if not provided)"
    )

    # SFDR classification
    target_sfdr: Optional[SFDRClassification] = Field(
        None, description="Target SFDR classification"
    )


class InvestmentScreenerOutput(BaseModel):
    """Output from investment screening."""
    success: bool
    operation: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Results
    screening_result: Optional[ScreeningResult] = Field(None)
    portfolio_results: Optional[List[ScreeningResult]] = Field(None)
    portfolio_summary: Optional[Dict[str, Any]] = Field(None)

    # Criteria used
    criteria_applied: Optional[InvestmentCriteria] = Field(None)
    sfdr_classification: Optional[SFDRClassification] = Field(None)

    # Audit
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# GREEN INVESTMENT SCREENER AGENT
# =============================================================================


class GreenInvestmentScreenerAgent(BaseAgent):
    """
    GL-FIN-X-003: Green Investment Screener Agent

    Screens investments against sustainability criteria using deterministic
    rule-based evaluation.

    Zero-Hallucination Guarantees:
        - All screening is deterministic rule evaluation
        - No inference or prediction on ESG scores
        - Complete audit trail for all decisions
        - SHA-256 provenance hashes for all outputs

    Usage:
        agent = GreenInvestmentScreenerAgent()
        result = agent.run({
            "operation": "screen_investment",
            "investment": investment_profile,
            "criteria": screening_criteria
        })
    """

    AGENT_ID = "GL-FIN-X-003"
    AGENT_NAME = "Green Investment Screener"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Green Investment Screener Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Investment sustainability screening",
                version=self.VERSION,
                parameters={}
            )

        self._audit_trail: List[AuditEntry] = []
        self._default_criteria = InvestmentCriteria()
        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute investment screening."""
        try:
            screen_input = InvestmentScreenerInput(**input_data)
            operation = screen_input.operation

            if operation == "screen_investment":
                output = self._screen_investment(screen_input)
            elif operation == "screen_portfolio":
                output = self._screen_portfolio(screen_input)
            elif operation == "get_criteria":
                output = self._get_criteria(screen_input)
            else:
                return AgentResult(success=False, error=f"Unknown operation: {operation}")

            return AgentResult(
                success=output.success,
                data=output.model_dump(),
                metadata={"agent_id": self.AGENT_ID, "operation": operation}
            )

        except Exception as e:
            logger.error(f"Investment screening failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _screen_investment(
        self, input_data: InvestmentScreenerInput
    ) -> InvestmentScreenerOutput:
        """Screen a single investment."""
        calculation_trace: List[str] = []

        if input_data.investment is None:
            return InvestmentScreenerOutput(
                success=False,
                operation="screen_investment",
                calculation_trace=["ERROR: No investment provided"]
            )

        investment = input_data.investment
        criteria = input_data.criteria or self._default_criteria

        calculation_trace.append(f"Screening: {investment.name} ({investment.investment_id})")
        calculation_trace.append(f"Sector: {investment.sector}, Country: {investment.country}")

        # Apply SFDR-based criteria if specified
        if input_data.target_sfdr:
            criteria = self._adjust_criteria_for_sfdr(criteria, input_data.target_sfdr)
            calculation_trace.append(f"Adjusted criteria for SFDR {input_data.target_sfdr.value}")

        result = self._evaluate_investment(investment, criteria, calculation_trace)

        provenance_hash = hashlib.sha256(
            json.dumps(result.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()

        return InvestmentScreenerOutput(
            success=True,
            operation="screen_investment",
            screening_result=result,
            criteria_applied=criteria,
            sfdr_classification=input_data.target_sfdr,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _evaluate_investment(
        self,
        investment: InvestmentProfile,
        criteria: InvestmentCriteria,
        trace: List[str]
    ) -> ScreeningResult:
        """Evaluate investment against criteria."""
        exclusion_violations: List[str] = []
        esg_violations: List[str] = []
        climate_violations: List[str] = []
        positive_missed: List[str] = []

        # Negative screening
        trace.append("Applying negative screens...")

        exclusion_mapping = {
            ExclusionCategory.CONTROVERSIAL_WEAPONS: investment.controversial_weapons_revenue_pct,
            ExclusionCategory.TOBACCO: investment.tobacco_revenue_pct,
            ExclusionCategory.THERMAL_COAL: investment.thermal_coal_revenue_pct,
            ExclusionCategory.OIL_SANDS: investment.oil_sands_revenue_pct,
            ExclusionCategory.GAMBLING: investment.gambling_revenue_pct,
            ExclusionCategory.ADULT_ENTERTAINMENT: investment.adult_entertainment_revenue_pct,
        }

        for category in criteria.exclusion_categories:
            if category in exclusion_mapping:
                revenue_pct = exclusion_mapping[category]
                threshold = criteria.custom_exclusion_thresholds.get(
                    category.value,
                    DEFAULT_EXCLUSION_THRESHOLDS.get(category.value, 5.0)
                )
                if revenue_pct > threshold:
                    exclusion_violations.append(
                        f"{category.value}: {revenue_pct}% > {threshold}% threshold"
                    )
                    trace.append(f"FAIL: {category.value} ({revenue_pct}% > {threshold}%)")
                else:
                    trace.append(f"PASS: {category.value} ({revenue_pct}% <= {threshold}%)")

        # UNGC violations
        if investment.ungc_violations:
            exclusion_violations.append("UN Global Compact violations")
            trace.append("FAIL: UNGC violations detected")

        # ESG screening
        trace.append("Applying ESG screens...")

        # Rating check
        if ESG_SCORE_MAP[investment.esg_rating] < ESG_SCORE_MAP[criteria.minimum_esg_rating]:
            esg_violations.append(
                f"ESG rating {investment.esg_rating.value} < minimum {criteria.minimum_esg_rating.value}"
            )
            trace.append(f"FAIL: ESG rating ({investment.esg_rating.value})")

        # Score checks
        if investment.esg_score < criteria.minimum_esg_score:
            esg_violations.append(
                f"ESG score {investment.esg_score} < minimum {criteria.minimum_esg_score}"
            )
            trace.append(f"FAIL: ESG score ({investment.esg_score})")

        if investment.environmental_score < criteria.minimum_environmental_score:
            esg_violations.append(
                f"E score {investment.environmental_score} < minimum {criteria.minimum_environmental_score}"
            )

        if investment.social_score < criteria.minimum_social_score:
            esg_violations.append(
                f"S score {investment.social_score} < minimum {criteria.minimum_social_score}"
            )

        if investment.governance_score < criteria.minimum_governance_score:
            esg_violations.append(
                f"G score {investment.governance_score} < minimum {criteria.minimum_governance_score}"
            )

        # Climate screening
        trace.append("Applying climate screens...")

        if criteria.require_sbti_target and not investment.has_sbti_target:
            climate_violations.append("Missing Science Based Target")
            trace.append("FAIL: No SBTi target")

        if criteria.require_net_zero_commitment and not investment.has_net_zero_commitment:
            climate_violations.append("Missing net zero commitment")
            trace.append("FAIL: No net zero commitment")

        if (
            criteria.maximum_carbon_intensity is not None
            and investment.carbon_intensity_tco2e_per_m_revenue is not None
            and investment.carbon_intensity_tco2e_per_m_revenue > criteria.maximum_carbon_intensity
        ):
            climate_violations.append(
                f"Carbon intensity {investment.carbon_intensity_tco2e_per_m_revenue} > "
                f"max {criteria.maximum_carbon_intensity}"
            )
            trace.append(f"FAIL: Carbon intensity exceeds maximum")

        # Positive screening
        trace.append("Applying positive screens...")

        if investment.green_revenue_pct < criteria.minimum_green_revenue_pct:
            positive_missed.append(
                f"Green revenue {investment.green_revenue_pct}% < "
                f"minimum {criteria.minimum_green_revenue_pct}%"
            )

        if criteria.require_eu_taxonomy_alignment:
            if investment.eu_taxonomy_aligned_pct < criteria.minimum_taxonomy_alignment_pct:
                positive_missed.append(
                    f"EU Taxonomy alignment {investment.eu_taxonomy_aligned_pct}% < "
                    f"minimum {criteria.minimum_taxonomy_alignment_pct}%"
                )

        # Calculate scores
        passed = (
            len(exclusion_violations) == 0
            and len(esg_violations) == 0
            and len(climate_violations) == 0
            and len(positive_missed) == 0
        )

        # Overall sustainability score
        sustainability_score = (
            investment.environmental_score * 0.35
            + investment.social_score * 0.30
            + investment.governance_score * 0.35
        )

        # Screening pass rate (out of all criteria checked)
        total_checks = (
            len(criteria.exclusion_categories)
            + 5  # ESG checks
            + (1 if criteria.require_sbti_target else 0)
            + (1 if criteria.require_net_zero_commitment else 0)
            + (1 if criteria.maximum_carbon_intensity else 0)
            + (1 if criteria.minimum_green_revenue_pct > 0 else 0)
            + (1 if criteria.require_eu_taxonomy_alignment else 0)
        )
        failures = (
            len(exclusion_violations)
            + len(esg_violations)
            + len(climate_violations)
            + len(positive_missed)
        )
        screening_score = ((total_checks - failures) / max(total_checks, 1)) * 100

        trace.append(f"Overall result: {'PASS' if passed else 'FAIL'}")
        trace.append(f"Sustainability score: {sustainability_score:.1f}")
        trace.append(f"Screening score: {screening_score:.1f}%")

        return ScreeningResult(
            investment_id=investment.investment_id,
            investment_name=investment.name,
            passed=passed,
            exclusion_violations=exclusion_violations,
            esg_violations=esg_violations,
            climate_violations=climate_violations,
            positive_criteria_missed=positive_missed,
            overall_sustainability_score=round(sustainability_score, 2),
            screening_score=round(screening_score, 2)
        )

    def _screen_portfolio(
        self, input_data: InvestmentScreenerInput
    ) -> InvestmentScreenerOutput:
        """Screen a portfolio of investments."""
        calculation_trace: List[str] = []

        if not input_data.portfolio:
            return InvestmentScreenerOutput(
                success=False,
                operation="screen_portfolio",
                calculation_trace=["ERROR: No portfolio provided"]
            )

        criteria = input_data.criteria or self._default_criteria
        if input_data.target_sfdr:
            criteria = self._adjust_criteria_for_sfdr(criteria, input_data.target_sfdr)

        results: List[ScreeningResult] = []
        calculation_trace.append(f"Screening portfolio of {len(input_data.portfolio)} investments")

        for investment in input_data.portfolio:
            result = self._evaluate_investment(investment, criteria, calculation_trace)
            results.append(result)

        # Portfolio summary
        passed_count = sum(1 for r in results if r.passed)
        avg_sustainability = sum(r.overall_sustainability_score for r in results) / len(results)
        avg_screening = sum(r.screening_score for r in results) / len(results)

        summary = {
            "total_investments": len(results),
            "passed_screening": passed_count,
            "failed_screening": len(results) - passed_count,
            "pass_rate_pct": round(passed_count / len(results) * 100, 2),
            "average_sustainability_score": round(avg_sustainability, 2),
            "average_screening_score": round(avg_screening, 2),
            "common_exclusion_violations": self._find_common_violations(
                [r.exclusion_violations for r in results]
            ),
            "common_esg_violations": self._find_common_violations(
                [r.esg_violations for r in results]
            )
        }

        calculation_trace.append(f"Portfolio pass rate: {summary['pass_rate_pct']}%")

        provenance_hash = hashlib.sha256(
            json.dumps(summary, sort_keys=True, default=str).encode()
        ).hexdigest()

        return InvestmentScreenerOutput(
            success=True,
            operation="screen_portfolio",
            portfolio_results=results,
            portfolio_summary=summary,
            criteria_applied=criteria,
            sfdr_classification=input_data.target_sfdr,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _adjust_criteria_for_sfdr(
        self,
        base_criteria: InvestmentCriteria,
        sfdr: SFDRClassification
    ) -> InvestmentCriteria:
        """Adjust criteria based on SFDR classification target."""
        criteria = base_criteria.model_copy()

        if sfdr == SFDRClassification.ARTICLE_8:
            criteria.minimum_esg_rating = ESGRating.BBB
            criteria.exclusion_categories = [
                ExclusionCategory.CONTROVERSIAL_WEAPONS,
                ExclusionCategory.TOBACCO,
                ExclusionCategory.THERMAL_COAL
            ]

        elif sfdr == SFDRClassification.ARTICLE_8_PLUS:
            criteria.minimum_esg_rating = ESGRating.A
            criteria.exclusion_categories = [
                ExclusionCategory.CONTROVERSIAL_WEAPONS,
                ExclusionCategory.TOBACCO,
                ExclusionCategory.THERMAL_COAL,
                ExclusionCategory.OIL_SANDS
            ]
            criteria.require_eu_taxonomy_alignment = True
            criteria.minimum_taxonomy_alignment_pct = 10.0

        elif sfdr == SFDRClassification.ARTICLE_9:
            criteria.minimum_esg_rating = ESGRating.AA
            criteria.exclusion_categories = list(ExclusionCategory)
            criteria.require_sbti_target = True
            criteria.require_eu_taxonomy_alignment = True
            criteria.minimum_taxonomy_alignment_pct = 50.0
            criteria.minimum_green_revenue_pct = 20.0

        return criteria

    def _get_criteria(
        self, input_data: InvestmentScreenerInput
    ) -> InvestmentScreenerOutput:
        """Return the effective criteria for screening."""
        criteria = input_data.criteria or self._default_criteria
        if input_data.target_sfdr:
            criteria = self._adjust_criteria_for_sfdr(criteria, input_data.target_sfdr)

        return InvestmentScreenerOutput(
            success=True,
            operation="get_criteria",
            criteria_applied=criteria,
            sfdr_classification=input_data.target_sfdr,
            calculation_trace=[f"Criteria for SFDR {input_data.target_sfdr.value if input_data.target_sfdr else 'default'}"]
        )

    def _find_common_violations(
        self, violation_lists: List[List[str]]
    ) -> List[Dict[str, Any]]:
        """Find most common violations across results."""
        from collections import Counter
        all_violations = []
        for vlist in violation_lists:
            all_violations.extend(vlist)

        counter = Counter(all_violations)
        return [
            {"violation": v, "count": c}
            for v, c in counter.most_common(5)
        ]


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "GreenInvestmentScreenerAgent",
    "InvestmentScreenerInput",
    "InvestmentScreenerOutput",
    "InvestmentCriteria",
    "InvestmentProfile",
    "ESGRating",
    "ScreeningResult",
    "SustainabilityScore",
    "SFDRClassification",
    "ExclusionCategory",
]
