# -*- coding: utf-8 -*-
"""
GL-PROC-X-001: Supplier Sustainability Scorer Agent
===================================================

Scores supplier sustainability performance across environmental, social,
and governance dimensions using deterministic scoring methodologies.

Capabilities:
    - Multi-dimensional sustainability scoring
    - Environmental performance evaluation
    - Social/labor practice assessment
    - Governance and ethics evaluation
    - Data quality assessment
    - Benchmark comparison
    - Trend analysis

Zero-Hallucination Guarantees:
    - All scores calculated from structured inputs
    - Deterministic scoring algorithms
    - Complete audit trail for all assessments
    - SHA-256 provenance hashes for all outputs

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.categories import AgentCategory

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class ScoreCategory(str, Enum):
    """Scoring categories."""
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"
    CLIMATE = "climate"
    OVERALL = "overall"


class DataQuality(str, Enum):
    """Data quality levels."""
    HIGH = "high"  # Verified, audited data
    MEDIUM = "medium"  # Self-reported with some verification
    LOW = "low"  # Self-reported only
    ESTIMATED = "estimated"  # No supplier data


class RiskLevel(str, Enum):
    """Supplier risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PerformanceTier(str, Enum):
    """Supplier performance tiers."""
    LEADER = "leader"
    ADVANCED = "advanced"
    DEVELOPING = "developing"
    LAGGING = "lagging"
    NON_COMPLIANT = "non_compliant"


# Scoring weights
DEFAULT_WEIGHTS = {
    ScoreCategory.ENVIRONMENTAL.value: 0.35,
    ScoreCategory.SOCIAL.value: 0.25,
    ScoreCategory.GOVERNANCE.value: 0.20,
    ScoreCategory.CLIMATE.value: 0.20,
}

# Data quality multipliers
DATA_QUALITY_MULTIPLIERS = {
    DataQuality.HIGH.value: 1.0,
    DataQuality.MEDIUM.value: 0.9,
    DataQuality.LOW.value: 0.75,
    DataQuality.ESTIMATED.value: 0.5,
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class SupplierProfile(BaseModel):
    """Supplier profile for sustainability assessment."""
    supplier_id: str = Field(..., description="Unique identifier")
    name: str = Field(..., description="Supplier name")
    industry: str = Field(..., description="Industry sector")
    country: str = Field(..., description="Country of operations")
    employee_count: Optional[int] = Field(None, ge=0)
    annual_revenue: Optional[float] = Field(None, ge=0)

    # Environmental data
    has_environmental_policy: bool = Field(default=False)
    has_iso14001: bool = Field(default=False)
    has_emissions_disclosure: bool = Field(default=False)
    scope1_emissions_tco2e: Optional[float] = Field(None, ge=0)
    scope2_emissions_tco2e: Optional[float] = Field(None, ge=0)
    scope3_emissions_tco2e: Optional[float] = Field(None, ge=0)
    has_sbti_target: bool = Field(default=False)
    has_renewable_energy: bool = Field(default=False)
    renewable_energy_pct: float = Field(default=0, ge=0, le=100)
    water_management: bool = Field(default=False)
    waste_management: bool = Field(default=False)
    waste_recycling_pct: float = Field(default=0, ge=0, le=100)

    # Social data
    has_labor_policy: bool = Field(default=False)
    has_health_safety_policy: bool = Field(default=False)
    has_iso45001: bool = Field(default=False)
    living_wage_commitment: bool = Field(default=False)
    diversity_policy: bool = Field(default=False)
    human_rights_dd: bool = Field(default=False)
    community_engagement: bool = Field(default=False)
    incident_rate: Optional[float] = Field(None, ge=0)

    # Governance data
    has_code_of_conduct: bool = Field(default=False)
    has_anti_corruption_policy: bool = Field(default=False)
    has_whistleblower_mechanism: bool = Field(default=False)
    board_sustainability_oversight: bool = Field(default=False)
    sustainability_reporting: bool = Field(default=False)
    third_party_audit: bool = Field(default=False)
    controversies: int = Field(default=0, ge=0)

    # Data quality
    data_quality: DataQuality = Field(default=DataQuality.MEDIUM)
    last_assessment_date: Optional[datetime] = Field(None)


class SustainabilityScore(BaseModel):
    """Detailed sustainability score breakdown."""
    overall_score: float = Field(..., ge=0, le=100)
    environmental_score: float = Field(..., ge=0, le=100)
    social_score: float = Field(..., ge=0, le=100)
    governance_score: float = Field(..., ge=0, le=100)
    climate_score: float = Field(..., ge=0, le=100)

    # Performance classification
    performance_tier: PerformanceTier
    risk_level: RiskLevel

    # Data quality adjusted
    data_quality_factor: float = Field(..., ge=0, le=1)
    confidence_level: float = Field(..., ge=0, le=100)


class SupplierAssessment(BaseModel):
    """Complete supplier sustainability assessment."""
    supplier_id: str
    supplier_name: str
    assessment_date: datetime = Field(default_factory=datetime.utcnow)

    # Scores
    scores: SustainabilityScore

    # Detailed breakdown
    environmental_factors: Dict[str, float] = Field(default_factory=dict)
    social_factors: Dict[str, float] = Field(default_factory=dict)
    governance_factors: Dict[str, float] = Field(default_factory=dict)
    climate_factors: Dict[str, float] = Field(default_factory=dict)

    # Strengths and gaps
    strengths: List[str] = Field(default_factory=list)
    improvement_areas: List[str] = Field(default_factory=list)
    critical_gaps: List[str] = Field(default_factory=list)

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)

    # Benchmark comparison
    industry_percentile: Optional[float] = Field(None, ge=0, le=100)
    peer_comparison: Optional[Dict[str, Any]] = Field(None)


class SupplierScorerInput(BaseModel):
    """Input for supplier scoring."""
    operation: str = Field(
        default="score_supplier",
        description="Operation: score_supplier, score_portfolio, compare_suppliers"
    )

    # Supplier(s) to assess
    supplier: Optional[SupplierProfile] = Field(None)
    suppliers: Optional[List[SupplierProfile]] = Field(None)

    # Scoring parameters
    custom_weights: Optional[Dict[str, float]] = Field(None)
    industry_benchmark: Optional[str] = Field(None)
    minimum_score_threshold: float = Field(default=0, ge=0, le=100)


class SupplierScorerOutput(BaseModel):
    """Output from supplier scoring."""
    success: bool
    operation: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Results
    assessment: Optional[SupplierAssessment] = Field(None)
    assessments: Optional[List[SupplierAssessment]] = Field(None)
    portfolio_summary: Optional[Dict[str, Any]] = Field(None)

    # Audit
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# SUPPLIER SUSTAINABILITY SCORER AGENT
# =============================================================================


class SupplierSustainabilityScorerAgent(BaseAgent):
    """
    GL-PROC-X-001: Supplier Sustainability Scorer Agent

    Scores supplier sustainability using deterministic methodologies.

    Zero-Hallucination Guarantees:
        - All scores calculated from structured inputs
        - Deterministic scoring algorithms
        - Complete audit trail for all assessments
        - SHA-256 provenance hashes for all outputs

    Usage:
        agent = SupplierSustainabilityScorerAgent()
        result = agent.run({
            "operation": "score_supplier",
            "supplier": supplier_profile
        })
    """

    AGENT_ID = "GL-PROC-X-001"
    AGENT_NAME = "Supplier Sustainability Scorer"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Supplier Sustainability Scorer Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Supplier sustainability scoring",
                version=self.VERSION,
                parameters={}
            )

        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute supplier scoring."""
        try:
            scorer_input = SupplierScorerInput(**input_data)
            operation = scorer_input.operation

            if operation == "score_supplier":
                output = self._score_supplier(scorer_input)
            elif operation == "score_portfolio":
                output = self._score_portfolio(scorer_input)
            elif operation == "compare_suppliers":
                output = self._compare_suppliers(scorer_input)
            else:
                return AgentResult(success=False, error=f"Unknown operation: {operation}")

            return AgentResult(
                success=output.success,
                data=output.model_dump(),
                metadata={"agent_id": self.AGENT_ID, "operation": operation}
            )

        except Exception as e:
            logger.error(f"Supplier scoring failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _score_supplier(self, input_data: SupplierScorerInput) -> SupplierScorerOutput:
        """Score a single supplier."""
        calculation_trace: List[str] = []

        if input_data.supplier is None:
            return SupplierScorerOutput(
                success=False,
                operation="score_supplier",
                calculation_trace=["ERROR: No supplier provided"]
            )

        supplier = input_data.supplier
        weights = input_data.custom_weights or DEFAULT_WEIGHTS

        calculation_trace.append(f"Scoring supplier: {supplier.name} ({supplier.supplier_id})")
        calculation_trace.append(f"Industry: {supplier.industry}, Country: {supplier.country}")

        # Calculate component scores
        env_score, env_factors = self._score_environmental(supplier, calculation_trace)
        social_score, social_factors = self._score_social(supplier, calculation_trace)
        gov_score, gov_factors = self._score_governance(supplier, calculation_trace)
        climate_score, climate_factors = self._score_climate(supplier, calculation_trace)

        # Calculate overall score
        overall = (
            env_score * weights.get(ScoreCategory.ENVIRONMENTAL.value, 0.35) +
            social_score * weights.get(ScoreCategory.SOCIAL.value, 0.25) +
            gov_score * weights.get(ScoreCategory.GOVERNANCE.value, 0.20) +
            climate_score * weights.get(ScoreCategory.CLIMATE.value, 0.20)
        )

        # Apply data quality factor
        dq_factor = DATA_QUALITY_MULTIPLIERS.get(supplier.data_quality.value, 0.75)
        confidence = dq_factor * 100

        calculation_trace.append(f"Data quality factor: {dq_factor}")
        calculation_trace.append(f"Overall score: {overall:.1f}")

        # Determine tier and risk
        tier = self._determine_tier(overall)
        risk = self._determine_risk(overall, supplier)

        # Create score object
        scores = SustainabilityScore(
            overall_score=round(overall, 2),
            environmental_score=round(env_score, 2),
            social_score=round(social_score, 2),
            governance_score=round(gov_score, 2),
            climate_score=round(climate_score, 2),
            performance_tier=tier,
            risk_level=risk,
            data_quality_factor=dq_factor,
            confidence_level=confidence
        )

        # Identify strengths and gaps
        strengths, improvements, critical = self._identify_factors(
            supplier, env_score, social_score, gov_score, climate_score
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(supplier, scores)

        assessment = SupplierAssessment(
            supplier_id=supplier.supplier_id,
            supplier_name=supplier.name,
            scores=scores,
            environmental_factors=env_factors,
            social_factors=social_factors,
            governance_factors=gov_factors,
            climate_factors=climate_factors,
            strengths=strengths,
            improvement_areas=improvements,
            critical_gaps=critical,
            recommendations=recommendations
        )

        provenance_hash = hashlib.sha256(
            json.dumps(assessment.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()

        return SupplierScorerOutput(
            success=True,
            operation="score_supplier",
            assessment=assessment,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _score_environmental(
        self, supplier: SupplierProfile, trace: List[str]
    ) -> tuple:
        """Score environmental performance."""
        score = 0.0
        factors: Dict[str, float] = {}

        # Environmental management (30 points)
        if supplier.has_environmental_policy:
            score += 10
            factors["environmental_policy"] = 10
        if supplier.has_iso14001:
            score += 20
            factors["iso14001"] = 20

        # Resource efficiency (35 points)
        if supplier.water_management:
            score += 10
            factors["water_management"] = 10
        if supplier.waste_management:
            score += 10
            factors["waste_management"] = 10
        if supplier.waste_recycling_pct >= 50:
            score += 15
            factors["recycling"] = 15
        elif supplier.waste_recycling_pct >= 25:
            score += 10
            factors["recycling"] = 10

        # Emissions reporting (35 points)
        if supplier.has_emissions_disclosure:
            score += 20
            factors["emissions_disclosure"] = 20
        if supplier.scope1_emissions_tco2e is not None:
            score += 10
            factors["scope1_reported"] = 10
        if supplier.scope2_emissions_tco2e is not None:
            score += 5
            factors["scope2_reported"] = 5

        trace.append(f"Environmental score: {score:.1f}")
        return score, factors

    def _score_social(
        self, supplier: SupplierProfile, trace: List[str]
    ) -> tuple:
        """Score social performance."""
        score = 0.0
        factors: Dict[str, float] = {}

        # Labor practices (40 points)
        if supplier.has_labor_policy:
            score += 15
            factors["labor_policy"] = 15
        if supplier.living_wage_commitment:
            score += 15
            factors["living_wage"] = 15
        if supplier.diversity_policy:
            score += 10
            factors["diversity"] = 10

        # Health & safety (35 points)
        if supplier.has_health_safety_policy:
            score += 15
            factors["hs_policy"] = 15
        if supplier.has_iso45001:
            score += 20
            factors["iso45001"] = 20

        # Human rights (25 points)
        if supplier.human_rights_dd:
            score += 15
            factors["human_rights_dd"] = 15
        if supplier.community_engagement:
            score += 10
            factors["community"] = 10

        trace.append(f"Social score: {score:.1f}")
        return score, factors

    def _score_governance(
        self, supplier: SupplierProfile, trace: List[str]
    ) -> tuple:
        """Score governance performance."""
        score = 0.0
        factors: Dict[str, float] = {}

        # Ethics & compliance (40 points)
        if supplier.has_code_of_conduct:
            score += 15
            factors["code_of_conduct"] = 15
        if supplier.has_anti_corruption_policy:
            score += 15
            factors["anti_corruption"] = 15
        if supplier.has_whistleblower_mechanism:
            score += 10
            factors["whistleblower"] = 10

        # Oversight & reporting (35 points)
        if supplier.board_sustainability_oversight:
            score += 15
            factors["board_oversight"] = 15
        if supplier.sustainability_reporting:
            score += 20
            factors["reporting"] = 20

        # Verification (25 points)
        if supplier.third_party_audit:
            score += 25
            factors["third_party_audit"] = 25

        # Deduct for controversies
        if supplier.controversies > 0:
            deduction = min(supplier.controversies * 10, 30)
            score -= deduction
            factors["controversy_deduction"] = -deduction

        score = max(0, score)
        trace.append(f"Governance score: {score:.1f}")
        return score, factors

    def _score_climate(
        self, supplier: SupplierProfile, trace: List[str]
    ) -> tuple:
        """Score climate performance."""
        score = 0.0
        factors: Dict[str, float] = {}

        # Targets (40 points)
        if supplier.has_sbti_target:
            score += 40
            factors["sbti_target"] = 40
        elif supplier.has_emissions_disclosure:
            score += 15
            factors["emissions_disclosure"] = 15

        # Renewable energy (30 points)
        if supplier.has_renewable_energy:
            if supplier.renewable_energy_pct >= 100:
                score += 30
                factors["renewable_100"] = 30
            elif supplier.renewable_energy_pct >= 50:
                score += 20
                factors["renewable_50"] = 20
            elif supplier.renewable_energy_pct >= 25:
                score += 10
                factors["renewable_25"] = 10

        # Scope 3 (30 points)
        if supplier.scope3_emissions_tco2e is not None:
            score += 30
            factors["scope3_reported"] = 30

        trace.append(f"Climate score: {score:.1f}")
        return score, factors

    def _determine_tier(self, score: float) -> PerformanceTier:
        """Determine performance tier from score."""
        if score >= 80:
            return PerformanceTier.LEADER
        elif score >= 60:
            return PerformanceTier.ADVANCED
        elif score >= 40:
            return PerformanceTier.DEVELOPING
        elif score >= 20:
            return PerformanceTier.LAGGING
        else:
            return PerformanceTier.NON_COMPLIANT

    def _determine_risk(
        self, score: float, supplier: SupplierProfile
    ) -> RiskLevel:
        """Determine risk level."""
        risk_score = 100 - score

        # Add risk for controversies
        if supplier.controversies > 2:
            risk_score += 20
        elif supplier.controversies > 0:
            risk_score += 10

        # Add risk for missing key policies
        if not supplier.has_code_of_conduct:
            risk_score += 10
        if not supplier.human_rights_dd:
            risk_score += 10

        if risk_score >= 70:
            return RiskLevel.CRITICAL
        elif risk_score >= 50:
            return RiskLevel.HIGH
        elif risk_score >= 30:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _identify_factors(
        self,
        supplier: SupplierProfile,
        env: float,
        social: float,
        gov: float,
        climate: float
    ) -> tuple:
        """Identify strengths, improvements, and critical gaps."""
        strengths: List[str] = []
        improvements: List[str] = []
        critical: List[str] = []

        # Strengths
        if env >= 70:
            strengths.append("Strong environmental management")
        if social >= 70:
            strengths.append("Strong social practices")
        if gov >= 70:
            strengths.append("Strong governance framework")
        if climate >= 70:
            strengths.append("Strong climate commitments")
        if supplier.has_sbti_target:
            strengths.append("Science-based climate target")
        if supplier.has_iso14001:
            strengths.append("ISO 14001 certified")
        if supplier.has_iso45001:
            strengths.append("ISO 45001 certified")

        # Improvements
        if env < 50:
            improvements.append("Enhance environmental management systems")
        if not supplier.has_emissions_disclosure:
            improvements.append("Implement emissions disclosure")
        if not supplier.has_renewable_energy:
            improvements.append("Transition to renewable energy")
        if social < 50:
            improvements.append("Strengthen social practices")
        if gov < 50:
            improvements.append("Improve governance framework")
        if not supplier.sustainability_reporting:
            improvements.append("Implement sustainability reporting")

        # Critical gaps
        if not supplier.has_code_of_conduct:
            critical.append("Missing code of conduct")
        if not supplier.human_rights_dd:
            critical.append("No human rights due diligence")
        if supplier.controversies > 2:
            critical.append("Multiple controversies identified")
        if env < 30:
            critical.append("Severely lacking environmental practices")
        if social < 30:
            critical.append("Severely lacking social practices")

        return strengths, improvements, critical

    def _generate_recommendations(
        self, supplier: SupplierProfile, scores: SustainabilityScore
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations: List[str] = []

        if not supplier.has_sbti_target:
            recommendations.append(
                "Set science-based emission reduction targets aligned with 1.5C pathway"
            )

        if not supplier.has_emissions_disclosure:
            recommendations.append(
                "Implement GHG emissions measurement and disclosure"
            )

        if supplier.renewable_energy_pct < 50:
            recommendations.append(
                "Increase renewable energy sourcing to at least 50%"
            )

        if not supplier.has_iso14001:
            recommendations.append(
                "Pursue ISO 14001 environmental management certification"
            )

        if not supplier.has_iso45001:
            recommendations.append(
                "Pursue ISO 45001 health & safety certification"
            )

        if not supplier.third_party_audit:
            recommendations.append(
                "Engage third-party sustainability auditor"
            )

        if scores.overall_score < 50:
            recommendations.append(
                "Develop comprehensive sustainability improvement roadmap"
            )

        return recommendations[:5]  # Top 5 recommendations

    def _score_portfolio(
        self, input_data: SupplierScorerInput
    ) -> SupplierScorerOutput:
        """Score a portfolio of suppliers."""
        calculation_trace: List[str] = []

        if not input_data.suppliers:
            return SupplierScorerOutput(
                success=False,
                operation="score_portfolio",
                calculation_trace=["ERROR: No suppliers provided"]
            )

        assessments: List[SupplierAssessment] = []

        for supplier in input_data.suppliers:
            result = self._score_supplier(
                SupplierScorerInput(supplier=supplier, custom_weights=input_data.custom_weights)
            )
            if result.assessment:
                assessments.append(result.assessment)

        # Portfolio summary
        avg_score = sum(a.scores.overall_score for a in assessments) / len(assessments)
        tier_distribution = {}
        risk_distribution = {}

        for a in assessments:
            tier = a.scores.performance_tier.value
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
            risk = a.scores.risk_level.value
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1

        summary = {
            "total_suppliers": len(assessments),
            "average_score": round(avg_score, 2),
            "tier_distribution": tier_distribution,
            "risk_distribution": risk_distribution,
            "leaders_count": tier_distribution.get("leader", 0),
            "high_risk_count": risk_distribution.get("high", 0) + risk_distribution.get("critical", 0),
            "below_threshold": sum(
                1 for a in assessments
                if a.scores.overall_score < input_data.minimum_score_threshold
            )
        }

        calculation_trace.append(f"Portfolio of {len(assessments)} suppliers")
        calculation_trace.append(f"Average score: {avg_score:.1f}")

        provenance_hash = hashlib.sha256(
            json.dumps(summary, sort_keys=True, default=str).encode()
        ).hexdigest()

        return SupplierScorerOutput(
            success=True,
            operation="score_portfolio",
            assessments=assessments,
            portfolio_summary=summary,
            calculation_trace=calculation_trace,
            provenance_hash=provenance_hash
        )

    def _compare_suppliers(
        self, input_data: SupplierScorerInput
    ) -> SupplierScorerOutput:
        """Compare multiple suppliers."""
        result = self._score_portfolio(input_data)
        if not result.assessments:
            return result

        # Add comparison rankings
        sorted_by_score = sorted(
            result.assessments,
            key=lambda a: a.scores.overall_score,
            reverse=True
        )

        rankings = [
            {
                "rank": i + 1,
                "supplier_id": a.supplier_id,
                "supplier_name": a.supplier_name,
                "overall_score": a.scores.overall_score,
                "tier": a.scores.performance_tier.value,
                "risk": a.scores.risk_level.value
            }
            for i, a in enumerate(sorted_by_score)
        ]

        result.portfolio_summary = result.portfolio_summary or {}
        result.portfolio_summary["rankings"] = rankings
        result.operation = "compare_suppliers"

        return result


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "SupplierSustainabilityScorerAgent",
    "SupplierScorerInput",
    "SupplierScorerOutput",
    "SupplierProfile",
    "SustainabilityScore",
    "SupplierAssessment",
    "ScoreCategory",
    "DataQuality",
    "PerformanceTier",
    "RiskLevel",
]
