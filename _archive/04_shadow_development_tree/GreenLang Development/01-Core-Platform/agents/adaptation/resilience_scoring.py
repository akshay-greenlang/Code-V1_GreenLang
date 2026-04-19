# -*- coding: utf-8 -*-
"""
GL-ADAPT-X-006: Resilience Scoring Agent
=========================================

Scores the resilience of assets and operations against climate hazards
using deterministic multi-factor assessment.

Capabilities:
    - Multi-dimensional resilience scoring
    - Absorptive capacity assessment
    - Adaptive capacity evaluation
    - Transformative capacity measurement
    - Benchmark comparison
    - Gap identification
    - Improvement prioritization

Zero-Hallucination Guarantees:
    - All scores from deterministic calculations
    - Factor weights from validated frameworks
    - Complete provenance tracking
    - No LLM-based scoring

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class ResilienceCapacity(str, Enum):
    """Types of resilience capacity."""
    ABSORPTIVE = "absorptive"  # Ability to absorb shocks
    ADAPTIVE = "adaptive"  # Ability to adapt to changes
    TRANSFORMATIVE = "transformative"  # Ability to transform fundamentally


class ResilienceLevel(str, Enum):
    """Resilience level classifications."""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


class ResilienceDimension(str, Enum):
    """Dimensions of resilience assessment."""
    PHYSICAL = "physical"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    ORGANIZATIONAL = "organizational"
    SOCIAL = "social"
    TECHNOLOGICAL = "technological"


# Resilience thresholds
RESILIENCE_THRESHOLDS = {
    ResilienceLevel.VERY_HIGH: 0.8,
    ResilienceLevel.HIGH: 0.6,
    ResilienceLevel.MODERATE: 0.4,
    ResilienceLevel.LOW: 0.2,
    ResilienceLevel.VERY_LOW: 0.0
}

# Sector benchmarks
SECTOR_BENCHMARKS = {
    "manufacturing": 0.55,
    "agriculture": 0.45,
    "energy": 0.60,
    "real_estate": 0.50,
    "financial_services": 0.65,
    "healthcare": 0.60,
    "technology": 0.65,
    "retail": 0.50,
}


# =============================================================================
# Pydantic Models
# =============================================================================

class DimensionScore(BaseModel):
    """Score for a single resilience dimension."""
    dimension: ResilienceDimension = Field(...)
    score: float = Field(..., ge=0, le=1)
    weight: float = Field(default=1.0, ge=0)
    sub_scores: Dict[str, float] = Field(default_factory=dict)
    gaps: List[str] = Field(default_factory=list)


class CapacityScore(BaseModel):
    """Score for a resilience capacity type."""
    capacity: ResilienceCapacity = Field(...)
    score: float = Field(..., ge=0, le=1)
    contributing_factors: List[str] = Field(default_factory=list)
    improvement_actions: List[str] = Field(default_factory=list)


class ResilienceProfile(BaseModel):
    """Complete resilience profile for an asset."""
    asset_id: str = Field(...)
    asset_name: str = Field(...)
    sector: str = Field(default="general")

    # Overall scores
    resilience_score: float = Field(..., ge=0, le=1)
    resilience_level: ResilienceLevel = Field(...)

    # Capacity breakdown
    absorptive_capacity: CapacityScore = Field(...)
    adaptive_capacity: CapacityScore = Field(...)
    transformative_capacity: CapacityScore = Field(...)

    # Dimension breakdown
    dimension_scores: List[DimensionScore] = Field(default_factory=list)

    # Benchmarking
    sector_benchmark: float = Field(default=0.5, ge=0, le=1)
    benchmark_gap: float = Field(default=0.0)
    benchmark_percentile: Optional[float] = Field(None, ge=0, le=100)

    # Gaps and priorities
    priority_gaps: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)

    # Metadata
    assessed_at: datetime = Field(default_factory=DeterministicClock.now)
    calculation_trace: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class ResilienceInput(BaseModel):
    """Input for resilience assessment of a single asset."""
    asset_id: str = Field(...)
    asset_name: str = Field(...)
    sector: str = Field(default="general")

    # Physical resilience factors
    structural_integrity: float = Field(default=0.5, ge=0, le=1)
    redundancy_level: float = Field(default=0.5, ge=0, le=1)
    protective_measures: float = Field(default=0.5, ge=0, le=1)

    # Operational resilience factors
    business_continuity_plan: bool = Field(default=False)
    backup_systems: float = Field(default=0.5, ge=0, le=1)
    supply_chain_flexibility: float = Field(default=0.5, ge=0, le=1)
    response_capabilities: float = Field(default=0.5, ge=0, le=1)

    # Financial resilience factors
    insurance_coverage: float = Field(default=0.5, ge=0, le=1)
    financial_reserves: float = Field(default=0.5, ge=0, le=1)
    revenue_diversification: float = Field(default=0.5, ge=0, le=1)

    # Organizational resilience factors
    leadership_commitment: float = Field(default=0.5, ge=0, le=1)
    staff_training: float = Field(default=0.5, ge=0, le=1)
    adaptation_planning: float = Field(default=0.5, ge=0, le=1)

    # Technology factors
    monitoring_systems: float = Field(default=0.5, ge=0, le=1)
    early_warning: float = Field(default=0.5, ge=0, le=1)
    data_backup: float = Field(default=0.5, ge=0, le=1)


class ResilienceScoringInput(BaseModel):
    """Input model for Resilience Scoring Agent."""
    assessment_id: str = Field(...)
    assets: List[ResilienceInput] = Field(..., min_length=1)
    include_benchmarks: bool = Field(default=True)
    include_recommendations: bool = Field(default=True)


class ResilienceScoringOutput(BaseModel):
    """Output model for Resilience Scoring Agent."""
    assessment_id: str = Field(...)
    completed_at: datetime = Field(default_factory=DeterministicClock.now)

    # Results
    resilience_profiles: List[ResilienceProfile] = Field(default_factory=list)

    # Summary
    total_assets_scored: int = Field(default=0)
    average_resilience: float = Field(default=0.0, ge=0, le=1)
    very_low_resilience_count: int = Field(default=0)
    low_resilience_count: int = Field(default=0)

    # Portfolio metrics
    portfolio_resilience_score: float = Field(default=0.0, ge=0, le=1)
    weakest_dimension: Optional[ResilienceDimension] = Field(None)
    weakest_capacity: Optional[ResilienceCapacity] = Field(None)

    # Common gaps
    common_gaps: List[str] = Field(default_factory=list)
    priority_actions: List[str] = Field(default_factory=list)

    # Processing info
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# Resilience Scoring Agent Implementation
# =============================================================================

class ResilienceScoringAgent(BaseAgent):
    """
    GL-ADAPT-X-006: Resilience Scoring Agent

    Scores resilience of assets using deterministic multi-factor assessment
    across absorptive, adaptive, and transformative capacities.

    Zero-Hallucination Implementation:
        - All scores from deterministic calculations
        - Factor weights from validated frameworks
        - Complete audit trail
        - No LLM-based scoring

    Example:
        >>> agent = ResilienceScoringAgent()
        >>> result = agent.run({
        ...     "assessment_id": "RS001",
        ...     "assets": [{"asset_id": "A1", "asset_name": "Factory", ...}]
        ... })
    """

    AGENT_ID = "GL-ADAPT-X-006"
    AGENT_NAME = "Resilience Scoring Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the Resilience Scoring Agent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Scores asset and operational resilience",
                version=self.VERSION,
                parameters={}
            )

        # Dimension weights
        self._dimension_weights = {
            ResilienceDimension.PHYSICAL: 1.2,
            ResilienceDimension.OPERATIONAL: 1.1,
            ResilienceDimension.FINANCIAL: 1.0,
            ResilienceDimension.ORGANIZATIONAL: 0.9,
            ResilienceDimension.TECHNOLOGICAL: 1.0,
        }

        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def initialize(self):
        """Initialize agent resources."""
        logger.info("Resilience Scoring Agent initialized")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute resilience scoring.

        Args:
            input_data: Input containing assets to score

        Returns:
            AgentResult with ResilienceScoringOutput
        """
        start_time = time.time()

        try:
            # Parse input
            scoring_input = ResilienceScoringInput(**input_data)
            self.logger.info(
                f"Starting resilience scoring: {scoring_input.assessment_id}, "
                f"{len(scoring_input.assets)} assets"
            )

            # Score each asset
            profiles: List[ResilienceProfile] = []
            for asset in scoring_input.assets:
                profile = self._score_asset(
                    asset,
                    include_benchmarks=scoring_input.include_benchmarks,
                    include_recommendations=scoring_input.include_recommendations
                )
                profiles.append(profile)

            # Calculate summary metrics
            avg_resilience = sum(p.resilience_score for p in profiles) / len(profiles) if profiles else 0
            very_low_count = sum(1 for p in profiles if p.resilience_level == ResilienceLevel.VERY_LOW)
            low_count = sum(1 for p in profiles if p.resilience_level == ResilienceLevel.LOW)

            # Find weakest dimension and capacity
            weakest_dim, weakest_cap = self._find_weakest_areas(profiles)

            # Identify common gaps
            common_gaps = self._identify_common_gaps(profiles)
            priority_actions = self._prioritize_actions(profiles)

            # Build output
            processing_time = (time.time() - start_time) * 1000

            output = ResilienceScoringOutput(
                assessment_id=scoring_input.assessment_id,
                resilience_profiles=profiles,
                total_assets_scored=len(profiles),
                average_resilience=avg_resilience,
                very_low_resilience_count=very_low_count,
                low_resilience_count=low_count,
                portfolio_resilience_score=avg_resilience,
                weakest_dimension=weakest_dim,
                weakest_capacity=weakest_cap,
                common_gaps=common_gaps,
                priority_actions=priority_actions,
                processing_time_ms=processing_time,
            )

            output.provenance_hash = self._calculate_provenance_hash(scoring_input, output)

            self.logger.info(
                f"Resilience scoring complete: {len(profiles)} assets, "
                f"avg resilience: {avg_resilience:.2f}"
            )

            return AgentResult(
                success=True,
                data=output.model_dump(),
                metadata={
                    "agent_id": self.AGENT_ID,
                    "version": self.VERSION,
                    "low_resilience_count": very_low_count + low_count
                }
            )

        except Exception as e:
            self.logger.error(f"Resilience scoring failed: {str(e)}", exc_info=True)
            return AgentResult(
                success=False,
                error=str(e),
                metadata={"agent_id": self.AGENT_ID, "version": self.VERSION}
            )

    def _score_asset(
        self,
        asset: ResilienceInput,
        include_benchmarks: bool,
        include_recommendations: bool
    ) -> ResilienceProfile:
        """Score a single asset's resilience."""
        trace = []

        # Calculate dimension scores
        dimension_scores = self._calculate_dimension_scores(asset)

        # Calculate capacity scores
        absorptive = self._calculate_absorptive_capacity(asset)
        adaptive = self._calculate_adaptive_capacity(asset)
        transformative = self._calculate_transformative_capacity(asset)

        trace.append(f"absorptive={absorptive.score:.4f}")
        trace.append(f"adaptive={adaptive.score:.4f}")
        trace.append(f"transformative={transformative.score:.4f}")

        # Calculate overall score
        resilience_score = (
            absorptive.score * 0.35 +
            adaptive.score * 0.40 +
            transformative.score * 0.25
        )
        trace.append(f"resilience_score={resilience_score:.4f}")

        # Classify level
        resilience_level = self._classify_resilience(resilience_score)

        # Benchmark comparison
        benchmark = SECTOR_BENCHMARKS.get(asset.sector, 0.5)
        benchmark_gap = resilience_score - benchmark

        # Identify gaps
        priority_gaps = self._identify_gaps(dimension_scores, absorptive, adaptive, transformative)

        # Generate recommendations
        recommendations = []
        if include_recommendations:
            recommendations = self._generate_recommendations(
                asset, dimension_scores, priority_gaps
            )

        profile = ResilienceProfile(
            asset_id=asset.asset_id,
            asset_name=asset.asset_name,
            sector=asset.sector,
            resilience_score=resilience_score,
            resilience_level=resilience_level,
            absorptive_capacity=absorptive,
            adaptive_capacity=adaptive,
            transformative_capacity=transformative,
            dimension_scores=dimension_scores,
            sector_benchmark=benchmark if include_benchmarks else 0.5,
            benchmark_gap=benchmark_gap if include_benchmarks else 0.0,
            priority_gaps=priority_gaps,
            recommended_actions=recommendations,
            calculation_trace=trace,
        )

        profile.provenance_hash = hashlib.sha256(
            json.dumps({
                "asset_id": asset.asset_id,
                "resilience_score": resilience_score,
            }, sort_keys=True).encode()
        ).hexdigest()[:16]

        return profile

    def _calculate_dimension_scores(
        self,
        asset: ResilienceInput
    ) -> List[DimensionScore]:
        """Calculate scores for each dimension."""
        scores = []

        # Physical dimension
        physical_score = (
            asset.structural_integrity * 0.4 +
            asset.redundancy_level * 0.3 +
            asset.protective_measures * 0.3
        )
        physical_gaps = []
        if asset.structural_integrity < 0.5:
            physical_gaps.append("Structural integrity needs improvement")
        scores.append(DimensionScore(
            dimension=ResilienceDimension.PHYSICAL,
            score=physical_score,
            weight=self._dimension_weights[ResilienceDimension.PHYSICAL],
            sub_scores={
                "structural_integrity": asset.structural_integrity,
                "redundancy": asset.redundancy_level,
                "protective_measures": asset.protective_measures
            },
            gaps=physical_gaps
        ))

        # Operational dimension
        bcp_score = 0.7 if asset.business_continuity_plan else 0.3
        operational_score = (
            bcp_score * 0.3 +
            asset.backup_systems * 0.25 +
            asset.supply_chain_flexibility * 0.25 +
            asset.response_capabilities * 0.2
        )
        operational_gaps = []
        if not asset.business_continuity_plan:
            operational_gaps.append("No business continuity plan")
        scores.append(DimensionScore(
            dimension=ResilienceDimension.OPERATIONAL,
            score=operational_score,
            weight=self._dimension_weights[ResilienceDimension.OPERATIONAL],
            gaps=operational_gaps
        ))

        # Financial dimension
        financial_score = (
            asset.insurance_coverage * 0.4 +
            asset.financial_reserves * 0.35 +
            asset.revenue_diversification * 0.25
        )
        financial_gaps = []
        if asset.insurance_coverage < 0.5:
            financial_gaps.append("Inadequate insurance coverage")
        scores.append(DimensionScore(
            dimension=ResilienceDimension.FINANCIAL,
            score=financial_score,
            weight=self._dimension_weights[ResilienceDimension.FINANCIAL],
            gaps=financial_gaps
        ))

        # Organizational dimension
        organizational_score = (
            asset.leadership_commitment * 0.35 +
            asset.staff_training * 0.35 +
            asset.adaptation_planning * 0.3
        )
        scores.append(DimensionScore(
            dimension=ResilienceDimension.ORGANIZATIONAL,
            score=organizational_score,
            weight=self._dimension_weights[ResilienceDimension.ORGANIZATIONAL],
            gaps=[]
        ))

        # Technological dimension
        tech_score = (
            asset.monitoring_systems * 0.35 +
            asset.early_warning * 0.35 +
            asset.data_backup * 0.3
        )
        scores.append(DimensionScore(
            dimension=ResilienceDimension.TECHNOLOGICAL,
            score=tech_score,
            weight=self._dimension_weights[ResilienceDimension.TECHNOLOGICAL],
            gaps=[]
        ))

        return scores

    def _calculate_absorptive_capacity(self, asset: ResilienceInput) -> CapacityScore:
        """Calculate absorptive capacity score."""
        score = (
            asset.structural_integrity * 0.3 +
            asset.protective_measures * 0.25 +
            asset.insurance_coverage * 0.25 +
            asset.financial_reserves * 0.2
        )

        factors = ["Structural integrity", "Protective measures", "Insurance", "Reserves"]
        actions = []
        if score < 0.5:
            actions.append("Strengthen physical protections")
            actions.append("Increase insurance coverage")

        return CapacityScore(
            capacity=ResilienceCapacity.ABSORPTIVE,
            score=score,
            contributing_factors=factors,
            improvement_actions=actions
        )

    def _calculate_adaptive_capacity(self, asset: ResilienceInput) -> CapacityScore:
        """Calculate adaptive capacity score."""
        bcp_val = 0.7 if asset.business_continuity_plan else 0.3
        score = (
            asset.supply_chain_flexibility * 0.25 +
            asset.response_capabilities * 0.25 +
            bcp_val * 0.25 +
            asset.adaptation_planning * 0.25
        )

        factors = ["Supply chain flexibility", "Response capabilities", "BCP", "Adaptation planning"]
        actions = []
        if not asset.business_continuity_plan:
            actions.append("Develop business continuity plan")
        if asset.supply_chain_flexibility < 0.5:
            actions.append("Diversify supply chain")

        return CapacityScore(
            capacity=ResilienceCapacity.ADAPTIVE,
            score=score,
            contributing_factors=factors,
            improvement_actions=actions
        )

    def _calculate_transformative_capacity(self, asset: ResilienceInput) -> CapacityScore:
        """Calculate transformative capacity score."""
        score = (
            asset.leadership_commitment * 0.35 +
            asset.staff_training * 0.25 +
            asset.monitoring_systems * 0.2 +
            asset.early_warning * 0.2
        )

        factors = ["Leadership", "Training", "Monitoring", "Early warning"]
        actions = []
        if asset.leadership_commitment < 0.5:
            actions.append("Strengthen leadership commitment to resilience")

        return CapacityScore(
            capacity=ResilienceCapacity.TRANSFORMATIVE,
            score=score,
            contributing_factors=factors,
            improvement_actions=actions
        )

    def _classify_resilience(self, score: float) -> ResilienceLevel:
        """Classify resilience score into level."""
        if score >= RESILIENCE_THRESHOLDS[ResilienceLevel.VERY_HIGH]:
            return ResilienceLevel.VERY_HIGH
        elif score >= RESILIENCE_THRESHOLDS[ResilienceLevel.HIGH]:
            return ResilienceLevel.HIGH
        elif score >= RESILIENCE_THRESHOLDS[ResilienceLevel.MODERATE]:
            return ResilienceLevel.MODERATE
        elif score >= RESILIENCE_THRESHOLDS[ResilienceLevel.LOW]:
            return ResilienceLevel.LOW
        else:
            return ResilienceLevel.VERY_LOW

    def _identify_gaps(
        self,
        dimensions: List[DimensionScore],
        absorptive: CapacityScore,
        adaptive: CapacityScore,
        transformative: CapacityScore
    ) -> List[str]:
        """Identify priority resilience gaps."""
        gaps = []

        # Dimension gaps
        for dim in dimensions:
            if dim.score < 0.4:
                gaps.append(f"Low {dim.dimension.value} resilience")
            gaps.extend(dim.gaps)

        # Capacity gaps
        if absorptive.score < 0.4:
            gaps.append("Limited absorptive capacity")
        if adaptive.score < 0.4:
            gaps.append("Limited adaptive capacity")
        if transformative.score < 0.4:
            gaps.append("Limited transformative capacity")

        return gaps[:5]

    def _generate_recommendations(
        self,
        asset: ResilienceInput,
        dimensions: List[DimensionScore],
        gaps: List[str]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        # Find lowest scoring dimension
        lowest_dim = min(dimensions, key=lambda d: d.score)
        if lowest_dim.score < 0.5:
            recommendations.append(f"Prioritize {lowest_dim.dimension.value} resilience improvements")

        # Specific recommendations
        if not asset.business_continuity_plan:
            recommendations.append("Develop comprehensive business continuity plan")
        if asset.insurance_coverage < 0.5:
            recommendations.append("Review and enhance climate risk insurance coverage")
        if asset.early_warning < 0.5:
            recommendations.append("Implement early warning monitoring systems")

        return recommendations[:5]

    def _find_weakest_areas(
        self,
        profiles: List[ResilienceProfile]
    ) -> tuple:
        """Find weakest dimension and capacity across portfolio."""
        if not profiles:
            return None, None

        # Aggregate dimension scores
        dim_totals: Dict[ResilienceDimension, List[float]] = {}
        cap_totals: Dict[ResilienceCapacity, List[float]] = {}

        for profile in profiles:
            for dim_score in profile.dimension_scores:
                if dim_score.dimension not in dim_totals:
                    dim_totals[dim_score.dimension] = []
                dim_totals[dim_score.dimension].append(dim_score.score)

            cap_totals.setdefault(ResilienceCapacity.ABSORPTIVE, []).append(
                profile.absorptive_capacity.score
            )
            cap_totals.setdefault(ResilienceCapacity.ADAPTIVE, []).append(
                profile.adaptive_capacity.score
            )
            cap_totals.setdefault(ResilienceCapacity.TRANSFORMATIVE, []).append(
                profile.transformative_capacity.score
            )

        # Find weakest
        weakest_dim = min(
            dim_totals.items(),
            key=lambda x: sum(x[1]) / len(x[1])
        )[0] if dim_totals else None

        weakest_cap = min(
            cap_totals.items(),
            key=lambda x: sum(x[1]) / len(x[1])
        )[0] if cap_totals else None

        return weakest_dim, weakest_cap

    def _identify_common_gaps(self, profiles: List[ResilienceProfile]) -> List[str]:
        """Identify gaps common across multiple assets."""
        gap_counts: Dict[str, int] = {}
        for profile in profiles:
            for gap in profile.priority_gaps:
                gap_counts[gap] = gap_counts.get(gap, 0) + 1

        # Return gaps appearing in >50% of assets
        threshold = len(profiles) * 0.5
        common = [g for g, c in gap_counts.items() if c >= threshold]
        return common[:5]

    def _prioritize_actions(self, profiles: List[ResilienceProfile]) -> List[str]:
        """Prioritize actions across portfolio."""
        action_counts: Dict[str, int] = {}
        for profile in profiles:
            for action in profile.recommended_actions:
                action_counts[action] = action_counts.get(action, 0) + 1

        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        return [a[0] for a in sorted_actions[:5]]

    def _calculate_provenance_hash(
        self,
        input_data: ResilienceScoringInput,
        output: ResilienceScoringOutput
    ) -> str:
        """Calculate SHA-256 hash for provenance."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "assessment_id": input_data.assessment_id,
            "asset_count": len(input_data.assets),
            "average_resilience": output.average_resilience,
            "timestamp": output.completed_at.isoformat(),
        }
        return hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "ResilienceScoringAgent",
    "ResilienceCapacity",
    "ResilienceLevel",
    "ResilienceDimension",
    "DimensionScore",
    "CapacityScore",
    "ResilienceProfile",
    "ResilienceInput",
    "ResilienceScoringInput",
    "ResilienceScoringOutput",
    "RESILIENCE_THRESHOLDS",
    "SECTOR_BENCHMARKS",
]
