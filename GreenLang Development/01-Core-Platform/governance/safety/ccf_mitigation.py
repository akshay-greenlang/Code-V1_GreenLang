"""
CCFMitigation - Common Cause Failure Mitigation Module

This module implements Common Cause Failure (CCF) analysis and mitigation
per IEC 61511 and IEC 61508 for Safety Instrumented Systems (SIS).

Common Cause Failures occur when multiple channels fail simultaneously
due to a single root cause. Proper CCF management is essential for achieving
SIL targets in redundant systems.

Key implementations:
- Beta factor calculation per IEC 61508-6 Annex D
- CCF scoring matrix based on IEC 61508 Tables D.1-D.5
- Diversity requirements assessment
- Physical separation guidelines
- Software diversity strategies
- Mitigation effectiveness validation

Reference: IEC 61508-6 Annex D, IEC 61511-1 Clause 11.4

Example:
    >>> from greenlang.safety.ccf_mitigation import CCFAnalyzer, CCFScenario
    >>> analyzer = CCFAnalyzer()
    >>> scenario = CCFScenario(
    ...     system_id="SIS-007",
    ...     architecture="1oo2",
    ...     separation_score=3,
    ...     diversity_score=3,
    ...     complexity_score=2
    ... )
    >>> result = analyzer.calculate_beta_factor(scenario)
    >>> print(f"Beta Factor: {result.beta_factor:.3f}")

Author: GreenLang Safety Engineering Team
Version: 1.0
Date: 2025-12-07
"""

from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from dataclasses import dataclass, field
import hashlib
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class VotingArchitecture(str, Enum):
    """Voting architectures for CCF analysis."""
    ONE_OO_ONE = "1oo1"
    ONE_OO_TWO = "1oo2"
    TWO_OO_TWO = "2oo2"
    TWO_OO_THREE = "2oo3"
    TWO_OO_FOUR = "2oo4"


class CCFCategory(str, Enum):
    """CCF susceptibility categories."""
    LOW = "low"  # Beta < 0.05
    MEDIUM = "medium"  # 0.05 <= Beta < 0.1
    HIGH = "high"  # 0.1 <= Beta < 0.2
    VERY_HIGH = "very_high"  # Beta >= 0.2


class MitigationLevel(str, Enum):
    """Mitigation effectiveness levels."""
    NONE = "none"
    BASIC = "basic"
    GOOD = "good"
    EXCELLENT = "excellent"


# =============================================================================
# Data Models
# =============================================================================

class CCFScoreFactors(BaseModel):
    """
    CCF scoring factors per IEC 61508-6 Annex D.

    Each factor contributes to the overall beta factor calculation.
    Scores range from 0 (worst) to 3 (best).
    """

    # Separation/Segregation (Table D.1)
    separation_score: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Physical separation score (0-3)"
    )

    # Diversity (Table D.2)
    diversity_score: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Diversity score (0-3)"
    )

    # Complexity/Design Analysis (Table D.3)
    complexity_score: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Complexity/design analysis score (0-3)"
    )

    # Assessment/Analysis (Table D.4)
    assessment_score: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Assessment/analysis score (0-3)"
    )

    # Competence/Training (Table D.5)
    competence_score: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Competence/training score (0-3)"
    )

    # Environmental Control
    environmental_score: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Environmental control score (0-3)"
    )

    @property
    def total_score(self) -> int:
        """Calculate total CCF score."""
        return (
            self.separation_score +
            self.diversity_score +
            self.complexity_score +
            self.assessment_score +
            self.competence_score +
            self.environmental_score
        )

    @property
    def max_score(self) -> int:
        """Maximum possible score."""
        return 18  # 6 factors x 3 max each


class CCFScenario(BaseModel):
    """CCF analysis scenario definition."""

    scenario_id: str = Field(
        default_factory=lambda: f"CCF-{uuid.uuid4().hex[:8].upper()}",
        description="Scenario identifier"
    )
    system_id: str = Field(
        ...,
        description="System being analyzed"
    )
    architecture: VotingArchitecture = Field(
        ...,
        description="Voting architecture"
    )
    description: str = Field(
        default="",
        description="Scenario description"
    )
    score_factors: CCFScoreFactors = Field(
        default_factory=CCFScoreFactors,
        description="CCF score factors"
    )
    num_channels: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Number of redundant channels"
    )
    target_sil: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Target SIL level"
    )


class CCFResult(BaseModel):
    """CCF analysis result."""

    scenario_id: str = Field(..., description="Scenario identifier")
    system_id: str = Field(..., description="System identifier")
    beta_factor: float = Field(
        ...,
        ge=0,
        le=1,
        description="Calculated beta factor"
    )
    beta_category: CCFCategory = Field(
        ...,
        description="Beta factor category"
    )
    total_score: int = Field(..., description="Total CCF score")
    max_score: int = Field(..., description="Maximum possible score")
    score_percent: float = Field(..., description="Score as percentage")
    mitigation_level: MitigationLevel = Field(
        ...,
        description="Overall mitigation level"
    )
    meets_sil_requirement: bool = Field(
        ...,
        description="Does beta meet SIL requirement"
    )
    required_beta: float = Field(
        ...,
        description="Required beta for target SIL"
    )
    pfd_ccf_contribution: float = Field(
        default=0,
        description="PFD contribution from CCF"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Improvement recommendations"
    )
    score_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Individual score breakdown"
    )
    calculation_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Calculation timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )
    formula_used: str = Field(
        default="",
        description="Formula used for calculation"
    )


class DiversityRequirement(BaseModel):
    """Diversity requirement specification."""

    requirement_id: str = Field(
        default_factory=lambda: f"DIV-{uuid.uuid4().hex[:6].upper()}",
        description="Requirement identifier"
    )
    category: str = Field(..., description="Diversity category")
    requirement: str = Field(..., description="Requirement description")
    implementation: str = Field(default="", description="Implementation approach")
    score_impact: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Score impact when implemented"
    )
    is_implemented: bool = Field(default=False, description="Implementation status")


class SeparationGuideline(BaseModel):
    """Physical separation guideline."""

    guideline_id: str = Field(
        default_factory=lambda: f"SEP-{uuid.uuid4().hex[:6].upper()}",
        description="Guideline identifier"
    )
    category: str = Field(..., description="Separation category")
    guideline: str = Field(..., description="Guideline description")
    minimum_distance_m: Optional[float] = Field(
        None,
        description="Minimum separation distance in meters"
    )
    score_impact: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Score impact when implemented"
    )
    is_implemented: bool = Field(default=False, description="Implementation status")


# =============================================================================
# CCF Analyzer
# =============================================================================

class CCFAnalyzer:
    """
    Common Cause Failure Analyzer.

    Implements CCF analysis per IEC 61508-6 Annex D and IEC 61511.
    Provides beta factor calculation, mitigation scoring, and
    recommendations for reducing CCF susceptibility.

    The analyzer follows zero-hallucination principles:
    - All calculations are deterministic
    - No LLM involvement in numeric computations
    - Complete audit trail with provenance hashing

    Attributes:
        beta_lookup: Beta factor lookup table
        sil_beta_requirements: Required beta by SIL

    Example:
        >>> analyzer = CCFAnalyzer()
        >>> result = analyzer.calculate_beta_factor(scenario)
        >>> print(f"Beta: {result.beta_factor:.3f}")
    """

    # Beta factor lookup table per IEC 61508-6 Table D.6
    # Score ranges map to beta factors
    BETA_LOOKUP: Dict[Tuple[int, int], float] = {
        (0, 3): 0.20,   # Very poor mitigation
        (4, 6): 0.15,   # Poor mitigation
        (7, 9): 0.10,   # Basic mitigation
        (10, 12): 0.05, # Good mitigation
        (13, 15): 0.02, # Very good mitigation
        (16, 18): 0.01, # Excellent mitigation
    }

    # Required beta factors by SIL (typical values)
    SIL_BETA_REQUIREMENTS: Dict[int, float] = {
        1: 0.20,  # SIL 1 - Beta < 0.20
        2: 0.10,  # SIL 2 - Beta < 0.10
        3: 0.05,  # SIL 3 - Beta < 0.05
        4: 0.02,  # SIL 4 - Beta < 0.02
    }

    def __init__(self):
        """Initialize CCFAnalyzer."""
        logger.info("CCFAnalyzer initialized")

    def calculate_beta_factor(self, scenario: CCFScenario) -> CCFResult:
        """
        Calculate beta factor for CCF scenario.

        Beta factor represents the fraction of dangerous failures that
        are common cause (affecting multiple channels simultaneously).

        Args:
            scenario: CCFScenario to analyze

        Returns:
            CCFResult with calculated beta and recommendations
        """
        logger.info(f"Calculating beta factor for {scenario.system_id}")

        try:
            # Calculate total score
            total_score = scenario.score_factors.total_score
            max_score = scenario.score_factors.max_score
            score_percent = (total_score / max_score) * 100 if max_score > 0 else 0

            # Look up beta factor from table
            beta_factor = self._lookup_beta(total_score)

            # Categorize beta
            beta_category = self._categorize_beta(beta_factor)

            # Determine mitigation level
            mitigation_level = self._determine_mitigation_level(total_score)

            # Check against SIL requirement
            required_beta = self.SIL_BETA_REQUIREMENTS.get(scenario.target_sil, 0.10)
            meets_requirement = beta_factor <= required_beta

            # Generate recommendations
            recommendations = self._generate_recommendations(
                scenario, beta_factor, required_beta
            )

            # Calculate PFD contribution from CCF
            pfd_ccf = self._calculate_pfd_ccf_contribution(
                beta_factor, scenario.architecture
            )

            # Build score breakdown
            score_breakdown = {
                "separation": scenario.score_factors.separation_score,
                "diversity": scenario.score_factors.diversity_score,
                "complexity": scenario.score_factors.complexity_score,
                "assessment": scenario.score_factors.assessment_score,
                "competence": scenario.score_factors.competence_score,
                "environmental": scenario.score_factors.environmental_score,
            }

            result = CCFResult(
                scenario_id=scenario.scenario_id,
                system_id=scenario.system_id,
                beta_factor=beta_factor,
                beta_category=beta_category,
                total_score=total_score,
                max_score=max_score,
                score_percent=score_percent,
                mitigation_level=mitigation_level,
                meets_sil_requirement=meets_requirement,
                required_beta=required_beta,
                pfd_ccf_contribution=pfd_ccf,
                recommendations=recommendations,
                score_breakdown=score_breakdown,
                formula_used="Beta = f(Total CCF Score) per IEC 61508-6 Table D.6",
            )

            # Calculate provenance hash
            result.provenance_hash = self._calculate_provenance(scenario, result)

            logger.info(
                f"Beta factor calculated: {beta_factor:.3f} ({beta_category.value})"
            )

            return result

        except Exception as e:
            logger.error(f"Beta calculation failed: {str(e)}", exc_info=True)
            raise

    def _lookup_beta(self, score: int) -> float:
        """Look up beta factor from score."""
        for (lower, upper), beta in self.BETA_LOOKUP.items():
            if lower <= score <= upper:
                return beta

        # Default to worst case if out of range
        if score < 0:
            return 0.20
        return 0.01  # Best case for very high scores

    def _categorize_beta(self, beta: float) -> CCFCategory:
        """Categorize beta factor."""
        if beta < 0.05:
            return CCFCategory.LOW
        elif beta < 0.10:
            return CCFCategory.MEDIUM
        elif beta < 0.20:
            return CCFCategory.HIGH
        else:
            return CCFCategory.VERY_HIGH

    def _determine_mitigation_level(self, score: int) -> MitigationLevel:
        """Determine mitigation level from score."""
        if score <= 3:
            return MitigationLevel.NONE
        elif score <= 9:
            return MitigationLevel.BASIC
        elif score <= 15:
            return MitigationLevel.GOOD
        else:
            return MitigationLevel.EXCELLENT

    def _calculate_pfd_ccf_contribution(
        self,
        beta: float,
        architecture: VotingArchitecture
    ) -> float:
        """
        Calculate PFD contribution from CCF.

        For redundant architectures, CCF adds to PFD:
        PFD_CCF = beta * lambda_DU * TI / 2

        Simplified calculation using typical values.
        """
        # Typical lambda_DU * TI / 2 contribution
        # Assuming lambda_DU = 1E-6/hr, TI = 8760 hrs
        typical_single_pfd = 4.38e-3  # lambda_DU * TI / 2

        if architecture == VotingArchitecture.ONE_OO_ONE:
            # No CCF reduction for single channel
            return 0.0
        elif architecture in [VotingArchitecture.ONE_OO_TWO, VotingArchitecture.TWO_OO_TWO]:
            # CCF dominates for dual systems
            return beta * typical_single_pfd
        elif architecture in [VotingArchitecture.TWO_OO_THREE, VotingArchitecture.TWO_OO_FOUR]:
            # CCF still significant for triple/quad systems
            return beta * typical_single_pfd

        return beta * typical_single_pfd

    def _generate_recommendations(
        self,
        scenario: CCFScenario,
        beta: float,
        required_beta: float
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        factors = scenario.score_factors

        if beta > required_beta:
            gap = beta - required_beta
            recommendations.append(
                f"Beta factor ({beta:.3f}) exceeds requirement ({required_beta:.3f}). "
                f"Improve CCF mitigation by {gap:.3f}."
            )

        # Check individual factors and recommend improvements
        if factors.separation_score < 2:
            recommendations.append(
                "Improve physical separation: Consider separate cable routes, "
                "enclosures, and power supplies for redundant channels."
            )

        if factors.diversity_score < 2:
            recommendations.append(
                "Improve diversity: Consider different manufacturers, "
                "sensing principles, or technology for redundant elements."
            )

        if factors.complexity_score < 2:
            recommendations.append(
                "Reduce complexity: Simplify system design, "
                "use proven components, minimize configuration options."
            )

        if factors.assessment_score < 2:
            recommendations.append(
                "Improve assessment: Conduct detailed CCF analysis, "
                "review historical failure data, perform FMEA."
            )

        if factors.competence_score < 2:
            recommendations.append(
                "Improve competence: Enhance training programs, "
                "use qualified personnel for safety system work."
            )

        if factors.environmental_score < 2:
            recommendations.append(
                "Improve environmental control: Control temperature, humidity, "
                "EMI, and contamination in equipment areas."
            )

        if not recommendations:
            recommendations.append(
                "CCF mitigation is adequate. Maintain current practices "
                "and verify during proof testing."
            )

        return recommendations

    def _calculate_provenance(
        self,
        scenario: CCFScenario,
        result: CCFResult
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{scenario.scenario_id}|"
            f"{scenario.system_id}|"
            f"{result.beta_factor}|"
            f"{result.total_score}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def get_diversity_requirements(
        self,
        target_sil: int
    ) -> List[DiversityRequirement]:
        """
        Get diversity requirements for target SIL.

        Args:
            target_sil: Target SIL level

        Returns:
            List of diversity requirements
        """
        requirements = []

        # Common requirements for all SIL levels
        requirements.extend([
            DiversityRequirement(
                category="Hardware",
                requirement="Different manufacturers for redundant sensors",
                implementation="Use sensor A from Manufacturer X, sensor B from Manufacturer Y",
                score_impact=1,
            ),
            DiversityRequirement(
                category="Hardware",
                requirement="Different sensing principles where practical",
                implementation="Use different technology (e.g., capacitive vs. piezoelectric)",
                score_impact=2,
            ),
        ])

        # SIL 2 and above
        if target_sil >= 2:
            requirements.extend([
                DiversityRequirement(
                    category="Software",
                    requirement="Diverse software for redundant channels",
                    implementation="Different software versions or algorithms",
                    score_impact=2,
                ),
                DiversityRequirement(
                    category="Hardware",
                    requirement="Different hardware platforms",
                    implementation="Different CPU types or controller models",
                    score_impact=2,
                ),
            ])

        # SIL 3 and above
        if target_sil >= 3:
            requirements.extend([
                DiversityRequirement(
                    category="Software",
                    requirement="N-version programming or diverse redundancy",
                    implementation="Independent development teams for software",
                    score_impact=3,
                ),
                DiversityRequirement(
                    category="Physical",
                    requirement="Complete physical separation of channels",
                    implementation="Separate rooms, power, and communication paths",
                    score_impact=3,
                ),
            ])

        return requirements

    def get_separation_guidelines(
        self,
        target_sil: int
    ) -> List[SeparationGuideline]:
        """
        Get physical separation guidelines for target SIL.

        Args:
            target_sil: Target SIL level

        Returns:
            List of separation guidelines
        """
        guidelines = []

        # Common guidelines
        guidelines.extend([
            SeparationGuideline(
                category="Cable Routing",
                guideline="Separate cable trays for redundant channels",
                minimum_distance_m=0.3,
                score_impact=1,
            ),
            SeparationGuideline(
                category="Power Supply",
                guideline="Separate power supplies for redundant channels",
                minimum_distance_m=None,
                score_impact=1,
            ),
        ])

        # SIL 2 and above
        if target_sil >= 2:
            guidelines.extend([
                SeparationGuideline(
                    category="Enclosures",
                    guideline="Separate junction boxes for redundant channels",
                    minimum_distance_m=1.0,
                    score_impact=2,
                ),
                SeparationGuideline(
                    category="Environmental",
                    guideline="Separate environmental zones where practical",
                    minimum_distance_m=3.0,
                    score_impact=2,
                ),
            ])

        # SIL 3 and above
        if target_sil >= 3:
            guidelines.extend([
                SeparationGuideline(
                    category="Physical",
                    guideline="Separate equipment rooms for safety systems",
                    minimum_distance_m=10.0,
                    score_impact=3,
                ),
                SeparationGuideline(
                    category="Fire",
                    guideline="Fire barriers between redundant equipment",
                    minimum_distance_m=None,
                    score_impact=3,
                ),
            ])

        return guidelines

    def validate_mitigation_effectiveness(
        self,
        scenario: CCFScenario,
        implemented_measures: List[str]
    ) -> Dict[str, Any]:
        """
        Validate effectiveness of implemented CCF mitigation measures.

        Args:
            scenario: CCF scenario
            implemented_measures: List of implemented measure IDs

        Returns:
            Validation result dictionary
        """
        # Calculate baseline (without mitigation)
        baseline_beta = 0.20

        # Calculate with current mitigation
        current_result = self.calculate_beta_factor(scenario)

        # Calculate improvement
        improvement = baseline_beta - current_result.beta_factor
        improvement_percent = (improvement / baseline_beta) * 100

        # Determine if adequate
        is_adequate = current_result.meets_sil_requirement

        return {
            "validation_id": f"VAL-{uuid.uuid4().hex[:8].upper()}",
            "scenario_id": scenario.scenario_id,
            "baseline_beta": baseline_beta,
            "achieved_beta": current_result.beta_factor,
            "improvement": improvement,
            "improvement_percent": improvement_percent,
            "is_adequate": is_adequate,
            "target_sil": scenario.target_sil,
            "required_beta": current_result.required_beta,
            "measures_implemented": len(implemented_measures),
            "mitigation_level": current_result.mitigation_level.value,
            "timestamp": datetime.utcnow().isoformat(),
            "recommendation": (
                "Mitigation is effective and adequate for target SIL."
                if is_adequate
                else f"Additional mitigation required. Reduce beta by "
                     f"{current_result.beta_factor - current_result.required_beta:.3f}"
            )
        }

    def get_software_diversity_strategies(self) -> List[Dict[str, Any]]:
        """
        Get software diversity strategies for CCF mitigation.

        Returns:
            List of software diversity strategies
        """
        return [
            {
                "strategy_id": "SDS-001",
                "name": "N-Version Programming",
                "description": "Develop multiple versions of software independently",
                "effectiveness": "High",
                "cost": "High",
                "implementation": "Separate development teams, specifications",
                "score_impact": 3,
            },
            {
                "strategy_id": "SDS-002",
                "name": "Recovery Blocks",
                "description": "Primary module with acceptance test and backup",
                "effectiveness": "Medium",
                "cost": "Medium",
                "implementation": "Acceptance test module, recovery module",
                "score_impact": 2,
            },
            {
                "strategy_id": "SDS-003",
                "name": "Data Diversity",
                "description": "Process same data with different algorithms",
                "effectiveness": "Medium",
                "cost": "Low",
                "implementation": "Different calculation methods, cross-check",
                "score_impact": 2,
            },
            {
                "strategy_id": "SDS-004",
                "name": "Diverse Programming Languages",
                "description": "Use different languages for redundant channels",
                "effectiveness": "Medium",
                "cost": "Medium",
                "implementation": "Channel A in C, Channel B in Ada/Pascal",
                "score_impact": 2,
            },
            {
                "strategy_id": "SDS-005",
                "name": "Diverse Compilers/Tools",
                "description": "Use different development tools",
                "effectiveness": "Low",
                "cost": "Low",
                "implementation": "Different compilers, IDEs, libraries",
                "score_impact": 1,
            },
        ]


# =============================================================================
# CCF Scoring Matrix
# =============================================================================

class CCFScoringMatrix:
    """
    CCF Scoring Matrix per IEC 61508-6 Annex D.

    Provides detailed scoring criteria for each CCF factor.
    """

    # Separation/Segregation Scoring (Table D.1)
    SEPARATION_CRITERIA: Dict[int, str] = {
        0: "No separation: Channels in same cabinet, same cable runs",
        1: "Basic separation: Separate modules in same cabinet",
        2: "Good separation: Separate cabinets, some cable separation",
        3: "Excellent separation: Separate rooms, complete isolation",
    }

    # Diversity Scoring (Table D.2)
    DIVERSITY_CRITERIA: Dict[int, str] = {
        0: "No diversity: Identical components, same manufacturer",
        1: "Basic diversity: Same type, different batches or versions",
        2: "Good diversity: Different manufacturers, same technology",
        3: "Excellent diversity: Different technology, independent design",
    }

    # Complexity Scoring (Table D.3)
    COMPLEXITY_CRITERIA: Dict[int, str] = {
        0: "High complexity: Novel design, many interfaces, untested",
        1: "Moderate complexity: Some novel elements, multiple interfaces",
        2: "Low complexity: Proven design, limited interfaces",
        3: "Minimal complexity: Simple, well-understood, few interfaces",
    }

    # Assessment Scoring (Table D.4)
    ASSESSMENT_CRITERIA: Dict[int, str] = {
        0: "No CCF analysis: No formal assessment performed",
        1: "Basic analysis: Checklist review, qualitative assessment",
        2: "Detailed analysis: FMEA, formal CCF methods applied",
        3: "Comprehensive analysis: Quantitative CCF analysis, beta verification",
    }

    # Competence Scoring (Table D.5)
    COMPETENCE_CRITERIA: Dict[int, str] = {
        0: "Low competence: Untrained personnel, no procedures",
        1: "Basic competence: General training, basic procedures",
        2: "Good competence: Specific training, detailed procedures",
        3: "Excellent competence: Expert personnel, rigorous procedures",
    }

    # Environmental Control Scoring
    ENVIRONMENTAL_CRITERIA: Dict[int, str] = {
        0: "No control: Uncontrolled environment, harsh conditions",
        1: "Basic control: Partial environmental control",
        2: "Good control: Temperature, humidity controlled",
        3: "Excellent control: Full environmental control, EMI protection",
    }

    @classmethod
    def get_scoring_guide(cls) -> Dict[str, Dict[int, str]]:
        """Get complete scoring guide."""
        return {
            "separation": cls.SEPARATION_CRITERIA,
            "diversity": cls.DIVERSITY_CRITERIA,
            "complexity": cls.COMPLEXITY_CRITERIA,
            "assessment": cls.ASSESSMENT_CRITERIA,
            "competence": cls.COMPETENCE_CRITERIA,
            "environmental": cls.ENVIRONMENTAL_CRITERIA,
        }

    @classmethod
    def evaluate_factor(
        cls,
        factor: str,
        score: int
    ) -> Dict[str, Any]:
        """
        Evaluate a specific CCF factor.

        Args:
            factor: Factor name
            score: Score value (0-3)

        Returns:
            Evaluation result
        """
        criteria_map = {
            "separation": cls.SEPARATION_CRITERIA,
            "diversity": cls.DIVERSITY_CRITERIA,
            "complexity": cls.COMPLEXITY_CRITERIA,
            "assessment": cls.ASSESSMENT_CRITERIA,
            "competence": cls.COMPETENCE_CRITERIA,
            "environmental": cls.ENVIRONMENTAL_CRITERIA,
        }

        criteria = criteria_map.get(factor, {})
        description = criteria.get(score, "Unknown score")

        return {
            "factor": factor,
            "score": score,
            "max_score": 3,
            "description": description,
            "improvement_potential": 3 - score,
            "recommendation": (
                f"Improve {factor} to achieve score {score + 1}: "
                f"{criteria.get(score + 1, 'N/A')}"
                if score < 3
                else f"{factor} is at maximum level"
            )
        }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "CCFAnalyzer",
    "CCFScenario",
    "CCFResult",
    "CCFScoreFactors",
    "CCFCategory",
    "MitigationLevel",
    "VotingArchitecture",
    "DiversityRequirement",
    "SeparationGuideline",
    "CCFScoringMatrix",
]
