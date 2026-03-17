# -*- coding: utf-8 -*-
"""
AdverseImpactEngine - PACK-019 CSDDD Adverse Impact Assessment Engine
======================================================================

Identifies, assesses, and prioritises adverse human rights and
environmental impacts across a company's own operations and value
chain, in accordance with CSDDD Articles 6 and 7.

The EU Corporate Sustainability Due Diligence Directive (CSDDD /
Directive 2024/1760) requires in-scope companies to identify actual
and potential adverse impacts on human rights and the environment
arising from their own operations, those of their subsidiaries, and
those carried out by their business partners in the chain of activities.

CSDDD Art 6 - Identifying Adverse Impacts:
    - Para 1: Companies shall take appropriate measures to identify
      actual and potential adverse impacts on human rights and the
      environment.
    - Para 2: Identification shall cover own operations, subsidiaries,
      and business partners in the chain of activities.
    - Para 3: Use appropriate resources and information for mapping.

CSDDD Art 7 - Prioritising Identified Adverse Impacts:
    - Para 1: Where it is not feasible to address all adverse impacts
      simultaneously, companies shall prioritise based on severity
      and likelihood.
    - Para 2: Severity shall be assessed based on scale, scope, and
      irremediable character.
    - Para 3: Once the most severe impacts are addressed, the company
      shall address impacts of lesser severity.

Human Rights Impacts (Annex Part I):
    - Rights derived from international human rights instruments
      including ICCPR, ICESCR, UDHR, ILO conventions, UNDRIP, etc.
    - Categories include forced labour, child labour, workplace safety,
      freedom of association, living wages, land rights, etc.

Environmental Impacts (Annex Part II):
    - Environmental prohibitions and obligations from MEAs including
      Minamata Convention, Stockholm Convention, Basel Convention,
      CBD, CITES, Paris Agreement, etc.

Risk Scoring Methodology:
    - Severity: CRITICAL=4, HIGH=3, MEDIUM=2, LOW=1
    - Likelihood: VERY_LIKELY=5, LIKELY=4, POSSIBLE=3, UNLIKELY=2, RARE=1
    - Risk score = severity_score x likelihood_score (range 1-20)
    - Priority ranking by descending risk score

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D), Articles 6-7
    - CSDDD Annex Part I (Human Rights instruments)
    - CSDDD Annex Part II (Environmental instruments)
    - UN Guiding Principles on Business and Human Rights
    - OECD Due Diligence Guidance for Responsible Business Conduct
    - ILO Declaration on Fundamental Principles and Rights at Work
    - Paris Agreement (2015)

Zero-Hallucination:
    - Risk scores computed via integer multiplication
    - Priority ranking uses deterministic sort
    - Summary statistics use count-based aggregation
    - All percentages use Decimal arithmetic
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-019 CSDDD Readiness
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value using ROUND_HALF_UP.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))


def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))


def _pct(part: int, total: int) -> Decimal:
    """Calculate percentage as Decimal, rounded to 1 decimal place."""
    if total == 0:
        return Decimal("0.0")
    return _round_val(
        _decimal(part) / _decimal(total) * Decimal("100"), 1
    )


def _pct_dec(part: Decimal, total: Decimal) -> Decimal:
    """Calculate percentage from Decimal values, rounded to 1 dp."""
    if total == Decimal("0"):
        return Decimal("0.0")
    return _round_val(part / total * Decimal("100"), 1)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AdverseImpactType(str, Enum):
    """Type of adverse impact under CSDDD.

    CSDDD distinguishes between adverse impacts on human rights
    (Annex Part I) and adverse environmental impacts (Annex Part II).
    """
    HUMAN_RIGHTS = "human_rights"
    ENVIRONMENTAL = "environmental"


class ImpactSeverity(str, Enum):
    """Severity classification for an adverse impact.

    Per CSDDD Art 7 Para 2, severity is assessed based on the
    scale, scope, and irremediable character of the adverse impact.
    Mapped to numeric scores for risk matrix calculation.

    - CRITICAL: Systemic, widespread, irreversible impact (score 4)
    - HIGH: Significant impact with limited reversibility (score 3)
    - MEDIUM: Moderate impact, largely reversible (score 2)
    - LOW: Minor, fully reversible impact (score 1)
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ImpactLikelihood(str, Enum):
    """Likelihood classification for a potential adverse impact.

    Probability of the adverse impact occurring, used together
    with severity for risk scoring per CSDDD Art 7.

    - VERY_LIKELY: Expected to occur (>90% probability, score 5)
    - LIKELY: More likely than not (60-90% probability, score 4)
    - POSSIBLE: Moderate chance (30-60% probability, score 3)
    - UNLIKELY: Low chance (10-30% probability, score 2)
    - RARE: Exceptional circumstances (<10% probability, score 1)
    """
    VERY_LIKELY = "very_likely"
    LIKELY = "likely"
    POSSIBLE = "possible"
    UNLIKELY = "unlikely"
    RARE = "rare"


class ImpactStatus(str, Enum):
    """Status of the adverse impact: actual or potential.

    CSDDD distinguishes between impacts that have already occurred
    (actual) and impacts that are reasonably foreseeable (potential).
    """
    ACTUAL = "actual"
    POTENTIAL = "potential"


class ValueChainPosition(str, Enum):
    """Position in the value chain where the impact occurs.

    CSDDD requires identification across own operations, upstream
    (suppliers) and downstream (customers/end users) value chain
    including both direct and indirect business relationships.
    """
    OWN_OPERATIONS = "own_operations"
    UPSTREAM_DIRECT = "upstream_direct"
    UPSTREAM_INDIRECT = "upstream_indirect"
    DOWNSTREAM_DIRECT = "downstream_direct"
    DOWNSTREAM_INDIRECT = "downstream_indirect"


class HumanRightsCategory(str, Enum):
    """Human rights impact categories from CSDDD Annex Part I.

    Derived from international human rights instruments including
    the UDHR, ICCPR, ICESCR, and ILO conventions referenced in
    Annex Part I of Directive 2024/1760.
    """
    FORCED_LABOUR = "forced_labour"
    CHILD_LABOUR = "child_labour"
    WORKPLACE_SAFETY = "workplace_safety"
    FREEDOM_OF_ASSOCIATION = "freedom_of_association"
    COLLECTIVE_BARGAINING = "collective_bargaining"
    LIVING_WAGE = "living_wage"
    DISCRIMINATION = "discrimination"
    WORKING_HOURS = "working_hours"
    LAND_RIGHTS = "land_rights"
    INDIGENOUS_PEOPLES_RIGHTS = "indigenous_peoples_rights"
    RIGHT_TO_PRIVACY = "right_to_privacy"
    SECURITY_FORCES = "security_forces"


class EnvironmentalCategory(str, Enum):
    """Environmental impact categories from CSDDD Annex Part II.

    Derived from multilateral environmental agreements referenced
    in Annex Part II of Directive 2024/1760, including the
    Minamata Convention, Stockholm Convention, Basel Convention,
    CBD, and Paris Agreement.
    """
    MERCURY_POLLUTION = "mercury_pollution"
    PERSISTENT_ORGANIC_POLLUTANTS = "persistent_organic_pollutants"
    HAZARDOUS_WASTE = "hazardous_waste"
    BIODIVERSITY_LOSS = "biodiversity_loss"
    WILDLIFE_TRAFFICKING = "wildlife_trafficking"
    GHG_EMISSIONS = "ghg_emissions"
    DEFORESTATION = "deforestation"
    WATER_POLLUTION = "water_pollution"
    SOIL_CONTAMINATION = "soil_contamination"


class RiskLevel(str, Enum):
    """Qualitative risk level derived from the risk matrix.

    Computed from the product of severity and likelihood scores.
    Used for categorisation and dashboard reporting.
    """
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Severity numeric scores
SEVERITY_SCORES: Dict[str, int] = {
    ImpactSeverity.CRITICAL.value: 4,
    ImpactSeverity.HIGH.value: 3,
    ImpactSeverity.MEDIUM.value: 2,
    ImpactSeverity.LOW.value: 1,
}

# Likelihood numeric scores
LIKELIHOOD_SCORES: Dict[str, int] = {
    ImpactLikelihood.VERY_LIKELY.value: 5,
    ImpactLikelihood.LIKELY.value: 4,
    ImpactLikelihood.POSSIBLE.value: 3,
    ImpactLikelihood.UNLIKELY.value: 2,
    ImpactLikelihood.RARE.value: 1,
}

# Risk level thresholds (risk_score = severity * likelihood)
# Max risk score is 4 * 5 = 20
RISK_LEVEL_THRESHOLDS: List[Tuple[int, RiskLevel]] = [
    (16, RiskLevel.CRITICAL),     # 16-20
    (12, RiskLevel.HIGH),         # 12-15
    (6, RiskLevel.MEDIUM),        # 6-11
    (3, RiskLevel.LOW),           # 3-5
    (0, RiskLevel.VERY_LOW),      # 1-2
]

# Human rights category labels for reporting
HR_CATEGORY_LABELS: Dict[str, str] = {
    HumanRightsCategory.FORCED_LABOUR.value: "Forced labour and modern slavery",
    HumanRightsCategory.CHILD_LABOUR.value: "Child labour",
    HumanRightsCategory.WORKPLACE_SAFETY.value: "Occupational health and safety",
    HumanRightsCategory.FREEDOM_OF_ASSOCIATION.value: "Freedom of association",
    HumanRightsCategory.COLLECTIVE_BARGAINING.value: "Collective bargaining rights",
    HumanRightsCategory.LIVING_WAGE.value: "Adequate living wage",
    HumanRightsCategory.DISCRIMINATION.value: "Non-discrimination and equal treatment",
    HumanRightsCategory.WORKING_HOURS.value: "Working hours and rest periods",
    HumanRightsCategory.LAND_RIGHTS.value: "Land, forest, and water rights",
    HumanRightsCategory.INDIGENOUS_PEOPLES_RIGHTS.value: "Rights of indigenous peoples",
    HumanRightsCategory.RIGHT_TO_PRIVACY.value: "Right to privacy",
    HumanRightsCategory.SECURITY_FORCES.value: "Use of security forces",
}

# Environmental category labels for reporting
ENV_CATEGORY_LABELS: Dict[str, str] = {
    EnvironmentalCategory.MERCURY_POLLUTION.value: "Mercury pollution (Minamata Convention)",
    EnvironmentalCategory.PERSISTENT_ORGANIC_POLLUTANTS.value: "POPs (Stockholm Convention)",
    EnvironmentalCategory.HAZARDOUS_WASTE.value: "Hazardous waste (Basel Convention)",
    EnvironmentalCategory.BIODIVERSITY_LOSS.value: "Biodiversity loss (CBD)",
    EnvironmentalCategory.WILDLIFE_TRAFFICKING.value: "Wildlife trafficking (CITES)",
    EnvironmentalCategory.GHG_EMISSIONS.value: "GHG emissions (Paris Agreement)",
    EnvironmentalCategory.DEFORESTATION.value: "Deforestation and land degradation",
    EnvironmentalCategory.WATER_POLLUTION.value: "Water pollution and depletion",
    EnvironmentalCategory.SOIL_CONTAMINATION.value: "Soil contamination",
}


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class AdverseImpact(BaseModel):
    """A single identified adverse impact per CSDDD Art 6.

    Represents one actual or potential adverse impact on human rights
    or the environment, identified within the company's own operations
    or value chain.

    Attributes:
        impact_id: Unique identifier for this impact.
        impact_type: Whether the impact is on human rights or environment.
        category: Specific category of the impact.
        description: Narrative description of the impact.
        severity: Severity classification (CRITICAL/HIGH/MEDIUM/LOW).
        likelihood: Likelihood classification.
        status: Whether the impact is actual or potential.
        value_chain_position: Where in the value chain the impact occurs.
        affected_stakeholders: Description of affected groups.
        affected_stakeholder_count: Estimated number of affected people.
        linked_rights: List of linked international rights instruments.
        country: ISO 3166-1 alpha-2 country code.
        sector: Sector where impact occurs.
        subsidiary_name: Name of subsidiary if applicable.
        business_partner_name: Name of business partner if applicable.
        identified_date: Date the impact was identified.
        last_assessed_date: Date of most recent assessment.
    """
    impact_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this impact",
    )
    impact_type: AdverseImpactType = Field(
        ...,
        description="Whether the impact is on human rights or the environment",
    )
    category: str = Field(
        ...,
        description="Specific category of the adverse impact",
        max_length=200,
    )
    description: str = Field(
        default="",
        description="Narrative description of the adverse impact",
        max_length=5000,
    )
    severity: ImpactSeverity = Field(
        ...,
        description="Severity classification of the impact",
    )
    likelihood: ImpactLikelihood = Field(
        ...,
        description="Likelihood of the impact occurring or recurring",
    )
    status: ImpactStatus = Field(
        ...,
        description="Whether the impact is actual or potential",
    )
    value_chain_position: ValueChainPosition = Field(
        ...,
        description="Position in the value chain where the impact occurs",
    )
    affected_stakeholders: str = Field(
        default="",
        description="Description of affected stakeholder groups",
        max_length=2000,
    )
    affected_stakeholder_count: int = Field(
        default=0,
        description="Estimated number of affected persons",
        ge=0,
    )
    linked_rights: List[str] = Field(
        default_factory=list,
        description="Linked international rights instruments or conventions",
    )
    country: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 country code where impact occurs",
        max_length=3,
    )
    sector: str = Field(
        default="",
        description="Sector or industry where the impact occurs",
        max_length=200,
    )
    subsidiary_name: str = Field(
        default="",
        description="Name of subsidiary where the impact occurs (if applicable)",
        max_length=500,
    )
    business_partner_name: str = Field(
        default="",
        description="Name of business partner (if applicable)",
        max_length=500,
    )
    identified_date: Optional[datetime] = Field(
        default=None,
        description="Date the impact was first identified",
    )
    last_assessed_date: Optional[datetime] = Field(
        default=None,
        description="Date of the most recent assessment",
    )
    is_salient: bool = Field(
        default=False,
        description="Whether this is a salient human rights issue",
    )
    scale: str = Field(
        default="",
        description="Scale of the impact (e.g. number of people, area affected)",
        max_length=500,
    )
    scope: str = Field(
        default="",
        description="Scope of the impact (geographic/operational breadth)",
        max_length=500,
    )
    irremediable_character: bool = Field(
        default=False,
        description="Whether the impact is irremediable or irreversible",
    )

    @field_validator("country")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country code is uppercase alphabetic."""
        if v and not v.isalpha():
            raise ValueError("Country code must be alphabetic")
        return v.upper()


class RiskMatrix(BaseModel):
    """Risk matrix result for a single adverse impact.

    Computes the quantitative risk score from severity and likelihood
    and maps it to a qualitative risk level for prioritisation
    per CSDDD Art 7.
    """
    impact_id: str = Field(
        default="",
        description="Reference to the adverse impact",
    )
    severity_score: int = Field(
        ...,
        description="Numeric severity score (1-4)",
        ge=1,
        le=4,
    )
    likelihood_score: int = Field(
        ...,
        description="Numeric likelihood score (1-5)",
        ge=1,
        le=5,
    )
    risk_score: int = Field(
        ...,
        description="Combined risk score (severity x likelihood, 1-20)",
        ge=1,
        le=20,
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Qualitative risk level derived from risk score",
    )
    priority_rank: int = Field(
        default=0,
        description="Priority rank (1 = highest priority)",
        ge=0,
    )


class SummaryStatistics(BaseModel):
    """Summary statistics for a collection of adverse impacts.

    Provides aggregated counts and distributions across impact type,
    severity, likelihood, status, and value chain position for
    management reporting and dashboard presentation.
    """
    total_impacts: int = Field(
        default=0,
        description="Total number of identified adverse impacts",
        ge=0,
    )
    human_rights_count: int = Field(
        default=0,
        description="Count of human rights impacts",
        ge=0,
    )
    environmental_count: int = Field(
        default=0,
        description="Count of environmental impacts",
        ge=0,
    )
    actual_count: int = Field(
        default=0,
        description="Count of actual (occurred) impacts",
        ge=0,
    )
    potential_count: int = Field(
        default=0,
        description="Count of potential impacts",
        ge=0,
    )
    by_severity: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of impacts by severity level",
    )
    by_likelihood: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of impacts by likelihood level",
    )
    by_risk_level: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of impacts by risk level",
    )
    by_value_chain_position: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of impacts by value chain position",
    )
    by_category: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of impacts by category",
    )
    by_country: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of impacts by country",
    )
    salient_issues_count: int = Field(
        default=0,
        description="Count of salient human rights issues",
        ge=0,
    )
    irremediable_count: int = Field(
        default=0,
        description="Count of irremediable impacts",
        ge=0,
    )
    total_affected_stakeholders: int = Field(
        default=0,
        description="Total estimated affected stakeholders",
        ge=0,
    )
    average_risk_score: Decimal = Field(
        default=Decimal("0"),
        description="Average risk score across all impacts",
    )
    critical_and_high_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of impacts rated critical or high",
    )
    own_operations_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of impacts in own operations",
    )
    value_chain_pct: Decimal = Field(
        default=Decimal("0.0"),
        description="Percentage of impacts in value chain",
    )


class ImpactAssessmentResult(BaseModel):
    """Complete adverse impact assessment result.

    Aggregates all identified impacts, risk matrix calculations,
    prioritised impact list, and summary statistics into a single
    auditable result with provenance tracking.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this assessment",
    )
    impacts: List[AdverseImpact] = Field(
        default_factory=list,
        description="All identified adverse impacts",
    )
    risk_matrices: List[RiskMatrix] = Field(
        default_factory=list,
        description="Risk matrix calculation for each impact",
    )
    priority_impacts: List[str] = Field(
        default_factory=list,
        description="Impact IDs ordered by priority (highest first)",
    )
    summary_stats: SummaryStatistics = Field(
        default_factory=SummaryStatistics,
        description="Aggregate summary statistics",
    )
    top_risks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top 10 highest-priority impacts with details",
    )
    category_risk_summary: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Risk summary aggregated by category",
    )
    value_chain_risk_summary: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Risk summary aggregated by value chain position",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Assessment timestamp (UTC)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail provenance",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AdverseImpactEngine:
    """CSDDD Adverse Impact assessment engine.

    Provides deterministic, zero-hallucination assessment and
    prioritisation of adverse human rights and environmental impacts
    per CSDDD Articles 6 and 7.

    The engine:
    1. Calculates risk matrix scores for each identified impact
       using severity x likelihood integer multiplication.
    2. Prioritises impacts by descending risk score with
       deterministic tie-breaking by severity then impact_id.
    3. Computes summary statistics across all dimensions.
    4. Generates category-level and value-chain-level risk summaries.

    All calculations use deterministic formulas with no LLM involvement.
    Every result includes a SHA-256 provenance hash for audit trails.

    Usage::

        engine = AdverseImpactEngine()
        impacts = [
            AdverseImpact(
                impact_type=AdverseImpactType.HUMAN_RIGHTS,
                category=HumanRightsCategory.CHILD_LABOUR.value,
                severity=ImpactSeverity.CRITICAL,
                likelihood=ImpactLikelihood.LIKELY,
                status=ImpactStatus.POTENTIAL,
                value_chain_position=ValueChainPosition.UPSTREAM_DIRECT,
            ),
        ]
        result = engine.assess_impacts(impacts)
        assert result.provenance_hash != ""
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Risk Matrix Calculation                                              #
    # ------------------------------------------------------------------ #

    def calculate_risk_matrix(self, impact: AdverseImpact) -> RiskMatrix:
        """Calculate the risk matrix for a single adverse impact.

        Computes severity_score * likelihood_score and maps the
        result to a qualitative risk level using fixed thresholds.

        Args:
            impact: AdverseImpact to assess.

        Returns:
            RiskMatrix with numeric scores and risk level.
        """
        severity_score = SEVERITY_SCORES.get(impact.severity.value, 1)
        likelihood_score = LIKELIHOOD_SCORES.get(impact.likelihood.value, 1)
        risk_score = severity_score * likelihood_score

        risk_level = self._score_to_risk_level(risk_score)

        return RiskMatrix(
            impact_id=impact.impact_id,
            severity_score=severity_score,
            likelihood_score=likelihood_score,
            risk_score=risk_score,
            risk_level=risk_level,
            priority_rank=0,  # Set during prioritisation
        )

    @staticmethod
    def _score_to_risk_level(risk_score: int) -> RiskLevel:
        """Map a numeric risk score to a qualitative RiskLevel.

        Uses fixed threshold ranges:
        - 16-20: CRITICAL
        - 12-15: HIGH
        - 6-11:  MEDIUM
        - 3-5:   LOW
        - 1-2:   VERY_LOW

        Args:
            risk_score: Numeric risk score (1-20).

        Returns:
            RiskLevel enum value.
        """
        for threshold, level in RISK_LEVEL_THRESHOLDS:
            if risk_score >= threshold:
                return level
        return RiskLevel.VERY_LOW

    # ------------------------------------------------------------------ #
    # Impact Prioritisation                                                #
    # ------------------------------------------------------------------ #

    def prioritize_impacts(
        self,
        impacts: List[AdverseImpact],
        risk_matrices: List[RiskMatrix],
    ) -> Tuple[List[str], List[RiskMatrix]]:
        """Prioritise impacts by risk score (highest first).

        Sorts impacts in descending order of risk score per CSDDD
        Art 7.  Tie-breaking uses severity_score (descending) then
        impact_id (ascending, deterministic).

        Args:
            impacts: List of AdverseImpact instances.
            risk_matrices: Corresponding RiskMatrix instances.

        Returns:
            Tuple of (priority_impact_ids, updated_risk_matrices).
        """
        if not risk_matrices:
            return [], []

        # Sort by risk_score desc, severity_score desc, impact_id asc
        sorted_matrices = sorted(
            risk_matrices,
            key=lambda rm: (-rm.risk_score, -rm.severity_score, rm.impact_id),
        )

        # Assign priority ranks (1-based)
        for rank, rm in enumerate(sorted_matrices, start=1):
            rm.priority_rank = rank

        priority_ids = [rm.impact_id for rm in sorted_matrices]

        logger.info(
            "Prioritised %d impacts: top risk_score=%d (%s)",
            len(sorted_matrices),
            sorted_matrices[0].risk_score if sorted_matrices else 0,
            sorted_matrices[0].risk_level.value if sorted_matrices else "none",
        )

        return priority_ids, sorted_matrices

    # ------------------------------------------------------------------ #
    # Summary Statistics                                                   #
    # ------------------------------------------------------------------ #

    def get_summary_statistics(
        self,
        impacts: List[AdverseImpact],
        risk_matrices: List[RiskMatrix],
    ) -> SummaryStatistics:
        """Compute summary statistics across all impacts.

        Aggregates counts by type, severity, likelihood, status,
        value chain position, category, and country.  Computes
        average risk score and key percentages.

        Args:
            impacts: List of AdverseImpact instances.
            risk_matrices: Corresponding RiskMatrix instances.

        Returns:
            SummaryStatistics with all aggregations.
        """
        if not impacts:
            logger.warning("No impacts provided for summary statistics")
            return SummaryStatistics()

        n = len(impacts)

        # Type counts
        hr_count = sum(
            1 for i in impacts
            if i.impact_type == AdverseImpactType.HUMAN_RIGHTS
        )
        env_count = sum(
            1 for i in impacts
            if i.impact_type == AdverseImpactType.ENVIRONMENTAL
        )

        # Status counts
        actual_count = sum(
            1 for i in impacts if i.status == ImpactStatus.ACTUAL
        )
        potential_count = sum(
            1 for i in impacts if i.status == ImpactStatus.POTENTIAL
        )

        # Severity distribution
        by_severity: Dict[str, int] = {}
        for sev in ImpactSeverity:
            by_severity[sev.value] = sum(
                1 for i in impacts if i.severity == sev
            )

        # Likelihood distribution
        by_likelihood: Dict[str, int] = {}
        for lh in ImpactLikelihood:
            by_likelihood[lh.value] = sum(
                1 for i in impacts if i.likelihood == lh
            )

        # Risk level distribution (from matrices)
        rm_by_id: Dict[str, RiskMatrix] = {
            rm.impact_id: rm for rm in risk_matrices
        }
        by_risk_level: Dict[str, int] = {}
        for rl in RiskLevel:
            by_risk_level[rl.value] = sum(
                1 for rm in risk_matrices if rm.risk_level == rl
            )

        # Value chain distribution
        by_vc: Dict[str, int] = {}
        for vcp in ValueChainPosition:
            by_vc[vcp.value] = sum(
                1 for i in impacts if i.value_chain_position == vcp
            )

        # Category distribution
        by_category: Dict[str, int] = {}
        for i in impacts:
            cat = i.category
            by_category[cat] = by_category.get(cat, 0) + 1

        # Country distribution
        by_country: Dict[str, int] = {}
        for i in impacts:
            if i.country:
                by_country[i.country] = by_country.get(i.country, 0) + 1

        # Salient issues
        salient_count = sum(1 for i in impacts if i.is_salient)

        # Irremediable
        irremediable_count = sum(
            1 for i in impacts if i.irremediable_character
        )

        # Total affected stakeholders
        total_affected = sum(
            i.affected_stakeholder_count for i in impacts
        )

        # Average risk score
        if risk_matrices:
            total_risk = sum(rm.risk_score for rm in risk_matrices)
            avg_risk = _round_val(
                _decimal(total_risk) / _decimal(len(risk_matrices)), 1
            )
        else:
            avg_risk = Decimal("0")

        # Critical + High percentage
        crit_high_count = (
            by_risk_level.get(RiskLevel.CRITICAL.value, 0)
            + by_risk_level.get(RiskLevel.HIGH.value, 0)
        )
        crit_high_pct = _pct(crit_high_count, n)

        # Own operations vs value chain
        own_ops_count = by_vc.get(ValueChainPosition.OWN_OPERATIONS.value, 0)
        own_ops_pct = _pct(own_ops_count, n)
        vc_count = n - own_ops_count
        vc_pct = _pct(vc_count, n)

        return SummaryStatistics(
            total_impacts=n,
            human_rights_count=hr_count,
            environmental_count=env_count,
            actual_count=actual_count,
            potential_count=potential_count,
            by_severity=by_severity,
            by_likelihood=by_likelihood,
            by_risk_level=by_risk_level,
            by_value_chain_position=by_vc,
            by_category=by_category,
            by_country=by_country,
            salient_issues_count=salient_count,
            irremediable_count=irremediable_count,
            total_affected_stakeholders=total_affected,
            average_risk_score=avg_risk,
            critical_and_high_pct=crit_high_pct,
            own_operations_pct=own_ops_pct,
            value_chain_pct=vc_pct,
        )

    # ------------------------------------------------------------------ #
    # Top Risks                                                            #
    # ------------------------------------------------------------------ #

    def _get_top_risks(
        self,
        impacts: List[AdverseImpact],
        risk_matrices: List[RiskMatrix],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get the top N highest-priority impacts with details.

        Returns a list of dictionaries containing impact details
        and risk matrix data for the highest-priority impacts.

        Args:
            impacts: All AdverseImpact instances.
            risk_matrices: All RiskMatrix instances (prioritised).
            top_n: Number of top risks to return (default 10).

        Returns:
            List of dicts with impact and risk details.
        """
        impact_by_id: Dict[str, AdverseImpact] = {
            i.impact_id: i for i in impacts
        }

        top_risks: List[Dict[str, Any]] = []
        for rm in risk_matrices[:top_n]:
            impact = impact_by_id.get(rm.impact_id)
            if impact is None:
                continue

            # Determine category label
            if impact.impact_type == AdverseImpactType.HUMAN_RIGHTS:
                label = HR_CATEGORY_LABELS.get(
                    impact.category, impact.category
                )
            else:
                label = ENV_CATEGORY_LABELS.get(
                    impact.category, impact.category
                )

            top_risks.append({
                "rank": rm.priority_rank,
                "impact_id": impact.impact_id,
                "impact_type": impact.impact_type.value,
                "category": impact.category,
                "category_label": label,
                "description": impact.description[:500],
                "severity": impact.severity.value,
                "likelihood": impact.likelihood.value,
                "status": impact.status.value,
                "value_chain_position": impact.value_chain_position.value,
                "risk_score": rm.risk_score,
                "risk_level": rm.risk_level.value,
                "country": impact.country,
                "affected_stakeholders": impact.affected_stakeholder_count,
                "is_salient": impact.is_salient,
                "irremediable": impact.irremediable_character,
            })

        return top_risks

    # ------------------------------------------------------------------ #
    # Category Risk Summary                                                #
    # ------------------------------------------------------------------ #

    def _get_category_risk_summary(
        self,
        impacts: List[AdverseImpact],
        risk_matrices: List[RiskMatrix],
    ) -> List[Dict[str, Any]]:
        """Aggregate risk by impact category.

        Groups impacts by category and computes average risk score,
        maximum risk level, and count per category.

        Args:
            impacts: All AdverseImpact instances.
            risk_matrices: All RiskMatrix instances.

        Returns:
            List of dicts with category-level risk summaries.
        """
        rm_by_id: Dict[str, RiskMatrix] = {
            rm.impact_id: rm for rm in risk_matrices
        }

        # Group by category
        cat_data: Dict[str, List[int]] = {}
        cat_type: Dict[str, str] = {}
        for impact in impacts:
            cat = impact.category
            rm = rm_by_id.get(impact.impact_id)
            if rm is None:
                continue
            if cat not in cat_data:
                cat_data[cat] = []
                cat_type[cat] = impact.impact_type.value
            cat_data[cat].append(rm.risk_score)

        summaries: List[Dict[str, Any]] = []
        for cat, scores in cat_data.items():
            count = len(scores)
            total_score = sum(scores)
            max_score = max(scores) if scores else 0
            avg_score = _round_val(
                _decimal(total_score) / _decimal(count), 1
            )
            max_risk_level = self._score_to_risk_level(max_score)

            # Determine label
            itype = cat_type.get(cat, "")
            if itype == AdverseImpactType.HUMAN_RIGHTS.value:
                label = HR_CATEGORY_LABELS.get(cat, cat)
            else:
                label = ENV_CATEGORY_LABELS.get(cat, cat)

            summaries.append({
                "category": cat,
                "category_label": label,
                "impact_type": itype,
                "count": count,
                "average_risk_score": avg_score,
                "max_risk_score": max_score,
                "max_risk_level": max_risk_level.value,
            })

        # Sort by max_risk_score descending
        summaries.sort(key=lambda s: -s["max_risk_score"])

        return summaries

    # ------------------------------------------------------------------ #
    # Value Chain Risk Summary                                             #
    # ------------------------------------------------------------------ #

    def _get_value_chain_risk_summary(
        self,
        impacts: List[AdverseImpact],
        risk_matrices: List[RiskMatrix],
    ) -> List[Dict[str, Any]]:
        """Aggregate risk by value chain position.

        Groups impacts by value chain position and computes average
        risk score, count, and distribution of risk levels.

        Args:
            impacts: All AdverseImpact instances.
            risk_matrices: All RiskMatrix instances.

        Returns:
            List of dicts with value-chain-level risk summaries.
        """
        rm_by_id: Dict[str, RiskMatrix] = {
            rm.impact_id: rm for rm in risk_matrices
        }

        # Group by value chain position
        vc_data: Dict[str, List[int]] = {}
        vc_risk_levels: Dict[str, Dict[str, int]] = {}
        for impact in impacts:
            vcp = impact.value_chain_position.value
            rm = rm_by_id.get(impact.impact_id)
            if rm is None:
                continue
            if vcp not in vc_data:
                vc_data[vcp] = []
                vc_risk_levels[vcp] = {rl.value: 0 for rl in RiskLevel}
            vc_data[vcp].append(rm.risk_score)
            vc_risk_levels[vcp][rm.risk_level.value] += 1

        summaries: List[Dict[str, Any]] = []
        total_impacts = len(impacts)

        for vcp in ValueChainPosition:
            scores = vc_data.get(vcp.value, [])
            count = len(scores)
            if count == 0:
                continue

            total_score = sum(scores)
            avg_score = _round_val(
                _decimal(total_score) / _decimal(count), 1
            )
            max_score = max(scores)
            max_risk_level = self._score_to_risk_level(max_score)
            pct_of_total = _pct(count, total_impacts)

            summaries.append({
                "value_chain_position": vcp.value,
                "count": count,
                "pct_of_total": pct_of_total,
                "average_risk_score": avg_score,
                "max_risk_score": max_score,
                "max_risk_level": max_risk_level.value,
                "risk_level_distribution": vc_risk_levels.get(
                    vcp.value, {}
                ),
            })

        return summaries

    # ------------------------------------------------------------------ #
    # Main Assessment Entry Point                                          #
    # ------------------------------------------------------------------ #

    def assess_impacts(
        self, impacts: List[AdverseImpact]
    ) -> ImpactAssessmentResult:
        """Run a complete adverse impact assessment.

        Calculates risk matrices for all impacts, prioritises them
        by risk score, and computes comprehensive summary statistics.

        This is the primary entry point for the engine.

        Args:
            impacts: List of AdverseImpact instances to assess.

        Returns:
            ImpactAssessmentResult with risk matrices, priority
            ordering, summary statistics, and provenance hash.
        """
        start_time = time.time()
        logger.info(
            "Starting adverse impact assessment for %d impacts",
            len(impacts),
        )

        if not impacts:
            logger.warning("No impacts provided for assessment")
            empty_result = ImpactAssessmentResult(
                impacts=[],
                risk_matrices=[],
                priority_impacts=[],
                summary_stats=SummaryStatistics(),
                top_risks=[],
                category_risk_summary=[],
                value_chain_risk_summary=[],
                processing_time_ms=0.0,
                assessed_at=_utcnow(),
            )
            empty_result.provenance_hash = _compute_hash(empty_result)
            return empty_result

        # Step 1: Calculate risk matrices for each impact
        risk_matrices: List[RiskMatrix] = []
        for impact in impacts:
            rm = self.calculate_risk_matrix(impact)
            risk_matrices.append(rm)

        # Step 2: Prioritise impacts
        priority_ids, prioritised_matrices = self.prioritize_impacts(
            impacts, risk_matrices
        )

        # Step 3: Compute summary statistics
        summary_stats = self.get_summary_statistics(
            impacts, prioritised_matrices
        )

        # Step 4: Get top risks
        top_risks = self._get_top_risks(
            impacts, prioritised_matrices, top_n=10
        )

        # Step 5: Category risk summary
        category_summary = self._get_category_risk_summary(
            impacts, prioritised_matrices
        )

        # Step 6: Value chain risk summary
        vc_summary = self._get_value_chain_risk_summary(
            impacts, prioritised_matrices
        )

        processing_time_ms = (time.time() - start_time) * 1000

        # Step 7: Build result
        result = ImpactAssessmentResult(
            impacts=impacts,
            risk_matrices=prioritised_matrices,
            priority_impacts=priority_ids,
            summary_stats=summary_stats,
            top_risks=top_risks,
            category_risk_summary=category_summary,
            value_chain_risk_summary=vc_summary,
            processing_time_ms=_round2(processing_time_ms),
            assessed_at=_utcnow(),
        )

        # Step 8: Compute provenance hash
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Adverse impact assessment complete: %d impacts, "
            "avg_risk=%.1f, critical+high=%.1f%%, time=%.2fms",
            len(impacts),
            float(summary_stats.average_risk_score),
            float(summary_stats.critical_and_high_pct),
            processing_time_ms,
        )

        return result

    # ------------------------------------------------------------------ #
    # Utility: Assess Single Impact                                        #
    # ------------------------------------------------------------------ #

    def assess_single_impact(
        self, impact: AdverseImpact
    ) -> Dict[str, Any]:
        """Assess and score a single adverse impact.

        Convenience method for ad-hoc assessment of a single impact
        without needing to call the full assess_impacts pipeline.

        Args:
            impact: AdverseImpact to assess.

        Returns:
            Dict with risk matrix data and category information.
        """
        rm = self.calculate_risk_matrix(impact)
        rm.priority_rank = 1

        if impact.impact_type == AdverseImpactType.HUMAN_RIGHTS:
            label = HR_CATEGORY_LABELS.get(
                impact.category, impact.category
            )
        else:
            label = ENV_CATEGORY_LABELS.get(
                impact.category, impact.category
            )

        result = {
            "impact_id": impact.impact_id,
            "impact_type": impact.impact_type.value,
            "category": impact.category,
            "category_label": label,
            "severity": impact.severity.value,
            "likelihood": impact.likelihood.value,
            "status": impact.status.value,
            "value_chain_position": impact.value_chain_position.value,
            "severity_score": rm.severity_score,
            "likelihood_score": rm.likelihood_score,
            "risk_score": rm.risk_score,
            "risk_level": rm.risk_level.value,
            "is_salient": impact.is_salient,
            "irremediable": impact.irremediable_character,
            "provenance_hash": _compute_hash(rm),
        }

        logger.info(
            "Single impact assessed: %s severity=%s likelihood=%s "
            "risk_score=%d risk_level=%s",
            impact.impact_id, impact.severity.value,
            impact.likelihood.value, rm.risk_score, rm.risk_level.value,
        )

        return result

    # ------------------------------------------------------------------ #
    # Utility: Filter Impacts                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def filter_impacts_by_risk_level(
        impacts: List[AdverseImpact],
        risk_matrices: List[RiskMatrix],
        min_risk_level: RiskLevel,
    ) -> List[AdverseImpact]:
        """Filter impacts to those at or above a minimum risk level.

        Args:
            impacts: All AdverseImpact instances.
            risk_matrices: Corresponding RiskMatrix instances.
            min_risk_level: Minimum risk level to include.

        Returns:
            Filtered list of AdverseImpact instances.
        """
        level_order = {
            RiskLevel.CRITICAL: 5,
            RiskLevel.HIGH: 4,
            RiskLevel.MEDIUM: 3,
            RiskLevel.LOW: 2,
            RiskLevel.VERY_LOW: 1,
        }
        min_order = level_order.get(min_risk_level, 1)

        rm_by_id: Dict[str, RiskMatrix] = {
            rm.impact_id: rm for rm in risk_matrices
        }

        filtered: List[AdverseImpact] = []
        for impact in impacts:
            rm = rm_by_id.get(impact.impact_id)
            if rm is None:
                continue
            order = level_order.get(rm.risk_level, 1)
            if order >= min_order:
                filtered.append(impact)

        return filtered

    @staticmethod
    def filter_impacts_by_type(
        impacts: List[AdverseImpact],
        impact_type: AdverseImpactType,
    ) -> List[AdverseImpact]:
        """Filter impacts by type (human rights or environmental).

        Args:
            impacts: All AdverseImpact instances.
            impact_type: The type to filter for.

        Returns:
            Filtered list of AdverseImpact instances.
        """
        return [i for i in impacts if i.impact_type == impact_type]

    @staticmethod
    def filter_impacts_by_value_chain(
        impacts: List[AdverseImpact],
        position: ValueChainPosition,
    ) -> List[AdverseImpact]:
        """Filter impacts by value chain position.

        Args:
            impacts: All AdverseImpact instances.
            position: The value chain position to filter for.

        Returns:
            Filtered list of AdverseImpact instances.
        """
        return [i for i in impacts if i.value_chain_position == position]
