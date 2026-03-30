# -*- coding: utf-8 -*-
"""
SustainableObjectiveEngine - PACK-011 SFDR Article 9 Engine 1
================================================================

Verify ALL investments qualify as sustainable per SFDR Article 2(17).

Article 9 products have sustainable investment as their objective, meaning
the portfolio must consist (near) entirely of sustainable investments.
Each holding must pass the three-part test:

    1. **Contribution** -- contributes to an environmental objective (measured
       by key resource efficiency indicators) OR a social objective (tackling
       inequality, fostering social cohesion, investing in human capital).
    2. **DNSH** -- does not significantly harm any of the environmental or
       social objectives.
    3. **Good governance** -- the investee company follows good governance
       practices (sound management, employee relations, remuneration, tax).

The engine classifies every holding, computes the portfolio-level sustainable
proportion (which must approach 100 % for a genuine Article 9 product), and
generates a compliance report with per-objective breakdowns.

Key Regulatory References:
    - Regulation (EU) 2019/2088 (SFDR) Article 2(17), Article 9
    - Delegated Regulation (EU) 2022/1288 (SFDR RTS) Articles 15-19
    - Regulation (EU) 2020/852 (Taxonomy Regulation) Article 9

Zero-Hallucination:
    - All classification steps use deterministic rule evaluation
    - Proportion calculations are pure arithmetic on NAV figures
    - No LLM involvement in classification or calculation paths
    - SHA-256 provenance hash on every result

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely, returning 0.0 on zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        Percentage value or 0.0.
    """
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0

def _round_val(value: float, places: int = 4) -> float:
    """Round a float to specified decimal places."""
    return round(value, places)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ObjectiveType(str, Enum):
    """Top-level objective classification."""
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    NONE = "none"

class EnvironmentalObjective(str, Enum):
    """EU Taxonomy six environmental objectives plus OTHER for non-Taxonomy."""
    CLIMATE_MITIGATION = "climate_mitigation"
    CLIMATE_ADAPTATION = "climate_adaptation"
    WATER_MARINE = "water_marine"
    CIRCULAR_ECONOMY = "circular_economy"
    POLLUTION_PREVENTION = "pollution_prevention"
    BIODIVERSITY = "biodiversity"
    OTHER = "other"

class SocialObjective(str, Enum):
    """Social objectives per SFDR Article 2(17)."""
    INEQUALITY = "inequality"
    SOCIAL_COHESION = "social_cohesion"
    HUMAN_CAPITAL = "human_capital"
    COMMUNITY_DEVELOPMENT = "community_development"
    HEALTH_WELLBEING = "health_wellbeing"
    AFFORDABLE_HOUSING = "affordable_housing"

class ComplianceStatus(str, Enum):
    """Overall compliance status for the portfolio."""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    MARGINAL = "MARGINAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

class HoldingClassificationType(str, Enum):
    """Classification of a holding under Article 2(17) for Article 9."""
    TAXONOMY_ALIGNED = "taxonomy_aligned"
    OTHER_ENVIRONMENTAL = "other_environmental"
    SOCIAL = "social"
    NOT_SUSTAINABLE = "not_sustainable"
    CASH_EQUIVALENT = "cash_equivalent"
    HEDGING = "hedging"

# ---------------------------------------------------------------------------
# DNSH PAI Mapping & Governance Criteria
# ---------------------------------------------------------------------------

ENVIRONMENTAL_DNSH_INDICATORS: Dict[str, List[str]] = {
    EnvironmentalObjective.CLIMATE_MITIGATION.value: [
        "ghg_emissions", "carbon_footprint", "ghg_intensity",
        "fossil_fuel_exposure", "non_renewable_energy",
    ],
    EnvironmentalObjective.CLIMATE_ADAPTATION.value: [
        "ghg_emissions", "carbon_footprint", "biodiversity_impact",
    ],
    EnvironmentalObjective.WATER_MARINE.value: [
        "water_emissions", "water_recycling", "hazardous_waste",
    ],
    EnvironmentalObjective.CIRCULAR_ECONOMY.value: [
        "hazardous_waste", "waste_recycling",
    ],
    EnvironmentalObjective.POLLUTION_PREVENTION.value: [
        "water_emissions", "hazardous_waste",
    ],
    EnvironmentalObjective.BIODIVERSITY.value: [
        "biodiversity_impact", "deforestation",
    ],
    EnvironmentalObjective.OTHER.value: [
        "ghg_emissions", "carbon_footprint",
    ],
}

SOCIAL_DNSH_INDICATORS: Dict[str, List[str]] = {
    SocialObjective.INEQUALITY.value: [
        "gender_pay_gap", "board_gender_diversity", "human_rights_violations",
    ],
    SocialObjective.SOCIAL_COHESION.value: [
        "human_rights_violations", "controversies",
    ],
    SocialObjective.HUMAN_CAPITAL.value: [
        "gender_pay_gap", "board_gender_diversity",
    ],
    SocialObjective.COMMUNITY_DEVELOPMENT.value: [
        "human_rights_violations", "controversies",
    ],
    SocialObjective.HEALTH_WELLBEING.value: [
        "human_rights_violations",
    ],
    SocialObjective.AFFORDABLE_HOUSING.value: [
        "human_rights_violations",
    ],
}

GOVERNANCE_CRITERIA: List[str] = [
    "sound_management_structures",
    "employee_relations",
    "remuneration_compliance",
    "tax_compliance",
]

# ---------------------------------------------------------------------------
# Pydantic Data Models
# ---------------------------------------------------------------------------

class HoldingData(BaseModel):
    """Input data for a single portfolio holding.

    Represents all data needed to classify a holding under Article 2(17)
    for an Article 9 product.

    Attributes:
        holding_id: Unique holding identifier (ISIN, internal ID).
        holding_name: Name of the holding / investee company.
        nav_value: Net Asset Value of this holding in EUR.
        weight_pct: Portfolio weight as percentage.
        sector: Sector classification (NACE/GICS).
        country: Country of domicile (ISO 3166-1 alpha-2).
        is_cash_equivalent: Whether this is a cash or cash-equivalent position.
        is_hedging: Whether this is a hedging or derivatives position.
        taxonomy_eligible: Whether the activity is EU Taxonomy eligible.
        taxonomy_aligned_pct: Percentage of revenue Taxonomy-aligned.
        environmental_objective: Primary environmental objective contribution.
        social_objective: Primary social objective contribution.
        pai_data: PAI indicator numeric values.
        pai_boolean_flags: Boolean PAI indicator flags.
        governance_data: Governance criteria assessment flags.
        dnsh_passed: Pre-assessed DNSH status (if available externally).
        good_governance_passed: Pre-assessed governance status.
        contribution_evidence: Evidence supporting the contribution claim.
    """
    holding_id: str = Field(
        default_factory=_new_uuid,
        description="Unique holding identifier",
    )
    holding_name: str = Field(
        default="", description="Holding / investee company name",
    )
    nav_value: float = Field(
        default=0.0, ge=0.0,
        description="Net Asset Value of this holding in EUR",
    )
    weight_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Portfolio weight as percentage",
    )
    sector: str = Field(default="", description="Sector classification")
    country: str = Field(default="", description="Country of domicile (ISO 3166)")
    is_cash_equivalent: bool = Field(
        default=False,
        description="Whether this is a cash or cash-equivalent position",
    )
    is_hedging: bool = Field(
        default=False,
        description="Whether this is a hedging or derivatives position",
    )
    taxonomy_eligible: bool = Field(
        default=False,
        description="EU Taxonomy eligible activity",
    )
    taxonomy_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of revenue Taxonomy-aligned",
    )
    environmental_objective: Optional[EnvironmentalObjective] = Field(
        default=None,
        description="Primary environmental objective contribution",
    )
    social_objective: Optional[SocialObjective] = Field(
        default=None,
        description="Primary social objective contribution",
    )
    pai_data: Dict[str, float] = Field(
        default_factory=dict,
        description="PAI indicator numeric values",
    )
    pai_boolean_flags: Dict[str, bool] = Field(
        default_factory=dict,
        description="Boolean PAI indicator flags",
    )
    governance_data: Dict[str, bool] = Field(
        default_factory=dict,
        description="Governance criteria assessment flags",
    )
    dnsh_passed: Optional[bool] = Field(
        default=None,
        description="Pre-assessed DNSH status (if available)",
    )
    good_governance_passed: Optional[bool] = Field(
        default=None,
        description="Pre-assessed governance status",
    )
    contribution_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence supporting the contribution claim",
    )

class HoldingClassification(BaseModel):
    """Classification result for a single holding under Article 2(17).

    Contains the final classification, the three-part test results,
    confidence level, and evidence trail.
    """
    classification_id: str = Field(
        default_factory=_new_uuid,
        description="Unique classification identifier",
    )
    holding_id: str = Field(description="Classified holding identifier")
    holding_name: str = Field(default="", description="Holding name")
    classification: HoldingClassificationType = Field(
        description="Final classification per Article 2(17)",
    )
    objective_type: ObjectiveType = Field(
        default=ObjectiveType.NONE,
        description="Top-level objective type (environmental / social / none)",
    )
    environmental_objective: Optional[EnvironmentalObjective] = Field(
        default=None, description="Environmental objective if applicable",
    )
    social_objective: Optional[SocialObjective] = Field(
        default=None, description="Social objective if applicable",
    )
    contribution_passed: bool = Field(
        default=False,
        description="Whether the holding passes the contribution test",
    )
    dnsh_passed: bool = Field(
        default=False,
        description="Whether the holding passes the DNSH test",
    )
    good_governance_passed: bool = Field(
        default=False,
        description="Whether the holding passes the good governance test",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Classification confidence score (0.0 to 1.0)",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Evidence supporting the classification",
    )
    nav_value: float = Field(
        default=0.0,
        description="NAV value for proportion calculation",
    )
    weight_pct: float = Field(
        default=0.0,
        description="Portfolio weight for proportion calculation",
    )
    classified_at: datetime = Field(
        default_factory=utcnow,
        description="Classification timestamp",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

class ObjectiveBreakdownEntry(BaseModel):
    """Breakdown of sustainable holdings by a specific objective.

    Provides aggregate metrics for holdings contributing to a single
    environmental or social objective.
    """
    objective_type: ObjectiveType = Field(
        description="Top-level objective type",
    )
    objective_name: str = Field(
        description="Human-readable objective name",
    )
    objective_value: str = Field(
        description="Objective enum value for programmatic use",
    )
    holding_count: int = Field(
        default=0, ge=0,
        description="Number of holdings contributing to this objective",
    )
    nav_value: float = Field(
        default=0.0, ge=0.0,
        description="Total NAV for this objective in EUR",
    )
    proportion_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Proportion of total portfolio NAV (%)",
    )
    average_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average classification confidence",
    )

class NonSustainableBreakdown(BaseModel):
    """Breakdown of non-sustainable holdings in the portfolio.

    For Article 9 products this should be minimal (only cash, hedging,
    or transitional residual positions).
    """
    total_non_sustainable_nav: float = Field(
        default=0.0, ge=0.0,
        description="Total NAV of non-sustainable holdings in EUR",
    )
    total_non_sustainable_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Non-sustainable holdings as % of NAV",
    )
    cash_equivalent_nav: float = Field(
        default=0.0, ge=0.0,
        description="Cash and cash-equivalent NAV",
    )
    cash_equivalent_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Cash and cash-equivalent as % of NAV",
    )
    hedging_nav: float = Field(
        default=0.0, ge=0.0,
        description="Hedging / derivatives NAV",
    )
    hedging_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Hedging / derivatives as % of NAV",
    )
    other_non_sustainable_nav: float = Field(
        default=0.0, ge=0.0,
        description="Other non-sustainable NAV (should be near zero)",
    )
    other_non_sustainable_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Other non-sustainable as % of NAV",
    )
    holdings: List[str] = Field(
        default_factory=list,
        description="List of non-sustainable holding IDs",
    )

class CommitmentStatus(BaseModel):
    """Status of the Article 9 sustainable investment commitment.

    Article 9 products commit to near-100 % sustainable investment.
    This tracks adherence to that commitment.
    """
    commitment_id: str = Field(
        default_factory=_new_uuid,
        description="Unique commitment status identifier",
    )
    minimum_sustainable_pct: float = Field(
        default=90.0, ge=0.0, le=100.0,
        description="Minimum sustainable investment % for Article 9 compliance",
    )
    actual_sustainable_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Actual sustainable investment %",
    )
    taxonomy_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Taxonomy-aligned investment %",
    )
    other_environmental_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Other environmental sustainable investment %",
    )
    social_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Social sustainable investment %",
    )
    gap_pct: float = Field(
        default=0.0,
        description="Gap between minimum and actual (negative = shortfall)",
    )
    meets_commitment: bool = Field(
        default=False,
        description="Whether actual >= minimum",
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.INSUFFICIENT_DATA,
        description="Overall compliance status",
    )
    checked_at: datetime = Field(
        default_factory=utcnow,
        description="Check timestamp",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

class ComplianceReport(BaseModel):
    """Full Article 9 compliance report.

    Consolidates all classification results, breakdowns, and compliance
    assessments into a single disclosure-ready report.
    """
    report_id: str = Field(
        default_factory=_new_uuid,
        description="Unique report identifier",
    )
    product_name: str = Field(
        default="",
        description="Financial product name",
    )
    reporting_date: datetime = Field(
        default_factory=utcnow,
        description="Reporting reference date",
    )
    total_holdings: int = Field(
        default=0, ge=0,
        description="Total number of holdings assessed",
    )
    total_nav: float = Field(
        default=0.0, ge=0.0,
        description="Total portfolio NAV in EUR",
    )
    sustainable_holdings: int = Field(
        default=0, ge=0,
        description="Number of holdings classified as sustainable",
    )
    sustainable_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Sustainable holdings as % of NAV",
    )
    taxonomy_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Taxonomy-aligned as % of NAV",
    )
    other_environmental_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Other environmental as % of NAV",
    )
    social_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Social as % of NAV",
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.INSUFFICIENT_DATA,
        description="Overall Article 9 compliance status",
    )
    objective_breakdown: List[ObjectiveBreakdownEntry] = Field(
        default_factory=list,
        description="Breakdown by sustainability objective",
    )
    non_sustainable_breakdown: Optional[NonSustainableBreakdown] = Field(
        default=None,
        description="Breakdown of non-sustainable positions",
    )
    commitment_status: Optional[CommitmentStatus] = Field(
        default=None,
        description="Commitment adherence status",
    )
    classifications: List[HoldingClassification] = Field(
        default_factory=list,
        description="Individual holding classification results",
    )
    contribution_pass_rate: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of holdings passing contribution test",
    )
    dnsh_pass_rate: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of holdings passing DNSH test",
    )
    governance_pass_rate: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of holdings passing governance test",
    )
    average_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average classification confidence across holdings",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version",
    )
    generated_at: datetime = Field(
        default_factory=utcnow,
        description="Report generation timestamp",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

class SustainableObjectiveResult(BaseModel):
    """Result of the sustainable objective assessment for the full portfolio.

    Contains proportion calculations, all classifications, and the
    overall compliance determination.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    total_nav: float = Field(
        default=0.0, ge=0.0,
        description="Total portfolio NAV in EUR",
    )
    sustainable_nav: float = Field(
        default=0.0, ge=0.0,
        description="Sustainable portion NAV in EUR",
    )
    sustainable_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Sustainable proportion as % of NAV",
    )
    taxonomy_aligned_nav: float = Field(
        default=0.0, ge=0.0,
        description="Taxonomy-aligned NAV in EUR",
    )
    taxonomy_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Taxonomy-aligned as % of NAV",
    )
    other_environmental_nav: float = Field(
        default=0.0, ge=0.0,
        description="Other environmental NAV in EUR",
    )
    other_environmental_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Other environmental as % of NAV",
    )
    social_nav: float = Field(
        default=0.0, ge=0.0,
        description="Social NAV in EUR",
    )
    social_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Social as % of NAV",
    )
    not_sustainable_nav: float = Field(
        default=0.0, ge=0.0,
        description="Non-sustainable NAV in EUR",
    )
    not_sustainable_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Non-sustainable as % of NAV",
    )
    total_holdings: int = Field(
        default=0, ge=0,
        description="Total holdings assessed",
    )
    sustainable_holdings: int = Field(
        default=0, ge=0,
        description="Number of sustainable holdings",
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.INSUFFICIENT_DATA,
        description="Overall compliance determination",
    )
    classifications: List[HoldingClassification] = Field(
        default_factory=list,
        description="All holding classifications",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Calculation timestamp",
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash",
    )

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class SustainableObjectiveConfig(BaseModel):
    """Configuration for the SustainableObjectiveEngine.

    Controls the thresholds, minimum proportions, and assessment
    parameters for Article 9 sustainable investment classification.

    Attributes:
        minimum_sustainable_pct: Minimum sustainable proportion for compliance.
        marginal_threshold_pct: Threshold for MARGINAL compliance status.
        dnsh_coverage_threshold: Minimum PAI coverage % to pass DNSH.
        governance_min_criteria: Minimum governance criteria to pass (out of 4).
        taxonomy_alignment_threshold: Min alignment % for TAXONOMY_ALIGNED.
        max_non_sustainable_pct: Maximum allowed non-sustainable proportion.
        allow_cash_hedging_exemption: Whether cash/hedging are exempt.
        cash_hedging_max_pct: Maximum allowed cash/hedging proportion.
        pai_thresholds: PAI indicator thresholds for DNSH.
        higher_is_better_indicators: PAI indicators where higher values pass.
    """
    minimum_sustainable_pct: float = Field(
        default=90.0, ge=0.0, le=100.0,
        description="Minimum sustainable investment % for Article 9 compliance",
    )
    marginal_threshold_pct: float = Field(
        default=85.0, ge=0.0, le=100.0,
        description="Threshold below which compliance is MARGINAL (not outright fail)",
    )
    dnsh_coverage_threshold: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="Minimum PAI coverage % to pass DNSH",
    )
    governance_min_criteria: int = Field(
        default=3, ge=1, le=4,
        description="Minimum governance criteria to pass (out of 4)",
    )
    taxonomy_alignment_threshold: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Minimum Taxonomy alignment % for TAXONOMY_ALIGNED class",
    )
    max_non_sustainable_pct: float = Field(
        default=10.0, ge=0.0, le=100.0,
        description="Maximum allowed non-sustainable proportion for Article 9",
    )
    allow_cash_hedging_exemption: bool = Field(
        default=True,
        description="Whether cash/hedging positions are exempt from sustainability test",
    )
    cash_hedging_max_pct: float = Field(
        default=10.0, ge=0.0, le=100.0,
        description="Maximum allowed cash/hedging proportion",
    )
    pai_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "ghg_emissions": 800000.0,
            "carbon_footprint": 400.0,
            "ghg_intensity": 600.0,
            "fossil_fuel_exposure": 5.0,
            "non_renewable_energy": 40.0,
            "water_emissions": 3.0,
            "hazardous_waste": 8.0,
            "biodiversity_impact": 0.0,
            "gender_pay_gap": 12.0,
            "board_gender_diversity": 33.0,
            "human_rights_violations": 0.0,
            "controversies": 0.0,
            "deforestation": 0.0,
            "water_recycling": 55.0,
            "waste_recycling": 45.0,
        },
        description="PAI indicator thresholds for DNSH (stricter than Article 8)",
    )
    higher_is_better_indicators: List[str] = Field(
        default_factory=lambda: [
            "board_gender_diversity",
            "water_recycling",
            "waste_recycling",
        ],
        description="PAI indicators where higher values are better",
    )

# ---------------------------------------------------------------------------
# Pydantic model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

SustainableObjectiveConfig.model_rebuild()
HoldingData.model_rebuild()
HoldingClassification.model_rebuild()
ObjectiveBreakdownEntry.model_rebuild()
NonSustainableBreakdown.model_rebuild()
CommitmentStatus.model_rebuild()
ComplianceReport.model_rebuild()
SustainableObjectiveResult.model_rebuild()

# ---------------------------------------------------------------------------
# SustainableObjectiveEngine
# ---------------------------------------------------------------------------

class SustainableObjectiveEngine:
    """
    Sustainable objective verification engine for SFDR Article 9 products.

    Implements the three-part test per SFDR Article 2(17) for EVERY holding
    in the portfolio, then computes portfolio-level proportions and
    compliance status.  Article 9 products must have near-100 % sustainable
    investment, so the engine enforces strict thresholds (configurable but
    defaulting to 90 % minimum).

    Zero-Hallucination Guarantees:
        - All classification steps use deterministic rule evaluation
        - Proportion calculations are pure arithmetic on NAV figures
        - No LLM involvement in classification or calculation paths
        - SHA-256 provenance hash on every result

    Attributes:
        config: Engine configuration.
        _holdings: Input holding data.
        _classifications: Computed classifications keyed by holding_id.
        _classification_count: Running count of classifications performed.

    Example:
        >>> config = SustainableObjectiveConfig(minimum_sustainable_pct=90.0)
        >>> engine = SustainableObjectiveEngine(config)
        >>> holdings = [HoldingData(holding_id="H1", nav_value=1e6, ...)]
        >>> result = engine.classify_portfolio(holdings)
        >>> assert result.compliance_status == ComplianceStatus.COMPLIANT
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SustainableObjectiveEngine.

        Args:
            config: Optional configuration dictionary or SustainableObjectiveConfig.
        """
        if config and isinstance(config, dict):
            self.config = SustainableObjectiveConfig(**config)
        elif config and isinstance(config, SustainableObjectiveConfig):
            self.config = config
        else:
            self.config = SustainableObjectiveConfig()

        self._holdings: List[HoldingData] = []
        self._classifications: Dict[str, HoldingClassification] = {}
        self._classification_count: int = 0

        logger.info(
            "SustainableObjectiveEngine initialized (version=%s, "
            "min_sustainable=%.1f%%)",
            _MODULE_VERSION,
            self.config.minimum_sustainable_pct,
        )

    # ------------------------------------------------------------------
    # Public API: Portfolio Classification
    # ------------------------------------------------------------------

    def classify_portfolio(
        self,
        holdings: List[HoldingData],
    ) -> SustainableObjectiveResult:
        """Classify all holdings and compute portfolio-level proportions.

        Applies the three-part Article 2(17) test to every holding,
        then aggregates results into proportions and determines overall
        compliance status.

        Args:
            holdings: List of HoldingData to classify.

        Returns:
            SustainableObjectiveResult with proportions and compliance.

        Raises:
            ValueError: If holdings list is empty.
        """
        start = utcnow()

        if not holdings:
            raise ValueError("Holdings list cannot be empty")

        self._holdings = holdings
        self._classifications = {}

        logger.info(
            "Classifying %d holdings for Article 9 compliance",
            len(holdings),
        )

        # Step 1: Classify each holding
        classifications: List[HoldingClassification] = []
        for holding in holdings:
            classification = self._classify_single(holding)
            self._classifications[holding.holding_id] = classification
            classifications.append(classification)
            self._classification_count += 1

        # Step 2: Compute proportions
        total_nav = sum(h.nav_value for h in holdings)
        taxonomy_nav = sum(
            c.nav_value for c in classifications
            if c.classification == HoldingClassificationType.TAXONOMY_ALIGNED
        )
        other_env_nav = sum(
            c.nav_value for c in classifications
            if c.classification == HoldingClassificationType.OTHER_ENVIRONMENTAL
        )
        social_nav = sum(
            c.nav_value for c in classifications
            if c.classification == HoldingClassificationType.SOCIAL
        )
        not_sustainable_nav = sum(
            c.nav_value for c in classifications
            if c.classification == HoldingClassificationType.NOT_SUSTAINABLE
        )
        cash_nav = sum(
            c.nav_value for c in classifications
            if c.classification == HoldingClassificationType.CASH_EQUIVALENT
        )
        hedging_nav = sum(
            c.nav_value for c in classifications
            if c.classification == HoldingClassificationType.HEDGING
        )

        sustainable_nav = taxonomy_nav + other_env_nav + social_nav
        sustainable_count = sum(
            1 for c in classifications
            if c.classification in (
                HoldingClassificationType.TAXONOMY_ALIGNED,
                HoldingClassificationType.OTHER_ENVIRONMENTAL,
                HoldingClassificationType.SOCIAL,
            )
        )

        # Step 3: Determine compliance status
        sustainable_pct = _safe_pct(sustainable_nav, total_nav)
        compliance_status = self._determine_compliance(
            sustainable_pct, cash_nav + hedging_nav, total_nav,
        )

        elapsed_ms = (utcnow() - start).total_seconds() * 1000

        result = SustainableObjectiveResult(
            total_nav=_round_val(total_nav, 2),
            sustainable_nav=_round_val(sustainable_nav, 2),
            sustainable_pct=_round_val(sustainable_pct, 2),
            taxonomy_aligned_nav=_round_val(taxonomy_nav, 2),
            taxonomy_aligned_pct=_round_val(_safe_pct(taxonomy_nav, total_nav), 2),
            other_environmental_nav=_round_val(other_env_nav, 2),
            other_environmental_pct=_round_val(_safe_pct(other_env_nav, total_nav), 2),
            social_nav=_round_val(social_nav, 2),
            social_pct=_round_val(_safe_pct(social_nav, total_nav), 2),
            not_sustainable_nav=_round_val(
                not_sustainable_nav + cash_nav + hedging_nav, 2,
            ),
            not_sustainable_pct=_round_val(
                _safe_pct(not_sustainable_nav + cash_nav + hedging_nav, total_nav), 2,
            ),
            total_holdings=len(classifications),
            sustainable_holdings=sustainable_count,
            compliance_status=compliance_status,
            classifications=classifications,
            processing_time_ms=round(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Portfolio classification complete: %.1f%% sustainable "
            "(%d/%d holdings), status=%s, time=%.1fms",
            result.sustainable_pct,
            sustainable_count,
            len(classifications),
            compliance_status.value,
            elapsed_ms,
        )
        return result

    def classify_single_holding(
        self,
        holding: HoldingData,
    ) -> HoldingClassification:
        """Classify a single holding under Article 2(17).

        Public method for classifying individual holdings outside of
        portfolio context.

        Args:
            holding: HoldingData to classify.

        Returns:
            HoldingClassification result.
        """
        classification = self._classify_single(holding)
        self._classification_count += 1
        return classification

    # ------------------------------------------------------------------
    # Public API: Compliance Report
    # ------------------------------------------------------------------

    def generate_compliance_report(
        self,
        holdings: List[HoldingData],
        product_name: str = "",
    ) -> ComplianceReport:
        """Generate a full Article 9 compliance report.

        Classifies all holdings, computes breakdowns, and assembles a
        complete compliance report suitable for regulatory disclosure.

        Args:
            holdings: List of HoldingData to assess.
            product_name: Name of the financial product.

        Returns:
            ComplianceReport with all disclosures.
        """
        start = utcnow()

        # Classify portfolio
        result = self.classify_portfolio(holdings)

        # Objective breakdown
        objective_breakdown = self._build_objective_breakdown(
            result.classifications, result.total_nav,
        )

        # Non-sustainable breakdown
        non_sustainable = self._build_non_sustainable_breakdown(
            result.classifications, result.total_nav,
        )

        # Commitment status
        commitment = self._check_commitment(result)

        # Pass rates
        assessable = [
            c for c in result.classifications
            if c.classification not in (
                HoldingClassificationType.CASH_EQUIVALENT,
                HoldingClassificationType.HEDGING,
            )
        ]
        total_assessable = len(assessable) if assessable else 1

        contribution_pass = sum(1 for c in assessable if c.contribution_passed)
        dnsh_pass = sum(1 for c in assessable if c.dnsh_passed)
        gov_pass = sum(1 for c in assessable if c.good_governance_passed)

        avg_confidence = (
            sum(c.confidence for c in result.classifications) / len(result.classifications)
            if result.classifications else 0.0
        )

        elapsed_ms = (utcnow() - start).total_seconds() * 1000

        report = ComplianceReport(
            product_name=product_name,
            total_holdings=result.total_holdings,
            total_nav=result.total_nav,
            sustainable_holdings=result.sustainable_holdings,
            sustainable_pct=result.sustainable_pct,
            taxonomy_aligned_pct=result.taxonomy_aligned_pct,
            other_environmental_pct=result.other_environmental_pct,
            social_pct=result.social_pct,
            compliance_status=result.compliance_status,
            objective_breakdown=objective_breakdown,
            non_sustainable_breakdown=non_sustainable,
            commitment_status=commitment,
            classifications=result.classifications,
            contribution_pass_rate=_round_val(
                _safe_pct(contribution_pass, total_assessable), 2,
            ),
            dnsh_pass_rate=_round_val(
                _safe_pct(dnsh_pass, total_assessable), 2,
            ),
            governance_pass_rate=_round_val(
                _safe_pct(gov_pass, total_assessable), 2,
            ),
            average_confidence=_round_val(avg_confidence, 3),
            processing_time_ms=round(elapsed_ms, 2),
        )
        report.provenance_hash = _compute_hash(report)

        logger.info(
            "Compliance report generated for '%s': status=%s, "
            "sustainable=%.1f%%, time=%.1fms",
            product_name,
            report.compliance_status.value,
            report.sustainable_pct,
            elapsed_ms,
        )
        return report

    # ------------------------------------------------------------------
    # Public API: Commitment Check
    # ------------------------------------------------------------------

    def check_commitment(
        self,
        result: Optional[SustainableObjectiveResult] = None,
    ) -> CommitmentStatus:
        """Check whether portfolio meets Article 9 commitment.

        Args:
            result: Optional pre-computed result.

        Returns:
            CommitmentStatus with adherence assessment.
        """
        if result is None:
            if not self._classifications:
                raise ValueError(
                    "No classifications available. Call classify_portfolio first."
                )
            classifications = list(self._classifications.values())
            total_nav = sum(c.nav_value for c in classifications)
            sustainable_nav = sum(
                c.nav_value for c in classifications
                if c.classification in (
                    HoldingClassificationType.TAXONOMY_ALIGNED,
                    HoldingClassificationType.OTHER_ENVIRONMENTAL,
                    HoldingClassificationType.SOCIAL,
                )
            )
            sustainable_pct = _safe_pct(sustainable_nav, total_nav)
            taxonomy_pct = _safe_pct(
                sum(c.nav_value for c in classifications
                    if c.classification == HoldingClassificationType.TAXONOMY_ALIGNED),
                total_nav,
            )
            other_env_pct = _safe_pct(
                sum(c.nav_value for c in classifications
                    if c.classification == HoldingClassificationType.OTHER_ENVIRONMENTAL),
                total_nav,
            )
            social_pct = _safe_pct(
                sum(c.nav_value for c in classifications
                    if c.classification == HoldingClassificationType.SOCIAL),
                total_nav,
            )
        else:
            sustainable_pct = result.sustainable_pct
            taxonomy_pct = result.taxonomy_aligned_pct
            other_env_pct = result.other_environmental_pct
            social_pct = result.social_pct

        return self._build_commitment_status(
            sustainable_pct, taxonomy_pct, other_env_pct, social_pct,
        )

    # ------------------------------------------------------------------
    # Public API: Objective Breakdown
    # ------------------------------------------------------------------

    def get_objective_breakdown(
        self,
        classifications: Optional[List[HoldingClassification]] = None,
        total_nav: Optional[float] = None,
    ) -> List[ObjectiveBreakdownEntry]:
        """Get breakdown of sustainable holdings by objective.

        Args:
            classifications: Optional list (uses stored if not provided).
            total_nav: Total portfolio NAV.

        Returns:
            List of ObjectiveBreakdownEntry.
        """
        if classifications is None:
            classifications = list(self._classifications.values())
        if total_nav is None:
            total_nav = sum(c.nav_value for c in classifications)
        return self._build_objective_breakdown(classifications, total_nav)

    # ------------------------------------------------------------------
    # Private: Single Holding Classification
    # ------------------------------------------------------------------

    def _classify_single(self, holding: HoldingData) -> HoldingClassification:
        """Classify a single holding through the three-part test.

        Args:
            holding: Holding data to classify.

        Returns:
            HoldingClassification result.
        """
        evidence: List[str] = []

        # Handle cash/hedging exemptions
        if holding.is_cash_equivalent and self.config.allow_cash_hedging_exemption:
            evidence.append("Cash/cash-equivalent position (exempt from sustainability test)")
            result = HoldingClassification(
                holding_id=holding.holding_id,
                holding_name=holding.holding_name,
                classification=HoldingClassificationType.CASH_EQUIVALENT,
                objective_type=ObjectiveType.NONE,
                confidence=1.0,
                evidence=evidence,
                nav_value=holding.nav_value,
                weight_pct=holding.weight_pct,
            )
            result.provenance_hash = _compute_hash(result)
            return result

        if holding.is_hedging and self.config.allow_cash_hedging_exemption:
            evidence.append("Hedging/derivatives position (exempt from sustainability test)")
            result = HoldingClassification(
                holding_id=holding.holding_id,
                holding_name=holding.holding_name,
                classification=HoldingClassificationType.HEDGING,
                objective_type=ObjectiveType.NONE,
                confidence=1.0,
                evidence=evidence,
                nav_value=holding.nav_value,
                weight_pct=holding.weight_pct,
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Part 1: Contribution test
        contribution_passed, objective_type, env_obj, soc_obj = (
            self._test_contribution(holding, evidence)
        )

        # Part 2: DNSH test
        dnsh_passed = self._test_dnsh(holding, env_obj, soc_obj, evidence)

        # Part 3: Good governance test
        gov_passed = self._test_governance(holding, evidence)

        # Determine classification
        classification_type, confidence = self._determine_classification(
            holding, contribution_passed, dnsh_passed, gov_passed,
            objective_type, env_obj,
        )

        result = HoldingClassification(
            holding_id=holding.holding_id,
            holding_name=holding.holding_name,
            classification=classification_type,
            objective_type=objective_type,
            environmental_objective=env_obj,
            social_objective=soc_obj,
            contribution_passed=contribution_passed,
            dnsh_passed=dnsh_passed,
            good_governance_passed=gov_passed,
            confidence=_round_val(confidence, 3),
            evidence=evidence,
            nav_value=holding.nav_value,
            weight_pct=holding.weight_pct,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Private: Three-Part Test Components
    # ------------------------------------------------------------------

    def _test_contribution(
        self,
        holding: HoldingData,
        evidence: List[str],
    ) -> Tuple[bool, ObjectiveType, Optional[EnvironmentalObjective], Optional[SocialObjective]]:
        """Test Part 1: Does the holding contribute to a sustainable objective?

        Args:
            holding: Holding data.
            evidence: Evidence list to append to.

        Returns:
            Tuple of (passed, objective_type, env_objective, social_objective).
        """
        env_obj = holding.environmental_objective
        soc_obj = holding.social_objective

        if env_obj is not None:
            evidence.append(
                f"Contribution: Environmental objective '{env_obj.value}' identified"
            )
            if holding.contribution_evidence:
                evidence.append(
                    f"  Supporting evidence: {'; '.join(holding.contribution_evidence)}"
                )
            return True, ObjectiveType.ENVIRONMENTAL, env_obj, soc_obj

        if soc_obj is not None:
            evidence.append(
                f"Contribution: Social objective '{soc_obj.value}' identified"
            )
            if holding.contribution_evidence:
                evidence.append(
                    f"  Supporting evidence: {'; '.join(holding.contribution_evidence)}"
                )
            return True, ObjectiveType.SOCIAL, env_obj, soc_obj

        evidence.append("Contribution: No sustainable objective identified - FAIL")
        return False, ObjectiveType.NONE, None, None

    def _test_dnsh(
        self,
        holding: HoldingData,
        env_obj: Optional[EnvironmentalObjective],
        soc_obj: Optional[SocialObjective],
        evidence: List[str],
    ) -> bool:
        """Test Part 2: Does the holding pass the DNSH check?

        Uses pre-assessed DNSH status if available, otherwise evaluates
        PAI indicator data against configured thresholds.

        Args:
            holding: Holding data.
            env_obj: Environmental objective.
            soc_obj: Social objective.
            evidence: Evidence list to append to.

        Returns:
            True if DNSH passes.
        """
        # Use pre-assessed status if available
        if holding.dnsh_passed is not None:
            if holding.dnsh_passed:
                evidence.append("DNSH: Passed (pre-assessed)")
            else:
                evidence.append("DNSH: Failed (pre-assessed)")
            return holding.dnsh_passed

        # Determine required indicators
        required_indicators = self._get_required_dnsh_indicators(env_obj, soc_obj)

        checked: List[str] = []
        passed: List[str] = []
        failed: List[str] = []

        for indicator in required_indicators:
            if indicator in holding.pai_data:
                checked.append(indicator)
                threshold = self.config.pai_thresholds.get(indicator)
                value = holding.pai_data[indicator]

                if threshold is not None:
                    if self._pai_passes(indicator, value, threshold):
                        passed.append(indicator)
                    else:
                        failed.append(indicator)
                else:
                    passed.append(indicator)

            elif indicator in holding.pai_boolean_flags:
                checked.append(indicator)
                flag = holding.pai_boolean_flags[indicator]
                # For boolean flags: True typically means violation
                if flag:
                    failed.append(indicator)
                else:
                    passed.append(indicator)

        # Check coverage
        coverage = _safe_pct(len(checked), len(required_indicators)) if required_indicators else 100.0

        if len(failed) > 0:
            evidence.append(
                f"DNSH: Failed - {len(failed)} indicators breached: {failed}"
            )
            return False

        if coverage < self.config.dnsh_coverage_threshold:
            evidence.append(
                f"DNSH: Insufficient data - coverage {coverage:.1f}% "
                f"< threshold {self.config.dnsh_coverage_threshold}%"
            )
            return False

        if len(checked) > 0:
            evidence.append(
                f"DNSH: Passed - {len(passed)}/{len(checked)} indicators checked"
            )
            return True

        evidence.append("DNSH: No indicators available to check")
        return False

    def _test_governance(
        self,
        holding: HoldingData,
        evidence: List[str],
    ) -> bool:
        """Test Part 3: Does the investee follow good governance?

        Uses pre-assessed governance status if available, otherwise
        evaluates governance criteria data.

        Args:
            holding: Holding data.
            evidence: Evidence list to append to.

        Returns:
            True if governance passes.
        """
        # Use pre-assessed status if available
        if holding.good_governance_passed is not None:
            if holding.good_governance_passed:
                evidence.append("Governance: Passed (pre-assessed)")
            else:
                evidence.append("Governance: Failed (pre-assessed)")
            return holding.good_governance_passed

        gov = holding.governance_data
        if not gov:
            evidence.append("Governance: Insufficient data")
            return False

        mgmt = gov.get("sound_management_structures", False)
        employee = gov.get("employee_relations", False)
        remuneration = gov.get("remuneration_compliance", False)
        tax = gov.get("tax_compliance", False)
        has_controversies = gov.get("controversies", False)

        criteria_met = sum([mgmt, employee, remuneration, tax])

        if has_controversies:
            evidence.append(
                f"Governance: Failed - active controversies flagged "
                f"({criteria_met}/4 criteria met)"
            )
            return False

        if criteria_met >= self.config.governance_min_criteria:
            evidence.append(
                f"Governance: Passed ({criteria_met}/4 criteria met)"
            )
            return True

        evidence.append(
            f"Governance: Failed ({criteria_met}/4 criteria met, "
            f"minimum={self.config.governance_min_criteria})"
        )
        return False

    # ------------------------------------------------------------------
    # Private: Classification Determination
    # ------------------------------------------------------------------

    def _determine_classification(
        self,
        holding: HoldingData,
        contribution_passed: bool,
        dnsh_passed: bool,
        gov_passed: bool,
        objective_type: ObjectiveType,
        env_obj: Optional[EnvironmentalObjective],
    ) -> Tuple[HoldingClassificationType, float]:
        """Determine final classification and confidence.

        All three parts must pass for sustainable classification.
        Taxonomy alignment requires additional Taxonomy criteria.

        Args:
            holding: Holding data.
            contribution_passed: Part 1 result.
            dnsh_passed: Part 2 result.
            gov_passed: Part 3 result.
            objective_type: Top-level objective type.
            env_obj: Environmental objective if applicable.

        Returns:
            Tuple of (classification_type, confidence).
        """
        all_passed = contribution_passed and dnsh_passed and gov_passed

        if not all_passed:
            confidence = self._calculate_confidence(
                contribution_passed, dnsh_passed, gov_passed, 0.0,
            )
            return HoldingClassificationType.NOT_SUSTAINABLE, confidence

        # All three tests passed -- determine specific category
        confidence = self._calculate_confidence(
            contribution_passed, dnsh_passed, gov_passed,
            holding.taxonomy_aligned_pct,
        )

        # Check for Taxonomy alignment
        if (
            objective_type == ObjectiveType.ENVIRONMENTAL
            and holding.taxonomy_eligible
            and holding.taxonomy_aligned_pct > self.config.taxonomy_alignment_threshold
            and env_obj is not None
            and env_obj != EnvironmentalObjective.OTHER
        ):
            return HoldingClassificationType.TAXONOMY_ALIGNED, confidence

        # Environmental but not Taxonomy-aligned
        if objective_type == ObjectiveType.ENVIRONMENTAL:
            return HoldingClassificationType.OTHER_ENVIRONMENTAL, confidence

        # Social
        if objective_type == ObjectiveType.SOCIAL:
            return HoldingClassificationType.SOCIAL, confidence

        return HoldingClassificationType.NOT_SUSTAINABLE, confidence

    def _calculate_confidence(
        self,
        contribution_passed: bool,
        dnsh_passed: bool,
        gov_passed: bool,
        taxonomy_pct: float,
    ) -> float:
        """Calculate classification confidence.

        Confidence factors:
        - Contribution test (35% weight)
        - DNSH test (30% weight)
        - Governance test (25% weight)
        - Taxonomy alignment data (10% weight)

        Args:
            contribution_passed: Part 1 result.
            dnsh_passed: Part 2 result.
            gov_passed: Part 3 result.
            taxonomy_pct: Taxonomy alignment percentage.

        Returns:
            Confidence score (0.0 to 1.0).
        """
        contrib_score = 1.0 if contribution_passed else 0.0
        dnsh_score = 1.0 if dnsh_passed else 0.0
        gov_score = 1.0 if gov_passed else 0.0
        taxonomy_score = min(taxonomy_pct / 100.0, 1.0) if taxonomy_pct > 0 else 0.0

        confidence = (
            contrib_score * 0.35
            + dnsh_score * 0.30
            + gov_score * 0.25
            + taxonomy_score * 0.10
        )
        return min(confidence, 1.0)

    # ------------------------------------------------------------------
    # Private: DNSH Helpers
    # ------------------------------------------------------------------

    def _get_required_dnsh_indicators(
        self,
        env_obj: Optional[EnvironmentalObjective],
        soc_obj: Optional[SocialObjective],
    ) -> List[str]:
        """Get the list of PAI indicators required for DNSH assessment.

        Args:
            env_obj: Environmental objective.
            soc_obj: Social objective.

        Returns:
            List of required PAI indicator names.
        """
        indicators: List[str] = []

        if env_obj is not None:
            indicators.extend(
                ENVIRONMENTAL_DNSH_INDICATORS.get(env_obj.value, [])
            )

        if soc_obj is not None:
            indicators.extend(
                SOCIAL_DNSH_INDICATORS.get(soc_obj.value, [])
            )

        if not indicators:
            # Fall back to all known indicators
            indicators = list(self.config.pai_thresholds.keys())

        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for ind in indicators:
            if ind not in seen:
                seen.add(ind)
                unique.append(ind)
        return unique

    def _pai_passes(
        self,
        indicator: str,
        value: float,
        threshold: float,
    ) -> bool:
        """Check if a PAI indicator value passes the threshold.

        Args:
            indicator: PAI indicator name.
            value: Measured value.
            threshold: Threshold value.

        Returns:
            True if the indicator passes.
        """
        if indicator in self.config.higher_is_better_indicators:
            return value >= threshold
        return value <= threshold

    # ------------------------------------------------------------------
    # Private: Compliance Determination
    # ------------------------------------------------------------------

    def _determine_compliance(
        self,
        sustainable_pct: float,
        cash_hedging_nav: float,
        total_nav: float,
    ) -> ComplianceStatus:
        """Determine overall Article 9 compliance status.

        Args:
            sustainable_pct: Sustainable proportion of NAV.
            cash_hedging_nav: Cash and hedging NAV.
            total_nav: Total portfolio NAV.

        Returns:
            ComplianceStatus enum value.
        """
        if total_nav == 0.0:
            return ComplianceStatus.INSUFFICIENT_DATA

        # If cash/hedging exemption is enabled, adjust the effective threshold
        if self.config.allow_cash_hedging_exemption:
            cash_hedging_pct = _safe_pct(cash_hedging_nav, total_nav)
            if cash_hedging_pct > self.config.cash_hedging_max_pct:
                logger.warning(
                    "Cash/hedging proportion %.1f%% exceeds maximum %.1f%%",
                    cash_hedging_pct,
                    self.config.cash_hedging_max_pct,
                )

        if sustainable_pct >= self.config.minimum_sustainable_pct:
            return ComplianceStatus.COMPLIANT

        if sustainable_pct >= self.config.marginal_threshold_pct:
            return ComplianceStatus.MARGINAL

        return ComplianceStatus.NON_COMPLIANT

    # ------------------------------------------------------------------
    # Private: Breakdown Builders
    # ------------------------------------------------------------------

    def _build_objective_breakdown(
        self,
        classifications: List[HoldingClassification],
        total_nav: float,
    ) -> List[ObjectiveBreakdownEntry]:
        """Build breakdown of sustainable holdings by objective.

        Args:
            classifications: All holding classifications.
            total_nav: Total portfolio NAV.

        Returns:
            List of ObjectiveBreakdownEntry.
        """
        entries: List[ObjectiveBreakdownEntry] = []

        # Environmental objectives
        for obj in EnvironmentalObjective:
            matching = [
                c for c in classifications
                if c.environmental_objective == obj
                and c.classification in (
                    HoldingClassificationType.TAXONOMY_ALIGNED,
                    HoldingClassificationType.OTHER_ENVIRONMENTAL,
                )
            ]
            if matching:
                nav = sum(c.nav_value for c in matching)
                avg_conf = sum(c.confidence for c in matching) / len(matching)
                entries.append(ObjectiveBreakdownEntry(
                    objective_type=ObjectiveType.ENVIRONMENTAL,
                    objective_name=obj.value.replace("_", " ").title(),
                    objective_value=obj.value,
                    holding_count=len(matching),
                    nav_value=_round_val(nav, 2),
                    proportion_pct=_round_val(_safe_pct(nav, total_nav), 2),
                    average_confidence=_round_val(avg_conf, 3),
                ))

        # Social objectives
        for obj in SocialObjective:
            matching = [
                c for c in classifications
                if c.social_objective == obj
                and c.classification == HoldingClassificationType.SOCIAL
            ]
            if matching:
                nav = sum(c.nav_value for c in matching)
                avg_conf = sum(c.confidence for c in matching) / len(matching)
                entries.append(ObjectiveBreakdownEntry(
                    objective_type=ObjectiveType.SOCIAL,
                    objective_name=obj.value.replace("_", " ").title(),
                    objective_value=obj.value,
                    holding_count=len(matching),
                    nav_value=_round_val(nav, 2),
                    proportion_pct=_round_val(_safe_pct(nav, total_nav), 2),
                    average_confidence=_round_val(avg_conf, 3),
                ))

        return entries

    def _build_non_sustainable_breakdown(
        self,
        classifications: List[HoldingClassification],
        total_nav: float,
    ) -> NonSustainableBreakdown:
        """Build breakdown of non-sustainable holdings.

        Args:
            classifications: All holding classifications.
            total_nav: Total portfolio NAV.

        Returns:
            NonSustainableBreakdown.
        """
        cash_holdings = [
            c for c in classifications
            if c.classification == HoldingClassificationType.CASH_EQUIVALENT
        ]
        hedging_holdings = [
            c for c in classifications
            if c.classification == HoldingClassificationType.HEDGING
        ]
        other_ns_holdings = [
            c for c in classifications
            if c.classification == HoldingClassificationType.NOT_SUSTAINABLE
        ]

        cash_nav = sum(c.nav_value for c in cash_holdings)
        hedging_nav = sum(c.nav_value for c in hedging_holdings)
        other_nav = sum(c.nav_value for c in other_ns_holdings)
        total_ns_nav = cash_nav + hedging_nav + other_nav

        all_ns_ids = (
            [c.holding_id for c in cash_holdings]
            + [c.holding_id for c in hedging_holdings]
            + [c.holding_id for c in other_ns_holdings]
        )

        return NonSustainableBreakdown(
            total_non_sustainable_nav=_round_val(total_ns_nav, 2),
            total_non_sustainable_pct=_round_val(_safe_pct(total_ns_nav, total_nav), 2),
            cash_equivalent_nav=_round_val(cash_nav, 2),
            cash_equivalent_pct=_round_val(_safe_pct(cash_nav, total_nav), 2),
            hedging_nav=_round_val(hedging_nav, 2),
            hedging_pct=_round_val(_safe_pct(hedging_nav, total_nav), 2),
            other_non_sustainable_nav=_round_val(other_nav, 2),
            other_non_sustainable_pct=_round_val(_safe_pct(other_nav, total_nav), 2),
            holdings=all_ns_ids,
        )

    # ------------------------------------------------------------------
    # Private: Commitment
    # ------------------------------------------------------------------

    def _check_commitment(
        self,
        result: SustainableObjectiveResult,
    ) -> CommitmentStatus:
        """Check commitment adherence from a classification result.

        Args:
            result: SustainableObjectiveResult.

        Returns:
            CommitmentStatus.
        """
        return self._build_commitment_status(
            result.sustainable_pct,
            result.taxonomy_aligned_pct,
            result.other_environmental_pct,
            result.social_pct,
        )

    def _build_commitment_status(
        self,
        sustainable_pct: float,
        taxonomy_pct: float,
        other_env_pct: float,
        social_pct: float,
    ) -> CommitmentStatus:
        """Build a CommitmentStatus from proportion values.

        Args:
            sustainable_pct: Total sustainable proportion.
            taxonomy_pct: Taxonomy-aligned proportion.
            other_env_pct: Other environmental proportion.
            social_pct: Social proportion.

        Returns:
            CommitmentStatus.
        """
        minimum = self.config.minimum_sustainable_pct
        gap = sustainable_pct - minimum
        meets = sustainable_pct >= minimum

        if meets:
            compliance = ComplianceStatus.COMPLIANT
        elif sustainable_pct >= self.config.marginal_threshold_pct:
            compliance = ComplianceStatus.MARGINAL
        else:
            compliance = ComplianceStatus.NON_COMPLIANT

        status = CommitmentStatus(
            minimum_sustainable_pct=minimum,
            actual_sustainable_pct=_round_val(sustainable_pct, 2),
            taxonomy_aligned_pct=_round_val(taxonomy_pct, 2),
            other_environmental_pct=_round_val(other_env_pct, 2),
            social_pct=_round_val(social_pct, 2),
            gap_pct=_round_val(gap, 2),
            meets_commitment=meets,
            compliance_status=compliance,
        )
        status.provenance_hash = _compute_hash(status)
        return status

    # ------------------------------------------------------------------
    # Private: Investment Lookup
    # ------------------------------------------------------------------

    def _find_holding(self, holding_id: str) -> Optional[HoldingData]:
        """Find a holding by ID.

        Args:
            holding_id: Target holding identifier.

        Returns:
            HoldingData if found, None otherwise.
        """
        for h in self._holdings:
            if h.holding_id == holding_id:
                return h
        return None

    # ------------------------------------------------------------------
    # Read-only Properties
    # ------------------------------------------------------------------

    @property
    def classification_count(self) -> int:
        """Number of classifications performed since initialization."""
        return self._classification_count

    @property
    def stored_classifications(self) -> Dict[str, HoldingClassification]:
        """Return the current stored classifications."""
        return dict(self._classifications)
