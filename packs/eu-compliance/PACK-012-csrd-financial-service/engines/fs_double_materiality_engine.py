# -*- coding: utf-8 -*-
"""
FSDoubleMaterialityEngine - PACK-012 CSRD Financial Service Engine 6
======================================================================

Financial-services-specific double materiality assessment engine for
CSRD/ESRS reporting by credit institutions, asset managers, and insurers.

Implements dual-lens materiality assessment: financial materiality
(outside-in: how sustainability matters affect the entity) and impact
materiality (inside-out: how the entity affects sustainability matters).
Includes IRO identification per ESRS, FI-specific topics such as
financed emissions, responsible lending, financial inclusion, and fair
pricing, plus cross-reference to ESRS datapoints (E1-E5, S1-S4, G1).

Key Regulatory References:
    - ESRS 1 Chapter 3 (Double Materiality)
    - ESRS 2 IRO-1 / IRO-2 (Description of processes)
    - ESRS 2 SBM-3 (Material impacts, risks and opportunities)
    - ESRS E1-E5, S1-S4, G1 (Topical standards)
    - EBA Guidelines on ESG Risks Management (EBA/GL/2025/01)
    - ECB Guide on Climate and Environmental Risks (2020)
    - EFRAG IG 1 (Materiality Assessment Implementation Guidance)

Formulas:
    Financial Materiality Score = likelihood * magnitude * scope_factor
    Impact Materiality Score = severity * likelihood * scope * irremediability
    Overall Topic Score = max(financial_score, impact_score) (per ESRS 1 para 38)
    Stakeholder Relevance = SUM(stakeholder_weight * relevance_rating)
    Material Topic Threshold = score >= threshold (configurable, default 50/100)

Zero-Hallucination:
    - All scores use deterministic weighted-product formulae
    - ESRS datapoint mappings from published EFRAG taxonomy
    - FI-specific topics from EBA/ECB regulatory guidance
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-012 CSRD Financial Service
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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _safe_divide(
    numerator: float, denominator: float, default: float = 0.0,
) -> float:
    """Safely divide two numbers, returning default on zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator

def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely."""
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0

def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp a value to [low, high] range."""
    return max(low, min(high, value))

def _round_val(value: float, places: int = 4) -> float:
    """Round a float to specified decimal places."""
    return round(value, places)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MaterialityDimension(str, Enum):
    """Materiality assessment dimension per ESRS 1."""
    FINANCIAL = "financial"         # Outside-in
    IMPACT = "impact"               # Inside-out

class IROType(str, Enum):
    """Impact, Risk, Opportunity classification per ESRS 2."""
    IMPACT = "impact"
    RISK = "risk"
    OPPORTUNITY = "opportunity"

class ESRSStandard(str, Enum):
    """ESRS topical standards."""
    E1 = "E1"   # Climate change
    E2 = "E2"   # Pollution
    E3 = "E3"   # Water and marine resources
    E4 = "E4"   # Biodiversity and ecosystems
    E5 = "E5"   # Resource use and circular economy
    S1 = "S1"   # Own workforce
    S2 = "S2"   # Workers in the value chain
    S3 = "S3"   # Affected communities
    S4 = "S4"   # Consumers and end-users
    G1 = "G1"   # Business conduct

class SeverityScale(str, Enum):
    """Severity scale for impact assessment (ESRS 1 para 45)."""
    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"

class LikelihoodScale(str, Enum):
    """Likelihood scale for risk/opportunity assessment."""
    VERY_UNLIKELY = "very_unlikely"
    UNLIKELY = "unlikely"
    POSSIBLE = "possible"
    LIKELY = "likely"
    VERY_LIKELY = "very_likely"

class MaterialityOutcome(str, Enum):
    """Whether a topic is material or not."""
    MATERIAL = "material"
    NOT_MATERIAL = "not_material"
    UNDER_REVIEW = "under_review"

class StakeholderGroup(str, Enum):
    """Stakeholder groups for FI materiality assessment."""
    REGULATORS = "regulators"
    INVESTORS = "investors"
    CUSTOMERS = "customers"
    EMPLOYEES = "employees"
    COMMUNITIES = "communities"
    NGOS = "ngos"
    SUPPLIERS = "suppliers"
    INDUSTRY_BODIES = "industry_bodies"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Numeric mapping for severity scale (deterministic)
SEVERITY_SCORES: Dict[str, float] = {
    SeverityScale.NEGLIGIBLE.value: 10.0,
    SeverityScale.MINOR.value: 30.0,
    SeverityScale.MODERATE.value: 50.0,
    SeverityScale.SIGNIFICANT.value: 75.0,
    SeverityScale.CRITICAL.value: 100.0,
}

# Numeric mapping for likelihood scale
LIKELIHOOD_SCORES: Dict[str, float] = {
    LikelihoodScale.VERY_UNLIKELY.value: 10.0,
    LikelihoodScale.UNLIKELY.value: 25.0,
    LikelihoodScale.POSSIBLE.value: 50.0,
    LikelihoodScale.LIKELY.value: 75.0,
    LikelihoodScale.VERY_LIKELY.value: 95.0,
}

# Default materiality threshold (score 0-100)
DEFAULT_MATERIALITY_THRESHOLD: float = 50.0

# Default stakeholder weights (sum to 1.0)
DEFAULT_STAKEHOLDER_WEIGHTS: Dict[str, float] = {
    StakeholderGroup.REGULATORS.value: 0.20,
    StakeholderGroup.INVESTORS.value: 0.20,
    StakeholderGroup.CUSTOMERS.value: 0.20,
    StakeholderGroup.EMPLOYEES.value: 0.10,
    StakeholderGroup.COMMUNITIES.value: 0.10,
    StakeholderGroup.NGOS.value: 0.10,
    StakeholderGroup.SUPPLIERS.value: 0.05,
    StakeholderGroup.INDUSTRY_BODIES.value: 0.05,
}

# FI-specific materiality topics (beyond standard ESRS topics)
FI_SPECIFIC_TOPICS: Dict[str, Dict[str, Any]] = {
    "financed_emissions": {
        "name": "Financed Emissions (Scope 3 Cat 15)",
        "esrs_standard": ESRSStandard.E1.value,
        "description": "GHG emissions from lending and investment portfolios",
        "fi_relevance": "critical",
    },
    "insured_emissions": {
        "name": "Insured Emissions",
        "esrs_standard": ESRSStandard.E1.value,
        "description": "GHG emissions from insurance underwriting",
        "fi_relevance": "high",
    },
    "financial_inclusion": {
        "name": "Financial Inclusion",
        "esrs_standard": ESRSStandard.S4.value,
        "description": "Access to financial services for underserved populations",
        "fi_relevance": "high",
    },
    "responsible_lending": {
        "name": "Responsible Lending",
        "esrs_standard": ESRSStandard.S4.value,
        "description": "Responsible lending practices and over-indebtedness prevention",
        "fi_relevance": "high",
    },
    "fair_pricing": {
        "name": "Fair Pricing",
        "esrs_standard": ESRSStandard.G1.value,
        "description": "Fair and transparent pricing of financial products",
        "fi_relevance": "high",
    },
    "data_protection": {
        "name": "Data Protection and Privacy",
        "esrs_standard": ESRSStandard.S4.value,
        "description": "Protection of customer financial data and privacy",
        "fi_relevance": "critical",
    },
    "climate_risk_integration": {
        "name": "Climate Risk in Risk Management",
        "esrs_standard": ESRSStandard.E1.value,
        "description": "Integration of climate risk in credit/investment decisions",
        "fi_relevance": "critical",
    },
    "stranded_assets": {
        "name": "Stranded Asset Exposure",
        "esrs_standard": ESRSStandard.E1.value,
        "description": "Exposure to potentially stranded fossil fuel assets",
        "fi_relevance": "high",
    },
    "green_product_design": {
        "name": "Green Product Design",
        "esrs_standard": ESRSStandard.E1.value,
        "description": "Development of green bonds, ESG funds, sustainable loans",
        "fi_relevance": "high",
    },
    "aml_kyc": {
        "name": "AML/KYC and Financial Crime Prevention",
        "esrs_standard": ESRSStandard.G1.value,
        "description": "Anti-money laundering and know-your-customer processes",
        "fi_relevance": "critical",
    },
    "regulatory_capital": {
        "name": "Regulatory Capital Adequacy (ESG Risks)",
        "esrs_standard": ESRSStandard.E1.value,
        "description": "Impact of ESG risks on capital requirements (CRR3)",
        "fi_relevance": "critical",
    },
    "taxonomy_alignment": {
        "name": "EU Taxonomy Alignment",
        "esrs_standard": ESRSStandard.E1.value,
        "description": "Alignment of lending/investment book with EU Taxonomy",
        "fi_relevance": "critical",
    },
}

# ESRS datapoint mapping: topic -> list of relevant datapoints
ESRS_DATAPOINT_REGISTRY: Dict[str, List[str]] = {
    ESRSStandard.E1.value: [
        "E1-1: Transition plan for climate change mitigation",
        "E1-2: Policies related to climate change mitigation and adaptation",
        "E1-3: Actions and resources in relation to climate change",
        "E1-4: Targets related to climate change mitigation and adaptation",
        "E1-5: Energy consumption and mix",
        "E1-6: Gross Scopes 1, 2, 3 and Total GHG emissions",
        "E1-7: GHG removals and GHG mitigation projects (carbon credits)",
        "E1-8: Internal carbon pricing",
        "E1-9: Anticipated financial effects from climate change",
    ],
    ESRSStandard.E2.value: [
        "E2-1: Policies related to pollution",
        "E2-2: Actions and resources related to pollution",
        "E2-3: Targets related to pollution",
        "E2-4: Pollution of air, water and soil",
        "E2-5: Substances of concern and substances of very high concern",
        "E2-6: Anticipated financial effects from pollution-related impacts",
    ],
    ESRSStandard.E3.value: [
        "E3-1: Policies related to water and marine resources",
        "E3-2: Actions and resources related to water and marine resources",
        "E3-3: Targets related to water and marine resources",
        "E3-4: Water consumption",
        "E3-5: Anticipated financial effects from water-related impacts",
    ],
    ESRSStandard.E4.value: [
        "E4-1: Transition plan on biodiversity and ecosystems",
        "E4-2: Policies related to biodiversity and ecosystems",
        "E4-3: Actions and resources related to biodiversity",
        "E4-4: Targets related to biodiversity and ecosystems",
        "E4-5: Impact metrics related to biodiversity change",
        "E4-6: Anticipated financial effects from biodiversity impacts",
    ],
    ESRSStandard.E5.value: [
        "E5-1: Policies related to resource use and circular economy",
        "E5-2: Actions and resources related to resource use",
        "E5-3: Targets related to resource use and circular economy",
        "E5-4: Resource inflows",
        "E5-5: Resource outflows",
        "E5-6: Anticipated financial effects from resource use impacts",
    ],
    ESRSStandard.S1.value: [
        "S1-1: Policies related to own workforce",
        "S1-2: Processes for engaging with own workforce",
        "S1-3: Processes to remediate negative impacts on own workforce",
        "S1-4: Taking action on material impacts on own workforce",
        "S1-5: Targets related to managing impacts on own workforce",
        "S1-6: Characteristics of employees",
        "S1-7: Characteristics of non-employee workers",
        "S1-8: Collective bargaining coverage and social dialogue",
        "S1-9: Diversity metrics",
        "S1-10: Adequate wages",
        "S1-11: Social protection",
        "S1-12: Persons with disabilities",
        "S1-13: Training and skills development metrics",
        "S1-14: Health and safety metrics",
        "S1-15: Work-life balance metrics",
        "S1-16: Remuneration metrics",
        "S1-17: Incidents, complaints and severe human rights impacts",
    ],
    ESRSStandard.S2.value: [
        "S2-1: Policies related to value chain workers",
        "S2-2: Processes for engaging with value chain workers",
        "S2-3: Processes to remediate negative impacts",
        "S2-4: Taking action on material impacts",
        "S2-5: Targets related to value chain workers",
    ],
    ESRSStandard.S3.value: [
        "S3-1: Policies related to affected communities",
        "S3-2: Processes for engaging with affected communities",
        "S3-3: Processes to remediate negative impacts",
        "S3-4: Taking action on material impacts",
        "S3-5: Targets related to affected communities",
    ],
    ESRSStandard.S4.value: [
        "S4-1: Policies related to consumers and end-users",
        "S4-2: Processes for engaging with consumers and end-users",
        "S4-3: Processes to remediate negative impacts",
        "S4-4: Taking action on material impacts",
        "S4-5: Targets related to consumers and end-users",
    ],
    ESRSStandard.G1.value: [
        "G1-1: Business conduct policies and corporate culture",
        "G1-2: Management of relationships with suppliers",
        "G1-3: Prevention and detection of corruption and bribery",
        "G1-4: Incidents of corruption or bribery",
        "G1-5: Political influence and lobbying activities",
        "G1-6: Payment practices",
    ],
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class MaterialityTopicData(BaseModel):
    """Input data for a single materiality topic.

    Attributes:
        topic_id: Unique topic identifier.
        topic_name: Human-readable topic name.
        esrs_standard: Applicable ESRS standard.
        is_fi_specific: Whether this is an FI-specific topic.
        financial_likelihood: Likelihood of financial impact.
        financial_magnitude: Magnitude of financial impact (0-100).
        financial_scope: Scope of financial impact (0-1 fraction).
        impact_severity: Severity of sustainability impact.
        impact_likelihood: Likelihood of sustainability impact.
        impact_scope: Breadth of impact (0-1 fraction).
        impact_irremediability: Irremediability factor (0-1).
        stakeholder_relevance: Stakeholder relevance ratings.
        description: Topic description.
        iro_type: Predominant IRO classification.
    """
    topic_id: str = Field(
        default_factory=_new_uuid, description="Unique topic ID",
    )
    topic_name: str = Field(default="", description="Topic name")
    esrs_standard: ESRSStandard = Field(
        default=ESRSStandard.E1, description="Applicable ESRS standard",
    )
    is_fi_specific: bool = Field(
        default=False, description="Whether FI-specific topic",
    )

    # Financial materiality inputs
    financial_likelihood: LikelihoodScale = Field(
        default=LikelihoodScale.POSSIBLE,
        description="Likelihood of financial impact",
    )
    financial_magnitude: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Magnitude of financial impact (0-100)",
    )
    financial_scope: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Scope of financial impact (0-1)",
    )

    # Impact materiality inputs
    impact_severity: SeverityScale = Field(
        default=SeverityScale.MODERATE,
        description="Severity of sustainability impact",
    )
    impact_likelihood: LikelihoodScale = Field(
        default=LikelihoodScale.POSSIBLE,
        description="Likelihood of sustainability impact",
    )
    impact_scope: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Breadth of impact (0-1)",
    )
    impact_irremediability: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Irremediability factor (0-1)",
    )

    # Stakeholder inputs
    stakeholder_relevance: Dict[str, float] = Field(
        default_factory=dict,
        description="Stakeholder relevance ratings (0-100 per group)",
    )

    # Metadata
    description: str = Field(default="", description="Topic description")
    iro_type: IROType = Field(
        default=IROType.IMPACT, description="Predominant IRO type",
    )

class StakeholderInput(BaseModel):
    """Stakeholder engagement input for materiality assessment."""
    stakeholder_id: str = Field(
        default_factory=_new_uuid, description="Stakeholder input ID",
    )
    stakeholder_group: StakeholderGroup = Field(
        default=StakeholderGroup.REGULATORS, description="Stakeholder group",
    )
    topic_ratings: Dict[str, float] = Field(
        default_factory=dict,
        description="Ratings by topic_id (0-100)",
    )
    weight_override: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Optional weight override",
    )
    engagement_method: str = Field(
        default="survey", description="Engagement method",
    )
    response_date: datetime = Field(
        default_factory=utcnow, description="Response date",
    )

class IROAssessment(BaseModel):
    """Impact, Risk, or Opportunity assessment result per topic."""
    iro_id: str = Field(default_factory=_new_uuid, description="IRO ID")
    topic_id: str = Field(default="", description="Source topic ID")
    topic_name: str = Field(default="", description="Topic name")
    iro_type: IROType = Field(
        default=IROType.IMPACT, description="IRO type",
    )
    esrs_standard: ESRSStandard = Field(
        default=ESRSStandard.E1, description="ESRS standard",
    )
    description: str = Field(default="", description="IRO description")
    severity: SeverityScale = Field(
        default=SeverityScale.MODERATE, description="Severity rating",
    )
    likelihood: LikelihoodScale = Field(
        default=LikelihoodScale.POSSIBLE, description="Likelihood rating",
    )
    financial_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Financial materiality score",
    )
    impact_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Impact materiality score",
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall materiality score",
    )
    is_material: bool = Field(default=False, description="Is topic material")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class FinancedImpactAssessment(BaseModel):
    """Assessment of financed/insured/advisory impact.

    FI-specific assessment of impact through financial intermediation.
    """
    assessment_id: str = Field(
        default_factory=_new_uuid, description="Assessment ID",
    )
    impact_channel: str = Field(
        default="lending",
        description="Impact channel: lending, investment, insurance, advisory",
    )
    total_exposure_eur: float = Field(
        default=0.0, ge=0.0, description="Total exposure (EUR)",
    )
    high_impact_exposure_eur: float = Field(
        default=0.0, ge=0.0,
        description="Exposure to high-impact sectors (EUR)",
    )
    high_impact_ratio_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="High-impact exposure ratio (%)",
    )
    financed_emissions_tco2e: float = Field(
        default=0.0, ge=0.0,
        description="Attributed financed emissions (tCO2e)",
    )
    taxonomy_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="EU Taxonomy aligned exposure (%)",
    )
    impact_severity: SeverityScale = Field(
        default=SeverityScale.MODERATE,
        description="Assessed impact severity",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class DatapointMapping(BaseModel):
    """Mapping of a material topic to ESRS datapoints."""
    mapping_id: str = Field(
        default_factory=_new_uuid, description="Mapping ID",
    )
    topic_id: str = Field(default="", description="Source topic ID")
    topic_name: str = Field(default="", description="Topic name")
    esrs_standard: ESRSStandard = Field(
        default=ESRSStandard.E1, description="ESRS standard",
    )
    required_datapoints: List[str] = Field(
        default_factory=list,
        description="List of required ESRS datapoints",
    )
    total_datapoints: int = Field(
        default=0, ge=0, description="Total datapoints for this standard",
    )
    is_material: bool = Field(default=False, description="Is topic material")

class MaterialityMatrix(BaseModel):
    """Materiality matrix visualization data.

    Provides coordinates for each topic on a 2D materiality matrix
    (financial materiality on x-axis, impact materiality on y-axis).
    """
    matrix_id: str = Field(
        default_factory=_new_uuid, description="Matrix ID",
    )
    topics: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Topic coordinates [{topic_id, name, x, y, material}]",
    )
    threshold_x: float = Field(
        default=50.0, description="Financial materiality threshold",
    )
    threshold_y: float = Field(
        default=50.0, description="Impact materiality threshold",
    )
    quadrant_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of topics per quadrant",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class FSMaterialityResult(BaseModel):
    """Complete double materiality assessment result for a financial institution."""
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    institution_name: str = Field(
        default="", description="Financial institution name",
    )
    reporting_date: datetime = Field(
        default_factory=utcnow, description="Reporting date",
    )

    # IRO assessments
    iro_assessments: List[IROAssessment] = Field(
        default_factory=list, description="IRO assessments per topic",
    )

    # Material topics
    material_topics: List[str] = Field(
        default_factory=list, description="List of material topic names",
    )
    material_topic_count: int = Field(
        default=0, ge=0, description="Number of material topics",
    )
    total_topics_assessed: int = Field(
        default=0, ge=0, description="Total topics assessed",
    )

    # FI-specific assessments
    financed_impact_assessments: List[FinancedImpactAssessment] = Field(
        default_factory=list, description="Financed impact assessments",
    )

    # Datapoint mappings
    datapoint_mappings: List[DatapointMapping] = Field(
        default_factory=list, description="ESRS datapoint mappings",
    )
    total_required_datapoints: int = Field(
        default=0, ge=0, description="Total required ESRS datapoints",
    )

    # Materiality matrix
    materiality_matrix: Optional[MaterialityMatrix] = Field(
        default=None, description="Materiality matrix data",
    )

    # Standards coverage
    material_standards: List[str] = Field(
        default_factory=list,
        description="ESRS standards determined material",
    )
    standards_coverage: Dict[str, bool] = Field(
        default_factory=dict,
        description="Materiality by ESRS standard",
    )

    # Stakeholder analysis
    stakeholder_alignment_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Alignment between stakeholder views and final assessment",
    )

    # Metadata
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp",
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class FSMaterialityConfig(BaseModel):
    """Configuration for the FSDoubleMaterialityEngine.

    Attributes:
        institution_name: Name of the financial institution.
        materiality_threshold: Score threshold for material topics (0-100).
        financial_weight_in_overall: Weight of financial materiality (0-1).
        impact_weight_in_overall: Weight of impact materiality (0-1).
        use_max_rule: Use max(financial, impact) per ESRS 1 para 38.
        include_fi_specific_topics: Include FI-specific topics.
        stakeholder_weights: Custom stakeholder weights.
        scope_factor_financial: Default scope factor for financial materiality.
        irremediability_weight: Weight of irremediability in impact score.
    """
    institution_name: str = Field(
        default="Financial Institution", description="Institution name",
    )
    materiality_threshold: float = Field(
        default=DEFAULT_MATERIALITY_THRESHOLD, ge=0.0, le=100.0,
        description="Materiality threshold (0-100)",
    )
    financial_weight_in_overall: float = Field(
        default=0.50, ge=0.0, le=1.0,
        description="Financial materiality weight",
    )
    impact_weight_in_overall: float = Field(
        default=0.50, ge=0.0, le=1.0,
        description="Impact materiality weight",
    )
    use_max_rule: bool = Field(
        default=True,
        description="Use max(fin, impact) per ESRS 1 para 38",
    )
    include_fi_specific_topics: bool = Field(
        default=True, description="Include FI-specific topics",
    )
    stakeholder_weights: Dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_STAKEHOLDER_WEIGHTS),
        description="Stakeholder group weights",
    )
    scope_factor_financial: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Default scope factor for financial materiality",
    )
    irremediability_weight: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Weight of irremediability in impact score",
    )

    @model_validator(mode="after")
    def _validate_weights(self) -> "FSMaterialityConfig":
        if not self.use_max_rule:
            total = (
                self.financial_weight_in_overall
                + self.impact_weight_in_overall
            )
            if abs(total - 1.0) > 0.001:
                raise ValueError(
                    "financial + impact weights must equal 1.0 "
                    f"when use_max_rule=False, got {total}"
                )
        return self

# ---------------------------------------------------------------------------
# model_rebuild for forward reference resolution
# ---------------------------------------------------------------------------

FSMaterialityConfig.model_rebuild()
MaterialityTopicData.model_rebuild()
StakeholderInput.model_rebuild()
IROAssessment.model_rebuild()
FinancedImpactAssessment.model_rebuild()
DatapointMapping.model_rebuild()
MaterialityMatrix.model_rebuild()
FSMaterialityResult.model_rebuild()

# ---------------------------------------------------------------------------
# FSDoubleMaterialityEngine
# ---------------------------------------------------------------------------

class FSDoubleMaterialityEngine:
    """
    Financial-services-specific double materiality assessment engine.

    Implements the ESRS double materiality assessment for credit
    institutions, asset managers, and insurers, including FI-specific
    topics, IRO identification, ESRS datapoint mapping, stakeholder
    relevance analysis, and materiality matrix generation.

    Zero-Hallucination Guarantees:
        - All scores use deterministic formulae (products of scaled inputs)
        - ESRS datapoint mappings from published EFRAG taxonomy
        - FI topics from EBA/ECB regulatory guidance
        - SHA-256 provenance hash on every result
        - No LLM involvement in any calculation path

    Attributes:
        config: Engine configuration.
    """

    def __init__(self, config: FSMaterialityConfig) -> None:
        """Initialize FSDoubleMaterialityEngine.

        Args:
            config: Engine configuration.
        """
        self.config = config
        logger.info(
            "FSDoubleMaterialityEngine initialized (v%s) for '%s'",
            _MODULE_VERSION, config.institution_name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_materiality(
        self,
        topics: List[MaterialityTopicData],
        stakeholder_inputs: Optional[List[StakeholderInput]] = None,
        financed_impacts: Optional[List[FinancedImpactAssessment]] = None,
        reporting_date: Optional[datetime] = None,
    ) -> FSMaterialityResult:
        """Run the complete double materiality assessment.

        Args:
            topics: List of topics to assess.
            stakeholder_inputs: Optional stakeholder engagement data.
            financed_impacts: Optional financed impact assessments.
            reporting_date: Optional reporting date.

        Returns:
            Complete FSMaterialityResult.
        """
        import time

        start = time.perf_counter()

        r_date = reporting_date or utcnow()
        stakeholder_inputs = stakeholder_inputs or []
        financed_impacts = financed_impacts or []

        # 1. Score each topic on both dimensions
        iro_assessments: List[IROAssessment] = []
        for topic in topics:
            iro = self._assess_topic(topic, stakeholder_inputs)
            iro_assessments.append(iro)

        # 2. Determine material topics
        material_topics = [
            iro.topic_name for iro in iro_assessments if iro.is_material
        ]
        material_topic_ids = {
            iro.topic_id for iro in iro_assessments if iro.is_material
        }

        # 3. Map material topics to ESRS datapoints
        datapoint_mappings = self._map_datapoints(iro_assessments)
        total_dps = sum(len(m.required_datapoints) for m in datapoint_mappings)

        # 4. Determine which ESRS standards are material
        material_standards_set: set = set()
        for iro in iro_assessments:
            if iro.is_material:
                material_standards_set.add(iro.esrs_standard.value)
        standards_coverage = {
            std.value: std.value in material_standards_set
            for std in ESRSStandard
        }

        # 5. Build materiality matrix
        matrix = self._build_materiality_matrix(iro_assessments)

        # 6. Stakeholder alignment
        alignment = self._compute_stakeholder_alignment(
            iro_assessments, stakeholder_inputs,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        result = FSMaterialityResult(
            institution_name=self.config.institution_name,
            reporting_date=r_date,
            iro_assessments=iro_assessments,
            material_topics=material_topics,
            material_topic_count=len(material_topics),
            total_topics_assessed=len(topics),
            financed_impact_assessments=financed_impacts,
            datapoint_mappings=datapoint_mappings,
            total_required_datapoints=total_dps,
            materiality_matrix=matrix,
            material_standards=sorted(material_standards_set),
            standards_coverage=standards_coverage,
            stakeholder_alignment_score=_round_val(alignment, 2),
            processing_time_ms=_round_val(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Topic Assessment
    # ------------------------------------------------------------------

    def _assess_topic(
        self,
        topic: MaterialityTopicData,
        stakeholder_inputs: List[StakeholderInput],
    ) -> IROAssessment:
        """Assess a single materiality topic on both dimensions.

        Financial Materiality Score:
            = (likelihood_score / 100) * (magnitude / 100) * scope * 100

        Impact Materiality Score:
            base = (severity_score / 100) * (likelihood_score / 100) * scope
            irremediability_adj = base * (1 + irremediability_weight * irremediability)
            score = irremediability_adj * 100

        Overall Score:
            if use_max_rule: max(financial, impact)
            else: weighted average

        Args:
            topic: Topic input data.
            stakeholder_inputs: Stakeholder engagement data.

        Returns:
            IROAssessment with scores and materiality determination.
        """
        # Financial materiality
        fin_likelihood = LIKELIHOOD_SCORES.get(
            topic.financial_likelihood.value, 50.0,
        )
        fin_score = (
            (fin_likelihood / 100.0)
            * (topic.financial_magnitude / 100.0)
            * topic.financial_scope
            * 100.0
        )
        fin_score = _clamp(_round_val(fin_score, 2))

        # Impact materiality
        imp_severity = SEVERITY_SCORES.get(
            topic.impact_severity.value, 50.0,
        )
        imp_likelihood = LIKELIHOOD_SCORES.get(
            topic.impact_likelihood.value, 50.0,
        )
        base_impact = (
            (imp_severity / 100.0)
            * (imp_likelihood / 100.0)
            * topic.impact_scope
        )
        irrem_adj = base_impact * (
            1.0
            + self.config.irremediability_weight * topic.impact_irremediability
        )
        imp_score = _clamp(_round_val(irrem_adj * 100.0, 2))

        # Stakeholder adjustment (optional boost up to 10%)
        stakeholder_boost = self._compute_stakeholder_boost(
            topic.topic_id, stakeholder_inputs,
        )
        imp_score = _clamp(_round_val(imp_score + stakeholder_boost, 2))
        fin_score = _clamp(_round_val(fin_score + stakeholder_boost * 0.5, 2))

        # Overall score
        if self.config.use_max_rule:
            overall = max(fin_score, imp_score)
        else:
            overall = (
                self.config.financial_weight_in_overall * fin_score
                + self.config.impact_weight_in_overall * imp_score
            )
        overall = _clamp(_round_val(overall, 2))

        is_material = overall >= self.config.materiality_threshold

        result = IROAssessment(
            topic_id=topic.topic_id,
            topic_name=topic.topic_name,
            iro_type=topic.iro_type,
            esrs_standard=topic.esrs_standard,
            description=topic.description,
            severity=topic.impact_severity,
            likelihood=topic.financial_likelihood,
            financial_score=fin_score,
            impact_score=imp_score,
            overall_score=overall,
            is_material=is_material,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Stakeholder Analysis
    # ------------------------------------------------------------------

    def _compute_stakeholder_boost(
        self,
        topic_id: str,
        stakeholder_inputs: List[StakeholderInput],
    ) -> float:
        """Compute stakeholder relevance boost for a topic.

        Calculates a weighted-average stakeholder relevance score,
        then maps it to a 0-10 boost.

        Formula:
            raw = SUM(stakeholder_weight * topic_rating) / SUM(weights)
            boost = raw / 10 (max 10 points)

        Args:
            topic_id: Topic being assessed.
            stakeholder_inputs: Stakeholder engagement data.

        Returns:
            Boost value (0-10).
        """
        if not stakeholder_inputs:
            return 0.0

        weighted_sum = 0.0
        weight_total = 0.0

        for si in stakeholder_inputs:
            rating = si.topic_ratings.get(topic_id, 0.0)
            if rating <= 0.0:
                continue
            weight = (
                si.weight_override
                if si.weight_override is not None
                else self.config.stakeholder_weights.get(
                    si.stakeholder_group.value, 0.1,
                )
            )
            weighted_sum += weight * rating
            weight_total += weight

        if weight_total == 0.0:
            return 0.0

        raw = weighted_sum / weight_total  # 0-100
        boost = raw / 10.0  # 0-10
        return min(boost, 10.0)

    def _compute_stakeholder_alignment(
        self,
        iro_assessments: List[IROAssessment],
        stakeholder_inputs: List[StakeholderInput],
    ) -> float:
        """Compute alignment between stakeholder views and assessment outcome.

        Measures how well the final materiality determination aligns
        with stakeholder priorities. Higher = better alignment.

        Args:
            iro_assessments: Completed IRO assessments.
            stakeholder_inputs: Stakeholder engagement data.

        Returns:
            Alignment score (0-100).
        """
        if not stakeholder_inputs or not iro_assessments:
            return 0.0

        # Build average stakeholder rating per topic
        topic_ratings: Dict[str, List[float]] = defaultdict(list)
        for si in stakeholder_inputs:
            for tid, rating in si.topic_ratings.items():
                topic_ratings[tid].append(rating)

        avg_ratings: Dict[str, float] = {
            tid: sum(ratings) / len(ratings)
            for tid, ratings in topic_ratings.items()
            if ratings
        }

        if not avg_ratings:
            return 0.0

        # Compare: stakeholders say material (>50) vs assessment says material
        aligned_count = 0
        total_count = 0

        for iro in iro_assessments:
            if iro.topic_id in avg_ratings:
                total_count += 1
                stakeholder_says_material = avg_ratings[iro.topic_id] >= 50.0
                if stakeholder_says_material == iro.is_material:
                    aligned_count += 1

        return _safe_pct(aligned_count, total_count)

    # ------------------------------------------------------------------
    # Datapoint Mapping
    # ------------------------------------------------------------------

    def _map_datapoints(
        self,
        iro_assessments: List[IROAssessment],
    ) -> List[DatapointMapping]:
        """Map material topics to required ESRS datapoints.

        For each material standard, lists the required disclosure datapoints.

        Args:
            iro_assessments: Completed IRO assessments.

        Returns:
            List of DatapointMapping entries.
        """
        mappings: List[DatapointMapping] = []

        # Collect material standards and their topics
        standard_topics: Dict[str, List[IROAssessment]] = defaultdict(list)
        for iro in iro_assessments:
            standard_topics[iro.esrs_standard.value].append(iro)

        for std_value, topics_in_std in standard_topics.items():
            is_any_material = any(t.is_material for t in topics_in_std)
            datapoints = ESRS_DATAPOINT_REGISTRY.get(std_value, [])

            # Only require datapoints if the standard is material
            required = datapoints if is_any_material else []

            for topic in topics_in_std:
                mappings.append(DatapointMapping(
                    topic_id=topic.topic_id,
                    topic_name=topic.topic_name,
                    esrs_standard=topic.esrs_standard,
                    required_datapoints=required,
                    total_datapoints=len(datapoints),
                    is_material=topic.is_material,
                ))

        return mappings

    # ------------------------------------------------------------------
    # Materiality Matrix
    # ------------------------------------------------------------------

    def _build_materiality_matrix(
        self,
        iro_assessments: List[IROAssessment],
    ) -> MaterialityMatrix:
        """Build materiality matrix visualization data.

        Places each topic on a 2D matrix with financial materiality on
        the x-axis and impact materiality on the y-axis, with quadrant
        classification.

        Args:
            iro_assessments: Completed IRO assessments.

        Returns:
            MaterialityMatrix with topic coordinates.
        """
        threshold = self.config.materiality_threshold
        topics_data: List[Dict[str, Any]] = []
        quadrants: Dict[str, int] = {
            "Q1_both_material": 0,       # top-right
            "Q2_impact_only": 0,         # top-left
            "Q3_neither_material": 0,    # bottom-left
            "Q4_financial_only": 0,      # bottom-right
        }

        for iro in iro_assessments:
            fin_material = iro.financial_score >= threshold
            imp_material = iro.impact_score >= threshold

            if fin_material and imp_material:
                quadrants["Q1_both_material"] += 1
            elif imp_material:
                quadrants["Q2_impact_only"] += 1
            elif fin_material:
                quadrants["Q4_financial_only"] += 1
            else:
                quadrants["Q3_neither_material"] += 1

            topics_data.append({
                "topic_id": iro.topic_id,
                "topic_name": iro.topic_name,
                "x": iro.financial_score,
                "y": iro.impact_score,
                "overall_score": iro.overall_score,
                "material": iro.is_material,
                "esrs_standard": iro.esrs_standard.value,
            })

        matrix = MaterialityMatrix(
            topics=topics_data,
            threshold_x=threshold,
            threshold_y=threshold,
            quadrant_counts=quadrants,
        )
        matrix.provenance_hash = _compute_hash(matrix)
        return matrix

    # ------------------------------------------------------------------
    # FI-Specific Topic Generation
    # ------------------------------------------------------------------

    def generate_fi_specific_topics(self) -> List[MaterialityTopicData]:
        """Generate FI-specific materiality topics for assessment.

        Creates pre-populated MaterialityTopicData records for all
        FI-specific topics defined in FI_SPECIFIC_TOPICS.

        Returns:
            List of MaterialityTopicData with FI-specific defaults.
        """
        topics: List[MaterialityTopicData] = []

        for topic_key, meta in FI_SPECIFIC_TOPICS.items():
            # Map fi_relevance to default severity
            rel = meta.get("fi_relevance", "high")
            severity_map = {
                "critical": SeverityScale.CRITICAL,
                "high": SeverityScale.SIGNIFICANT,
                "medium": SeverityScale.MODERATE,
                "low": SeverityScale.MINOR,
            }
            default_severity = severity_map.get(rel, SeverityScale.MODERATE)

            # Map fi_relevance to default magnitude
            magnitude_map = {
                "critical": 80.0,
                "high": 65.0,
                "medium": 45.0,
                "low": 25.0,
            }
            default_magnitude = magnitude_map.get(rel, 50.0)

            esrs_std = ESRSStandard(meta["esrs_standard"])

            topics.append(MaterialityTopicData(
                topic_name=meta["name"],
                esrs_standard=esrs_std,
                is_fi_specific=True,
                financial_likelihood=LikelihoodScale.LIKELY,
                financial_magnitude=default_magnitude,
                financial_scope=0.6,
                impact_severity=default_severity,
                impact_likelihood=LikelihoodScale.LIKELY,
                impact_scope=0.7,
                impact_irremediability=0.4,
                description=meta["description"],
                iro_type=IROType.IMPACT,
            ))

        return topics

    # ------------------------------------------------------------------
    # Convenience Methods
    # ------------------------------------------------------------------

    def get_material_esrs_standards(
        self,
        result: FSMaterialityResult,
    ) -> List[str]:
        """Get list of material ESRS standards from a result.

        Args:
            result: Completed materiality assessment result.

        Returns:
            Sorted list of material ESRS standard identifiers.
        """
        return sorted(result.material_standards)

    def get_required_datapoints_count(
        self,
        result: FSMaterialityResult,
    ) -> Dict[str, int]:
        """Get count of required datapoints per ESRS standard.

        Args:
            result: Completed materiality assessment result.

        Returns:
            Dict mapping ESRS standard to required datapoint count.
        """
        counts: Dict[str, int] = defaultdict(int)
        seen: Dict[str, set] = defaultdict(set)

        for m in result.datapoint_mappings:
            if m.is_material:
                std = m.esrs_standard.value
                for dp in m.required_datapoints:
                    if dp not in seen[std]:
                        seen[std].add(dp)
                        counts[std] += 1

        return dict(counts)

    def assess_single_topic(
        self,
        topic: MaterialityTopicData,
    ) -> IROAssessment:
        """Assess a single topic without full portfolio context.

        Convenience method for ad-hoc topic assessment.

        Args:
            topic: Topic to assess.

        Returns:
            IROAssessment for the topic.
        """
        return self._assess_topic(topic, [])
