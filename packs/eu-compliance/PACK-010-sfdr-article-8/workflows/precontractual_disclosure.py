# -*- coding: utf-8 -*-
"""
Pre-contractual Disclosure Workflow
=======================================

Five-phase workflow for generating SFDR Annex II pre-contractual disclosures
for Article 8 financial products. Orchestrates product classification,
investment strategy definition, sustainability assessment, template population,
and review/approval into a single auditable pipeline.

Regulatory Context:
    Per EU SFDR Regulation 2019/2088 and Delegated Regulation 2022/1288 (RTS):
    - Article 8: Products promoting environmental or social characteristics.
    - Article 6: Pre-contractual disclosure requirements for all products.
    - Annex II: Template for Article 8 pre-contractual disclosures.
    - 12 mandatory sections per RTS covering investment strategy, asset
      allocation, sustainability indicators, DNSH methodology, data sources,
      engagement policies, and designated reference benchmark.
    - Taxonomy Regulation 2020/852 alignment for sustainable investments.

    Key Disclosure Elements:
    - Environmental/social characteristics promoted
    - Binding elements of the investment strategy
    - Asset allocation (minimum proportions: sustainable investments,
      taxonomy-aligned, other E/S characteristics)
    - DNSH assessment methodology
    - Good governance assessment approach
    - Reference benchmark designation (if applicable)
    - Data sources and processing methodology
    - Limitations and due diligence procedures

Phases:
    1. ProductClassification - Verify Article 8 eligibility, determine
       sustainable investment scope, set SFDR classification level
    2. InvestmentStrategy - Define binding elements (exclusions, ESG
       thresholds, engagement), asset allocation targets, derivatives policy
    3. SustainabilityAssessment - Calculate taxonomy alignment commitment,
       sustainable investment minimum, DNSH methodology, good governance
    4. TemplatePopulation - Generate Annex II template with all 12 required
       sections populated with computed data
    5. ReviewApproval - Compliance completeness check, legal review status,
       version tracking, sign-off workflow

Author: GreenLang Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# UTILITIES
# =============================================================================

def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"

class SFDRClassification(str, Enum):
    """SFDR product classification level."""
    ARTICLE_6 = "ARTICLE_6"
    ARTICLE_8 = "ARTICLE_8"
    ARTICLE_8_PLUS = "ARTICLE_8_PLUS"
    ARTICLE_9 = "ARTICLE_9"

class CharacteristicType(str, Enum):
    """Environmental or social characteristic type."""
    ENVIRONMENTAL = "ENVIRONMENTAL"
    SOCIAL = "SOCIAL"
    GOVERNANCE = "GOVERNANCE"

class ReviewStatus(str, Enum):
    """Document review workflow status."""
    DRAFT = "DRAFT"
    UNDER_REVIEW = "UNDER_REVIEW"
    LEGAL_APPROVED = "LEGAL_APPROVED"
    COMPLIANCE_APPROVED = "COMPLIANCE_APPROVED"
    FINAL_APPROVED = "FINAL_APPROVED"
    PUBLISHED = "PUBLISHED"

# =============================================================================
# DATA MODELS - SHARED
# =============================================================================

class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        """Store phase outputs for downstream consumption."""
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        """Retrieve outputs from a previous phase."""
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Record phase status for checkpoint/resume."""
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase has already completed."""
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)

class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")

# =============================================================================
# DATA MODELS - PRE-CONTRACTUAL DISCLOSURE
# =============================================================================

class ESCharacteristic(BaseModel):
    """Environmental or social characteristic promoted by the product."""
    characteristic_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Characteristic name")
    characteristic_type: CharacteristicType = Field(
        ..., description="Environmental or social"
    )
    description: str = Field(default="", description="Detailed description")
    sustainability_indicators: List[str] = Field(
        default_factory=list,
        description="KPIs used to measure attainment"
    )
    binding: bool = Field(
        default=True, description="Whether this is a binding element"
    )

class ExclusionCriteria(BaseModel):
    """Negative screening exclusion criterion."""
    criterion_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Exclusion criterion name")
    description: str = Field(default="")
    threshold_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Revenue threshold percentage for exclusion"
    )
    applies_to: str = Field(
        default="all", description="Scope: all, direct, indirect"
    )
    binding: bool = Field(default=True)

class AssetAllocationTarget(BaseModel):
    """Target asset allocation proportions."""
    minimum_sustainable_investment_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Minimum % in sustainable investments"
    )
    minimum_taxonomy_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Minimum % taxonomy-aligned investments"
    )
    minimum_environmentally_sustainable_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Minimum % environmentally sustainable"
    )
    minimum_socially_sustainable_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Minimum % socially sustainable"
    )
    other_es_characteristics_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="% promoting E/S characteristics but not sustainable investments"
    )
    remaining_not_sustainable_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="% not aligned with E/S characteristics"
    )

class PrecontractualDisclosureInput(BaseModel):
    """Input configuration for the pre-contractual disclosure workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    product_isin: Optional[str] = Field(None, description="ISIN if applicable")
    product_lei: Optional[str] = Field(None, description="LEI of product manager")
    reporting_date: str = Field(
        ..., description="Disclosure date in YYYY-MM-DD format"
    )
    sfdr_classification: SFDRClassification = Field(
        default=SFDRClassification.ARTICLE_8,
        description="Target SFDR classification"
    )
    makes_sustainable_investments: bool = Field(
        default=False,
        description="Whether the product makes sustainable investments (Art 8+)"
    )
    characteristics: List[ESCharacteristic] = Field(
        default_factory=list,
        description="E/S characteristics promoted"
    )
    exclusion_criteria: List[ExclusionCriteria] = Field(
        default_factory=list,
        description="Negative screening criteria"
    )
    asset_allocation_targets: Optional[AssetAllocationTarget] = Field(
        None, description="Target asset allocation"
    )
    esg_rating_threshold: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Minimum ESG rating for investees"
    )
    engagement_commitment: bool = Field(
        default=False, description="Whether product commits to engagement"
    )
    engagement_description: str = Field(
        default="", description="Engagement policy description"
    )
    derivatives_policy: str = Field(
        default="not_used",
        description="Role of derivatives: not_used, hedging, es_attainment"
    )
    reference_benchmark: Optional[str] = Field(
        None, description="Designated reference benchmark if any"
    )
    benchmark_aligned_eu_climate: bool = Field(
        default=False, description="Whether benchmark is EU Climate Benchmark"
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Primary data sources"
    )
    data_processing_description: str = Field(
        default="", description="Data processing methodology"
    )
    limitations_description: str = Field(
        default="", description="Limitations of methodology"
    )
    due_diligence_description: str = Field(
        default="", description="Due diligence procedures"
    )
    previous_version_id: Optional[str] = Field(
        None, description="ID of previous disclosure version"
    )
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate reporting date is valid ISO format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("reporting_date must be YYYY-MM-DD format")
        return v

class PrecontractualDisclosureResult(WorkflowResult):
    """Complete result from the pre-contractual disclosure workflow."""
    product_name: str = Field(default="")
    sfdr_classification: str = Field(default="ARTICLE_8")
    is_article_8_eligible: bool = Field(default=False)
    makes_sustainable_investments: bool = Field(default=False)
    taxonomy_alignment_commitment_pct: float = Field(default=0.0)
    sustainable_investment_minimum_pct: float = Field(default=0.0)
    binding_elements_count: int = Field(default=0)
    template_sections_completed: int = Field(default=0)
    template_sections_total: int = Field(default=12)
    review_status: str = Field(default="DRAFT")
    completeness_pct: float = Field(default=0.0)

# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================

class ProductClassificationPhase:
    """
    Phase 1: Product Classification.

    Verifies Article 8 eligibility by checking that the product promotes
    environmental or social characteristics, determines whether it makes
    sustainable investments (Article 8+), and sets the SFDR classification.
    """

    PHASE_NAME = "product_classification"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute product classification phase.

        Args:
            context: Workflow context with product configuration.

        Returns:
            PhaseResult with classification determination.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            product_name = config.get("product_name", "")
            classification = config.get("sfdr_classification", "ARTICLE_8")
            makes_si = config.get("makes_sustainable_investments", False)
            characteristics = config.get("characteristics", [])
            exclusions = config.get("exclusion_criteria", [])

            outputs["product_name"] = product_name
            outputs["product_isin"] = config.get("product_isin", "")
            outputs["product_lei"] = config.get("product_lei", "")

            # Check Article 8 eligibility requirements
            eligibility_checks = []

            # Requirement 1: Must promote at least one E/S characteristic
            has_es_characteristics = len(characteristics) > 0
            eligibility_checks.append({
                "check": "es_characteristics_defined",
                "passed": has_es_characteristics,
                "detail": f"{len(characteristics)} characteristic(s) defined"
            })

            # Requirement 2: Must have binding elements
            binding_characteristics = [
                c for c in characteristics
                if c.get("binding", True)
            ]
            binding_exclusions = [
                e for e in exclusions if e.get("binding", True)
            ]
            has_binding = len(binding_characteristics) + len(binding_exclusions) > 0
            eligibility_checks.append({
                "check": "binding_elements_defined",
                "passed": has_binding,
                "detail": (
                    f"{len(binding_characteristics)} binding characteristic(s), "
                    f"{len(binding_exclusions)} binding exclusion(s)"
                )
            })

            # Requirement 3: Must describe how characteristics are met
            has_indicators = any(
                len(c.get("sustainability_indicators", [])) > 0
                for c in characteristics
            ) if characteristics else False
            eligibility_checks.append({
                "check": "sustainability_indicators_defined",
                "passed": has_indicators,
                "detail": "Sustainability indicators present" if has_indicators
                else "No sustainability indicators defined"
            })

            # Determine eligibility
            is_eligible = all(c["passed"] for c in eligibility_checks)
            outputs["eligibility_checks"] = eligibility_checks
            outputs["is_article_8_eligible"] = is_eligible

            if not is_eligible:
                failed_checks = [
                    c["check"] for c in eligibility_checks if not c["passed"]
                ]
                warnings.append(
                    f"Product may not meet Article 8 requirements: "
                    f"failed checks: {', '.join(failed_checks)}"
                )

            # Determine classification level
            if makes_si:
                effective_classification = SFDRClassification.ARTICLE_8_PLUS.value
            else:
                effective_classification = SFDRClassification.ARTICLE_8.value
            outputs["effective_classification"] = effective_classification
            outputs["makes_sustainable_investments"] = makes_si

            # Categorize characteristics
            env_chars = [
                c for c in characteristics
                if c.get("characteristic_type") == CharacteristicType.ENVIRONMENTAL.value
            ]
            social_chars = [
                c for c in characteristics
                if c.get("characteristic_type") == CharacteristicType.SOCIAL.value
            ]
            outputs["environmental_characteristics_count"] = len(env_chars)
            outputs["social_characteristics_count"] = len(social_chars)
            outputs["total_characteristics_count"] = len(characteristics)
            outputs["binding_elements_count"] = (
                len(binding_characteristics) + len(binding_exclusions)
            )
            outputs["exclusion_criteria_count"] = len(exclusions)

            status = PhaseStatus.COMPLETED
            records = len(characteristics) + len(exclusions)

        except Exception as exc:
            logger.error("ProductClassification failed: %s", exc, exc_info=True)
            errors.append(f"Product classification failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

class InvestmentStrategyPhase:
    """
    Phase 2: Investment Strategy.

    Defines the binding elements of the investment strategy including
    exclusion criteria, ESG thresholds, engagement commitments, asset
    allocation targets, and the derivatives policy.
    """

    PHASE_NAME = "investment_strategy"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute investment strategy phase.

        Args:
            context: Workflow context with strategy configuration.

        Returns:
            PhaseResult with strategy definition and allocation targets.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            classification_output = context.get_phase_output(
                "product_classification"
            )
            makes_si = classification_output.get(
                "makes_sustainable_investments", False
            )

            # Binding elements compilation
            binding_elements = []

            # Exclusion criteria as binding elements
            exclusions = config.get("exclusion_criteria", [])
            for exc_item in exclusions:
                if exc_item.get("binding", True):
                    binding_elements.append({
                        "type": "exclusion",
                        "name": exc_item.get("name", ""),
                        "description": exc_item.get("description", ""),
                        "threshold_pct": exc_item.get("threshold_pct"),
                        "applies_to": exc_item.get("applies_to", "all"),
                    })

            # ESG rating threshold as binding element
            esg_threshold = config.get("esg_rating_threshold")
            if esg_threshold is not None:
                binding_elements.append({
                    "type": "esg_threshold",
                    "name": "Minimum ESG Rating",
                    "description": (
                        f"Investees must maintain minimum ESG rating of "
                        f"{esg_threshold}"
                    ),
                    "threshold_value": esg_threshold,
                })

            # Engagement commitment as binding element
            if config.get("engagement_commitment", False):
                binding_elements.append({
                    "type": "engagement",
                    "name": "Active Engagement Commitment",
                    "description": config.get("engagement_description", ""),
                })

            outputs["binding_elements"] = binding_elements
            outputs["binding_elements_count"] = len(binding_elements)

            # Asset allocation targets
            allocation_config = config.get("asset_allocation_targets")
            if allocation_config:
                allocation = {
                    "minimum_sustainable_investment_pct": allocation_config.get(
                        "minimum_sustainable_investment_pct", 0.0
                    ),
                    "minimum_taxonomy_aligned_pct": allocation_config.get(
                        "minimum_taxonomy_aligned_pct", 0.0
                    ),
                    "minimum_environmentally_sustainable_pct": allocation_config.get(
                        "minimum_environmentally_sustainable_pct", 0.0
                    ),
                    "minimum_socially_sustainable_pct": allocation_config.get(
                        "minimum_socially_sustainable_pct", 0.0
                    ),
                    "other_es_characteristics_pct": allocation_config.get(
                        "other_es_characteristics_pct", 0.0
                    ),
                    "remaining_not_sustainable_pct": allocation_config.get(
                        "remaining_not_sustainable_pct", 0.0
                    ),
                }
            else:
                allocation = {
                    "minimum_sustainable_investment_pct": 0.0,
                    "minimum_taxonomy_aligned_pct": 0.0,
                    "minimum_environmentally_sustainable_pct": 0.0,
                    "minimum_socially_sustainable_pct": 0.0,
                    "other_es_characteristics_pct": 100.0,
                    "remaining_not_sustainable_pct": 0.0,
                }

            # Validate allocation totals
            total_allocation = (
                allocation["minimum_sustainable_investment_pct"]
                + allocation["other_es_characteristics_pct"]
                + allocation["remaining_not_sustainable_pct"]
            )
            if total_allocation > 0 and abs(total_allocation - 100.0) > 0.1:
                warnings.append(
                    f"Asset allocation does not sum to 100%: "
                    f"total={total_allocation:.1f}%"
                )

            # Article 8+ validation
            if makes_si and allocation["minimum_sustainable_investment_pct"] <= 0:
                warnings.append(
                    "Product classified as Article 8+ but minimum sustainable "
                    "investment percentage is 0%"
                )

            outputs["asset_allocation"] = allocation

            # Derivatives policy
            derivatives_policy = config.get("derivatives_policy", "not_used")
            outputs["derivatives_policy"] = derivatives_policy
            outputs["derivatives_for_es_attainment"] = (
                derivatives_policy == "es_attainment"
            )

            if derivatives_policy == "es_attainment":
                outputs["derivatives_disclosure_required"] = True
                outputs["derivatives_description"] = (
                    "Derivatives are used to attain the environmental or "
                    "social characteristics promoted by the product."
                )
            elif derivatives_policy == "hedging":
                outputs["derivatives_disclosure_required"] = True
                outputs["derivatives_description"] = (
                    "Derivatives are used for hedging purposes only and do "
                    "not contribute to the attainment of E/S characteristics."
                )
            else:
                outputs["derivatives_disclosure_required"] = False
                outputs["derivatives_description"] = (
                    "The product does not use derivatives."
                )

            # Reference benchmark
            benchmark = config.get("reference_benchmark")
            outputs["has_reference_benchmark"] = benchmark is not None
            outputs["reference_benchmark"] = benchmark or ""
            outputs["benchmark_aligned_eu_climate"] = config.get(
                "benchmark_aligned_eu_climate", False
            )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("InvestmentStrategy failed: %s", exc, exc_info=True)
            errors.append(f"Investment strategy definition failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

class SustainabilityAssessmentPhase:
    """
    Phase 3: Sustainability Assessment.

    Calculates taxonomy alignment commitment percentage, sustainable
    investment minimum, describes the DNSH methodology, and defines
    the good governance assessment approach.
    """

    PHASE_NAME = "sustainability_assessment"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute sustainability assessment phase.

        Args:
            context: Workflow context with strategy and classification data.

        Returns:
            PhaseResult with sustainability metrics and DNSH approach.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            classification_output = context.get_phase_output(
                "product_classification"
            )
            strategy_output = context.get_phase_output("investment_strategy")
            allocation = strategy_output.get("asset_allocation", {})
            makes_si = classification_output.get(
                "makes_sustainable_investments", False
            )

            # Taxonomy alignment commitment
            taxonomy_pct = allocation.get("minimum_taxonomy_aligned_pct", 0.0)
            outputs["taxonomy_alignment_commitment_pct"] = taxonomy_pct
            outputs["taxonomy_alignment_basis"] = "revenue"

            if taxonomy_pct > 0:
                outputs["taxonomy_alignment_description"] = (
                    f"The product commits to a minimum of {taxonomy_pct}% "
                    f"of investments in taxonomy-aligned economic activities, "
                    f"measured on a revenue basis."
                )
                outputs["taxonomy_environmental_objectives"] = [
                    "climate_change_mitigation",
                    "climate_change_adaptation",
                ]
            else:
                outputs["taxonomy_alignment_description"] = (
                    "The product does not commit to a minimum proportion of "
                    "taxonomy-aligned investments. The investments underlying "
                    "this financial product do not take into account the EU "
                    "criteria for environmentally sustainable economic activities."
                )
                outputs["taxonomy_environmental_objectives"] = []

            # Sustainable investment minimum
            si_pct = allocation.get("minimum_sustainable_investment_pct", 0.0)
            outputs["sustainable_investment_minimum_pct"] = si_pct

            if makes_si and si_pct > 0:
                env_si_pct = allocation.get(
                    "minimum_environmentally_sustainable_pct", 0.0
                )
                social_si_pct = allocation.get(
                    "minimum_socially_sustainable_pct", 0.0
                )
                outputs["sustainable_investment_breakdown"] = {
                    "total_pct": si_pct,
                    "environmental_pct": env_si_pct,
                    "social_pct": social_si_pct,
                    "taxonomy_aligned_pct": taxonomy_pct,
                }
            else:
                outputs["sustainable_investment_breakdown"] = {
                    "total_pct": 0.0,
                    "environmental_pct": 0.0,
                    "social_pct": 0.0,
                    "taxonomy_aligned_pct": 0.0,
                }

            # DNSH methodology
            dnsh_methodology = self._build_dnsh_methodology(makes_si)
            outputs["dnsh_methodology"] = dnsh_methodology

            # Good governance assessment
            governance_approach = self._build_governance_approach()
            outputs["good_governance_approach"] = governance_approach

            # Data sources and processing
            data_sources = config.get("data_sources", [])
            outputs["data_sources"] = data_sources
            outputs["data_sources_count"] = len(data_sources)
            outputs["data_processing_description"] = config.get(
                "data_processing_description", ""
            )

            # Limitations
            outputs["limitations_description"] = config.get(
                "limitations_description", ""
            )
            outputs["due_diligence_description"] = config.get(
                "due_diligence_description", ""
            )

            # Engagement policy
            outputs["engagement_policy"] = {
                "has_engagement": config.get("engagement_commitment", False),
                "description": config.get("engagement_description", ""),
            }

            if not data_sources:
                warnings.append(
                    "No data sources specified. Data sources disclosure "
                    "is mandatory under Annex II."
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "SustainabilityAssessment failed: %s", exc, exc_info=True
            )
            errors.append(f"Sustainability assessment failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

    def _build_dnsh_methodology(self, makes_si: bool) -> Dict[str, Any]:
        """Build DNSH methodology description for sustainable investments."""
        if not makes_si:
            return {
                "applicable": False,
                "description": (
                    "DNSH assessment is not applicable as this product does "
                    "not make sustainable investments."
                ),
                "indicators": [],
            }

        return {
            "applicable": True,
            "description": (
                "Sustainable investments are assessed to ensure they do not "
                "significantly harm any of the environmental or social "
                "objectives set out in the Taxonomy Regulation."
            ),
            "indicators": [
                {
                    "objective": "Climate change mitigation",
                    "assessment": "GHG emissions intensity analysis",
                },
                {
                    "objective": "Climate change adaptation",
                    "assessment": "Physical risk exposure assessment",
                },
                {
                    "objective": "Water and marine resources",
                    "assessment": "Water stress and pollution metrics",
                },
                {
                    "objective": "Circular economy",
                    "assessment": "Waste management and recycling rates",
                },
                {
                    "objective": "Pollution prevention",
                    "assessment": "Pollutant emission monitoring",
                },
                {
                    "objective": "Biodiversity and ecosystems",
                    "assessment": "Biodiversity impact assessment",
                },
            ],
            "social_safeguards": [
                "OECD Guidelines for Multinational Enterprises",
                "UN Guiding Principles on Business and Human Rights",
                "ILO Core Labour Conventions",
                "International Bill of Human Rights",
            ],
        }

    def _build_governance_approach(self) -> Dict[str, Any]:
        """Build good governance assessment approach."""
        return {
            "description": (
                "Good governance practices of investee companies are assessed "
                "through evaluation of sound management structures, employee "
                "relations, remuneration of staff, and tax compliance."
            ),
            "assessment_areas": [
                {
                    "area": "Management structures",
                    "criteria": "Board independence, oversight effectiveness",
                },
                {
                    "area": "Employee relations",
                    "criteria": "Labour practices, health and safety",
                },
                {
                    "area": "Remuneration of staff",
                    "criteria": "Pay equity, performance alignment",
                },
                {
                    "area": "Tax compliance",
                    "criteria": "Tax transparency, no aggressive avoidance",
                },
            ],
            "minimum_standards": [
                "UN Global Compact principles adherence",
                "No severe controversies (UNGC violations)",
            ],
        }

class TemplatePopulationPhase:
    """
    Phase 4: Template Population.

    Generates the complete Annex II template with all 12 required sections
    populated from computed data in previous phases.
    """

    PHASE_NAME = "template_population"

    ANNEX_II_SECTIONS = [
        "summary",
        "no_sustainable_investment_objective",
        "environmental_social_characteristics",
        "investment_strategy",
        "proportion_of_investments",
        "monitoring_of_characteristics",
        "methodologies",
        "data_sources_and_processing",
        "limitations_to_methodologies",
        "due_diligence",
        "engagement_policies",
        "designated_reference_benchmark",
    ]

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute template population phase.

        Args:
            context: Workflow context with all prior phase outputs.

        Returns:
            PhaseResult with populated Annex II template.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            classification_output = context.get_phase_output(
                "product_classification"
            )
            strategy_output = context.get_phase_output("investment_strategy")
            assessment_output = context.get_phase_output(
                "sustainability_assessment"
            )

            product_name = classification_output.get("product_name", "")
            makes_si = classification_output.get(
                "makes_sustainable_investments", False
            )
            allocation = strategy_output.get("asset_allocation", {})

            # Build template sections
            sections: Dict[str, Any] = {}
            completed_sections = 0

            # Section 1: Summary
            sections["summary"] = self._build_summary_section(
                product_name, classification_output, strategy_output,
                assessment_output
            )
            completed_sections += 1

            # Section 2: No sustainable investment objective disclaimer
            sections["no_sustainable_investment_objective"] = (
                self._build_no_si_objective_section(makes_si)
            )
            completed_sections += 1

            # Section 3: Environmental/social characteristics
            sections["environmental_social_characteristics"] = (
                self._build_es_characteristics_section(
                    config.get("characteristics", [])
                )
            )
            completed_sections += 1

            # Section 4: Investment strategy
            sections["investment_strategy"] = (
                self._build_investment_strategy_section(
                    strategy_output, assessment_output
                )
            )
            completed_sections += 1

            # Section 5: Proportion of investments
            sections["proportion_of_investments"] = (
                self._build_proportion_section(allocation, makes_si)
            )
            completed_sections += 1

            # Section 6: Monitoring of E/S characteristics
            indicators = []
            for char in config.get("characteristics", []):
                indicators.extend(
                    char.get("sustainability_indicators", [])
                )
            sections["monitoring_of_characteristics"] = {
                "description": (
                    "The attainment of E/S characteristics is monitored "
                    "through sustainability indicators measured on an "
                    "ongoing basis."
                ),
                "sustainability_indicators": indicators,
                "monitoring_frequency": "quarterly",
            }
            completed_sections += 1

            # Section 7: Methodologies
            sections["methodologies"] = {
                "description": (
                    "The following methodologies are used to measure "
                    "the attainment of E/S characteristics."
                ),
                "measurement_approach": "quantitative_and_qualitative",
                "data_quality_framework": "tiered",
            }
            completed_sections += 1

            # Section 8: Data sources and processing
            sections["data_sources_and_processing"] = {
                "data_sources": assessment_output.get("data_sources", []),
                "processing_description": assessment_output.get(
                    "data_processing_description", ""
                ),
                "data_quality_measures": [
                    "Automated validation checks",
                    "Cross-referencing multiple data providers",
                    "Regular data completeness audits",
                ],
                "estimated_data_proportion": "Estimated data does not "
                "exceed 30% of total data used.",
            }
            completed_sections += 1

            # Section 9: Limitations to methodologies
            limitations = assessment_output.get(
                "limitations_description", ""
            )
            sections["limitations_to_methodologies"] = {
                "description": limitations if limitations else (
                    "Limitations may arise from data availability gaps, "
                    "differences in reporting standards across jurisdictions, "
                    "and estimation methodologies applied where reported "
                    "data is unavailable."
                ),
                "mitigation_measures": [
                    "Use of multiple data sources for cross-validation",
                    "Conservative estimation approaches",
                    "Regular methodology reviews and updates",
                ],
            }
            completed_sections += 1

            # Section 10: Due diligence
            dd = assessment_output.get("due_diligence_description", "")
            sections["due_diligence"] = {
                "description": dd if dd else (
                    "Due diligence is carried out on the underlying assets "
                    "of the financial product through ESG integration in "
                    "the investment decision-making process."
                ),
                "processes": [
                    "Pre-investment ESG screening",
                    "Ongoing monitoring of ESG performance",
                    "Incident and controversy monitoring",
                    "Regular portfolio review against binding elements",
                ],
            }
            completed_sections += 1

            # Section 11: Engagement policies
            engagement = assessment_output.get("engagement_policy", {})
            sections["engagement_policies"] = {
                "has_engagement": engagement.get("has_engagement", False),
                "description": engagement.get("description", ""),
                "engagement_types": [
                    "Direct dialogue with investee companies",
                    "Proxy voting",
                    "Collaborative engagement initiatives",
                ] if engagement.get("has_engagement", False) else [],
            }
            completed_sections += 1

            # Section 12: Designated reference benchmark
            sections["designated_reference_benchmark"] = {
                "has_benchmark": strategy_output.get(
                    "has_reference_benchmark", False
                ),
                "benchmark_name": strategy_output.get(
                    "reference_benchmark", ""
                ),
                "eu_climate_benchmark": strategy_output.get(
                    "benchmark_aligned_eu_climate", False
                ),
                "benchmark_methodology": (
                    "The reference benchmark is aligned with the E/S "
                    "characteristics promoted by this product."
                ) if strategy_output.get("has_reference_benchmark", False)
                else "No reference benchmark has been designated.",
            }
            completed_sections += 1

            outputs["template_sections"] = sections
            outputs["sections_completed"] = completed_sections
            outputs["sections_total"] = len(self.ANNEX_II_SECTIONS)
            outputs["completeness_pct"] = round(
                completed_sections / len(self.ANNEX_II_SECTIONS) * 100, 1
            )
            outputs["template_version"] = "1.0"
            outputs["template_format"] = "structured_json"
            outputs["generated_at"] = utcnow().isoformat()

            # Check for incomplete sections
            incomplete = []
            for section_name in self.ANNEX_II_SECTIONS:
                section_data = sections.get(section_name, {})
                if not section_data:
                    incomplete.append(section_name)
            if incomplete:
                warnings.append(
                    f"Incomplete template sections: {', '.join(incomplete)}"
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("TemplatePopulation failed: %s", exc, exc_info=True)
            errors.append(f"Template population failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

    def _build_summary_section(
        self,
        product_name: str,
        classification: Dict[str, Any],
        strategy: Dict[str, Any],
        assessment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the summary section of Annex II."""
        return {
            "product_name": product_name,
            "classification": classification.get(
                "effective_classification", "ARTICLE_8"
            ),
            "characteristics_summary": (
                f"This product promotes {classification.get('environmental_characteristics_count', 0)} "
                f"environmental and {classification.get('social_characteristics_count', 0)} "
                f"social characteristic(s)."
            ),
            "sustainable_investment_commitment": assessment.get(
                "taxonomy_alignment_description", ""
            ),
            "binding_elements_summary": (
                f"{strategy.get('binding_elements_count', 0)} binding "
                f"elements govern the investment strategy."
            ),
        }

    def _build_no_si_objective_section(
        self, makes_si: bool
    ) -> Dict[str, Any]:
        """Build the 'no sustainable investment objective' section."""
        if makes_si:
            return {
                "applicable": False,
                "description": (
                    "This product makes sustainable investments. The "
                    "sustainable investments are subject to the DNSH "
                    "principle and good governance assessment."
                ),
            }
        return {
            "applicable": True,
            "description": (
                "This financial product promotes environmental or social "
                "characteristics, but does not have as its objective "
                "sustainable investment. It will have a minimum proportion "
                "of investments aligned with E/S characteristics."
            ),
        }

    def _build_es_characteristics_section(
        self, characteristics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build the E/S characteristics section."""
        return {
            "characteristics": [
                {
                    "name": c.get("name", ""),
                    "type": c.get("characteristic_type", ""),
                    "description": c.get("description", ""),
                    "indicators": c.get("sustainability_indicators", []),
                    "binding": c.get("binding", True),
                }
                for c in characteristics
            ],
            "total_count": len(characteristics),
        }

    def _build_investment_strategy_section(
        self,
        strategy: Dict[str, Any],
        assessment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the investment strategy section."""
        return {
            "binding_elements": strategy.get("binding_elements", []),
            "good_governance": assessment.get("good_governance_approach", {}),
            "derivatives_policy": strategy.get("derivatives_description", ""),
        }

    def _build_proportion_section(
        self,
        allocation: Dict[str, Any],
        makes_si: bool,
    ) -> Dict[str, Any]:
        """Build the proportion of investments section."""
        return {
            "asset_allocation": allocation,
            "makes_sustainable_investments": makes_si,
            "taxonomy_alignment_diagram": {
                "taxonomy_aligned_pct": allocation.get(
                    "minimum_taxonomy_aligned_pct", 0.0
                ),
                "other_environmental_pct": allocation.get(
                    "minimum_environmentally_sustainable_pct", 0.0
                ) - allocation.get("minimum_taxonomy_aligned_pct", 0.0),
                "social_pct": allocation.get(
                    "minimum_socially_sustainable_pct", 0.0
                ),
                "other_es_pct": allocation.get(
                    "other_es_characteristics_pct", 0.0
                ),
                "not_sustainable_pct": allocation.get(
                    "remaining_not_sustainable_pct", 0.0
                ),
            },
        }

class ReviewApprovalPhase:
    """
    Phase 5: Review and Approval.

    Performs compliance completeness check, tracks legal review status,
    manages version control, and orchestrates the sign-off workflow.
    """

    PHASE_NAME = "review_approval"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute review and approval phase.

        Args:
            context: Workflow context with template and prior outputs.

        Returns:
            PhaseResult with completeness assessment and review status.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            template_output = context.get_phase_output("template_population")
            classification_output = context.get_phase_output(
                "product_classification"
            )

            # Completeness assessment
            sections_completed = template_output.get("sections_completed", 0)
            sections_total = template_output.get("sections_total", 12)
            completeness_pct = template_output.get("completeness_pct", 0.0)

            completeness_checks = []

            # Check each mandatory section
            template_sections = template_output.get("template_sections", {})
            for section_name in TemplatePopulationPhase.ANNEX_II_SECTIONS:
                section = template_sections.get(section_name, {})
                is_populated = bool(section)
                completeness_checks.append({
                    "section": section_name,
                    "populated": is_populated,
                    "status": "complete" if is_populated else "missing",
                })

            outputs["completeness_checks"] = completeness_checks
            outputs["completeness_pct"] = completeness_pct
            outputs["sections_completed"] = sections_completed
            outputs["sections_total"] = sections_total

            # Compliance validation
            compliance_issues = []

            # Check: Article 8 eligibility confirmed
            if not classification_output.get("is_article_8_eligible", False):
                compliance_issues.append({
                    "severity": "critical",
                    "issue": "Product does not meet Article 8 eligibility criteria",
                    "remediation": "Review and address failing eligibility checks",
                })

            # Check: Binding elements defined
            if classification_output.get("binding_elements_count", 0) == 0:
                compliance_issues.append({
                    "severity": "critical",
                    "issue": "No binding elements defined",
                    "remediation": (
                        "Define at least one binding element (exclusion, "
                        "ESG threshold, or engagement commitment)"
                    ),
                })

            # Check: Data sources disclosed
            assessment_output = context.get_phase_output(
                "sustainability_assessment"
            )
            if assessment_output.get("data_sources_count", 0) == 0:
                compliance_issues.append({
                    "severity": "high",
                    "issue": "No data sources disclosed",
                    "remediation": "Specify data sources used for ESG assessment",
                })

            outputs["compliance_issues"] = compliance_issues
            outputs["compliance_issues_count"] = len(compliance_issues)
            outputs["has_critical_issues"] = any(
                i["severity"] == "critical" for i in compliance_issues
            )

            if compliance_issues:
                warnings.append(
                    f"{len(compliance_issues)} compliance issue(s) identified"
                )

            # Version tracking
            previous_version = config.get("previous_version_id")
            version_info = {
                "version_id": str(uuid.uuid4()),
                "version_number": "1.0" if not previous_version else "1.1",
                "previous_version_id": previous_version,
                "created_at": utcnow().isoformat(),
                "created_by": "system",
                "change_summary": (
                    "Initial pre-contractual disclosure" if not previous_version
                    else "Updated pre-contractual disclosure"
                ),
            }
            outputs["version_info"] = version_info

            # Review workflow status
            if outputs["has_critical_issues"]:
                review_status = ReviewStatus.DRAFT.value
            elif completeness_pct >= 100.0 and len(compliance_issues) == 0:
                review_status = ReviewStatus.UNDER_REVIEW.value
            else:
                review_status = ReviewStatus.DRAFT.value

            outputs["review_status"] = review_status
            outputs["sign_off_workflow"] = {
                "status": review_status,
                "compliance_officer": {"approved": False, "timestamp": None},
                "legal_counsel": {"approved": False, "timestamp": None},
                "product_manager": {"approved": False, "timestamp": None},
                "senior_management": {"approved": False, "timestamp": None},
            }

            # Disclosure metadata
            outputs["disclosure_metadata"] = {
                "product_name": classification_output.get("product_name", ""),
                "product_isin": classification_output.get("product_isin", ""),
                "reporting_date": config.get("reporting_date", ""),
                "classification": classification_output.get(
                    "effective_classification", ""
                ),
                "language": "en",
                "format": "structured_json",
            }

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("ReviewApproval failed: %s", exc, exc_info=True)
            errors.append(f"Review and approval failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================

class PrecontractualDisclosureWorkflow:
    """
    Five-phase pre-contractual disclosure workflow for SFDR Article 8.

    Orchestrates the complete Annex II disclosure generation from product
    classification through template population and review approval.
    Supports checkpoint/resume via phase-level status tracking and
    phase skipping via configuration.

    Attributes:
        workflow_id: Unique execution identifier.
        _phases: Ordered mapping of phase name to executor instance.
        _progress_callback: Optional progress notification callback.

    Example:
        >>> wf = PrecontractualDisclosureWorkflow()
        >>> input_data = PrecontractualDisclosureInput(
        ...     organization_id="org-123",
        ...     product_name="Green Bond Fund",
        ...     reporting_date="2026-01-01",
        ...     characteristics=[
        ...         ESCharacteristic(
        ...             name="Carbon Reduction",
        ...             characteristic_type=CharacteristicType.ENVIRONMENTAL,
        ...         )
        ...     ],
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "precontractual_disclosure"

    PHASE_ORDER = [
        "product_classification",
        "investment_strategy",
        "sustainability_assessment",
        "template_population",
        "review_approval",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the pre-contractual disclosure workflow.

        Args:
            progress_callback: Optional callback(phase, message, pct).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "product_classification": ProductClassificationPhase(),
            "investment_strategy": InvestmentStrategyPhase(),
            "sustainability_assessment": SustainabilityAssessmentPhase(),
            "template_population": TemplatePopulationPhase(),
            "review_approval": ReviewApprovalPhase(),
        }

    async def run(
        self, input_data: PrecontractualDisclosureInput
    ) -> PrecontractualDisclosureResult:
        """
        Execute the complete 5-phase pre-contractual disclosure workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            PrecontractualDisclosureResult with per-phase details and summary.
        """
        started_at = utcnow()
        logger.info(
            "Starting precontractual disclosure workflow %s for org=%s product=%s",
            self.workflow_id, input_data.organization_id,
            input_data.product_name,
        )

        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=self._build_config(input_data),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            if context.is_phase_completed(phase_name):
                logger.info("Phase '%s' already completed, skipping", phase_name)
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(phase_name, f"Starting: {phase_name}", pct)
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_executor = self._phases[phase_name]
                phase_result = await phase_executor.execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, phase_result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, phase_result.status)
                    if phase_name == "product_classification":
                        overall_status = WorkflowStatus.FAILED
                        logger.error(
                            "Critical phase '%s' failed, aborting workflow",
                            phase_name,
                        )
                        break

                context.errors.extend(phase_result.errors)
                context.warnings.extend(phase_result.warnings)

            except Exception as exc:
                logger.error(
                    "Phase '%s' raised unhandled exception: %s",
                    phase_name, exc, exc_info=True,
                )
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    started_at=utcnow(),
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = (
                WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL
            )

        completed_at = utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress(
            "workflow", f"Workflow {overall_status.value}", 1.0
        )
        logger.info(
            "Precontractual disclosure workflow %s finished status=%s in %.1fs",
            self.workflow_id, overall_status.value, total_duration,
        )

        return PrecontractualDisclosureResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            product_name=summary.get("product_name", ""),
            sfdr_classification=summary.get("sfdr_classification", "ARTICLE_8"),
            is_article_8_eligible=summary.get("is_article_8_eligible", False),
            makes_sustainable_investments=summary.get(
                "makes_sustainable_investments", False
            ),
            taxonomy_alignment_commitment_pct=summary.get(
                "taxonomy_alignment_commitment_pct", 0.0
            ),
            sustainable_investment_minimum_pct=summary.get(
                "sustainable_investment_minimum_pct", 0.0
            ),
            binding_elements_count=summary.get("binding_elements_count", 0),
            template_sections_completed=summary.get(
                "template_sections_completed", 0
            ),
            template_sections_total=summary.get(
                "template_sections_total", 12
            ),
            review_status=summary.get("review_status", "DRAFT"),
            completeness_pct=summary.get("completeness_pct", 0.0),
        )

    def _build_config(
        self, input_data: PrecontractualDisclosureInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        config = input_data.model_dump()
        config["sfdr_classification"] = input_data.sfdr_classification.value
        if input_data.characteristics:
            config["characteristics"] = [
                c.model_dump() for c in input_data.characteristics
            ]
            for c in config["characteristics"]:
                c["characteristic_type"] = c["characteristic_type"].value if isinstance(
                    c["characteristic_type"], CharacteristicType
                ) else c["characteristic_type"]
        if input_data.exclusion_criteria:
            config["exclusion_criteria"] = [
                e.model_dump() for e in input_data.exclusion_criteria
            ]
        if input_data.asset_allocation_targets:
            config["asset_allocation_targets"] = (
                input_data.asset_allocation_targets.model_dump()
            )
        return config

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        classification = context.get_phase_output("product_classification")
        strategy = context.get_phase_output("investment_strategy")
        assessment = context.get_phase_output("sustainability_assessment")
        template = context.get_phase_output("template_population")
        review = context.get_phase_output("review_approval")

        return {
            "product_name": classification.get("product_name", ""),
            "sfdr_classification": classification.get(
                "effective_classification", "ARTICLE_8"
            ),
            "is_article_8_eligible": classification.get(
                "is_article_8_eligible", False
            ),
            "makes_sustainable_investments": classification.get(
                "makes_sustainable_investments", False
            ),
            "taxonomy_alignment_commitment_pct": assessment.get(
                "taxonomy_alignment_commitment_pct", 0.0
            ),
            "sustainable_investment_minimum_pct": assessment.get(
                "sustainable_investment_minimum_pct", 0.0
            ),
            "binding_elements_count": strategy.get(
                "binding_elements_count", 0
            ),
            "template_sections_completed": template.get(
                "sections_completed", 0
            ),
            "template_sections_total": template.get("sections_total", 12),
            "review_status": review.get("review_status", "DRAFT"),
            "completeness_pct": review.get("completeness_pct", 0.0),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
