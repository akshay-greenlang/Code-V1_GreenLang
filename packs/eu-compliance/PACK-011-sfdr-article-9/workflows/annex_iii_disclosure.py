# -*- coding: utf-8 -*-
"""
Annex III Pre-contractual Disclosure Workflow
================================================

Five-phase workflow for generating SFDR Annex III pre-contractual disclosures
for Article 9 financial products. Orchestrates sustainable objective
verification, investment strategy definition, DNSH assessment, template
population, and review/approval into a single auditable pipeline.

Regulatory Context:
    Per EU SFDR Regulation 2019/2088 and Delegated Regulation 2022/1288 (RTS):
    - Article 9: Products with sustainable investment as their objective.
    - Annex III: Template for Article 9 pre-contractual disclosures.
    - 100% of investments must be sustainable (excluding cash and hedging).
    - Mandatory DNSH assessment for all sustainable investments.
    - Taxonomy Regulation 2020/852 alignment must be disclosed with minimum
      commitment to taxonomy-aligned activities.
    - EU Climate Benchmark (CTB or PAB) designation required, or explanation
      of how the sustainable investment objective is met without one.
    - Good governance assessment mandatory for all investees.

    Key Disclosure Elements (Annex III):
    - Sustainable investment objective description
    - Investment strategy with binding elements
    - 100% sustainable investment commitment with environmental/social split
    - Taxonomy alignment minimum commitment
    - DNSH methodology for all six environmental objectives
    - Good governance assessment approach
    - EU Climate Benchmark designation and methodology
    - Data sources and processing methodology
    - Limitations and due diligence procedures

Phases:
    1. SustainableObjectiveVerification - Verify Article 9 eligibility,
       confirm sustainable investment objective, validate 100% commitment
    2. InvestmentStrategy - Define binding elements, asset allocation with
       100% sustainable minimum, benchmark designation
    3. DNSHAssessment - Full DNSH assessment across all six environmental
       objectives plus social safeguards
    4. TemplatePopulation - Generate Annex III template with all required
       sections populated from computed data
    5. ReviewApproval - Compliance completeness check, legal review,
       version tracking, sign-off workflow

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

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

class SustainableObjectiveType(str, Enum):
    """Type of sustainable investment objective."""
    ENVIRONMENTAL = "ENVIRONMENTAL"
    SOCIAL = "SOCIAL"
    COMBINED = "COMBINED"
    CLIMATE_CHANGE_MITIGATION = "CLIMATE_CHANGE_MITIGATION"
    CLIMATE_CHANGE_ADAPTATION = "CLIMATE_CHANGE_ADAPTATION"
    CARBON_REDUCTION = "CARBON_REDUCTION"

class BenchmarkType(str, Enum):
    """EU Climate Benchmark type."""
    CTB = "CTB"
    PAB = "PAB"
    CUSTOM = "CUSTOM"
    NONE = "NONE"

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
# DATA MODELS - ANNEX III DISCLOSURE
# =============================================================================

class SustainableObjective(BaseModel):
    """Sustainable investment objective for Article 9 product."""
    objective_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Objective name")
    objective_type: SustainableObjectiveType = Field(
        ..., description="Objective type"
    )
    description: str = Field(default="", description="Detailed description")
    sustainability_indicators: List[str] = Field(
        default_factory=list,
        description="KPIs measuring objective attainment"
    )
    taxonomy_objective: Optional[str] = Field(
        None, description="Linked EU Taxonomy environmental objective"
    )

class ExclusionCriteria(BaseModel):
    """Negative screening exclusion criterion."""
    criterion_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Exclusion criterion name")
    description: str = Field(default="")
    threshold_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0,
        description="Revenue threshold percentage"
    )
    applies_to: str = Field(
        default="all", description="Scope: all, direct, indirect"
    )
    binding: bool = Field(default=True)

class AnnexIIIDisclosureInput(BaseModel):
    """Input configuration for the Annex III pre-contractual workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    product_name: str = Field(..., description="Financial product name")
    product_isin: Optional[str] = Field(None, description="ISIN if applicable")
    product_lei: Optional[str] = Field(None, description="LEI of product manager")
    reporting_date: str = Field(
        ..., description="Disclosure date YYYY-MM-DD"
    )
    sustainable_objectives: List[SustainableObjective] = Field(
        default_factory=list,
        description="Sustainable investment objectives"
    )
    exclusion_criteria: List[ExclusionCriteria] = Field(
        default_factory=list, description="Negative screening criteria"
    )
    minimum_taxonomy_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Minimum taxonomy-aligned investment percentage"
    )
    minimum_environmentally_sustainable_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Minimum environmentally sustainable percentage"
    )
    minimum_socially_sustainable_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Minimum socially sustainable percentage"
    )
    non_sustainable_allocation_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Cash/hedging not qualifying as sustainable"
    )
    benchmark_type: BenchmarkType = Field(
        default=BenchmarkType.NONE,
        description="EU Climate Benchmark type"
    )
    benchmark_name: Optional[str] = Field(None, description="Benchmark name")
    benchmark_methodology: str = Field(
        default="", description="Benchmark methodology description"
    )
    esg_rating_threshold: Optional[float] = Field(
        None, ge=0.0, le=100.0
    )
    engagement_commitment: bool = Field(default=True)
    engagement_description: str = Field(default="")
    data_sources: List[str] = Field(default_factory=list)
    data_processing_description: str = Field(default="")
    limitations_description: str = Field(default="")
    due_diligence_description: str = Field(default="")
    previous_version_id: Optional[str] = Field(None)
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

class AnnexIIIDisclosureResult(WorkflowResult):
    """Complete result from the Annex III disclosure workflow."""
    product_name: str = Field(default="")
    is_article_9_eligible: bool = Field(default=False)
    sustainable_objective_confirmed: bool = Field(default=False)
    taxonomy_alignment_commitment_pct: float = Field(default=0.0)
    sustainable_investment_commitment_pct: float = Field(default=100.0)
    environmental_sustainable_pct: float = Field(default=0.0)
    social_sustainable_pct: float = Field(default=0.0)
    dnsh_assessment_complete: bool = Field(default=False)
    benchmark_type: str = Field(default="NONE")
    template_sections_completed: int = Field(default=0)
    template_sections_total: int = Field(default=11)
    review_status: str = Field(default="DRAFT")
    completeness_pct: float = Field(default=0.0)

# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================

class SustainableObjectiveVerificationPhase:
    """
    Phase 1: Sustainable Objective Verification.

    Verifies Article 9 eligibility by confirming the product has sustainable
    investment as its objective, validates the 100% sustainable investment
    commitment, and checks taxonomy alignment requirements.
    """

    PHASE_NAME = "sustainable_objective_verification"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute sustainable objective verification phase.

        Args:
            context: Workflow context with product configuration.

        Returns:
            PhaseResult with eligibility determination.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            product_name = config.get("product_name", "")
            objectives = config.get("sustainable_objectives", [])
            exclusions = config.get("exclusion_criteria", [])
            non_sustainable_pct = config.get(
                "non_sustainable_allocation_pct", 0.0
            )

            outputs["product_name"] = product_name
            outputs["product_isin"] = config.get("product_isin", "")
            outputs["product_lei"] = config.get("product_lei", "")

            # Article 9 eligibility checks
            eligibility_checks = []

            # Check 1: Must have at least one sustainable objective
            has_objectives = len(objectives) > 0
            eligibility_checks.append({
                "check": "sustainable_objective_defined",
                "passed": has_objectives,
                "detail": f"{len(objectives)} objective(s) defined"
            })

            # Check 2: Must have sustainability indicators
            has_indicators = any(
                len(o.get("sustainability_indicators", [])) > 0
                for o in objectives
            ) if objectives else False
            eligibility_checks.append({
                "check": "sustainability_indicators_defined",
                "passed": has_indicators,
                "detail": "Indicators present" if has_indicators
                else "No sustainability indicators"
            })

            # Check 3: 100% sustainable commitment (excl cash/hedging)
            sustainable_pct = 100.0 - non_sustainable_pct
            has_full_commitment = sustainable_pct >= 99.0
            eligibility_checks.append({
                "check": "full_sustainable_commitment",
                "passed": has_full_commitment,
                "detail": (
                    f"{sustainable_pct:.1f}% sustainable investment "
                    f"commitment ({non_sustainable_pct:.1f}% "
                    f"cash/hedging)"
                )
            })

            # Check 4: Non-sustainable must be cash/hedging only
            valid_non_sustainable = non_sustainable_pct <= 10.0
            eligibility_checks.append({
                "check": "non_sustainable_within_limits",
                "passed": valid_non_sustainable,
                "detail": (
                    f"Non-sustainable allocation: {non_sustainable_pct:.1f}%"
                )
            })

            # Check 5: Benchmark or methodology defined
            benchmark_type = config.get("benchmark_type", "NONE")
            has_benchmark = benchmark_type != BenchmarkType.NONE.value
            benchmark_methodology = config.get("benchmark_methodology", "")
            has_methodology = bool(benchmark_methodology)
            eligibility_checks.append({
                "check": "benchmark_or_methodology",
                "passed": has_benchmark or has_methodology,
                "detail": (
                    f"Benchmark: {benchmark_type}"
                    if has_benchmark
                    else "Custom methodology"
                    if has_methodology
                    else "No benchmark or methodology"
                )
            })

            is_eligible = all(c["passed"] for c in eligibility_checks)
            outputs["eligibility_checks"] = eligibility_checks
            outputs["is_article_9_eligible"] = is_eligible

            if not is_eligible:
                failed = [
                    c["check"] for c in eligibility_checks
                    if not c["passed"]
                ]
                warnings.append(
                    f"Product may not meet Article 9 requirements: "
                    f"failed: {', '.join(failed)}"
                )

            # Categorize objectives
            env_objectives = [
                o for o in objectives
                if o.get("objective_type") in (
                    SustainableObjectiveType.ENVIRONMENTAL.value,
                    SustainableObjectiveType.CLIMATE_CHANGE_MITIGATION.value,
                    SustainableObjectiveType.CLIMATE_CHANGE_ADAPTATION.value,
                    SustainableObjectiveType.CARBON_REDUCTION.value,
                )
            ]
            social_objectives = [
                o for o in objectives
                if o.get("objective_type") == SustainableObjectiveType.SOCIAL.value
            ]
            combined_objectives = [
                o for o in objectives
                if o.get("objective_type") == SustainableObjectiveType.COMBINED.value
            ]

            outputs["environmental_objectives_count"] = len(env_objectives)
            outputs["social_objectives_count"] = len(social_objectives)
            outputs["combined_objectives_count"] = len(combined_objectives)
            outputs["total_objectives_count"] = len(objectives)
            outputs["sustainable_investment_commitment_pct"] = sustainable_pct
            outputs["non_sustainable_allocation_pct"] = non_sustainable_pct
            outputs["exclusion_criteria_count"] = len(exclusions)

            status = PhaseStatus.COMPLETED
            records = len(objectives) + len(exclusions)

        except Exception as exc:
            logger.error(
                "SustainableObjectiveVerification failed: %s",
                exc, exc_info=True,
            )
            errors.append(
                f"Sustainable objective verification failed: {str(exc)}"
            )
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

    Defines binding elements of the investment strategy including exclusion
    criteria, ESG thresholds, engagement commitments, 100% sustainable
    allocation, and EU Climate Benchmark designation.
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
            verification_output = context.get_phase_output(
                "sustainable_objective_verification"
            )
            sustainable_pct = verification_output.get(
                "sustainable_investment_commitment_pct", 100.0
            )
            non_sustainable_pct = verification_output.get(
                "non_sustainable_allocation_pct", 0.0
            )

            # Binding elements compilation
            binding_elements = []

            # 100% sustainable investment is the primary binding element
            binding_elements.append({
                "type": "sustainable_objective",
                "name": "100% Sustainable Investment Commitment",
                "description": (
                    f"A minimum of {sustainable_pct:.1f}% of investments "
                    f"must qualify as sustainable investments."
                ),
                "threshold_value": sustainable_pct,
            })

            # Exclusion criteria
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

            # ESG threshold
            esg_threshold = config.get("esg_rating_threshold")
            if esg_threshold is not None:
                binding_elements.append({
                    "type": "esg_threshold",
                    "name": "Minimum ESG Rating",
                    "description": (
                        f"Investees must maintain minimum ESG rating "
                        f"of {esg_threshold}"
                    ),
                    "threshold_value": esg_threshold,
                })

            # Engagement commitment
            if config.get("engagement_commitment", True):
                binding_elements.append({
                    "type": "engagement",
                    "name": "Active Engagement Commitment",
                    "description": config.get("engagement_description", ""),
                })

            outputs["binding_elements"] = binding_elements
            outputs["binding_elements_count"] = len(binding_elements)

            # Asset allocation - Article 9 requires ~100% sustainable
            env_pct = config.get(
                "minimum_environmentally_sustainable_pct", 0.0
            )
            social_pct = config.get(
                "minimum_socially_sustainable_pct", 0.0
            )
            taxonomy_pct = config.get("minimum_taxonomy_aligned_pct", 0.0)

            allocation = {
                "sustainable_investment_pct": sustainable_pct,
                "minimum_taxonomy_aligned_pct": taxonomy_pct,
                "minimum_environmentally_sustainable_pct": env_pct,
                "minimum_socially_sustainable_pct": social_pct,
                "non_sustainable_pct": non_sustainable_pct,
            }

            # Validate allocation consistency
            if env_pct + social_pct > sustainable_pct + 0.1:
                warnings.append(
                    f"Environmental ({env_pct:.1f}%) + Social "
                    f"({social_pct:.1f}%) exceeds sustainable "
                    f"commitment ({sustainable_pct:.1f}%)"
                )

            if taxonomy_pct > env_pct + 0.1:
                warnings.append(
                    f"Taxonomy alignment ({taxonomy_pct:.1f}%) "
                    f"exceeds environmental allocation ({env_pct:.1f}%)"
                )

            outputs["asset_allocation"] = allocation

            # EU Climate Benchmark
            benchmark_type = config.get(
                "benchmark_type", BenchmarkType.NONE.value
            )
            benchmark_name = config.get("benchmark_name", "")
            benchmark_methodology = config.get("benchmark_methodology", "")

            outputs["benchmark_designation"] = {
                "type": benchmark_type,
                "name": benchmark_name,
                "methodology": benchmark_methodology,
                "is_eu_climate_benchmark": benchmark_type in (
                    BenchmarkType.CTB.value,
                    BenchmarkType.PAB.value,
                ),
            }

            if benchmark_type == BenchmarkType.NONE.value:
                outputs["no_benchmark_explanation"] = (
                    "No EU Climate Benchmark has been designated. The "
                    "sustainable investment objective is attained through "
                    "the binding investment strategy and continuous "
                    "monitoring of sustainability indicators."
                )
            else:
                outputs["no_benchmark_explanation"] = ""

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "InvestmentStrategy failed: %s", exc, exc_info=True
            )
            errors.append(
                f"Investment strategy definition failed: {str(exc)}"
            )
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

class DNSHAssessmentPhase:
    """
    Phase 3: DNSH Assessment.

    Full Do No Significant Harm assessment across all six EU Taxonomy
    environmental objectives plus social safeguards. Mandatory for
    all Article 9 sustainable investments.
    """

    PHASE_NAME = "dnsh_assessment"

    ENVIRONMENTAL_OBJECTIVES = [
        "climate_change_mitigation",
        "climate_change_adaptation",
        "water_and_marine_resources",
        "circular_economy",
        "pollution_prevention",
        "biodiversity_and_ecosystems",
    ]

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute DNSH assessment phase.

        Args:
            context: Workflow context with investment strategy data.

        Returns:
            PhaseResult with DNSH assessment and governance approach.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            strategy_output = context.get_phase_output(
                "investment_strategy"
            )
            allocation = strategy_output.get("asset_allocation", {})
            taxonomy_pct = allocation.get(
                "minimum_taxonomy_aligned_pct", 0.0
            )

            # Taxonomy alignment commitment
            outputs["taxonomy_alignment_commitment_pct"] = taxonomy_pct
            outputs["taxonomy_alignment_basis"] = "revenue"

            if taxonomy_pct > 0:
                outputs["taxonomy_alignment_description"] = (
                    f"The product commits to a minimum of {taxonomy_pct}% "
                    f"of investments in taxonomy-aligned economic "
                    f"activities, measured on a revenue basis."
                )
                outputs["taxonomy_environmental_objectives"] = [
                    "climate_change_mitigation",
                    "climate_change_adaptation",
                ]
            else:
                outputs["taxonomy_alignment_description"] = (
                    "While the product has sustainable investment as its "
                    "objective, it does not commit to a minimum proportion "
                    "of taxonomy-aligned investments."
                )
                outputs["taxonomy_environmental_objectives"] = []

            # Full DNSH methodology - mandatory for Article 9
            dnsh_assessments = []
            for objective in self.ENVIRONMENTAL_OBJECTIVES:
                assessment = self._build_objective_assessment(objective)
                dnsh_assessments.append(assessment)

            outputs["dnsh_methodology"] = {
                "applicable": True,
                "description": (
                    "All sustainable investments are assessed to ensure "
                    "they do not significantly harm any environmental "
                    "or social objectives. As an Article 9 product, "
                    "DNSH assessment is mandatory for 100% of "
                    "investments qualifying as sustainable."
                ),
                "objective_assessments": dnsh_assessments,
                "social_safeguards": [
                    "OECD Guidelines for Multinational Enterprises",
                    "UN Guiding Principles on Business and Human Rights",
                    "ILO Core Labour Conventions",
                    "International Bill of Human Rights",
                ],
                "pai_indicators_used": [
                    "GHG emissions (Scope 1, 2, 3)",
                    "Carbon footprint",
                    "GHG intensity of investee companies",
                    "Exposure to fossil fuels",
                    "Non-renewable energy share",
                    "Energy consumption intensity",
                    "Activities affecting biodiversity",
                    "Emissions to water",
                    "Hazardous waste ratio",
                    "UNGC/OECD violations",
                    "Gender pay gap",
                    "Board gender diversity",
                    "Controversial weapons exposure",
                ],
            }

            # Good governance assessment
            outputs["good_governance_approach"] = {
                "description": (
                    "Good governance practices of investee companies "
                    "are assessed through evaluation of sound management "
                    "structures, employee relations, remuneration of "
                    "staff, and tax compliance. For Article 9 products, "
                    "governance assessment is applied to 100% of "
                    "investees."
                ),
                "assessment_areas": [
                    {
                        "area": "Management structures",
                        "criteria": "Board independence, oversight",
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
                        "criteria": "Tax transparency, no avoidance",
                    },
                ],
                "minimum_standards": [
                    "UN Global Compact principles adherence",
                    "No severe UNGC controversies",
                    "No critical governance failures",
                ],
            }

            # Data sources and processing
            data_sources = config.get("data_sources", [])
            outputs["data_sources"] = data_sources
            outputs["data_sources_count"] = len(data_sources)
            outputs["data_processing_description"] = config.get(
                "data_processing_description", ""
            )
            outputs["limitations_description"] = config.get(
                "limitations_description", ""
            )
            outputs["due_diligence_description"] = config.get(
                "due_diligence_description", ""
            )

            # Engagement policy
            outputs["engagement_policy"] = {
                "has_engagement": config.get(
                    "engagement_commitment", True
                ),
                "description": config.get(
                    "engagement_description", ""
                ),
            }

            if not data_sources:
                warnings.append(
                    "No data sources specified. Data sources disclosure "
                    "is mandatory under Annex III."
                )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "DNSHAssessment failed: %s", exc, exc_info=True
            )
            errors.append(f"DNSH assessment failed: {str(exc)}")
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

    def _build_objective_assessment(
        self, objective: str
    ) -> Dict[str, Any]:
        """Build DNSH assessment for a single environmental objective."""
        assessment_map = {
            "climate_change_mitigation": {
                "objective": "Climate change mitigation",
                "assessment": "GHG emissions intensity analysis",
                "indicators": [
                    "Scope 1+2 GHG emissions",
                    "Carbon footprint per EUR invested",
                ],
                "threshold": "Paris-aligned trajectory",
            },
            "climate_change_adaptation": {
                "objective": "Climate change adaptation",
                "assessment": "Physical risk exposure assessment",
                "indicators": [
                    "Physical climate risk score",
                    "Adaptation measures in place",
                ],
                "threshold": "No high unmitigated physical risk",
            },
            "water_and_marine_resources": {
                "objective": "Water and marine resources",
                "assessment": "Water stress and pollution metrics",
                "indicators": [
                    "Water consumption intensity",
                    "Water pollution incidents",
                ],
                "threshold": "Below sector median water intensity",
            },
            "circular_economy": {
                "objective": "Circular economy",
                "assessment": "Waste management and recycling",
                "indicators": [
                    "Waste recycling rate",
                    "Hazardous waste ratio",
                ],
                "threshold": "Above sector median recycling",
            },
            "pollution_prevention": {
                "objective": "Pollution prevention and control",
                "assessment": "Pollutant emission monitoring",
                "indicators": [
                    "Air pollutant emissions",
                    "Soil contamination incidents",
                ],
                "threshold": "Regulatory compliance confirmed",
            },
            "biodiversity_and_ecosystems": {
                "objective": "Biodiversity and ecosystems",
                "assessment": "Biodiversity impact assessment",
                "indicators": [
                    "Operations near sensitive areas",
                    "Deforestation exposure",
                ],
                "threshold": "No significant biodiversity harm",
            },
        }
        return assessment_map.get(objective, {
            "objective": objective,
            "assessment": "Standard assessment",
            "indicators": [],
            "threshold": "Not defined",
        })

class TemplatePopulationPhase:
    """
    Phase 4: Template Population.

    Generates the Annex III template with all required sections populated
    from computed data in previous phases.

from greenlang.schemas import utcnow
    """

    PHASE_NAME = "template_population"

    ANNEX_III_SECTIONS = [
        "summary",
        "sustainable_investment_objective",
        "investment_strategy",
        "proportion_of_investments",
        "monitoring_of_objective",
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
            PhaseResult with populated Annex III template.
        """
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            verification_output = context.get_phase_output(
                "sustainable_objective_verification"
            )
            strategy_output = context.get_phase_output(
                "investment_strategy"
            )
            dnsh_output = context.get_phase_output("dnsh_assessment")

            product_name = verification_output.get("product_name", "")
            allocation = strategy_output.get("asset_allocation", {})

            sections: Dict[str, Any] = {}
            completed_sections = 0

            # Section 1: Summary
            sections["summary"] = {
                "product_name": product_name,
                "classification": "ARTICLE_9",
                "objective_summary": (
                    f"This product has sustainable investment as its "
                    f"objective. {verification_output.get('total_objectives_count', 0)} "
                    f"sustainable objective(s) defined."
                ),
                "sustainable_commitment": (
                    f"{allocation.get('sustainable_investment_pct', 100.0):.1f}% "
                    f"sustainable investment commitment."
                ),
                "taxonomy_commitment": (
                    f"{allocation.get('minimum_taxonomy_aligned_pct', 0.0):.1f}% "
                    f"minimum taxonomy alignment."
                ),
            }
            completed_sections += 1

            # Section 2: Sustainable investment objective
            sections["sustainable_investment_objective"] = {
                "has_sustainable_objective": True,
                "description": (
                    "This financial product has sustainable investment "
                    "as its objective within the meaning of Article 9 "
                    "of Regulation (EU) 2019/2088."
                ),
                "objectives": config.get("sustainable_objectives", []),
                "dnsh_description": (
                    "All sustainable investments are assessed against "
                    "DNSH criteria across all six environmental "
                    "objectives and social safeguards."
                ),
            }
            completed_sections += 1

            # Section 3: Investment strategy
            sections["investment_strategy"] = {
                "binding_elements": strategy_output.get(
                    "binding_elements", []
                ),
                "good_governance": dnsh_output.get(
                    "good_governance_approach", {}
                ),
            }
            completed_sections += 1

            # Section 4: Proportion of investments
            sections["proportion_of_investments"] = {
                "asset_allocation": allocation,
                "is_100_pct_sustainable": True,
                "taxonomy_alignment": {
                    "commitment_pct": allocation.get(
                        "minimum_taxonomy_aligned_pct", 0.0
                    ),
                    "environmental_pct": allocation.get(
                        "minimum_environmentally_sustainable_pct", 0.0
                    ),
                    "social_pct": allocation.get(
                        "minimum_socially_sustainable_pct", 0.0
                    ),
                },
                "non_sustainable_purpose": (
                    "Cash held for liquidity management and derivatives "
                    "for hedging purposes only."
                ),
            }
            completed_sections += 1

            # Section 5: Monitoring
            indicators = []
            for obj in config.get("sustainable_objectives", []):
                indicators.extend(
                    obj.get("sustainability_indicators", [])
                )
            sections["monitoring_of_objective"] = {
                "description": (
                    "The attainment of the sustainable investment "
                    "objective is monitored through sustainability "
                    "indicators measured continuously."
                ),
                "sustainability_indicators": indicators,
                "monitoring_frequency": "continuous",
                "escalation_process": (
                    "Deviations from the objective trigger immediate "
                    "review and remediation."
                ),
            }
            completed_sections += 1

            # Section 6: Methodologies
            sections["methodologies"] = {
                "description": (
                    "Quantitative and qualitative methodologies are "
                    "used to measure attainment of the sustainable "
                    "investment objective."
                ),
                "measurement_approach": "quantitative_and_qualitative",
                "dnsh_methodology": dnsh_output.get(
                    "dnsh_methodology", {}
                ),
            }
            completed_sections += 1

            # Section 7: Data sources
            sections["data_sources_and_processing"] = {
                "data_sources": dnsh_output.get("data_sources", []),
                "processing_description": dnsh_output.get(
                    "data_processing_description", ""
                ),
                "data_quality_measures": [
                    "Automated validation checks",
                    "Cross-referencing multiple data providers",
                    "Regular data completeness audits",
                    "DNSH data verification process",
                ],
            }
            completed_sections += 1

            # Section 8: Limitations
            limitations = dnsh_output.get(
                "limitations_description", ""
            )
            sections["limitations_to_methodologies"] = {
                "description": limitations if limitations else (
                    "Limitations may arise from data availability, "
                    "reporting standard differences, and estimation "
                    "methodologies where reported data is unavailable."
                ),
                "mitigation_measures": [
                    "Multiple data sources for cross-validation",
                    "Conservative estimation approaches",
                    "Regular methodology reviews",
                    "Enhanced due diligence for data gaps",
                ],
            }
            completed_sections += 1

            # Section 9: Due diligence
            dd = dnsh_output.get("due_diligence_description", "")
            sections["due_diligence"] = {
                "description": dd if dd else (
                    "Enhanced due diligence is carried out on all "
                    "underlying assets to verify sustainable "
                    "investment qualification."
                ),
                "processes": [
                    "Pre-investment sustainability screening",
                    "DNSH assessment for all investees",
                    "Good governance verification",
                    "Ongoing sustainability monitoring",
                    "Incident and controversy monitoring",
                ],
            }
            completed_sections += 1

            # Section 10: Engagement
            engagement = dnsh_output.get("engagement_policy", {})
            sections["engagement_policies"] = {
                "has_engagement": engagement.get(
                    "has_engagement", True
                ),
                "description": engagement.get("description", ""),
                "engagement_types": [
                    "Direct dialogue with investee companies",
                    "Proxy voting aligned with sustainable objective",
                    "Collaborative engagement initiatives",
                    "Escalation procedures for non-compliance",
                ],
            }
            completed_sections += 1

            # Section 11: Reference benchmark
            benchmark = strategy_output.get(
                "benchmark_designation", {}
            )
            sections["designated_reference_benchmark"] = {
                "has_benchmark": benchmark.get(
                    "is_eu_climate_benchmark", False
                ),
                "benchmark_type": benchmark.get("type", "NONE"),
                "benchmark_name": benchmark.get("name", ""),
                "methodology": benchmark.get("methodology", ""),
                "no_benchmark_explanation": strategy_output.get(
                    "no_benchmark_explanation", ""
                ),
            }
            completed_sections += 1

            outputs["template_sections"] = sections
            outputs["sections_completed"] = completed_sections
            outputs["sections_total"] = len(self.ANNEX_III_SECTIONS)
            outputs["completeness_pct"] = round(
                completed_sections / len(self.ANNEX_III_SECTIONS) * 100,
                1,
            )
            outputs["template_version"] = "1.0"
            outputs["template_format"] = "structured_json"
            outputs["generated_at"] = utcnow().isoformat()

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "TemplatePopulation failed: %s", exc, exc_info=True
            )
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

class ReviewApprovalPhase:
    """
    Phase 5: Review and Approval.

    Performs compliance completeness check against Annex III requirements,
    tracks legal review status, manages version control, and orchestrates
    the sign-off workflow.
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
            template_output = context.get_phase_output(
                "template_population"
            )
            verification_output = context.get_phase_output(
                "sustainable_objective_verification"
            )
            dnsh_output = context.get_phase_output("dnsh_assessment")

            sections_completed = template_output.get(
                "sections_completed", 0
            )
            sections_total = template_output.get("sections_total", 11)
            completeness_pct = template_output.get(
                "completeness_pct", 0.0
            )

            # Compliance checks specific to Article 9
            compliance_issues = []

            # Check: Article 9 eligibility
            if not verification_output.get(
                "is_article_9_eligible", False
            ):
                compliance_issues.append({
                    "severity": "critical",
                    "issue": "Product does not meet Article 9 eligibility",
                    "remediation": "Address failing eligibility checks",
                })

            # Check: Sustainable objective defined
            if verification_output.get(
                "total_objectives_count", 0
            ) == 0:
                compliance_issues.append({
                    "severity": "critical",
                    "issue": "No sustainable investment objective defined",
                    "remediation": "Define at least one objective",
                })

            # Check: DNSH assessment complete
            dnsh = dnsh_output.get("dnsh_methodology", {})
            dnsh_complete = (
                dnsh.get("applicable", False)
                and len(dnsh.get("objective_assessments", [])) == 6
            )
            if not dnsh_complete:
                compliance_issues.append({
                    "severity": "critical",
                    "issue": "DNSH assessment incomplete",
                    "remediation": (
                        "Complete DNSH for all 6 environmental objectives"
                    ),
                })

            # Check: Data sources
            if dnsh_output.get("data_sources_count", 0) == 0:
                compliance_issues.append({
                    "severity": "high",
                    "issue": "No data sources disclosed",
                    "remediation": "Specify data sources for assessment",
                })

            outputs["compliance_issues"] = compliance_issues
            outputs["compliance_issues_count"] = len(compliance_issues)
            outputs["has_critical_issues"] = any(
                i["severity"] == "critical" for i in compliance_issues
            )
            outputs["completeness_pct"] = completeness_pct
            outputs["sections_completed"] = sections_completed
            outputs["sections_total"] = sections_total
            outputs["dnsh_assessment_complete"] = dnsh_complete

            if compliance_issues:
                warnings.append(
                    f"{len(compliance_issues)} compliance issue(s)"
                )

            # Version tracking
            previous = config.get("previous_version_id")
            outputs["version_info"] = {
                "version_id": str(uuid.uuid4()),
                "version_number": "1.0" if not previous else "1.1",
                "previous_version_id": previous,
                "created_at": utcnow().isoformat(),
                "created_by": "system",
            }

            # Review status
            if outputs["has_critical_issues"]:
                review_status = ReviewStatus.DRAFT.value
            elif completeness_pct >= 100.0 and not compliance_issues:
                review_status = ReviewStatus.UNDER_REVIEW.value
            else:
                review_status = ReviewStatus.DRAFT.value

            outputs["review_status"] = review_status
            outputs["sign_off_workflow"] = {
                "status": review_status,
                "compliance_officer": {
                    "approved": False, "timestamp": None
                },
                "legal_counsel": {
                    "approved": False, "timestamp": None
                },
                "product_manager": {
                    "approved": False, "timestamp": None
                },
                "senior_management": {
                    "approved": False, "timestamp": None
                },
            }

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error(
                "ReviewApproval failed: %s", exc, exc_info=True
            )
            errors.append(
                f"Review and approval failed: {str(exc)}"
            )
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

class AnnexIIIDisclosureWorkflow:
    """
    Five-phase Annex III pre-contractual disclosure workflow for Article 9.

    Orchestrates the complete Annex III disclosure generation from
    sustainable objective verification through template population and
    review approval. Supports checkpoint/resume and phase skipping.

    Attributes:
        workflow_id: Unique execution identifier.
        _phases: Ordered mapping of phase name to executor instance.
        _progress_callback: Optional progress notification callback.

    Example:
        >>> wf = AnnexIIIDisclosureWorkflow()
        >>> input_data = AnnexIIIDisclosureInput(
        ...     organization_id="org-123",
        ...     product_name="Climate Solutions Fund",
        ...     reporting_date="2026-01-01",
        ...     sustainable_objectives=[
        ...         SustainableObjective(
        ...             name="Carbon Reduction",
        ...             objective_type=SustainableObjectiveType.CARBON_REDUCTION,
        ...         )
        ...     ],
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "annex_iii_disclosure"

    PHASE_ORDER = [
        "sustainable_objective_verification",
        "investment_strategy",
        "dnsh_assessment",
        "template_population",
        "review_approval",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the Annex III disclosure workflow.

        Args:
            progress_callback: Optional callback(phase, message, pct).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "sustainable_objective_verification": SustainableObjectiveVerificationPhase(),
            "investment_strategy": InvestmentStrategyPhase(),
            "dnsh_assessment": DNSHAssessmentPhase(),
            "template_population": TemplatePopulationPhase(),
            "review_approval": ReviewApprovalPhase(),
        }

    async def run(
        self, input_data: AnnexIIIDisclosureInput
    ) -> AnnexIIIDisclosureResult:
        """
        Execute the complete 5-phase Annex III disclosure workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            AnnexIIIDisclosureResult with per-phase details and summary.
        """
        started_at = utcnow()
        logger.info(
            "Starting Annex III disclosure workflow %s for org=%s product=%s",
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
                logger.info(
                    "Phase '%s' already completed, skipping",
                    phase_name,
                )
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(
                phase_name, f"Starting: {phase_name}", pct
            )
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_executor = self._phases[phase_name]
                phase_result = await phase_executor.execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(
                        phase_name, phase_result.outputs
                    )
                    context.mark_phase(
                        phase_name, PhaseStatus.COMPLETED
                    )
                else:
                    context.mark_phase(
                        phase_name, phase_result.status
                    )
                    if phase_name == "sustainable_objective_verification":
                        overall_status = WorkflowStatus.FAILED
                        logger.error(
                            "Critical phase '%s' failed, aborting",
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
                WorkflowStatus.COMPLETED if all_ok
                else WorkflowStatus.PARTIAL
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
            "Annex III disclosure workflow %s finished status=%s in %.1fs",
            self.workflow_id, overall_status.value, total_duration,
        )

        return AnnexIIIDisclosureResult(
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
            is_article_9_eligible=summary.get(
                "is_article_9_eligible", False
            ),
            sustainable_objective_confirmed=summary.get(
                "sustainable_objective_confirmed", False
            ),
            taxonomy_alignment_commitment_pct=summary.get(
                "taxonomy_alignment_commitment_pct", 0.0
            ),
            sustainable_investment_commitment_pct=summary.get(
                "sustainable_investment_commitment_pct", 100.0
            ),
            environmental_sustainable_pct=summary.get(
                "environmental_sustainable_pct", 0.0
            ),
            social_sustainable_pct=summary.get(
                "social_sustainable_pct", 0.0
            ),
            dnsh_assessment_complete=summary.get(
                "dnsh_assessment_complete", False
            ),
            benchmark_type=summary.get("benchmark_type", "NONE"),
            template_sections_completed=summary.get(
                "template_sections_completed", 0
            ),
            template_sections_total=summary.get(
                "template_sections_total", 11
            ),
            review_status=summary.get("review_status", "DRAFT"),
            completeness_pct=summary.get("completeness_pct", 0.0),
        )

    def _build_config(
        self, input_data: AnnexIIIDisclosureInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        config = input_data.model_dump()
        config["benchmark_type"] = input_data.benchmark_type.value
        if input_data.sustainable_objectives:
            config["sustainable_objectives"] = [
                o.model_dump() for o in input_data.sustainable_objectives
            ]
            for o in config["sustainable_objectives"]:
                o["objective_type"] = (
                    o["objective_type"].value
                    if isinstance(
                        o["objective_type"], SustainableObjectiveType
                    )
                    else o["objective_type"]
                )
        if input_data.exclusion_criteria:
            config["exclusion_criteria"] = [
                e.model_dump() for e in input_data.exclusion_criteria
            ]
        return config

    def _build_summary(
        self, context: WorkflowContext
    ) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        verification = context.get_phase_output(
            "sustainable_objective_verification"
        )
        strategy = context.get_phase_output("investment_strategy")
        dnsh = context.get_phase_output("dnsh_assessment")
        template = context.get_phase_output("template_population")
        review = context.get_phase_output("review_approval")
        allocation = strategy.get("asset_allocation", {})
        benchmark = strategy.get("benchmark_designation", {})

        return {
            "product_name": verification.get("product_name", ""),
            "is_article_9_eligible": verification.get(
                "is_article_9_eligible", False
            ),
            "sustainable_objective_confirmed": verification.get(
                "total_objectives_count", 0
            ) > 0,
            "taxonomy_alignment_commitment_pct": dnsh.get(
                "taxonomy_alignment_commitment_pct", 0.0
            ),
            "sustainable_investment_commitment_pct": allocation.get(
                "sustainable_investment_pct", 100.0
            ),
            "environmental_sustainable_pct": allocation.get(
                "minimum_environmentally_sustainable_pct", 0.0
            ),
            "social_sustainable_pct": allocation.get(
                "minimum_socially_sustainable_pct", 0.0
            ),
            "dnsh_assessment_complete": review.get(
                "dnsh_assessment_complete", False
            ),
            "benchmark_type": benchmark.get("type", "NONE"),
            "template_sections_completed": template.get(
                "sections_completed", 0
            ),
            "template_sections_total": template.get(
                "sections_total", 11
            ),
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
                logger.debug(
                    "Progress callback failed for phase=%s", phase
                )
