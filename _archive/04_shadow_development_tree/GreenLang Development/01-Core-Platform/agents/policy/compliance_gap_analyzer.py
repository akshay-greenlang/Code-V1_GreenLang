# -*- coding: utf-8 -*-
"""
GL-POL-X-002: Compliance Gap Analyzer
=====================================

Identifies compliance gaps between current organizational state and regulatory
requirements. This agent is CRITICAL PATH - all gap assessments are deterministic
with full audit trails.

Capabilities:
    - Gap identification against regulatory requirements
    - Maturity assessment for compliance domains
    - Remediation priority scoring
    - Timeline-based gap closure tracking
    - Cross-regulation gap aggregation
    - Quantified compliance risk assessment

Zero-Hallucination Guarantees:
    - All gap assessments derived from deterministic rules
    - Maturity scores calculated from defined criteria
    - Complete audit trail for all assessments
    - No LLM inference in compliance gap determination

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import DeterministicClock, deterministic_uuid

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class ComplianceDomain(str, Enum):
    """Domains of compliance assessment."""
    EMISSIONS_MEASUREMENT = "emissions_measurement"
    EMISSIONS_REPORTING = "emissions_reporting"
    DATA_COLLECTION = "data_collection"
    DATA_QUALITY = "data_quality"
    GOVERNANCE = "governance"
    ASSURANCE = "assurance"
    DISCLOSURE = "disclosure"
    SUPPLY_CHAIN = "supply_chain"
    TARGETS = "targets"
    STRATEGY = "strategy"
    RISK_MANAGEMENT = "risk_management"
    BIODIVERSITY = "biodiversity"


class MaturityLevel(str, Enum):
    """Maturity levels for compliance domains."""
    NONE = "none"           # 0 - No capability
    INITIAL = "initial"     # 1 - Ad-hoc processes
    DEVELOPING = "developing"  # 2 - Basic processes
    DEFINED = "defined"     # 3 - Standardized processes
    MANAGED = "managed"     # 4 - Measured and controlled
    OPTIMIZED = "optimized" # 5 - Continuous improvement


class GapSeverity(str, Enum):
    """Severity of compliance gaps."""
    CRITICAL = "critical"     # Immediate action required
    HIGH = "high"             # Action required within 30 days
    MEDIUM = "medium"         # Action required within 90 days
    LOW = "low"               # Action required within 180 days
    INFORMATIONAL = "informational"  # For awareness only


class GapStatus(str, Enum):
    """Status of a compliance gap."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    REMEDIATED = "remediated"
    ACCEPTED = "accepted"  # Risk accepted
    DEFERRED = "deferred"


class RemediationEffort(str, Enum):
    """Effort level for remediation."""
    MINIMAL = "minimal"       # < 1 week
    LOW = "low"               # 1-4 weeks
    MEDIUM = "medium"         # 1-3 months
    HIGH = "high"             # 3-6 months
    EXTENSIVE = "extensive"   # > 6 months


# Maturity level numeric mapping
MATURITY_SCORES: Dict[MaturityLevel, int] = {
    MaturityLevel.NONE: 0,
    MaturityLevel.INITIAL: 1,
    MaturityLevel.DEVELOPING: 2,
    MaturityLevel.DEFINED: 3,
    MaturityLevel.MANAGED: 4,
    MaturityLevel.OPTIMIZED: 5,
}

# Severity numeric mapping for prioritization
SEVERITY_SCORES: Dict[GapSeverity, int] = {
    GapSeverity.CRITICAL: 100,
    GapSeverity.HIGH: 75,
    GapSeverity.MEDIUM: 50,
    GapSeverity.LOW: 25,
    GapSeverity.INFORMATIONAL: 10,
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class ComplianceRequirement(BaseModel):
    """A specific compliance requirement to assess."""

    requirement_id: str = Field(..., description="Unique requirement identifier")
    regulation_id: str = Field(..., description="Parent regulation ID")
    domain: ComplianceDomain = Field(..., description="Compliance domain")
    name: str = Field(..., description="Requirement name")
    description: str = Field(..., description="Detailed description")

    # Thresholds
    minimum_maturity_level: MaturityLevel = Field(
        default=MaturityLevel.DEFINED,
        description="Minimum required maturity level"
    )

    # Deadlines
    deadline: Optional[date] = Field(
        None,
        description="Compliance deadline"
    )

    # Mandatory vs voluntary
    mandatory: bool = Field(
        default=True,
        description="Whether requirement is mandatory"
    )

    # Weighting
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Weight for scoring"
    )

    # Evidence requirements
    evidence_required: List[str] = Field(
        default_factory=list,
        description="Required evidence types"
    )


class CurrentStateAssessment(BaseModel):
    """Current state of compliance for a domain."""

    domain: ComplianceDomain = Field(..., description="Compliance domain")
    current_maturity: MaturityLevel = Field(
        ...,
        description="Current maturity level"
    )

    # Evidence
    evidence_available: List[str] = Field(
        default_factory=list,
        description="Available evidence"
    )
    evidence_quality: str = Field(
        default="medium",
        description="Quality: high, medium, low"
    )

    # Processes
    documented_processes: bool = Field(
        default=False,
        description="Whether processes are documented"
    )
    automated_processes: bool = Field(
        default=False,
        description="Whether processes are automated"
    )

    # Resources
    dedicated_resources: bool = Field(
        default=False,
        description="Whether dedicated resources exist"
    )
    external_support: bool = Field(
        default=False,
        description="Whether external support is engaged"
    )

    # Notes
    notes: Optional[str] = Field(None, description="Additional notes")


class ComplianceGap(BaseModel):
    """A specific compliance gap identified."""

    gap_id: str = Field(
        default_factory=lambda: deterministic_uuid("gap"),
        description="Unique gap identifier"
    )
    requirement_id: str = Field(..., description="Related requirement ID")
    regulation_id: str = Field(..., description="Related regulation ID")
    domain: ComplianceDomain = Field(..., description="Compliance domain")

    # Gap details
    gap_title: str = Field(..., description="Gap title")
    gap_description: str = Field(..., description="Detailed gap description")

    # Severity and priority
    severity: GapSeverity = Field(..., description="Gap severity")
    priority_score: float = Field(
        default=0.0,
        description="Calculated priority score"
    )

    # Maturity gap
    current_maturity: MaturityLevel = Field(
        ...,
        description="Current maturity level"
    )
    required_maturity: MaturityLevel = Field(
        ...,
        description="Required maturity level"
    )
    maturity_gap: int = Field(
        default=0,
        description="Numeric maturity gap"
    )

    # Remediation
    remediation_effort: RemediationEffort = Field(
        ...,
        description="Estimated remediation effort"
    )
    remediation_actions: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )
    estimated_cost_eur: Optional[Decimal] = Field(
        None,
        description="Estimated remediation cost"
    )

    # Timeline
    deadline: Optional[date] = Field(
        None,
        description="Compliance deadline"
    )
    days_until_deadline: Optional[int] = Field(
        None,
        description="Days until deadline"
    )

    # Status
    status: GapStatus = Field(
        default=GapStatus.OPEN,
        description="Current status"
    )

    # Evidence
    missing_evidence: List[str] = Field(
        default_factory=list,
        description="Missing evidence items"
    )

    # Audit
    identified_at: datetime = Field(
        default_factory=DeterministicClock.now,
        description="When gap was identified"
    )
    assessment_trace: List[str] = Field(
        default_factory=list,
        description="Assessment trace"
    )


class GapAnalysisResult(BaseModel):
    """Complete gap analysis result."""

    result_id: str = Field(
        default_factory=lambda: deterministic_uuid("gap_analysis"),
        description="Unique result identifier"
    )
    organization_id: str = Field(..., description="Organization identifier")
    analysis_date: date = Field(
        default_factory=lambda: DeterministicClock.now().date(),
        description="Analysis date"
    )

    # Gaps by severity
    critical_gaps: List[ComplianceGap] = Field(
        default_factory=list,
        description="Critical severity gaps"
    )
    high_gaps: List[ComplianceGap] = Field(
        default_factory=list,
        description="High severity gaps"
    )
    medium_gaps: List[ComplianceGap] = Field(
        default_factory=list,
        description="Medium severity gaps"
    )
    low_gaps: List[ComplianceGap] = Field(
        default_factory=list,
        description="Low severity gaps"
    )

    # Summary metrics
    total_gaps: int = Field(default=0)
    total_requirements_assessed: int = Field(default=0)
    compliance_score: float = Field(
        default=0.0,
        description="Overall compliance score (0-100)"
    )

    # Maturity summary
    maturity_by_domain: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Maturity assessment by domain"
    )
    average_maturity_score: float = Field(default=0.0)

    # Risk assessment
    compliance_risk_score: float = Field(
        default=0.0,
        description="Overall risk score (0-100)"
    )
    estimated_penalty_exposure_eur: Decimal = Field(
        default=Decimal("0"),
        description="Estimated penalty exposure"
    )

    # Remediation summary
    total_remediation_cost_eur: Decimal = Field(
        default=Decimal("0"),
        description="Total estimated remediation cost"
    )
    prioritized_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Prioritized remediation actions"
    )

    # Timeline
    nearest_deadline: Optional[date] = Field(
        None,
        description="Nearest compliance deadline"
    )
    gaps_by_deadline: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Gaps grouped by deadline"
    )

    # Provenance
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        content = {
            "organization_id": self.organization_id,
            "analysis_date": self.analysis_date.isoformat(),
            "total_gaps": self.total_gaps,
            "compliance_score": self.compliance_score,
        }
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()


class GapAnalysisInput(BaseModel):
    """Input for gap analysis."""

    organization_id: str = Field(..., description="Organization identifier")
    regulation_ids: List[str] = Field(
        ...,
        description="Regulations to assess against"
    )
    current_state: Dict[str, CurrentStateAssessment] = Field(
        ...,
        description="Current state by domain"
    )
    analysis_date: Optional[date] = Field(
        None,
        description="Date for analysis"
    )
    include_voluntary: bool = Field(
        default=False,
        description="Include voluntary requirements"
    )


class GapAnalysisOutput(BaseModel):
    """Output from gap analysis."""

    success: bool = Field(..., description="Whether analysis succeeded")
    result: Optional[GapAnalysisResult] = Field(None, description="Analysis result")
    error: Optional[str] = Field(None, description="Error message")
    warnings: List[str] = Field(default_factory=list)


# =============================================================================
# REQUIREMENTS DATABASE
# =============================================================================


# Standard compliance requirements (curated)
COMPLIANCE_REQUIREMENTS: Dict[str, ComplianceRequirement] = {}


def _initialize_requirements_database() -> None:
    """Initialize compliance requirements database."""
    global COMPLIANCE_REQUIREMENTS

    requirements = [
        # CSRD Requirements
        ComplianceRequirement(
            requirement_id="CSRD-E1-1",
            regulation_id="EU-CSRD",
            domain=ComplianceDomain.EMISSIONS_MEASUREMENT,
            name="Scope 1 GHG Emissions",
            description="Measure and report Scope 1 GHG emissions in tCO2e",
            minimum_maturity_level=MaturityLevel.MANAGED,
            deadline=date(2025, 1, 1),
            evidence_required=["emission_inventory", "calculation_methodology", "source_documentation"],
        ),
        ComplianceRequirement(
            requirement_id="CSRD-E1-2",
            regulation_id="EU-CSRD",
            domain=ComplianceDomain.EMISSIONS_MEASUREMENT,
            name="Scope 2 GHG Emissions",
            description="Measure and report Scope 2 GHG emissions (location and market-based)",
            minimum_maturity_level=MaturityLevel.MANAGED,
            deadline=date(2025, 1, 1),
            evidence_required=["energy_consumption_data", "emission_factors", "calculation_methodology"],
        ),
        ComplianceRequirement(
            requirement_id="CSRD-E1-3",
            regulation_id="EU-CSRD",
            domain=ComplianceDomain.EMISSIONS_MEASUREMENT,
            name="Scope 3 GHG Emissions",
            description="Measure and report material Scope 3 categories",
            minimum_maturity_level=MaturityLevel.DEFINED,
            deadline=date(2025, 1, 1),
            evidence_required=["scope3_screening", "category_calculations", "data_sources"],
        ),
        ComplianceRequirement(
            requirement_id="CSRD-G1-1",
            regulation_id="EU-CSRD",
            domain=ComplianceDomain.GOVERNANCE,
            name="Sustainability Governance",
            description="Board-level oversight of sustainability matters",
            minimum_maturity_level=MaturityLevel.DEFINED,
            deadline=date(2025, 1, 1),
            evidence_required=["board_charter", "sustainability_committee", "meeting_minutes"],
        ),
        ComplianceRequirement(
            requirement_id="CSRD-A1-1",
            regulation_id="EU-CSRD",
            domain=ComplianceDomain.ASSURANCE,
            name="Limited Assurance",
            description="Obtain limited assurance on sustainability statement",
            minimum_maturity_level=MaturityLevel.MANAGED,
            deadline=date(2025, 1, 1),
            evidence_required=["assurance_engagement", "assurance_report"],
        ),
        # CBAM Requirements
        ComplianceRequirement(
            requirement_id="CBAM-R1-1",
            regulation_id="EU-CBAM",
            domain=ComplianceDomain.EMISSIONS_MEASUREMENT,
            name="Embedded Emissions",
            description="Calculate embedded emissions in imported goods",
            minimum_maturity_level=MaturityLevel.MANAGED,
            deadline=date(2024, 1, 31),
            evidence_required=["product_emissions", "supplier_data", "calculation_methodology"],
        ),
        ComplianceRequirement(
            requirement_id="CBAM-R1-2",
            regulation_id="EU-CBAM",
            domain=ComplianceDomain.SUPPLY_CHAIN,
            name="Supplier Emissions Data",
            description="Collect actual emissions data from suppliers",
            minimum_maturity_level=MaturityLevel.DEFINED,
            deadline=date(2026, 1, 1),
            evidence_required=["supplier_declarations", "verification_records"],
        ),
        # SB253 Requirements
        ComplianceRequirement(
            requirement_id="SB253-R1-1",
            regulation_id="US-CA-SB253",
            domain=ComplianceDomain.EMISSIONS_REPORTING,
            name="Full GHG Disclosure",
            description="Annual disclosure of Scopes 1, 2, and 3 emissions",
            minimum_maturity_level=MaturityLevel.MANAGED,
            deadline=date(2027, 1, 1),
            evidence_required=["ghg_inventory", "third_party_verification"],
        ),
        ComplianceRequirement(
            requirement_id="SB253-R1-2",
            regulation_id="US-CA-SB253",
            domain=ComplianceDomain.ASSURANCE,
            name="Third Party Verification",
            description="Obtain third-party verification of GHG report",
            minimum_maturity_level=MaturityLevel.MANAGED,
            deadline=date(2030, 1, 1),
            evidence_required=["verification_statement", "verifier_accreditation"],
        ),
    ]

    for req in requirements:
        COMPLIANCE_REQUIREMENTS[req.requirement_id] = req


# Initialize
_initialize_requirements_database()


# =============================================================================
# COMPLIANCE GAP ANALYZER AGENT
# =============================================================================


class ComplianceGapAnalyzer(BaseAgent):
    """
    GL-POL-X-002: Compliance Gap Analyzer

    Identifies compliance gaps between current organizational state and
    regulatory requirements. CRITICAL PATH agent with zero-hallucination guarantees.

    All gap assessments are:
    - Based on deterministic maturity comparisons
    - Derived from curated requirements database
    - Fully auditable with assessment traces
    - No LLM inference involved

    Usage:
        agent = ComplianceGapAnalyzer()
        result = agent.run({
            'organization_id': 'org-123',
            'regulation_ids': ['EU-CSRD', 'EU-CBAM'],
            'current_state': {...}
        })
    """

    AGENT_ID = "GL-POL-X-002"
    AGENT_NAME = "Compliance Gap Analyzer"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name=AGENT_NAME,
        category=AgentCategory.CRITICAL,
        uses_chat_session=False,
        uses_rag=False,
        uses_tools=False,
        critical_for_compliance=True,
        audit_trail_required=True,
        description="Identifies compliance gaps using deterministic assessment"
    )

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize Compliance Gap Analyzer."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Deterministic compliance gap analyzer",
                version=self.VERSION,
                parameters={
                    "include_remediation_costs": True,
                    "calculate_risk_score": True,
                }
            )

        self._requirements = COMPLIANCE_REQUIREMENTS.copy()
        self._audit_trail: List[Dict[str, Any]] = []

        super().__init__(config)
        logger.info(f"Initialized {self.AGENT_NAME} v{self.VERSION}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute gap analysis."""
        import time
        start_time = time.time()

        try:
            agent_input = GapAnalysisInput(**input_data)
            result = self._analyze_gaps(agent_input)
            result.provenance_hash = result.calculate_provenance_hash()
            result.processing_time_ms = (time.time() - start_time) * 1000

            output = GapAnalysisOutput(success=True, result=result)

            return AgentResult(
                success=True,
                data=output.model_dump(),
            )

        except Exception as e:
            logger.error(f"Gap analysis failed: {str(e)}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _analyze_gaps(self, input_data: GapAnalysisInput) -> GapAnalysisResult:
        """Perform gap analysis - 100% deterministic."""
        analysis_date = input_data.analysis_date or DeterministicClock.now().date()

        result = GapAnalysisResult(
            organization_id=input_data.organization_id,
            analysis_date=analysis_date,
        )

        # Get requirements to assess
        requirements = self._get_requirements(
            input_data.regulation_ids,
            input_data.include_voluntary
        )

        all_gaps: List[ComplianceGap] = []
        maturity_by_domain: Dict[str, Dict[str, Any]] = {}

        # Assess each requirement
        for req_id, requirement in requirements.items():
            domain = requirement.domain
            current_state = input_data.current_state.get(
                domain.value,
                CurrentStateAssessment(
                    domain=domain,
                    current_maturity=MaturityLevel.NONE
                )
            )

            # Track maturity by domain
            if domain.value not in maturity_by_domain:
                maturity_by_domain[domain.value] = {
                    "domain": domain.value,
                    "current_maturity": current_state.current_maturity.value,
                    "current_score": MATURITY_SCORES[current_state.current_maturity],
                    "requirements_assessed": 0,
                    "gaps_found": 0,
                }

            maturity_by_domain[domain.value]["requirements_assessed"] += 1

            # Check for gap
            gap = self._assess_requirement(
                requirement, current_state, analysis_date
            )

            if gap:
                all_gaps.append(gap)
                maturity_by_domain[domain.value]["gaps_found"] += 1

        # Categorize gaps by severity
        for gap in all_gaps:
            if gap.severity == GapSeverity.CRITICAL:
                result.critical_gaps.append(gap)
            elif gap.severity == GapSeverity.HIGH:
                result.high_gaps.append(gap)
            elif gap.severity == GapSeverity.MEDIUM:
                result.medium_gaps.append(gap)
            else:
                result.low_gaps.append(gap)

        # Calculate summaries
        result.total_gaps = len(all_gaps)
        result.total_requirements_assessed = len(requirements)
        result.maturity_by_domain = maturity_by_domain

        # Calculate compliance score
        if requirements:
            gaps_weighted = sum(
                SEVERITY_SCORES[g.severity] * g.maturity_gap
                for g in all_gaps
            )
            max_possible = len(requirements) * SEVERITY_SCORES[GapSeverity.CRITICAL] * 5
            result.compliance_score = max(0, 100 - (gaps_weighted / max_possible * 100)) if max_possible > 0 else 100

        # Calculate average maturity
        if maturity_by_domain:
            total_score = sum(d["current_score"] for d in maturity_by_domain.values())
            result.average_maturity_score = total_score / len(maturity_by_domain)

        # Calculate risk score
        if self.config.parameters.get("calculate_risk_score", True):
            result.compliance_risk_score = self._calculate_risk_score(all_gaps)

        # Calculate remediation costs
        if self.config.parameters.get("include_remediation_costs", True):
            result.total_remediation_cost_eur = sum(
                g.estimated_cost_eur or Decimal("0") for g in all_gaps
            )

        # Build prioritized actions
        result.prioritized_actions = self._build_prioritized_actions(all_gaps)

        # Find nearest deadline
        deadlines = [g.deadline for g in all_gaps if g.deadline]
        if deadlines:
            result.nearest_deadline = min(deadlines)

        return result

    def _get_requirements(
        self,
        regulation_ids: List[str],
        include_voluntary: bool
    ) -> Dict[str, ComplianceRequirement]:
        """Get requirements for specified regulations."""
        requirements = {}

        for req_id, req in self._requirements.items():
            if req.regulation_id in regulation_ids:
                if req.mandatory or include_voluntary:
                    requirements[req_id] = req

        return requirements

    def _assess_requirement(
        self,
        requirement: ComplianceRequirement,
        current_state: CurrentStateAssessment,
        analysis_date: date
    ) -> Optional[ComplianceGap]:
        """Assess a single requirement for gaps."""
        trace: List[str] = []

        current_score = MATURITY_SCORES[current_state.current_maturity]
        required_score = MATURITY_SCORES[requirement.minimum_maturity_level]

        trace.append(
            f"Assessing {requirement.name}: "
            f"current={current_state.current_maturity.value} ({current_score}), "
            f"required={requirement.minimum_maturity_level.value} ({required_score})"
        )

        # No gap if maturity is sufficient
        if current_score >= required_score:
            trace.append("No gap: current maturity meets requirement")
            return None

        maturity_gap = required_score - current_score
        trace.append(f"Maturity gap: {maturity_gap} levels")

        # Determine severity based on gap size and deadline proximity
        severity = self._determine_severity(
            maturity_gap, requirement.deadline, analysis_date, trace
        )

        # Calculate priority score
        priority_score = self._calculate_priority(
            severity, maturity_gap, requirement.deadline, analysis_date
        )

        # Determine remediation effort
        effort = self._estimate_effort(maturity_gap, current_state)

        # Estimate cost
        estimated_cost = self._estimate_cost(effort, requirement.domain)

        # Check missing evidence
        missing_evidence = [
            e for e in requirement.evidence_required
            if e not in current_state.evidence_available
        ]

        # Build remediation actions
        remediation_actions = self._build_remediation_actions(
            requirement, current_state, maturity_gap
        )

        # Calculate days until deadline
        days_until = None
        if requirement.deadline:
            days_until = (requirement.deadline - analysis_date).days

        return ComplianceGap(
            requirement_id=requirement.requirement_id,
            regulation_id=requirement.regulation_id,
            domain=requirement.domain,
            gap_title=f"Gap in {requirement.name}",
            gap_description=(
                f"Current maturity ({current_state.current_maturity.value}) "
                f"does not meet required level ({requirement.minimum_maturity_level.value})"
            ),
            severity=severity,
            priority_score=priority_score,
            current_maturity=current_state.current_maturity,
            required_maturity=requirement.minimum_maturity_level,
            maturity_gap=maturity_gap,
            remediation_effort=effort,
            remediation_actions=remediation_actions,
            estimated_cost_eur=estimated_cost,
            deadline=requirement.deadline,
            days_until_deadline=days_until,
            missing_evidence=missing_evidence,
            assessment_trace=trace,
        )

    def _determine_severity(
        self,
        maturity_gap: int,
        deadline: Optional[date],
        analysis_date: date,
        trace: List[str]
    ) -> GapSeverity:
        """Determine gap severity based on gap size and deadline."""
        # Base severity on maturity gap
        if maturity_gap >= 4:
            base_severity = GapSeverity.CRITICAL
        elif maturity_gap >= 3:
            base_severity = GapSeverity.HIGH
        elif maturity_gap >= 2:
            base_severity = GapSeverity.MEDIUM
        else:
            base_severity = GapSeverity.LOW

        # Adjust for deadline proximity
        if deadline:
            days_until = (deadline - analysis_date).days
            if days_until <= 30:
                trace.append(f"Escalating severity: deadline in {days_until} days")
                if base_severity != GapSeverity.CRITICAL:
                    return GapSeverity.CRITICAL
            elif days_until <= 90:
                if base_severity in [GapSeverity.MEDIUM, GapSeverity.LOW]:
                    return GapSeverity.HIGH

        return base_severity

    def _calculate_priority(
        self,
        severity: GapSeverity,
        maturity_gap: int,
        deadline: Optional[date],
        analysis_date: date
    ) -> float:
        """Calculate priority score for gap ordering."""
        score = SEVERITY_SCORES[severity] * maturity_gap

        # Add deadline urgency factor
        if deadline:
            days_until = (deadline - analysis_date).days
            if days_until <= 0:
                score *= 2.0  # Past due
            elif days_until <= 30:
                score *= 1.5
            elif days_until <= 90:
                score *= 1.2

        return round(score, 2)

    def _estimate_effort(
        self,
        maturity_gap: int,
        current_state: CurrentStateAssessment
    ) -> RemediationEffort:
        """Estimate remediation effort."""
        if maturity_gap >= 4:
            return RemediationEffort.EXTENSIVE
        elif maturity_gap >= 3:
            return RemediationEffort.HIGH
        elif maturity_gap >= 2:
            if current_state.documented_processes:
                return RemediationEffort.MEDIUM
            return RemediationEffort.HIGH
        else:
            if current_state.automated_processes:
                return RemediationEffort.MINIMAL
            return RemediationEffort.LOW

    def _estimate_cost(
        self,
        effort: RemediationEffort,
        domain: ComplianceDomain
    ) -> Decimal:
        """Estimate remediation cost based on effort and domain."""
        # Base costs by effort level (EUR)
        effort_costs = {
            RemediationEffort.MINIMAL: Decimal("5000"),
            RemediationEffort.LOW: Decimal("25000"),
            RemediationEffort.MEDIUM: Decimal("75000"),
            RemediationEffort.HIGH: Decimal("200000"),
            RemediationEffort.EXTENSIVE: Decimal("500000"),
        }

        # Domain multipliers
        domain_multipliers = {
            ComplianceDomain.ASSURANCE: Decimal("1.5"),
            ComplianceDomain.SUPPLY_CHAIN: Decimal("2.0"),
            ComplianceDomain.EMISSIONS_MEASUREMENT: Decimal("1.2"),
            ComplianceDomain.DATA_COLLECTION: Decimal("1.3"),
        }

        base_cost = effort_costs.get(effort, Decimal("50000"))
        multiplier = domain_multipliers.get(domain, Decimal("1.0"))

        return base_cost * multiplier

    def _build_remediation_actions(
        self,
        requirement: ComplianceRequirement,
        current_state: CurrentStateAssessment,
        maturity_gap: int
    ) -> List[str]:
        """Build recommended remediation actions."""
        actions = []

        current_level = MATURITY_SCORES[current_state.current_maturity]

        # Level 0 -> 1: Establish basic capability
        if current_level < 1:
            actions.append("Establish initial data collection process")
            actions.append("Define roles and responsibilities")

        # Level 1 -> 2: Develop basic processes
        if current_level < 2 and maturity_gap >= 1:
            actions.append("Document standard operating procedures")
            actions.append("Implement basic quality controls")

        # Level 2 -> 3: Standardize processes
        if current_level < 3 and maturity_gap >= 2:
            actions.append("Standardize processes across organization")
            actions.append("Implement formal training program")

        # Level 3 -> 4: Measure and control
        if current_level < 4 and maturity_gap >= 3:
            actions.append("Implement monitoring and measurement systems")
            actions.append("Establish KPIs and performance tracking")

        # Level 4 -> 5: Optimize
        if current_level < 5 and maturity_gap >= 4:
            actions.append("Implement continuous improvement program")
            actions.append("Automate where possible")

        # Add evidence-specific actions
        for evidence in requirement.evidence_required:
            if evidence not in current_state.evidence_available:
                actions.append(f"Obtain required evidence: {evidence}")

        return actions

    def _calculate_risk_score(self, gaps: List[ComplianceGap]) -> float:
        """Calculate overall compliance risk score."""
        if not gaps:
            return 0.0

        # Weight by severity and maturity gap
        total_risk = sum(
            SEVERITY_SCORES[g.severity] * g.maturity_gap
            for g in gaps
        )

        # Normalize to 0-100
        max_risk = len(gaps) * 100 * 5  # Max severity * max gap
        return min(100, (total_risk / max_risk * 100) if max_risk > 0 else 0)

    def _build_prioritized_actions(
        self,
        gaps: List[ComplianceGap]
    ) -> List[Dict[str, Any]]:
        """Build prioritized list of remediation actions."""
        # Sort gaps by priority score
        sorted_gaps = sorted(gaps, key=lambda g: g.priority_score, reverse=True)

        actions = []
        for gap in sorted_gaps[:10]:  # Top 10 priorities
            actions.append({
                "priority_rank": len(actions) + 1,
                "gap_id": gap.gap_id,
                "requirement_id": gap.requirement_id,
                "domain": gap.domain.value,
                "severity": gap.severity.value,
                "priority_score": gap.priority_score,
                "actions": gap.remediation_actions[:3],  # Top 3 actions
                "effort": gap.remediation_effort.value,
                "deadline": gap.deadline.isoformat() if gap.deadline else None,
            })

        return actions

    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================

    def add_requirement(self, requirement: ComplianceRequirement) -> str:
        """Add a custom requirement."""
        self._requirements[requirement.requirement_id] = requirement
        return requirement.requirement_id

    def get_requirement(self, requirement_id: str) -> Optional[ComplianceRequirement]:
        """Get a requirement by ID."""
        return self._requirements.get(requirement_id)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ComplianceGapAnalyzer",
    "ComplianceDomain",
    "MaturityLevel",
    "GapSeverity",
    "GapStatus",
    "RemediationEffort",
    "ComplianceRequirement",
    "CurrentStateAssessment",
    "ComplianceGap",
    "GapAnalysisResult",
    "GapAnalysisInput",
    "GapAnalysisOutput",
    "COMPLIANCE_REQUIREMENTS",
]
