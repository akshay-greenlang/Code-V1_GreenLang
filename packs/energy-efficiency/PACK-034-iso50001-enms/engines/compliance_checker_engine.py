# -*- coding: utf-8 -*-
"""
ComplianceCheckerEngine - PACK-034 ISO 50001 EnMS Engine 8
===========================================================

ISO 50001:2018 Clauses 4-10 gap analysis and certification readiness
assessment engine.  Builds the complete ISO 50001:2018 clause tree
(Clauses 4.1 through 10.2), assesses each clause against supplied
evidence and documentation, identifies nonconformities, calculates
weighted compliance scores by category, determines certification
readiness, and generates prioritised gap-analysis reports.

Calculation Methodology:
    Clause Scoring (0-100):
        evidence_score = min(100, evidence_count / required_evidence * 100)
        document_score = min(100, document_count / required_documents * 100)
        clause_score   = evidence_weight * evidence_score
                       + document_weight * document_score
        Default weights: evidence_weight = 0.6, document_weight = 0.4

    Category Score (context, leadership, planning, ...):
        category_score = avg(clause_scores within category)

    Overall Score:
        Category weights per ISO 50001 emphasis:
            context: 0.10, leadership: 0.15, planning: 0.20,
            support: 0.10, operation: 0.15, performance: 0.20,
            improvement: 0.10
        overall_score = sum(category_score * category_weight)

    Nonconformity Severity:
        clause_score < 25  => CRITICAL
        clause_score 25-49 => MAJOR
        clause_score 50-74 => MINOR
        clause_score 75-89 => OBSERVATION
        clause_score >= 90 => compliant (no NC)

    Certification Readiness:
        NOT_READY:    overall < 40 OR any CRITICAL/MAJOR NC
        NEEDS_WORK:   overall 40-60 OR > 3 MINOR NCs
        NEARLY_READY: overall 60-80 AND <= 3 MINOR NCs
        READY:        overall >= 80 AND no open NCs
        CERTIFIED:    already certified (external flag)

Regulatory References:
    - ISO 50001:2018 - Energy management systems - Requirements
    - ISO 50003:2021 - Requirements for auditing and certification bodies
    - ISO 19011:2018 - Guidelines for auditing management systems
    - ISO 50004:2020 - Guidance for implementation, maintenance, improvement

Zero-Hallucination:
    - All thresholds derived from ISO 50001/50003 guidance
    - Deterministic Decimal arithmetic throughout
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-034 ISO 50001 EnMS
Engine:  8 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

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
    """Compute deterministic SHA-256 hash excluding volatile fields."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "assessed_date",
                         "calculation_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(num: Decimal, den: Decimal,
                 default: Decimal = Decimal("0")) -> Decimal:
    """Safely divide; return default on zero denominator."""
    if den == Decimal("0"):
        return default
    return num / den

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute part/whole*100 safely."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 2) -> Decimal:
    """Round Decimal to *places* using ROUND_HALF_UP."""
    q = "0." + "0" * places
    return value.quantize(Decimal(q), rounding=ROUND_HALF_UP)

def _today_str() -> str:
    """Return current UTC date as ISO string."""
    return datetime.now(timezone.utc).date().isoformat()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ClauseStatus(str, Enum):
    """Assessment status of an ISO 50001 clause."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PARTIALLY_COMPLIANT = "partially_compliant"
    COMPLIANT = "compliant"
    NOT_APPLICABLE = "not_applicable"

class AssessmentType(str, Enum):
    """Type of compliance assessment."""
    INITIAL_GAP = "initial_gap"
    SURVEILLANCE = "surveillance"
    RECERTIFICATION = "recertification"
    INTERNAL_AUDIT = "internal_audit"
    PRE_CERTIFICATION = "pre_certification"

class NonconformitySeverity(str, Enum):
    """Severity classification of a nonconformity."""
    OBSERVATION = "observation"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"

class CorrectionStatus(str, Enum):
    """Lifecycle status of a corrective action."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    CLOSED = "closed"

class CertificationReadiness(str, Enum):
    """Overall certification readiness level."""
    NOT_READY = "not_ready"
    NEEDS_WORK = "needs_work"
    NEARLY_READY = "nearly_ready"
    READY = "ready"
    CERTIFIED = "certified"

class DocumentType(str, Enum):
    """Type of documented information per ISO 50001."""
    POLICY = "policy"
    PROCEDURE = "procedure"
    RECORD = "record"
    MANUAL = "manual"
    FORM = "form"
    PLAN = "plan"
    REPORT = "report"
    REGISTER = "register"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_WEIGHTS: Dict[str, Decimal] = {
    "context": Decimal("0.10"), "leadership": Decimal("0.15"),
    "planning": Decimal("0.20"), "support": Decimal("0.10"),
    "operation": Decimal("0.15"), "performance": Decimal("0.20"),
    "improvement": Decimal("0.10"),
}

CLAUSE_CATEGORY_MAP: Dict[str, str] = {
    "4": "context", "5": "leadership", "6": "planning",
    "7": "support", "8": "operation", "9": "performance",
    "10": "improvement",
}

EVIDENCE_WEIGHT: Decimal = Decimal("0.60")
DOCUMENT_WEIGHT: Decimal = Decimal("0.40")

NC_THRESHOLD_CRITICAL: Decimal = Decimal("25")
NC_THRESHOLD_MAJOR: Decimal = Decimal("50")
NC_THRESHOLD_MINOR: Decimal = Decimal("75")
NC_THRESHOLD_OBS: Decimal = Decimal("90")

READINESS_NOT_READY: Decimal = Decimal("40")
READINESS_NEEDS_WORK: Decimal = Decimal("60")
READINESS_NEARLY_READY: Decimal = Decimal("80")

MAX_MINOR_NCS_NEARLY_READY: int = 3

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ISO50001Clause(BaseModel):
    """ISO 50001:2018 clause definition with audit requirements."""
    clause_number: str = Field(..., description="Clause identifier e.g. '4.1'")
    title: str = Field(..., description="Official clause title")
    description: str = Field(..., description="Clause requirement summary")
    parent_clause: Optional[str] = Field(default=None, description="Parent clause number")
    mandatory_documents: List[str] = Field(default_factory=list, description="Documents to be maintained")
    mandatory_records: List[str] = Field(default_factory=list, description="Records to be retained")
    audit_questions: List[str] = Field(default_factory=list, description="Audit questions")

    @field_validator("clause_number", mode="before")
    @classmethod
    def _val_cn(cls, v: Any) -> str:
        v = str(v).strip()
        if not v:
            raise ValueError("clause_number must not be empty")
        return v

class ClauseAssessment(BaseModel):
    """Assessment result for a single clause."""
    clause: ISO50001Clause = Field(..., description="Assessed clause")
    status: ClauseStatus = Field(..., description="Compliance status")
    score: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"), description="Score 0-100")
    evidence: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    assessed_date: str = Field(default_factory=_today_str)
    assessor: str = Field(default="system")

    @field_validator("status", mode="before")
    @classmethod
    def _val_st(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return ClauseStatus(v)
            except ValueError:
                pass
        return v

class DocumentChecklist(BaseModel):
    """Document completeness checklist item."""
    document_type: DocumentType = Field(...)
    title: str = Field(...)
    clause_references: List[str] = Field(default_factory=list)
    required: bool = Field(default=True)
    exists: bool = Field(default=False)
    current: bool = Field(default=False)
    adequate: bool = Field(default=False)
    notes: str = Field(default="")

    @field_validator("document_type", mode="before")
    @classmethod
    def _val_dt(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return DocumentType(v)
            except ValueError:
                pass
        return v

class Nonconformity(BaseModel):
    """Nonconformity identified during assessment."""
    nc_id: str = Field(default_factory=_new_uuid)
    clause_reference: str = Field(...)
    severity: NonconformitySeverity = Field(...)
    description: str = Field(...)
    objective_evidence: str = Field(default="")
    root_cause: Optional[str] = Field(default=None)
    correction_status: CorrectionStatus = Field(default=CorrectionStatus.OPEN)
    corrective_action: Optional[str] = Field(default=None)
    due_date: Optional[str] = Field(default=None)
    closed_date: Optional[str] = Field(default=None)

    @field_validator("severity", mode="before")
    @classmethod
    def _val_sev(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return NonconformitySeverity(v)
            except ValueError:
                pass
        return v

    @field_validator("correction_status", mode="before")
    @classmethod
    def _val_cs(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return CorrectionStatus(v)
            except ValueError:
                pass
        return v

class AuditFinding(BaseModel):
    """Audit finding generated during assessment."""
    finding_id: str = Field(default_factory=_new_uuid)
    clause_reference: str = Field(...)
    finding_type: str = Field(...)
    description: str = Field(...)
    evidence: str = Field(default="")
    nonconformity: Optional[Nonconformity] = Field(default=None)

    @field_validator("finding_type", mode="before")
    @classmethod
    def _val_ft(cls, v: Any) -> str:
        valid = {"positive", "observation", "nc_minor", "nc_major"}
        v = str(v).strip().lower()
        if v not in valid:
            return "observation"
        return v

class ComplianceScore(BaseModel):
    """Overall compliance score with category breakdown."""
    overall_score: Decimal = Field(default=Decimal("0"), ge=0, le=Decimal("100"))
    clause_scores: Dict[str, Decimal] = Field(default_factory=dict)
    category_scores: Dict[str, Decimal] = Field(default_factory=dict)
    readiness: CertificationReadiness = Field(default=CertificationReadiness.NOT_READY)
    critical_gaps: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)

class ComplianceResult(BaseModel):
    """Complete compliance assessment result."""
    assessment_id: str = Field(default_factory=_new_uuid)
    assessment_type: AssessmentType = Field(default=AssessmentType.INITIAL_GAP)
    organization_name: str = Field(...)
    scope: str = Field(...)
    assessed_date: str = Field(default_factory=_today_str)
    clause_assessments: List[ClauseAssessment] = Field(default_factory=list)
    document_checklist: List[DocumentChecklist] = Field(default_factory=list)
    nonconformities: List[Nonconformity] = Field(default_factory=list)
    findings: List[AuditFinding] = Field(default_factory=list)
    compliance_score: ComplianceScore = Field(default_factory=ComplianceScore)
    recommendations: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    calculation_time_ms: int = Field(default=0, ge=0)

    @field_validator("assessment_type", mode="before")
    @classmethod
    def _val_at(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return AssessmentType(v)
            except ValueError:
                pass
        return v

# ---------------------------------------------------------------------------
# Model Rebuild
# ---------------------------------------------------------------------------
ISO50001Clause.model_rebuild()
ClauseAssessment.model_rebuild()
DocumentChecklist.model_rebuild()
Nonconformity.model_rebuild()
AuditFinding.model_rebuild()
ComplianceScore.model_rebuild()
ComplianceResult.model_rebuild()

# ---------------------------------------------------------------------------
# ISO 50001:2018 Clause Definitions (25 sub-clauses: 4.1-10.2)
# ---------------------------------------------------------------------------

_CLAUSE_DEFS: List[Dict[str, Any]] = [
    {"clause_number": "4.1", "title": "Understanding the organisation and its context", "description": "Determine external and internal issues relevant to the EnMS.", "parent_clause": "4", "mandatory_documents": ["Documented context analysis (PESTLE or equivalent)", "List of internal and external issues affecting the EnMS"], "mandatory_records": ["Records of context review meetings", "Evidence of periodic context reassessment"], "audit_questions": ["How has the organisation identified external issues affecting energy performance?", "What internal issues are relevant to the EnMS?", "How often is the context analysis reviewed?", "Are context issues considered in EnMS planning?"]},
    {"clause_number": "4.2", "title": "Understanding the needs and expectations of interested parties", "description": "Determine interested parties, their requirements, and compliance obligations.", "parent_clause": "4", "mandatory_documents": ["List of interested parties and their requirements", "Register of compliance obligations"], "mandatory_records": ["Records of stakeholder consultation", "Evidence of compliance obligation identification"], "audit_questions": ["Who are the interested parties relevant to the EnMS?", "What are their energy management requirements?", "Which requirements are compliance obligations?", "How are changes in requirements tracked?"]},
    {"clause_number": "4.3", "title": "Determining the scope of the energy management system", "description": "Determine the boundaries and applicability of the EnMS.", "parent_clause": "4", "mandatory_documents": ["Documented EnMS scope statement", "Scope boundary diagram or description"], "mandatory_records": ["Records of scope determination rationale", "Records of scope exclusions with justification"], "audit_questions": ["What is the defined scope of the EnMS?", "What boundaries have been established?", "Are there exclusions and how are they justified?", "Is the scope documented and available?"]},
    {"clause_number": "4.4", "title": "Energy management system", "description": "Establish, implement, maintain, and continually improve an EnMS.", "parent_clause": "4", "mandatory_documents": ["EnMS manual or system description", "Process interaction map or diagram"], "mandatory_records": ["Evidence of EnMS establishment", "Records of process interaction definition"], "audit_questions": ["Has the organisation established the EnMS?", "Are EnMS processes and interactions defined?", "How is continual improvement ensured?", "Is the EnMS aligned with ISO 50001:2018?"]},
    {"clause_number": "5.1", "title": "Leadership and commitment", "description": "Top management shall demonstrate leadership and commitment.", "parent_clause": "5", "mandatory_documents": ["Evidence of top management commitment", "Communication of energy management importance"], "mandatory_records": ["Management review minutes showing leadership", "Records of resource allocation by top management", "Records of energy performance accountability"], "audit_questions": ["How does top management demonstrate leadership?", "Is top management accountable for EnMS effectiveness?", "Are energy targets compatible with strategic direction?", "What evidence shows continual improvement promotion?", "How are adequate resources ensured?"]},
    {"clause_number": "5.2", "title": "Energy policy", "description": "Establish energy policy with commitment to continual improvement.", "parent_clause": "5", "mandatory_documents": ["Documented energy policy", "Evidence of policy communication"], "mandatory_records": ["Records of policy review and approval", "Records of policy communication", "Records of policy availability to interested parties"], "audit_questions": ["Does the policy include commitment to continual improvement?", "Does it provide a framework for objectives and targets?", "Is the policy communicated within the organisation?", "Is the policy available to interested parties?", "When was the policy last reviewed?"]},
    {"clause_number": "5.3", "title": "Organisational roles, responsibilities and authorities", "description": "Assign responsibility and authority for the EnMS.", "parent_clause": "5", "mandatory_documents": ["Documented roles and responsibilities for the EnMS", "Organisation chart with energy management structure", "Energy management representative appointment"], "mandatory_records": ["Records of role assignments and acceptance", "Records of authority delegations", "Evidence of energy management team meetings"], "audit_questions": ["Has an energy management team been established?", "Are roles and authorities clearly defined?", "Who is the energy management representative?", "How does the team report to top management?"]},
    {"clause_number": "6.1", "title": "Actions to address risks and opportunities", "description": "Determine risks and opportunities that need to be addressed.", "parent_clause": "6", "mandatory_documents": ["Risk and opportunity register", "Risk assessment methodology", "Action plans for risks and opportunities"], "mandatory_records": ["Records of risk identification and evaluation", "Records of risk treatment decisions", "Evidence of actions taken"], "audit_questions": ["How are risks and opportunities identified?", "What methodology is used for risk assessment?", "What actions address identified risks?", "How is effectiveness of actions evaluated?"]},
    {"clause_number": "6.2", "title": "Objectives, energy targets and planning to achieve them", "description": "Establish energy objectives and targets at relevant functions.", "parent_clause": "6", "mandatory_documents": ["Documented energy objectives and targets", "Action plans for objectives", "Method for evaluating results against targets"], "mandatory_records": ["Records of objective setting and approval", "Records of progress against targets", "Evidence of action plan implementation"], "audit_questions": ["Are objectives established at relevant functions?", "Are objectives consistent with the energy policy?", "Are targets measurable and time-bound?", "What action plans exist?", "How is progress monitored?"]},
    {"clause_number": "6.3", "title": "Energy review", "description": "Analyse energy use, identify SEUs, and identify improvement opportunities.", "parent_clause": "6", "mandatory_documents": ["Documented energy review methodology", "Energy review report", "Register of significant energy uses (SEUs)", "Register of energy improvement opportunities"], "mandatory_records": ["Records of energy data analysis", "Records of SEU determination criteria", "Records of relevant variables affecting SEUs", "Records of current energy performance of SEUs"], "audit_questions": ["How is the energy review conducted and how often?", "What data sources are used?", "How are SEUs identified and determined?", "What relevant variables are identified?", "What improvement opportunities were found?", "How is the energy review kept up to date?"]},
    {"clause_number": "6.4", "title": "Energy baseline", "description": "Establish energy baselines using information from the energy review.", "parent_clause": "6", "mandatory_documents": ["Documented energy baselines with methodology", "Baseline adjustment criteria and procedures", "Static factor documentation"], "mandatory_records": ["Records of baseline data and period selection", "Records of baseline normalisation calculations", "Records of baseline adjustments and justification"], "audit_questions": ["What energy baselines have been established?", "What data period was used and why?", "How are baselines normalised?", "When are baselines adjusted?", "What static factors are identified?"]},
    {"clause_number": "6.5", "title": "Energy performance indicators", "description": "Determine EnPIs appropriate for measuring energy performance.", "parent_clause": "6", "mandatory_documents": ["Documented EnPIs with calculation methodology", "EnPI validation methodology", "EnPI reporting format and frequency"], "mandatory_records": ["Records of EnPI values over time", "Records of EnPI validation results", "Records of EnPI review decisions"], "audit_questions": ["What EnPIs have been established?", "How do they demonstrate improvement?", "What calculation methodology is used?", "How are EnPIs statistically validated?", "How often are EnPIs reviewed?"]},
    {"clause_number": "6.6", "title": "Planning for collection of energy data", "description": "Ensure key characteristics affecting energy performance are measured.", "parent_clause": "6", "mandatory_documents": ["Energy data collection plan", "Metering and monitoring plan", "Data management procedures"], "mandatory_records": ["Records of energy consumption data", "Records of relevant variable measurements", "Meter calibration and maintenance records", "Data quality assessment records"], "audit_questions": ["What data collection plan exists?", "What metering infrastructure is in place?", "How is data quality assured?", "Are meters calibrated on schedule?", "How is energy data protected?"]},
    {"clause_number": "7.1", "title": "Resources", "description": "Provide resources needed for the EnMS.", "parent_clause": "7", "mandatory_documents": ["Resource allocation plan for the EnMS", "Budget documentation for energy management"], "mandatory_records": ["Records of resource allocation and expenditure", "Records of resource adequacy assessments"], "audit_questions": ["What resources have been allocated?", "Is the budget adequate?", "How are resource needs identified?", "Are human and financial resources sufficient?"]},
    {"clause_number": "7.2", "title": "Competence", "description": "Ensure persons affecting energy performance are competent.", "parent_clause": "7", "mandatory_documents": ["Competence requirements for energy-related roles", "Training plan for energy management"], "mandatory_records": ["Training records and certificates", "Competence assessment records", "Records of actions to acquire competence"], "audit_questions": ["How are competence requirements determined?", "What training programmes exist?", "How is competence evaluated?", "What happens when gaps are identified?"]},
    {"clause_number": "7.3", "title": "Awareness", "description": "Ensure personnel are aware of the energy policy and their contribution.", "parent_clause": "7", "mandatory_documents": ["Energy awareness programme documentation", "Communication materials"], "mandatory_records": ["Records of awareness sessions and attendance", "Records of awareness effectiveness evaluation"], "audit_questions": ["How are personnel made aware of the energy policy?", "Do personnel understand their contribution?", "How is awareness maintained?", "Can staff explain implications of non-conformance?"]},
    {"clause_number": "7.4", "title": "Communication", "description": "Determine internal and external communications for the EnMS.", "parent_clause": "7", "mandatory_documents": ["Communication plan/procedure for the EnMS", "Templates for energy communications"], "mandatory_records": ["Records of internal energy communications", "Records of external communications"], "audit_questions": ["What is the communication plan?", "How is energy performance communicated internally?", "What external communications are made?", "How is feedback processed?"]},
    {"clause_number": "7.5", "title": "Documented information", "description": "Include and control documented information required by the standard.", "parent_clause": "7", "mandatory_documents": ["Document control procedure", "Master list of EnMS documents", "Document retention and disposal procedure"], "mandatory_records": ["Records of document approvals and revisions", "Document distribution records", "Records of obsolete document management"], "audit_questions": ["What document control procedure exists?", "How is information created and approved?", "Is there a master document list?", "How are documents stored and protected?", "What is the retention policy?"]},
    {"clause_number": "8.1", "title": "Operational planning and control", "description": "Plan and control processes to meet requirements.", "parent_clause": "8", "mandatory_documents": ["Operational control procedures for SEUs", "Maintenance procedures affecting energy performance", "Procurement specifications for energy equipment", "Operating criteria for SEU-related processes"], "mandatory_records": ["Records of operational control implementation", "Maintenance logs for energy-significant equipment", "Records of deviations from operational criteria", "Commissioning records for new equipment"], "audit_questions": ["How are operations planned to achieve objectives?", "What operational criteria exist for SEUs?", "How is maintenance managed for energy equipment?", "How are deviations detected and addressed?", "What procurement criteria ensure efficiency?", "How is design of new facilities considered?"]},
    {"clause_number": "8.2", "title": "Design", "description": "Consider energy performance in design of new or modified facilities.", "parent_clause": "8", "mandatory_documents": ["Design review procedure with energy criteria", "Energy performance specifications", "Life cycle cost analysis methodology"], "mandatory_records": ["Records of design reviews for energy performance", "Energy impact assessments for design changes", "Life cycle cost analysis records"], "audit_questions": ["How is energy considered in design of new facilities?", "Are improvement opportunities evaluated during design?", "What criteria assess energy impact of changes?", "Are life cycle costs considered?", "How are design review outcomes documented?"]},
    {"clause_number": "9.1", "title": "Monitoring, measurement, analysis and evaluation", "description": "Monitor, measure, analyse and evaluate energy performance.", "parent_clause": "9", "mandatory_documents": ["Monitoring and measurement plan", "Analysis and evaluation methodology", "EnPI calculation and comparison methodology", "Investigation procedure for significant deviations"], "mandatory_records": ["Records of monitoring and measurement results", "Energy performance evaluation reports", "EnPI comparison results against baselines", "Records of deviation investigations", "Equipment calibration records"], "audit_questions": ["What is monitored for energy performance?", "What methods are used for measurement?", "How often is performance analysed?", "How are EnPIs compared to baselines?", "How are deviations investigated?", "How is measurement equipment calibrated?"]},
    {"clause_number": "9.2", "title": "Internal audit", "description": "Conduct internal audits at planned intervals.", "parent_clause": "9", "mandatory_documents": ["Internal audit programme", "Internal audit procedure", "Audit checklists aligned with ISO 50001"], "mandatory_records": ["Audit plans and schedules", "Audit reports with findings", "Records of auditor competence", "Records of corrective actions from audits", "Records of audit programme review"], "audit_questions": ["Is there a planned audit programme?", "How frequently are audits conducted?", "Are criteria, scope, and methods defined?", "Are auditors objective and impartial?", "How are findings reported to management?", "How are corrective actions tracked?"]},
    {"clause_number": "9.3", "title": "Management review", "description": "Top management shall review the EnMS at planned intervals.", "parent_clause": "9", "mandatory_documents": ["Management review procedure", "Review input template covering all required items"], "mandatory_records": ["Management review meeting minutes", "Records of decisions and actions", "Records of energy policy review", "Records of EnPI results presented", "Records of resource adequacy assessment"], "audit_questions": ["How often does top management review the EnMS?", "Are all required inputs included?", "Are EnPI trends reviewed?", "What decisions result from reviews?", "Is the energy policy reviewed for suitability?", "Are resource needs assessed?"]},
    {"clause_number": "10.1", "title": "Nonconformity and corrective action", "description": "React to nonconformities, determine root causes, implement corrections.", "parent_clause": "10", "mandatory_documents": ["Nonconformity and corrective action procedure", "Root cause analysis methodology"], "mandatory_records": ["Nonconformity reports", "Root cause analysis records", "Corrective action records with effectiveness review", "Records of EnMS changes from corrective actions"], "audit_questions": ["What is the NC management procedure?", "How are root causes determined?", "How are corrective actions implemented?", "How is effectiveness verified?", "Are NC records complete and current?", "Have EnMS changes resulted from CAs?"]},
    {"clause_number": "10.2", "title": "Continual improvement", "description": "Continually improve the EnMS and demonstrate energy performance improvement.", "parent_clause": "10", "mandatory_documents": ["Continual improvement plan or programme", "Methodology for demonstrating improvement"], "mandatory_records": ["Records demonstrating continual improvement", "EnPI trend data showing improvement", "Records of improvement initiatives and outcomes", "Year-on-year energy performance comparisons"], "audit_questions": ["How does the organisation demonstrate continual improvement?", "What evidence shows improvement over time?", "How are initiatives identified and prioritised?", "Are EnPI trends showing sustained improvement?", "How is EnMS improvement demonstrated?"]},
]

def _build_clause_tree() -> List[ISO50001Clause]:
    """Build complete ISO 50001:2018 clause tree."""
    return [ISO50001Clause(**d) for d in _CLAUSE_DEFS]

_MANDATORY_DOC_DEFS: List[Dict[str, Any]] = [
    {"document_type": DocumentType.REPORT, "title": "Context analysis (internal and external issues)", "clause_references": ["4.1"], "required": True},
    {"document_type": DocumentType.REGISTER, "title": "Interested parties register and compliance obligations", "clause_references": ["4.2"], "required": True},
    {"document_type": DocumentType.POLICY, "title": "EnMS scope statement", "clause_references": ["4.3"], "required": True},
    {"document_type": DocumentType.MANUAL, "title": "EnMS manual or system description", "clause_references": ["4.4"], "required": True},
    {"document_type": DocumentType.POLICY, "title": "Energy policy", "clause_references": ["5.2"], "required": True},
    {"document_type": DocumentType.PROCEDURE, "title": "Roles, responsibilities and authorities documentation", "clause_references": ["5.3"], "required": True},
    {"document_type": DocumentType.REGISTER, "title": "Risk and opportunity register", "clause_references": ["6.1"], "required": True},
    {"document_type": DocumentType.PLAN, "title": "Energy objectives and targets with action plans", "clause_references": ["6.2"], "required": True},
    {"document_type": DocumentType.REPORT, "title": "Energy review report", "clause_references": ["6.3"], "required": True},
    {"document_type": DocumentType.REGISTER, "title": "Significant energy uses (SEU) register", "clause_references": ["6.3"], "required": True},
    {"document_type": DocumentType.REPORT, "title": "Energy baseline(s) documentation", "clause_references": ["6.4"], "required": True},
    {"document_type": DocumentType.REPORT, "title": "Energy performance indicators (EnPIs) documentation", "clause_references": ["6.5"], "required": True},
    {"document_type": DocumentType.PLAN, "title": "Energy data collection plan", "clause_references": ["6.6"], "required": True},
    {"document_type": DocumentType.PLAN, "title": "Resource allocation plan", "clause_references": ["7.1"], "required": True},
    {"document_type": DocumentType.RECORD, "title": "Competence and training records", "clause_references": ["7.2"], "required": True},
    {"document_type": DocumentType.PROCEDURE, "title": "Energy awareness programme", "clause_references": ["7.3"], "required": True},
    {"document_type": DocumentType.PROCEDURE, "title": "Communication procedure", "clause_references": ["7.4"], "required": True},
    {"document_type": DocumentType.PROCEDURE, "title": "Document control procedure", "clause_references": ["7.5"], "required": True},
    {"document_type": DocumentType.PROCEDURE, "title": "Operational control procedures for SEUs", "clause_references": ["8.1"], "required": True},
    {"document_type": DocumentType.PROCEDURE, "title": "Procurement specifications for energy-related equipment", "clause_references": ["8.1"], "required": True},
    {"document_type": DocumentType.PROCEDURE, "title": "Design review procedure (energy performance criteria)", "clause_references": ["8.2"], "required": True},
    {"document_type": DocumentType.PLAN, "title": "Monitoring and measurement plan", "clause_references": ["9.1"], "required": True},
    {"document_type": DocumentType.PROCEDURE, "title": "Internal audit programme and procedure", "clause_references": ["9.2"], "required": True},
    {"document_type": DocumentType.PROCEDURE, "title": "Management review procedure", "clause_references": ["9.3"], "required": True},
    {"document_type": DocumentType.PROCEDURE, "title": "Nonconformity and corrective action procedure", "clause_references": ["10.1"], "required": True},
    {"document_type": DocumentType.PLAN, "title": "Continual improvement plan", "clause_references": ["10.2"], "required": True},
]

class ComplianceCheckerEngine:
    """ISO 50001:2018 Clauses 4-10 gap analysis and certification readiness engine.

    Assesses every clause against evidence and documentation, identifies
    nonconformities, calculates weighted compliance scores, determines
    certification readiness, and generates gap-analysis reports.

    All arithmetic uses Decimal. Every result carries SHA-256 provenance.

    Usage::

        engine = ComplianceCheckerEngine()
        result = engine.run_full_assessment(
            organization_name="Acme Ltd", scope="Main facility",
            evidence_map={"4.1": ["PESTLE analysis"]},
            document_map={"4.1": ["Context report"]},
            available_documents=[{"title": "Context report", "type": "report", "current": True}],
        )
        assert result.provenance_hash != ""
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise ComplianceCheckerEngine."""
        self.config = config or {}
        self._ev_wt = _decimal(self.config.get("evidence_weight", EVIDENCE_WEIGHT))
        self._doc_wt = _decimal(self.config.get("document_weight", DOCUMENT_WEIGHT))
        self._cat_wts: Dict[str, Decimal] = {}
        for c, w in self.config.get("category_weights", CATEGORY_WEIGHTS).items():
            self._cat_wts[str(c)] = _decimal(w)
        self._nc_auto = bool(self.config.get("nc_auto_generate", True))
        self._assessor = str(self.config.get("assessor_name", "system"))
        self._clauses = _build_clause_tree()
        self._idx: Dict[str, ISO50001Clause] = {c.clause_number: c for c in self._clauses}
        self._mandatory_docs = list(_MANDATORY_DOC_DEFS)
        logger.info("ComplianceCheckerEngine v%s init (clauses=%d)", self.engine_version, len(self._clauses))

    def assess_clause(self, clause_number: str, evidence: List[str], documents: List[str]) -> ClauseAssessment:
        """Assess a single ISO 50001 clause.

        Args:
            clause_number: Clause to assess (e.g. '4.1').
            evidence: Evidence items provided.
            documents: Document titles provided.

        Returns:
            ClauseAssessment with score, status, gaps, recommendations.

        Raises:
            ValueError: If clause_number not found.
        """
        t0 = time.perf_counter()
        cn = str(clause_number).strip()
        if cn not in self._idx:
            raise ValueError(f"Clause '{cn}' not found. Valid: {sorted(self._idx.keys())}")
        clause = self._idx[cn]
        ev_score = self._calc_ev_score(clause, evidence)
        doc_score = self._calc_doc_score(clause, documents)
        combined = self._ev_wt * ev_score + self._doc_wt * doc_score
        combined = _round_val(min(combined, Decimal("100")), 2)
        status = self._status_from_score(combined)
        gaps = self._find_gaps(clause, evidence, documents)
        recs = self._make_recs(clause, combined, gaps)
        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.debug("Clause %s: %s, %s (%.1f ms)", cn, combined, status.value, elapsed)
        return ClauseAssessment(clause=clause, status=status, score=combined, evidence=evidence, gaps=gaps, recommendations=recs, assessed_date=_today_str(), assessor=self._assessor)

    def assess_all_clauses(self, evidence_map: Dict[str, List[str]], document_map: Dict[str, List[str]]) -> List[ClauseAssessment]:
        """Assess all clauses. Returns list of ClauseAssessment for all 25 clauses."""
        t0 = time.perf_counter()
        results = [self.assess_clause(c.clause_number, evidence_map.get(c.clause_number, []), document_map.get(c.clause_number, [])) for c in self._clauses]
        logger.info("Assessed %d clauses (%.1f ms)", len(results), (time.perf_counter() - t0) * 1000)
        return results

    def check_documents(self, available_documents: List[Dict[str, Any]]) -> List[DocumentChecklist]:
        """Check document completeness against mandatory requirements."""
        t0 = time.perf_counter()
        lookup: Dict[str, Dict[str, Any]] = {}
        for doc in available_documents:
            k = str(doc.get("title", "")).strip().lower()
            if k:
                lookup[k] = doc
        checklist: List[DocumentChecklist] = []
        for m in self._mandatory_docs:
            title = str(m["title"])
            k = title.strip().lower()
            matched = lookup.get(k) or self._fuzzy_match(k, lookup)
            exists = matched is not None
            current = bool(matched.get("current", False)) if matched else False
            adequate = bool(matched.get("adequate", current)) if matched else False
            notes = str(matched.get("notes", "")) if matched else "Not found"
            checklist.append(DocumentChecklist(document_type=m["document_type"], title=title, clause_references=m["clause_references"], required=m["required"], exists=exists, current=current, adequate=adequate, notes=notes))
        logger.info("Doc check: %d/%d exist (%.1f ms)", sum(1 for c in checklist if c.exists), len(checklist), (time.perf_counter() - t0) * 1000)
        return checklist

    def identify_nonconformities(self, clause_assessments: List[ClauseAssessment]) -> List[Nonconformity]:
        """Identify nonconformities from clause assessments."""
        t0 = time.perf_counter()
        ncs: List[Nonconformity] = []
        counter = 0
        for a in clause_assessments:
            if a.status in (ClauseStatus.NOT_APPLICABLE, ClauseStatus.COMPLIANT):
                continue
            sev = self._nc_severity(a.score)
            if sev is None:
                continue
            counter += 1
            gap_text = "; ".join(a.gaps) if a.gaps else "Insufficient evidence and documentation"
            ncs.append(Nonconformity(nc_id=f"NC-{counter:04d}", clause_reference=a.clause.clause_number, severity=sev, description=f"Clause {a.clause.clause_number} ({a.clause.title}): {gap_text}", objective_evidence=f"Score: {a.score}/100. Evidence: {len(a.evidence)}. Gaps: {len(a.gaps)}."))
        logger.info("NCs: %d (%.1f ms)", len(ncs), (time.perf_counter() - t0) * 1000)
        return ncs

    def calculate_compliance_score(self, clause_assessments: List[ClauseAssessment]) -> ComplianceScore:
        """Calculate overall and category compliance scores."""
        t0 = time.perf_counter()
        cs: Dict[str, Decimal] = {a.clause.clause_number: a.score for a in clause_assessments}
        cat_scores: Dict[str, Decimal] = {}
        for cat in self._cat_wts:
            nums = self._clauses_for_cat(cat)
            vals = [cs[n] for n in nums if n in cs]
            cat_scores[cat] = _round_val(_safe_divide(sum(vals, Decimal("0")), _decimal(len(vals))), 2) if vals else Decimal("0")
        overall = Decimal("0")
        tw = Decimal("0")
        for cat, s in cat_scores.items():
            w = self._cat_wts.get(cat, Decimal("0"))
            overall += s * w
            tw += w
        if tw > Decimal("0"):
            overall = _safe_divide(overall, tw)
        overall = _round_val(overall, 2)
        crit = [f"Clause {a.clause.clause_number} ({a.clause.title}): score {a.score}/100" for a in clause_assessments if a.status != ClauseStatus.NOT_APPLICABLE and a.score < Decimal("50")]
        strengths = [f"Clause {a.clause.clause_number} ({a.clause.title}): score {a.score}/100" for a in clause_assessments if a.status != ClauseStatus.NOT_APPLICABLE and a.score >= Decimal("80")]
        readiness = self._readiness_from_score(overall)
        logger.info("Score: %.2f, readiness=%s (%.1f ms)", float(overall), readiness.value, (time.perf_counter() - t0) * 1000)
        return ComplianceScore(overall_score=overall, clause_scores=cs, category_scores=cat_scores, readiness=readiness, critical_gaps=crit, strengths=strengths)

    def assess_certification_readiness(self, score: ComplianceScore, nonconformities: List[Nonconformity]) -> CertificationReadiness:
        """Determine certification readiness from score and NCs."""
        overall = score.overall_score
        open_ncs = [nc for nc in nonconformities if nc.correction_status not in (CorrectionStatus.VERIFIED, CorrectionStatus.CLOSED)]
        crit = sum(1 for nc in open_ncs if nc.severity == NonconformitySeverity.CRITICAL)
        major = sum(1 for nc in open_ncs if nc.severity == NonconformitySeverity.MAJOR)
        minor = sum(1 for nc in open_ncs if nc.severity == NonconformitySeverity.MINOR)
        if overall < READINESS_NOT_READY or crit > 0 or major > 0:
            return CertificationReadiness.NOT_READY
        if overall < READINESS_NEEDS_WORK or minor > MAX_MINOR_NCS_NEARLY_READY:
            return CertificationReadiness.NEEDS_WORK
        if overall < READINESS_NEARLY_READY:
            return CertificationReadiness.NEARLY_READY
        if len(open_ncs) == 0:
            return CertificationReadiness.READY
        return CertificationReadiness.NEARLY_READY

    def generate_gap_analysis(self, clause_assessments: List[ClauseAssessment]) -> List[Dict[str, Any]]:
        """Generate prioritised gap analysis report."""
        t0 = time.perf_counter()
        items: List[Dict[str, Any]] = []
        for a in clause_assessments:
            if a.status in (ClauseStatus.NOT_APPLICABLE, ClauseStatus.COMPLIANT):
                continue
            cn = a.clause.clause_number
            cat = self._cat_for_clause(cn)
            wt = self._cat_wts.get(cat, Decimal("0.10"))
            gap = Decimal("100") - a.score
            pri = _round_val(gap * wt * Decimal("100"), 2)
            items.append({"clause_number": cn, "title": a.clause.title, "score": str(a.score), "status": a.status.value, "category": cat, "category_weight": str(wt), "priority_score": str(pri), "gaps": a.gaps, "recommendations": a.recommendations, "evidence_count": len(a.evidence)})
        items.sort(key=lambda x: Decimal(x["priority_score"]), reverse=True)
        logger.info("Gap analysis: %d items (%.1f ms)", len(items), (time.perf_counter() - t0) * 1000)
        return items

    def run_full_assessment(self, organization_name: str, scope: str, evidence_map: Dict[str, List[str]], document_map: Dict[str, List[str]], available_documents: List[Dict[str, Any]], assessment_type: AssessmentType = AssessmentType.INITIAL_GAP) -> ComplianceResult:
        """Run complete ISO 50001 compliance assessment.

        Raises:
            ValueError: If organization_name or scope is empty.
        """
        t0 = time.perf_counter()
        if not organization_name or not organization_name.strip():
            raise ValueError("organization_name must not be empty")
        if not scope or not scope.strip():
            raise ValueError("scope must not be empty")
        logger.info("Running %s for '%s'", assessment_type.value, organization_name)
        ca = self.assess_all_clauses(evidence_map, document_map)
        dc = self.check_documents(available_documents)
        ncs = self.identify_nonconformities(ca) if self._nc_auto else []
        cs = self.calculate_compliance_score(ca)
        readiness = self.assess_certification_readiness(cs, ncs)
        cs.readiness = readiness
        findings = self._build_findings(ca, ncs)
        gap = self.generate_gap_analysis(ca)
        recs = self._compile_recs(gap, ncs, cs)
        ns = self._next_steps(readiness, ncs)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        result = ComplianceResult(assessment_id=_new_uuid(), assessment_type=assessment_type, organization_name=organization_name.strip(), scope=scope.strip(), assessed_date=_today_str(), clause_assessments=ca, document_checklist=dc, nonconformities=ncs, findings=findings, compliance_score=cs, recommendations=recs, next_steps=ns, provenance_hash="", calculation_time_ms=elapsed_ms)
        result.provenance_hash = _compute_hash(result)
        logger.info("Done: score=%.2f, readiness=%s, ncs=%d (%d ms)", float(cs.overall_score), readiness.value, len(ncs), elapsed_ms)
        return result

    # -- Private scoring --

    def _calc_ev_score(self, clause: ISO50001Clause, evidence: List[str]) -> Decimal:
        """Evidence coverage score (0-100)."""
        if not evidence:
            return Decimal("0")
        expected = len(clause.audit_questions) + len(clause.mandatory_records)
        if expected == 0:
            return Decimal("100")
        cov = min(len(evidence), expected)
        return _round_val(min(_safe_pct(_decimal(cov), _decimal(expected)), Decimal("100")), 2)

    def _calc_doc_score(self, clause: ISO50001Clause, documents: List[str]) -> Decimal:
        """Document coverage score (0-100)."""
        if not documents:
            return Decimal("0")
        expected = len(clause.mandatory_documents)
        if expected == 0:
            return Decimal("100")
        cov = min(len(documents), expected)
        return _round_val(min(_safe_pct(_decimal(cov), _decimal(expected)), Decimal("100")), 2)

    def _status_from_score(self, score: Decimal) -> ClauseStatus:
        """Clause status from score."""
        if score >= Decimal("90"):
            return ClauseStatus.COMPLIANT
        if score >= Decimal("50"):
            return ClauseStatus.PARTIALLY_COMPLIANT
        if score >= Decimal("25"):
            return ClauseStatus.IN_PROGRESS
        return ClauseStatus.NOT_STARTED

    def _nc_severity(self, score: Decimal) -> Optional[NonconformitySeverity]:
        """NC severity from score. None if compliant."""
        if score < NC_THRESHOLD_CRITICAL:
            return NonconformitySeverity.CRITICAL
        if score < NC_THRESHOLD_MAJOR:
            return NonconformitySeverity.MAJOR
        if score < NC_THRESHOLD_MINOR:
            return NonconformitySeverity.MINOR
        if score < NC_THRESHOLD_OBS:
            return NonconformitySeverity.OBSERVATION
        return None

    def _readiness_from_score(self, overall: Decimal) -> CertificationReadiness:
        """Preliminary readiness from score alone."""
        if overall < READINESS_NOT_READY:
            return CertificationReadiness.NOT_READY
        if overall < READINESS_NEEDS_WORK:
            return CertificationReadiness.NEEDS_WORK
        if overall < READINESS_NEARLY_READY:
            return CertificationReadiness.NEARLY_READY
        return CertificationReadiness.READY

    # -- Private gap/rec helpers --

    def _find_gaps(self, clause: ISO50001Clause, evidence: List[str], documents: List[str]) -> List[str]:
        """Identify specific gaps for a clause."""
        gaps: List[str] = []
        nd = len(documents)
        rd = len(clause.mandatory_documents)
        if nd < rd:
            gaps.append(f"{rd - nd} of {rd} mandatory documents not provided")
            for i, name in enumerate(clause.mandatory_documents):
                if i >= nd:
                    gaps.append(f"Missing document: {name}")
        rem_ev = max(len(evidence) - rd, 0)
        rr = len(clause.mandatory_records)
        if rem_ev < rr:
            gaps.append(f"{rr - rem_ev} of {rr} mandatory records not evidenced")
        if not evidence and not documents:
            gaps.append(f"No evidence or documentation for Clause {clause.clause_number} ({clause.title})")
        return gaps

    def _make_recs(self, clause: ISO50001Clause, score: Decimal, gaps: List[str]) -> List[str]:
        """Generate recommendations for a clause."""
        recs: List[str] = []
        cn = clause.clause_number
        if score < Decimal("25"):
            recs.append(f"Priority: Commence Clause {cn} ({clause.title}) immediately")
            recs.append(f"Assign responsible person for Clause {cn}")
            for doc in clause.mandatory_documents:
                recs.append(f"Create: {doc}")
        elif score < Decimal("50"):
            recs.append(f"Complete remaining documentation for Clause {cn}")
            recs.append(f"Gather additional evidence for Clause {cn}")
        elif score < Decimal("75"):
            recs.append(f"Address remaining gaps in Clause {cn}")
            if gaps:
                recs.append(f"Specific: {gaps[0]}")
        elif score < Decimal("90"):
            recs.append(f"Fine-tune Clause {cn} for full compliance")
            recs.append(f"Verify all records are current for Clause {cn}")
        return recs

    # -- Private category helpers --

    def _cat_for_clause(self, clause_number: str) -> str:
        """Category name for a clause number."""
        return CLAUSE_CATEGORY_MAP.get(clause_number.split(".")[0], "context")

    def _clauses_for_cat(self, category: str) -> List[str]:
        """Clause numbers belonging to a category."""
        tops = [cn for cn, cat in CLAUSE_CATEGORY_MAP.items() if cat == category]
        return [c.clause_number for c in self._clauses if c.clause_number.split(".")[0] in tops]

    # -- Private document matching --

    def _fuzzy_match(self, target: str, lookup: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Fuzzy match document title by >=50% word overlap."""
        tw = set(target.split())
        best: Optional[Dict[str, Any]] = None
        best_n = 0
        for k, doc in lookup.items():
            aw = set(k.split())
            overlap = len(tw & aw)
            mn = min(len(tw), len(aw))
            if mn > 0 and overlap / mn >= 0.5 and overlap > best_n:
                best_n = overlap
                best = doc
        return best

    # -- Private findings --

    def _build_findings(self, assessments: List[ClauseAssessment], ncs: List[Nonconformity]) -> List[AuditFinding]:
        """Generate audit findings from assessments and NCs."""
        findings: List[AuditFinding] = []
        nc_map: Dict[str, Nonconformity] = {nc.clause_reference: nc for nc in ncs}
        ctr = 0
        for a in assessments:
            if a.status == ClauseStatus.NOT_APPLICABLE:
                continue
            ctr += 1
            cn = a.clause.clause_number
            if a.status == ClauseStatus.COMPLIANT:
                findings.append(AuditFinding(finding_id=f"F-{ctr:04d}", clause_reference=cn, finding_type="positive", description=f"Clause {cn} ({a.clause.title}) fully compliant, score {a.score}/100.", evidence=f"{len(a.evidence)} evidence items"))
            elif cn in nc_map:
                nc = nc_map[cn]
                ft = "nc_major" if nc.severity in (NonconformitySeverity.CRITICAL, NonconformitySeverity.MAJOR) else ("nc_minor" if nc.severity == NonconformitySeverity.MINOR else "observation")
                findings.append(AuditFinding(finding_id=f"F-{ctr:04d}", clause_reference=cn, finding_type=ft, description=nc.description, evidence=nc.objective_evidence, nonconformity=nc))
            else:
                findings.append(AuditFinding(finding_id=f"F-{ctr:04d}", clause_reference=cn, finding_type="observation", description=f"Clause {cn} ({a.clause.title}) scored {a.score}/100.", evidence=f"{len(a.evidence)} evidence items"))
        return findings

    # -- Private recs & next steps --

    def _compile_recs(self, gap_analysis: List[Dict[str, Any]], ncs: List[Nonconformity], score: ComplianceScore) -> List[str]:
        """Compile prioritised recommendations."""
        recs: List[str] = []
        crit_ncs = [nc for nc in ncs if nc.severity == NonconformitySeverity.CRITICAL]
        major_ncs = [nc for nc in ncs if nc.severity == NonconformitySeverity.MAJOR]
        nc_refs = {nc.clause_reference for nc in crit_ncs + major_ncs}
        for nc in crit_ncs:
            recs.append(f"CRITICAL: Address {nc.nc_id} (Clause {nc.clause_reference}) immediately")
        for nc in major_ncs:
            recs.append(f"MAJOR: Resolve {nc.nc_id} (Clause {nc.clause_reference}) within 30 days")
        for item in gap_analysis[:10]:
            if item["clause_number"] not in nc_refs:
                recs.append(f"Clause {item['clause_number']} ({item['title']}): Score {item['score']}/100, priority {item['priority_score']}")
        for cat, cs in score.category_scores.items():
            if cs < Decimal("50"):
                recs.append(f"Category '{cat}' needs improvement (score: {cs}/100)")
        return recs

    def _next_steps(self, readiness: CertificationReadiness, ncs: List[Nonconformity]) -> List[str]:
        """Determine next steps based on readiness."""
        steps: List[str] = []
        if readiness == CertificationReadiness.NOT_READY:
            steps.extend(["Develop comprehensive EnMS implementation plan", "Assign resources to address critical gaps", "Engage ISO 50001 consultant", "Establish internal energy management team", "Follow-up assessment in 3-6 months"])
        elif readiness == CertificationReadiness.NEEDS_WORK:
            steps.extend(["Focus on closing nonconformities", "Complete missing mandatory documentation", "Conduct internal audits to verify corrective actions", "Follow-up assessment in 2-3 months"])
        elif readiness == CertificationReadiness.NEARLY_READY:
            steps.extend(["Address remaining minor nonconformities", "Conduct full internal audit cycle", "Complete management review with all inputs", "Consider pre-certification (Stage 1) assessment", "Schedule certification audit within 1-2 months"])
        elif readiness == CertificationReadiness.READY:
            steps.extend(["Proceed with Stage 1 certification audit", "Schedule Stage 2 certification audit (on-site)", "Prepare staff for auditor interviews", "Ensure records are readily accessible", "Brief top management on audit expectations"])
        elif readiness == CertificationReadiness.CERTIFIED:
            steps.extend(["Maintain surveillance audit schedule", "Continue monitoring EnPIs", "Prepare for next surveillance/recertification"])
        open_ncs = [nc for nc in ncs if nc.correction_status not in (CorrectionStatus.VERIFIED, CorrectionStatus.CLOSED)]
        if open_ncs:
            steps.append(f"Close {len(open_ncs)} open NCs before certification")
        return steps

    # -- Public utility --

    def get_clause_tree(self) -> List[ISO50001Clause]:
        """Return the complete clause tree."""
        return list(self._clauses)

    def get_clause(self, clause_number: str) -> Optional[ISO50001Clause]:
        """Look up a single clause by number."""
        return self._idx.get(clause_number.strip())

    def get_mandatory_documents(self) -> List[Dict[str, Any]]:
        """Return the mandatory document registry."""
        return list(self._mandatory_docs)

    def get_category_weights(self) -> Dict[str, Decimal]:
        """Return current category weights."""
        return dict(self._cat_wts)

    def get_clause_count(self) -> int:
        """Return total number of clauses."""
        return len(self._clauses)

    def get_categories(self) -> List[str]:
        """Return sorted list of category names."""
        return sorted(self._cat_wts.keys())
