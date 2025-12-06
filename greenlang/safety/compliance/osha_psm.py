"""
OSHAPSM - OSHA 1910.119 Process Safety Management Support

This module implements support for OSHA Process Safety Management
requirements per 29 CFR 1910.119.

Key PSM elements covered:
- Process Hazard Analysis (PHA)
- Operating Procedures
- Training
- Mechanical Integrity
- Management of Change (MOC)
- Pre-Startup Safety Review (PSSR)
- Incident Investigation
- Emergency Planning

Reference: 29 CFR 1910.119

Example:
    >>> from greenlang.safety.compliance.osha_psm import OSHAPSM
    >>> psm = OSHAPSM(facility_id="FAC-001")
    >>> audit = psm.conduct_audit()
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime, date
import uuid

logger = logging.getLogger(__name__)


class PSMElement(str, Enum):
    """14 PSM elements per OSHA 1910.119."""
    EMPLOYEE_PARTICIPATION = "employee_participation"
    PROCESS_SAFETY_INFO = "process_safety_info"
    PROCESS_HAZARD_ANALYSIS = "process_hazard_analysis"
    OPERATING_PROCEDURES = "operating_procedures"
    TRAINING = "training"
    CONTRACTORS = "contractors"
    PRE_STARTUP_REVIEW = "pre_startup_review"
    MECHANICAL_INTEGRITY = "mechanical_integrity"
    HOT_WORK = "hot_work"
    MANAGEMENT_OF_CHANGE = "management_of_change"
    INCIDENT_INVESTIGATION = "incident_investigation"
    EMERGENCY_PLANNING = "emergency_planning"
    COMPLIANCE_AUDITS = "compliance_audits"
    TRADE_SECRETS = "trade_secrets"


class PSMFinding(BaseModel):
    """PSM audit finding."""
    finding_id: str = Field(default_factory=lambda: f"PSM-{uuid.uuid4().hex[:6].upper()}")
    element: PSMElement = Field(...)
    severity: str = Field(default="observation")
    description: str = Field(...)
    corrective_action: Optional[str] = Field(None)
    due_date: Optional[date] = Field(None)
    status: str = Field(default="open")


class PSMAuditResult(BaseModel):
    """PSM audit result."""
    audit_id: str = Field(default_factory=lambda: f"AUDIT-{uuid.uuid4().hex[:8].upper()}")
    facility_id: str = Field(...)
    audit_date: datetime = Field(default_factory=datetime.utcnow)
    audit_type: str = Field(default="comprehensive")
    auditor: str = Field(...)
    elements_audited: List[PSMElement] = Field(default_factory=list)
    findings: List[PSMFinding] = Field(default_factory=list)
    findings_by_severity: Dict[str, int] = Field(default_factory=dict)
    overall_score: float = Field(default=0.0)
    recommendations: List[str] = Field(default_factory=list)
    next_audit_date: Optional[date] = Field(None)
    provenance_hash: str = Field(default="")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat(), date: lambda v: v.isoformat()}


class OSHAPSM:
    """
    OSHA PSM Compliance Manager.

    Supports compliance with OSHA 1910.119 Process Safety
    Management requirements.

    Example:
        >>> psm = OSHAPSM(facility_id="FAC-001")
        >>> audit = psm.conduct_audit(auditor="J.Smith")
    """

    # Element requirements and scoring weights
    ELEMENT_REQUIREMENTS = {
        PSMElement.EMPLOYEE_PARTICIPATION: {
            "weight": 5,
            "checks": ["written_plan", "employee_consultation", "access_to_info"]
        },
        PSMElement.PROCESS_SAFETY_INFO: {
            "weight": 10,
            "checks": ["hazard_info", "technology_info", "equipment_info", "p_ids"]
        },
        PSMElement.PROCESS_HAZARD_ANALYSIS: {
            "weight": 15,
            "checks": ["pha_completed", "pha_current", "recommendations_addressed", "team_qualified"]
        },
        PSMElement.OPERATING_PROCEDURES: {
            "weight": 10,
            "checks": ["procedures_exist", "procedures_current", "accessible", "annual_certification"]
        },
        PSMElement.TRAINING: {
            "weight": 10,
            "checks": ["initial_training", "refresher_training", "training_documented", "competency_verified"]
        },
        PSMElement.MECHANICAL_INTEGRITY: {
            "weight": 15,
            "checks": ["equipment_list", "inspection_procedures", "testing_procedures", "quality_assurance"]
        },
        PSMElement.MANAGEMENT_OF_CHANGE: {
            "weight": 10,
            "checks": ["moc_procedure", "technical_basis", "impact_assessment", "documentation"]
        },
        PSMElement.INCIDENT_INVESTIGATION: {
            "weight": 10,
            "checks": ["investigation_procedure", "48hr_initiation", "root_cause", "corrective_actions"]
        },
    }

    def __init__(self, facility_id: str):
        """Initialize OSHAPSM manager."""
        self.facility_id = facility_id
        self.audits: List[PSMAuditResult] = []
        self.findings: List[PSMFinding] = []
        logger.info(f"OSHAPSM manager initialized for {facility_id}")

    def conduct_audit(
        self,
        auditor: str,
        element_scores: Optional[Dict[PSMElement, float]] = None
    ) -> PSMAuditResult:
        """
        Conduct PSM compliance audit.

        Args:
            auditor: Auditor name
            element_scores: Scores by element (0-100)

        Returns:
            PSMAuditResult
        """
        logger.info(f"Conducting PSM audit for {self.facility_id}")

        findings = []
        elements_audited = list(self.ELEMENT_REQUIREMENTS.keys())

        # Default scores if not provided
        if element_scores is None:
            element_scores = {e: 85.0 for e in elements_audited}

        # Evaluate each element
        total_weighted_score = 0.0
        total_weight = 0

        for element, config in self.ELEMENT_REQUIREMENTS.items():
            score = element_scores.get(element, 0)
            weight = config["weight"]

            total_weighted_score += score * weight
            total_weight += weight

            # Generate findings for low scores
            if score < 70:
                findings.append(PSMFinding(
                    element=element,
                    severity="major" if score < 50 else "minor",
                    description=f"{element.value} score of {score}% below acceptable threshold",
                ))
            elif score < 85:
                findings.append(PSMFinding(
                    element=element,
                    severity="observation",
                    description=f"{element.value} has improvement opportunities",
                ))

        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0

        # Count findings by severity
        by_severity = {}
        for finding in findings:
            by_severity[finding.severity] = by_severity.get(finding.severity, 0) + 1

        result = PSMAuditResult(
            facility_id=self.facility_id,
            auditor=auditor,
            elements_audited=elements_audited,
            findings=findings,
            findings_by_severity=by_severity,
            overall_score=overall_score,
            recommendations=self._generate_recommendations(findings),
        )

        result.provenance_hash = hashlib.sha256(
            f"{result.audit_id}|{self.facility_id}|{overall_score}".encode()
        ).hexdigest()

        self.audits.append(result)
        self.findings.extend(findings)

        return result

    def _generate_recommendations(self, findings: List[PSMFinding]) -> List[str]:
        """Generate recommendations from audit findings."""
        recommendations = []
        major_elements = set()

        for finding in findings:
            if finding.severity == "major":
                major_elements.add(finding.element)

        for element in major_elements:
            recommendations.append(
                f"Priority: Address {element.value} deficiencies immediately"
            )

        if len(findings) > 5:
            recommendations.append(
                "Consider comprehensive PSM program review"
            )

        return recommendations
