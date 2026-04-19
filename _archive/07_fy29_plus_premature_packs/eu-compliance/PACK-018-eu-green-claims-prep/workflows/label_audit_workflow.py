# -*- coding: utf-8 -*-
"""
Label Audit Workflow - PACK-018 EU Green Claims Prep
=====================================================

4-phase workflow that audits environmental labels and eco-label schemes
used by an organisation against the EU Green Claims Directive Article 8
and Article 10 requirements. Inventories all labels in use, validates
each scheme against Article 10 criteria, verifies certificate validity,
and produces a per-label compliance assessment report.

Phases:
    1. LabelInventory          -- Catalogue all eco-labels in use
    2. SchemeValidation        -- Verify Article 10 criteria for each scheme
    3. CertificateVerification -- Validate certificates and expiry status
    4. ComplianceReport        -- Generate per-label compliance assessment

Reference:
    EU Green Claims Directive (COM/2023/166), Articles 8 and 10
    PACK-018 Solution Pack specification

Author: GreenLang Team
Version: 18.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID-4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hex digest for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    """Execution status for a single workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class LabelAuditPhase(str, Enum):
    """Label audit workflow phase identifiers."""
    LABEL_INVENTORY = "LabelInventory"
    SCHEME_VALIDATION = "SchemeValidation"
    CERTIFICATE_VERIFICATION = "CertificateVerification"
    COMPLIANCE_REPORT = "ComplianceReport"

class LabelSchemeType(str, Enum):
    """ISO label scheme type classification."""
    TYPE_I = "type_i_ecolabel"
    TYPE_II = "type_ii_self_declared"
    TYPE_III = "type_iii_epd"
    EU_OFFICIAL = "eu_official"
    PRIVATE = "private_scheme"
    NATIONAL = "national_scheme"
    UNKNOWN = "unknown"

class CertificateStatus(str, Enum):
    """Certificate validity status."""
    VALID = "valid"
    EXPIRED = "expired"
    EXPIRING_SOON = "expiring_soon"
    NOT_PROVIDED = "not_provided"
    REVOKED = "revoked"

# =============================================================================
# DATA MODELS
# =============================================================================

class LabelAuditConfig(BaseModel):
    """Configuration for LabelAuditWorkflow."""
    expiry_warning_days: int = Field(
        default=90, ge=1, le=365,
        description="Days before expiry to trigger expiring_soon status",
    )
    require_third_party_audit: bool = Field(
        default=True, description="Whether third-party audit is mandatory",
    )
    eu_approved_schemes: List[str] = Field(
        default_factory=lambda: [
            "eu_ecolabel", "eu_organic", "energy_label", "eu_flower",
        ],
        description="List of EU-approved scheme identifiers",
    )

class LabelAuditResult(BaseModel):
    """Final result model for per-label audit assessment."""
    label_id: str = Field(..., description="Unique label identifier")
    label_name: str = Field(..., description="Display name of the label")
    scheme_type: str = Field(..., description="ISO label type classification")
    article_10_compliant: bool = Field(..., description="Meets Article 10 criteria")
    certificate_status: str = Field(..., description="Certificate validity status")
    compliance_score: float = Field(..., ge=0.0, le=100.0, description="Per-label score")
    issues: List[str] = Field(default_factory=list, description="Identified issues")
    recommendation: str = Field(default="", description="Remediation recommendation")

class WorkflowInput(BaseModel):
    """Input model for LabelAuditWorkflow."""
    labels_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of label/certification objects to audit",
    )
    entity_name: str = Field(default="", description="Reporting entity name")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    config: Dict[str, Any] = Field(default_factory=dict)

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    result_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = Field(default=None)

class WorkflowResult(BaseModel):
    """Complete result from LabelAuditWorkflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="label_audit")
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    phases: List[PhaseResult] = Field(default_factory=list)
    overall_result: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class LabelAuditWorkflow:
    """
    4-phase label audit workflow for EU Green Claims Directive compliance.

    Catalogues eco-labels in use, validates each scheme against Article 10
    criteria, verifies certificate validity and expiry, and produces a
    per-label compliance assessment with provenance tracking.

    Zero-hallucination: all scoring and classification uses deterministic
    rules and keyword matching. No LLM calls in calculation paths.

    Example:
        >>> wf = LabelAuditWorkflow()
        >>> result = wf.execute(
        ...     labels_data=[{"name": "EU Ecolabel", "scheme": "eu_ecolabel",
        ...                   "certificate_expiry": "2027-01-01"}],
        ... )
        >>> assert result["status"] == "completed"
    """

    WORKFLOW_NAME: str = "label_audit"

    # Article 10 criteria checklist
    ARTICLE_10_CRITERIA: List[str] = [
        "independent_certification_body",
        "publicly_accessible_criteria",
        "transparent_governance",
        "complaint_mechanism",
        "periodic_review",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize LabelAuditWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config: Dict[str, Any] = config or {}
        self.audit_config = LabelAuditConfig(**self.config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> dict:
        """
        Execute the 4-phase label audit pipeline.

        Keyword Args:
            labels_data: List of label/certification dictionaries.
            entity_name: Organisation name.
            reporting_year: Assessment year.

        Returns:
            Serialised WorkflowResult dictionary with provenance hash.
        """
        input_data = WorkflowInput(
            labels_data=kwargs.get("labels_data", []),
            entity_name=kwargs.get("entity_name", ""),
            reporting_year=kwargs.get("reporting_year", 2025),
            config=kwargs.get("config", {}),
        )

        started_at = utcnow()
        self.logger.info("Starting %s workflow %s", self.WORKFLOW_NAME, self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = PhaseStatus.RUNNING

        try:
            # Phase 1 -- Label Inventory
            phase_results.append(self._run_label_inventory(input_data))

            # Phase 2 -- Scheme Validation
            inventory = phase_results[0].result_data
            phase_results.append(self._run_scheme_validation(inventory))

            # Phase 3 -- Certificate Verification
            phase_results.append(self._run_certificate_verification(inventory))

            # Phase 4 -- Compliance Report
            scheme_data = phase_results[1].result_data
            cert_data = phase_results[2].result_data
            phase_results.append(
                self._run_compliance_report(input_data, inventory, scheme_data, cert_data)
            )

            overall_status = PhaseStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Workflow %s failed: %s", self.workflow_id, exc, exc_info=True)
            overall_status = PhaseStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error_capture",
                status=PhaseStatus.FAILED,
                started_at=utcnow(),
                completed_at=utcnow(),
                error_message=str(exc),
            ))

        completed_at = utcnow()

        completed_phases = [p for p in phase_results if p.status == PhaseStatus.COMPLETED]
        overall_result: Dict[str, Any] = {
            "total_labels_audited": len(input_data.labels_data),
            "phases_completed": len(completed_phases),
            "phases_total": 4,
        }
        if phase_results and phase_results[-1].status == PhaseStatus.COMPLETED:
            overall_result.update(phase_results[-1].result_data)

        result = WorkflowResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            phases=phase_results,
            overall_result=overall_result,
            started_at=started_at,
            completed_at=completed_at,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Workflow %s %s in %.1fs -- %d labels audited",
            self.workflow_id,
            overall_status.value,
            (completed_at - started_at).total_seconds(),
            len(input_data.labels_data),
        )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # run_phase dispatcher
    # ------------------------------------------------------------------

    def run_phase(self, phase: LabelAuditPhase, **kwargs: Any) -> PhaseResult:
        """
        Run a single named phase independently.

        Args:
            phase: The LabelAuditPhase to execute.
            **kwargs: Phase-specific keyword arguments.

        Returns:
            PhaseResult for the executed phase.
        """
        dispatch: Dict[LabelAuditPhase, Any] = {
            LabelAuditPhase.LABEL_INVENTORY: lambda: self._run_label_inventory(
                WorkflowInput(labels_data=kwargs.get("labels_data", []))
            ),
            LabelAuditPhase.SCHEME_VALIDATION: lambda: self._run_scheme_validation(
                kwargs.get("inventory", {})
            ),
            LabelAuditPhase.CERTIFICATE_VERIFICATION: lambda: self._run_certificate_verification(
                kwargs.get("inventory", {})
            ),
            LabelAuditPhase.COMPLIANCE_REPORT: lambda: self._run_compliance_report(
                WorkflowInput(labels_data=kwargs.get("labels_data", [])),
                kwargs.get("inventory", {}),
                kwargs.get("scheme_data", {}),
                kwargs.get("cert_data", {}),
            ),
        }
        handler = dispatch.get(phase)
        if handler is None:
            return PhaseResult(
                phase_name=phase.value,
                status=PhaseStatus.FAILED,
                error_message=f"Unknown phase: {phase.value}",
            )
        return handler()

    # ------------------------------------------------------------------
    # Phase 1: Label Inventory
    # ------------------------------------------------------------------

    def _run_label_inventory(self, input_data: WorkflowInput) -> PhaseResult:
        """Catalogue all eco-labels and certifications in use."""
        started = utcnow()
        self.logger.info("Phase 1/4 LabelInventory -- cataloguing %d labels",
                         len(input_data.labels_data))

        inventory: List[Dict[str, Any]] = []
        scheme_types: Dict[str, int] = {t.value: 0 for t in LabelSchemeType}

        for idx, label in enumerate(input_data.labels_data):
            label_type = self._classify_scheme_type(label)
            entry = {
                "label_id": label.get("id", f"lbl-{idx}"),
                "name": label.get("name", "unnamed"),
                "scheme": label.get("scheme", "unknown"),
                "scheme_type": label_type.value,
                "issuer": label.get("issuer", ""),
                "certificate_id": label.get("certificate_id", ""),
                "certificate_expiry": label.get("certificate_expiry", ""),
                "products_covered": label.get("products_covered", []),
                "article_10_metadata": label.get("article_10_metadata", {}),
            }
            inventory.append(entry)
            scheme_types[label_type.value] += 1

        result_data: Dict[str, Any] = {
            "inventory": inventory,
            "total_labels": len(inventory),
            "scheme_type_distribution": {k: v for k, v in scheme_types.items() if v > 0},
            "unique_issuers": len({e["issuer"] for e in inventory if e["issuer"]}),
        }

        return PhaseResult(
            phase_name=LabelAuditPhase.LABEL_INVENTORY.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 2: Scheme Validation
    # ------------------------------------------------------------------

    def _run_scheme_validation(self, inventory_data: Dict[str, Any]) -> PhaseResult:
        """Verify each label scheme against Article 10 criteria."""
        started = utcnow()
        self.logger.info("Phase 2/4 SchemeValidation -- checking Article 10 criteria")

        validations: List[Dict[str, Any]] = []
        compliant_count = 0

        for item in inventory_data.get("inventory", []):
            criteria_results = self._evaluate_article_10(item)
            met_count = sum(1 for v in criteria_results.values() if v)
            total_criteria = len(criteria_results)
            is_compliant = met_count == total_criteria

            if is_compliant:
                compliant_count += 1

            validations.append({
                "label_id": item["label_id"],
                "name": item["name"],
                "scheme_type": item["scheme_type"],
                "criteria_results": criteria_results,
                "criteria_met": met_count,
                "criteria_total": total_criteria,
                "article_10_compliant": is_compliant,
                "compliance_pct": round(
                    (met_count / total_criteria * 100) if total_criteria else 0.0, 1
                ),
            })

        total = len(validations)
        result_data: Dict[str, Any] = {
            "validations": validations,
            "article_10_compliant_count": compliant_count,
            "article_10_non_compliant_count": total - compliant_count,
            "overall_compliance_rate_pct": round(
                (compliant_count / total * 100) if total else 0.0, 1
            ),
        }

        return PhaseResult(
            phase_name=LabelAuditPhase.SCHEME_VALIDATION.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 3: Certificate Verification
    # ------------------------------------------------------------------

    def _run_certificate_verification(self, inventory_data: Dict[str, Any]) -> PhaseResult:
        """Validate certificates and determine expiry status."""
        started = utcnow()
        self.logger.info("Phase 3/4 CertificateVerification -- validating certificates")

        verifications: List[Dict[str, Any]] = []
        status_counts: Dict[str, int] = {s.value: 0 for s in CertificateStatus}
        now = utcnow()

        for item in inventory_data.get("inventory", []):
            cert_status = self._verify_certificate(item, now)
            status_counts[cert_status.value] += 1

            verifications.append({
                "label_id": item["label_id"],
                "name": item["name"],
                "certificate_id": item.get("certificate_id", ""),
                "certificate_expiry": item.get("certificate_expiry", ""),
                "certificate_status": cert_status.value,
                "days_until_expiry": self._days_until_expiry(
                    item.get("certificate_expiry", ""), now,
                ),
                "requires_renewal": cert_status in (
                    CertificateStatus.EXPIRED, CertificateStatus.EXPIRING_SOON,
                ),
            })

        valid_count = status_counts.get(CertificateStatus.VALID.value, 0)
        total = len(verifications)

        result_data: Dict[str, Any] = {
            "verifications": verifications,
            "status_distribution": status_counts,
            "valid_certificates": valid_count,
            "certificates_needing_action": total - valid_count,
            "validity_rate_pct": round(
                (valid_count / total * 100) if total else 0.0, 1
            ),
        }

        return PhaseResult(
            phase_name=LabelAuditPhase.CERTIFICATE_VERIFICATION.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 4: Compliance Report
    # ------------------------------------------------------------------

    def _run_compliance_report(
        self,
        input_data: WorkflowInput,
        inventory_data: Dict[str, Any],
        scheme_data: Dict[str, Any],
        cert_data: Dict[str, Any],
    ) -> PhaseResult:
        """Generate per-label compliance assessment report."""
        started = utcnow()
        self.logger.info("Phase 4/4 ComplianceReport -- assembling per-label assessments")

        scheme_by_id = {
            v["label_id"]: v for v in scheme_data.get("validations", [])
        }
        cert_by_id = {
            v["label_id"]: v for v in cert_data.get("verifications", [])
        }

        assessments: List[Dict[str, Any]] = []
        pass_count = 0

        for item in inventory_data.get("inventory", []):
            label_id = item["label_id"]
            scheme_info = scheme_by_id.get(label_id, {})
            cert_info = cert_by_id.get(label_id, {})

            issues = self._identify_issues(item, scheme_info, cert_info)
            score = self._calculate_compliance_score(scheme_info, cert_info)
            is_pass = score >= 70.0 and len(issues) == 0

            if is_pass:
                pass_count += 1

            audit_result = LabelAuditResult(
                label_id=label_id,
                label_name=item.get("name", ""),
                scheme_type=item.get("scheme_type", LabelSchemeType.UNKNOWN.value),
                article_10_compliant=scheme_info.get("article_10_compliant", False),
                certificate_status=cert_info.get("certificate_status", CertificateStatus.NOT_PROVIDED.value),
                compliance_score=score,
                issues=issues,
                recommendation=self._generate_recommendation(issues, score),
            )
            assessments.append(audit_result.model_dump())

        total = len(assessments)
        result_data: Dict[str, Any] = {
            "report_id": _new_uuid(),
            "entity_name": input_data.entity_name,
            "reporting_year": input_data.reporting_year,
            "assessments": assessments,
            "total_labels_audited": total,
            "labels_passing": pass_count,
            "labels_failing": total - pass_count,
            "pass_rate_pct": round((pass_count / total * 100) if total else 0.0, 1),
            "audit_outcome": self._determine_audit_outcome(pass_count, total),
        }

        return PhaseResult(
            phase_name=LabelAuditPhase.COMPLIANCE_REPORT.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_scheme_type(self, label: Dict[str, Any]) -> LabelSchemeType:
        """Classify label by ISO type or scheme origin."""
        scheme = label.get("scheme", "").lower()
        name = label.get("name", "").lower()

        if scheme in {s.lower() for s in self.audit_config.eu_approved_schemes}:
            return LabelSchemeType.EU_OFFICIAL
        if "type i" in name or "ecolabel" in name or scheme == "type_i":
            return LabelSchemeType.TYPE_I
        if "type ii" in name or "self" in name or scheme == "type_ii":
            return LabelSchemeType.TYPE_II
        if "type iii" in name or "epd" in name or scheme == "type_iii":
            return LabelSchemeType.TYPE_III
        if "national" in scheme:
            return LabelSchemeType.NATIONAL
        if "private" in scheme or label.get("issuer"):
            return LabelSchemeType.PRIVATE
        return LabelSchemeType.UNKNOWN

    def _evaluate_article_10(self, item: Dict[str, Any]) -> Dict[str, bool]:
        """Evaluate Article 10 criteria for a label scheme."""
        metadata = item.get("article_10_metadata", {})
        scheme_type = item.get("scheme_type", "")

        results: Dict[str, bool] = {}
        for criterion in self.ARTICLE_10_CRITERIA:
            if scheme_type == LabelSchemeType.EU_OFFICIAL.value:
                results[criterion] = True
            else:
                results[criterion] = bool(metadata.get(criterion, False))

        return results

    def _verify_certificate(
        self, item: Dict[str, Any], now: datetime,
    ) -> CertificateStatus:
        """Determine certificate validity status."""
        expiry_str = item.get("certificate_expiry", "")
        cert_id = item.get("certificate_id", "")

        if not cert_id and not expiry_str:
            return CertificateStatus.NOT_PROVIDED

        if item.get("revoked", False):
            return CertificateStatus.REVOKED

        if not expiry_str:
            return CertificateStatus.NOT_PROVIDED

        try:
            expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)

            if expiry < now:
                return CertificateStatus.EXPIRED

            days_remaining = (expiry - now).days
            if days_remaining <= self.audit_config.expiry_warning_days:
                return CertificateStatus.EXPIRING_SOON

            return CertificateStatus.VALID
        except (ValueError, TypeError):
            return CertificateStatus.NOT_PROVIDED

    def _days_until_expiry(self, expiry_str: str, now: datetime) -> Optional[int]:
        """Calculate days until certificate expiry."""
        if not expiry_str:
            return None
        try:
            expiry = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            return (expiry - now).days
        except (ValueError, TypeError):
            return None

    def _identify_issues(
        self,
        item: Dict[str, Any],
        scheme_info: Dict[str, Any],
        cert_info: Dict[str, Any],
    ) -> List[str]:
        """Identify compliance issues for a label."""
        issues: List[str] = []

        if not scheme_info.get("article_10_compliant", False):
            failed = [
                k for k, v in scheme_info.get("criteria_results", {}).items()
                if not v
            ]
            issues.append(f"Article 10 criteria not met: {', '.join(failed)}")

        cert_status = cert_info.get("certificate_status", "")
        if cert_status == CertificateStatus.EXPIRED.value:
            issues.append("Certificate has expired -- renewal required")
        elif cert_status == CertificateStatus.EXPIRING_SOON.value:
            days = cert_info.get("days_until_expiry")
            issues.append(f"Certificate expiring within {days} days")
        elif cert_status == CertificateStatus.NOT_PROVIDED.value:
            issues.append("No certificate provided for verification")
        elif cert_status == CertificateStatus.REVOKED.value:
            issues.append("Certificate has been revoked")

        if item.get("scheme_type") == LabelSchemeType.TYPE_II.value:
            issues.append("Self-declared label (Type II) requires additional substantiation")

        if not item.get("issuer"):
            issues.append("No issuing body identified")

        return issues

    def _calculate_compliance_score(
        self, scheme_info: Dict[str, Any], cert_info: Dict[str, Any],
    ) -> float:
        """Calculate per-label compliance score (0-100)."""
        score = 0.0

        # Article 10 compliance (50 points max)
        criteria_met = scheme_info.get("criteria_met", 0)
        criteria_total = scheme_info.get("criteria_total", 1)
        score += (criteria_met / criteria_total) * 50.0

        # Certificate validity (30 points)
        cert_status = cert_info.get("certificate_status", "")
        cert_scores: Dict[str, float] = {
            CertificateStatus.VALID.value: 30.0,
            CertificateStatus.EXPIRING_SOON.value: 20.0,
            CertificateStatus.EXPIRED.value: 5.0,
            CertificateStatus.NOT_PROVIDED.value: 0.0,
            CertificateStatus.REVOKED.value: 0.0,
        }
        score += cert_scores.get(cert_status, 0.0)

        # Article 10 full compliance bonus (20 points)
        if scheme_info.get("article_10_compliant", False):
            score += 20.0

        return min(round(score, 1), 100.0)

    def _generate_recommendation(self, issues: List[str], score: float) -> str:
        """Generate a recommendation based on audit findings."""
        if not issues and score >= 80.0:
            return "Label is compliant. Maintain current certification and schedule periodic reviews."
        if not issues:
            return "Label meets minimum requirements. Consider strengthening governance documentation."
        if len(issues) >= 3:
            return "Multiple compliance issues detected. Consider replacing with an EU-approved scheme."
        if score < 40.0:
            return "Significant compliance gaps. Withdraw label pending remediation."
        return "Address identified issues and re-evaluate before Directive enforcement date."

    def _determine_audit_outcome(self, pass_count: int, total: int) -> str:
        """Determine overall audit outcome."""
        if total == 0:
            return "NO_LABELS_ASSESSED"
        rate = (pass_count / total) * 100
        if rate >= 90.0:
            return "PASS"
        if rate >= 70.0:
            return "CONDITIONAL_PASS"
        return "FAIL"
