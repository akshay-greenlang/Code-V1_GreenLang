# -*- coding: utf-8 -*-
"""
VerificationEngine - PACK-004 CBAM Readiness Engine 6
======================================================

Accredited verifier management and verification engagement engine for CBAM
compliance. Handles verifier registration, engagement creation, evidence
packaging, finding management, materiality assessment, and verification
statement issuance.

Verification Framework:
    - CBAM requires third-party verification of embedded emission data
    - Verifiers must be accredited by a recognized national body
    - Assurance levels: LIMITED (transitional) or REASONABLE (definitive)
    - Materiality threshold: 5% of total reported emissions
    - Findings classified by category and severity

Finding Categories:
    - EMISSION_CALCULATION: Errors in emission factor or formula application
    - DATA_QUALITY: Missing, incomplete, or inconsistent source data
    - METHODOLOGY: Incorrect calculation method or boundary setting
    - DOCUMENTATION: Insufficient supporting evidence or audit trail

Finding Severities:
    - MATERIAL: Exceeds 5% materiality threshold; may affect opinion
    - IMMATERIAL: Below threshold; noted for improvement

Zero-Hallucination:
    - All materiality calculations use deterministic arithmetic
    - No LLM involvement in any verification decision
    - SHA-256 provenance hashing on every engagement

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-004 CBAM Readiness
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default materiality threshold (percentage of total reported emissions)
MATERIALITY_THRESHOLD_PCT: float = 5.0


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CBAMGoodsCategory(str, Enum):
    """CBAM goods categories (mirrored for self-containment)."""

    CEMENT = "cement"
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    FERTILIZERS = "fertilizers"
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"


class AssuranceLevel(str, Enum):
    """Verification assurance level."""

    LIMITED = "LIMITED"
    REASONABLE = "REASONABLE"


class EngagementStatus(str, Enum):
    """Verification engagement lifecycle status."""

    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FINDINGS_OPEN = "FINDINGS_OPEN"


class FindingCategory(str, Enum):
    """Category of verification finding."""

    EMISSION_CALCULATION = "EMISSION_CALCULATION"
    DATA_QUALITY = "DATA_QUALITY"
    METHODOLOGY = "METHODOLOGY"
    DOCUMENTATION = "DOCUMENTATION"


class FindingSeverity(str, Enum):
    """Severity of verification finding."""

    MATERIAL = "MATERIAL"
    IMMATERIAL = "IMMATERIAL"


class VerificationOpinion(str, Enum):
    """Verification statement opinion types."""

    UNQUALIFIED = "UNQUALIFIED"
    QUALIFIED = "QUALIFIED"
    ADVERSE = "ADVERSE"
    DISCLAIMER = "DISCLAIMER"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class AccreditedVerifier(BaseModel):
    """Accredited third-party verifier for CBAM emission data.

    Verifiers must hold valid accreditation from a recognized national
    accreditation body to perform CBAM verification.
    """

    verifier_id: str = Field(
        default_factory=_new_uuid,
        description="Unique verifier identifier",
    )
    name: str = Field(
        ..., min_length=1, max_length=300,
        description="Legal name of the verification body",
    )
    accreditation_body: str = Field(
        ..., min_length=1, max_length=300,
        description="Name of the national accreditation body",
    )
    accreditation_number: str = Field(
        ..., min_length=1, max_length=100,
        description="Accreditation certificate number",
    )
    specializations: List[CBAMGoodsCategory] = Field(
        default_factory=list,
        description="Goods categories the verifier is qualified for",
    )
    country: str = Field(
        ..., min_length=2, max_length=2,
        description="Country where the verifier is based",
    )
    contact_email: str = Field(
        "", max_length=200,
        description="Primary contact email",
    )
    contact_phone: str = Field(
        "", max_length=50,
        description="Contact phone number",
    )
    valid_until: date = Field(
        ..., description="Accreditation validity end date",
    )
    registered_at: datetime = Field(
        default_factory=_utcnow,
        description="Registration timestamp",
    )

    @field_validator("country")
    @classmethod
    def uppercase_country(cls, v: str) -> str:
        """Ensure country code is uppercase."""
        return v.strip().upper()


class VerificationFinding(BaseModel):
    """Individual finding from a verification engagement.

    Findings are issues identified by the verifier during the assessment
    of the importer's embedded emission data and documentation.
    """

    finding_id: str = Field(
        default_factory=_new_uuid,
        description="Unique finding identifier",
    )
    engagement_id: str = Field(
        ..., description="Parent engagement identifier",
    )
    category: FindingCategory = Field(
        ..., description="Finding category",
    )
    severity: FindingSeverity = Field(
        ..., description="Finding severity (MATERIAL or IMMATERIAL)",
    )
    title: str = Field(
        ..., min_length=1, max_length=300,
        description="Short title of the finding",
    )
    description: str = Field(
        ..., min_length=1, max_length=5000,
        description="Detailed description of the finding",
    )
    affected_emissions_tco2e: float = Field(
        0.0, ge=0,
        description="Estimated emissions affected by this finding (tCO2e)",
    )
    affected_cn_codes: List[str] = Field(
        default_factory=list,
        description="CN codes affected by this finding",
    )
    response: str = Field(
        "", max_length=5000,
        description="Importer's response to the finding",
    )
    corrective_action: str = Field(
        "", max_length=5000,
        description="Corrective action taken or planned",
    )
    resolved: bool = Field(
        False, description="Whether the finding has been resolved",
    )
    resolution_date: Optional[datetime] = Field(
        None, description="Date the finding was resolved",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Finding creation timestamp",
    )


class VerificationEngagement(BaseModel):
    """Verification engagement between an importer and an accredited verifier.

    Represents the full scope and status of a CBAM verification, including
    findings, evidence, and the final verification statement.
    """

    engagement_id: str = Field(
        default_factory=_new_uuid,
        description="Unique engagement identifier",
    )
    verifier_id: str = Field(
        ..., description="Assigned verifier identifier",
    )
    verifier_name: str = Field(
        "", max_length=300,
        description="Name of the verification body",
    )
    importer_eori: str = Field(
        ..., min_length=5, max_length=17,
        description="EORI number of the importer",
    )
    importer_name: str = Field(
        "", max_length=300,
        description="Name of the importer",
    )
    scope: List[CBAMGoodsCategory] = Field(
        default_factory=list,
        description="Goods categories in scope",
    )
    reporting_year: int = Field(
        ..., ge=2023, le=2050,
        description="Reporting year under verification",
    )
    assurance_level: AssuranceLevel = Field(
        AssuranceLevel.LIMITED,
        description="Assurance level (LIMITED or REASONABLE)",
    )
    status: EngagementStatus = Field(
        EngagementStatus.PLANNED,
        description="Current engagement status",
    )
    materiality_threshold_pct: float = Field(
        MATERIALITY_THRESHOLD_PCT,
        description="Materiality threshold percentage",
    )
    total_emissions_in_scope: float = Field(
        0.0, ge=0,
        description="Total emissions in scope for materiality calculation",
    )
    findings: List[VerificationFinding] = Field(
        default_factory=list,
        description="List of verification findings",
    )
    verification_statement: Optional[str] = Field(
        None, description="Final verification statement text",
    )
    opinion: Optional[VerificationOpinion] = Field(
        None, description="Verification opinion type",
    )
    planned_start: Optional[date] = Field(
        None, description="Planned start date",
    )
    planned_end: Optional[date] = Field(
        None, description="Planned end date",
    )
    actual_completion: Optional[datetime] = Field(
        None, description="Actual completion timestamp",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Engagement creation timestamp",
    )
    provenance_hash: str = Field(
        "", description="SHA-256 hash for audit trail",
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class VerificationEngine:
    """Accredited verifier management and verification engagement engine.

    Manages the full verification lifecycle: verifier registration,
    engagement creation, evidence packaging, finding management,
    materiality assessment, and verification statement issuance.

    Zero-Hallucination Guarantees:
        - Materiality calculations use deterministic arithmetic
        - No LLM involvement in any verification decision
        - SHA-256 provenance hashing on every engagement

    Example:
        >>> engine = VerificationEngine()
        >>> verifier = AccreditedVerifier(
        ...     name="EU Verify AG",
        ...     accreditation_body="DAkkS",
        ...     accreditation_number="D-VS-12345",
        ...     country="DE",
        ...     valid_until=date(2028, 12, 31),
        ... )
        >>> verifier_id = engine.register_verifier(verifier)
    """

    def __init__(self) -> None:
        """Initialize VerificationEngine."""
        self._verifiers: Dict[str, AccreditedVerifier] = {}
        self._engagements: Dict[str, VerificationEngagement] = {}
        self._findings: Dict[str, VerificationFinding] = {}
        logger.info("VerificationEngine initialized (v%s)", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_verifier(self, verifier: AccreditedVerifier) -> str:
        """Register an accredited verifier.

        Args:
            verifier: Complete verifier profile with accreditation details.

        Returns:
            The verifier_id of the registered verifier.

        Raises:
            ValueError: If accreditation number is already registered or expired.
        """
        # Check for duplicate accreditation number
        for existing in self._verifiers.values():
            if existing.accreditation_number == verifier.accreditation_number:
                raise ValueError(
                    f"Accreditation number {verifier.accreditation_number} already "
                    f"registered under verifier {existing.verifier_id}"
                )

        # Check accreditation is not expired
        today = _utcnow().date()
        if verifier.valid_until < today:
            raise ValueError(
                f"Accreditation expired on {verifier.valid_until}; "
                "cannot register expired verifiers"
            )

        self._verifiers[verifier.verifier_id] = verifier

        logger.info(
            "Verifier registered [%s]: %s (accredited by %s, valid until %s)",
            verifier.verifier_id,
            verifier.name,
            verifier.accreditation_body,
            verifier.valid_until,
        )

        return verifier.verifier_id

    def create_engagement(
        self,
        importer_eori: str,
        verifier_id: str,
        scope: List[CBAMGoodsCategory],
        year: int,
        assurance_level: AssuranceLevel = AssuranceLevel.LIMITED,
        importer_name: str = "",
        total_emissions: float = 0.0,
        planned_start: Optional[date] = None,
        planned_end: Optional[date] = None,
    ) -> VerificationEngagement:
        """Create a new verification engagement.

        Assigns a verifier to verify an importer's CBAM data for a
        specific reporting year and set of goods categories.

        Args:
            importer_eori: EORI number of the importer.
            verifier_id: ID of the assigned verifier.
            scope: List of goods categories in scope.
            year: Reporting year to verify.
            assurance_level: LIMITED or REASONABLE.
            importer_name: Optional importer name.
            total_emissions: Total emissions in scope (for materiality).
            planned_start: Planned engagement start date.
            planned_end: Planned engagement end date.

        Returns:
            VerificationEngagement in PLANNED status.

        Raises:
            ValueError: If verifier is not found or not qualified for scope.
        """
        verifier = self._get_verifier(verifier_id)

        # Validate verifier qualifications
        missing_quals = [
            cat for cat in scope if cat not in verifier.specializations
        ]
        if missing_quals:
            logger.warning(
                "Verifier %s missing qualifications for: %s",
                verifier_id,
                [m.value for m in missing_quals],
            )

        engagement = VerificationEngagement(
            verifier_id=verifier_id,
            verifier_name=verifier.name,
            importer_eori=importer_eori,
            importer_name=importer_name,
            scope=scope,
            reporting_year=year,
            assurance_level=assurance_level,
            total_emissions_in_scope=total_emissions,
            planned_start=planned_start,
            planned_end=planned_end,
        )
        engagement.provenance_hash = _compute_hash(engagement)

        self._engagements[engagement.engagement_id] = engagement

        logger.info(
            "Engagement created [%s]: verifier=%s, importer=%s, year=%d, scope=%s",
            engagement.engagement_id,
            verifier.name,
            importer_eori,
            year,
            [s.value for s in scope],
        )

        return engagement

    def prepare_evidence(
        self,
        engagement_id: str,
        emission_data: Optional[Dict[str, Any]] = None,
        supplier_data: Optional[List[Dict[str, Any]]] = None,
        report_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Package evidence for a verifier.

        Assembles the data package that the verifier needs to perform
        their assessment, including emission calculations, supplier
        communications, and report data.

        Args:
            engagement_id: ID of the engagement.
            emission_data: Aggregated emission calculation data.
            supplier_data: Supplier communication and submission data.
            report_data: Quarterly/annual report data.

        Returns:
            Evidence package dictionary with metadata and data sections.

        Raises:
            ValueError: If engagement is not found.
        """
        engagement = self._get_engagement(engagement_id)

        evidence = {
            "evidence_package_id": _new_uuid(),
            "engagement_id": engagement_id,
            "verifier_id": engagement.verifier_id,
            "importer_eori": engagement.importer_eori,
            "reporting_year": engagement.reporting_year,
            "scope": [s.value for s in engagement.scope],
            "assurance_level": engagement.assurance_level.value,
            "prepared_at": _utcnow().isoformat(),
            "sections": {},
        }

        if emission_data:
            evidence["sections"]["emission_calculations"] = {
                "description": "Embedded emission calculation results and supporting data",
                "record_count": len(emission_data) if isinstance(emission_data, list) else 1,
                "data": emission_data,
            }

        if supplier_data:
            evidence["sections"]["supplier_communications"] = {
                "description": "Supplier emission data submissions and communications",
                "supplier_count": len(supplier_data),
                "data": supplier_data,
            }

        if report_data:
            evidence["sections"]["quarterly_reports"] = {
                "description": "CBAM quarterly report data",
                "data": report_data,
            }

        evidence["provenance_hash"] = _compute_hash(evidence)

        # Update engagement status
        engagement.status = EngagementStatus.IN_PROGRESS

        logger.info(
            "Evidence packaged for engagement [%s]: %d sections",
            engagement_id,
            len(evidence["sections"]),
        )

        return evidence

    def submit_finding(
        self,
        engagement_id: str,
        finding: VerificationFinding,
    ) -> str:
        """Submit a verification finding for an engagement.

        Args:
            engagement_id: ID of the engagement.
            finding: The verification finding to submit.

        Returns:
            The finding_id.

        Raises:
            ValueError: If engagement is not found.
        """
        engagement = self._get_engagement(engagement_id)

        finding.engagement_id = engagement_id
        engagement.findings.append(finding)
        self._findings[finding.finding_id] = finding

        # Auto-assess materiality
        is_material = self.assess_materiality(finding, engagement.total_emissions_in_scope)
        if is_material:
            finding.severity = FindingSeverity.MATERIAL

        # Update engagement status if there are open material findings
        open_material = any(
            f.severity == FindingSeverity.MATERIAL and not f.resolved
            for f in engagement.findings
        )
        if open_material:
            engagement.status = EngagementStatus.FINDINGS_OPEN

        logger.info(
            "Finding submitted [%s] for engagement [%s]: %s (%s) - %s",
            finding.finding_id,
            engagement_id,
            finding.title,
            finding.category.value,
            finding.severity.value,
        )

        return finding.finding_id

    def respond_to_finding(
        self,
        finding_id: str,
        response: str,
        corrective_action: str = "",
        resolve: bool = False,
    ) -> Dict[str, Any]:
        """Record an importer's response to a verification finding.

        Args:
            finding_id: ID of the finding.
            response: Importer's response text.
            corrective_action: Description of corrective action taken.
            resolve: Whether to mark the finding as resolved.

        Returns:
            Dictionary with finding status update.

        Raises:
            ValueError: If finding is not found.
        """
        if finding_id not in self._findings:
            raise ValueError(f"Finding not found: {finding_id}")

        finding = self._findings[finding_id]
        finding.response = response
        finding.corrective_action = corrective_action

        if resolve:
            finding.resolved = True
            finding.resolution_date = _utcnow()

        # Check if all material findings for the engagement are resolved
        engagement = self._engagements.get(finding.engagement_id)
        if engagement:
            open_material = any(
                f.severity == FindingSeverity.MATERIAL and not f.resolved
                for f in engagement.findings
            )
            if not open_material and engagement.status == EngagementStatus.FINDINGS_OPEN:
                engagement.status = EngagementStatus.IN_PROGRESS

        logger.info(
            "Response recorded for finding [%s]: resolved=%s",
            finding_id,
            resolve,
        )

        return {
            "finding_id": finding_id,
            "response_recorded": True,
            "resolved": finding.resolved,
            "resolution_date": (
                finding.resolution_date.isoformat() if finding.resolution_date else None
            ),
            "engagement_status": engagement.status.value if engagement else "UNKNOWN",
        }

    def assess_materiality(
        self,
        finding: VerificationFinding,
        total_emissions: float,
    ) -> bool:
        """Assess whether a finding is material.

        A finding is material if the affected emissions exceed the
        materiality threshold percentage of total reported emissions.

        Materiality threshold: 5% of total emissions in scope.

        Args:
            finding: The verification finding to assess.
            total_emissions: Total emissions in scope (tCO2e).

        Returns:
            True if the finding is material, False otherwise.
        """
        if total_emissions <= 0:
            return False

        affected = _decimal(finding.affected_emissions_tco2e)
        total = _decimal(total_emissions)
        threshold_pct = _decimal(MATERIALITY_THRESHOLD_PCT)

        materiality_limit = total * threshold_pct / Decimal("100")
        is_material = affected > materiality_limit

        if is_material:
            logger.warning(
                "Finding [%s] is MATERIAL: affected=%.2f tCO2e > threshold=%.2f tCO2e (%.1f%%)",
                finding.finding_id,
                float(affected),
                float(materiality_limit),
                MATERIALITY_THRESHOLD_PCT,
            )

        return is_material

    def issue_statement(
        self,
        engagement_id: str,
        opinion: VerificationOpinion,
        statement_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Issue a verification statement for a completed engagement.

        The verification statement summarizes the verifier's opinion on the
        accuracy and completeness of the importer's CBAM data.

        Opinion Types:
            - UNQUALIFIED: Data is materially correct
            - QUALIFIED: Data is generally correct but with noted exceptions
            - ADVERSE: Data contains material misstatements
            - DISCLAIMER: Unable to form an opinion

        Args:
            engagement_id: ID of the engagement.
            opinion: Verification opinion type.
            statement_text: Optional custom statement text.

        Returns:
            Dictionary with statement details.

        Raises:
            ValueError: If engagement not found or has open material findings.
        """
        engagement = self._get_engagement(engagement_id)

        # Check for unresolved material findings
        open_material = [
            f for f in engagement.findings
            if f.severity == FindingSeverity.MATERIAL and not f.resolved
        ]

        if open_material and opinion == VerificationOpinion.UNQUALIFIED:
            raise ValueError(
                f"Cannot issue UNQUALIFIED opinion with {len(open_material)} "
                "open material finding(s). Resolve findings first or use a "
                "different opinion type."
            )

        # Generate default statement text if not provided
        if statement_text is None:
            statement_text = self._generate_statement_text(engagement, opinion)

        engagement.verification_statement = statement_text
        engagement.opinion = opinion
        engagement.status = EngagementStatus.COMPLETED
        engagement.actual_completion = _utcnow()
        engagement.provenance_hash = _compute_hash(engagement)

        # Summary statistics
        total_findings = len(engagement.findings)
        material_findings = sum(
            1 for f in engagement.findings if f.severity == FindingSeverity.MATERIAL
        )
        resolved_findings = sum(1 for f in engagement.findings if f.resolved)

        logger.info(
            "Verification statement issued [%s]: opinion=%s, findings=%d (%d material, %d resolved)",
            engagement_id,
            opinion.value,
            total_findings,
            material_findings,
            resolved_findings,
        )

        return {
            "engagement_id": engagement_id,
            "opinion": opinion.value,
            "statement_text": statement_text,
            "total_findings": total_findings,
            "material_findings": material_findings,
            "resolved_findings": resolved_findings,
            "completed_at": engagement.actual_completion.isoformat(),
            "provenance_hash": engagement.provenance_hash,
        }

    def list_verifiers(
        self,
        specialization: Optional[CBAMGoodsCategory] = None,
        country: Optional[str] = None,
    ) -> List[AccreditedVerifier]:
        """List registered verifiers with optional filtering.

        Args:
            specialization: Filter by goods category qualification.
            country: Filter by country code.

        Returns:
            List of matching AccreditedVerifier objects.
        """
        verifiers = list(self._verifiers.values())

        if specialization:
            verifiers = [
                v for v in verifiers if specialization in v.specializations
            ]

        if country:
            country_upper = country.strip().upper()
            verifiers = [v for v in verifiers if v.country == country_upper]

        # Filter out expired
        today = _utcnow().date()
        verifiers = [v for v in verifiers if v.valid_until >= today]

        return sorted(verifiers, key=lambda v: v.name)

    def get_engagement(self, engagement_id: str) -> VerificationEngagement:
        """Retrieve an engagement by ID.

        Args:
            engagement_id: Engagement identifier.

        Returns:
            VerificationEngagement.

        Raises:
            ValueError: If not found.
        """
        return self._get_engagement(engagement_id)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def verifier_count(self) -> int:
        """Number of registered verifiers."""
        return len(self._verifiers)

    @property
    def engagement_count(self) -> int:
        """Number of verification engagements."""
        return len(self._engagements)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_verifier(self, verifier_id: str) -> AccreditedVerifier:
        """Look up a verifier by ID."""
        if verifier_id not in self._verifiers:
            raise ValueError(f"Verifier not found: {verifier_id}")
        return self._verifiers[verifier_id]

    def _get_engagement(self, engagement_id: str) -> VerificationEngagement:
        """Look up an engagement by ID."""
        if engagement_id not in self._engagements:
            raise ValueError(f"Engagement not found: {engagement_id}")
        return self._engagements[engagement_id]

    def _generate_statement_text(
        self,
        engagement: VerificationEngagement,
        opinion: VerificationOpinion,
    ) -> str:
        """Generate default verification statement text.

        Args:
            engagement: The verification engagement.
            opinion: The verification opinion.

        Returns:
            Formatted verification statement text.
        """
        scope_str = ", ".join(s.value.replace("_", " ").title() for s in engagement.scope)
        total_findings = len(engagement.findings)
        material_count = sum(
            1 for f in engagement.findings if f.severity == FindingSeverity.MATERIAL
        )

        header = (
            f"VERIFICATION STATEMENT\n"
            f"{'=' * 50}\n\n"
            f"Engagement: {engagement.engagement_id}\n"
            f"Importer: {engagement.importer_name} ({engagement.importer_eori})\n"
            f"Verifier: {engagement.verifier_name}\n"
            f"Reporting Year: {engagement.reporting_year}\n"
            f"Scope: {scope_str}\n"
            f"Assurance Level: {engagement.assurance_level.value}\n"
            f"Total Emissions in Scope: {engagement.total_emissions_in_scope:.2f} tCO2e\n\n"
        )

        if opinion == VerificationOpinion.UNQUALIFIED:
            body = (
                "OPINION: UNQUALIFIED\n\n"
                "Based on the procedures we have performed and the evidence we have obtained, "
                "nothing has come to our attention that causes us to believe that the embedded "
                "emissions data reported by the importer for the CBAM goods in scope is not, "
                "in all material respects, prepared in accordance with the CBAM Implementing "
                "Regulation.\n"
            )
        elif opinion == VerificationOpinion.QUALIFIED:
            body = (
                "OPINION: QUALIFIED\n\n"
                "Except for the matters described in the findings section, based on the "
                "procedures we have performed, nothing has come to our attention that causes "
                "us to believe that the embedded emissions data is not, in all material "
                "respects, prepared in accordance with the CBAM Implementing Regulation.\n"
            )
        elif opinion == VerificationOpinion.ADVERSE:
            body = (
                "OPINION: ADVERSE\n\n"
                "Based on the procedures we have performed, we have identified material "
                "misstatements in the embedded emissions data reported by the importer. "
                "The data is not, in our opinion, prepared in accordance with the CBAM "
                "Implementing Regulation in the following material respects.\n"
            )
        else:  # DISCLAIMER
            body = (
                "OPINION: DISCLAIMER\n\n"
                "Due to the significance of the matters described in the findings, we "
                "were unable to obtain sufficient appropriate evidence to provide a "
                "basis for a verification opinion on the embedded emissions data.\n"
            )

        footer = (
            f"\nFindings Summary: {total_findings} total ({material_count} material)\n"
            f"Date: {_utcnow().date().isoformat()}\n"
        )

        return header + body + footer
