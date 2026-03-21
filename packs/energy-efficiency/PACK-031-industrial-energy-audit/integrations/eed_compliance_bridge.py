# -*- coding: utf-8 -*-
"""
EEDComplianceBridge - EU Energy Efficiency Directive Compliance for PACK-031
==============================================================================

This module manages compliance with the EU Energy Efficiency Directive (EED),
including Article 8 mandatory energy audit obligations, audit cycle scheduling,
ISO 50001 / EMAS exemption tracking, national transposition requirements by
EU member state, and compliance reporting for national authorities.

EED Article 8 Requirements:
    - Mandatory energy audits for non-SME enterprises every 4 years
    - Exemption for organisations with certified ISO 50001 or EMAS
    - Audit must cover at least 90% of total energy consumption
    - Must comply with EN 16247-1 (general) and sector parts (2-5)
    - National authority notification within specified deadline

National Transposition Variations (selected):
    DE: BAFA notification, EDL-G transposition
    FR: ADEME notification, Code de l'energie
    NL: EED audit + EML (Energy Saving Obligation)
    IT: ENEA notification, D.Lgs 102/2014
    ES: National register, Royal Decree 56/2016

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-031 Industrial Energy Audit
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EEDExemptionType(str, Enum):
    """Exemption types from EED Article 8 mandatory audit."""

    NONE = "none"
    ISO_50001 = "iso_50001"
    EMAS = "emas"
    SME = "sme"


class ComplianceStatus(str, Enum):
    """EED compliance status values."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    EXEMPT = "exempt"
    PENDING = "pending"
    OVERDUE = "overdue"


class AuditStandard(str, Enum):
    """Applicable EN 16247 audit standard parts."""

    EN_16247_1 = "EN_16247-1"  # General requirements
    EN_16247_2 = "EN_16247-2"  # Buildings
    EN_16247_3 = "EN_16247-3"  # Processes
    EN_16247_4 = "EN_16247-4"  # Transport
    EN_16247_5 = "EN_16247-5"  # Competence of auditors


class EUMemberState(str, Enum):
    """EU member states with EED transposition details."""

    DE = "DE"
    FR = "FR"
    NL = "NL"
    IT = "IT"
    ES = "ES"
    AT = "AT"
    BE = "BE"
    PL = "PL"
    SE = "SE"
    DK = "DK"
    FI = "FI"
    IE = "IE"
    PT = "PT"
    CZ = "CZ"
    RO = "RO"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class EEDObligationAssessment(BaseModel):
    """Assessment of EED Article 8 obligation for an organisation."""

    assessment_id: str = Field(default_factory=_new_uuid)
    organisation_name: str = Field(default="")
    member_state: str = Field(default="DE")
    is_sme: bool = Field(default=False, description="True if < 250 employees and < EUR 50M revenue")
    employee_count: int = Field(default=0, ge=0)
    annual_revenue_eur: float = Field(default=0.0, ge=0)
    balance_sheet_total_eur: float = Field(default=0.0, ge=0)
    has_iso_50001: bool = Field(default=False)
    has_emas: bool = Field(default=False)
    exemption_type: EEDExemptionType = Field(default=EEDExemptionType.NONE)
    audit_obligation: bool = Field(default=True)
    obligation_reason: str = Field(default="")
    next_audit_deadline: Optional[date] = Field(None)
    compliance_status: ComplianceStatus = Field(default=ComplianceStatus.PENDING)
    provenance_hash: str = Field(default="")


class AuditCycleRecord(BaseModel):
    """Record of a 4-year EED audit cycle."""

    cycle_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field(default="")
    cycle_number: int = Field(default=1, ge=1)
    audit_date: Optional[date] = Field(None)
    next_audit_due: Optional[date] = Field(None)
    auditor_name: str = Field(default="")
    auditor_qualification: str = Field(default="")
    energy_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    en_16247_parts_applied: List[str] = Field(default_factory=list)
    authority_notified: bool = Field(default=False)
    authority_notification_date: Optional[date] = Field(None)
    status: ComplianceStatus = Field(default=ComplianceStatus.PENDING)
    provenance_hash: str = Field(default="")


class NationalTransposition(BaseModel):
    """National transposition details for a member state."""

    member_state: EUMemberState = Field(...)
    national_law: str = Field(default="")
    national_authority: str = Field(default="")
    notification_portal: str = Field(default="")
    notification_deadline_days: int = Field(default=90, description="Days after audit to notify")
    additional_requirements: List[str] = Field(default_factory=list)
    energy_coverage_threshold_pct: float = Field(default=90.0)
    auditor_certification_required: bool = Field(default=True)


class EEDComplianceReport(BaseModel):
    """Compliance report for national authority submission."""

    report_id: str = Field(default_factory=_new_uuid)
    organisation_name: str = Field(default="")
    member_state: str = Field(default="")
    audit_date: Optional[date] = Field(None)
    energy_consumption_mwh: float = Field(default=0.0)
    energy_coverage_pct: float = Field(default=0.0)
    savings_identified_mwh: float = Field(default=0.0)
    savings_identified_eur: float = Field(default=0.0)
    en_16247_compliance: bool = Field(default=False)
    recommendations_count: int = Field(default=0)
    status: ComplianceStatus = Field(default=ComplianceStatus.PENDING)
    generated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="")


class EEDComplianceBridgeConfig(BaseModel):
    """Configuration for the EED Compliance Bridge."""

    pack_id: str = Field(default="PACK-031")
    default_member_state: EUMemberState = Field(default=EUMemberState.DE)
    audit_cycle_years: int = Field(default=4, ge=1, le=10)
    energy_coverage_threshold_pct: float = Field(default=90.0, ge=0.0, le=100.0)
    enable_provenance: bool = Field(default=True)


# ---------------------------------------------------------------------------
# National Transposition Database
# ---------------------------------------------------------------------------

NATIONAL_TRANSPOSITIONS: Dict[EUMemberState, NationalTransposition] = {
    EUMemberState.DE: NationalTransposition(
        member_state=EUMemberState.DE,
        national_law="EDL-G (Energiedienstleistungsgesetz)",
        national_authority="BAFA (Federal Office for Economic Affairs and Export Control)",
        notification_portal="https://edl.bafa.de",
        notification_deadline_days=90,
        additional_requirements=["Online portal registration", "Auditor must be BAFA-listed"],
        auditor_certification_required=True,
    ),
    EUMemberState.FR: NationalTransposition(
        member_state=EUMemberState.FR,
        national_law="Code de l'energie - Article L233-1",
        national_authority="ADEME",
        notification_portal="https://audit-energetique.ademe.fr",
        notification_deadline_days=120,
        additional_requirements=["COFRAC-accredited auditor", "Action plan mandatory"],
        auditor_certification_required=True,
    ),
    EUMemberState.NL: NationalTransposition(
        member_state=EUMemberState.NL,
        national_law="Activiteitenbesluit + EED audit obligation",
        national_authority="RVO (Netherlands Enterprise Agency)",
        notification_portal="https://rvo.nl/eed",
        notification_deadline_days=60,
        additional_requirements=["EML energy saving obligation", "EED + informatieplicht combined"],
        auditor_certification_required=True,
    ),
    EUMemberState.IT: NationalTransposition(
        member_state=EUMemberState.IT,
        national_law="D.Lgs 102/2014",
        national_authority="ENEA",
        notification_portal="https://audit102.enea.it",
        notification_deadline_days=90,
        additional_requirements=["ENEA-registered auditor (EGE)", "Online portal submission"],
        auditor_certification_required=True,
    ),
    EUMemberState.ES: NationalTransposition(
        member_state=EUMemberState.ES,
        national_law="Royal Decree 56/2016",
        national_authority="IDAE",
        notification_portal="https://sede.idae.gob.es",
        notification_deadline_days=90,
        additional_requirements=["National registry inscription", "Regional authority may also apply"],
        auditor_certification_required=True,
    ),
}


# ---------------------------------------------------------------------------
# EEDComplianceBridge
# ---------------------------------------------------------------------------


class EEDComplianceBridge:
    """EU Energy Efficiency Directive compliance management.

    Manages Article 8 obligation assessment, 4-year audit cycle scheduling,
    ISO 50001/EMAS exemption tracking, and national authority reporting.

    Attributes:
        config: Bridge configuration.
        _cycles: Historical audit cycle records.
        _assessments: Obligation assessments.

    Example:
        >>> bridge = EEDComplianceBridge()
        >>> assessment = bridge.assess_obligation(
        ...     organisation_name="Acme Manufacturing",
        ...     employee_count=500, annual_revenue_eur=150_000_000
        ... )
        >>> assert assessment.audit_obligation is True
    """

    def __init__(self, config: Optional[EEDComplianceBridgeConfig] = None) -> None:
        """Initialize the EED Compliance Bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or EEDComplianceBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cycles: List[AuditCycleRecord] = []
        self._assessments: List[EEDObligationAssessment] = []
        self.logger.info(
            "EEDComplianceBridge initialized: member_state=%s, cycle=%d years",
            self.config.default_member_state.value,
            self.config.audit_cycle_years,
        )

    def assess_obligation(
        self,
        organisation_name: str,
        employee_count: int,
        annual_revenue_eur: float,
        balance_sheet_total_eur: float = 0.0,
        has_iso_50001: bool = False,
        has_emas: bool = False,
        member_state: Optional[str] = None,
    ) -> EEDObligationAssessment:
        """Assess whether an organisation has an EED Article 8 audit obligation.

        Deterministic assessment based on EU SME definition:
        - SME: < 250 employees AND (revenue < EUR 50M OR balance sheet < EUR 43M)
        - ISO 50001 or EMAS certified organisations are exempt

        Args:
            organisation_name: Name of the organisation.
            employee_count: Number of employees.
            annual_revenue_eur: Annual revenue in EUR.
            balance_sheet_total_eur: Balance sheet total in EUR.
            has_iso_50001: Whether the organisation has ISO 50001 certification.
            has_emas: Whether the organisation has EMAS registration.
            member_state: EU member state (default from config).

        Returns:
            EEDObligationAssessment with obligation determination.
        """
        ms = member_state or self.config.default_member_state.value

        # Deterministic SME assessment per EU Recommendation 2003/361/EC
        is_sme = (
            employee_count < 250
            and (annual_revenue_eur < 50_000_000 or balance_sheet_total_eur < 43_000_000)
        )

        # Determine exemption
        exemption = EEDExemptionType.NONE
        obligation = True
        reason = ""

        if is_sme:
            exemption = EEDExemptionType.SME
            obligation = False
            reason = "SME exemption: below 250 employees and EUR 50M revenue threshold"
        elif has_iso_50001:
            exemption = EEDExemptionType.ISO_50001
            obligation = False
            reason = "Exempt: certified ISO 50001 Energy Management System"
        elif has_emas:
            exemption = EEDExemptionType.EMAS
            obligation = False
            reason = "Exempt: registered under EU EMAS regulation"
        else:
            reason = "Non-SME enterprise without ISO 50001/EMAS: audit mandatory"

        status = ComplianceStatus.EXEMPT if not obligation else ComplianceStatus.PENDING

        assessment = EEDObligationAssessment(
            organisation_name=organisation_name,
            member_state=ms,
            is_sme=is_sme,
            employee_count=employee_count,
            annual_revenue_eur=annual_revenue_eur,
            balance_sheet_total_eur=balance_sheet_total_eur,
            has_iso_50001=has_iso_50001,
            has_emas=has_emas,
            exemption_type=exemption,
            audit_obligation=obligation,
            obligation_reason=reason,
            compliance_status=status,
        )
        if self.config.enable_provenance:
            assessment.provenance_hash = _compute_hash(assessment)

        self._assessments.append(assessment)
        self.logger.info(
            "EED obligation assessed: org=%s, obligation=%s, exemption=%s",
            organisation_name, obligation, exemption.value,
        )
        return assessment

    def schedule_audit_cycle(
        self,
        organisation_id: str,
        last_audit_date: Optional[date] = None,
    ) -> AuditCycleRecord:
        """Schedule the next audit in the 4-year cycle.

        Args:
            organisation_id: Organisation identifier.
            last_audit_date: Date of the most recent audit (None if first).

        Returns:
            AuditCycleRecord with next audit deadline.
        """
        cycle_number = 1
        if last_audit_date:
            existing = [c for c in self._cycles if c.organisation_id == organisation_id]
            cycle_number = len(existing) + 1
            next_due = date(
                last_audit_date.year + self.config.audit_cycle_years,
                last_audit_date.month,
                last_audit_date.day,
            )
        else:
            next_due = date.today()

        record = AuditCycleRecord(
            organisation_id=organisation_id,
            cycle_number=cycle_number,
            audit_date=last_audit_date,
            next_audit_due=next_due,
            status=ComplianceStatus.PENDING,
        )
        if self.config.enable_provenance:
            record.provenance_hash = _compute_hash(record)

        self._cycles.append(record)
        self.logger.info(
            "Audit cycle scheduled: org=%s, cycle=%d, next_due=%s",
            organisation_id, cycle_number, next_due.isoformat(),
        )
        return record

    def get_national_requirements(
        self, member_state: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Get national transposition requirements for a member state.

        Args:
            member_state: EU member state code (default from config).

        Returns:
            Dict with national requirements, or None if not found.
        """
        ms_code = member_state or self.config.default_member_state.value
        try:
            ms_enum = EUMemberState(ms_code)
        except ValueError:
            return None

        transposition = NATIONAL_TRANSPOSITIONS.get(ms_enum)
        if transposition is None:
            return None

        return {
            "member_state": transposition.member_state.value,
            "national_law": transposition.national_law,
            "national_authority": transposition.national_authority,
            "notification_portal": transposition.notification_portal,
            "notification_deadline_days": transposition.notification_deadline_days,
            "additional_requirements": transposition.additional_requirements,
            "energy_coverage_threshold_pct": transposition.energy_coverage_threshold_pct,
            "auditor_certification_required": transposition.auditor_certification_required,
        }

    def generate_compliance_report(
        self,
        organisation_name: str,
        audit_date: date,
        energy_consumption_mwh: float,
        energy_coverage_pct: float,
        savings_identified_mwh: float,
        savings_identified_eur: float,
        recommendations_count: int,
        member_state: Optional[str] = None,
    ) -> EEDComplianceReport:
        """Generate a compliance report for national authority submission.

        Args:
            organisation_name: Organisation name.
            audit_date: Date the audit was completed.
            energy_consumption_mwh: Total energy consumption in MWh.
            energy_coverage_pct: Percentage of energy covered by audit.
            savings_identified_mwh: Savings identified in MWh.
            savings_identified_eur: Savings identified in EUR.
            recommendations_count: Number of recommendations.
            member_state: EU member state.

        Returns:
            EEDComplianceReport ready for authority submission.
        """
        ms = member_state or self.config.default_member_state.value

        en_16247_compliance = energy_coverage_pct >= self.config.energy_coverage_threshold_pct
        status = ComplianceStatus.COMPLIANT if en_16247_compliance else ComplianceStatus.NON_COMPLIANT

        report = EEDComplianceReport(
            organisation_name=organisation_name,
            member_state=ms,
            audit_date=audit_date,
            energy_consumption_mwh=energy_consumption_mwh,
            energy_coverage_pct=energy_coverage_pct,
            savings_identified_mwh=savings_identified_mwh,
            savings_identified_eur=savings_identified_eur,
            en_16247_compliance=en_16247_compliance,
            recommendations_count=recommendations_count,
            status=status,
        )
        if self.config.enable_provenance:
            report.provenance_hash = _compute_hash(report)

        self.logger.info(
            "EED compliance report generated: org=%s, status=%s, coverage=%.1f%%",
            organisation_name, status.value, energy_coverage_pct,
        )
        return report

    def get_audit_history(self, organisation_id: str) -> List[Dict[str, Any]]:
        """Get audit cycle history for an organisation.

        Args:
            organisation_id: Organisation identifier.

        Returns:
            List of audit cycle summaries.
        """
        return [
            {
                "cycle_id": c.cycle_id,
                "cycle_number": c.cycle_number,
                "audit_date": c.audit_date.isoformat() if c.audit_date else None,
                "next_audit_due": c.next_audit_due.isoformat() if c.next_audit_due else None,
                "status": c.status.value,
                "authority_notified": c.authority_notified,
            }
            for c in self._cycles
            if c.organisation_id == organisation_id
        ]

    def check_compliance_health(self) -> Dict[str, Any]:
        """Check the health of compliance tracking.

        Returns:
            Dict with compliance health metrics.
        """
        total = len(self._assessments)
        obligated = sum(1 for a in self._assessments if a.audit_obligation)
        exempt = sum(1 for a in self._assessments if not a.audit_obligation)

        overdue_cycles = [
            c for c in self._cycles
            if c.next_audit_due and c.next_audit_due < date.today()
            and c.status == ComplianceStatus.PENDING
        ]

        return {
            "total_organisations": total,
            "obligated": obligated,
            "exempt": exempt,
            "total_cycles": len(self._cycles),
            "overdue_audits": len(overdue_cycles),
            "member_states_covered": len(NATIONAL_TRANSPOSITIONS),
            "status": "healthy" if len(overdue_cycles) == 0 else "attention_needed",
        }
