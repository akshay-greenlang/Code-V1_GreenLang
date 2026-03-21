# -*- coding: utf-8 -*-
"""
Regulatory Compliance Workflow (Energy Audit)
==================================================

3-phase workflow for EED energy audit compliance within
PACK-031 Industrial Energy Audit Pack.

Phases:
    1. EEDObligationCheck    -- Assess facility obligation under EED Article 8,
                                check ISO 50001/EMAS exemptions
    2. AuditScheduling       -- Calculate next audit due date, track deadlines
    3. ComplianceReporting   -- Generate EED compliance report, national authority data

The workflow follows GreenLang zero-hallucination principles: all
obligation checks use deterministic threshold logic per EED Article 8
and national transposition rules. No LLM calls in numeric path.

Schedule: quarterly
Estimated duration: 30 minutes

Author: GreenLang Team
Version: 31.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ObligationStatus(str, Enum):
    """EED obligation status."""

    OBLIGATED = "obligated"
    EXEMPT_ISO50001 = "exempt_iso_50001"
    EXEMPT_EMAS = "exempt_emas"
    EXEMPT_SME = "exempt_sme"
    NOT_APPLICABLE = "not_applicable"
    UNDER_REVIEW = "under_review"


class AuditStatus(str, Enum):
    """Audit compliance status."""

    COMPLIANT = "compliant"
    DUE_SOON = "due_soon"  # Within 6 months
    OVERDUE = "overdue"
    NOT_REQUIRED = "not_required"
    FIRST_AUDIT_NEEDED = "first_audit_needed"


class ComplianceLevel(str, Enum):
    """Overall compliance level."""

    FULLY_COMPLIANT = "fully_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


class MemberState(str, Enum):
    """EU member state for national transposition rules."""

    AT = "AT"
    BE = "BE"
    BG = "BG"
    CZ = "CZ"
    DE = "DE"
    DK = "DK"
    EE = "EE"
    ES = "ES"
    FI = "FI"
    FR = "FR"
    GR = "GR"
    HR = "HR"
    HU = "HU"
    IE = "IE"
    IT = "IT"
    LT = "LT"
    LU = "LU"
    LV = "LV"
    MT = "MT"
    NL = "NL"
    PL = "PL"
    PT = "PT"
    RO = "RO"
    SE = "SE"
    SI = "SI"
    SK = "SK"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class FacilityObligationData(BaseModel):
    """Facility data for obligation assessment."""

    facility_id: str = Field(default="", description="Facility identifier")
    facility_name: str = Field(default="", description="Facility name")
    company_name: str = Field(default="", description="Legal entity name")
    member_state: str = Field(default="DE", description="EU member state ISO code")
    employee_count: int = Field(default=0, ge=0, description="Total employees")
    annual_turnover_eur: float = Field(default=0.0, ge=0.0, description="Annual turnover EUR")
    annual_balance_sheet_eur: float = Field(default=0.0, ge=0.0, description="Balance sheet total EUR")
    annual_energy_consumption_mwh: float = Field(default=0.0, ge=0.0, description="Annual MWh")
    is_large_enterprise: bool = Field(default=False, description="Meets large enterprise criteria")
    has_iso_50001: bool = Field(default=False, description="ISO 50001 certified")
    iso_50001_expiry_date: str = Field(default="", description="ISO 50001 cert expiry YYYY-MM-DD")
    has_emas: bool = Field(default=False, description="EMAS registered")
    emas_registration_number: str = Field(default="", description="EMAS registration number")
    last_audit_date: str = Field(default="", description="Last EN 16247 audit YYYY-MM-DD")
    last_audit_type: str = Field(default="", description="standard|detailed")
    nace_code: str = Field(default="", description="NACE code")
    is_energy_intensive: bool = Field(default=False, description="Energy-intensive industry")


class EEDObligationResult(BaseModel):
    """Result of EED obligation assessment."""

    facility_id: str = Field(default="")
    obligation_status: ObligationStatus = Field(default=ObligationStatus.UNDER_REVIEW)
    is_large_enterprise: bool = Field(default=False)
    is_sme: bool = Field(default=False)
    exemption_reason: str = Field(default="")
    eed_article_reference: str = Field(default="EED Article 8")
    national_law_reference: str = Field(default="")
    energy_threshold_mwh: float = Field(default=0.0, description="National energy threshold MWh")
    assessment_notes: List[str] = Field(default_factory=list)


class AuditScheduleEntry(BaseModel):
    """Audit schedule and deadline tracking."""

    facility_id: str = Field(default="")
    audit_status: AuditStatus = Field(default=AuditStatus.NOT_REQUIRED)
    last_audit_date: str = Field(default="", description="YYYY-MM-DD")
    next_audit_due: str = Field(default="", description="YYYY-MM-DD")
    days_until_due: int = Field(default=0, description="Days until due (negative=overdue)")
    audit_cycle_years: int = Field(default=4, description="Audit cycle in years")
    iso_50001_covers: bool = Field(default=False)
    emas_covers: bool = Field(default=False)
    notes: List[str] = Field(default_factory=list)


class ComplianceReportData(BaseModel):
    """Compliance report data for national authority submission."""

    report_id: str = Field(default_factory=lambda: f"rpt-{uuid.uuid4().hex[:8]}")
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    company_name: str = Field(default="")
    member_state: str = Field(default="")
    reporting_year: int = Field(default=2025)
    obligation_status: str = Field(default="")
    exemption_type: str = Field(default="none")
    last_audit_date: str = Field(default="")
    next_audit_due: str = Field(default="")
    audit_standard: str = Field(default="EN 16247-1:2022")
    auditor_qualification: str = Field(default="")
    total_energy_consumption_mwh: float = Field(default=0.0)
    identified_savings_mwh: float = Field(default=0.0)
    implemented_savings_mwh: float = Field(default=0.0)
    compliance_level: ComplianceLevel = Field(default=ComplianceLevel.NOT_ASSESSED)
    submission_ready: bool = Field(default=False)


class RegulatoryComplianceInput(BaseModel):
    """Input data model for RegulatoryComplianceWorkflow."""

    facilities: List[FacilityObligationData] = Field(default_factory=list)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    identified_savings_mwh: float = Field(default=0.0, ge=0.0, description="From audit")
    implemented_savings_mwh: float = Field(default=0.0, ge=0.0)
    auditor_name: str = Field(default="")
    auditor_qualification: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class RegulatoryComplianceResult(BaseModel):
    """Complete result from regulatory compliance workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="regulatory_compliance")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    obligations: List[EEDObligationResult] = Field(default_factory=list)
    schedules: List[AuditScheduleEntry] = Field(default_factory=list)
    reports: List[ComplianceReportData] = Field(default_factory=list)
    facilities_obligated: int = Field(default=0)
    facilities_exempt: int = Field(default=0)
    facilities_overdue: int = Field(default=0)
    facilities_compliant: int = Field(default=0)
    overall_compliance: ComplianceLevel = Field(default=ComplianceLevel.NOT_ASSESSED)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# EED NATIONAL TRANSPOSITION RULES (Zero-Hallucination)
# =============================================================================

# EED Article 8: large enterprises must conduct energy audits every 4 years
# Large enterprise: >= 250 employees OR turnover > EUR 50M AND balance sheet > EUR 43M
EED_EMPLOYEE_THRESHOLD = 250
EED_TURNOVER_THRESHOLD_EUR = 50_000_000.0
EED_BALANCE_SHEET_THRESHOLD_EUR = 43_000_000.0
EED_AUDIT_CYCLE_YEARS = 4

# National variations
NATIONAL_RULES: Dict[str, Dict[str, Any]] = {
    "DE": {
        "law": "Energiedienstleistungsgesetz (EDL-G)",
        "cycle_years": 4,
        "energy_threshold_mwh": 0.0,  # All large enterprises
        "allows_iso50001_exemption": True,
        "allows_emas_exemption": True,
        "national_register": "BAFA",
    },
    "FR": {
        "law": "Code de l'Energie L.233-1",
        "cycle_years": 4,
        "energy_threshold_mwh": 0.0,
        "allows_iso50001_exemption": True,
        "allows_emas_exemption": True,
        "national_register": "ADEME",
    },
    "IT": {
        "law": "D.Lgs. 102/2014",
        "cycle_years": 4,
        "energy_threshold_mwh": 0.0,
        "allows_iso50001_exemption": True,
        "allows_emas_exemption": True,
        "national_register": "ENEA",
    },
    "ES": {
        "law": "Real Decreto 56/2016",
        "cycle_years": 4,
        "energy_threshold_mwh": 0.0,
        "allows_iso50001_exemption": True,
        "allows_emas_exemption": True,
        "national_register": "IDAE",
    },
    "NL": {
        "law": "Wet milieubeheer / EED Implementation",
        "cycle_years": 4,
        "energy_threshold_mwh": 0.0,
        "allows_iso50001_exemption": True,
        "allows_emas_exemption": True,
        "national_register": "RVO",
    },
    "PL": {
        "law": "Ustawa o efektywnosci energetycznej",
        "cycle_years": 4,
        "energy_threshold_mwh": 0.0,
        "allows_iso50001_exemption": True,
        "allows_emas_exemption": True,
        "national_register": "URE",
    },
    "SE": {
        "law": "Lag om energikartlaggning i stora foretag",
        "cycle_years": 4,
        "energy_threshold_mwh": 0.0,
        "allows_iso50001_exemption": True,
        "allows_emas_exemption": True,
        "national_register": "Energimyndigheten",
    },
}

# Default rules for member states not explicitly listed
DEFAULT_NATIONAL_RULES: Dict[str, Any] = {
    "law": "EED Article 8 national transposition",
    "cycle_years": 4,
    "energy_threshold_mwh": 0.0,
    "allows_iso50001_exemption": True,
    "allows_emas_exemption": True,
    "national_register": "National energy authority",
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class RegulatoryComplianceWorkflow:
    """
    3-phase EED energy audit compliance workflow.

    Checks facility obligations under EED Article 8, calculates
    audit scheduling and deadlines, and generates compliance
    reports for national authority submission.

    Zero-hallucination: all obligation checks use deterministic
    threshold comparisons per EED and national transposition rules.

    Attributes:
        workflow_id: Unique execution identifier.
        _obligations: Per-facility obligation results.
        _schedules: Audit schedule entries.
        _reports: Compliance report data.

    Example:
        >>> wf = RegulatoryComplianceWorkflow()
        >>> inp = RegulatoryComplianceInput(facilities=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize RegulatoryComplianceWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._obligations: List[EEDObligationResult] = []
        self._schedules: List[AuditScheduleEntry] = []
        self._reports: List[ComplianceReportData] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[RegulatoryComplianceInput] = None,
        facilities: Optional[List[FacilityObligationData]] = None,
    ) -> RegulatoryComplianceResult:
        """
        Execute the 3-phase regulatory compliance workflow.

        Args:
            input_data: Full input model (preferred).
            facilities: Facility list (fallback).

        Returns:
            RegulatoryComplianceResult with obligations, schedules, reports.
        """
        if input_data is None:
            input_data = RegulatoryComplianceInput(facilities=facilities or [])

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting regulatory compliance workflow %s for %d facilities",
            self.workflow_id, len(input_data.facilities),
        )

        self._phase_results = []
        self._obligations = []
        self._schedules = []
        self._reports = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_eed_obligation_check(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_audit_scheduling(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_compliance_reporting(input_data)
            self._phase_results.append(phase3)

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Regulatory compliance workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()
        obligated = sum(1 for o in self._obligations if o.obligation_status == ObligationStatus.OBLIGATED)
        exempt = sum(1 for o in self._obligations if o.obligation_status in (
            ObligationStatus.EXEMPT_ISO50001, ObligationStatus.EXEMPT_EMAS, ObligationStatus.EXEMPT_SME
        ))
        overdue = sum(1 for s in self._schedules if s.audit_status == AuditStatus.OVERDUE)
        compliant = sum(1 for r in self._reports if r.compliance_level == ComplianceLevel.FULLY_COMPLIANT)

        if obligated + exempt == len(input_data.facilities) and overdue == 0:
            overall = ComplianceLevel.FULLY_COMPLIANT
        elif overdue > 0:
            overall = ComplianceLevel.NON_COMPLIANT
        elif compliant > 0:
            overall = ComplianceLevel.PARTIALLY_COMPLIANT
        else:
            overall = ComplianceLevel.NOT_ASSESSED

        result = RegulatoryComplianceResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            obligations=self._obligations,
            schedules=self._schedules,
            reports=self._reports,
            facilities_obligated=obligated,
            facilities_exempt=exempt,
            facilities_overdue=overdue,
            facilities_compliant=compliant,
            overall_compliance=overall,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Regulatory compliance workflow %s completed in %.2fs obligated=%d exempt=%d overdue=%d",
            self.workflow_id, elapsed, obligated, exempt, overdue,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: EED Obligation Check
    # -------------------------------------------------------------------------

    async def _phase_eed_obligation_check(
        self, input_data: RegulatoryComplianceInput
    ) -> PhaseResult:
        """Assess each facility's obligation under EED Article 8."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for facility in input_data.facilities:
            obligation = self._assess_obligation(facility)
            self._obligations.append(obligation)

        outputs["facilities_assessed"] = len(self._obligations)
        outputs["obligated"] = sum(
            1 for o in self._obligations if o.obligation_status == ObligationStatus.OBLIGATED
        )
        outputs["exempt_iso50001"] = sum(
            1 for o in self._obligations if o.obligation_status == ObligationStatus.EXEMPT_ISO50001
        )
        outputs["exempt_emas"] = sum(
            1 for o in self._obligations if o.obligation_status == ObligationStatus.EXEMPT_EMAS
        )
        outputs["exempt_sme"] = sum(
            1 for o in self._obligations if o.obligation_status == ObligationStatus.EXEMPT_SME
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 EEDObligationCheck: %d assessed, %d obligated",
            len(self._obligations), outputs["obligated"],
        )
        return PhaseResult(
            phase_name="eed_obligation_check", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _assess_obligation(self, facility: FacilityObligationData) -> EEDObligationResult:
        """Assess a single facility's EED obligation."""
        rules = NATIONAL_RULES.get(facility.member_state, DEFAULT_NATIONAL_RULES)
        notes: List[str] = []

        # Step 1: Determine if large enterprise
        is_large = facility.is_large_enterprise
        if not is_large:
            is_large = (
                facility.employee_count >= EED_EMPLOYEE_THRESHOLD
                or (
                    facility.annual_turnover_eur > EED_TURNOVER_THRESHOLD_EUR
                    and facility.annual_balance_sheet_eur > EED_BALANCE_SHEET_THRESHOLD_EUR
                )
            )
        is_sme = not is_large

        if is_sme:
            notes.append("SME: below EED Article 8 thresholds")
            return EEDObligationResult(
                facility_id=facility.facility_id,
                obligation_status=ObligationStatus.EXEMPT_SME,
                is_large_enterprise=False,
                is_sme=True,
                exemption_reason="SME: does not meet large enterprise thresholds",
                eed_article_reference="EED Article 8(4)",
                national_law_reference=rules["law"],
                assessment_notes=notes,
            )

        notes.append(f"Large enterprise: {facility.employee_count} employees, "
                      f"turnover {facility.annual_turnover_eur/1e6:.1f}M EUR")

        # Step 2: Check ISO 50001 exemption
        if facility.has_iso_50001 and rules.get("allows_iso50001_exemption", True):
            expiry_valid = True
            if facility.iso_50001_expiry_date:
                try:
                    expiry = datetime.strptime(facility.iso_50001_expiry_date, "%Y-%m-%d")
                    if expiry < datetime.utcnow():
                        expiry_valid = False
                        notes.append("ISO 50001 certificate has expired")
                except ValueError:
                    expiry_valid = False
                    notes.append("Invalid ISO 50001 expiry date format")

            if expiry_valid:
                notes.append("ISO 50001 certified: exempt from mandatory audit")
                return EEDObligationResult(
                    facility_id=facility.facility_id,
                    obligation_status=ObligationStatus.EXEMPT_ISO50001,
                    is_large_enterprise=True,
                    is_sme=False,
                    exemption_reason="ISO 50001:2018 certified (covers EnMS requirements)",
                    eed_article_reference="EED Article 8(6)",
                    national_law_reference=rules["law"],
                    assessment_notes=notes,
                )

        # Step 3: Check EMAS exemption
        if facility.has_emas and rules.get("allows_emas_exemption", True):
            notes.append("EMAS registered: exempt from mandatory audit")
            return EEDObligationResult(
                facility_id=facility.facility_id,
                obligation_status=ObligationStatus.EXEMPT_EMAS,
                is_large_enterprise=True,
                is_sme=False,
                exemption_reason="EMAS registered (includes energy audit in environmental review)",
                eed_article_reference="EED Article 8(6)",
                national_law_reference=rules["law"],
                assessment_notes=notes,
            )

        # Step 4: Obligated
        notes.append(f"Subject to mandatory energy audit per {rules['law']}")
        return EEDObligationResult(
            facility_id=facility.facility_id,
            obligation_status=ObligationStatus.OBLIGATED,
            is_large_enterprise=True,
            is_sme=False,
            eed_article_reference="EED Article 8(4)",
            national_law_reference=rules["law"],
            energy_threshold_mwh=rules.get("energy_threshold_mwh", 0.0),
            assessment_notes=notes,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Audit Scheduling
    # -------------------------------------------------------------------------

    async def _phase_audit_scheduling(
        self, input_data: RegulatoryComplianceInput
    ) -> PhaseResult:
        """Calculate next audit due dates and track compliance deadlines."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for i, facility in enumerate(input_data.facilities):
            obligation = self._obligations[i] if i < len(self._obligations) else None
            schedule = self._calculate_schedule(facility, obligation)
            self._schedules.append(schedule)

            if schedule.audit_status == AuditStatus.OVERDUE:
                warnings.append(
                    f"Facility {facility.facility_id}: audit overdue by "
                    f"{abs(schedule.days_until_due)} days"
                )
            elif schedule.audit_status == AuditStatus.DUE_SOON:
                warnings.append(
                    f"Facility {facility.facility_id}: audit due in "
                    f"{schedule.days_until_due} days"
                )

        outputs["schedules_calculated"] = len(self._schedules)
        outputs["overdue_count"] = sum(
            1 for s in self._schedules if s.audit_status == AuditStatus.OVERDUE
        )
        outputs["due_soon_count"] = sum(
            1 for s in self._schedules if s.audit_status == AuditStatus.DUE_SOON
        )
        outputs["compliant_count"] = sum(
            1 for s in self._schedules if s.audit_status == AuditStatus.COMPLIANT
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 AuditScheduling: %d schedules, %d overdue, %d due soon",
            len(self._schedules), outputs["overdue_count"], outputs["due_soon_count"],
        )
        return PhaseResult(
            phase_name="audit_scheduling", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _calculate_schedule(
        self,
        facility: FacilityObligationData,
        obligation: Optional[EEDObligationResult],
    ) -> AuditScheduleEntry:
        """Calculate audit schedule for a facility."""
        notes: List[str] = []

        # Not obligated
        if obligation and obligation.obligation_status != ObligationStatus.OBLIGATED:
            covers_iso = obligation.obligation_status == ObligationStatus.EXEMPT_ISO50001
            covers_emas = obligation.obligation_status == ObligationStatus.EXEMPT_EMAS
            return AuditScheduleEntry(
                facility_id=facility.facility_id,
                audit_status=AuditStatus.NOT_REQUIRED,
                iso_50001_covers=covers_iso,
                emas_covers=covers_emas,
                notes=[f"Exempt: {obligation.exemption_reason}"],
            )

        rules = NATIONAL_RULES.get(facility.member_state, DEFAULT_NATIONAL_RULES)
        cycle_years = rules.get("cycle_years", EED_AUDIT_CYCLE_YEARS)

        # No previous audit
        if not facility.last_audit_date:
            notes.append("No previous audit on record; first audit required")
            return AuditScheduleEntry(
                facility_id=facility.facility_id,
                audit_status=AuditStatus.FIRST_AUDIT_NEEDED,
                audit_cycle_years=cycle_years,
                days_until_due=0,
                notes=notes,
            )

        # Calculate next due date
        try:
            last_audit = datetime.strptime(facility.last_audit_date, "%Y-%m-%d")
        except ValueError:
            notes.append("Invalid last_audit_date format")
            return AuditScheduleEntry(
                facility_id=facility.facility_id,
                audit_status=AuditStatus.FIRST_AUDIT_NEEDED,
                notes=notes,
            )

        next_due = last_audit + timedelta(days=cycle_years * 365)
        days_until = (next_due - datetime.utcnow()).days

        if days_until < 0:
            status = AuditStatus.OVERDUE
        elif days_until <= 180:
            status = AuditStatus.DUE_SOON
        else:
            status = AuditStatus.COMPLIANT

        return AuditScheduleEntry(
            facility_id=facility.facility_id,
            audit_status=status,
            last_audit_date=facility.last_audit_date,
            next_audit_due=next_due.strftime("%Y-%m-%d"),
            days_until_due=days_until,
            audit_cycle_years=cycle_years,
            notes=notes,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Compliance Reporting
    # -------------------------------------------------------------------------

    async def _phase_compliance_reporting(
        self, input_data: RegulatoryComplianceInput
    ) -> PhaseResult:
        """Generate EED compliance reports for national authority submission."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for i, facility in enumerate(input_data.facilities):
            obligation = self._obligations[i] if i < len(self._obligations) else None
            schedule = self._schedules[i] if i < len(self._schedules) else None
            report = self._generate_compliance_report(facility, obligation, schedule, input_data)
            self._reports.append(report)

        outputs["reports_generated"] = len(self._reports)
        outputs["submission_ready_count"] = sum(1 for r in self._reports if r.submission_ready)
        outputs["fully_compliant_count"] = sum(
            1 for r in self._reports if r.compliance_level == ComplianceLevel.FULLY_COMPLIANT
        )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ComplianceReporting: %d reports, %d submission-ready",
            len(self._reports), outputs["submission_ready_count"],
        )
        return PhaseResult(
            phase_name="compliance_reporting", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _generate_compliance_report(
        self,
        facility: FacilityObligationData,
        obligation: Optional[EEDObligationResult],
        schedule: Optional[AuditScheduleEntry],
        input_data: RegulatoryComplianceInput,
    ) -> ComplianceReportData:
        """Generate compliance report for a single facility."""
        # Determine compliance level
        if obligation and obligation.obligation_status != ObligationStatus.OBLIGATED:
            compliance = ComplianceLevel.FULLY_COMPLIANT
            exemption = obligation.obligation_status.value
        elif schedule and schedule.audit_status == AuditStatus.COMPLIANT:
            compliance = ComplianceLevel.FULLY_COMPLIANT
            exemption = "none"
        elif schedule and schedule.audit_status in (AuditStatus.OVERDUE, AuditStatus.FIRST_AUDIT_NEEDED):
            compliance = ComplianceLevel.NON_COMPLIANT
            exemption = "none"
        elif schedule and schedule.audit_status == AuditStatus.DUE_SOON:
            compliance = ComplianceLevel.PARTIALLY_COMPLIANT
            exemption = "none"
        else:
            compliance = ComplianceLevel.NOT_ASSESSED
            exemption = "none"

        submission_ready = compliance in (ComplianceLevel.FULLY_COMPLIANT, ComplianceLevel.PARTIALLY_COMPLIANT)

        return ComplianceReportData(
            facility_id=facility.facility_id,
            facility_name=facility.facility_name,
            company_name=facility.company_name,
            member_state=facility.member_state,
            reporting_year=input_data.reporting_year,
            obligation_status=obligation.obligation_status.value if obligation else "unknown",
            exemption_type=exemption,
            last_audit_date=facility.last_audit_date,
            next_audit_due=schedule.next_audit_due if schedule else "",
            audit_standard="EN 16247-1:2022",
            auditor_qualification=input_data.auditor_qualification,
            total_energy_consumption_mwh=facility.annual_energy_consumption_mwh,
            identified_savings_mwh=input_data.identified_savings_mwh,
            implemented_savings_mwh=input_data.implemented_savings_mwh,
            compliance_level=compliance,
            submission_ready=submission_ready,
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: RegulatoryComplianceResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
