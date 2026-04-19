# -*- coding: utf-8 -*-
"""Compliance Rules Schemas for GL-010 EmissionsGuardian"""

from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import hashlib


class AveragingPeriod(str, Enum):
    HOURLY = "hourly"
    ROLLING_3HOUR = "rolling_3hour"
    ROLLING_24HOUR = "rolling_24hour"
    DAILY = "daily"
    ROLLING_30DAY = "rolling_30day"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    OZONE_SEASON = "ozone_season"


class OperatingState(str, Enum):
    NORMAL = "normal"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    MALFUNCTION = "malfunction"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    LOAD_FOLLOWING = "load_following"
    OFFLINE = "offline"


class ExceedanceSeverity(str, Enum):
    INFORMATIONAL = "informational"
    WARNING = "warning"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class CorrectiveActionState(str, Enum):
    IDENTIFIED = "identified"
    ACKNOWLEDGED = "acknowledged"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    CORRECTIVE_ACTION_PLANNED = "corrective_action_planned"
    CORRECTIVE_ACTION_IN_PROGRESS = "corrective_action_in_progress"
    VERIFICATION = "verification"
    CLOSED = "closed"
    REOPENED = "reopened"


class RegulatoryProgram(str, Enum):
    ARP = "arp"
    CSAPR = "csapr"
    MATS = "mats"
    NSPS = "nsps"
    NESHAP = "neshap"
    SIP = "sip"
    TITLE_V = "title_v"
    PSD = "psd"


class RuleVersion(BaseModel):
    """Version tracking for regulatory rules."""
    version: str = Field(..., description="Semantic version string")
    effective_date: date = Field(..., description="Date this version becomes effective")
    expiration_date: Optional[date] = Field(None, description="Date this version expires")
    change_reason: str = Field(..., description="Reason for version change")
    approved_by: str = Field(..., description="Approver name/ID")
    approval_date: datetime = Field(default_factory=datetime.now)
    supersedes_version: Optional[str] = Field(None, description="Previous version")

    @field_validator("version")
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        parts = v.split(".")
        if len(parts) < 2:
            raise ValueError("Version must be in semantic format")
        return v

    def is_active(self, reference_date: Optional[date] = None) -> bool:
        check_date = reference_date or date.today()
        if check_date < self.effective_date:
            return False
        if self.expiration_date and check_date > self.expiration_date:
            return False
        return True


class OperatingCondition(BaseModel):
    """Operating conditions that affect rule applicability."""
    parameter: str = Field(..., description="Parameter name")
    operator: str = Field(..., description="Comparison operator")
    value: Union[float, str, List[str]] = Field(..., description="Condition value")

    @field_validator("operator")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        valid_operators = {"eq", "ne", "gt", "lt", "ge", "le", "in", "not_in", "between"}
        if v.lower() not in valid_operators:
            raise ValueError(f"Operator must be one of: {valid_operators}")
        return v.lower()

    def evaluate(self, actual_value: Any) -> bool:
        op = self.operator
        if op == "eq":
            return actual_value == self.value
        elif op == "ne":
            return actual_value != self.value
        elif op == "gt":
            return actual_value > self.value
        elif op == "lt":
            return actual_value < self.value
        elif op == "ge":
            return actual_value >= self.value
        elif op == "le":
            return actual_value <= self.value
        elif op == "in":
            return actual_value in self.value
        elif op == "not_in":
            return actual_value not in self.value
        elif op == "between":
            return self.value[0] <= actual_value <= self.value[1]
        return False


class EffectiveDateRange(BaseModel):
    """Date range for permit/rule applicability."""
    start_date: date = Field(..., description="Start date of applicability")
    end_date: Optional[date] = Field(None, description="End date of applicability")
    description: Optional[str] = Field(None, description="Description of date range")

    def is_active(self, check_date: Optional[date] = None) -> bool:
        dt = check_date or date.today()
        if dt < self.start_date:
            return False
        if self.end_date and dt > self.end_date:
            return False
        return True


class PermitRule(BaseModel):
    """Permit rule defining emission limits and conditions."""
    rule_id: str = Field(..., description="Unique rule identifier")
    permit_id: str = Field(..., description="Parent permit identifier")
    pollutant: str = Field(..., description="Pollutant code")
    unit_id: Optional[str] = Field(None, description="Specific unit")
    limit_value: Decimal = Field(..., gt=0, description="Emission limit value")
    limit_unit: str = Field(..., description="Unit of measurement")
    averaging_period: AveragingPeriod = Field(..., description="Averaging period")
    warning_threshold_pct: float = Field(90.0, ge=0, le=100)
    action_threshold_pct: float = Field(100.0, ge=0, le=100)
    operating_conditions: List[OperatingCondition] = Field(default_factory=list)
    applicable_states: List[OperatingState] = Field(default=[OperatingState.NORMAL])
    exemption_states: List[OperatingState] = Field(default_factory=list)
    effective_dates: EffectiveDateRange = Field(...)
    version: RuleVersion = Field(...)
    regulatory_program: RegulatoryProgram = Field(...)
    regulation_citation: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    notes: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def is_applicable(
        self,
        operating_state: OperatingState,
        operating_params: Dict[str, Any],
        check_date: Optional[date] = None
    ) -> bool:
        if not self.effective_dates.is_active(check_date):
            return False
        if not self.version.is_active(check_date):
            return False
        if operating_state in self.exemption_states:
            return False
        if self.applicable_states and operating_state not in self.applicable_states:
            return False
        for condition in self.operating_conditions:
            param_value = operating_params.get(condition.parameter)
            if param_value is not None and not condition.evaluate(param_value):
                return False
        return True

    def calculate_provenance_hash(self) -> str:
        content = (
            f"{self.rule_id}|{self.permit_id}|{self.pollutant}|"
            f"{self.limit_value}|{self.limit_unit}|{self.averaging_period}|"
            f"{self.version.version}|{self.effective_dates.start_date}"
        )
        return hashlib.sha256(content.encode()).hexdigest()


class ComplianceSchedule(BaseModel):
    """Compliance schedule for reporting deadlines."""
    schedule_id: str = Field(..., description="Unique schedule identifier")
    regulatory_program: RegulatoryProgram = Field(...)
    reporting_period: str = Field(...)
    period_start: date = Field(...)
    period_end: date = Field(...)
    submission_deadline: datetime = Field(...)
    certification_deadline: Optional[datetime] = Field(None)
    is_submitted: bool = Field(False)
    submission_date: Optional[datetime] = Field(None)
    is_certified: bool = Field(False)
    certification_date: Optional[datetime] = Field(None)
    report_type: str = Field(...)
    facility_id: str = Field(...)
    unit_ids: List[str] = Field(default_factory=list)
    version: RuleVersion = Field(...)

    def days_until_deadline(self) -> int:
        delta = self.submission_deadline - datetime.now()
        return delta.days

    def is_overdue(self) -> bool:
        return not self.is_submitted and datetime.now() > self.submission_deadline

    def get_status(self) -> str:
        if self.is_certified:
            return "CERTIFIED"
        elif self.is_submitted:
            return "SUBMITTED"
        elif self.is_overdue():
            return "OVERDUE"
        elif self.days_until_deadline() <= 7:
            return "DUE_SOON"
        else:
            return "PENDING"


class ExceedanceEvent(BaseModel):
    """Exceedance event for tracking permit violations."""
    event_id: str = Field(...)
    rule_id: str = Field(...)
    permit_id: str = Field(...)
    facility_id: str = Field(...)
    unit_id: str = Field(...)
    pollutant: str = Field(...)
    limit_value: Decimal = Field(...)
    measured_value: Decimal = Field(...)
    limit_unit: str = Field(...)
    severity: ExceedanceSeverity = Field(...)
    exceedance_pct: float = Field(...)
    threshold_exceeded: str = Field(...)
    start_time: datetime = Field(...)
    end_time: Optional[datetime] = Field(None)
    duration_hours: Optional[float] = Field(None)
    operating_state: OperatingState = Field(...)
    operating_conditions: Dict[str, Any] = Field(default_factory=dict)
    root_cause: Optional[str] = Field(None)
    root_cause_category: Optional[str] = Field(None)
    data_quality_flag: Optional[str] = Field(None)
    is_substituted_data: bool = Field(False)
    requires_deviation_report: bool = Field(False)
    deviation_report_submitted: bool = Field(False)
    notification_sent: bool = Field(False)
    notification_recipients: List[str] = Field(default_factory=list)
    detected_at: datetime = Field(default_factory=datetime.now)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def calculate_duration(self) -> Optional[float]:
        if self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() / 3600
        return None

    def determine_severity(self) -> ExceedanceSeverity:
        pct = self.exceedance_pct
        duration = self.duration_hours or 0
        if pct > 100 or duration > 24:
            return ExceedanceSeverity.CRITICAL
        elif pct > 50 or duration > 12:
            return ExceedanceSeverity.MAJOR
        elif pct > 10 or duration > 6:
            return ExceedanceSeverity.MODERATE
        elif pct > 0:
            return ExceedanceSeverity.MINOR
        elif pct > -10:
            return ExceedanceSeverity.WARNING
        else:
            return ExceedanceSeverity.INFORMATIONAL


class CorrectiveAction(BaseModel):
    """Corrective action workflow for exceedance events."""
    action_id: str = Field(...)
    event_id: str = Field(...)
    title: str = Field(...)
    description: str = Field(...)
    action_type: str = Field(...)
    state: CorrectiveActionState = Field(default=CorrectiveActionState.IDENTIFIED)
    state_history: List[Dict[str, Any]] = Field(default_factory=list)
    assigned_to: Optional[str] = Field(None)
    assigned_by: Optional[str] = Field(None)
    assigned_date: Optional[datetime] = Field(None)
    target_completion_date: Optional[date] = Field(None)
    actual_completion_date: Optional[date] = Field(None)
    is_overdue: bool = Field(False)
    root_cause_analysis: Optional[str] = Field(None)
    root_cause_category: Optional[str] = Field(None)
    contributing_factors: List[str] = Field(default_factory=list)
    immediate_actions: List[str] = Field(default_factory=list)
    long_term_actions: List[str] = Field(default_factory=list)
    preventive_measures: List[str] = Field(default_factory=list)
    verification_method: Optional[str] = Field(None)
    verification_date: Optional[datetime] = Field(None)
    verified_by: Optional[str] = Field(None)
    verification_notes: Optional[str] = Field(None)
    attachments: List[str] = Field(default_factory=list)
    comments: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(...)

    def transition_state(
        self,
        new_state: CorrectiveActionState,
        transitioned_by: str,
        notes: Optional[str] = None
    ) -> None:
        old_state = self.state
        self.state_history.append({
            "from_state": old_state.value,
            "to_state": new_state.value,
            "transitioned_by": transitioned_by,
            "transitioned_at": datetime.now().isoformat(),
            "notes": notes
        })
        self.state = new_state
        self.updated_at = datetime.now()

    def is_closed(self) -> bool:
        return self.state == CorrectiveActionState.CLOSED

    def days_open(self) -> int:
        end_date = self.actual_completion_date or date.today()
        return (end_date - self.created_at.date()).days


class ComplianceStatus(BaseModel):
    """Aggregated compliance status for a facility or unit."""
    facility_id: str = Field(...)
    unit_id: Optional[str] = Field(None)
    status_date: datetime = Field(default_factory=datetime.now)
    overall_status: str = Field(...)
    compliance_score: float = Field(..., ge=0, le=100)
    pollutant_status: Dict[str, str] = Field(default_factory=dict)
    program_status: Dict[str, str] = Field(default_factory=dict)
    active_exceedances: int = Field(0)
    active_warnings: int = Field(0)
    open_corrective_actions: int = Field(0)
    overdue_actions: int = Field(0)
    pending_submissions: int = Field(0)
    overdue_submissions: int = Field(0)
    days_in_compliance: int = Field(0)
    last_exceedance_date: Optional[datetime] = Field(None)
    next_deadline: Optional[datetime] = Field(None)
    exceedances_this_quarter: int = Field(0)
    exceedances_this_year: int = Field(0)
    total_applicable_rules: int = Field(0)
    rules_in_compliance: int = Field(0)
    rules_in_warning: int = Field(0)
    rules_exceeded: int = Field(0)
    provenance_hash: str = Field(...)
    calculated_at: datetime = Field(default_factory=datetime.now)

    def calculate_compliance_score(self) -> float:
        if self.total_applicable_rules == 0:
            return 100.0
        compliant_weight = self.rules_in_compliance * 100
        warning_weight = self.rules_in_warning * 50
        total_weight = compliant_weight + warning_weight
        return total_weight / self.total_applicable_rules

    def determine_overall_status(self) -> str:
        if self.active_exceedances > 0:
            return "NON_COMPLIANT"
        elif self.active_warnings > 0 or self.overdue_actions > 0:
            return "WARNING"
        else:
            return "COMPLIANT"

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "facility_id": self.facility_id,
            "unit_id": self.unit_id,
            "status": self.overall_status,
            "score": self.compliance_score,
            "active_exceedances": self.active_exceedances,
            "open_actions": self.open_corrective_actions,
            "days_in_compliance": self.days_in_compliance,
            "next_deadline": self.next_deadline.isoformat() if self.next_deadline else None,
            "status_date": self.status_date.isoformat()
        }


class SubstitutionDataRecord(BaseModel):
    """Substitution data record per EPA 40 CFR 75 Appendix D."""
    record_id: str = Field(...)
    facility_id: str = Field(...)
    unit_id: str = Field(...)
    monitor_id: str = Field(...)
    parameter: str = Field(...)
    original_value: Optional[Decimal] = Field(None)
    substituted_value: Decimal = Field(...)
    substitution_type: str = Field(...)
    start_hour: datetime = Field(...)
    end_hour: datetime = Field(...)
    hours_substituted: int = Field(..., ge=1)
    quarter: int = Field(..., ge=1, le=4)
    year: int = Field(...)
    cumulative_substitute_hours: int = Field(0)
    regulation_citation: str = Field(default="40 CFR 75 Appendix D")
    reason_code: str = Field(...)
    applied_at: datetime = Field(default_factory=datetime.now)
    applied_by: str = Field(...)
    provenance_hash: str = Field(...)

    def calculate_provenance_hash(self) -> str:
        content = (
            f"{self.record_id}|{self.unit_id}|{self.monitor_id}|"
            f"{self.parameter}|{self.substituted_value}|"
            f"{self.start_hour.isoformat()}|{self.end_hour.isoformat()}"
        )
        return hashlib.sha256(content.encode()).hexdigest()
