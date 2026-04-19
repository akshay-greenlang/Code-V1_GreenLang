"""
GL-071: Regulatory Guardian Agent (REGULATORY-GUARDIAN)

This module implements the RegulatoryGuardianAgent for comprehensive regulatory
compliance monitoring and reporting for EPA and OSHA standards.

Standards Reference:
    - EPA Clean Air Act (CAA)
    - EPA Clean Water Act (CWA)
    - OSHA 29 CFR 1910
    - EPA 40 CFR (Environmental Regulations)

Example:
    >>> agent = RegulatoryGuardianAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Compliance Score: {result.overall_compliance_score}")
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class RegulatoryAgency(str, Enum):
    EPA = "EPA"
    OSHA = "OSHA"
    STATE_EPA = "STATE_EPA"
    LOCAL = "LOCAL"


class ComplianceStatus(str, Enum):
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    AT_RISK = "AT_RISK"
    UNDER_REVIEW = "UNDER_REVIEW"


class ViolationSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    ADVISORY = "ADVISORY"


class ReportingFrequency(str, Enum):
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    ANNUALLY = "ANNUALLY"


# =============================================================================
# INPUT MODELS
# =============================================================================

class EmissionLimit(BaseModel):
    pollutant: str = Field(..., description="Pollutant name (e.g., NOx, SOx, PM2.5)")
    limit_value: float = Field(..., ge=0, description="Emission limit value")
    limit_unit: str = Field(..., description="Unit (e.g., ppm, lb/hr, tons/year)")
    current_value: float = Field(..., ge=0, description="Current emission level")
    measurement_timestamp: datetime = Field(default_factory=datetime.utcnow)


class SafetyRequirement(BaseModel):
    requirement_id: str = Field(..., description="OSHA requirement identifier")
    category: str = Field(..., description="Safety category (e.g., PPE, Training, Equipment)")
    description: str = Field(..., description="Requirement description")
    is_met: bool = Field(..., description="Whether requirement is currently met")
    last_inspection_date: Optional[datetime] = Field(None, description="Last inspection date")
    next_inspection_due: Optional[datetime] = Field(None, description="Next inspection due date")


class PermitRequirement(BaseModel):
    permit_id: str = Field(..., description="Permit identifier")
    permit_type: str = Field(..., description="Permit type (e.g., Air, Water, Waste)")
    issuing_agency: RegulatoryAgency = Field(..., description="Issuing regulatory agency")
    issue_date: datetime = Field(..., description="Permit issue date")
    expiration_date: datetime = Field(..., description="Permit expiration date")
    reporting_frequency: ReportingFrequency = Field(..., description="Required reporting frequency")
    last_report_date: Optional[datetime] = Field(None, description="Last report submission date")
    is_current: bool = Field(default=True, description="Whether permit is current")


class TrainingRecord(BaseModel):
    employee_id: str = Field(..., description="Employee identifier")
    training_topic: str = Field(..., description="Training topic/course name")
    completion_date: datetime = Field(..., description="Training completion date")
    expiration_date: Optional[datetime] = Field(None, description="Training expiration date")
    certification_number: Optional[str] = Field(None, description="Certification number if applicable")


class RegulatoryGuardianInput(BaseModel):
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(default="Industrial Facility", description="Facility name")
    emission_limits: List[EmissionLimit] = Field(default_factory=list, description="Emission limits and current values")
    safety_requirements: List[SafetyRequirement] = Field(default_factory=list, description="OSHA safety requirements")
    permits: List[PermitRequirement] = Field(default_factory=list, description="Regulatory permits")
    training_records: List[TrainingRecord] = Field(default_factory=list, description="Employee training records")
    inspection_date: datetime = Field(default_factory=datetime.utcnow, description="Inspection/analysis date")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class ComplianceViolation(BaseModel):
    violation_id: str
    agency: RegulatoryAgency
    regulation_reference: str
    severity: ViolationSeverity
    description: str
    affected_area: str
    detection_date: datetime
    corrective_action_required: str
    deadline: Optional[datetime] = None


class ComplianceRecommendation(BaseModel):
    recommendation_id: str
    priority: str
    category: str
    description: str
    estimated_cost_usd: Optional[float] = None
    implementation_timeframe_days: Optional[int] = None
    regulatory_benefit: str


class ComplianceWarning(BaseModel):
    warning_id: str
    warning_type: str
    description: str
    days_until_deadline: Optional[int] = None
    action_required: str


class EmissionComplianceReport(BaseModel):
    pollutant: str
    limit_value: float
    current_value: float
    unit: str
    compliance_margin_percent: float
    status: ComplianceStatus
    trend: str


class PermitStatusReport(BaseModel):
    permit_id: str
    permit_type: str
    agency: str
    days_until_expiration: int
    reporting_overdue: bool
    days_overdue: int
    status: ComplianceStatus


class ProvenanceRecord(BaseModel):
    operation: str
    timestamp: datetime
    input_hash: str
    output_hash: str
    tool_name: str


class RegulatoryGuardianOutput(BaseModel):
    analysis_id: str
    facility_id: str
    facility_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Compliance Scores
    overall_compliance_score: float = Field(..., ge=0, le=100, description="Overall compliance score 0-100")
    epa_compliance_score: float = Field(..., ge=0, le=100)
    osha_compliance_score: float = Field(..., ge=0, le=100)

    # Violations
    violations: List[ComplianceViolation] = Field(default_factory=list)
    critical_violations_count: int
    major_violations_count: int
    minor_violations_count: int

    # Recommendations and Warnings
    recommendations: List[ComplianceRecommendation] = Field(default_factory=list)
    warnings: List[ComplianceWarning] = Field(default_factory=list)

    # Detailed Reports
    emission_compliance: List[EmissionComplianceReport] = Field(default_factory=list)
    permit_status: List[PermitStatusReport] = Field(default_factory=list)

    # Training Compliance
    training_compliance_percent: float = Field(..., ge=0, le=100)
    employees_with_expired_training: int

    # Risk Assessment
    regulatory_risk_level: str
    estimated_penalty_exposure_usd: float

    # Provenance
    provenance_chain: List[ProvenanceRecord] = Field(default_factory=list)
    provenance_hash: str

    # Processing Metadata
    processing_time_ms: float
    validation_status: str
    validation_errors: List[str] = Field(default_factory=list)


# =============================================================================
# REGULATORY GUARDIAN AGENT
# =============================================================================

class RegulatoryGuardianAgent:
    """GL-071: Regulatory Guardian Agent - EPA/OSHA Compliance Monitoring."""

    AGENT_ID = "GL-071"
    AGENT_NAME = "REGULATORY-GUARDIAN"
    VERSION = "1.0.0"

    # Thresholds
    EMISSION_WARNING_THRESHOLD_PERCENT = 80.0  # Warn if emissions > 80% of limit
    PERMIT_RENEWAL_WARNING_DAYS = 90  # Warn if permit expires in < 90 days
    TRAINING_WARNING_DAYS = 30  # Warn if training expires in < 30 days

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []
        logger.info(f"RegulatoryGuardianAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: RegulatoryGuardianInput) -> RegulatoryGuardianOutput:
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []

        logger.info(f"Starting regulatory compliance analysis for {input_data.facility_id}")

        # Step 1: Analyze emission compliance
        emission_reports, emission_violations = self._analyze_emissions(input_data.emission_limits)
        self._track_provenance(
            "emission_compliance_check",
            {"pollutants_count": len(input_data.emission_limits)},
            {"violations": len(emission_violations)},
            "emission_analyzer"
        )

        # Step 2: Analyze safety requirements
        safety_violations = self._analyze_safety_requirements(input_data.safety_requirements)
        self._track_provenance(
            "safety_compliance_check",
            {"requirements_count": len(input_data.safety_requirements)},
            {"violations": len(safety_violations)},
            "safety_analyzer"
        )

        # Step 3: Analyze permits
        permit_reports, permit_violations, permit_warnings = self._analyze_permits(
            input_data.permits, input_data.inspection_date
        )
        self._track_provenance(
            "permit_compliance_check",
            {"permits_count": len(input_data.permits)},
            {"violations": len(permit_violations), "warnings": len(permit_warnings)},
            "permit_analyzer"
        )

        # Step 4: Analyze training records
        training_compliance, training_warnings, expired_count = self._analyze_training(
            input_data.training_records, input_data.inspection_date
        )
        self._track_provenance(
            "training_compliance_check",
            {"records_count": len(input_data.training_records)},
            {"compliance_percent": training_compliance, "expired": expired_count},
            "training_analyzer"
        )

        # Step 5: Combine all violations
        all_violations = emission_violations + safety_violations + permit_violations

        # Count violations by severity
        critical_count = sum(1 for v in all_violations if v.severity == ViolationSeverity.CRITICAL)
        major_count = sum(1 for v in all_violations if v.severity == ViolationSeverity.MAJOR)
        minor_count = sum(1 for v in all_violations if v.severity == ViolationSeverity.MINOR)

        # Step 6: Calculate compliance scores
        epa_violations = [v for v in all_violations if v.agency in [RegulatoryAgency.EPA, RegulatoryAgency.STATE_EPA]]
        osha_violations = [v for v in all_violations if v.agency == RegulatoryAgency.OSHA]

        epa_score = self._calculate_compliance_score(epa_violations, len(input_data.emission_limits) + len(input_data.permits))
        osha_score = self._calculate_compliance_score(osha_violations, len(input_data.safety_requirements))
        overall_score = (epa_score + osha_score) / 2.0

        # Step 7: Generate recommendations
        recommendations = self._generate_recommendations(all_violations, emission_reports, permit_reports)

        # Step 8: Assess risk and penalty exposure
        risk_level = self._assess_risk_level(critical_count, major_count, minor_count)
        penalty_exposure = self._estimate_penalty_exposure(all_violations)

        # Combine all warnings
        all_warnings = permit_warnings + training_warnings

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return RegulatoryGuardianOutput(
            analysis_id=f"REG-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            facility_id=input_data.facility_id,
            facility_name=input_data.facility_name,
            overall_compliance_score=round(overall_score, 2),
            epa_compliance_score=round(epa_score, 2),
            osha_compliance_score=round(osha_score, 2),
            violations=all_violations,
            critical_violations_count=critical_count,
            major_violations_count=major_count,
            minor_violations_count=minor_count,
            recommendations=recommendations,
            warnings=all_warnings,
            emission_compliance=emission_reports,
            permit_status=permit_reports,
            training_compliance_percent=round(training_compliance, 2),
            employees_with_expired_training=expired_count,
            regulatory_risk_level=risk_level,
            estimated_penalty_exposure_usd=round(penalty_exposure, 2),
            provenance_chain=[ProvenanceRecord(**s) for s in self._provenance_steps],
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS" if not self._validation_errors else "FAIL",
            validation_errors=self._validation_errors
        )

    def _analyze_emissions(self, limits: List[EmissionLimit]) -> tuple:
        reports = []
        violations = []

        for limit in limits:
            compliance_margin = ((limit.limit_value - limit.current_value) / limit.limit_value * 100.0) if limit.limit_value > 0 else 0.0

            if limit.current_value > limit.limit_value:
                status = ComplianceStatus.NON_COMPLIANT
                violations.append(ComplianceViolation(
                    violation_id=f"EPA-EM-{limit.pollutant}",
                    agency=RegulatoryAgency.EPA,
                    regulation_reference="40 CFR Part 60",
                    severity=ViolationSeverity.CRITICAL,
                    description=f"{limit.pollutant} emissions exceed limit: {limit.current_value} {limit.limit_unit} > {limit.limit_value} {limit.limit_unit}",
                    affected_area=limit.pollutant,
                    detection_date=limit.measurement_timestamp,
                    corrective_action_required="Reduce emissions immediately through operational adjustments or emission controls",
                    deadline=datetime.utcnow() + timedelta(days=30)
                ))
                trend = "EXCEEDED"
            elif limit.current_value / limit.limit_value >= (self.EMISSION_WARNING_THRESHOLD_PERCENT / 100.0):
                status = ComplianceStatus.AT_RISK
                trend = "APPROACHING_LIMIT"
            else:
                status = ComplianceStatus.COMPLIANT
                trend = "NORMAL"

            reports.append(EmissionComplianceReport(
                pollutant=limit.pollutant,
                limit_value=limit.limit_value,
                current_value=limit.current_value,
                unit=limit.limit_unit,
                compliance_margin_percent=round(compliance_margin, 2),
                status=status,
                trend=trend
            ))

        return reports, violations

    def _analyze_safety_requirements(self, requirements: List[SafetyRequirement]) -> List[ComplianceViolation]:
        violations = []

        for req in requirements:
            if not req.is_met:
                violations.append(ComplianceViolation(
                    violation_id=f"OSHA-{req.requirement_id}",
                    agency=RegulatoryAgency.OSHA,
                    regulation_reference="29 CFR 1910",
                    severity=ViolationSeverity.MAJOR,
                    description=f"OSHA requirement not met: {req.description}",
                    affected_area=req.category,
                    detection_date=datetime.utcnow(),
                    corrective_action_required=f"Implement required {req.category} measures",
                    deadline=datetime.utcnow() + timedelta(days=60)
                ))

        return violations

    def _analyze_permits(self, permits: List[PermitRequirement], inspection_date: datetime) -> tuple:
        reports = []
        violations = []
        warnings = []

        for permit in permits:
            days_until_expiration = (permit.expiration_date - inspection_date).days

            # Check if permit is expired
            if days_until_expiration < 0:
                violations.append(ComplianceViolation(
                    violation_id=f"PERMIT-EXP-{permit.permit_id}",
                    agency=permit.issuing_agency,
                    regulation_reference=f"{permit.permit_type} Permit Requirements",
                    severity=ViolationSeverity.CRITICAL,
                    description=f"Permit {permit.permit_id} expired {abs(days_until_expiration)} days ago",
                    affected_area=permit.permit_type,
                    detection_date=inspection_date,
                    corrective_action_required="Cease operations or obtain emergency permit extension immediately",
                    deadline=inspection_date + timedelta(days=7)
                ))
                status = ComplianceStatus.NON_COMPLIANT
            elif days_until_expiration < self.PERMIT_RENEWAL_WARNING_DAYS:
                warnings.append(ComplianceWarning(
                    warning_id=f"WARN-PERMIT-{permit.permit_id}",
                    warning_type="PERMIT_EXPIRING",
                    description=f"Permit {permit.permit_id} expires in {days_until_expiration} days",
                    days_until_deadline=days_until_expiration,
                    action_required="Begin permit renewal process immediately"
                ))
                status = ComplianceStatus.AT_RISK
            else:
                status = ComplianceStatus.COMPLIANT

            # Check reporting compliance
            reporting_overdue = False
            days_overdue = 0
            if permit.last_report_date:
                if permit.reporting_frequency == ReportingFrequency.MONTHLY:
                    next_due = permit.last_report_date + timedelta(days=30)
                elif permit.reporting_frequency == ReportingFrequency.QUARTERLY:
                    next_due = permit.last_report_date + timedelta(days=90)
                elif permit.reporting_frequency == ReportingFrequency.ANNUALLY:
                    next_due = permit.last_report_date + timedelta(days=365)
                else:
                    next_due = permit.last_report_date + timedelta(days=7)

                if inspection_date > next_due:
                    reporting_overdue = True
                    days_overdue = (inspection_date - next_due).days
                    violations.append(ComplianceViolation(
                        violation_id=f"REPORT-{permit.permit_id}",
                        agency=permit.issuing_agency,
                        regulation_reference=f"{permit.permit_type} Reporting Requirements",
                        severity=ViolationSeverity.MAJOR,
                        description=f"Required report for permit {permit.permit_id} is {days_overdue} days overdue",
                        affected_area=permit.permit_type,
                        detection_date=inspection_date,
                        corrective_action_required="Submit overdue report immediately",
                        deadline=inspection_date + timedelta(days=14)
                    ))

            reports.append(PermitStatusReport(
                permit_id=permit.permit_id,
                permit_type=permit.permit_type,
                agency=permit.issuing_agency.value,
                days_until_expiration=days_until_expiration,
                reporting_overdue=reporting_overdue,
                days_overdue=days_overdue,
                status=status
            ))

        return reports, violations, warnings

    def _analyze_training(self, records: List[TrainingRecord], inspection_date: datetime) -> tuple:
        if not records:
            return 0.0, [], 0

        total_records = len(records)
        expired_count = 0
        warnings = []

        for record in records:
            if record.expiration_date:
                days_until_expiration = (record.expiration_date - inspection_date).days

                if days_until_expiration < 0:
                    expired_count += 1
                elif days_until_expiration < self.TRAINING_WARNING_DAYS:
                    warnings.append(ComplianceWarning(
                        warning_id=f"WARN-TRAIN-{record.employee_id}-{record.training_topic[:20]}",
                        warning_type="TRAINING_EXPIRING",
                        description=f"Training '{record.training_topic}' for employee {record.employee_id} expires in {days_until_expiration} days",
                        days_until_deadline=days_until_expiration,
                        action_required="Schedule refresher training"
                    ))

        compliance_percent = ((total_records - expired_count) / total_records * 100.0) if total_records > 0 else 100.0

        return compliance_percent, warnings, expired_count

    def _calculate_compliance_score(self, violations: List[ComplianceViolation], total_items: int) -> float:
        if total_items == 0:
            return 100.0

        # Weight violations by severity
        penalty_points = sum([
            10.0 if v.severity == ViolationSeverity.CRITICAL else
            5.0 if v.severity == ViolationSeverity.MAJOR else
            2.0 if v.severity == ViolationSeverity.MINOR else 0.5
            for v in violations
        ])

        # Calculate score (penalize based on violations)
        max_score = 100.0
        score = max(0.0, max_score - (penalty_points / total_items * 20.0))

        return score

    def _generate_recommendations(
        self,
        violations: List[ComplianceViolation],
        emission_reports: List[EmissionComplianceReport],
        permit_reports: List[PermitStatusReport]
    ) -> List[ComplianceRecommendation]:
        recommendations = []
        rec_id = 0

        # Recommendations for emissions approaching limits
        for report in emission_reports:
            if report.status == ComplianceStatus.AT_RISK:
                rec_id += 1
                recommendations.append(ComplianceRecommendation(
                    recommendation_id=f"REC-{rec_id:03d}",
                    priority="HIGH",
                    category="EMISSION_CONTROL",
                    description=f"Implement additional controls for {report.pollutant} - currently at {report.compliance_margin_percent}% margin",
                    estimated_cost_usd=50000.0,
                    implementation_timeframe_days=60,
                    regulatory_benefit="Prevent future emission limit violations and potential fines"
                ))

        # Recommendations for expiring permits
        for report in permit_reports:
            if report.days_until_expiration < 180 and report.days_until_expiration > 0:
                rec_id += 1
                recommendations.append(ComplianceRecommendation(
                    recommendation_id=f"REC-{rec_id:03d}",
                    priority="MEDIUM",
                    category="PERMIT_RENEWAL",
                    description=f"Begin renewal process for {report.permit_type} permit {report.permit_id}",
                    estimated_cost_usd=15000.0,
                    implementation_timeframe_days=90,
                    regulatory_benefit="Ensure continuous regulatory authorization"
                ))

        # General compliance improvement recommendations
        if len(violations) > 5:
            rec_id += 1
            recommendations.append(ComplianceRecommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                priority="HIGH",
                category="COMPLIANCE_MANAGEMENT",
                description="Implement comprehensive environmental management system (EMS) to improve compliance tracking",
                estimated_cost_usd=100000.0,
                implementation_timeframe_days=180,
                regulatory_benefit="Systematic approach to preventing violations and maintaining compliance"
            ))

        return recommendations

    def _assess_risk_level(self, critical: int, major: int, minor: int) -> str:
        if critical > 0:
            return "CRITICAL"
        elif major >= 3:
            return "HIGH"
        elif major > 0 or minor >= 5:
            return "MEDIUM"
        elif minor > 0:
            return "LOW"
        else:
            return "MINIMAL"

    def _estimate_penalty_exposure(self, violations: List[ComplianceViolation]) -> float:
        # Estimate potential penalties based on typical EPA/OSHA fines
        total_exposure = 0.0

        for violation in violations:
            if violation.severity == ViolationSeverity.CRITICAL:
                # Critical EPA violations can be $25,000-$50,000 per day
                total_exposure += 37500.0
            elif violation.severity == ViolationSeverity.MAJOR:
                # Major violations typically $10,000-$25,000
                total_exposure += 17500.0
            elif violation.severity == ViolationSeverity.MINOR:
                # Minor violations typically $1,000-$10,000
                total_exposure += 5000.0

        return total_exposure

    def _track_provenance(self, operation: str, inputs: Dict, outputs: Dict, tool_name: str) -> None:
        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
            "output_hash": hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest(),
            "tool_name": tool_name
        })

    def _calculate_provenance_hash(self) -> str:
        data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": [
                {
                    "operation": s["operation"],
                    "input_hash": s["input_hash"],
                    "output_hash": s["output_hash"]
                }
                for s in self._provenance_steps
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-071",
    "name": "REGULATORY-GUARDIAN",
    "version": "1.0.0",
    "summary": "EPA/OSHA regulatory compliance monitoring and violation tracking",
    "tags": ["regulatory", "compliance", "EPA", "OSHA", "environmental", "safety"],
    "standards": [
        {"ref": "EPA Clean Air Act", "description": "Air quality regulations"},
        {"ref": "EPA Clean Water Act", "description": "Water quality regulations"},
        {"ref": "OSHA 29 CFR 1910", "description": "Occupational safety standards"},
        {"ref": "EPA 40 CFR", "description": "Environmental protection regulations"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True
    }
}
