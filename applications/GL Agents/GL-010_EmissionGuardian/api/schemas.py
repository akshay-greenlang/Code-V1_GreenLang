# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - API Request/Response Schemas

Pydantic models for all REST API endpoints including:
- CEMS data models
- Compliance models
- RATA models
- Fugitive detection models
- Trading models
- Reporting models
- Pagination and filtering models
- Error response models

Standards Compliance:
    - EPA 40 CFR Part 75: Continuous Emissions Monitoring
    - EPA 40 CFR Part 60: NSPS
    - EPA 40 CFR Part 63: NESHAP
    - OpenAPI 3.1.0 compatible

Zero-Hallucination Principle:
    - All response models include provenance_hash for traceability
    - Timestamps are ISO 8601 compliant
    - All numeric values use appropriate precision
"""

from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import hashlib
import uuid


# ============================================================================
# Generic Type Variables
# ============================================================================

T = TypeVar("T")


# ============================================================================
# Enumerations
# ============================================================================

class Pollutant(str, Enum):
    """Monitored pollutants per EPA 40 CFR Part 75."""
    NOX = "nox"
    SO2 = "so2"
    CO2 = "co2"
    CO = "co"
    PM = "pm"
    PM10 = "pm10"
    PM25 = "pm25"
    VOC = "voc"
    NH3 = "nh3"
    HCL = "hcl"
    HG = "hg"
    O2 = "o2"
    FLOW = "flow"
    OPACITY = "opacity"
    MOISTURE = "moisture"
    CH4 = "ch4"


class DataQuality(str, Enum):
    """CEMS data quality indicators per 40 CFR Part 75."""
    VALID = "valid"
    SUBSTITUTED = "substituted"
    MISSING = "missing"
    MAINTENANCE = "maintenance"
    CALIBRATION = "calibration"
    QA_FAILED = "qa_failed"
    OUT_OF_RANGE = "out_of_range"
    SUSPECT = "suspect"


class AveragingPeriod(str, Enum):
    """Compliance averaging periods."""
    HOURLY = "hourly"
    ROLLING_3HR = "rolling_3hr"
    ROLLING_24HR = "rolling_24hr"
    DAILY = "daily"
    BLOCK_30DAY = "block_30day"
    ROLLING_30DAY = "rolling_30day"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    OZONE_SEASON = "ozone_season"


class OperatingState(str, Enum):
    """Unit operating states for emissions monitoring."""
    OPERATING = "operating"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    STANDBY = "standby"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


class ComplianceStatusType(str, Enum):
    """Compliance status types."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    NON_COMPLIANT = "non_compliant"
    PENDING = "pending"
    UNKNOWN = "unknown"


class ExceedanceSeverity(str, Enum):
    """Exceedance severity levels."""
    INFORMATIONAL = "informational"
    WARNING = "warning"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class CorrectiveActionState(str, Enum):
    """Corrective action workflow states."""
    IDENTIFIED = "identified"
    ACKNOWLEDGED = "acknowledged"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    CORRECTIVE_ACTION_PLANNED = "corrective_action_planned"
    CORRECTIVE_ACTION_IN_PROGRESS = "corrective_action_in_progress"
    VERIFICATION = "verification"
    CLOSED = "closed"
    REOPENED = "reopened"


class RATATestType(str, Enum):
    """RATA test types per 40 CFR Part 75."""
    STANDARD = "standard"
    ABBREVIATED = "abbreviated"
    SINGLE_LOAD = "single_load"
    THREE_LOAD = "three_load"
    CYLINDER_GAS = "cylinder_gas"


class RATAStatus(str, Enum):
    """RATA test status."""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PASSED = "passed"
    FAILED = "failed"
    CONDITIONAL = "conditional"
    CANCELLED = "cancelled"


class AlertSeverity(str, Enum):
    """Fugitive alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Fugitive alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    CONFIRMED = "confirmed"
    FALSE_POSITIVE = "false_positive"
    RESOLVED = "resolved"


class SensorStatus(str, Enum):
    """Sensor health status."""
    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    CALIBRATING = "calibrating"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class ReviewDecision(str, Enum):
    """Fugitive review decision."""
    CONFIRMED_LEAK = "confirmed_leak"
    FALSE_POSITIVE = "false_positive"
    NEEDS_INVESTIGATION = "needs_investigation"
    DEFERRED = "deferred"


class TradeType(str, Enum):
    """Carbon trade types."""
    BUY = "buy"
    SELL = "sell"
    TRANSFER = "transfer"
    RETIRE = "retire"


class TradeStatus(str, Enum):
    """Trade status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    SETTLED = "settled"


class CarbonMarket(str, Enum):
    """Carbon market types."""
    EU_ETS = "eu_ets"
    CA_CAT = "ca_cat"
    RGGI = "rggi"
    WCI = "wci"
    CORSIA = "corsia"
    VOLUNTARY = "voluntary"
    INTERNAL = "internal"


class OffsetStandard(str, Enum):
    """Carbon offset verification standards."""
    VERRA_VCS = "verra_vcs"
    GOLD_STANDARD = "gold_standard"
    ACR = "acr"
    CAR = "car"
    CDM = "cdm"
    CORSIA_ELIGIBLE = "corsia_eligible"
    INTERNAL = "internal"


class ReportType(str, Enum):
    """Report types."""
    QUARTERLY_EMISSIONS = "quarterly_emissions"
    ANNUAL_INVENTORY = "annual_inventory"
    DEVIATION = "deviation"
    EXCESS_EMISSIONS = "excess_emissions"
    AUDIT_TRAIL = "audit_trail"
    RATA_SUMMARY = "rata_summary"
    LDAR_SUMMARY = "ldar_summary"


class SortOrder(str, Enum):
    """Sort order."""
    ASC = "asc"
    DESC = "desc"


# ============================================================================
# Base Models
# ============================================================================

class ProvenanceBase(BaseModel):
    """Base model with provenance tracking."""
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for data integrity verification"
    )

    def calculate_provenance_hash(self, content: str) -> str:
        """Calculate SHA-256 provenance hash."""
        return hashlib.sha256(content.encode()).hexdigest()


class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Record creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Record last update timestamp"
    )


# ============================================================================
# Pagination Models
# ============================================================================

class PaginationParams(BaseModel):
    """Pagination query parameters."""
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(50, ge=1, le=1000, description="Items per page")

    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.page_size


class PaginationMeta(BaseModel):
    """Pagination metadata for responses."""
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")

    @classmethod
    def from_pagination(cls, page: int, page_size: int, total_items: int) -> "PaginationMeta":
        """Create pagination meta from query parameters."""
        total_pages = (total_items + page_size - 1) // page_size if page_size > 0 else 0
        return cls(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1
        )


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""
    data: List[T] = Field(..., description="List of items")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")
    provenance_hash: Optional[str] = Field(None, description="Response data hash")


# ============================================================================
# Filter Models
# ============================================================================

class DateTimeRange(BaseModel):
    """Date/time range filter."""
    start: datetime = Field(..., description="Start of date/time range")
    end: datetime = Field(..., description="End of date/time range")

    @model_validator(mode="after")
    def validate_range(self) -> "DateTimeRange":
        if self.end < self.start:
            raise ValueError("End datetime must be after start datetime")
        return self


class DateRange(BaseModel):
    """Date range filter."""
    start: date = Field(..., description="Start date")
    end: date = Field(..., description="End date")

    @model_validator(mode="after")
    def validate_range(self) -> "DateRange":
        if self.end < self.start:
            raise ValueError("End date must be after start date")
        return self


class NumericRange(BaseModel):
    """Numeric range filter."""
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")

    @model_validator(mode="after")
    def validate_range(self) -> "NumericRange":
        if self.min_value is not None and self.max_value is not None:
            if self.max_value < self.min_value:
                raise ValueError("max_value must be >= min_value")
        return self


# ============================================================================
# Error Response Models
# ============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information."""
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed error information")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier for tracing"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Request validation failed",
                "details": [
                    {"field": "start_time", "message": "Start time is required", "code": "required"}
                ],
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2025-11-09T10:30:00Z"
            }
        }



# CEMS Models
class CEMSReadingBase(BaseModel):
    facility_id: str = Field(...)
    unit_id: str = Field(...)
    stack_id: str = Field(...)
    timestamp: datetime = Field(...)

class CEMSReadingResponse(CEMSReadingBase, ProvenanceBase):
    reading_id: str = Field(...)
    pollutant: Pollutant = Field(...)
    value: float = Field(...)
    units: str = Field(...)
    data_quality: DataQuality = Field(...)
    operating_state: OperatingState = Field(...)
    is_substituted: bool = Field(False)

class CEMSReadingCreate(BaseModel):
    facility_id: str = Field(...)
    unit_id: str = Field(...)
    stack_id: str = Field(...)
    pollutant: Pollutant = Field(...)
    timestamp: datetime = Field(...)
    value: float = Field(...)
    units: str = Field(...)
    reason: str = Field(..., min_length=10)

class CEMSReadingFilter(BaseModel):
    facility_id: Optional[str] = None
    unit_id: Optional[str] = None
    start_time: datetime = Field(...)
    end_time: datetime = Field(...)
    include_substituted: bool = True

class HourlyAverageResponse(CEMSReadingBase, ProvenanceBase):
    hour: datetime = Field(...)
    pollutant: Pollutant = Field(...)
    average_value: float = Field(...)
    data_availability_percent: float = Field(..., ge=0, le=100)

class DailySummaryResponse(ProvenanceBase):
    facility_id: str = Field(...)
    unit_id: str = Field(...)
    date: date = Field(...)
    pollutant: Pollutant = Field(...)
    daily_average: float = Field(...)
    compliance_status: ComplianceStatusType = Field(...)

class DataAvailabilityResponse(ProvenanceBase):
    facility_id: str = Field(...)
    unit_id: str = Field(...)
    monitor_id: str = Field(...)
    pollutant: Pollutant = Field(...)
    availability_percent: float = Field(..., ge=0, le=100)
    meets_minimum: bool = Field(...)



# Compliance Models
class PermitLimitResponse(ProvenanceBase):
    limit_id: str = Field(...)
    permit_id: str = Field(...)
    facility_id: str = Field(...)
    pollutant: Pollutant = Field(...)
    limit_value: float = Field(...)
    units: str = Field(...)
    averaging_period: AveragingPeriod = Field(...)
    is_active: bool = Field(...)

class ComplianceStatusResponse(ProvenanceBase):
    facility_id: str = Field(...)
    unit_id: Optional[str] = None
    status_timestamp: datetime = Field(...)
    overall_status: ComplianceStatusType = Field(...)
    compliance_score: float = Field(..., ge=0, le=100)
    active_exceedances: int = Field(0)
    active_warnings: int = Field(0)
    days_in_compliance: int = Field(0)

class ExceedanceResponse(ProvenanceBase, TimestampMixin):
    exceedance_id: str = Field(...)
    facility_id: str = Field(...)
    unit_id: str = Field(...)
    pollutant: Pollutant = Field(...)
    limit_value: float = Field(...)
    measured_value: float = Field(...)
    severity: ExceedanceSeverity = Field(...)
    start_time: datetime = Field(...)
    is_resolved: bool = Field(False)

class ExceedanceFilter(BaseModel):
    facility_id: Optional[str] = None
    unit_id: Optional[str] = None
    severity: Optional[List[ExceedanceSeverity]] = None
    is_resolved: Optional[bool] = None

class ComplianceScheduleResponse(ProvenanceBase):
    schedule_id: str = Field(...)
    facility_id: str = Field(...)
    report_type: ReportType = Field(...)
    submission_deadline: datetime = Field(...)
    days_until_deadline: int = Field(...)
    is_overdue: bool = Field(False)

class CorrectiveActionCreate(BaseModel):
    exceedance_id: str = Field(...)
    title: str = Field(..., min_length=5)
    description: str = Field(..., min_length=20)
    action_type: str = Field(...)

class CorrectiveActionResponse(ProvenanceBase, TimestampMixin):
    action_id: str = Field(...)
    exceedance_id: str = Field(...)
    title: str = Field(...)
    state: CorrectiveActionState = Field(...)
    is_overdue: bool = Field(False)

class CorrectiveActionUpdate(BaseModel):
    state: Optional[CorrectiveActionState] = None
    assigned_to: Optional[str] = None
    verification_notes: Optional[str] = None



# RATA Models
class RATAScheduleResponse(ProvenanceBase):
    schedule_id: str = Field(...)
    facility_id: str = Field(...)
    unit_id: str = Field(...)
    monitor_id: str = Field(...)
    pollutant: Pollutant = Field(...)
    test_type: RATATestType = Field(...)
    scheduled_date: date = Field(...)
    deadline_date: date = Field(...)
    days_until_deadline: int = Field(...)
    status: RATAStatus = Field(...)

class RATARunData(BaseModel):
    run_number: int = Field(..., ge=1)
    start_time: datetime = Field(...)
    end_time: datetime = Field(...)
    duration_minutes: int = Field(..., ge=21)
    reference_value: float = Field(...)
    cems_value: float = Field(...)
    units: str = Field(...)
    is_valid: bool = Field(True)

class RATATestCreate(BaseModel):
    facility_id: str = Field(...)
    unit_id: str = Field(...)
    monitor_id: str = Field(...)
    pollutant: Pollutant = Field(...)
    test_type: RATATestType = Field(...)
    test_date: date = Field(...)
    contractor: str = Field(...)
    reference_method: str = Field(...)
    runs: List[RATARunData] = Field(..., min_length=3)

class RATAResultResponse(ProvenanceBase, TimestampMixin):
    test_id: str = Field(...)
    facility_id: str = Field(...)
    unit_id: str = Field(...)
    pollutant: Pollutant = Field(...)
    test_date: date = Field(...)
    status: RATAStatus = Field(...)
    relative_accuracy: float = Field(...)
    ra_limit: float = Field(...)
    passed: bool = Field(...)
    next_rata_due: date = Field(...)

class RATAReportResponse(ProvenanceBase):
    test_id: str = Field(...)
    report_generated_at: datetime = Field(...)
    report_url: str = Field(...)
    summary: Dict[str, Any] = Field(...)



# Fugitive Detection Models
class FugitiveAlertResponse(ProvenanceBase, TimestampMixin):
    alert_id: str = Field(...)
    facility_id: str = Field(...)
    location_id: str = Field(...)
    location_name: str = Field(...)
    source_type: str = Field(...)
    detection_time: datetime = Field(...)
    severity: AlertSeverity = Field(...)
    status: AlertStatus = Field(...)
    concentration_ppm: float = Field(...)
    confidence_score: float = Field(..., ge=0, le=100)
    explanation: Optional[str] = None

class FugitiveAlertFilter(BaseModel):
    facility_id: Optional[str] = None
    severity: Optional[List[AlertSeverity]] = None
    status: Optional[List[AlertStatus]] = None
    min_confidence: Optional[float] = Field(None, ge=0, le=100)

class SensorStatusResponse(ProvenanceBase):
    sensor_id: str = Field(...)
    facility_id: str = Field(...)
    location_id: str = Field(...)
    sensor_type: str = Field(...)
    status: SensorStatus = Field(...)
    last_reading_time: Optional[datetime] = None
    last_reading_value: Optional[float] = None
    battery_level_percent: Optional[float] = Field(None, ge=0, le=100)

class ReviewDecisionCreate(BaseModel):
    alert_id: str = Field(...)
    decision: ReviewDecision = Field(...)
    reviewer_notes: str = Field(..., min_length=10)
    repair_required: bool = Field(False)

class ReviewDecisionResponse(ProvenanceBase, TimestampMixin):
    review_id: str = Field(...)
    alert_id: str = Field(...)
    decision: ReviewDecision = Field(...)
    reviewer_id: str = Field(...)
    reviewer_notes: str = Field(...)
    repair_required: bool = Field(False)

class LDARStatusResponse(ProvenanceBase):
    facility_id: str = Field(...)
    program_id: str = Field(...)
    overall_status: ComplianceStatusType = Field(...)
    total_components: int = Field(...)
    active_leaks: int = Field(...)
    leak_rate_percent: float = Field(..., ge=0, le=100)



# Trading Models
class TradingPositionResponse(ProvenanceBase):
    position_id: str = Field(...)
    facility_id: str = Field(...)
    market: CarbonMarket = Field(...)
    position_type: str = Field(...)
    instrument: str = Field(...)
    quantity_mtco2e: float = Field(...)
    average_cost_usd: float = Field(...)
    current_price_usd: float = Field(...)
    market_value_usd: float = Field(...)
    unrealized_pnl_usd: float = Field(...)

class TradingRecommendationResponse(ProvenanceBase, TimestampMixin):
    recommendation_id: str = Field(...)
    facility_id: str = Field(...)
    recommendation_type: TradeType = Field(...)
    market: CarbonMarket = Field(...)
    quantity_mtco2e: float = Field(...)
    confidence_score: float = Field(..., ge=0, le=100)
    rationale: str = Field(...)
    status: str = Field(...)

class TradeApprovalCreate(BaseModel):
    recommendation_id: str = Field(...)
    approved: bool = Field(...)
    approver_notes: str = Field(..., min_length=10)

class TradeApprovalResponse(ProvenanceBase, TimestampMixin):
    approval_id: str = Field(...)
    recommendation_id: str = Field(...)
    approved: bool = Field(...)
    approver_id: str = Field(...)
    execution_status: str = Field(...)

class OffsetCertificateResponse(ProvenanceBase, TimestampMixin):
    certificate_id: str = Field(...)
    registry_id: str = Field(...)
    facility_id: str = Field(...)
    standard: OffsetStandard = Field(...)
    project_name: str = Field(...)
    vintage_year: int = Field(...)
    quantity_mtco2e: float = Field(...)
    status: str = Field(...)



# Reporting Models
class QuarterlyReportRequest(BaseModel):
    facility_id: str = Field(...)
    year: int = Field(..., ge=2000, le=2100)
    quarter: int = Field(..., ge=1, le=4)
    include_details: bool = Field(True)
    format: str = Field("json")

class QuarterlyReportResponse(ProvenanceBase):
    report_id: str = Field(...)
    facility_id: str = Field(...)
    year: int = Field(...)
    quarter: int = Field(...)
    generated_at: datetime = Field(...)
    status: str = Field(...)
    emissions_summary: Dict[str, Any] = Field(...)
    compliance_summary: Dict[str, Any] = Field(...)
    total_emissions: Dict[str, float] = Field(...)

class AnnualReportResponse(ProvenanceBase):
    report_id: str = Field(...)
    facility_id: str = Field(...)
    year: int = Field(...)
    generated_at: datetime = Field(...)
    total_ghg_emissions_mtco2e: float = Field(...)
    emissions_by_pollutant: Dict[str, float] = Field(...)
    net_emissions_mtco2e: float = Field(...)

class DeviationReportResponse(ProvenanceBase, TimestampMixin):
    report_id: str = Field(...)
    facility_id: str = Field(...)
    exceedance_id: str = Field(...)
    pollutant: Pollutant = Field(...)
    duration_hours: float = Field(...)
    excess_emissions: float = Field(...)
    cause: str = Field(...)
    status: str = Field(...)

class AuditTrailFilter(BaseModel):
    facility_id: Optional[str] = None
    entity_type: Optional[str] = None
    action: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class AuditTrailResponse(ProvenanceBase):
    audit_id: str = Field(...)
    timestamp: datetime = Field(...)
    facility_id: str = Field(...)
    entity_type: str = Field(...)
    entity_id: str = Field(...)
    action: str = Field(...)
    user_id: str = Field(...)
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None



# Authentication Models
class TokenResponse(BaseModel):
    access_token: str = Field(...)
    token_type: str = Field("bearer")
    expires_in: int = Field(...)
    refresh_token: Optional[str] = None
    scope: str = Field(...)

class UserInfo(BaseModel):
    user_id: str = Field(...)
    email: str = Field(...)
    name: str = Field(...)
    roles: List[str] = Field(default_factory=list)
    facilities: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    is_active: bool = Field(True)


# Health Check Models
class HealthCheckResponse(BaseModel):
    status: str = Field(...)
    version: str = Field(...)
    timestamp: datetime = Field(default_factory=datetime.now)
    components: Dict[str, str] = Field(default_factory=dict)
    uptime_seconds: float = Field(...)

class ReadinessCheckResponse(BaseModel):
    ready: bool = Field(...)
    checks: Dict[str, bool] = Field(default_factory=dict)
    message: Optional[str] = None
