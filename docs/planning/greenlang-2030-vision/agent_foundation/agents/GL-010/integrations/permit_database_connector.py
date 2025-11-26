"""
Permit Database Connector for GL-010 EMISSIONWATCH.

Provides integration with permit and regulatory limits database for
multi-jurisdiction compliance management, permit limit tracking,
and applicability determinations.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import asyncio
import json
import logging
import time
import uuid

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    ConnectionState,
    ConnectorType,
    HealthCheckResult,
    HealthStatus,
    ConnectorError,
    ConnectionError,
    ConfigurationError,
    ValidationError,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class PermitType(str, Enum):
    """Types of environmental permits."""

    TITLE_V = "title_v"  # Federal operating permit
    PSD = "psd"  # Prevention of Significant Deterioration
    NNSR = "nnsr"  # Nonattainment New Source Review
    MINOR_NSR = "minor_nsr"  # Minor new source review
    SIP = "sip"  # State Implementation Plan
    NSPS = "nsps"  # New Source Performance Standard
    NESHAP = "neshap"  # National Emission Standards for HAPs
    MACT = "mact"  # Maximum Achievable Control Technology
    CONSENT_DECREE = "consent_decree"
    ADMINISTRATIVE_ORDER = "administrative_order"
    STATE_PERMIT = "state_permit"
    LOCAL_PERMIT = "local_permit"


class PermitStatus(str, Enum):
    """Permit status codes."""

    DRAFT = "draft"
    PENDING_APPLICATION = "pending_application"
    UNDER_REVIEW = "under_review"
    PUBLIC_COMMENT = "public_comment"
    ISSUED = "issued"
    ACTIVE = "active"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"
    SUSPENDED = "suspended"
    MODIFIED = "modified"
    RENEWED = "renewed"


class LimitType(str, Enum):
    """Types of emission limits."""

    EMISSION_RATE = "emission_rate"  # lb/hr, tons/yr
    CONCENTRATION = "concentration"  # ppm, gr/dscf
    OPACITY = "opacity"  # percent
    WORK_PRACTICE = "work_practice"  # operational standard
    PARAMETRIC = "parametric"  # surrogate monitoring
    EQUIPMENT_STANDARD = "equipment_standard"
    FUEL_RESTRICTION = "fuel_restriction"
    HOURS_RESTRICTION = "hours_restriction"
    THROUGHPUT_LIMIT = "throughput_limit"
    VISIBLE_EMISSIONS = "visible_emissions"


class AveragingPeriod(str, Enum):
    """Averaging periods for limits."""

    INSTANTANEOUS = "instantaneous"
    ONE_MINUTE = "1_minute"
    THREE_MINUTE = "3_minute"
    SIX_MINUTE = "6_minute"
    ONE_HOUR = "1_hour"
    THREE_HOUR = "3_hour"
    EIGHT_HOUR = "8_hour"
    TWENTY_FOUR_HOUR = "24_hour"
    THIRTY_DAY = "30_day"
    CALENDAR_MONTH = "calendar_month"
    TWELVE_MONTH_ROLLING = "12_month_rolling"
    CALENDAR_YEAR = "calendar_year"
    CONSECUTIVE_HOURS = "consecutive_hours"


class Pollutant(str, Enum):
    """Regulated pollutants."""

    NOX = "nox"
    SO2 = "so2"
    CO = "co"
    PM = "pm"
    PM10 = "pm10"
    PM25 = "pm25"
    VOC = "voc"
    HAP = "hap"
    CO2 = "co2"
    CO2E = "co2e"
    N2O = "n2o"
    CH4 = "ch4"
    NH3 = "nh3"
    HCL = "hcl"
    HF = "hf"
    H2SO4 = "h2so4"
    PB = "pb"
    HG = "hg"
    OPACITY = "opacity"
    VE = "visible_emissions"


class Jurisdiction(str, Enum):
    """Regulatory jurisdictions."""

    FEDERAL_EPA = "federal_epa"
    STATE = "state"
    LOCAL = "local"
    AIR_DISTRICT = "air_district"
    TRIBAL = "tribal"
    INTERNATIONAL = "international"


class OperatingMode(str, Enum):
    """Operating modes for different limit applicability."""

    NORMAL = "normal"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    MALFUNCTION = "malfunction"
    EMERGENCY = "emergency"
    COMMISSIONING = "commissioning"
    MAINTENANCE = "maintenance"
    LOW_LOAD = "low_load"
    HIGH_LOAD = "high_load"


class ComplianceStatus(str, Enum):
    """Compliance status."""

    IN_COMPLIANCE = "in_compliance"
    OUT_OF_COMPLIANCE = "out_of_compliance"
    DEVIATION = "deviation"
    EXCEEDANCE = "exceedance"
    UNDER_REVIEW = "under_review"
    PENDING = "pending"
    NOT_APPLICABLE = "not_applicable"


# =============================================================================
# Pydantic Models
# =============================================================================


class EmissionLimit(BaseModel):
    """Emission limit definition."""

    model_config = ConfigDict(frozen=True)

    limit_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Limit identifier"
    )
    permit_id: str = Field(..., description="Associated permit ID")
    unit_id: Optional[str] = Field(default=None, description="Emission unit ID")
    stack_id: Optional[str] = Field(default=None, description="Stack ID")

    # Pollutant and type
    pollutant: Pollutant = Field(..., description="Regulated pollutant")
    limit_type: LimitType = Field(..., description="Type of limit")

    # Limit value
    limit_value: float = Field(..., description="Limit value")
    limit_unit: str = Field(..., description="Limit unit")
    averaging_period: AveragingPeriod = Field(..., description="Averaging period")

    # Applicability
    applicable_modes: List[OperatingMode] = Field(
        default_factory=lambda: [OperatingMode.NORMAL],
        description="Applicable operating modes"
    )
    effective_date: date = Field(..., description="Effective date")
    expiration_date: Optional[date] = Field(default=None, description="Expiration date")

    # Regulatory basis
    regulatory_citation: str = Field(..., description="Regulatory citation")
    permit_condition: str = Field(..., description="Permit condition reference")

    # Monitoring requirements
    monitoring_method: Optional[str] = Field(default=None, description="Monitoring method")
    monitoring_frequency: Optional[str] = Field(default=None, description="Monitoring frequency")

    # Compliance margin
    warning_threshold_percent: float = Field(
        default=80.0,
        ge=0,
        le=100,
        description="Warning threshold %"
    )

    # Flags
    is_federally_enforceable: bool = Field(
        default=True,
        description="Federally enforceable"
    )
    is_synthetic_minor: bool = Field(
        default=False,
        description="Synthetic minor limit"
    )

    notes: Optional[str] = Field(default=None, description="Notes")


class PermitCondition(BaseModel):
    """General permit condition."""

    model_config = ConfigDict(frozen=True)

    condition_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Condition identifier"
    )
    permit_id: str = Field(..., description="Permit ID")
    condition_number: str = Field(..., description="Condition number")
    condition_type: str = Field(..., description="Condition type")

    # Condition text
    title: str = Field(..., description="Condition title")
    text: str = Field(..., description="Full condition text")

    # Applicability
    unit_ids: List[str] = Field(default_factory=list, description="Applicable units")
    effective_date: date = Field(..., description="Effective date")
    expiration_date: Optional[date] = Field(default=None)

    # Compliance
    compliance_method: Optional[str] = Field(default=None)
    recordkeeping_requirements: Optional[str] = Field(default=None)
    reporting_requirements: Optional[str] = Field(default=None)

    is_federally_enforceable: bool = Field(default=True)


class StartupShutdownProvision(BaseModel):
    """Startup/shutdown/malfunction provision."""

    model_config = ConfigDict(frozen=True)

    provision_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Provision identifier"
    )
    permit_id: str = Field(..., description="Permit ID")
    unit_id: str = Field(..., description="Unit ID")

    # Operating mode
    operating_mode: OperatingMode = Field(..., description="Operating mode")

    # Duration limits
    max_duration_hours: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum duration"
    )
    max_events_per_year: Optional[int] = Field(
        default=None,
        ge=0,
        description="Max events per year"
    )

    # Alternate limits during mode
    alternate_limits: List[EmissionLimit] = Field(
        default_factory=list,
        description="Alternate limits"
    )

    # Requirements
    notification_required: bool = Field(default=False)
    notification_timing: Optional[str] = Field(default=None)
    documentation_requirements: Optional[str] = Field(default=None)
    work_practice_standards: Optional[str] = Field(default=None)


class MalfunctionProvision(BaseModel):
    """Malfunction provision definition."""

    model_config = ConfigDict(frozen=True)

    provision_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Provision identifier"
    )
    permit_id: str = Field(..., description="Permit ID")

    # Malfunction definition
    malfunction_definition: str = Field(..., description="Definition of malfunction")
    excluded_conditions: List[str] = Field(
        default_factory=list,
        description="Conditions not considered malfunction"
    )

    # Affirmative defense
    affirmative_defense_available: bool = Field(
        default=False,
        description="Affirmative defense available"
    )
    affirmative_defense_criteria: Optional[str] = Field(default=None)

    # Reporting requirements
    immediate_notification_required: bool = Field(default=True)
    written_report_required: bool = Field(default=True)
    report_deadline_days: int = Field(default=2, ge=0)

    # Documentation
    required_records: List[str] = Field(default_factory=list)


class Permit(BaseModel):
    """Complete permit record."""

    model_config = ConfigDict(frozen=True)

    permit_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Permit identifier"
    )
    permit_number: str = Field(..., description="Official permit number")
    permit_type: PermitType = Field(..., description="Permit type")
    permit_status: PermitStatus = Field(..., description="Permit status")

    # Facility
    facility_id: str = Field(..., description="Facility ID")
    facility_name: str = Field(..., description="Facility name")

    # Jurisdiction
    jurisdiction: Jurisdiction = Field(..., description="Issuing jurisdiction")
    issuing_authority: str = Field(..., description="Issuing authority")
    regulatory_contact: Optional[str] = Field(default=None)

    # Dates
    issue_date: date = Field(..., description="Issue date")
    effective_date: date = Field(..., description="Effective date")
    expiration_date: Optional[date] = Field(default=None, description="Expiration date")
    last_modified_date: Optional[date] = Field(default=None)

    # Version tracking
    version: str = Field(default="1.0", description="Permit version")
    revision_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Revision history"
    )

    # Content
    emission_limits: List[EmissionLimit] = Field(
        default_factory=list,
        description="Emission limits"
    )
    conditions: List[PermitCondition] = Field(
        default_factory=list,
        description="Permit conditions"
    )
    ssm_provisions: List[StartupShutdownProvision] = Field(
        default_factory=list,
        description="SSM provisions"
    )
    malfunction_provisions: List[MalfunctionProvision] = Field(
        default_factory=list,
        description="Malfunction provisions"
    )

    # Applicable regulations
    applicable_regulations: List[str] = Field(
        default_factory=list,
        description="Applicable regulations"
    )
    applicable_nsps: List[str] = Field(
        default_factory=list,
        description="Applicable NSPS subparts"
    )
    applicable_neshap: List[str] = Field(
        default_factory=list,
        description="Applicable NESHAP subparts"
    )

    # Documents
    permit_document_url: Optional[str] = Field(default=None)
    attachments: List[str] = Field(default_factory=list)

    def is_expired(self) -> bool:
        """Check if permit is expired."""
        if self.expiration_date is None:
            return False
        return date.today() > self.expiration_date

    def days_until_expiration(self) -> Optional[int]:
        """Calculate days until expiration."""
        if self.expiration_date is None:
            return None
        return (self.expiration_date - date.today()).days


class ApplicabilityDetermination(BaseModel):
    """Applicability determination record."""

    model_config = ConfigDict(frozen=True)

    determination_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Determination identifier"
    )
    facility_id: str = Field(..., description="Facility ID")
    unit_id: Optional[str] = Field(default=None, description="Unit ID")

    # Regulation
    regulation: str = Field(..., description="Regulation evaluated")
    subpart: Optional[str] = Field(default=None, description="Subpart")

    # Determination
    is_applicable: bool = Field(..., description="Is regulation applicable")
    determination_date: date = Field(..., description="Determination date")
    determination_basis: str = Field(..., description="Basis for determination")

    # Supporting information
    affected_source_definition: Optional[str] = Field(default=None)
    exemptions_evaluated: List[str] = Field(default_factory=list)
    calculations: Optional[str] = Field(default=None)
    documentation_references: List[str] = Field(default_factory=list)

    # Review
    prepared_by: str = Field(..., description="Prepared by")
    reviewed_by: Optional[str] = Field(default=None, description="Reviewed by")
    review_date: Optional[date] = Field(default=None)


class ComplianceRecord(BaseModel):
    """Compliance evaluation record."""

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Record identifier"
    )
    limit_id: str = Field(..., description="Limit ID")
    evaluation_period_start: datetime = Field(..., description="Period start")
    evaluation_period_end: datetime = Field(..., description="Period end")

    # Measured/calculated values
    actual_value: float = Field(..., description="Actual value")
    actual_unit: str = Field(..., description="Unit")
    limit_value: float = Field(..., description="Applicable limit")

    # Compliance status
    compliance_status: ComplianceStatus = Field(..., description="Status")
    percent_of_limit: float = Field(..., ge=0, description="Percent of limit")

    # Exceedance details
    exceedance_amount: Optional[float] = Field(default=None, ge=0)
    exceedance_duration_minutes: Optional[int] = Field(default=None, ge=0)

    # Data quality
    data_completeness_percent: Optional[float] = Field(default=None, ge=0, le=100)
    data_quality_flag: Optional[str] = Field(default=None)


class PermitDatabaseConnectorConfig(BaseConnectorConfig):
    """Configuration for permit database connector."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    connector_type: ConnectorType = Field(
        default=ConnectorType.PERMIT_DATABASE,
        description="Connector type"
    )

    # Database settings
    database_type: str = Field(
        default="postgresql",
        description="Database type"
    )
    database_host: str = Field(default="localhost", description="Database host")
    database_port: int = Field(default=5432, ge=1, le=65535)
    database_name: str = Field(..., description="Database name")
    database_schema: str = Field(default="permits", description="Schema name")

    # Facility
    facility_id: str = Field(..., description="Facility identifier")

    # Cache settings
    permit_cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="Permit cache TTL"
    )

    # Compliance evaluation
    default_warning_threshold_percent: float = Field(
        default=80.0,
        ge=0,
        le=100,
        description="Default warning threshold"
    )
    compliance_check_frequency_minutes: int = Field(
        default=60,
        ge=1,
        description="Compliance check frequency"
    )


# =============================================================================
# Permit Repository
# =============================================================================


class PermitRepository:
    """
    Repository for permit data storage and retrieval.

    In production, this would interface with a database.
    For this implementation, uses in-memory storage.
    """

    def __init__(self) -> None:
        """Initialize repository."""
        self._permits: Dict[str, Permit] = {}
        self._limits: Dict[str, EmissionLimit] = {}
        self._determinations: Dict[str, ApplicabilityDetermination] = {}
        self._compliance_records: List[ComplianceRecord] = []
        self._logger = logging.getLogger("permit.repository")

    async def save_permit(self, permit: Permit) -> str:
        """
        Save permit to repository.

        Args:
            permit: Permit to save

        Returns:
            Permit ID
        """
        self._permits[permit.permit_id] = permit

        # Index limits
        for limit in permit.emission_limits:
            self._limits[limit.limit_id] = limit

        self._logger.info(f"Saved permit {permit.permit_number}")
        return permit.permit_id

    async def get_permit(self, permit_id: str) -> Optional[Permit]:
        """
        Get permit by ID.

        Args:
            permit_id: Permit identifier

        Returns:
            Permit or None
        """
        return self._permits.get(permit_id)

    async def get_permit_by_number(self, permit_number: str) -> Optional[Permit]:
        """
        Get permit by permit number.

        Args:
            permit_number: Official permit number

        Returns:
            Permit or None
        """
        for permit in self._permits.values():
            if permit.permit_number == permit_number:
                return permit
        return None

    async def get_facility_permits(
        self,
        facility_id: str,
        active_only: bool = True,
    ) -> List[Permit]:
        """
        Get all permits for a facility.

        Args:
            facility_id: Facility identifier
            active_only: Only return active permits

        Returns:
            List of permits
        """
        permits = [
            p for p in self._permits.values()
            if p.facility_id == facility_id
        ]

        if active_only:
            permits = [
                p for p in permits
                if p.permit_status == PermitStatus.ACTIVE and not p.is_expired()
            ]

        return permits

    async def get_limit(self, limit_id: str) -> Optional[EmissionLimit]:
        """
        Get emission limit by ID.

        Args:
            limit_id: Limit identifier

        Returns:
            Emission limit or None
        """
        return self._limits.get(limit_id)

    async def get_limits_for_unit(
        self,
        unit_id: str,
        pollutant: Optional[Pollutant] = None,
    ) -> List[EmissionLimit]:
        """
        Get emission limits for a unit.

        Args:
            unit_id: Emission unit ID
            pollutant: Optional pollutant filter

        Returns:
            List of limits
        """
        limits = [
            l for l in self._limits.values()
            if l.unit_id == unit_id
        ]

        if pollutant:
            limits = [l for l in limits if l.pollutant == pollutant]

        # Filter by effective date
        today = date.today()
        limits = [
            l for l in limits
            if l.effective_date <= today and (
                l.expiration_date is None or l.expiration_date >= today
            )
        ]

        return limits

    async def save_determination(
        self,
        determination: ApplicabilityDetermination,
    ) -> str:
        """
        Save applicability determination.

        Args:
            determination: Determination to save

        Returns:
            Determination ID
        """
        self._determinations[determination.determination_id] = determination
        return determination.determination_id

    async def get_determinations_for_facility(
        self,
        facility_id: str,
    ) -> List[ApplicabilityDetermination]:
        """
        Get applicability determinations for facility.

        Args:
            facility_id: Facility ID

        Returns:
            List of determinations
        """
        return [
            d for d in self._determinations.values()
            if d.facility_id == facility_id
        ]

    async def save_compliance_record(
        self,
        record: ComplianceRecord,
    ) -> str:
        """
        Save compliance record.

        Args:
            record: Compliance record

        Returns:
            Record ID
        """
        self._compliance_records.append(record)
        return record.record_id

    async def get_compliance_history(
        self,
        limit_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[ComplianceRecord]:
        """
        Get compliance history for a limit.

        Args:
            limit_id: Limit ID
            start_date: Start date
            end_date: End date

        Returns:
            List of compliance records
        """
        return [
            r for r in self._compliance_records
            if r.limit_id == limit_id
            and r.evaluation_period_start >= start_date
            and r.evaluation_period_end <= end_date
        ]


# =============================================================================
# Compliance Evaluator
# =============================================================================


class ComplianceEvaluator:
    """
    Evaluates compliance against permit limits.
    """

    def __init__(self, warning_threshold_percent: float = 80.0) -> None:
        """
        Initialize evaluator.

        Args:
            warning_threshold_percent: Default warning threshold
        """
        self._warning_threshold = warning_threshold_percent
        self._logger = logging.getLogger("permit.compliance")

    def evaluate_limit(
        self,
        limit: EmissionLimit,
        actual_value: float,
        actual_unit: str,
        period_start: datetime,
        period_end: datetime,
    ) -> ComplianceRecord:
        """
        Evaluate compliance against a single limit.

        Args:
            limit: Emission limit
            actual_value: Measured/calculated value
            actual_unit: Unit of actual value
            period_start: Evaluation period start
            period_end: Evaluation period end

        Returns:
            Compliance record
        """
        # TODO: Unit conversion if needed

        # Calculate percent of limit
        percent_of_limit = (actual_value / limit.limit_value * 100) if limit.limit_value > 0 else 0

        # Determine status
        if actual_value > limit.limit_value:
            status = ComplianceStatus.EXCEEDANCE
            exceedance_amount = actual_value - limit.limit_value
        elif percent_of_limit >= limit.warning_threshold_percent:
            status = ComplianceStatus.DEVIATION
            exceedance_amount = None
        else:
            status = ComplianceStatus.IN_COMPLIANCE
            exceedance_amount = None

        record = ComplianceRecord(
            limit_id=limit.limit_id,
            evaluation_period_start=period_start,
            evaluation_period_end=period_end,
            actual_value=actual_value,
            actual_unit=actual_unit,
            limit_value=limit.limit_value,
            compliance_status=status,
            percent_of_limit=percent_of_limit,
            exceedance_amount=exceedance_amount,
        )

        self._logger.info(
            f"Compliance evaluation: {limit.pollutant.value} - "
            f"{actual_value}/{limit.limit_value} ({percent_of_limit:.1f}%) - "
            f"{status.value}"
        )

        return record

    def get_applicable_limit(
        self,
        limits: List[EmissionLimit],
        operating_mode: OperatingMode = OperatingMode.NORMAL,
        timestamp: Optional[datetime] = None,
    ) -> Optional[EmissionLimit]:
        """
        Get the applicable limit for current conditions.

        Args:
            limits: Available limits
            operating_mode: Current operating mode
            timestamp: Evaluation timestamp

        Returns:
            Most restrictive applicable limit
        """
        ts = timestamp or datetime.utcnow()
        eval_date = ts.date()

        # Filter by date and operating mode
        applicable = [
            l for l in limits
            if l.effective_date <= eval_date
            and (l.expiration_date is None or l.expiration_date >= eval_date)
            and operating_mode in l.applicable_modes
        ]

        if not applicable:
            return None

        # Return most restrictive (lowest) limit
        return min(applicable, key=lambda l: l.limit_value)


# =============================================================================
# Permit Database Connector
# =============================================================================


class PermitDatabaseConnector(BaseConnector):
    """
    Permit Database Connector.

    Provides comprehensive permit and regulatory limits management:
    - Permit storage and retrieval
    - Multi-jurisdiction support
    - Limit version tracking
    - Applicability determinations
    - SSM provisions
    - Compliance evaluation

    Features:
    - Hierarchical permit structure
    - Limit applicability by operating mode
    - Warning threshold monitoring
    - Regulatory citation tracking
    - Version history
    """

    def __init__(self, config: PermitDatabaseConnectorConfig) -> None:
        """
        Initialize permit database connector.

        Args:
            config: Connector configuration
        """
        super().__init__(config)
        self._permit_config = config

        # Initialize components
        self._repository = PermitRepository()
        self._compliance_evaluator = ComplianceEvaluator(
            config.default_warning_threshold_percent
        )

        # Cached data
        self._permit_cache: Dict[str, Tuple[datetime, Permit]] = {}

        self._logger = logging.getLogger(f"permit.connector.{config.facility_id}")

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Establish connection to permit database.

        Raises:
            ConnectionError: If connection fails
        """
        self._state = ConnectionState.CONNECTING
        self._logger.info("Connecting to permit database")

        try:
            # In production, establish database connection
            self._state = ConnectionState.CONNECTED
            self._logger.info("Permit database connected")

            await self._audit_logger.log_operation(
                operation="connect",
                status="success",
            )

        except Exception as e:
            self._state = ConnectionState.ERROR
            raise ConnectionError(f"Failed to connect to permit database: {e}")

    async def disconnect(self) -> None:
        """Disconnect from permit database."""
        self._logger.info("Disconnecting from permit database")

        # Clear cache
        self._permit_cache.clear()

        self._state = ConnectionState.DISCONNECTED

        await self._audit_logger.log_operation(
            operation="disconnect",
            status="success",
        )

    async def health_check(self) -> HealthCheckResult:
        """
        Perform health check on permit database.

        Returns:
            Health check result
        """
        start_time = time.time()

        try:
            # Test database connection
            # In production, execute test query

            latency_ms = (time.time() - start_time) * 1000

            # Check for expiring permits
            permits = await self._repository.get_facility_permits(
                self._permit_config.facility_id
            )

            expiring_soon = [
                p for p in permits
                if p.days_until_expiration() is not None
                and p.days_until_expiration() <= 90
            ]

            status = HealthStatus.HEALTHY
            message = "Permit database healthy"

            if expiring_soon:
                status = HealthStatus.DEGRADED
                message = f"{len(expiring_soon)} permits expiring within 90 days"

            return HealthCheckResult(
                status=status,
                latency_ms=latency_ms,
                message=message,
                details={
                    "total_permits": len(permits),
                    "expiring_soon": len(expiring_soon),
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Health check failed: {e}",
            )

    async def validate_configuration(self) -> bool:
        """
        Validate permit database configuration.

        Returns:
            True if configuration is valid
        """
        issues: List[str] = []

        if not self._permit_config.database_name:
            issues.append("database_name is required")

        if not self._permit_config.facility_id:
            issues.append("facility_id is required")

        if issues:
            raise ConfigurationError(
                f"Invalid permit database configuration: {issues}",
                connector_id=self._config.connector_id,
            )

        return True

    # -------------------------------------------------------------------------
    # Permit Management Methods
    # -------------------------------------------------------------------------

    async def add_permit(self, permit: Permit) -> str:
        """
        Add a permit to the database.

        Args:
            permit: Permit to add

        Returns:
            Permit ID
        """
        permit_id = await self._repository.save_permit(permit)

        await self._audit_logger.log_operation(
            operation="add_permit",
            status="success",
            request_data={
                "permit_number": permit.permit_number,
                "permit_type": permit.permit_type.value,
            },
            response_summary=f"Added permit {permit.permit_number}",
        )

        return permit_id

    async def get_permit(self, permit_id: str) -> Optional[Permit]:
        """
        Get permit by ID.

        Args:
            permit_id: Permit identifier

        Returns:
            Permit or None
        """
        # Check cache
        if permit_id in self._permit_cache:
            cached_time, permit = self._permit_cache[permit_id]
            cache_age = (datetime.utcnow() - cached_time).total_seconds()
            if cache_age < self._permit_config.permit_cache_ttl_seconds:
                return permit

        permit = await self._repository.get_permit(permit_id)

        if permit:
            self._permit_cache[permit_id] = (datetime.utcnow(), permit)

        return permit

    async def get_facility_permits(
        self,
        active_only: bool = True,
    ) -> List[Permit]:
        """
        Get all permits for the configured facility.

        Args:
            active_only: Only return active permits

        Returns:
            List of permits
        """
        return await self._repository.get_facility_permits(
            self._permit_config.facility_id,
            active_only,
        )

    async def get_limit(
        self,
        unit_id: str,
        pollutant: Pollutant,
        operating_mode: OperatingMode = OperatingMode.NORMAL,
    ) -> Optional[EmissionLimit]:
        """
        Get applicable emission limit.

        Args:
            unit_id: Emission unit ID
            pollutant: Pollutant
            operating_mode: Operating mode

        Returns:
            Applicable limit or None
        """
        limits = await self._repository.get_limits_for_unit(unit_id, pollutant)

        return self._compliance_evaluator.get_applicable_limit(
            limits,
            operating_mode,
        )

    async def get_all_limits_for_unit(
        self,
        unit_id: str,
    ) -> Dict[Pollutant, List[EmissionLimit]]:
        """
        Get all emission limits for a unit grouped by pollutant.

        Args:
            unit_id: Emission unit ID

        Returns:
            Dictionary of pollutant to limits
        """
        limits = await self._repository.get_limits_for_unit(unit_id)

        result: Dict[Pollutant, List[EmissionLimit]] = {}
        for limit in limits:
            if limit.pollutant not in result:
                result[limit.pollutant] = []
            result[limit.pollutant].append(limit)

        return result

    # -------------------------------------------------------------------------
    # Applicability Methods
    # -------------------------------------------------------------------------

    async def add_applicability_determination(
        self,
        determination: ApplicabilityDetermination,
    ) -> str:
        """
        Add applicability determination.

        Args:
            determination: Determination to add

        Returns:
            Determination ID
        """
        det_id = await self._repository.save_determination(determination)

        await self._audit_logger.log_operation(
            operation="add_applicability_determination",
            status="success",
            request_data={
                "regulation": determination.regulation,
                "is_applicable": determination.is_applicable,
            },
        )

        return det_id

    async def get_applicable_regulations(
        self,
        unit_id: Optional[str] = None,
    ) -> List[ApplicabilityDetermination]:
        """
        Get applicable regulations for facility/unit.

        Args:
            unit_id: Optional unit filter

        Returns:
            List of applicable determinations
        """
        determinations = await self._repository.get_determinations_for_facility(
            self._permit_config.facility_id
        )

        applicable = [d for d in determinations if d.is_applicable]

        if unit_id:
            applicable = [
                d for d in applicable
                if d.unit_id is None or d.unit_id == unit_id
            ]

        return applicable

    # -------------------------------------------------------------------------
    # Compliance Methods
    # -------------------------------------------------------------------------

    async def evaluate_compliance(
        self,
        unit_id: str,
        pollutant: Pollutant,
        actual_value: float,
        actual_unit: str,
        period_start: datetime,
        period_end: datetime,
        operating_mode: OperatingMode = OperatingMode.NORMAL,
    ) -> ComplianceRecord:
        """
        Evaluate compliance against applicable limit.

        Args:
            unit_id: Emission unit ID
            pollutant: Pollutant
            actual_value: Measured value
            actual_unit: Unit of measurement
            period_start: Evaluation period start
            period_end: Evaluation period end
            operating_mode: Operating mode

        Returns:
            Compliance record

        Raises:
            ValidationError: If no applicable limit found
        """
        limit = await self.get_limit(unit_id, pollutant, operating_mode)

        if not limit:
            raise ValidationError(
                f"No applicable limit found for {pollutant.value} on unit {unit_id}"
            )

        record = self._compliance_evaluator.evaluate_limit(
            limit,
            actual_value,
            actual_unit,
            period_start,
            period_end,
        )

        await self._repository.save_compliance_record(record)

        await self._audit_logger.log_operation(
            operation="evaluate_compliance",
            status="success" if record.compliance_status == ComplianceStatus.IN_COMPLIANCE else "warning",
            request_data={
                "unit_id": unit_id,
                "pollutant": pollutant.value,
                "actual_value": actual_value,
            },
            response_summary=f"Status: {record.compliance_status.value} ({record.percent_of_limit:.1f}%)",
        )

        return record

    async def get_compliance_summary(
        self,
        unit_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """
        Get compliance summary for a unit.

        Args:
            unit_id: Emission unit ID
            start_date: Start date
            end_date: End date

        Returns:
            Compliance summary
        """
        limits = await self._repository.get_limits_for_unit(unit_id)

        summary = {
            "unit_id": unit_id,
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_limits": len(limits),
            "by_pollutant": {},
            "exceedances": 0,
            "deviations": 0,
        }

        for limit in limits:
            records = await self._repository.get_compliance_history(
                limit.limit_id,
                start_date,
                end_date,
            )

            exceedances = sum(
                1 for r in records
                if r.compliance_status == ComplianceStatus.EXCEEDANCE
            )
            deviations = sum(
                1 for r in records
                if r.compliance_status == ComplianceStatus.DEVIATION
            )

            if limit.pollutant.value not in summary["by_pollutant"]:
                summary["by_pollutant"][limit.pollutant.value] = {
                    "evaluations": 0,
                    "exceedances": 0,
                    "deviations": 0,
                }

            summary["by_pollutant"][limit.pollutant.value]["evaluations"] += len(records)
            summary["by_pollutant"][limit.pollutant.value]["exceedances"] += exceedances
            summary["by_pollutant"][limit.pollutant.value]["deviations"] += deviations

            summary["exceedances"] += exceedances
            summary["deviations"] += deviations

        return summary

    # -------------------------------------------------------------------------
    # SSM Provisions Methods
    # -------------------------------------------------------------------------

    async def get_ssm_provisions(
        self,
        unit_id: str,
        operating_mode: OperatingMode,
    ) -> List[StartupShutdownProvision]:
        """
        Get SSM provisions for a unit and mode.

        Args:
            unit_id: Emission unit ID
            operating_mode: Operating mode

        Returns:
            Applicable SSM provisions
        """
        permits = await self.get_facility_permits()

        provisions = []
        for permit in permits:
            for provision in permit.ssm_provisions:
                if provision.unit_id == unit_id and provision.operating_mode == operating_mode:
                    provisions.append(provision)

        return provisions

    async def check_permit_expiration(self) -> List[Dict[str, Any]]:
        """
        Check for permits approaching expiration.

        Returns:
            List of expiring permits with details
        """
        permits = await self.get_facility_permits()

        expiring = []
        for permit in permits:
            days = permit.days_until_expiration()
            if days is not None and days <= 180:  # Within 6 months
                expiring.append({
                    "permit_id": permit.permit_id,
                    "permit_number": permit.permit_number,
                    "permit_type": permit.permit_type.value,
                    "expiration_date": permit.expiration_date.isoformat() if permit.expiration_date else None,
                    "days_remaining": days,
                    "urgency": "critical" if days <= 30 else "high" if days <= 90 else "medium",
                })

        await self._audit_logger.log_operation(
            operation="check_permit_expiration",
            status="success",
            response_summary=f"Found {len(expiring)} expiring permits",
        )

        return expiring


# =============================================================================
# Factory Function
# =============================================================================


def create_permit_database_connector(
    facility_id: str,
    database_name: str,
    database_host: str = "localhost",
    **kwargs: Any,
) -> PermitDatabaseConnector:
    """
    Factory function to create permit database connector.

    Args:
        facility_id: Facility identifier
        database_name: Database name
        database_host: Database host
        **kwargs: Additional configuration

    Returns:
        Configured permit database connector
    """
    config = PermitDatabaseConnectorConfig(
        connector_name=f"PermitDB_{facility_id}",
        facility_id=facility_id,
        database_name=database_name,
        database_host=database_host,
        **kwargs,
    )

    return PermitDatabaseConnector(config)
