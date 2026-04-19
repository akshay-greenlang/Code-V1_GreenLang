"""
LOPA/HAZOP Integration Module - GL-007_FurnacePulse Safety Module

This module provides integration between operational telemetry and Layer of
Protection Analysis (LOPA) / Hazard and Operability Study (HAZOP) systems.

It maintains a Protection Layer Register, tracks Independent Protection Layer (IPL)
health, links to HAZOP recommendations, and supports Management of Change (MOC)
processes.

Key Features:
    - Protection Layer Register with unique IPL IDs
    - Telemetry-to-IPL degradation indicator mapping
    - HAZOP recommendation tracking
    - MOC reference mapping
    - IPL health dashboard (overdue tests, overrides, demands)
    - Incident investigation support with event reconstruction

Example:
    >>> integrator = LOPAHAZOPIntegrator(config)
    >>> integrator.register_ipl(ipl_data)
    >>> health = integrator.get_ipl_health_dashboard("FRN-001")
"""

from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import hashlib
import logging
import json

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class IPLCategory(str, Enum):
    """Categories of Independent Protection Layers."""
    BPCS = "bpcs"  # Basic Process Control System
    ALARM = "alarm"  # Alarm with operator response
    SIS = "sis"  # Safety Instrumented System
    RELIEF = "relief"  # Relief device
    MECHANICAL = "mechanical"  # Mechanical integrity
    HUMAN = "human"  # Human intervention
    DIKE = "dike"  # Containment dike
    FLAME_SAFEGUARD = "flame_safeguard"  # Flame safeguard system


class IPLStatus(str, Enum):
    """Status of an IPL."""
    ACTIVE = "active"
    DEGRADED = "degraded"
    BYPASSED = "bypassed"
    FAILED = "failed"
    UNDER_TEST = "under_test"
    MAINTENANCE = "maintenance"


class HAZOPRecommendationStatus(str, Enum):
    """Status of HAZOP recommendations."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    CLOSED = "closed"
    DEFERRED = "deferred"


class MOCStatus(str, Enum):
    """Management of Change status."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    IMPLEMENTED = "implemented"
    CLOSED = "closed"
    REJECTED = "rejected"


class DemandType(str, Enum):
    """Types of IPL demands."""
    TRUE_DEMAND = "true_demand"  # Actual process demand
    TEST_DEMAND = "test_demand"  # Proof test
    SPURIOUS = "spurious"  # Spurious/false trip
    UNKNOWN = "unknown"


# =============================================================================
# Pydantic Models
# =============================================================================

class LOPAHAZOPConfig(BaseModel):
    """Configuration for LOPA/HAZOP integration."""

    site_id: str = Field(..., description="Site identifier")
    proof_test_overdue_threshold_days: int = Field(
        default=30, ge=1, le=365,
        description="Days past due date to flag as overdue"
    )
    max_bypass_duration_hours: int = Field(
        default=8, ge=1, le=168,
        description="Maximum allowed bypass duration in hours"
    )
    demand_rate_window_days: int = Field(
        default=365, ge=30, le=1825,
        description="Window for calculating demand rates"
    )
    pfd_threshold_warning: float = Field(
        default=0.1, ge=0.001, le=0.5,
        description="PFD threshold for warning alerts"
    )
    audit_retention_years: int = Field(
        default=10, ge=1, le=25,
        description="Audit record retention period"
    )


class DegradationIndicator(BaseModel):
    """Telemetry-based degradation indicator for IPL."""

    indicator_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique indicator ID"
    )
    ipl_id: str = Field(..., description="Associated IPL ID")
    parameter_name: str = Field(..., description="Telemetry parameter name")
    parameter_tag: str = Field(..., description="Telemetry tag ID")
    threshold_warning: float = Field(..., description="Warning threshold")
    threshold_critical: float = Field(..., description="Critical threshold")
    comparison: str = Field(
        ..., pattern="^(gt|lt|gte|lte|eq|ne)$",
        description="Comparison operator"
    )
    current_value: Optional[float] = Field(None, description="Current value")
    last_updated: Optional[datetime] = Field(None, description="Last update time")
    is_degraded: bool = Field(default=False, description="Degradation status")
    degradation_level: Optional[str] = Field(
        None, pattern="^(warning|critical)$",
        description="Degradation severity"
    )


class IPLDemand(BaseModel):
    """Record of an IPL demand event."""

    demand_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique demand ID"
    )
    ipl_id: str = Field(..., description="IPL that was demanded")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Demand timestamp"
    )
    demand_type: DemandType = Field(..., description="Type of demand")
    success: bool = Field(..., description="Whether IPL functioned correctly")
    response_time_seconds: Optional[float] = Field(
        None, description="Time to respond"
    )
    initiating_event: str = Field(..., description="What triggered the demand")
    process_value_at_demand: Optional[float] = Field(
        None, description="Process value when demand occurred"
    )
    notes: Optional[str] = Field(None, description="Additional notes")
    recorded_by: str = Field(..., description="Who recorded this demand")


class ProofTest(BaseModel):
    """Record of IPL proof testing."""

    test_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique test ID"
    )
    ipl_id: str = Field(..., description="IPL being tested")
    test_date: datetime = Field(..., description="Test date")
    next_test_due: datetime = Field(..., description="Next test due date")
    test_result: str = Field(
        ..., pattern="^(pass|fail|partial)$",
        description="Test result"
    )
    tester_id: str = Field(..., description="Person who performed test")
    test_procedure: str = Field(..., description="Procedure reference")
    findings: Optional[str] = Field(None, description="Test findings")
    corrective_actions: Optional[str] = Field(
        None, description="Corrective actions if failed"
    )
    cmms_work_order: Optional[str] = Field(None, description="CMMS work order")


class BypassRecord(BaseModel):
    """Record of IPL bypass."""

    bypass_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique bypass ID"
    )
    ipl_id: str = Field(..., description="IPL being bypassed")
    start_time: datetime = Field(..., description="Bypass start time")
    planned_end_time: datetime = Field(..., description="Planned end time")
    actual_end_time: Optional[datetime] = Field(None, description="Actual end time")
    reason: str = Field(..., description="Reason for bypass")
    risk_assessment: str = Field(..., description="Risk assessment summary")
    compensating_measures: List[str] = Field(
        ..., description="Compensating measures in place"
    )
    approved_by: str = Field(..., description="Approver")
    requested_by: str = Field(..., description="Requester")
    is_active: bool = Field(default=True, description="Whether bypass is active")
    exceeded_duration: bool = Field(
        default=False, description="Whether duration exceeded plan"
    )


class IPL(BaseModel):
    """Independent Protection Layer definition."""

    ipl_id: str = Field(..., description="Unique IPL identifier")
    name: str = Field(..., description="IPL name")
    description: str = Field(..., description="IPL description")
    category: IPLCategory = Field(..., description="IPL category")
    asset_id: str = Field(..., description="Associated asset ID")
    furnace_id: str = Field(..., description="Associated furnace ID")
    sif_reference: Optional[str] = Field(
        None, description="SIF reference if applicable"
    )
    pfd_avg: float = Field(
        ..., ge=0.0, le=1.0,
        description="Probability of Failure on Demand"
    )
    sil_level: Optional[int] = Field(
        None, ge=1, le=4,
        description="SIL level if applicable"
    )
    proof_test_interval_days: int = Field(
        ..., ge=1, le=3650,
        description="Proof test interval in days"
    )
    status: IPLStatus = Field(default=IPLStatus.ACTIVE, description="Current status")
    hazop_references: List[str] = Field(
        default_factory=list,
        description="Related HAZOP item references"
    )
    lopa_scenario_ids: List[str] = Field(
        default_factory=list,
        description="Related LOPA scenario IDs"
    )
    degradation_indicators: List[DegradationIndicator] = Field(
        default_factory=list,
        description="Telemetry-based degradation indicators"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    last_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last modification timestamp"
    )


class HAZOPRecommendation(BaseModel):
    """HAZOP recommendation tracking."""

    recommendation_id: str = Field(..., description="Unique recommendation ID")
    hazop_study_id: str = Field(..., description="Parent HAZOP study ID")
    node_id: str = Field(..., description="Process node reference")
    deviation: str = Field(..., description="Deviation studied")
    cause: str = Field(..., description="Potential cause")
    consequence: str = Field(..., description="Potential consequence")
    safeguards: List[str] = Field(..., description="Existing safeguards")
    recommendation: str = Field(..., description="Recommendation text")
    priority: str = Field(
        ..., pattern="^(high|medium|low)$",
        description="Priority level"
    )
    status: HAZOPRecommendationStatus = Field(
        default=HAZOPRecommendationStatus.OPEN,
        description="Current status"
    )
    owner: str = Field(..., description="Responsible person")
    due_date: datetime = Field(..., description="Due date")
    completion_date: Optional[datetime] = Field(None, description="Completion date")
    verification_date: Optional[datetime] = Field(None, description="Verification date")
    linked_ipl_ids: List[str] = Field(
        default_factory=list,
        description="Related IPL IDs"
    )
    moc_reference: Optional[str] = Field(None, description="Related MOC reference")
    furnace_id: str = Field(..., description="Associated furnace ID")


class MOCRecord(BaseModel):
    """Management of Change record."""

    moc_id: str = Field(..., description="Unique MOC ID")
    title: str = Field(..., description="MOC title")
    description: str = Field(..., description="Change description")
    change_type: str = Field(
        ..., pattern="^(temporary|permanent)$",
        description="Change type"
    )
    status: MOCStatus = Field(default=MOCStatus.DRAFT, description="Current status")
    requester: str = Field(..., description="Requester")
    approvers: List[str] = Field(..., description="Required approvers")
    approval_date: Optional[datetime] = Field(None, description="Approval date")
    implementation_date: Optional[datetime] = Field(
        None, description="Implementation date"
    )
    reversion_date: Optional[datetime] = Field(
        None, description="Reversion date for temporary changes"
    )
    affected_ipls: List[str] = Field(
        default_factory=list,
        description="Affected IPL IDs"
    )
    affected_hazop_items: List[str] = Field(
        default_factory=list,
        description="Affected HAZOP items"
    )
    risk_assessment_summary: str = Field(..., description="Risk assessment")
    furnace_id: str = Field(..., description="Associated furnace ID")
    pssr_required: bool = Field(
        default=False, description="Pre-startup safety review required"
    )
    pssr_completed: bool = Field(default=False, description="PSSR completed")


class EventTimelineEntry(BaseModel):
    """Entry in event reconstruction timeline."""

    timestamp: datetime = Field(..., description="Event timestamp")
    event_type: str = Field(..., description="Type of event")
    description: str = Field(..., description="Event description")
    source: str = Field(..., description="Data source")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    ipl_involvement: Optional[str] = Field(None, description="Related IPL ID")


class IPLHealthDashboard(BaseModel):
    """Dashboard data for IPL health status."""

    furnace_id: str = Field(..., description="Furnace ID")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Generation timestamp"
    )
    total_ipls: int = Field(..., description="Total IPL count")
    active_ipls: int = Field(..., description="Active IPL count")
    degraded_ipls: int = Field(..., description="Degraded IPL count")
    bypassed_ipls: int = Field(..., description="Bypassed IPL count")
    overdue_tests: List[Dict[str, Any]] = Field(
        ..., description="IPLs with overdue tests"
    )
    active_bypasses: List[Dict[str, Any]] = Field(
        ..., description="Active bypass records"
    )
    exceeded_bypasses: List[Dict[str, Any]] = Field(
        ..., description="Bypasses exceeding duration"
    )
    recent_demands: List[Dict[str, Any]] = Field(
        ..., description="Recent IPL demands"
    )
    demand_rate_summary: Dict[str, float] = Field(
        ..., description="Demand rates by IPL"
    )
    hazop_recommendations_open: int = Field(
        ..., description="Open HAZOP recommendations"
    )
    hazop_recommendations_overdue: int = Field(
        ..., description="Overdue HAZOP recommendations"
    )
    overall_health_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Overall health score percentage"
    )


class AuditEntry(BaseModel):
    """Audit log entry for LOPA/HAZOP activities."""

    audit_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique audit ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    action: str = Field(..., description="Action performed")
    entity_type: str = Field(..., description="Entity type affected")
    entity_id: str = Field(..., description="Entity ID affected")
    actor_id: str = Field(..., description="Actor user or system ID")
    furnace_id: str = Field(..., description="Associated furnace ID")
    changes: Dict[str, Any] = Field(default_factory=dict, description="Change details")
    data_hash: str = Field(..., description="SHA-256 hash of change data")


# =============================================================================
# LOPA/HAZOP Integrator
# =============================================================================

class LOPAHAZOPIntegrator:
    """
    LOPA/HAZOP Integration Manager for furnace safety systems.

    This integrator maintains the Protection Layer Register, tracks IPL health,
    links telemetry to degradation indicators, manages HAZOP recommendations,
    and supports incident investigation through event reconstruction.

    Attributes:
        config: Integration configuration
        ipl_register: Protection Layer Register
        demands: IPL demand history
        proof_tests: Proof test records
        bypasses: Bypass records
        hazop_recommendations: HAZOP recommendation tracking
        moc_records: Management of Change records
        audit_log: Audit trail

    Example:
        >>> config = LOPAHAZOPConfig(site_id="SITE-001")
        >>> integrator = LOPAHAZOPIntegrator(config)
        >>> integrator.register_ipl(ipl)
        >>> health = integrator.get_ipl_health_dashboard("FRN-001")
    """

    def __init__(self, config: LOPAHAZOPConfig):
        """
        Initialize LOPAHAZOPIntegrator.

        Args:
            config: Integration configuration
        """
        self.config = config
        self.ipl_register: Dict[str, IPL] = {}
        self.demands: Dict[str, List[IPLDemand]] = {}
        self.proof_tests: Dict[str, List[ProofTest]] = {}
        self.bypasses: Dict[str, List[BypassRecord]] = {}
        self.hazop_recommendations: Dict[str, HAZOPRecommendation] = {}
        self.moc_records: Dict[str, MOCRecord] = {}
        self.audit_log: List[AuditEntry] = []

        logger.info(f"LOPAHAZOPIntegrator initialized for site {config.site_id}")

    # =========================================================================
    # IPL Registration and Management
    # =========================================================================

    def register_ipl(self, ipl: IPL, registered_by: str) -> IPL:
        """
        Register a new IPL in the Protection Layer Register.

        Args:
            ipl: IPL to register
            registered_by: User registering the IPL

        Returns:
            Registered IPL
        """
        if ipl.ipl_id in self.ipl_register:
            logger.warning(f"IPL {ipl.ipl_id} already exists, updating instead")

        self.ipl_register[ipl.ipl_id] = ipl
        self.demands[ipl.ipl_id] = []
        self.proof_tests[ipl.ipl_id] = []
        self.bypasses[ipl.ipl_id] = []

        self._log_audit(
            action="IPL_REGISTERED",
            entity_type="IPL",
            entity_id=ipl.ipl_id,
            actor_id=registered_by,
            furnace_id=ipl.furnace_id,
            changes={
                "name": ipl.name,
                "category": ipl.category.value,
                "pfd_avg": ipl.pfd_avg,
                "proof_test_interval_days": ipl.proof_test_interval_days,
            }
        )

        logger.info(f"IPL registered: {ipl.ipl_id} ({ipl.name})")
        return ipl

    def update_ipl_status(
        self,
        ipl_id: str,
        new_status: IPLStatus,
        updated_by: str,
        reason: Optional[str] = None
    ) -> IPL:
        """
        Update IPL status.

        Args:
            ipl_id: IPL identifier
            new_status: New status
            updated_by: User making update
            reason: Reason for status change

        Returns:
            Updated IPL

        Raises:
            ValueError: If IPL not found
        """
        if ipl_id not in self.ipl_register:
            raise ValueError(f"IPL {ipl_id} not found")

        ipl = self.ipl_register[ipl_id]
        old_status = ipl.status
        ipl.status = new_status
        ipl.last_modified = datetime.now(timezone.utc)

        self._log_audit(
            action="IPL_STATUS_CHANGED",
            entity_type="IPL",
            entity_id=ipl_id,
            actor_id=updated_by,
            furnace_id=ipl.furnace_id,
            changes={
                "old_status": old_status.value,
                "new_status": new_status.value,
                "reason": reason,
            }
        )

        logger.info(f"IPL {ipl_id} status changed: {old_status.value} -> {new_status.value}")
        return ipl

    def add_degradation_indicator(
        self,
        ipl_id: str,
        indicator: DegradationIndicator,
        added_by: str
    ) -> DegradationIndicator:
        """
        Add degradation indicator to IPL.

        Args:
            ipl_id: IPL identifier
            indicator: Degradation indicator to add
            added_by: User adding indicator

        Returns:
            Added DegradationIndicator

        Raises:
            ValueError: If IPL not found
        """
        if ipl_id not in self.ipl_register:
            raise ValueError(f"IPL {ipl_id} not found")

        indicator.ipl_id = ipl_id
        self.ipl_register[ipl_id].degradation_indicators.append(indicator)
        self.ipl_register[ipl_id].last_modified = datetime.now(timezone.utc)

        self._log_audit(
            action="DEGRADATION_INDICATOR_ADDED",
            entity_type="IPL",
            entity_id=ipl_id,
            actor_id=added_by,
            furnace_id=self.ipl_register[ipl_id].furnace_id,
            changes={
                "indicator_id": indicator.indicator_id,
                "parameter_name": indicator.parameter_name,
                "parameter_tag": indicator.parameter_tag,
            }
        )

        logger.info(
            f"Degradation indicator added to IPL {ipl_id}: {indicator.parameter_name}"
        )
        return indicator

    def update_degradation_from_telemetry(
        self,
        ipl_id: str,
        parameter_tag: str,
        value: float,
        timestamp: datetime
    ) -> Optional[DegradationIndicator]:
        """
        Update degradation indicator from telemetry data.

        Args:
            ipl_id: IPL identifier
            parameter_tag: Telemetry tag
            value: Current value
            timestamp: Telemetry timestamp

        Returns:
            Updated DegradationIndicator if found, None otherwise
        """
        if ipl_id not in self.ipl_register:
            return None

        ipl = self.ipl_register[ipl_id]
        for indicator in ipl.degradation_indicators:
            if indicator.parameter_tag == parameter_tag:
                indicator.current_value = value
                indicator.last_updated = timestamp

                # Check thresholds
                is_degraded, level = self._check_degradation(indicator, value)
                old_degraded = indicator.is_degraded
                indicator.is_degraded = is_degraded
                indicator.degradation_level = level

                # Update IPL status if degradation changed
                if is_degraded and not old_degraded:
                    ipl.status = IPLStatus.DEGRADED
                    logger.warning(
                        f"IPL {ipl_id} degraded: {indicator.parameter_name} = {value} "
                        f"(threshold: {level})"
                    )

                return indicator

        return None

    def _check_degradation(
        self,
        indicator: DegradationIndicator,
        value: float
    ) -> tuple:
        """Check if value triggers degradation."""
        comparison_funcs = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
            "eq": lambda v, t: v == t,
            "ne": lambda v, t: v != t,
        }

        compare = comparison_funcs.get(indicator.comparison, lambda v, t: False)

        if compare(value, indicator.threshold_critical):
            return True, "critical"
        elif compare(value, indicator.threshold_warning):
            return True, "warning"

        return False, None

    # =========================================================================
    # Demand Tracking
    # =========================================================================

    def record_demand(self, demand: IPLDemand) -> IPLDemand:
        """
        Record an IPL demand event.

        Args:
            demand: Demand event to record

        Returns:
            Recorded IPLDemand

        Raises:
            ValueError: If IPL not found
        """
        if demand.ipl_id not in self.ipl_register:
            raise ValueError(f"IPL {demand.ipl_id} not found")

        self.demands[demand.ipl_id].append(demand)

        ipl = self.ipl_register[demand.ipl_id]
        self._log_audit(
            action="IPL_DEMAND_RECORDED",
            entity_type="IPL_DEMAND",
            entity_id=demand.demand_id,
            actor_id=demand.recorded_by,
            furnace_id=ipl.furnace_id,
            changes={
                "ipl_id": demand.ipl_id,
                "demand_type": demand.demand_type.value,
                "success": demand.success,
                "initiating_event": demand.initiating_event,
            }
        )

        logger.info(
            f"IPL demand recorded: {demand.ipl_id} - {demand.demand_type.value} "
            f"({'success' if demand.success else 'FAILURE'})"
        )

        if not demand.success:
            logger.critical(
                f"IPL FAILURE: {demand.ipl_id} failed to respond to {demand.demand_type.value}"
            )

        return demand

    def get_demand_rate(self, ipl_id: str) -> float:
        """
        Calculate demand rate for an IPL.

        Args:
            ipl_id: IPL identifier

        Returns:
            Demands per year

        Raises:
            ValueError: If IPL not found
        """
        if ipl_id not in self.demands:
            raise ValueError(f"IPL {ipl_id} not found")

        demands = self.demands[ipl_id]
        window_start = datetime.now(timezone.utc) - timedelta(
            days=self.config.demand_rate_window_days
        )

        true_demands = [
            d for d in demands
            if d.demand_type == DemandType.TRUE_DEMAND and d.timestamp >= window_start
        ]

        days_in_window = self.config.demand_rate_window_days
        demand_rate = len(true_demands) / days_in_window * 365

        return round(demand_rate, 4)

    # =========================================================================
    # Proof Testing
    # =========================================================================

    def record_proof_test(self, test: ProofTest) -> ProofTest:
        """
        Record a proof test for an IPL.

        Args:
            test: Proof test record

        Returns:
            Recorded ProofTest

        Raises:
            ValueError: If IPL not found
        """
        if test.ipl_id not in self.ipl_register:
            raise ValueError(f"IPL {test.ipl_id} not found")

        self.proof_tests[test.ipl_id].append(test)

        ipl = self.ipl_register[test.ipl_id]
        self._log_audit(
            action="PROOF_TEST_RECORDED",
            entity_type="PROOF_TEST",
            entity_id=test.test_id,
            actor_id=test.tester_id,
            furnace_id=ipl.furnace_id,
            changes={
                "ipl_id": test.ipl_id,
                "test_result": test.test_result,
                "next_test_due": test.next_test_due.isoformat(),
            }
        )

        logger.info(
            f"Proof test recorded for IPL {test.ipl_id}: {test.test_result.upper()}"
        )

        return test

    def get_overdue_tests(self, furnace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of IPLs with overdue proof tests.

        Args:
            furnace_id: Optional filter by furnace

        Returns:
            List of overdue test info
        """
        overdue = []
        threshold = datetime.now(timezone.utc)

        for ipl_id, tests in self.proof_tests.items():
            ipl = self.ipl_register.get(ipl_id)
            if not ipl:
                continue

            if furnace_id and ipl.furnace_id != furnace_id:
                continue

            if tests:
                latest = max(tests, key=lambda t: t.test_date)
                if latest.next_test_due < threshold:
                    days_overdue = (threshold - latest.next_test_due).days
                    overdue.append({
                        "ipl_id": ipl_id,
                        "ipl_name": ipl.name,
                        "last_test_date": latest.test_date.isoformat(),
                        "due_date": latest.next_test_due.isoformat(),
                        "days_overdue": days_overdue,
                        "furnace_id": ipl.furnace_id,
                    })
            else:
                # Never tested
                overdue.append({
                    "ipl_id": ipl_id,
                    "ipl_name": ipl.name,
                    "last_test_date": None,
                    "due_date": None,
                    "days_overdue": None,
                    "never_tested": True,
                    "furnace_id": ipl.furnace_id,
                })

        return overdue

    # =========================================================================
    # Bypass Management
    # =========================================================================

    def create_bypass(self, bypass: BypassRecord, approved_by: str) -> BypassRecord:
        """
        Create a bypass record for an IPL.

        Args:
            bypass: Bypass record
            approved_by: Approver ID

        Returns:
            Created BypassRecord

        Raises:
            ValueError: If IPL not found
        """
        if bypass.ipl_id not in self.ipl_register:
            raise ValueError(f"IPL {bypass.ipl_id} not found")

        bypass.approved_by = approved_by
        self.bypasses[bypass.ipl_id].append(bypass)

        # Update IPL status
        self.ipl_register[bypass.ipl_id].status = IPLStatus.BYPASSED
        self.ipl_register[bypass.ipl_id].last_modified = datetime.now(timezone.utc)

        ipl = self.ipl_register[bypass.ipl_id]
        self._log_audit(
            action="BYPASS_CREATED",
            entity_type="BYPASS",
            entity_id=bypass.bypass_id,
            actor_id=approved_by,
            furnace_id=ipl.furnace_id,
            changes={
                "ipl_id": bypass.ipl_id,
                "reason": bypass.reason,
                "planned_duration_hours": (
                    bypass.planned_end_time - bypass.start_time
                ).total_seconds() / 3600,
                "compensating_measures": bypass.compensating_measures,
            }
        )

        logger.warning(
            f"IPL bypass created: {bypass.ipl_id} - Reason: {bypass.reason}"
        )

        return bypass

    def end_bypass(self, bypass_id: str, ended_by: str) -> BypassRecord:
        """
        End an active bypass.

        Args:
            bypass_id: Bypass record ID
            ended_by: User ending bypass

        Returns:
            Updated BypassRecord

        Raises:
            ValueError: If bypass not found
        """
        for ipl_id, bypasses in self.bypasses.items():
            for bypass in bypasses:
                if bypass.bypass_id == bypass_id:
                    bypass.actual_end_time = datetime.now(timezone.utc)
                    bypass.is_active = False
                    bypass.exceeded_duration = (
                        bypass.actual_end_time > bypass.planned_end_time
                    )

                    # Check if any other active bypasses
                    active_bypasses = [
                        b for b in bypasses if b.is_active and b.bypass_id != bypass_id
                    ]
                    if not active_bypasses:
                        self.ipl_register[ipl_id].status = IPLStatus.ACTIVE
                        self.ipl_register[ipl_id].last_modified = datetime.now(timezone.utc)

                    ipl = self.ipl_register[ipl_id]
                    self._log_audit(
                        action="BYPASS_ENDED",
                        entity_type="BYPASS",
                        entity_id=bypass_id,
                        actor_id=ended_by,
                        furnace_id=ipl.furnace_id,
                        changes={
                            "ipl_id": ipl_id,
                            "actual_duration_hours": (
                                bypass.actual_end_time - bypass.start_time
                            ).total_seconds() / 3600,
                            "exceeded_duration": bypass.exceeded_duration,
                        }
                    )

                    logger.info(f"IPL bypass ended: {bypass_id}")
                    return bypass

        raise ValueError(f"Bypass {bypass_id} not found")

    def get_active_bypasses(
        self,
        furnace_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of active bypasses.

        Args:
            furnace_id: Optional filter by furnace

        Returns:
            List of active bypass info
        """
        active = []
        now = datetime.now(timezone.utc)

        for ipl_id, bypasses in self.bypasses.items():
            ipl = self.ipl_register.get(ipl_id)
            if not ipl:
                continue

            if furnace_id and ipl.furnace_id != furnace_id:
                continue

            for bypass in bypasses:
                if bypass.is_active:
                    duration_hours = (now - bypass.start_time).total_seconds() / 3600
                    exceeded = now > bypass.planned_end_time

                    active.append({
                        "bypass_id": bypass.bypass_id,
                        "ipl_id": ipl_id,
                        "ipl_name": ipl.name,
                        "reason": bypass.reason,
                        "start_time": bypass.start_time.isoformat(),
                        "planned_end_time": bypass.planned_end_time.isoformat(),
                        "duration_hours": round(duration_hours, 2),
                        "exceeded_planned": exceeded,
                        "compensating_measures": bypass.compensating_measures,
                        "furnace_id": ipl.furnace_id,
                    })

        return active

    # =========================================================================
    # HAZOP Recommendation Tracking
    # =========================================================================

    def add_hazop_recommendation(
        self,
        recommendation: HAZOPRecommendation,
        added_by: str
    ) -> HAZOPRecommendation:
        """
        Add a HAZOP recommendation.

        Args:
            recommendation: Recommendation to add
            added_by: User adding recommendation

        Returns:
            Added HAZOPRecommendation
        """
        self.hazop_recommendations[recommendation.recommendation_id] = recommendation

        self._log_audit(
            action="HAZOP_RECOMMENDATION_ADDED",
            entity_type="HAZOP_RECOMMENDATION",
            entity_id=recommendation.recommendation_id,
            actor_id=added_by,
            furnace_id=recommendation.furnace_id,
            changes={
                "hazop_study_id": recommendation.hazop_study_id,
                "deviation": recommendation.deviation,
                "priority": recommendation.priority,
                "due_date": recommendation.due_date.isoformat(),
            }
        )

        logger.info(
            f"HAZOP recommendation added: {recommendation.recommendation_id} "
            f"(Priority: {recommendation.priority})"
        )

        return recommendation

    def update_hazop_recommendation_status(
        self,
        recommendation_id: str,
        new_status: HAZOPRecommendationStatus,
        updated_by: str,
        notes: Optional[str] = None
    ) -> HAZOPRecommendation:
        """
        Update HAZOP recommendation status.

        Args:
            recommendation_id: Recommendation ID
            new_status: New status
            updated_by: User updating
            notes: Optional notes

        Returns:
            Updated HAZOPRecommendation

        Raises:
            ValueError: If recommendation not found
        """
        if recommendation_id not in self.hazop_recommendations:
            raise ValueError(f"HAZOP recommendation {recommendation_id} not found")

        rec = self.hazop_recommendations[recommendation_id]
        old_status = rec.status
        rec.status = new_status

        if new_status == HAZOPRecommendationStatus.IMPLEMENTED:
            rec.completion_date = datetime.now(timezone.utc)
        elif new_status == HAZOPRecommendationStatus.VERIFIED:
            rec.verification_date = datetime.now(timezone.utc)

        self._log_audit(
            action="HAZOP_RECOMMENDATION_STATUS_CHANGED",
            entity_type="HAZOP_RECOMMENDATION",
            entity_id=recommendation_id,
            actor_id=updated_by,
            furnace_id=rec.furnace_id,
            changes={
                "old_status": old_status.value,
                "new_status": new_status.value,
                "notes": notes,
            }
        )

        logger.info(
            f"HAZOP recommendation {recommendation_id} status changed: "
            f"{old_status.value} -> {new_status.value}"
        )

        return rec

    def link_hazop_to_ipl(
        self,
        recommendation_id: str,
        ipl_id: str,
        linked_by: str
    ) -> None:
        """
        Link a HAZOP recommendation to an IPL.

        Args:
            recommendation_id: Recommendation ID
            ipl_id: IPL ID to link
            linked_by: User creating link
        """
        if recommendation_id not in self.hazop_recommendations:
            raise ValueError(f"HAZOP recommendation {recommendation_id} not found")
        if ipl_id not in self.ipl_register:
            raise ValueError(f"IPL {ipl_id} not found")

        rec = self.hazop_recommendations[recommendation_id]
        if ipl_id not in rec.linked_ipl_ids:
            rec.linked_ipl_ids.append(ipl_id)

        ipl = self.ipl_register[ipl_id]
        if recommendation_id not in ipl.hazop_references:
            ipl.hazop_references.append(recommendation_id)

        self._log_audit(
            action="HAZOP_IPL_LINKED",
            entity_type="HAZOP_RECOMMENDATION",
            entity_id=recommendation_id,
            actor_id=linked_by,
            furnace_id=rec.furnace_id,
            changes={"linked_ipl_id": ipl_id}
        )

        logger.info(f"HAZOP recommendation {recommendation_id} linked to IPL {ipl_id}")

    # =========================================================================
    # MOC Management
    # =========================================================================

    def create_moc(self, moc: MOCRecord, created_by: str) -> MOCRecord:
        """
        Create a Management of Change record.

        Args:
            moc: MOC record to create
            created_by: User creating MOC

        Returns:
            Created MOCRecord
        """
        self.moc_records[moc.moc_id] = moc

        self._log_audit(
            action="MOC_CREATED",
            entity_type="MOC",
            entity_id=moc.moc_id,
            actor_id=created_by,
            furnace_id=moc.furnace_id,
            changes={
                "title": moc.title,
                "change_type": moc.change_type,
                "affected_ipls": moc.affected_ipls,
                "pssr_required": moc.pssr_required,
            }
        )

        logger.info(f"MOC created: {moc.moc_id} - {moc.title}")
        return moc

    def update_moc_status(
        self,
        moc_id: str,
        new_status: MOCStatus,
        updated_by: str
    ) -> MOCRecord:
        """
        Update MOC status.

        Args:
            moc_id: MOC ID
            new_status: New status
            updated_by: User updating

        Returns:
            Updated MOCRecord

        Raises:
            ValueError: If MOC not found
        """
        if moc_id not in self.moc_records:
            raise ValueError(f"MOC {moc_id} not found")

        moc = self.moc_records[moc_id]
        old_status = moc.status
        moc.status = new_status

        if new_status == MOCStatus.APPROVED:
            moc.approval_date = datetime.now(timezone.utc)
        elif new_status == MOCStatus.IMPLEMENTED:
            moc.implementation_date = datetime.now(timezone.utc)

        self._log_audit(
            action="MOC_STATUS_CHANGED",
            entity_type="MOC",
            entity_id=moc_id,
            actor_id=updated_by,
            furnace_id=moc.furnace_id,
            changes={
                "old_status": old_status.value,
                "new_status": new_status.value,
            }
        )

        logger.info(f"MOC {moc_id} status changed: {old_status.value} -> {new_status.value}")
        return moc

    def get_mocs_affecting_ipl(self, ipl_id: str) -> List[MOCRecord]:
        """
        Get all MOCs affecting a specific IPL.

        Args:
            ipl_id: IPL identifier

        Returns:
            List of related MOC records
        """
        return [
            moc for moc in self.moc_records.values()
            if ipl_id in moc.affected_ipls
        ]

    # =========================================================================
    # Health Dashboard
    # =========================================================================

    def get_ipl_health_dashboard(self, furnace_id: str) -> IPLHealthDashboard:
        """
        Generate IPL health dashboard data.

        Args:
            furnace_id: Furnace identifier

        Returns:
            IPLHealthDashboard with comprehensive health metrics
        """
        start_time = datetime.now(timezone.utc)

        # Filter IPLs for this furnace
        furnace_ipls = {
            ipl_id: ipl for ipl_id, ipl in self.ipl_register.items()
            if ipl.furnace_id == furnace_id
        }

        # Count by status
        total_ipls = len(furnace_ipls)
        active_ipls = sum(1 for ipl in furnace_ipls.values() if ipl.status == IPLStatus.ACTIVE)
        degraded_ipls = sum(1 for ipl in furnace_ipls.values() if ipl.status == IPLStatus.DEGRADED)
        bypassed_ipls = sum(1 for ipl in furnace_ipls.values() if ipl.status == IPLStatus.BYPASSED)

        # Get overdue tests
        overdue_tests = self.get_overdue_tests(furnace_id)

        # Get active bypasses
        active_bypasses = self.get_active_bypasses(furnace_id)

        # Find exceeded bypasses
        exceeded_bypasses = [b for b in active_bypasses if b.get("exceeded_planned", False)]

        # Get recent demands (last 30 days)
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        recent_demands = []
        for ipl_id, demands in self.demands.items():
            if ipl_id not in furnace_ipls:
                continue
            for demand in demands:
                if demand.timestamp >= thirty_days_ago:
                    recent_demands.append({
                        "demand_id": demand.demand_id,
                        "ipl_id": demand.ipl_id,
                        "ipl_name": furnace_ipls[ipl_id].name,
                        "timestamp": demand.timestamp.isoformat(),
                        "demand_type": demand.demand_type.value,
                        "success": demand.success,
                    })

        # Calculate demand rates
        demand_rates = {}
        for ipl_id in furnace_ipls:
            try:
                demand_rates[ipl_id] = self.get_demand_rate(ipl_id)
            except Exception:
                demand_rates[ipl_id] = 0.0

        # HAZOP recommendations
        furnace_hazop = [
            rec for rec in self.hazop_recommendations.values()
            if rec.furnace_id == furnace_id
        ]
        open_recommendations = sum(
            1 for rec in furnace_hazop
            if rec.status in [HAZOPRecommendationStatus.OPEN, HAZOPRecommendationStatus.IN_PROGRESS]
        )
        overdue_recommendations = sum(
            1 for rec in furnace_hazop
            if rec.status in [HAZOPRecommendationStatus.OPEN, HAZOPRecommendationStatus.IN_PROGRESS]
            and rec.due_date < datetime.now(timezone.utc)
        )

        # Calculate overall health score
        health_score = self._calculate_health_score(
            total_ipls=total_ipls,
            active_ipls=active_ipls,
            degraded_ipls=degraded_ipls,
            bypassed_ipls=bypassed_ipls,
            overdue_tests_count=len(overdue_tests),
            exceeded_bypasses_count=len(exceeded_bypasses),
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(f"IPL health dashboard generated for {furnace_id} in {processing_time:.2f}ms")

        return IPLHealthDashboard(
            furnace_id=furnace_id,
            total_ipls=total_ipls,
            active_ipls=active_ipls,
            degraded_ipls=degraded_ipls,
            bypassed_ipls=bypassed_ipls,
            overdue_tests=overdue_tests,
            active_bypasses=active_bypasses,
            exceeded_bypasses=exceeded_bypasses,
            recent_demands=recent_demands,
            demand_rate_summary=demand_rates,
            hazop_recommendations_open=open_recommendations,
            hazop_recommendations_overdue=overdue_recommendations,
            overall_health_score=health_score,
        )

    def _calculate_health_score(
        self,
        total_ipls: int,
        active_ipls: int,
        degraded_ipls: int,
        bypassed_ipls: int,
        overdue_tests_count: int,
        exceeded_bypasses_count: int
    ) -> float:
        """Calculate overall IPL health score."""
        if total_ipls == 0:
            return 100.0

        # Base score from active ratio
        base_score = (active_ipls / total_ipls) * 100

        # Penalties
        degraded_penalty = (degraded_ipls / total_ipls) * 20
        bypassed_penalty = (bypassed_ipls / total_ipls) * 30
        overdue_penalty = min(overdue_tests_count * 5, 25)
        exceeded_penalty = min(exceeded_bypasses_count * 10, 20)

        score = base_score - degraded_penalty - bypassed_penalty - overdue_penalty - exceeded_penalty
        return max(0.0, min(100.0, round(score, 2)))

    # =========================================================================
    # Incident Investigation Support
    # =========================================================================

    def reconstruct_event_timeline(
        self,
        furnace_id: str,
        start_time: datetime,
        end_time: datetime,
        include_telemetry: bool = True
    ) -> List[EventTimelineEntry]:
        """
        Reconstruct event timeline for incident investigation.

        Args:
            furnace_id: Furnace identifier
            start_time: Timeline start
            end_time: Timeline end
            include_telemetry: Include telemetry changes

        Returns:
            Chronological list of EventTimelineEntry
        """
        timeline: List[EventTimelineEntry] = []

        # Add demand events
        for ipl_id, demands in self.demands.items():
            ipl = self.ipl_register.get(ipl_id)
            if not ipl or ipl.furnace_id != furnace_id:
                continue

            for demand in demands:
                if start_time <= demand.timestamp <= end_time:
                    timeline.append(EventTimelineEntry(
                        timestamp=demand.timestamp,
                        event_type="IPL_DEMAND",
                        description=(
                            f"IPL {ipl.name} received {demand.demand_type.value} - "
                            f"{'SUCCESS' if demand.success else 'FAILURE'}"
                        ),
                        source="LOPA_INTEGRATOR",
                        data={
                            "demand_id": demand.demand_id,
                            "initiating_event": demand.initiating_event,
                            "success": demand.success,
                            "response_time_seconds": demand.response_time_seconds,
                        },
                        ipl_involvement=ipl_id,
                    ))

        # Add bypass events
        for ipl_id, bypasses in self.bypasses.items():
            ipl = self.ipl_register.get(ipl_id)
            if not ipl or ipl.furnace_id != furnace_id:
                continue

            for bypass in bypasses:
                if start_time <= bypass.start_time <= end_time:
                    timeline.append(EventTimelineEntry(
                        timestamp=bypass.start_time,
                        event_type="IPL_BYPASS_START",
                        description=f"IPL {ipl.name} bypass started: {bypass.reason}",
                        source="LOPA_INTEGRATOR",
                        data={
                            "bypass_id": bypass.bypass_id,
                            "reason": bypass.reason,
                            "compensating_measures": bypass.compensating_measures,
                        },
                        ipl_involvement=ipl_id,
                    ))

                if bypass.actual_end_time and start_time <= bypass.actual_end_time <= end_time:
                    timeline.append(EventTimelineEntry(
                        timestamp=bypass.actual_end_time,
                        event_type="IPL_BYPASS_END",
                        description=f"IPL {ipl.name} bypass ended",
                        source="LOPA_INTEGRATOR",
                        data={
                            "bypass_id": bypass.bypass_id,
                            "exceeded_duration": bypass.exceeded_duration,
                        },
                        ipl_involvement=ipl_id,
                    ))

        # Add audit log entries
        for entry in self.audit_log:
            if entry.furnace_id == furnace_id and start_time <= entry.timestamp <= end_time:
                timeline.append(EventTimelineEntry(
                    timestamp=entry.timestamp,
                    event_type=f"AUDIT_{entry.action}",
                    description=f"{entry.action} on {entry.entity_type} {entry.entity_id}",
                    source="AUDIT_LOG",
                    data=entry.changes,
                    ipl_involvement=entry.entity_id if entry.entity_type == "IPL" else None,
                ))

        # Sort by timestamp
        timeline.sort(key=lambda e: e.timestamp)

        logger.info(
            f"Event timeline reconstructed for {furnace_id}: "
            f"{len(timeline)} events from {start_time} to {end_time}"
        )

        return timeline

    def get_ipl_history(
        self,
        ipl_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get complete history for an IPL (for investigation).

        Args:
            ipl_id: IPL identifier
            start_time: Optional start filter
            end_time: Optional end filter

        Returns:
            Dictionary with complete IPL history

        Raises:
            ValueError: If IPL not found
        """
        if ipl_id not in self.ipl_register:
            raise ValueError(f"IPL {ipl_id} not found")

        ipl = self.ipl_register[ipl_id]
        start = start_time or datetime.min.replace(tzinfo=timezone.utc)
        end = end_time or datetime.now(timezone.utc)

        # Filter records by time
        demands = [
            d.dict() for d in self.demands.get(ipl_id, [])
            if start <= d.timestamp <= end
        ]
        tests = [
            t.dict() for t in self.proof_tests.get(ipl_id, [])
            if start <= t.test_date <= end
        ]
        bypasses = [
            b.dict() for b in self.bypasses.get(ipl_id, [])
            if start <= b.start_time <= end
        ]

        # Related MOCs
        mocs = [moc.dict() for moc in self.get_mocs_affecting_ipl(ipl_id)]

        # Related HAZOP recommendations
        hazop_recs = [
            rec.dict() for rec in self.hazop_recommendations.values()
            if ipl_id in rec.linked_ipl_ids
        ]

        # Audit entries
        audit_entries = [
            e.dict() for e in self.audit_log
            if e.entity_id == ipl_id and start <= e.timestamp <= end
        ]

        return {
            "ipl": ipl.dict(),
            "demands": demands,
            "proof_tests": tests,
            "bypasses": bypasses,
            "mocs": mocs,
            "hazop_recommendations": hazop_recs,
            "audit_entries": audit_entries,
            "query_period": {
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        }

    # =========================================================================
    # Audit Logging
    # =========================================================================

    def _log_audit(
        self,
        action: str,
        entity_type: str,
        entity_id: str,
        actor_id: str,
        furnace_id: str,
        changes: Dict[str, Any]
    ) -> None:
        """Add entry to audit log with integrity hash."""
        changes_str = json.dumps(changes, sort_keys=True, default=str)
        data_hash = hashlib.sha256(changes_str.encode()).hexdigest()

        entry = AuditEntry(
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            actor_id=actor_id,
            furnace_id=furnace_id,
            changes=changes,
            data_hash=data_hash,
        )
        self.audit_log.append(entry)

    def get_audit_log(
        self,
        furnace_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditEntry]:
        """
        Retrieve audit log with optional filters.

        Args:
            furnace_id: Optional furnace filter
            entity_type: Optional entity type filter
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of matching AuditEntry records
        """
        filtered = self.audit_log

        if furnace_id:
            filtered = [e for e in filtered if e.furnace_id == furnace_id]
        if entity_type:
            filtered = [e for e in filtered if e.entity_type == entity_type]
        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time]
        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time]

        return filtered
