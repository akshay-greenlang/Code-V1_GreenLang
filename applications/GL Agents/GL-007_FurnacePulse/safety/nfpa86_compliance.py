"""
NFPA86 Compliance Manager - GL-007_FurnacePulse Safety Module

This module implements NFPA86 (Standard for Ovens and Furnaces) compliance
management with configurable checklists, evidence collection, and audit trails.

NFPA86 is the primary standard governing safety requirements for Class A, B, C,
and D ovens and furnaces, including fuel safety systems, purge sequences,
flame safeguards, and interlocks.

Example:
    >>> config = NFPA86Config(furnace_class="B", fuel_type="natural_gas")
    >>> manager = NFPA86ComplianceManager(config)
    >>> evidence = manager.collect_evidence("FRN-001", EvidenceType.PURGE_SEQUENCE)
    >>> package = manager.generate_evidence_package("FRN-001")
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timezone
from uuid import uuid4
import hashlib
import logging
import json

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class FurnaceClass(str, Enum):
    """NFPA86 furnace classifications."""
    CLASS_A = "A"  # Ovens with flammable volatiles
    CLASS_B = "B"  # Ovens with heat-producing equipment
    CLASS_C = "C"  # Furnaces with special atmospheres
    CLASS_D = "D"  # Vacuum furnaces


class FuelType(str, Enum):
    """Supported fuel types for NFPA86 compliance."""
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    FUEL_OIL = "fuel_oil"
    ELECTRIC = "electric"
    HYDROGEN = "hydrogen"


class EvidenceType(str, Enum):
    """Types of evidence collected for NFPA86 compliance."""
    PURGE_SEQUENCE = "purge_sequence"
    STARTUP_SEQUENCE = "startup_sequence"
    FLAME_SAFEGUARD = "flame_safeguard"
    INTERLOCK_HEALTH = "interlock_health"
    FUEL_AIR_RATIO = "fuel_air_ratio"
    ALARM_HISTORY = "alarm_history"
    TRIP_HISTORY = "trip_history"
    PRESSURE_TEST = "pressure_test"
    COMBUSTION_ANALYSIS = "combustion_analysis"


class ComplianceStatus(str, Enum):
    """Compliance status values."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    EVIDENCE_REQUIRED = "evidence_required"
    WAIVER_GRANTED = "waiver_granted"


class ChecklistItemStatus(str, Enum):
    """Status of individual checklist items."""
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"
    REQUIRES_ATTENTION = "requires_attention"


# =============================================================================
# Pydantic Models
# =============================================================================

class NFPA86Config(BaseModel):
    """Configuration for NFPA86 compliance management."""

    furnace_class: FurnaceClass = Field(
        ..., description="NFPA86 furnace classification"
    )
    fuel_type: FuelType = Field(
        ..., description="Primary fuel type"
    )
    purge_volume_multiplier: int = Field(
        default=4, ge=1, le=10,
        description="Required purge volume multiplier (default 4x)"
    )
    flame_failure_response_seconds: float = Field(
        default=4.0, ge=0.5, le=10.0,
        description="Maximum flame failure response time in seconds"
    )
    interlock_test_interval_days: int = Field(
        default=30, ge=1, le=365,
        description="Required interlock testing interval in days"
    )
    evidence_retention_years: int = Field(
        default=7, ge=1, le=25,
        description="Evidence retention period in years"
    )
    custom_checklist_path: Optional[str] = Field(
        None, description="Path to custom checklist YAML file"
    )


class ChecklistItem(BaseModel):
    """Individual NFPA86 compliance checklist item."""

    item_id: str = Field(..., description="Unique checklist item ID")
    nfpa_section: str = Field(..., description="NFPA86 section reference")
    requirement: str = Field(..., description="Requirement description")
    category: str = Field(..., description="Requirement category")
    status: ChecklistItemStatus = Field(
        default=ChecklistItemStatus.PENDING,
        description="Current compliance status"
    )
    evidence_types_required: List[EvidenceType] = Field(
        default_factory=list,
        description="Types of evidence required"
    )
    last_verified: Optional[datetime] = Field(
        None, description="Last verification timestamp"
    )
    verified_by: Optional[str] = Field(None, description="Verifier user ID")
    notes: Optional[str] = Field(None, description="Additional notes")


class EvidenceRecord(BaseModel):
    """Evidence record for NFPA86 compliance."""

    evidence_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique evidence ID"
    )
    furnace_id: str = Field(..., description="Associated furnace ID")
    evidence_type: EvidenceType = Field(..., description="Type of evidence")
    collected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Immutable collection timestamp"
    )
    collected_by: str = Field(..., description="Collector user ID or system ID")
    data: Dict[str, Any] = Field(..., description="Evidence data payload")
    data_hash: str = Field(..., description="SHA-256 hash of evidence data")
    source_system: str = Field(..., description="Source system identifier")
    checklist_items: List[str] = Field(
        default_factory=list,
        description="Related checklist item IDs"
    )
    attachments: List[str] = Field(
        default_factory=list,
        description="Attachment file references"
    )
    is_validated: bool = Field(default=False, description="Validation status")
    validation_notes: Optional[str] = Field(None, description="Validation notes")

    @validator('data_hash', pre=True, always=True)
    def compute_hash(cls, v, values):
        """Compute SHA-256 hash of evidence data if not provided."""
        if v:
            return v
        if 'data' in values:
            data_str = json.dumps(values['data'], sort_keys=True, default=str)
            return hashlib.sha256(data_str.encode()).hexdigest()
        return ""


class EvidencePackage(BaseModel):
    """Immutable evidence package for audits."""

    package_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique package ID"
    )
    furnace_id: str = Field(..., description="Furnace ID")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Package generation timestamp"
    )
    generated_by: str = Field(..., description="Generator user ID")
    compliance_period_start: datetime = Field(
        ..., description="Compliance period start"
    )
    compliance_period_end: datetime = Field(
        ..., description="Compliance period end"
    )
    checklist_items: List[ChecklistItem] = Field(
        ..., description="Checklist items with status"
    )
    evidence_records: List[EvidenceRecord] = Field(
        ..., description="Collected evidence records"
    )
    overall_status: ComplianceStatus = Field(
        ..., description="Overall compliance status"
    )
    package_hash: str = Field(..., description="SHA-256 hash of entire package")
    retention_until: datetime = Field(..., description="Retention expiry date")


class ComplianceDashboardData(BaseModel):
    """Dashboard data for compliance status display."""

    furnace_id: str = Field(..., description="Furnace ID")
    last_updated: datetime = Field(..., description="Last update timestamp")
    overall_status: ComplianceStatus = Field(..., description="Overall status")
    checklist_summary: Dict[str, int] = Field(
        ..., description="Count by checklist status"
    )
    evidence_summary: Dict[str, int] = Field(
        ..., description="Count by evidence type"
    )
    pending_items: List[Dict[str, str]] = Field(
        ..., description="Items requiring attention"
    )
    upcoming_deadlines: List[Dict[str, Any]] = Field(
        ..., description="Upcoming compliance deadlines"
    )
    compliance_score: float = Field(
        ..., ge=0.0, le=100.0,
        description="Compliance score percentage"
    )


class AuditLogEntry(BaseModel):
    """Audit log entry for compliance activities."""

    log_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique log ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    action: str = Field(..., description="Action performed")
    actor_id: str = Field(..., description="Actor user or system ID")
    furnace_id: str = Field(..., description="Associated furnace ID")
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action details"
    )
    ip_address: Optional[str] = Field(None, description="Source IP address")
    session_id: Optional[str] = Field(None, description="Session identifier")


# =============================================================================
# Default Checklist Definition
# =============================================================================

DEFAULT_NFPA86_CHECKLIST: List[Dict[str, Any]] = [
    {
        "item_id": "NFPA86-001",
        "nfpa_section": "8.2.1",
        "requirement": "Pre-purge cycle completes minimum 4 volume changes",
        "category": "Purge Sequence",
        "evidence_types_required": [EvidenceType.PURGE_SEQUENCE],
    },
    {
        "item_id": "NFPA86-002",
        "nfpa_section": "8.2.2",
        "requirement": "Purge airflow verified by flow switch before ignition",
        "category": "Purge Sequence",
        "evidence_types_required": [EvidenceType.PURGE_SEQUENCE],
    },
    {
        "item_id": "NFPA86-003",
        "nfpa_section": "8.3.1",
        "requirement": "Flame safeguard system responds within 4 seconds",
        "category": "Flame Safeguard",
        "evidence_types_required": [EvidenceType.FLAME_SAFEGUARD],
    },
    {
        "item_id": "NFPA86-004",
        "nfpa_section": "8.3.2",
        "requirement": "Flame detection sensors tested per schedule",
        "category": "Flame Safeguard",
        "evidence_types_required": [EvidenceType.FLAME_SAFEGUARD],
    },
    {
        "item_id": "NFPA86-005",
        "nfpa_section": "8.4.1",
        "requirement": "Safety shutoff valves tested monthly",
        "category": "Interlock Health",
        "evidence_types_required": [EvidenceType.INTERLOCK_HEALTH],
    },
    {
        "item_id": "NFPA86-006",
        "nfpa_section": "8.4.2",
        "requirement": "High/low gas pressure interlocks functional",
        "category": "Interlock Health",
        "evidence_types_required": [EvidenceType.INTERLOCK_HEALTH, EvidenceType.PRESSURE_TEST],
    },
    {
        "item_id": "NFPA86-007",
        "nfpa_section": "8.5.1",
        "requirement": "Fuel/air ratio maintained within design limits",
        "category": "Combustion Control",
        "evidence_types_required": [EvidenceType.FUEL_AIR_RATIO, EvidenceType.COMBUSTION_ANALYSIS],
    },
    {
        "item_id": "NFPA86-008",
        "nfpa_section": "8.6.1",
        "requirement": "All alarms logged and acknowledged",
        "category": "Alarm Management",
        "evidence_types_required": [EvidenceType.ALARM_HISTORY],
    },
    {
        "item_id": "NFPA86-009",
        "nfpa_section": "8.6.2",
        "requirement": "Safety trips investigated and documented",
        "category": "Trip Management",
        "evidence_types_required": [EvidenceType.TRIP_HISTORY],
    },
    {
        "item_id": "NFPA86-010",
        "nfpa_section": "8.7.1",
        "requirement": "Startup sequence follows approved procedure",
        "category": "Startup Sequence",
        "evidence_types_required": [EvidenceType.STARTUP_SEQUENCE],
    },
]


# =============================================================================
# NFPA86 Compliance Manager
# =============================================================================

class NFPA86ComplianceManager:
    """
    NFPA86 Compliance Manager for industrial furnace safety.

    This manager handles NFPA86 compliance checklists, evidence collection,
    package generation, and audit trail maintenance. It provides a configurable
    framework for tracking compliance with NFPA86 requirements.

    Attributes:
        config: NFPA86 configuration settings
        checklists: Mapping of furnace IDs to their checklists
        evidence_store: Collected evidence records
        audit_log: Audit trail entries

    Example:
        >>> config = NFPA86Config(furnace_class=FurnaceClass.CLASS_B, fuel_type=FuelType.NATURAL_GAS)
        >>> manager = NFPA86ComplianceManager(config)
        >>> manager.initialize_checklist("FRN-001")
        >>> evidence = manager.collect_evidence(
        ...     furnace_id="FRN-001",
        ...     evidence_type=EvidenceType.PURGE_SEQUENCE,
        ...     data={"purge_cycles": 4, "airflow_cfm": 5000},
        ...     collector_id="SYS-PLC-001"
        ... )
    """

    def __init__(self, config: NFPA86Config):
        """
        Initialize NFPA86ComplianceManager.

        Args:
            config: NFPA86 configuration settings
        """
        self.config = config
        self.checklists: Dict[str, List[ChecklistItem]] = {}
        self.evidence_store: Dict[str, List[EvidenceRecord]] = {}
        self.audit_log: List[AuditLogEntry] = []

        logger.info(
            f"NFPA86ComplianceManager initialized for {config.furnace_class.value} "
            f"furnace with {config.fuel_type.value} fuel"
        )

    def initialize_checklist(
        self,
        furnace_id: str,
        custom_items: Optional[List[Dict[str, Any]]] = None
    ) -> List[ChecklistItem]:
        """
        Initialize compliance checklist for a furnace.

        Args:
            furnace_id: Unique furnace identifier
            custom_items: Optional custom checklist items to add

        Returns:
            List of initialized ChecklistItem objects
        """
        start_time = datetime.now(timezone.utc)

        # Build checklist from defaults
        checklist_items = []
        for item_dict in DEFAULT_NFPA86_CHECKLIST:
            item = ChecklistItem(
                item_id=item_dict["item_id"],
                nfpa_section=item_dict["nfpa_section"],
                requirement=item_dict["requirement"],
                category=item_dict["category"],
                evidence_types_required=item_dict["evidence_types_required"],
            )
            checklist_items.append(item)

        # Add custom items if provided
        if custom_items:
            for custom_dict in custom_items:
                custom_item = ChecklistItem(**custom_dict)
                checklist_items.append(custom_item)

        self.checklists[furnace_id] = checklist_items
        self.evidence_store[furnace_id] = []

        # Log audit entry
        self._log_audit(
            action="CHECKLIST_INITIALIZED",
            actor_id="SYSTEM",
            furnace_id=furnace_id,
            details={
                "item_count": len(checklist_items),
                "custom_item_count": len(custom_items) if custom_items else 0,
            }
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            f"Checklist initialized for {furnace_id} with {len(checklist_items)} items "
            f"in {processing_time:.2f}ms"
        )

        return checklist_items

    def collect_evidence(
        self,
        furnace_id: str,
        evidence_type: EvidenceType,
        data: Dict[str, Any],
        collector_id: str,
        source_system: str = "SCADA",
        attachments: Optional[List[str]] = None
    ) -> EvidenceRecord:
        """
        Collect evidence for NFPA86 compliance.

        Args:
            furnace_id: Furnace identifier
            evidence_type: Type of evidence being collected
            data: Evidence data payload
            collector_id: User or system ID collecting evidence
            source_system: Source system identifier
            attachments: Optional list of attachment file references

        Returns:
            Created EvidenceRecord

        Raises:
            ValueError: If furnace not initialized
        """
        if furnace_id not in self.checklists:
            raise ValueError(f"Furnace {furnace_id} not initialized. Call initialize_checklist first.")

        # Compute data hash for integrity
        data_str = json.dumps(data, sort_keys=True, default=str)
        data_hash = hashlib.sha256(data_str.encode()).hexdigest()

        # Find related checklist items
        related_items = [
            item.item_id
            for item in self.checklists[furnace_id]
            if evidence_type in item.evidence_types_required
        ]

        # Create evidence record
        evidence = EvidenceRecord(
            furnace_id=furnace_id,
            evidence_type=evidence_type,
            collected_by=collector_id,
            data=data,
            data_hash=data_hash,
            source_system=source_system,
            checklist_items=related_items,
            attachments=attachments or [],
        )

        self.evidence_store[furnace_id].append(evidence)

        # Log audit entry
        self._log_audit(
            action="EVIDENCE_COLLECTED",
            actor_id=collector_id,
            furnace_id=furnace_id,
            details={
                "evidence_id": evidence.evidence_id,
                "evidence_type": evidence_type.value,
                "data_hash": data_hash,
                "source_system": source_system,
                "related_checklist_items": related_items,
            }
        )

        logger.info(
            f"Evidence collected: {evidence_type.value} for {furnace_id} "
            f"(hash: {data_hash[:16]}...)"
        )

        return evidence

    def collect_purge_sequence_evidence(
        self,
        furnace_id: str,
        purge_cycles: int,
        airflow_cfm: float,
        purge_duration_seconds: float,
        flow_switch_verified: bool,
        collector_id: str
    ) -> EvidenceRecord:
        """
        Collect purge sequence evidence with specific validation.

        Args:
            furnace_id: Furnace identifier
            purge_cycles: Number of volume changes completed
            airflow_cfm: Airflow rate in CFM
            purge_duration_seconds: Total purge duration
            flow_switch_verified: Flow switch verification status
            collector_id: Collector ID

        Returns:
            Created EvidenceRecord
        """
        data = {
            "purge_cycles": purge_cycles,
            "airflow_cfm": airflow_cfm,
            "purge_duration_seconds": purge_duration_seconds,
            "flow_switch_verified": flow_switch_verified,
            "minimum_required_cycles": self.config.purge_volume_multiplier,
            "compliant": purge_cycles >= self.config.purge_volume_multiplier and flow_switch_verified,
        }

        return self.collect_evidence(
            furnace_id=furnace_id,
            evidence_type=EvidenceType.PURGE_SEQUENCE,
            data=data,
            collector_id=collector_id,
            source_system="BMS",
        )

    def collect_flame_safeguard_evidence(
        self,
        furnace_id: str,
        response_time_seconds: float,
        sensor_test_passed: bool,
        sensor_serial: str,
        test_date: datetime,
        collector_id: str
    ) -> EvidenceRecord:
        """
        Collect flame safeguard evidence.

        Args:
            furnace_id: Furnace identifier
            response_time_seconds: Flame failure response time
            sensor_test_passed: Sensor test result
            sensor_serial: Flame sensor serial number
            test_date: Date of sensor test
            collector_id: Collector ID

        Returns:
            Created EvidenceRecord
        """
        data = {
            "response_time_seconds": response_time_seconds,
            "max_allowed_seconds": self.config.flame_failure_response_seconds,
            "sensor_test_passed": sensor_test_passed,
            "sensor_serial": sensor_serial,
            "test_date": test_date.isoformat(),
            "compliant": (
                response_time_seconds <= self.config.flame_failure_response_seconds
                and sensor_test_passed
            ),
        }

        return self.collect_evidence(
            furnace_id=furnace_id,
            evidence_type=EvidenceType.FLAME_SAFEGUARD,
            data=data,
            collector_id=collector_id,
            source_system="FSG",
        )

    def collect_interlock_health_evidence(
        self,
        furnace_id: str,
        interlock_id: str,
        test_result: bool,
        test_date: datetime,
        next_test_due: datetime,
        tester_id: str
    ) -> EvidenceRecord:
        """
        Collect interlock health test evidence.

        Args:
            furnace_id: Furnace identifier
            interlock_id: Interlock identifier
            test_result: Test pass/fail result
            test_date: Date of test
            next_test_due: Next scheduled test date
            tester_id: Person who performed test

        Returns:
            Created EvidenceRecord
        """
        data = {
            "interlock_id": interlock_id,
            "test_result": "PASS" if test_result else "FAIL",
            "test_date": test_date.isoformat(),
            "next_test_due": next_test_due.isoformat(),
            "test_interval_days": self.config.interlock_test_interval_days,
            "compliant": test_result,
        }

        return self.collect_evidence(
            furnace_id=furnace_id,
            evidence_type=EvidenceType.INTERLOCK_HEALTH,
            data=data,
            collector_id=tester_id,
            source_system="CMMS",
        )

    def collect_fuel_air_ratio_evidence(
        self,
        furnace_id: str,
        fuel_flow_scfh: float,
        air_flow_cfm: float,
        measured_ratio: float,
        design_ratio_min: float,
        design_ratio_max: float,
        o2_percentage: float,
        collector_id: str
    ) -> EvidenceRecord:
        """
        Collect fuel/air ratio evidence from combustion analysis.

        Args:
            furnace_id: Furnace identifier
            fuel_flow_scfh: Fuel flow in SCFH
            air_flow_cfm: Combustion air flow in CFM
            measured_ratio: Measured fuel/air ratio
            design_ratio_min: Minimum design ratio
            design_ratio_max: Maximum design ratio
            o2_percentage: Stack O2 percentage
            collector_id: Collector ID

        Returns:
            Created EvidenceRecord
        """
        data = {
            "fuel_flow_scfh": fuel_flow_scfh,
            "air_flow_cfm": air_flow_cfm,
            "measured_ratio": measured_ratio,
            "design_ratio_min": design_ratio_min,
            "design_ratio_max": design_ratio_max,
            "o2_percentage": o2_percentage,
            "within_limits": design_ratio_min <= measured_ratio <= design_ratio_max,
            "compliant": design_ratio_min <= measured_ratio <= design_ratio_max,
        }

        return self.collect_evidence(
            furnace_id=furnace_id,
            evidence_type=EvidenceType.FUEL_AIR_RATIO,
            data=data,
            collector_id=collector_id,
            source_system="COMBUSTION_ANALYZER",
        )

    def collect_alarm_history_evidence(
        self,
        furnace_id: str,
        alarm_records: List[Dict[str, Any]],
        period_start: datetime,
        period_end: datetime,
        collector_id: str
    ) -> EvidenceRecord:
        """
        Collect alarm history evidence.

        Args:
            furnace_id: Furnace identifier
            alarm_records: List of alarm records
            period_start: Period start datetime
            period_end: Period end datetime
            collector_id: Collector ID

        Returns:
            Created EvidenceRecord
        """
        # Analyze alarm data
        total_alarms = len(alarm_records)
        unacknowledged = sum(1 for a in alarm_records if not a.get("acknowledged", False))
        safety_alarms = sum(1 for a in alarm_records if a.get("priority") == "safety")

        data = {
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "total_alarms": total_alarms,
            "unacknowledged_count": unacknowledged,
            "safety_alarm_count": safety_alarms,
            "alarm_records": alarm_records,
            "all_acknowledged": unacknowledged == 0,
            "compliant": unacknowledged == 0,
        }

        return self.collect_evidence(
            furnace_id=furnace_id,
            evidence_type=EvidenceType.ALARM_HISTORY,
            data=data,
            collector_id=collector_id,
            source_system="HISTORIAN",
        )

    def collect_trip_history_evidence(
        self,
        furnace_id: str,
        trip_records: List[Dict[str, Any]],
        period_start: datetime,
        period_end: datetime,
        collector_id: str
    ) -> EvidenceRecord:
        """
        Collect trip history evidence.

        Args:
            furnace_id: Furnace identifier
            trip_records: List of trip records
            period_start: Period start datetime
            period_end: Period end datetime
            collector_id: Collector ID

        Returns:
            Created EvidenceRecord
        """
        total_trips = len(trip_records)
        investigated = sum(1 for t in trip_records if t.get("investigated", False))
        pending_investigation = total_trips - investigated

        data = {
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "total_trips": total_trips,
            "investigated_count": investigated,
            "pending_investigation_count": pending_investigation,
            "trip_records": trip_records,
            "all_investigated": pending_investigation == 0,
            "compliant": pending_investigation == 0,
        }

        return self.collect_evidence(
            furnace_id=furnace_id,
            evidence_type=EvidenceType.TRIP_HISTORY,
            data=data,
            collector_id=collector_id,
            source_system="INCIDENT_DB",
        )

    def update_checklist_item(
        self,
        furnace_id: str,
        item_id: str,
        status: ChecklistItemStatus,
        verified_by: str,
        notes: Optional[str] = None
    ) -> ChecklistItem:
        """
        Update checklist item status.

        Args:
            furnace_id: Furnace identifier
            item_id: Checklist item ID
            status: New status
            verified_by: Verifier user ID
            notes: Optional notes

        Returns:
            Updated ChecklistItem

        Raises:
            ValueError: If furnace or item not found
        """
        if furnace_id not in self.checklists:
            raise ValueError(f"Furnace {furnace_id} not found")

        for item in self.checklists[furnace_id]:
            if item.item_id == item_id:
                old_status = item.status
                item.status = status
                item.verified_by = verified_by
                item.last_verified = datetime.now(timezone.utc)
                if notes:
                    item.notes = notes

                self._log_audit(
                    action="CHECKLIST_ITEM_UPDATED",
                    actor_id=verified_by,
                    furnace_id=furnace_id,
                    details={
                        "item_id": item_id,
                        "old_status": old_status.value,
                        "new_status": status.value,
                        "notes": notes,
                    }
                )

                logger.info(
                    f"Checklist item {item_id} updated: {old_status.value} -> {status.value}"
                )
                return item

        raise ValueError(f"Checklist item {item_id} not found for furnace {furnace_id}")

    def generate_evidence_package(
        self,
        furnace_id: str,
        period_start: datetime,
        period_end: datetime,
        generator_id: str
    ) -> EvidencePackage:
        """
        Generate immutable evidence package for audit.

        Args:
            furnace_id: Furnace identifier
            period_start: Compliance period start
            period_end: Compliance period end
            generator_id: User generating package

        Returns:
            EvidencePackage with SHA-256 integrity hash

        Raises:
            ValueError: If furnace not found
        """
        if furnace_id not in self.checklists:
            raise ValueError(f"Furnace {furnace_id} not found")

        start_time = datetime.now(timezone.utc)

        # Get checklist and evidence
        checklist = self.checklists[furnace_id]
        evidence = [
            e for e in self.evidence_store.get(furnace_id, [])
            if period_start <= e.collected_at <= period_end
        ]

        # Determine overall compliance status
        overall_status = self._calculate_overall_status(checklist)

        # Calculate retention date
        retention_until = datetime.now(timezone.utc).replace(
            year=datetime.now(timezone.utc).year + self.config.evidence_retention_years
        )

        # Create package
        package = EvidencePackage(
            furnace_id=furnace_id,
            generated_by=generator_id,
            compliance_period_start=period_start,
            compliance_period_end=period_end,
            checklist_items=checklist,
            evidence_records=evidence,
            overall_status=overall_status,
            package_hash="",  # Will be computed
            retention_until=retention_until,
        )

        # Compute package hash (excluding the hash field itself)
        package_dict = package.dict(exclude={"package_hash"})
        package_str = json.dumps(package_dict, sort_keys=True, default=str)
        package.package_hash = hashlib.sha256(package_str.encode()).hexdigest()

        # Log audit entry
        self._log_audit(
            action="EVIDENCE_PACKAGE_GENERATED",
            actor_id=generator_id,
            furnace_id=furnace_id,
            details={
                "package_id": package.package_id,
                "package_hash": package.package_hash,
                "evidence_count": len(evidence),
                "checklist_items": len(checklist),
                "overall_status": overall_status.value,
                "retention_until": retention_until.isoformat(),
            }
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            f"Evidence package generated for {furnace_id}: {package.package_id} "
            f"(hash: {package.package_hash[:16]}...) in {processing_time:.2f}ms"
        )

        return package

    def get_compliance_dashboard_data(self, furnace_id: str) -> ComplianceDashboardData:
        """
        Get compliance status dashboard data.

        Args:
            furnace_id: Furnace identifier

        Returns:
            ComplianceDashboardData for dashboard display

        Raises:
            ValueError: If furnace not found
        """
        if furnace_id not in self.checklists:
            raise ValueError(f"Furnace {furnace_id} not found")

        checklist = self.checklists[furnace_id]
        evidence = self.evidence_store.get(furnace_id, [])

        # Checklist summary
        checklist_summary = {}
        for item in checklist:
            status_key = item.status.value
            checklist_summary[status_key] = checklist_summary.get(status_key, 0) + 1

        # Evidence summary
        evidence_summary = {}
        for e in evidence:
            type_key = e.evidence_type.value
            evidence_summary[type_key] = evidence_summary.get(type_key, 0) + 1

        # Pending items
        pending_items = [
            {"item_id": item.item_id, "requirement": item.requirement}
            for item in checklist
            if item.status in [ChecklistItemStatus.PENDING, ChecklistItemStatus.REQUIRES_ATTENTION]
        ]

        # Upcoming deadlines (from evidence requiring renewal)
        upcoming_deadlines = []
        for e in evidence:
            if e.evidence_type == EvidenceType.INTERLOCK_HEALTH:
                next_due = e.data.get("next_test_due")
                if next_due:
                    upcoming_deadlines.append({
                        "type": "Interlock Test",
                        "due_date": next_due,
                        "interlock_id": e.data.get("interlock_id"),
                    })

        # Calculate compliance score
        total_items = len(checklist)
        passed_items = sum(1 for item in checklist if item.status == ChecklistItemStatus.PASS)
        compliance_score = (passed_items / total_items * 100) if total_items > 0 else 0.0

        return ComplianceDashboardData(
            furnace_id=furnace_id,
            last_updated=datetime.now(timezone.utc),
            overall_status=self._calculate_overall_status(checklist),
            checklist_summary=checklist_summary,
            evidence_summary=evidence_summary,
            pending_items=pending_items,
            upcoming_deadlines=upcoming_deadlines,
            compliance_score=round(compliance_score, 2),
        )

    def get_audit_log(
        self,
        furnace_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        action_filter: Optional[str] = None
    ) -> List[AuditLogEntry]:
        """
        Retrieve audit log entries with optional filters.

        Args:
            furnace_id: Optional filter by furnace
            start_date: Optional start date filter
            end_date: Optional end date filter
            action_filter: Optional action type filter

        Returns:
            List of matching AuditLogEntry records
        """
        filtered = self.audit_log

        if furnace_id:
            filtered = [e for e in filtered if e.furnace_id == furnace_id]
        if start_date:
            filtered = [e for e in filtered if e.timestamp >= start_date]
        if end_date:
            filtered = [e for e in filtered if e.timestamp <= end_date]
        if action_filter:
            filtered = [e for e in filtered if e.action == action_filter]

        return filtered

    def validate_evidence_integrity(self, evidence_id: str, furnace_id: str) -> bool:
        """
        Validate evidence record integrity by recomputing hash.

        Args:
            evidence_id: Evidence record ID
            furnace_id: Furnace identifier

        Returns:
            True if integrity verified, False if tampered
        """
        evidence_list = self.evidence_store.get(furnace_id, [])
        for evidence in evidence_list:
            if evidence.evidence_id == evidence_id:
                # Recompute hash
                data_str = json.dumps(evidence.data, sort_keys=True, default=str)
                computed_hash = hashlib.sha256(data_str.encode()).hexdigest()

                is_valid = computed_hash == evidence.data_hash

                self._log_audit(
                    action="EVIDENCE_INTEGRITY_CHECK",
                    actor_id="SYSTEM",
                    furnace_id=furnace_id,
                    details={
                        "evidence_id": evidence_id,
                        "stored_hash": evidence.data_hash,
                        "computed_hash": computed_hash,
                        "integrity_valid": is_valid,
                    }
                )

                if not is_valid:
                    logger.warning(
                        f"Evidence integrity check FAILED for {evidence_id}: "
                        f"stored={evidence.data_hash[:16]}... computed={computed_hash[:16]}..."
                    )
                else:
                    logger.info(f"Evidence integrity verified for {evidence_id}")

                return is_valid

        raise ValueError(f"Evidence {evidence_id} not found for furnace {furnace_id}")

    def _calculate_overall_status(self, checklist: List[ChecklistItem]) -> ComplianceStatus:
        """Calculate overall compliance status from checklist items."""
        statuses = [item.status for item in checklist]

        if any(s == ChecklistItemStatus.FAIL for s in statuses):
            return ComplianceStatus.NON_COMPLIANT
        if any(s == ChecklistItemStatus.PENDING for s in statuses):
            return ComplianceStatus.PENDING_REVIEW
        if any(s == ChecklistItemStatus.REQUIRES_ATTENTION for s in statuses):
            return ComplianceStatus.EVIDENCE_REQUIRED
        if all(s in [ChecklistItemStatus.PASS, ChecklistItemStatus.NOT_APPLICABLE] for s in statuses):
            return ComplianceStatus.COMPLIANT

        return ComplianceStatus.PENDING_REVIEW

    def _log_audit(
        self,
        action: str,
        actor_id: str,
        furnace_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Add entry to audit log."""
        entry = AuditLogEntry(
            action=action,
            actor_id=actor_id,
            furnace_id=furnace_id,
            details=details,
        )
        self.audit_log.append(entry)
