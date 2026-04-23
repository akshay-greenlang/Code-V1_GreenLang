"""
GL-012 STEAMQUAL - GL-003 UNIFIEDSTEAM Interface

Interface module for exporting steam quality data to GL-003 UNIFIEDSTEAM:
- QualityConstraints (x_min, DeltaT_min, max_drain_duty_cycle, max_ramp_rate)
- QualityState (x_est, R_carry, eta_sep, data_quality_flags, uncertainty)
- QualityEvents (event_type, severity, timestamps, evidence, playbook)
- QualityCosts (energy penalty, CO2e impact)

Playbook Requirements:
- Time synchronization between GL-012 and GL-003
- Unit normalization (SI units)
- Data quality flag propagation
- Provenance tracking with SHA-256 hashing
- Event correlation for root cause analysis
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, IntFlag
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import asyncio
import logging
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class QualityEventType(Enum):
    """Types of steam quality events."""
    # Quality excursions
    LOW_DRYNESS = "low_dryness"
    LOW_SUPERHEAT = "low_superheat"
    HIGH_CARRYOVER = "high_carryover"
    WET_STEAM_DETECTED = "wet_steam_detected"

    # Separator events
    SEPARATOR_FOULING = "separator_fouling"
    SEPARATOR_EFFICIENCY_LOW = "separator_efficiency_low"
    SEPARATOR_DP_HIGH = "separator_dp_high"
    SEPARATOR_DP_LOW = "separator_dp_low"

    # Drain system events
    DRAIN_DUTY_HIGH = "drain_duty_high"
    DRAIN_VALVE_STUCK = "drain_valve_stuck"
    DRAIN_CYCLING_EXCESSIVE = "drain_cycling_excessive"

    # Control events
    RAMP_RATE_EXCEEDED = "ramp_rate_exceeded"
    CONSTRAINT_VIOLATED = "constraint_violated"
    SETPOINT_CHANGE = "setpoint_change"

    # Data quality events
    DATA_QUALITY_DEGRADED = "data_quality_degraded"
    SENSOR_FAULT = "sensor_fault"
    COMMUNICATION_LOSS = "communication_loss"

    # Advisory events
    QUALITY_ADVISORY = "quality_advisory"
    MAINTENANCE_RECOMMENDED = "maintenance_recommended"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"


class EventSeverity(Enum):
    """Event severity levels per ISA-18.2."""
    DIAGNOSTIC = 0  # Information only
    LOW = 1  # Minor deviation
    MEDIUM = 2  # Action recommended
    HIGH = 3  # Immediate action required
    CRITICAL = 4  # Safety or major loss


class DataQualityFlag(IntFlag):
    """Data quality flags for GL-003 export."""
    GOOD = 0x00
    UNCERTAIN = 0x01
    BAD = 0x02
    STALE = 0x04
    ESTIMATED = 0x08
    MANUAL = 0x10
    COMMUNICATION_ERROR = 0x20
    SENSOR_FAULT = 0x40
    OUT_OF_RANGE = 0x80

    def is_usable(self) -> bool:
        """Check if data is usable for optimization."""
        return not (self & (DataQualityFlag.BAD | DataQualityFlag.SENSOR_FAULT))


@dataclass
class QualityConstraints:
    """
    Quality constraints exported to GL-003 UNIFIEDSTEAM.

    These constraints define operational limits for steam quality control
    that GL-003 must respect in its optimization decisions.

    Attributes:
        x_min: Minimum allowable steam dryness fraction (0.95-1.0)
        delta_t_min: Minimum superheat in degrees Celsius
        max_drain_duty_cycle: Maximum drain valve duty cycle (0.0-0.5)
        max_ramp_rate: Maximum quality change rate per second
        effective_from: Constraint validity start time
        effective_until: Constraint validity end time
        source_agent: Originating agent ID
        provenance_hash: SHA-256 hash for audit trail
    """
    x_min: float  # Minimum dryness fraction (e.g., 0.97)
    delta_t_min: float  # Minimum superheat, degC (e.g., 5.0)
    max_drain_duty_cycle: float  # Maximum drain duty (e.g., 0.25)
    max_ramp_rate: float  # Maximum ramp rate, fraction/s (e.g., 0.01)

    # Validity window
    effective_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    effective_until: Optional[datetime] = None

    # Provenance
    source_agent: str = "GL-012_STEAMQUAL"
    provenance_hash: str = ""

    # Additional constraints
    min_separator_efficiency: float = 0.95
    max_carryover_rate: float = 0.02
    min_data_quality_score: float = 0.90

    def __post_init__(self):
        """Calculate provenance hash."""
        if not self.provenance_hash:
            hash_data = (
                f"{self.x_min}:{self.delta_t_min}:{self.max_drain_duty_cycle}:"
                f"{self.max_ramp_rate}:{self.effective_from.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(hash_data.encode()).hexdigest()

    def is_valid(self) -> bool:
        """Check if constraints are currently valid."""
        now = datetime.now(timezone.utc)
        if now < self.effective_from:
            return False
        if self.effective_until and now > self.effective_until:
            return False
        return True

    def validate_values(self) -> List[str]:
        """Validate constraint values are within acceptable ranges."""
        errors = []

        if not (0.90 <= self.x_min <= 1.0):
            errors.append(f"x_min {self.x_min} outside range [0.90, 1.0]")

        if not (0.0 <= self.delta_t_min <= 50.0):
            errors.append(f"delta_t_min {self.delta_t_min} outside range [0, 50]")

        if not (0.0 <= self.max_drain_duty_cycle <= 0.5):
            errors.append(f"max_drain_duty_cycle {self.max_drain_duty_cycle} outside range [0, 0.5]")

        if not (0.001 <= self.max_ramp_rate <= 0.1):
            errors.append(f"max_ramp_rate {self.max_ramp_rate} outside range [0.001, 0.1]")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "x_min": self.x_min,
            "delta_t_min": self.delta_t_min,
            "max_drain_duty_cycle": self.max_drain_duty_cycle,
            "max_ramp_rate": self.max_ramp_rate,
            "effective_from": self.effective_from.isoformat(),
            "effective_until": self.effective_until.isoformat() if self.effective_until else None,
            "source_agent": self.source_agent,
            "provenance_hash": self.provenance_hash,
            "min_separator_efficiency": self.min_separator_efficiency,
            "max_carryover_rate": self.max_carryover_rate,
            "min_data_quality_score": self.min_data_quality_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityConstraints":
        """Create from dictionary."""
        return cls(
            x_min=data["x_min"],
            delta_t_min=data["delta_t_min"],
            max_drain_duty_cycle=data["max_drain_duty_cycle"],
            max_ramp_rate=data["max_ramp_rate"],
            effective_from=datetime.fromisoformat(data["effective_from"]),
            effective_until=datetime.fromisoformat(data["effective_until"]) if data.get("effective_until") else None,
            source_agent=data.get("source_agent", "GL-012_STEAMQUAL"),
            min_separator_efficiency=data.get("min_separator_efficiency", 0.95),
            max_carryover_rate=data.get("max_carryover_rate", 0.02),
            min_data_quality_score=data.get("min_data_quality_score", 0.90),
        )


@dataclass
class QualityState:
    """
    Current steam quality state exported to GL-003 UNIFIEDSTEAM.

    Provides real-time steam quality measurements and estimates
    for GL-003 optimization decisions.

    Attributes:
        x_est: Estimated steam dryness fraction (0.0-1.0)
        r_carry: Moisture carryover rate (fraction)
        eta_sep: Separator efficiency (0.0-1.0)
        data_quality_flags: Quality flags for measurements
        uncertainty: Estimation uncertainty (sigma)
        timestamp: State timestamp (UTC)
        provenance_hash: SHA-256 hash for audit trail
    """
    x_est: float  # Estimated dryness fraction
    r_carry: float  # Carryover rate
    eta_sep: float  # Separator efficiency

    # Data quality
    data_quality_flags: DataQualityFlag = DataQualityFlag.GOOD
    quality_score: float = 100.0  # 0-100

    # Uncertainty
    uncertainty: float = 0.005  # Standard deviation

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Provenance
    source_agent: str = "GL-012_STEAMQUAL"
    provenance_hash: str = ""

    # Additional state
    superheat_delta_t: float = 0.0  # Superheat in degC
    drain_valve_position: float = 0.0  # Drain valve position %
    drain_duty_cycle: float = 0.0  # Current duty cycle
    separator_dp_kpa: float = 0.0  # Separator DP

    # Process conditions
    header_pressure_kpa: float = 0.0
    header_temperature_degc: float = 0.0
    steam_flow_kgs: float = 0.0

    def __post_init__(self):
        """Calculate provenance hash."""
        if not self.provenance_hash:
            hash_data = (
                f"{self.x_est}:{self.r_carry}:{self.eta_sep}:"
                f"{self.timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(hash_data.encode()).hexdigest()

    def is_quality_acceptable(self, constraints: QualityConstraints) -> bool:
        """Check if current state meets constraints."""
        if self.x_est < constraints.x_min:
            return False
        if self.superheat_delta_t < constraints.delta_t_min:
            return False
        if self.drain_duty_cycle > constraints.max_drain_duty_cycle:
            return False
        if self.r_carry > constraints.max_carryover_rate:
            return False
        return True

    def get_quality_margin(self, constraints: QualityConstraints) -> Dict[str, float]:
        """Calculate margin to constraint limits."""
        return {
            "x_margin": self.x_est - constraints.x_min,
            "superheat_margin": self.superheat_delta_t - constraints.delta_t_min,
            "duty_margin": constraints.max_drain_duty_cycle - self.drain_duty_cycle,
            "carryover_margin": constraints.max_carryover_rate - self.r_carry,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "x_est": self.x_est,
            "r_carry": self.r_carry,
            "eta_sep": self.eta_sep,
            "data_quality_flags": int(self.data_quality_flags),
            "quality_score": self.quality_score,
            "uncertainty": self.uncertainty,
            "timestamp": self.timestamp.isoformat(),
            "source_agent": self.source_agent,
            "provenance_hash": self.provenance_hash,
            "superheat_delta_t": self.superheat_delta_t,
            "drain_valve_position": self.drain_valve_position,
            "drain_duty_cycle": self.drain_duty_cycle,
            "separator_dp_kpa": self.separator_dp_kpa,
            "header_pressure_kpa": self.header_pressure_kpa,
            "header_temperature_degc": self.header_temperature_degc,
            "steam_flow_kgs": self.steam_flow_kgs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityState":
        """Create from dictionary."""
        return cls(
            x_est=data["x_est"],
            r_carry=data["r_carry"],
            eta_sep=data["eta_sep"],
            data_quality_flags=DataQualityFlag(data.get("data_quality_flags", 0)),
            quality_score=data.get("quality_score", 100.0),
            uncertainty=data.get("uncertainty", 0.005),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_agent=data.get("source_agent", "GL-012_STEAMQUAL"),
            superheat_delta_t=data.get("superheat_delta_t", 0.0),
            drain_valve_position=data.get("drain_valve_position", 0.0),
            drain_duty_cycle=data.get("drain_duty_cycle", 0.0),
            separator_dp_kpa=data.get("separator_dp_kpa", 0.0),
            header_pressure_kpa=data.get("header_pressure_kpa", 0.0),
            header_temperature_degc=data.get("header_temperature_degc", 0.0),
            steam_flow_kgs=data.get("steam_flow_kgs", 0.0),
        )


@dataclass
class QualityEvent:
    """
    Steam quality event exported to GL-003 UNIFIEDSTEAM.

    Captures quality excursions, alerts, and operational events
    for coordination with GL-003 optimization.

    Attributes:
        event_type: Type of quality event
        severity: Event severity level
        timestamp: Event occurrence time
        evidence: Supporting data for the event
        playbook: Recommended response actions
        provenance_hash: SHA-256 hash for audit trail
    """
    event_type: QualityEventType
    severity: EventSeverity
    timestamp: datetime

    # Event details
    message: str = ""
    description: str = ""

    # Evidence
    evidence: Dict[str, Any] = field(default_factory=dict)

    # Response
    playbook: List[str] = field(default_factory=list)
    recommended_action: str = ""

    # Correlation
    correlation_id: str = ""
    related_events: List[str] = field(default_factory=list)

    # Timing
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    # Provenance
    source_agent: str = "GL-012_STEAMQUAL"
    provenance_hash: str = ""

    def __post_init__(self):
        """Calculate provenance hash and correlation ID."""
        if not self.provenance_hash:
            hash_data = (
                f"{self.event_type.value}:{self.severity.value}:"
                f"{self.timestamp.isoformat()}:{json.dumps(self.evidence, sort_keys=True)}"
            )
            self.provenance_hash = hashlib.sha256(hash_data.encode()).hexdigest()

        if not self.correlation_id:
            self.correlation_id = self.provenance_hash[:16]

    def acknowledge(self, operator_id: str = "") -> None:
        """Acknowledge the event."""
        self.acknowledged_at = datetime.now(timezone.utc)
        self.evidence["acknowledged_by"] = operator_id

    def resolve(self, resolution: str = "") -> None:
        """Mark event as resolved."""
        self.resolved_at = datetime.now(timezone.utc)
        self.evidence["resolution"] = resolution

    @property
    def is_active(self) -> bool:
        """Check if event is still active."""
        return self.resolved_at is None

    @property
    def duration_s(self) -> float:
        """Get event duration in seconds."""
        end = self.resolved_at or datetime.now(timezone.utc)
        return (end - self.timestamp).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "severity_name": self.severity.name,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "description": self.description,
            "evidence": self.evidence,
            "playbook": self.playbook,
            "recommended_action": self.recommended_action,
            "correlation_id": self.correlation_id,
            "related_events": self.related_events,
            "detected_at": self.detected_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "source_agent": self.source_agent,
            "provenance_hash": self.provenance_hash,
            "is_active": self.is_active,
            "duration_s": self.duration_s,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityEvent":
        """Create from dictionary."""
        return cls(
            event_type=QualityEventType(data["event_type"]),
            severity=EventSeverity(data["severity"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message=data.get("message", ""),
            description=data.get("description", ""),
            evidence=data.get("evidence", {}),
            playbook=data.get("playbook", []),
            recommended_action=data.get("recommended_action", ""),
            correlation_id=data.get("correlation_id", ""),
            related_events=data.get("related_events", []),
            detected_at=datetime.fromisoformat(data["detected_at"]) if data.get("detected_at") else datetime.now(timezone.utc),
            acknowledged_at=datetime.fromisoformat(data["acknowledged_at"]) if data.get("acknowledged_at") else None,
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None,
            source_agent=data.get("source_agent", "GL-012_STEAMQUAL"),
        )


@dataclass
class QualityCosts:
    """
    Quality-related costs exported to GL-003 UNIFIEDSTEAM.

    Quantifies energy and emissions impacts of steam quality
    for GL-003 economic optimization.

    Attributes:
        energy_penalty_kw: Energy penalty due to wet steam (kW)
        co2e_impact_kghr: CO2e impact rate (kg/hr)
        steam_loss_kgs: Steam loss rate due to wet steam (kg/s)
        condensate_loss_kgs: Condensate loss rate (kg/s)
        timestamp: Cost calculation timestamp
        provenance_hash: SHA-256 hash for audit trail
    """
    energy_penalty_kw: float  # Energy penalty, kW
    co2e_impact_kghr: float  # CO2e impact, kg/hr

    # Detailed losses
    steam_loss_kgs: float = 0.0  # Steam loss, kg/s
    condensate_loss_kgs: float = 0.0  # Condensate loss, kg/s
    makeup_water_lps: float = 0.0  # Makeup water, L/s
    chemical_cost_usdhr: float = 0.0  # Chemical treatment cost, $/hr

    # Economic impact
    hourly_cost_usd: float = 0.0  # Total hourly cost, $
    annual_cost_usd: float = 0.0  # Annualized cost, $

    # Calculation basis
    steam_price_usd_per_klb: float = 12.0  # Steam cost, $/klb
    electricity_price_usd_per_kwh: float = 0.08  # Electricity cost, $/kWh
    co2_price_usd_per_ton: float = 50.0  # Carbon price, $/ton

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    calculation_period_hours: float = 1.0

    # Provenance
    source_agent: str = "GL-012_STEAMQUAL"
    provenance_hash: str = ""

    def __post_init__(self):
        """Calculate provenance hash and derived values."""
        # Calculate total hourly cost if not set
        if self.hourly_cost_usd == 0.0:
            self.hourly_cost_usd = self._calculate_hourly_cost()

        # Annualize
        if self.annual_cost_usd == 0.0:
            self.annual_cost_usd = self.hourly_cost_usd * 8760  # Hours per year

        if not self.provenance_hash:
            hash_data = (
                f"{self.energy_penalty_kw}:{self.co2e_impact_kghr}:"
                f"{self.timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(hash_data.encode()).hexdigest()

    def _calculate_hourly_cost(self) -> float:
        """Calculate total hourly cost."""
        # Energy cost
        energy_cost = self.energy_penalty_kw * self.electricity_price_usd_per_kwh

        # Steam loss cost (convert kg/s to klb/hr)
        steam_cost = self.steam_loss_kgs * 3600 * 2.205 / 1000 * self.steam_price_usd_per_klb

        # Carbon cost (convert kg/hr to ton/hr)
        carbon_cost = self.co2e_impact_kghr / 1000 * self.co2_price_usd_per_ton

        # Chemical cost
        chemical_cost = self.chemical_cost_usdhr

        return energy_cost + steam_cost + carbon_cost + chemical_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "energy_penalty_kw": self.energy_penalty_kw,
            "co2e_impact_kghr": self.co2e_impact_kghr,
            "steam_loss_kgs": self.steam_loss_kgs,
            "condensate_loss_kgs": self.condensate_loss_kgs,
            "makeup_water_lps": self.makeup_water_lps,
            "chemical_cost_usdhr": self.chemical_cost_usdhr,
            "hourly_cost_usd": self.hourly_cost_usd,
            "annual_cost_usd": self.annual_cost_usd,
            "timestamp": self.timestamp.isoformat(),
            "source_agent": self.source_agent,
            "provenance_hash": self.provenance_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityCosts":
        """Create from dictionary."""
        return cls(
            energy_penalty_kw=data["energy_penalty_kw"],
            co2e_impact_kghr=data["co2e_impact_kghr"],
            steam_loss_kgs=data.get("steam_loss_kgs", 0.0),
            condensate_loss_kgs=data.get("condensate_loss_kgs", 0.0),
            makeup_water_lps=data.get("makeup_water_lps", 0.0),
            chemical_cost_usdhr=data.get("chemical_cost_usdhr", 0.0),
            hourly_cost_usd=data.get("hourly_cost_usd", 0.0),
            annual_cost_usd=data.get("annual_cost_usd", 0.0),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_agent=data.get("source_agent", "GL-012_STEAMQUAL"),
        )


@dataclass
class GL003ExportResult:
    """Result of export to GL-003."""
    success: bool
    timestamp: datetime
    export_type: str  # constraints, state, events, costs

    # Content
    data: Dict[str, Any] = field(default_factory=dict)

    # Errors
    errors: List[str] = field(default_factory=list)

    # Timing
    processing_time_ms: float = 0.0

    # Provenance
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "export_type": self.export_type,
            "data": self.data,
            "errors": self.errors,
            "processing_time_ms": self.processing_time_ms,
            "provenance_hash": self.provenance_hash,
        }


@dataclass
class SteamQualitySnapshot:
    """
    Complete snapshot of steam quality state for GL-003 export.

    Bundles constraints, state, active events, and costs
    into a single export package.
    """
    constraints: QualityConstraints
    state: QualityState
    events: List[QualityEvent]
    costs: QualityCosts

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_agent: str = "GL-012_STEAMQUAL"
    provenance_hash: str = ""

    # Synchronization
    sequence_number: int = 0
    last_sync_time: Optional[datetime] = None

    def __post_init__(self):
        """Calculate provenance hash."""
        if not self.provenance_hash:
            hash_data = (
                f"{self.constraints.provenance_hash}:"
                f"{self.state.provenance_hash}:"
                f"{self.costs.provenance_hash}:"
                f"{self.timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(hash_data.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "constraints": self.constraints.to_dict(),
            "state": self.state.to_dict(),
            "events": [e.to_dict() for e in self.events],
            "costs": self.costs.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "source_agent": self.source_agent,
            "provenance_hash": self.provenance_hash,
            "sequence_number": self.sequence_number,
            "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
        }


@dataclass
class GL003InterfaceConfig:
    """GL-003 interface configuration."""
    # Connection settings
    gl003_endpoint: str = "http://localhost:8003"
    api_key: Optional[str] = None
    timeout_s: float = 30.0

    # Export settings
    export_interval_s: float = 5.0  # State export interval
    constraint_update_interval_s: float = 60.0  # Constraint update interval
    event_batch_size: int = 100

    # Quality thresholds
    min_export_quality_score: float = 50.0

    # Retry settings
    retry_enabled: bool = True
    retry_count: int = 3
    retry_delay_s: float = 1.0

    # Buffering
    buffer_enabled: bool = True
    buffer_size: int = 1000


class GL003Interface:
    """
    Interface for GL-012 STEAMQUAL to GL-003 UNIFIEDSTEAM communication.

    Exports steam quality data (constraints, state, events, costs) to
    GL-003 for coordination with steam system optimization.

    Example:
        config = GL003InterfaceConfig(
            gl003_endpoint="http://gl003.company.com:8003",
            api_key="secret_key",
        )
        interface = GL003Interface(config)

        # Export constraints
        constraints = QualityConstraints(
            x_min=0.97,
            delta_t_min=5.0,
            max_drain_duty_cycle=0.25,
            max_ramp_rate=0.01,
        )
        result = await interface.export_constraints(constraints)

        # Export state
        state = QualityState(
            x_est=0.98,
            r_carry=0.01,
            eta_sep=0.97,
        )
        result = await interface.export_state(state)

        # Export events
        event = QualityEvent(
            event_type=QualityEventType.LOW_DRYNESS,
            severity=EventSeverity.MEDIUM,
            timestamp=datetime.now(timezone.utc),
            message="Steam dryness below threshold",
        )
        result = await interface.export_event(event)
    """

    def __init__(
        self,
        config: Optional[GL003InterfaceConfig] = None,
        vault_client: Optional[Any] = None,
    ) -> None:
        """Initialize GL-003 interface."""
        self.config = config or GL003InterfaceConfig()
        self._vault_client = vault_client

        # Retrieve API key from vault
        if vault_client and not self.config.api_key:
            try:
                self.config.api_key = vault_client.get_secret("gl003/api_key")
            except Exception as e:
                logger.warning(f"Failed to retrieve GL-003 API key: {e}")

        # Connection state
        self._connected = False
        self._session = None

        # Export state
        self._last_constraint_export: Optional[datetime] = None
        self._last_state_export: Optional[datetime] = None
        self._sequence_number = 0

        # Buffer for retries
        self._buffer: List[Dict[str, Any]] = []

        # Active events
        self._active_events: Dict[str, QualityEvent] = {}

        # Statistics
        self._stats = {
            "constraints_exported": 0,
            "states_exported": 0,
            "events_exported": 0,
            "costs_exported": 0,
            "snapshots_exported": 0,
            "errors": 0,
            "retries": 0,
        }

        logger.info(f"GL003Interface initialized: {self.config.gl003_endpoint}")

    async def connect(self) -> bool:
        """Connect to GL-003."""
        try:
            # In production: create HTTP session and test connection
            # async with aiohttp.ClientSession() as session:
            #     async with session.get(f"{self.config.gl003_endpoint}/health") as resp:
            #         self._connected = resp.status == 200

            self._connected = True
            logger.info("Connected to GL-003 UNIFIEDSTEAM")
            return True

        except Exception as e:
            logger.error(f"GL-003 connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from GL-003."""
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    async def export_constraints(
        self,
        constraints: QualityConstraints,
    ) -> GL003ExportResult:
        """
        Export quality constraints to GL-003.

        Args:
            constraints: Quality constraints to export

        Returns:
            GL003ExportResult
        """
        import time
        start = time.perf_counter()

        # Validate constraints
        errors = constraints.validate_values()
        if errors:
            return GL003ExportResult(
                success=False,
                timestamp=datetime.now(timezone.utc),
                export_type="constraints",
                errors=errors,
            )

        try:
            # Export to GL-003
            data = constraints.to_dict()
            success = await self._send_to_gl003("/api/quality/constraints", data)

            self._stats["constraints_exported"] += 1
            self._last_constraint_export = datetime.now(timezone.utc)

            return GL003ExportResult(
                success=success,
                timestamp=datetime.now(timezone.utc),
                export_type="constraints",
                data=data,
                processing_time_ms=(time.perf_counter() - start) * 1000,
                provenance_hash=constraints.provenance_hash,
            )

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Constraint export failed: {e}")
            return GL003ExportResult(
                success=False,
                timestamp=datetime.now(timezone.utc),
                export_type="constraints",
                errors=[str(e)],
            )

    async def export_state(self, state: QualityState) -> GL003ExportResult:
        """
        Export quality state to GL-003.

        Args:
            state: Current quality state

        Returns:
            GL003ExportResult
        """
        import time
        start = time.perf_counter()

        # Check data quality
        if state.quality_score < self.config.min_export_quality_score:
            logger.warning(
                f"State quality score {state.quality_score} below threshold "
                f"{self.config.min_export_quality_score}"
            )

        try:
            data = state.to_dict()
            data["sequence_number"] = self._sequence_number
            self._sequence_number += 1

            success = await self._send_to_gl003("/api/quality/state", data)

            self._stats["states_exported"] += 1
            self._last_state_export = datetime.now(timezone.utc)

            return GL003ExportResult(
                success=success,
                timestamp=datetime.now(timezone.utc),
                export_type="state",
                data=data,
                processing_time_ms=(time.perf_counter() - start) * 1000,
                provenance_hash=state.provenance_hash,
            )

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"State export failed: {e}")
            return GL003ExportResult(
                success=False,
                timestamp=datetime.now(timezone.utc),
                export_type="state",
                errors=[str(e)],
            )

    async def export_event(self, event: QualityEvent) -> GL003ExportResult:
        """
        Export quality event to GL-003.

        Args:
            event: Quality event to export

        Returns:
            GL003ExportResult
        """
        import time
        start = time.perf_counter()

        try:
            data = event.to_dict()
            success = await self._send_to_gl003("/api/quality/events", data)

            # Track active events
            if event.is_active:
                self._active_events[event.correlation_id] = event
            else:
                self._active_events.pop(event.correlation_id, None)

            self._stats["events_exported"] += 1

            return GL003ExportResult(
                success=success,
                timestamp=datetime.now(timezone.utc),
                export_type="events",
                data=data,
                processing_time_ms=(time.perf_counter() - start) * 1000,
                provenance_hash=event.provenance_hash,
            )

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Event export failed: {e}")
            return GL003ExportResult(
                success=False,
                timestamp=datetime.now(timezone.utc),
                export_type="events",
                errors=[str(e)],
            )

    async def export_costs(self, costs: QualityCosts) -> GL003ExportResult:
        """
        Export quality costs to GL-003.

        Args:
            costs: Quality-related costs

        Returns:
            GL003ExportResult
        """
        import time
        start = time.perf_counter()

        try:
            data = costs.to_dict()
            success = await self._send_to_gl003("/api/quality/costs", data)

            self._stats["costs_exported"] += 1

            return GL003ExportResult(
                success=success,
                timestamp=datetime.now(timezone.utc),
                export_type="costs",
                data=data,
                processing_time_ms=(time.perf_counter() - start) * 1000,
                provenance_hash=costs.provenance_hash,
            )

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Cost export failed: {e}")
            return GL003ExportResult(
                success=False,
                timestamp=datetime.now(timezone.utc),
                export_type="costs",
                errors=[str(e)],
            )

    async def export_snapshot(
        self,
        snapshot: SteamQualitySnapshot,
    ) -> GL003ExportResult:
        """
        Export complete quality snapshot to GL-003.

        Args:
            snapshot: Complete quality snapshot

        Returns:
            GL003ExportResult
        """
        import time
        start = time.perf_counter()

        try:
            data = snapshot.to_dict()
            success = await self._send_to_gl003("/api/quality/snapshot", data)

            self._stats["snapshots_exported"] += 1

            return GL003ExportResult(
                success=success,
                timestamp=datetime.now(timezone.utc),
                export_type="snapshot",
                data=data,
                processing_time_ms=(time.perf_counter() - start) * 1000,
                provenance_hash=snapshot.provenance_hash,
            )

        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Snapshot export failed: {e}")
            return GL003ExportResult(
                success=False,
                timestamp=datetime.now(timezone.utc),
                export_type="snapshot",
                errors=[str(e)],
            )

    async def _send_to_gl003(self, endpoint: str, data: Dict[str, Any]) -> bool:
        """Send data to GL-003 API."""
        if not self._connected:
            if self.config.buffer_enabled:
                self._buffer_data(endpoint, data)
            return False

        # In production: HTTP POST to GL-003
        # headers = {"Authorization": f"Bearer {self.config.api_key}"}
        # async with self._session.post(
        #     f"{self.config.gl003_endpoint}{endpoint}",
        #     json=data,
        #     headers=headers,
        #     timeout=self.config.timeout_s,
        # ) as resp:
        #     return resp.status == 200

        logger.debug(f"Sent to GL-003 {endpoint}: {len(json.dumps(data))} bytes")
        return True

    def _buffer_data(self, endpoint: str, data: Dict[str, Any]) -> None:
        """Buffer data for later retry."""
        if len(self._buffer) >= self.config.buffer_size:
            self._buffer.pop(0)  # Remove oldest

        self._buffer.append({
            "endpoint": endpoint,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    async def flush_buffer(self) -> int:
        """Flush buffered data to GL-003."""
        if not self._connected or not self._buffer:
            return 0

        flushed = 0
        while self._buffer:
            item = self._buffer.pop(0)
            try:
                success = await self._send_to_gl003(item["endpoint"], item["data"])
                if success:
                    flushed += 1
            except Exception as e:
                logger.error(f"Buffer flush error: {e}")
                self._buffer.insert(0, item)  # Put back
                break

        return flushed

    def create_low_dryness_event(
        self,
        current_dryness: float,
        threshold: float,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> QualityEvent:
        """Create low dryness event with standard playbook."""
        return QualityEvent(
            event_type=QualityEventType.LOW_DRYNESS,
            severity=EventSeverity.HIGH if current_dryness < threshold - 0.02 else EventSeverity.MEDIUM,
            timestamp=datetime.now(timezone.utc),
            message=f"Steam dryness {current_dryness:.3f} below threshold {threshold:.3f}",
            description=(
                f"Steam quality has dropped below the minimum dryness constraint. "
                f"Current: {current_dryness:.1%}, Threshold: {threshold:.1%}"
            ),
            evidence={
                "current_dryness": current_dryness,
                "threshold": threshold,
                "deviation": threshold - current_dryness,
                **(evidence or {}),
            },
            playbook=[
                "1. Check separator differential pressure",
                "2. Verify drain valve operation",
                "3. Check upstream steam conditions",
                "4. Review recent load changes",
                "5. Consider reducing steam demand temporarily",
            ],
            recommended_action="Increase separator efficiency or reduce load",
        )

    def create_separator_dp_event(
        self,
        current_dp: float,
        threshold: float,
        high: bool = True,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> QualityEvent:
        """Create separator DP event."""
        event_type = QualityEventType.SEPARATOR_DP_HIGH if high else QualityEventType.SEPARATOR_DP_LOW
        severity = EventSeverity.MEDIUM

        return QualityEvent(
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(timezone.utc),
            message=f"Separator DP {'high' if high else 'low'}: {current_dp:.1f} kPa",
            evidence={
                "current_dp_kpa": current_dp,
                "threshold_kpa": threshold,
                **(evidence or {}),
            },
            playbook=[
                "1. Check separator internals for fouling",
                "2. Verify flow measurement accuracy",
                "3. Check for liquid accumulation",
                "4. Schedule maintenance if persistent",
            ],
            recommended_action="Schedule separator inspection",
        )

    def get_active_events(self) -> List[QualityEvent]:
        """Get list of active events."""
        return list(self._active_events.values())

    def get_statistics(self) -> Dict[str, Any]:
        """Get interface statistics."""
        return {
            **self._stats,
            "connected": self._connected,
            "buffer_size": len(self._buffer),
            "active_events": len(self._active_events),
            "last_constraint_export": self._last_constraint_export.isoformat() if self._last_constraint_export else None,
            "last_state_export": self._last_state_export.isoformat() if self._last_state_export else None,
            "sequence_number": self._sequence_number,
        }


def create_gl003_interface(
    endpoint: str = "http://localhost:8003",
    **kwargs: Any,
) -> GL003Interface:
    """
    Create GL-003 interface with common defaults.

    Args:
        endpoint: GL-003 API endpoint
        **kwargs: Additional configuration

    Returns:
        Configured GL003Interface
    """
    config = GL003InterfaceConfig(
        gl003_endpoint=endpoint,
        **kwargs,
    )
    return GL003Interface(config)
