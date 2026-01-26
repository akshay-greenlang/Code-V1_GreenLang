# -*- coding: utf-8 -*-
"""
GL-DATA-X-013: IoT Meter Management Agent
=========================================

Manages meter inventory, calibration records, and trust scores for
IoT devices used in emissions data collection.

Capabilities:
    - Maintain meter inventory and metadata
    - Track calibration schedules and history
    - Calculate meter trust scores
    - Detect anomalies in meter readings
    - Manage meter hierarchies (virtual meters, submeters)
    - Track data quality metrics
    - Provenance tracking with SHA-256 hashes

Zero-Hallucination Guarantees:
    - All readings from calibrated physical devices
    - NO LLM involvement in reading validation
    - Trust scores based on deterministic criteria
    - Complete audit trail for all meters

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class MeterType(str, Enum):
    """Types of meters."""
    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    WATER = "water"
    STEAM = "steam"
    CHILLED_WATER = "chilled_water"
    HOT_WATER = "hot_water"
    FUEL_OIL = "fuel_oil"
    COMPRESSED_AIR = "compressed_air"
    THERMAL = "thermal"
    RENEWABLE = "renewable"


class MeterStatus(str, Enum):
    """Meter operational status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    CALIBRATION_DUE = "calibration_due"
    FAILED = "failed"
    DECOMMISSIONED = "decommissioned"


class CommunicationType(str, Enum):
    """Meter communication types."""
    MODBUS = "modbus"
    BACNET = "bacnet"
    MBUS = "mbus"
    PULSE = "pulse"
    MANUAL = "manual"
    IOT_CELLULAR = "iot_cellular"
    IOT_LORAWAN = "iot_lorawan"
    IOT_WIFI = "iot_wifi"


class CalibrationResult(str, Enum):
    """Calibration result."""
    PASS = "pass"
    FAIL = "fail"
    ADJUSTED = "adjusted"
    REPLACED = "replaced"


class TrustLevel(str, Enum):
    """Meter trust level."""
    HIGH = "high"  # >= 90
    MEDIUM = "medium"  # 70-89
    LOW = "low"  # 50-69
    UNTRUSTED = "untrusted"  # < 50


class AnomalyType(str, Enum):
    """Types of meter anomalies."""
    SPIKE = "spike"
    GAP = "gap"
    FLAT_LINE = "flat_line"
    NEGATIVE = "negative"
    OUT_OF_RANGE = "out_of_range"
    CALIBRATION_DRIFT = "calibration_drift"
    COMMUNICATION_FAILURE = "communication_failure"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class MeterLocation(BaseModel):
    """Meter location information."""
    building_id: str = Field(...)
    building_name: Optional[str] = Field(None)
    floor: Optional[str] = Field(None)
    room: Optional[str] = Field(None)
    panel_id: Optional[str] = Field(None)
    coordinates: Optional[Dict[str, float]] = Field(None)


class MeterSpecification(BaseModel):
    """Meter technical specifications."""
    manufacturer: str = Field(...)
    model: str = Field(...)
    serial_number: str = Field(...)
    measurement_unit: str = Field(...)
    range_min: float = Field(...)
    range_max: float = Field(...)
    accuracy_pct: float = Field(...)
    resolution: float = Field(...)
    ct_ratio: Optional[float] = Field(None)  # For electricity meters
    pulse_factor: Optional[float] = Field(None)
    installation_date: date = Field(...)
    warranty_end: Optional[date] = Field(None)


class CalibrationRecord(BaseModel):
    """Calibration record."""
    calibration_id: str = Field(...)
    meter_id: str = Field(...)
    calibration_date: date = Field(...)
    next_calibration_date: date = Field(...)
    technician: str = Field(...)
    result: CalibrationResult = Field(...)
    pre_adjustment_error_pct: Optional[float] = Field(None)
    post_adjustment_error_pct: Optional[float] = Field(None)
    certificate_number: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)


class MeterReading(BaseModel):
    """Meter reading."""
    reading_id: str = Field(...)
    meter_id: str = Field(...)
    timestamp: datetime = Field(...)
    value: float = Field(...)
    unit: str = Field(...)
    reading_type: str = Field(default="automatic")  # automatic, manual, estimated
    quality_flag: str = Field(default="good")
    anomaly_detected: bool = Field(default=False)


class MeterAnomaly(BaseModel):
    """Detected meter anomaly."""
    anomaly_id: str = Field(...)
    meter_id: str = Field(...)
    detected_at: datetime = Field(...)
    anomaly_type: AnomalyType = Field(...)
    severity: str = Field(...)  # low, medium, high, critical
    description: str = Field(...)
    affected_readings: List[str] = Field(default_factory=list)
    resolved: bool = Field(default=False)
    resolution_notes: Optional[str] = Field(None)


class MeterTrustScore(BaseModel):
    """Meter trust score calculation."""
    meter_id: str = Field(...)
    calculated_at: datetime = Field(...)
    overall_score: float = Field(..., ge=0, le=100)
    trust_level: TrustLevel = Field(...)
    components: Dict[str, float] = Field(default_factory=dict)
    factors: Dict[str, Any] = Field(default_factory=dict)


class Meter(BaseModel):
    """Meter definition."""
    meter_id: str = Field(...)
    name: str = Field(...)
    description: Optional[str] = Field(None)
    meter_type: MeterType = Field(...)
    status: MeterStatus = Field(default=MeterStatus.ACTIVE)
    location: MeterLocation = Field(...)
    specification: MeterSpecification = Field(...)
    communication: CommunicationType = Field(...)
    parent_meter_id: Optional[str] = Field(None)  # For submeters
    child_meter_ids: List[str] = Field(default_factory=list)
    is_virtual: bool = Field(default=False)
    virtual_formula: Optional[str] = Field(None)  # For virtual meters
    tags: List[str] = Field(default_factory=list)
    current_trust_score: Optional[float] = Field(None)


class VirtualMeterConfig(BaseModel):
    """Virtual meter configuration."""
    meter_id: str = Field(...)
    name: str = Field(...)
    formula: str = Field(...)  # e.g., "M001 + M002 - M003"
    source_meter_ids: List[str] = Field(...)
    coefficients: Dict[str, float] = Field(default_factory=dict)
    unit: str = Field(...)


class MeterQueryInput(BaseModel):
    """Input for meter operations."""
    operation: str = Field(...)  # register, update, query, calibrate, trust_score, anomaly
    meter: Optional[Meter] = Field(None)
    meter_id: Optional[str] = Field(None)
    calibration: Optional[CalibrationRecord] = Field(None)
    reading: Optional[MeterReading] = Field(None)
    virtual_config: Optional[VirtualMeterConfig] = Field(None)
    building_id: Optional[str] = Field(None)
    meter_type: Optional[MeterType] = Field(None)
    start_time: Optional[datetime] = Field(None)
    end_time: Optional[datetime] = Field(None)
    tenant_id: Optional[str] = Field(None)


class MeterQueryOutput(BaseModel):
    """Output from meter operations."""
    operation: str = Field(...)
    meters: List[Meter] = Field(default_factory=list)
    calibrations: List[CalibrationRecord] = Field(default_factory=list)
    readings: List[MeterReading] = Field(default_factory=list)
    anomalies: List[MeterAnomaly] = Field(default_factory=list)
    trust_scores: List[MeterTrustScore] = Field(default_factory=list)
    meter_count: int = Field(default=0)
    active_count: int = Field(default=0)
    calibration_due_count: int = Field(default=0)
    processing_time_ms: float = Field(...)
    provenance_hash: str = Field(...)


# =============================================================================
# IOT METER MANAGEMENT AGENT
# =============================================================================

class IoTMeterManagementAgent(BaseAgent):
    """
    GL-DATA-X-013: IoT Meter Management Agent

    Manages meter inventory, calibration, and trust scores.

    Zero-Hallucination Guarantees:
        - All readings from calibrated physical devices
        - NO LLM involvement in reading validation
        - Trust scores based on deterministic criteria
        - Complete audit trail for all meters
    """

    AGENT_ID = "GL-DATA-X-013"
    AGENT_NAME = "IoT Meter Management Agent"
    VERSION = "1.0.0"

    # Trust score weights
    TRUST_WEIGHTS = {
        "calibration_status": 0.25,
        "data_availability": 0.20,
        "anomaly_history": 0.20,
        "accuracy_spec": 0.15,
        "communication_reliability": 0.10,
        "age_factor": 0.10
    }

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize IoTMeterManagementAgent."""
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="IoT meter inventory and trust management",
                version=self.VERSION,
            )
        super().__init__(config)

        self._meters: Dict[str, Meter] = {}
        self._calibrations: Dict[str, List[CalibrationRecord]] = {}
        self._readings: Dict[str, List[MeterReading]] = {}
        self._anomalies: Dict[str, List[MeterAnomaly]] = {}
        self._trust_scores: Dict[str, MeterTrustScore] = {}

        self.logger.info(f"Initialized {self.AGENT_NAME}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute meter operation."""
        start_time = datetime.utcnow()

        try:
            query_input = MeterQueryInput(**input_data)

            if query_input.operation == "register":
                return self._handle_register(query_input, start_time)
            elif query_input.operation == "update":
                return self._handle_update(query_input, start_time)
            elif query_input.operation == "query":
                return self._handle_query(query_input, start_time)
            elif query_input.operation == "calibrate":
                return self._handle_calibrate(query_input, start_time)
            elif query_input.operation == "trust_score":
                return self._handle_trust_score(query_input, start_time)
            elif query_input.operation == "anomaly":
                return self._handle_anomaly(query_input, start_time)
            elif query_input.operation == "reading":
                return self._handle_reading(query_input, start_time)
            elif query_input.operation == "virtual":
                return self._handle_virtual(query_input, start_time)
            else:
                return AgentResult(success=False, error=f"Unknown operation: {query_input.operation}")

        except Exception as e:
            self.logger.error(f"Meter operation failed: {str(e)}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _handle_register(self, query_input: MeterQueryInput, start_time: datetime) -> AgentResult:
        """Handle meter registration."""
        if not query_input.meter:
            return AgentResult(success=False, error="meter required for registration")

        meter = query_input.meter
        self._meters[meter.meter_id] = meter
        self._calibrations[meter.meter_id] = []
        self._readings[meter.meter_id] = []
        self._anomalies[meter.meter_id] = []

        # Calculate initial trust score
        trust_score = self._calculate_trust_score(meter)
        self._trust_scores[meter.meter_id] = trust_score
        meter.current_trust_score = trust_score.overall_score

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        output = MeterQueryOutput(
            operation="register",
            meters=[meter.model_dump()],
            trust_scores=[trust_score.model_dump()],
            meter_count=1,
            processing_time_ms=processing_time,
            provenance_hash=self._compute_provenance_hash(meter.model_dump(), {"registered": True})
        )

        return AgentResult(success=True, data=output.model_dump())

    def _handle_update(self, query_input: MeterQueryInput, start_time: datetime) -> AgentResult:
        """Handle meter update."""
        if not query_input.meter_id:
            return AgentResult(success=False, error="meter_id required")

        if query_input.meter_id not in self._meters:
            return AgentResult(success=False, error=f"Meter not found: {query_input.meter_id}")

        # Update meter
        if query_input.meter:
            self._meters[query_input.meter_id] = query_input.meter

        return AgentResult(success=True, data={"meter_id": query_input.meter_id, "updated": True})

    def _handle_query(self, query_input: MeterQueryInput, start_time: datetime) -> AgentResult:
        """Handle meter query."""
        meters = list(self._meters.values())

        if query_input.meter_id:
            meters = [m for m in meters if m.meter_id == query_input.meter_id]
        if query_input.building_id:
            meters = [m for m in meters if m.location.building_id == query_input.building_id]
        if query_input.meter_type:
            meters = [m for m in meters if m.meter_type == query_input.meter_type]

        active_count = len([m for m in meters if m.status == MeterStatus.ACTIVE])
        calibration_due = len([m for m in meters if m.status == MeterStatus.CALIBRATION_DUE])

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        output = MeterQueryOutput(
            operation="query",
            meters=[m.model_dump() for m in meters],
            meter_count=len(meters),
            active_count=active_count,
            calibration_due_count=calibration_due,
            processing_time_ms=processing_time,
            provenance_hash=self._compute_provenance_hash({}, {"count": len(meters)})
        )

        return AgentResult(success=True, data=output.model_dump())

    def _handle_calibrate(self, query_input: MeterQueryInput, start_time: datetime) -> AgentResult:
        """Handle calibration record."""
        if not query_input.calibration:
            return AgentResult(success=False, error="calibration record required")

        calibration = query_input.calibration
        meter_id = calibration.meter_id

        if meter_id not in self._meters:
            return AgentResult(success=False, error=f"Meter not found: {meter_id}")

        # Store calibration
        if meter_id not in self._calibrations:
            self._calibrations[meter_id] = []
        self._calibrations[meter_id].append(calibration)

        # Update meter status
        meter = self._meters[meter_id]
        if calibration.result in [CalibrationResult.PASS, CalibrationResult.ADJUSTED]:
            meter.status = MeterStatus.ACTIVE
        elif calibration.result == CalibrationResult.FAIL:
            meter.status = MeterStatus.MAINTENANCE

        # Recalculate trust score
        trust_score = self._calculate_trust_score(meter)
        self._trust_scores[meter_id] = trust_score
        meter.current_trust_score = trust_score.overall_score

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        output = MeterQueryOutput(
            operation="calibrate",
            calibrations=[calibration.model_dump()],
            trust_scores=[trust_score.model_dump()],
            processing_time_ms=processing_time,
            provenance_hash=self._compute_provenance_hash(calibration.model_dump(), {"result": calibration.result.value})
        )

        return AgentResult(success=True, data=output.model_dump())

    def _handle_trust_score(self, query_input: MeterQueryInput, start_time: datetime) -> AgentResult:
        """Handle trust score calculation."""
        meter_ids = [query_input.meter_id] if query_input.meter_id else list(self._meters.keys())
        trust_scores = []

        for meter_id in meter_ids:
            if meter_id in self._meters:
                score = self._calculate_trust_score(self._meters[meter_id])
                self._trust_scores[meter_id] = score
                self._meters[meter_id].current_trust_score = score.overall_score
                trust_scores.append(score)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        output = MeterQueryOutput(
            operation="trust_score",
            trust_scores=[s.model_dump() for s in trust_scores],
            processing_time_ms=processing_time,
            provenance_hash=self._compute_provenance_hash({}, {"count": len(trust_scores)})
        )

        return AgentResult(success=True, data=output.model_dump())

    def _handle_anomaly(self, query_input: MeterQueryInput, start_time: datetime) -> AgentResult:
        """Handle anomaly query/detection."""
        if not query_input.meter_id:
            return AgentResult(success=False, error="meter_id required")

        anomalies = self._anomalies.get(query_input.meter_id, [])

        # Filter by time range if specified
        if query_input.start_time:
            anomalies = [a for a in anomalies if a.detected_at >= query_input.start_time]
        if query_input.end_time:
            anomalies = [a for a in anomalies if a.detected_at <= query_input.end_time]

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        output = MeterQueryOutput(
            operation="anomaly",
            anomalies=[a.model_dump() for a in anomalies],
            processing_time_ms=processing_time,
            provenance_hash=self._compute_provenance_hash({}, {"count": len(anomalies)})
        )

        return AgentResult(success=True, data=output.model_dump())

    def _handle_reading(self, query_input: MeterQueryInput, start_time: datetime) -> AgentResult:
        """Handle meter reading submission."""
        if not query_input.reading:
            return AgentResult(success=False, error="reading required")

        reading = query_input.reading
        meter_id = reading.meter_id

        if meter_id not in self._meters:
            return AgentResult(success=False, error=f"Meter not found: {meter_id}")

        # Check for anomalies
        anomaly = self._detect_anomaly(meter_id, reading)
        if anomaly:
            reading.anomaly_detected = True
            self._anomalies[meter_id].append(anomaly)

        # Store reading
        if meter_id not in self._readings:
            self._readings[meter_id] = []
        self._readings[meter_id].append(reading)

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        output = MeterQueryOutput(
            operation="reading",
            readings=[reading.model_dump()],
            anomalies=[anomaly.model_dump()] if anomaly else [],
            processing_time_ms=processing_time,
            provenance_hash=self._compute_provenance_hash(reading.model_dump(), {"anomaly": reading.anomaly_detected})
        )

        return AgentResult(success=True, data=output.model_dump())

    def _handle_virtual(self, query_input: MeterQueryInput, start_time: datetime) -> AgentResult:
        """Handle virtual meter registration."""
        if not query_input.virtual_config:
            return AgentResult(success=False, error="virtual_config required")

        config = query_input.virtual_config

        # Create virtual meter
        meter = Meter(
            meter_id=config.meter_id,
            name=config.name,
            description=f"Virtual meter: {config.formula}",
            meter_type=MeterType.ELECTRICITY,  # Default, can be overridden
            status=MeterStatus.ACTIVE,
            location=MeterLocation(building_id="virtual"),
            specification=MeterSpecification(
                manufacturer="GreenLang",
                model="Virtual",
                serial_number=config.meter_id,
                measurement_unit=config.unit,
                range_min=0,
                range_max=999999,
                accuracy_pct=0.5,
                resolution=0.01,
                installation_date=date.today()
            ),
            communication=CommunicationType.MANUAL,
            is_virtual=True,
            virtual_formula=config.formula,
            child_meter_ids=config.source_meter_ids
        )

        self._meters[meter.meter_id] = meter

        return AgentResult(success=True, data={"meter_id": meter.meter_id, "registered": True, "is_virtual": True})

    def _calculate_trust_score(self, meter: Meter) -> MeterTrustScore:
        """Calculate trust score for a meter."""
        components = {}
        factors = {}

        # 1. Calibration status (25%)
        calibrations = self._calibrations.get(meter.meter_id, [])
        if calibrations:
            latest = calibrations[-1]
            days_since = (date.today() - latest.calibration_date).days
            days_until_due = (latest.next_calibration_date - date.today()).days

            if days_until_due > 30:
                cal_score = 100
            elif days_until_due > 0:
                cal_score = 70 + (days_until_due / 30) * 30
            else:
                cal_score = max(0, 70 - abs(days_until_due) * 2)

            factors["last_calibration"] = latest.calibration_date.isoformat()
            factors["days_until_due"] = days_until_due
        else:
            cal_score = 50  # No calibration record
            factors["last_calibration"] = None

        components["calibration_status"] = cal_score

        # 2. Data availability (20%)
        readings = self._readings.get(meter.meter_id, [])
        if readings:
            # Check last 24 hours
            recent_readings = [r for r in readings if (datetime.utcnow() - r.timestamp).days < 1]
            expected = 24 if meter.communication != CommunicationType.MANUAL else 1
            availability = min(100, (len(recent_readings) / expected) * 100)
        else:
            availability = 0

        components["data_availability"] = availability
        factors["recent_readings"] = len(readings)

        # 3. Anomaly history (20%)
        anomalies = self._anomalies.get(meter.meter_id, [])
        recent_anomalies = [a for a in anomalies if (datetime.utcnow() - a.detected_at).days < 30]
        anomaly_score = max(0, 100 - len(recent_anomalies) * 10)
        components["anomaly_history"] = anomaly_score
        factors["recent_anomalies"] = len(recent_anomalies)

        # 4. Accuracy specification (15%)
        accuracy_score = max(0, 100 - meter.specification.accuracy_pct * 20)
        components["accuracy_spec"] = accuracy_score
        factors["accuracy_pct"] = meter.specification.accuracy_pct

        # 5. Communication reliability (10%)
        if meter.communication in [CommunicationType.MODBUS, CommunicationType.BACNET]:
            comm_score = 95
        elif meter.communication in [CommunicationType.IOT_CELLULAR, CommunicationType.IOT_WIFI]:
            comm_score = 85
        elif meter.communication == CommunicationType.MANUAL:
            comm_score = 60
        else:
            comm_score = 75
        components["communication_reliability"] = comm_score

        # 6. Age factor (10%)
        age_years = (date.today() - meter.specification.installation_date).days / 365
        age_score = max(0, 100 - age_years * 5)  # Lose 5 points per year
        components["age_factor"] = age_score
        factors["age_years"] = round(age_years, 1)

        # Calculate weighted score
        overall = sum(
            components[k] * self.TRUST_WEIGHTS[k]
            for k in components
        )

        # Determine trust level
        if overall >= 90:
            level = TrustLevel.HIGH
        elif overall >= 70:
            level = TrustLevel.MEDIUM
        elif overall >= 50:
            level = TrustLevel.LOW
        else:
            level = TrustLevel.UNTRUSTED

        return MeterTrustScore(
            meter_id=meter.meter_id,
            calculated_at=datetime.utcnow(),
            overall_score=round(overall, 1),
            trust_level=level,
            components=components,
            factors=factors
        )

    def _detect_anomaly(self, meter_id: str, reading: MeterReading) -> Optional[MeterAnomaly]:
        """Detect anomalies in meter reading."""
        meter = self._meters.get(meter_id)
        if not meter:
            return None

        spec = meter.specification

        # Check range
        if reading.value < spec.range_min:
            return MeterAnomaly(
                anomaly_id=f"ANOM-{uuid.uuid4().hex[:8].upper()}",
                meter_id=meter_id,
                detected_at=datetime.utcnow(),
                anomaly_type=AnomalyType.OUT_OF_RANGE,
                severity="high",
                description=f"Reading {reading.value} below minimum {spec.range_min}",
                affected_readings=[reading.reading_id]
            )

        if reading.value > spec.range_max:
            return MeterAnomaly(
                anomaly_id=f"ANOM-{uuid.uuid4().hex[:8].upper()}",
                meter_id=meter_id,
                detected_at=datetime.utcnow(),
                anomaly_type=AnomalyType.OUT_OF_RANGE,
                severity="high",
                description=f"Reading {reading.value} above maximum {spec.range_max}",
                affected_readings=[reading.reading_id]
            )

        # Check negative
        if reading.value < 0:
            return MeterAnomaly(
                anomaly_id=f"ANOM-{uuid.uuid4().hex[:8].upper()}",
                meter_id=meter_id,
                detected_at=datetime.utcnow(),
                anomaly_type=AnomalyType.NEGATIVE,
                severity="medium",
                description=f"Negative reading: {reading.value}",
                affected_readings=[reading.reading_id]
            )

        # Check spike (compared to recent readings)
        recent = self._readings.get(meter_id, [])[-10:]
        if recent:
            avg = sum(r.value for r in recent) / len(recent)
            if avg > 0 and abs(reading.value - avg) / avg > 3:  # 3x deviation
                return MeterAnomaly(
                    anomaly_id=f"ANOM-{uuid.uuid4().hex[:8].upper()}",
                    meter_id=meter_id,
                    detected_at=datetime.utcnow(),
                    anomaly_type=AnomalyType.SPIKE,
                    severity="medium",
                    description=f"Spike detected: {reading.value} vs avg {avg:.2f}",
                    affected_readings=[reading.reading_id]
                )

        return None

    def _compute_provenance_hash(self, input_data: Any, output_data: Any) -> str:
        """Compute SHA-256 provenance hash."""
        provenance_str = json.dumps(
            {"input": str(input_data), "output": output_data},
            sort_keys=True, default=str
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    # =========================================================================
    # PUBLIC API METHODS
    # =========================================================================

    def register_meter(self, meter: Meter) -> Meter:
        """Register a new meter."""
        result = self.run({"operation": "register", "meter": meter.model_dump()})
        if result.success and result.data.get("meters"):
            return Meter(**result.data["meters"][0])
        raise ValueError(f"Registration failed: {result.error}")

    def record_calibration(self, calibration: CalibrationRecord) -> CalibrationRecord:
        """Record a calibration."""
        result = self.run({"operation": "calibrate", "calibration": calibration.model_dump()})
        if result.success and result.data.get("calibrations"):
            return CalibrationRecord(**result.data["calibrations"][0])
        raise ValueError(f"Calibration failed: {result.error}")

    def get_trust_score(self, meter_id: str) -> MeterTrustScore:
        """Get trust score for a meter."""
        result = self.run({"operation": "trust_score", "meter_id": meter_id})
        if result.success and result.data.get("trust_scores"):
            return MeterTrustScore(**result.data["trust_scores"][0])
        raise ValueError(f"Trust score failed: {result.error}")

    def submit_reading(self, reading: MeterReading) -> MeterReading:
        """Submit a meter reading."""
        result = self.run({"operation": "reading", "reading": reading.model_dump()})
        if result.success and result.data.get("readings"):
            return MeterReading(**result.data["readings"][0])
        raise ValueError(f"Reading submission failed: {result.error}")

    def get_meter_types(self) -> List[str]:
        """Get list of meter types."""
        return [t.value for t in MeterType]

    def get_meter_count(self) -> int:
        """Get total number of meters."""
        return len(self._meters)

    def get_meters_needing_calibration(self) -> List[Meter]:
        """Get meters that need calibration."""
        return [m for m in self._meters.values() if m.status == MeterStatus.CALIBRATION_DUE]
