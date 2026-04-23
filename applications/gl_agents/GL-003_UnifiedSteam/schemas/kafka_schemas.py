"""
Kafka Schema Definitions for GL-003 UNIFIEDSTEAM

Provides Pydantic models for all Kafka message types used in
steam system data streaming.

Author: GL-003 Data Engineering Team
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================

class SensorQuality(str, Enum):
    """Sensor quality status."""
    GOOD = "GOOD"
    UNCERTAIN = "UNCERTAIN"
    BAD = "BAD"
    STALE = "STALE"
    NOT_CONNECTED = "NOT_CONNECTED"


class ValidationStatus(str, Enum):
    """Validation result status."""
    VALID = "VALID"
    RANGE_WARNING = "RANGE_WARNING"
    RANGE_ERROR = "RANGE_ERROR"
    RATE_WARNING = "RATE_WARNING"
    RATE_ERROR = "RATE_ERROR"
    CONSISTENCY_ERROR = "CONSISTENCY_ERROR"
    QUARANTINED = "QUARANTINED"


class RecommendationType(str, Enum):
    """Types of recommendations."""
    DESUPERHEATER_SETPOINT = "desuperheater_setpoint"
    TRAP_INSPECTION = "trap_inspection"
    TRAP_REPLACEMENT = "trap_replacement"
    PRV_ADJUSTMENT = "prv_adjustment"
    CONDENSATE_ROUTING = "condensate_routing"
    INSULATION_REPAIR = "insulation_repair"
    BLOWDOWN_ADJUSTMENT = "blowdown_adjustment"
    HEADER_PRESSURE = "header_pressure"
    MAINTENANCE_SCHEDULE = "maintenance_schedule"
    OPERATIONAL_CHANGE = "operational_change"


class RecommendationPriority(str, Enum):
    """Recommendation priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class EventType(str, Enum):
    """Event types."""
    ALARM = "alarm"
    MAINTENANCE = "maintenance"
    SETPOINT_CHANGE = "setpoint_change"
    MODE_CHANGE = "mode_change"
    CALIBRATION = "calibration"
    CONFIGURATION = "configuration"
    SYSTEM = "system"


# =============================================================================
# RAW SIGNAL SCHEMAS
# =============================================================================

class SensorMetadata(BaseModel):
    """Metadata about the sensor."""
    sensor_id: str
    sensor_type: str
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    accuracy_pct_fs: Optional[float] = Field(None, ge=0, le=100)
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    calibration_date: Optional[datetime] = None
    calibration_due: Optional[datetime] = None


class RawSignalSchema(BaseModel):
    """
    Raw signal from OT system.

    Represents minimally processed sensor data from OPC-UA or historian.
    """
    ts: datetime = Field(..., description="Timestamp (ISO 8601)")
    site: str = Field(..., description="Site identifier")
    area: str = Field(..., description="Area/unit identifier")
    asset: str = Field(..., description="Asset identifier")
    tag: str = Field(..., description="Tag name")
    value: float = Field(..., description="Signal value")
    unit: str = Field(..., description="Engineering unit")
    quality: SensorQuality = Field(default=SensorQuality.GOOD)
    sensor: Optional[SensorMetadata] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return {
            "ts": self.ts.isoformat(),
            "site": self.site,
            "area": self.area,
            "asset": self.asset,
            "tag": self.tag,
            "value": self.value,
            "unit": self.unit,
            "quality": {"status": self.quality.value, "flags": []},
            "sensor": self.sensor.dict() if self.sensor else None,
        }


# =============================================================================
# VALIDATED SIGNAL SCHEMAS
# =============================================================================

class QualityFlags(BaseModel):
    """Quality flags from validation."""
    in_range: bool = True
    rate_ok: bool = True
    consistent: bool = True
    flags: List[str] = Field(default_factory=list)


class ValidatedSignalSchema(BaseModel):
    """
    Validated and normalized signal.

    After range checks, rate-of-change validation, and consistency checks.
    """
    ts: datetime
    site: str
    area: str
    asset: str
    tag: str
    value: float
    unit: str
    original_value: float
    original_unit: str
    status: ValidationStatus
    quality_flags: QualityFlags
    sensor_accuracy_pct: Optional[float] = None
    derived_uncertainty: Optional[float] = None
    validation_hash: Optional[str] = None

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return {
            "ts": self.ts.isoformat(),
            "site": self.site,
            "area": self.area,
            "asset": self.asset,
            "tag": self.tag,
            "value": self.value,
            "unit": self.unit,
            "original_value": self.original_value,
            "original_unit": self.original_unit,
            "status": self.status.value,
            "quality_flags": self.quality_flags.dict(),
            "sensor_accuracy_pct": self.sensor_accuracy_pct,
            "derived_uncertainty": self.derived_uncertainty,
            "validation_hash": self.validation_hash,
        }


# =============================================================================
# FEATURE SCHEMAS
# =============================================================================

class FeatureSchema(BaseModel):
    """
    Base feature schema for ML models.
    """
    ts: datetime
    site: str
    area: str
    asset: str
    feature_set_id: str
    feature_version: str
    features: Dict[str, float]
    feature_quality: Dict[str, float] = Field(default_factory=dict)
    computation_hash: Optional[str] = None

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return {
            "ts": self.ts.isoformat(),
            "site": self.site,
            "area": self.area,
            "asset": self.asset,
            "feature_set_id": self.feature_set_id,
            "feature_version": self.feature_version,
            "features": self.features,
            "feature_quality": self.feature_quality,
            "computation_hash": self.computation_hash,
        }


class TrapFeatureSchema(BaseModel):
    """
    Feature schema for steam trap models.
    """
    ts: datetime
    site: str
    area: str
    trap_id: str
    trap_type: str

    # Acoustic features
    acoustic_rms_db: Optional[float] = None
    acoustic_peak_freq_hz: Optional[float] = None
    acoustic_spectral_entropy: Optional[float] = None
    acoustic_crest_factor: Optional[float] = None

    # Process features
    inlet_pressure_kpa: Optional[float] = None
    outlet_pressure_kpa: Optional[float] = None
    pressure_differential_kpa: Optional[float] = None
    inlet_temperature_c: Optional[float] = None

    # Derived features
    trap_age_days: Optional[int] = None
    time_since_service_days: Optional[int] = None
    operating_hours: Optional[float] = None
    cycle_count: Optional[int] = None

    # Context
    process_load_pct: Optional[float] = None
    ambient_temperature_c: Optional[float] = None

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return self.dict(exclude_none=True)


class HeaderFeatureSchema(BaseModel):
    """
    Feature schema for steam header models.
    """
    ts: datetime
    site: str
    area: str
    header_id: str
    header_pressure_class: str

    # Measurements
    pressure_kpa: float
    temperature_c: float
    flow_kg_s: Optional[float] = None

    # Computed
    saturation_temp_c: Optional[float] = None
    superheat_c: Optional[float] = None
    steam_quality: Optional[float] = None
    enthalpy_kj_kg: Optional[float] = None
    density_kg_m3: Optional[float] = None

    # KPIs
    pressure_stability_pct: Optional[float] = None
    demand_variability_pct: Optional[float] = None

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return self.dict(exclude_none=True)


class DesuperheaterFeatureSchema(BaseModel):
    """
    Feature schema for desuperheater models.
    """
    ts: datetime
    site: str
    area: str
    desuperheater_id: str

    # Inlet conditions
    inlet_pressure_kpa: float
    inlet_temperature_c: float
    inlet_flow_kg_s: Optional[float] = None
    inlet_enthalpy_kj_kg: Optional[float] = None

    # Outlet conditions
    outlet_pressure_kpa: Optional[float] = None
    outlet_temperature_c: float
    target_temperature_c: float

    # Spray water
    spray_flow_kg_s: Optional[float] = None
    spray_temperature_c: Optional[float] = None
    spray_valve_position_pct: Optional[float] = None

    # Computed
    superheat_inlet_c: Optional[float] = None
    superheat_outlet_c: Optional[float] = None
    approach_to_saturation_c: Optional[float] = None
    temp_deviation_from_target_c: Optional[float] = None

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return self.dict(exclude_none=True)


# =============================================================================
# COMPUTED PROPERTIES SCHEMAS
# =============================================================================

class SteamPropertiesComputed(BaseModel):
    """
    Computed steam properties from IAPWS-IF97.
    """
    ts: datetime
    site: str
    area: str
    asset: str

    # Input conditions
    pressure_kpa: float
    temperature_c: float

    # Computed properties
    enthalpy_kj_kg: float
    entropy_kj_kg_k: float
    specific_volume_m3_kg: float
    density_kg_m3: float
    saturation_temp_c: float
    superheat_c: float
    steam_quality: Optional[float] = None
    if97_region: int

    # Uncertainty
    enthalpy_uncertainty: Optional[float] = None
    computation_hash: str

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return {
            "ts": self.ts.isoformat(),
            "site": self.site,
            "area": self.area,
            "asset": self.asset,
            "input": {
                "pressure_kpa": self.pressure_kpa,
                "temperature_c": self.temperature_c,
            },
            "properties": {
                "enthalpy_kj_kg": self.enthalpy_kj_kg,
                "entropy_kj_kg_k": self.entropy_kj_kg_k,
                "specific_volume_m3_kg": self.specific_volume_m3_kg,
                "density_kg_m3": self.density_kg_m3,
                "saturation_temp_c": self.saturation_temp_c,
                "superheat_c": self.superheat_c,
                "steam_quality": self.steam_quality,
                "if97_region": self.if97_region,
            },
            "uncertainty": {
                "enthalpy": self.enthalpy_uncertainty,
            },
            "computation_hash": self.computation_hash,
        }


class EnthalpyBalanceComputed(BaseModel):
    """
    Computed enthalpy balance results.
    """
    ts: datetime
    site: str
    area: str
    balance_node: str

    # Mass balance
    mass_in_kg_s: float
    mass_out_kg_s: float
    mass_balance_kg_s: float

    # Energy balance
    enthalpy_in_kw: float
    enthalpy_out_kw: float
    heat_loss_kw: float
    heat_added_kw: float
    energy_balance_kw: float

    # Closure
    mass_closure_pct: float
    energy_closure_pct: float

    # Uncertainty
    mass_uncertainty_pct: Optional[float] = None
    energy_uncertainty_pct: Optional[float] = None
    computation_hash: str

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return {
            "ts": self.ts.isoformat(),
            "site": self.site,
            "area": self.area,
            "balance_node": self.balance_node,
            "mass_balance": {
                "in_kg_s": self.mass_in_kg_s,
                "out_kg_s": self.mass_out_kg_s,
                "balance_kg_s": self.mass_balance_kg_s,
                "closure_pct": self.mass_closure_pct,
            },
            "energy_balance": {
                "in_kw": self.enthalpy_in_kw,
                "out_kw": self.enthalpy_out_kw,
                "loss_kw": self.heat_loss_kw,
                "added_kw": self.heat_added_kw,
                "balance_kw": self.energy_balance_kw,
                "closure_pct": self.energy_closure_pct,
            },
            "uncertainty": {
                "mass_pct": self.mass_uncertainty_pct,
                "energy_pct": self.energy_uncertainty_pct,
            },
            "computation_hash": self.computation_hash,
        }


class KPIComputed(BaseModel):
    """
    Computed KPIs for steam system.
    """
    ts: datetime
    site: str
    area: str
    kpi_type: str

    # KPI value
    value: float
    unit: str
    target: Optional[float] = None
    baseline: Optional[float] = None

    # Performance
    vs_target_pct: Optional[float] = None
    vs_baseline_pct: Optional[float] = None
    trend: str = "neutral"  # up, down, neutral

    # Context
    period_start: datetime
    period_end: datetime
    sample_count: int

    # Uncertainty
    confidence_lower: Optional[float] = None
    confidence_upper: Optional[float] = None

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return {
            "ts": self.ts.isoformat(),
            "site": self.site,
            "area": self.area,
            "kpi_type": self.kpi_type,
            "value": self.value,
            "unit": self.unit,
            "target": self.target,
            "baseline": self.baseline,
            "performance": {
                "vs_target_pct": self.vs_target_pct,
                "vs_baseline_pct": self.vs_baseline_pct,
                "trend": self.trend,
            },
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
                "sample_count": self.sample_count,
            },
            "confidence": {
                "lower": self.confidence_lower,
                "upper": self.confidence_upper,
            },
        }


class ComputedPropertiesSchema(BaseModel):
    """
    Combined computed properties message.
    """
    ts: datetime
    site: str
    area: str
    computation_id: str

    steam_properties: List[SteamPropertiesComputed] = Field(default_factory=list)
    enthalpy_balances: List[EnthalpyBalanceComputed] = Field(default_factory=list)
    kpis: List[KPIComputed] = Field(default_factory=list)

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return {
            "ts": self.ts.isoformat(),
            "site": self.site,
            "area": self.area,
            "computation_id": self.computation_id,
            "steam_properties": [p.to_kafka_dict() for p in self.steam_properties],
            "enthalpy_balances": [b.to_kafka_dict() for b in self.enthalpy_balances],
            "kpis": [k.to_kafka_dict() for k in self.kpis],
        }


# =============================================================================
# RECOMMENDATION SCHEMAS
# =============================================================================

class RecommendationSchema(BaseModel):
    """
    Optimization recommendation message.
    """
    ts: datetime
    recommendation_id: str
    site: str
    area: str
    asset: str

    # Recommendation details
    recommendation_type: RecommendationType
    priority: RecommendationPriority
    action: str
    rationale: str

    # Expected impact
    expected_steam_savings_kg_hr: Optional[float] = None
    expected_energy_savings_kw: Optional[float] = None
    expected_cost_savings_usd: Optional[float] = None
    expected_co2e_reduction_kg: Optional[float] = None

    # Uncertainty
    impact_uncertainty_pct: Optional[float] = None
    confidence_score: float = Field(ge=0, le=1)

    # Constraints and checks
    constraints_checked: List[str] = Field(default_factory=list)
    safety_envelope_ok: bool = True

    # Verification
    verification_plan: str = ""
    escalation_path: str = ""

    # Explainability
    primary_drivers: List[str] = Field(default_factory=list)
    supporting_signals: List[str] = Field(default_factory=list)

    # Disposition
    disposition: str = "pending"  # pending, accepted, rejected, implemented
    disposition_by: Optional[str] = None
    disposition_at: Optional[datetime] = None
    disposition_notes: Optional[str] = None

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return {
            "ts": self.ts.isoformat(),
            "recommendation_id": self.recommendation_id,
            "site": self.site,
            "area": self.area,
            "asset": self.asset,
            "type": self.recommendation_type.value,
            "priority": self.priority.value,
            "action": self.action,
            "rationale": self.rationale,
            "impact": {
                "steam_savings_kg_hr": self.expected_steam_savings_kg_hr,
                "energy_savings_kw": self.expected_energy_savings_kw,
                "cost_savings_usd": self.expected_cost_savings_usd,
                "co2e_reduction_kg": self.expected_co2e_reduction_kg,
                "uncertainty_pct": self.impact_uncertainty_pct,
            },
            "confidence_score": self.confidence_score,
            "constraints_checked": self.constraints_checked,
            "safety_envelope_ok": self.safety_envelope_ok,
            "verification_plan": self.verification_plan,
            "escalation_path": self.escalation_path,
            "explainability": {
                "primary_drivers": self.primary_drivers,
                "supporting_signals": self.supporting_signals,
            },
            "disposition": {
                "status": self.disposition,
                "by": self.disposition_by,
                "at": self.disposition_at.isoformat() if self.disposition_at else None,
                "notes": self.disposition_notes,
            },
        }


# =============================================================================
# EVENT SCHEMAS
# =============================================================================

class EventSchema(BaseModel):
    """
    Base event schema.
    """
    ts: datetime
    event_id: str
    site: str
    area: str
    event_type: EventType
    source: str
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return {
            "ts": self.ts.isoformat(),
            "event_id": self.event_id,
            "site": self.site,
            "area": self.area,
            "event_type": self.event_type.value,
            "source": self.source,
            "description": self.description,
            "details": self.details,
        }


class AlarmSchema(BaseModel):
    """
    Alarm event schema.
    """
    ts: datetime
    alarm_id: str
    site: str
    area: str
    asset: str
    tag: str

    # Alarm details
    alarm_type: str
    severity: str  # critical, high, medium, low
    message: str
    value: float
    threshold: float
    unit: str

    # Status
    status: str = "active"  # active, acknowledged, cleared
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    cleared_at: Optional[datetime] = None

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return {
            "ts": self.ts.isoformat(),
            "alarm_id": self.alarm_id,
            "site": self.site,
            "area": self.area,
            "asset": self.asset,
            "tag": self.tag,
            "alarm_type": self.alarm_type,
            "severity": self.severity,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "unit": self.unit,
            "status": self.status,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "cleared_at": self.cleared_at.isoformat() if self.cleared_at else None,
        }


class MaintenanceEventSchema(BaseModel):
    """
    Maintenance event schema.
    """
    ts: datetime
    event_id: str
    site: str
    area: str
    asset: str

    # Event details
    event_type: str  # inspection, repair, replacement, calibration
    work_order_id: Optional[str] = None
    description: str

    # Before/after
    condition_before: Optional[str] = None
    condition_after: Optional[str] = None
    findings: List[str] = Field(default_factory=list)
    actions_taken: List[str] = Field(default_factory=list)

    # Cost
    labor_hours: Optional[float] = None
    parts_cost_usd: Optional[float] = None

    # Personnel
    performed_by: str = ""
    verified_by: Optional[str] = None

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return {
            "ts": self.ts.isoformat(),
            "event_id": self.event_id,
            "site": self.site,
            "area": self.area,
            "asset": self.asset,
            "event_type": self.event_type,
            "work_order_id": self.work_order_id,
            "description": self.description,
            "condition": {
                "before": self.condition_before,
                "after": self.condition_after,
            },
            "findings": self.findings,
            "actions_taken": self.actions_taken,
            "cost": {
                "labor_hours": self.labor_hours,
                "parts_cost_usd": self.parts_cost_usd,
            },
            "personnel": {
                "performed_by": self.performed_by,
                "verified_by": self.verified_by,
            },
        }


class SetpointChangeSchema(BaseModel):
    """
    Setpoint change event schema.
    """
    ts: datetime
    event_id: str
    site: str
    area: str
    asset: str
    tag: str

    # Change details
    old_value: float
    new_value: float
    unit: str
    reason: str

    # Authorization
    changed_by: str
    authorized_by: Optional[str] = None
    moc_reference: Optional[str] = None  # Management of Change reference

    # Recommendation link
    recommendation_id: Optional[str] = None

    def to_kafka_dict(self) -> Dict[str, Any]:
        """Convert to Kafka-ready dictionary."""
        return {
            "ts": self.ts.isoformat(),
            "event_id": self.event_id,
            "site": self.site,
            "area": self.area,
            "asset": self.asset,
            "tag": self.tag,
            "change": {
                "old_value": self.old_value,
                "new_value": self.new_value,
                "unit": self.unit,
            },
            "reason": self.reason,
            "authorization": {
                "changed_by": self.changed_by,
                "authorized_by": self.authorized_by,
                "moc_reference": self.moc_reference,
            },
            "recommendation_id": self.recommendation_id,
        }
