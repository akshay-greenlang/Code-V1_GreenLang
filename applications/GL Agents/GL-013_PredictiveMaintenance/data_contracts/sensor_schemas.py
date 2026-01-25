"""GL-013 PredictiveMaintenance: Sensor Reading Schemas - Version 1.0"""
from __future__ import annotations
import hashlib, uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class SensorModality(str, Enum):
    VIBRATION = "vibration"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    CURRENT = "current"
    CONTEXT = "context"
    ENVIRONMENT = "environment"

class SensorQualityStatus(str, Enum):
    OK = "ok"
    SUSPECT = "suspect"
    BAD = "bad"
    MISSING = "missing"

class AxisType(str, Enum):
    X = "x"
    Y = "y"
    Z = "z"
    RADIAL = "radial"
    AXIAL = "axial"
    TANGENTIAL = "tangential"

class RunState(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    IDLE = "idle"
    TRIPPED = "tripped"

class CalibrationInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    calibration_id: str
    calibration_date: datetime
    calibration_due: Optional[datetime] = None
    sensitivity: Optional[float] = None
    offset: Optional[float] = None
    is_valid: bool = True

class SamplingInfo(BaseModel):
    model_config = ConfigDict(frozen=True)
    sampling_rate_hz: float = Field(..., gt=0, le=102400)
    samples_per_window: int = Field(..., gt=0)
    window_duration_ms: float = Field(..., gt=0)
    anti_alias_filter_hz: Optional[float] = None
    averaging_type: Optional[Literal["linear", "exponential", "peak_hold"]] = "linear"
    num_averages: int = Field(default=1, ge=1)

class SensorMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    sensor_id: str
    location: str
    axis: Optional[AxisType] = None
    calibration: Optional[CalibrationInfo] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None

class VibrationReading(BaseModel):
    model_config = ConfigDict(frozen=True)
    axis: AxisType
    velocity_mm_s: Optional[float] = Field(default=None, ge=0, le=500)
    acceleration_g: Optional[float] = Field(default=None, ge=0, le=100)
    displacement_um: Optional[float] = Field(default=None, ge=0, le=2000)
    peak_velocity_mm_s: Optional[float] = Field(default=None, ge=0)
    crest_factor: Optional[float] = Field(default=None, ge=1, le=20)
    kurtosis: Optional[float] = None
    quality: SensorQualityStatus = SensorQualityStatus.OK
    alarm_level: Optional[Literal["normal", "alert", "danger"]] = "normal"

class TemperatureReading(BaseModel):
    model_config = ConfigDict(frozen=True)
    temperature_c: float = Field(..., ge=-273.15, le=1000)
    temperature_type: Literal["bearing", "ambient", "process", "surface", "oil"] = "bearing"
    delta_t_from_ambient_c: Optional[float] = None
    rate_of_change_c_per_min: Optional[float] = None
    quality: SensorQualityStatus = SensorQualityStatus.OK
    alarm_level: Optional[Literal["normal", "warning", "alarm", "trip"]] = "normal"

class PressureReading(BaseModel):
    model_config = ConfigDict(frozen=True)
    pressure_bar: float = Field(..., ge=-1, le=1000)
    pressure_type: Literal["suction", "discharge", "differential", "casing", "oil"] = "discharge"
    differential_bar: Optional[float] = None
    quality: SensorQualityStatus = SensorQualityStatus.OK

class CurrentReading(BaseModel):
    model_config = ConfigDict(frozen=True)
    current_a: float = Field(..., ge=0, le=10000)
    phase: Literal["A", "B", "C", "average"] = "average"
    power_factor: Optional[float] = Field(default=None, ge=0, le=1)
    thd_percent: Optional[float] = Field(default=None, ge=0, le=100)
    unbalance_percent: Optional[float] = Field(default=None, ge=0, le=100)
    quality: SensorQualityStatus = SensorQualityStatus.OK

class ContextReading(BaseModel):
    model_config = ConfigDict(frozen=True)
    run_state: RunState
    load_percent: Optional[float] = Field(default=None, ge=0, le=150)
    speed_rpm: Optional[float] = Field(default=None, ge=0, le=100000)
    speed_percent: Optional[float] = Field(default=None, ge=0, le=150)
    flow_rate_m3h: Optional[float] = Field(default=None, ge=0)
    runtime_hours: Optional[float] = Field(default=None, ge=0)
    start_count: Optional[int] = Field(default=None, ge=0)
    quality: SensorQualityStatus = SensorQualityStatus.OK

class EnvironmentReading(BaseModel):
    model_config = ConfigDict(frozen=True)
    ambient_temp_c: Optional[float] = Field(default=None, ge=-60, le=80)
    humidity_percent: Optional[float] = Field(default=None, ge=0, le=100)
    atmospheric_pressure_mbar: Optional[float] = Field(default=None, ge=800, le=1200)
    dust_level: Optional[Literal["low", "medium", "high"]] = None
    quality: SensorQualityStatus = SensorQualityStatus.OK

class SensorReadingEvent(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True, extra="forbid")
    schema_version: str = Field(default="1.0")
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    asset_id: str = Field(..., min_length=1)
    component_id: Optional[str] = None
    timestamp: datetime
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None
    modality: SensorModality
    vibration_readings: Optional[List[VibrationReading]] = None
    temperature_readings: Optional[List[TemperatureReading]] = None
    pressure_readings: Optional[List[PressureReading]] = None
    current_readings: Optional[List[CurrentReading]] = None
    context_reading: Optional[ContextReading] = None
    environment_reading: Optional[EnvironmentReading] = None
    sensor_metadata: Optional[SensorMetadata] = None
    sampling_info: Optional[SamplingInfo] = None
    overall_quality: SensorQualityStatus = SensorQualityStatus.OK
    quality_notes: Optional[str] = Field(default=None, max_length=500)
    source_system: str = "unknown"
    provenance_hash: Optional[str] = None

class SensorReadingBatch(BaseModel):
    model_config = ConfigDict(frozen=True)
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    asset_id: str
    readings: List[SensorReadingEvent] = Field(..., min_length=1)
    batch_start: datetime
    batch_end: datetime
    reading_count: int = Field(default=0, ge=0)

SENSOR_READING_TYPES = {"VibrationReading": VibrationReading, "TemperatureReading": TemperatureReading, "PressureReading": PressureReading, "CurrentReading": CurrentReading, "ContextReading": ContextReading, "EnvironmentReading": EnvironmentReading, "SensorReadingEvent": SensorReadingEvent, "SensorReadingBatch": SensorReadingBatch}
__all__ = ["SensorModality", "SensorQualityStatus", "AxisType", "RunState", "CalibrationInfo", "SamplingInfo", "SensorMetadata", "VibrationReading", "TemperatureReading", "PressureReading", "CurrentReading", "ContextReading", "EnvironmentReading", "SensorReadingEvent", "SensorReadingBatch", "SENSOR_READING_TYPES"]
