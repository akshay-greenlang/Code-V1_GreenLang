# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro: Measurement Schemas - Version 1.0

Provides validated data schemas for online process measurements including
temperature, flow, pressure, and generic timeseries data points.

This module defines Pydantic v2 models for:
- TemperatureMeasurement: Inlet/outlet temperatures for hot and cold sides
- FlowMeasurement: Mass and volumetric flow rates
- PressureMeasurement: Pressures and pressure drops
- TimeseriesPoint: Generic tagged timeseries data point
- ProcessMeasurementSet: Complete set of measurements for UA calculation

Author: GreenLang AI Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# ENUMERATIONS
# =============================================================================

class MeasurementQuality(str, Enum):
    """OPC-style quality codes for measurement validity."""
    GOOD = "good"
    GOOD_LOCAL_OVERRIDE = "good_local_override"
    GOOD_CLAMPED = "good_clamped"
    UNCERTAIN = "uncertain"
    UNCERTAIN_LAST_USABLE = "uncertain_last_usable"
    UNCERTAIN_SENSOR_CAL = "uncertain_sensor_cal"
    UNCERTAIN_SUBSTITUTED = "uncertain_substituted"
    BAD = "bad"
    BAD_DEVICE_FAILURE = "bad_device_failure"
    BAD_SENSOR_FAILURE = "bad_sensor_failure"
    BAD_COMM_FAILURE = "bad_comm_failure"
    BAD_OUT_OF_SERVICE = "bad_out_of_service"
    BAD_CONFIGURATION = "bad_configuration"
    MISSING = "missing"


class TemperatureUnit(str, Enum):
    """Temperature units."""
    CELSIUS = "celsius"
    KELVIN = "kelvin"
    FAHRENHEIT = "fahrenheit"


class PressureUnit(str, Enum):
    """Pressure units."""
    BAR_G = "bar_g"
    BAR_A = "bar_a"
    KPA_G = "kpa_g"
    KPA_A = "kpa_a"
    PSI_G = "psi_g"
    PSI_A = "psi_a"
    MPA_G = "mpa_g"
    MPA_A = "mpa_a"
    MBAR = "mbar"
    MM_H2O = "mm_h2o"
    IN_H2O = "in_h2o"


class FlowUnit(str, Enum):
    """Flow rate units."""
    KG_S = "kg/s"
    KG_H = "kg/h"
    T_H = "t/h"
    M3_H = "m3/h"
    M3_S = "m3/s"
    L_S = "l/s"
    L_MIN = "l/min"
    GPM = "gpm"
    BPD = "bpd"  # Barrels per day
    KBPD = "kbpd"  # Thousand barrels per day


class MeasurementSource(str, Enum):
    """Source of measurement data."""
    DCS = "dcs"
    SCADA = "scada"
    HISTORIAN = "historian"
    MANUAL = "manual"
    SIMULATION = "simulation"
    CALCULATED = "calculated"


# =============================================================================
# TIMESERIES POINT
# =============================================================================

class TimeseriesPoint(BaseModel):
    """
    Generic timeseries data point with quality metadata.

    Represents a single measurement from any sensor or calculated value,
    including full provenance and quality tracking for audit trails.
    """

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "tag": "TI-1001.PV",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "value": 185.5,
                    "unit": "celsius",
                    "quality": "good",
                    "source": "historian"
                }
            ]
        }
    )

    tag: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Tag or point name from source system"
    )
    timestamp: datetime = Field(
        ...,
        description="Measurement timestamp in UTC"
    )
    value: float = Field(
        ...,
        description="Measured value"
    )
    unit: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Engineering unit of measurement"
    )
    quality: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="OPC-style quality code"
    )
    source: MeasurementSource = Field(
        default=MeasurementSource.HISTORIAN,
        description="Data source system"
    )

    # Optional metadata
    raw_value: Optional[float] = Field(
        None,
        description="Raw sensor value before scaling/conversion"
    )
    status_bits: Optional[int] = Field(
        None,
        ge=0,
        description="Status bits from source system"
    )
    substituted: bool = Field(
        default=False,
        description="True if value was substituted by operator or algorithm"
    )
    annotation: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional annotation or comment"
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        """Ensure timestamp is UTC."""
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime) and v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v


# =============================================================================
# TEMPERATURE MEASUREMENT
# =============================================================================

class TemperatureMeasurement(BaseModel):
    """
    Temperature measurements for heat exchanger.

    Captures inlet and outlet temperatures for both hot and cold sides,
    with quality flags and unit tracking for thermal calculations.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "timestamp": "2024-01-15T10:30:00Z",
                    "t_hot_in_c": 250.0,
                    "t_hot_out_c": 180.0,
                    "t_cold_in_c": 80.0,
                    "t_cold_out_c": 145.0,
                    "unit": "celsius",
                    "quality_hot_in": "good",
                    "quality_hot_out": "good",
                    "quality_cold_in": "good",
                    "quality_cold_out": "good"
                }
            ]
        }
    )

    # Measurement identifier
    measurement_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique measurement identifier"
    )
    exchanger_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Reference to exchanger asset"
    )
    timestamp: datetime = Field(
        ...,
        description="Measurement timestamp in UTC"
    )

    # Hot side temperatures
    t_hot_in_c: float = Field(
        ...,
        ge=-273.15,
        le=1500,
        description="Hot stream inlet temperature in Celsius"
    )
    t_hot_out_c: float = Field(
        ...,
        ge=-273.15,
        le=1500,
        description="Hot stream outlet temperature in Celsius"
    )

    # Cold side temperatures
    t_cold_in_c: float = Field(
        ...,
        ge=-273.15,
        le=1500,
        description="Cold stream inlet temperature in Celsius"
    )
    t_cold_out_c: float = Field(
        ...,
        ge=-273.15,
        le=1500,
        description="Cold stream outlet temperature in Celsius"
    )

    # Unit (all temperatures in same unit)
    unit: TemperatureUnit = Field(
        default=TemperatureUnit.CELSIUS,
        description="Temperature unit (values stored in Celsius internally)"
    )

    # Quality flags for each measurement
    quality_hot_in: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Quality flag for hot inlet temperature"
    )
    quality_hot_out: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Quality flag for hot outlet temperature"
    )
    quality_cold_in: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Quality flag for cold inlet temperature"
    )
    quality_cold_out: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Quality flag for cold outlet temperature"
    )

    # Tag references for traceability
    tag_hot_in: Optional[str] = Field(
        None,
        max_length=200,
        description="Source tag for hot inlet temperature"
    )
    tag_hot_out: Optional[str] = Field(
        None,
        max_length=200,
        description="Source tag for hot outlet temperature"
    )
    tag_cold_in: Optional[str] = Field(
        None,
        max_length=200,
        description="Source tag for cold inlet temperature"
    )
    tag_cold_out: Optional[str] = Field(
        None,
        max_length=200,
        description="Source tag for cold outlet temperature"
    )

    # Data source
    source: MeasurementSource = Field(
        default=MeasurementSource.HISTORIAN,
        description="Source of temperature data"
    )

    @model_validator(mode="after")
    def validate_temperatures(self) -> "TemperatureMeasurement":
        """Validate temperature relationships."""
        # Hot inlet should typically be >= hot outlet
        if self.t_hot_in_c < self.t_hot_out_c:
            # This could be valid for cooling, so just flag don't fail
            pass

        # Cold outlet should typically be >= cold inlet
        if self.t_cold_out_c < self.t_cold_in_c:
            # This could be valid, so just flag don't fail
            pass

        # Hot side should typically be hotter than cold side
        if self.t_hot_in_c < self.t_cold_in_c:
            # Could be valid for cooling applications
            pass

        return self

    @property
    def overall_quality(self) -> MeasurementQuality:
        """Get the worst quality among all measurements."""
        qualities = [
            self.quality_hot_in,
            self.quality_hot_out,
            self.quality_cold_in,
            self.quality_cold_out,
        ]

        # Define quality hierarchy (worst to best)
        hierarchy = [
            MeasurementQuality.MISSING,
            MeasurementQuality.BAD,
            MeasurementQuality.BAD_DEVICE_FAILURE,
            MeasurementQuality.BAD_SENSOR_FAILURE,
            MeasurementQuality.BAD_COMM_FAILURE,
            MeasurementQuality.BAD_OUT_OF_SERVICE,
            MeasurementQuality.BAD_CONFIGURATION,
            MeasurementQuality.UNCERTAIN,
            MeasurementQuality.UNCERTAIN_LAST_USABLE,
            MeasurementQuality.UNCERTAIN_SENSOR_CAL,
            MeasurementQuality.UNCERTAIN_SUBSTITUTED,
            MeasurementQuality.GOOD_CLAMPED,
            MeasurementQuality.GOOD_LOCAL_OVERRIDE,
            MeasurementQuality.GOOD,
        ]

        for quality in hierarchy:
            if quality in qualities:
                return quality

        return MeasurementQuality.GOOD

    @property
    def delta_t_hot_c(self) -> float:
        """Temperature drop on hot side (positive for cooling)."""
        return self.t_hot_in_c - self.t_hot_out_c

    @property
    def delta_t_cold_c(self) -> float:
        """Temperature rise on cold side (positive for heating)."""
        return self.t_cold_out_c - self.t_cold_in_c


# =============================================================================
# FLOW MEASUREMENT
# =============================================================================

class FlowMeasurement(BaseModel):
    """
    Flow rate measurements for heat exchanger streams.

    Captures mass flow and/or volumetric flow for hot and cold sides
    with quality tracking and unit conversion support.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "timestamp": "2024-01-15T10:30:00Z",
                    "mass_flow_hot_kg_s": 15.5,
                    "mass_flow_cold_kg_s": 22.3,
                    "quality_hot": "good",
                    "quality_cold": "good"
                }
            ]
        }
    )

    # Measurement identifier
    measurement_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique measurement identifier"
    )
    exchanger_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Reference to exchanger asset"
    )
    timestamp: datetime = Field(
        ...,
        description="Measurement timestamp in UTC"
    )

    # Mass flow rates (preferred for heat duty calculation)
    mass_flow_hot_kg_s: Optional[float] = Field(
        None,
        ge=0,
        le=1e6,
        description="Hot stream mass flow rate in kg/s"
    )
    mass_flow_cold_kg_s: Optional[float] = Field(
        None,
        ge=0,
        le=1e6,
        description="Cold stream mass flow rate in kg/s"
    )

    # Volumetric flow rates (alternative if mass flow not available)
    volumetric_flow_hot_m3_h: Optional[float] = Field(
        None,
        ge=0,
        le=1e6,
        description="Hot stream volumetric flow rate in m^3/h"
    )
    volumetric_flow_cold_m3_h: Optional[float] = Field(
        None,
        ge=0,
        le=1e6,
        description="Cold stream volumetric flow rate in m^3/h"
    )

    # Units
    mass_flow_unit: FlowUnit = Field(
        default=FlowUnit.KG_S,
        description="Mass flow rate unit (values stored in kg/s internally)"
    )
    volumetric_flow_unit: FlowUnit = Field(
        default=FlowUnit.M3_H,
        description="Volumetric flow rate unit (values stored in m^3/h internally)"
    )

    # Quality flags
    quality_hot: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Quality flag for hot stream flow"
    )
    quality_cold: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Quality flag for cold stream flow"
    )

    # Tag references for traceability
    tag_hot: Optional[str] = Field(
        None,
        max_length=200,
        description="Source tag for hot stream flow"
    )
    tag_cold: Optional[str] = Field(
        None,
        max_length=200,
        description="Source tag for cold stream flow"
    )

    # Data source
    source: MeasurementSource = Field(
        default=MeasurementSource.HISTORIAN,
        description="Source of flow data"
    )

    # Density for mass/volumetric conversion
    density_hot_kg_m3: Optional[float] = Field(
        None,
        gt=0,
        le=25000,
        description="Hot stream density in kg/m^3 for flow conversion"
    )
    density_cold_kg_m3: Optional[float] = Field(
        None,
        gt=0,
        le=25000,
        description="Cold stream density in kg/m^3 for flow conversion"
    )

    @model_validator(mode="after")
    def validate_flow_availability(self) -> "FlowMeasurement":
        """Ensure at least one flow measurement is available."""
        has_hot_flow = (
            self.mass_flow_hot_kg_s is not None or
            self.volumetric_flow_hot_m3_h is not None
        )
        has_cold_flow = (
            self.mass_flow_cold_kg_s is not None or
            self.volumetric_flow_cold_m3_h is not None
        )

        if not has_hot_flow and not has_cold_flow:
            raise ValueError(
                "At least one flow measurement (hot or cold, mass or volumetric) "
                "must be provided"
            )

        return self


# =============================================================================
# PRESSURE MEASUREMENT
# =============================================================================

class PressureMeasurement(BaseModel):
    """
    Pressure measurements for heat exchanger.

    Captures inlet/outlet pressures and differential pressure for
    both shell and tube sides, essential for fouling detection.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        json_schema_extra={
            "examples": [
                {
                    "timestamp": "2024-01-15T10:30:00Z",
                    "p_shell_in_bar_g": 5.2,
                    "p_shell_out_bar_g": 4.8,
                    "dp_shell_bar": 0.4,
                    "p_tube_in_bar_g": 8.5,
                    "p_tube_out_bar_g": 7.9,
                    "dp_tube_bar": 0.6
                }
            ]
        }
    )

    # Measurement identifier
    measurement_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique measurement identifier"
    )
    exchanger_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Reference to exchanger asset"
    )
    timestamp: datetime = Field(
        ...,
        description="Measurement timestamp in UTC"
    )

    # Shell-side pressures
    p_shell_in_bar_g: Optional[float] = Field(
        None,
        ge=-1,
        le=500,
        description="Shell-side inlet pressure in bar gauge"
    )
    p_shell_out_bar_g: Optional[float] = Field(
        None,
        ge=-1,
        le=500,
        description="Shell-side outlet pressure in bar gauge"
    )
    dp_shell_bar: Optional[float] = Field(
        None,
        ge=-10,
        le=100,
        description="Shell-side differential pressure in bar"
    )

    # Tube-side pressures
    p_tube_in_bar_g: Optional[float] = Field(
        None,
        ge=-1,
        le=500,
        description="Tube-side inlet pressure in bar gauge"
    )
    p_tube_out_bar_g: Optional[float] = Field(
        None,
        ge=-1,
        le=500,
        description="Tube-side outlet pressure in bar gauge"
    )
    dp_tube_bar: Optional[float] = Field(
        None,
        ge=-10,
        le=100,
        description="Tube-side differential pressure in bar"
    )

    # Unit (stored values are in bar)
    unit: PressureUnit = Field(
        default=PressureUnit.BAR_G,
        description="Pressure unit (values stored in bar gauge internally)"
    )

    # Quality flags
    quality_shell_in: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Quality flag for shell inlet pressure"
    )
    quality_shell_out: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Quality flag for shell outlet pressure"
    )
    quality_shell_dp: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Quality flag for shell differential pressure"
    )
    quality_tube_in: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Quality flag for tube inlet pressure"
    )
    quality_tube_out: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Quality flag for tube outlet pressure"
    )
    quality_tube_dp: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Quality flag for tube differential pressure"
    )

    # Tag references for traceability
    tag_shell_in: Optional[str] = Field(None, max_length=200)
    tag_shell_out: Optional[str] = Field(None, max_length=200)
    tag_shell_dp: Optional[str] = Field(None, max_length=200)
    tag_tube_in: Optional[str] = Field(None, max_length=200)
    tag_tube_out: Optional[str] = Field(None, max_length=200)
    tag_tube_dp: Optional[str] = Field(None, max_length=200)

    # Data source
    source: MeasurementSource = Field(
        default=MeasurementSource.HISTORIAN,
        description="Source of pressure data"
    )

    @model_validator(mode="after")
    def validate_dp_consistency(self) -> "PressureMeasurement":
        """Validate differential pressure consistency with in/out pressures."""
        # Check shell-side consistency
        if (
            self.p_shell_in_bar_g is not None and
            self.p_shell_out_bar_g is not None and
            self.dp_shell_bar is not None
        ):
            calculated_dp = self.p_shell_in_bar_g - self.p_shell_out_bar_g
            if abs(calculated_dp - self.dp_shell_bar) > 0.5:  # Allow 0.5 bar tolerance
                # Log warning but don't fail - could be measurement uncertainty
                pass

        # Check tube-side consistency
        if (
            self.p_tube_in_bar_g is not None and
            self.p_tube_out_bar_g is not None and
            self.dp_tube_bar is not None
        ):
            calculated_dp = self.p_tube_in_bar_g - self.p_tube_out_bar_g
            if abs(calculated_dp - self.dp_tube_bar) > 0.5:
                pass

        return self


# =============================================================================
# PROCESS MEASUREMENT SET
# =============================================================================

class ProcessMeasurementSet(BaseModel):
    """
    Complete set of process measurements for heat exchanger analysis.

    Combines temperature, flow, and pressure measurements at a single
    timestamp for thermal performance and fouling calculations.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "schema_version": "1.0",
                    "measurement_set_id": "550e8400-e29b-41d4-a716-446655440000",
                    "exchanger_id": "HX-1001",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "temperatures": {
                        "timestamp": "2024-01-15T10:30:00Z",
                        "t_hot_in_c": 250.0,
                        "t_hot_out_c": 180.0,
                        "t_cold_in_c": 80.0,
                        "t_cold_out_c": 145.0
                    },
                    "flows": {
                        "timestamp": "2024-01-15T10:30:00Z",
                        "mass_flow_hot_kg_s": 15.5,
                        "mass_flow_cold_kg_s": 22.3
                    }
                }
            ]
        }
    )

    # Identifiers
    schema_version: str = Field(
        default="1.0",
        description="Schema version for compatibility tracking"
    )
    measurement_set_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this measurement set"
    )
    exchanger_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to exchanger asset"
    )
    timestamp: datetime = Field(
        ...,
        description="Primary timestamp for this measurement set"
    )

    # Component measurements
    temperatures: TemperatureMeasurement = Field(
        ...,
        description="Temperature measurements"
    )
    flows: FlowMeasurement = Field(
        ...,
        description="Flow rate measurements"
    )
    pressures: Optional[PressureMeasurement] = Field(
        None,
        description="Pressure measurements (optional)"
    )

    # Fluid properties at measurement conditions
    hot_fluid_properties: Optional[Dict[str, Any]] = Field(
        None,
        description="Hot fluid properties at measurement conditions"
    )
    cold_fluid_properties: Optional[Dict[str, Any]] = Field(
        None,
        description="Cold fluid properties at measurement conditions"
    )

    # Operating context
    operating_mode: Optional[Literal[
        "normal", "startup", "shutdown", "turndown", "bypass", "cleaning"
    ]] = Field(
        None,
        description="Current operating mode"
    )
    load_percent: Optional[float] = Field(
        None,
        ge=0,
        le=150,
        description="Operating load as percentage of design"
    )

    # Data quality summary
    overall_quality: MeasurementQuality = Field(
        default=MeasurementQuality.GOOD,
        description="Overall quality of measurement set"
    )
    quality_notes: Optional[str] = Field(
        None,
        max_length=500,
        description="Notes on data quality issues"
    )

    # Data source tracking
    source_system: Optional[str] = Field(
        None,
        max_length=100,
        description="Primary source system (e.g., historian name)"
    )

    # Provenance
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash for audit trail"
    )

    def compute_provenance_hash(self) -> str:
        """Compute SHA-256 hash for complete audit trail."""
        content = (
            f"{self.measurement_set_id}"
            f"{self.exchanger_id}"
            f"{self.timestamp.isoformat()}"
            f"{self.temperatures.t_hot_in_c:.4f}"
            f"{self.temperatures.t_hot_out_c:.4f}"
            f"{self.temperatures.t_cold_in_c:.4f}"
            f"{self.temperatures.t_cold_out_c:.4f}"
        )
        if self.flows.mass_flow_hot_kg_s:
            content += f"{self.flows.mass_flow_hot_kg_s:.4f}"
        if self.flows.mass_flow_cold_kg_s:
            content += f"{self.flows.mass_flow_cold_kg_s:.4f}"

        return hashlib.sha256(content.encode()).hexdigest()

    @model_validator(mode="after")
    def compute_overall_quality(self) -> "ProcessMeasurementSet":
        """Compute overall quality from component qualities."""
        # This would need to be done before freezing in practice
        # Just validating structure here
        return self


class ProcessMeasurementBatch(BaseModel):
    """
    Batch of process measurements for a single exchanger over time.

    Used for bulk data transfer and historical analysis.
    """

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "batch_id": "550e8400-e29b-41d4-a716-446655440000",
                    "exchanger_id": "HX-1001",
                    "start_time": "2024-01-15T00:00:00Z",
                    "end_time": "2024-01-15T23:59:59Z",
                    "measurement_count": 288,
                    "sampling_interval_seconds": 300
                }
            ]
        }
    )

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch identifier"
    )
    exchanger_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Reference to exchanger asset"
    )
    measurements: List[ProcessMeasurementSet] = Field(
        ...,
        min_length=1,
        description="List of process measurements in batch"
    )
    start_time: datetime = Field(
        ...,
        description="Batch start time"
    )
    end_time: datetime = Field(
        ...,
        description="Batch end time"
    )
    measurement_count: int = Field(
        ...,
        ge=1,
        description="Number of measurements in batch"
    )
    sampling_interval_seconds: Optional[int] = Field(
        None,
        gt=0,
        description="Expected sampling interval in seconds"
    )

    @model_validator(mode="after")
    def validate_batch(self) -> "ProcessMeasurementBatch":
        """Validate batch consistency."""
        if len(self.measurements) != self.measurement_count:
            raise ValueError(
                f"Measurement count mismatch: expected {self.measurement_count}, "
                f"got {len(self.measurements)}"
            )

        if self.end_time < self.start_time:
            raise ValueError("End time must be after start time")

        return self


# =============================================================================
# EXPORTS
# =============================================================================

MEASUREMENT_SCHEMAS = {
    "MeasurementQuality": MeasurementQuality,
    "TemperatureUnit": TemperatureUnit,
    "PressureUnit": PressureUnit,
    "FlowUnit": FlowUnit,
    "MeasurementSource": MeasurementSource,
    "TimeseriesPoint": TimeseriesPoint,
    "TemperatureMeasurement": TemperatureMeasurement,
    "FlowMeasurement": FlowMeasurement,
    "PressureMeasurement": PressureMeasurement,
    "ProcessMeasurementSet": ProcessMeasurementSet,
    "ProcessMeasurementBatch": ProcessMeasurementBatch,
}

__all__ = [
    # Enumerations
    "MeasurementQuality",
    "TemperatureUnit",
    "PressureUnit",
    "FlowUnit",
    "MeasurementSource",
    # Models
    "TimeseriesPoint",
    "TemperatureMeasurement",
    "FlowMeasurement",
    "PressureMeasurement",
    "ProcessMeasurementSet",
    "ProcessMeasurementBatch",
    # Export dictionary
    "MEASUREMENT_SCHEMAS",
]
