"""
GL-007 FurnacePulse Pydantic Schemas

This module defines all Pydantic data models for the FurnacePulse
Furnace Performance Monitoring Agent. These schemas provide type-safe
data validation for telemetry ingestion, analytics processing, and
output generation.

Schema Categories:
    - Telemetry: Raw sensor data models (TelemetrySignal, TMTReading)
    - State: Furnace state representations (FurnaceState)
    - Alerts: Alert and notification models (HotspotAlert, MaintenanceAlert)
    - KPIs: Performance metric models (EfficiencyKPI, EfficiencyTrend)
    - Predictions: RUL and forecast models (RULPrediction)
    - Compliance: NFPA 86 compliance models (ComplianceEvidence, SafetyEvent)
    - Output: Agent output container (FurnacePulseOutput)

All schemas include:
    - Type hints for all fields
    - Field validators for data quality
    - Computed properties where applicable
    - SHA-256 provenance hashing support

Example:
    >>> from core.schemas import TelemetrySignal, SignalQuality
    >>> signal = TelemetrySignal(
    ...     tag_id="FIC-101.TMT",
    ...     value=875.5,
    ...     quality=SignalQuality.GOOD,
    ...     timestamp=datetime.now(timezone.utc)
    ... )
    >>> signal.is_valid
    True

Author: GreenLang Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator, computed_field, model_validator
import hashlib
import json

from core.config import (
    AlertTier,
    SignalQuality,
    FurnaceZone,
    OperatingMode,
    FlameQuality,
    TrendDirection,
    ComplianceStatus,
    HotspotSeverity,
)


# =============================================================================
# BASE SCHEMAS
# =============================================================================

class TelemetrySignal(BaseModel):
    """
    Base telemetry signal from OPC-UA or other data sources.

    Represents a single sensor reading with quality metadata
    for use in furnace monitoring calculations.

    Attributes:
        tag_id: Unique sensor tag identifier (e.g., "FIC-101.TMT")
        value: Numeric sensor value
        quality: Signal quality indicator
        timestamp: Reading timestamp (UTC)
        unit: Engineering unit (optional)
        source: Data source identifier (optional)

    Example:
        >>> signal = TelemetrySignal(
        ...     tag_id="TI-101",
        ...     value=875.5,
        ...     quality=SignalQuality.GOOD,
        ...     timestamp=datetime.now(timezone.utc)
        ... )
        >>> print(signal.is_usable)
        True
    """
    tag_id: str = Field(
        ...,
        description="Unique sensor tag identifier",
        min_length=1,
        max_length=100,
        examples=["FIC-101.TMT", "TI-201", "PI-301"]
    )
    value: float = Field(
        ...,
        description="Numeric sensor value"
    )
    quality: SignalQuality = Field(
        default=SignalQuality.GOOD,
        description="Signal quality indicator"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp in UTC"
    )
    unit: Optional[str] = Field(
        default=None,
        description="Engineering unit (e.g., C, kPa, kg/h)"
    )
    source: Optional[str] = Field(
        default=None,
        description="Data source identifier"
    )

    @field_validator("timestamp")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """Ensure timestamp is UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

    @computed_field
    @property
    def is_usable(self) -> bool:
        """Check if signal is usable for calculations."""
        return self.quality.is_usable

    @computed_field
    @property
    def age_seconds(self) -> float:
        """Get signal age in seconds."""
        now = datetime.now(timezone.utc)
        return (now - self.timestamp).total_seconds()

    def to_hash_string(self) -> str:
        """Generate string for provenance hashing."""
        return f"{self.tag_id}:{self.value}:{self.timestamp.isoformat()}"


# =============================================================================
# TMT SCHEMAS
# =============================================================================

class TMTReading(BaseModel):
    """
    Tube Metal Temperature reading for fired heater monitoring.

    TMT readings are critical for heater tube life management per
    API 530 and API 560 standards. Rate of rise is monitored to
    detect abnormal heating conditions.

    Attributes:
        tube_id: Unique tube identifier
        temperature_C: Current tube metal temperature in Celsius
        rate_of_rise_C_min: Temperature rate of rise in C/min
        zone: Furnace zone location
        design_limit_C: Maximum allowable TMT per design
        signal_quality: Quality of temperature measurement
        timestamp: Reading timestamp

    Example:
        >>> tmt = TMTReading(
        ...     tube_id="R-1-01",
        ...     temperature_C=875.0,
        ...     rate_of_rise_C_min=2.5,
        ...     zone=FurnaceZone.RADIANT,
        ...     design_limit_C=950.0
        ... )
        >>> print(tmt.utilization_percent)
        92.1
    """
    tube_id: str = Field(
        ...,
        description="Unique tube identifier",
        min_length=1,
        examples=["R-1-01", "C-2-05", "S-1-03"]
    )
    temperature_C: float = Field(
        ...,
        description="Current tube metal temperature in Celsius",
        ge=0.0,
        le=1200.0
    )
    rate_of_rise_C_min: float = Field(
        default=0.0,
        description="Temperature rate of rise in C/min",
        ge=-50.0,
        le=50.0
    )
    zone: FurnaceZone = Field(
        ...,
        description="Furnace zone location"
    )
    design_limit_C: float = Field(
        default=950.0,
        description="Maximum allowable TMT per design",
        ge=500.0,
        le=1100.0
    )
    signal_quality: SignalQuality = Field(
        default=SignalQuality.GOOD,
        description="Quality of temperature measurement"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp"
    )

    @computed_field
    @property
    def utilization_percent(self) -> float:
        """Calculate TMT utilization as percentage of design limit."""
        if self.design_limit_C <= 0:
            return 0.0
        return round((self.temperature_C / self.design_limit_C) * 100.0, 1)

    @computed_field
    @property
    def margin_C(self) -> float:
        """Calculate margin to design limit in Celsius."""
        return round(self.design_limit_C - self.temperature_C, 1)

    @computed_field
    @property
    def is_critical(self) -> bool:
        """Check if TMT is in critical range (>95% of design)."""
        return self.utilization_percent > 95.0

    def get_alert_tier(self, warning_C: float = 900.0, urgent_C: float = 950.0) -> Optional[AlertTier]:
        """
        Determine alert tier based on TMT thresholds.

        Args:
            warning_C: Temperature threshold for WARNING tier
            urgent_C: Temperature threshold for URGENT tier

        Returns:
            Alert tier or None if within normal range
        """
        if self.temperature_C >= urgent_C:
            return AlertTier.URGENT
        elif self.temperature_C >= warning_C:
            return AlertTier.WARNING
        elif self.utilization_percent > 85.0:
            return AlertTier.ADVISORY
        return None


# =============================================================================
# STATE SCHEMAS
# =============================================================================

class FlameStatus(BaseModel):
    """
    Flame detection and quality status for burner monitoring.

    Per NFPA 86 flame supervision requirements.

    Attributes:
        burner_id: Burner identifier
        flame_present: Whether flame is detected
        uv_intensity: UV sensor intensity reading
        ir_intensity: IR sensor intensity reading
        flame_quality: Assessed flame quality
        timestamp: Status timestamp
    """
    burner_id: str = Field(..., description="Burner identifier")
    flame_present: bool = Field(..., description="Flame detected flag")
    uv_intensity: float = Field(
        default=0.0,
        description="UV sensor intensity (0-100)",
        ge=0.0,
        le=100.0
    )
    ir_intensity: float = Field(
        default=0.0,
        description="IR sensor intensity (0-100)",
        ge=0.0,
        le=100.0
    )
    flame_quality: FlameQuality = Field(
        default=FlameQuality.STABLE,
        description="Assessed flame quality"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @computed_field
    @property
    def is_safe(self) -> bool:
        """Check if flame status is safe for operation."""
        return self.flame_present and self.flame_quality.is_safe


class DraftReading(BaseModel):
    """
    Furnace draft pressure reading.

    Draft control is critical for combustion efficiency and safety.

    Attributes:
        location: Measurement location
        pressure_Pa: Draft pressure in Pascals
        is_balanced: Whether draft is balanced
        timestamp: Reading timestamp
    """
    location: str = Field(..., description="Measurement location")
    pressure_Pa: float = Field(
        ...,
        description="Draft pressure in Pascals",
        ge=-500.0,
        le=500.0
    )
    is_balanced: bool = Field(
        default=True,
        description="Draft balance status"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @computed_field
    @property
    def is_negative(self) -> bool:
        """Check if draft is negative (suction)."""
        return self.pressure_Pa < 0

    @computed_field
    @property
    def is_in_range(self) -> bool:
        """Check if draft is in normal operating range."""
        return -250.0 <= self.pressure_Pa <= 25.0


class FuelConsumption(BaseModel):
    """
    Fuel consumption data for efficiency calculations.

    Attributes:
        fuel_type: Type of fuel
        flow_rate_kg_h: Mass flow rate in kg/h
        heating_value_MJ_kg: Lower heating value in MJ/kg
        pressure_kPa: Fuel pressure in kPa
        temperature_C: Fuel temperature in Celsius
        timestamp: Reading timestamp
    """
    fuel_type: str = Field(
        ...,
        description="Fuel type (natural_gas, fuel_oil, hydrogen, mixed)"
    )
    flow_rate_kg_h: float = Field(
        ...,
        description="Mass flow rate in kg/h",
        ge=0.0
    )
    heating_value_MJ_kg: float = Field(
        default=50.0,
        description="Lower heating value in MJ/kg",
        ge=10.0,
        le=150.0
    )
    pressure_kPa: float = Field(
        default=200.0,
        description="Fuel pressure in kPa",
        ge=0.0
    )
    temperature_C: float = Field(
        default=25.0,
        description="Fuel temperature in Celsius"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @computed_field
    @property
    def heat_input_MW(self) -> float:
        """Calculate heat input in MW."""
        # Q = m_dot * LHV / 3600
        return round((self.flow_rate_kg_h * self.heating_value_MJ_kg) / 3600.0, 3)


class FurnaceState(BaseModel):
    """
    Complete furnace state snapshot for monitoring and analysis.

    Aggregates all sensor readings into a unified state representation
    for efficiency calculations and compliance checking.

    Attributes:
        furnace_id: Unique furnace identifier
        operating_mode: Current operating mode
        zone_temperatures: Temperature readings by zone
        fuel_consumption: Current fuel consumption data
        production_throughput_kg_h: Current production rate
        tmt_readings: List of TMT readings
        flame_status: List of flame status readings
        draft_readings: List of draft pressure readings
        timestamp: State snapshot timestamp
        provenance_hash: SHA-256 hash for audit trail

    Example:
        >>> state = FurnaceState(
        ...     furnace_id="FH-101",
        ...     operating_mode=OperatingMode.NORMAL,
        ...     zone_temperatures={"RADIANT": 850.0, "CONVECTION": 650.0},
        ...     fuel_consumption=FuelConsumption(...),
        ...     tmt_readings=[TMTReading(...), ...]
        ... )
    """
    furnace_id: str = Field(
        ...,
        description="Unique furnace identifier",
        min_length=1
    )
    operating_mode: OperatingMode = Field(
        default=OperatingMode.NORMAL,
        description="Current operating mode"
    )
    zone_temperatures: Dict[str, float] = Field(
        default_factory=dict,
        description="Temperature readings by zone (zone_name: temp_C)"
    )
    fuel_consumption: Optional[FuelConsumption] = Field(
        default=None,
        description="Current fuel consumption data"
    )
    production_throughput_kg_h: float = Field(
        default=0.0,
        description="Current production rate in kg/h",
        ge=0.0
    )
    tmt_readings: List[TMTReading] = Field(
        default_factory=list,
        description="List of TMT readings"
    )
    flame_status: List[FlameStatus] = Field(
        default_factory=list,
        description="List of flame status readings"
    )
    draft_readings: List[DraftReading] = Field(
        default_factory=list,
        description="List of draft pressure readings"
    )
    excess_air_percent: Optional[float] = Field(
        default=None,
        description="Current excess air percentage",
        ge=0.0,
        le=100.0
    )
    stack_temperature_C: Optional[float] = Field(
        default=None,
        description="Stack exhaust temperature in Celsius"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="State snapshot timestamp"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit trail"
    )

    @computed_field
    @property
    def max_tmt_C(self) -> Optional[float]:
        """Get maximum TMT from all readings."""
        if not self.tmt_readings:
            return None
        return max(r.temperature_C for r in self.tmt_readings)

    @computed_field
    @property
    def tmt_count(self) -> int:
        """Get count of TMT readings."""
        return len(self.tmt_readings)

    @computed_field
    @property
    def all_flames_present(self) -> bool:
        """Check if all flames are present."""
        if not self.flame_status:
            return False
        return all(f.flame_present for f in self.flame_status)

    @computed_field
    @property
    def heat_input_MW(self) -> Optional[float]:
        """Get heat input from fuel consumption."""
        if self.fuel_consumption is None:
            return None
        return self.fuel_consumption.heat_input_MW

    def calculate_provenance_hash(self) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Returns:
            SHA-256 hex digest of state data
        """
        # Build deterministic string representation
        data_str = json.dumps({
            "furnace_id": self.furnace_id,
            "operating_mode": self.operating_mode.value,
            "zone_temperatures": self.zone_temperatures,
            "max_tmt_C": self.max_tmt_C,
            "production_throughput_kg_h": self.production_throughput_kg_h,
            "timestamp": self.timestamp.isoformat(),
        }, sort_keys=True)

        return hashlib.sha256(data_str.encode()).hexdigest()

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to calculate provenance hash."""
        if self.provenance_hash is None:
            object.__setattr__(self, 'provenance_hash', self.calculate_provenance_hash())


# =============================================================================
# ALERT SCHEMAS
# =============================================================================

class HotspotAlert(BaseModel):
    """
    Thermal hotspot detection alert from IR camera analysis.

    Hotspots indicate potential refractory damage or tube overheating
    requiring maintenance attention.

    Attributes:
        alert_id: Unique alert identifier
        tier: Alert severity tier
        furnace_id: Affected furnace
        location: Hotspot location description
        tmt_readings: Associated TMT readings
        ir_snapshot_ref: Reference to IR camera snapshot
        max_temperature_C: Maximum detected temperature
        delta_t_C: Temperature differential from baseline
        severity: Hotspot severity classification
        confidence_percent: Detection confidence
        timestamp: Alert generation timestamp
        recommended_action: Suggested maintenance action

    Example:
        >>> alert = HotspotAlert(
        ...     alert_id="HS-2024-001",
        ...     tier=AlertTier.WARNING,
        ...     furnace_id="FH-101",
        ...     location="Radiant section, east wall",
        ...     max_temperature_C=925.0,
        ...     delta_t_C=75.0,
        ...     severity=HotspotSeverity.HIGH,
        ...     confidence_percent=95.5
        ... )
    """
    alert_id: str = Field(
        ...,
        description="Unique alert identifier"
    )
    tier: AlertTier = Field(
        ...,
        description="Alert severity tier"
    )
    furnace_id: str = Field(
        ...,
        description="Affected furnace identifier"
    )
    location: str = Field(
        ...,
        description="Hotspot location description"
    )
    tmt_readings: List[TMTReading] = Field(
        default_factory=list,
        description="Associated TMT readings"
    )
    ir_snapshot_ref: Optional[str] = Field(
        default=None,
        description="Reference to IR camera snapshot"
    )
    max_temperature_C: float = Field(
        ...,
        description="Maximum detected temperature",
        ge=0.0
    )
    delta_t_C: float = Field(
        ...,
        description="Temperature differential from baseline",
        ge=0.0
    )
    severity: HotspotSeverity = Field(
        ...,
        description="Hotspot severity classification"
    )
    confidence_percent: float = Field(
        default=100.0,
        description="Detection confidence percentage",
        ge=0.0,
        le=100.0
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Alert generation timestamp"
    )
    recommended_action: Optional[str] = Field(
        default=None,
        description="Suggested maintenance action"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit trail"
    )

    @model_validator(mode='after')
    def set_provenance(self) -> 'HotspotAlert':
        """Calculate provenance hash after validation."""
        if self.provenance_hash is None:
            data_str = f"{self.alert_id}:{self.furnace_id}:{self.max_temperature_C}:{self.timestamp.isoformat()}"
            object.__setattr__(self, 'provenance_hash', hashlib.sha256(data_str.encode()).hexdigest())
        return self


class MaintenanceAlert(BaseModel):
    """
    Maintenance alert for scheduled or urgent work.

    Integrates with CMMS for work order generation.

    Attributes:
        alert_id: Unique alert identifier
        tier: Alert severity tier
        furnace_id: Affected furnace
        component: Component requiring attention
        description: Alert description
        recommended_action: Suggested maintenance action
        due_date: Target completion date
        work_order_id: CMMS work order reference
        timestamp: Alert generation timestamp
    """
    alert_id: str = Field(..., description="Unique alert identifier")
    tier: AlertTier = Field(..., description="Alert severity tier")
    furnace_id: str = Field(..., description="Affected furnace")
    component: str = Field(..., description="Component requiring attention")
    description: str = Field(..., description="Alert description")
    recommended_action: str = Field(..., description="Suggested maintenance action")
    due_date: Optional[datetime] = Field(
        default=None,
        description="Target completion date"
    )
    work_order_id: Optional[str] = Field(
        default=None,
        description="CMMS work order reference"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# =============================================================================
# KPI SCHEMAS
# =============================================================================

class EfficiencyKPI(BaseModel):
    """
    Furnace efficiency Key Performance Indicators.

    All calculations are deterministic (zero-hallucination) using
    standard thermodynamic formulas per API 560.

    Attributes:
        furnace_id: Furnace identifier
        thermal_efficiency_percent: Overall thermal efficiency
        specific_fuel_consumption_MJ_kg: Fuel per unit production
        excess_air_percent: Combustion excess air percentage
        stack_loss_percent: Stack heat loss percentage
        radiation_loss_percent: Radiation heat loss percentage
        calculation_timestamp: KPI calculation timestamp
        calculation_hash: SHA-256 hash of calculation inputs

    Example:
        >>> kpi = EfficiencyKPI(
        ...     furnace_id="FH-101",
        ...     thermal_efficiency_percent=87.5,
        ...     specific_fuel_consumption_MJ_kg=3.2,
        ...     excess_air_percent=15.0,
        ...     stack_loss_percent=8.5
        ... )
        >>> print(kpi.efficiency_grade)
        "A"
    """
    furnace_id: str = Field(..., description="Furnace identifier")
    thermal_efficiency_percent: float = Field(
        ...,
        description="Overall thermal efficiency",
        ge=0.0,
        le=100.0
    )
    specific_fuel_consumption_MJ_kg: float = Field(
        ...,
        description="Fuel consumption per unit production (MJ/kg)",
        ge=0.0
    )
    excess_air_percent: float = Field(
        ...,
        description="Combustion excess air percentage",
        ge=0.0,
        le=100.0
    )
    stack_loss_percent: float = Field(
        default=0.0,
        description="Stack heat loss percentage",
        ge=0.0,
        le=50.0
    )
    radiation_loss_percent: float = Field(
        default=2.0,
        description="Radiation heat loss percentage",
        ge=0.0,
        le=10.0
    )
    heat_absorbed_MW: Optional[float] = Field(
        default=None,
        description="Heat absorbed by process in MW"
    )
    heat_released_MW: Optional[float] = Field(
        default=None,
        description="Heat released from combustion in MW"
    )
    calculation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="KPI calculation timestamp"
    )
    calculation_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of calculation inputs"
    )

    @computed_field
    @property
    def efficiency_grade(self) -> str:
        """
        Get efficiency grade based on thermal efficiency.

        Returns:
            Grade from A (best) to F (worst)
        """
        eff = self.thermal_efficiency_percent
        if eff >= 90.0:
            return "A"
        elif eff >= 85.0:
            return "B"
        elif eff >= 80.0:
            return "C"
        elif eff >= 75.0:
            return "D"
        else:
            return "F"

    @computed_field
    @property
    def is_optimal(self) -> bool:
        """Check if efficiency is in optimal range."""
        return (
            self.thermal_efficiency_percent >= 85.0 and
            10.0 <= self.excess_air_percent <= 20.0
        )


class EfficiencyTrend(BaseModel):
    """
    Efficiency trend analysis over time period.

    Attributes:
        furnace_id: Furnace identifier
        trend_period_days: Analysis period in days
        trend_direction: Trend direction indicator
        degradation_rate_percent_month: Monthly degradation rate
        forecast_30d_percent: 30-day efficiency forecast
        confidence_percent: Forecast confidence
        data_points_count: Number of data points in analysis
        timestamp: Analysis timestamp
    """
    furnace_id: str = Field(..., description="Furnace identifier")
    trend_period_days: int = Field(
        ...,
        description="Analysis period in days",
        ge=1
    )
    trend_direction: TrendDirection = Field(
        ...,
        description="Trend direction indicator"
    )
    degradation_rate_percent_month: float = Field(
        default=0.0,
        description="Monthly degradation rate (negative = improvement)"
    )
    forecast_30d_percent: float = Field(
        ...,
        description="30-day efficiency forecast",
        ge=0.0,
        le=100.0
    )
    confidence_percent: float = Field(
        default=95.0,
        description="Forecast confidence",
        ge=0.0,
        le=100.0
    )
    data_points_count: int = Field(
        default=0,
        description="Number of data points in analysis",
        ge=0
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# =============================================================================
# RUL PREDICTION SCHEMAS
# =============================================================================

class ConfidenceBounds(BaseModel):
    """
    Confidence interval bounds for predictions.

    Attributes:
        lower: Lower bound value
        upper: Upper bound value
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    """
    lower: float = Field(..., description="Lower bound value")
    upper: float = Field(..., description="Upper bound value")
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level",
        ge=0.0,
        le=1.0
    )

    @field_validator("upper")
    @classmethod
    def upper_gte_lower(cls, v: float, info) -> float:
        """Validate upper bound is >= lower bound."""
        if info.data and "lower" in info.data:
            if v < info.data["lower"]:
                raise ValueError("upper must be >= lower")
        return v


class RULPrediction(BaseModel):
    """
    Remaining Useful Life prediction for furnace components.

    RUL predictions use survival analysis and degradation models
    for predictive maintenance scheduling.

    Attributes:
        component_id: Component identifier
        component_type: Type of component
        furnace_id: Parent furnace identifier
        remaining_hours: Predicted remaining useful life in hours
        confidence_bounds: Confidence interval for prediction
        prediction_date: Date prediction was made
        next_inspection_date: Recommended next inspection
        failure_mode: Predicted failure mode
        data_quality_score: Quality score of input data

    Example:
        >>> rul = RULPrediction(
        ...     component_id="TUBE-R1-01",
        ...     component_type="radiant_tube",
        ...     furnace_id="FH-101",
        ...     remaining_hours=8760.0,
        ...     confidence_bounds=ConfidenceBounds(lower=7000, upper=10000)
        ... )
        >>> print(rul.remaining_days)
        365.0
    """
    component_id: str = Field(..., description="Component identifier")
    component_type: str = Field(
        ...,
        description="Type of component (radiant_tube, refractory, burner, etc.)"
    )
    furnace_id: str = Field(..., description="Parent furnace identifier")
    remaining_hours: float = Field(
        ...,
        description="Predicted remaining useful life in hours",
        ge=0.0
    )
    confidence_bounds: ConfidenceBounds = Field(
        ...,
        description="Confidence interval for prediction"
    )
    prediction_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Date prediction was made"
    )
    next_inspection_date: Optional[datetime] = Field(
        default=None,
        description="Recommended next inspection date"
    )
    failure_mode: Optional[str] = Field(
        default=None,
        description="Predicted failure mode"
    )
    data_quality_score: float = Field(
        default=1.0,
        description="Quality score of input data (0-1)",
        ge=0.0,
        le=1.0
    )
    model_version: str = Field(
        default="1.0.0",
        description="RUL model version"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit trail"
    )

    @computed_field
    @property
    def remaining_days(self) -> float:
        """Get remaining life in days."""
        return round(self.remaining_hours / 24.0, 1)

    @computed_field
    @property
    def remaining_months(self) -> float:
        """Get remaining life in months (30-day months)."""
        return round(self.remaining_hours / (24.0 * 30.0), 1)

    def get_alert_tier(
        self,
        warning_hours: float = 2000.0,
        urgent_hours: float = 500.0
    ) -> Optional[AlertTier]:
        """
        Get alert tier based on RUL thresholds.

        Args:
            warning_hours: RUL threshold for WARNING tier
            urgent_hours: RUL threshold for URGENT tier

        Returns:
            Alert tier or None if RUL is healthy
        """
        if self.remaining_hours <= urgent_hours:
            return AlertTier.URGENT
        elif self.remaining_hours <= warning_hours:
            return AlertTier.WARNING
        elif self.remaining_hours <= warning_hours * 2:
            return AlertTier.ADVISORY
        return None


# =============================================================================
# COMPLIANCE SCHEMAS
# =============================================================================

class ChecklistItem(BaseModel):
    """
    NFPA 86 compliance checklist item.

    Attributes:
        item_id: Checklist item identifier
        description: Item description
        is_passed: Pass/fail status
        evidence_ref: Reference to supporting evidence
        verified_by: Verifier identifier
        verified_at: Verification timestamp
        notes: Additional notes
    """
    item_id: str = Field(..., description="Checklist item identifier")
    description: str = Field(..., description="Item description")
    is_passed: bool = Field(..., description="Pass/fail status")
    evidence_ref: Optional[str] = Field(
        default=None,
        description="Reference to supporting evidence"
    )
    verified_by: Optional[str] = Field(
        default=None,
        description="Verifier identifier"
    )
    verified_at: Optional[datetime] = Field(
        default=None,
        description="Verification timestamp"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes"
    )
    is_critical: bool = Field(
        default=False,
        description="Whether item is safety-critical"
    )


class ComplianceEvidence(BaseModel):
    """
    NFPA 86 compliance evidence package.

    Collects all evidence for regulatory compliance verification
    including checklist items, sensor data, and timestamps.

    Attributes:
        event_id: Unique evidence package identifier
        furnace_id: Furnace identifier
        compliance_status: Overall compliance status
        checklist_items: List of checklist items with results
        signals: Supporting telemetry signals
        timestamps: Key event timestamps
        auditor_id: Auditor identifier
        audit_date: Audit date
        next_audit_due: Next audit due date
        provenance_hash: SHA-256 hash for audit trail

    Example:
        >>> evidence = ComplianceEvidence(
        ...     event_id="AUDIT-2024-001",
        ...     furnace_id="FH-101",
        ...     compliance_status=ComplianceStatus.COMPLIANT,
        ...     checklist_items=[ChecklistItem(...), ...]
        ... )
    """
    event_id: str = Field(
        ...,
        description="Unique evidence package identifier"
    )
    furnace_id: str = Field(..., description="Furnace identifier")
    compliance_status: ComplianceStatus = Field(
        ...,
        description="Overall compliance status"
    )
    checklist_items: List[ChecklistItem] = Field(
        default_factory=list,
        description="List of checklist items with results"
    )
    signals: List[TelemetrySignal] = Field(
        default_factory=list,
        description="Supporting telemetry signals"
    )
    timestamps: Dict[str, datetime] = Field(
        default_factory=dict,
        description="Key event timestamps"
    )
    auditor_id: Optional[str] = Field(
        default=None,
        description="Auditor identifier"
    )
    audit_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Audit date"
    )
    next_audit_due: Optional[datetime] = Field(
        default=None,
        description="Next audit due date"
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Audit findings and observations"
    )
    corrective_actions: List[str] = Field(
        default_factory=list,
        description="Required corrective actions"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit trail"
    )

    @computed_field
    @property
    def items_passed(self) -> int:
        """Count of passed checklist items."""
        return sum(1 for item in self.checklist_items if item.is_passed)

    @computed_field
    @property
    def items_failed(self) -> int:
        """Count of failed checklist items."""
        return sum(1 for item in self.checklist_items if not item.is_passed)

    @computed_field
    @property
    def critical_items_failed(self) -> int:
        """Count of failed critical checklist items."""
        return sum(
            1 for item in self.checklist_items
            if item.is_critical and not item.is_passed
        )

    @model_validator(mode='after')
    def calculate_provenance(self) -> 'ComplianceEvidence':
        """Calculate provenance hash after validation."""
        if self.provenance_hash is None:
            data_str = json.dumps({
                "event_id": self.event_id,
                "furnace_id": self.furnace_id,
                "status": self.compliance_status.value,
                "items_passed": self.items_passed,
                "items_failed": self.items_failed,
                "audit_date": self.audit_date.isoformat(),
            }, sort_keys=True)
            object.__setattr__(
                self,
                'provenance_hash',
                hashlib.sha256(data_str.encode()).hexdigest()
            )
        return self


class SafetyEvent(BaseModel):
    """
    Safety event with HAZOP/LOPA references.

    Records safety-critical events for incident investigation
    and regulatory reporting.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of safety event
        furnace_id: Affected furnace
        hazop_ref: HAZOP study reference
        lopa_layer: LOPA protection layer that responded
        severity: Event severity
        description: Event description
        evidence_package_id: Reference to evidence package
        timestamp: Event timestamp
        resolution: Event resolution description
        lessons_learned: Lessons learned notes

    Example:
        >>> event = SafetyEvent(
        ...     event_id="SE-2024-001",
        ...     event_type="HIGH_TMT_ALARM",
        ...     furnace_id="FH-101",
        ...     hazop_ref="HAZOP-FH101-N3",
        ...     lopa_layer="SIF",
        ...     severity="HIGH"
        ... )
    """
    event_id: str = Field(..., description="Unique event identifier")
    event_type: str = Field(
        ...,
        description="Type of safety event (e.g., HIGH_TMT_ALARM, FLAME_FAILURE)"
    )
    furnace_id: str = Field(..., description="Affected furnace")
    hazop_ref: Optional[str] = Field(
        default=None,
        description="HAZOP study reference"
    )
    lopa_layer: Optional[str] = Field(
        default=None,
        description="LOPA protection layer (BPCS, Operator, SIF, etc.)"
    )
    severity: str = Field(
        default="MEDIUM",
        description="Event severity (LOW, MEDIUM, HIGH, CRITICAL)"
    )
    description: str = Field(
        default="",
        description="Event description"
    )
    evidence_package_id: Optional[str] = Field(
        default=None,
        description="Reference to evidence package"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    duration_seconds: Optional[float] = Field(
        default=None,
        description="Event duration in seconds"
    )
    resolution: Optional[str] = Field(
        default=None,
        description="Event resolution description"
    )
    resolved_at: Optional[datetime] = Field(
        default=None,
        description="Resolution timestamp"
    )
    lessons_learned: Optional[str] = Field(
        default=None,
        description="Lessons learned notes"
    )
    root_cause: Optional[str] = Field(
        default=None,
        description="Root cause analysis result"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit trail"
    )

    @computed_field
    @property
    def is_resolved(self) -> bool:
        """Check if event has been resolved."""
        return self.resolved_at is not None


# =============================================================================
# OUTPUT SCHEMA
# =============================================================================

class FurnacePulseOutput(BaseModel):
    """
    Complete output from FurnacePulse agent processing.

    Aggregates all outputs from a single processing cycle
    for downstream consumption and dashboard display.

    Attributes:
        furnace_id: Furnace identifier
        processing_timestamp: Output generation timestamp
        furnace_state: Current furnace state
        efficiency_kpis: Calculated efficiency KPIs
        efficiency_trend: Efficiency trend analysis
        hotspot_alerts: Detected hotspots
        maintenance_alerts: Maintenance notifications
        rul_predictions: RUL predictions
        compliance_status: Compliance status
        safety_events: Safety events recorded
        processing_time_ms: Processing duration
        provenance_hash: SHA-256 hash for audit trail

    Example:
        >>> output = FurnacePulseOutput(
        ...     furnace_id="FH-101",
        ...     furnace_state=state,
        ...     efficiency_kpis=kpis,
        ...     hotspot_alerts=[alert1, alert2],
        ...     rul_predictions=[rul1]
        ... )
    """
    furnace_id: str = Field(..., description="Furnace identifier")
    processing_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Output generation timestamp"
    )
    furnace_state: Optional[FurnaceState] = Field(
        default=None,
        description="Current furnace state"
    )
    efficiency_kpis: Optional[EfficiencyKPI] = Field(
        default=None,
        description="Calculated efficiency KPIs"
    )
    efficiency_trend: Optional[EfficiencyTrend] = Field(
        default=None,
        description="Efficiency trend analysis"
    )
    hotspot_alerts: List[HotspotAlert] = Field(
        default_factory=list,
        description="Detected hotspots"
    )
    maintenance_alerts: List[MaintenanceAlert] = Field(
        default_factory=list,
        description="Maintenance notifications"
    )
    rul_predictions: List[RULPrediction] = Field(
        default_factory=list,
        description="RUL predictions"
    )
    compliance_status: Optional[ComplianceStatus] = Field(
        default=None,
        description="Compliance status"
    )
    compliance_evidence: Optional[ComplianceEvidence] = Field(
        default=None,
        description="Compliance evidence package"
    )
    safety_events: List[SafetyEvent] = Field(
        default_factory=list,
        description="Safety events recorded"
    )
    explainability: Dict[str, Any] = Field(
        default_factory=dict,
        description="SHAP/LIME explainability data"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing duration in milliseconds"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit trail"
    )

    @computed_field
    @property
    def alert_count(self) -> int:
        """Total count of all alerts."""
        return len(self.hotspot_alerts) + len(self.maintenance_alerts)

    @computed_field
    @property
    def urgent_alert_count(self) -> int:
        """Count of urgent tier alerts."""
        return sum(
            1 for a in self.hotspot_alerts + self.maintenance_alerts
            if a.tier == AlertTier.URGENT
        )

    @computed_field
    @property
    def has_critical_issues(self) -> bool:
        """Check if output contains critical issues requiring attention."""
        return (
            self.urgent_alert_count > 0 or
            self.compliance_status == ComplianceStatus.NON_COMPLIANT or
            any(p.remaining_hours < 500 for p in self.rul_predictions)
        )

    @model_validator(mode='after')
    def calculate_provenance(self) -> 'FurnacePulseOutput':
        """Calculate provenance hash after validation."""
        if self.provenance_hash is None:
            data_str = json.dumps({
                "furnace_id": self.furnace_id,
                "timestamp": self.processing_timestamp.isoformat(),
                "alert_count": self.alert_count,
                "compliance_status": self.compliance_status.value if self.compliance_status else None,
                "rul_count": len(self.rul_predictions),
            }, sort_keys=True)
            object.__setattr__(
                self,
                'provenance_hash',
                hashlib.sha256(data_str.encode()).hexdigest()
            )
        return self
