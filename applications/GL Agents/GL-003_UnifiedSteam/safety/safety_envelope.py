"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Safety Envelope

This module defines safety envelopes for steam system equipment including
pressure, temperature, quality, and rate limits with equipment rating integration.

Safety Architecture:
    - Multi-parameter safety envelopes per equipment
    - Equipment rating database integration
    - Alarm margin management
    - Real-time envelope checking with provenance

Reference Standards:
    - ASME B31.1 Power Piping
    - ASME BPVC Section VIII Pressure Vessels
    - IEC 61511 Functional Safety
    - API 520/521 Pressure Relieving Systems

Author: GreenLang Control Systems Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime
import hashlib
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class LimitType(str, Enum):
    """Limit type enumeration."""
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    QUALITY = "quality"
    RATE = "rate"
    FLOW = "flow"
    LEVEL = "level"


class EnvelopeStatus(str, Enum):
    """Envelope check status enumeration."""
    WITHIN_ENVELOPE = "within_envelope"
    ALARM_LOW = "alarm_low"
    ALARM_HIGH = "alarm_high"
    WARNING_LOW = "warning_low"
    WARNING_HIGH = "warning_high"
    TRIP_LOW = "trip_low"
    TRIP_HIGH = "trip_high"


class AlarmSeverity(str, Enum):
    """Alarm severity enumeration."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    TRIP = "trip"


class EquipmentType(str, Enum):
    """Equipment type enumeration."""
    BOILER = "boiler"
    DESUPERHEATER = "desuperheater"
    PRV = "prv"
    TURBINE = "turbine"
    HEADER = "header"
    HEAT_EXCHANGER = "heat_exchanger"
    SEPARATOR = "separator"
    TRAP = "trap"
    VALVE = "valve"


# =============================================================================
# DATA MODELS
# =============================================================================

class AlarmMargins(BaseModel):
    """Alarm and warning margins relative to limits."""

    warning_pct: float = Field(
        default=10.0,
        ge=0,
        le=50,
        description="Warning margin as percentage from limit"
    )
    alarm_pct: float = Field(
        default=5.0,
        ge=0,
        le=50,
        description="Alarm margin as percentage from limit"
    )
    trip_pct: float = Field(
        default=0.0,
        ge=0,
        le=50,
        description="Trip margin as percentage from limit"
    )


class PressureLimits(BaseModel):
    """Pressure limits for equipment."""

    equipment_id: str = Field(..., description="Equipment identifier")
    min_kpa: float = Field(..., description="Minimum pressure (kPa)")
    max_kpa: float = Field(..., description="Maximum pressure (kPa)")
    design_pressure_kpa: float = Field(
        ...,
        description="Design pressure (kPa)"
    )
    test_pressure_kpa: Optional[float] = Field(
        None,
        description="Test pressure (kPa)"
    )
    alarm_margins: AlarmMargins = Field(
        default_factory=AlarmMargins,
        description="Alarm margins"
    )
    unit: str = Field(default="kPa", description="Pressure unit")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")

    @validator("max_kpa")
    def validate_max_above_min(cls, v, values):
        """Ensure max is above min."""
        if "min_kpa" in values and v <= values["min_kpa"]:
            raise ValueError("max_kpa must be greater than min_kpa")
        return v


class TemperatureLimits(BaseModel):
    """Temperature limits for equipment."""

    equipment_id: str = Field(..., description="Equipment identifier")
    min_c: float = Field(..., description="Minimum temperature (C)")
    max_c: float = Field(..., description="Maximum temperature (C)")
    design_temperature_c: float = Field(
        ...,
        description="Design temperature (C)"
    )
    alarm_margins: AlarmMargins = Field(
        default_factory=AlarmMargins,
        description="Alarm margins"
    )
    unit: str = Field(default="C", description="Temperature unit")
    saturation_margin_c: Optional[float] = Field(
        None,
        description="Minimum margin above saturation (C)"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class QualityLimits(BaseModel):
    """Steam quality limits for equipment."""

    equipment_id: str = Field(..., description="Equipment identifier")
    min_dryness: float = Field(
        ...,
        ge=0,
        le=1,
        description="Minimum dryness fraction"
    )
    erosion_threshold: float = Field(
        default=0.90,
        ge=0,
        le=1,
        description="Erosion threshold dryness"
    )
    target_dryness: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Target dryness fraction"
    )
    min_superheat_c: Optional[float] = Field(
        None,
        ge=0,
        description="Minimum superheat (C)"
    )
    alarm_margins: AlarmMargins = Field(
        default_factory=AlarmMargins,
        description="Alarm margins"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class RateLimits(BaseModel):
    """Rate of change limits for equipment."""

    equipment_id: str = Field(..., description="Equipment identifier")
    parameter: str = Field(..., description="Parameter name")
    max_rate_per_min: float = Field(
        ...,
        ge=0,
        description="Maximum rate per minute"
    )
    unit: str = Field(..., description="Parameter unit")
    reason: str = Field(
        default="thermal_stress",
        description="Reason for rate limit"
    )
    ramp_up_limit: Optional[float] = Field(
        None,
        ge=0,
        description="Ramp up limit (may differ from down)"
    )
    ramp_down_limit: Optional[float] = Field(
        None,
        ge=0,
        description="Ramp down limit (may differ from up)"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Creation timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class EnvelopeCheckResult(BaseModel):
    """Result of envelope check."""

    check_id: str = Field(..., description="Check ID")
    equipment_id: str = Field(..., description="Equipment ID")
    parameter: str = Field(..., description="Parameter checked")
    value: float = Field(..., description="Value checked")
    unit: str = Field(..., description="Value unit")
    status: EnvelopeStatus = Field(..., description="Envelope status")
    severity: AlarmSeverity = Field(..., description="Alarm severity")
    limit_type: LimitType = Field(..., description="Type of limit checked")
    min_limit: float = Field(..., description="Minimum limit")
    max_limit: float = Field(..., description="Maximum limit")
    distance_to_limit: float = Field(
        ...,
        description="Distance to nearest limit"
    )
    distance_pct: float = Field(
        ...,
        description="Distance as percentage of range"
    )
    message: str = Field(default="", description="Status message")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Check timestamp"
    )
    provenance_hash: str = Field(default="", description="Provenance hash")


class EquipmentRating(BaseModel):
    """Equipment rating from database."""

    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_type: EquipmentType = Field(..., description="Equipment type")
    manufacturer: str = Field(default="", description="Manufacturer")
    model: str = Field(default="", description="Model number")
    design_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Design pressure (kPa)"
    )
    design_temperature_c: float = Field(
        ...,
        description="Design temperature (C)"
    )
    max_allowable_working_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="MAWP (kPa)"
    )
    min_design_metal_temperature_c: float = Field(
        default=-29,
        description="MDMT (C)"
    )
    material_spec: str = Field(default="", description="Material specification")
    corrosion_allowance_mm: float = Field(
        default=3.0,
        ge=0,
        description="Corrosion allowance (mm)"
    )


# =============================================================================
# SAFETY ENVELOPE
# =============================================================================

class SafetyEnvelope:
    """
    Safety envelope definitions for steam system equipment.

    This class manages safety envelopes for all equipment in the steam system,
    including pressure, temperature, quality, and rate limits with equipment
    rating integration.

    Features:
        - Multi-parameter safety envelopes per equipment
        - Equipment rating database integration
        - Alarm margin management (warning, alarm, trip)
        - Real-time envelope checking with provenance

    Safety Philosophy:
        - Safety envelopes are conservative by design
        - Margins provide early warning before limits
        - All checks are logged for audit trail
        - Integration with plant equipment ratings

    Attributes:
        _pressure_limits: Pressure limits by equipment ID
        _temperature_limits: Temperature limits by equipment ID
        _quality_limits: Quality limits by equipment ID
        _rate_limits: Rate limits by equipment ID and parameter
        _equipment_ratings: Equipment ratings from database

    Example:
        >>> envelope = SafetyEnvelope()
        >>> envelope.define_pressure_limits(
        ...     equipment_id="DS-001",
        ...     min_kpa=500,
        ...     max_kpa=4500,
        ...     alarm_margins=AlarmMargins(warning_pct=10, alarm_pct=5)
        ... )
        >>> result = envelope.check_within_envelope("pressure", 4200)
    """

    def __init__(self):
        """Initialize SafetyEnvelope."""
        self._pressure_limits: Dict[str, PressureLimits] = {}
        self._temperature_limits: Dict[str, TemperatureLimits] = {}
        self._quality_limits: Dict[str, QualityLimits] = {}
        self._rate_limits: Dict[str, Dict[str, RateLimits]] = {}
        self._equipment_ratings: Dict[str, EquipmentRating] = {}
        self._check_history: List[EnvelopeCheckResult] = []
        self._max_history_size = 10000

        logger.info("SafetyEnvelope initialized")

    def define_pressure_limits(
        self,
        equipment_id: str,
        min_kpa: float,
        max_kpa: float,
        alarm_margins: Optional[AlarmMargins] = None,
        design_pressure_kpa: Optional[float] = None
    ) -> PressureLimits:
        """
        Define pressure limits for equipment.

        This method establishes pressure safety limits for a piece of equipment,
        including alarm and warning margins.

        Args:
            equipment_id: Equipment identifier
            min_kpa: Minimum pressure (kPa)
            max_kpa: Maximum pressure (kPa)
            alarm_margins: Alarm and warning margins
            design_pressure_kpa: Design pressure (defaults to max)

        Returns:
            PressureLimits: Defined pressure limits

        Raises:
            ValueError: If limits are invalid
        """
        if min_kpa >= max_kpa:
            raise ValueError(f"min_kpa ({min_kpa}) must be less than max_kpa ({max_kpa})")

        margins = alarm_margins or AlarmMargins()
        design = design_pressure_kpa or max_kpa

        limits = PressureLimits(
            equipment_id=equipment_id,
            min_kpa=min_kpa,
            max_kpa=max_kpa,
            design_pressure_kpa=design,
            alarm_margins=margins
        )

        # Calculate provenance hash
        limits.provenance_hash = hashlib.sha256(
            f"PRES_{equipment_id}|{min_kpa}|{max_kpa}|{design}".encode()
        ).hexdigest()

        self._pressure_limits[equipment_id] = limits

        logger.info(
            f"Defined pressure limits for {equipment_id}: "
            f"[{min_kpa}, {max_kpa}] kPa, design={design} kPa"
        )

        return limits

    def define_temperature_limits(
        self,
        equipment_id: str,
        min_c: float,
        max_c: float,
        alarm_margins: Optional[AlarmMargins] = None,
        design_temperature_c: Optional[float] = None,
        saturation_margin_c: Optional[float] = None
    ) -> TemperatureLimits:
        """
        Define temperature limits for equipment.

        This method establishes temperature safety limits for a piece of
        equipment, including alarm margins and saturation margin requirements.

        Args:
            equipment_id: Equipment identifier
            min_c: Minimum temperature (C)
            max_c: Maximum temperature (C)
            alarm_margins: Alarm and warning margins
            design_temperature_c: Design temperature (defaults to max)
            saturation_margin_c: Minimum margin above saturation

        Returns:
            TemperatureLimits: Defined temperature limits
        """
        if min_c >= max_c:
            raise ValueError(f"min_c ({min_c}) must be less than max_c ({max_c})")

        margins = alarm_margins or AlarmMargins()
        design = design_temperature_c or max_c

        limits = TemperatureLimits(
            equipment_id=equipment_id,
            min_c=min_c,
            max_c=max_c,
            design_temperature_c=design,
            alarm_margins=margins,
            saturation_margin_c=saturation_margin_c
        )

        # Calculate provenance hash
        limits.provenance_hash = hashlib.sha256(
            f"TEMP_{equipment_id}|{min_c}|{max_c}|{design}".encode()
        ).hexdigest()

        self._temperature_limits[equipment_id] = limits

        logger.info(
            f"Defined temperature limits for {equipment_id}: "
            f"[{min_c}, {max_c}] C, design={design} C"
        )

        return limits

    def define_quality_limits(
        self,
        equipment_id: str,
        min_dryness: float,
        erosion_threshold: float,
        alarm_margins: Optional[AlarmMargins] = None,
        min_superheat_c: Optional[float] = None
    ) -> QualityLimits:
        """
        Define steam quality limits for equipment.

        This method establishes steam quality (dryness fraction) limits
        for equipment with erosion protection thresholds.

        Args:
            equipment_id: Equipment identifier
            min_dryness: Minimum dryness fraction (0-1)
            erosion_threshold: Erosion threshold dryness
            alarm_margins: Alarm and warning margins
            min_superheat_c: Minimum superheat margin (C)

        Returns:
            QualityLimits: Defined quality limits
        """
        if not 0 <= min_dryness <= 1:
            raise ValueError(f"min_dryness ({min_dryness}) must be between 0 and 1")

        if not 0 <= erosion_threshold <= 1:
            raise ValueError(
                f"erosion_threshold ({erosion_threshold}) must be between 0 and 1"
            )

        margins = alarm_margins or AlarmMargins()

        limits = QualityLimits(
            equipment_id=equipment_id,
            min_dryness=min_dryness,
            erosion_threshold=erosion_threshold,
            alarm_margins=margins,
            min_superheat_c=min_superheat_c
        )

        # Calculate provenance hash
        limits.provenance_hash = hashlib.sha256(
            f"QUAL_{equipment_id}|{min_dryness}|{erosion_threshold}".encode()
        ).hexdigest()

        self._quality_limits[equipment_id] = limits

        logger.info(
            f"Defined quality limits for {equipment_id}: "
            f"min_dryness={min_dryness}, erosion_threshold={erosion_threshold}"
        )

        return limits

    def define_rate_limits(
        self,
        equipment_id: str,
        parameter: str,
        max_rate_per_min: float,
        unit: str,
        reason: str = "thermal_stress",
        ramp_up_limit: Optional[float] = None,
        ramp_down_limit: Optional[float] = None
    ) -> RateLimits:
        """
        Define rate of change limits for equipment parameter.

        This method establishes rate limits to prevent thermal shock
        and equipment damage from rapid changes.

        Args:
            equipment_id: Equipment identifier
            parameter: Parameter name (e.g., "temperature", "pressure")
            max_rate_per_min: Maximum rate of change per minute
            unit: Parameter unit
            reason: Reason for rate limit
            ramp_up_limit: Ramp up limit (if different from max_rate)
            ramp_down_limit: Ramp down limit (if different from max_rate)

        Returns:
            RateLimits: Defined rate limits
        """
        limits = RateLimits(
            equipment_id=equipment_id,
            parameter=parameter,
            max_rate_per_min=max_rate_per_min,
            unit=unit,
            reason=reason,
            ramp_up_limit=ramp_up_limit,
            ramp_down_limit=ramp_down_limit
        )

        # Calculate provenance hash
        limits.provenance_hash = hashlib.sha256(
            f"RATE_{equipment_id}|{parameter}|{max_rate_per_min}".encode()
        ).hexdigest()

        # Store rate limits by equipment_id and parameter
        if equipment_id not in self._rate_limits:
            self._rate_limits[equipment_id] = {}
        self._rate_limits[equipment_id][parameter] = limits

        logger.info(
            f"Defined rate limits for {equipment_id}.{parameter}: "
            f"max={max_rate_per_min} {unit}/min ({reason})"
        )

        return limits

    def check_within_envelope(
        self,
        equipment_id: str,
        parameter: str,
        value: float,
        unit: Optional[str] = None
    ) -> EnvelopeCheckResult:
        """
        Check if value is within safety envelope.

        This method performs comprehensive envelope checking including
        warning, alarm, and trip thresholds based on defined limits.

        Args:
            equipment_id: Equipment identifier
            parameter: Parameter name ("pressure", "temperature", "quality")
            value: Value to check
            unit: Unit of value (optional, for validation)

        Returns:
            EnvelopeCheckResult: Envelope check result

        Raises:
            KeyError: If equipment or parameter limits not defined
        """
        start_time = datetime.now()

        # Determine limit type and get limits
        if parameter == "pressure":
            result = self._check_pressure_envelope(equipment_id, value)
        elif parameter == "temperature":
            result = self._check_temperature_envelope(equipment_id, value)
        elif parameter == "quality" or parameter == "dryness":
            result = self._check_quality_envelope(equipment_id, value)
        else:
            raise KeyError(f"Unknown parameter type: {parameter}")

        # Store in history
        self._add_to_history(result)

        return result

    def load_from_equipment_ratings(
        self,
        ratings: List[EquipmentRating]
    ) -> int:
        """
        Load safety limits from equipment ratings database.

        This method automatically defines safety limits based on equipment
        ratings from the plant's equipment database.

        Args:
            ratings: List of equipment ratings

        Returns:
            int: Number of equipment limits defined
        """
        count = 0

        for rating in ratings:
            self._equipment_ratings[rating.equipment_id] = rating

            # Define pressure limits from MAWP
            self.define_pressure_limits(
                equipment_id=rating.equipment_id,
                min_kpa=0,  # Vacuum protection if needed
                max_kpa=rating.max_allowable_working_pressure_kpa,
                design_pressure_kpa=rating.design_pressure_kpa
            )

            # Define temperature limits
            self.define_temperature_limits(
                equipment_id=rating.equipment_id,
                min_c=rating.min_design_metal_temperature_c,
                max_c=rating.design_temperature_c
            )

            count += 1

        logger.info(f"Loaded limits from {count} equipment ratings")

        return count

    def get_pressure_limits(self, equipment_id: str) -> Optional[PressureLimits]:
        """Get pressure limits for equipment."""
        return self._pressure_limits.get(equipment_id)

    def get_temperature_limits(self, equipment_id: str) -> Optional[TemperatureLimits]:
        """Get temperature limits for equipment."""
        return self._temperature_limits.get(equipment_id)

    def get_quality_limits(self, equipment_id: str) -> Optional[QualityLimits]:
        """Get quality limits for equipment."""
        return self._quality_limits.get(equipment_id)

    def get_rate_limits(
        self,
        equipment_id: str,
        parameter: Optional[str] = None
    ) -> Union[RateLimits, Dict[str, RateLimits], None]:
        """Get rate limits for equipment."""
        if equipment_id not in self._rate_limits:
            return None

        if parameter:
            return self._rate_limits[equipment_id].get(parameter)

        return self._rate_limits[equipment_id]

    def get_all_equipment_ids(self) -> List[str]:
        """Get all equipment IDs with defined limits."""
        all_ids = set()
        all_ids.update(self._pressure_limits.keys())
        all_ids.update(self._temperature_limits.keys())
        all_ids.update(self._quality_limits.keys())
        all_ids.update(self._rate_limits.keys())
        return list(all_ids)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _check_pressure_envelope(
        self,
        equipment_id: str,
        value: float
    ) -> EnvelopeCheckResult:
        """Check pressure value against envelope."""
        if equipment_id not in self._pressure_limits:
            raise KeyError(f"No pressure limits defined for {equipment_id}")

        limits = self._pressure_limits[equipment_id]
        return self._evaluate_limits(
            equipment_id=equipment_id,
            parameter="pressure",
            value=value,
            unit=limits.unit,
            min_limit=limits.min_kpa,
            max_limit=limits.max_kpa,
            margins=limits.alarm_margins,
            limit_type=LimitType.PRESSURE
        )

    def _check_temperature_envelope(
        self,
        equipment_id: str,
        value: float
    ) -> EnvelopeCheckResult:
        """Check temperature value against envelope."""
        if equipment_id not in self._temperature_limits:
            raise KeyError(f"No temperature limits defined for {equipment_id}")

        limits = self._temperature_limits[equipment_id]
        return self._evaluate_limits(
            equipment_id=equipment_id,
            parameter="temperature",
            value=value,
            unit=limits.unit,
            min_limit=limits.min_c,
            max_limit=limits.max_c,
            margins=limits.alarm_margins,
            limit_type=LimitType.TEMPERATURE
        )

    def _check_quality_envelope(
        self,
        equipment_id: str,
        value: float
    ) -> EnvelopeCheckResult:
        """Check quality value against envelope."""
        if equipment_id not in self._quality_limits:
            raise KeyError(f"No quality limits defined for {equipment_id}")

        limits = self._quality_limits[equipment_id]

        # Quality is special - we mainly care about minimum
        return self._evaluate_limits(
            equipment_id=equipment_id,
            parameter="quality",
            value=value,
            unit="fraction",
            min_limit=limits.min_dryness,
            max_limit=1.0,
            margins=limits.alarm_margins,
            limit_type=LimitType.QUALITY
        )

    def _evaluate_limits(
        self,
        equipment_id: str,
        parameter: str,
        value: float,
        unit: str,
        min_limit: float,
        max_limit: float,
        margins: AlarmMargins,
        limit_type: LimitType
    ) -> EnvelopeCheckResult:
        """Evaluate value against limits with margins."""
        range_size = max_limit - min_limit

        # Calculate threshold values
        warning_low = min_limit + (range_size * margins.warning_pct / 100)
        warning_high = max_limit - (range_size * margins.warning_pct / 100)
        alarm_low = min_limit + (range_size * margins.alarm_pct / 100)
        alarm_high = max_limit - (range_size * margins.alarm_pct / 100)
        trip_low = min_limit + (range_size * margins.trip_pct / 100)
        trip_high = max_limit - (range_size * margins.trip_pct / 100)

        # Determine status and severity
        if value <= min_limit:
            status = EnvelopeStatus.TRIP_LOW
            severity = AlarmSeverity.TRIP
            message = f"{parameter} {value} at or below minimum limit {min_limit}"
        elif value >= max_limit:
            status = EnvelopeStatus.TRIP_HIGH
            severity = AlarmSeverity.TRIP
            message = f"{parameter} {value} at or above maximum limit {max_limit}"
        elif value <= trip_low:
            status = EnvelopeStatus.TRIP_LOW
            severity = AlarmSeverity.TRIP
            message = f"{parameter} {value} in trip zone (< {trip_low})"
        elif value >= trip_high:
            status = EnvelopeStatus.TRIP_HIGH
            severity = AlarmSeverity.TRIP
            message = f"{parameter} {value} in trip zone (> {trip_high})"
        elif value <= alarm_low:
            status = EnvelopeStatus.ALARM_LOW
            severity = AlarmSeverity.ALARM
            message = f"{parameter} {value} in alarm zone (< {alarm_low})"
        elif value >= alarm_high:
            status = EnvelopeStatus.ALARM_HIGH
            severity = AlarmSeverity.ALARM
            message = f"{parameter} {value} in alarm zone (> {alarm_high})"
        elif value <= warning_low:
            status = EnvelopeStatus.WARNING_LOW
            severity = AlarmSeverity.WARNING
            message = f"{parameter} {value} in warning zone (< {warning_low})"
        elif value >= warning_high:
            status = EnvelopeStatus.WARNING_HIGH
            severity = AlarmSeverity.WARNING
            message = f"{parameter} {value} in warning zone (> {warning_high})"
        else:
            status = EnvelopeStatus.WITHIN_ENVELOPE
            severity = AlarmSeverity.INFO
            message = f"{parameter} {value} within normal operating envelope"

        # Calculate distance to nearest limit
        distance_to_min = value - min_limit
        distance_to_max = max_limit - value
        distance_to_limit = min(abs(distance_to_min), abs(distance_to_max))
        distance_pct = (distance_to_limit / range_size) * 100 if range_size > 0 else 0

        # Generate check ID
        check_id = hashlib.sha256(
            f"CHK_{equipment_id}|{parameter}|{value}|{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        result = EnvelopeCheckResult(
            check_id=check_id,
            equipment_id=equipment_id,
            parameter=parameter,
            value=value,
            unit=unit,
            status=status,
            severity=severity,
            limit_type=limit_type,
            min_limit=min_limit,
            max_limit=max_limit,
            distance_to_limit=distance_to_limit,
            distance_pct=distance_pct,
            message=message
        )

        result.provenance_hash = hashlib.sha256(
            f"{check_id}|{status.value}|{severity.value}".encode()
        ).hexdigest()

        # Log based on severity
        if severity in (AlarmSeverity.TRIP, AlarmSeverity.CRITICAL):
            logger.error(message)
        elif severity == AlarmSeverity.ALARM:
            logger.warning(message)
        elif severity == AlarmSeverity.WARNING:
            logger.info(message)
        else:
            logger.debug(message)

        return result

    def _add_to_history(self, result: EnvelopeCheckResult) -> None:
        """Add check result to history with size limit."""
        self._check_history.append(result)
        if len(self._check_history) > self._max_history_size:
            self._check_history = self._check_history[-self._max_history_size:]
