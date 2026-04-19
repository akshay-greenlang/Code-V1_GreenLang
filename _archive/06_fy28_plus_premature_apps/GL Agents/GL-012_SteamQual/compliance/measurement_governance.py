"""
GL-012 SteamQual - Measurement Governance

Comprehensive measurement governance framework for steam quality monitoring
including sensor selection, calibration workflow tracking, data quality scoring,
and uncertainty reporting standards.

Regulatory References:
- ASME PTC 19.1: Test Uncertainty
- ISO/IEC 17025: Calibration Laboratory Requirements
- NIST Traceability Requirements
- ISA-67.04.01: Setpoints for Nuclear Safety Systems

This module provides:
1. Sensor selection requirements and validation
2. Calibration workflow tracking with due date management
3. Data quality scoring algorithms
4. Uncertainty propagation and reporting
5. Measurement provenance for audit trails

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging
import math

logger = logging.getLogger(__name__)


# =============================================================================
# MEASUREMENT GOVERNANCE ENUMERATIONS
# =============================================================================

class SensorType(Enum):
    """Steam quality measurement sensor types."""

    # Dryness/Quality Sensors
    THROTTLING_CALORIMETER = "throttling_calorimeter"
    SEPARATING_CALORIMETER = "separating_calorimeter"
    ELECTRICAL_CALORIMETER = "electrical_calorimeter"
    OPTICAL_DRYNESS = "optical_dryness"

    # Temperature Sensors
    RTD_PT100 = "rtd_pt100"
    RTD_PT1000 = "rtd_pt1000"
    THERMOCOUPLE_K = "thermocouple_k"
    THERMOCOUPLE_J = "thermocouple_j"
    THERMOCOUPLE_T = "thermocouple_t"

    # Pressure Sensors
    PRESSURE_TRANSMITTER = "pressure_transmitter"
    DIFFERENTIAL_PRESSURE = "differential_pressure"
    ABSOLUTE_PRESSURE = "absolute_pressure"

    # Flow Sensors
    ORIFICE_PLATE = "orifice_plate"
    VORTEX_METER = "vortex_meter"
    ULTRASONIC_METER = "ultrasonic_meter"
    CORIOLIS_METER = "coriolis_meter"

    # Water Quality Sensors
    CONDUCTIVITY_PROBE = "conductivity_probe"
    PH_PROBE = "ph_probe"
    SILICA_ANALYZER = "silica_analyzer"
    SODIUM_ANALYZER = "sodium_analyzer"
    DO_ANALYZER = "dissolved_oxygen_analyzer"
    TDS_ANALYZER = "tds_analyzer"


class CalibrationStatus(Enum):
    """Sensor calibration status."""

    VALID = "valid"                  # Within calibration period
    DUE_SOON = "due_soon"            # Within 30 days of due date
    OVERDUE = "overdue"              # Past calibration due date
    FAILED = "failed"                # Failed last calibration
    NOT_CALIBRATED = "not_calibrated"  # Never calibrated
    BYPASSED = "bypassed"            # Temporarily bypassed


class AccuracyClass(Enum):
    """Sensor accuracy classification per ASME standards."""

    CLASS_A = "class_a"    # +/- 0.1%
    CLASS_B = "class_b"    # +/- 0.25%
    CLASS_C = "class_c"    # +/- 0.5%
    CLASS_D = "class_d"    # +/- 1.0%
    CLASS_E = "class_e"    # +/- 2.0%


class DataQualityGrade(Enum):
    """Data quality grade classification."""

    EXCELLENT = "excellent"    # Score >= 95
    GOOD = "good"              # Score >= 85
    ACCEPTABLE = "acceptable"  # Score >= 70
    MARGINAL = "marginal"      # Score >= 50
    POOR = "poor"              # Score < 50
    INVALID = "invalid"        # Data unusable


class MeasurementApplication(Enum):
    """Application context for measurement requirements."""

    PROCESS_CONTROL = "process_control"
    CUSTODY_TRANSFER = "custody_transfer"
    REGULATORY_REPORTING = "regulatory_reporting"
    PERFORMANCE_MONITORING = "performance_monitoring"
    SAFETY_CRITICAL = "safety_critical"
    OPTIMIZATION = "optimization"


# =============================================================================
# SENSOR SELECTION REQUIREMENTS
# =============================================================================

@dataclass
class SensorRequirement:
    """
    Sensor selection requirement specification.

    Defines minimum requirements for sensors based on
    measurement application and regulatory standards.
    """

    requirement_id: str
    sensor_type: SensorType
    application: MeasurementApplication
    min_accuracy_class: AccuracyClass
    min_range: Decimal
    max_range: Decimal
    unit: str
    response_time_max_sec: Decimal
    calibration_interval_days: int
    environmental_rating: str  # IP rating
    standard_reference: str
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "requirement_id": self.requirement_id,
            "sensor_type": self.sensor_type.value,
            "application": self.application.value,
            "min_accuracy_class": self.min_accuracy_class.value,
            "min_range": str(self.min_range),
            "max_range": str(self.max_range),
            "unit": self.unit,
            "response_time_max_sec": str(self.response_time_max_sec),
            "calibration_interval_days": self.calibration_interval_days,
            "environmental_rating": self.environmental_rating,
            "standard_reference": self.standard_reference,
            "notes": self.notes,
        }


# Standard sensor requirements for steam quality measurement
SENSOR_REQUIREMENTS: List[SensorRequirement] = [
    # Temperature measurement for steam quality
    SensorRequirement(
        requirement_id="REQ-TEMP-001",
        sensor_type=SensorType.RTD_PT100,
        application=MeasurementApplication.PROCESS_CONTROL,
        min_accuracy_class=AccuracyClass.CLASS_A,
        min_range=Decimal("0"),
        max_range=Decimal("400"),
        unit="deg_C",
        response_time_max_sec=Decimal("2.0"),
        calibration_interval_days=365,
        environmental_rating="IP65",
        standard_reference="ASME PTC 19.3",
        notes="Class A RTD per IEC 60751",
    ),
    SensorRequirement(
        requirement_id="REQ-TEMP-002",
        sensor_type=SensorType.THERMOCOUPLE_K,
        application=MeasurementApplication.PERFORMANCE_MONITORING,
        min_accuracy_class=AccuracyClass.CLASS_B,
        min_range=Decimal("-40"),
        max_range=Decimal("1000"),
        unit="deg_C",
        response_time_max_sec=Decimal("5.0"),
        calibration_interval_days=180,
        environmental_rating="IP65",
        standard_reference="ASTM E230",
    ),

    # Pressure measurement
    SensorRequirement(
        requirement_id="REQ-PRESS-001",
        sensor_type=SensorType.PRESSURE_TRANSMITTER,
        application=MeasurementApplication.PROCESS_CONTROL,
        min_accuracy_class=AccuracyClass.CLASS_B,
        min_range=Decimal("0"),
        max_range=Decimal("250"),
        unit="bar",
        response_time_max_sec=Decimal("0.5"),
        calibration_interval_days=365,
        environmental_rating="IP67",
        standard_reference="ASME PTC 19.2",
    ),

    # Conductivity measurement
    SensorRequirement(
        requirement_id="REQ-COND-001",
        sensor_type=SensorType.CONDUCTIVITY_PROBE,
        application=MeasurementApplication.PROCESS_CONTROL,
        min_accuracy_class=AccuracyClass.CLASS_C,
        min_range=Decimal("0"),
        max_range=Decimal("100"),
        unit="umhos/cm",
        response_time_max_sec=Decimal("10.0"),
        calibration_interval_days=90,
        environmental_rating="IP68",
        standard_reference="ASTM D1125",
        notes="Requires temperature compensation",
    ),

    # Silica analyzer
    SensorRequirement(
        requirement_id="REQ-SIO2-001",
        sensor_type=SensorType.SILICA_ANALYZER,
        application=MeasurementApplication.PROCESS_CONTROL,
        min_accuracy_class=AccuracyClass.CLASS_C,
        min_range=Decimal("0"),
        max_range=Decimal("1000"),
        unit="ppb",
        response_time_max_sec=Decimal("60.0"),
        calibration_interval_days=30,
        environmental_rating="IP54",
        standard_reference="ASTM D859",
        notes="Colorimetric or photometric method",
    ),

    # Flow measurement for steam
    SensorRequirement(
        requirement_id="REQ-FLOW-001",
        sensor_type=SensorType.VORTEX_METER,
        application=MeasurementApplication.CUSTODY_TRANSFER,
        min_accuracy_class=AccuracyClass.CLASS_B,
        min_range=Decimal("0"),
        max_range=Decimal("100000"),
        unit="kg/h",
        response_time_max_sec=Decimal("1.0"),
        calibration_interval_days=365,
        environmental_rating="IP67",
        standard_reference="ASME PTC 19.5",
    ),
]


# =============================================================================
# SENSOR REGISTRATION AND TRACKING
# =============================================================================

@dataclass
class RegisteredSensor:
    """
    Registered sensor with calibration tracking.

    Maintains complete lifecycle information for a sensor
    including calibration history and current status.
    """

    sensor_id: str
    sensor_type: SensorType
    tag_number: str
    manufacturer: str
    model: str
    serial_number: str
    accuracy_class: AccuracyClass
    range_min: Decimal
    range_max: Decimal
    unit: str
    location: str
    installation_date: datetime
    last_calibration_date: Optional[datetime]
    next_calibration_due: Optional[datetime]
    calibration_interval_days: int
    calibration_status: CalibrationStatus
    nist_traceable: bool
    certificate_number: Optional[str] = None
    application: MeasurementApplication = MeasurementApplication.PROCESS_CONTROL
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "tag_number": self.tag_number,
            "manufacturer": self.manufacturer,
            "model": self.model,
            "serial_number": self.serial_number,
            "accuracy_class": self.accuracy_class.value,
            "range_min": str(self.range_min),
            "range_max": str(self.range_max),
            "unit": self.unit,
            "location": self.location,
            "installation_date": self.installation_date.isoformat(),
            "last_calibration_date": self.last_calibration_date.isoformat() if self.last_calibration_date else None,
            "next_calibration_due": self.next_calibration_due.isoformat() if self.next_calibration_due else None,
            "calibration_interval_days": self.calibration_interval_days,
            "calibration_status": self.calibration_status.value,
            "nist_traceable": self.nist_traceable,
            "certificate_number": self.certificate_number,
            "application": self.application.value,
            "notes": self.notes,
        }

    def update_calibration_status(self) -> CalibrationStatus:
        """Update calibration status based on current date."""
        now = datetime.now(timezone.utc)

        if self.calibration_status == CalibrationStatus.FAILED:
            return CalibrationStatus.FAILED

        if self.calibration_status == CalibrationStatus.BYPASSED:
            return CalibrationStatus.BYPASSED

        if self.next_calibration_due is None:
            self.calibration_status = CalibrationStatus.NOT_CALIBRATED
            return self.calibration_status

        if now > self.next_calibration_due:
            self.calibration_status = CalibrationStatus.OVERDUE
        elif now > self.next_calibration_due - timedelta(days=30):
            self.calibration_status = CalibrationStatus.DUE_SOON
        else:
            self.calibration_status = CalibrationStatus.VALID

        return self.calibration_status


# =============================================================================
# CALIBRATION WORKFLOW
# =============================================================================

@dataclass
class CalibrationRecord:
    """
    Calibration record with full traceability.

    Records complete calibration event including
    as-found/as-left data and NIST traceability.
    """

    calibration_id: str
    sensor_id: str
    calibration_date: datetime
    calibration_due_date: datetime
    technician_id: str
    calibration_lab: str
    certificate_number: str
    nist_traceable: bool
    reference_standard_id: str
    reference_standard_cert: str

    # Calibration data
    test_points: List[Dict[str, Decimal]]  # {reference, as_found, as_left}
    as_found_error_max: Decimal
    as_left_error_max: Decimal
    tolerance: Decimal

    # Result
    passed: bool
    adjusted: bool
    provenance_hash: str

    notes: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "calibration_id": self.calibration_id,
            "sensor_id": self.sensor_id,
            "calibration_date": self.calibration_date.isoformat(),
            "calibration_due_date": self.calibration_due_date.isoformat(),
            "technician_id": self.technician_id,
            "calibration_lab": self.calibration_lab,
            "certificate_number": self.certificate_number,
            "nist_traceable": self.nist_traceable,
            "reference_standard_id": self.reference_standard_id,
            "reference_standard_cert": self.reference_standard_cert,
            "test_points": [
                {k: str(v) for k, v in tp.items()}
                for tp in self.test_points
            ],
            "as_found_error_max": str(self.as_found_error_max),
            "as_left_error_max": str(self.as_left_error_max),
            "tolerance": str(self.tolerance),
            "passed": self.passed,
            "adjusted": self.adjusted,
            "provenance_hash": self.provenance_hash,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# DATA QUALITY SCORING
# =============================================================================

@dataclass
class DataQualityScore:
    """
    Data quality score with component breakdown.

    Provides transparent scoring with individual
    component contributions for audit.
    """

    score_id: str
    timestamp: datetime
    sensor_id: str
    measurement_value: Decimal
    unit: str

    # Component scores (0-100)
    calibration_score: Decimal     # Based on calibration status
    range_score: Decimal           # Value within sensor range
    rate_of_change_score: Decimal  # Rate of change reasonable
    redundancy_score: Decimal      # Comparison to redundant sensors
    environmental_score: Decimal   # Environmental conditions

    # Composite
    overall_score: Decimal
    grade: DataQualityGrade

    # Flags
    flags: List[str]
    provenance_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "score_id": self.score_id,
            "timestamp": self.timestamp.isoformat(),
            "sensor_id": self.sensor_id,
            "measurement_value": str(self.measurement_value),
            "unit": self.unit,
            "calibration_score": str(self.calibration_score),
            "range_score": str(self.range_score),
            "rate_of_change_score": str(self.rate_of_change_score),
            "redundancy_score": str(self.redundancy_score),
            "environmental_score": str(self.environmental_score),
            "overall_score": str(self.overall_score),
            "grade": self.grade.value,
            "flags": self.flags,
            "provenance_hash": self.provenance_hash,
        }


class DataQualityScorer:
    """
    Data quality scoring engine.

    Calculates quality scores using deterministic rules
    following zero-hallucination principles.

    Example:
        >>> scorer = DataQualityScorer()
        >>> score = scorer.calculate_score(
        ...     sensor=registered_sensor,
        ...     value=Decimal("150.5"),
        ...     previous_value=Decimal("150.2"),
        ...     time_delta_sec=60
        ... )
        >>> print(f"Grade: {score.grade.value}")
    """

    VERSION = "1.0.0"

    # Scoring weights
    WEIGHTS = {
        "calibration": Decimal("0.30"),
        "range": Decimal("0.25"),
        "rate_of_change": Decimal("0.20"),
        "redundancy": Decimal("0.15"),
        "environmental": Decimal("0.10"),
    }

    # Rate of change limits by sensor type (per minute)
    RATE_LIMITS = {
        SensorType.RTD_PT100: Decimal("5.0"),        # 5 deg C/min
        SensorType.PRESSURE_TRANSMITTER: Decimal("10.0"),  # 10 bar/min
        SensorType.CONDUCTIVITY_PROBE: Decimal("5.0"),     # 5 umhos/cm/min
        SensorType.VORTEX_METER: Decimal("1000.0"),        # 1000 kg/h/min
    }

    def __init__(self) -> None:
        """Initialize data quality scorer."""
        logger.info("DataQualityScorer initialized")

    def calculate_score(
        self,
        sensor: RegisteredSensor,
        value: Union[Decimal, float],
        previous_value: Optional[Union[Decimal, float]] = None,
        time_delta_sec: Optional[float] = None,
        redundant_values: Optional[List[Decimal]] = None,
        ambient_temperature: Optional[Decimal] = None,
    ) -> DataQualityScore:
        """
        Calculate data quality score for measurement.

        Uses deterministic scoring rules - no ML/LLM.

        Args:
            sensor: Registered sensor information
            value: Current measurement value
            previous_value: Previous measurement for rate check
            time_delta_sec: Time since previous measurement
            redundant_values: Values from redundant sensors
            ambient_temperature: Ambient temperature for env check

        Returns:
            DataQualityScore with grade and component breakdown
        """
        timestamp = datetime.now(timezone.utc)
        value_dec = Decimal(str(value))
        flags: List[str] = []

        # Calculate component scores
        cal_score = self._score_calibration(sensor, flags)
        range_score = self._score_range(sensor, value_dec, flags)
        rate_score = self._score_rate_of_change(
            sensor, value_dec, previous_value, time_delta_sec, flags
        )
        redundancy_score = self._score_redundancy(value_dec, redundant_values, flags)
        env_score = self._score_environmental(sensor, ambient_temperature, flags)

        # Calculate weighted overall score
        overall = (
            cal_score * self.WEIGHTS["calibration"] +
            range_score * self.WEIGHTS["range"] +
            rate_score * self.WEIGHTS["rate_of_change"] +
            redundancy_score * self.WEIGHTS["redundancy"] +
            env_score * self.WEIGHTS["environmental"]
        )

        # Determine grade
        grade = self._determine_grade(overall)

        # Calculate provenance hash
        provenance_data = {
            "sensor_id": sensor.sensor_id,
            "value": str(value_dec),
            "timestamp": timestamp.isoformat(),
            "overall_score": str(overall),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return DataQualityScore(
            score_id=f"DQS-{timestamp.strftime('%Y%m%d%H%M%S')}",
            timestamp=timestamp,
            sensor_id=sensor.sensor_id,
            measurement_value=value_dec,
            unit=sensor.unit,
            calibration_score=cal_score,
            range_score=range_score,
            rate_of_change_score=rate_score,
            redundancy_score=redundancy_score,
            environmental_score=env_score,
            overall_score=overall.quantize(Decimal("0.01")),
            grade=grade,
            flags=flags,
            provenance_hash=provenance_hash,
        )

    def _score_calibration(
        self,
        sensor: RegisteredSensor,
        flags: List[str],
    ) -> Decimal:
        """Score based on calibration status."""
        sensor.update_calibration_status()

        if sensor.calibration_status == CalibrationStatus.VALID:
            return Decimal("100")
        elif sensor.calibration_status == CalibrationStatus.DUE_SOON:
            flags.append("CALIBRATION_DUE_SOON")
            return Decimal("80")
        elif sensor.calibration_status == CalibrationStatus.OVERDUE:
            flags.append("CALIBRATION_OVERDUE")
            return Decimal("40")
        elif sensor.calibration_status == CalibrationStatus.FAILED:
            flags.append("CALIBRATION_FAILED")
            return Decimal("0")
        elif sensor.calibration_status == CalibrationStatus.BYPASSED:
            flags.append("SENSOR_BYPASSED")
            return Decimal("30")
        else:  # NOT_CALIBRATED
            flags.append("NOT_CALIBRATED")
            return Decimal("0")

    def _score_range(
        self,
        sensor: RegisteredSensor,
        value: Decimal,
        flags: List[str],
    ) -> Decimal:
        """Score based on value within sensor range."""
        range_span = sensor.range_max - sensor.range_min

        if value < sensor.range_min:
            flags.append("BELOW_RANGE")
            return Decimal("0")
        elif value > sensor.range_max:
            flags.append("ABOVE_RANGE")
            return Decimal("0")

        # Score based on how centered the value is (penalize extremes)
        position = (value - sensor.range_min) / range_span

        # Full score for middle 80% of range
        if Decimal("0.1") <= position <= Decimal("0.9"):
            return Decimal("100")
        elif Decimal("0.05") <= position < Decimal("0.1") or Decimal("0.9") < position <= Decimal("0.95"):
            flags.append("NEAR_RANGE_LIMIT")
            return Decimal("80")
        else:
            flags.append("AT_RANGE_EXTREME")
            return Decimal("60")

    def _score_rate_of_change(
        self,
        sensor: RegisteredSensor,
        value: Decimal,
        previous_value: Optional[Union[Decimal, float]],
        time_delta_sec: Optional[float],
        flags: List[str],
    ) -> Decimal:
        """Score based on rate of change reasonableness."""
        if previous_value is None or time_delta_sec is None or time_delta_sec == 0:
            return Decimal("100")  # Cannot evaluate

        prev_dec = Decimal(str(previous_value))
        rate_per_min = abs(value - prev_dec) / (Decimal(str(time_delta_sec)) / Decimal("60"))

        # Get rate limit for sensor type
        rate_limit = self.RATE_LIMITS.get(sensor.sensor_type, Decimal("100"))

        if rate_per_min <= rate_limit:
            return Decimal("100")
        elif rate_per_min <= rate_limit * Decimal("2"):
            flags.append("HIGH_RATE_OF_CHANGE")
            return Decimal("70")
        elif rate_per_min <= rate_limit * Decimal("5"):
            flags.append("VERY_HIGH_RATE_OF_CHANGE")
            return Decimal("40")
        else:
            flags.append("SUSPECT_RATE_OF_CHANGE")
            return Decimal("10")

    def _score_redundancy(
        self,
        value: Decimal,
        redundant_values: Optional[List[Decimal]],
        flags: List[str],
    ) -> Decimal:
        """Score based on comparison to redundant sensors."""
        if not redundant_values or len(redundant_values) == 0:
            return Decimal("100")  # No redundancy to check

        # Calculate average of redundant values
        avg_redundant = sum(redundant_values) / len(redundant_values)

        if avg_redundant == Decimal("0"):
            return Decimal("100")

        # Calculate deviation percentage
        deviation = abs((value - avg_redundant) / avg_redundant) * Decimal("100")

        if deviation <= Decimal("1"):
            return Decimal("100")
        elif deviation <= Decimal("2"):
            return Decimal("90")
        elif deviation <= Decimal("5"):
            flags.append("REDUNDANCY_DEVIATION")
            return Decimal("70")
        elif deviation <= Decimal("10"):
            flags.append("HIGH_REDUNDANCY_DEVIATION")
            return Decimal("40")
        else:
            flags.append("REDUNDANCY_MISMATCH")
            return Decimal("10")

    def _score_environmental(
        self,
        sensor: RegisteredSensor,
        ambient_temperature: Optional[Decimal],
        flags: List[str],
    ) -> Decimal:
        """Score based on environmental conditions."""
        if ambient_temperature is None:
            return Decimal("100")  # Cannot evaluate

        # Check if ambient is within typical operating range
        if Decimal("-20") <= ambient_temperature <= Decimal("50"):
            return Decimal("100")
        elif Decimal("-40") <= ambient_temperature <= Decimal("70"):
            flags.append("MARGINAL_AMBIENT_TEMP")
            return Decimal("70")
        else:
            flags.append("EXTREME_AMBIENT_TEMP")
            return Decimal("30")

    def _determine_grade(self, score: Decimal) -> DataQualityGrade:
        """Determine quality grade from score."""
        if score >= Decimal("95"):
            return DataQualityGrade.EXCELLENT
        elif score >= Decimal("85"):
            return DataQualityGrade.GOOD
        elif score >= Decimal("70"):
            return DataQualityGrade.ACCEPTABLE
        elif score >= Decimal("50"):
            return DataQualityGrade.MARGINAL
        elif score > Decimal("0"):
            return DataQualityGrade.POOR
        else:
            return DataQualityGrade.INVALID


# =============================================================================
# UNCERTAINTY REPORTING
# =============================================================================

@dataclass
class UncertaintyBudget:
    """
    Measurement uncertainty budget per GUM/ASME PTC 19.1.

    Documents all uncertainty contributors with
    proper propagation methodology.
    """

    budget_id: str
    measurement_type: str
    measured_value: Decimal
    unit: str
    coverage_factor: Decimal  # k-factor, typically 2 for 95%

    # Uncertainty components
    components: List[Dict[str, Any]]  # {name, type, value, unit, distribution}

    # Combined and expanded
    combined_standard_uncertainty: Decimal
    expanded_uncertainty: Decimal
    confidence_level_percent: Decimal

    # Provenance
    methodology: str
    provenance_hash: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "budget_id": self.budget_id,
            "measurement_type": self.measurement_type,
            "measured_value": str(self.measured_value),
            "unit": self.unit,
            "coverage_factor": str(self.coverage_factor),
            "components": self.components,
            "combined_standard_uncertainty": str(self.combined_standard_uncertainty),
            "expanded_uncertainty": str(self.expanded_uncertainty),
            "confidence_level_percent": str(self.confidence_level_percent),
            "methodology": self.methodology,
            "provenance_hash": self.provenance_hash,
            "created_at": self.created_at.isoformat(),
        }


class UncertaintyCalculator:
    """
    Measurement uncertainty calculator per ASME PTC 19.1.

    Implements Type A (statistical) and Type B (other)
    uncertainty evaluation methods.

    Example:
        >>> calc = UncertaintyCalculator()
        >>> budget = calc.calculate_uncertainty(
        ...     measured_value=Decimal("150.5"),
        ...     sensor_accuracy_pct=Decimal("0.1"),
        ...     repeatability_pct=Decimal("0.05"),
        ...     calibration_uncertainty_pct=Decimal("0.02")
        ... )
        >>> print(f"Expanded uncertainty: +/- {budget.expanded_uncertainty}")
    """

    VERSION = "1.0.0"

    def __init__(self, default_coverage_factor: Decimal = Decimal("2.0")) -> None:
        """
        Initialize uncertainty calculator.

        Args:
            default_coverage_factor: k-factor for expanded uncertainty
        """
        self.default_k = default_coverage_factor
        logger.info(f"UncertaintyCalculator initialized with k={default_coverage_factor}")

    def calculate_uncertainty(
        self,
        measured_value: Union[Decimal, float],
        unit: str,
        sensor_accuracy_pct: Union[Decimal, float],
        repeatability_pct: Optional[Union[Decimal, float]] = None,
        calibration_uncertainty_pct: Optional[Union[Decimal, float]] = None,
        resolution_value: Optional[Union[Decimal, float]] = None,
        environmental_pct: Optional[Union[Decimal, float]] = None,
        measurement_type: str = "steam_quality",
        coverage_factor: Optional[Decimal] = None,
    ) -> UncertaintyBudget:
        """
        Calculate measurement uncertainty budget.

        Uses root-sum-square (RSS) combination per GUM.
        Zero-hallucination: Pure arithmetic calculation.

        Args:
            measured_value: Measured value
            unit: Measurement unit
            sensor_accuracy_pct: Sensor accuracy as % of reading
            repeatability_pct: Measurement repeatability as %
            calibration_uncertainty_pct: Calibration uncertainty as %
            resolution_value: Resolution in measurement units
            environmental_pct: Environmental influence as %
            measurement_type: Type of measurement
            coverage_factor: k-factor (default 2.0 for 95%)

        Returns:
            Complete UncertaintyBudget with provenance
        """
        timestamp = datetime.now(timezone.utc)
        value_dec = Decimal(str(measured_value))
        k = coverage_factor or self.default_k

        components = []
        variances = []

        # Convert percentages to absolute values and build components
        accuracy = Decimal(str(sensor_accuracy_pct)) / Decimal("100") * value_dec
        components.append({
            "name": "sensor_accuracy",
            "type": "B",
            "value": str(accuracy),
            "unit": unit,
            "distribution": "rectangular",
            "divisor": "sqrt(3)",
        })
        # For rectangular distribution, divide by sqrt(3)
        std_accuracy = accuracy / Decimal(str(math.sqrt(3)))
        variances.append(std_accuracy ** 2)

        if repeatability_pct is not None:
            repeat = Decimal(str(repeatability_pct)) / Decimal("100") * value_dec
            components.append({
                "name": "repeatability",
                "type": "A",
                "value": str(repeat),
                "unit": unit,
                "distribution": "normal",
                "divisor": "1",
            })
            variances.append(repeat ** 2)

        if calibration_uncertainty_pct is not None:
            cal = Decimal(str(calibration_uncertainty_pct)) / Decimal("100") * value_dec
            components.append({
                "name": "calibration",
                "type": "B",
                "value": str(cal),
                "unit": unit,
                "distribution": "normal",
                "divisor": "k=2",
            })
            # Calibration uncertainty is usually at k=2, so divide by 2
            std_cal = cal / Decimal("2")
            variances.append(std_cal ** 2)

        if resolution_value is not None:
            res = Decimal(str(resolution_value))
            components.append({
                "name": "resolution",
                "type": "B",
                "value": str(res),
                "unit": unit,
                "distribution": "rectangular",
                "divisor": "sqrt(3)",
            })
            std_res = res / Decimal(str(math.sqrt(3)))
            variances.append(std_res ** 2)

        if environmental_pct is not None:
            env = Decimal(str(environmental_pct)) / Decimal("100") * value_dec
            components.append({
                "name": "environmental",
                "type": "B",
                "value": str(env),
                "unit": unit,
                "distribution": "rectangular",
                "divisor": "sqrt(3)",
            })
            std_env = env / Decimal(str(math.sqrt(3)))
            variances.append(std_env ** 2)

        # Combine using RSS
        combined_variance = sum(variances)
        combined_std = Decimal(str(math.sqrt(float(combined_variance))))
        expanded = combined_std * k

        # Calculate confidence level (k=2 -> 95.45%)
        confidence = Decimal("95.45") if k == Decimal("2") else Decimal("99.73") if k == Decimal("3") else Decimal("68.27")

        # Provenance hash
        provenance_data = {
            "measured_value": str(value_dec),
            "components": len(components),
            "combined_uncertainty": str(combined_std),
            "timestamp": timestamp.isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return UncertaintyBudget(
            budget_id=f"UB-{timestamp.strftime('%Y%m%d%H%M%S')}",
            measurement_type=measurement_type,
            measured_value=value_dec,
            unit=unit,
            coverage_factor=k,
            components=components,
            combined_standard_uncertainty=combined_std.quantize(Decimal("0.0001")),
            expanded_uncertainty=expanded.quantize(Decimal("0.0001")),
            confidence_level_percent=confidence,
            methodology="GUM/ASME PTC 19.1 RSS combination",
            provenance_hash=provenance_hash,
        )


# =============================================================================
# MEASUREMENT GOVERNANCE MANAGER
# =============================================================================

class MeasurementGovernanceManager:
    """
    Central management for measurement governance.

    Coordinates sensor registration, calibration tracking,
    data quality scoring, and uncertainty reporting.

    Example:
        >>> mgr = MeasurementGovernanceManager()
        >>> mgr.register_sensor(sensor)
        >>> mgr.record_calibration(calibration_record)
        >>> score = mgr.score_measurement(sensor_id, value)
    """

    VERSION = "1.0.0"

    def __init__(self) -> None:
        """Initialize measurement governance manager."""
        self._sensors: Dict[str, RegisteredSensor] = {}
        self._calibration_history: Dict[str, List[CalibrationRecord]] = {}
        self._scorer = DataQualityScorer()
        self._uncertainty_calc = UncertaintyCalculator()

        logger.info("MeasurementGovernanceManager initialized")

    def register_sensor(self, sensor: RegisteredSensor) -> None:
        """
        Register a sensor for governance tracking.

        Args:
            sensor: Sensor to register
        """
        self._sensors[sensor.sensor_id] = sensor
        self._calibration_history[sensor.sensor_id] = []
        logger.info(f"Registered sensor: {sensor.sensor_id} ({sensor.tag_number})")

    def get_sensor(self, sensor_id: str) -> Optional[RegisteredSensor]:
        """Get registered sensor by ID."""
        return self._sensors.get(sensor_id)

    def record_calibration(self, record: CalibrationRecord) -> None:
        """
        Record calibration event for a sensor.

        Args:
            record: Calibration record to store
        """
        sensor = self._sensors.get(record.sensor_id)
        if sensor is None:
            raise ValueError(f"Unknown sensor: {record.sensor_id}")

        # Update sensor calibration info
        sensor.last_calibration_date = record.calibration_date
        sensor.next_calibration_due = record.calibration_due_date
        sensor.certificate_number = record.certificate_number

        if record.passed:
            sensor.calibration_status = CalibrationStatus.VALID
        else:
            sensor.calibration_status = CalibrationStatus.FAILED

        # Store in history
        self._calibration_history[record.sensor_id].append(record)

        logger.info(
            f"Recorded calibration for {record.sensor_id}: "
            f"{'PASSED' if record.passed else 'FAILED'}"
        )

    def get_calibration_history(
        self,
        sensor_id: str,
    ) -> List[CalibrationRecord]:
        """Get calibration history for sensor."""
        return self._calibration_history.get(sensor_id, [])

    def get_sensors_due_calibration(
        self,
        within_days: int = 30,
    ) -> List[RegisteredSensor]:
        """
        Get sensors due for calibration within specified days.

        Args:
            within_days: Number of days to look ahead

        Returns:
            List of sensors needing calibration
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=within_days)

        due_sensors = []
        for sensor in self._sensors.values():
            sensor.update_calibration_status()
            if sensor.calibration_status in [
                CalibrationStatus.OVERDUE,
                CalibrationStatus.DUE_SOON,
                CalibrationStatus.NOT_CALIBRATED,
            ]:
                if sensor.next_calibration_due is None or sensor.next_calibration_due <= cutoff:
                    due_sensors.append(sensor)

        return due_sensors

    def score_measurement(
        self,
        sensor_id: str,
        value: Union[Decimal, float],
        previous_value: Optional[Union[Decimal, float]] = None,
        time_delta_sec: Optional[float] = None,
    ) -> DataQualityScore:
        """
        Score measurement data quality.

        Args:
            sensor_id: Sensor identifier
            value: Measured value
            previous_value: Previous value for rate check
            time_delta_sec: Time since previous measurement

        Returns:
            DataQualityScore with grade
        """
        sensor = self._sensors.get(sensor_id)
        if sensor is None:
            raise ValueError(f"Unknown sensor: {sensor_id}")

        return self._scorer.calculate_score(
            sensor=sensor,
            value=value,
            previous_value=previous_value,
            time_delta_sec=time_delta_sec,
        )

    def calculate_uncertainty(
        self,
        sensor_id: str,
        measured_value: Union[Decimal, float],
    ) -> UncertaintyBudget:
        """
        Calculate measurement uncertainty for sensor reading.

        Args:
            sensor_id: Sensor identifier
            measured_value: Measured value

        Returns:
            UncertaintyBudget with expanded uncertainty
        """
        sensor = self._sensors.get(sensor_id)
        if sensor is None:
            raise ValueError(f"Unknown sensor: {sensor_id}")

        # Derive accuracy from accuracy class
        accuracy_map = {
            AccuracyClass.CLASS_A: Decimal("0.1"),
            AccuracyClass.CLASS_B: Decimal("0.25"),
            AccuracyClass.CLASS_C: Decimal("0.5"),
            AccuracyClass.CLASS_D: Decimal("1.0"),
            AccuracyClass.CLASS_E: Decimal("2.0"),
        }
        accuracy_pct = accuracy_map.get(sensor.accuracy_class, Decimal("1.0"))

        return self._uncertainty_calc.calculate_uncertainty(
            measured_value=measured_value,
            unit=sensor.unit,
            sensor_accuracy_pct=accuracy_pct,
            repeatability_pct=accuracy_pct / Decimal("2"),  # Estimate
            calibration_uncertainty_pct=accuracy_pct / Decimal("3"),  # Estimate
        )

    def generate_governance_report(
        self,
        site_id: str,
    ) -> Dict[str, Any]:
        """
        Generate measurement governance status report.

        Args:
            site_id: Site identifier

        Returns:
            Comprehensive governance report
        """
        timestamp = datetime.now(timezone.utc)

        # Summarize calibration status
        status_counts = {status.value: 0 for status in CalibrationStatus}
        for sensor in self._sensors.values():
            sensor.update_calibration_status()
            status_counts[sensor.calibration_status.value] += 1

        # Get overdue sensors
        overdue = [
            s.to_dict() for s in self._sensors.values()
            if s.calibration_status == CalibrationStatus.OVERDUE
        ]

        # Calculate report hash
        report_data = {
            "site_id": site_id,
            "timestamp": timestamp.isoformat(),
            "sensor_count": len(self._sensors),
            "status_counts": status_counts,
        }
        report_hash = hashlib.sha256(
            json.dumps(report_data, sort_keys=True).encode()
        ).hexdigest()

        return {
            "report_metadata": {
                "report_id": f"MGR-{timestamp.strftime('%Y%m%d%H%M%S')}",
                "site_id": site_id,
                "generated_at": timestamp.isoformat(),
                "report_hash": report_hash,
                "generator_version": self.VERSION,
            },
            "summary": {
                "total_sensors": len(self._sensors),
                "calibration_status": status_counts,
                "overdue_count": len(overdue),
                "compliance_rate": (
                    f"{(status_counts[CalibrationStatus.VALID.value] / len(self._sensors) * 100):.1f}%"
                    if self._sensors else "N/A"
                ),
            },
            "overdue_sensors": overdue,
            "sensors": [s.to_dict() for s in self._sensors.values()],
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_governance_manager() -> MeasurementGovernanceManager:
    """Factory function to create governance manager."""
    return MeasurementGovernanceManager()


def create_data_quality_scorer() -> DataQualityScorer:
    """Factory function to create data quality scorer."""
    return DataQualityScorer()


def create_uncertainty_calculator(
    coverage_factor: Decimal = Decimal("2.0"),
) -> UncertaintyCalculator:
    """
    Factory function to create uncertainty calculator.

    Args:
        coverage_factor: k-factor for expanded uncertainty

    Returns:
        Configured UncertaintyCalculator
    """
    return UncertaintyCalculator(coverage_factor)


def get_sensor_requirements(
    sensor_type: Optional[SensorType] = None,
    application: Optional[MeasurementApplication] = None,
) -> List[SensorRequirement]:
    """
    Get sensor requirements, optionally filtered.

    Args:
        sensor_type: Filter by sensor type
        application: Filter by application

    Returns:
        List of matching SensorRequirement objects
    """
    requirements = SENSOR_REQUIREMENTS

    if sensor_type:
        requirements = [r for r in requirements if r.sensor_type == sensor_type]

    if application:
        requirements = [r for r in requirements if r.application == application]

    return requirements
