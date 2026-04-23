"""
GL-012_SteamQual - Domain Models

Domain-specific enumerations and types for steam quality monitoring and control.
These models define the semantic vocabulary for the SteamQual agent.

Standards Reference:
- IAPWS-IF97: Industrial Formulation for Water and Steam
- ASME PTC 19.11: Steam and Water Sampling, Conditioning, and Analysis
- ISO 9806: Solar thermal collectors

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from enum import Enum, IntEnum
from typing import Optional


# =============================================================================
# Steam State Enumerations
# =============================================================================


class SteamState(str, Enum):
    """
    Thermodynamic state of steam based on IAPWS-IF97 regions.

    Determines the phase and calculation region for steam properties.
    Critical for accurate enthalpy and quality calculations.
    """

    SUBCOOLED = "subcooled"
    """Compressed liquid below saturation temperature. Quality undefined."""

    SATURATED = "saturated"
    """Two-phase mixture at saturation. Quality x in [0, 1]."""

    SUPERHEATED = "superheated"
    """Vapor above saturation temperature. Quality = 1.0 by definition."""

    SUPERCRITICAL = "supercritical"
    """Above critical point (Pc=22.064 MPa, Tc=647.096 K). Quality undefined."""

    UNKNOWN = "unknown"
    """State cannot be determined from available data."""

    def is_two_phase(self) -> bool:
        """Check if state is in two-phase region where quality is defined."""
        return self == SteamState.SATURATED

    def requires_quality(self) -> bool:
        """Check if state requires quality specification to be fully defined."""
        return self == SteamState.SATURATED


class SteamRegion(IntEnum):
    """
    IAPWS-IF97 thermodynamic regions for water and steam.

    Used for selecting appropriate calculation formulas.
    """

    REGION_1 = 1
    """Compressed liquid (subcooled water)."""

    REGION_2 = 2
    """Superheated steam (vapor phase)."""

    REGION_3 = 3
    """Near-critical region (requires special treatment)."""

    REGION_4 = 4
    """Two-phase (wet steam, saturation curve)."""

    REGION_5 = 5
    """High-temperature steam (>800 C)."""


# =============================================================================
# Event Type Enumerations
# =============================================================================


class EventType(str, Enum):
    """
    Types of quality events detected by the SteamQual agent.

    Events are generated when steam quality deviates from expected
    operating conditions or when risk thresholds are exceeded.
    """

    # Quality degradation events
    LOW_DRYNESS = "low_dryness"
    """Dryness fraction below minimum threshold (x < x_min)."""

    HIGH_MOISTURE = "high_moisture"
    """Moisture content exceeds acceptable limit."""

    CARRYOVER_RISK = "carryover_risk"
    """Risk of liquid carryover to consumers detected."""

    CARRYOVER_DETECTED = "carryover_detected"
    """Liquid carryover to consumers confirmed."""

    # Process condition events
    SUBCOOLED_DETECTED = "subcooled_detected"
    """Steam appears subcooled (condensate accumulation)."""

    EXCESS_SUPERHEAT = "excess_superheat"
    """Superheat exceeds optimal range."""

    FLASH_STEAM_LOSS = "flash_steam_loss"
    """Unrecovered flash steam detected."""

    # Sensor and data events
    SENSOR_DISAGREEMENT = "sensor_disagreement"
    """Multiple sensors show inconsistent readings."""

    DATA_QUALITY_LOW = "data_quality_low"
    """Input data quality below acceptable threshold."""

    STALE_DATA = "stale_data"
    """Data age exceeds maximum allowed latency."""

    # Separator performance events
    SEPARATOR_INEFFICIENT = "separator_inefficient"
    """Moisture separator not performing adequately."""

    DRAIN_FLOW_ABNORMAL = "drain_flow_abnormal"
    """Separator drain flow outside expected range."""

    # Header pressure events
    HEADER_PRESSURE_LOW = "header_pressure_low"
    """Header pressure below minimum for consumer requirements."""

    HEADER_PRESSURE_HIGH = "header_pressure_high"
    """Header pressure exceeds maximum safe operating limit."""

    # Consumer protection events
    TURBINE_PROTECTION = "turbine_protection"
    """Quality alert for turbine protection."""

    HEAT_EXCHANGER_PROTECTION = "heat_exchanger_protection"
    """Quality alert for heat exchanger protection."""

    # System events
    QUALITY_RECOVERED = "quality_recovered"
    """Steam quality returned to acceptable range."""

    CONSTRAINT_VIOLATED = "constraint_violated"
    """Operating constraint violated."""

    CONSTRAINT_WARNING = "constraint_warning"
    """Approaching constraint limit."""


# =============================================================================
# Severity Enumerations
# =============================================================================


class Severity(str, Enum):
    """
    Severity levels for quality events.

    Follows ISA-18.2 alarm severity levels adapted for steam quality.
    Lower numeric value = less severe.
    """

    S0_INFO = "S0_INFO"
    """Informational - no action required. For logging and trending only."""

    S1_ADVISORY = "S1_ADVISORY"
    """Advisory - operator awareness recommended. No immediate action."""

    S2_WARNING = "S2_WARNING"
    """Warning - action should be taken to prevent escalation."""

    S3_CRITICAL = "S3_CRITICAL"
    """Critical - immediate action required. Equipment protection at risk."""

    @property
    def numeric_level(self) -> int:
        """Get numeric severity level (0-3) for comparisons."""
        level_map = {
            Severity.S0_INFO: 0,
            Severity.S1_ADVISORY: 1,
            Severity.S2_WARNING: 2,
            Severity.S3_CRITICAL: 3,
        }
        return level_map[self]

    def __lt__(self, other: "Severity") -> bool:
        """Compare severity levels."""
        return self.numeric_level < other.numeric_level

    def __gt__(self, other: "Severity") -> bool:
        """Compare severity levels."""
        return self.numeric_level > other.numeric_level

    def __le__(self, other: "Severity") -> bool:
        """Compare severity levels."""
        return self.numeric_level <= other.numeric_level

    def __ge__(self, other: "Severity") -> bool:
        """Compare severity levels."""
        return self.numeric_level >= other.numeric_level

    @classmethod
    def from_numeric(cls, level: int) -> "Severity":
        """Create severity from numeric level."""
        level_map = {
            0: cls.S0_INFO,
            1: cls.S1_ADVISORY,
            2: cls.S2_WARNING,
            3: cls.S3_CRITICAL,
        }
        return level_map.get(level, cls.S2_WARNING)


class AlarmPriority(str, Enum):
    """
    Alarm priority for operator response (ISA-18.2 aligned).

    Determines response time requirements and escalation paths.
    """

    DIAGNOSTIC = "diagnostic"
    """Diagnostic information - no operator action required."""

    LOW = "low"
    """Low priority - respond within shift."""

    MEDIUM = "medium"
    """Medium priority - respond within 30 minutes."""

    HIGH = "high"
    """High priority - respond within 10 minutes."""

    EMERGENCY = "emergency"
    """Emergency - immediate response required."""


# =============================================================================
# Consumer Class Enumerations
# =============================================================================


class ConsumerClass(str, Enum):
    """
    Classification of steam consumers by quality requirements.

    Different consumer types have vastly different tolerance for
    moisture in steam. Turbines are most sensitive, while some
    heating applications can tolerate significant moisture.
    """

    TURBINE = "turbine"
    """Steam turbines - highest quality requirements (x > 0.995).
    Moisture causes blade erosion and efficiency loss."""

    HEAT_EXCHANGER = "heat_exchanger"
    """Shell-and-tube heat exchangers - moderate requirements (x > 0.95).
    Moisture affects heat transfer and can cause water hammer."""

    STERILIZER = "sterilizer"
    """Autoclaves and sterilization - high requirements (x > 0.97).
    Dry steam required for effective sterilization."""

    HUMIDIFIER = "humidifier"
    """Direct steam humidification - moderate requirements (x > 0.90).
    Some moisture acceptable for humidity applications."""

    PROCESS_HEATING = "process_heating"
    """General process heating - lower requirements (x > 0.85).
    Direct heating applications with condensate drainage."""

    DESUPERHEATER = "desuperheater"
    """Desuperheating stations - special handling.
    May intentionally operate with saturated steam."""

    EJECTOR = "ejector"
    """Steam ejectors/vacuum systems - moderate requirements (x > 0.95).
    Moisture reduces ejector efficiency."""

    REBOILER = "reboiler"
    """Distillation reboilers - moderate requirements (x > 0.92).
    Latent heat transfer applications."""

    DRYER = "dryer"
    """Industrial dryers - varies by application (x > 0.90).
    Paper, textile, and food drying processes."""

    REFORMER = "reformer"
    """Steam methane reformers - high requirements (x > 0.97).
    Process steam for hydrogen production."""

    ATOMIZATION = "atomization"
    """Steam atomization (burners, nozzles) - moderate (x > 0.93).
    Fuel atomization in combustion systems."""

    TRACING = "tracing"
    """Steam tracing/jacketing - lower requirements (x > 0.85).
    Pipe and vessel heating."""

    CLEANING = "cleaning"
    """Steam cleaning/CIP - moderate requirements (x > 0.90).
    Clean-in-place and washdown systems."""

    GENERAL = "general"
    """Unclassified consumers - default moderate requirements (x > 0.90)."""

    @property
    def min_quality(self) -> float:
        """Get minimum acceptable steam quality (dryness fraction) for this class."""
        quality_map = {
            ConsumerClass.TURBINE: 0.995,
            ConsumerClass.HEAT_EXCHANGER: 0.95,
            ConsumerClass.STERILIZER: 0.97,
            ConsumerClass.HUMIDIFIER: 0.90,
            ConsumerClass.PROCESS_HEATING: 0.85,
            ConsumerClass.DESUPERHEATER: 0.85,
            ConsumerClass.EJECTOR: 0.95,
            ConsumerClass.REBOILER: 0.92,
            ConsumerClass.DRYER: 0.90,
            ConsumerClass.REFORMER: 0.97,
            ConsumerClass.ATOMIZATION: 0.93,
            ConsumerClass.TRACING: 0.85,
            ConsumerClass.CLEANING: 0.90,
            ConsumerClass.GENERAL: 0.90,
        }
        return quality_map.get(self, 0.90)

    @property
    def is_critical(self) -> bool:
        """Check if this consumer class has critical quality requirements."""
        return self in {
            ConsumerClass.TURBINE,
            ConsumerClass.STERILIZER,
            ConsumerClass.REFORMER,
        }


# =============================================================================
# Data Quality Enumerations
# =============================================================================


class DataQualityFlag(str, Enum):
    """
    Data quality indicators for sensor measurements.

    Used to assess reliability of input data for quality estimation.
    """

    GOOD = "good"
    """Data is valid and within expected range."""

    UNCERTAIN = "uncertain"
    """Data may have quality issues but is usable."""

    BAD = "bad"
    """Data is invalid or outside physical bounds."""

    STALE = "stale"
    """Data age exceeds maximum allowed latency."""

    FROZEN = "frozen"
    """Data value has not changed for extended period (sensor stuck)."""

    OUT_OF_RANGE = "out_of_range"
    """Data outside configured sensor range."""

    SUBSTITUTED = "substituted"
    """Original value replaced with estimate or default."""

    MANUAL = "manual"
    """Manually entered value (not from sensor)."""

    @property
    def is_usable(self) -> bool:
        """Check if data with this flag can be used for calculations."""
        return self in {
            DataQualityFlag.GOOD,
            DataQualityFlag.UNCERTAIN,
            DataQualityFlag.SUBSTITUTED,
            DataQualityFlag.MANUAL,
        }


class EstimationMethod(str, Enum):
    """
    Methods for estimating steam quality (dryness fraction).

    Different methods have different accuracy and data requirements.
    """

    ENTHALPY = "enthalpy"
    """From specific enthalpy: x = (h - hf) / hfg. Most accurate."""

    ENTROPY = "entropy"
    """From specific entropy: x = (s - sf) / sfg."""

    SPECIFIC_VOLUME = "specific_volume"
    """From specific volume: x = (v - vf) / (vg - vf)."""

    TEMPERATURE_PRESSURE = "temperature_pressure"
    """From T-P relationship relative to saturation curve."""

    SEPARATOR_MASS_BALANCE = "separator_mass_balance"
    """From moisture separator mass balance."""

    ORIFICE_CORRELATION = "orifice_correlation"
    """From orifice plate differential with wet steam correction."""

    CALORIMETRIC = "calorimetric"
    """From throttling or barrel calorimeter measurement."""

    MACHINE_LEARNING = "machine_learning"
    """ML model-based estimation (non-deterministic, for advisory only)."""

    HISTORICAL = "historical"
    """Based on historical average for similar conditions."""

    ASSUMED = "assumed"
    """Assumed value when no measurement available."""

    @property
    def is_deterministic(self) -> bool:
        """Check if this method produces deterministic results."""
        return self not in {
            EstimationMethod.MACHINE_LEARNING,
            EstimationMethod.HISTORICAL,
            EstimationMethod.ASSUMED,
        }


class ConstraintType(str, Enum):
    """
    Types of constraints for steam quality optimization.

    Used when providing constraints to GL-003 optimizer.
    """

    QUALITY_MIN = "quality_min"
    """Minimum steam quality constraint."""

    QUALITY_MAX = "quality_max"
    """Maximum steam quality constraint (rare, for desuperheating)."""

    PRESSURE_MIN = "pressure_min"
    """Minimum header pressure constraint."""

    PRESSURE_MAX = "pressure_max"
    """Maximum header pressure constraint."""

    TEMPERATURE_MIN = "temperature_min"
    """Minimum temperature constraint."""

    TEMPERATURE_MAX = "temperature_max"
    """Maximum temperature constraint."""

    FLOW_MIN = "flow_min"
    """Minimum flow rate constraint."""

    FLOW_MAX = "flow_max"
    """Maximum flow rate constraint."""

    SUPERHEAT_MIN = "superheat_min"
    """Minimum superheat constraint."""

    MOISTURE_MAX = "moisture_max"
    """Maximum moisture content constraint."""


# =============================================================================
# Header Type Enumerations
# =============================================================================


class HeaderType(str, Enum):
    """
    Types of steam headers in distribution system.

    Defines pressure class and typical operating conditions.
    """

    HIGH_PRESSURE = "high_pressure"
    """HP header, typically >40 bar / 580 psig."""

    MEDIUM_PRESSURE = "medium_pressure"
    """MP header, typically 10-40 bar / 150-580 psig."""

    LOW_PRESSURE = "low_pressure"
    """LP header, typically 1-10 bar / 15-150 psig."""

    VERY_LOW_PRESSURE = "very_low_pressure"
    """VLP header, typically <1 bar / <15 psig."""

    VACUUM = "vacuum"
    """Vacuum steam systems, <1 atm."""


class SeparatorType(str, Enum):
    """
    Types of moisture separators in steam systems.
    """

    CYCLONE = "cyclone"
    """Cyclone/centrifugal separator."""

    MESH_PAD = "mesh_pad"
    """Wire mesh demister pad."""

    VANE = "vane"
    """Vane-type separator."""

    BAFFLE = "baffle"
    """Baffle plate separator."""

    COMBINED = "combined"
    """Combined separator (multiple mechanisms)."""


# =============================================================================
# Recommendation Action Enumerations
# =============================================================================


class RecommendationAction(str, Enum):
    """
    Recommended actions for quality issues.
    """

    NO_ACTION = "no_action"
    """No action required - conditions acceptable."""

    MONITOR = "monitor"
    """Continue monitoring - no immediate action."""

    ADJUST_BLOWDOWN = "adjust_blowdown"
    """Adjust boiler blowdown rate."""

    ADJUST_SEPARATOR_DRAIN = "adjust_separator_drain"
    """Adjust separator drain valve."""

    INCREASE_SUPERHEAT = "increase_superheat"
    """Increase superheat at source."""

    REDUCE_FLOW = "reduce_flow"
    """Reduce steam flow rate."""

    DIVERT_FLOW = "divert_flow"
    """Divert steam to different header."""

    ISOLATE_CONSUMER = "isolate_consumer"
    """Isolate affected consumer."""

    ALERT_OPERATOR = "alert_operator"
    """Alert operator for manual intervention."""

    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    """Emergency shutdown of affected equipment."""

    MAINTENANCE_REQUIRED = "maintenance_required"
    """Schedule maintenance for equipment."""

    INVESTIGATE = "investigate"
    """Investigate root cause of quality issue."""
