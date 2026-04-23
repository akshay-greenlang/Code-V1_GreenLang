"""
GL-012 STEAMQUAL - Tag Mapping Configuration

Maps OT/SCADA tags to internal GreenLang names with:
- OT tag to internal name translation
- Unit conversion specifications
- Tag quality flags
- Validation rules

Tag Naming Convention:
    OT format:  <system>_<equipment>_<measurement>_<instance>
    Internal:   <domain>.<asset>.<parameter>

Example Mappings:
    OT: STEAM_SEP_DP_001      -> Internal: separator.dp.primary
    OT: STEAM_HDR_P_001       -> Internal: header.pressure.main
    OT: STEAM_QUAL_X_001      -> Internal: quality.dryness.estimated
    OT: DRAIN_VLV_POS_001     -> Internal: drain.valve.position

Playbook Requirements:
- Support multiple tag naming conventions (ISA-5.1, site-specific)
- Unit normalization to SI (kPa, degC, kg/s, fraction)
- Quality flag propagation from OT systems
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntFlag
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import re
import json
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class TagQualityFlag(IntFlag):
    """Tag quality flags for data validation."""
    GOOD = 0x00
    STALE = 0x01
    UNCERTAIN = 0x02
    BAD = 0x04
    SUBSTITUTED = 0x08
    MANUAL_OVERRIDE = 0x10
    CONFIG_ERROR = 0x20
    COMM_ERROR = 0x40
    OUT_OF_RANGE = 0x80

    def is_usable(self) -> bool:
        """Check if data is usable for calculations."""
        return not (self & (TagQualityFlag.BAD | TagQualityFlag.CONFIG_ERROR))

    def is_good(self) -> bool:
        """Check if data quality is good."""
        return self == TagQualityFlag.GOOD


class UnitCategory(Enum):
    """Engineering unit categories."""
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    MASS_FLOW = "mass_flow"
    VOLUMETRIC_FLOW = "volumetric_flow"
    DIFFERENTIAL_PRESSURE = "differential_pressure"
    PERCENTAGE = "percentage"
    CONCENTRATION = "concentration"
    POWER = "power"
    ENERGY = "energy"
    DIMENSIONLESS = "dimensionless"
    TIME = "time"


@dataclass
class UnitConversion:
    """
    Unit conversion specification.

    Defines conversion from source OT unit to target internal unit.
    Supports linear transformations: target = (source + offset) * factor
    """
    from_unit: str
    to_unit: str
    factor: float = 1.0
    offset: float = 0.0
    category: UnitCategory = UnitCategory.DIMENSIONLESS

    def convert(self, value: float) -> float:
        """Convert value from source to target unit."""
        return (value + self.offset) * self.factor

    def reverse(self, value: float) -> float:
        """Convert value from target back to source unit."""
        if self.factor == 0:
            return 0.0
        return (value / self.factor) - self.offset


# Standard unit conversions for steam quality control
STANDARD_CONVERSIONS: Dict[Tuple[str, str], UnitConversion] = {
    # Pressure
    ("psig", "kPa"): UnitConversion("psig", "kPa", 6.89476, 14.696, UnitCategory.PRESSURE),
    ("psia", "kPa"): UnitConversion("psia", "kPa", 6.89476, 0.0, UnitCategory.PRESSURE),
    ("bar", "kPa"): UnitConversion("bar", "kPa", 100.0, 0.0, UnitCategory.PRESSURE),
    ("barg", "kPa"): UnitConversion("barg", "kPa", 100.0, 1.01325, UnitCategory.PRESSURE),
    ("kPa", "kPa"): UnitConversion("kPa", "kPa", 1.0, 0.0, UnitCategory.PRESSURE),
    ("MPa", "kPa"): UnitConversion("MPa", "kPa", 1000.0, 0.0, UnitCategory.PRESSURE),
    ("inH2O", "kPa"): UnitConversion("inH2O", "kPa", 0.249089, 0.0, UnitCategory.DIFFERENTIAL_PRESSURE),

    # Temperature
    ("degF", "degC"): UnitConversion("degF", "degC", 5/9, -32.0, UnitCategory.TEMPERATURE),
    ("F", "degC"): UnitConversion("F", "degC", 5/9, -32.0, UnitCategory.TEMPERATURE),
    ("degC", "degC"): UnitConversion("degC", "degC", 1.0, 0.0, UnitCategory.TEMPERATURE),
    ("C", "degC"): UnitConversion("C", "degC", 1.0, 0.0, UnitCategory.TEMPERATURE),
    ("K", "degC"): UnitConversion("K", "degC", 1.0, -273.15, UnitCategory.TEMPERATURE),

    # Mass flow
    ("lb/hr", "kg/s"): UnitConversion("lb/hr", "kg/s", 0.000125998, 0.0, UnitCategory.MASS_FLOW),
    ("klb/hr", "kg/s"): UnitConversion("klb/hr", "kg/s", 0.125998, 0.0, UnitCategory.MASS_FLOW),
    ("kg/hr", "kg/s"): UnitConversion("kg/hr", "kg/s", 1/3600, 0.0, UnitCategory.MASS_FLOW),
    ("kg/s", "kg/s"): UnitConversion("kg/s", "kg/s", 1.0, 0.0, UnitCategory.MASS_FLOW),
    ("t/hr", "kg/s"): UnitConversion("t/hr", "kg/s", 1000/3600, 0.0, UnitCategory.MASS_FLOW),

    # Percentage/fraction
    ("%", "fraction"): UnitConversion("%", "fraction", 0.01, 0.0, UnitCategory.PERCENTAGE),
    ("fraction", "fraction"): UnitConversion("fraction", "fraction", 1.0, 0.0, UnitCategory.PERCENTAGE),
    ("%", "%"): UnitConversion("%", "%", 1.0, 0.0, UnitCategory.PERCENTAGE),

    # Concentration
    ("ppm", "ppm"): UnitConversion("ppm", "ppm", 1.0, 0.0, UnitCategory.CONCENTRATION),
    ("ppb", "ppb"): UnitConversion("ppb", "ppb", 1.0, 0.0, UnitCategory.CONCENTRATION),
    ("ppb", "ppm"): UnitConversion("ppb", "ppm", 0.001, 0.0, UnitCategory.CONCENTRATION),
    ("uS/cm", "uS/cm"): UnitConversion("uS/cm", "uS/cm", 1.0, 0.0, UnitCategory.CONCENTRATION),

    # Power
    ("kW", "kW"): UnitConversion("kW", "kW", 1.0, 0.0, UnitCategory.POWER),
    ("MW", "kW"): UnitConversion("MW", "kW", 1000.0, 0.0, UnitCategory.POWER),
    ("hp", "kW"): UnitConversion("hp", "kW", 0.7457, 0.0, UnitCategory.POWER),
}


@dataclass
class TagMetadata:
    """Metadata for a mapped tag."""
    internal_name: str
    ot_tag: str
    description: str = ""

    # Type information
    data_type: str = "float64"
    unit: str = ""
    target_unit: str = ""

    # Validation limits
    low_limit: Optional[float] = None
    high_limit: Optional[float] = None
    low_alarm: Optional[float] = None
    high_alarm: Optional[float] = None

    # Rate of change limits
    max_rate_per_second: Optional[float] = None

    # Quality settings
    stale_timeout_s: float = 10.0
    quality_threshold: TagQualityFlag = TagQualityFlag.GOOD

    # Scaling
    raw_min: float = 0.0
    raw_max: float = 100.0
    eng_min: float = 0.0
    eng_max: float = 100.0

    # Source information
    source_system: str = ""
    sample_rate_ms: int = 1000

    # Classification
    is_critical: bool = False
    is_calculated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "internal_name": self.internal_name,
            "ot_tag": self.ot_tag,
            "description": self.description,
            "data_type": self.data_type,
            "unit": self.unit,
            "target_unit": self.target_unit,
            "low_limit": self.low_limit,
            "high_limit": self.high_limit,
            "stale_timeout_s": self.stale_timeout_s,
            "is_critical": self.is_critical,
        }


@dataclass
class TagMapping:
    """
    Single tag mapping from OT to internal name.

    Includes unit conversion and validation specifications.
    """
    ot_tag: str
    internal_name: str
    metadata: TagMetadata
    conversion: Optional[UnitConversion] = None

    # Pattern matching for dynamic tag resolution
    ot_pattern: Optional[str] = None
    internal_template: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    modified_at: Optional[datetime] = None

    def apply(self, value: float) -> Tuple[float, str]:
        """
        Apply conversion to value.

        Returns:
            Tuple of (converted_value, target_unit)
        """
        if self.conversion:
            return self.conversion.convert(value), self.conversion.to_unit
        return value, self.metadata.unit

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ot_tag": self.ot_tag,
            "internal_name": self.internal_name,
            "metadata": self.metadata.to_dict(),
            "conversion": {
                "from_unit": self.conversion.from_unit,
                "to_unit": self.conversion.to_unit,
                "factor": self.conversion.factor,
                "offset": self.conversion.offset,
            } if self.conversion else None,
        }


@dataclass
class TagValidationResult:
    """Result of tag value validation."""
    is_valid: bool
    quality_flag: TagQualityFlag
    issues: List[str] = field(default_factory=list)
    corrected_value: Optional[float] = None

    # Detailed checks
    range_valid: bool = True
    rate_valid: bool = True
    staleness_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "quality_flag": int(self.quality_flag),
            "issues": self.issues,
            "corrected_value": self.corrected_value,
            "range_valid": self.range_valid,
            "rate_valid": self.rate_valid,
            "staleness_valid": self.staleness_valid,
        }


@dataclass
class TagMapperConfig:
    """Tag mapper configuration."""
    # Mapping file path
    mapping_file: Optional[str] = None

    # Default units (SI)
    default_pressure_unit: str = "kPa"
    default_temperature_unit: str = "degC"
    default_flow_unit: str = "kg/s"
    default_quality_unit: str = "fraction"

    # Validation settings
    enable_range_validation: bool = True
    enable_rate_validation: bool = True
    enable_staleness_check: bool = True

    # Quality settings
    min_quality_for_calculations: TagQualityFlag = TagQualityFlag.UNCERTAIN

    # Caching
    cache_enabled: bool = True
    cache_ttl_s: float = 60.0


class SteamQualityTagSet:
    """
    Standard tag set for steam quality control.

    Defines the canonical internal names for steam quality parameters.
    """

    # Header/Process measurements
    HEADER_PRESSURE = "header.pressure.main"
    HEADER_TEMPERATURE = "header.temperature.main"
    HEADER_FLOW = "header.flow.main"
    HEADER_ENTHALPY = "header.enthalpy.calculated"

    # Steam quality measurements
    QUALITY_DRYNESS = "quality.dryness.estimated"
    QUALITY_DRYNESS_MIN = "quality.dryness.minimum"
    QUALITY_SUPERHEAT = "quality.superheat.delta_t"
    QUALITY_CARRYOVER = "quality.carryover.rate"

    # Separator measurements
    SEPARATOR_DP = "separator.dp.primary"
    SEPARATOR_EFFICIENCY = "separator.efficiency.calculated"
    SEPARATOR_INLET_PRESSURE = "separator.pressure.inlet"
    SEPARATOR_OUTLET_PRESSURE = "separator.pressure.outlet"

    # Drain system
    DRAIN_VALVE_POSITION = "drain.valve.position"
    DRAIN_DUTY_CYCLE = "drain.duty.cycle"
    DRAIN_FLOW = "drain.flow.condensate"
    DRAIN_TEMPERATURE = "drain.temperature.condensate"

    # Chemistry
    CHEM_CONDUCTIVITY = "chemistry.conductivity.cation"
    CHEM_PH = "chemistry.ph.value"
    CHEM_SILICA = "chemistry.silica.concentration"
    CHEM_DISSOLVED_O2 = "chemistry.do2.concentration"

    # Constraints (from GL-003)
    CONSTRAINT_X_MIN = "constraint.quality.x_min"
    CONSTRAINT_DELTA_T_MIN = "constraint.superheat.delta_t_min"
    CONSTRAINT_MAX_DRAIN_DUTY = "constraint.drain.max_duty_cycle"
    CONSTRAINT_MAX_RAMP_RATE = "constraint.ramp.max_rate"

    # Uncertainty
    UNCERTAINTY_QUALITY = "uncertainty.quality.estimate"
    UNCERTAINTY_CARRYOVER = "uncertainty.carryover.estimate"

    @classmethod
    def get_all_tags(cls) -> List[str]:
        """Get all defined tag names."""
        return [
            v for k, v in cls.__dict__.items()
            if not k.startswith("_") and isinstance(v, str)
        ]

    @classmethod
    def get_critical_tags(cls) -> List[str]:
        """Get tags critical for steam quality control."""
        return [
            cls.HEADER_PRESSURE,
            cls.HEADER_TEMPERATURE,
            cls.HEADER_FLOW,
            cls.QUALITY_DRYNESS,
            cls.SEPARATOR_DP,
            cls.DRAIN_VALVE_POSITION,
        ]


class TagMapper:
    """
    Tag mapping service for GL-012 STEAMQUAL.

    Maps OT/SCADA tags to internal GreenLang names with:
    - Bidirectional mapping (OT <-> internal)
    - Unit conversion
    - Quality flag management
    - Value validation

    Example:
        mapper = TagMapper(TagMapperConfig())

        # Load mappings
        await mapper.load_mappings("config/tags.json")

        # Map OT tag to internal
        internal = mapper.to_internal("STEAM_HDR_P_001")  # -> "header.pressure.main"

        # Convert value with units
        value, unit = mapper.convert_value("STEAM_HDR_P_001", 150.0)  # psig -> kPa

        # Validate value
        result = mapper.validate("header.pressure.main", 1034.0)
    """

    def __init__(self, config: Optional[TagMapperConfig] = None) -> None:
        """Initialize tag mapper."""
        self.config = config or TagMapperConfig()

        # Mapping dictionaries
        self._ot_to_internal: Dict[str, TagMapping] = {}
        self._internal_to_ot: Dict[str, TagMapping] = {}

        # Pattern-based mappings
        self._patterns: List[Tuple[re.Pattern, str, TagMetadata]] = []

        # Value history for rate validation
        self._value_history: Dict[str, Tuple[float, datetime]] = {}

        # Cache
        self._cache: Dict[str, Tuple[Any, datetime]] = {}

        # Statistics
        self._stats = {
            "mappings_loaded": 0,
            "conversions": 0,
            "validations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        logger.info("TagMapper initialized")

    async def load_mappings(self, config_path: str) -> int:
        """
        Load tag mappings from configuration file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            Number of mappings loaded
        """
        path = Path(config_path)

        if not path.exists():
            logger.warning(f"Mapping file not found: {config_path}, using defaults")
            self._load_default_mappings()
            return self._stats["mappings_loaded"]

        try:
            with open(path, "r") as f:
                data = json.load(f)

            for mapping_data in data.get("mappings", []):
                self._add_mapping_from_dict(mapping_data)

            for pattern_data in data.get("patterns", []):
                self._add_pattern_from_dict(pattern_data)

            logger.info(f"Loaded {self._stats['mappings_loaded']} tag mappings")
            return self._stats["mappings_loaded"]

        except Exception as e:
            logger.error(f"Error loading mappings: {e}")
            self._load_default_mappings()
            return self._stats["mappings_loaded"]

    def _add_mapping_from_dict(self, data: Dict[str, Any]) -> None:
        """Add mapping from dictionary data."""
        try:
            metadata = TagMetadata(
                internal_name=data["internal_name"],
                ot_tag=data["ot_tag"],
                description=data.get("description", ""),
                unit=data.get("unit", ""),
                target_unit=data.get("target_unit", ""),
                low_limit=data.get("low_limit"),
                high_limit=data.get("high_limit"),
                max_rate_per_second=data.get("max_rate_per_second"),
                stale_timeout_s=data.get("stale_timeout_s", 10.0),
                is_critical=data.get("is_critical", False),
            )

            # Find conversion
            conversion = None
            if metadata.unit and metadata.target_unit:
                key = (metadata.unit, metadata.target_unit)
                conversion = STANDARD_CONVERSIONS.get(key)

            mapping = TagMapping(
                ot_tag=data["ot_tag"],
                internal_name=data["internal_name"],
                metadata=metadata,
                conversion=conversion,
            )

            self.add_mapping(mapping)

        except Exception as e:
            logger.warning(f"Error parsing mapping: {e}")

    def _add_pattern_from_dict(self, data: Dict[str, Any]) -> None:
        """Add pattern-based mapping."""
        try:
            pattern = re.compile(data["pattern"])
            template = data["template"]
            metadata = TagMetadata(
                internal_name=template,
                ot_tag=data["pattern"],
                description=data.get("description", ""),
                unit=data.get("unit", ""),
                target_unit=data.get("target_unit", ""),
            )
            self._patterns.append((pattern, template, metadata))
        except Exception as e:
            logger.warning(f"Error parsing pattern: {e}")

    def _load_default_mappings(self) -> None:
        """Load default steam quality tag mappings."""
        default_mappings = [
            # Header measurements
            {
                "ot_tag": "STEAM_HDR_P_001",
                "internal_name": SteamQualityTagSet.HEADER_PRESSURE,
                "description": "Main steam header pressure",
                "unit": "psig",
                "target_unit": "kPa",
                "low_limit": 0.0,
                "high_limit": 3000.0,
                "is_critical": True,
            },
            {
                "ot_tag": "STEAM_HDR_T_001",
                "internal_name": SteamQualityTagSet.HEADER_TEMPERATURE,
                "description": "Main steam header temperature",
                "unit": "degF",
                "target_unit": "degC",
                "low_limit": 100.0,
                "high_limit": 350.0,
                "is_critical": True,
            },
            {
                "ot_tag": "STEAM_HDR_F_001",
                "internal_name": SteamQualityTagSet.HEADER_FLOW,
                "description": "Main steam header flow",
                "unit": "klb/hr",
                "target_unit": "kg/s",
                "low_limit": 0.0,
                "high_limit": 100.0,
                "is_critical": True,
            },
            # Steam quality
            {
                "ot_tag": "STEAM_QUAL_X_001",
                "internal_name": SteamQualityTagSet.QUALITY_DRYNESS,
                "description": "Estimated steam dryness fraction",
                "unit": "fraction",
                "target_unit": "fraction",
                "low_limit": 0.9,
                "high_limit": 1.0,
                "is_critical": True,
            },
            {
                "ot_tag": "STEAM_QUAL_DT_001",
                "internal_name": SteamQualityTagSet.QUALITY_SUPERHEAT,
                "description": "Superheat delta-T",
                "unit": "degC",
                "target_unit": "degC",
                "low_limit": 0.0,
                "high_limit": 100.0,
            },
            {
                "ot_tag": "STEAM_CARRY_R_001",
                "internal_name": SteamQualityTagSet.QUALITY_CARRYOVER,
                "description": "Moisture carryover rate",
                "unit": "fraction",
                "target_unit": "fraction",
                "low_limit": 0.0,
                "high_limit": 0.1,
            },
            # Separator
            {
                "ot_tag": "STEAM_SEP_DP_001",
                "internal_name": SteamQualityTagSet.SEPARATOR_DP,
                "description": "Separator differential pressure",
                "unit": "inH2O",
                "target_unit": "kPa",
                "low_limit": 0.0,
                "high_limit": 25.0,
                "is_critical": True,
            },
            {
                "ot_tag": "STEAM_SEP_EFF_001",
                "internal_name": SteamQualityTagSet.SEPARATOR_EFFICIENCY,
                "description": "Separator efficiency",
                "unit": "%",
                "target_unit": "fraction",
                "low_limit": 0.9,
                "high_limit": 1.0,
            },
            # Drain system
            {
                "ot_tag": "DRAIN_VLV_POS_001",
                "internal_name": SteamQualityTagSet.DRAIN_VALVE_POSITION,
                "description": "Drain valve position",
                "unit": "%",
                "target_unit": "%",
                "low_limit": 0.0,
                "high_limit": 100.0,
                "is_critical": True,
            },
            {
                "ot_tag": "DRAIN_DUTY_001",
                "internal_name": SteamQualityTagSet.DRAIN_DUTY_CYCLE,
                "description": "Drain duty cycle",
                "unit": "%",
                "target_unit": "fraction",
                "low_limit": 0.0,
                "high_limit": 0.5,
            },
            {
                "ot_tag": "DRAIN_FLOW_001",
                "internal_name": SteamQualityTagSet.DRAIN_FLOW,
                "description": "Condensate drain flow",
                "unit": "lb/hr",
                "target_unit": "kg/s",
                "low_limit": 0.0,
                "high_limit": 10.0,
            },
            # Chemistry
            {
                "ot_tag": "CHEM_COND_001",
                "internal_name": SteamQualityTagSet.CHEM_CONDUCTIVITY,
                "description": "Cation conductivity",
                "unit": "uS/cm",
                "target_unit": "uS/cm",
                "low_limit": 0.0,
                "high_limit": 50.0,
            },
            {
                "ot_tag": "CHEM_PH_001",
                "internal_name": SteamQualityTagSet.CHEM_PH,
                "description": "pH value",
                "unit": "pH",
                "target_unit": "pH",
                "low_limit": 7.0,
                "high_limit": 11.0,
            },
            {
                "ot_tag": "CHEM_SILICA_001",
                "internal_name": SteamQualityTagSet.CHEM_SILICA,
                "description": "Silica concentration",
                "unit": "ppb",
                "target_unit": "ppb",
                "low_limit": 0.0,
                "high_limit": 100.0,
            },
            # Constraints (from GL-003)
            {
                "ot_tag": "GL003_X_MIN",
                "internal_name": SteamQualityTagSet.CONSTRAINT_X_MIN,
                "description": "Minimum steam quality constraint",
                "unit": "fraction",
                "target_unit": "fraction",
                "low_limit": 0.95,
                "high_limit": 1.0,
            },
            {
                "ot_tag": "GL003_DT_MIN",
                "internal_name": SteamQualityTagSet.CONSTRAINT_DELTA_T_MIN,
                "description": "Minimum superheat constraint",
                "unit": "degC",
                "target_unit": "degC",
                "low_limit": 0.0,
                "high_limit": 50.0,
            },
            {
                "ot_tag": "GL003_DRAIN_MAX",
                "internal_name": SteamQualityTagSet.CONSTRAINT_MAX_DRAIN_DUTY,
                "description": "Maximum drain duty cycle constraint",
                "unit": "fraction",
                "target_unit": "fraction",
                "low_limit": 0.0,
                "high_limit": 0.5,
            },
        ]

        for mapping_data in default_mappings:
            self._add_mapping_from_dict(mapping_data)

        logger.info(f"Loaded {len(default_mappings)} default mappings")

    def add_mapping(self, mapping: TagMapping) -> None:
        """Add tag mapping."""
        self._ot_to_internal[mapping.ot_tag] = mapping
        self._internal_to_ot[mapping.internal_name] = mapping
        self._stats["mappings_loaded"] += 1

    def remove_mapping(self, ot_tag: str) -> Optional[TagMapping]:
        """Remove tag mapping."""
        if ot_tag not in self._ot_to_internal:
            return None

        mapping = self._ot_to_internal.pop(ot_tag)
        self._internal_to_ot.pop(mapping.internal_name, None)
        return mapping

    def to_internal(self, ot_tag: str) -> Optional[str]:
        """
        Map OT tag to internal name.

        Args:
            ot_tag: OT/SCADA tag name

        Returns:
            Internal tag name or None if not mapped
        """
        # Check direct mapping
        if ot_tag in self._ot_to_internal:
            return self._ot_to_internal[ot_tag].internal_name

        # Check pattern mappings
        for pattern, template, _ in self._patterns:
            match = pattern.match(ot_tag)
            if match:
                return template.format(**match.groupdict())

        return None

    def to_ot(self, internal_name: str) -> Optional[str]:
        """
        Map internal name to OT tag.

        Args:
            internal_name: Internal tag name

        Returns:
            OT tag name or None if not mapped
        """
        if internal_name in self._internal_to_ot:
            return self._internal_to_ot[internal_name].ot_tag
        return None

    def get_mapping(self, tag: str) -> Optional[TagMapping]:
        """Get mapping by either OT tag or internal name."""
        if tag in self._ot_to_internal:
            return self._ot_to_internal[tag]
        if tag in self._internal_to_ot:
            return self._internal_to_ot[tag]
        return None

    def get_metadata(self, tag: str) -> Optional[TagMetadata]:
        """Get metadata for tag."""
        mapping = self.get_mapping(tag)
        return mapping.metadata if mapping else None

    def convert_value(
        self,
        tag: str,
        value: float,
    ) -> Tuple[float, str]:
        """
        Convert value with unit conversion.

        Args:
            tag: OT tag or internal name
            value: Raw value

        Returns:
            Tuple of (converted_value, target_unit)
        """
        self._stats["conversions"] += 1

        mapping = self.get_mapping(tag)
        if not mapping:
            return value, ""

        return mapping.apply(value)

    def validate(
        self,
        tag: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ) -> TagValidationResult:
        """
        Validate tag value.

        Args:
            tag: OT tag or internal name
            value: Value to validate
            timestamp: Value timestamp

        Returns:
            TagValidationResult
        """
        self._stats["validations"] += 1
        ts = timestamp or datetime.now(timezone.utc)
        issues: List[str] = []
        quality = TagQualityFlag.GOOD

        mapping = self.get_mapping(tag)
        if not mapping:
            return TagValidationResult(
                is_valid=True,
                quality_flag=quality,
            )

        metadata = mapping.metadata
        range_valid = True
        rate_valid = True
        staleness_valid = True

        # Range validation
        if self.config.enable_range_validation:
            if metadata.low_limit is not None and value < metadata.low_limit:
                range_valid = False
                quality |= TagQualityFlag.OUT_OF_RANGE
                issues.append(f"Value {value} below limit {metadata.low_limit}")

            if metadata.high_limit is not None and value > metadata.high_limit:
                range_valid = False
                quality |= TagQualityFlag.OUT_OF_RANGE
                issues.append(f"Value {value} above limit {metadata.high_limit}")

        # Rate of change validation
        if self.config.enable_rate_validation and metadata.max_rate_per_second:
            internal_name = mapping.internal_name
            if internal_name in self._value_history:
                last_value, last_ts = self._value_history[internal_name]
                dt = (ts - last_ts).total_seconds()
                if dt > 0:
                    rate = abs(value - last_value) / dt
                    if rate > metadata.max_rate_per_second:
                        rate_valid = False
                        quality |= TagQualityFlag.UNCERTAIN
                        issues.append(
                            f"Rate {rate:.2f}/s exceeds limit {metadata.max_rate_per_second}/s"
                        )

            self._value_history[internal_name] = (value, ts)

        # Staleness check
        if self.config.enable_staleness_check:
            internal_name = mapping.internal_name
            if internal_name in self._value_history:
                _, last_ts = self._value_history[internal_name]
                age = (ts - last_ts).total_seconds()
                if age > metadata.stale_timeout_s:
                    staleness_valid = False
                    quality |= TagQualityFlag.STALE
                    issues.append(f"Value stale: {age:.1f}s > {metadata.stale_timeout_s}s")

        is_valid = quality.is_usable()

        return TagValidationResult(
            is_valid=is_valid,
            quality_flag=quality,
            issues=issues,
            range_valid=range_valid,
            rate_valid=rate_valid,
            staleness_valid=staleness_valid,
        )

    def get_quality_flag(self, ot_quality: int) -> TagQualityFlag:
        """
        Convert OT quality code to internal flag.

        Args:
            ot_quality: OPC-UA or native quality code

        Returns:
            TagQualityFlag
        """
        # OPC-UA quality mapping
        if ot_quality == 0:
            return TagQualityFlag.GOOD
        if ot_quality & 0x80000000:  # Bad quality
            if ot_quality & 0x00130000:  # Comm failure
                return TagQualityFlag.COMM_ERROR
            if ot_quality & 0x00040000:  # Config error
                return TagQualityFlag.CONFIG_ERROR
            return TagQualityFlag.BAD
        if ot_quality & 0x40000000:  # Uncertain
            return TagQualityFlag.UNCERTAIN
        if ot_quality & 0x00D80000:  # Local override
            return TagQualityFlag.MANUAL_OVERRIDE

        return TagQualityFlag.GOOD

    def get_all_mappings(self) -> List[TagMapping]:
        """Get all tag mappings."""
        return list(self._ot_to_internal.values())

    def get_critical_tags(self) -> List[TagMapping]:
        """Get critical tag mappings."""
        return [m for m in self._ot_to_internal.values() if m.metadata.is_critical]

    def export_mappings(self, output_path: str) -> None:
        """Export mappings to JSON file."""
        data = {
            "version": "1.0",
            "mappings": [m.to_dict() for m in self._ot_to_internal.values()],
            "patterns": [
                {
                    "pattern": p.pattern,
                    "template": t,
                    "description": m.description,
                }
                for p, t, m in self._patterns
            ],
        }

        path = Path(output_path)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(data['mappings'])} mappings to {output_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get mapper statistics."""
        return {
            **self._stats,
            "ot_tags": len(self._ot_to_internal),
            "internal_tags": len(self._internal_to_ot),
            "patterns": len(self._patterns),
        }


def create_default_tag_mapping() -> TagMapper:
    """
    Create tag mapper with default steam quality mappings.

    Returns:
        Configured TagMapper instance
    """
    mapper = TagMapper(TagMapperConfig())
    mapper._load_default_mappings()
    return mapper
