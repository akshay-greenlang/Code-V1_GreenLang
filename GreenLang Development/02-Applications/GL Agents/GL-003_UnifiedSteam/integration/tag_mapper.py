"""
GL-003 UNIFIEDSTEAM - Tag Mapping Service

Maps OT/SCADA tags to canonical GreenLang naming conventions.

Tag Naming Convention:
    <site>.<area>.<asset_type>.<asset_id>.<measurement>.<qualifier>

Example:
    PLANT1.UTIL.BOILER.B001.STEAM_FLOW.PV
    PLANT1.UTIL.HEADER.H001.PRESSURE.PV
    PLANT1.PROD.TRAP.ST042.TEMPERATURE.PV

Components:
- Site: Physical site/plant identifier
- Area: Process area (UTIL=utilities, PROD=production, etc.)
- Asset Type: Equipment type (BOILER, HEADER, TURBINE, TRAP, etc.)
- Asset ID: Unique equipment identifier
- Measurement: Physical quantity measured
- Qualifier: PV (process variable), SP (setpoint), MV (manipulated), etc.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import re
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


class AssetType(Enum):
    """Steam system asset types."""
    BOILER = "BOILER"
    HEADER = "HEADER"
    TURBINE = "TURBINE"
    TRAP = "TRAP"
    VALVE = "VALVE"
    PUMP = "PUMP"
    HEAT_EXCHANGER = "HX"
    DEAERATOR = "DA"
    ECONOMIZER = "ECON"
    SUPERHEATER = "SH"
    BLOWDOWN = "BD"
    CONDENSATE = "COND"
    FEEDWATER = "FW"
    DESUPERHEATER = "DSH"
    PRV = "PRV"  # Pressure reducing valve
    GENERAL = "GEN"


class MeasurementType(Enum):
    """Standard measurement types for steam systems."""
    # Process variables
    PRESSURE = "PRESSURE"
    TEMPERATURE = "TEMPERATURE"
    FLOW = "FLOW"
    LEVEL = "LEVEL"
    POSITION = "POSITION"

    # Calculated/derived
    ENTHALPY = "ENTHALPY"
    ENTROPY = "ENTROPY"
    QUALITY = "QUALITY"  # Steam quality (dryness)
    SUPERHEAT = "SUPERHEAT"
    EFFICIENCY = "EFFICIENCY"

    # Combustion
    O2 = "O2"
    CO = "CO"
    CO2 = "CO2"
    NOX = "NOX"
    EXCESS_AIR = "EXCESS_AIR"

    # Electrical
    POWER = "POWER"
    CURRENT = "CURRENT"
    VOLTAGE = "VOLTAGE"
    FREQUENCY = "FREQUENCY"

    # Acoustics (steam traps)
    ACOUSTIC_LEVEL = "ACOUSTIC_LEVEL"
    ACOUSTIC_FREQ = "ACOUSTIC_FREQ"

    # Status
    STATUS = "STATUS"
    ALARM = "ALARM"
    MODE = "MODE"


class SignalQualifier(Enum):
    """Signal qualifiers."""
    PV = "PV"  # Process variable
    SP = "SP"  # Setpoint
    MV = "MV"  # Manipulated variable (output)
    CV = "CV"  # Controlled variable
    DV = "DV"  # Disturbance variable
    HI = "HI"  # High limit
    LO = "LO"  # Low limit
    HH = "HH"  # High-high alarm
    LL = "LL"  # Low-low alarm
    DEV = "DEV"  # Deviation
    RAW = "RAW"  # Raw/uncalibrated


@dataclass
class ValidationError:
    """Tag mapping validation error."""
    tag: str
    error_type: str
    message: str
    severity: str = "error"  # error, warning, info
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "tag": self.tag,
            "error_type": self.error_type,
            "message": self.message,
            "severity": self.severity,
            "suggestion": self.suggestion,
        }


@dataclass
class SensorMetadata:
    """Metadata for a mapped sensor tag."""
    canonical_tag: str
    raw_tag: str

    # Location
    site: str
    area: str
    asset_type: AssetType
    asset_id: str

    # Measurement
    measurement_type: MeasurementType
    qualifier: SignalQualifier = SignalQualifier.PV

    # Engineering
    engineering_unit: str = ""
    description: str = ""

    # Limits
    low_limit: Optional[float] = None
    high_limit: Optional[float] = None
    low_alarm: Optional[float] = None
    high_alarm: Optional[float] = None

    # Data characteristics
    data_type: str = "float64"
    scan_rate_ms: int = 1000
    deadband: float = 0.0

    # Source system
    source_system: str = ""  # OPC-UA server, historian, etc.
    source_node_id: str = ""

    # Maintenance
    last_calibration: Optional[datetime] = None
    next_calibration: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "canonical_tag": self.canonical_tag,
            "raw_tag": self.raw_tag,
            "site": self.site,
            "area": self.area,
            "asset_type": self.asset_type.value,
            "asset_id": self.asset_id,
            "measurement_type": self.measurement_type.value,
            "qualifier": self.qualifier.value,
            "engineering_unit": self.engineering_unit,
            "description": self.description,
            "low_limit": self.low_limit,
            "high_limit": self.high_limit,
            "source_system": self.source_system,
        }


@dataclass
class TagMapping:
    """Complete tag mapping configuration."""
    mappings: Dict[str, SensorMetadata] = field(default_factory=dict)
    reverse_mappings: Dict[str, str] = field(default_factory=dict)  # raw -> canonical

    # Metadata
    version: str = "1.0"
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    site_id: str = ""
    description: str = ""

    # Statistics
    total_tags: int = 0
    tags_by_asset_type: Dict[str, int] = field(default_factory=dict)
    tags_by_measurement: Dict[str, int] = field(default_factory=dict)

    def add_mapping(self, metadata: SensorMetadata) -> None:
        """Add tag mapping."""
        self.mappings[metadata.canonical_tag] = metadata
        self.reverse_mappings[metadata.raw_tag] = metadata.canonical_tag
        self._update_statistics()

    def remove_mapping(self, canonical_tag: str) -> Optional[SensorMetadata]:
        """Remove tag mapping."""
        if canonical_tag in self.mappings:
            metadata = self.mappings.pop(canonical_tag)
            self.reverse_mappings.pop(metadata.raw_tag, None)
            self._update_statistics()
            return metadata
        return None

    def _update_statistics(self) -> None:
        """Update mapping statistics."""
        self.total_tags = len(self.mappings)

        self.tags_by_asset_type = {}
        self.tags_by_measurement = {}

        for metadata in self.mappings.values():
            asset_type = metadata.asset_type.value
            measurement = metadata.measurement_type.value

            self.tags_by_asset_type[asset_type] = \
                self.tags_by_asset_type.get(asset_type, 0) + 1
            self.tags_by_measurement[measurement] = \
                self.tags_by_measurement.get(measurement, 0) + 1

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "site_id": self.site_id,
            "description": self.description,
            "total_tags": self.total_tags,
            "tags_by_asset_type": self.tags_by_asset_type,
            "tags_by_measurement": self.tags_by_measurement,
            "mappings": {k: v.to_dict() for k, v in self.mappings.items()},
        }


class TagNamingConvention:
    """
    Defines and validates tag naming conventions.

    Pattern: <site>.<area>.<asset_type>.<asset_id>.<measurement>.<qualifier>
    """

    # Valid patterns for each component
    SITE_PATTERN = r"^[A-Z][A-Z0-9_]{1,15}$"
    AREA_PATTERN = r"^[A-Z]{2,10}$"
    ASSET_ID_PATTERN = r"^[A-Z0-9]{2,10}$"

    # Complete canonical tag pattern
    CANONICAL_PATTERN = (
        r"^(?P<site>[A-Z][A-Z0-9_]{1,15})\."
        r"(?P<area>[A-Z]{2,10})\."
        r"(?P<asset_type>[A-Z_]{2,15})\."
        r"(?P<asset_id>[A-Z0-9]{2,10})\."
        r"(?P<measurement>[A-Z0-9_]{2,20})\."
        r"(?P<qualifier>[A-Z]{2,4})$"
    )

    def __init__(self):
        self._canonical_regex = re.compile(self.CANONICAL_PATTERN)

    def build_canonical_tag(
        self,
        site: str,
        area: str,
        asset_type: AssetType,
        asset_id: str,
        measurement: MeasurementType,
        qualifier: SignalQualifier = SignalQualifier.PV,
    ) -> str:
        """
        Build canonical tag from components.

        Args:
            site: Site identifier
            area: Process area
            asset_type: Asset type enum
            asset_id: Equipment identifier
            measurement: Measurement type enum
            qualifier: Signal qualifier enum

        Returns:
            Canonical tag string
        """
        return ".".join([
            site.upper(),
            area.upper(),
            asset_type.value,
            asset_id.upper(),
            measurement.value,
            qualifier.value,
        ])

    def parse_canonical_tag(self, tag: str) -> Optional[Dict[str, str]]:
        """
        Parse canonical tag into components.

        Args:
            tag: Canonical tag string

        Returns:
            Dict of components or None if invalid
        """
        match = self._canonical_regex.match(tag)
        if match:
            return match.groupdict()
        return None

    def validate_tag(self, tag: str) -> List[ValidationError]:
        """
        Validate tag against naming convention.

        Args:
            tag: Tag to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors: List[ValidationError] = []

        # Check overall pattern
        if not self._canonical_regex.match(tag):
            errors.append(ValidationError(
                tag=tag,
                error_type="pattern_mismatch",
                message="Tag does not match canonical pattern",
                suggestion="Use format: SITE.AREA.ASSET_TYPE.ASSET_ID.MEASUREMENT.QUALIFIER",
            ))
            return errors

        # Parse and validate components
        components = self.parse_canonical_tag(tag)
        if not components:
            return errors

        # Validate site
        if not re.match(self.SITE_PATTERN, components["site"]):
            errors.append(ValidationError(
                tag=tag,
                error_type="invalid_site",
                message=f"Invalid site identifier: {components['site']}",
                severity="warning",
            ))

        # Validate asset type is known
        try:
            AssetType(components["asset_type"])
        except ValueError:
            errors.append(ValidationError(
                tag=tag,
                error_type="unknown_asset_type",
                message=f"Unknown asset type: {components['asset_type']}",
                severity="warning",
                suggestion=f"Valid types: {[t.value for t in AssetType]}",
            ))

        # Validate measurement type is known
        try:
            MeasurementType(components["measurement"])
        except ValueError:
            errors.append(ValidationError(
                tag=tag,
                error_type="unknown_measurement",
                message=f"Unknown measurement type: {components['measurement']}",
                severity="warning",
                suggestion=f"Valid types: {[m.value for m in MeasurementType]}",
            ))

        # Validate qualifier
        try:
            SignalQualifier(components["qualifier"])
        except ValueError:
            errors.append(ValidationError(
                tag=tag,
                error_type="unknown_qualifier",
                message=f"Unknown qualifier: {components['qualifier']}",
                severity="warning",
                suggestion=f"Valid qualifiers: {[q.value for q in SignalQualifier]}",
            ))

        return errors


class TagMapper:
    """
    Tag mapping service for GreenLang steam system integration.

    Maps raw OT/SCADA tags to canonical naming convention and provides
    metadata lookup for sensor configuration.

    Example:
        mapper = TagMapper()
        await mapper.load_tag_mapping("config/steam_tags.json")

        # Get canonical name for raw tag
        canonical = mapper.get_canonical_tag("PI_100A")  # -> "PLANT1.UTIL.HEADER.H001.PRESSURE.PV"

        # Get sensor metadata
        metadata = mapper.get_sensor_metadata("PLANT1.UTIL.HEADER.H001.PRESSURE.PV")
        print(metadata.engineering_unit)  # -> "psig"
    """

    def __init__(self) -> None:
        """Initialize tag mapper."""
        self._mapping: TagMapping = TagMapping()
        self._convention = TagNamingConvention()

        # Pattern-based mappings for bulk conversion
        self._pattern_rules: List[Dict] = []

        # Cache
        self._canonical_cache: Dict[str, str] = {}
        self._metadata_cache: Dict[str, SensorMetadata] = {}

        logger.info("TagMapper initialized")

    async def load_tag_mapping(self, config_path: str) -> TagMapping:
        """
        Load tag mapping from configuration file.

        Args:
            config_path: Path to JSON/YAML mapping file

        Returns:
            Loaded TagMapping

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config format invalid
        """
        path = Path(config_path)

        if not path.exists():
            # Create default mapping if file doesn't exist
            logger.warning(f"Tag mapping file not found: {config_path}, using defaults")
            self._mapping = self._create_default_mapping()
            return self._mapping

        try:
            with open(path, "r") as f:
                if path.suffix in [".json"]:
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {path.suffix}")

            self._mapping = self._parse_mapping_config(data)

            # Build caches
            self._build_caches()

            logger.info(f"Loaded tag mapping with {self._mapping.total_tags} tags")
            return self._mapping

        except Exception as e:
            logger.error(f"Error loading tag mapping: {e}")
            raise

    def _parse_mapping_config(self, data: Dict) -> TagMapping:
        """Parse mapping configuration data."""
        mapping = TagMapping(
            version=data.get("version", "1.0"),
            site_id=data.get("site_id", ""),
            description=data.get("description", ""),
            created_at=datetime.now(timezone.utc),
        )

        # Parse mappings
        for tag_data in data.get("mappings", []):
            try:
                metadata = SensorMetadata(
                    canonical_tag=tag_data["canonical_tag"],
                    raw_tag=tag_data["raw_tag"],
                    site=tag_data.get("site", ""),
                    area=tag_data.get("area", ""),
                    asset_type=AssetType(tag_data.get("asset_type", "GEN")),
                    asset_id=tag_data.get("asset_id", ""),
                    measurement_type=MeasurementType(tag_data.get("measurement_type", "STATUS")),
                    qualifier=SignalQualifier(tag_data.get("qualifier", "PV")),
                    engineering_unit=tag_data.get("engineering_unit", ""),
                    description=tag_data.get("description", ""),
                    low_limit=tag_data.get("low_limit"),
                    high_limit=tag_data.get("high_limit"),
                    source_system=tag_data.get("source_system", ""),
                    source_node_id=tag_data.get("source_node_id", ""),
                )
                mapping.add_mapping(metadata)
            except Exception as e:
                logger.warning(f"Error parsing tag mapping: {e}")

        # Parse pattern rules
        for rule in data.get("pattern_rules", []):
            self._pattern_rules.append(rule)

        return mapping

    def _create_default_mapping(self) -> TagMapping:
        """Create default tag mapping for demo/testing."""
        mapping = TagMapping(
            version="1.0",
            site_id="DEMO",
            description="Default steam system tag mapping",
            created_at=datetime.now(timezone.utc),
        )

        # Add common steam system tags
        default_tags = [
            # Steam header
            {
                "raw_tag": "PI_100A",
                "canonical_tag": "DEMO.UTIL.HEADER.H001.PRESSURE.PV",
                "site": "DEMO", "area": "UTIL",
                "asset_type": AssetType.HEADER, "asset_id": "H001",
                "measurement_type": MeasurementType.PRESSURE,
                "engineering_unit": "psig",
                "description": "Main steam header pressure",
                "low_limit": 0.0, "high_limit": 200.0,
            },
            {
                "raw_tag": "TI_100A",
                "canonical_tag": "DEMO.UTIL.HEADER.H001.TEMPERATURE.PV",
                "site": "DEMO", "area": "UTIL",
                "asset_type": AssetType.HEADER, "asset_id": "H001",
                "measurement_type": MeasurementType.TEMPERATURE,
                "engineering_unit": "degF",
                "description": "Main steam header temperature",
                "low_limit": 200.0, "high_limit": 600.0,
            },
            {
                "raw_tag": "FI_100A",
                "canonical_tag": "DEMO.UTIL.HEADER.H001.FLOW.PV",
                "site": "DEMO", "area": "UTIL",
                "asset_type": AssetType.HEADER, "asset_id": "H001",
                "measurement_type": MeasurementType.FLOW,
                "engineering_unit": "klb/hr",
                "description": "Main steam header flow",
                "low_limit": 0.0, "high_limit": 500.0,
            },

            # Boiler 1
            {
                "raw_tag": "FI_B1_STEAM",
                "canonical_tag": "DEMO.UTIL.BOILER.B001.FLOW.PV",
                "site": "DEMO", "area": "UTIL",
                "asset_type": AssetType.BOILER, "asset_id": "B001",
                "measurement_type": MeasurementType.FLOW,
                "engineering_unit": "klb/hr",
                "description": "Boiler 1 steam flow",
            },
            {
                "raw_tag": "AI_B1_O2",
                "canonical_tag": "DEMO.UTIL.BOILER.B001.O2.PV",
                "site": "DEMO", "area": "UTIL",
                "asset_type": AssetType.BOILER, "asset_id": "B001",
                "measurement_type": MeasurementType.O2,
                "engineering_unit": "%",
                "description": "Boiler 1 flue gas O2",
                "low_limit": 0.0, "high_limit": 21.0,
            },

            # Steam traps
            {
                "raw_tag": "TI_ST001",
                "canonical_tag": "DEMO.PROD.TRAP.ST001.TEMPERATURE.PV",
                "site": "DEMO", "area": "PROD",
                "asset_type": AssetType.TRAP, "asset_id": "ST001",
                "measurement_type": MeasurementType.TEMPERATURE,
                "engineering_unit": "degF",
                "description": "Steam trap ST001 temperature",
            },
            {
                "raw_tag": "ACOUSTIC_ST001",
                "canonical_tag": "DEMO.PROD.TRAP.ST001.ACOUSTIC_LEVEL.PV",
                "site": "DEMO", "area": "PROD",
                "asset_type": AssetType.TRAP, "asset_id": "ST001",
                "measurement_type": MeasurementType.ACOUSTIC_LEVEL,
                "engineering_unit": "dB",
                "description": "Steam trap ST001 acoustic level",
            },
        ]

        for tag_config in default_tags:
            metadata = SensorMetadata(
                canonical_tag=tag_config["canonical_tag"],
                raw_tag=tag_config["raw_tag"],
                site=tag_config["site"],
                area=tag_config["area"],
                asset_type=tag_config["asset_type"],
                asset_id=tag_config["asset_id"],
                measurement_type=tag_config["measurement_type"],
                qualifier=tag_config.get("qualifier", SignalQualifier.PV),
                engineering_unit=tag_config.get("engineering_unit", ""),
                description=tag_config.get("description", ""),
                low_limit=tag_config.get("low_limit"),
                high_limit=tag_config.get("high_limit"),
            )
            mapping.add_mapping(metadata)

        return mapping

    def _build_caches(self) -> None:
        """Build lookup caches for performance."""
        self._canonical_cache.clear()
        self._metadata_cache.clear()

        for canonical, metadata in self._mapping.mappings.items():
            self._canonical_cache[metadata.raw_tag] = canonical
            self._metadata_cache[canonical] = metadata

    def get_canonical_tag(self, raw_tag: str) -> Optional[str]:
        """
        Get canonical tag name for raw SCADA tag.

        Args:
            raw_tag: Raw tag from OPC-UA/SCADA

        Returns:
            Canonical tag name or None if not mapped
        """
        # Check direct mapping
        if raw_tag in self._mapping.reverse_mappings:
            return self._mapping.reverse_mappings[raw_tag]

        # Check cache
        if raw_tag in self._canonical_cache:
            return self._canonical_cache[raw_tag]

        # Try pattern-based mapping
        for rule in self._pattern_rules:
            pattern = rule.get("pattern", "")
            if re.match(pattern, raw_tag):
                # Apply transformation
                canonical = self._apply_pattern_rule(raw_tag, rule)
                if canonical:
                    self._canonical_cache[raw_tag] = canonical
                    return canonical

        logger.warning(f"No mapping found for tag: {raw_tag}")
        return None

    def _apply_pattern_rule(self, raw_tag: str, rule: Dict) -> Optional[str]:
        """Apply pattern-based mapping rule."""
        pattern = rule.get("pattern", "")
        template = rule.get("template", "")

        match = re.match(pattern, raw_tag)
        if not match:
            return None

        # Substitute captured groups into template
        canonical = template
        for i, group in enumerate(match.groups(), 1):
            canonical = canonical.replace(f"${i}", group)

        # Also support named groups
        for name, value in match.groupdict().items():
            if value:
                canonical = canonical.replace(f"${{{name}}}", value)

        return canonical

    def get_sensor_metadata(self, tag_name: str) -> Optional[SensorMetadata]:
        """
        Get sensor metadata by tag name.

        Args:
            tag_name: Canonical or raw tag name

        Returns:
            SensorMetadata or None if not found
        """
        # Check if canonical
        if tag_name in self._mapping.mappings:
            return self._mapping.mappings[tag_name]

        # Check if raw tag
        canonical = self.get_canonical_tag(tag_name)
        if canonical:
            return self._mapping.mappings.get(canonical)

        return None

    def validate_mapping(self, mapping: Optional[TagMapping] = None) -> List[ValidationError]:
        """
        Validate tag mapping for errors and warnings.

        Args:
            mapping: Mapping to validate (uses loaded mapping if None)

        Returns:
            List of validation errors
        """
        errors: List[ValidationError] = []
        target_mapping = mapping or self._mapping

        # Check for required fields
        if not target_mapping.site_id:
            errors.append(ValidationError(
                tag="(global)",
                error_type="missing_site_id",
                message="Mapping missing site_id",
                severity="warning",
            ))

        # Validate each tag
        for canonical, metadata in target_mapping.mappings.items():
            # Validate canonical tag format
            tag_errors = self._convention.validate_tag(canonical)
            errors.extend(tag_errors)

            # Check for duplicate raw tags
            raw_count = sum(
                1 for m in target_mapping.mappings.values()
                if m.raw_tag == metadata.raw_tag
            )
            if raw_count > 1:
                errors.append(ValidationError(
                    tag=canonical,
                    error_type="duplicate_raw_tag",
                    message=f"Raw tag '{metadata.raw_tag}' mapped to multiple canonical tags",
                    severity="error",
                ))

            # Check for missing engineering unit
            if not metadata.engineering_unit:
                errors.append(ValidationError(
                    tag=canonical,
                    error_type="missing_unit",
                    message="No engineering unit specified",
                    severity="warning",
                ))

            # Check limits consistency
            if metadata.low_limit is not None and metadata.high_limit is not None:
                if metadata.low_limit >= metadata.high_limit:
                    errors.append(ValidationError(
                        tag=canonical,
                        error_type="invalid_limits",
                        message=f"Low limit ({metadata.low_limit}) >= high limit ({metadata.high_limit})",
                        severity="error",
                    ))

        return errors

    def add_mapping(
        self,
        raw_tag: str,
        canonical_tag: str,
        metadata: Optional[Dict] = None,
    ) -> SensorMetadata:
        """
        Add tag mapping dynamically.

        Args:
            raw_tag: Raw SCADA tag
            canonical_tag: Canonical tag name
            metadata: Optional additional metadata

        Returns:
            Created SensorMetadata
        """
        # Parse canonical tag for components
        components = self._convention.parse_canonical_tag(canonical_tag)

        if not components:
            raise ValueError(f"Invalid canonical tag format: {canonical_tag}")

        sensor_metadata = SensorMetadata(
            canonical_tag=canonical_tag,
            raw_tag=raw_tag,
            site=components["site"],
            area=components["area"],
            asset_type=AssetType(components["asset_type"]) if components["asset_type"] in [t.value for t in AssetType] else AssetType.GENERAL,
            asset_id=components["asset_id"],
            measurement_type=MeasurementType(components["measurement"]) if components["measurement"] in [m.value for m in MeasurementType] else MeasurementType.STATUS,
            qualifier=SignalQualifier(components["qualifier"]) if components["qualifier"] in [q.value for q in SignalQualifier] else SignalQualifier.PV,
            **(metadata or {}),
        )

        self._mapping.add_mapping(sensor_metadata)
        self._canonical_cache[raw_tag] = canonical_tag
        self._metadata_cache[canonical_tag] = sensor_metadata

        logger.debug(f"Added mapping: {raw_tag} -> {canonical_tag}")
        return sensor_metadata

    def get_tags_by_asset(self, asset_type: AssetType) -> List[SensorMetadata]:
        """Get all tags for an asset type."""
        return [
            m for m in self._mapping.mappings.values()
            if m.asset_type == asset_type
        ]

    def get_tags_by_measurement(self, measurement_type: MeasurementType) -> List[SensorMetadata]:
        """Get all tags for a measurement type."""
        return [
            m for m in self._mapping.mappings.values()
            if m.measurement_type == measurement_type
        ]

    def get_tags_by_site_area(self, site: str, area: str) -> List[SensorMetadata]:
        """Get all tags for a site and area."""
        return [
            m for m in self._mapping.mappings.values()
            if m.site == site and m.area == area
        ]

    def export_mapping(self, output_path: str) -> None:
        """Export tag mapping to JSON file."""
        path = Path(output_path)

        data = {
            "version": self._mapping.version,
            "site_id": self._mapping.site_id,
            "description": self._mapping.description,
            "created_at": self._mapping.created_at.isoformat() if self._mapping.created_at else None,
            "mappings": [m.to_dict() for m in self._mapping.mappings.values()],
            "pattern_rules": self._pattern_rules,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported tag mapping to {output_path}")

    def get_statistics(self) -> Dict:
        """Get mapping statistics."""
        return {
            "total_tags": self._mapping.total_tags,
            "tags_by_asset_type": self._mapping.tags_by_asset_type,
            "tags_by_measurement": self._mapping.tags_by_measurement,
            "pattern_rules": len(self._pattern_rules),
            "cache_size": len(self._canonical_cache),
        }
