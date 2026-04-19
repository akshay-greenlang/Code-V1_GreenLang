"""
GL-016 Waterguard OPC-UA Tag Mapping

Version-controlled tag definitions mapping OPC-UA node IDs to logical
tag names with scaling, units, and validation parameters.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Tag Data Types
# =============================================================================

class TagDataType(str, Enum):
    """OPC-UA data types for tags."""
    FLOAT = "Float"
    DOUBLE = "Double"
    INT16 = "Int16"
    INT32 = "Int32"
    INT64 = "Int64"
    UINT16 = "UInt16"
    UINT32 = "UInt32"
    BOOLEAN = "Boolean"
    STRING = "String"
    DATETIME = "DateTime"


class TagCategory(str, Enum):
    """Categories for organizing tags."""
    CHEMISTRY = "chemistry"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW = "flow"
    LEVEL = "level"
    PUMP = "pump"
    VALVE = "valve"
    ANALYZER = "analyzer"
    DOSING = "dosing"
    ALARM = "alarm"
    STATUS = "status"
    SETPOINT = "setpoint"
    OUTPUT = "output"


class TagAccessMode(str, Enum):
    """Tag access modes."""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    WRITE_ONLY = "write_only"


# =============================================================================
# Scaling Configuration
# =============================================================================

class ScalingType(str, Enum):
    """Types of scaling transformations."""
    NONE = "none"
    LINEAR = "linear"
    SQUARE_ROOT = "square_root"
    LOGARITHMIC = "logarithmic"
    LOOKUP = "lookup"


class ScalingConfig(BaseModel):
    """Configuration for value scaling/transformation."""

    scaling_type: ScalingType = Field(
        default=ScalingType.NONE,
        description="Type of scaling"
    )

    # Linear scaling: eu_value = raw_value * slope + offset
    raw_min: float = Field(default=0.0, description="Raw value minimum")
    raw_max: float = Field(default=100.0, description="Raw value maximum")
    eu_min: float = Field(default=0.0, description="Engineering units minimum")
    eu_max: float = Field(default=100.0, description="Engineering units maximum")

    # Lookup table (for discrete mappings)
    lookup_table: Optional[Dict[str, float]] = Field(
        default=None,
        description="Lookup table for value mapping"
    )

    @property
    def slope(self) -> float:
        """Calculate linear scaling slope."""
        raw_range = self.raw_max - self.raw_min
        if raw_range == 0:
            return 1.0
        eu_range = self.eu_max - self.eu_min
        return eu_range / raw_range

    @property
    def offset(self) -> float:
        """Calculate linear scaling offset."""
        return self.eu_min - (self.slope * self.raw_min)

    def apply(self, raw_value: float) -> float:
        """Apply scaling to raw value."""
        if self.scaling_type == ScalingType.NONE:
            return raw_value

        elif self.scaling_type == ScalingType.LINEAR:
            return raw_value * self.slope + self.offset

        elif self.scaling_type == ScalingType.SQUARE_ROOT:
            import math
            # Normalize, take square root, scale
            normalized = (raw_value - self.raw_min) / (self.raw_max - self.raw_min)
            normalized = max(0.0, min(1.0, normalized))
            sqrt_val = math.sqrt(normalized)
            return self.eu_min + sqrt_val * (self.eu_max - self.eu_min)

        elif self.scaling_type == ScalingType.LOGARITHMIC:
            import math
            if raw_value <= 0:
                return self.eu_min
            normalized = (raw_value - self.raw_min) / (self.raw_max - self.raw_min)
            normalized = max(0.001, min(1.0, normalized))
            log_val = math.log10(normalized * 9 + 1) / math.log10(10)
            return self.eu_min + log_val * (self.eu_max - self.eu_min)

        return raw_value

    def reverse(self, eu_value: float) -> float:
        """Reverse scaling to get raw value."""
        if self.scaling_type == ScalingType.NONE:
            return eu_value

        elif self.scaling_type == ScalingType.LINEAR:
            if self.slope == 0:
                return self.raw_min
            return (eu_value - self.offset) / self.slope

        # Square root and logarithmic reverse scaling would be implemented here
        return eu_value


# =============================================================================
# Tag Definition
# =============================================================================

class TagDefinition(BaseModel):
    """
    Complete definition of an OPC-UA tag.

    Provides all metadata needed for reading, writing, and interpreting
    tag values correctly.
    """

    # Identification
    tag_id: str = Field(..., description="Unique tag identifier")
    node_id: str = Field(..., description="OPC-UA node ID")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(default="", description="Tag description")

    # Classification
    category: TagCategory = Field(..., description="Tag category")
    data_type: TagDataType = Field(default=TagDataType.FLOAT)
    access_mode: TagAccessMode = Field(default=TagAccessMode.READ_ONLY)

    # Engineering units
    engineering_units: str = Field(default="", description="Units (e.g., 'ppm')")
    engineering_units_description: str = Field(default="")

    # Scaling
    scaling: ScalingConfig = Field(
        default_factory=ScalingConfig,
        description="Value scaling configuration"
    )

    # Limits
    eu_range_low: Optional[float] = Field(default=None, description="EU range low")
    eu_range_high: Optional[float] = Field(default=None, description="EU range high")
    alarm_low_low: Optional[float] = Field(default=None, description="Low-low alarm")
    alarm_low: Optional[float] = Field(default=None, description="Low alarm")
    alarm_high: Optional[float] = Field(default=None, description="High alarm")
    alarm_high_high: Optional[float] = Field(default=None, description="High-high alarm")

    # Control limits (for setpoints)
    control_min: Optional[float] = Field(default=None, description="Control minimum")
    control_max: Optional[float] = Field(default=None, description="Control maximum")
    rate_limit: Optional[float] = Field(default=None, description="Rate limit per second")
    deadband: Optional[float] = Field(default=None, description="Change deadband")

    # Quality
    stale_timeout_seconds: int = Field(default=60, description="Staleness timeout")
    expected_update_rate_ms: int = Field(default=1000, description="Expected update rate")

    # Metadata
    equipment_id: Optional[str] = Field(default=None, description="Parent equipment ID")
    process_area: Optional[str] = Field(default=None, description="Process area")
    tags: List[str] = Field(default_factory=list, description="Custom tags")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Custom properties")

    # Versioning
    version: str = Field(default="1.0", description="Definition version")
    last_modified: datetime = Field(default_factory=datetime.utcnow)
    modified_by: Optional[str] = Field(default=None)

    @field_validator("node_id")
    @classmethod
    def validate_node_id(cls, v: str) -> str:
        """Validate OPC-UA node ID format."""
        # Basic validation - should contain namespace and identifier
        if "ns=" not in v.lower() and "i=" not in v.lower() and "s=" not in v.lower():
            # Could be a simple string identifier
            pass
        return v

    def is_in_alarm(self, value: float) -> tuple[bool, str]:
        """Check if value triggers any alarm condition."""
        if self.alarm_high_high is not None and value >= self.alarm_high_high:
            return True, "high_high"
        if self.alarm_high is not None and value >= self.alarm_high:
            return True, "high"
        if self.alarm_low_low is not None and value <= self.alarm_low_low:
            return True, "low_low"
        if self.alarm_low is not None and value <= self.alarm_low:
            return True, "low"
        return False, ""

    def is_in_control_range(self, value: float) -> bool:
        """Check if value is within control range."""
        if self.control_min is not None and value < self.control_min:
            return False
        if self.control_max is not None and value > self.control_max:
            return False
        return True

    def scale_value(self, raw_value: float) -> float:
        """Apply scaling to raw value."""
        return self.scaling.apply(raw_value)

    def unscale_value(self, eu_value: float) -> float:
        """Reverse scaling to get raw value."""
        return self.scaling.reverse(eu_value)


# =============================================================================
# Tag Mapping
# =============================================================================

class TagMapping(BaseModel):
    """
    Collection of tag definitions for a system or process area.

    Provides version-controlled tag configuration that can be loaded
    from files or databases.
    """

    # Identification
    mapping_id: UUID = Field(default_factory=uuid4, description="Mapping ID")
    name: str = Field(..., description="Mapping name")
    description: str = Field(default="", description="Mapping description")

    # Tags
    tags: List[TagDefinition] = Field(default_factory=list, description="Tag definitions")

    # Metadata
    version: str = Field(default="1.0", description="Mapping version")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified: datetime = Field(default_factory=datetime.utcnow)
    modified_by: Optional[str] = Field(default=None)

    # Index
    _tag_index: Dict[str, TagDefinition] = {}
    _node_index: Dict[str, TagDefinition] = {}

    def model_post_init(self, __context: Any) -> None:
        """Build indices after initialization."""
        self._build_indices()

    def _build_indices(self) -> None:
        """Build tag lookup indices."""
        self._tag_index = {tag.tag_id: tag for tag in self.tags}
        self._node_index = {tag.node_id: tag for tag in self.tags}

    def get_by_tag_id(self, tag_id: str) -> Optional[TagDefinition]:
        """Get tag definition by tag ID."""
        return self._tag_index.get(tag_id)

    def get_by_node_id(self, node_id: str) -> Optional[TagDefinition]:
        """Get tag definition by OPC-UA node ID."""
        return self._node_index.get(node_id)

    def get_by_category(self, category: TagCategory) -> List[TagDefinition]:
        """Get all tags in a category."""
        return [tag for tag in self.tags if tag.category == category]

    def get_by_equipment(self, equipment_id: str) -> List[TagDefinition]:
        """Get all tags for an equipment."""
        return [tag for tag in self.tags if tag.equipment_id == equipment_id]

    def get_writeable_tags(self) -> List[TagDefinition]:
        """Get all writeable tags."""
        return [
            tag for tag in self.tags
            if tag.access_mode in [TagAccessMode.READ_WRITE, TagAccessMode.WRITE_ONLY]
        ]

    def add_tag(self, tag: TagDefinition) -> None:
        """Add a tag definition."""
        if tag.tag_id in self._tag_index:
            raise ValueError(f"Tag {tag.tag_id} already exists")
        self.tags.append(tag)
        self._tag_index[tag.tag_id] = tag
        self._node_index[tag.node_id] = tag
        self.last_modified = datetime.utcnow()

    def update_tag(self, tag: TagDefinition) -> None:
        """Update an existing tag definition."""
        if tag.tag_id not in self._tag_index:
            raise ValueError(f"Tag {tag.tag_id} not found")

        # Remove old entry
        old_tag = self._tag_index[tag.tag_id]
        self.tags.remove(old_tag)
        del self._node_index[old_tag.node_id]

        # Add new entry
        tag.last_modified = datetime.utcnow()
        self.tags.append(tag)
        self._tag_index[tag.tag_id] = tag
        self._node_index[tag.node_id] = tag
        self.last_modified = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "TagMapping":
        """Create from JSON string."""
        return cls.model_validate_json(json_str)

    def save(self, path: Union[str, Path]) -> None:
        """Save mapping to file."""
        path = Path(path)
        path.write_text(self.to_json())
        logger.info(f"Saved tag mapping to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TagMapping":
        """Load mapping from file."""
        path = Path(path)
        json_str = path.read_text()
        mapping = cls.from_json(json_str)
        logger.info(f"Loaded tag mapping from {path}: {len(mapping.tags)} tags")
        return mapping


# =============================================================================
# Default Tag Mappings for GL-016 Waterguard
# =============================================================================

def create_default_chemistry_tags() -> List[TagDefinition]:
    """Create default chemistry analyzer tags."""
    return [
        TagDefinition(
            tag_id="AI_PO4_001",
            node_id="ns=2;s=BOILER1.CHEM.AI_PO4_001",
            name="Phosphate Analyzer",
            description="Boiler drum phosphate concentration",
            category=TagCategory.CHEMISTRY,
            access_mode=TagAccessMode.READ_ONLY,
            engineering_units="ppm",
            scaling=ScalingConfig(
                scaling_type=ScalingType.LINEAR,
                raw_min=4.0,
                raw_max=20.0,
                eu_min=0.0,
                eu_max=50.0,
            ),
            eu_range_low=0.0,
            eu_range_high=50.0,
            alarm_low_low=1.0,
            alarm_low=2.0,
            alarm_high=12.0,
            alarm_high_high=15.0,
            stale_timeout_seconds=60,
        ),
        TagDefinition(
            tag_id="AI_COND_001",
            node_id="ns=2;s=BOILER1.CHEM.AI_COND_001",
            name="Conductivity Analyzer",
            description="Boiler drum conductivity",
            category=TagCategory.CHEMISTRY,
            access_mode=TagAccessMode.READ_ONLY,
            engineering_units="uS/cm",
            scaling=ScalingConfig(
                scaling_type=ScalingType.LINEAR,
                raw_min=4.0,
                raw_max=20.0,
                eu_min=0.0,
                eu_max=5000.0,
            ),
            eu_range_low=0.0,
            eu_range_high=5000.0,
            alarm_low=100.0,
            alarm_high=3000.0,
            alarm_high_high=3500.0,
            stale_timeout_seconds=60,
        ),
        TagDefinition(
            tag_id="AI_PH_001",
            node_id="ns=2;s=BOILER1.CHEM.AI_PH_001",
            name="pH Analyzer",
            description="Boiler drum pH",
            category=TagCategory.CHEMISTRY,
            access_mode=TagAccessMode.READ_ONLY,
            engineering_units="pH",
            eu_range_low=0.0,
            eu_range_high=14.0,
            alarm_low_low=8.0,
            alarm_low=8.5,
            alarm_high=10.5,
            alarm_high_high=11.0,
            stale_timeout_seconds=60,
        ),
        TagDefinition(
            tag_id="AI_DO_001",
            node_id="ns=2;s=BOILER1.CHEM.AI_DO_001",
            name="Dissolved Oxygen Analyzer",
            description="Feedwater dissolved oxygen",
            category=TagCategory.CHEMISTRY,
            access_mode=TagAccessMode.READ_ONLY,
            engineering_units="ppb",
            eu_range_low=0.0,
            eu_range_high=100.0,
            alarm_high=10.0,
            alarm_high_high=20.0,
            stale_timeout_seconds=60,
        ),
    ]


def create_default_dosing_tags() -> List[TagDefinition]:
    """Create default dosing pump tags."""
    return [
        TagDefinition(
            tag_id="AO_PHOS_PUMP_001",
            node_id="ns=2;s=BOILER1.DOSING.AO_PHOS_PUMP_001",
            name="Phosphate Dosing Pump Setpoint",
            description="Phosphate dosing pump speed setpoint",
            category=TagCategory.DOSING,
            access_mode=TagAccessMode.READ_WRITE,
            engineering_units="%",
            eu_range_low=0.0,
            eu_range_high=100.0,
            control_min=0.0,
            control_max=100.0,
            rate_limit=10.0,  # Max 10% change per second
            deadband=0.5,
        ),
        TagDefinition(
            tag_id="AI_PHOS_PUMP_FB_001",
            node_id="ns=2;s=BOILER1.DOSING.AI_PHOS_PUMP_FB_001",
            name="Phosphate Dosing Pump Feedback",
            description="Phosphate dosing pump actual speed",
            category=TagCategory.DOSING,
            access_mode=TagAccessMode.READ_ONLY,
            engineering_units="%",
        ),
        TagDefinition(
            tag_id="AO_AMINE_PUMP_001",
            node_id="ns=2;s=BOILER1.DOSING.AO_AMINE_PUMP_001",
            name="Amine Dosing Pump Setpoint",
            description="Amine dosing pump speed setpoint",
            category=TagCategory.DOSING,
            access_mode=TagAccessMode.READ_WRITE,
            engineering_units="%",
            eu_range_low=0.0,
            eu_range_high=100.0,
            control_min=0.0,
            control_max=100.0,
            rate_limit=10.0,
        ),
    ]


def create_default_blowdown_tags() -> List[TagDefinition]:
    """Create default blowdown valve tags."""
    return [
        TagDefinition(
            tag_id="AO_CBD_001",
            node_id="ns=2;s=BOILER1.BLOWDOWN.AO_CBD_001",
            name="Continuous Blowdown Valve",
            description="Continuous blowdown valve position setpoint",
            category=TagCategory.VALVE,
            access_mode=TagAccessMode.READ_WRITE,
            engineering_units="%",
            eu_range_low=0.0,
            eu_range_high=100.0,
            control_min=0.0,
            control_max=50.0,  # Safety limit
            rate_limit=5.0,
        ),
        TagDefinition(
            tag_id="AI_CBD_FB_001",
            node_id="ns=2;s=BOILER1.BLOWDOWN.AI_CBD_FB_001",
            name="Continuous Blowdown Valve Feedback",
            description="Continuous blowdown valve actual position",
            category=TagCategory.VALVE,
            access_mode=TagAccessMode.READ_ONLY,
            engineering_units="%",
        ),
    ]


def load_tag_mappings(path: Optional[Union[str, Path]] = None) -> TagMapping:
    """
    Load tag mappings from file or create defaults.

    Args:
        path: Optional path to JSON mapping file

    Returns:
        TagMapping instance
    """
    if path:
        path = Path(path)
        if path.exists():
            return TagMapping.load(path)

    # Create default mapping
    all_tags = (
        create_default_chemistry_tags() +
        create_default_dosing_tags() +
        create_default_blowdown_tags()
    )

    mapping = TagMapping(
        name="GL-016 Waterguard Default Tags",
        description="Default OPC-UA tag mapping for boiler water chemistry monitoring",
        tags=all_tags,
    )

    logger.info(f"Created default tag mapping with {len(mapping.tags)} tags")
    return mapping
