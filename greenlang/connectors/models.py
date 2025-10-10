"""
Connector Data Models
=====================

Common data models for connectors, especially time-series data.

Key Design Decisions:
- Decimal for values (precision-critical)
- AwareDatetime for timestamps (UTC enforced)
- ISO 3166-2 for region codes
- Literal types for enums (type safety)
"""

from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone
from decimal import Decimal
from typing import Literal, List, Optional
import re


class TimeWindow(BaseModel):
    """
    Time window specification

    Always UTC, timezone-aware.
    """
    start: datetime = Field(..., description="Start time (UTC, timezone-aware)")
    end: datetime = Field(..., description="End time (UTC, timezone-aware)")
    resolution: Literal["hour", "day", "month"] = Field(
        default="hour",
        description="Time resolution"
    )

    @validator('start', 'end')
    def ensure_utc(cls, v):
        """Ensure timestamps are UTC and timezone-aware"""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware (include tzinfo)")
        # Convert to UTC
        return v.astimezone(timezone.utc)

    @validator('end')
    def end_after_start(cls, v, values):
        """Ensure end is after start"""
        if 'start' in values and v <= values['start']:
            raise ValueError("End time must be after start time")
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TSPoint(BaseModel):
    """
    Time series data point

    Generic point for any time-series data.
    Uses Decimal for exact precision.
    """
    ts: datetime = Field(..., description="Timestamp (UTC boundary)")
    value: Decimal = Field(..., description="Value (exact precision)")
    unit: str = Field(..., description="Unit of measurement")
    quality: Literal["estimated", "measured", "simulated", "forecast"] = Field(
        default="measured",
        description="Data quality indicator"
    )

    @validator('ts')
    def ensure_utc(cls, v):
        """Ensure timestamp is UTC"""
        if v.tzinfo is None:
            raise ValueError("Timestamp must be timezone-aware")
        return v.astimezone(timezone.utc)

    class Config:
        json_encoders = {
            Decimal: lambda v: str(v),
            datetime: lambda v: v.isoformat()
        }


class GridIntensityQuery(BaseModel):
    """
    Query for grid carbon intensity data

    Region codes follow ISO 3166-2 with extensions:
    - CA-ON: Canada - Ontario
    - US-CAISO: US - California ISO
    - IN-NO: India - Northern Grid
    - EU-DE: Europe - Germany
    """
    region: str = Field(
        ...,
        description="Region code (ISO 3166-2 or grid operator code)",
        pattern=r"^[A-Z]{2}(-[A-Z0-9]+)?$"
    )
    window: TimeWindow = Field(..., description="Time window")

    @validator('region')
    def validate_region_format(cls, v):
        """Validate region code format"""
        if not re.match(r"^[A-Z]{2}(-[A-Z0-9]+)?$", v):
            raise ValueError(
                f"Invalid region code format: {v}. "
                f"Expected ISO 3166-2 format (e.g., CA-ON, US-CAISO, IN-NO)"
            )
        return v


class GridIntensityPayload(BaseModel):
    """
    Grid carbon intensity response payload

    Time-series of carbon intensity values.
    """
    series: List[TSPoint] = Field(..., description="Time series data points")
    region: str = Field(..., description="Region code")
    unit: Literal["gCO2/kWh", "kgCO2/MWh", "gCO2e/kWh"] = Field(
        default="gCO2/kWh",
        description="Carbon intensity unit"
    )
    resolution: Literal["hour", "day", "month"] = Field(
        default="hour",
        description="Time resolution"
    )
    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Additional metadata (grid operator, methodology, etc.)"
    )

    @validator('series')
    def series_not_empty(cls, v):
        """Ensure series has data"""
        if not v:
            raise ValueError("Series must contain at least one data point")
        return v

    @validator('series')
    def series_ordered(cls, v):
        """Ensure series is time-ordered"""
        timestamps = [point.ts for point in v]
        if timestamps != sorted(timestamps):
            raise ValueError("Series must be ordered by timestamp (ascending)")
        return v

    class Config:
        json_encoders = {
            Decimal: lambda v: str(v),
            datetime: lambda v: v.isoformat()
        }


# Region metadata for validation and documentation
REGION_METADATA = {
    "CA-ON": {
        "name": "Ontario, Canada",
        "grid_operator": "IESO",
        "timezone": "America/Toronto"
    },
    "US-CAISO": {
        "name": "California ISO, USA",
        "grid_operator": "CAISO",
        "timezone": "America/Los_Angeles"
    },
    "US-PJM": {
        "name": "PJM Interconnection, USA",
        "grid_operator": "PJM",
        "timezone": "America/New_York"
    },
    "EU-DE": {
        "name": "Germany",
        "grid_operator": "Multiple",
        "timezone": "Europe/Berlin"
    },
    "IN-NO": {
        "name": "Northern Grid, India",
        "grid_operator": "NRLDC",
        "timezone": "Asia/Kolkata"
    },
    "UK-GB": {
        "name": "Great Britain",
        "grid_operator": "National Grid ESO",
        "timezone": "Europe/London"
    }
}


def validate_region(region: str) -> bool:
    """
    Validate region code

    Args:
        region: Region code to validate

    Returns:
        True if valid, False otherwise
    """
    # Check format
    if not re.match(r"^[A-Z]{2}(-[A-Z0-9]+)?$", region):
        return False

    # Known regions are valid
    if region in REGION_METADATA:
        return True

    # ISO 3166-1 country codes are valid (2-letter)
    if len(region) == 2:
        return True

    # ISO 3166-2 subdivision codes are valid (XX-YY format)
    if "-" in region and len(region.split("-")[0]) == 2:
        return True

    return False


def get_region_metadata(region: str) -> Optional[dict]:
    """
    Get metadata for a region

    Args:
        region: Region code

    Returns:
        Region metadata if available, None otherwise
    """
    return REGION_METADATA.get(region)
