"""
PACK-049 GHG Multi-Site Management Pack - Site Registry Engine
====================================================================

Manages the authoritative registry of all facilities, plants, offices,
warehouses, and other sites within an organisation's GHG reporting
boundary. Provides lifecycle management (register, update, decommission),
site classification and grouping, portfolio-level summarisation, and
geographic/facility-type analytics.

Regulatory Basis:
    - GHG Protocol Corporate Standard (Chapter 3): Setting Organizational
      Boundaries - requires a complete list of all facilities/operations.
    - GHG Protocol Scope 3 Standard (Chapter 3): Value chain boundary
      determination requires an accurate site inventory.
    - ISO 14064-1:2018 (Clause 5.1): Organisation shall define its
      organisational boundaries and list all facilities.
    - ESRS E1-6: Gross Scopes 1, 2 and 3 require per-site disaggregation.

Capabilities:
    - Register new sites with full facility characteristics
    - Update site metadata and characteristics over time
    - Decommission sites with audit trail
    - Classify sites against a configurable taxonomy
    - Create and manage site groups (by region, business unit, etc.)
    - Generate portfolio-level summaries with aggregated metrics
    - Filter sites by arbitrary criteria
    - Validate site data completeness

Zero-Hallucination:
    - All aggregations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation or classification path
    - SHA-256 provenance hash on every result object

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-049 GHG Multi-Site Management
Engine:  1 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash, excluding volatile fields."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("created_at", "updated_at", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert any value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Divide safely, returning *default* when denominator is zero."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round2(value: Any) -> Decimal:
    """Round a value to two decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FacilityType(str, Enum):
    """Standard facility type taxonomy for GHG reporting."""
    MANUFACTURING = "MANUFACTURING"
    OFFICE = "OFFICE"
    WAREHOUSE = "WAREHOUSE"
    DATA_CENTER = "DATA_CENTER"
    RETAIL = "RETAIL"
    LABORATORY = "LABORATORY"
    HOSPITAL = "HOSPITAL"
    DISTRIBUTION_CENTER = "DISTRIBUTION_CENTER"
    REFINERY = "REFINERY"
    POWER_PLANT = "POWER_PLANT"
    MINE = "MINE"
    AGRICULTURAL = "AGRICULTURAL"
    MIXED_USE = "MIXED_USE"
    OTHER = "OTHER"


class LifecycleStatus(str, Enum):
    """Site lifecycle stages."""
    PLANNED = "PLANNED"
    UNDER_CONSTRUCTION = "UNDER_CONSTRUCTION"
    COMMISSIONING = "COMMISSIONING"
    OPERATIONAL = "OPERATIONAL"
    MOTHBALLED = "MOTHBALLED"
    DECOMMISSIONING = "DECOMMISSIONING"
    DECOMMISSIONED = "DECOMMISSIONED"
    DIVESTED = "DIVESTED"


class GroupType(str, Enum):
    """Standard site grouping types."""
    REGION = "REGION"
    BUSINESS_UNIT = "BUSINESS_UNIT"
    COUNTRY = "COUNTRY"
    FACILITY_TYPE = "FACILITY_TYPE"
    REPORTING_SEGMENT = "REPORTING_SEGMENT"
    LEGAL_ENTITY = "LEGAL_ENTITY"
    CUSTOM = "CUSTOM"


class CompletenessLevel(str, Enum):
    """Site data completeness levels."""
    COMPLETE = "COMPLETE"
    MOSTLY_COMPLETE = "MOSTLY_COMPLETE"
    PARTIAL = "PARTIAL"
    MINIMAL = "MINIMAL"
    INSUFFICIENT = "INSUFFICIENT"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class FacilityCharacteristics(BaseModel):
    """Physical and operational characteristics of a facility.

    These characteristics are used for intensity metrics, benchmarking,
    and as proxies when activity data is unavailable for estimation.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    floor_area_m2: Decimal = Field(
        ...,
        ge=0,
        description="Total conditioned floor area in square metres.",
    )
    headcount: int = Field(
        ...,
        ge=0,
        description="Full-time equivalent head count at the site.",
    )
    operating_hours_per_year: int = Field(
        ...,
        ge=0,
        le=8784,
        description="Annual operating hours (max 8784 for leap year).",
    )
    production_output: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Annual production output in the site's primary unit.",
    )
    production_unit: Optional[str] = Field(
        None,
        description="Unit of production (e.g. 'tonnes', 'units', 'MWh').",
    )
    grid_region: Optional[str] = Field(
        None,
        description="Electricity grid region identifier.",
    )
    climate_zone: Optional[str] = Field(
        None,
        description="Koppen-Geiger climate classification code.",
    )
    electricity_provider: Optional[str] = Field(
        None,
        description="Name of the electricity utility provider.",
    )
    gas_provider: Optional[str] = Field(
        None,
        description="Name of the natural gas provider.",
    )

    @field_validator("floor_area_m2", "production_output", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        if v is not None:
            return Decimal(str(v))
        return v


class SiteRecord(BaseModel):
    """Represents a single site in the GHG site registry.

    A site is any physical location (facility, plant, office, warehouse)
    that is part of the organisation's operational or financial boundary.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    site_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for the site.",
    )
    site_code: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="Short code for the site (e.g. 'US-CHI-MFG-01').",
    )
    site_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name for the site.",
    )
    facility_type: str = Field(
        ...,
        description="Type of facility (maps to FacilityType enum).",
    )
    legal_entity_id: str = Field(
        ...,
        description="ID of the owning legal entity.",
    )
    business_unit: Optional[str] = Field(
        None,
        description="Business unit or division.",
    )
    country: str = Field(
        ...,
        min_length=2,
        max_length=3,
        description="ISO 3166-1 alpha-2 or alpha-3 country code.",
    )
    region: Optional[str] = Field(
        None,
        description="Sub-national region or state.",
    )
    city: Optional[str] = Field(
        None,
        description="City or municipality.",
    )
    postal_code: Optional[str] = Field(
        None,
        description="Postal / ZIP code.",
    )
    latitude: Optional[Decimal] = Field(
        None,
        ge=Decimal("-90"),
        le=Decimal("90"),
        description="Latitude in decimal degrees.",
    )
    longitude: Optional[Decimal] = Field(
        None,
        ge=Decimal("-180"),
        le=Decimal("180"),
        description="Longitude in decimal degrees.",
    )
    characteristics: Optional[FacilityCharacteristics] = Field(
        None,
        description="Physical and operational characteristics.",
    )
    lifecycle_status: str = Field(
        default="OPERATIONAL",
        description="Current lifecycle status of the site.",
    )
    acquisition_date: Optional[date] = Field(
        None,
        description="Date site was acquired (for M&A tracking).",
    )
    commissioning_date: Optional[date] = Field(
        None,
        description="Date site was commissioned / went operational.",
    )
    decommissioning_date: Optional[date] = Field(
        None,
        description="Date site was or will be decommissioned.",
    )
    is_active: bool = Field(
        default=True,
        description="Whether the site is currently active for reporting.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Freeform tags for filtering and grouping.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the site record was created.",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the site record was last updated.",
    )

    @field_validator("latitude", "longitude", mode="before")
    @classmethod
    def _coerce_decimal_coords(cls, v: Any) -> Any:
        if v is not None:
            return Decimal(str(v))
        return v

    @field_validator("facility_type")
    @classmethod
    def _validate_facility_type(cls, v: str) -> str:
        """Validate facility_type is a recognised value."""
        valid = {ft.value for ft in FacilityType}
        if v.upper() not in valid:
            logger.warning(
                "Facility type '%s' not in standard taxonomy; accepted as CUSTOM.",
                v,
            )
        return v.upper()

    @field_validator("lifecycle_status")
    @classmethod
    def _validate_lifecycle_status(cls, v: str) -> str:
        """Validate lifecycle_status is a recognised value."""
        valid = {ls.value for ls in LifecycleStatus}
        if v.upper() not in valid:
            raise ValueError(
                f"Invalid lifecycle_status '{v}'. Must be one of {sorted(valid)}."
            )
        return v.upper()


class SiteGroup(BaseModel):
    """A logical grouping of sites for aggregated reporting.

    Site groups enable roll-up reporting by region, business unit,
    or any custom grouping. A site can belong to multiple groups.
    """
    model_config = ConfigDict(validate_default=True)

    group_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for the group.",
    )
    group_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Name of the site group.",
    )
    group_type: str = Field(
        ...,
        description="Type of grouping (maps to GroupType enum).",
    )
    member_site_ids: List[str] = Field(
        default_factory=list,
        description="List of site IDs that belong to this group.",
    )
    description: Optional[str] = Field(
        None,
        description="Optional description of the group purpose.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of group creation.",
    )

    @field_validator("group_type")
    @classmethod
    def _validate_group_type(cls, v: str) -> str:
        valid = {gt.value for gt in GroupType}
        if v.upper() not in valid:
            logger.warning("Group type '%s' not in standard types; accepted as CUSTOM.", v)
            return "CUSTOM"
        return v.upper()


class SiteRegistryResult(BaseModel):
    """Portfolio-level summary of the entire site registry.

    Aggregates key metrics across all active sites for dashboard
    display and executive reporting.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    registry_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this registry snapshot.",
    )
    sites: List[SiteRecord] = Field(
        default_factory=list,
        description="Complete list of site records.",
    )
    groups: List[SiteGroup] = Field(
        default_factory=list,
        description="Defined site groups.",
    )
    total_active_sites: int = Field(
        default=0,
        description="Count of active sites.",
    )
    total_floor_area: Decimal = Field(
        default=Decimal("0"),
        description="Sum of floor_area_m2 across all active sites.",
    )
    total_headcount: int = Field(
        default=0,
        description="Sum of headcount across all active sites.",
    )
    countries_covered: int = Field(
        default=0,
        description="Number of distinct countries.",
    )
    facility_type_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of active sites by facility type.",
    )
    lifecycle_status_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of sites by lifecycle status.",
    )
    geographic_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of active sites by country.",
    )
    total_production_output: Decimal = Field(
        default=Decimal("0"),
        description="Sum of production output across all active sites.",
    )
    avg_floor_area: Decimal = Field(
        default=Decimal("0"),
        description="Average floor area per active site.",
    )
    avg_headcount: Decimal = Field(
        default=Decimal("0"),
        description="Average headcount per active site.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of this summary.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash of the registry snapshot.",
    )


class SiteClassification(BaseModel):
    """Result of classifying a site against a taxonomy."""
    model_config = ConfigDict(validate_default=True)

    site_id: str = Field(..., description="The classified site's ID.")
    primary_category: str = Field(
        ..., description="Primary category from taxonomy."
    )
    secondary_categories: List[str] = Field(
        default_factory=list,
        description="Secondary taxonomy categories.",
    )
    size_class: str = Field(
        ..., description="Size classification (SMALL, MEDIUM, LARGE, MEGA).",
    )
    emission_intensity_class: Optional[str] = Field(
        None,
        description="Expected emission intensity class (LOW, MEDIUM, HIGH).",
    )
    reporting_priority: str = Field(
        default="STANDARD",
        description="Reporting priority (CRITICAL, HIGH, STANDARD, LOW).",
    )
    classification_rationale: Dict[str, str] = Field(
        default_factory=dict,
        description="Rationale for each classification decision.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of classification.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash.",
    )


class SiteCompletenessResult(BaseModel):
    """Completeness assessment for a single site record."""
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    site_id: str = Field(..., description="The assessed site.")
    completeness_level: str = Field(
        ..., description="Overall completeness level."
    )
    completeness_pct: Decimal = Field(
        ..., description="Percentage of required fields populated."
    )
    required_fields_total: int = Field(
        default=0, description="Total required fields."
    )
    required_fields_present: int = Field(
        default=0, description="Required fields that are populated."
    )
    optional_fields_total: int = Field(
        default=0, description="Total optional fields."
    )
    optional_fields_present: int = Field(
        default=0, description="Optional fields that are populated."
    )
    missing_required: List[str] = Field(
        default_factory=list, description="Names of missing required fields."
    )
    missing_optional: List[str] = Field(
        default_factory=list, description="Names of missing optional fields."
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations to improve completeness."
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash."
    )


# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------


# Default size-class thresholds based on floor area (m2)
SIZE_CLASS_THRESHOLDS: Dict[str, Tuple[Decimal, Decimal]] = {
    "SMALL": (Decimal("0"), Decimal("1000")),
    "MEDIUM": (Decimal("1000"), Decimal("10000")),
    "LARGE": (Decimal("10000"), Decimal("100000")),
    "MEGA": (Decimal("100000"), Decimal("999999999")),
}

# Facility types that are typically high emission intensity
HIGH_INTENSITY_FACILITY_TYPES: set = {
    FacilityType.MANUFACTURING.value,
    FacilityType.REFINERY.value,
    FacilityType.POWER_PLANT.value,
    FacilityType.MINE.value,
    FacilityType.DATA_CENTER.value,
}

# Facility types that are typically medium emission intensity
MEDIUM_INTENSITY_FACILITY_TYPES: set = {
    FacilityType.WAREHOUSE.value,
    FacilityType.DISTRIBUTION_CENTER.value,
    FacilityType.HOSPITAL.value,
    FacilityType.LABORATORY.value,
    FacilityType.AGRICULTURAL.value,
    FacilityType.MIXED_USE.value,
}

# Required fields for completeness assessment
REQUIRED_SITE_FIELDS: List[str] = [
    "site_code",
    "site_name",
    "facility_type",
    "legal_entity_id",
    "country",
]

# Important optional fields
OPTIONAL_SITE_FIELDS: List[str] = [
    "business_unit",
    "region",
    "city",
    "postal_code",
    "latitude",
    "longitude",
    "characteristics",
    "commissioning_date",
]

# Characteristics sub-fields used in completeness
REQUIRED_CHARACTERISTICS_FIELDS: List[str] = [
    "floor_area_m2",
    "headcount",
    "operating_hours_per_year",
]

OPTIONAL_CHARACTERISTICS_FIELDS: List[str] = [
    "production_output",
    "production_unit",
    "grid_region",
    "climate_zone",
    "electricity_provider",
    "gas_provider",
]

# Reporting priority thresholds based on floor area (m2)
PRIORITY_THRESHOLDS: Dict[str, Decimal] = {
    "CRITICAL": Decimal("50000"),
    "HIGH": Decimal("10000"),
    "STANDARD": Decimal("1000"),
    # Below 1000 -> LOW
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SiteRegistryEngine:
    """Manages the authoritative GHG site registry.

    Provides CRUD operations for site records, site grouping,
    portfolio summarisation, classification, and completeness
    validation. All numeric aggregations use Decimal arithmetic
    with SHA-256 provenance hashing on every result.

    Attributes:
        _sites: Internal dict mapping site_id to SiteRecord.
        _groups: Internal dict mapping group_id to SiteGroup.
        _change_log: Append-only list of change events.

    Example:
        >>> engine = SiteRegistryEngine()
        >>> site = engine.register_site({
        ...     "site_code": "US-CHI-MFG-01",
        ...     "site_name": "Chicago Manufacturing",
        ...     "facility_type": "MANUFACTURING",
        ...     "legal_entity_id": "LE-001",
        ...     "country": "US",
        ...     "characteristics": {
        ...         "floor_area_m2": "25000",
        ...         "headcount": 350,
        ...         "operating_hours_per_year": 6000,
        ...     },
        ... })
        >>> assert site.is_active is True
    """

    def __init__(self) -> None:
        """Initialise the SiteRegistryEngine with empty state."""
        self._sites: Dict[str, SiteRecord] = {}
        self._groups: Dict[str, SiteGroup] = {}
        self._change_log: List[Dict[str, Any]] = []
        logger.info("SiteRegistryEngine v%s initialised.", _MODULE_VERSION)

    # ------------------------------------------------------------------
    # Site CRUD
    # ------------------------------------------------------------------

    def register_site(self, site_data: Dict[str, Any]) -> SiteRecord:
        """Register a new site in the registry.

        Validates the incoming data, assigns a unique site_id if not
        provided, and stores the record. Logs the creation event.

        Args:
            site_data: Dictionary of site attributes. Must include at
                minimum: site_code, site_name, facility_type,
                legal_entity_id, country.

        Returns:
            The created SiteRecord with generated site_id.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        logger.info("Registering new site with code '%s'.", site_data.get("site_code", "N/A"))
        start = _utcnow()

        # Ensure site_id
        if "site_id" not in site_data or not site_data["site_id"]:
            site_data["site_id"] = _new_uuid()

        # Check for duplicate site_code
        for existing in self._sites.values():
            if existing.site_code == site_data.get("site_code", ""):
                raise ValueError(
                    f"Site code '{site_data['site_code']}' already exists "
                    f"(site_id={existing.site_id})."
                )

        # Parse characteristics if dict
        if "characteristics" in site_data and isinstance(
            site_data["characteristics"], dict
        ):
            site_data["characteristics"] = FacilityCharacteristics(
                **site_data["characteristics"]
            )

        # Set timestamps
        now = _utcnow()
        site_data["created_at"] = now
        site_data["updated_at"] = now

        site = SiteRecord(**site_data)
        self._sites[site.site_id] = site

        self._change_log.append({
            "event": "SITE_REGISTERED",
            "site_id": site.site_id,
            "site_code": site.site_code,
            "timestamp": now.isoformat(),
        })

        logger.info(
            "Site '%s' registered successfully (id=%s) in %s.",
            site.site_name,
            site.site_id,
            _utcnow() - start,
        )
        return site

    def update_site(self, site_id: str, updates: Dict[str, Any]) -> SiteRecord:
        """Update an existing site record.

        Applies partial updates to the site's fields. Immutable fields
        (site_id, created_at) are silently ignored. Characteristics can
        be partially updated.

        Args:
            site_id: The ID of the site to update.
            updates: Dictionary of fields to update.

        Returns:
            The updated SiteRecord.

        Raises:
            KeyError: If the site_id does not exist.
            ValueError: If the update values are invalid.
        """
        if site_id not in self._sites:
            raise KeyError(f"Site '{site_id}' not found in registry.")

        site = self._sites[site_id]
        logger.info("Updating site '%s' (id=%s).", site.site_name, site_id)

        # Fields that cannot be changed via update
        immutable_fields = {"site_id", "created_at"}

        # Build the update dict from current values
        current_data = site.model_dump()
        changes_applied: Dict[str, Any] = {}

        for key, value in updates.items():
            if key in immutable_fields:
                logger.warning("Ignoring immutable field '%s' in update.", key)
                continue

            # Handle nested characteristics update
            if key == "characteristics" and isinstance(value, dict):
                if site.characteristics is not None:
                    char_data = site.characteristics.model_dump()
                    char_data.update(value)
                    current_data["characteristics"] = FacilityCharacteristics(**char_data)
                else:
                    current_data["characteristics"] = FacilityCharacteristics(**value)
                changes_applied[key] = value
            else:
                if key in current_data:
                    old_value = current_data[key]
                    current_data[key] = value
                    changes_applied[key] = {"old": old_value, "new": value}
                else:
                    logger.warning("Unknown field '%s' ignored in update.", key)

        current_data["updated_at"] = _utcnow()

        # Handle characteristics that may be dict or model
        if isinstance(current_data.get("characteristics"), dict):
            current_data["characteristics"] = FacilityCharacteristics(
                **current_data["characteristics"]
            )

        updated_site = SiteRecord(**current_data)
        self._sites[site_id] = updated_site

        self._change_log.append({
            "event": "SITE_UPDATED",
            "site_id": site_id,
            "changes": changes_applied,
            "timestamp": _utcnow().isoformat(),
        })

        logger.info("Site '%s' updated with %d field(s).", site_id, len(changes_applied))
        return updated_site

    def decommission_site(
        self,
        site_id: str,
        decommission_date: date,
        reason: str,
    ) -> SiteRecord:
        """Decommission a site, marking it inactive for future reporting.

        Sets the lifecycle status to DECOMMISSIONED, records the
        decommission date, and marks the site inactive. The site record
        is preserved for historical reporting.

        Args:
            site_id: The site to decommission.
            decommission_date: Effective date of decommissioning.
            reason: Reason for decommissioning (audit trail).

        Returns:
            The updated SiteRecord with decommissioned status.

        Raises:
            KeyError: If site not found.
            ValueError: If site is already decommissioned.
        """
        if site_id not in self._sites:
            raise KeyError(f"Site '{site_id}' not found in registry.")

        site = self._sites[site_id]

        if site.lifecycle_status == LifecycleStatus.DECOMMISSIONED.value:
            raise ValueError(
                f"Site '{site_id}' is already decommissioned "
                f"(date={site.decommissioning_date})."
            )

        logger.info(
            "Decommissioning site '%s' (id=%s), reason: %s.",
            site.site_name,
            site_id,
            reason,
        )

        updated = self.update_site(site_id, {
            "lifecycle_status": LifecycleStatus.DECOMMISSIONED.value,
            "decommissioning_date": decommission_date,
            "is_active": False,
        })

        self._change_log.append({
            "event": "SITE_DECOMMISSIONED",
            "site_id": site_id,
            "decommission_date": decommission_date.isoformat(),
            "reason": reason,
            "timestamp": _utcnow().isoformat(),
        })

        return updated

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_site(
        self,
        site_id: str,
        taxonomy: Optional[Dict[str, Any]] = None,
    ) -> SiteClassification:
        """Classify a site against a facility taxonomy.

        Determines the site's size class, expected emission intensity,
        and reporting priority based on its characteristics and
        facility type.

        Args:
            site_id: ID of the site to classify.
            taxonomy: Optional custom taxonomy overrides.
                Keys: 'size_thresholds', 'high_intensity_types',
                'medium_intensity_types', 'priority_thresholds'.

        Returns:
            SiteClassification with all classification fields.

        Raises:
            KeyError: If site not found.
        """
        if site_id not in self._sites:
            raise KeyError(f"Site '{site_id}' not found in registry.")

        site = self._sites[site_id]
        taxonomy = taxonomy or {}
        logger.info("Classifying site '%s' (id=%s).", site.site_name, site_id)

        # Resolve taxonomy parameters with defaults
        size_thresholds = taxonomy.get("size_thresholds", SIZE_CLASS_THRESHOLDS)
        high_intensity = taxonomy.get(
            "high_intensity_types", HIGH_INTENSITY_FACILITY_TYPES
        )
        medium_intensity = taxonomy.get(
            "medium_intensity_types", MEDIUM_INTENSITY_FACILITY_TYPES
        )
        priority_thresholds = taxonomy.get(
            "priority_thresholds", PRIORITY_THRESHOLDS
        )

        # Determine size class
        floor_area = Decimal("0")
        if site.characteristics is not None:
            floor_area = site.characteristics.floor_area_m2

        size_class = "SMALL"
        rationale: Dict[str, str] = {}
        for cls_name, (lower, upper) in sorted(
            size_thresholds.items(), key=lambda x: x[1][0]
        ):
            if lower <= floor_area < upper:
                size_class = cls_name
                break
        else:
            # If floor_area exceeds all thresholds, use the largest class
            if floor_area >= Decimal("100000"):
                size_class = "MEGA"
        rationale["size_class"] = (
            f"Floor area {floor_area} m2 falls in {size_class} range."
        )

        # Determine emission intensity class
        ftype = site.facility_type.upper()
        if ftype in high_intensity:
            emission_intensity_class = "HIGH"
        elif ftype in medium_intensity:
            emission_intensity_class = "MEDIUM"
        else:
            emission_intensity_class = "LOW"
        rationale["emission_intensity"] = (
            f"Facility type {ftype} mapped to {emission_intensity_class} intensity."
        )

        # Determine reporting priority
        reporting_priority = "LOW"
        for priority, threshold in sorted(
            priority_thresholds.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if floor_area >= threshold:
                reporting_priority = priority
                break
        rationale["reporting_priority"] = (
            f"Floor area {floor_area} m2 yields {reporting_priority} priority."
        )

        # Primary and secondary categories
        primary_category = ftype
        secondary_categories: List[str] = []
        if site.business_unit:
            secondary_categories.append(f"BU:{site.business_unit}")
        if site.region:
            secondary_categories.append(f"REGION:{site.region}")
        if size_class in ("LARGE", "MEGA"):
            secondary_categories.append("MAJOR_FACILITY")

        result = SiteClassification(
            site_id=site_id,
            primary_category=primary_category,
            secondary_categories=secondary_categories,
            size_class=size_class,
            emission_intensity_class=emission_intensity_class,
            reporting_priority=reporting_priority,
            classification_rationale=rationale,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Groups
    # ------------------------------------------------------------------

    def create_site_group(
        self,
        group_name: str,
        group_type: str,
        site_ids: List[str],
        description: Optional[str] = None,
    ) -> SiteGroup:
        """Create a new site group.

        Groups are logical collections of sites for aggregated
        reporting. A site can belong to multiple groups.

        Args:
            group_name: Name for the group.
            group_type: Type of grouping (REGION, BUSINESS_UNIT, etc.).
            site_ids: List of site IDs to include.
            description: Optional group description.

        Returns:
            The created SiteGroup.

        Raises:
            ValueError: If any site_id is not found.
        """
        logger.info(
            "Creating site group '%s' of type '%s' with %d sites.",
            group_name,
            group_type,
            len(site_ids),
        )

        # Validate all site IDs exist
        missing = [sid for sid in site_ids if sid not in self._sites]
        if missing:
            raise ValueError(
                f"Site IDs not found in registry: {missing}"
            )

        # Check for duplicate group names within same type
        for existing_group in self._groups.values():
            if (
                existing_group.group_name == group_name
                and existing_group.group_type == group_type.upper()
            ):
                raise ValueError(
                    f"Group '{group_name}' of type '{group_type}' already exists."
                )

        group = SiteGroup(
            group_name=group_name,
            group_type=group_type,
            member_site_ids=list(site_ids),
            description=description,
        )
        self._groups[group.group_id] = group

        self._change_log.append({
            "event": "GROUP_CREATED",
            "group_id": group.group_id,
            "group_name": group_name,
            "member_count": len(site_ids),
            "timestamp": _utcnow().isoformat(),
        })

        logger.info("Group '%s' created (id=%s).", group_name, group.group_id)
        return group

    def add_sites_to_group(
        self,
        group_id: str,
        site_ids: List[str],
    ) -> SiteGroup:
        """Add sites to an existing group.

        Args:
            group_id: The group to add sites to.
            site_ids: Sites to add.

        Returns:
            The updated SiteGroup.

        Raises:
            KeyError: If the group does not exist.
            ValueError: If any site_id is invalid.
        """
        if group_id not in self._groups:
            raise KeyError(f"Group '{group_id}' not found.")

        missing = [sid for sid in site_ids if sid not in self._sites]
        if missing:
            raise ValueError(f"Site IDs not found: {missing}")

        group = self._groups[group_id]
        existing_ids = set(group.member_site_ids)
        added: List[str] = []
        for sid in site_ids:
            if sid not in existing_ids:
                existing_ids.add(sid)
                added.append(sid)

        updated_group = group.model_copy(
            update={"member_site_ids": sorted(existing_ids)}
        )
        self._groups[group_id] = updated_group

        logger.info("Added %d site(s) to group '%s'.", len(added), group.group_name)
        return updated_group

    def remove_sites_from_group(
        self,
        group_id: str,
        site_ids: List[str],
    ) -> SiteGroup:
        """Remove sites from an existing group.

        Args:
            group_id: The group to remove sites from.
            site_ids: Sites to remove.

        Returns:
            The updated SiteGroup.

        Raises:
            KeyError: If the group does not exist.
        """
        if group_id not in self._groups:
            raise KeyError(f"Group '{group_id}' not found.")

        group = self._groups[group_id]
        remove_set = set(site_ids)
        new_members = [sid for sid in group.member_site_ids if sid not in remove_set]

        updated_group = group.model_copy(
            update={"member_site_ids": new_members}
        )
        self._groups[group_id] = updated_group

        removed_count = len(group.member_site_ids) - len(new_members)
        logger.info(
            "Removed %d site(s) from group '%s'.", removed_count, group.group_name
        )
        return updated_group

    # ------------------------------------------------------------------
    # Portfolio Summary & Analytics
    # ------------------------------------------------------------------

    def get_portfolio_summary(
        self,
        sites: Optional[List[SiteRecord]] = None,
        groups: Optional[List[SiteGroup]] = None,
    ) -> SiteRegistryResult:
        """Generate a portfolio-level summary of sites.

        Aggregates floor area, headcount, production output, and
        produces distribution analytics by facility type, lifecycle
        status, and geography.

        Args:
            sites: List of sites to summarise. If None, uses the
                internal registry.
            groups: List of groups. If None, uses internal groups.

        Returns:
            SiteRegistryResult with all aggregated metrics.
        """
        if sites is None:
            sites = list(self._sites.values())
        if groups is None:
            groups = list(self._groups.values())

        logger.info("Generating portfolio summary for %d site(s).", len(sites))

        active_sites = [s for s in sites if s.is_active]
        total_active = len(active_sites)

        # Aggregate floor area
        total_floor_area = Decimal("0")
        total_headcount = 0
        total_production = Decimal("0")
        countries: set = set()
        facility_types: Dict[str, int] = {}
        lifecycle_statuses: Dict[str, int] = {}
        geo_dist: Dict[str, int] = {}

        for site in sites:
            # Lifecycle distribution (all sites)
            status = site.lifecycle_status
            lifecycle_statuses[status] = lifecycle_statuses.get(status, 0) + 1

        for site in active_sites:
            # Floor area and headcount
            if site.characteristics is not None:
                total_floor_area += site.characteristics.floor_area_m2
                total_headcount += site.characteristics.headcount
                if site.characteristics.production_output is not None:
                    total_production += site.characteristics.production_output

            # Country
            countries.add(site.country)

            # Facility type distribution
            ftype = site.facility_type
            facility_types[ftype] = facility_types.get(ftype, 0) + 1

            # Geographic distribution
            geo_dist[site.country] = geo_dist.get(site.country, 0) + 1

        # Averages
        avg_floor = _safe_divide(
            total_floor_area,
            _decimal(total_active),
        )
        avg_headcount = _safe_divide(
            _decimal(total_headcount),
            _decimal(total_active),
        )

        result = SiteRegistryResult(
            sites=sites,
            groups=groups,
            total_active_sites=total_active,
            total_floor_area=_round2(total_floor_area),
            total_headcount=total_headcount,
            countries_covered=len(countries),
            facility_type_distribution=facility_types,
            lifecycle_status_distribution=lifecycle_statuses,
            geographic_distribution=geo_dist,
            total_production_output=_round2(total_production),
            avg_floor_area=_round2(avg_floor),
            avg_headcount=_round2(avg_headcount),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Portfolio summary: %d active sites, %d countries, "
            "total floor area=%s m2, total headcount=%d.",
            total_active,
            len(countries),
            result.total_floor_area,
            total_headcount,
        )
        return result

    def get_sites_by_filter(
        self,
        sites: Optional[List[SiteRecord]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SiteRecord]:
        """Filter sites by arbitrary criteria.

        Supported filter keys:
            - country: str or List[str]
            - facility_type: str or List[str]
            - lifecycle_status: str or List[str]
            - is_active: bool
            - business_unit: str
            - legal_entity_id: str
            - tag: str (matches if tag is in site.tags)
            - min_floor_area: Decimal
            - max_floor_area: Decimal
            - min_headcount: int
            - max_headcount: int

        Args:
            sites: Sites to filter. If None, uses internal registry.
            filters: Dictionary of filter criteria.

        Returns:
            List of SiteRecords matching all filters.
        """
        if sites is None:
            sites = list(self._sites.values())
        if not filters:
            return list(sites)

        results: List[SiteRecord] = []

        for site in sites:
            if not self._matches_filters(site, filters):
                continue
            results.append(site)

        logger.info(
            "Filter returned %d of %d sites.", len(results), len(sites)
        )
        return results

    def _matches_filters(
        self,
        site: SiteRecord,
        filters: Dict[str, Any],
    ) -> bool:
        """Check if a site matches all filter criteria.

        Args:
            site: The site to check.
            filters: The filter criteria.

        Returns:
            True if all filters match.
        """
        for key, value in filters.items():
            if key == "country":
                acceptable = value if isinstance(value, list) else [value]
                if site.country not in acceptable:
                    return False

            elif key == "facility_type":
                acceptable = value if isinstance(value, list) else [value]
                acceptable_upper = [v.upper() for v in acceptable]
                if site.facility_type.upper() not in acceptable_upper:
                    return False

            elif key == "lifecycle_status":
                acceptable = value if isinstance(value, list) else [value]
                acceptable_upper = [v.upper() for v in acceptable]
                if site.lifecycle_status.upper() not in acceptable_upper:
                    return False

            elif key == "is_active":
                if site.is_active != value:
                    return False

            elif key == "business_unit":
                if site.business_unit != value:
                    return False

            elif key == "legal_entity_id":
                if site.legal_entity_id != value:
                    return False

            elif key == "tag":
                if value not in site.tags:
                    return False

            elif key == "min_floor_area":
                threshold = _decimal(value)
                actual = Decimal("0")
                if site.characteristics is not None:
                    actual = site.characteristics.floor_area_m2
                if actual < threshold:
                    return False

            elif key == "max_floor_area":
                threshold = _decimal(value)
                actual = Decimal("0")
                if site.characteristics is not None:
                    actual = site.characteristics.floor_area_m2
                if actual > threshold:
                    return False

            elif key == "min_headcount":
                actual = 0
                if site.characteristics is not None:
                    actual = site.characteristics.headcount
                if actual < int(value):
                    return False

            elif key == "max_headcount":
                actual = 0
                if site.characteristics is not None:
                    actual = site.characteristics.headcount
                if actual > int(value):
                    return False

            else:
                logger.warning("Unknown filter key '%s' ignored.", key)

        return True

    def get_facility_type_distribution(
        self,
        sites: Optional[List[SiteRecord]] = None,
    ) -> Dict[str, int]:
        """Get the distribution of active sites by facility type.

        Args:
            sites: Sites to analyse. If None, uses internal registry.

        Returns:
            Dictionary mapping facility_type to count.
        """
        if sites is None:
            sites = list(self._sites.values())

        distribution: Dict[str, int] = {}
        for site in sites:
            if not site.is_active:
                continue
            ftype = site.facility_type
            distribution[ftype] = distribution.get(ftype, 0) + 1
        return distribution

    def get_geographic_distribution(
        self,
        sites: Optional[List[SiteRecord]] = None,
    ) -> Dict[str, int]:
        """Get the distribution of active sites by country.

        Args:
            sites: Sites to analyse. If None, uses internal registry.

        Returns:
            Dictionary mapping country code to count of active sites.
        """
        if sites is None:
            sites = list(self._sites.values())

        distribution: Dict[str, int] = {}
        for site in sites:
            if not site.is_active:
                continue
            distribution[site.country] = distribution.get(site.country, 0) + 1
        return distribution

    def get_region_distribution(
        self,
        sites: Optional[List[SiteRecord]] = None,
    ) -> Dict[str, int]:
        """Get the distribution of active sites by sub-national region.

        Args:
            sites: Sites to analyse. If None, uses internal registry.

        Returns:
            Dictionary mapping region to count of active sites.
        """
        if sites is None:
            sites = list(self._sites.values())

        distribution: Dict[str, int] = {}
        for site in sites:
            if not site.is_active:
                continue
            region_key = site.region or "UNKNOWN"
            distribution[region_key] = distribution.get(region_key, 0) + 1
        return distribution

    # ------------------------------------------------------------------
    # Completeness Validation
    # ------------------------------------------------------------------

    def validate_site_completeness(
        self,
        site: SiteRecord,
    ) -> SiteCompletenessResult:
        """Validate the data completeness of a single site record.

        Checks required and optional fields against the standard
        completeness criteria. Returns a detailed report with
        completeness percentage and recommendations.

        Completeness Thresholds:
            >= 95%: COMPLETE
            >= 80%: MOSTLY_COMPLETE
            >= 60%: PARTIAL
            >= 40%: MINIMAL
            <  40%: INSUFFICIENT

        Args:
            site: The SiteRecord to validate.

        Returns:
            SiteCompletenessResult with detailed breakdown.
        """
        logger.info("Validating completeness for site '%s'.", site.site_id)

        missing_required: List[str] = []
        missing_optional: List[str] = []
        recommendations: List[str] = []

        # Check top-level required fields
        required_total = len(REQUIRED_SITE_FIELDS)
        required_present = 0

        for field_name in REQUIRED_SITE_FIELDS:
            val = getattr(site, field_name, None)
            if val is not None and val != "":
                required_present += 1
            else:
                missing_required.append(field_name)

        # Check optional top-level fields
        optional_total = len(OPTIONAL_SITE_FIELDS)
        optional_present = 0

        for field_name in OPTIONAL_SITE_FIELDS:
            val = getattr(site, field_name, None)
            if val is not None and val != "":
                optional_present += 1
            else:
                missing_optional.append(field_name)

        # Check characteristics sub-fields
        if site.characteristics is not None:
            required_total += len(REQUIRED_CHARACTERISTICS_FIELDS)
            for field_name in REQUIRED_CHARACTERISTICS_FIELDS:
                val = getattr(site.characteristics, field_name, None)
                if val is not None:
                    required_present += 1
                else:
                    missing_required.append(f"characteristics.{field_name}")

            optional_total += len(OPTIONAL_CHARACTERISTICS_FIELDS)
            for field_name in OPTIONAL_CHARACTERISTICS_FIELDS:
                val = getattr(site.characteristics, field_name, None)
                if val is not None and val != "":
                    optional_present += 1
                else:
                    missing_optional.append(f"characteristics.{field_name}")
        else:
            # No characteristics at all - all those fields are missing
            required_total += len(REQUIRED_CHARACTERISTICS_FIELDS)
            for field_name in REQUIRED_CHARACTERISTICS_FIELDS:
                missing_required.append(f"characteristics.{field_name}")
            optional_total += len(OPTIONAL_CHARACTERISTICS_FIELDS)
            for field_name in OPTIONAL_CHARACTERISTICS_FIELDS:
                missing_optional.append(f"characteristics.{field_name}")

        # Calculate weighted completeness (required: 70%, optional: 30%)
        total_fields = required_total + optional_total
        if total_fields == 0:
            completeness_pct = Decimal("100")
        else:
            required_score = _safe_divide(
                _decimal(required_present),
                _decimal(required_total),
            ) * Decimal("70")
            optional_score = _safe_divide(
                _decimal(optional_present),
                _decimal(optional_total),
            ) * Decimal("30")
            completeness_pct = _round2(required_score + optional_score)

        # Determine completeness level
        if completeness_pct >= Decimal("95"):
            level = CompletenessLevel.COMPLETE.value
        elif completeness_pct >= Decimal("80"):
            level = CompletenessLevel.MOSTLY_COMPLETE.value
        elif completeness_pct >= Decimal("60"):
            level = CompletenessLevel.PARTIAL.value
        elif completeness_pct >= Decimal("40"):
            level = CompletenessLevel.MINIMAL.value
        else:
            level = CompletenessLevel.INSUFFICIENT.value

        # Generate recommendations
        if missing_required:
            recommendations.append(
                f"Complete {len(missing_required)} required field(s): "
                f"{', '.join(missing_required[:5])}"
                + ("..." if len(missing_required) > 5 else "")
            )
        if site.characteristics is None:
            recommendations.append(
                "Add facility characteristics (floor area, headcount, "
                "operating hours) for intensity calculations."
            )
        if site.latitude is None or site.longitude is None:
            recommendations.append(
                "Add geographic coordinates for regional factor assignment."
            )
        if not site.tags:
            recommendations.append(
                "Add tags for easier filtering and grouping."
            )
        if site.commissioning_date is None:
            recommendations.append(
                "Add commissioning date for lifecycle tracking."
            )

        result = SiteCompletenessResult(
            site_id=site.site_id,
            completeness_level=level,
            completeness_pct=completeness_pct,
            required_fields_total=required_total,
            required_fields_present=required_present,
            optional_fields_total=optional_total,
            optional_fields_present=optional_present,
            missing_required=missing_required,
            missing_optional=missing_optional,
            recommendations=recommendations,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Site '%s' completeness: %s (%s%%).",
            site.site_id,
            level,
            completeness_pct,
        )
        return result

    def validate_all_sites_completeness(
        self,
        sites: Optional[List[SiteRecord]] = None,
    ) -> Dict[str, SiteCompletenessResult]:
        """Validate completeness for all sites in the registry.

        Args:
            sites: Sites to validate. If None, uses internal registry.

        Returns:
            Dictionary mapping site_id to SiteCompletenessResult.
        """
        if sites is None:
            sites = list(self._sites.values())

        results: Dict[str, SiteCompletenessResult] = {}
        for site in sites:
            results[site.site_id] = self.validate_site_completeness(site)

        summary_counts: Dict[str, int] = {}
        for r in results.values():
            lvl = r.completeness_level
            summary_counts[lvl] = summary_counts.get(lvl, 0) + 1

        logger.info(
            "Completeness validation complete for %d sites: %s.",
            len(sites),
            summary_counts,
        )
        return results

    # ------------------------------------------------------------------
    # Accessors & Utilities
    # ------------------------------------------------------------------

    def get_site(self, site_id: str) -> SiteRecord:
        """Retrieve a site by ID.

        Args:
            site_id: The site ID.

        Returns:
            The SiteRecord.

        Raises:
            KeyError: If not found.
        """
        if site_id not in self._sites:
            raise KeyError(f"Site '{site_id}' not found.")
        return self._sites[site_id]

    def get_all_sites(self) -> List[SiteRecord]:
        """Return all sites in the registry.

        Returns:
            List of all SiteRecords.
        """
        return list(self._sites.values())

    def get_active_sites(self) -> List[SiteRecord]:
        """Return only active sites.

        Returns:
            List of active SiteRecords.
        """
        return [s for s in self._sites.values() if s.is_active]

    def get_group(self, group_id: str) -> SiteGroup:
        """Retrieve a group by ID.

        Args:
            group_id: The group ID.

        Returns:
            The SiteGroup.

        Raises:
            KeyError: If not found.
        """
        if group_id not in self._groups:
            raise KeyError(f"Group '{group_id}' not found.")
        return self._groups[group_id]

    def get_all_groups(self) -> List[SiteGroup]:
        """Return all groups.

        Returns:
            List of all SiteGroups.
        """
        return list(self._groups.values())

    def get_sites_in_group(self, group_id: str) -> List[SiteRecord]:
        """Get all sites belonging to a specific group.

        Args:
            group_id: The group to query.

        Returns:
            List of SiteRecords in the group.

        Raises:
            KeyError: If group not found.
        """
        group = self.get_group(group_id)
        return [
            self._sites[sid]
            for sid in group.member_site_ids
            if sid in self._sites
        ]

    def get_change_log(self) -> List[Dict[str, Any]]:
        """Return the complete change log.

        Returns:
            List of change log entries in chronological order.
        """
        return list(self._change_log)

    def get_site_count(self) -> int:
        """Return total number of sites in registry.

        Returns:
            Integer count of all sites (active + inactive).
        """
        return len(self._sites)

    def export_registry(self) -> Dict[str, Any]:
        """Export the entire registry as a serialisable dictionary.

        Returns:
            Dictionary with sites, groups, and metadata.
        """
        sites_data = [s.model_dump(mode="json") for s in self._sites.values()]
        groups_data = [g.model_dump(mode="json") for g in self._groups.values()]

        export = {
            "version": _MODULE_VERSION,
            "exported_at": _utcnow().isoformat(),
            "total_sites": len(self._sites),
            "total_groups": len(self._groups),
            "sites": sites_data,
            "groups": groups_data,
        }
        export["provenance_hash"] = _compute_hash(export)
        return export

    def import_sites(self, site_dicts: List[Dict[str, Any]]) -> int:
        """Bulk import sites from a list of dictionaries.

        Skips sites whose site_code already exists. Returns the number
        of sites successfully imported.

        Args:
            site_dicts: List of site data dictionaries.

        Returns:
            Number of sites imported.
        """
        imported = 0
        existing_codes = {s.site_code for s in self._sites.values()}

        for sd in site_dicts:
            code = sd.get("site_code", "")
            if code in existing_codes:
                logger.warning("Skipping duplicate site_code '%s'.", code)
                continue
            try:
                site = self.register_site(sd)
                existing_codes.add(site.site_code)
                imported += 1
            except (ValueError, TypeError) as exc:
                logger.error("Failed to import site '%s': %s.", code, exc)

        logger.info("Imported %d of %d sites.", imported, len(site_dicts))
        return imported
