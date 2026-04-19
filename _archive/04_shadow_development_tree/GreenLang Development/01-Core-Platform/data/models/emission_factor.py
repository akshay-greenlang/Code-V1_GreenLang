# -*- coding: utf-8 -*-
"""
Emission Factor Data Models

This module defines the core data models for emission factors used throughout
the GreenLang ecosystem. All models use Pydantic for validation and type safety.

Example:
    >>> factor = EmissionFactor(
    ...     factor_id="diesel_fuel",
    ...     name="Diesel Fuel",
    ...     emission_factor_kg_co2e=2.68,
    ...     unit="liter",
    ...     scope="Scope 1",
    ...     source=SourceProvenance(...)
    ... )
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime, date
from enum import Enum
import hashlib
import json
from greenlang.utilities.determinism import DeterministicClock


class DataQualityTier(str, Enum):
    """Data quality tiers for emission factors."""
    TIER_1 = "Tier 1"
    TIER_2 = "Tier 2"
    TIER_3 = "Tier 3"
    FACILITY_SPECIFIC = "Facility Specific"


class Scope(str, Enum):
    """GHG Protocol scopes."""
    SCOPE_1 = "Scope 1"
    SCOPE_2_LOCATION = "Scope 2 - Location-Based"
    SCOPE_2_MARKET = "Scope 2 - Market-Based"
    SCOPE_3 = "Scope 3"


class GeographyLevel(str, Enum):
    """Geographic granularity levels."""
    FACILITY = "Facility"
    CITY = "City"
    STATE = "State"
    COUNTRY = "Country"
    REGION = "Region"
    GLOBAL = "Global"


class Geography(BaseModel):
    """Geographic scope for emission factor."""

    geographic_scope: str = Field(..., description="Primary geographic identifier (e.g., 'United States', 'California')")
    geography_level: GeographyLevel = Field(..., description="Granularity level")
    country_code: Optional[str] = Field(None, max_length=3, description="ISO 3166-1 alpha-3 country code")
    state_province: Optional[str] = Field(None, description="State or province")
    region: Optional[str] = Field(None, description="Regional grouping (e.g., 'North America', 'EU')")

    @validator('country_code')
    def validate_country_code(cls, v):
        """Validate country code format."""
        if v and len(v) not in [2, 3]:
            raise ValueError("Country code must be 2 or 3 characters")
        return v.upper() if v else v


class SourceProvenance(BaseModel):
    """Source provenance for audit trail."""

    source_org: str = Field(..., description="Source organization (e.g., 'EPA', 'IPCC', 'DEFRA')")
    source_publication: Optional[str] = Field(None, description="Publication name")
    source_uri: str = Field(..., description="URI to source document for verification")
    standard: Optional[str] = Field(None, description="Reporting standard (e.g., 'GHG Protocol')")
    year_published: Optional[int] = Field(None, ge=1990, le=2030, description="Year source was published")

    @validator('source_uri')
    def validate_uri(cls, v):
        """Validate URI format."""
        if not v.startswith(('http://', 'https://', 'ftp://')):
            raise ValueError("source_uri must be a valid URI")
        return v


class DataQualityScore(BaseModel):
    """Data quality assessment."""

    tier: DataQualityTier = Field(..., description="Data quality tier")
    uncertainty_percent: Optional[float] = Field(None, ge=0, le=100, description="Uncertainty as percentage")
    confidence_95ci: Optional[float] = Field(None, ge=0, description="95% confidence interval")
    completeness_score: Optional[float] = Field(None, ge=0, le=1, description="Data completeness (0-1)")

    def is_acceptable_quality(self, min_tier: DataQualityTier = DataQualityTier.TIER_2) -> bool:
        """Check if data quality meets minimum requirement."""
        tier_order = {
            DataQualityTier.FACILITY_SPECIFIC: 4,
            DataQualityTier.TIER_3: 3,
            DataQualityTier.TIER_2: 2,
            DataQualityTier.TIER_1: 1
        }
        return tier_order.get(self.tier, 0) >= tier_order.get(min_tier, 0)


class EmissionFactorUnit(BaseModel):
    """Unit conversion for emission factor."""

    unit_name: str = Field(..., description="Unit name (e.g., 'liter', 'kwh', 'kg')")
    emission_factor_value: float = Field(..., gt=0, description="Emission factor in kg CO2e per unit")
    conversion_to_base: Optional[float] = Field(None, gt=0, description="Conversion factor to base unit")

    @validator('unit_name')
    def validate_unit_name(cls, v):
        """Normalize unit name."""
        return v.lower().strip()


class GasVector(BaseModel):
    """Individual gas contribution to total CO2e."""

    gas_type: str = Field(..., description="GHG type (CO2, CH4, N2O, etc.)")
    kg_per_unit: float = Field(..., ge=0, description="kg of this gas per activity unit")
    gwp: Optional[int] = Field(None, ge=1, description="Global Warming Potential (AR6)")

    @validator('gas_type')
    def validate_gas_type(cls, v):
        """Validate gas type."""
        valid_gases = ['CO2', 'CH4', 'N2O', 'HFCs', 'PFCs', 'SF6', 'NF3']
        if v.upper() not in valid_gases:
            raise ValueError(f"gas_type must be one of {valid_gases}")
        return v.upper()


class EmissionFactor(BaseModel):
    """
    Core emission factor model.

    This is the primary data model for all emission factors in the GreenLang system.
    Every factor must have a unique ID, value, unit, source, and geography.

    Attributes:
        factor_id: Unique identifier (e.g., 'diesel_fuel', 'us_grid_caiso')
        name: Human-readable name
        category: Primary category (e.g., 'fuels', 'grids', 'materials')
        subcategory: Optional subcategory for finer granularity
        emission_factor_kg_co2e: Primary emission factor value in kg CO2e
        unit: Primary unit for this factor
        scope: GHG Protocol scope
        source: Source provenance for audit trail
        geography: Geographic scope
        data_quality: Data quality assessment
        last_updated: Date factor was last updated
        year_applicable: Year this factor applies to
        renewable_share: Optional renewable energy share (0-1)
        notes: Optional additional information
        metadata: Optional additional structured metadata

    Example:
        >>> factor = EmissionFactor(
        ...     factor_id="diesel_fuel",
        ...     name="Diesel Fuel",
        ...     category="fuels",
        ...     emission_factor_kg_co2e=2.68,
        ...     unit="liter",
        ...     scope=Scope.SCOPE_1,
        ...     source=SourceProvenance(...),
        ...     geography=Geography(...),
        ...     data_quality=DataQualityScore(...),
        ...     last_updated=date.today()
        ... )
    """

    factor_id: str = Field(..., description="Unique factor identifier")
    name: str = Field(..., min_length=1, description="Human-readable name")
    category: str = Field(..., description="Primary category")
    subcategory: Optional[str] = Field(None, description="Subcategory")

    # Emission factor value
    emission_factor_kg_co2e: float = Field(..., gt=0, description="Emission factor in kg CO2e per unit")
    unit: str = Field(..., description="Unit for this factor")

    # Additional units (for factors with multiple unit representations)
    additional_units: List[EmissionFactorUnit] = Field(default_factory=list, description="Additional unit conversions")

    # GHG scope and source
    scope: str = Field(..., description="GHG Protocol scope")
    source: SourceProvenance = Field(..., description="Source provenance")
    geography: Geography = Field(..., description="Geographic scope")
    data_quality: DataQualityScore = Field(..., description="Data quality")

    # Temporal information
    last_updated: date = Field(..., description="Date last updated")
    year_applicable: Optional[int] = Field(None, ge=1990, le=2030, description="Applicable year")

    # Optional attributes
    renewable_share: Optional[float] = Field(None, ge=0, le=1, description="Renewable share (0-1)")
    gas_vectors: List[GasVector] = Field(default_factory=list, description="Individual gas contributions")
    notes: Optional[str] = Field(None, description="Additional notes")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator('factor_id')
    def validate_factor_id(cls, v):
        """Validate factor ID format."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("factor_id must be alphanumeric with underscores or hyphens")
        return v.lower()

    @validator('unit')
    def validate_unit(cls, v):
        """Normalize unit."""
        return v.lower().strip()

    def get_factor_for_unit(self, target_unit: str) -> float:
        """
        Get emission factor for a specific unit.

        Args:
            target_unit: Target unit to convert to

        Returns:
            Emission factor in kg CO2e per target unit

        Raises:
            ValueError: If unit not available
        """
        target_unit = target_unit.lower().strip()

        # Check primary unit
        if self.unit == target_unit:
            return self.emission_factor_kg_co2e

        # Check additional units
        for unit_conversion in self.additional_units:
            if unit_conversion.unit_name == target_unit:
                return unit_conversion.emission_factor_value

        raise ValueError(f"Unit '{target_unit}' not available for factor '{self.factor_id}'")

    def calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        provenance_data = {
            'factor_id': self.factor_id,
            'emission_factor': self.emission_factor_kg_co2e,
            'unit': self.unit,
            'source_uri': self.source.source_uri,
            'last_updated': self.last_updated.isoformat()
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def is_stale(self, max_age_years: int = 3) -> bool:
        """Check if factor is stale (older than max_age_years)."""
        years_old = (DeterministicClock.now().date() - self.last_updated).days / 365.25
        return years_old > max_age_years


class EmissionResult(BaseModel):
    """
    Result of an emission calculation.

    This model captures the complete audit trail for an emission calculation,
    including the input activity data, the factor used, and the calculation result.

    Attributes:
        activity_amount: Amount of activity (e.g., 100 gallons, 500 kWh)
        activity_unit: Unit of activity
        emissions_kg_co2e: Calculated emissions in kg CO2e
        factor_used: The emission factor that was applied
        calculation_timestamp: When calculation was performed
        audit_trail: SHA-256 hash for complete audit trail
        warnings: Optional warnings (e.g., stale factor)

    Example:
        >>> result = EmissionResult(
        ...     activity_amount=100.0,
        ...     activity_unit="gallons",
        ...     emissions_kg_co2e=1021.0,
        ...     factor_used=diesel_factor,
        ...     calculation_timestamp=DeterministicClock.now(),
        ...     audit_trail="abc123..."
        ... )
    """

    activity_amount: float = Field(..., description="Activity amount")
    activity_unit: str = Field(..., description="Activity unit")
    emissions_kg_co2e: float = Field(..., ge=0, description="Calculated emissions in kg CO2e")
    emissions_metric_tons_co2e: float = Field(..., ge=0, description="Calculated emissions in metric tons CO2e")

    factor_used: EmissionFactor = Field(..., description="Emission factor applied")
    factor_value_applied: float = Field(..., description="Actual factor value used (kg CO2e/unit)")

    calculation_timestamp: datetime = Field(..., description="Calculation timestamp")
    audit_trail: str = Field(..., description="SHA-256 hash for audit trail")

    warnings: List[str] = Field(default_factory=list, description="Calculation warnings")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator('emissions_metric_tons_co2e', always=True)
    def calculate_metric_tons(cls, v, values):
        """Calculate metric tons from kg."""
        if 'emissions_kg_co2e' in values:
            return values['emissions_kg_co2e'] / 1000.0
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'activity_amount': self.activity_amount,
            'activity_unit': self.activity_unit,
            'emissions_kg_co2e': self.emissions_kg_co2e,
            'emissions_metric_tons_co2e': self.emissions_metric_tons_co2e,
            'factor_id': self.factor_used.factor_id,
            'factor_name': self.factor_used.name,
            'factor_value': self.factor_value_applied,
            'calculation_timestamp': self.calculation_timestamp.isoformat(),
            'audit_trail': self.audit_trail,
            'warnings': self.warnings
        }


class FactorSearchCriteria(BaseModel):
    """Search criteria for emission factors."""

    category: Optional[str] = Field(None, description="Filter by category")
    subcategory: Optional[str] = Field(None, description="Filter by subcategory")
    scope: Optional[str] = Field(None, description="Filter by scope")
    geographic_scope: Optional[str] = Field(None, description="Filter by geography")
    source_org: Optional[str] = Field(None, description="Filter by source organization")
    min_quality_tier: Optional[DataQualityTier] = Field(None, description="Minimum quality tier")
    max_age_years: Optional[int] = Field(None, ge=1, le=10, description="Maximum age in years")
    search_text: Optional[str] = Field(None, description="Free text search in name/notes")

    def to_sql_where(self) -> tuple[str, Dict[str, Any]]:
        """Convert search criteria to SQL WHERE clause."""
        conditions = []
        params = {}

        if self.category:
            conditions.append("category = :category")
            params['category'] = self.category

        if self.subcategory:
            conditions.append("subcategory = :subcategory")
            params['subcategory'] = self.subcategory

        if self.scope:
            conditions.append("scope = :scope")
            params['scope'] = self.scope

        if self.geographic_scope:
            conditions.append("geographic_scope LIKE :geographic_scope")
            params['geographic_scope'] = f"%{self.geographic_scope}%"

        if self.source_org:
            conditions.append("source_org = :source_org")
            params['source_org'] = self.source_org

        if self.min_quality_tier:
            conditions.append("data_quality_tier >= :min_tier")
            params['min_tier'] = self.min_quality_tier.value

        if self.max_age_years:
            conditions.append(f"julianday('now') - julianday(last_updated) <= :max_age_days")
            params['max_age_days'] = self.max_age_years * 365

        if self.search_text:
            conditions.append("(name LIKE :search_text OR notes LIKE :search_text)")
            params['search_text'] = f"%{self.search_text}%"

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return where_clause, params
