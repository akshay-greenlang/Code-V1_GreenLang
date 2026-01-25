"""
Emission Factor Database Schema
===============================

Comprehensive schema design for 100,000+ emission factors with versioning,
audit trails, and query optimization support.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime, date
from decimal import Decimal
from pydantic import BaseModel, Field, validator, root_validator
import hashlib
import json


# =============================================================================
# ENUMERATIONS
# =============================================================================

class GHGType(str, Enum):
    """Greenhouse gas types per GHG Protocol."""
    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    HFCS = "HFCs"
    PFCS = "PFCs"
    SF6 = "SF6"
    NF3 = "NF3"
    CO2E = "CO2e"  # CO2 equivalent (combined)


class ScopeType(str, Enum):
    """GHG Protocol scope types."""
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"
    WTT = "well_to_tank"  # Upstream


class IndustryCategory(str, Enum):
    """Industry categories for emission factors."""
    STEEL = "steel"
    CEMENT = "cement"
    ALUMINUM = "aluminum"
    CHEMICALS = "chemicals"
    FERTILIZER = "fertilizer"
    HYDROGEN = "hydrogen"
    ELECTRICITY = "electricity"
    AUTOMOTIVE = "automotive"
    AVIATION = "aviation"
    SHIPPING = "shipping"
    RAIL = "rail"
    ROAD_FREIGHT = "road_freight"
    AGRICULTURE = "agriculture"
    TEXTILES = "textiles"
    ELECTRONICS = "electronics"
    CONSTRUCTION = "construction"
    WASTE = "waste"
    FOOD_BEVERAGE = "food_beverage"
    PAPER_PULP = "paper_pulp"
    GLASS = "glass"
    GENERAL = "general"


class GeographicRegion(str, Enum):
    """Geographic regions for emission factors."""
    GLOBAL = "global"
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST = "middle_east"
    AFRICA = "africa"
    USA = "usa"
    CANADA = "canada"
    MEXICO = "mexico"
    UK = "uk"
    GERMANY = "germany"
    FRANCE = "france"
    ITALY = "italy"
    SPAIN = "spain"
    CHINA = "china"
    INDIA = "india"
    JAPAN = "japan"
    SOUTH_KOREA = "south_korea"
    AUSTRALIA = "australia"
    BRAZIL = "brazil"
    RUSSIA = "russia"
    TURKEY = "turkey"


class DataSourceType(str, Enum):
    """Data source types for emission factors."""
    DEFRA = "defra"
    EPA_EGRID = "epa_egrid"
    EPA_GHG = "epa_ghg"
    IPCC_AR6 = "ipcc_ar6"
    IPCC_AR7 = "ipcc_ar7"
    ECOINVENT = "ecoinvent"
    WORLD_BANK = "world_bank"
    IEA = "iea"
    FAO = "fao"
    CUSTOMER_PROVIDED = "customer_provided"
    CALCULATED = "calculated"


class QualityTier(str, Enum):
    """Data quality tier classification."""
    TIER_1 = "tier_1"  # Default values (lowest quality)
    TIER_2 = "tier_2"  # Country-specific
    TIER_3 = "tier_3"  # Facility-specific (highest quality)


class VersionStatus(str, Enum):
    """Version status for emission factors."""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


# =============================================================================
# CORE SCHEMA MODELS
# =============================================================================

class EmissionFactorSource(BaseModel):
    """Source information for emission factor."""
    source_type: DataSourceType
    source_name: str = Field(..., description="Full name of data source")
    source_url: Optional[str] = Field(None, description="URL to source document")
    publication_date: Optional[date] = Field(None, description="Publication date")
    access_date: date = Field(default_factory=date.today, description="Date accessed")
    version: Optional[str] = Field(None, description="Source version")
    page_reference: Optional[str] = Field(None, description="Page or section reference")
    methodology: Optional[str] = Field(None, description="Calculation methodology")

    class Config:
        use_enum_values = True


class EmissionFactorUncertainty(BaseModel):
    """Uncertainty quantification for emission factor."""
    uncertainty_type: str = Field(..., description="statistical, expert_judgment, default")
    min_value: Optional[Decimal] = Field(None, description="Minimum bound")
    max_value: Optional[Decimal] = Field(None, description="Maximum bound")
    mean_value: Optional[Decimal] = Field(None, description="Mean value")
    std_deviation: Optional[Decimal] = Field(None, description="Standard deviation")
    confidence_level: Optional[float] = Field(None, ge=0, le=100, description="Confidence %")
    distribution_type: Optional[str] = Field(None, description="normal, lognormal, uniform")
    sample_size: Optional[int] = Field(None, ge=0, description="Sample size if measured")

    @validator('confidence_level')
    def validate_confidence_level(cls, v):
        if v is not None and not (0 <= v <= 100):
            raise ValueError("confidence_level must be between 0 and 100")
        return v


class EmissionFactorQuality(BaseModel):
    """Data quality indicators (DQI) for emission factor."""
    quality_tier: QualityTier = Field(default=QualityTier.TIER_1)
    reliability_score: int = Field(ge=1, le=5, description="1=verified, 5=non-qualified estimate")
    completeness_score: int = Field(ge=1, le=5, description="1=complete, 5=unknown")
    temporal_score: int = Field(ge=1, le=5, description="1=recent, 5=>15 years old")
    geographic_score: int = Field(ge=1, le=5, description="1=exact match, 5=global default")
    technology_score: int = Field(ge=1, le=5, description="1=exact technology, 5=generic")
    aggregate_dqi: Optional[float] = Field(None, ge=0, le=100, description="Aggregate DQI score")

    def calculate_aggregate_dqi(self) -> float:
        """Calculate aggregate DQI score (0-100, higher is better)."""
        # Invert scores (1=best becomes 5, 5=worst becomes 1)
        inverted = [6 - s for s in [
            self.reliability_score,
            self.completeness_score,
            self.temporal_score,
            self.geographic_score,
            self.technology_score,
        ]]
        # Calculate percentage
        self.aggregate_dqi = (sum(inverted) / 25) * 100
        return self.aggregate_dqi

    class Config:
        use_enum_values = True


class EmissionFactorVersion(BaseModel):
    """Version tracking for emission factor."""
    version_id: str = Field(..., description="Semantic version (e.g., 1.0.0)")
    previous_version_id: Optional[str] = Field(None, description="Previous version")
    status: VersionStatus = Field(default=VersionStatus.ACTIVE)
    effective_from: date = Field(..., description="Version effective from date")
    effective_to: Optional[date] = Field(None, description="Version effective to date")
    change_reason: Optional[str] = Field(None, description="Reason for version change")
    change_magnitude: Optional[float] = Field(None, description="% change from previous")
    approved_by: Optional[str] = Field(None, description="Approver user ID")
    approved_at: Optional[datetime] = Field(None, description="Approval timestamp")

    @root_validator
    def validate_dates(cls, values):
        eff_from = values.get('effective_from')
        eff_to = values.get('effective_to')
        if eff_from and eff_to and eff_to < eff_from:
            raise ValueError("effective_to must be after effective_from")
        return values

    class Config:
        use_enum_values = True


class EmissionFactorSchema(BaseModel):
    """
    Comprehensive emission factor schema supporting 100K+ factors.

    Designed for:
    - Multi-industry coverage (steel, cement, aluminum, etc.)
    - Multi-regional support (200+ countries)
    - Full versioning and audit trail
    - Data quality scoring (DQI)
    - Uncertainty quantification
    - Regulatory compliance (CBAM, CSRD, EUDR)
    """

    # Primary identifiers
    factor_id: str = Field(..., description="Unique factor identifier (UUID)")
    factor_hash: Optional[str] = Field(None, description="Content hash for deduplication")

    # Classification
    industry: IndustryCategory = Field(..., description="Industry category")
    product_code: Optional[str] = Field(None, description="Product code (CN, HS, NACE)")
    product_name: str = Field(..., description="Product/material name")
    product_subcategory: Optional[str] = Field(None, description="Product subcategory")
    production_route: Optional[str] = Field(None, description="Production route/process")

    # Geographic
    region: GeographicRegion = Field(..., description="Geographic region")
    country_code: Optional[str] = Field(None, max_length=3, description="ISO 3166-1 alpha-3")
    state_province: Optional[str] = Field(None, description="State/province code")
    facility_id: Optional[str] = Field(None, description="Facility identifier")

    # Emission factor values
    ghg_type: GHGType = Field(default=GHGType.CO2E, description="GHG type")
    scope_type: ScopeType = Field(..., description="GHG Protocol scope")
    factor_value: Decimal = Field(..., ge=0, description="Emission factor value")
    factor_unit: str = Field(..., description="Unit (e.g., kgCO2e/kg, kgCO2e/kWh)")
    input_unit: str = Field(..., description="Input unit for calculation")
    output_unit: str = Field(default="kgCO2e", description="Output unit")

    # GWP values
    gwp_source: str = Field(default="IPCC AR6", description="GWP source")
    gwp_timeframe: int = Field(default=100, description="GWP timeframe (years)")
    gwp_value: Optional[Decimal] = Field(None, description="GWP value if not CO2e")

    # Temporal
    reference_year: int = Field(..., ge=1990, le=2050, description="Reference year")
    valid_from: date = Field(..., description="Valid from date")
    valid_to: Optional[date] = Field(None, description="Valid to date")

    # Quality and uncertainty
    quality: EmissionFactorQuality = Field(default_factory=EmissionFactorQuality)
    uncertainty: Optional[EmissionFactorUncertainty] = None

    # Source and versioning
    source: EmissionFactorSource = Field(..., description="Data source information")
    version: EmissionFactorVersion = Field(..., description="Version information")

    # Regulatory flags
    cbam_eligible: bool = Field(default=False, description="CBAM eligible factor")
    csrd_compliant: bool = Field(default=False, description="CSRD compliant factor")
    iso14064_compliant: bool = Field(default=False, description="ISO 14064 compliant")
    ghg_protocol_compliant: bool = Field(default=True, description="GHG Protocol compliant")

    # Audit trail
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="Creator user ID")
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = Field(None, description="Updater user ID")

    # Additional metadata
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    notes: Optional[str] = Field(None, description="Additional notes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extended metadata")

    def __init__(self, **data):
        super().__init__(**data)
        self.calculate_hash()
        if self.quality:
            self.quality.calculate_aggregate_dqi()

    def calculate_hash(self) -> str:
        """Calculate content hash for deduplication."""
        content = f"{self.industry}:{self.product_code}:{self.region}:{self.country_code}:"
        content += f"{self.ghg_type}:{self.scope_type}:{self.factor_value}:{self.reference_year}"
        self.factor_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self.factor_hash

    def to_calculation_input(self) -> Dict[str, Any]:
        """Convert to calculation engine input format."""
        return {
            "factor_id": self.factor_id,
            "factor_value": float(self.factor_value),
            "factor_unit": self.factor_unit,
            "input_unit": self.input_unit,
            "output_unit": self.output_unit,
            "ghg_type": self.ghg_type,
            "scope_type": self.scope_type,
            "uncertainty": {
                "min": float(self.uncertainty.min_value) if self.uncertainty and self.uncertainty.min_value else None,
                "max": float(self.uncertainty.max_value) if self.uncertainty and self.uncertainty.max_value else None,
            } if self.uncertainty else None,
            "quality_score": self.quality.aggregate_dqi if self.quality else None,
        }

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            Decimal: lambda v: float(v),
        }


# =============================================================================
# SQL DDL GENERATION
# =============================================================================

EMISSION_FACTOR_DDL = """
-- =============================================================================
-- GreenLang Emission Factor Database Schema
-- Version: 1.0.0
-- Target: 100,000+ emission factors
-- Database: PostgreSQL 14+
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For fuzzy text search

-- =============================================================================
-- ENUMERATION TYPES
-- =============================================================================

CREATE TYPE ghg_type AS ENUM (
    'CO2', 'CH4', 'N2O', 'HFCs', 'PFCs', 'SF6', 'NF3', 'CO2e'
);

CREATE TYPE scope_type AS ENUM (
    'scope_1', 'scope_2_location', 'scope_2_market', 'scope_3', 'well_to_tank'
);

CREATE TYPE industry_category AS ENUM (
    'steel', 'cement', 'aluminum', 'chemicals', 'fertilizer', 'hydrogen',
    'electricity', 'automotive', 'aviation', 'shipping', 'rail', 'road_freight',
    'agriculture', 'textiles', 'electronics', 'construction', 'waste',
    'food_beverage', 'paper_pulp', 'glass', 'general'
);

CREATE TYPE geographic_region AS ENUM (
    'global', 'north_america', 'europe', 'asia_pacific', 'latin_america',
    'middle_east', 'africa', 'usa', 'canada', 'mexico', 'uk', 'germany',
    'france', 'italy', 'spain', 'china', 'india', 'japan', 'south_korea',
    'australia', 'brazil', 'russia', 'turkey'
);

CREATE TYPE data_source_type AS ENUM (
    'defra', 'epa_egrid', 'epa_ghg', 'ipcc_ar6', 'ipcc_ar7', 'ecoinvent',
    'world_bank', 'iea', 'fao', 'customer_provided', 'calculated'
);

CREATE TYPE quality_tier AS ENUM ('tier_1', 'tier_2', 'tier_3');

CREATE TYPE version_status AS ENUM ('draft', 'active', 'deprecated', 'archived');

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Emission Factor Sources (dimension table)
CREATE TABLE emission_factor_sources (
    source_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type data_source_type NOT NULL,
    source_name VARCHAR(255) NOT NULL,
    source_url TEXT,
    publication_date DATE,
    access_date DATE NOT NULL DEFAULT CURRENT_DATE,
    version VARCHAR(50),
    page_reference VARCHAR(255),
    methodology TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_sources_type ON emission_factor_sources(source_type);
CREATE INDEX idx_sources_name ON emission_factor_sources USING gin(source_name gin_trgm_ops);

-- Emission Factor Versions (SCD Type 2 tracking)
CREATE TABLE emission_factor_versions (
    version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    factor_id UUID NOT NULL,
    version_number VARCHAR(20) NOT NULL,
    previous_version_id UUID REFERENCES emission_factor_versions(version_id),
    status version_status NOT NULL DEFAULT 'active',
    effective_from DATE NOT NULL,
    effective_to DATE,
    change_reason TEXT,
    change_magnitude DECIMAL(10,4),
    approved_by VARCHAR(255),
    approved_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_versions_factor ON emission_factor_versions(factor_id);
CREATE INDEX idx_versions_status ON emission_factor_versions(status);
CREATE INDEX idx_versions_effective ON emission_factor_versions(effective_from, effective_to);

-- Emission Factor Quality (dimension table)
CREATE TABLE emission_factor_quality (
    quality_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    factor_id UUID NOT NULL,
    quality_tier quality_tier NOT NULL DEFAULT 'tier_1',
    reliability_score SMALLINT NOT NULL CHECK (reliability_score BETWEEN 1 AND 5),
    completeness_score SMALLINT NOT NULL CHECK (completeness_score BETWEEN 1 AND 5),
    temporal_score SMALLINT NOT NULL CHECK (temporal_score BETWEEN 1 AND 5),
    geographic_score SMALLINT NOT NULL CHECK (geographic_score BETWEEN 1 AND 5),
    technology_score SMALLINT NOT NULL CHECK (technology_score BETWEEN 1 AND 5),
    aggregate_dqi DECIMAL(5,2) GENERATED ALWAYS AS (
        ((6 - reliability_score) + (6 - completeness_score) + (6 - temporal_score) +
         (6 - geographic_score) + (6 - technology_score)) / 25.0 * 100
    ) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_quality_factor ON emission_factor_quality(factor_id);
CREATE INDEX idx_quality_tier ON emission_factor_quality(quality_tier);
CREATE INDEX idx_quality_dqi ON emission_factor_quality(aggregate_dqi);

-- Emission Factor Uncertainty
CREATE TABLE emission_factor_uncertainty (
    uncertainty_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    factor_id UUID NOT NULL,
    uncertainty_type VARCHAR(50) NOT NULL,
    min_value DECIMAL(20,10),
    max_value DECIMAL(20,10),
    mean_value DECIMAL(20,10),
    std_deviation DECIMAL(20,10),
    confidence_level DECIMAL(5,2) CHECK (confidence_level BETWEEN 0 AND 100),
    distribution_type VARCHAR(50),
    sample_size INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_uncertainty_factor ON emission_factor_uncertainty(factor_id);

-- Main Emission Factors Table (fact table)
CREATE TABLE emission_factors (
    factor_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    factor_hash VARCHAR(16) NOT NULL,

    -- Classification
    industry industry_category NOT NULL,
    product_code VARCHAR(20),
    product_name VARCHAR(500) NOT NULL,
    product_subcategory VARCHAR(255),
    production_route VARCHAR(255),

    -- Geographic
    region geographic_region NOT NULL,
    country_code CHAR(3),
    state_province VARCHAR(100),
    facility_id VARCHAR(100),

    -- Emission factor values
    ghg_type ghg_type NOT NULL DEFAULT 'CO2e',
    scope_type scope_type NOT NULL,
    factor_value DECIMAL(20,10) NOT NULL CHECK (factor_value >= 0),
    factor_unit VARCHAR(50) NOT NULL,
    input_unit VARCHAR(50) NOT NULL,
    output_unit VARCHAR(50) NOT NULL DEFAULT 'kgCO2e',

    -- GWP
    gwp_source VARCHAR(50) DEFAULT 'IPCC AR6',
    gwp_timeframe SMALLINT DEFAULT 100,
    gwp_value DECIMAL(10,4),

    -- Temporal
    reference_year SMALLINT NOT NULL CHECK (reference_year BETWEEN 1990 AND 2050),
    valid_from DATE NOT NULL,
    valid_to DATE,

    -- Foreign keys
    source_id UUID REFERENCES emission_factor_sources(source_id),
    current_version_id UUID,
    quality_id UUID,
    uncertainty_id UUID,

    -- Regulatory flags
    cbam_eligible BOOLEAN DEFAULT FALSE,
    csrd_compliant BOOLEAN DEFAULT FALSE,
    iso14064_compliant BOOLEAN DEFAULT FALSE,
    ghg_protocol_compliant BOOLEAN DEFAULT TRUE,

    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_by VARCHAR(255),

    -- Additional
    tags TEXT[],
    notes TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Constraints
    UNIQUE (factor_hash),
    CHECK (valid_to IS NULL OR valid_to > valid_from)
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Primary lookup indexes
CREATE INDEX idx_ef_industry ON emission_factors(industry);
CREATE INDEX idx_ef_region ON emission_factors(region);
CREATE INDEX idx_ef_country ON emission_factors(country_code);
CREATE INDEX idx_ef_product_code ON emission_factors(product_code);
CREATE INDEX idx_ef_scope ON emission_factors(scope_type);
CREATE INDEX idx_ef_ghg ON emission_factors(ghg_type);
CREATE INDEX idx_ef_year ON emission_factors(reference_year);

-- Composite indexes for common queries
CREATE INDEX idx_ef_industry_region ON emission_factors(industry, region);
CREATE INDEX idx_ef_industry_country ON emission_factors(industry, country_code);
CREATE INDEX idx_ef_product_region ON emission_factors(product_code, region);
CREATE INDEX idx_ef_product_country ON emission_factors(product_code, country_code);
CREATE INDEX idx_ef_product_year ON emission_factors(product_code, reference_year);

-- Regulatory compliance indexes
CREATE INDEX idx_ef_cbam ON emission_factors(cbam_eligible) WHERE cbam_eligible = TRUE;
CREATE INDEX idx_ef_csrd ON emission_factors(csrd_compliant) WHERE csrd_compliant = TRUE;

-- Text search indexes
CREATE INDEX idx_ef_product_name ON emission_factors USING gin(product_name gin_trgm_ops);
CREATE INDEX idx_ef_tags ON emission_factors USING gin(tags);
CREATE INDEX idx_ef_metadata ON emission_factors USING gin(metadata);

-- Temporal indexes
CREATE INDEX idx_ef_valid_range ON emission_factors(valid_from, valid_to);
CREATE INDEX idx_ef_created ON emission_factors(created_at);

-- Hash index for deduplication
CREATE INDEX idx_ef_hash ON emission_factors USING hash(factor_hash);

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- Active emission factors view
CREATE OR REPLACE VIEW v_active_emission_factors AS
SELECT
    ef.*,
    efq.quality_tier,
    efq.aggregate_dqi,
    efs.source_name,
    efs.publication_date
FROM emission_factors ef
LEFT JOIN emission_factor_quality efq ON ef.quality_id = efq.quality_id
LEFT JOIN emission_factor_sources efs ON ef.source_id = efs.source_id
WHERE ef.valid_to IS NULL OR ef.valid_to > CURRENT_DATE;

-- CBAM emission factors view
CREATE OR REPLACE VIEW v_cbam_emission_factors AS
SELECT * FROM v_active_emission_factors
WHERE cbam_eligible = TRUE
AND industry IN ('steel', 'cement', 'aluminum', 'fertilizer', 'hydrogen', 'electricity');

-- Regional grid factors view
CREATE OR REPLACE VIEW v_grid_factors AS
SELECT * FROM v_active_emission_factors
WHERE industry = 'electricity'
AND scope_type = 'scope_2_location'
ORDER BY region, country_code, reference_year DESC;

-- High quality factors view (DQI > 70)
CREATE OR REPLACE VIEW v_high_quality_factors AS
SELECT ef.*, efq.aggregate_dqi
FROM emission_factors ef
JOIN emission_factor_quality efq ON ef.quality_id = efq.quality_id
WHERE efq.aggregate_dqi >= 70
AND (ef.valid_to IS NULL OR ef.valid_to > CURRENT_DATE);

-- =============================================================================
-- FUNCTIONS AND TRIGGERS
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for emission_factors
CREATE TRIGGER trg_ef_updated_at
BEFORE UPDATE ON emission_factors
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Trigger for sources
CREATE TRIGGER trg_sources_updated_at
BEFORE UPDATE ON emission_factor_sources
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate factor hash
CREATE OR REPLACE FUNCTION calculate_factor_hash(
    p_industry industry_category,
    p_product_code VARCHAR,
    p_region geographic_region,
    p_country_code VARCHAR,
    p_ghg_type ghg_type,
    p_scope_type scope_type,
    p_factor_value DECIMAL,
    p_reference_year SMALLINT
) RETURNS VARCHAR AS $$
BEGIN
    RETURN LEFT(
        encode(
            sha256(
                (p_industry || ':' || COALESCE(p_product_code, '') || ':' || p_region || ':' ||
                 COALESCE(p_country_code, '') || ':' || p_ghg_type || ':' || p_scope_type || ':' ||
                 p_factor_value::TEXT || ':' || p_reference_year::TEXT)::bytea
            ),
            'hex'
        ),
        16
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to look up emission factor
CREATE OR REPLACE FUNCTION lookup_emission_factor(
    p_product_code VARCHAR,
    p_country_code VARCHAR DEFAULT NULL,
    p_region geographic_region DEFAULT NULL,
    p_reference_year SMALLINT DEFAULT EXTRACT(YEAR FROM CURRENT_DATE)::SMALLINT
) RETURNS TABLE (
    factor_id UUID,
    factor_value DECIMAL,
    factor_unit VARCHAR,
    quality_score DECIMAL,
    source_name VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ef.factor_id,
        ef.factor_value,
        ef.factor_unit,
        efq.aggregate_dqi,
        efs.source_name
    FROM emission_factors ef
    LEFT JOIN emission_factor_quality efq ON ef.quality_id = efq.quality_id
    LEFT JOIN emission_factor_sources efs ON ef.source_id = efs.source_id
    WHERE ef.product_code = p_product_code
    AND (p_country_code IS NULL OR ef.country_code = p_country_code)
    AND (p_region IS NULL OR ef.region = p_region)
    AND ef.reference_year <= p_reference_year
    AND (ef.valid_to IS NULL OR ef.valid_to > CURRENT_DATE)
    ORDER BY
        CASE WHEN ef.country_code = p_country_code THEN 0 ELSE 1 END,
        ef.reference_year DESC,
        efq.aggregate_dqi DESC NULLS LAST
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- AUDIT TRAIL TABLE
-- =============================================================================

CREATE TABLE emission_factor_audit (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    factor_id UUID NOT NULL,
    operation VARCHAR(10) NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values JSONB,
    new_values JSONB,
    changed_fields TEXT[],
    changed_by VARCHAR(255),
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    session_id VARCHAR(255),
    change_reason TEXT
);

CREATE INDEX idx_audit_factor ON emission_factor_audit(factor_id);
CREATE INDEX idx_audit_operation ON emission_factor_audit(operation);
CREATE INDEX idx_audit_changed_at ON emission_factor_audit(changed_at);
CREATE INDEX idx_audit_changed_by ON emission_factor_audit(changed_by);

-- Audit trigger function
CREATE OR REPLACE FUNCTION audit_emission_factor_changes()
RETURNS TRIGGER AS $$
DECLARE
    v_old_values JSONB;
    v_new_values JSONB;
    v_changed_fields TEXT[];
BEGIN
    IF TG_OP = 'DELETE' THEN
        v_old_values = to_jsonb(OLD);
        INSERT INTO emission_factor_audit (factor_id, operation, old_values, changed_by)
        VALUES (OLD.factor_id, 'DELETE', v_old_values, current_user);
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        v_old_values = to_jsonb(OLD);
        v_new_values = to_jsonb(NEW);

        -- Get changed fields
        SELECT array_agg(key)
        INTO v_changed_fields
        FROM jsonb_each(v_new_values)
        WHERE v_new_values->key IS DISTINCT FROM v_old_values->key;

        INSERT INTO emission_factor_audit (factor_id, operation, old_values, new_values, changed_fields, changed_by)
        VALUES (NEW.factor_id, 'UPDATE', v_old_values, v_new_values, v_changed_fields, NEW.updated_by);
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        v_new_values = to_jsonb(NEW);
        INSERT INTO emission_factor_audit (factor_id, operation, new_values, changed_by)
        VALUES (NEW.factor_id, 'INSERT', v_new_values, NEW.created_by);
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_audit_emission_factors
AFTER INSERT OR UPDATE OR DELETE ON emission_factors
FOR EACH ROW
EXECUTE FUNCTION audit_emission_factor_changes();

-- =============================================================================
-- PARTITIONING (for scale to 100K+ factors)
-- =============================================================================

-- Note: For production with 100K+ factors, consider partitioning by:
-- 1. industry (list partitioning)
-- 2. reference_year (range partitioning)

-- Example partitioned table (uncomment for production):
/*
CREATE TABLE emission_factors_partitioned (
    LIKE emission_factors INCLUDING ALL
) PARTITION BY LIST (industry);

CREATE TABLE ef_part_steel PARTITION OF emission_factors_partitioned FOR VALUES IN ('steel');
CREATE TABLE ef_part_cement PARTITION OF emission_factors_partitioned FOR VALUES IN ('cement');
CREATE TABLE ef_part_aluminum PARTITION OF emission_factors_partitioned FOR VALUES IN ('aluminum');
CREATE TABLE ef_part_electricity PARTITION OF emission_factors_partitioned FOR VALUES IN ('electricity');
-- ... more partitions
*/

-- =============================================================================
-- STATISTICS FOR QUERY OPTIMIZER
-- =============================================================================

-- Analyze tables after bulk load
-- ANALYZE emission_factors;
-- ANALYZE emission_factor_sources;
-- ANALYZE emission_factor_quality;
-- ANALYZE emission_factor_versions;
-- ANALYZE emission_factor_uncertainty;

-- =============================================================================
-- GRANTS (adjust roles as needed)
-- =============================================================================

-- Read-only role
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO greenlang_readonly;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO greenlang_readonly;

-- Read-write role
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO greenlang_readwrite;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO greenlang_readwrite;

-- Admin role
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO greenlang_admin;
-- GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO greenlang_admin;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO greenlang_admin;
"""


def get_ddl() -> str:
    """Return the full DDL for emission factor database."""
    return EMISSION_FACTOR_DDL


def generate_create_table_sqlite() -> str:
    """Generate SQLite-compatible DDL (simplified)."""
    return """
-- SQLite Schema for Emission Factors (Simplified)
-- For development/testing - use PostgreSQL for production

CREATE TABLE IF NOT EXISTS emission_factor_sources (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_url TEXT,
    publication_date TEXT,
    access_date TEXT NOT NULL,
    version TEXT,
    methodology TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS emission_factors (
    factor_id TEXT PRIMARY KEY,
    factor_hash TEXT NOT NULL UNIQUE,
    industry TEXT NOT NULL,
    product_code TEXT,
    product_name TEXT NOT NULL,
    product_subcategory TEXT,
    production_route TEXT,
    region TEXT NOT NULL,
    country_code TEXT,
    state_province TEXT,
    ghg_type TEXT NOT NULL DEFAULT 'CO2e',
    scope_type TEXT NOT NULL,
    factor_value REAL NOT NULL CHECK (factor_value >= 0),
    factor_unit TEXT NOT NULL,
    input_unit TEXT NOT NULL,
    output_unit TEXT NOT NULL DEFAULT 'kgCO2e',
    gwp_source TEXT DEFAULT 'IPCC AR6',
    gwp_timeframe INTEGER DEFAULT 100,
    reference_year INTEGER NOT NULL CHECK (reference_year BETWEEN 1990 AND 2050),
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    source_id TEXT REFERENCES emission_factor_sources(source_id),
    quality_tier TEXT DEFAULT 'tier_1',
    aggregate_dqi REAL,
    cbam_eligible INTEGER DEFAULT 0,
    csrd_compliant INTEGER DEFAULT 0,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    tags TEXT,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_ef_industry ON emission_factors(industry);
CREATE INDEX IF NOT EXISTS idx_ef_region ON emission_factors(region);
CREATE INDEX IF NOT EXISTS idx_ef_country ON emission_factors(country_code);
CREATE INDEX IF NOT EXISTS idx_ef_product_code ON emission_factors(product_code);
CREATE INDEX IF NOT EXISTS idx_ef_year ON emission_factors(reference_year);
CREATE INDEX IF NOT EXISTS idx_ef_cbam ON emission_factors(cbam_eligible);
"""
