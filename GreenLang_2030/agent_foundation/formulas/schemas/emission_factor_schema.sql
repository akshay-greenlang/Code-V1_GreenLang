-- GreenLang Emission Factor Database Schema
-- Supports 100K+ factors from DEFRA, EPA, Ecoinvent, IPCC
-- Zero-hallucination guarantee with complete provenance tracking

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- Enum types for controlled vocabularies
CREATE TYPE emission_source AS ENUM (
    'DEFRA_2023',
    'DEFRA_2024',
    'EPA_EGRID_2023',
    'ECOINVENT_3.9.1',
    'IPCC_AR6',
    'IEA_2023',
    'GABI_2023',
    'CUSTOM'
);

CREATE TYPE emission_category AS ENUM (
    'stationary_combustion',
    'mobile_combustion',
    'electricity',
    'fugitive_emissions',
    'industrial_processes',
    'agriculture',
    'waste',
    'purchased_goods',
    'transportation',
    'employee_commuting',
    'business_travel',
    'investments'
);

CREATE TYPE emission_scope AS ENUM (
    'scope_1',
    'scope_2_location',
    'scope_2_market',
    'scope_3_cat1',
    'scope_3_cat2',
    'scope_3_cat3',
    'scope_3_cat4',
    'scope_3_cat5',
    'scope_3_cat6',
    'scope_3_cat7',
    'scope_3_cat8',
    'scope_3_cat9',
    'scope_3_cat10',
    'scope_3_cat11',
    'scope_3_cat12',
    'scope_3_cat13',
    'scope_3_cat14',
    'scope_3_cat15'
);

CREATE TYPE methodology AS ENUM (
    'GHG_PROTOCOL',
    'ISO_14064',
    'ISO_14067',
    'PAS_2050',
    'CDP',
    'TCFD',
    'SBTI'
);

-- Main emission factors table
CREATE TABLE emission_factors (
    -- Primary identification
    factor_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id VARCHAR(255) UNIQUE, -- Original ID from source

    -- Source and versioning
    source emission_source NOT NULL,
    source_version VARCHAR(50) NOT NULL,
    source_document VARCHAR(500),

    -- Factor classification
    category emission_category NOT NULL,
    subcategory VARCHAR(255) NOT NULL,
    scope emission_scope NOT NULL,

    -- Factor details
    name VARCHAR(500) NOT NULL,
    description TEXT,

    -- Emission factor values
    factor_value DECIMAL(20, 10) NOT NULL,
    factor_unit VARCHAR(100) NOT NULL,

    -- Conversion to standard units (kg CO2e)
    co2e_factor DECIMAL(20, 10) NOT NULL, -- Normalized to kg CO2e
    co2e_unit VARCHAR(100) NOT NULL DEFAULT 'kg CO2e',

    -- Component gases (for detailed reporting)
    co2_factor DECIMAL(20, 10),
    ch4_factor DECIMAL(20, 10),
    n2o_factor DECIMAL(20, 10),

    -- Geographic scope
    country_code VARCHAR(3), -- ISO 3166-1 alpha-3
    region VARCHAR(255),
    city VARCHAR(255),
    coordinates POINT, -- For location-specific factors

    -- Temporal validity
    valid_from DATE NOT NULL,
    valid_until DATE,
    reporting_year INTEGER NOT NULL,

    -- Uncertainty and quality
    uncertainty_percentage DECIMAL(5, 2), -- ± percentage
    quality_rating INTEGER CHECK (quality_rating BETWEEN 1 AND 5),
    data_quality_flags JSONB,

    -- Methodology
    methodology methodology NOT NULL,
    calculation_method TEXT,
    assumptions TEXT,

    -- Additional attributes
    tags TEXT[],
    metadata JSONB,

    -- Audit trail
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    updated_by VARCHAR(255),

    -- Provenance hash for reproducibility
    provenance_hash VARCHAR(64) GENERATED ALWAYS AS (
        encode(sha256(
            (factor_id::text ||
             source::text ||
             factor_value::text ||
             factor_unit::text)::bytea
        ), 'hex')
    ) STORED
);

-- Create indexes for fast lookups
CREATE INDEX idx_emission_factors_category ON emission_factors(category);
CREATE INDEX idx_emission_factors_subcategory ON emission_factors(subcategory);
CREATE INDEX idx_emission_factors_scope ON emission_factors(scope);
CREATE INDEX idx_emission_factors_country ON emission_factors(country_code);
CREATE INDEX idx_emission_factors_region ON emission_factors(region);
CREATE INDEX idx_emission_factors_valid_dates ON emission_factors(valid_from, valid_until);
CREATE INDEX idx_emission_factors_source ON emission_factors(source, source_version);
CREATE INDEX idx_emission_factors_tags ON emission_factors USING GIN(tags);
CREATE INDEX idx_emission_factors_metadata ON emission_factors USING GIN(metadata);

-- Spatial index for location-based queries
CREATE INDEX idx_emission_factors_location ON emission_factors USING GIST(coordinates);

-- Full-text search index
CREATE INDEX idx_emission_factors_search ON emission_factors USING GIN(
    to_tsvector('english', name || ' ' || COALESCE(description, ''))
);

-- Unit conversion table
CREATE TABLE unit_conversions (
    conversion_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_unit VARCHAR(100) NOT NULL,
    to_unit VARCHAR(100) NOT NULL,
    conversion_factor DECIMAL(20, 10) NOT NULL,
    category VARCHAR(100),
    notes TEXT,
    UNIQUE(from_unit, to_unit)
);

-- Factor relationships (for complex calculations)
CREATE TABLE factor_relationships (
    relationship_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_factor_id UUID REFERENCES emission_factors(factor_id),
    child_factor_id UUID REFERENCES emission_factors(factor_id),
    relationship_type VARCHAR(50), -- 'component', 'alternative', 'supersedes'
    weight DECIMAL(10, 5),
    notes TEXT
);

-- Audit log for factor changes
CREATE TABLE emission_factor_audit (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    factor_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL, -- 'INSERT', 'UPDATE', 'DELETE'
    old_values JSONB,
    new_values JSONB,
    changed_by VARCHAR(255),
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    change_reason TEXT
);

-- Cached calculations for performance
CREATE TABLE factor_cache (
    cache_key VARCHAR(255) PRIMARY KEY,
    factor_id UUID REFERENCES emission_factors(factor_id),
    calculation_params JSONB,
    result_value DECIMAL(20, 10),
    result_unit VARCHAR(100),
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    hit_count INTEGER DEFAULT 0
);

-- Create materialized view for common lookups
CREATE MATERIALIZED VIEW common_emission_factors AS
SELECT
    factor_id,
    source,
    category,
    subcategory,
    scope,
    name,
    factor_value,
    factor_unit,
    co2e_factor,
    country_code,
    region,
    valid_from,
    valid_until,
    uncertainty_percentage,
    provenance_hash
FROM emission_factors
WHERE valid_from <= CURRENT_DATE
  AND (valid_until IS NULL OR valid_until >= CURRENT_DATE)
  AND quality_rating >= 3
ORDER BY source, category, subcategory;

-- Create index on materialized view
CREATE INDEX idx_common_factors_lookup ON common_emission_factors(
    category, subcategory, country_code, region
);

-- Function to get the best available factor
CREATE OR REPLACE FUNCTION get_best_emission_factor(
    p_category emission_category,
    p_subcategory VARCHAR,
    p_country VARCHAR DEFAULT NULL,
    p_region VARCHAR DEFAULT NULL,
    p_date DATE DEFAULT CURRENT_DATE
)
RETURNS TABLE (
    factor_id UUID,
    factor_value DECIMAL,
    factor_unit VARCHAR,
    uncertainty DECIMAL,
    source emission_source,
    quality_rating INTEGER
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Try exact match first
    RETURN QUERY
    SELECT
        ef.factor_id,
        ef.co2e_factor,
        ef.co2e_unit::VARCHAR,
        ef.uncertainty_percentage,
        ef.source,
        ef.quality_rating
    FROM emission_factors ef
    WHERE ef.category = p_category
      AND ef.subcategory = p_subcategory
      AND (p_country IS NULL OR ef.country_code = p_country)
      AND (p_region IS NULL OR ef.region = p_region)
      AND ef.valid_from <= p_date
      AND (ef.valid_until IS NULL OR ef.valid_until >= p_date)
    ORDER BY
        ef.quality_rating DESC,
        ef.source_version DESC,
        ef.uncertainty_percentage ASC
    LIMIT 1;

    -- If no exact match, try regional average
    IF NOT FOUND AND p_country IS NOT NULL THEN
        RETURN QUERY
        SELECT
            ef.factor_id,
            ef.co2e_factor,
            ef.factor_unit::VARCHAR,
            ef.uncertainty_percentage,
            ef.source,
            ef.quality_rating
        FROM emission_factors ef
        WHERE ef.category = p_category
          AND ef.subcategory = p_subcategory
          AND ef.region = p_region
          AND ef.country_code IS NULL
          AND ef.valid_from <= p_date
          AND (ef.valid_until IS NULL OR ef.valid_until >= p_date)
        ORDER BY
            ef.quality_rating DESC,
            ef.source_version DESC
        LIMIT 1;
    END IF;

    -- If still no match, use global average
    IF NOT FOUND THEN
        RETURN QUERY
        SELECT
            ef.factor_id,
            ef.co2e_factor,
            ef.co2e_unit::VARCHAR,
            ef.uncertainty_percentage,
            ef.source,
            ef.quality_rating
        FROM emission_factors ef
        WHERE ef.category = p_category
          AND ef.subcategory = p_subcategory
          AND ef.country_code IS NULL
          AND ef.region IS NULL
          AND ef.valid_from <= p_date
          AND (ef.valid_until IS NULL OR ef.valid_until >= p_date)
        ORDER BY
            ef.quality_rating DESC,
            ef.source_version DESC
        LIMIT 1;
    END IF;
END;
$$;

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_emission_factors_updated_at
BEFORE UPDATE ON emission_factors
FOR EACH ROW
EXECUTE FUNCTION update_updated_at();

-- Audit trigger
CREATE OR REPLACE FUNCTION audit_emission_factor_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO emission_factor_audit(factor_id, action, new_values, changed_by)
        VALUES (NEW.factor_id, 'INSERT', row_to_json(NEW), NEW.created_by);
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO emission_factor_audit(factor_id, action, old_values, new_values, changed_by)
        VALUES (NEW.factor_id, 'UPDATE', row_to_json(OLD), row_to_json(NEW), NEW.updated_by);
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO emission_factor_audit(factor_id, action, old_values)
        VALUES (OLD.factor_id, 'DELETE', row_to_json(OLD));
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_emission_factors
AFTER INSERT OR UPDATE OR DELETE ON emission_factors
FOR EACH ROW
EXECUTE FUNCTION audit_emission_factor_changes();