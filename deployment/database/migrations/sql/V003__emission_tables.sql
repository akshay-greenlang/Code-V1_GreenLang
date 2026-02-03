-- =============================================================================
-- V003: Emission Tables
-- =============================================================================
-- Description: Creates emission measurement tables, emission factors,
--              calculation results, and continuous aggregates for GreenLang.
-- Author: GreenLang Data Integration Team
-- Requires: TimescaleDB (V002)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Emission Measurement Types
-- -----------------------------------------------------------------------------

-- GHG Protocol scopes
CREATE TYPE public.emission_scope AS ENUM (
    'scope1',           -- Direct emissions
    'scope2_location',  -- Indirect - location-based
    'scope2_market',    -- Indirect - market-based
    'scope3_upstream',  -- Value chain - upstream
    'scope3_downstream' -- Value chain - downstream
);

-- Emission categories (GHG Protocol aligned)
CREATE TYPE public.emission_category AS ENUM (
    -- Scope 1
    'stationary_combustion',
    'mobile_combustion',
    'fugitive_emissions',
    'process_emissions',

    -- Scope 2
    'purchased_electricity',
    'purchased_heat',
    'purchased_steam',
    'purchased_cooling',

    -- Scope 3 Upstream
    'purchased_goods_services',
    'capital_goods',
    'fuel_energy_activities',
    'upstream_transportation',
    'waste_generated',
    'business_travel',
    'employee_commuting',
    'upstream_leased_assets',

    -- Scope 3 Downstream
    'downstream_transportation',
    'processing_sold_products',
    'use_sold_products',
    'eol_sold_products',
    'downstream_leased_assets',
    'franchises',
    'investments'
);

-- Data quality indicators
CREATE TYPE public.data_quality_level AS ENUM (
    'primary_measured',    -- Direct measurement
    'primary_calculated',  -- Calculated from primary data
    'secondary_specific',  -- Industry-specific secondary data
    'secondary_average',   -- Industry average data
    'estimated',           -- Estimated/extrapolated
    'default'              -- Default factors used
);

-- Calculation methodology
CREATE TYPE public.calculation_method AS ENUM (
    'spend_based',
    'activity_based',
    'supplier_specific',
    'hybrid',
    'direct_measurement',
    'mass_balance',
    'engineering_estimate'
);

-- -----------------------------------------------------------------------------
-- Emission Factors Table
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS public.emission_factors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Factor Identification
    factor_name VARCHAR(255) NOT NULL,
    factor_code VARCHAR(100) NOT NULL UNIQUE,
    factor_version VARCHAR(20) DEFAULT '1.0',

    -- Classification
    scope emission_scope NOT NULL,
    category emission_category NOT NULL,

    -- Activity and Unit
    activity_type VARCHAR(100) NOT NULL,
    activity_unit VARCHAR(50) NOT NULL,  -- e.g., 'kWh', 'liter', 'kg', 'USD'

    -- GHG Emissions (kg CO2e per activity unit)
    co2_factor NUMERIC(20, 10),          -- kg CO2 per unit
    ch4_factor NUMERIC(20, 10),          -- kg CH4 per unit
    n2o_factor NUMERIC(20, 10),          -- kg N2O per unit
    hfc_factor NUMERIC(20, 10),          -- kg HFC per unit
    pfc_factor NUMERIC(20, 10),          -- kg PFC per unit
    sf6_factor NUMERIC(20, 10),          -- kg SF6 per unit
    nf3_factor NUMERIC(20, 10),          -- kg NF3 per unit

    -- Total CO2 equivalent (GWP-weighted)
    co2e_factor NUMERIC(20, 10) NOT NULL, -- kg CO2e per unit

    -- GWP Values Used
    gwp_co2 NUMERIC(10, 2) DEFAULT 1,
    gwp_ch4 NUMERIC(10, 2) DEFAULT 28,    -- AR5 value
    gwp_n2o NUMERIC(10, 2) DEFAULT 265,   -- AR5 value

    -- Source and Validity
    source_database VARCHAR(100) NOT NULL,  -- e.g., 'EPA', 'DEFRA', 'ecoinvent'
    source_reference TEXT,
    source_url VARCHAR(500),
    publication_year INTEGER,
    valid_from DATE NOT NULL,
    valid_to DATE,

    -- Geographic Scope
    country_code CHAR(2),                   -- ISO 3166-1 alpha-2
    region VARCHAR(100),
    geographic_scope VARCHAR(255),

    -- Industry/Sector
    sector_code VARCHAR(20),                -- NAICS or ISIC
    sector_name VARCHAR(255),

    -- Quality and Uncertainty
    data_quality_level data_quality_level DEFAULT 'secondary_average',
    uncertainty_pct NUMERIC(5, 2),          -- e.g., 10.00 for 10%
    uncertainty_min NUMERIC(20, 10),
    uncertainty_max NUMERIC(20, 10),

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    tags TEXT[],

    -- Audit Fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_by UUID,
    is_active BOOLEAN DEFAULT TRUE,

    -- Constraints
    CONSTRAINT emission_factors_validity_check
        CHECK (valid_to IS NULL OR valid_to >= valid_from),
    CONSTRAINT emission_factors_uncertainty_check
        CHECK (uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 100))
);

-- Indexes for emission factors
CREATE INDEX idx_emission_factors_scope_category
    ON public.emission_factors(scope, category) WHERE is_active;
CREATE INDEX idx_emission_factors_activity
    ON public.emission_factors(activity_type) WHERE is_active;
CREATE INDEX idx_emission_factors_source
    ON public.emission_factors(source_database, publication_year) WHERE is_active;
CREATE INDEX idx_emission_factors_geography
    ON public.emission_factors(country_code, region) WHERE is_active;
CREATE INDEX idx_emission_factors_validity
    ON public.emission_factors(valid_from, valid_to) WHERE is_active;
CREATE INDEX idx_emission_factors_tags
    ON public.emission_factors USING gin(tags);

-- -----------------------------------------------------------------------------
-- Emission Measurements Hypertable
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS metrics.emission_measurements (
    -- Time dimension (required for hypertable)
    timestamp TIMESTAMPTZ NOT NULL,

    -- Identifiers
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL,
    project_id UUID NOT NULL,
    facility_id UUID,
    asset_id UUID,

    -- Classification
    scope emission_scope NOT NULL,
    category emission_category NOT NULL,

    -- Activity Data
    activity_type VARCHAR(100) NOT NULL,
    activity_value NUMERIC(20, 6) NOT NULL,
    activity_unit VARCHAR(50) NOT NULL,

    -- Emission Factor Used
    emission_factor_id UUID REFERENCES public.emission_factors(id),
    emission_factor_value NUMERIC(20, 10) NOT NULL,
    emission_factor_source VARCHAR(100),

    -- Calculated Emissions (kg CO2e)
    co2_kg NUMERIC(20, 6),
    ch4_kg NUMERIC(20, 6),
    n2o_kg NUMERIC(20, 6),
    other_ghg_kg NUMERIC(20, 6),
    co2e_kg NUMERIC(20, 6) NOT NULL,

    -- Uncertainty
    uncertainty_pct NUMERIC(5, 2),
    co2e_min_kg NUMERIC(20, 6),
    co2e_max_kg NUMERIC(20, 6),

    -- Data Quality
    data_quality data_quality_level NOT NULL DEFAULT 'secondary_average',
    data_quality_score INTEGER CHECK (data_quality_score BETWEEN 0 AND 100),
    calculation_method calculation_method NOT NULL DEFAULT 'activity_based',

    -- Source Information
    source_type VARCHAR(50),  -- 'manual', 'api', 'sensor', 'invoice', 'estimate'
    source_reference VARCHAR(255),
    source_document_id UUID,

    -- Temporal Details
    reporting_period_start DATE,
    reporting_period_end DATE,

    -- Verification
    verified BOOLEAN DEFAULT FALSE,
    verified_at TIMESTAMPTZ,
    verified_by UUID,
    verification_notes TEXT,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,
    tags TEXT[],

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Primary key for hypertable (must include time column)
    PRIMARY KEY (timestamp, id, organization_id)
);

-- Convert to hypertable with space partitioning by organization
SELECT create_hypertable(
    'metrics.emission_measurements',
    'timestamp',
    'organization_id',
    4,  -- Number of space partitions
    chunk_time_interval => INTERVAL '${chunk_interval}',
    if_not_exists => TRUE
);

-- Enable compression
ALTER TABLE metrics.emission_measurements SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'organization_id, scope, category',
    timescaledb.compress_orderby = 'timestamp DESC, id'
);

-- Add compression policy (compress after 7 days)
SELECT add_compression_policy(
    'metrics.emission_measurements',
    INTERVAL '${compression_after_days} days',
    if_not_exists => TRUE
);

-- Add retention policy
SELECT add_retention_policy(
    'metrics.emission_measurements',
    INTERVAL '${retention_days} days',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- Calculation Results Hypertable
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS metrics.calculation_results (
    -- Time dimension
    timestamp TIMESTAMPTZ NOT NULL,

    -- Identifiers
    id UUID NOT NULL DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL,
    project_id UUID NOT NULL,
    calculation_run_id UUID NOT NULL,

    -- Aggregation Level
    aggregation_level VARCHAR(50) NOT NULL,  -- 'facility', 'project', 'organization', 'scope', 'category'
    aggregation_key VARCHAR(255),

    -- Scope and Category
    scope emission_scope,
    category emission_category,

    -- Calculated Totals (tonnes CO2e)
    total_co2e_tonnes NUMERIC(20, 6) NOT NULL,
    total_co2_tonnes NUMERIC(20, 6),
    total_ch4_tonnes NUMERIC(20, 6),
    total_n2o_tonnes NUMERIC(20, 6),
    total_other_ghg_tonnes NUMERIC(20, 6),

    -- Comparison to Baseline
    baseline_co2e_tonnes NUMERIC(20, 6),
    change_from_baseline_pct NUMERIC(10, 4),
    change_from_baseline_tonnes NUMERIC(20, 6),

    -- Intensity Metrics
    revenue NUMERIC(20, 2),
    revenue_currency CHAR(3) DEFAULT 'USD',
    intensity_per_revenue NUMERIC(20, 10),  -- tonnes CO2e per unit revenue

    headcount INTEGER,
    intensity_per_employee NUMERIC(20, 10), -- tonnes CO2e per employee

    production_volume NUMERIC(20, 6),
    production_unit VARCHAR(50),
    intensity_per_production NUMERIC(20, 10),

    -- Uncertainty and Quality
    data_quality_score NUMERIC(5, 2),
    coverage_pct NUMERIC(5, 2),             -- % of emissions covered
    uncertainty_pct NUMERIC(5, 2),

    -- Methodology
    calculation_method calculation_method,
    methodology_notes TEXT,

    -- Status
    status VARCHAR(50) DEFAULT 'draft',     -- 'draft', 'reviewed', 'approved', 'published'
    approved_by UUID,
    approved_at TIMESTAMPTZ,

    -- Metadata
    metadata JSONB DEFAULT '{}'::jsonb,

    -- Audit
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID,

    PRIMARY KEY (timestamp, id, organization_id)
);

-- Convert to hypertable
SELECT create_hypertable(
    'metrics.calculation_results',
    'timestamp',
    'organization_id',
    4,
    chunk_time_interval => INTERVAL '${chunk_interval}',
    if_not_exists => TRUE
);

-- Enable compression
ALTER TABLE metrics.calculation_results SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'organization_id, aggregation_level',
    timescaledb.compress_orderby = 'timestamp DESC, id'
);

SELECT add_compression_policy(
    'metrics.calculation_results',
    INTERVAL '${compression_after_days} days',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- Indexes for Emission Measurements
-- -----------------------------------------------------------------------------

CREATE INDEX idx_emission_measurements_org_time
    ON metrics.emission_measurements (organization_id, timestamp DESC);

CREATE INDEX idx_emission_measurements_project_time
    ON metrics.emission_measurements (project_id, timestamp DESC);

CREATE INDEX idx_emission_measurements_scope_category
    ON metrics.emission_measurements (scope, category, timestamp DESC);

CREATE INDEX idx_emission_measurements_facility
    ON metrics.emission_measurements (facility_id, timestamp DESC)
    WHERE facility_id IS NOT NULL;

CREATE INDEX idx_emission_measurements_quality
    ON metrics.emission_measurements (data_quality, timestamp DESC);

CREATE INDEX idx_emission_measurements_tags
    ON metrics.emission_measurements USING gin(tags);

CREATE INDEX idx_emission_measurements_metadata
    ON metrics.emission_measurements USING gin(metadata jsonb_path_ops);

-- -----------------------------------------------------------------------------
-- Indexes for Calculation Results
-- -----------------------------------------------------------------------------

CREATE INDEX idx_calculation_results_org_time
    ON metrics.calculation_results (organization_id, timestamp DESC);

CREATE INDEX idx_calculation_results_project
    ON metrics.calculation_results (project_id, timestamp DESC);

CREATE INDEX idx_calculation_results_scope
    ON metrics.calculation_results (scope, timestamp DESC);

CREATE INDEX idx_calculation_results_aggregation
    ON metrics.calculation_results (aggregation_level, timestamp DESC);

CREATE INDEX idx_calculation_results_run
    ON metrics.calculation_results (calculation_run_id);

-- -----------------------------------------------------------------------------
-- Continuous Aggregates
-- -----------------------------------------------------------------------------

-- Daily emissions by organization and scope
CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.daily_emissions_by_scope
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS bucket,
    organization_id,
    scope,
    COUNT(*) AS measurement_count,
    SUM(co2e_kg) AS total_co2e_kg,
    SUM(co2_kg) AS total_co2_kg,
    SUM(ch4_kg) AS total_ch4_kg,
    SUM(n2o_kg) AS total_n2o_kg,
    AVG(data_quality_score) AS avg_quality_score,
    COUNT(DISTINCT facility_id) AS facility_count
FROM metrics.emission_measurements
GROUP BY time_bucket('1 day', timestamp), organization_id, scope
WITH NO DATA;

-- Add refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy(
    'metrics.daily_emissions_by_scope',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Monthly emissions by organization, scope, and category
CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.monthly_emissions_by_category
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 month', timestamp) AS bucket,
    organization_id,
    project_id,
    scope,
    category,
    COUNT(*) AS measurement_count,
    SUM(co2e_kg) / 1000 AS total_co2e_tonnes,  -- Convert to tonnes
    SUM(co2_kg) / 1000 AS total_co2_tonnes,
    SUM(ch4_kg) / 1000 AS total_ch4_tonnes,
    SUM(n2o_kg) / 1000 AS total_n2o_tonnes,
    AVG(data_quality_score) AS avg_quality_score,
    MIN(data_quality) AS min_quality_level,
    COUNT(DISTINCT source_type) AS source_type_count,
    SUM(CASE WHEN verified THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS verification_rate
FROM metrics.emission_measurements
GROUP BY time_bucket('1 month', timestamp), organization_id, project_id, scope, category
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'metrics.monthly_emissions_by_category',
    start_offset => INTERVAL '3 months',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Yearly emissions summary
CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.yearly_emissions_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 year', timestamp) AS bucket,
    organization_id,
    scope,
    SUM(co2e_kg) / 1000 AS total_co2e_tonnes,
    SUM(CASE WHEN scope = 'scope1' THEN co2e_kg ELSE 0 END) / 1000 AS scope1_tonnes,
    SUM(CASE WHEN scope IN ('scope2_location', 'scope2_market') THEN co2e_kg ELSE 0 END) / 1000 AS scope2_tonnes,
    SUM(CASE WHEN scope IN ('scope3_upstream', 'scope3_downstream') THEN co2e_kg ELSE 0 END) / 1000 AS scope3_tonnes,
    AVG(data_quality_score) AS avg_quality_score,
    COUNT(DISTINCT category) AS category_count,
    COUNT(DISTINCT facility_id) AS facility_count
FROM metrics.emission_measurements
GROUP BY time_bucket('1 year', timestamp), organization_id, scope
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'metrics.yearly_emissions_summary',
    start_offset => INTERVAL '2 years',
    end_offset => INTERVAL '1 month',
    schedule_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- -----------------------------------------------------------------------------
-- Emission Calculation Helper Functions
-- -----------------------------------------------------------------------------

-- Function to calculate emissions using emission factors
CREATE OR REPLACE FUNCTION public.calculate_emissions(
    p_activity_type VARCHAR(100),
    p_activity_value NUMERIC,
    p_activity_unit VARCHAR(50),
    p_country_code CHAR(2) DEFAULT NULL,
    p_sector_code VARCHAR(20) DEFAULT NULL,
    p_reference_date DATE DEFAULT CURRENT_DATE
) RETURNS TABLE (
    emission_factor_id UUID,
    emission_factor_value NUMERIC,
    co2e_kg NUMERIC,
    uncertainty_pct NUMERIC,
    data_quality data_quality_level,
    source_database VARCHAR(100)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ef.id,
        ef.co2e_factor,
        (p_activity_value * ef.co2e_factor)::NUMERIC,
        ef.uncertainty_pct,
        ef.data_quality_level,
        ef.source_database
    FROM public.emission_factors ef
    WHERE ef.activity_type = p_activity_type
      AND ef.activity_unit = p_activity_unit
      AND ef.is_active = TRUE
      AND p_reference_date >= ef.valid_from
      AND (ef.valid_to IS NULL OR p_reference_date <= ef.valid_to)
      AND (p_country_code IS NULL OR ef.country_code = p_country_code OR ef.country_code IS NULL)
      AND (p_sector_code IS NULL OR ef.sector_code = p_sector_code OR ef.sector_code IS NULL)
    ORDER BY
        -- Prefer more specific factors
        CASE WHEN ef.country_code = p_country_code THEN 0 ELSE 1 END,
        CASE WHEN ef.sector_code = p_sector_code THEN 0 ELSE 1 END,
        -- Prefer higher quality data
        CASE ef.data_quality_level
            WHEN 'primary_measured' THEN 0
            WHEN 'primary_calculated' THEN 1
            WHEN 'secondary_specific' THEN 2
            WHEN 'secondary_average' THEN 3
            WHEN 'estimated' THEN 4
            WHEN 'default' THEN 5
        END,
        -- Prefer more recent factors
        ef.publication_year DESC NULLS LAST
    LIMIT 1;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function to get emissions summary for an organization
CREATE OR REPLACE FUNCTION public.get_emissions_summary(
    p_organization_id UUID,
    p_start_date TIMESTAMPTZ,
    p_end_date TIMESTAMPTZ
) RETURNS TABLE (
    scope emission_scope,
    category emission_category,
    total_co2e_tonnes NUMERIC,
    measurement_count BIGINT,
    avg_quality_score NUMERIC,
    data_coverage_pct NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        em.scope,
        em.category,
        SUM(em.co2e_kg) / 1000 AS total_co2e_tonnes,
        COUNT(*) AS measurement_count,
        AVG(em.data_quality_score)::NUMERIC AS avg_quality_score,
        (COUNT(em.id)::NUMERIC /
         NULLIF(COUNT(DISTINCT DATE_TRUNC('day', em.timestamp)), 0)
        )::NUMERIC AS data_coverage_pct
    FROM metrics.emission_measurements em
    WHERE em.organization_id = p_organization_id
      AND em.timestamp >= p_start_date
      AND em.timestamp < p_end_date
    GROUP BY em.scope, em.category
    ORDER BY em.scope, total_co2e_tonnes DESC;
END;
$$ LANGUAGE plpgsql STABLE;

-- -----------------------------------------------------------------------------
-- Row Level Security
-- -----------------------------------------------------------------------------

ALTER TABLE metrics.emission_measurements ENABLE ROW LEVEL SECURITY;
ALTER TABLE metrics.calculation_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.emission_factors ENABLE ROW LEVEL SECURITY;

-- Emission measurements: Users can only see their organization's data
CREATE POLICY emission_measurements_select_policy ON metrics.emission_measurements
    FOR SELECT
    USING (
        organization_id IN (
            SELECT organization_id
            FROM public.organization_members
            WHERE user_id = NULLIF(current_setting('app.current_user_id', true), '')::UUID
        )
    );

-- Calculation results: Users can only see their organization's results
CREATE POLICY calculation_results_select_policy ON metrics.calculation_results
    FOR SELECT
    USING (
        organization_id IN (
            SELECT organization_id
            FROM public.organization_members
            WHERE user_id = NULLIF(current_setting('app.current_user_id', true), '')::UUID
        )
    );

-- Emission factors: All active users can read
CREATE POLICY emission_factors_select_policy ON public.emission_factors
    FOR SELECT
    USING (is_active = TRUE);

-- -----------------------------------------------------------------------------
-- Comments
-- -----------------------------------------------------------------------------

COMMENT ON TABLE public.emission_factors IS 'Repository of emission factors from various sources (EPA, DEFRA, ecoinvent, etc.)';
COMMENT ON TABLE metrics.emission_measurements IS 'Time-series hypertable storing individual emission measurements';
COMMENT ON TABLE metrics.calculation_results IS 'Time-series hypertable storing aggregated calculation results';

COMMENT ON MATERIALIZED VIEW metrics.daily_emissions_by_scope IS 'Continuous aggregate: Daily emissions rolled up by organization and scope';
COMMENT ON MATERIALIZED VIEW metrics.monthly_emissions_by_category IS 'Continuous aggregate: Monthly emissions by organization, scope, and category';
COMMENT ON MATERIALIZED VIEW metrics.yearly_emissions_summary IS 'Continuous aggregate: Yearly emissions summary by organization';

COMMENT ON FUNCTION public.calculate_emissions IS 'Calculate emissions using best-matching emission factor';
COMMENT ON FUNCTION public.get_emissions_summary IS 'Get emissions summary for an organization within a date range';
