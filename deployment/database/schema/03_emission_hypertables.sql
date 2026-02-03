-- =============================================================================
-- GreenLang Climate OS - Emission Hypertables
-- =============================================================================
-- File: 03_emission_hypertables.sql
-- Description: TimescaleDB hypertables for emission measurements, emission
--              factors, and calculation results with appropriate chunking,
--              compression, and retention policies.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Emission Sources Table (Regular table, not hypertable)
-- -----------------------------------------------------------------------------
-- Defines the sources of emissions within projects (facilities, equipment, etc.)
CREATE TABLE IF NOT EXISTS metrics.emission_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Relationships
    org_id UUID NOT NULL REFERENCES public.organizations(id) ON DELETE CASCADE,
    project_id UUID NOT NULL REFERENCES public.projects(id) ON DELETE CASCADE,

    -- Source identification
    name VARCHAR(255) NOT NULL,
    code VARCHAR(100),
    description TEXT,

    -- Source classification
    -- Categories: stationary_combustion, mobile_combustion, fugitive, process, purchased_energy, transport, waste
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),

    -- GHG Protocol scope (1, 2, 3)
    scope INTEGER NOT NULL CHECK (scope IN (1, 2, 3)),

    -- Scope 3 category (if applicable)
    -- Categories 1-15 per GHG Protocol
    scope3_category INTEGER CHECK (scope3_category BETWEEN 1 AND 15),

    -- Source metadata
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Location information
    location JSONB DEFAULT '{}',

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT true,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT emission_sources_org_code_unique UNIQUE (org_id, code)
);

CREATE INDEX IF NOT EXISTS idx_emission_sources_org ON metrics.emission_sources(org_id);
CREATE INDEX IF NOT EXISTS idx_emission_sources_project ON metrics.emission_sources(project_id);
CREATE INDEX IF NOT EXISTS idx_emission_sources_scope ON metrics.emission_sources(scope);
CREATE INDEX IF NOT EXISTS idx_emission_sources_category ON metrics.emission_sources(category);

COMMENT ON TABLE metrics.emission_sources IS 'Emission sources within projects (facilities, equipment, vehicles, etc.)';

-- -----------------------------------------------------------------------------
-- Emission Measurements Hypertable
-- -----------------------------------------------------------------------------
-- Time-series data for actual emission measurements and activity data.
-- This is the primary table for storing emission data points.
CREATE TABLE IF NOT EXISTS metrics.emission_measurements (
    -- Time is the primary partitioning column
    time TIMESTAMPTZ NOT NULL,

    -- Relationships (denormalized for query performance)
    org_id UUID NOT NULL,
    project_id UUID NOT NULL,
    source_id UUID NOT NULL,

    -- Emission scope (1, 2, or 3)
    scope INTEGER NOT NULL CHECK (scope IN (1, 2, 3)),

    -- Measurement type
    -- Types: activity_data, direct_measurement, calculated, estimated
    measurement_type VARCHAR(50) NOT NULL DEFAULT 'activity_data',

    -- Activity data (input for calculations)
    activity_value DOUBLE PRECISION,
    activity_unit VARCHAR(50),

    -- Calculated emissions
    emission_value DOUBLE PRECISION NOT NULL,
    emission_unit VARCHAR(50) NOT NULL DEFAULT 'kgCO2e',

    -- Individual GHG values (optional breakdown)
    co2_value DOUBLE PRECISION,
    ch4_value DOUBLE PRECISION,
    n2o_value DOUBLE PRECISION,
    hfc_value DOUBLE PRECISION,
    pfc_value DOUBLE PRECISION,
    sf6_value DOUBLE PRECISION,
    nf3_value DOUBLE PRECISION,

    -- Emission factor used
    emission_factor_id UUID,
    emission_factor_value DOUBLE PRECISION,
    emission_factor_source VARCHAR(255),

    -- Data quality
    data_quality_score INTEGER CHECK (data_quality_score BETWEEN 0 AND 100),
    uncertainty_percent DOUBLE PRECISION,
    verification_status VARCHAR(50) DEFAULT 'unverified',

    -- Source metadata
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Data lineage
    data_source VARCHAR(255),
    import_batch_id UUID,

    -- Record tracking
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by UUID
);

-- Create hypertable with 1 day chunks
-- 1 day chunks are optimal for emission data that is typically reported daily/weekly
SELECT create_hypertable(
    'metrics.emission_measurements',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_emission_measurements_org
    ON metrics.emission_measurements(org_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_emission_measurements_project
    ON metrics.emission_measurements(project_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_emission_measurements_source
    ON metrics.emission_measurements(source_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_emission_measurements_scope
    ON metrics.emission_measurements(scope, time DESC);

-- Composite index for common filtered queries
CREATE INDEX IF NOT EXISTS idx_emission_measurements_org_scope
    ON metrics.emission_measurements(org_id, scope, time DESC);

-- Index for data quality filtering
CREATE INDEX IF NOT EXISTS idx_emission_measurements_quality
    ON metrics.emission_measurements(data_quality_score, time DESC)
    WHERE data_quality_score IS NOT NULL;

COMMENT ON TABLE metrics.emission_measurements IS 'Time-series emission measurements with 1-day chunks, 7-day compression, 7-year retention';

-- -----------------------------------------------------------------------------
-- Emission Factors Table (Regular table)
-- -----------------------------------------------------------------------------
-- Reference data for emission factors from various sources (EPA, IPCC, etc.)
CREATE TABLE IF NOT EXISTS metrics.emission_factors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Factor identification
    name VARCHAR(255) NOT NULL,
    code VARCHAR(100) UNIQUE,

    -- Classification
    category VARCHAR(100) NOT NULL,
    subcategory VARCHAR(100),

    -- Geographic scope
    region VARCHAR(100) NOT NULL DEFAULT 'global',
    country_code CHAR(2),

    -- Factor values
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(100) NOT NULL,

    -- Individual GHG factors (optional)
    co2_factor DOUBLE PRECISION,
    ch4_factor DOUBLE PRECISION,
    n2o_factor DOUBLE PRECISION,

    -- Global Warming Potentials used
    gwp_source VARCHAR(100) DEFAULT 'AR5',

    -- Source and validity
    source VARCHAR(255) NOT NULL,
    source_url VARCHAR(500),
    methodology TEXT,

    -- Validity period
    valid_from DATE NOT NULL,
    valid_to DATE,

    -- Data quality
    uncertainty_percent DOUBLE PRECISION,
    data_quality_rating VARCHAR(20),

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT true,

    -- Metadata
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT emission_factors_validity CHECK (valid_to IS NULL OR valid_to >= valid_from)
);

CREATE INDEX IF NOT EXISTS idx_emission_factors_category ON metrics.emission_factors(category);
CREATE INDEX IF NOT EXISTS idx_emission_factors_region ON metrics.emission_factors(region);
CREATE INDEX IF NOT EXISTS idx_emission_factors_valid ON metrics.emission_factors(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_emission_factors_active ON metrics.emission_factors(is_active) WHERE is_active = true;

-- GIN index for metadata search
CREATE INDEX IF NOT EXISTS idx_emission_factors_metadata ON metrics.emission_factors USING GIN(metadata jsonb_path_ops);

COMMENT ON TABLE metrics.emission_factors IS 'Emission factors from various sources (EPA, IPCC, regional databases)';

-- -----------------------------------------------------------------------------
-- Calculation Results Hypertable
-- -----------------------------------------------------------------------------
-- Stores results of emission calculations, aggregations, and analyses.
CREATE TABLE IF NOT EXISTS metrics.calculation_results (
    -- Time of calculation
    time TIMESTAMPTZ NOT NULL,

    -- Calculation run identifier
    run_id UUID NOT NULL,

    -- Relationships
    org_id UUID NOT NULL,
    project_id UUID,

    -- Result type
    -- Types: emission_total, scope_breakdown, category_breakdown, forecast, scenario, benchmark
    result_type VARCHAR(100) NOT NULL,

    -- Result identification
    result_key VARCHAR(255) NOT NULL,

    -- Calculated values
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(50) NOT NULL DEFAULT 'kgCO2e',

    -- Breakdown (if applicable)
    breakdown JSONB DEFAULT '{}',

    -- Calculation parameters
    calculation_method VARCHAR(100),
    parameters JSONB NOT NULL DEFAULT '{}',

    -- Time range covered by calculation
    period_start TIMESTAMPTZ,
    period_end TIMESTAMPTZ,

    -- Metadata
    metadata JSONB NOT NULL DEFAULT '{}',

    -- Audit trail
    created_by UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create hypertable with 1 hour chunks
-- Shorter chunks because calculation results are generated frequently
SELECT create_hypertable(
    'metrics.calculation_results',
    'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_calculation_results_run
    ON metrics.calculation_results(run_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_calculation_results_org
    ON metrics.calculation_results(org_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_calculation_results_project
    ON metrics.calculation_results(project_id, time DESC)
    WHERE project_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_calculation_results_type
    ON metrics.calculation_results(result_type, time DESC);

-- Composite index for fetching specific results
CREATE INDEX IF NOT EXISTS idx_calculation_results_org_type
    ON metrics.calculation_results(org_id, result_type, result_key, time DESC);

COMMENT ON TABLE metrics.calculation_results IS 'Calculation results with 1-hour chunks, 1-day compression';

-- -----------------------------------------------------------------------------
-- Helper function to get latest emission factor
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION metrics.get_emission_factor(
    p_category VARCHAR,
    p_region VARCHAR,
    p_date DATE DEFAULT CURRENT_DATE
)
RETURNS TABLE (
    factor_id UUID,
    factor_value DOUBLE PRECISION,
    factor_unit VARCHAR,
    factor_source VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ef.id,
        ef.value,
        ef.unit,
        ef.source
    FROM metrics.emission_factors ef
    WHERE ef.category = p_category
      AND ef.region = p_region
      AND ef.is_active = true
      AND ef.valid_from <= p_date
      AND (ef.valid_to IS NULL OR ef.valid_to >= p_date)
    ORDER BY ef.valid_from DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION metrics.get_emission_factor IS 'Get the most recent valid emission factor for a category and region';
