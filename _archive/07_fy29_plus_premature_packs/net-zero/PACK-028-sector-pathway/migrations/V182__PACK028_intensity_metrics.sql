-- =============================================================================
-- V182: PACK-028 Sector Pathway Pack - Intensity Metrics
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    002 of 015
-- Date:         March 2026
--
-- Sector-specific intensity metric definitions, historical intensity data,
-- normalization rules, and 20+ metric types for SBTi SDA intensity
-- convergence tracking across power, steel, cement, aluminum, chemicals,
-- pulp/paper, aviation, shipping, road transport, rail, and buildings.
--
-- Tables (1):
--   1. pack028_sector_pathway.gl_sector_intensity_metrics
--
-- Previous: V181__PACK028_sector_classification.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack028_sector_pathway.gl_sector_intensity_metrics
-- =============================================================================
-- Sector-specific intensity metric definitions and historical data with
-- support for 20+ metric types, data normalization rules, trend analysis,
-- and SBTi SDA intensity convergence tracking.

CREATE TABLE pack028_sector_pathway.gl_sector_intensity_metrics (
    metric_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID            NOT NULL,
    classification_id           UUID            REFERENCES pack028_sector_pathway.gl_sector_classifications(classification_id) ON DELETE SET NULL,
    -- Sector and metric type
    sector                      VARCHAR(80)     NOT NULL,
    sector_code                 VARCHAR(20)     NOT NULL,
    metric_type                 VARCHAR(60)     NOT NULL,
    metric_code                 VARCHAR(40)     NOT NULL,
    -- Metric definition
    numerator_unit              VARCHAR(30)     NOT NULL,
    denominator_unit            VARCHAR(50)     NOT NULL,
    display_unit                VARCHAR(80)     NOT NULL,
    metric_description          TEXT,
    -- Reporting period
    reporting_year              INTEGER         NOT NULL,
    reporting_period            VARCHAR(10)     DEFAULT 'ANNUAL',
    period_start                DATE,
    period_end                  DATE,
    -- Activity data (denominator)
    activity_value              DECIMAL(18,4)   NOT NULL,
    activity_unit               VARCHAR(50)     NOT NULL,
    activity_description        VARCHAR(500),
    activity_data_source        VARCHAR(200),
    activity_data_quality       VARCHAR(20)     DEFAULT 'MEASURED',
    -- Emissions data (numerator)
    scope1_emissions_tco2e      DECIMAL(18,4)   DEFAULT 0,
    scope2_location_tco2e       DECIMAL(18,4)   DEFAULT 0,
    scope2_market_tco2e         DECIMAL(18,4)   DEFAULT 0,
    scope3_upstream_tco2e       DECIMAL(18,4)   DEFAULT 0,
    scope3_downstream_tco2e     DECIMAL(18,4)   DEFAULT 0,
    total_emissions_tco2e       DECIMAL(18,4)   NOT NULL,
    emissions_boundary          VARCHAR(30)     NOT NULL DEFAULT 'SCOPE_1_2',
    -- Calculated intensity
    intensity_value             DECIMAL(18,8)   NOT NULL,
    intensity_unit              VARCHAR(80)     NOT NULL,
    -- Normalization
    normalization_method        VARCHAR(30)     DEFAULT 'DIRECT',
    normalization_factor        DECIMAL(18,8)   DEFAULT 1.0,
    normalized_intensity        DECIMAL(18,8),
    climate_correction_applied  BOOLEAN         DEFAULT FALSE,
    climate_correction_factor   DECIMAL(8,4),
    -- Base year comparison
    is_base_year                BOOLEAN         DEFAULT FALSE,
    base_year_intensity         DECIMAL(18,8),
    yoy_change_pct              DECIMAL(8,4),
    cumulative_change_pct       DECIMAL(8,4),
    cagr_pct                    DECIMAL(8,4),
    -- Sector-specific sub-metrics (for decomposition)
    sub_metrics                 JSONB           DEFAULT '{}',
    -- Power sector specifics
    generation_mix              JSONB           DEFAULT '{}',
    capacity_mw                 DECIMAL(12,2),
    load_factor_pct             DECIMAL(6,2),
    -- Steel sector specifics
    production_route            VARCHAR(50),
    scrap_share_pct             DECIMAL(6,2),
    -- Cement sector specifics
    clinker_ratio               DECIMAL(6,4),
    -- Aviation sector specifics
    load_factor_passenger_pct   DECIMAL(6,2),
    average_stage_length_km     DECIMAL(10,2),
    -- Buildings sector specifics
    floor_area_m2               DECIMAL(14,2),
    heating_degree_days         DECIMAL(8,2),
    cooling_degree_days         DECIMAL(8,2),
    -- Data quality
    data_quality_score          DECIMAL(5,2),
    data_quality_level          INTEGER         DEFAULT 3,
    uncertainty_pct             DECIMAL(6,2),
    data_completeness_pct       DECIMAL(5,2)    DEFAULT 100.00,
    -- Verification
    verification_status         VARCHAR(20)     DEFAULT 'UNVERIFIED',
    verified_by                 VARCHAR(255),
    verification_date           DATE,
    verification_standard       VARCHAR(50),
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_im_metric_type CHECK (
        metric_type IN (
            -- Power
            'GCO2_PER_KWH', 'TCO2E_PER_MWH', 'CAPACITY_WEIGHTED_INTENSITY',
            -- Steel
            'TCO2E_PER_TONNE_CRUDE_STEEL', 'TCO2E_PER_TONNE_STEEL_BF_BOF',
            'TCO2E_PER_TONNE_STEEL_EAF', 'TCO2E_PER_TONNE_DRI',
            -- Cement
            'TCO2E_PER_TONNE_CLINKER', 'TCO2E_PER_TONNE_CEMENT', 'TCO2E_PER_M3_CONCRETE',
            -- Aluminum
            'TCO2E_PER_TONNE_ALUMINIUM', 'TCO2E_PER_TONNE_PRIMARY_AL',
            'TCO2E_PER_TONNE_SECONDARY_AL',
            -- Chemicals
            'TCO2E_PER_TONNE_PRODUCT', 'TCO2E_PER_TONNE_HVC',
            -- Pulp & Paper
            'TCO2E_PER_TONNE_PULP', 'TCO2E_PER_TONNE_PAPER',
            -- Aviation
            'GCO2_PER_PKM', 'GCO2_PER_RTK', 'L_FUEL_PER_100_PKM',
            -- Shipping
            'GCO2_PER_TKM', 'EEOI', 'AER',
            -- Road transport
            'GCO2_PER_VKM', 'GCO2_PER_PKM_ROAD', 'GCO2_PER_TKM_ROAD',
            -- Rail
            'GCO2_PER_PKM_RAIL', 'GCO2_PER_TKM_RAIL',
            -- Buildings
            'KGCO2_PER_M2_YEAR', 'KGCO2_PER_M2_EMBODIED', 'KWH_PER_M2_YEAR',
            -- Agriculture
            'TCO2E_PER_TONNE_FOOD', 'TCO2E_PER_HECTARE',
            -- Oil & Gas
            'GCO2_PER_MJ', 'KGCO2E_PER_BOE',
            -- Generic
            'TCO2E_PER_REVENUE', 'TCO2E_PER_EMPLOYEE', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p028_im_reporting_period CHECK (
        reporting_period IN ('ANNUAL', 'SEMI_ANNUAL', 'QUARTERLY', 'MONTHLY')
    ),
    CONSTRAINT chk_p028_im_activity_value CHECK (
        activity_value > 0
    ),
    CONSTRAINT chk_p028_im_total_emissions CHECK (
        total_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p028_im_intensity CHECK (
        intensity_value >= 0
    ),
    CONSTRAINT chk_p028_im_emissions_boundary CHECK (
        emissions_boundary IN ('SCOPE_1', 'SCOPE_1_2', 'SCOPE_1_2_3', 'SCOPE_2', 'SCOPE_3', 'FULL')
    ),
    CONSTRAINT chk_p028_im_normalization CHECK (
        normalization_method IN ('DIRECT', 'CLIMATE_CORRECTED', 'PPP_ADJUSTED',
                                 'PRODUCTION_WEIGHTED', 'REVENUE_WEIGHTED', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_im_activity_quality CHECK (
        activity_data_quality IN ('MEASURED', 'CALCULATED', 'ESTIMATED', 'PROXY', 'DEFAULT')
    ),
    CONSTRAINT chk_p028_im_data_quality_level CHECK (
        data_quality_level >= 1 AND data_quality_level <= 5
    ),
    CONSTRAINT chk_p028_im_data_quality_score CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    ),
    CONSTRAINT chk_p028_im_uncertainty CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 100)
    ),
    CONSTRAINT chk_p028_im_completeness CHECK (
        data_completeness_pct >= 0 AND data_completeness_pct <= 100
    ),
    CONSTRAINT chk_p028_im_verification CHECK (
        verification_status IN ('UNVERIFIED', 'INTERNALLY_VERIFIED', 'THIRD_PARTY_LIMITED',
                                'THIRD_PARTY_REASONABLE', 'SBTI_VALIDATED')
    ),
    CONSTRAINT chk_p028_im_reporting_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p028_im_production_route CHECK (
        production_route IS NULL OR production_route IN (
            'BF_BOF', 'EAF_SCRAP', 'EAF_DRI', 'DRI_GAS', 'DRI_HYDROGEN',
            'INTEGRATED', 'SECONDARY', 'PRIMARY', 'MIXED'
        )
    ),
    CONSTRAINT chk_p028_im_clinker_ratio CHECK (
        clinker_ratio IS NULL OR (clinker_ratio >= 0 AND clinker_ratio <= 1.0)
    ),
    CONSTRAINT chk_p028_im_yoy CHECK (
        yoy_change_pct IS NULL OR (yoy_change_pct >= -100 AND yoy_change_pct <= 1000)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p028_im_tenant             ON pack028_sector_pathway.gl_sector_intensity_metrics(tenant_id);
CREATE INDEX idx_p028_im_company            ON pack028_sector_pathway.gl_sector_intensity_metrics(company_id);
CREATE INDEX idx_p028_im_classification     ON pack028_sector_pathway.gl_sector_intensity_metrics(classification_id);
CREATE INDEX idx_p028_im_sector             ON pack028_sector_pathway.gl_sector_intensity_metrics(sector);
CREATE INDEX idx_p028_im_sector_code        ON pack028_sector_pathway.gl_sector_intensity_metrics(sector_code);
CREATE INDEX idx_p028_im_metric_type        ON pack028_sector_pathway.gl_sector_intensity_metrics(metric_type);
CREATE INDEX idx_p028_im_metric_code        ON pack028_sector_pathway.gl_sector_intensity_metrics(metric_code);
CREATE INDEX idx_p028_im_year               ON pack028_sector_pathway.gl_sector_intensity_metrics(reporting_year);
CREATE INDEX idx_p028_im_company_year       ON pack028_sector_pathway.gl_sector_intensity_metrics(company_id, reporting_year);
CREATE INDEX idx_p028_im_company_metric     ON pack028_sector_pathway.gl_sector_intensity_metrics(company_id, metric_type, reporting_year);
CREATE INDEX idx_p028_im_sector_metric_year ON pack028_sector_pathway.gl_sector_intensity_metrics(sector_code, metric_type, reporting_year);
CREATE INDEX idx_p028_im_base_year          ON pack028_sector_pathway.gl_sector_intensity_metrics(company_id, is_base_year) WHERE is_base_year = TRUE;
CREATE INDEX idx_p028_im_intensity          ON pack028_sector_pathway.gl_sector_intensity_metrics(intensity_value);
CREATE INDEX idx_p028_im_verification       ON pack028_sector_pathway.gl_sector_intensity_metrics(verification_status);
CREATE INDEX idx_p028_im_data_quality       ON pack028_sector_pathway.gl_sector_intensity_metrics(data_quality_level);
CREATE INDEX idx_p028_im_emissions_bnd      ON pack028_sector_pathway.gl_sector_intensity_metrics(emissions_boundary);
CREATE INDEX idx_p028_im_created            ON pack028_sector_pathway.gl_sector_intensity_metrics(created_at DESC);
CREATE INDEX idx_p028_im_sub_metrics        ON pack028_sector_pathway.gl_sector_intensity_metrics USING GIN(sub_metrics);
CREATE INDEX idx_p028_im_gen_mix            ON pack028_sector_pathway.gl_sector_intensity_metrics USING GIN(generation_mix);
CREATE INDEX idx_p028_im_metadata           ON pack028_sector_pathway.gl_sector_intensity_metrics USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p028_intensity_metrics_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_sector_intensity_metrics
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack028_sector_pathway.gl_sector_intensity_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_im_tenant_isolation
    ON pack028_sector_pathway.gl_sector_intensity_metrics
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_im_service_bypass
    ON pack028_sector_pathway.gl_sector_intensity_metrics
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_sector_intensity_metrics TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack028_sector_pathway.gl_sector_intensity_metrics IS
    'Sector-specific intensity metric definitions and historical data with 20+ metric types for SBTi SDA intensity convergence tracking across all supported sectors.';

COMMENT ON COLUMN pack028_sector_pathway.gl_sector_intensity_metrics.metric_id IS 'Unique intensity metric record identifier.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_intensity_metrics.metric_type IS 'Intensity metric type (e.g., GCO2_PER_KWH for power, TCO2E_PER_TONNE_CRUDE_STEEL for steel).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_intensity_metrics.activity_value IS 'Activity data denominator value (e.g., MWh generated, tonnes produced).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_intensity_metrics.intensity_value IS 'Calculated intensity value (emissions / activity).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_intensity_metrics.normalization_method IS 'Method used to normalize intensity (DIRECT, CLIMATE_CORRECTED, PPP_ADJUSTED, etc.).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_intensity_metrics.is_base_year IS 'Whether this record represents the base year intensity for SDA pathway calculation.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_intensity_metrics.yoy_change_pct IS 'Year-over-year change in intensity (negative = improvement).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_intensity_metrics.generation_mix IS 'Power sector only: JSONB breakdown of generation mix by source.';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_intensity_metrics.production_route IS 'Steel sector only: production route (BF_BOF, EAF_SCRAP, EAF_DRI, DRI_HYDROGEN).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_intensity_metrics.clinker_ratio IS 'Cement sector only: clinker-to-cement ratio (0-1.0).';
COMMENT ON COLUMN pack028_sector_pathway.gl_sector_intensity_metrics.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
