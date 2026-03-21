-- =============================================================================
-- V192: PACK-028 Sector Pathway Pack - Sector Reference Data
-- =============================================================================
-- Pack:         PACK-028 (Sector Pathway Pack)
-- Migration:    012 of 015
-- Date:         March 2026
--
-- Sector reference data tables for emission factors, activity data
-- normalization, and technology catalogs supporting sector pathway
-- calculations and benchmarking.
--
-- Tables (3):
--   1. pack028_sector_pathway.gl_sector_emission_factors
--   2. pack028_sector_pathway.gl_sector_activity_data
--   3. pack028_sector_pathway.gl_sector_technology_catalog
--
-- Previous: V191__PACK028_ipcc_ar6_pathways.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack028_sector_pathway.gl_sector_emission_factors
-- =============================================================================
-- Sector-specific emission factors from IPCC, EPA, DEFRA, and industry
-- sources with vintage tracking, regional variants, and uncertainty ranges.

CREATE TABLE pack028_sector_pathway.gl_sector_emission_factors (
    factor_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    -- Sector classification
    sector                      VARCHAR(80)     NOT NULL,
    sector_code                 VARCHAR(20)     NOT NULL,
    sub_sector                  VARCHAR(80),
    -- Factor definition
    factor_name                 VARCHAR(255)    NOT NULL,
    factor_code                 VARCHAR(60)     NOT NULL,
    factor_category             VARCHAR(50)     NOT NULL,
    -- Emission factor values
    factor_value                DECIMAL(18,8)   NOT NULL,
    factor_unit                 VARCHAR(80)     NOT NULL,
    factor_numerator_unit       VARCHAR(30)     NOT NULL,
    factor_denominator_unit     VARCHAR(50)     NOT NULL,
    -- GHG breakdown
    co2_factor                  DECIMAL(18,8),
    ch4_factor                  DECIMAL(18,8),
    n2o_factor                  DECIMAL(18,8),
    hfc_factor                  DECIMAL(18,8),
    pfc_factor                  DECIMAL(18,8),
    sf6_factor                  DECIMAL(18,8),
    nf3_factor                  DECIMAL(18,8),
    -- Uncertainty
    uncertainty_pct             DECIMAL(6,2),
    factor_low                  DECIMAL(18,8),
    factor_high                 DECIMAL(18,8),
    confidence_level            VARCHAR(20)     DEFAULT 'MEDIUM',
    -- Source and vintage
    source                      VARCHAR(100)    NOT NULL,
    source_version              VARCHAR(50),
    source_year                 INTEGER         NOT NULL,
    data_vintage                INTEGER         NOT NULL,
    valid_from                  DATE,
    valid_to                    DATE,
    -- Regional specificity
    region                      VARCHAR(30)     DEFAULT 'GLOBAL',
    country_code                VARCHAR(3),
    regional_variants           JSONB           DEFAULT '{}',
    -- Applicability
    applicable_scopes           TEXT[]          DEFAULT '{SCOPE_1}',
    applicable_activities       TEXT[]          DEFAULT '{}',
    technology_specificity      VARCHAR(50),
    -- GWP basis
    gwp_source                  VARCHAR(20)     DEFAULT 'AR6',
    gwp_timeframe               VARCHAR(10)     DEFAULT '100yr',
    -- Metadata
    is_default                  BOOLEAN         DEFAULT FALSE,
    is_active                   BOOLEAN         DEFAULT TRUE,
    superseded_by               UUID,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_sef_factor_category CHECK (
        factor_category IN (
            'COMBUSTION', 'PROCESS', 'FUGITIVE', 'ELECTRICITY_GRID', 'TRANSPORT',
            'WASTE', 'AGRICULTURE', 'LULUCF', 'INDUSTRIAL_PROCESS', 'REFRIGERANT',
            'UPSTREAM', 'DOWNSTREAM', 'LIFECYCLE', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p028_sef_source CHECK (
        source IN (
            'IPCC_2006', 'IPCC_2006_2019_REF', 'EPA', 'DEFRA', 'ECOINVENT',
            'GHG_PROTOCOL', 'IEA', 'EU_ETS', 'ADEME', 'INDUSTRY_SPECIFIC',
            'NATIONAL_INVENTORY', 'COMPANY_SPECIFIC', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p028_sef_confidence CHECK (
        confidence_level IN ('VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW', 'VERY_LOW')
    ),
    CONSTRAINT chk_p028_sef_region CHECK (
        region IN ('GLOBAL', 'OECD', 'NON_OECD', 'EU', 'NORTH_AMERICA', 'ASIA_PACIFIC',
                   'LATIN_AMERICA', 'AFRICA', 'MIDDLE_EAST', 'CHINA', 'INDIA',
                   'COUNTRY_SPECIFIC', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_sef_gwp_source CHECK (
        gwp_source IN ('AR4', 'AR5', 'AR6', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_sef_factor_positive CHECK (
        factor_value >= 0
    ),
    CONSTRAINT chk_p028_sef_uncertainty CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 200)
    ),
    CONSTRAINT chk_p028_sef_source_year CHECK (
        source_year >= 1990 AND source_year <= 2100
    )
);

-- Indexes for gl_sector_emission_factors
CREATE INDEX idx_p028_sef_tenant            ON pack028_sector_pathway.gl_sector_emission_factors(tenant_id);
CREATE INDEX idx_p028_sef_sector            ON pack028_sector_pathway.gl_sector_emission_factors(sector_code);
CREATE INDEX idx_p028_sef_factor_code       ON pack028_sector_pathway.gl_sector_emission_factors(factor_code);
CREATE INDEX idx_p028_sef_category          ON pack028_sector_pathway.gl_sector_emission_factors(factor_category);
CREATE INDEX idx_p028_sef_source            ON pack028_sector_pathway.gl_sector_emission_factors(source);
CREATE INDEX idx_p028_sef_vintage           ON pack028_sector_pathway.gl_sector_emission_factors(data_vintage);
CREATE INDEX idx_p028_sef_region            ON pack028_sector_pathway.gl_sector_emission_factors(region);
CREATE INDEX idx_p028_sef_country           ON pack028_sector_pathway.gl_sector_emission_factors(country_code) WHERE country_code IS NOT NULL;
CREATE INDEX idx_p028_sef_default           ON pack028_sector_pathway.gl_sector_emission_factors(is_default) WHERE is_default = TRUE;
CREATE INDEX idx_p028_sef_active            ON pack028_sector_pathway.gl_sector_emission_factors(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p028_sef_sector_cat        ON pack028_sector_pathway.gl_sector_emission_factors(sector_code, factor_category);
CREATE INDEX idx_p028_sef_sector_source     ON pack028_sector_pathway.gl_sector_emission_factors(sector_code, source, data_vintage);
CREATE INDEX idx_p028_sef_scopes            ON pack028_sector_pathway.gl_sector_emission_factors USING GIN(applicable_scopes);
CREATE INDEX idx_p028_sef_created           ON pack028_sector_pathway.gl_sector_emission_factors(created_at DESC);
CREATE INDEX idx_p028_sef_metadata          ON pack028_sector_pathway.gl_sector_emission_factors USING GIN(metadata);

-- Trigger
CREATE TRIGGER trg_p028_emission_factors_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_sector_emission_factors
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- RLS
ALTER TABLE pack028_sector_pathway.gl_sector_emission_factors ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_sef_tenant_isolation
    ON pack028_sector_pathway.gl_sector_emission_factors
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_sef_service_bypass
    ON pack028_sector_pathway.gl_sector_emission_factors
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_sector_emission_factors TO PUBLIC;

-- =============================================================================
-- Table 2: pack028_sector_pathway.gl_sector_activity_data
-- =============================================================================
-- Sector-specific activity data for intensity metric calculation including
-- production volumes, energy consumption, transport metrics, and building
-- area data with normalization and quality tracking.

CREATE TABLE pack028_sector_pathway.gl_sector_activity_data (
    activity_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    company_id                  UUID            NOT NULL,
    classification_id           UUID            REFERENCES pack028_sector_pathway.gl_sector_classifications(classification_id) ON DELETE SET NULL,
    -- Sector context
    sector                      VARCHAR(80)     NOT NULL,
    sector_code                 VARCHAR(20)     NOT NULL,
    -- Activity definition
    activity_name               VARCHAR(255)    NOT NULL,
    activity_code               VARCHAR(60)     NOT NULL,
    activity_type               VARCHAR(50)     NOT NULL,
    -- Activity data
    reporting_year              INTEGER         NOT NULL,
    reporting_period            VARCHAR(10)     DEFAULT 'ANNUAL',
    activity_value              DECIMAL(18,4)   NOT NULL,
    activity_unit               VARCHAR(50)     NOT NULL,
    -- Normalized value
    normalized_value            DECIMAL(18,4),
    normalized_unit             VARCHAR(50),
    normalization_factor        DECIMAL(18,8)   DEFAULT 1.0,
    normalization_method        VARCHAR(30)     DEFAULT 'DIRECT',
    -- Year-over-year
    yoy_change_pct              DECIMAL(8,4),
    cagr_3yr_pct                DECIMAL(8,4),
    cagr_5yr_pct                DECIMAL(8,4),
    -- Sub-activity breakdown
    sub_activities              JSONB           DEFAULT '{}',
    -- Production specifics
    production_volume           DECIMAL(18,4),
    production_unit             VARCHAR(50),
    capacity_utilization_pct    DECIMAL(6,2),
    -- Energy specifics
    energy_consumption_mwh      DECIMAL(18,4),
    energy_intensity            DECIMAL(18,8),
    energy_mix                  JSONB           DEFAULT '{}',
    -- Transport specifics
    distance_km                 DECIMAL(18,2),
    load_factor_pct             DECIMAL(6,2),
    fleet_size                  INTEGER,
    -- Buildings specifics
    floor_area_m2               DECIMAL(14,2),
    building_count              INTEGER,
    occupancy_rate_pct          DECIMAL(6,2),
    -- Data quality
    data_source                 VARCHAR(200),
    data_quality                VARCHAR(20)     DEFAULT 'MEASURED',
    data_quality_score          DECIMAL(5,2),
    uncertainty_pct             DECIMAL(6,2),
    completeness_pct            DECIMAL(5,2)    DEFAULT 100.00,
    -- Verification
    verification_status         VARCHAR(20)     DEFAULT 'UNVERIFIED',
    verified_by                 VARCHAR(255),
    verification_date           DATE,
    -- Metadata
    is_active                   BOOLEAN         DEFAULT TRUE,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_sad_activity_type CHECK (
        activity_type IN (
            'PRODUCTION_VOLUME', 'ENERGY_GENERATION', 'ENERGY_CONSUMPTION',
            'TRANSPORT_DISTANCE', 'TRANSPORT_LOAD', 'FLOOR_AREA',
            'REVENUE', 'EMPLOYEES', 'FUEL_CONSUMPTION', 'RAW_MATERIAL',
            'WASTE_VOLUME', 'WATER_CONSUMPTION', 'AGRICULTURAL_OUTPUT',
            'PASSENGER_KM', 'TONNE_KM', 'VEHICLE_KM', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p028_sad_data_quality CHECK (
        data_quality IN ('MEASURED', 'CALCULATED', 'ESTIMATED', 'PROXY', 'DEFAULT')
    ),
    CONSTRAINT chk_p028_sad_verification CHECK (
        verification_status IN ('UNVERIFIED', 'INTERNALLY_VERIFIED',
                                'THIRD_PARTY_LIMITED', 'THIRD_PARTY_REASONABLE')
    ),
    CONSTRAINT chk_p028_sad_normalization CHECK (
        normalization_method IN ('DIRECT', 'CLIMATE_CORRECTED', 'PPP_ADJUSTED',
                                 'PRODUCTION_WEIGHTED', 'REVENUE_WEIGHTED', 'CUSTOM')
    ),
    CONSTRAINT chk_p028_sad_reporting_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p028_sad_activity_positive CHECK (
        activity_value >= 0
    ),
    CONSTRAINT chk_p028_sad_completeness CHECK (
        completeness_pct >= 0 AND completeness_pct <= 100
    ),
    CONSTRAINT chk_p028_sad_quality_score CHECK (
        data_quality_score IS NULL OR (data_quality_score >= 0 AND data_quality_score <= 100)
    )
);

-- Indexes for gl_sector_activity_data
CREATE INDEX idx_p028_sad_tenant            ON pack028_sector_pathway.gl_sector_activity_data(tenant_id);
CREATE INDEX idx_p028_sad_company           ON pack028_sector_pathway.gl_sector_activity_data(company_id);
CREATE INDEX idx_p028_sad_classification    ON pack028_sector_pathway.gl_sector_activity_data(classification_id);
CREATE INDEX idx_p028_sad_sector            ON pack028_sector_pathway.gl_sector_activity_data(sector_code);
CREATE INDEX idx_p028_sad_activity_code     ON pack028_sector_pathway.gl_sector_activity_data(activity_code);
CREATE INDEX idx_p028_sad_activity_type     ON pack028_sector_pathway.gl_sector_activity_data(activity_type);
CREATE INDEX idx_p028_sad_year              ON pack028_sector_pathway.gl_sector_activity_data(reporting_year);
CREATE INDEX idx_p028_sad_company_year      ON pack028_sector_pathway.gl_sector_activity_data(company_id, reporting_year);
CREATE INDEX idx_p028_sad_company_sector    ON pack028_sector_pathway.gl_sector_activity_data(company_id, sector_code, reporting_year);
CREATE INDEX idx_p028_sad_data_quality      ON pack028_sector_pathway.gl_sector_activity_data(data_quality);
CREATE INDEX idx_p028_sad_verification      ON pack028_sector_pathway.gl_sector_activity_data(verification_status);
CREATE INDEX idx_p028_sad_active            ON pack028_sector_pathway.gl_sector_activity_data(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p028_sad_created           ON pack028_sector_pathway.gl_sector_activity_data(created_at DESC);
CREATE INDEX idx_p028_sad_sub_activities    ON pack028_sector_pathway.gl_sector_activity_data USING GIN(sub_activities);
CREATE INDEX idx_p028_sad_energy_mix        ON pack028_sector_pathway.gl_sector_activity_data USING GIN(energy_mix);
CREATE INDEX idx_p028_sad_metadata          ON pack028_sector_pathway.gl_sector_activity_data USING GIN(metadata);

-- Trigger
CREATE TRIGGER trg_p028_activity_data_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_sector_activity_data
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- RLS
ALTER TABLE pack028_sector_pathway.gl_sector_activity_data ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_sad_tenant_isolation
    ON pack028_sector_pathway.gl_sector_activity_data
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_sad_service_bypass
    ON pack028_sector_pathway.gl_sector_activity_data
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_sector_activity_data TO PUBLIC;

-- =============================================================================
-- Table 3: pack028_sector_pathway.gl_sector_technology_catalog
-- =============================================================================
-- Sector-specific technology catalog with TRL, cost data, emission reduction
-- potential, and deployment timeline data for technology roadmap generation.

CREATE TABLE pack028_sector_pathway.gl_sector_technology_catalog (
    technology_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    -- Sector
    sector                      VARCHAR(80)     NOT NULL,
    sector_code                 VARCHAR(20)     NOT NULL,
    -- Technology definition
    technology_name             VARCHAR(255)    NOT NULL,
    technology_code             VARCHAR(60)     NOT NULL,
    technology_category         VARCHAR(50)     NOT NULL,
    technology_sub_category     VARCHAR(80),
    technology_description      TEXT,
    -- TRL and maturity
    current_trl                 INTEGER         NOT NULL DEFAULT 1,
    commercial_readiness        VARCHAR(30),
    first_commercial_year       INTEGER,
    expected_mature_year        INTEGER,
    -- Cost data
    current_cost                DECIMAL(14,2),
    projected_2030_cost         DECIMAL(14,2),
    projected_2050_cost         DECIMAL(14,2),
    cost_unit                   VARCHAR(50),
    learning_rate_pct           DECIMAL(6,2),
    cost_decline_trajectory     JSONB           DEFAULT '{}',
    -- Performance
    emission_reduction_pct      DECIMAL(6,2),
    energy_efficiency_pct       DECIMAL(6,2),
    max_penetration_pct         DECIMAL(6,2)    DEFAULT 100.00,
    -- Regional availability
    availability_regions        JSONB           DEFAULT '{}',
    -- IEA reference
    iea_reference               VARCHAR(100),
    iea_scenario_alignment      VARCHAR(30),
    -- Constraints and dependencies
    prerequisites               TEXT[]          DEFAULT '{}',
    infrastructure_needs        TEXT[]          DEFAULT '{}',
    supply_chain_requirements   TEXT[]          DEFAULT '{}',
    -- Metadata
    is_reference_data           BOOLEAN         DEFAULT TRUE,
    is_active                   BOOLEAN         DEFAULT TRUE,
    data_source                 VARCHAR(200),
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p028_stc_category CHECK (
        technology_category IN (
            'RENEWABLE_ENERGY', 'ENERGY_STORAGE', 'HYDROGEN', 'CCS_CCUS',
            'ELECTRIFICATION', 'FUEL_SWITCHING', 'ENERGY_EFFICIENCY',
            'PROCESS_INNOVATION', 'CIRCULAR_ECONOMY', 'DIGITALIZATION',
            'FLEET_TRANSITION', 'BUILDING_RETROFIT', 'SUSTAINABLE_FUELS',
            'NUCLEAR', 'GRID_INFRASTRUCTURE', 'HEAT_PUMPS', 'BIOMASS',
            'CARBON_REMOVAL', 'NATURE_BASED', 'OTHER'
        )
    ),
    CONSTRAINT chk_p028_stc_trl CHECK (
        current_trl >= 1 AND current_trl <= 9
    ),
    CONSTRAINT chk_p028_stc_readiness CHECK (
        commercial_readiness IS NULL OR commercial_readiness IN (
            'CONCEPT', 'RESEARCH', 'PROTOTYPE', 'DEMONSTRATION', 'EARLY_COMMERCIAL',
            'COMMERCIALLY_AVAILABLE', 'MATURE', 'DECLINING'
        )
    ),
    CONSTRAINT chk_p028_stc_iea_alignment CHECK (
        iea_scenario_alignment IS NULL OR iea_scenario_alignment IN (
            'NZE_CRITICAL', 'NZE_IMPORTANT', 'APS_RELEVANT', 'STEPS_BASELINE', 'NOT_MAPPED'
        )
    )
);

-- Indexes for gl_sector_technology_catalog
CREATE INDEX idx_p028_stc_tenant            ON pack028_sector_pathway.gl_sector_technology_catalog(tenant_id);
CREATE INDEX idx_p028_stc_sector            ON pack028_sector_pathway.gl_sector_technology_catalog(sector_code);
CREATE INDEX idx_p028_stc_tech_code         ON pack028_sector_pathway.gl_sector_technology_catalog(technology_code);
CREATE INDEX idx_p028_stc_category          ON pack028_sector_pathway.gl_sector_technology_catalog(technology_category);
CREATE INDEX idx_p028_stc_trl               ON pack028_sector_pathway.gl_sector_technology_catalog(current_trl);
CREATE INDEX idx_p028_stc_readiness         ON pack028_sector_pathway.gl_sector_technology_catalog(commercial_readiness);
CREATE INDEX idx_p028_stc_iea_alignment     ON pack028_sector_pathway.gl_sector_technology_catalog(iea_scenario_alignment);
CREATE INDEX idx_p028_stc_reference         ON pack028_sector_pathway.gl_sector_technology_catalog(is_reference_data) WHERE is_reference_data = TRUE;
CREATE INDEX idx_p028_stc_active            ON pack028_sector_pathway.gl_sector_technology_catalog(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p028_stc_sector_cat        ON pack028_sector_pathway.gl_sector_technology_catalog(sector_code, technology_category);
CREATE INDEX idx_p028_stc_created           ON pack028_sector_pathway.gl_sector_technology_catalog(created_at DESC);
CREATE INDEX idx_p028_stc_prereqs           ON pack028_sector_pathway.gl_sector_technology_catalog USING GIN(prerequisites);
CREATE INDEX idx_p028_stc_cost_traj         ON pack028_sector_pathway.gl_sector_technology_catalog USING GIN(cost_decline_trajectory);
CREATE INDEX idx_p028_stc_metadata          ON pack028_sector_pathway.gl_sector_technology_catalog USING GIN(metadata);

-- Trigger
CREATE TRIGGER trg_p028_technology_catalog_updated
    BEFORE UPDATE ON pack028_sector_pathway.gl_sector_technology_catalog
    FOR EACH ROW EXECUTE FUNCTION pack028_sector_pathway.fn_set_updated_at();

-- RLS
ALTER TABLE pack028_sector_pathway.gl_sector_technology_catalog ENABLE ROW LEVEL SECURITY;

CREATE POLICY p028_stc_tenant_isolation
    ON pack028_sector_pathway.gl_sector_technology_catalog
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p028_stc_service_bypass
    ON pack028_sector_pathway.gl_sector_technology_catalog
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

GRANT SELECT, INSERT, UPDATE, DELETE ON pack028_sector_pathway.gl_sector_technology_catalog TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack028_sector_pathway.gl_sector_emission_factors IS
    'Sector-specific emission factors from IPCC, EPA, DEFRA, and industry sources with vintage tracking, regional variants, GHG breakdown, and uncertainty ranges.';

COMMENT ON TABLE pack028_sector_pathway.gl_sector_activity_data IS
    'Sector-specific activity data for intensity metric calculation including production volumes, energy consumption, transport metrics, and building area data.';

COMMENT ON TABLE pack028_sector_pathway.gl_sector_technology_catalog IS
    'Sector-specific technology catalog with TRL levels, cost projections, emission reduction potential, and IEA scenario alignment for technology roadmap generation.';
