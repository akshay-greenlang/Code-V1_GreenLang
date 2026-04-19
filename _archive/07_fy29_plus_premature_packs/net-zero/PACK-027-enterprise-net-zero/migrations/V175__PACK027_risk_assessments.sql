-- =============================================================================
-- V175: PACK-027 Enterprise Net Zero - Climate Risk Assessments
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    010 of 015
-- Date:         March 2026
--
-- Climate risk assessment framework with physical and transition risk
-- modeling, asset-level exposure scoring, financial impact quantification,
-- and mitigation strategy tracking per TCFD/ISSB S2 requirements.
--
-- Tables (2):
--   1. pack027_enterprise_net_zero.gl_climate_risks
--   2. pack027_enterprise_net_zero.gl_asset_risk_exposure
--
-- Previous: V174__PACK027_financial_integration.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack027_enterprise_net_zero.gl_climate_risks
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_climate_risks (
    risk_id                     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    -- Risk identification
    risk_name                   VARCHAR(500)    NOT NULL,
    risk_type                   VARCHAR(30)     NOT NULL,
    risk_subtype                VARCHAR(50),
    risk_category               VARCHAR(50),
    description                 TEXT,
    -- Assessment
    severity                    VARCHAR(20)     NOT NULL,
    likelihood                  VARCHAR(20)     NOT NULL,
    velocity                    VARCHAR(20),
    time_horizon                VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM_TERM',
    -- Financial impact
    financial_impact_usd        DECIMAL(18,2),
    financial_impact_range_low  DECIMAL(18,2),
    financial_impact_range_high DECIMAL(18,2),
    impact_as_pct_revenue       DECIMAL(8,4),
    impact_type                 VARCHAR(30),
    -- Scenario linkage
    scenario_id                 UUID            REFERENCES pack027_enterprise_net_zero.gl_scenario_models(scenario_id) ON DELETE SET NULL,
    scenario_temperature        VARCHAR(10),
    rcp_pathway                 VARCHAR(10),
    -- Affected scope
    affected_geographies        TEXT[]          DEFAULT '{}',
    affected_business_units     TEXT[]          DEFAULT '{}',
    affected_asset_classes      TEXT[]          DEFAULT '{}',
    affected_value_chain        TEXT[]          DEFAULT '{}',
    -- Mitigation
    mitigation_strategy         TEXT,
    mitigation_status           VARCHAR(30)     DEFAULT 'IDENTIFIED',
    mitigation_cost_usd         DECIMAL(18,2),
    mitigation_timeline         VARCHAR(50),
    residual_risk_level         VARCHAR(20),
    -- Opportunity (risks may also present opportunities)
    has_opportunity              BOOLEAN         DEFAULT FALSE,
    opportunity_description     TEXT,
    opportunity_value_usd       DECIMAL(18,2),
    -- Governance
    risk_owner                  VARCHAR(255),
    review_frequency            VARCHAR(30)     DEFAULT 'ANNUAL',
    last_review_date            DATE,
    next_review_date            DATE,
    board_reported              BOOLEAN         DEFAULT FALSE,
    -- Regulatory mapping
    tcfd_category               VARCHAR(50),
    issb_s2_reference           VARCHAR(50),
    esrs_e1_reference           VARCHAR(50),
    -- Metadata
    assessment_methodology      VARCHAR(100),
    data_sources                JSONB           DEFAULT '{}',
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_cr_risk_type CHECK (
        risk_type IN ('PHYSICAL_ACUTE', 'PHYSICAL_CHRONIC', 'TRANSITION_POLICY',
                       'TRANSITION_LEGAL', 'TRANSITION_TECHNOLOGY', 'TRANSITION_MARKET',
                       'TRANSITION_REPUTATION', 'SYSTEMIC')
    ),
    CONSTRAINT chk_p027_cr_severity CHECK (
        severity IN ('NEGLIGIBLE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p027_cr_likelihood CHECK (
        likelihood IN ('RARE', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'ALMOST_CERTAIN')
    ),
    CONSTRAINT chk_p027_cr_velocity CHECK (
        velocity IS NULL OR velocity IN ('GRADUAL', 'MODERATE', 'RAPID', 'IMMEDIATE')
    ),
    CONSTRAINT chk_p027_cr_time_horizon CHECK (
        time_horizon IN ('SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM', 'BEYOND_2050')
    ),
    CONSTRAINT chk_p027_cr_impact_type CHECK (
        impact_type IS NULL OR impact_type IN (
            'REVENUE_LOSS', 'COST_INCREASE', 'ASSET_IMPAIRMENT', 'STRANDED_ASSET',
            'REGULATORY_FINE', 'LITIGATION', 'INSURANCE_COST', 'SUPPLY_DISRUPTION',
            'MARKET_ACCESS', 'REPUTATION', 'COMBINED'
        )
    ),
    CONSTRAINT chk_p027_cr_scenario_temp CHECK (
        scenario_temperature IS NULL OR scenario_temperature IN ('1.5C', '2.0C', '2.5C', '3.0C', '4.0C')
    ),
    CONSTRAINT chk_p027_cr_rcp CHECK (
        rcp_pathway IS NULL OR rcp_pathway IN ('RCP2.6', 'RCP4.5', 'RCP6.0', 'RCP8.5', 'SSP1', 'SSP2', 'SSP3', 'SSP5')
    ),
    CONSTRAINT chk_p027_cr_mitigation_status CHECK (
        mitigation_status IN ('IDENTIFIED', 'PLANNED', 'IN_PROGRESS', 'IMPLEMENTED', 'MONITORED', 'NOT_APPLICABLE')
    ),
    CONSTRAINT chk_p027_cr_residual CHECK (
        residual_risk_level IS NULL OR residual_risk_level IN ('NEGLIGIBLE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p027_cr_impact_range CHECK (
        financial_impact_range_low IS NULL OR financial_impact_range_high IS NULL
        OR financial_impact_range_low <= financial_impact_range_high
    ),
    CONSTRAINT chk_p027_cr_review_frequency CHECK (
        review_frequency IN ('QUARTERLY', 'SEMI_ANNUAL', 'ANNUAL', 'BIENNIAL', 'AD_HOC')
    )
);

-- =============================================================================
-- Table 2: pack027_enterprise_net_zero.gl_asset_risk_exposure
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_asset_risk_exposure (
    exposure_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    -- Asset identification
    asset_id                    UUID            NOT NULL,
    asset_name                  VARCHAR(500)    NOT NULL,
    asset_type                  VARCHAR(50)     NOT NULL,
    asset_location_country      VARCHAR(3),
    asset_location_region       VARCHAR(100),
    asset_location_lat          DECIMAL(10,7),
    asset_location_lon          DECIMAL(10,7),
    asset_book_value_usd        DECIMAL(18,2),
    asset_remaining_life_years  INTEGER,
    -- Physical risk scores (0-100)
    physical_risk_score         DECIMAL(5,2)    NOT NULL DEFAULT 0,
    flood_risk_score            DECIMAL(5,2)    DEFAULT 0,
    wildfire_risk_score         DECIMAL(5,2)    DEFAULT 0,
    hurricane_risk_score        DECIMAL(5,2)    DEFAULT 0,
    drought_risk_score          DECIMAL(5,2)    DEFAULT 0,
    sea_level_risk_score        DECIMAL(5,2)    DEFAULT 0,
    heat_stress_risk_score      DECIMAL(5,2)    DEFAULT 0,
    water_stress_risk_score     DECIMAL(5,2)    DEFAULT 0,
    -- Transition risk scores (0-100)
    transition_risk_score       DECIMAL(5,2)    NOT NULL DEFAULT 0,
    policy_risk_score           DECIMAL(5,2)    DEFAULT 0,
    technology_risk_score       DECIMAL(5,2)    DEFAULT 0,
    market_risk_score           DECIMAL(5,2)    DEFAULT 0,
    stranded_asset_risk_score   DECIMAL(5,2)    DEFAULT 0,
    -- Combined exposure
    total_exposure_usd          DECIMAL(18,2),
    physical_exposure_usd       DECIMAL(18,2),
    transition_exposure_usd     DECIMAL(18,2),
    -- Insurance
    insured                     BOOLEAN         DEFAULT FALSE,
    insurance_coverage_usd      DECIMAL(18,2),
    insurance_gap_usd           DECIMAL(18,2),
    -- Adaptation
    adaptation_measures         JSONB           DEFAULT '{}',
    adaptation_investment_usd   DECIMAL(18,2),
    -- Assessment metadata
    assessment_date             DATE            NOT NULL DEFAULT CURRENT_DATE,
    assessment_scenario         VARCHAR(10),
    assessment_time_horizon     VARCHAR(20),
    data_source                 VARCHAR(100),
    -- Metadata
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_are_asset_type CHECK (
        asset_type IN ('FACILITY', 'FACTORY', 'OFFICE', 'WAREHOUSE', 'DATA_CENTER',
                        'RETAIL_STORE', 'INFRASTRUCTURE', 'LAND', 'EQUIPMENT',
                        'VEHICLE_FLEET', 'PIPELINE', 'PORT', 'OTHER')
    ),
    CONSTRAINT chk_p027_are_physical_score CHECK (
        physical_risk_score >= 0 AND physical_risk_score <= 100
    ),
    CONSTRAINT chk_p027_are_transition_score CHECK (
        transition_risk_score >= 0 AND transition_risk_score <= 100
    ),
    CONSTRAINT chk_p027_are_sub_scores CHECK (
        (flood_risk_score IS NULL OR (flood_risk_score >= 0 AND flood_risk_score <= 100)) AND
        (wildfire_risk_score IS NULL OR (wildfire_risk_score >= 0 AND wildfire_risk_score <= 100)) AND
        (hurricane_risk_score IS NULL OR (hurricane_risk_score >= 0 AND hurricane_risk_score <= 100)) AND
        (drought_risk_score IS NULL OR (drought_risk_score >= 0 AND drought_risk_score <= 100)) AND
        (sea_level_risk_score IS NULL OR (sea_level_risk_score >= 0 AND sea_level_risk_score <= 100)) AND
        (heat_stress_risk_score IS NULL OR (heat_stress_risk_score >= 0 AND heat_stress_risk_score <= 100)) AND
        (water_stress_risk_score IS NULL OR (water_stress_risk_score >= 0 AND water_stress_risk_score <= 100)) AND
        (policy_risk_score IS NULL OR (policy_risk_score >= 0 AND policy_risk_score <= 100)) AND
        (technology_risk_score IS NULL OR (technology_risk_score >= 0 AND technology_risk_score <= 100)) AND
        (market_risk_score IS NULL OR (market_risk_score >= 0 AND market_risk_score <= 100)) AND
        (stranded_asset_risk_score IS NULL OR (stranded_asset_risk_score >= 0 AND stranded_asset_risk_score <= 100))
    ),
    CONSTRAINT chk_p027_are_exposure_non_neg CHECK (
        total_exposure_usd IS NULL OR total_exposure_usd >= 0
    ),
    CONSTRAINT chk_p027_are_assessment_scenario CHECK (
        assessment_scenario IS NULL OR assessment_scenario IN ('1.5C', '2.0C', '3.0C', '4.0C', 'BAU')
    ),
    CONSTRAINT chk_p027_are_time_horizon CHECK (
        assessment_time_horizon IS NULL OR assessment_time_horizon IN ('2030', '2040', '2050', '2100')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for gl_climate_risks
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_cr_company            ON pack027_enterprise_net_zero.gl_climate_risks(company_id);
CREATE INDEX idx_p027_cr_tenant             ON pack027_enterprise_net_zero.gl_climate_risks(tenant_id);
CREATE INDEX idx_p027_cr_type               ON pack027_enterprise_net_zero.gl_climate_risks(risk_type);
CREATE INDEX idx_p027_cr_severity           ON pack027_enterprise_net_zero.gl_climate_risks(severity);
CREATE INDEX idx_p027_cr_likelihood         ON pack027_enterprise_net_zero.gl_climate_risks(likelihood);
CREATE INDEX idx_p027_cr_time_horizon       ON pack027_enterprise_net_zero.gl_climate_risks(time_horizon);
CREATE INDEX idx_p027_cr_mitigation         ON pack027_enterprise_net_zero.gl_climate_risks(mitigation_status);
CREATE INDEX idx_p027_cr_scenario           ON pack027_enterprise_net_zero.gl_climate_risks(scenario_id);
CREATE INDEX idx_p027_cr_owner              ON pack027_enterprise_net_zero.gl_climate_risks(risk_owner);
CREATE INDEX idx_p027_cr_next_review        ON pack027_enterprise_net_zero.gl_climate_risks(next_review_date);
CREATE INDEX idx_p027_cr_board              ON pack027_enterprise_net_zero.gl_climate_risks(board_reported) WHERE board_reported = TRUE;
CREATE INDEX idx_p027_cr_geographies        ON pack027_enterprise_net_zero.gl_climate_risks USING GIN(affected_geographies);
CREATE INDEX idx_p027_cr_bus                ON pack027_enterprise_net_zero.gl_climate_risks USING GIN(affected_business_units);
CREATE INDEX idx_p027_cr_created            ON pack027_enterprise_net_zero.gl_climate_risks(created_at DESC);
CREATE INDEX idx_p027_cr_metadata           ON pack027_enterprise_net_zero.gl_climate_risks USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Indexes for gl_asset_risk_exposure
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_are_company           ON pack027_enterprise_net_zero.gl_asset_risk_exposure(company_id);
CREATE INDEX idx_p027_are_tenant            ON pack027_enterprise_net_zero.gl_asset_risk_exposure(tenant_id);
CREATE INDEX idx_p027_are_asset             ON pack027_enterprise_net_zero.gl_asset_risk_exposure(asset_id);
CREATE INDEX idx_p027_are_type              ON pack027_enterprise_net_zero.gl_asset_risk_exposure(asset_type);
CREATE INDEX idx_p027_are_country           ON pack027_enterprise_net_zero.gl_asset_risk_exposure(asset_location_country);
CREATE INDEX idx_p027_are_physical          ON pack027_enterprise_net_zero.gl_asset_risk_exposure(physical_risk_score DESC);
CREATE INDEX idx_p027_are_transition        ON pack027_enterprise_net_zero.gl_asset_risk_exposure(transition_risk_score DESC);
CREATE INDEX idx_p027_are_exposure          ON pack027_enterprise_net_zero.gl_asset_risk_exposure(total_exposure_usd DESC);
CREATE INDEX idx_p027_are_stranded          ON pack027_enterprise_net_zero.gl_asset_risk_exposure(stranded_asset_risk_score DESC);
CREATE INDEX idx_p027_are_assessment_date   ON pack027_enterprise_net_zero.gl_asset_risk_exposure(assessment_date DESC);
CREATE INDEX idx_p027_are_scenario          ON pack027_enterprise_net_zero.gl_asset_risk_exposure(assessment_scenario);
CREATE INDEX idx_p027_are_adaptation        ON pack027_enterprise_net_zero.gl_asset_risk_exposure USING GIN(adaptation_measures);
CREATE INDEX idx_p027_are_created           ON pack027_enterprise_net_zero.gl_asset_risk_exposure(created_at DESC);
CREATE INDEX idx_p027_are_metadata          ON pack027_enterprise_net_zero.gl_asset_risk_exposure USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Triggers
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p027_climate_risks_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_climate_risks
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

CREATE TRIGGER trg_p027_asset_risk_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_asset_risk_exposure
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack027_enterprise_net_zero.gl_climate_risks ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack027_enterprise_net_zero.gl_asset_risk_exposure ENABLE ROW LEVEL SECURITY;

CREATE POLICY p027_cr_tenant_isolation
    ON pack027_enterprise_net_zero.gl_climate_risks
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p027_cr_service_bypass
    ON pack027_enterprise_net_zero.gl_climate_risks
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p027_are_tenant_isolation
    ON pack027_enterprise_net_zero.gl_asset_risk_exposure
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p027_are_service_bypass
    ON pack027_enterprise_net_zero.gl_asset_risk_exposure
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_climate_risks TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_asset_risk_exposure TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack027_enterprise_net_zero.gl_climate_risks IS
    'Climate risk register with physical and transition risk assessments, financial impact quantification, mitigation tracking, and TCFD/ISSB S2 mapping.';
COMMENT ON TABLE pack027_enterprise_net_zero.gl_asset_risk_exposure IS
    'Asset-level climate risk exposure with physical hazard scores (flood, wildfire, drought, etc.), transition risk scores, and financial exposure quantification.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_climate_risks.risk_id IS 'Unique climate risk identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_climate_risks.risk_type IS 'Climate risk taxonomy: PHYSICAL_ACUTE, PHYSICAL_CHRONIC, TRANSITION_POLICY, TRANSITION_LEGAL, etc.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_climate_risks.severity IS 'Risk severity: NEGLIGIBLE, LOW, MEDIUM, HIGH, CRITICAL.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_climate_risks.financial_impact_usd IS 'Estimated financial impact in USD.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_climate_risks.mitigation_strategy IS 'Description of risk mitigation approach.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_asset_risk_exposure.exposure_id IS 'Unique asset risk exposure record identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_asset_risk_exposure.physical_risk_score IS 'Composite physical risk score (0-100).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_asset_risk_exposure.transition_risk_score IS 'Composite transition risk score (0-100).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_asset_risk_exposure.total_exposure_usd IS 'Total financial exposure in USD combining physical and transition risks.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_asset_risk_exposure.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
