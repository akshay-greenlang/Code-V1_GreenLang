-- =============================================================================
-- V172: PACK-027 Enterprise Net Zero - Scope 4 Avoided Emissions Projects
-- =============================================================================
-- Pack:         PACK-027 (Enterprise Net Zero Pack)
-- Migration:    007 of 015
-- Date:         March 2026
--
-- Scope 4 avoided emissions quantification per WBCSD Avoided Emissions
-- Guidance. Tracks projects whose products/services displace higher-emission
-- alternatives with baseline scenario definition, attributional calculation,
-- additionality proof, and conservative estimation.
--
-- Tables (1):
--   1. pack027_enterprise_net_zero.gl_scope4_projects
--
-- Previous: V171__PACK027_carbon_pricing.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack027_enterprise_net_zero.gl_scope4_projects
-- =============================================================================

CREATE TABLE pack027_enterprise_net_zero.gl_scope4_projects (
    project_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id                  UUID            NOT NULL REFERENCES pack027_enterprise_net_zero.gl_enterprise_profiles(company_id) ON DELETE CASCADE,
    tenant_id                   UUID            NOT NULL,
    -- Project identification
    project_name                VARCHAR(500)    NOT NULL,
    project_code                VARCHAR(50),
    project_type                VARCHAR(50)     NOT NULL,
    description                 TEXT,
    -- Product/service details
    product_name                VARCHAR(255),
    product_category            VARCHAR(100),
    units_deployed              DECIMAL(18,2),
    unit_type                   VARCHAR(50),
    deployment_geography        TEXT[]          DEFAULT '{}',
    -- Avoided emissions
    avoided_tco2e               DECIMAL(18,4)   NOT NULL DEFAULT 0,
    avoided_per_unit_tco2e      DECIMAL(18,8),
    reporting_year              INTEGER         NOT NULL,
    lifetime_avoided_tco2e      DECIMAL(18,4),
    product_lifetime_years      INTEGER,
    -- Baseline scenario
    baseline_scenario           TEXT            NOT NULL,
    baseline_emissions_tco2e    DECIMAL(18,4)   NOT NULL,
    baseline_methodology        VARCHAR(100),
    baseline_data_source        TEXT,
    baseline_year               INTEGER,
    -- Project scenario
    project_scenario            TEXT            NOT NULL,
    project_emissions_tco2e     DECIMAL(18,4)   NOT NULL DEFAULT 0,
    project_methodology         VARCHAR(100),
    -- Additionality
    additionality_proof         TEXT,
    additionality_type          VARCHAR(50),
    additionality_verified      BOOLEAN         DEFAULT FALSE,
    -- Attribution
    attribution_approach        VARCHAR(30)     DEFAULT 'ATTRIBUTIONAL',
    attribution_share_pct       DECIMAL(6,2)    DEFAULT 100.00,
    double_counting_check       BOOLEAN         DEFAULT FALSE,
    -- Conservativeness
    conservativeness_adjustment DECIMAL(6,2)    DEFAULT 0,
    uncertainty_range_pct       DECIMAL(6,2),
    confidence_level            VARCHAR(20),
    -- Verification
    verification_status         VARCHAR(30)     DEFAULT 'unverified',
    verified_by                 VARCHAR(255),
    verification_date           DATE,
    verification_standard       VARCHAR(100),
    -- Governance
    approved_by                 VARCHAR(255),
    approval_date               DATE,
    -- Metadata
    calculation_details         JSONB           DEFAULT '{}',
    supporting_evidence         JSONB           DEFAULT '{}',
    warnings                    TEXT[]          DEFAULT '{}',
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p027_s4_project_type CHECK (
        project_type IN ('PRODUCT_SUBSTITUTION', 'EFFICIENCY_IMPROVEMENT', 'ENABLING_EFFECT',
                          'SYSTEMIC_CHANGE', 'RENEWABLE_ENERGY', 'ELECTRIFICATION',
                          'CIRCULAR_ECONOMY', 'CARBON_CAPTURE', 'OTHER')
    ),
    CONSTRAINT chk_p027_s4_avoided_non_neg CHECK (
        avoided_tco2e >= 0
    ),
    CONSTRAINT chk_p027_s4_baseline_positive CHECK (
        baseline_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p027_s4_project_non_neg CHECK (
        project_emissions_tco2e >= 0
    ),
    CONSTRAINT chk_p027_s4_avoided_consistent CHECK (
        avoided_tco2e <= baseline_emissions_tco2e
    ),
    CONSTRAINT chk_p027_s4_reporting_year CHECK (
        reporting_year >= 2015 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p027_s4_attribution CHECK (
        attribution_approach IN ('ATTRIBUTIONAL', 'CONSEQUENTIAL')
    ),
    CONSTRAINT chk_p027_s4_attribution_share CHECK (
        attribution_share_pct >= 0 AND attribution_share_pct <= 100
    ),
    CONSTRAINT chk_p027_s4_additionality_type CHECK (
        additionality_type IS NULL OR additionality_type IN (
            'REGULATORY', 'FINANCIAL', 'TECHNOLOGICAL', 'BARRIER', 'COMMON_PRACTICE', 'COMBINED'
        )
    ),
    CONSTRAINT chk_p027_s4_verification CHECK (
        verification_status IN ('unverified', 'self_verified', 'third_party_verified', 'pending', 'rejected')
    ),
    CONSTRAINT chk_p027_s4_confidence CHECK (
        confidence_level IS NULL OR confidence_level IN ('LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p027_s4_company            ON pack027_enterprise_net_zero.gl_scope4_projects(company_id);
CREATE INDEX idx_p027_s4_tenant             ON pack027_enterprise_net_zero.gl_scope4_projects(tenant_id);
CREATE INDEX idx_p027_s4_type               ON pack027_enterprise_net_zero.gl_scope4_projects(project_type);
CREATE INDEX idx_p027_s4_year               ON pack027_enterprise_net_zero.gl_scope4_projects(reporting_year);
CREATE INDEX idx_p027_s4_company_year       ON pack027_enterprise_net_zero.gl_scope4_projects(company_id, reporting_year);
CREATE INDEX idx_p027_s4_verification       ON pack027_enterprise_net_zero.gl_scope4_projects(verification_status);
CREATE INDEX idx_p027_s4_additionality      ON pack027_enterprise_net_zero.gl_scope4_projects(additionality_verified) WHERE additionality_verified = TRUE;
CREATE INDEX idx_p027_s4_attribution        ON pack027_enterprise_net_zero.gl_scope4_projects(attribution_approach);
CREATE INDEX idx_p027_s4_product            ON pack027_enterprise_net_zero.gl_scope4_projects(product_category);
CREATE INDEX idx_p027_s4_geography          ON pack027_enterprise_net_zero.gl_scope4_projects USING GIN(deployment_geography);
CREATE INDEX idx_p027_s4_created            ON pack027_enterprise_net_zero.gl_scope4_projects(created_at DESC);
CREATE INDEX idx_p027_s4_calculation        ON pack027_enterprise_net_zero.gl_scope4_projects USING GIN(calculation_details);
CREATE INDEX idx_p027_s4_metadata           ON pack027_enterprise_net_zero.gl_scope4_projects USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p027_scope4_projects_updated
    BEFORE UPDATE ON pack027_enterprise_net_zero.gl_scope4_projects
    FOR EACH ROW EXECUTE FUNCTION pack027_enterprise_net_zero.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack027_enterprise_net_zero.gl_scope4_projects ENABLE ROW LEVEL SECURITY;

CREATE POLICY p027_s4_tenant_isolation
    ON pack027_enterprise_net_zero.gl_scope4_projects
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p027_s4_service_bypass
    ON pack027_enterprise_net_zero.gl_scope4_projects
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack027_enterprise_net_zero.gl_scope4_projects TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack027_enterprise_net_zero.gl_scope4_projects IS
    'Scope 4 avoided emissions projects per WBCSD guidance with baseline/project scenarios, additionality proof, attribution, and conservative estimation.';

COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scope4_projects.project_id IS 'Unique Scope 4 project identifier.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scope4_projects.project_type IS 'Avoided emissions mechanism: PRODUCT_SUBSTITUTION, EFFICIENCY_IMPROVEMENT, ENABLING_EFFECT, etc.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scope4_projects.avoided_tco2e IS 'Total avoided emissions in tCO2e for the reporting year.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scope4_projects.baseline_scenario IS 'Description of what would happen without the project/product (counterfactual).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scope4_projects.project_scenario IS 'Description of actual emissions with the project/product deployed.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scope4_projects.additionality_proof IS 'Evidence that avoided emissions are additional to business-as-usual.';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scope4_projects.attribution_approach IS 'Attribution methodology: ATTRIBUTIONAL (product-level) or CONSEQUENTIAL (system-level).';
COMMENT ON COLUMN pack027_enterprise_net_zero.gl_scope4_projects.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
