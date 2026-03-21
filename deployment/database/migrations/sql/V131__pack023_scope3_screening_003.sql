-- =============================================================================
-- V131: PACK-023-sbti-alignment-003: Scope 3 Screening and Coverage Tracking
-- =============================================================================
-- Pack:         PACK-023 (SBTi Alignment Pack)
-- Date:         March 2026
--
-- Pack-level tables for Scope 3 materiality screening and coverage tracking.
-- Covers 15-category Scope 3 assessment with 40% materiality trigger,
-- 67%/90% coverage requirements for near-term/long-term targets, supplier
-- engagement target validation, and per-category data quality scoring.
--
-- EXTENDS:
--   V087: GL-SBTi-APP v1.0 (scope 3 baseline)
--   V080: Scope 3 Category Mapper Service
--   V129: PACK-023 Target Definitions
--
-- These tables sit in the pack023_sbti_alignment schema and provide
-- the Scope 3 materiality and coverage assessment layer for the pack.
-- =============================================================================
-- Tables (4):
--   1. pack023_scope3_materiality_screening  - Scope 3 materiality assessment
--   2. pack023_scope3_category_details       - Per-category emissions breakdown
--   3. pack023_scope3_coverage_analysis      - Coverage calculation and tracking
--   4. pack023_scope3_supplier_engagement    - Supplier engagement target details
--
-- Hypertables (1):
--   pack023_scope3_materiality_screening on screening_date (chunk: 3 months)
--
-- Also includes: 40+ indexes, update triggers, security grants, and comments.
-- Previous: V130__pack023_validation_002.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack023_sbti_alignment.pack023_scope3_materiality_screening
-- =============================================================================
-- Scope 3 materiality screening assessment identifying which categories are
-- material (above 40% threshold) and require targets, with overall assessment.

CREATE TABLE pack023_sbti_alignment.pack023_scope3_materiality_screening (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    screening_date          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    reporting_year          INTEGER         NOT NULL,
    target_type             VARCHAR(30),
    total_scope1_emissions  DECIMAL(18,6),
    total_scope2_emissions  DECIMAL(18,6),
    total_scope3_emissions  DECIMAL(18,6),
    total_emissions         DECIMAL(18,6),
    scope3_percentage       DECIMAL(6,2),
    materiality_threshold   DECIMAL(6,2)    DEFAULT 40.0,
    is_material             BOOLEAN         DEFAULT FALSE,
    material_threshold_met  BOOLEAN         DEFAULT FALSE,
    total_categories        INTEGER         DEFAULT 15,
    material_categories     INTEGER         DEFAULT 0,
    material_category_list  VARCHAR(50)[],
    categories_with_data    INTEGER         DEFAULT 0,
    categories_with_gap     INTEGER         DEFAULT 0,
    data_quality_assessment VARCHAR(30),
    assessment_comments     TEXT,
    assessed_by             VARCHAR(255),
    assessment_notes        TEXT,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_s3_threshold CHECK (
        scope3_percentage >= 0 AND scope3_percentage <= 100
    ),
    CONSTRAINT chk_pk_s3_mat_threshold CHECK (
        materiality_threshold > 0 AND materiality_threshold <= 100
    ),
    CONSTRAINT chk_pk_s3_categories CHECK (
        total_categories = 15
    )
);

-- Hypertable
SELECT create_hypertable(
    'pack023_sbti_alignment.pack023_scope3_materiality_screening',
    'screening_date',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '3 months'
);

-- Indexes
CREATE INDEX idx_pk_s3_tenant ON pack023_sbti_alignment.pack023_scope3_materiality_screening(tenant_id);
CREATE INDEX idx_pk_s3_org ON pack023_sbti_alignment.pack023_scope3_materiality_screening(org_id);
CREATE INDEX idx_pk_s3_date ON pack023_sbti_alignment.pack023_scope3_materiality_screening(screening_date DESC);
CREATE INDEX idx_pk_s3_year ON pack023_sbti_alignment.pack023_scope3_materiality_screening(reporting_year);
CREATE INDEX idx_pk_s3_type ON pack023_sbti_alignment.pack023_scope3_materiality_screening(target_type);
CREATE INDEX idx_pk_s3_material ON pack023_sbti_alignment.pack023_scope3_materiality_screening(is_material);
CREATE INDEX idx_pk_s3_mat_threshold ON pack023_sbti_alignment.pack023_scope3_materiality_screening(material_threshold_met);
CREATE INDEX idx_pk_s3_org_year ON pack023_sbti_alignment.pack023_scope3_materiality_screening(org_id, reporting_year);
CREATE INDEX idx_pk_s3_categories_list ON pack023_sbti_alignment.pack023_scope3_materiality_screening USING GIN(material_category_list);
CREATE INDEX idx_pk_s3_metadata ON pack023_sbti_alignment.pack023_scope3_materiality_screening USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_pk_s3_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_scope3_materiality_screening
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 2: pack023_sbti_alignment.pack023_scope3_category_details
-- =============================================================================
-- Per-category Scope 3 emissions breakdown with category-specific metrics,
-- data quality tier assessment, and materiality indicators.

CREATE TABLE pack023_sbti_alignment.pack023_scope3_category_details (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    materiality_screening_id UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_scope3_materiality_screening(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    category_number         INTEGER         NOT NULL,
    category_name           VARCHAR(200)    NOT NULL,
    category_description    TEXT,
    emissions_mt_co2e       DECIMAL(18,6),
    percentage_of_scope3    DECIMAL(6,2),
    percentage_of_total     DECIMAL(6,2),
    materiality_flag        BOOLEAN         DEFAULT FALSE,
    is_targeted             BOOLEAN         DEFAULT FALSE,
    data_quality_tier       VARCHAR(30),
    data_sources            VARCHAR(100)[],
    calculation_method      VARCHAR(100),
    activity_data_available BOOLEAN         DEFAULT FALSE,
    emission_factors_available BOOLEAN      DEFAULT FALSE,
    data_collection_status  VARCHAR(30),
    data_gap_description    TEXT,
    supplier_engagement_applicable BOOLEAN DEFAULT FALSE,
    supplier_count          INTEGER,
    supplier_coverage_percentage DECIMAL(6,2),
    primary_activity        VARCHAR(500),
    risks_identified        TEXT[],
    uncertainties           DECIMAL(6,2),
    comments                TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_cat_number CHECK (
        category_number >= 1 AND category_number <= 15
    ),
    CONSTRAINT chk_pk_cat_percentage CHECK (
        percentage_of_scope3 IS NULL OR (percentage_of_scope3 >= 0 AND percentage_of_scope3 <= 100)
    ),
    CONSTRAINT chk_pk_cat_total CHECK (
        percentage_of_total IS NULL OR (percentage_of_total >= 0 AND percentage_of_total <= 100)
    ),
    CONSTRAINT chk_pk_cat_tier CHECK (
        data_quality_tier IN ('PRIMARY', 'SECONDARY', 'PROXY', 'SPEND', 'NONE')
    )
);

-- Indexes
CREATE INDEX idx_pk_cat_screening_id ON pack023_sbti_alignment.pack023_scope3_category_details(materiality_screening_id);
CREATE INDEX idx_pk_cat_tenant ON pack023_sbti_alignment.pack023_scope3_category_details(tenant_id);
CREATE INDEX idx_pk_cat_org ON pack023_sbti_alignment.pack023_scope3_category_details(org_id);
CREATE INDEX idx_pk_cat_number ON pack023_sbti_alignment.pack023_scope3_category_details(category_number);
CREATE INDEX idx_pk_cat_material ON pack023_sbti_alignment.pack023_scope3_category_details(materiality_flag);
CREATE INDEX idx_pk_cat_targeted ON pack023_sbti_alignment.pack023_scope3_category_details(is_targeted);
CREATE INDEX idx_pk_cat_tier ON pack023_sbti_alignment.pack023_scope3_category_details(data_quality_tier);
CREATE INDEX idx_pk_cat_status ON pack023_sbti_alignment.pack023_scope3_category_details(data_collection_status);
CREATE INDEX idx_pk_cat_created_at ON pack023_sbti_alignment.pack023_scope3_category_details(created_at DESC);
CREATE INDEX idx_pk_cat_sources ON pack023_sbti_alignment.pack023_scope3_category_details USING GIN(data_sources);
CREATE INDEX idx_pk_cat_risks ON pack023_sbti_alignment.pack023_scope3_category_details USING GIN(risks_identified);

-- Updated_at trigger
CREATE TRIGGER trg_pk_cat_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_scope3_category_details
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 3: pack023_sbti_alignment.pack023_scope3_coverage_analysis
-- =============================================================================
-- Coverage calculation and tracking for Scope 3 targets showing % of emissions
-- covered by targets and assessment against 67%/90% thresholds.

CREATE TABLE pack023_sbti_alignment.pack023_scope3_coverage_analysis (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    materiality_screening_id UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_scope3_materiality_screening(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    target_type             VARCHAR(30),
    analysis_date           TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    total_scope3_emissions  DECIMAL(18,6),
    targeted_scope3_emissions DECIMAL(18,6),
    untargeted_scope3_emissions DECIMAL(18,6),
    coverage_percentage     DECIMAL(6,2),
    required_coverage_pct   DECIMAL(6,2),
    coverage_requirement_met BOOLEAN        DEFAULT FALSE,
    near_term_requirement   DECIMAL(6,2)    DEFAULT 67.0,
    long_term_requirement   DECIMAL(6,2)    DEFAULT 90.0,
    gap_to_requirement      DECIMAL(6,2),
    categories_targeted     INTEGER,
    categories_with_targets INTEGER,
    remaining_work          TEXT,
    target_setting_status   VARCHAR(30),
    coverage_assessment     VARCHAR(500),
    assessed_by             VARCHAR(255),
    approved_by             VARCHAR(255),
    approved_date           DATE,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_cov_pct CHECK (
        coverage_percentage >= 0 AND coverage_percentage <= 100
    ),
    CONSTRAINT chk_pk_cov_requirement CHECK (
        required_coverage_pct > 0 AND required_coverage_pct <= 100
    )
);

-- Indexes
CREATE INDEX idx_pk_cov_screening_id ON pack023_sbti_alignment.pack023_scope3_coverage_analysis(materiality_screening_id);
CREATE INDEX idx_pk_cov_tenant ON pack023_sbti_alignment.pack023_scope3_coverage_analysis(tenant_id);
CREATE INDEX idx_pk_cov_org ON pack023_sbti_alignment.pack023_scope3_coverage_analysis(org_id);
CREATE INDEX idx_pk_cov_type ON pack023_sbti_alignment.pack023_scope3_coverage_analysis(target_type);
CREATE INDEX idx_pk_cov_date ON pack023_sbti_alignment.pack023_scope3_coverage_analysis(analysis_date DESC);
CREATE INDEX idx_pk_cov_percentage ON pack023_sbti_alignment.pack023_scope3_coverage_analysis(coverage_percentage);
CREATE INDEX idx_pk_cov_met ON pack023_sbti_alignment.pack023_scope3_coverage_analysis(coverage_requirement_met);
CREATE INDEX idx_pk_cov_created_at ON pack023_sbti_alignment.pack023_scope3_coverage_analysis(created_at DESC);
CREATE INDEX idx_pk_cov_metadata ON pack023_sbti_alignment.pack023_scope3_coverage_analysis USING GIN(metadata);

-- Updated_at trigger
CREATE TRIGGER trg_pk_cov_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_scope3_coverage_analysis
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Table 4: pack023_sbti_alignment.pack023_scope3_supplier_engagement
-- =============================================================================
-- Supplier engagement target details and progress tracking for Scope 3
-- category-specific targets requiring supplier emissions disclosure.

CREATE TABLE pack023_sbti_alignment.pack023_scope3_supplier_engagement (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    scope3_category_id      UUID            NOT NULL REFERENCES pack023_sbti_alignment.pack023_scope3_category_details(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    org_id                  UUID            NOT NULL,
    category_number         INTEGER,
    category_name           VARCHAR(200),
    engagement_applicable   BOOLEAN         DEFAULT FALSE,
    engagement_required     BOOLEAN         DEFAULT FALSE,
    target_supplier_count   INTEGER,
    target_emissions_coverage DECIMAL(6,2),
    engagement_objective    TEXT,
    engagement_method       VARCHAR(200)[],
    communication_approach  VARCHAR(500),
    disclosure_expectation  TEXT,
    target_disclosure_pct   DECIMAL(6,2),
    suppliers_contacted     INTEGER         DEFAULT 0,
    suppliers_responded     INTEGER         DEFAULT 0,
    response_rate_percentage DECIMAL(6,2),
    suppliers_with_targets  INTEGER         DEFAULT 0,
    suppliers_with_sda      INTEGER         DEFAULT 0,
    average_ambition        VARCHAR(50),
    engagement_start_date   DATE,
    engagement_end_date     DATE,
    engagement_progress     VARCHAR(30),
    progress_percentage     DECIMAL(6,2),
    challenges              TEXT[],
    mitigation_actions      TEXT[],
    evidence_of_engagement  TEXT[],
    status                  VARCHAR(30)     DEFAULT 'planned',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_pk_supp_coverage CHECK (
        target_emissions_coverage IS NULL OR (target_emissions_coverage > 0 AND target_emissions_coverage <= 100)
    ),
    CONSTRAINT chk_pk_supp_disclosure CHECK (
        target_disclosure_pct IS NULL OR (target_disclosure_pct >= 0 AND target_disclosure_pct <= 100)
    ),
    CONSTRAINT chk_pk_supp_status CHECK (
        status IN ('planned', 'in_progress', 'on_track', 'at_risk', 'complete', 'discontinued')
    )
);

-- Indexes
CREATE INDEX idx_pk_supp_cat_id ON pack023_sbti_alignment.pack023_scope3_supplier_engagement(scope3_category_id);
CREATE INDEX idx_pk_supp_tenant ON pack023_sbti_alignment.pack023_scope3_supplier_engagement(tenant_id);
CREATE INDEX idx_pk_supp_org ON pack023_sbti_alignment.pack023_scope3_supplier_engagement(org_id);
CREATE INDEX idx_pk_supp_category ON pack023_sbti_alignment.pack023_scope3_supplier_engagement(category_number);
CREATE INDEX idx_pk_supp_applicable ON pack023_sbti_alignment.pack023_scope3_supplier_engagement(engagement_applicable);
CREATE INDEX idx_pk_supp_required ON pack023_sbti_alignment.pack023_scope3_supplier_engagement(engagement_required);
CREATE INDEX idx_pk_supp_status ON pack023_sbti_alignment.pack023_scope3_supplier_engagement(status);
CREATE INDEX idx_pk_supp_progress ON pack023_sbti_alignment.pack023_scope3_supplier_engagement(progress_percentage);
CREATE INDEX idx_pk_supp_created_at ON pack023_sbti_alignment.pack023_scope3_supplier_engagement(created_at DESC);
CREATE INDEX idx_pk_supp_methods ON pack023_sbti_alignment.pack023_scope3_supplier_engagement USING GIN(engagement_method);
CREATE INDEX idx_pk_supp_challenges ON pack023_sbti_alignment.pack023_scope3_supplier_engagement USING GIN(challenges);

-- Updated_at trigger
CREATE TRIGGER trg_pk_supp_updated_at
    BEFORE UPDATE ON pack023_sbti_alignment.pack023_scope3_supplier_engagement
    FOR EACH ROW
    EXECUTE FUNCTION pack023_sbti_alignment.set_updated_at();

-- =============================================================================
-- Permissions & Grants
-- =============================================================================

GRANT USAGE ON SCHEMA pack023_sbti_alignment TO public;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA pack023_sbti_alignment TO public;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA pack023_sbti_alignment TO public;

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE pack023_sbti_alignment.pack023_scope3_materiality_screening IS
'Scope 3 materiality assessment determining which categories are material (>40%) and require targets, with overall assessment results.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_scope3_category_details IS
'Detailed per-category emissions breakdown for all 15 Scope 3 categories with data quality tier, collection status, and materiality flags.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_scope3_coverage_analysis IS
'Coverage calculation and tracking showing % of Scope 3 emissions covered by targets against 67%/90% thresholds for near-term/long-term.';

COMMENT ON TABLE pack023_sbti_alignment.pack023_scope3_supplier_engagement IS
'Supplier engagement target details and progress tracking for categories requiring supplier emissions disclosure and target setting.';
