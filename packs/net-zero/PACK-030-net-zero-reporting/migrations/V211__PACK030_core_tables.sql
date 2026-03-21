-- =============================================================================
-- V211: PACK-030 Net Zero Reporting Pack - Core Report Tables
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    001 of 015
-- Date:         March 2026
--
-- Creates the pack030_nz_reporting schema, the shared updated_at trigger
-- function, and the three core report tables: reports, report_sections,
-- and report_metrics.
--
-- Tables (3):
--   1. pack030_nz_reporting.gl_nz_reports
--   2. pack030_nz_reporting.gl_nz_report_sections
--   3. pack030_nz_reporting.gl_nz_report_metrics
--
-- Also includes: schema, update trigger function, per-table indexes,
-- per-table RLS, per-table triggers, grants, and column comments.
--
-- Previous: V210__PACK029_views_and_indexes.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Schema
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS pack030_nz_reporting;

-- ---------------------------------------------------------------------------
-- Trigger function: auto-update updated_at
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION pack030_nz_reporting.fn_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

-- =============================================================================
-- Table 1: pack030_nz_reporting.gl_nz_reports
-- =============================================================================
-- Core report metadata with framework identification, reporting period,
-- lifecycle status, SHA-256 provenance, and approval workflow tracking.

CREATE TABLE pack030_nz_reporting.gl_nz_reports (
    report_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    -- Framework identification
    framework                   VARCHAR(50)     NOT NULL,
    framework_version           VARCHAR(30),
    -- Reporting period
    reporting_period_start      DATE            NOT NULL,
    reporting_period_end        DATE            NOT NULL,
    reporting_year              INTEGER         NOT NULL,
    fiscal_year_end             DATE,
    -- Report classification
    report_type                 VARCHAR(50)     NOT NULL DEFAULT 'ANNUAL',
    report_title                VARCHAR(500),
    report_description          TEXT,
    -- Lifecycle status
    status                      VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    status_changed_at           TIMESTAMPTZ,
    status_changed_by           UUID,
    -- Data sources
    source_packs                JSONB           NOT NULL DEFAULT '[]',
    source_apps                 JSONB           NOT NULL DEFAULT '[]',
    data_completeness_pct       DECIMAL(5,2)    DEFAULT 0,
    -- Provenance
    provenance_hash             CHAR(64)        NOT NULL,
    content_hash                CHAR(64),
    -- Approval workflow
    created_by                  UUID            NOT NULL,
    approved_by                 UUID,
    approved_at                 TIMESTAMPTZ,
    published_by                UUID,
    published_at                TIMESTAMPTZ,
    -- Branding
    branding_config             JSONB           NOT NULL DEFAULT '{}',
    -- Output tracking
    output_formats              JSONB           NOT NULL DEFAULT '[]',
    generated_files             JSONB           NOT NULL DEFAULT '[]',
    -- Versioning
    version_number              INTEGER         NOT NULL DEFAULT 1,
    previous_version_id         UUID,
    is_latest                   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    archived                    BOOLEAN         NOT NULL DEFAULT FALSE,
    archived_at                 TIMESTAMPTZ,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           NOT NULL DEFAULT '{}',
    tags                        JSONB           NOT NULL DEFAULT '[]',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_rpt_framework CHECK (
        framework IN ('SBTi', 'CDP', 'TCFD', 'GRI', 'ISSB', 'SEC', 'CSRD', 'MULTI_FRAMEWORK', 'CUSTOM')
    ),
    CONSTRAINT chk_p030_rpt_status CHECK (
        status IN ('DRAFT', 'IN_PROGRESS', 'REVIEW', 'APPROVED', 'PUBLISHED', 'ARCHIVED', 'REJECTED')
    ),
    CONSTRAINT chk_p030_rpt_report_type CHECK (
        report_type IN ('ANNUAL', 'INTERIM', 'QUARTERLY', 'AD_HOC', 'SUBMISSION', 'PROGRESS_UPDATE')
    ),
    CONSTRAINT chk_p030_rpt_period_valid CHECK (
        reporting_period_end > reporting_period_start
    ),
    CONSTRAINT chk_p030_rpt_reporting_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    ),
    CONSTRAINT chk_p030_rpt_completeness CHECK (
        data_completeness_pct >= 0 AND data_completeness_pct <= 100
    ),
    CONSTRAINT chk_p030_rpt_version CHECK (
        version_number >= 1
    )
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_reports
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_rpt_tenant              ON pack030_nz_reporting.gl_nz_reports(tenant_id);
CREATE INDEX idx_p030_rpt_org                 ON pack030_nz_reporting.gl_nz_reports(organization_id);
CREATE INDEX idx_p030_rpt_org_framework       ON pack030_nz_reporting.gl_nz_reports(organization_id, framework);
CREATE INDEX idx_p030_rpt_org_fw_year         ON pack030_nz_reporting.gl_nz_reports(organization_id, framework, reporting_year);
CREATE INDEX idx_p030_rpt_framework           ON pack030_nz_reporting.gl_nz_reports(framework);
CREATE INDEX idx_p030_rpt_status              ON pack030_nz_reporting.gl_nz_reports(status);
CREATE INDEX idx_p030_rpt_reporting_year      ON pack030_nz_reporting.gl_nz_reports(reporting_year);
CREATE INDEX idx_p030_rpt_period_start        ON pack030_nz_reporting.gl_nz_reports(reporting_period_start);
CREATE INDEX idx_p030_rpt_period_end          ON pack030_nz_reporting.gl_nz_reports(reporting_period_end);
CREATE INDEX idx_p030_rpt_provenance          ON pack030_nz_reporting.gl_nz_reports(provenance_hash);
CREATE INDEX idx_p030_rpt_latest              ON pack030_nz_reporting.gl_nz_reports(organization_id, framework) WHERE is_latest = TRUE;
CREATE INDEX idx_p030_rpt_active              ON pack030_nz_reporting.gl_nz_reports(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p030_rpt_published           ON pack030_nz_reporting.gl_nz_reports(organization_id, published_at DESC) WHERE status = 'PUBLISHED';
CREATE INDEX idx_p030_rpt_created             ON pack030_nz_reporting.gl_nz_reports(created_at DESC);
CREATE INDEX idx_p030_rpt_source_packs        ON pack030_nz_reporting.gl_nz_reports USING GIN(source_packs);
CREATE INDEX idx_p030_rpt_source_apps         ON pack030_nz_reporting.gl_nz_reports USING GIN(source_apps);
CREATE INDEX idx_p030_rpt_metadata            ON pack030_nz_reporting.gl_nz_reports USING GIN(metadata);
CREATE INDEX idx_p030_rpt_tags                ON pack030_nz_reporting.gl_nz_reports USING GIN(tags);
CREATE INDEX idx_p030_rpt_output_formats      ON pack030_nz_reporting.gl_nz_reports USING GIN(output_formats);

-- ---------------------------------------------------------------------------
-- Trigger: gl_nz_reports
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p030_reports_updated
    BEFORE UPDATE ON pack030_nz_reporting.gl_nz_reports
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_reports
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_rpt_tenant_isolation
    ON pack030_nz_reporting.gl_nz_reports
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_rpt_service_bypass
    ON pack030_nz_reporting.gl_nz_reports
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- =============================================================================
-- Table 2: pack030_nz_reporting.gl_nz_report_sections
-- =============================================================================
-- Report section content with framework-specific section typing, ordering,
-- language support, citation linkage, and consistency scoring.

CREATE TABLE pack030_nz_reporting.gl_nz_report_sections (
    section_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    report_id                   UUID            NOT NULL REFERENCES pack030_nz_reporting.gl_nz_reports(report_id) ON DELETE CASCADE,
    -- Section identification
    section_type                VARCHAR(100)    NOT NULL,
    section_code                VARCHAR(50),
    section_title               VARCHAR(500)    NOT NULL,
    section_order               INTEGER         NOT NULL,
    parent_section_id           UUID            REFERENCES pack030_nz_reporting.gl_nz_report_sections(section_id) ON DELETE SET NULL,
    depth_level                 INTEGER         NOT NULL DEFAULT 0,
    -- Content
    content                     TEXT            NOT NULL,
    content_format              VARCHAR(20)     NOT NULL DEFAULT 'MARKDOWN',
    word_count                  INTEGER,
    -- Language
    language                    VARCHAR(5)      NOT NULL DEFAULT 'en',
    is_translated               BOOLEAN         NOT NULL DEFAULT FALSE,
    source_language             VARCHAR(5),
    translation_quality_score   DECIMAL(5,2),
    -- Citations
    citations                   JSONB           NOT NULL DEFAULT '[]',
    citation_count              INTEGER         NOT NULL DEFAULT 0,
    -- Consistency
    consistency_score           DECIMAL(5,2),
    consistency_issues          JSONB           NOT NULL DEFAULT '[]',
    -- Review
    review_status               VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    reviewed_by                 UUID,
    reviewed_at                 TIMESTAMPTZ,
    review_comments             TEXT,
    -- Versioning
    version_number              INTEGER         NOT NULL DEFAULT 1,
    is_latest                   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Metadata
    metadata                    JSONB           NOT NULL DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_sec_content_format CHECK (
        content_format IN ('MARKDOWN', 'HTML', 'PLAIN_TEXT', 'RICH_TEXT')
    ),
    CONSTRAINT chk_p030_sec_language CHECK (
        language IN ('en', 'de', 'fr', 'es', 'pt', 'it', 'nl', 'ja', 'zh')
    ),
    CONSTRAINT chk_p030_sec_review_status CHECK (
        review_status IN ('DRAFT', 'PENDING_REVIEW', 'APPROVED', 'REJECTED', 'NEEDS_REVISION')
    ),
    CONSTRAINT chk_p030_sec_consistency_score CHECK (
        consistency_score IS NULL OR (consistency_score >= 0 AND consistency_score <= 100)
    ),
    CONSTRAINT chk_p030_sec_translation_quality CHECK (
        translation_quality_score IS NULL OR (translation_quality_score >= 0 AND translation_quality_score <= 100)
    ),
    CONSTRAINT chk_p030_sec_depth CHECK (
        depth_level >= 0 AND depth_level <= 10
    ),
    CONSTRAINT chk_p030_sec_order CHECK (
        section_order >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_report_sections
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_sec_tenant              ON pack030_nz_reporting.gl_nz_report_sections(tenant_id);
CREATE INDEX idx_p030_sec_report              ON pack030_nz_reporting.gl_nz_report_sections(report_id);
CREATE INDEX idx_p030_sec_report_order        ON pack030_nz_reporting.gl_nz_report_sections(report_id, section_order);
CREATE INDEX idx_p030_sec_report_type         ON pack030_nz_reporting.gl_nz_report_sections(report_id, section_type);
CREATE INDEX idx_p030_sec_report_lang         ON pack030_nz_reporting.gl_nz_report_sections(report_id, language);
CREATE INDEX idx_p030_sec_section_type        ON pack030_nz_reporting.gl_nz_report_sections(section_type);
CREATE INDEX idx_p030_sec_language            ON pack030_nz_reporting.gl_nz_report_sections(language);
CREATE INDEX idx_p030_sec_parent              ON pack030_nz_reporting.gl_nz_report_sections(parent_section_id);
CREATE INDEX idx_p030_sec_review_status       ON pack030_nz_reporting.gl_nz_report_sections(review_status);
CREATE INDEX idx_p030_sec_consistency         ON pack030_nz_reporting.gl_nz_report_sections(consistency_score);
CREATE INDEX idx_p030_sec_active              ON pack030_nz_reporting.gl_nz_report_sections(report_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p030_sec_latest              ON pack030_nz_reporting.gl_nz_report_sections(report_id, section_type) WHERE is_latest = TRUE;
CREATE INDEX idx_p030_sec_translated          ON pack030_nz_reporting.gl_nz_report_sections(report_id) WHERE is_translated = TRUE;
CREATE INDEX idx_p030_sec_created             ON pack030_nz_reporting.gl_nz_report_sections(created_at DESC);
CREATE INDEX idx_p030_sec_content_fts         ON pack030_nz_reporting.gl_nz_report_sections USING GIN(to_tsvector('english', content));
CREATE INDEX idx_p030_sec_citations           ON pack030_nz_reporting.gl_nz_report_sections USING GIN(citations);
CREATE INDEX idx_p030_sec_metadata            ON pack030_nz_reporting.gl_nz_report_sections USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger: gl_nz_report_sections
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p030_sections_updated
    BEFORE UPDATE ON pack030_nz_reporting.gl_nz_report_sections
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_report_sections
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_report_sections ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_sec_tenant_isolation
    ON pack030_nz_reporting.gl_nz_report_sections
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_sec_service_bypass
    ON pack030_nz_reporting.gl_nz_report_sections
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- =============================================================================
-- Table 3: pack030_nz_reporting.gl_nz_report_metrics
-- =============================================================================
-- Reported metrics with value, unit, scope, source system provenance,
-- calculation method documentation, and uncertainty range tracking.

CREATE TABLE pack030_nz_reporting.gl_nz_report_metrics (
    metric_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    report_id                   UUID            NOT NULL REFERENCES pack030_nz_reporting.gl_nz_reports(report_id) ON DELETE CASCADE,
    section_id                  UUID            REFERENCES pack030_nz_reporting.gl_nz_report_sections(section_id) ON DELETE SET NULL,
    -- Metric identification
    metric_name                 VARCHAR(200)    NOT NULL,
    metric_code                 VARCHAR(100),
    metric_category             VARCHAR(100),
    -- Value
    metric_value                NUMERIC         NOT NULL,
    unit                        VARCHAR(50)     NOT NULL,
    precision_decimals          INTEGER         DEFAULT 2,
    -- Scope
    scope                       VARCHAR(20),
    scope_category              VARCHAR(100),
    -- Source
    source_system               VARCHAR(100)    NOT NULL,
    source_pack                 VARCHAR(50),
    source_record_id            UUID,
    -- Calculation
    calculation_method          TEXT,
    emission_factor_used        DECIMAL(18,8),
    emission_factor_source      VARCHAR(200),
    -- Provenance
    provenance_hash             CHAR(64)        NOT NULL,
    -- Uncertainty
    uncertainty_lower           NUMERIC,
    uncertainty_upper           NUMERIC,
    uncertainty_confidence_pct  DECIMAL(5,2),
    -- Data quality
    data_quality_tier           VARCHAR(20),
    verification_status         VARCHAR(30)     DEFAULT 'UNVERIFIED',
    -- Comparison
    prior_period_value          NUMERIC,
    variance_from_prior         NUMERIC,
    variance_pct_from_prior     DECIMAL(8,4),
    -- Framework-specific
    framework_element_ref       VARCHAR(200),
    xbrl_tag                    VARCHAR(200),
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           NOT NULL DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_met_scope CHECK (
        scope IS NULL OR scope IN ('SCOPE_1', 'SCOPE_2', 'SCOPE_3', 'SCOPE_1_2', 'SCOPE_1_2_3', 'TOTAL')
    ),
    CONSTRAINT chk_p030_met_data_quality CHECK (
        data_quality_tier IS NULL OR data_quality_tier IN ('TIER_1', 'TIER_2', 'TIER_3', 'ESTIMATED', 'MEASURED', 'CALCULATED')
    ),
    CONSTRAINT chk_p030_met_verification CHECK (
        verification_status IN ('UNVERIFIED', 'SELF_VERIFIED', 'INTERNAL_AUDIT', 'EXTERNAL_LIMITED', 'EXTERNAL_REASONABLE')
    ),
    CONSTRAINT chk_p030_met_uncertainty_conf CHECK (
        uncertainty_confidence_pct IS NULL OR (uncertainty_confidence_pct >= 0 AND uncertainty_confidence_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_report_metrics
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_met_tenant              ON pack030_nz_reporting.gl_nz_report_metrics(tenant_id);
CREATE INDEX idx_p030_met_report              ON pack030_nz_reporting.gl_nz_report_metrics(report_id);
CREATE INDEX idx_p030_met_report_name         ON pack030_nz_reporting.gl_nz_report_metrics(report_id, metric_name);
CREATE INDEX idx_p030_met_report_scope        ON pack030_nz_reporting.gl_nz_report_metrics(report_id, scope);
CREATE INDEX idx_p030_met_section             ON pack030_nz_reporting.gl_nz_report_metrics(section_id);
CREATE INDEX idx_p030_met_metric_name         ON pack030_nz_reporting.gl_nz_report_metrics(metric_name);
CREATE INDEX idx_p030_met_metric_code         ON pack030_nz_reporting.gl_nz_report_metrics(metric_code);
CREATE INDEX idx_p030_met_scope               ON pack030_nz_reporting.gl_nz_report_metrics(scope);
CREATE INDEX idx_p030_met_source_system       ON pack030_nz_reporting.gl_nz_report_metrics(source_system);
CREATE INDEX idx_p030_met_source_pack         ON pack030_nz_reporting.gl_nz_report_metrics(source_pack);
CREATE INDEX idx_p030_met_provenance          ON pack030_nz_reporting.gl_nz_report_metrics(provenance_hash);
CREATE INDEX idx_p030_met_dq_tier             ON pack030_nz_reporting.gl_nz_report_metrics(data_quality_tier);
CREATE INDEX idx_p030_met_verification        ON pack030_nz_reporting.gl_nz_report_metrics(verification_status);
CREATE INDEX idx_p030_met_xbrl_tag            ON pack030_nz_reporting.gl_nz_report_metrics(xbrl_tag) WHERE xbrl_tag IS NOT NULL;
CREATE INDEX idx_p030_met_active              ON pack030_nz_reporting.gl_nz_report_metrics(report_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p030_met_created             ON pack030_nz_reporting.gl_nz_report_metrics(created_at DESC);
CREATE INDEX idx_p030_met_metadata            ON pack030_nz_reporting.gl_nz_report_metrics USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger: gl_nz_report_metrics
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p030_metrics_updated
    BEFORE UPDATE ON pack030_nz_reporting.gl_nz_report_metrics
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_report_metrics
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_report_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_met_tenant_isolation
    ON pack030_nz_reporting.gl_nz_report_metrics
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_met_service_bypass
    ON pack030_nz_reporting.gl_nz_report_metrics
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA pack030_nz_reporting TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_reports TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_report_sections TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_report_metrics TO PUBLIC;
GRANT EXECUTE ON FUNCTION pack030_nz_reporting.fn_set_updated_at() TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON SCHEMA pack030_nz_reporting IS
    'PACK-030 Net Zero Reporting Pack - Multi-framework climate disclosure reporting with automated report generation for SBTi, CDP, TCFD, GRI, ISSB, SEC, and CSRD frameworks, including narrative generation, XBRL tagging, assurance packaging, and interactive dashboards.';

COMMENT ON TABLE pack030_nz_reporting.gl_nz_reports IS
    'Core report metadata with framework identification, reporting period, lifecycle status (draft/review/approved/published), SHA-256 provenance, approval workflow, branding configuration, and multi-format output tracking.';

COMMENT ON COLUMN pack030_nz_reporting.gl_nz_reports.report_id IS 'Unique report identifier.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_reports.tenant_id IS 'Multi-tenant isolation key.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_reports.organization_id IS 'Reference to the organization owning this report.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_reports.framework IS 'Reporting framework: SBTi, CDP, TCFD, GRI, ISSB, SEC, CSRD, MULTI_FRAMEWORK, CUSTOM.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_reports.status IS 'Report lifecycle status: DRAFT, IN_PROGRESS, REVIEW, APPROVED, PUBLISHED, ARCHIVED, REJECTED.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_reports.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_reports.data_completeness_pct IS 'Percentage of required data fields populated (0-100).';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_reports.source_packs IS 'JSONB array of source pack IDs contributing data to this report.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_reports.source_apps IS 'JSONB array of source application IDs contributing data to this report.';

COMMENT ON TABLE pack030_nz_reporting.gl_nz_report_sections IS
    'Report section content with framework-specific section typing, hierarchical ordering, multi-language support, citation linkage, consistency scoring, and review workflow.';

COMMENT ON COLUMN pack030_nz_reporting.gl_nz_report_sections.section_id IS 'Unique section identifier.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_report_sections.section_type IS 'Framework-specific section type (e.g., governance, strategy, metrics, E1-1).';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_report_sections.consistency_score IS 'Cross-framework narrative consistency score (0-100).';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_report_sections.citations IS 'JSONB array of citation references linking content to source data.';

COMMENT ON TABLE pack030_nz_reporting.gl_nz_report_metrics IS
    'Reported metrics with numeric value, unit, GHG scope, source system provenance, calculation method documentation, XBRL tag mapping, uncertainty range, and data quality classification.';

COMMENT ON COLUMN pack030_nz_reporting.gl_nz_report_metrics.metric_id IS 'Unique metric identifier.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_report_metrics.metric_name IS 'Human-readable metric name (e.g., Total Scope 1 Emissions).';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_report_metrics.provenance_hash IS 'SHA-256 hash for calculation provenance and audit trail.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_report_metrics.scope IS 'GHG Protocol scope: SCOPE_1, SCOPE_2, SCOPE_3, SCOPE_1_2, SCOPE_1_2_3, TOTAL.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_report_metrics.xbrl_tag IS 'XBRL element reference for SEC/CSRD digital taxonomy tagging.';
