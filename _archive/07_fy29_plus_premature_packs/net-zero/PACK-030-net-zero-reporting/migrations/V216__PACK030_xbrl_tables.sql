-- =============================================================================
-- V216: PACK-030 Net Zero Reporting Pack - XBRL Tagging Tables
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    006 of 015
-- Date:         March 2026
--
-- XBRL/iXBRL tagging for SEC and CSRD digital taxonomy compliance with
-- element mapping, namespace management, taxonomy version tracking, and
-- validation status.
--
-- Tables (1):
--   1. pack030_nz_reporting.gl_nz_xbrl_tags
--
-- Previous: V215__PACK030_audit_tables.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack030_nz_reporting.gl_nz_xbrl_tags
-- =============================================================================
-- XBRL/iXBRL tag definitions linking report metrics to official taxonomy
-- elements for SEC and CSRD digital reporting with namespace, context,
-- unit references, validation, and taxonomy version tracking.

CREATE TABLE pack030_nz_reporting.gl_nz_xbrl_tags (
    tag_id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    report_id                   UUID            NOT NULL REFERENCES pack030_nz_reporting.gl_nz_reports(report_id) ON DELETE CASCADE,
    metric_id                   UUID            REFERENCES pack030_nz_reporting.gl_nz_report_metrics(metric_id) ON DELETE SET NULL,
    section_id                  UUID            REFERENCES pack030_nz_reporting.gl_nz_report_sections(section_id) ON DELETE SET NULL,
    -- Metric reference
    metric_name                 VARCHAR(200)    NOT NULL,
    metric_value                TEXT,
    -- XBRL element
    xbrl_element                VARCHAR(200)    NOT NULL,
    xbrl_namespace              VARCHAR(500)    NOT NULL,
    xbrl_prefix                 VARCHAR(50),
    -- Taxonomy
    taxonomy_framework          VARCHAR(50)     NOT NULL,
    taxonomy_version            VARCHAR(50)     NOT NULL,
    taxonomy_url                VARCHAR(500),
    -- Context
    context_ref                 VARCHAR(200),
    context_period_start        DATE,
    context_period_end          DATE,
    context_instant             DATE,
    context_entity              VARCHAR(200),
    context_segment             JSONB,
    -- Unit
    unit_ref                    VARCHAR(100),
    unit_measure                VARCHAR(200),
    -- Precision
    decimals                    INTEGER,
    precision_value             INTEGER,
    -- Tag type
    tag_type                    VARCHAR(30)     NOT NULL DEFAULT 'NUMERIC',
    -- Inline XBRL
    ixbrl_format                VARCHAR(100),
    ixbrl_sign                  VARCHAR(10),
    ixbrl_scale                 INTEGER,
    -- Validation
    validation_status           VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    validation_errors           JSONB           NOT NULL DEFAULT '[]',
    validated_at                TIMESTAMPTZ,
    -- Extension taxonomy
    is_extension                BOOLEAN         NOT NULL DEFAULT FALSE,
    extension_definition        JSONB,
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           NOT NULL DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_xb_taxonomy_framework CHECK (
        taxonomy_framework IN ('SEC', 'CSRD', 'ISSB', 'ESEF', 'CUSTOM')
    ),
    CONSTRAINT chk_p030_xb_tag_type CHECK (
        tag_type IN ('NUMERIC', 'TEXT', 'DATE', 'BOOLEAN', 'ENUM', 'MONETARY', 'SHARES', 'PER_SHARE')
    ),
    CONSTRAINT chk_p030_xb_validation_status CHECK (
        validation_status IN ('PENDING', 'VALID', 'INVALID', 'WARNING', 'SKIPPED')
    ),
    CONSTRAINT chk_p030_xb_ixbrl_sign CHECK (
        ixbrl_sign IS NULL OR ixbrl_sign IN ('-', '')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_xbrl_tags
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_xb_tenant               ON pack030_nz_reporting.gl_nz_xbrl_tags(tenant_id);
CREATE INDEX idx_p030_xb_report               ON pack030_nz_reporting.gl_nz_xbrl_tags(report_id);
CREATE INDEX idx_p030_xb_metric               ON pack030_nz_reporting.gl_nz_xbrl_tags(metric_id);
CREATE INDEX idx_p030_xb_section              ON pack030_nz_reporting.gl_nz_xbrl_tags(section_id);
CREATE INDEX idx_p030_xb_report_element       ON pack030_nz_reporting.gl_nz_xbrl_tags(report_id, xbrl_element);
CREATE INDEX idx_p030_xb_element              ON pack030_nz_reporting.gl_nz_xbrl_tags(xbrl_element);
CREATE INDEX idx_p030_xb_namespace            ON pack030_nz_reporting.gl_nz_xbrl_tags(xbrl_namespace);
CREATE INDEX idx_p030_xb_taxonomy_fw          ON pack030_nz_reporting.gl_nz_xbrl_tags(taxonomy_framework);
CREATE INDEX idx_p030_xb_taxonomy_version     ON pack030_nz_reporting.gl_nz_xbrl_tags(taxonomy_framework, taxonomy_version);
CREATE INDEX idx_p030_xb_context_ref          ON pack030_nz_reporting.gl_nz_xbrl_tags(context_ref);
CREATE INDEX idx_p030_xb_tag_type             ON pack030_nz_reporting.gl_nz_xbrl_tags(tag_type);
CREATE INDEX idx_p030_xb_validation           ON pack030_nz_reporting.gl_nz_xbrl_tags(validation_status);
CREATE INDEX idx_p030_xb_invalid              ON pack030_nz_reporting.gl_nz_xbrl_tags(report_id) WHERE validation_status = 'INVALID';
CREATE INDEX idx_p030_xb_extensions           ON pack030_nz_reporting.gl_nz_xbrl_tags(report_id) WHERE is_extension = TRUE;
CREATE INDEX idx_p030_xb_active               ON pack030_nz_reporting.gl_nz_xbrl_tags(report_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p030_xb_created              ON pack030_nz_reporting.gl_nz_xbrl_tags(created_at DESC);
CREATE INDEX idx_p030_xb_validation_errors    ON pack030_nz_reporting.gl_nz_xbrl_tags USING GIN(validation_errors);
CREATE INDEX idx_p030_xb_context_segment      ON pack030_nz_reporting.gl_nz_xbrl_tags USING GIN(context_segment);
CREATE INDEX idx_p030_xb_metadata             ON pack030_nz_reporting.gl_nz_xbrl_tags USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger: gl_nz_xbrl_tags
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p030_xbrl_tags_updated
    BEFORE UPDATE ON pack030_nz_reporting.gl_nz_xbrl_tags
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_xbrl_tags
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_xbrl_tags ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_xb_tenant_isolation
    ON pack030_nz_reporting.gl_nz_xbrl_tags
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_xb_service_bypass
    ON pack030_nz_reporting.gl_nz_xbrl_tags
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_xbrl_tags TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack030_nz_reporting.gl_nz_xbrl_tags IS
    'XBRL/iXBRL tag definitions linking report metrics to official SEC, CSRD, and ISSB taxonomy elements with namespace management, context/unit references, inline XBRL formatting, validation tracking, and extension taxonomy support.';

COMMENT ON COLUMN pack030_nz_reporting.gl_nz_xbrl_tags.tag_id IS 'Unique XBRL tag identifier.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_xbrl_tags.xbrl_element IS 'XBRL element name from official taxonomy (e.g., ifrs-full:GrossScope1Emissions).';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_xbrl_tags.xbrl_namespace IS 'XBRL namespace URI for the taxonomy element.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_xbrl_tags.taxonomy_framework IS 'Target taxonomy framework: SEC, CSRD, ISSB, ESEF, CUSTOM.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_xbrl_tags.taxonomy_version IS 'Taxonomy version identifier (e.g., 2024, 2025-Q1).';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_xbrl_tags.context_ref IS 'XBRL context reference linking to reporting period and entity.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_xbrl_tags.validation_status IS 'Tag validation status: PENDING, VALID, INVALID, WARNING, SKIPPED.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_xbrl_tags.is_extension IS 'Whether this tag uses an extension taxonomy element.';
