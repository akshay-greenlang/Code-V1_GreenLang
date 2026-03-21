-- =============================================================================
-- V212: PACK-030 Net Zero Reporting Pack - Framework Tables
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    002 of 015
-- Date:         March 2026
--
-- Framework schema definitions, cross-framework metric mappings, and
-- framework submission deadline tracking.
--
-- Tables (3):
--   1. pack030_nz_reporting.gl_nz_framework_schemas
--   2. pack030_nz_reporting.gl_nz_framework_mappings
--   3. pack030_nz_reporting.gl_nz_framework_deadlines
--
-- Previous: V211__PACK030_core_tables.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack030_nz_reporting.gl_nz_framework_schemas
-- =============================================================================
-- Framework schema definitions with JSON Schema validation, version tracking,
-- effective/deprecated dates, and required/optional field classification.

CREATE TABLE pack030_nz_reporting.gl_nz_framework_schemas (
    schema_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    -- Framework identification
    framework                   VARCHAR(50)     NOT NULL,
    version                     VARCHAR(30)     NOT NULL,
    schema_type                 VARCHAR(50)     NOT NULL,
    -- Schema content
    schema_name                 VARCHAR(200)    NOT NULL,
    schema_description          TEXT,
    json_schema                 JSONB           NOT NULL,
    -- Field requirements
    required_fields             JSONB           NOT NULL DEFAULT '[]',
    optional_fields             JSONB           NOT NULL DEFAULT '[]',
    total_field_count           INTEGER         NOT NULL DEFAULT 0,
    -- Lifecycle
    effective_date              DATE            NOT NULL,
    deprecated_date             DATE,
    is_current                  BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Versioning
    previous_version_id         UUID,
    change_summary              TEXT,
    -- Source
    source_url                  VARCHAR(500),
    source_organization         VARCHAR(200),
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Metadata
    metadata                    JSONB           NOT NULL DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_fs_framework CHECK (
        framework IN ('SBTi', 'CDP', 'TCFD', 'GRI', 'ISSB', 'SEC', 'CSRD', 'CUSTOM')
    ),
    CONSTRAINT chk_p030_fs_schema_type CHECK (
        schema_type IN ('QUESTIONNAIRE', 'REPORT', 'TAXONOMY', 'DISCLOSURE', 'SUBMISSION', 'DATA_TABLE')
    ),
    CONSTRAINT chk_p030_fs_deprecated_after_effective CHECK (
        deprecated_date IS NULL OR deprecated_date >= effective_date
    ),
    CONSTRAINT uq_p030_fs_framework_version_type UNIQUE (framework, version, schema_type)
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_framework_schemas
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_fs_tenant               ON pack030_nz_reporting.gl_nz_framework_schemas(tenant_id);
CREATE INDEX idx_p030_fs_framework            ON pack030_nz_reporting.gl_nz_framework_schemas(framework);
CREATE INDEX idx_p030_fs_framework_version    ON pack030_nz_reporting.gl_nz_framework_schemas(framework, version);
CREATE INDEX idx_p030_fs_schema_type          ON pack030_nz_reporting.gl_nz_framework_schemas(schema_type);
CREATE INDEX idx_p030_fs_current              ON pack030_nz_reporting.gl_nz_framework_schemas(framework, schema_type) WHERE is_current = TRUE;
CREATE INDEX idx_p030_fs_effective            ON pack030_nz_reporting.gl_nz_framework_schemas(effective_date);
CREATE INDEX idx_p030_fs_active               ON pack030_nz_reporting.gl_nz_framework_schemas(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p030_fs_not_deprecated       ON pack030_nz_reporting.gl_nz_framework_schemas(framework) WHERE deprecated_date IS NULL;
CREATE INDEX idx_p030_fs_created              ON pack030_nz_reporting.gl_nz_framework_schemas(created_at DESC);
CREATE INDEX idx_p030_fs_json_schema          ON pack030_nz_reporting.gl_nz_framework_schemas USING GIN(json_schema);
CREATE INDEX idx_p030_fs_required_fields      ON pack030_nz_reporting.gl_nz_framework_schemas USING GIN(required_fields);
CREATE INDEX idx_p030_fs_metadata             ON pack030_nz_reporting.gl_nz_framework_schemas USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger: gl_nz_framework_schemas
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p030_framework_schemas_updated
    BEFORE UPDATE ON pack030_nz_reporting.gl_nz_framework_schemas
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_framework_schemas
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_framework_schemas ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_fs_tenant_isolation
    ON pack030_nz_reporting.gl_nz_framework_schemas
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_fs_service_bypass
    ON pack030_nz_reporting.gl_nz_framework_schemas
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- =============================================================================
-- Table 2: pack030_nz_reporting.gl_nz_framework_mappings
-- =============================================================================
-- Cross-framework metric mappings with source-to-target translation,
-- mapping confidence, conversion formulas, and bidirectional sync support.

CREATE TABLE pack030_nz_reporting.gl_nz_framework_mappings (
    mapping_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    -- Source mapping
    source_framework            VARCHAR(50)     NOT NULL,
    source_metric               VARCHAR(200)    NOT NULL,
    source_section              VARCHAR(100),
    source_element_ref          VARCHAR(200),
    -- Target mapping
    target_framework            VARCHAR(50)     NOT NULL,
    target_metric               VARCHAR(200)    NOT NULL,
    target_section              VARCHAR(100),
    target_element_ref          VARCHAR(200),
    -- Mapping classification
    mapping_type                VARCHAR(50)     NOT NULL,
    mapping_direction           VARCHAR(20)     NOT NULL DEFAULT 'UNIDIRECTIONAL',
    -- Conversion
    conversion_formula          TEXT,
    conversion_notes            TEXT,
    unit_conversion_required    BOOLEAN         NOT NULL DEFAULT FALSE,
    source_unit                 VARCHAR(50),
    target_unit                 VARCHAR(50),
    -- Confidence
    confidence_score            DECIMAL(5,2),
    validation_status           VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    validated_by                VARCHAR(255),
    validated_at                TIMESTAMPTZ,
    -- Usage tracking
    usage_count                 INTEGER         NOT NULL DEFAULT 0,
    last_used_at                TIMESTAMPTZ,
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           NOT NULL DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_fm_source_framework CHECK (
        source_framework IN ('SBTi', 'CDP', 'TCFD', 'GRI', 'ISSB', 'SEC', 'CSRD', 'GHG_PROTOCOL', 'INTERNAL', 'CUSTOM')
    ),
    CONSTRAINT chk_p030_fm_target_framework CHECK (
        target_framework IN ('SBTi', 'CDP', 'TCFD', 'GRI', 'ISSB', 'SEC', 'CSRD', 'GHG_PROTOCOL', 'INTERNAL', 'CUSTOM')
    ),
    CONSTRAINT chk_p030_fm_mapping_type CHECK (
        mapping_type IN ('DIRECT', 'CALCULATED', 'APPROXIMATE', 'PARTIAL', 'COMPOSITE', 'DERIVED', 'NO_EQUIVALENT')
    ),
    CONSTRAINT chk_p030_fm_mapping_direction CHECK (
        mapping_direction IN ('UNIDIRECTIONAL', 'BIDIRECTIONAL')
    ),
    CONSTRAINT chk_p030_fm_validation_status CHECK (
        validation_status IN ('PENDING', 'VALIDATED', 'REJECTED', 'NEEDS_REVIEW')
    ),
    CONSTRAINT chk_p030_fm_confidence CHECK (
        confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 100)
    ),
    CONSTRAINT chk_p030_fm_not_self_mapping CHECK (
        source_framework != target_framework OR source_metric != target_metric
    )
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_framework_mappings
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_fm_tenant               ON pack030_nz_reporting.gl_nz_framework_mappings(tenant_id);
CREATE INDEX idx_p030_fm_source_fw            ON pack030_nz_reporting.gl_nz_framework_mappings(source_framework);
CREATE INDEX idx_p030_fm_target_fw            ON pack030_nz_reporting.gl_nz_framework_mappings(target_framework);
CREATE INDEX idx_p030_fm_source_target        ON pack030_nz_reporting.gl_nz_framework_mappings(source_framework, target_framework);
CREATE INDEX idx_p030_fm_source_metric        ON pack030_nz_reporting.gl_nz_framework_mappings(source_framework, source_metric);
CREATE INDEX idx_p030_fm_target_metric        ON pack030_nz_reporting.gl_nz_framework_mappings(target_framework, target_metric);
CREATE INDEX idx_p030_fm_mapping_type         ON pack030_nz_reporting.gl_nz_framework_mappings(mapping_type);
CREATE INDEX idx_p030_fm_confidence           ON pack030_nz_reporting.gl_nz_framework_mappings(confidence_score DESC);
CREATE INDEX idx_p030_fm_validation           ON pack030_nz_reporting.gl_nz_framework_mappings(validation_status);
CREATE INDEX idx_p030_fm_validated            ON pack030_nz_reporting.gl_nz_framework_mappings(source_framework, target_framework) WHERE validation_status = 'VALIDATED';
CREATE INDEX idx_p030_fm_active               ON pack030_nz_reporting.gl_nz_framework_mappings(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p030_fm_bidirectional        ON pack030_nz_reporting.gl_nz_framework_mappings(source_framework, target_framework) WHERE mapping_direction = 'BIDIRECTIONAL';
CREATE INDEX idx_p030_fm_created              ON pack030_nz_reporting.gl_nz_framework_mappings(created_at DESC);
CREATE INDEX idx_p030_fm_metadata             ON pack030_nz_reporting.gl_nz_framework_mappings USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger: gl_nz_framework_mappings
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p030_framework_mappings_updated
    BEFORE UPDATE ON pack030_nz_reporting.gl_nz_framework_mappings
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_framework_mappings
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_framework_mappings ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_fm_tenant_isolation
    ON pack030_nz_reporting.gl_nz_framework_mappings
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_fm_service_bypass
    ON pack030_nz_reporting.gl_nz_framework_mappings
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- =============================================================================
-- Table 3: pack030_nz_reporting.gl_nz_framework_deadlines
-- =============================================================================
-- Framework submission deadline tracking with notification scheduling,
-- organization-specific overrides, and status tracking.

CREATE TABLE pack030_nz_reporting.gl_nz_framework_deadlines (
    deadline_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID,
    -- Framework
    framework                   VARCHAR(50)     NOT NULL,
    reporting_year              INTEGER         NOT NULL,
    -- Deadline
    deadline_date               DATE            NOT NULL,
    deadline_type               VARCHAR(30)     NOT NULL DEFAULT 'SUBMISSION',
    description                 TEXT,
    -- Notification
    notification_days           INTEGER[]       NOT NULL DEFAULT ARRAY[90, 60, 30, 14, 7],
    notification_sent           JSONB           NOT NULL DEFAULT '{}',
    -- Status
    submission_status           VARCHAR(30)     NOT NULL DEFAULT 'NOT_SUBMITTED',
    submitted_at                TIMESTAMPTZ,
    submitted_by                UUID,
    -- Extension
    extension_granted           BOOLEAN         NOT NULL DEFAULT FALSE,
    original_deadline           DATE,
    extension_reason            TEXT,
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           NOT NULL DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_fd_framework CHECK (
        framework IN ('SBTi', 'CDP', 'TCFD', 'GRI', 'ISSB', 'SEC', 'CSRD', 'CUSTOM')
    ),
    CONSTRAINT chk_p030_fd_deadline_type CHECK (
        deadline_type IN ('SUBMISSION', 'FILING', 'REVIEW', 'PUBLICATION', 'DATA_COLLECTION', 'INTERNAL')
    ),
    CONSTRAINT chk_p030_fd_submission_status CHECK (
        submission_status IN ('NOT_SUBMITTED', 'IN_PROGRESS', 'SUBMITTED', 'ACCEPTED', 'REJECTED', 'OVERDUE')
    ),
    CONSTRAINT chk_p030_fd_reporting_year CHECK (
        reporting_year >= 2000 AND reporting_year <= 2100
    )
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_framework_deadlines
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_fd_tenant               ON pack030_nz_reporting.gl_nz_framework_deadlines(tenant_id);
CREATE INDEX idx_p030_fd_org                  ON pack030_nz_reporting.gl_nz_framework_deadlines(organization_id);
CREATE INDEX idx_p030_fd_framework            ON pack030_nz_reporting.gl_nz_framework_deadlines(framework);
CREATE INDEX idx_p030_fd_fw_year              ON pack030_nz_reporting.gl_nz_framework_deadlines(framework, reporting_year);
CREATE INDEX idx_p030_fd_deadline_date        ON pack030_nz_reporting.gl_nz_framework_deadlines(deadline_date);
CREATE INDEX idx_p030_fd_upcoming             ON pack030_nz_reporting.gl_nz_framework_deadlines(deadline_date) WHERE deadline_date >= CURRENT_DATE AND submission_status != 'SUBMITTED';
CREATE INDEX idx_p030_fd_overdue              ON pack030_nz_reporting.gl_nz_framework_deadlines(deadline_date) WHERE deadline_date < CURRENT_DATE AND submission_status = 'NOT_SUBMITTED';
CREATE INDEX idx_p030_fd_submission           ON pack030_nz_reporting.gl_nz_framework_deadlines(submission_status);
CREATE INDEX idx_p030_fd_active               ON pack030_nz_reporting.gl_nz_framework_deadlines(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p030_fd_created              ON pack030_nz_reporting.gl_nz_framework_deadlines(created_at DESC);
CREATE INDEX idx_p030_fd_metadata             ON pack030_nz_reporting.gl_nz_framework_deadlines USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger: gl_nz_framework_deadlines
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p030_framework_deadlines_updated
    BEFORE UPDATE ON pack030_nz_reporting.gl_nz_framework_deadlines
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_framework_deadlines
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_framework_deadlines ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_fd_tenant_isolation
    ON pack030_nz_reporting.gl_nz_framework_deadlines
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_fd_service_bypass
    ON pack030_nz_reporting.gl_nz_framework_deadlines
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_framework_schemas TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_framework_mappings TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_framework_deadlines TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack030_nz_reporting.gl_nz_framework_schemas IS
    'Framework schema definitions with JSON Schema validation, version tracking, effective/deprecated lifecycle, required/optional field classification for SBTi, CDP, TCFD, GRI, ISSB, SEC, and CSRD frameworks.';

COMMENT ON TABLE pack030_nz_reporting.gl_nz_framework_mappings IS
    'Cross-framework metric mappings with source-to-target translation, mapping type classification (direct/calculated/approximate), conversion formulas, confidence scoring, and bidirectional sync support for 7-framework harmonization.';

COMMENT ON TABLE pack030_nz_reporting.gl_nz_framework_deadlines IS
    'Framework submission deadline tracking with configurable notification scheduling, organization-specific overrides, extension management, and submission status tracking.';
