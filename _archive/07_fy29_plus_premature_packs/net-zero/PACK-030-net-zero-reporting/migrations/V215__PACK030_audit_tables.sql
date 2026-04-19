-- =============================================================================
-- V215: PACK-030 Net Zero Reporting Pack - Audit Trail & Data Lineage
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    005 of 015
-- Date:         March 2026
--
-- Immutable audit trail for all report lifecycle events and data lineage
-- tracking for source-to-report metric provenance.
--
-- Tables (2):
--   1. pack030_nz_reporting.gl_nz_audit_trail
--   2. pack030_nz_reporting.gl_nz_data_lineage
--
-- Previous: V214__PACK030_assurance_tables.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack030_nz_reporting.gl_nz_audit_trail
-- =============================================================================
-- Immutable audit trail recording all report lifecycle events (create,
-- update, approve, publish), actor identification, IP address, user agent,
-- change detail tracking, and cryptographic integrity.

CREATE TABLE pack030_nz_reporting.gl_nz_audit_trail (
    audit_id                    UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    report_id                   UUID            REFERENCES pack030_nz_reporting.gl_nz_reports(report_id) ON DELETE SET NULL,
    -- Event classification
    event_type                  VARCHAR(100)    NOT NULL,
    event_category              VARCHAR(50)     NOT NULL DEFAULT 'REPORT',
    event_severity              VARCHAR(20)     NOT NULL DEFAULT 'INFO',
    -- Actor
    actor_id                    UUID            NOT NULL,
    actor_type                  VARCHAR(50)     NOT NULL,
    actor_name                  VARCHAR(255),
    actor_role                  VARCHAR(100),
    -- Client info
    ip_address                  INET,
    user_agent                  TEXT,
    session_id                  VARCHAR(100),
    -- Change details
    resource_type               VARCHAR(50),
    resource_id                 UUID,
    change_summary              TEXT,
    previous_state              JSONB,
    new_state                   JSONB,
    changed_fields              JSONB           NOT NULL DEFAULT '[]',
    -- Framework context
    framework                   VARCHAR(50),
    section_type                VARCHAR(100),
    -- Integrity
    event_hash                  CHAR(64),
    previous_event_hash         CHAR(64),
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Metadata
    details                     JSONB           NOT NULL DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_at_event_type CHECK (
        event_type IN (
            'REPORT_CREATED', 'REPORT_UPDATED', 'REPORT_SUBMITTED_FOR_REVIEW',
            'REPORT_APPROVED', 'REPORT_REJECTED', 'REPORT_PUBLISHED', 'REPORT_ARCHIVED',
            'REPORT_VERSIONED', 'REPORT_DELETED',
            'SECTION_CREATED', 'SECTION_UPDATED', 'SECTION_APPROVED', 'SECTION_DELETED',
            'METRIC_ADDED', 'METRIC_UPDATED', 'METRIC_DELETED', 'METRIC_VERIFIED',
            'NARRATIVE_CREATED', 'NARRATIVE_UPDATED', 'NARRATIVE_REVIEWED',
            'NARRATIVE_TRANSLATED', 'NARRATIVE_CONSISTENCY_CHECK',
            'FRAMEWORK_MAPPING_APPLIED', 'FRAMEWORK_MAPPING_UPDATED',
            'XBRL_TAGGED', 'XBRL_VALIDATED',
            'EVIDENCE_UPLOADED', 'EVIDENCE_REVIEWED', 'EVIDENCE_REJECTED',
            'EVIDENCE_BUNDLE_GENERATED',
            'VALIDATION_RUN', 'VALIDATION_RESOLVED',
            'PDF_GENERATED', 'HTML_GENERATED', 'EXCEL_GENERATED',
            'XBRL_GENERATED', 'IXBRL_GENERATED', 'JSON_GENERATED',
            'DASHBOARD_VIEWED', 'DASHBOARD_EXPORTED',
            'DATA_AGGREGATED', 'DATA_RECONCILED',
            'CONFIG_CHANGED', 'BRANDING_UPDATED',
            'USER_ACCESS_GRANTED', 'USER_ACCESS_REVOKED',
            'SYSTEM_EVENT', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p030_at_event_category CHECK (
        event_category IN ('REPORT', 'SECTION', 'METRIC', 'NARRATIVE', 'FRAMEWORK',
                           'XBRL', 'EVIDENCE', 'VALIDATION', 'OUTPUT', 'DASHBOARD',
                           'DATA', 'CONFIG', 'ACCESS', 'SYSTEM')
    ),
    CONSTRAINT chk_p030_at_event_severity CHECK (
        event_severity IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    ),
    CONSTRAINT chk_p030_at_actor_type CHECK (
        actor_type IN ('USER', 'SYSTEM', 'API', 'SCHEDULER', 'INTEGRATION', 'ADMIN')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_audit_trail
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_at_tenant               ON pack030_nz_reporting.gl_nz_audit_trail(tenant_id);
CREATE INDEX idx_p030_at_org                  ON pack030_nz_reporting.gl_nz_audit_trail(organization_id);
CREATE INDEX idx_p030_at_report               ON pack030_nz_reporting.gl_nz_audit_trail(report_id);
CREATE INDEX idx_p030_at_report_time          ON pack030_nz_reporting.gl_nz_audit_trail(report_id, created_at DESC);
CREATE INDEX idx_p030_at_event_type           ON pack030_nz_reporting.gl_nz_audit_trail(event_type);
CREATE INDEX idx_p030_at_event_category       ON pack030_nz_reporting.gl_nz_audit_trail(event_category);
CREATE INDEX idx_p030_at_event_severity       ON pack030_nz_reporting.gl_nz_audit_trail(event_severity);
CREATE INDEX idx_p030_at_actor                ON pack030_nz_reporting.gl_nz_audit_trail(actor_id);
CREATE INDEX idx_p030_at_actor_type           ON pack030_nz_reporting.gl_nz_audit_trail(actor_type);
CREATE INDEX idx_p030_at_resource             ON pack030_nz_reporting.gl_nz_audit_trail(resource_type, resource_id);
CREATE INDEX idx_p030_at_framework            ON pack030_nz_reporting.gl_nz_audit_trail(framework);
CREATE INDEX idx_p030_at_event_hash           ON pack030_nz_reporting.gl_nz_audit_trail(event_hash);
CREATE INDEX idx_p030_at_org_time             ON pack030_nz_reporting.gl_nz_audit_trail(organization_id, created_at DESC);
CREATE INDEX idx_p030_at_created              ON pack030_nz_reporting.gl_nz_audit_trail(created_at DESC);
CREATE INDEX idx_p030_at_errors               ON pack030_nz_reporting.gl_nz_audit_trail(organization_id, created_at DESC) WHERE event_severity IN ('ERROR', 'CRITICAL');
CREATE INDEX idx_p030_at_details              ON pack030_nz_reporting.gl_nz_audit_trail USING GIN(details);
CREATE INDEX idx_p030_at_changed_fields       ON pack030_nz_reporting.gl_nz_audit_trail USING GIN(changed_fields);

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_audit_trail (no trigger - immutable)
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_audit_trail ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_at_tenant_isolation
    ON pack030_nz_reporting.gl_nz_audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_at_service_bypass
    ON pack030_nz_reporting.gl_nz_audit_trail
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- =============================================================================
-- Table 2: pack030_nz_reporting.gl_nz_data_lineage
-- =============================================================================
-- Data lineage tracking from source system through transformations to final
-- reported metric, with transformation step recording, source record
-- references, and visual lineage diagram support.

CREATE TABLE pack030_nz_reporting.gl_nz_data_lineage (
    lineage_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    report_id                   UUID            NOT NULL REFERENCES pack030_nz_reporting.gl_nz_reports(report_id) ON DELETE CASCADE,
    metric_id                   UUID            REFERENCES pack030_nz_reporting.gl_nz_report_metrics(metric_id) ON DELETE SET NULL,
    -- Metric reference
    metric_name                 VARCHAR(200)    NOT NULL,
    metric_value                NUMERIC,
    metric_unit                 VARCHAR(50),
    -- Source system
    source_system               VARCHAR(100)    NOT NULL,
    source_type                 VARCHAR(50)     NOT NULL,
    source_pack                 VARCHAR(50),
    source_app                  VARCHAR(50),
    source_table                VARCHAR(200),
    source_record_ids           JSONB           NOT NULL DEFAULT '[]',
    source_query                TEXT,
    -- Transformation
    transformation_steps        JSONB           NOT NULL DEFAULT '[]',
    transformation_count        INTEGER         NOT NULL DEFAULT 0,
    -- Calculation details
    calculation_formula         TEXT,
    emission_factors_used       JSONB           NOT NULL DEFAULT '[]',
    conversion_factors_used     JSONB           NOT NULL DEFAULT '[]',
    -- Aggregation
    aggregation_method          VARCHAR(30),
    aggregation_level           VARCHAR(50),
    records_aggregated          INTEGER,
    -- Data quality
    data_quality_tier           VARCHAR(20),
    confidence_level            DECIMAL(5,2),
    -- Temporal
    data_period_start           DATE,
    data_period_end             DATE,
    data_extraction_at          TIMESTAMPTZ,
    -- Diagram support
    lineage_graph               JSONB           NOT NULL DEFAULT '{}',
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           NOT NULL DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_dl_source_type CHECK (
        source_type IN ('DATABASE', 'API', 'FILE', 'MANUAL', 'CALCULATED', 'ESTIMATED', 'EXTERNAL')
    ),
    CONSTRAINT chk_p030_dl_aggregation_method CHECK (
        aggregation_method IS NULL OR aggregation_method IN ('SUM', 'AVERAGE', 'WEIGHTED_AVERAGE', 'MAX', 'MIN', 'MEDIAN', 'CUSTOM')
    ),
    CONSTRAINT chk_p030_dl_data_quality CHECK (
        data_quality_tier IS NULL OR data_quality_tier IN ('TIER_1', 'TIER_2', 'TIER_3', 'ESTIMATED', 'MEASURED', 'CALCULATED')
    ),
    CONSTRAINT chk_p030_dl_confidence CHECK (
        confidence_level IS NULL OR (confidence_level >= 0 AND confidence_level <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_data_lineage
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_dl_tenant               ON pack030_nz_reporting.gl_nz_data_lineage(tenant_id);
CREATE INDEX idx_p030_dl_org                  ON pack030_nz_reporting.gl_nz_data_lineage(organization_id);
CREATE INDEX idx_p030_dl_report               ON pack030_nz_reporting.gl_nz_data_lineage(report_id);
CREATE INDEX idx_p030_dl_metric               ON pack030_nz_reporting.gl_nz_data_lineage(metric_id);
CREATE INDEX idx_p030_dl_report_metric        ON pack030_nz_reporting.gl_nz_data_lineage(report_id, metric_name);
CREATE INDEX idx_p030_dl_metric_name          ON pack030_nz_reporting.gl_nz_data_lineage(metric_name);
CREATE INDEX idx_p030_dl_source_system        ON pack030_nz_reporting.gl_nz_data_lineage(source_system);
CREATE INDEX idx_p030_dl_source_type          ON pack030_nz_reporting.gl_nz_data_lineage(source_type);
CREATE INDEX idx_p030_dl_source_pack          ON pack030_nz_reporting.gl_nz_data_lineage(source_pack);
CREATE INDEX idx_p030_dl_source_app           ON pack030_nz_reporting.gl_nz_data_lineage(source_app);
CREATE INDEX idx_p030_dl_data_quality         ON pack030_nz_reporting.gl_nz_data_lineage(data_quality_tier);
CREATE INDEX idx_p030_dl_active               ON pack030_nz_reporting.gl_nz_data_lineage(report_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p030_dl_created              ON pack030_nz_reporting.gl_nz_data_lineage(created_at DESC);
CREATE INDEX idx_p030_dl_transformation       ON pack030_nz_reporting.gl_nz_data_lineage USING GIN(transformation_steps);
CREATE INDEX idx_p030_dl_source_records       ON pack030_nz_reporting.gl_nz_data_lineage USING GIN(source_record_ids);
CREATE INDEX idx_p030_dl_emission_factors     ON pack030_nz_reporting.gl_nz_data_lineage USING GIN(emission_factors_used);
CREATE INDEX idx_p030_dl_metadata             ON pack030_nz_reporting.gl_nz_data_lineage USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger: gl_nz_data_lineage
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p030_data_lineage_updated
    BEFORE UPDATE ON pack030_nz_reporting.gl_nz_data_lineage
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_data_lineage
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_data_lineage ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_dl_tenant_isolation
    ON pack030_nz_reporting.gl_nz_data_lineage
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_dl_service_bypass
    ON pack030_nz_reporting.gl_nz_data_lineage
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT ON pack030_nz_reporting.gl_nz_audit_trail TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_data_lineage TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack030_nz_reporting.gl_nz_audit_trail IS
    'Immutable audit trail recording all report lifecycle events with actor identification, IP tracking, change detail capture, cryptographic event chaining, and severity classification for compliance and forensic analysis.';

COMMENT ON COLUMN pack030_nz_reporting.gl_nz_audit_trail.audit_id IS 'Unique audit event identifier.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_audit_trail.event_type IS 'Event type classification covering report, section, metric, narrative, framework, XBRL, evidence, validation, output, and system events.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_audit_trail.event_hash IS 'SHA-256 hash of event content for immutability verification.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_audit_trail.previous_event_hash IS 'Hash of prior event for chain integrity verification.';

COMMENT ON TABLE pack030_nz_reporting.gl_nz_data_lineage IS
    'Data lineage tracking from source system through transformation steps to final reported metric, with source record references, calculation formula documentation, emission factor tracking, aggregation details, and visual lineage diagram support.';

COMMENT ON COLUMN pack030_nz_reporting.gl_nz_data_lineage.lineage_id IS 'Unique lineage record identifier.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_data_lineage.source_system IS 'Source system providing the raw data (e.g., PACK-021, GL-GHG-APP, ERP).';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_data_lineage.transformation_steps IS 'JSONB array of transformation steps applied from source to reported value.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_data_lineage.lineage_graph IS 'JSONB graph structure for visual lineage diagram rendering.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_data_lineage.provenance_hash IS 'SHA-256 hash for lineage integrity and audit provenance.';
