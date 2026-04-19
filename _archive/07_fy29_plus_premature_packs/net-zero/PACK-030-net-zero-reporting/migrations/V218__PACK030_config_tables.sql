-- =============================================================================
-- V218: PACK-030 Net Zero Reporting Pack - Configuration & Dashboard Tables
-- =============================================================================
-- Pack:         PACK-030 (Net Zero Reporting Pack)
-- Migration:    008 of 015
-- Date:         March 2026
--
-- Organization-level report configuration with branding, notification
-- preferences, and stakeholder-specific dashboard view definitions.
--
-- Tables (2):
--   1. pack030_nz_reporting.gl_nz_report_config
--   2. pack030_nz_reporting.gl_nz_dashboard_views
--
-- Previous: V217__PACK030_validation_tables.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack030_nz_reporting.gl_nz_report_config
-- =============================================================================
-- Organization-level reporting configuration with framework-specific
-- preferences, branding (logo, colors, fonts), content customization,
-- notification channels, and output format preferences.

CREATE TABLE pack030_nz_reporting.gl_nz_report_config (
    config_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    -- Framework
    framework                   VARCHAR(50)     NOT NULL,
    -- Branding
    branding_config             JSONB           NOT NULL DEFAULT '{}',
    -- Content
    content_config              JSONB           NOT NULL DEFAULT '{}',
    -- Notification
    notification_config         JSONB           NOT NULL DEFAULT '{}',
    -- Output preferences
    output_config               JSONB           NOT NULL DEFAULT '{}',
    preferred_formats           JSONB           NOT NULL DEFAULT '["PDF"]',
    preferred_language          VARCHAR(5)      NOT NULL DEFAULT 'en',
    additional_languages        JSONB           NOT NULL DEFAULT '[]',
    -- Data sources
    data_source_config          JSONB           NOT NULL DEFAULT '{}',
    -- Assurance
    assurance_config            JSONB           NOT NULL DEFAULT '{}',
    assurance_standard          VARCHAR(50),
    assurance_level             VARCHAR(30),
    -- Dashboard
    dashboard_config            JSONB           NOT NULL DEFAULT '{}',
    -- Automation
    auto_generate               BOOLEAN         NOT NULL DEFAULT FALSE,
    auto_generate_schedule      VARCHAR(100),
    auto_validate               BOOLEAN         NOT NULL DEFAULT TRUE,
    auto_notify                 BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Defaults
    default_report_type         VARCHAR(50)     DEFAULT 'ANNUAL',
    default_review_workflow     VARCHAR(50)     DEFAULT 'STANDARD',
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           NOT NULL DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_rc_framework CHECK (
        framework IN ('SBTi', 'CDP', 'TCFD', 'GRI', 'ISSB', 'SEC', 'CSRD', 'MULTI_FRAMEWORK', 'CUSTOM')
    ),
    CONSTRAINT chk_p030_rc_language CHECK (
        preferred_language IN ('en', 'de', 'fr', 'es', 'pt', 'it', 'nl', 'ja', 'zh')
    ),
    CONSTRAINT chk_p030_rc_assurance_std CHECK (
        assurance_standard IS NULL OR assurance_standard IN (
            'ISAE_3410', 'ISAE_3000', 'AA1000AS', 'ISO_14064_3', 'SOC_2', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p030_rc_assurance_level CHECK (
        assurance_level IS NULL OR assurance_level IN ('LIMITED', 'REASONABLE', 'HIGH', 'MODERATE', 'NONE')
    ),
    CONSTRAINT uq_p030_rc_org_framework UNIQUE (organization_id, framework)
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_report_config
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_rc_tenant               ON pack030_nz_reporting.gl_nz_report_config(tenant_id);
CREATE INDEX idx_p030_rc_org                  ON pack030_nz_reporting.gl_nz_report_config(organization_id);
CREATE INDEX idx_p030_rc_org_fw               ON pack030_nz_reporting.gl_nz_report_config(organization_id, framework);
CREATE INDEX idx_p030_rc_framework            ON pack030_nz_reporting.gl_nz_report_config(framework);
CREATE INDEX idx_p030_rc_auto_generate        ON pack030_nz_reporting.gl_nz_report_config(organization_id) WHERE auto_generate = TRUE;
CREATE INDEX idx_p030_rc_active               ON pack030_nz_reporting.gl_nz_report_config(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p030_rc_created              ON pack030_nz_reporting.gl_nz_report_config(created_at DESC);
CREATE INDEX idx_p030_rc_branding             ON pack030_nz_reporting.gl_nz_report_config USING GIN(branding_config);
CREATE INDEX idx_p030_rc_content              ON pack030_nz_reporting.gl_nz_report_config USING GIN(content_config);
CREATE INDEX idx_p030_rc_notification         ON pack030_nz_reporting.gl_nz_report_config USING GIN(notification_config);
CREATE INDEX idx_p030_rc_output               ON pack030_nz_reporting.gl_nz_report_config USING GIN(output_config);
CREATE INDEX idx_p030_rc_metadata             ON pack030_nz_reporting.gl_nz_report_config USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger: gl_nz_report_config
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p030_report_config_updated
    BEFORE UPDATE ON pack030_nz_reporting.gl_nz_report_config
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_report_config
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_report_config ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_rc_tenant_isolation
    ON pack030_nz_reporting.gl_nz_report_config
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_rc_service_bypass
    ON pack030_nz_reporting.gl_nz_report_config
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- =============================================================================
-- Table 2: pack030_nz_reporting.gl_nz_dashboard_views
-- =============================================================================
-- Stakeholder-specific dashboard view definitions with layout configuration,
-- widget selection, filter preferences, and access control.

CREATE TABLE pack030_nz_reporting.gl_nz_dashboard_views (
    view_id                     UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    organization_id             UUID            NOT NULL,
    -- View identification
    view_type                   VARCHAR(50)     NOT NULL,
    view_name                   VARCHAR(200)    NOT NULL,
    view_description            TEXT,
    -- Layout
    layout_config               JSONB           NOT NULL DEFAULT '{}',
    widgets                     JSONB           NOT NULL DEFAULT '[]',
    -- Filters
    default_filters             JSONB           NOT NULL DEFAULT '{}',
    available_filters           JSONB           NOT NULL DEFAULT '[]',
    -- Frameworks displayed
    frameworks_shown            JSONB           NOT NULL DEFAULT '[]',
    -- Data scope
    date_range_default          VARCHAR(30)     DEFAULT 'CURRENT_YEAR',
    scopes_shown                JSONB           NOT NULL DEFAULT '["SCOPE_1", "SCOPE_2", "SCOPE_3"]',
    -- Access
    access_roles                JSONB           NOT NULL DEFAULT '[]',
    is_public                   BOOLEAN         NOT NULL DEFAULT FALSE,
    share_link                  VARCHAR(500),
    -- Branding
    branding_override           JSONB,
    -- Creator
    created_by                  UUID            NOT NULL,
    -- Status
    is_active                   BOOLEAN         NOT NULL DEFAULT TRUE,
    is_default                  BOOLEAN         NOT NULL DEFAULT FALSE,
    -- Metadata
    metadata                    JSONB           NOT NULL DEFAULT '{}',
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p030_dv_view_type CHECK (
        view_type IN ('EXECUTIVE', 'INVESTOR', 'REGULATOR', 'CUSTOMER', 'EMPLOYEE',
                       'AUDITOR', 'BOARD', 'FRAMEWORK_SPECIFIC', 'CUSTOM')
    ),
    CONSTRAINT chk_p030_dv_date_range CHECK (
        date_range_default IN ('CURRENT_YEAR', 'LAST_YEAR', 'LAST_3_YEARS', 'LAST_5_YEARS', 'ALL_TIME', 'CUSTOM')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes: gl_nz_dashboard_views
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p030_dv_tenant               ON pack030_nz_reporting.gl_nz_dashboard_views(tenant_id);
CREATE INDEX idx_p030_dv_org                  ON pack030_nz_reporting.gl_nz_dashboard_views(organization_id);
CREATE INDEX idx_p030_dv_org_type             ON pack030_nz_reporting.gl_nz_dashboard_views(organization_id, view_type);
CREATE INDEX idx_p030_dv_view_type            ON pack030_nz_reporting.gl_nz_dashboard_views(view_type);
CREATE INDEX idx_p030_dv_created_by           ON pack030_nz_reporting.gl_nz_dashboard_views(created_by);
CREATE INDEX idx_p030_dv_active               ON pack030_nz_reporting.gl_nz_dashboard_views(organization_id, is_active) WHERE is_active = TRUE;
CREATE INDEX idx_p030_dv_default              ON pack030_nz_reporting.gl_nz_dashboard_views(organization_id, view_type) WHERE is_default = TRUE;
CREATE INDEX idx_p030_dv_public               ON pack030_nz_reporting.gl_nz_dashboard_views(organization_id) WHERE is_public = TRUE;
CREATE INDEX idx_p030_dv_created              ON pack030_nz_reporting.gl_nz_dashboard_views(created_at DESC);
CREATE INDEX idx_p030_dv_frameworks           ON pack030_nz_reporting.gl_nz_dashboard_views USING GIN(frameworks_shown);
CREATE INDEX idx_p030_dv_widgets              ON pack030_nz_reporting.gl_nz_dashboard_views USING GIN(widgets);
CREATE INDEX idx_p030_dv_access_roles         ON pack030_nz_reporting.gl_nz_dashboard_views USING GIN(access_roles);
CREATE INDEX idx_p030_dv_metadata             ON pack030_nz_reporting.gl_nz_dashboard_views USING GIN(metadata);

-- ---------------------------------------------------------------------------
-- Trigger: gl_nz_dashboard_views
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p030_dashboard_views_updated
    BEFORE UPDATE ON pack030_nz_reporting.gl_nz_dashboard_views
    FOR EACH ROW EXECUTE FUNCTION pack030_nz_reporting.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security: gl_nz_dashboard_views
-- ---------------------------------------------------------------------------
ALTER TABLE pack030_nz_reporting.gl_nz_dashboard_views ENABLE ROW LEVEL SECURITY;

CREATE POLICY p030_dv_tenant_isolation
    ON pack030_nz_reporting.gl_nz_dashboard_views
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE POLICY p030_dv_service_bypass
    ON pack030_nz_reporting.gl_nz_dashboard_views
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_report_config TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack030_nz_reporting.gl_nz_dashboard_views TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack030_nz_reporting.gl_nz_report_config IS
    'Organization-level reporting configuration with framework-specific preferences, branding (logo/colors/fonts), content customization, notification channels, output format preferences, assurance settings, and automation scheduling.';

COMMENT ON COLUMN pack030_nz_reporting.gl_nz_report_config.config_id IS 'Unique configuration identifier.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_report_config.branding_config IS 'JSONB branding: logo_path, primary_color, secondary_color, font_family, style.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_report_config.notification_config IS 'JSONB notification preferences: email, slack, teams, deadline reminders.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_report_config.auto_generate IS 'Whether reports are auto-generated on schedule.';

COMMENT ON TABLE pack030_nz_reporting.gl_nz_dashboard_views IS
    'Stakeholder-specific dashboard view definitions with layout configuration, widget selection, filter preferences, framework scope, date range defaults, and access control for executive/investor/regulator/customer/employee audiences.';

COMMENT ON COLUMN pack030_nz_reporting.gl_nz_dashboard_views.view_id IS 'Unique dashboard view identifier.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_dashboard_views.view_type IS 'Stakeholder type: EXECUTIVE, INVESTOR, REGULATOR, CUSTOMER, EMPLOYEE, AUDITOR, BOARD, FRAMEWORK_SPECIFIC, CUSTOM.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_dashboard_views.widgets IS 'JSONB array of dashboard widget configurations.';
COMMENT ON COLUMN pack030_nz_reporting.gl_nz_dashboard_views.frameworks_shown IS 'JSONB array of frameworks to display in this view.';
