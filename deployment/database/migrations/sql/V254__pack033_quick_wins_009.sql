-- =============================================================================
-- V254: PACK-033 Quick Wins Identifier - Reporting Configuration
-- =============================================================================
-- Pack:         PACK-033 (Quick Wins Identifier Pack)
-- Migration:    009 of 010
-- Date:         March 2026
--
-- Creates reporting configuration tables for scheduled report generation
-- and configurable dashboard widgets for quick-win analytics.
--
-- Tables (2):
--   1. pack033_quick_wins.report_configs
--   2. pack033_quick_wins.dashboard_widgets
--
-- Previous: V253__pack033_quick_wins_008.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack033_quick_wins.report_configs
-- =============================================================================
-- Report configuration for scheduled and on-demand report generation with
-- recipient lists, filters, and output format preferences.

CREATE TABLE pack033_quick_wins.report_configs (
    config_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    report_type             VARCHAR(50)     NOT NULL,
    report_name             VARCHAR(500)    NOT NULL,
    schedule                VARCHAR(50),
    recipients              JSONB           DEFAULT '[]',
    filters                 JSONB           DEFAULT '{}',
    format                  VARCHAR(20)     NOT NULL DEFAULT 'PDF',
    last_generated_at       TIMESTAMPTZ,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_rc_report_type CHECK (
        report_type IN ('SCAN_SUMMARY', 'ACTION_PLAN', 'FINANCIAL_ANALYSIS', 'SAVINGS_PROGRESS',
                          'REBATE_STATUS', 'BEHAVIORAL_PROGRAM', 'EXECUTIVE_DASHBOARD',
                          'CARBON_IMPACT', 'PRIORITY_MATRIX', 'CUSTOM')
    ),
    CONSTRAINT chk_p033_rc_schedule CHECK (
        schedule IS NULL OR schedule IN ('DAILY', 'WEEKLY', 'BIWEEKLY', 'MONTHLY',
                                           'QUARTERLY', 'ANNUALLY', 'ON_DEMAND')
    ),
    CONSTRAINT chk_p033_rc_format CHECK (
        format IN ('PDF', 'XLSX', 'CSV', 'JSON', 'HTML', 'PPTX')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_rc_tenant        ON pack033_quick_wins.report_configs(tenant_id);
CREATE INDEX idx_p033_rc_report_type   ON pack033_quick_wins.report_configs(report_type);
CREATE INDEX idx_p033_rc_schedule      ON pack033_quick_wins.report_configs(schedule);
CREATE INDEX idx_p033_rc_format        ON pack033_quick_wins.report_configs(format);
CREATE INDEX idx_p033_rc_last_gen      ON pack033_quick_wins.report_configs(last_generated_at DESC);
CREATE INDEX idx_p033_rc_recipients    ON pack033_quick_wins.report_configs USING GIN(recipients);
CREATE INDEX idx_p033_rc_filters       ON pack033_quick_wins.report_configs USING GIN(filters);
CREATE INDEX idx_p033_rc_metadata      ON pack033_quick_wins.report_configs USING GIN(metadata);
CREATE INDEX idx_p033_rc_created       ON pack033_quick_wins.report_configs(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_rc_updated
    BEFORE UPDATE ON pack033_quick_wins.report_configs
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack033_quick_wins.dashboard_widgets
-- =============================================================================
-- Configurable dashboard widget definitions linked to report configs,
-- specifying widget type, data source, layout position, and rendering config.

CREATE TABLE pack033_quick_wins.dashboard_widgets (
    widget_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    config_id               UUID            NOT NULL REFERENCES pack033_quick_wins.report_configs(config_id) ON DELETE CASCADE,
    widget_type             VARCHAR(50)     NOT NULL,
    title                   VARCHAR(255)    NOT NULL,
    data_source             VARCHAR(100)    NOT NULL,
    position_order          INTEGER         NOT NULL DEFAULT 0,
    config                  JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p033_dw_widget_type CHECK (
        widget_type IN ('KPI_CARD', 'BAR_CHART', 'LINE_CHART', 'PIE_CHART', 'DONUT_CHART',
                          'SCATTER_PLOT', 'HEATMAP', 'TABLE', 'WATERFALL', 'GAUGE',
                          'SANKEY', 'TREEMAP', 'PROGRESS_BAR', 'MAP', 'CUSTOM')
    ),
    CONSTRAINT chk_p033_dw_data_source CHECK (
        data_source IN ('SCAN_RESULTS', 'PRIORITY_SCORES', 'PAYBACK_ANALYSES',
                          'SAVINGS_ESTIMATES', 'CARBON_IMPACTS', 'IMPLEMENTATION_PROGRESS',
                          'SAVINGS_ACTUALS', 'REBATE_APPLICATIONS', 'BEHAVIORAL_PROGRAMS',
                          'ACTION_LIBRARY', 'INTERACTIVE_EFFECTS', 'CUSTOM_QUERY')
    ),
    CONSTRAINT chk_p033_dw_position CHECK (
        position_order >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p033_dw_config        ON pack033_quick_wins.dashboard_widgets(config_id);
CREATE INDEX idx_p033_dw_widget_type   ON pack033_quick_wins.dashboard_widgets(widget_type);
CREATE INDEX idx_p033_dw_data_source   ON pack033_quick_wins.dashboard_widgets(data_source);
CREATE INDEX idx_p033_dw_position      ON pack033_quick_wins.dashboard_widgets(config_id, position_order);
CREATE INDEX idx_p033_dw_config_json   ON pack033_quick_wins.dashboard_widgets USING GIN(config);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p033_dw_updated
    BEFORE UPDATE ON pack033_quick_wins.dashboard_widgets
    FOR EACH ROW EXECUTE FUNCTION pack033_quick_wins.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack033_quick_wins.report_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack033_quick_wins.dashboard_widgets ENABLE ROW LEVEL SECURITY;

CREATE POLICY p033_rc_tenant_isolation
    ON pack033_quick_wins.report_configs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p033_rc_service_bypass
    ON pack033_quick_wins.report_configs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p033_dw_tenant_isolation
    ON pack033_quick_wins.dashboard_widgets
    USING (config_id IN (
        SELECT config_id FROM pack033_quick_wins.report_configs
        WHERE tenant_id = current_setting('app.current_tenant')::UUID
    ));
CREATE POLICY p033_dw_service_bypass
    ON pack033_quick_wins.dashboard_widgets
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.report_configs TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack033_quick_wins.dashboard_widgets TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack033_quick_wins.report_configs IS
    'Report configuration for scheduled and on-demand report generation with recipient lists and output format preferences.';

COMMENT ON TABLE pack033_quick_wins.dashboard_widgets IS
    'Configurable dashboard widget definitions specifying widget type, data source, layout position, and rendering config.';

COMMENT ON COLUMN pack033_quick_wins.report_configs.report_type IS
    'Type of report: SCAN_SUMMARY, ACTION_PLAN, FINANCIAL_ANALYSIS, SAVINGS_PROGRESS, etc.';
COMMENT ON COLUMN pack033_quick_wins.report_configs.schedule IS
    'Report generation schedule: DAILY, WEEKLY, MONTHLY, QUARTERLY, ANNUALLY, ON_DEMAND.';
COMMENT ON COLUMN pack033_quick_wins.report_configs.recipients IS
    'JSONB array of recipient objects with email, name, and delivery preferences.';
COMMENT ON COLUMN pack033_quick_wins.report_configs.filters IS
    'JSONB filter criteria applied when generating the report (facility_ids, date_range, categories, etc.).';
COMMENT ON COLUMN pack033_quick_wins.dashboard_widgets.widget_type IS
    'Visual widget type: KPI_CARD, BAR_CHART, LINE_CHART, PIE_CHART, TABLE, WATERFALL, GAUGE, etc.';
COMMENT ON COLUMN pack033_quick_wins.dashboard_widgets.data_source IS
    'Source table or query for widget data (e.g., SCAN_RESULTS, SAVINGS_ACTUALS).';
COMMENT ON COLUMN pack033_quick_wins.dashboard_widgets.config IS
    'JSONB rendering configuration (colors, thresholds, axes, legend, drill-down settings, etc.).';
