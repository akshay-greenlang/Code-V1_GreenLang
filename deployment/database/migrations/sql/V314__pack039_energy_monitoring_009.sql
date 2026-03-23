-- =============================================================================
-- V314: PACK-039 Energy Monitoring Pack - Dashboards and Reports
-- =============================================================================
-- Pack:         PACK-039 (Energy Monitoring Pack)
-- Migration:    009 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates dashboard and reporting tables for energy monitoring visualization
-- and scheduled report generation. Includes dashboard configuration with
-- layout persistence, report scheduling with delivery options, report
-- output records, KPI definitions for dashboard widgets, and widget
-- configuration for flexible dashboard composition.
--
-- Tables (5):
--   1. pack039_energy_monitoring.em_dashboard_configs
--   2. pack039_energy_monitoring.em_report_schedules
--   3. pack039_energy_monitoring.em_report_outputs
--   4. pack039_energy_monitoring.em_kpi_definitions
--   5. pack039_energy_monitoring.em_widget_configs
--
-- Previous: V313__pack039_energy_monitoring_008.sql
-- =============================================================================

SET search_path TO pack039_energy_monitoring, public;

-- =============================================================================
-- Table 1: pack039_energy_monitoring.em_dashboard_configs
-- =============================================================================
-- Dashboard configuration and layout definitions for energy monitoring
-- visualizations. Each dashboard represents a named collection of widgets
-- with a specific layout, target audience, and access controls. Supports
-- multiple dashboard types (operational, executive, compliance) with
-- user-customizable layouts and shared/private visibility.

CREATE TABLE pack039_energy_monitoring.em_dashboard_configs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID,
    dashboard_name          VARCHAR(255)    NOT NULL,
    dashboard_code          VARCHAR(50)     NOT NULL,
    dashboard_type          VARCHAR(30)     NOT NULL DEFAULT 'OPERATIONAL',
    description             TEXT,
    target_audience         VARCHAR(30)     NOT NULL DEFAULT 'OPERATIONS',
    layout_type             VARCHAR(30)     NOT NULL DEFAULT 'GRID',
    layout_columns          INTEGER         NOT NULL DEFAULT 12,
    layout_config           JSONB           NOT NULL DEFAULT '{}',
    widget_ids              UUID[]          DEFAULT '{}',
    default_time_range      VARCHAR(30)     NOT NULL DEFAULT 'LAST_24H',
    default_refresh_seconds INTEGER         NOT NULL DEFAULT 300,
    auto_refresh            BOOLEAN         NOT NULL DEFAULT true,
    theme                   VARCHAR(20)     NOT NULL DEFAULT 'LIGHT',
    color_scheme            JSONB           DEFAULT '{}',
    filters_config          JSONB           DEFAULT '{}',
    global_filters          JSONB           DEFAULT '{}',
    is_default              BOOLEAN         NOT NULL DEFAULT false,
    is_public               BOOLEAN         NOT NULL DEFAULT false,
    is_shared               BOOLEAN         NOT NULL DEFAULT false,
    shared_with_roles       VARCHAR(50)[]   DEFAULT '{}',
    owner_id                UUID            NOT NULL,
    last_accessed_at        TIMESTAMPTZ,
    access_count            BIGINT          NOT NULL DEFAULT 0,
    is_template             BOOLEAN         NOT NULL DEFAULT false,
    template_source_id      UUID,
    dashboard_status        VARCHAR(20)     NOT NULL DEFAULT 'ACTIVE',
    version                 INTEGER         NOT NULL DEFAULT 1,
    tags                    JSONB           DEFAULT '[]',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_dbc_type CHECK (
        dashboard_type IN (
            'OPERATIONAL', 'EXECUTIVE', 'ENGINEERING', 'COMPLIANCE',
            'FINANCIAL', 'PORTFOLIO', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_dbc_audience CHECK (
        target_audience IN (
            'OPERATIONS', 'MANAGEMENT', 'ENGINEERING', 'FINANCE',
            'SUSTAINABILITY', 'TENANT', 'PUBLIC', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_dbc_layout CHECK (
        layout_type IN ('GRID', 'FREEFORM', 'TABS', 'SCROLL', 'FIXED')
    ),
    CONSTRAINT chk_p039_dbc_time_range CHECK (
        default_time_range IN (
            'LAST_1H', 'LAST_4H', 'LAST_8H', 'LAST_24H', 'LAST_7D',
            'LAST_30D', 'LAST_90D', 'LAST_12M', 'THIS_MONTH',
            'THIS_QUARTER', 'THIS_YEAR', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_dbc_theme CHECK (
        theme IN ('LIGHT', 'DARK', 'HIGH_CONTRAST', 'CUSTOM')
    ),
    CONSTRAINT chk_p039_dbc_status CHECK (
        dashboard_status IN ('ACTIVE', 'DRAFT', 'ARCHIVED', 'DISABLED')
    ),
    CONSTRAINT chk_p039_dbc_columns CHECK (
        layout_columns >= 1 AND layout_columns <= 24
    ),
    CONSTRAINT chk_p039_dbc_refresh CHECK (
        default_refresh_seconds >= 10 AND default_refresh_seconds <= 86400
    ),
    CONSTRAINT chk_p039_dbc_version CHECK (
        version >= 1
    ),
    CONSTRAINT chk_p039_dbc_access CHECK (
        access_count >= 0
    ),
    CONSTRAINT uq_p039_dbc_tenant_code UNIQUE (tenant_id, dashboard_code, version)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_dbc_tenant         ON pack039_energy_monitoring.em_dashboard_configs(tenant_id);
CREATE INDEX idx_p039_dbc_facility       ON pack039_energy_monitoring.em_dashboard_configs(facility_id);
CREATE INDEX idx_p039_dbc_code           ON pack039_energy_monitoring.em_dashboard_configs(dashboard_code);
CREATE INDEX idx_p039_dbc_type           ON pack039_energy_monitoring.em_dashboard_configs(dashboard_type);
CREATE INDEX idx_p039_dbc_audience       ON pack039_energy_monitoring.em_dashboard_configs(target_audience);
CREATE INDEX idx_p039_dbc_owner          ON pack039_energy_monitoring.em_dashboard_configs(owner_id);
CREATE INDEX idx_p039_dbc_default        ON pack039_energy_monitoring.em_dashboard_configs(is_default) WHERE is_default = true;
CREATE INDEX idx_p039_dbc_public         ON pack039_energy_monitoring.em_dashboard_configs(is_public) WHERE is_public = true;
CREATE INDEX idx_p039_dbc_template       ON pack039_energy_monitoring.em_dashboard_configs(is_template) WHERE is_template = true;
CREATE INDEX idx_p039_dbc_status         ON pack039_energy_monitoring.em_dashboard_configs(dashboard_status);
CREATE INDEX idx_p039_dbc_accessed       ON pack039_energy_monitoring.em_dashboard_configs(last_accessed_at DESC);
CREATE INDEX idx_p039_dbc_created        ON pack039_energy_monitoring.em_dashboard_configs(created_at DESC);
CREATE INDEX idx_p039_dbc_tags           ON pack039_energy_monitoring.em_dashboard_configs USING GIN(tags);
CREATE INDEX idx_p039_dbc_widgets        ON pack039_energy_monitoring.em_dashboard_configs USING GIN(widget_ids);
CREATE INDEX idx_p039_dbc_roles          ON pack039_energy_monitoring.em_dashboard_configs USING GIN(shared_with_roles);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_dbc_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_dashboard_configs
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack039_energy_monitoring.em_report_schedules
-- =============================================================================
-- Scheduled report definitions for automated energy report generation
-- and delivery. Each schedule defines the report type, content scope,
-- generation frequency, delivery channels, and recipient lists. Supports
-- PDF, Excel, and email report formats with custom templates.

CREATE TABLE pack039_energy_monitoring.em_report_schedules (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID,
    report_name             VARCHAR(255)    NOT NULL,
    report_code             VARCHAR(50)     NOT NULL,
    report_type             VARCHAR(50)     NOT NULL DEFAULT 'MONTHLY_SUMMARY',
    report_category         VARCHAR(30)     NOT NULL DEFAULT 'OPERATIONAL',
    description             TEXT,
    template_id             VARCHAR(100),
    template_version        INTEGER         DEFAULT 1,
    scope_type              VARCHAR(30)     NOT NULL DEFAULT 'FACILITY',
    scope_meter_ids         UUID[]          DEFAULT '{}',
    scope_account_ids       UUID[]          DEFAULT '{}',
    scope_enpi_ids          UUID[]          DEFAULT '{}',
    energy_types            VARCHAR(50)[]   DEFAULT '{ELECTRICITY}',
    include_sections        VARCHAR(50)[]   DEFAULT '{}',
    schedule_type           VARCHAR(20)     NOT NULL DEFAULT 'RECURRING',
    cron_expression         VARCHAR(100),
    generation_frequency    VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    report_period_type      VARCHAR(20)     NOT NULL DEFAULT 'PREVIOUS_MONTH',
    custom_period_days      INTEGER,
    timezone                VARCHAR(50)     NOT NULL DEFAULT 'UTC',
    output_format           VARCHAR(20)     NOT NULL DEFAULT 'PDF',
    output_formats          VARCHAR(20)[]   DEFAULT '{PDF}',
    delivery_method         VARCHAR(30)     NOT NULL DEFAULT 'EMAIL',
    delivery_channels       JSONB           DEFAULT '[]',
    recipients_to           JSONB           DEFAULT '[]',
    recipients_cc           JSONB           DEFAULT '[]',
    email_subject_template  VARCHAR(500),
    email_body_template     TEXT,
    upload_to_storage       BOOLEAN         NOT NULL DEFAULT true,
    storage_path_template   VARCHAR(500),
    retention_days          INTEGER         DEFAULT 365,
    include_raw_data        BOOLEAN         NOT NULL DEFAULT false,
    include_charts          BOOLEAN         NOT NULL DEFAULT true,
    include_recommendations BOOLEAN         NOT NULL DEFAULT false,
    branding_config         JSONB           DEFAULT '{}',
    language                VARCHAR(10)     NOT NULL DEFAULT 'en',
    is_enabled              BOOLEAN         NOT NULL DEFAULT true,
    last_generated_at       TIMESTAMPTZ,
    last_generation_status  VARCHAR(20),
    next_generation_at      TIMESTAMPTZ,
    total_generations       BIGINT          NOT NULL DEFAULT 0,
    total_failures          BIGINT          NOT NULL DEFAULT 0,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    created_by              UUID,
    -- Constraints
    CONSTRAINT chk_p039_rs_type CHECK (
        report_type IN (
            'MONTHLY_SUMMARY', 'WEEKLY_SUMMARY', 'DAILY_SUMMARY',
            'ENPI_REPORT', 'VARIANCE_REPORT', 'ANOMALY_REPORT',
            'COST_ALLOCATION', 'BILLING_REPORT', 'BUDGET_REPORT',
            'COMPLIANCE_REPORT', 'ALARM_SUMMARY', 'DATA_QUALITY',
            'EXECUTIVE_DASHBOARD', 'BENCHMARKING', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_rs_category CHECK (
        report_category IN (
            'OPERATIONAL', 'FINANCIAL', 'COMPLIANCE', 'EXECUTIVE',
            'ENGINEERING', 'SUSTAINABILITY', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_rs_scope CHECK (
        scope_type IN (
            'FACILITY', 'BUILDING', 'METER', 'ACCOUNT', 'PORTFOLIO', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_rs_schedule CHECK (
        schedule_type IN ('ONE_TIME', 'RECURRING', 'ON_DEMAND', 'EVENT_TRIGGERED')
    ),
    CONSTRAINT chk_p039_rs_frequency CHECK (
        generation_frequency IN (
            'DAILY', 'WEEKLY', 'BIWEEKLY', 'MONTHLY', 'QUARTERLY', 'ANNUAL', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_rs_period CHECK (
        report_period_type IN (
            'PREVIOUS_DAY', 'PREVIOUS_WEEK', 'PREVIOUS_MONTH',
            'PREVIOUS_QUARTER', 'PREVIOUS_YEAR', 'CUSTOM_DAYS',
            'MONTH_TO_DATE', 'YEAR_TO_DATE'
        )
    ),
    CONSTRAINT chk_p039_rs_format CHECK (
        output_format IN ('PDF', 'EXCEL', 'CSV', 'HTML', 'JSON')
    ),
    CONSTRAINT chk_p039_rs_delivery CHECK (
        delivery_method IN (
            'EMAIL', 'SFTP', 'API', 'STORAGE', 'WEBHOOK', 'PRINT'
        )
    ),
    CONSTRAINT chk_p039_rs_gen_status CHECK (
        last_generation_status IS NULL OR last_generation_status IN (
            'SUCCESS', 'PARTIAL', 'FAILED', 'TIMEOUT', 'CANCELLED'
        )
    ),
    CONSTRAINT chk_p039_rs_retention CHECK (
        retention_days IS NULL OR (retention_days >= 1 AND retention_days <= 3650)
    ),
    CONSTRAINT chk_p039_rs_counts CHECK (
        total_generations >= 0 AND total_failures >= 0
    ),
    CONSTRAINT uq_p039_rs_tenant_code UNIQUE (tenant_id, report_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_rs_tenant          ON pack039_energy_monitoring.em_report_schedules(tenant_id);
CREATE INDEX idx_p039_rs_facility        ON pack039_energy_monitoring.em_report_schedules(facility_id);
CREATE INDEX idx_p039_rs_code            ON pack039_energy_monitoring.em_report_schedules(report_code);
CREATE INDEX idx_p039_rs_type            ON pack039_energy_monitoring.em_report_schedules(report_type);
CREATE INDEX idx_p039_rs_category        ON pack039_energy_monitoring.em_report_schedules(report_category);
CREATE INDEX idx_p039_rs_frequency       ON pack039_energy_monitoring.em_report_schedules(generation_frequency);
CREATE INDEX idx_p039_rs_enabled         ON pack039_energy_monitoring.em_report_schedules(is_enabled) WHERE is_enabled = true;
CREATE INDEX idx_p039_rs_next_gen        ON pack039_energy_monitoring.em_report_schedules(next_generation_at);
CREATE INDEX idx_p039_rs_last_gen        ON pack039_energy_monitoring.em_report_schedules(last_generated_at DESC);
CREATE INDEX idx_p039_rs_created         ON pack039_energy_monitoring.em_report_schedules(created_at DESC);
CREATE INDEX idx_p039_rs_meters          ON pack039_energy_monitoring.em_report_schedules USING GIN(scope_meter_ids);
CREATE INDEX idx_p039_rs_recipients      ON pack039_energy_monitoring.em_report_schedules USING GIN(recipients_to);
CREATE INDEX idx_p039_rs_energy_types    ON pack039_energy_monitoring.em_report_schedules USING GIN(energy_types);

-- Composite: scheduled reports due for generation
CREATE INDEX idx_p039_rs_due             ON pack039_energy_monitoring.em_report_schedules(next_generation_at, generation_frequency)
    WHERE is_enabled = true AND schedule_type = 'RECURRING';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_rs_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_report_schedules
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack039_energy_monitoring.em_report_outputs
-- =============================================================================
-- Generated report output records. Each row represents a single report
-- generation event with file location, delivery status, and metadata.
-- Provides a complete history of all reports generated for audit and
-- retrieval purposes.

CREATE TABLE pack039_energy_monitoring.em_report_outputs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    report_schedule_id      UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_report_schedules(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    report_period_start     DATE            NOT NULL,
    report_period_end       DATE            NOT NULL,
    generated_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    generation_duration_ms  INTEGER,
    output_format           VARCHAR(20)     NOT NULL DEFAULT 'PDF',
    file_size_bytes         BIGINT,
    file_path               TEXT,
    file_url                TEXT,
    storage_bucket          VARCHAR(255),
    storage_key             VARCHAR(500),
    page_count              INTEGER,
    data_points_count       INTEGER,
    generation_status       VARCHAR(20)     NOT NULL DEFAULT 'SUCCESS',
    error_message           TEXT,
    delivery_status         VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    delivered_at            TIMESTAMPTZ,
    delivery_method         VARCHAR(30),
    delivery_reference      VARCHAR(255),
    recipients_delivered    JSONB           DEFAULT '[]',
    delivery_errors         JSONB           DEFAULT '[]',
    data_quality_pct        NUMERIC(5,2),
    report_hash             VARCHAR(64),
    download_count          INTEGER         NOT NULL DEFAULT 0,
    last_downloaded_at      TIMESTAMPTZ,
    last_downloaded_by      UUID,
    expires_at              TIMESTAMPTZ,
    is_archived             BOOLEAN         NOT NULL DEFAULT false,
    notes                   TEXT,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_ro_format CHECK (
        output_format IN ('PDF', 'EXCEL', 'CSV', 'HTML', 'JSON')
    ),
    CONSTRAINT chk_p039_ro_gen_status CHECK (
        generation_status IN ('SUCCESS', 'PARTIAL', 'FAILED', 'TIMEOUT', 'CANCELLED')
    ),
    CONSTRAINT chk_p039_ro_del_status CHECK (
        delivery_status IN (
            'PENDING', 'SENDING', 'DELIVERED', 'PARTIAL_DELIVERY',
            'FAILED', 'BOUNCED', 'NOT_APPLICABLE'
        )
    ),
    CONSTRAINT chk_p039_ro_delivery CHECK (
        delivery_method IS NULL OR delivery_method IN (
            'EMAIL', 'SFTP', 'API', 'STORAGE', 'WEBHOOK', 'PRINT'
        )
    ),
    CONSTRAINT chk_p039_ro_file_size CHECK (
        file_size_bytes IS NULL OR file_size_bytes >= 0
    ),
    CONSTRAINT chk_p039_ro_duration CHECK (
        generation_duration_ms IS NULL OR generation_duration_ms >= 0
    ),
    CONSTRAINT chk_p039_ro_downloads CHECK (
        download_count >= 0
    ),
    CONSTRAINT chk_p039_ro_quality CHECK (
        data_quality_pct IS NULL OR (data_quality_pct >= 0 AND data_quality_pct <= 100)
    ),
    CONSTRAINT chk_p039_ro_dates CHECK (
        report_period_start <= report_period_end
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_ro_schedule        ON pack039_energy_monitoring.em_report_outputs(report_schedule_id);
CREATE INDEX idx_p039_ro_tenant          ON pack039_energy_monitoring.em_report_outputs(tenant_id);
CREATE INDEX idx_p039_ro_generated       ON pack039_energy_monitoring.em_report_outputs(generated_at DESC);
CREATE INDEX idx_p039_ro_period_start    ON pack039_energy_monitoring.em_report_outputs(report_period_start DESC);
CREATE INDEX idx_p039_ro_gen_status      ON pack039_energy_monitoring.em_report_outputs(generation_status);
CREATE INDEX idx_p039_ro_del_status      ON pack039_energy_monitoring.em_report_outputs(delivery_status);
CREATE INDEX idx_p039_ro_format          ON pack039_energy_monitoring.em_report_outputs(output_format);
CREATE INDEX idx_p039_ro_archived        ON pack039_energy_monitoring.em_report_outputs(is_archived) WHERE is_archived = false;
CREATE INDEX idx_p039_ro_expires         ON pack039_energy_monitoring.em_report_outputs(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_p039_ro_created         ON pack039_energy_monitoring.em_report_outputs(created_at DESC);

-- Composite: recent outputs by schedule for history view
CREATE INDEX idx_p039_ro_sched_recent    ON pack039_energy_monitoring.em_report_outputs(report_schedule_id, generated_at DESC);

-- =============================================================================
-- Table 4: pack039_energy_monitoring.em_kpi_definitions
-- =============================================================================
-- Key Performance Indicator definitions for dashboard widgets. Each KPI
-- defines a metric calculation, target, and visualization parameters.
-- KPIs are referenced by dashboard widgets and provide the data model
-- for gauges, scorecards, sparklines, and trend indicators.

CREATE TABLE pack039_energy_monitoring.em_kpi_definitions (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID,
    kpi_name                VARCHAR(255)    NOT NULL,
    kpi_code                VARCHAR(50)     NOT NULL,
    kpi_category            VARCHAR(50)     NOT NULL DEFAULT 'ENERGY',
    description             TEXT,
    data_source             VARCHAR(50)     NOT NULL DEFAULT 'INTERVAL_DATA',
    calculation_method      VARCHAR(30)     NOT NULL DEFAULT 'SUM',
    formula_expression      TEXT,
    source_meter_ids        UUID[]          DEFAULT '{}',
    source_enpi_ids         UUID[]          DEFAULT '{}',
    unit                    VARCHAR(30)     NOT NULL DEFAULT 'kWh',
    display_format          VARCHAR(30)     NOT NULL DEFAULT 'NUMBER',
    decimal_places          INTEGER         NOT NULL DEFAULT 2,
    target_value            NUMERIC(18,6),
    target_direction        VARCHAR(10)     NOT NULL DEFAULT 'LOWER',
    warning_threshold       NUMERIC(18,6),
    critical_threshold      NUMERIC(18,6),
    benchmark_value         NUMERIC(18,6),
    benchmark_source        VARCHAR(100),
    comparison_period       VARCHAR(20)     DEFAULT 'PREVIOUS_PERIOD',
    trend_window_periods    INTEGER         DEFAULT 6,
    visualization_type      VARCHAR(30)     NOT NULL DEFAULT 'GAUGE',
    color_good              VARCHAR(10)     DEFAULT '#22C55E',
    color_warning           VARCHAR(10)     DEFAULT '#EAB308',
    color_critical          VARCHAR(10)     DEFAULT '#EF4444',
    current_value           NUMERIC(18,6),
    current_status          VARCHAR(20),
    current_trend           VARCHAR(20),
    last_calculated_at      TIMESTAMPTZ,
    is_published            BOOLEAN         NOT NULL DEFAULT false,
    is_active               BOOLEAN         NOT NULL DEFAULT true,
    sort_order              INTEGER         NOT NULL DEFAULT 100,
    tags                    JSONB           DEFAULT '[]',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_kd_category CHECK (
        kpi_category IN (
            'ENERGY', 'COST', 'DEMAND', 'ENPI', 'ENVIRONMENTAL',
            'OPERATIONAL', 'DATA_QUALITY', 'ALARM', 'BUDGET',
            'COMPLIANCE', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_kd_data_source CHECK (
        data_source IN (
            'INTERVAL_DATA', 'ENPI_VALUES', 'COST_ALLOCATIONS',
            'ANOMALIES', 'ALARMS', 'BUDGETS', 'QUALITY_SCORES',
            'BILLING', 'REGRESSION', 'FORMULA', 'EXTERNAL'
        )
    ),
    CONSTRAINT chk_p039_kd_calc_method CHECK (
        calculation_method IN (
            'SUM', 'AVERAGE', 'MAX', 'MIN', 'COUNT', 'RATIO',
            'WEIGHTED_AVERAGE', 'PERCENTILE', 'FORMULA', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_kd_display CHECK (
        display_format IN (
            'NUMBER', 'PERCENTAGE', 'CURRENCY', 'SCIENTIFIC',
            'COMPACT', 'DURATION', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_kd_direction CHECK (
        target_direction IN ('LOWER', 'HIGHER', 'TARGET')
    ),
    CONSTRAINT chk_p039_kd_comparison CHECK (
        comparison_period IS NULL OR comparison_period IN (
            'PREVIOUS_PERIOD', 'SAME_PERIOD_LAST_YEAR', 'BASELINE',
            'TARGET', 'BENCHMARK', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_kd_viz CHECK (
        visualization_type IN (
            'GAUGE', 'SCORECARD', 'SPARKLINE', 'TREND_ARROW',
            'BAR', 'PIE', 'NUMBER_CARD', 'COMPARISON', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_kd_status CHECK (
        current_status IS NULL OR current_status IN (
            'GOOD', 'WARNING', 'CRITICAL', 'UNKNOWN', 'NO_DATA'
        )
    ),
    CONSTRAINT chk_p039_kd_trend CHECK (
        current_trend IS NULL OR current_trend IN (
            'IMPROVING', 'STABLE', 'DEGRADING', 'INSUFFICIENT_DATA'
        )
    ),
    CONSTRAINT chk_p039_kd_decimals CHECK (
        decimal_places >= 0 AND decimal_places <= 10
    ),
    CONSTRAINT chk_p039_kd_sort CHECK (
        sort_order >= 1 AND sort_order <= 9999
    ),
    CONSTRAINT uq_p039_kd_tenant_code UNIQUE (tenant_id, kpi_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_kd_tenant          ON pack039_energy_monitoring.em_kpi_definitions(tenant_id);
CREATE INDEX idx_p039_kd_facility        ON pack039_energy_monitoring.em_kpi_definitions(facility_id);
CREATE INDEX idx_p039_kd_code            ON pack039_energy_monitoring.em_kpi_definitions(kpi_code);
CREATE INDEX idx_p039_kd_category        ON pack039_energy_monitoring.em_kpi_definitions(kpi_category);
CREATE INDEX idx_p039_kd_data_source     ON pack039_energy_monitoring.em_kpi_definitions(data_source);
CREATE INDEX idx_p039_kd_active          ON pack039_energy_monitoring.em_kpi_definitions(is_active) WHERE is_active = true;
CREATE INDEX idx_p039_kd_published       ON pack039_energy_monitoring.em_kpi_definitions(is_published) WHERE is_published = true;
CREATE INDEX idx_p039_kd_status          ON pack039_energy_monitoring.em_kpi_definitions(current_status);
CREATE INDEX idx_p039_kd_sort            ON pack039_energy_monitoring.em_kpi_definitions(sort_order);
CREATE INDEX idx_p039_kd_created         ON pack039_energy_monitoring.em_kpi_definitions(created_at DESC);
CREATE INDEX idx_p039_kd_tags            ON pack039_energy_monitoring.em_kpi_definitions USING GIN(tags);
CREATE INDEX idx_p039_kd_meters          ON pack039_energy_monitoring.em_kpi_definitions USING GIN(source_meter_ids);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_kd_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_kpi_definitions
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack039_energy_monitoring.em_widget_configs
-- =============================================================================
-- Individual widget configurations for dashboard composition. Each widget
-- defines a single visualization element (chart, gauge, table, map) with
-- its data source, rendering options, position within the dashboard
-- layout, and interaction settings.

CREATE TABLE pack039_energy_monitoring.em_widget_configs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    dashboard_id            UUID            NOT NULL REFERENCES pack039_energy_monitoring.em_dashboard_configs(id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    widget_name             VARCHAR(255)    NOT NULL,
    widget_code             VARCHAR(50)     NOT NULL,
    widget_type             VARCHAR(30)     NOT NULL DEFAULT 'LINE_CHART',
    kpi_id                  UUID            REFERENCES pack039_energy_monitoring.em_kpi_definitions(id),
    data_source             VARCHAR(50)     NOT NULL DEFAULT 'INTERVAL_DATA',
    data_query_config       JSONB           NOT NULL DEFAULT '{}',
    meter_ids               UUID[]          DEFAULT '{}',
    enpi_ids                UUID[]          DEFAULT '{}',
    time_range_override     VARCHAR(30),
    refresh_seconds_override INTEGER,
    position_x              INTEGER         NOT NULL DEFAULT 0,
    position_y              INTEGER         NOT NULL DEFAULT 0,
    width                   INTEGER         NOT NULL DEFAULT 4,
    height                  INTEGER         NOT NULL DEFAULT 3,
    min_width               INTEGER         DEFAULT 2,
    min_height              INTEGER         DEFAULT 2,
    chart_config            JSONB           DEFAULT '{}',
    color_config            JSONB           DEFAULT '{}',
    axis_config             JSONB           DEFAULT '{}',
    legend_config           JSONB           DEFAULT '{}',
    tooltip_config          JSONB           DEFAULT '{}',
    threshold_lines         JSONB           DEFAULT '[]',
    comparison_config       JSONB           DEFAULT '{}',
    drill_down_config       JSONB           DEFAULT '{}',
    click_action            VARCHAR(30),
    click_target_url        TEXT,
    is_interactive          BOOLEAN         NOT NULL DEFAULT true,
    is_resizable            BOOLEAN         NOT NULL DEFAULT true,
    is_visible              BOOLEAN         NOT NULL DEFAULT true,
    sort_order              INTEGER         NOT NULL DEFAULT 100,
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p039_wc_type CHECK (
        widget_type IN (
            'LINE_CHART', 'BAR_CHART', 'STACKED_BAR', 'AREA_CHART',
            'PIE_CHART', 'DONUT_CHART', 'GAUGE', 'SCORECARD',
            'NUMBER_CARD', 'TABLE', 'HEATMAP', 'SANKEY',
            'TREEMAP', 'SPARKLINE', 'MAP', 'METER_DIAGRAM',
            'ALARM_LIST', 'ANOMALY_LIST', 'TEXT', 'IMAGE', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p039_wc_data_source CHECK (
        data_source IN (
            'INTERVAL_DATA', 'ENPI_VALUES', 'COST_ALLOCATIONS',
            'ANOMALIES', 'ALARMS', 'BUDGETS', 'QUALITY_SCORES',
            'BILLING', 'COMPLETENESS', 'KPI', 'FORMULA', 'EXTERNAL'
        )
    ),
    CONSTRAINT chk_p039_wc_click CHECK (
        click_action IS NULL OR click_action IN (
            'DRILL_DOWN', 'NAVIGATE', 'FILTER', 'MODAL', 'NONE'
        )
    ),
    CONSTRAINT chk_p039_wc_position CHECK (
        position_x >= 0 AND position_y >= 0
    ),
    CONSTRAINT chk_p039_wc_size CHECK (
        width >= 1 AND width <= 24 AND height >= 1 AND height <= 24
    ),
    CONSTRAINT chk_p039_wc_min_size CHECK (
        (min_width IS NULL OR min_width >= 1) AND
        (min_height IS NULL OR min_height >= 1)
    ),
    CONSTRAINT chk_p039_wc_sort CHECK (
        sort_order >= 1 AND sort_order <= 9999
    ),
    CONSTRAINT uq_p039_wc_dashboard_code UNIQUE (dashboard_id, widget_code)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p039_wc_dashboard       ON pack039_energy_monitoring.em_widget_configs(dashboard_id);
CREATE INDEX idx_p039_wc_tenant          ON pack039_energy_monitoring.em_widget_configs(tenant_id);
CREATE INDEX idx_p039_wc_code            ON pack039_energy_monitoring.em_widget_configs(widget_code);
CREATE INDEX idx_p039_wc_type            ON pack039_energy_monitoring.em_widget_configs(widget_type);
CREATE INDEX idx_p039_wc_kpi             ON pack039_energy_monitoring.em_widget_configs(kpi_id);
CREATE INDEX idx_p039_wc_data_source     ON pack039_energy_monitoring.em_widget_configs(data_source);
CREATE INDEX idx_p039_wc_visible         ON pack039_energy_monitoring.em_widget_configs(is_visible) WHERE is_visible = true;
CREATE INDEX idx_p039_wc_sort            ON pack039_energy_monitoring.em_widget_configs(sort_order);
CREATE INDEX idx_p039_wc_created         ON pack039_energy_monitoring.em_widget_configs(created_at DESC);
CREATE INDEX idx_p039_wc_meters          ON pack039_energy_monitoring.em_widget_configs USING GIN(meter_ids);
CREATE INDEX idx_p039_wc_enpi            ON pack039_energy_monitoring.em_widget_configs USING GIN(enpi_ids);
CREATE INDEX idx_p039_wc_chart           ON pack039_energy_monitoring.em_widget_configs USING GIN(chart_config);

-- Composite: visible widgets by dashboard for rendering
CREATE INDEX idx_p039_wc_dash_visible    ON pack039_energy_monitoring.em_widget_configs(dashboard_id, sort_order)
    WHERE is_visible = true;

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p039_wc_updated
    BEFORE UPDATE ON pack039_energy_monitoring.em_widget_configs
    FOR EACH ROW EXECUTE FUNCTION pack039_energy_monitoring.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack039_energy_monitoring.em_dashboard_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_report_schedules ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_report_outputs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_kpi_definitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack039_energy_monitoring.em_widget_configs ENABLE ROW LEVEL SECURITY;

CREATE POLICY p039_dbc_tenant_isolation
    ON pack039_energy_monitoring.em_dashboard_configs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_dbc_service_bypass
    ON pack039_energy_monitoring.em_dashboard_configs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_rs_tenant_isolation
    ON pack039_energy_monitoring.em_report_schedules
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_rs_service_bypass
    ON pack039_energy_monitoring.em_report_schedules
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_ro_tenant_isolation
    ON pack039_energy_monitoring.em_report_outputs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_ro_service_bypass
    ON pack039_energy_monitoring.em_report_outputs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_kd_tenant_isolation
    ON pack039_energy_monitoring.em_kpi_definitions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_kd_service_bypass
    ON pack039_energy_monitoring.em_kpi_definitions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p039_wc_tenant_isolation
    ON pack039_energy_monitoring.em_widget_configs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p039_wc_service_bypass
    ON pack039_energy_monitoring.em_widget_configs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_dashboard_configs TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_report_schedules TO PUBLIC;
GRANT SELECT, INSERT ON pack039_energy_monitoring.em_report_outputs TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_kpi_definitions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack039_energy_monitoring.em_widget_configs TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack039_energy_monitoring.em_dashboard_configs IS
    'Dashboard configuration with layout, widget composition, access controls, and user customization.';
COMMENT ON TABLE pack039_energy_monitoring.em_report_schedules IS
    'Scheduled report definitions for automated generation and delivery with templates and recipient management.';
COMMENT ON TABLE pack039_energy_monitoring.em_report_outputs IS
    'Generated report output records with file locations, delivery tracking, and download history.';
COMMENT ON TABLE pack039_energy_monitoring.em_kpi_definitions IS
    'KPI definitions for dashboard widgets with calculation methods, targets, thresholds, and visualization parameters.';
COMMENT ON TABLE pack039_energy_monitoring.em_widget_configs IS
    'Individual widget configurations with chart type, data source, position, sizing, and interaction settings.';

COMMENT ON COLUMN pack039_energy_monitoring.em_dashboard_configs.dashboard_type IS 'Dashboard purpose: OPERATIONAL (real-time), EXECUTIVE (summary), ENGINEERING (detailed), COMPLIANCE, FINANCIAL.';
COMMENT ON COLUMN pack039_energy_monitoring.em_dashboard_configs.layout_config IS 'JSON layout configuration: {rows, cols, gaps, responsive_breakpoints}.';
COMMENT ON COLUMN pack039_energy_monitoring.em_dashboard_configs.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack039_energy_monitoring.em_report_schedules.report_type IS 'Report template: MONTHLY_SUMMARY, ENPI_REPORT, VARIANCE_REPORT, COST_ALLOCATION, etc.';
COMMENT ON COLUMN pack039_energy_monitoring.em_report_schedules.include_sections IS 'Report sections to include: consumption_summary, demand_analysis, enpi_tracking, anomalies, etc.';

COMMENT ON COLUMN pack039_energy_monitoring.em_report_outputs.report_hash IS 'SHA-256 hash of the generated report file for integrity verification.';
COMMENT ON COLUMN pack039_energy_monitoring.em_report_outputs.expires_at IS 'Report expiration date based on retention policy.';

COMMENT ON COLUMN pack039_energy_monitoring.em_kpi_definitions.visualization_type IS 'Widget rendering: GAUGE, SCORECARD, SPARKLINE, TREND_ARROW, BAR, NUMBER_CARD.';
COMMENT ON COLUMN pack039_energy_monitoring.em_kpi_definitions.current_trend IS 'Current trend direction: IMPROVING, STABLE, DEGRADING, INSUFFICIENT_DATA.';

COMMENT ON COLUMN pack039_energy_monitoring.em_widget_configs.widget_type IS 'Chart type: LINE_CHART, BAR_CHART, GAUGE, SCORECARD, HEATMAP, SANKEY, TABLE, MAP, etc.';
COMMENT ON COLUMN pack039_energy_monitoring.em_widget_configs.drill_down_config IS 'JSON drill-down configuration: {enabled, target_dashboard_id, filter_mapping}.';
