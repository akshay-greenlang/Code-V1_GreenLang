-- =============================================================================
-- V285: PACK-036 Utility Analysis Pack - Weather Normalization, Reports,
--       KPIs, Degree Days & Audit Trail
-- =============================================================================
-- Pack:         PACK-036 (Utility Analysis Pack)
-- Migration:    010 of 010
-- Date:         March 2026
--
-- Final migration for PACK-036 including weather normalization models,
-- degree day reference data, utility report generation tracking,
-- utility KPI dashboard table, pack-level audit trail, and materialized
-- views for dashboard performance.
--
-- Tables (5):
--   1. pack036_utility_analysis.gl_weather_normalizations
--   2. pack036_utility_analysis.gl_degree_days
--   3. pack036_utility_analysis.gl_utility_reports
--   4. pack036_utility_analysis.gl_utility_kpis
--   5. pack036_utility_analysis.pack036_audit_trail
--
-- Materialized Views (2):
--   1. pack036_utility_analysis.mv_facility_cost_summary
--   2. pack036_utility_analysis.mv_portfolio_utility_overview
--
-- Previous: V284__pack036_regulatory_charges.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack036_utility_analysis.gl_weather_normalizations
-- =============================================================================
-- Weather normalization model results per facility. Uses degree-day
-- regression or change-point models to separate weather-dependent from
-- weather-independent consumption. Validates against ASHRAE Guideline 14
-- statistical criteria (CV-RMSE, NMBE).

CREATE TABLE pack036_utility_analysis.gl_weather_normalizations (
    normalization_id        UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    weather_station_id      VARCHAR(50),
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    commodity               VARCHAR(30)     NOT NULL DEFAULT 'ELECTRICITY',
    model_type              VARCHAR(50)     NOT NULL,
    base_temp_heating_c     NUMERIC(5,2),
    base_temp_cooling_c     NUMERIC(5,2),
    heating_slope           NUMERIC(10,6),
    cooling_slope           NUMERIC(10,6),
    base_load_kwh_day       NUMERIC(12,4),
    r_squared               NUMERIC(6,4)    NOT NULL,
    cv_rmse_pct             NUMERIC(8,4)    NOT NULL,
    nmbe_pct                NUMERIC(8,4)    NOT NULL,
    ashrae14_passed         BOOLEAN         NOT NULL DEFAULT false,
    actual_consumption_kwh  NUMERIC(16,4)   NOT NULL,
    normalized_consumption_kwh NUMERIC(16,4) NOT NULL,
    weather_impact_kwh      NUMERIC(16,4)   NOT NULL DEFAULT 0,
    weather_impact_pct      NUMERIC(8,4),
    tmy_source              VARCHAR(100),
    data_points             INTEGER,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_wn_commodity CHECK (
        commodity IN (
            'ELECTRICITY', 'NATURAL_GAS', 'STEAM', 'TOTAL', 'ALL'
        )
    ),
    CONSTRAINT chk_p036_wn_model CHECK (
        model_type IN (
            'SIMPLE_REGRESSION', 'MULTIVARIABLE_REGRESSION',
            'THREE_PARAMETER_COOLING', 'THREE_PARAMETER_HEATING',
            'FOUR_PARAMETER', 'FIVE_PARAMETER', 'CHANGE_POINT',
            'DEGREE_DAY', 'INVERSE_MODEL', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p036_wn_r_squared CHECK (
        r_squared >= 0 AND r_squared <= 1
    ),
    CONSTRAINT chk_p036_wn_cv_rmse CHECK (
        cv_rmse_pct >= 0
    ),
    CONSTRAINT chk_p036_wn_actual CHECK (
        actual_consumption_kwh >= 0
    ),
    CONSTRAINT chk_p036_wn_normalized CHECK (
        normalized_consumption_kwh >= 0
    ),
    CONSTRAINT chk_p036_wn_period CHECK (
        period_end >= period_start
    ),
    CONSTRAINT chk_p036_wn_data_points CHECK (
        data_points IS NULL OR data_points >= 1
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_wn_tenant         ON pack036_utility_analysis.gl_weather_normalizations(tenant_id);
CREATE INDEX idx_p036_wn_facility       ON pack036_utility_analysis.gl_weather_normalizations(facility_id);
CREATE INDEX idx_p036_wn_period         ON pack036_utility_analysis.gl_weather_normalizations(period_start DESC);
CREATE INDEX idx_p036_wn_commodity      ON pack036_utility_analysis.gl_weather_normalizations(commodity);
CREATE INDEX idx_p036_wn_model          ON pack036_utility_analysis.gl_weather_normalizations(model_type);
CREATE INDEX idx_p036_wn_ashrae         ON pack036_utility_analysis.gl_weather_normalizations(ashrae14_passed);
CREATE INDEX idx_p036_wn_r_squared      ON pack036_utility_analysis.gl_weather_normalizations(r_squared DESC);
CREATE INDEX idx_p036_wn_created        ON pack036_utility_analysis.gl_weather_normalizations(created_at DESC);

-- Composite: facility + period for time-series normalization lookup
CREATE INDEX idx_p036_wn_fac_period     ON pack036_utility_analysis.gl_weather_normalizations(facility_id, period_start DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_wn_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_weather_normalizations
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack036_utility_analysis.gl_degree_days
-- =============================================================================
-- Heating and cooling degree day reference data by weather station and
-- period. Used for weather normalization calculations and degree-day
-- regression models.

CREATE TABLE pack036_utility_analysis.gl_degree_days (
    dd_id                   UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    station_id              VARCHAR(50)     NOT NULL,
    station_name            VARCHAR(255),
    country_code            CHAR(2),
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    period_type             VARCHAR(20)     NOT NULL DEFAULT 'MONTHLY',
    hdd                     NUMERIC(10,2)   NOT NULL DEFAULT 0,
    cdd                     NUMERIC(10,2)   NOT NULL DEFAULT 0,
    base_temp_heating_c     NUMERIC(5,2)    NOT NULL DEFAULT 15.5,
    base_temp_cooling_c     NUMERIC(5,2)    NOT NULL DEFAULT 18.0,
    avg_temp_c              NUMERIC(6,2),
    min_temp_c              NUMERIC(6,2),
    max_temp_c              NUMERIC(6,2),
    data_source             VARCHAR(100),
    is_tmy                  BOOLEAN         DEFAULT false,
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_dd_hdd CHECK (
        hdd >= 0
    ),
    CONSTRAINT chk_p036_dd_cdd CHECK (
        cdd >= 0
    ),
    CONSTRAINT chk_p036_dd_period CHECK (
        period_end >= period_start
    ),
    CONSTRAINT chk_p036_dd_period_type CHECK (
        period_type IN ('DAILY', 'WEEKLY', 'MONTHLY', 'ANNUAL')
    ),
    CONSTRAINT chk_p036_dd_country CHECK (
        country_code IS NULL OR LENGTH(country_code) = 2
    ),
    CONSTRAINT uq_p036_dd_station_period UNIQUE (station_id, period_start, period_end, base_temp_heating_c, base_temp_cooling_c)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_dd_station        ON pack036_utility_analysis.gl_degree_days(station_id);
CREATE INDEX idx_p036_dd_country        ON pack036_utility_analysis.gl_degree_days(country_code);
CREATE INDEX idx_p036_dd_period_start   ON pack036_utility_analysis.gl_degree_days(period_start DESC);
CREATE INDEX idx_p036_dd_period_type    ON pack036_utility_analysis.gl_degree_days(period_type);
CREATE INDEX idx_p036_dd_tmy            ON pack036_utility_analysis.gl_degree_days(is_tmy);

-- Composite: station + period for degree day lookup
CREATE INDEX idx_p036_dd_station_period ON pack036_utility_analysis.gl_degree_days(station_id, period_start DESC);

-- =============================================================================
-- Table 3: pack036_utility_analysis.gl_utility_reports
-- =============================================================================
-- Generated utility analysis reports. Tracks report type, format,
-- period, content (text/HTML), and structured data (JSONB) for
-- programmatic consumption.

CREATE TABLE pack036_utility_analysis.gl_utility_reports (
    report_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID,
    portfolio_id            UUID,
    report_type             VARCHAR(50)     NOT NULL,
    format                  VARCHAR(20)     NOT NULL DEFAULT 'PDF',
    period_start            DATE            NOT NULL,
    period_end              DATE            NOT NULL,
    title                   VARCHAR(500)    NOT NULL,
    description             TEXT,
    content                 TEXT,
    data                    JSONB           DEFAULT '{}',
    file_path               TEXT,
    file_size_bytes         BIGINT,
    generated_at            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    generated_by            UUID,
    status                  VARCHAR(30)     NOT NULL DEFAULT 'GENERATED',
    recipients              TEXT[],
    sent_at                 TIMESTAMPTZ,
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_ur_report_type CHECK (
        report_type IN (
            'MONTHLY_SUMMARY', 'QUARTERLY_REVIEW', 'ANNUAL_REVIEW',
            'BILL_AUDIT', 'RATE_ANALYSIS', 'DEMAND_ANALYSIS',
            'COST_ALLOCATION', 'BUDGET_VARIANCE', 'PROCUREMENT',
            'BENCHMARK', 'REGULATORY', 'WEATHER_NORMALIZATION',
            'EXECUTIVE_DASHBOARD', 'CUSTOM'
        )
    ),
    CONSTRAINT chk_p036_ur_format CHECK (
        format IN ('PDF', 'XLSX', 'CSV', 'HTML', 'JSON', 'DOCX')
    ),
    CONSTRAINT chk_p036_ur_period CHECK (
        period_end >= period_start
    ),
    CONSTRAINT chk_p036_ur_status CHECK (
        status IN ('GENERATING', 'GENERATED', 'SENT', 'FAILED', 'ARCHIVED')
    ),
    CONSTRAINT chk_p036_ur_file_size CHECK (
        file_size_bytes IS NULL OR file_size_bytes >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_ur_tenant         ON pack036_utility_analysis.gl_utility_reports(tenant_id);
CREATE INDEX idx_p036_ur_facility       ON pack036_utility_analysis.gl_utility_reports(facility_id);
CREATE INDEX idx_p036_ur_portfolio      ON pack036_utility_analysis.gl_utility_reports(portfolio_id);
CREATE INDEX idx_p036_ur_type           ON pack036_utility_analysis.gl_utility_reports(report_type);
CREATE INDEX idx_p036_ur_format         ON pack036_utility_analysis.gl_utility_reports(format);
CREATE INDEX idx_p036_ur_period         ON pack036_utility_analysis.gl_utility_reports(period_start DESC);
CREATE INDEX idx_p036_ur_status         ON pack036_utility_analysis.gl_utility_reports(status);
CREATE INDEX idx_p036_ur_generated      ON pack036_utility_analysis.gl_utility_reports(generated_at DESC);
CREATE INDEX idx_p036_ur_created        ON pack036_utility_analysis.gl_utility_reports(created_at DESC);
CREATE INDEX idx_p036_ur_data           ON pack036_utility_analysis.gl_utility_reports USING GIN(data);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_ur_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_utility_reports
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack036_utility_analysis.gl_utility_kpis
-- =============================================================================
-- Utility KPI dashboard data. Stores current, previous, target values,
-- and RAG status for key utility performance indicators by facility
-- and commodity.

CREATE TABLE pack036_utility_analysis.gl_utility_kpis (
    kpi_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id               UUID            NOT NULL,
    facility_id             UUID            NOT NULL,
    period                  DATE            NOT NULL,
    kpi_name                VARCHAR(100)    NOT NULL,
    kpi_code                VARCHAR(50),
    category                VARCHAR(50)     NOT NULL,
    commodity               VARCHAR(30),
    current_value           NUMERIC(16,4)   NOT NULL,
    previous_value          NUMERIC(16,4),
    change_pct              NUMERIC(8,4),
    target_value            NUMERIC(16,4),
    variance_from_target    NUMERIC(16,4),
    rag_status              VARCHAR(10)     NOT NULL DEFAULT 'GREEN',
    unit                    VARCHAR(30),
    trend                   VARCHAR(20),
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_uk_category CHECK (
        category IN (
            'COST', 'CONSUMPTION', 'DEMAND', 'EFFICIENCY',
            'BENCHMARK', 'BUDGET', 'PROCUREMENT', 'REGULATORY',
            'SUSTAINABILITY', 'QUALITY'
        )
    ),
    CONSTRAINT chk_p036_uk_commodity CHECK (
        commodity IS NULL OR commodity IN (
            'ELECTRICITY', 'NATURAL_GAS', 'WATER', 'STEAM',
            'CHILLED_WATER', 'ALL'
        )
    ),
    CONSTRAINT chk_p036_uk_rag CHECK (
        rag_status IN ('RED', 'AMBER', 'GREEN', 'GREY')
    ),
    CONSTRAINT chk_p036_uk_trend CHECK (
        trend IS NULL OR trend IN (
            'IMPROVING', 'STABLE', 'DECLINING', 'VOLATILE', 'INSUFFICIENT_DATA'
        )
    ),
    CONSTRAINT uq_p036_uk_fac_period_kpi UNIQUE (facility_id, period, kpi_name, commodity)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p036_uk_tenant         ON pack036_utility_analysis.gl_utility_kpis(tenant_id);
CREATE INDEX idx_p036_uk_facility       ON pack036_utility_analysis.gl_utility_kpis(facility_id);
CREATE INDEX idx_p036_uk_period         ON pack036_utility_analysis.gl_utility_kpis(period DESC);
CREATE INDEX idx_p036_uk_kpi_name       ON pack036_utility_analysis.gl_utility_kpis(kpi_name);
CREATE INDEX idx_p036_uk_category       ON pack036_utility_analysis.gl_utility_kpis(category);
CREATE INDEX idx_p036_uk_commodity      ON pack036_utility_analysis.gl_utility_kpis(commodity);
CREATE INDEX idx_p036_uk_rag            ON pack036_utility_analysis.gl_utility_kpis(rag_status);
CREATE INDEX idx_p036_uk_trend          ON pack036_utility_analysis.gl_utility_kpis(trend);
CREATE INDEX idx_p036_uk_created        ON pack036_utility_analysis.gl_utility_kpis(created_at DESC);

-- Composite: facility + period for KPI dashboard lookup
CREATE INDEX idx_p036_uk_fac_period     ON pack036_utility_analysis.gl_utility_kpis(facility_id, period DESC);

-- Partial: non-green KPIs for attention dashboard
CREATE INDEX idx_p036_uk_attention      ON pack036_utility_analysis.gl_utility_kpis(facility_id, rag_status, category)
    WHERE rag_status IN ('RED', 'AMBER');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p036_uk_updated
    BEFORE UPDATE ON pack036_utility_analysis.gl_utility_kpis
    FOR EACH ROW EXECUTE FUNCTION pack036_utility_analysis.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack036_utility_analysis.pack036_audit_trail
-- =============================================================================
-- Pack-level audit trail logging all significant actions across
-- PACK-036 entities for compliance and provenance tracking.

CREATE TABLE pack036_utility_analysis.pack036_audit_trail (
    audit_trail_id          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID,
    tenant_id               UUID,
    action                  VARCHAR(50)     NOT NULL,
    entity_type             VARCHAR(100)    NOT NULL,
    entity_id               UUID,
    actor                   TEXT            NOT NULL,
    actor_role              VARCHAR(50),
    ip_address              VARCHAR(45),
    old_values              JSONB,
    new_values              JSONB,
    details                 JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p036_trail_action CHECK (
        action IN ('CREATE', 'UPDATE', 'DELETE', 'READ', 'APPROVE', 'REJECT',
                   'SUBMIT', 'EXPORT', 'IMPORT', 'CALCULATE', 'VERIFY',
                   'ARCHIVE', 'RESTORE', 'LOCK', 'UNLOCK', 'ANALYSE',
                   'FORECAST', 'ALLOCATE', 'RECONCILE', 'ALERT')
    )
);

-- Indexes
CREATE INDEX idx_p036_trail_facility    ON pack036_utility_analysis.pack036_audit_trail(facility_id);
CREATE INDEX idx_p036_trail_tenant      ON pack036_utility_analysis.pack036_audit_trail(tenant_id);
CREATE INDEX idx_p036_trail_action      ON pack036_utility_analysis.pack036_audit_trail(action);
CREATE INDEX idx_p036_trail_entity      ON pack036_utility_analysis.pack036_audit_trail(entity_type, entity_id);
CREATE INDEX idx_p036_trail_actor       ON pack036_utility_analysis.pack036_audit_trail(actor);
CREATE INDEX idx_p036_trail_created     ON pack036_utility_analysis.pack036_audit_trail(created_at DESC);
CREATE INDEX idx_p036_trail_details     ON pack036_utility_analysis.pack036_audit_trail USING GIN(details);

-- =============================================================================
-- Materialized View 1: mv_facility_cost_summary
-- =============================================================================
-- Latest cost summary per facility for dashboard rendering.
-- Aggregates total spend, consumption, and cost metrics across commodities.

CREATE MATERIALIZED VIEW pack036_utility_analysis.mv_facility_cost_summary AS
SELECT
    ub.facility_id,
    ub.tenant_id,
    ub.commodity_type,
    DATE_TRUNC('month', ub.billing_period_start) AS billing_month,
    COUNT(ub.bill_id) AS bill_count,
    SUM(ub.total_consumption) AS total_consumption,
    ub.consumption_unit,
    SUM(ub.total_amount) AS total_amount_eur,
    SUM(ub.taxes) AS total_taxes_eur,
    SUM(ub.total_amount + COALESCE(ub.taxes, 0)) AS total_cost_eur,
    AVG(ub.peak_demand_kw) AS avg_peak_demand_kw,
    MAX(ub.peak_demand_kw) AS max_peak_demand_kw,
    AVG(ub.power_factor) AS avg_power_factor
FROM pack036_utility_analysis.gl_utility_bills ub
WHERE ub.bill_status IN ('VALIDATED', 'APPROVED')
GROUP BY ub.facility_id, ub.tenant_id, ub.commodity_type,
         DATE_TRUNC('month', ub.billing_period_start), ub.consumption_unit
WITH NO DATA;

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_p036_mv_fcs_uniq ON pack036_utility_analysis.mv_facility_cost_summary(facility_id, commodity_type, billing_month, consumption_unit);
CREATE INDEX idx_p036_mv_fcs_tenant ON pack036_utility_analysis.mv_facility_cost_summary(tenant_id);
CREATE INDEX idx_p036_mv_fcs_commodity ON pack036_utility_analysis.mv_facility_cost_summary(commodity_type);
CREATE INDEX idx_p036_mv_fcs_month ON pack036_utility_analysis.mv_facility_cost_summary(billing_month DESC);
CREATE INDEX idx_p036_mv_fcs_cost ON pack036_utility_analysis.mv_facility_cost_summary(total_cost_eur DESC);

-- =============================================================================
-- Materialized View 2: mv_portfolio_utility_overview
-- =============================================================================
-- Portfolio-wide utility overview aggregating facility costs and
-- consumption by commodity for portfolio dashboard.

CREATE MATERIALIZED VIEW pack036_utility_analysis.mv_portfolio_utility_overview AS
SELECT
    ub.tenant_id,
    ub.commodity_type,
    DATE_TRUNC('month', ub.billing_period_start) AS billing_month,
    COUNT(DISTINCT ub.facility_id) AS facilities_count,
    COUNT(ub.bill_id) AS total_bills,
    SUM(ub.total_consumption) AS total_consumption,
    SUM(ub.total_amount) AS total_amount_eur,
    SUM(ub.total_amount + COALESCE(ub.taxes, 0)) AS total_cost_eur,
    AVG(ub.total_amount) AS avg_bill_amount_eur,
    MAX(ub.peak_demand_kw) AS portfolio_peak_demand_kw
FROM pack036_utility_analysis.gl_utility_bills ub
WHERE ub.bill_status IN ('VALIDATED', 'APPROVED')
GROUP BY ub.tenant_id, ub.commodity_type,
         DATE_TRUNC('month', ub.billing_period_start)
WITH NO DATA;

-- Indexes on materialized view
CREATE UNIQUE INDEX idx_p036_mv_puo_uniq ON pack036_utility_analysis.mv_portfolio_utility_overview(tenant_id, commodity_type, billing_month);
CREATE INDEX idx_p036_mv_puo_tenant ON pack036_utility_analysis.mv_portfolio_utility_overview(tenant_id);
CREATE INDEX idx_p036_mv_puo_commodity ON pack036_utility_analysis.mv_portfolio_utility_overview(commodity_type);
CREATE INDEX idx_p036_mv_puo_month ON pack036_utility_analysis.mv_portfolio_utility_overview(billing_month DESC);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack036_utility_analysis.gl_weather_normalizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_degree_days ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_utility_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.gl_utility_kpis ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack036_utility_analysis.pack036_audit_trail ENABLE ROW LEVEL SECURITY;

CREATE POLICY p036_wn_tenant_isolation
    ON pack036_utility_analysis.gl_weather_normalizations
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_wn_service_bypass
    ON pack036_utility_analysis.gl_weather_normalizations
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- Degree days are shared reference data (no tenant isolation)
CREATE POLICY p036_dd_public_read
    ON pack036_utility_analysis.gl_degree_days
    USING (TRUE);
CREATE POLICY p036_dd_service_bypass
    ON pack036_utility_analysis.gl_degree_days
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_ur_tenant_isolation
    ON pack036_utility_analysis.gl_utility_reports
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_ur_service_bypass
    ON pack036_utility_analysis.gl_utility_reports
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_uk_tenant_isolation
    ON pack036_utility_analysis.gl_utility_kpis
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_uk_service_bypass
    ON pack036_utility_analysis.gl_utility_kpis
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p036_trail_tenant_isolation
    ON pack036_utility_analysis.pack036_audit_trail
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p036_trail_service_bypass
    ON pack036_utility_analysis.pack036_audit_trail
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_weather_normalizations TO PUBLIC;
GRANT SELECT, INSERT ON pack036_utility_analysis.gl_degree_days TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_utility_reports TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack036_utility_analysis.gl_utility_kpis TO PUBLIC;
GRANT SELECT, INSERT ON pack036_utility_analysis.pack036_audit_trail TO PUBLIC;
GRANT SELECT ON pack036_utility_analysis.mv_facility_cost_summary TO PUBLIC;
GRANT SELECT ON pack036_utility_analysis.mv_portfolio_utility_overview TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack036_utility_analysis.gl_weather_normalizations IS
    'Weather normalization model results per facility using degree-day regression or change-point models with ASHRAE 14 validation.';

COMMENT ON TABLE pack036_utility_analysis.gl_degree_days IS
    'Heating and cooling degree day reference data by weather station for normalization calculations.';

COMMENT ON TABLE pack036_utility_analysis.gl_utility_reports IS
    'Generated utility analysis reports with type, format, content, and structured data for programmatic consumption.';

COMMENT ON TABLE pack036_utility_analysis.gl_utility_kpis IS
    'Utility KPI dashboard data with current, previous, target values, RAG status, and trend by facility and commodity.';

COMMENT ON TABLE pack036_utility_analysis.pack036_audit_trail IS
    'Pack-level audit trail logging all significant actions across PACK-036 entities for compliance and provenance.';

COMMENT ON MATERIALIZED VIEW pack036_utility_analysis.mv_facility_cost_summary IS
    'Facility cost summary by commodity and month for dashboard rendering. Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY pack036_utility_analysis.mv_facility_cost_summary;';

COMMENT ON MATERIALIZED VIEW pack036_utility_analysis.mv_portfolio_utility_overview IS
    'Portfolio-wide utility overview by commodity and month. Refresh with: REFRESH MATERIALIZED VIEW CONCURRENTLY pack036_utility_analysis.mv_portfolio_utility_overview;';

COMMENT ON COLUMN pack036_utility_analysis.gl_weather_normalizations.normalization_id IS
    'Unique identifier for the weather normalization result.';
COMMENT ON COLUMN pack036_utility_analysis.gl_weather_normalizations.model_type IS
    'Normalization model type: SIMPLE_REGRESSION, THREE_PARAMETER_COOLING/HEATING, FOUR_PARAMETER, FIVE_PARAMETER, CHANGE_POINT, DEGREE_DAY.';
COMMENT ON COLUMN pack036_utility_analysis.gl_weather_normalizations.r_squared IS
    'Coefficient of determination (0 to 1). ASHRAE 14 monthly: R-squared >= 0.75.';
COMMENT ON COLUMN pack036_utility_analysis.gl_weather_normalizations.cv_rmse_pct IS
    'Coefficient of Variation of RMSE in percent. ASHRAE 14 monthly: CV-RMSE <= 15%.';
COMMENT ON COLUMN pack036_utility_analysis.gl_weather_normalizations.nmbe_pct IS
    'Normalized Mean Bias Error in percent. ASHRAE 14 monthly: |NMBE| <= 5%.';
COMMENT ON COLUMN pack036_utility_analysis.gl_weather_normalizations.ashrae14_passed IS
    'Whether the model meets ASHRAE Guideline 14 statistical criteria for M&V.';
COMMENT ON COLUMN pack036_utility_analysis.gl_weather_normalizations.actual_consumption_kwh IS
    'Actual metered consumption for the analysis period in kWh.';
COMMENT ON COLUMN pack036_utility_analysis.gl_weather_normalizations.normalized_consumption_kwh IS
    'Weather-normalized consumption adjusted to TMY or baseline weather conditions.';
COMMENT ON COLUMN pack036_utility_analysis.gl_weather_normalizations.weather_impact_kwh IS
    'Consumption difference attributable to weather variation (actual - normalized).';
COMMENT ON COLUMN pack036_utility_analysis.gl_weather_normalizations.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack036_utility_analysis.gl_degree_days.dd_id IS
    'Unique identifier for the degree day record.';
COMMENT ON COLUMN pack036_utility_analysis.gl_degree_days.station_id IS
    'Weather station identifier (e.g., ICAO code).';
COMMENT ON COLUMN pack036_utility_analysis.gl_degree_days.hdd IS
    'Heating degree days for the period.';
COMMENT ON COLUMN pack036_utility_analysis.gl_degree_days.cdd IS
    'Cooling degree days for the period.';
COMMENT ON COLUMN pack036_utility_analysis.gl_degree_days.base_temp_heating_c IS
    'Base temperature for HDD calculation in degrees Celsius (default 15.5).';
COMMENT ON COLUMN pack036_utility_analysis.gl_degree_days.base_temp_cooling_c IS
    'Base temperature for CDD calculation in degrees Celsius (default 18.0).';
COMMENT ON COLUMN pack036_utility_analysis.gl_degree_days.is_tmy IS
    'Whether this is Typical Meteorological Year (TMY) data.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_reports.report_type IS
    'Report type: MONTHLY_SUMMARY, QUARTERLY_REVIEW, ANNUAL_REVIEW, BILL_AUDIT, RATE_ANALYSIS, etc.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_reports.content IS
    'Report content as text/HTML for inline viewing.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_reports.data IS
    'Structured report data as JSONB for programmatic consumption and API responses.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_reports.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_kpis.kpi_name IS
    'KPI name (e.g., Cost per kWh, Peak Demand, Load Factor, EUI).';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_kpis.rag_status IS
    'RAG status: RED (off target), AMBER (warning), GREEN (on target), GREY (no data).';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_kpis.trend IS
    'Performance trend: IMPROVING, STABLE, DECLINING, VOLATILE, INSUFFICIENT_DATA.';
COMMENT ON COLUMN pack036_utility_analysis.gl_utility_kpis.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
