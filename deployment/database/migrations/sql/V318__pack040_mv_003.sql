-- =============================================================================
-- V318: PACK-040 M&V Pack - Adjustment Records
-- =============================================================================
-- Pack:         PACK-040 (Measurement & Verification Pack)
-- Migration:    003 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates tables for routine and non-routine adjustments per IPMVP
-- methodology. Routine adjustments normalize for weather, production, and
-- occupancy changes between baseline and reporting periods. Non-routine
-- adjustments account for structural changes (floor area, equipment, schedule).
-- Includes adjustment factor storage, documentation, and summaries.
--
-- Tables (5):
--   1. pack040_mv.mv_routine_adjustments
--   2. pack040_mv.mv_nonroutine_adjustments
--   3. pack040_mv.mv_adjustment_factors
--   4. pack040_mv.mv_adjustment_docs
--   5. pack040_mv.mv_adjustment_summaries
--
-- Previous: V317__pack040_mv_002.sql
-- =============================================================================

SET search_path TO pack040_mv, public;

-- =============================================================================
-- Table 1: pack040_mv.mv_routine_adjustments
-- =============================================================================
-- Routine adjustments per IPMVP that normalize the baseline prediction to
-- reporting period conditions. These adjustments account for expected changes
-- in independent variables (weather, production, occupancy) and are applied
-- automatically through the regression model's independent variables.

CREATE TABLE pack040_mv.mv_routine_adjustments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    baseline_id                 UUID            NOT NULL REFERENCES pack040_mv.mv_baselines(id) ON DELETE CASCADE,
    adjustment_name             VARCHAR(255)    NOT NULL,
    adjustment_type             VARCHAR(50)     NOT NULL DEFAULT 'WEATHER',
    adjustment_method           VARCHAR(50)     NOT NULL DEFAULT 'REGRESSION_MODEL',
    reporting_period_start      DATE            NOT NULL,
    reporting_period_end        DATE            NOT NULL,
    -- Weather adjustment details
    baseline_hdd                NUMERIC(12,3),
    reporting_hdd               NUMERIC(12,3),
    baseline_cdd                NUMERIC(12,3),
    reporting_cdd               NUMERIC(12,3),
    hdd_balance_point_f         NUMERIC(6,2),
    cdd_balance_point_f         NUMERIC(6,2),
    baseline_avg_temp_f         NUMERIC(8,3),
    reporting_avg_temp_f        NUMERIC(8,3),
    -- Production adjustment details
    baseline_production         NUMERIC(18,3),
    reporting_production        NUMERIC(18,3),
    production_unit             VARCHAR(50),
    production_normalization_factor NUMERIC(10,6),
    -- Occupancy adjustment details
    baseline_occupancy_pct      NUMERIC(7,4),
    reporting_occupancy_pct     NUMERIC(7,4),
    -- Operating hours adjustment
    baseline_operating_hours    NUMERIC(8,2),
    reporting_operating_hours   NUMERIC(8,2),
    -- Calculated values
    baseline_predicted_kwh      NUMERIC(18,3)   NOT NULL,
    adjusted_baseline_kwh       NUMERIC(18,3)   NOT NULL,
    adjustment_value_kwh        NUMERIC(18,3)   NOT NULL,
    adjustment_pct              NUMERIC(8,4),
    adjustment_direction        VARCHAR(10)     NOT NULL DEFAULT 'POSITIVE',
    -- Status
    adjustment_status           VARCHAR(20)     NOT NULL DEFAULT 'CALCULATED',
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_ra_type CHECK (
        adjustment_type IN (
            'WEATHER', 'PRODUCTION', 'OCCUPANCY', 'OPERATING_HOURS',
            'COMBINED_WEATHER_PRODUCTION', 'COMBINED_ALL', 'OTHER'
        )
    ),
    CONSTRAINT chk_p040_ra_method CHECK (
        adjustment_method IN (
            'REGRESSION_MODEL', 'DEGREE_DAY_REGRESSION', 'RATIO_NORMALIZATION',
            'SIMPLE_RATIO', 'INTERPOLATION', 'ENGINEERING_ESTIMATE'
        )
    ),
    CONSTRAINT chk_p040_ra_dates CHECK (
        reporting_period_start < reporting_period_end
    ),
    CONSTRAINT chk_p040_ra_direction CHECK (
        adjustment_direction IN ('POSITIVE', 'NEGATIVE', 'ZERO')
    ),
    CONSTRAINT chk_p040_ra_status CHECK (
        adjustment_status IN (
            'CALCULATED', 'REVIEWED', 'APPROVED', 'REJECTED', 'REVISED'
        )
    ),
    CONSTRAINT chk_p040_ra_occupancy_bl CHECK (
        baseline_occupancy_pct IS NULL OR
        (baseline_occupancy_pct >= 0 AND baseline_occupancy_pct <= 100)
    ),
    CONSTRAINT chk_p040_ra_occupancy_rp CHECK (
        reporting_occupancy_pct IS NULL OR
        (reporting_occupancy_pct >= 0 AND reporting_occupancy_pct <= 100)
    ),
    CONSTRAINT chk_p040_ra_hdd CHECK (
        baseline_hdd IS NULL OR baseline_hdd >= 0
    ),
    CONSTRAINT chk_p040_ra_cdd CHECK (
        baseline_cdd IS NULL OR baseline_cdd >= 0
    ),
    CONSTRAINT uq_p040_ra_project_period_type UNIQUE (project_id, baseline_id, adjustment_type, reporting_period_start)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_ra_tenant            ON pack040_mv.mv_routine_adjustments(tenant_id);
CREATE INDEX idx_p040_ra_project           ON pack040_mv.mv_routine_adjustments(project_id);
CREATE INDEX idx_p040_ra_baseline          ON pack040_mv.mv_routine_adjustments(baseline_id);
CREATE INDEX idx_p040_ra_type              ON pack040_mv.mv_routine_adjustments(adjustment_type);
CREATE INDEX idx_p040_ra_method            ON pack040_mv.mv_routine_adjustments(adjustment_method);
CREATE INDEX idx_p040_ra_period            ON pack040_mv.mv_routine_adjustments(reporting_period_start, reporting_period_end);
CREATE INDEX idx_p040_ra_status            ON pack040_mv.mv_routine_adjustments(adjustment_status);
CREATE INDEX idx_p040_ra_created           ON pack040_mv.mv_routine_adjustments(created_at DESC);

-- Composite: project + approved adjustments
CREATE INDEX idx_p040_ra_project_approved  ON pack040_mv.mv_routine_adjustments(project_id, reporting_period_start)
    WHERE adjustment_status = 'APPROVED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_ra_updated
    BEFORE UPDATE ON pack040_mv.mv_routine_adjustments
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack040_mv.mv_nonroutine_adjustments
-- =============================================================================
-- Non-routine adjustments per IPMVP that account for structural changes
-- between baseline and reporting periods that are not captured by the
-- regression model's independent variables. These include floor area changes,
-- equipment additions/removals, schedule changes, and other one-time events.

CREATE TABLE pack040_mv.mv_nonroutine_adjustments (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    baseline_id                 UUID            REFERENCES pack040_mv.mv_baselines(id) ON DELETE SET NULL,
    adjustment_name             VARCHAR(255)    NOT NULL,
    adjustment_type             VARCHAR(50)     NOT NULL DEFAULT 'EQUIPMENT_ADDITION',
    adjustment_method           VARCHAR(50)     NOT NULL DEFAULT 'ENGINEERING_ESTIMATE',
    effective_date              DATE            NOT NULL,
    end_date                    DATE,
    is_permanent                BOOLEAN         NOT NULL DEFAULT true,
    -- Change details
    change_description          TEXT            NOT NULL,
    affected_system             VARCHAR(100),
    affected_end_use            VARCHAR(50),
    -- Floor area changes
    floor_area_change_m2        NUMERIC(12,2),
    original_floor_area_m2      NUMERIC(12,2),
    new_floor_area_m2           NUMERIC(12,2),
    eui_for_scaling             NUMERIC(10,3),
    -- Equipment changes
    equipment_added             TEXT,
    equipment_removed           TEXT,
    equipment_rated_kw          NUMERIC(12,3),
    equipment_annual_kwh        NUMERIC(18,3),
    operating_hours_per_year    NUMERIC(8,2),
    load_factor                 NUMERIC(5,4),
    diversity_factor            NUMERIC(5,4),
    -- Schedule changes
    old_schedule_description    TEXT,
    new_schedule_description    TEXT,
    schedule_delta_hours        NUMERIC(8,2),
    -- Static factor
    static_factor_value_kwh     NUMERIC(18,3),
    static_factor_direction     VARCHAR(10),
    -- Calculated values
    adjustment_value_kwh        NUMERIC(18,3)   NOT NULL,
    adjustment_value_monthly_kwh NUMERIC(18,3),
    adjustment_pct_of_baseline  NUMERIC(8,4),
    uncertainty_pct             NUMERIC(8,4),
    -- Status
    adjustment_status           VARCHAR(20)     NOT NULL DEFAULT 'ESTIMATED',
    justification               TEXT,
    data_source                 VARCHAR(255),
    verified_by                 VARCHAR(255),
    verified_at                 TIMESTAMPTZ,
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    notes                       TEXT,
    supporting_docs             JSONB           DEFAULT '[]',
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_nra_type CHECK (
        adjustment_type IN (
            'FLOOR_AREA_CHANGE', 'EQUIPMENT_ADDITION', 'EQUIPMENT_REMOVAL',
            'SCHEDULE_CHANGE', 'OCCUPANCY_CHANGE', 'PROCESS_CHANGE',
            'FUEL_SWITCH', 'RATE_CHANGE', 'STATIC_FACTOR',
            'BUILDING_MODIFICATION', 'TENANT_CHANGE', 'OTHER'
        )
    ),
    CONSTRAINT chk_p040_nra_method CHECK (
        adjustment_method IN (
            'ENGINEERING_ESTIMATE', 'METERED_DATA', 'NAMEPLATE_CALCULATION',
            'PROPORTIONAL_SCALING', 'SIMULATION', 'STIPULATED',
            'SURVEY_DATA', 'MANUFACTURER_DATA', 'HISTORICAL_ANALYSIS'
        )
    ),
    CONSTRAINT chk_p040_nra_end_use CHECK (
        affected_end_use IS NULL OR affected_end_use IN (
            'HVAC', 'LIGHTING', 'PLUG_LOAD', 'PROCESS', 'DOMESTIC_HW',
            'REFRIGERATION', 'COOKING', 'LAUNDRY', 'ELEVATOR',
            'IT_EQUIPMENT', 'COMPRESSED_AIR', 'PUMPING', 'VENTILATION',
            'WHOLE_BUILDING', 'OTHER'
        )
    ),
    CONSTRAINT chk_p040_nra_status CHECK (
        adjustment_status IN (
            'ESTIMATED', 'MEASURED', 'VERIFIED', 'APPROVED',
            'REJECTED', 'REVISED', 'STIPULATED'
        )
    ),
    CONSTRAINT chk_p040_nra_direction CHECK (
        static_factor_direction IS NULL OR
        static_factor_direction IN ('INCREASE', 'DECREASE')
    ),
    CONSTRAINT chk_p040_nra_dates CHECK (
        end_date IS NULL OR effective_date <= end_date
    ),
    CONSTRAINT chk_p040_nra_load_factor CHECK (
        load_factor IS NULL OR (load_factor >= 0 AND load_factor <= 1)
    ),
    CONSTRAINT chk_p040_nra_diversity CHECK (
        diversity_factor IS NULL OR (diversity_factor >= 0 AND diversity_factor <= 1)
    ),
    CONSTRAINT chk_p040_nra_uncertainty CHECK (
        uncertainty_pct IS NULL OR (uncertainty_pct >= 0 AND uncertainty_pct <= 100)
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_nra_tenant           ON pack040_mv.mv_nonroutine_adjustments(tenant_id);
CREATE INDEX idx_p040_nra_project          ON pack040_mv.mv_nonroutine_adjustments(project_id);
CREATE INDEX idx_p040_nra_baseline         ON pack040_mv.mv_nonroutine_adjustments(baseline_id);
CREATE INDEX idx_p040_nra_type             ON pack040_mv.mv_nonroutine_adjustments(adjustment_type);
CREATE INDEX idx_p040_nra_method           ON pack040_mv.mv_nonroutine_adjustments(adjustment_method);
CREATE INDEX idx_p040_nra_effective        ON pack040_mv.mv_nonroutine_adjustments(effective_date);
CREATE INDEX idx_p040_nra_status           ON pack040_mv.mv_nonroutine_adjustments(adjustment_status);
CREATE INDEX idx_p040_nra_created          ON pack040_mv.mv_nonroutine_adjustments(created_at DESC);

-- Composite: project + approved non-routine adjustments
CREATE INDEX idx_p040_nra_project_appr     ON pack040_mv.mv_nonroutine_adjustments(project_id, effective_date)
    WHERE adjustment_status = 'APPROVED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_nra_updated
    BEFORE UPDATE ON pack040_mv.mv_nonroutine_adjustments
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack040_mv.mv_adjustment_factors
-- =============================================================================
-- Reusable adjustment factors that can be applied across multiple reporting
-- periods. Stores standardized correction factors for common adjustments
-- such as weather normalization coefficients, production scaling ratios,
-- and degree-day regression coefficients.

CREATE TABLE pack040_mv.mv_adjustment_factors (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    factor_name                 VARCHAR(255)    NOT NULL,
    factor_type                 VARCHAR(50)     NOT NULL DEFAULT 'WEATHER_COEFFICIENT',
    factor_category             VARCHAR(30)     NOT NULL DEFAULT 'ROUTINE',
    associated_variable         VARCHAR(100),
    factor_value                NUMERIC(18,8)   NOT NULL,
    factor_unit                 VARCHAR(50),
    standard_error              NUMERIC(18,8),
    confidence_lower            NUMERIC(18,8),
    confidence_upper            NUMERIC(18,8),
    confidence_level_pct        NUMERIC(5,2)    DEFAULT 90.0,
    effective_from              DATE            NOT NULL,
    effective_to                DATE,
    derivation_method           VARCHAR(50)     NOT NULL DEFAULT 'REGRESSION',
    source_baseline_id          UUID            REFERENCES pack040_mv.mv_baselines(id) ON DELETE SET NULL,
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_af_type CHECK (
        factor_type IN (
            'WEATHER_COEFFICIENT', 'HDD_COEFFICIENT', 'CDD_COEFFICIENT',
            'PRODUCTION_COEFFICIENT', 'OCCUPANCY_COEFFICIENT',
            'OPERATING_HOURS_COEFFICIENT', 'FLOOR_AREA_SCALING',
            'LOAD_DIVERSITY', 'DEGREE_DAY_RATIO', 'STATIC_CORRECTION',
            'SEASONAL_FACTOR', 'TIME_OF_USE_FACTOR'
        )
    ),
    CONSTRAINT chk_p040_af_category CHECK (
        factor_category IN ('ROUTINE', 'NON_ROUTINE', 'STATIC')
    ),
    CONSTRAINT chk_p040_af_derivation CHECK (
        derivation_method IN (
            'REGRESSION', 'ENGINEERING_CALCULATION', 'METERED',
            'STIPULATED', 'HISTORICAL_AVERAGE', 'MANUFACTURER'
        )
    ),
    CONSTRAINT chk_p040_af_dates CHECK (
        effective_to IS NULL OR effective_from <= effective_to
    ),
    CONSTRAINT chk_p040_af_se CHECK (
        standard_error IS NULL OR standard_error >= 0
    ),
    CONSTRAINT chk_p040_af_confidence CHECK (
        confidence_lower IS NULL OR confidence_upper IS NULL OR
        confidence_lower <= confidence_upper
    ),
    CONSTRAINT uq_p040_af_project_name_date UNIQUE (project_id, factor_name, effective_from)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_af_tenant            ON pack040_mv.mv_adjustment_factors(tenant_id);
CREATE INDEX idx_p040_af_project           ON pack040_mv.mv_adjustment_factors(project_id);
CREATE INDEX idx_p040_af_type              ON pack040_mv.mv_adjustment_factors(factor_type);
CREATE INDEX idx_p040_af_category          ON pack040_mv.mv_adjustment_factors(factor_category);
CREATE INDEX idx_p040_af_active            ON pack040_mv.mv_adjustment_factors(is_active) WHERE is_active = true;
CREATE INDEX idx_p040_af_effective         ON pack040_mv.mv_adjustment_factors(effective_from, effective_to);
CREATE INDEX idx_p040_af_baseline          ON pack040_mv.mv_adjustment_factors(source_baseline_id);
CREATE INDEX idx_p040_af_created           ON pack040_mv.mv_adjustment_factors(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_af_updated
    BEFORE UPDATE ON pack040_mv.mv_adjustment_factors
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 4: pack040_mv.mv_adjustment_docs
-- =============================================================================
-- Supporting documentation for non-routine adjustments. Stores references
-- to engineering analyses, equipment specifications, invoices, photos,
-- and other evidence that justifies the magnitude and direction of
-- non-routine adjustments per IPMVP documentation requirements.

CREATE TABLE pack040_mv.mv_adjustment_docs (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    adjustment_id               UUID            NOT NULL,
    adjustment_source           VARCHAR(20)     NOT NULL DEFAULT 'NON_ROUTINE',
    document_type               VARCHAR(50)     NOT NULL DEFAULT 'ENGINEERING_ANALYSIS',
    document_name               VARCHAR(255)    NOT NULL,
    document_ref                VARCHAR(255),
    file_path                   VARCHAR(500),
    file_size_bytes             BIGINT,
    mime_type                   VARCHAR(100),
    document_date               DATE,
    author                      VARCHAR(255),
    description                 TEXT,
    key_findings                TEXT,
    supporting_value_kwh        NUMERIC(18,3),
    is_primary_evidence         BOOLEAN         NOT NULL DEFAULT false,
    review_status               VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    reviewed_by                 VARCHAR(255),
    reviewed_at                 TIMESTAMPTZ,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_ad_source CHECK (
        adjustment_source IN ('ROUTINE', 'NON_ROUTINE')
    ),
    CONSTRAINT chk_p040_ad_doc_type CHECK (
        document_type IN (
            'ENGINEERING_ANALYSIS', 'EQUIPMENT_SPECIFICATION', 'INVOICE',
            'PHOTOGRAPH', 'UTILITY_BILL', 'COMMISSIONING_REPORT',
            'INSPECTION_REPORT', 'SURVEY_DATA', 'MANUFACTURER_DATA',
            'SIMULATION_REPORT', 'MEASUREMENT_DATA', 'CONTRACT',
            'APPROVAL_MEMO', 'CALCULATION_WORKSHEET', 'OTHER'
        )
    ),
    CONSTRAINT chk_p040_ad_review_status CHECK (
        review_status IN ('PENDING', 'REVIEWED', 'ACCEPTED', 'REJECTED', 'REVISED')
    ),
    CONSTRAINT chk_p040_ad_file_size CHECK (
        file_size_bytes IS NULL OR file_size_bytes >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_ad_tenant            ON pack040_mv.mv_adjustment_docs(tenant_id);
CREATE INDEX idx_p040_ad_adjustment        ON pack040_mv.mv_adjustment_docs(adjustment_id);
CREATE INDEX idx_p040_ad_source            ON pack040_mv.mv_adjustment_docs(adjustment_source);
CREATE INDEX idx_p040_ad_doc_type          ON pack040_mv.mv_adjustment_docs(document_type);
CREATE INDEX idx_p040_ad_review            ON pack040_mv.mv_adjustment_docs(review_status);
CREATE INDEX idx_p040_ad_primary           ON pack040_mv.mv_adjustment_docs(is_primary_evidence) WHERE is_primary_evidence = true;
CREATE INDEX idx_p040_ad_created           ON pack040_mv.mv_adjustment_docs(created_at DESC);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_ad_updated
    BEFORE UPDATE ON pack040_mv.mv_adjustment_docs
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- =============================================================================
-- Table 5: pack040_mv.mv_adjustment_summaries
-- =============================================================================
-- Aggregated adjustment summaries per reporting period combining all routine
-- and non-routine adjustments. Provides a single consolidated view of the
-- total adjustment applied to the baseline for each reporting period, used
-- as input to the savings calculation engine.

CREATE TABLE pack040_mv.mv_adjustment_summaries (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    project_id                  UUID            NOT NULL REFERENCES pack040_mv.mv_projects(id) ON DELETE CASCADE,
    baseline_id                 UUID            NOT NULL REFERENCES pack040_mv.mv_baselines(id) ON DELETE CASCADE,
    reporting_period_start      DATE            NOT NULL,
    reporting_period_end        DATE            NOT NULL,
    -- Routine adjustments total
    routine_adjustment_count    INTEGER         NOT NULL DEFAULT 0,
    routine_weather_kwh         NUMERIC(18,3)   DEFAULT 0,
    routine_production_kwh      NUMERIC(18,3)   DEFAULT 0,
    routine_occupancy_kwh       NUMERIC(18,3)   DEFAULT 0,
    routine_operating_hours_kwh NUMERIC(18,3)   DEFAULT 0,
    routine_total_kwh           NUMERIC(18,3)   NOT NULL DEFAULT 0,
    -- Non-routine adjustments total
    nonroutine_adjustment_count INTEGER         NOT NULL DEFAULT 0,
    nonroutine_floor_area_kwh   NUMERIC(18,3)   DEFAULT 0,
    nonroutine_equipment_kwh    NUMERIC(18,3)   DEFAULT 0,
    nonroutine_schedule_kwh     NUMERIC(18,3)   DEFAULT 0,
    nonroutine_other_kwh        NUMERIC(18,3)   DEFAULT 0,
    nonroutine_total_kwh        NUMERIC(18,3)   NOT NULL DEFAULT 0,
    -- Combined totals
    total_adjustment_kwh        NUMERIC(18,3)   NOT NULL DEFAULT 0,
    total_adjustment_pct        NUMERIC(8,4),
    -- Baseline values
    unadjusted_baseline_kwh     NUMERIC(18,3)   NOT NULL,
    adjusted_baseline_kwh       NUMERIC(18,3)   NOT NULL,
    -- Status
    summary_status              VARCHAR(20)     NOT NULL DEFAULT 'DRAFT',
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p040_as_dates CHECK (
        reporting_period_start < reporting_period_end
    ),
    CONSTRAINT chk_p040_as_routine_count CHECK (
        routine_adjustment_count >= 0
    ),
    CONSTRAINT chk_p040_as_nonroutine_count CHECK (
        nonroutine_adjustment_count >= 0
    ),
    CONSTRAINT chk_p040_as_status CHECK (
        summary_status IN ('DRAFT', 'CALCULATED', 'REVIEWED', 'APPROVED', 'REVISED')
    ),
    CONSTRAINT chk_p040_as_unadj CHECK (
        unadjusted_baseline_kwh >= 0
    ),
    CONSTRAINT chk_p040_as_adj CHECK (
        adjusted_baseline_kwh >= 0
    ),
    CONSTRAINT uq_p040_as_project_period UNIQUE (project_id, baseline_id, reporting_period_start)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p040_as_tenant            ON pack040_mv.mv_adjustment_summaries(tenant_id);
CREATE INDEX idx_p040_as_project           ON pack040_mv.mv_adjustment_summaries(project_id);
CREATE INDEX idx_p040_as_baseline          ON pack040_mv.mv_adjustment_summaries(baseline_id);
CREATE INDEX idx_p040_as_period            ON pack040_mv.mv_adjustment_summaries(reporting_period_start, reporting_period_end);
CREATE INDEX idx_p040_as_status            ON pack040_mv.mv_adjustment_summaries(summary_status);
CREATE INDEX idx_p040_as_created           ON pack040_mv.mv_adjustment_summaries(created_at DESC);

-- Composite: project + approved summaries
CREATE INDEX idx_p040_as_project_approved  ON pack040_mv.mv_adjustment_summaries(project_id, reporting_period_start)
    WHERE summary_status = 'APPROVED';

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p040_as_updated
    BEFORE UPDATE ON pack040_mv.mv_adjustment_summaries
    FOR EACH ROW EXECUTE FUNCTION pack040_mv.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack040_mv.mv_routine_adjustments ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_nonroutine_adjustments ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_adjustment_factors ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_adjustment_docs ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack040_mv.mv_adjustment_summaries ENABLE ROW LEVEL SECURITY;

CREATE POLICY p040_ra_tenant_isolation
    ON pack040_mv.mv_routine_adjustments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_ra_service_bypass
    ON pack040_mv.mv_routine_adjustments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_nra_tenant_isolation
    ON pack040_mv.mv_nonroutine_adjustments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_nra_service_bypass
    ON pack040_mv.mv_nonroutine_adjustments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_af_tenant_isolation
    ON pack040_mv.mv_adjustment_factors
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_af_service_bypass
    ON pack040_mv.mv_adjustment_factors
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_ad_tenant_isolation
    ON pack040_mv.mv_adjustment_docs
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_ad_service_bypass
    ON pack040_mv.mv_adjustment_docs
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p040_as_tenant_isolation
    ON pack040_mv.mv_adjustment_summaries
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p040_as_service_bypass
    ON pack040_mv.mv_adjustment_summaries
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_routine_adjustments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_nonroutine_adjustments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_adjustment_factors TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_adjustment_docs TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack040_mv.mv_adjustment_summaries TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack040_mv.mv_routine_adjustments IS
    'Routine adjustments normalizing baseline predictions for weather, production, occupancy, and operating hours per IPMVP.';
COMMENT ON TABLE pack040_mv.mv_nonroutine_adjustments IS
    'Non-routine adjustments for structural changes (floor area, equipment, schedule) between baseline and reporting periods per IPMVP.';
COMMENT ON TABLE pack040_mv.mv_adjustment_factors IS
    'Reusable adjustment factors (weather coefficients, production scaling ratios) derived from regression or engineering calculation.';
COMMENT ON TABLE pack040_mv.mv_adjustment_docs IS
    'Supporting documentation for adjustment justification including engineering analyses, equipment specs, and measurement data.';
COMMENT ON TABLE pack040_mv.mv_adjustment_summaries IS
    'Consolidated adjustment summaries per reporting period aggregating all routine and non-routine adjustments for savings calculation.';

COMMENT ON COLUMN pack040_mv.mv_routine_adjustments.adjustment_type IS 'Type of routine adjustment: WEATHER, PRODUCTION, OCCUPANCY, OPERATING_HOURS.';
COMMENT ON COLUMN pack040_mv.mv_routine_adjustments.adjusted_baseline_kwh IS 'Baseline energy prediction adjusted to reporting period conditions.';
COMMENT ON COLUMN pack040_mv.mv_routine_adjustments.provenance_hash IS 'SHA-256 hash for data integrity and audit provenance.';

COMMENT ON COLUMN pack040_mv.mv_nonroutine_adjustments.adjustment_type IS 'Type of non-routine change: FLOOR_AREA_CHANGE, EQUIPMENT_ADDITION, SCHEDULE_CHANGE, etc.';
COMMENT ON COLUMN pack040_mv.mv_nonroutine_adjustments.load_factor IS 'Ratio of actual load to rated capacity for equipment adjustments (0-1).';
COMMENT ON COLUMN pack040_mv.mv_nonroutine_adjustments.diversity_factor IS 'Fraction of equipment operating simultaneously (0-1).';

COMMENT ON COLUMN pack040_mv.mv_adjustment_factors.factor_type IS 'Factor type: WEATHER_COEFFICIENT, HDD_COEFFICIENT, PRODUCTION_COEFFICIENT, etc.';
COMMENT ON COLUMN pack040_mv.mv_adjustment_factors.derivation_method IS 'How the factor was derived: REGRESSION, ENGINEERING_CALCULATION, METERED, STIPULATED.';

COMMENT ON COLUMN pack040_mv.mv_adjustment_summaries.adjusted_baseline_kwh IS 'Baseline prediction after applying all routine and non-routine adjustments.';
COMMENT ON COLUMN pack040_mv.mv_adjustment_summaries.total_adjustment_pct IS 'Total adjustment as percentage of unadjusted baseline.';
