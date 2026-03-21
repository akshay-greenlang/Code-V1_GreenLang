-- =============================================================================
-- V199: PACK-032 Building Energy Assessment - Indoor Environment & Whole Life Carbon
-- =============================================================================
-- Pack:         PACK-032 (Building Energy Assessment Pack)
-- Migration:    009 of 010
-- Date:         March 2026
--
-- Creates indoor environmental quality (IEQ) assessment and whole life carbon
-- (WLC) analysis tables for holistic building performance evaluation.
--
-- Tables (3):
--   1. pack032_building_assessment.indoor_environment_assessments
--   2. pack032_building_assessment.whole_life_carbon
--   3. pack032_building_assessment.material_quantities
--
-- Previous: V198__pack032_building_assessment_008.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack032_building_assessment.indoor_environment_assessments
-- =============================================================================
-- Indoor environmental quality assessments including thermal comfort (PMV/PPD),
-- air quality (CO2), temperature, humidity, ventilation, overheating, and
-- daylight performance.

CREATE TABLE pack032_building_assessment.indoor_environment_assessments (
    assessment_id           UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    ieq_category            VARCHAR(100)    NOT NULL,
    pmv_avg                 NUMERIC(6,2),
    ppd_avg                 NUMERIC(6,2),
    co2_avg_ppm             NUMERIC(8,2),
    temperature_avg_c       NUMERIC(6,2),
    humidity_avg_pct        NUMERIC(6,2),
    ventilation_rate_ls_per_person NUMERIC(8,2),
    overheating_hours       INTEGER,
    daylight_factor_avg     NUMERIC(6,2),
    ieq_score               NUMERIC(6,2),
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_iea_pmv CHECK (
        pmv_avg IS NULL OR (pmv_avg >= -3 AND pmv_avg <= 3)
    ),
    CONSTRAINT chk_p032_iea_ppd CHECK (
        ppd_avg IS NULL OR (ppd_avg >= 0 AND ppd_avg <= 100)
    ),
    CONSTRAINT chk_p032_iea_co2 CHECK (
        co2_avg_ppm IS NULL OR co2_avg_ppm >= 0
    ),
    CONSTRAINT chk_p032_iea_temp CHECK (
        temperature_avg_c IS NULL OR (temperature_avg_c >= -20 AND temperature_avg_c <= 60)
    ),
    CONSTRAINT chk_p032_iea_humidity CHECK (
        humidity_avg_pct IS NULL OR (humidity_avg_pct >= 0 AND humidity_avg_pct <= 100)
    ),
    CONSTRAINT chk_p032_iea_ventilation CHECK (
        ventilation_rate_ls_per_person IS NULL OR ventilation_rate_ls_per_person >= 0
    ),
    CONSTRAINT chk_p032_iea_overheating CHECK (
        overheating_hours IS NULL OR overheating_hours >= 0
    ),
    CONSTRAINT chk_p032_iea_daylight CHECK (
        daylight_factor_avg IS NULL OR daylight_factor_avg >= 0
    ),
    CONSTRAINT chk_p032_iea_score CHECK (
        ieq_score IS NULL OR (ieq_score >= 0 AND ieq_score <= 100)
    ),
    CONSTRAINT chk_p032_iea_category CHECK (
        ieq_category IN ('THERMAL_COMFORT', 'AIR_QUALITY', 'VISUAL_COMFORT',
                           'ACOUSTIC_COMFORT', 'OVERALL', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_iea_building ON pack032_building_assessment.indoor_environment_assessments(building_id);
CREATE INDEX idx_p032_iea_tenant   ON pack032_building_assessment.indoor_environment_assessments(tenant_id);
CREATE INDEX idx_p032_iea_category ON pack032_building_assessment.indoor_environment_assessments(ieq_category);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_iea_updated
    BEFORE UPDATE ON pack032_building_assessment.indoor_environment_assessments
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack032_building_assessment.whole_life_carbon
-- =============================================================================
-- Whole life carbon analysis per EN 15978 life cycle modules (A1-A5, B1-B7,
-- C1-C4, D) with total WLC and target comparison.

CREATE TABLE pack032_building_assessment.whole_life_carbon (
    wlc_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    study_period_years      INTEGER         NOT NULL,
    a1_a3_kgco2e            NUMERIC(16,2),
    a4_kgco2e               NUMERIC(16,2),
    a5_kgco2e               NUMERIC(16,2),
    b1_b5_kgco2e            NUMERIC(16,2),
    b6_kgco2e               NUMERIC(16,2),
    b7_kgco2e               NUMERIC(16,2),
    c1_c4_kgco2e            NUMERIC(16,2),
    d_kgco2e                NUMERIC(16,2),
    total_wlc_kgco2e        NUMERIC(16,2),
    wlc_per_m2              NUMERIC(10,2),
    target_kgco2e_m2        NUMERIC(10,2),
    target_source           VARCHAR(255),
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_wlc_study_period CHECK (
        study_period_years > 0 AND study_period_years <= 150
    ),
    CONSTRAINT chk_p032_wlc_a1_a3 CHECK (
        a1_a3_kgco2e IS NULL OR a1_a3_kgco2e >= 0
    ),
    CONSTRAINT chk_p032_wlc_a4 CHECK (
        a4_kgco2e IS NULL OR a4_kgco2e >= 0
    ),
    CONSTRAINT chk_p032_wlc_a5 CHECK (
        a5_kgco2e IS NULL OR a5_kgco2e >= 0
    ),
    CONSTRAINT chk_p032_wlc_b1_b5 CHECK (
        b1_b5_kgco2e IS NULL OR b1_b5_kgco2e >= 0
    ),
    CONSTRAINT chk_p032_wlc_b6 CHECK (
        b6_kgco2e IS NULL OR b6_kgco2e >= 0
    ),
    CONSTRAINT chk_p032_wlc_b7 CHECK (
        b7_kgco2e IS NULL OR b7_kgco2e >= 0
    ),
    CONSTRAINT chk_p032_wlc_c1_c4 CHECK (
        c1_c4_kgco2e IS NULL OR c1_c4_kgco2e >= 0
    ),
    CONSTRAINT chk_p032_wlc_total CHECK (
        total_wlc_kgco2e IS NULL OR total_wlc_kgco2e >= 0
    ),
    CONSTRAINT chk_p032_wlc_per_m2 CHECK (
        wlc_per_m2 IS NULL OR wlc_per_m2 >= 0
    ),
    CONSTRAINT chk_p032_wlc_target CHECK (
        target_kgco2e_m2 IS NULL OR target_kgco2e_m2 >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_wlc_building ON pack032_building_assessment.whole_life_carbon(building_id);
CREATE INDEX idx_p032_wlc_tenant   ON pack032_building_assessment.whole_life_carbon(tenant_id);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_wlc_updated
    BEFORE UPDATE ON pack032_building_assessment.whole_life_carbon
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack032_building_assessment.material_quantities
-- =============================================================================
-- Material quantities for embodied carbon calculation linked to WLC studies
-- with EPD references and emission factors.

CREATE TABLE pack032_building_assessment.material_quantities (
    material_id             UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    wlc_id                  UUID            NOT NULL REFERENCES pack032_building_assessment.whole_life_carbon(wlc_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    material_name           VARCHAR(255)    NOT NULL,
    category                VARCHAR(100)    NOT NULL,
    quantity_kg             NUMERIC(14,2),
    ec_factor_kgco2e_per_kg NUMERIC(10,6),
    embodied_carbon_kgco2e  NUMERIC(14,2),
    epd_reference           VARCHAR(255),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_mq_quantity CHECK (
        quantity_kg IS NULL OR quantity_kg >= 0
    ),
    CONSTRAINT chk_p032_mq_ec_factor CHECK (
        ec_factor_kgco2e_per_kg IS NULL OR ec_factor_kgco2e_per_kg >= 0
    ),
    CONSTRAINT chk_p032_mq_embodied CHECK (
        embodied_carbon_kgco2e IS NULL OR embodied_carbon_kgco2e >= 0
    ),
    CONSTRAINT chk_p032_mq_category CHECK (
        category IN ('CONCRETE', 'STEEL', 'TIMBER', 'MASONRY', 'ALUMINIUM',
                       'GLASS', 'INSULATION', 'PLASTICS', 'COPPER', 'CLADDING',
                       'ROOFING', 'FLOORING', 'MEP', 'FINISHES', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_mq_wlc      ON pack032_building_assessment.material_quantities(wlc_id);
CREATE INDEX idx_p032_mq_tenant   ON pack032_building_assessment.material_quantities(tenant_id);
CREATE INDEX idx_p032_mq_category ON pack032_building_assessment.material_quantities(category);

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack032_building_assessment.indoor_environment_assessments ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.whole_life_carbon ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.material_quantities ENABLE ROW LEVEL SECURITY;

CREATE POLICY p032_iea_tenant_isolation
    ON pack032_building_assessment.indoor_environment_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_iea_service_bypass
    ON pack032_building_assessment.indoor_environment_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_wlc_tenant_isolation
    ON pack032_building_assessment.whole_life_carbon
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_wlc_service_bypass
    ON pack032_building_assessment.whole_life_carbon
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_mq_tenant_isolation
    ON pack032_building_assessment.material_quantities
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_mq_service_bypass
    ON pack032_building_assessment.material_quantities
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.indoor_environment_assessments TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.whole_life_carbon TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.material_quantities TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack032_building_assessment.indoor_environment_assessments IS
    'Indoor environmental quality assessments with thermal comfort (PMV/PPD), air quality, temperature, humidity, ventilation, overheating, and daylight.';

COMMENT ON TABLE pack032_building_assessment.whole_life_carbon IS
    'Whole life carbon analysis per EN 15978 life cycle modules (A1-D) with total WLC and target comparison.';

COMMENT ON TABLE pack032_building_assessment.material_quantities IS
    'Material quantities for embodied carbon calculation linked to WLC studies with EPD references.';

COMMENT ON COLUMN pack032_building_assessment.indoor_environment_assessments.pmv_avg IS
    'Predicted Mean Vote (-3 cold to +3 hot), ISO 7730.';
COMMENT ON COLUMN pack032_building_assessment.indoor_environment_assessments.ppd_avg IS
    'Predicted Percentage Dissatisfied (0-100%), ISO 7730.';
COMMENT ON COLUMN pack032_building_assessment.indoor_environment_assessments.overheating_hours IS
    'Number of occupied hours exceeding overheating threshold (CIBSE TM59/TM52).';
COMMENT ON COLUMN pack032_building_assessment.indoor_environment_assessments.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack032_building_assessment.whole_life_carbon.a1_a3_kgco2e IS
    'Product stage embodied carbon: raw material supply, transport, manufacturing (kgCO2e).';
COMMENT ON COLUMN pack032_building_assessment.whole_life_carbon.b6_kgco2e IS
    'Operational energy use carbon over the study period (kgCO2e).';
COMMENT ON COLUMN pack032_building_assessment.whole_life_carbon.b7_kgco2e IS
    'Operational water use carbon over the study period (kgCO2e).';
COMMENT ON COLUMN pack032_building_assessment.whole_life_carbon.d_kgco2e IS
    'Benefits and loads beyond the system boundary: reuse, recovery, recycling (kgCO2e).';
COMMENT ON COLUMN pack032_building_assessment.whole_life_carbon.target_source IS
    'Source of the WLC target (e.g., LETI 2020, RIBA 2030, GLA, RICS).';
COMMENT ON COLUMN pack032_building_assessment.material_quantities.epd_reference IS
    'Environmental Product Declaration reference or database source.';
