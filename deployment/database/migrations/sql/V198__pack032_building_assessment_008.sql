-- =============================================================================
-- V198: PACK-032 Building Energy Assessment - Retrofit & Certification
-- =============================================================================
-- Pack:         PACK-032 (Building Energy Assessment Pack)
-- Migration:    008 of 010
-- Date:         March 2026
--
-- Creates retrofit measures, retrofit plans, and green building certification
-- assessment tables for capital planning and compliance tracking.
--
-- Tables (3):
--   1. pack032_building_assessment.retrofit_measures
--   2. pack032_building_assessment.retrofit_plans
--   3. pack032_building_assessment.certification_assessments
--
-- Previous: V197__pack032_building_assessment_007.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack032_building_assessment.retrofit_measures
-- =============================================================================
-- Individual retrofit / energy conservation measures (ECMs) with energy
-- savings, cost, payback, NPV, IRR, and carbon savings.

CREATE TABLE pack032_building_assessment.retrofit_measures (
    measure_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    measure_name            VARCHAR(500)    NOT NULL,
    category                VARCHAR(100)    NOT NULL,
    description             TEXT,
    energy_savings_kwh      NUMERIC(14,2),
    cost_eur                NUMERIC(14,2),
    payback_years           NUMERIC(8,2),
    npv_eur                 NUMERIC(14,2),
    irr_pct                 NUMERIC(8,2),
    carbon_savings_kgco2    NUMERIC(14,2),
    priority                VARCHAR(20),
    implementation_status   VARCHAR(30)     DEFAULT 'proposed',
    phase                   VARCHAR(50),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_rm_energy_savings CHECK (
        energy_savings_kwh IS NULL OR energy_savings_kwh >= 0
    ),
    CONSTRAINT chk_p032_rm_cost CHECK (
        cost_eur IS NULL OR cost_eur >= 0
    ),
    CONSTRAINT chk_p032_rm_payback CHECK (
        payback_years IS NULL OR payback_years >= 0
    ),
    CONSTRAINT chk_p032_rm_irr CHECK (
        irr_pct IS NULL OR irr_pct >= -100
    ),
    CONSTRAINT chk_p032_rm_carbon CHECK (
        carbon_savings_kgco2 IS NULL OR carbon_savings_kgco2 >= 0
    ),
    CONSTRAINT chk_p032_rm_category CHECK (
        category IN ('ENVELOPE', 'HEATING', 'COOLING', 'VENTILATION', 'LIGHTING',
                      'DHW', 'RENEWABLES', 'CONTROLS', 'METERING', 'BEHAVIOUR',
                      'COMBINED', 'OTHER')
    ),
    CONSTRAINT chk_p032_rm_priority CHECK (
        priority IS NULL OR priority IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'DEFERRED')
    ),
    CONSTRAINT chk_p032_rm_status CHECK (
        implementation_status IN ('proposed', 'approved', 'in_progress', 'completed',
                                    'rejected', 'deferred')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_rm_building ON pack032_building_assessment.retrofit_measures(building_id);
CREATE INDEX idx_p032_rm_tenant   ON pack032_building_assessment.retrofit_measures(tenant_id);
CREATE INDEX idx_p032_rm_category ON pack032_building_assessment.retrofit_measures(category);
CREATE INDEX idx_p032_rm_priority ON pack032_building_assessment.retrofit_measures(priority);
CREATE INDEX idx_p032_rm_status   ON pack032_building_assessment.retrofit_measures(implementation_status);
CREATE INDEX idx_p032_rm_phase    ON pack032_building_assessment.retrofit_measures(phase);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_rm_updated
    BEFORE UPDATE ON pack032_building_assessment.retrofit_measures
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack032_building_assessment.retrofit_plans
-- =============================================================================
-- Aggregated retrofit plans with total CAPEX, savings, target EPC/EUI,
-- phased implementation, and financial analysis.

CREATE TABLE pack032_building_assessment.retrofit_plans (
    plan_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    plan_name               VARCHAR(500)    NOT NULL,
    total_capex_eur         NUMERIC(16,2),
    total_savings_kwh       NUMERIC(16,2),
    total_carbon_savings    NUMERIC(16,2),
    payback_years           NUMERIC(8,2),
    npv_eur                 NUMERIC(16,2),
    target_epc_rating       VARCHAR(5),
    target_eui              NUMERIC(10,2),
    phases                  JSONB           DEFAULT '[]',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_rp_capex CHECK (
        total_capex_eur IS NULL OR total_capex_eur >= 0
    ),
    CONSTRAINT chk_p032_rp_savings CHECK (
        total_savings_kwh IS NULL OR total_savings_kwh >= 0
    ),
    CONSTRAINT chk_p032_rp_carbon CHECK (
        total_carbon_savings IS NULL OR total_carbon_savings >= 0
    ),
    CONSTRAINT chk_p032_rp_payback CHECK (
        payback_years IS NULL OR payback_years >= 0
    ),
    CONSTRAINT chk_p032_rp_target_epc CHECK (
        target_epc_rating IS NULL OR target_epc_rating IN ('A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G')
    ),
    CONSTRAINT chk_p032_rp_target_eui CHECK (
        target_eui IS NULL OR target_eui >= 0
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_rp_building    ON pack032_building_assessment.retrofit_plans(building_id);
CREATE INDEX idx_p032_rp_tenant      ON pack032_building_assessment.retrofit_plans(tenant_id);
CREATE INDEX idx_p032_rp_target_epc  ON pack032_building_assessment.retrofit_plans(target_epc_rating);
CREATE INDEX idx_p032_rp_phases      ON pack032_building_assessment.retrofit_plans USING GIN(phases);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_rp_updated
    BEFORE UPDATE ON pack032_building_assessment.retrofit_plans
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 3: pack032_building_assessment.certification_assessments
-- =============================================================================
-- Green building certification tracking (BREEAM, LEED, NABERS, WELL, etc.)
-- with target level, current score, credits achieved, and gap analysis.

CREATE TABLE pack032_building_assessment.certification_assessments (
    cert_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    certification_type      VARCHAR(100)    NOT NULL,
    target_level            VARCHAR(100),
    current_score           NUMERIC(8,2),
    target_score            NUMERIC(8,2),
    credits_achieved        JSONB           DEFAULT '{}',
    gaps                    JSONB           DEFAULT '[]',
    action_plan             JSONB           DEFAULT '[]',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_ca_score CHECK (
        current_score IS NULL OR current_score >= 0
    ),
    CONSTRAINT chk_p032_ca_target CHECK (
        target_score IS NULL OR target_score >= 0
    ),
    CONSTRAINT chk_p032_ca_cert_type CHECK (
        certification_type IN ('BREEAM_NEW', 'BREEAM_REFURB', 'BREEAM_IN_USE',
                                 'LEED_BD_C', 'LEED_O_M', 'LEED_ID_C',
                                 'NABERS_ENERGY', 'NABERS_WATER', 'NABERS_INDOOR',
                                 'WELL', 'PASSIVHAUS', 'GREENSTAR', 'DGNB',
                                 'HQE', 'FITWEL', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_ca_building ON pack032_building_assessment.certification_assessments(building_id);
CREATE INDEX idx_p032_ca_tenant   ON pack032_building_assessment.certification_assessments(tenant_id);
CREATE INDEX idx_p032_ca_type     ON pack032_building_assessment.certification_assessments(certification_type);
CREATE INDEX idx_p032_ca_credits  ON pack032_building_assessment.certification_assessments USING GIN(credits_achieved);
CREATE INDEX idx_p032_ca_gaps     ON pack032_building_assessment.certification_assessments USING GIN(gaps);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_ca_updated
    BEFORE UPDATE ON pack032_building_assessment.certification_assessments
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack032_building_assessment.retrofit_measures ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.retrofit_plans ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.certification_assessments ENABLE ROW LEVEL SECURITY;

CREATE POLICY p032_rm_tenant_isolation
    ON pack032_building_assessment.retrofit_measures
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_rm_service_bypass
    ON pack032_building_assessment.retrofit_measures
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_rp_tenant_isolation
    ON pack032_building_assessment.retrofit_plans
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_rp_service_bypass
    ON pack032_building_assessment.retrofit_plans
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_ca_tenant_isolation
    ON pack032_building_assessment.certification_assessments
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_ca_service_bypass
    ON pack032_building_assessment.certification_assessments
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.retrofit_measures TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.retrofit_plans TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.certification_assessments TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack032_building_assessment.retrofit_measures IS
    'Individual retrofit / energy conservation measures (ECMs) with energy savings, cost, payback, NPV, IRR, and carbon savings.';

COMMENT ON TABLE pack032_building_assessment.retrofit_plans IS
    'Aggregated retrofit plans with total CAPEX, savings, target EPC/EUI, phased implementation, and financial analysis.';

COMMENT ON TABLE pack032_building_assessment.certification_assessments IS
    'Green building certification tracking (BREEAM, LEED, NABERS, WELL, etc.) with credits, gap analysis, and action plans.';

COMMENT ON COLUMN pack032_building_assessment.retrofit_measures.npv_eur IS
    'Net Present Value of the measure over its lifetime in EUR.';
COMMENT ON COLUMN pack032_building_assessment.retrofit_measures.irr_pct IS
    'Internal Rate of Return as a percentage.';
COMMENT ON COLUMN pack032_building_assessment.retrofit_plans.phases IS
    'JSON array of implementation phases with timelines, measures, and milestones.';
COMMENT ON COLUMN pack032_building_assessment.retrofit_plans.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack032_building_assessment.certification_assessments.credits_achieved IS
    'JSON object mapping credit categories to achieved points.';
COMMENT ON COLUMN pack032_building_assessment.certification_assessments.gaps IS
    'JSON array of identified gaps between current state and target certification level.';
