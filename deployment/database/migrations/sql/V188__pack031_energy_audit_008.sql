-- =============================================================================
-- V188: PACK-031 Industrial Energy Audit - Waste Heat Recovery
-- =============================================================================
-- Pack:         PACK-031 (Industrial Energy Audit Pack)
-- Migration:    008 of 010
-- Date:         March 2026
--
-- Waste heat recovery analysis tables including heat source inventory,
-- heat sink requirements, pinch analysis results, and heat recovery
-- project planning with technology selection and financial assessment.
--
-- Tables (4):
--   1. pack031_energy_audit.waste_heat_sources
--   2. pack031_energy_audit.heat_sinks
--   3. pack031_energy_audit.pinch_analyses
--   4. pack031_energy_audit.heat_recovery_projects
--
-- Previous: V187__pack031_energy_audit_007.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack031_energy_audit.waste_heat_sources
-- =============================================================================
-- Inventory of waste heat sources with temperature, flow rate,
-- available thermal energy, and temperature grade classification.

CREATE TABLE pack031_energy_audit.waste_heat_sources (
    source_id               UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    name                    VARCHAR(255)    NOT NULL,
    source_type             VARCHAR(100)    NOT NULL,
    medium                  VARCHAR(50),
    flow_rate_kg_s          NUMERIC(10,4),
    temperature_c           NUMERIC(8,2)    NOT NULL,
    return_temperature_c    NUMERIC(8,2),
    specific_heat_kj_kgk    NUMERIC(8,4)    DEFAULT 4.186,
    available_heat_kw       NUMERIC(12,4),
    operating_hours         INTEGER,
    annual_available_kwh    NUMERIC(14,4),
    temperature_grade       VARCHAR(20),
    variability             VARCHAR(20)     DEFAULT 'steady',
    corrosive               BOOLEAN         DEFAULT FALSE,
    contaminated            BOOLEAN         DEFAULT FALSE,
    equipment_id            UUID            REFERENCES pack031_energy_audit.equipment(equipment_id) ON DELETE SET NULL,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_whs_source_type CHECK (
        source_type IN ('FLUE_GAS', 'EXHAUST_AIR', 'PROCESS_WATER', 'CONDENSATE',
                        'COOLING_WATER', 'COMPRESSOR_HEAT', 'REFRIGERATION_REJECT',
                        'FURNACE_EXHAUST', 'DRYER_EXHAUST', 'OVEN_EXHAUST',
                        'STEAM_BLOWDOWN', 'FLASH_STEAM', 'RADIATION', 'OTHER')
    ),
    CONSTRAINT chk_p031_whs_flow CHECK (
        flow_rate_kg_s IS NULL OR flow_rate_kg_s >= 0
    ),
    CONSTRAINT chk_p031_whs_heat CHECK (
        available_heat_kw IS NULL OR available_heat_kw >= 0
    ),
    CONSTRAINT chk_p031_whs_grade CHECK (
        temperature_grade IS NULL OR temperature_grade IN ('HIGH', 'MEDIUM', 'LOW', 'VERY_LOW')
    ),
    CONSTRAINT chk_p031_whs_variability CHECK (
        variability IN ('steady', 'intermittent', 'batch', 'seasonal')
    ),
    CONSTRAINT chk_p031_whs_hours CHECK (
        operating_hours IS NULL OR (operating_hours >= 0 AND operating_hours <= 8784)
    )
);

-- Indexes
CREATE INDEX idx_p031_whs_facility     ON pack031_energy_audit.waste_heat_sources(facility_id);
CREATE INDEX idx_p031_whs_tenant       ON pack031_energy_audit.waste_heat_sources(tenant_id);
CREATE INDEX idx_p031_whs_type         ON pack031_energy_audit.waste_heat_sources(source_type);
CREATE INDEX idx_p031_whs_temp         ON pack031_energy_audit.waste_heat_sources(temperature_c);
CREATE INDEX idx_p031_whs_grade        ON pack031_energy_audit.waste_heat_sources(temperature_grade);
CREATE INDEX idx_p031_whs_heat         ON pack031_energy_audit.waste_heat_sources(available_heat_kw DESC);

-- Trigger
CREATE TRIGGER trg_p031_whs_updated
    BEFORE UPDATE ON pack031_energy_audit.waste_heat_sources
    FOR EACH ROW EXECUTE FUNCTION pack031_energy_audit.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack031_energy_audit.heat_sinks
-- =============================================================================
-- Heat demand points (sinks) that can potentially utilize recovered
-- waste heat, with required temperature and current energy source.

CREATE TABLE pack031_energy_audit.heat_sinks (
    sink_id                 UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    name                    VARCHAR(255)    NOT NULL,
    sink_type               VARCHAR(100),
    required_heat_kw        NUMERIC(12,4)   NOT NULL,
    temperature_required_c  NUMERIC(8,2)    NOT NULL,
    return_temperature_c    NUMERIC(8,2),
    current_source          VARCHAR(100),
    current_cost_eur_yr     NUMERIC(12,4),
    operating_hours         INTEGER,
    annual_demand_kwh       NUMERIC(14,4),
    medium                  VARCHAR(50),
    flow_rate_kg_s          NUMERIC(10,4),
    priority                VARCHAR(20)     DEFAULT 'medium',
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_hs_heat CHECK (required_heat_kw >= 0),
    CONSTRAINT chk_p031_hs_cost CHECK (
        current_cost_eur_yr IS NULL OR current_cost_eur_yr >= 0
    ),
    CONSTRAINT chk_p031_hs_hours CHECK (
        operating_hours IS NULL OR (operating_hours >= 0 AND operating_hours <= 8784)
    ),
    CONSTRAINT chk_p031_hs_priority CHECK (
        priority IN ('critical', 'high', 'medium', 'low')
    )
);

-- Indexes
CREATE INDEX idx_p031_hs_facility      ON pack031_energy_audit.heat_sinks(facility_id);
CREATE INDEX idx_p031_hs_tenant        ON pack031_energy_audit.heat_sinks(tenant_id);
CREATE INDEX idx_p031_hs_temp          ON pack031_energy_audit.heat_sinks(temperature_required_c);
CREATE INDEX idx_p031_hs_heat          ON pack031_energy_audit.heat_sinks(required_heat_kw DESC);

-- =============================================================================
-- Table 3: pack031_energy_audit.pinch_analyses
-- =============================================================================
-- Pinch analysis results determining theoretical maximum heat recovery,
-- minimum utility requirements, and composite curve data.

CREATE TABLE pack031_energy_audit.pinch_analyses (
    pinch_id                UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    analysis_date           DATE            NOT NULL,
    delta_t_min_c           NUMERIC(8,2)    NOT NULL DEFAULT 10.0,
    pinch_temperature_c     NUMERIC(8,2),
    min_heating_utility_kw  NUMERIC(12,4),
    min_cooling_utility_kw  NUMERIC(12,4),
    max_heat_recovery_kw    NUMERIC(12,4),
    current_heat_recovery_kw NUMERIC(12,4),
    recovery_gap_kw         NUMERIC(12,4),
    recovery_potential_pct  NUMERIC(5,2),
    hot_composite_curve     JSONB           DEFAULT '[]',
    cold_composite_curve    JSONB           DEFAULT '[]',
    grand_composite_curve   JSONB           DEFAULT '[]',
    analysis_software       VARCHAR(100),
    analyst_name            VARCHAR(255),
    methodology_notes       TEXT,
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_pinch_dt CHECK (delta_t_min_c > 0),
    CONSTRAINT chk_p031_pinch_heating CHECK (
        min_heating_utility_kw IS NULL OR min_heating_utility_kw >= 0
    ),
    CONSTRAINT chk_p031_pinch_cooling CHECK (
        min_cooling_utility_kw IS NULL OR min_cooling_utility_kw >= 0
    ),
    CONSTRAINT chk_p031_pinch_recovery CHECK (
        max_heat_recovery_kw IS NULL OR max_heat_recovery_kw >= 0
    ),
    CONSTRAINT chk_p031_pinch_pct CHECK (
        recovery_potential_pct IS NULL OR (recovery_potential_pct >= 0 AND recovery_potential_pct <= 100)
    )
);

-- Indexes
CREATE INDEX idx_p031_pinch_facility   ON pack031_energy_audit.pinch_analyses(facility_id);
CREATE INDEX idx_p031_pinch_tenant     ON pack031_energy_audit.pinch_analyses(tenant_id);
CREATE INDEX idx_p031_pinch_date       ON pack031_energy_audit.pinch_analyses(analysis_date);
CREATE INDEX idx_p031_pinch_recovery   ON pack031_energy_audit.pinch_analyses(max_heat_recovery_kw DESC);

-- =============================================================================
-- Table 4: pack031_energy_audit.heat_recovery_projects
-- =============================================================================
-- Heat recovery project definitions matching waste heat sources to
-- sinks with technology selection, sizing, and financial assessment.

CREATE TABLE pack031_energy_audit.heat_recovery_projects (
    project_id              UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    facility_id             UUID            NOT NULL REFERENCES pack031_energy_audit.energy_audit_facilities(facility_id) ON DELETE CASCADE,
    source_id               UUID            REFERENCES pack031_energy_audit.waste_heat_sources(source_id) ON DELETE SET NULL,
    sink_id                 UUID            REFERENCES pack031_energy_audit.heat_sinks(sink_id) ON DELETE SET NULL,
    tenant_id               UUID            NOT NULL,
    name                    VARCHAR(255)    NOT NULL,
    technology              VARCHAR(100)    NOT NULL,
    heat_duty_kw            NUMERIC(12,4),
    source_temp_in_c        NUMERIC(8,2),
    source_temp_out_c       NUMERIC(8,2),
    sink_temp_in_c          NUMERIC(8,2),
    sink_temp_out_c         NUMERIC(8,2),
    exchanger_area_m2       NUMERIC(10,4),
    annual_recovery_kwh     NUMERIC(14,4),
    capex_eur               NUMERIC(14,4),
    annual_savings_eur      NUMERIC(12,4),
    payback_years           NUMERIC(8,2),
    npv_eur                 NUMERIC(14,4),
    co2_savings_tonnes_yr   NUMERIC(12,4),
    status                  VARCHAR(30)     DEFAULT 'concept',
    priority                VARCHAR(20)     DEFAULT 'medium',
    metadata                JSONB           DEFAULT '{}',
    provenance_hash         VARCHAR(64),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p031_hrp_technology CHECK (
        technology IN ('PLATE_HX', 'SHELL_TUBE_HX', 'HEAT_PIPE', 'HEAT_PUMP',
                       'ORC', 'ECONOMIZER', 'AIR_PREHEATER', 'REGENERATOR',
                       'RECUPERATOR', 'RUNAROUND_COIL', 'THERMAL_WHEEL',
                       'CONDENSING_ECONOMIZER', 'WASTE_HEAT_BOILER', 'OTHER')
    ),
    CONSTRAINT chk_p031_hrp_duty CHECK (
        heat_duty_kw IS NULL OR heat_duty_kw >= 0
    ),
    CONSTRAINT chk_p031_hrp_area CHECK (
        exchanger_area_m2 IS NULL OR exchanger_area_m2 >= 0
    ),
    CONSTRAINT chk_p031_hrp_capex CHECK (
        capex_eur IS NULL OR capex_eur >= 0
    ),
    CONSTRAINT chk_p031_hrp_savings CHECK (
        annual_savings_eur IS NULL OR annual_savings_eur >= 0
    ),
    CONSTRAINT chk_p031_hrp_payback CHECK (
        payback_years IS NULL OR payback_years >= 0
    ),
    CONSTRAINT chk_p031_hrp_status CHECK (
        status IN ('concept', 'feasibility', 'design', 'approved', 'construction',
                   'operational', 'deferred', 'rejected')
    ),
    CONSTRAINT chk_p031_hrp_priority CHECK (
        priority IN ('critical', 'high', 'medium', 'low')
    )
);

-- Indexes
CREATE INDEX idx_p031_hrp_facility     ON pack031_energy_audit.heat_recovery_projects(facility_id);
CREATE INDEX idx_p031_hrp_source       ON pack031_energy_audit.heat_recovery_projects(source_id);
CREATE INDEX idx_p031_hrp_sink         ON pack031_energy_audit.heat_recovery_projects(sink_id);
CREATE INDEX idx_p031_hrp_tenant       ON pack031_energy_audit.heat_recovery_projects(tenant_id);
CREATE INDEX idx_p031_hrp_tech         ON pack031_energy_audit.heat_recovery_projects(technology);
CREATE INDEX idx_p031_hrp_status       ON pack031_energy_audit.heat_recovery_projects(status);
CREATE INDEX idx_p031_hrp_payback      ON pack031_energy_audit.heat_recovery_projects(payback_years);
CREATE INDEX idx_p031_hrp_savings      ON pack031_energy_audit.heat_recovery_projects(annual_savings_eur DESC);

-- Trigger
CREATE TRIGGER trg_p031_hrp_updated
    BEFORE UPDATE ON pack031_energy_audit.heat_recovery_projects
    FOR EACH ROW EXECUTE FUNCTION pack031_energy_audit.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack031_energy_audit.waste_heat_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.heat_sinks ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.pinch_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack031_energy_audit.heat_recovery_projects ENABLE ROW LEVEL SECURITY;

CREATE POLICY p031_whs_tenant_isolation ON pack031_energy_audit.waste_heat_sources
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_whs_service_bypass ON pack031_energy_audit.waste_heat_sources
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_hs_tenant_isolation ON pack031_energy_audit.heat_sinks
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_hs_service_bypass ON pack031_energy_audit.heat_sinks
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_pinch_tenant_isolation ON pack031_energy_audit.pinch_analyses
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_pinch_service_bypass ON pack031_energy_audit.pinch_analyses
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p031_hrp_tenant_isolation ON pack031_energy_audit.heat_recovery_projects
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p031_hrp_service_bypass ON pack031_energy_audit.heat_recovery_projects
    TO greenlang_service USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.waste_heat_sources TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.heat_sinks TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.pinch_analyses TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack031_energy_audit.heat_recovery_projects TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack031_energy_audit.waste_heat_sources IS
    'Waste heat source inventory with temperature, flow rate, available thermal energy, and temperature grade.';
COMMENT ON TABLE pack031_energy_audit.heat_sinks IS
    'Heat demand points (sinks) for waste heat utilization with required temperature and current source.';
COMMENT ON TABLE pack031_energy_audit.pinch_analyses IS
    'Pinch analysis results: theoretical max heat recovery, minimum utilities, and composite curve data.';
COMMENT ON TABLE pack031_energy_audit.heat_recovery_projects IS
    'Heat recovery projects matching sources to sinks with technology, sizing, and financial assessment.';

COMMENT ON COLUMN pack031_energy_audit.waste_heat_sources.temperature_grade IS
    'Temperature grade: HIGH (>400C), MEDIUM (150-400C), LOW (80-150C), VERY_LOW (<80C).';
COMMENT ON COLUMN pack031_energy_audit.pinch_analyses.delta_t_min_c IS
    'Minimum temperature approach (delta T min) for pinch analysis in degrees Celsius.';
