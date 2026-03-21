-- =============================================================================
-- V196: PACK-032 Building Energy Assessment - EPC & DEC Records
-- =============================================================================
-- Pack:         PACK-032 (Building Energy Assessment Pack)
-- Migration:    006 of 010
-- Date:         March 2026
--
-- Creates Energy Performance Certificate (EPC) and Display Energy Certificate
-- (DEC) tables for regulatory compliance tracking and assessment history.
--
-- Tables (2):
--   1. pack032_building_assessment.epc_certificates
--   2. pack032_building_assessment.dec_certificates
--
-- Previous: V195__pack032_building_assessment_005.sql
-- =============================================================================

-- =============================================================================
-- Table 1: pack032_building_assessment.epc_certificates
-- =============================================================================
-- Energy Performance Certificate records with rating, primary energy,
-- CO2 emissions, assessor details, lodgement reference, and improvement
-- recommendations.

CREATE TABLE pack032_building_assessment.epc_certificates (
    epc_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    epc_rating              VARCHAR(5)      NOT NULL,
    primary_energy_kwh_m2   NUMERIC(10,2),
    co2_emissions_kg_m2     NUMERIC(10,4),
    methodology             VARCHAR(100),
    assessment_date         DATE            NOT NULL,
    expiry_date             DATE,
    assessor_name           VARCHAR(255),
    assessor_accreditation  VARCHAR(100),
    lodgement_reference     VARCHAR(100),
    reference_building_energy NUMERIC(10,2),
    improvement_potential_kwh_m2 NUMERIC(10,2),
    recommendations         JSONB           DEFAULT '[]',
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_epc_rating CHECK (
        epc_rating IN ('A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G')
    ),
    CONSTRAINT chk_p032_epc_primary_energy CHECK (
        primary_energy_kwh_m2 IS NULL OR primary_energy_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p032_epc_co2 CHECK (
        co2_emissions_kg_m2 IS NULL OR co2_emissions_kg_m2 >= 0
    ),
    CONSTRAINT chk_p032_epc_ref_energy CHECK (
        reference_building_energy IS NULL OR reference_building_energy >= 0
    ),
    CONSTRAINT chk_p032_epc_improvement CHECK (
        improvement_potential_kwh_m2 IS NULL OR improvement_potential_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p032_epc_dates CHECK (
        expiry_date IS NULL OR expiry_date > assessment_date
    ),
    CONSTRAINT chk_p032_epc_methodology CHECK (
        methodology IS NULL OR methodology IN ('SAP', 'RdSAP', 'SBEM', 'DSM',
                                                  'EPBD_CALCULATION', 'PHPP', 'OTHER')
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_epc_building    ON pack032_building_assessment.epc_certificates(building_id);
CREATE INDEX idx_p032_epc_tenant      ON pack032_building_assessment.epc_certificates(tenant_id);
CREATE INDEX idx_p032_epc_rating      ON pack032_building_assessment.epc_certificates(epc_rating);
CREATE INDEX idx_p032_epc_assess_date ON pack032_building_assessment.epc_certificates(assessment_date DESC);
CREATE INDEX idx_p032_epc_expiry      ON pack032_building_assessment.epc_certificates(expiry_date);
CREATE INDEX idx_p032_epc_lodgement   ON pack032_building_assessment.epc_certificates(lodgement_reference);
CREATE INDEX idx_p032_epc_recs        ON pack032_building_assessment.epc_certificates USING GIN(recommendations);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_epc_updated
    BEFORE UPDATE ON pack032_building_assessment.epc_certificates
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- =============================================================================
-- Table 2: pack032_building_assessment.dec_certificates
-- =============================================================================
-- Display Energy Certificate records tracking actual operational energy
-- performance against benchmarks for public buildings.

CREATE TABLE pack032_building_assessment.dec_certificates (
    dec_id                  UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    building_id             UUID            NOT NULL REFERENCES pack032_building_assessment.building_profiles(building_id) ON DELETE CASCADE,
    tenant_id               UUID            NOT NULL,
    operational_rating      VARCHAR(5)      NOT NULL,
    electricity_kwh         NUMERIC(14,2),
    heating_kwh             NUMERIC(14,2),
    renewable_kwh           NUMERIC(14,2),
    benchmark_kwh_m2        NUMERIC(10,2),
    assessment_date         DATE            NOT NULL,
    display_until_date      DATE,
    advisory_report_id      VARCHAR(100),
    provenance_hash         VARCHAR(64),
    metadata                JSONB           DEFAULT '{}',
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p032_dec_rating CHECK (
        operational_rating IN ('A', 'B', 'C', 'D', 'E', 'F', 'G')
    ),
    CONSTRAINT chk_p032_dec_electricity CHECK (
        electricity_kwh IS NULL OR electricity_kwh >= 0
    ),
    CONSTRAINT chk_p032_dec_heating CHECK (
        heating_kwh IS NULL OR heating_kwh >= 0
    ),
    CONSTRAINT chk_p032_dec_renewable CHECK (
        renewable_kwh IS NULL OR renewable_kwh >= 0
    ),
    CONSTRAINT chk_p032_dec_benchmark CHECK (
        benchmark_kwh_m2 IS NULL OR benchmark_kwh_m2 >= 0
    ),
    CONSTRAINT chk_p032_dec_dates CHECK (
        display_until_date IS NULL OR display_until_date > assessment_date
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p032_dec_building    ON pack032_building_assessment.dec_certificates(building_id);
CREATE INDEX idx_p032_dec_tenant      ON pack032_building_assessment.dec_certificates(tenant_id);
CREATE INDEX idx_p032_dec_rating      ON pack032_building_assessment.dec_certificates(operational_rating);
CREATE INDEX idx_p032_dec_assess_date ON pack032_building_assessment.dec_certificates(assessment_date DESC);
CREATE INDEX idx_p032_dec_display     ON pack032_building_assessment.dec_certificates(display_until_date);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p032_dec_updated
    BEFORE UPDATE ON pack032_building_assessment.dec_certificates
    FOR EACH ROW EXECUTE FUNCTION pack032_building_assessment.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
ALTER TABLE pack032_building_assessment.epc_certificates ENABLE ROW LEVEL SECURITY;
ALTER TABLE pack032_building_assessment.dec_certificates ENABLE ROW LEVEL SECURITY;

CREATE POLICY p032_epc_tenant_isolation
    ON pack032_building_assessment.epc_certificates
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_epc_service_bypass
    ON pack032_building_assessment.epc_certificates
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p032_dec_tenant_isolation
    ON pack032_building_assessment.dec_certificates
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p032_dec_service_bypass
    ON pack032_building_assessment.dec_certificates
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.epc_certificates TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON pack032_building_assessment.dec_certificates TO PUBLIC;

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE pack032_building_assessment.epc_certificates IS
    'Energy Performance Certificate records with rating, primary energy, CO2 intensity, assessor details, and improvement recommendations.';

COMMENT ON TABLE pack032_building_assessment.dec_certificates IS
    'Display Energy Certificate records tracking actual operational energy performance against benchmarks for public buildings.';

COMMENT ON COLUMN pack032_building_assessment.epc_certificates.epc_rating IS
    'EPC energy efficiency rating band (A+ to G).';
COMMENT ON COLUMN pack032_building_assessment.epc_certificates.primary_energy_kwh_m2 IS
    'Primary energy consumption per m2 per year.';
COMMENT ON COLUMN pack032_building_assessment.epc_certificates.co2_emissions_kg_m2 IS
    'CO2 emissions per m2 per year in kgCO2.';
COMMENT ON COLUMN pack032_building_assessment.epc_certificates.methodology IS
    'Assessment methodology (SAP, RdSAP, SBEM, PHPP, etc.).';
COMMENT ON COLUMN pack032_building_assessment.epc_certificates.lodgement_reference IS
    'Unique reference from the national EPC register.';
COMMENT ON COLUMN pack032_building_assessment.epc_certificates.provenance_hash IS
    'SHA-256 hash for data integrity and audit provenance.';
COMMENT ON COLUMN pack032_building_assessment.dec_certificates.operational_rating IS
    'DEC operational rating band (A to G) based on actual energy use.';
COMMENT ON COLUMN pack032_building_assessment.dec_certificates.benchmark_kwh_m2 IS
    'Benchmark energy use per m2 for the building category.';
